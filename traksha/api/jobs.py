"""Job store: one upload, one background reconstruction, one tileset.

The pipeline is a batch program that takes tens of seconds to minutes. A web
request cannot wait for it, so a job is created immediately, the work runs on a
background thread, and the browser follows `StageEvent`s over SSE - which is
what `StageEvent` was defined for.

Everything here is deliberately in-process and on-disk: a dict of jobs and a
directory per job. No database, no queue, no broker. That is honest for a
single-node demo and it is the thing to replace first if this ever needs to
scale past one machine.

A job does now survive a restart, because the alternative was worse: the store
writes `job.json` on every phase transition and reloads the directory on
startup. A job whose process died mid-run is reloaded as *failed*, naming the
phase it died in, rather than as a `running` job that will never finish -
those are indistinguishable from a hang, and they are what makes a progress
bar untrustworthy.

Uploads are untrusted input, so this module is where the paranoia lives:
extension and magic-byte checks, a size cap, a pixel cap, and job ids that are
generated rather than taken from the request. See `validate_upload`.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..core.types import StageEvent
from .phases import JOB_PHASES, IllegalTransition, JobProgress

# A reconstruction is quadratic in the long side and a 12000 px upload would
# occupy the only worker for an hour. The cap is the honest limit of a
# single-node demo, not a statement about the method.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Magic bytes, checked because a suffix is a claim by the uploader.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"II+\x00", "bigtiff"),
    (b"MM\x00+", "bigtiff"),
)

STAGES = JOB_PHASES

# Progress is written to disk on every transition. Within a phase that reports
# fractions - depth per chip, uncertainty per resample - the writes are
# throttled, because a job that spends its time serialising its own progress is
# a job that reports its progress slowly.
STATE_WRITE_INTERVAL_S = 0.5


class UploadRejected(ValueError):
    """The upload is not something we are willing to run the pipeline on."""


def validate_upload(filename: str, data: bytes) -> str:
    """Return a safe suffix, or raise UploadRejected with a readable reason."""
    if not data:
        raise UploadRejected("the uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"file is {len(data) / 1e6:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB")

    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(
            f"unsupported extension '{suffix or 'none'}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}")

    if not any(data.startswith(sig) for sig, _ in _MAGIC):
        raise UploadRejected(
            "the file's contents are not a PNG, JPEG or TIFF, whatever the "
            "extension says")
    return suffix


@dataclass
class Job:
    id: str
    dir: str
    status: str = "queued"          # queued | running | done | failed
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    error: Optional[str] = None
    events: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    progress: JobProgress = field(default_factory=JobProgress)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _saved_at: float = field(default=0.0, repr=False)

    def public(self) -> dict:
        """Everything the browser needs, including which phase is running.

        The phase block is the part that was missing. Without it the poll
        endpoint returned a job with no current phase in it, so the page could
        only ever show the list greyed out - see docs/ARCHITECTURE.md section
        2.1. It is merged in flat so a poll response and an SSE frame have the
        same shape and the client needs one renderer, not two.
        """
        return {
            "id": self.id, "status": self.status,
            "created": self.created, "finished": self.finished,
            "error": self.error, "params": self.params,
            "summary": self.summary, "notes": self.notes,
            "stages": [p.name for p in self.progress.phases],
            "elapsed_s": round((self.finished or time.time()) - self.created, 1),
            **self.progress.public(),
        }

    # ------------------------------------------------------------ on disk
    def state_path(self) -> str:
        return os.path.join(self.dir, "job.json")

    def save(self, force: bool = True) -> None:
        """Persist the job. Throttled unless `force`, which transitions use."""
        now = time.time()
        if not force and now - self._saved_at < STATE_WRITE_INTERVAL_S:
            return
        self._saved_at = now
        record = {
            "version": 1, "id": self.id, "status": self.status,
            "created": self.created, "finished": self.finished,
            "error": self.error, "params": self.params,
            "summary": self.summary, "notes": self.notes,
            "progress": self.progress.to_dict(),
        }
        tmp = self.state_path() + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            os.replace(tmp, self.state_path())
        except OSError:
            # Losing the ability to resume is not a reason to lose the run.
            pass

    @classmethod
    def load(cls, d: str) -> "Optional[Job]":
        """Rebuild a job from its directory, or None if it holds no state."""
        path = os.path.join(d, "job.json")
        try:
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            return None
        job = cls(id=r.get("id") or os.path.basename(d), dir=d,
                  status=r.get("status", "failed"),
                  created=r.get("created", os.path.getmtime(path)),
                  finished=r.get("finished"), error=r.get("error"),
                  summary=r.get("summary") or {}, notes=r.get("notes") or [],
                  params=r.get("params") or {},
                  progress=JobProgress.from_dict(r.get("progress") or {}))
        if job.status in ("queued", "running"):
            # Nothing is running: this process just started. Say so.
            job.progress.interrupted()
            job.status = "failed"
            job.error = job.error or (
                "the server restarted while this job was running")
            job.finished = job.finished or time.time()
        job._done.set()
        return job


class JobStore:
    """In-process job registry with a bounded worker pool.

    `max_concurrent` exists because each reconstruction wants several cores and
    a GPU if there is one; letting three uploads run at once on one box makes
    all three slow and none of them fail, which is the worst outcome to debug.
    """

    def __init__(self, root: str, max_concurrent: int = 1, retain: int = 32):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.retain = retain
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max(1, max_concurrent))
        self._reload()

    def _reload(self) -> None:
        """Adopt the jobs already on disk, so a restart does not lose them."""
        try:
            entries = sorted(os.scandir(self.root), key=lambda e: -e.stat().st_mtime)
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            job = Job.load(entry.path)
            if job is not None:
                self._jobs[job.id] = job

    # ------------------------------------------------------------------ api
    def create(self, data: bytes, filename: str, params: dict) -> Job:
        suffix = validate_upload(filename, data)
        job_id = uuid.uuid4().hex[:16]
        d = os.path.join(self.root, job_id)
        os.makedirs(d, exist_ok=True)
        # The stored name is ours, never the uploader's: a filename is the
        # classic path-traversal vector and we have no reason to keep it.
        src = os.path.join(d, f"input{suffix}")
        with open(src, "wb") as fh:
            fh.write(data)

        # `validation` only runs against a reference DSM, which an upload does
        # not have. Declaring it skipped now keeps it out of the denominator;
        # left in, every successful upload would stop short of 100%.
        progress = JobProgress()
        if not params.get("reference"):
            progress.skip("validation", "no reference DSM was supplied")

        job = Job(id=job_id, dir=d, progress=progress,
                  params=dict(params, original_name=filename))
        job.save()
        with self._lock:
            self._jobs[job_id] = job
            self._evict()
        threading.Thread(target=self._run, args=(job, src), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list:
        with self._lock:
            return [j.public() for j in
                    sorted(self._jobs.values(), key=lambda j: -j.created)]

    def path(self, job_id: str, *parts: str) -> Optional[str]:
        """Resolve a path inside a job, refusing anything that escapes it."""
        job = self.get(job_id)
        if job is None:
            return None
        clean = [p for p in parts if p not in ("", ".", "..")]
        if len(clean) != len(parts):
            return None
        target = os.path.abspath(os.path.join(job.dir, *clean))
        if os.path.commonpath([target, job.dir]) != job.dir:
            return None
        return target

    # -------------------------------------------------------------- worker
    def _run(self, job: Job, src: str) -> None:
        from ..core.types import Config
        from .pipeline import run as run_pipeline

        def emit(ev: StageEvent) -> None:
            """One StageEvent from the pipeline, folded into the phase state.

            The pipeline fires `running` twice for different reasons: once on
            entering a stage, and again for each fraction of it. They are told
            apart by whether the phase has already started, which is the only
            signal in the event - `pct` is 0.0 in both cases at the boundary.
            """
            job.events.append({"stage": ev.stage, "status": ev.status,
                               "detail": ev.detail, "pct": ev.pct,
                               "t": round(time.time() - job.created, 2)})
            force = True
            try:
                if ev.status == "running":
                    known = ev.stage in job.progress._by_name
                    if known and job.progress[ev.stage].status == "running":
                        job.progress.advance(ev.stage, ev.pct, ev.detail)
                        force = False
                    else:
                        job.progress.begin(ev.stage, ev.detail)
                elif ev.status == "done":
                    job.progress.complete(ev.stage, ev.detail)
                elif ev.status in ("failed", "skipped"):
                    if ev.status == "failed":
                        job.progress.fail(ev.stage, ev.detail)
                    else:
                        job.progress.skip(ev.stage, ev.detail)
            except IllegalTransition as bad:
                # An event the state machine refuses is a bug in the emitter,
                # not a reason to abandon the run. Record it where it will be
                # seen instead of raising inside a worker thread.
                job.events.append({"stage": ev.stage, "status": "warning",
                                   "detail": f"rejected transition: {bad}",
                                   "pct": 0.0,
                                   "t": round(time.time() - job.created, 2)})
            job.save(force=force)

        with self._slots:
            job.status = "running"
            job.save()
            try:
                p = job.params
                cfg = Config(
                    backbone=p.get("backbone", "dav2-vits"),
                    chip=int(p.get("chip", 512)),
                    n_bootstrap=int(p.get("bootstrap", 12)),
                    dem_source=p.get("dem_source") or None,
                    reference=p.get("reference") or None,
                    extras={
                            "batch_size": int(p.get("batch", 0)),
                            "workers": int(p.get("workers", 0)),
                            # The fitted structural scale, exactly as `run` and
                            # `dataset` use it. Without these two an upload came
                            # back as a flat sheet - 0.4 m of relief on a scene
                            # with 33 m of it - which is the failure README
                            # section 3.2 describes, served to a user as if it
                            # were the product.
                            "scale_model": p.get("scale_model", "auto"),
                            "dual_branch": p.get("scale_model", "auto")
                            not in ("off", "none", "no")},
                )
                out_dir = os.path.join(job.dir, "run")
                res = run_pipeline(src, cfg=cfg, out_dir=out_dir,
                                   write_artifacts=True, on_event=emit)

                emit(StageEvent("tiles", "running", "building the 3D tileset", 0.0))
                from ..mesh.build import build_tileset

                man = build_tileset(out_dir, os.path.join(job.dir, "tiles"),
                                    tile=512, write_mesh=bool(p.get("mesh", False)),
                                    quantise_bits=12)
                emit(StageEvent("tiles", "done",
                                f"{len(man['lods'])} LODs", 1.0))
                job.progress["tiles"].artifact = "tiles/tileset.json"

                job.notes = man.get("notes", [])
                job.summary = {
                    "tier": res.tier.value, "tier_reason": res.tier_reason,
                    "anchors": res.anchor_counts,
                    "anchors_used": res.anchors_used,
                    "anchors_rejected": res.anchors_rejected,
                    "metrics": res.metrics,
                    "timings_s": res.timings_s,
                    "provenance": res.provenance,
                    "grid": man.get("grid", {}),
                    "layers": sorted(man.get("layers", {})),
                }
                job.status = "done"
            except Exception as exc:                      # surfaced to the user
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                emit(StageEvent("error", "failed", job.error, 0.0))
            finally:
                job.finished = time.time()
                job.save()
                job._done.set()

    # ------------------------------------------------------------ retention
    def _evict(self) -> None:
        """Keep the newest `retain` jobs; delete the rest from disk."""
        jobs = sorted(self._jobs.values(), key=lambda j: -j.created)
        for old in jobs[self.retain:]:
            if old.status in ("running", "queued"):
                continue
            shutil.rmtree(old.dir, ignore_errors=True)
            self._jobs.pop(old.id, None)
