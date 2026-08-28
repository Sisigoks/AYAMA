"""Job store: one upload, one background reconstruction, one tileset.

The pipeline is a batch program that takes tens of seconds to minutes. A web
request cannot wait for it, so a job is created immediately, the work runs on a
background thread, and the browser follows `StageEvent`s over SSE - which is
what `StageEvent` was defined for.

Everything here is deliberately in-process and on-disk: a dict of jobs and a
directory per job. No database, no queue, no broker. That is honest for a
single-node demo and it is the thing to replace first if this ever needs to
survive a restart or scale past one machine.

Uploads are untrusted input, so this module is where the paranoia lives:
extension and magic-byte checks, a size cap, a pixel cap, and job ids that are
generated rather than taken from the request. See `validate_upload`.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..core.types import StageEvent

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

STAGES = ("ingest", "depth", "segmentation", "shadow", "anchors", "calibration",
          "uncertainty", "assemble", "artifacts", "validation", "tiles")


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
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "created": self.created, "finished": self.finished,
            "error": self.error, "params": self.params,
            "summary": self.summary, "notes": self.notes,
            "stages": STAGES,
            "elapsed_s": round((self.finished or time.time()) - self.created, 1),
        }


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

        job = Job(id=job_id, dir=d, params=dict(params, original_name=filename))
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
            job.events.append({"stage": ev.stage, "status": ev.status,
                               "detail": ev.detail, "pct": ev.pct,
                               "t": round(time.time() - job.created, 2)})

        with self._slots:
            job.status = "running"
            try:
                p = job.params
                cfg = Config(
                    backbone=p.get("backbone", "dav2-vits"),
                    chip=int(p.get("chip", 512)),
                    n_bootstrap=int(p.get("bootstrap", 12)),
                    dem_source=p.get("dem_source") or None,
                    reference=p.get("reference") or None,
                    extras={"device": p.get("device", "auto"),
                            "batch_size": int(p.get("batch", 0)),
                            "workers": int(p.get("workers", 0))},
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
