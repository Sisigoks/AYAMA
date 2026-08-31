"""The job phase state machine: what the backend is doing, authoritatively.

The browser used to be told nothing. The server framed every message as a named
SSE event that the client's `onmessage` handler could not receive, and the poll
fallback returned a job record with no current-phase field in it, so the
progress list rendered every row grey from submission to completion. Nothing was
stuck; there was no channel. This module is the missing half: one object that
knows which phase is running, how far into it the work is, and what the overall
figure therefore is.

Three properties matter and each is here for a reason.

**Weights are measured, not assumed.** Depth is 80% of a run - 151.6 s of 189.3
on the reference scene - so an equal-weight bar would crawl to 10% and then sit
still for two and a half minutes. The weights below are median stage times over
the four delivered scenes; `scripts/phase_weights.py` regenerates them.

**Progress is monotonic by construction.** Every transition is validated, and a
phase cannot start once a later one has. A bar that goes backwards is worse than
one that stands still, because it tells the reader the number is invented.

**Skipped phases are removed from the denominator.** `validation` only runs when
a reference DSM was supplied, which an upload never has. Leaving it in the total
would cap every upload at 97%, and a bar that never reaches 100% on success is
a bug report waiting to be filed.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

# Median seconds per stage over the four delivered swisstopo scenes at 1024 px
# (results/dataset.json), plus a measured tileset+mesh build. Kept as seconds
# rather than percentages so re-measuring is a substitution, not arithmetic, and
# so the numbers stay legible as what they are.
MEASURED_SECONDS = {
    "ingest": 0.26,
    "instances": 48.6,
    "depth": 151.55,
    "segmentation": 0.57,
    "shadow": 0.21,
    "anchors": 0.44,
    "calibration": 1.46,
    "uncertainty": 3.03,
    "assemble": 0.64,
    "artifacts": 4.50,
    "validation": 5.76,
    "tiles": 20.83,
}

# The order a run executes in. `instances` is the structural segmentation and
# runs before depth, which is the architectural change: everything after it can
# be told where one object stops and the next begins. `segmentation` is the
# older five-class colour raster, which the DEM anchor gate still reads and
# which depends on nothing, so it stays where it was.
PIPELINE_PHASES = ("ingest", "instances", "depth", "segmentation", "shadow",
                   "anchors", "calibration", "uncertainty", "assemble",
                   "artifacts", "validation")
JOB_PHASES = PIPELINE_PHASES + ("tiles",)

PENDING, RUNNING, DONE, FAILED, SKIPPED = (
    "pending", "running", "done", "failed", "skipped")
TERMINAL = (DONE, FAILED, SKIPPED)

# What each phase is doing, in words a reader who did not write it can use.
DESCRIPTIONS = {
    "ingest": "Reading the image and its georeferencing",
    "instances": "Finding structural instances with SAM 2",
    "depth": "Running the depth backbone over the image",
    "segmentation": "Classifying ground, road, building, vegetation, water",
    "shadow": "Detecting cast shadow",
    "anchors": "Harvesting elevation anchors and choosing a calibration tier",
    "calibration": "Solving Chhaya/AGMC for the metric surface",
    "uncertainty": "Bootstrapping per-pixel uncertainty",
    "assemble": "Separating terrain from height above ground",
    "artifacts": "Writing the GeoTIFFs and previews",
    "validation": "Scoring against the reference DSM",
    "tiles": "Building the 3D tileset and mesh",
}


class IllegalTransition(RuntimeError):
    """A phase change that would make the reported state a lie."""


@dataclass
class PhaseState:
    name: str
    status: str = PENDING
    progress: float = 0.0                 # 0..1 within this phase
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    message: str = ""
    artifact: Optional[str] = None
    error: Optional[str] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at if self.completed_at is not None else time.time()
        return round(end - self.started_at, 2)

    def public(self) -> dict:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        d["description"] = DESCRIPTIONS.get(self.name, "")
        return d


@dataclass
class JobProgress:
    """Ordered phases, validated transitions, and one weighted overall figure."""

    expected: Sequence[str] = field(default_factory=lambda: list(JOB_PHASES))
    phases: list = field(default_factory=list)
    current: Optional[str] = None

    def __post_init__(self):
        if not self.phases:
            self.phases = [PhaseState(name=n) for n in self.expected]
        self._by_name = {p.name: p for p in self.phases}
        self.expected = [p.name for p in self.phases]

    # ------------------------------------------------------------- lookups
    def __getitem__(self, name: str) -> PhaseState:
        try:
            return self._by_name[name]
        except KeyError:
            raise IllegalTransition(
                f"'{name}' is not a phase of this job. Expected one of: "
                f"{', '.join(self.expected)}") from None

    def index(self, name: str) -> int:
        return self.expected.index(self[name].name)

    @property
    def failed(self) -> Optional[PhaseState]:
        return next((p for p in self.phases if p.status == FAILED), None)

    @property
    def finished(self) -> bool:
        """Every phase has reached a terminal state."""
        return all(p.status in TERMINAL for p in self.phases)

    # --------------------------------------------------------- transitions
    def begin(self, name: str, message: str = "") -> PhaseState:
        p = self[name]
        if self.failed is not None:
            raise IllegalTransition(
                f"cannot start '{name}': '{self.failed.name}' already failed")
        if p.status in (DONE, FAILED):
            raise IllegalTransition(
                f"cannot restart '{name}', it is already {p.status}")
        # The rule that makes the bar monotonic, and the one the plan names
        # explicitly: a completed job must never report itself back in `depth`.
        if self.current is not None and self.index(name) < self.index(self.current):
            raise IllegalTransition(
                f"cannot go back to '{name}' from '{self.current}'")
        p.status = RUNNING
        p.progress = 0.0
        p.started_at = p.started_at or time.time()
        p.message = message or DESCRIPTIONS.get(name, "")
        self.current = name
        return p

    def advance(self, name: str, fraction: float, message: str = "") -> PhaseState:
        """Report a fraction of one phase. Never moves backwards within it."""
        p = self[name]
        if p.status != RUNNING:
            raise IllegalTransition(
                f"cannot report progress on '{name}', it is {p.status}")
        f = 0.0 if fraction != fraction else float(fraction)   # NaN -> 0
        p.progress = max(p.progress, min(max(f, 0.0), 1.0))
        if message:
            p.message = message
        return p

    def complete(self, name: str, message: str = "",
                 artifact: Optional[str] = None) -> PhaseState:
        p = self[name]
        if p.status not in (RUNNING, PENDING):
            raise IllegalTransition(f"cannot complete '{name}', it is {p.status}")
        p.status = DONE
        p.progress = 1.0
        p.started_at = p.started_at or time.time()
        p.completed_at = time.time()
        if message:
            p.message = message
        if artifact:
            p.artifact = artifact
        return p

    def skip(self, name: str, why: str = "") -> PhaseState:
        """Mark a phase as one this run will not perform, and take it out of the total."""
        p = self[name]
        if p.status in (DONE, FAILED):
            raise IllegalTransition(f"cannot skip '{name}', it is {p.status}")
        p.status = SKIPPED
        p.progress = 0.0
        p.message = why or "not applicable to this run"
        return p

    def fail(self, name: str, error: str) -> PhaseState:
        """Fail a phase by name, or record a failure outside any phase."""
        p = self._by_name.get(name)
        if p is None:                       # e.g. the tileset builder blew up
            p = self[self.current] if self.current else self.phases[0]
        p.status = FAILED
        p.error = str(error)
        p.completed_at = time.time()
        p.started_at = p.started_at or p.completed_at
        self.current = p.name
        return p

    # ------------------------------------------------------------- reading
    def weight(self, name: str) -> float:
        # An unknown phase gets the median of the known ones rather than zero:
        # a new stage that contributes nothing to the bar is invisible until
        # someone notices the bar jumping over it.
        return MEASURED_SECONDS.get(name, 1.0)

    def overall(self) -> float:
        """Weighted completion in 0..1, over the phases this run will actually do."""
        live = [p for p in self.phases if p.status != SKIPPED]
        total = sum(self.weight(p.name) for p in live)
        if total <= 0:
            return 0.0
        acc = 0.0
        for p in live:
            if p.status == DONE:
                acc += self.weight(p.name)
            elif p.status == RUNNING:
                acc += self.weight(p.name) * p.progress
        return max(0.0, min(acc / total, 1.0))

    def public(self) -> dict:
        cur = self._by_name.get(self.current) if self.current else None
        return {
            "phase": self.current,
            "phase_status": cur.status if cur else PENDING,
            "phase_progress": round(cur.progress, 4) if cur else 0.0,
            "message": cur.message if cur else "",
            "progress": round(self.overall(), 4),
            "phases": [p.public() for p in self.phases],
        }

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"version": 1, "current": self.current,
                "phases": [asdict(p) for p in self.phases]}

    @classmethod
    def from_dict(cls, d: dict) -> "JobProgress":
        phases = [PhaseState(**p) for p in d.get("phases", [])]
        obj = cls(expected=[p.name for p in phases], phases=phases)
        obj.current = d.get("current")
        return obj

    def interrupted(self, why: str = "the server restarted while this job was running"):
        """Turn a phase left `running` by a crash into an honest failure.

        A job whose process died still says `running` on disk, and a job that
        reports itself running forever is indistinguishable from a hung one. On
        reload it is failed instead, naming the phase it died in.
        """
        for p in self.phases:
            if p.status == RUNNING:
                p.status = FAILED
                p.error = why
                p.completed_at = time.time()
        return self
