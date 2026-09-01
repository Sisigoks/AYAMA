"""The job phase state machine.

These exist because the progress screen was broken in a way no test could have
caught: there was no phase state to test. The state machine is now the thing the
UI renders, so what it refuses matters as much as what it records - a bar that
goes backwards, or that reports a finished job back in `depth`, is worse than no
bar, because it teaches the reader the number is invented.
"""
from __future__ import annotations

import math

import pytest

from traksha.api.phases import (JOB_PHASES, MEASURED_SECONDS, IllegalTransition,
                                JobProgress)


def run_through(p: JobProgress, phases=None):
    """Drive a job cleanly from first phase to last."""
    for name in (phases or [q.name for q in p.phases if q.status != "skipped"]):
        p.begin(name)
        p.advance(name, 0.5)
        p.complete(name)
    return p


# ------------------------------------------------------------ happy path
def test_a_clean_run_visits_every_phase_and_finishes_at_one():
    p = JobProgress()
    assert p.overall() == 0.0
    run_through(p)
    assert p.finished
    assert p.overall() == 1.0
    assert all(q.status == "done" for q in p.phases)
    assert all(q.duration_s is not None for q in p.phases)


def test_every_declared_phase_has_a_measured_weight():
    """A phase with no weight is invisible: the bar jumps over it."""
    for name in JOB_PHASES:
        assert name in MEASURED_SECONDS, f"{name} has no measured duration"


def test_overall_is_weighted_by_measured_duration_not_by_count():
    """Depth dominates a run. Equal weights would make the bar a lie.

    The expected figures are derived from the weight table rather than written
    down, so adding a phase changes what this asserts instead of breaking it -
    and the property under test, that the bar tracks time and not phase count,
    is the thing that survives.
    """
    # Facade refinement is skipped here, which is what a run without a GPU
    # does. With it, it is by far the largest phase and depth stops dominating -
    # correctly, and the skipped-phase test covers that case.
    p = JobProgress()
    p.skip("facades", "no GPU")
    total = sum(MEASURED_SECONDS[n] for n in JOB_PHASES if n != "facades")
    p.begin("ingest")
    p.complete("ingest")
    after_ingest = p.overall()
    p.begin("depth")
    p.complete("depth")
    after_depth = p.overall()

    assert after_ingest == pytest.approx(MEASURED_SECONDS["ingest"] / total, abs=1e-6)
    assert after_ingest < 0.01, f"ingest alone should be a sliver, got {after_ingest}"
    assert after_depth > 0.5, f"depth should dominate, got {after_depth}"
    # and the count-based figure is nowhere near it
    assert abs(after_depth - 2 / len(JOB_PHASES)) > 0.3


def test_progress_within_a_phase_moves_the_overall_figure():
    """Depth runs for minutes; without sub-progress the bar would stall there."""
    p = JobProgress()
    p.skip("facades", "no GPU")
    total = sum(MEASURED_SECONDS[n] for n in JOB_PHASES if n != "facades")
    p.begin("ingest")
    p.complete("ingest")
    p.begin("depth")
    at_zero = p.overall()
    p.advance("depth", 0.5)
    at_half = p.overall()
    assert at_half - at_zero == pytest.approx(
        0.5 * MEASURED_SECONDS["depth"] / total, abs=1e-6)


def test_a_skipped_phase_leaves_the_denominator():
    """An upload has no reference DSM, so validation never runs.

    Left in the total, every successful upload would stop short of 100%, and a
    bar that never completes on success is a bug report waiting to be filed.
    """
    p = JobProgress()
    p.skip("validation", "no reference DSM")
    run_through(p)
    assert p.overall() == 1.0
    assert p["validation"].status == "skipped"


# ------------------------------------------------------- refused transitions
def test_a_finished_job_cannot_report_itself_back_in_an_earlier_phase():
    """The transition the plan names explicitly: COMPLETED -> DEPTH."""
    p = JobProgress()
    run_through(p)
    with pytest.raises(IllegalTransition):
        p.begin("depth")


def test_a_completed_phase_cannot_restart():
    p = JobProgress()
    p.begin("ingest")
    p.complete("ingest")
    with pytest.raises(IllegalTransition):
        p.begin("ingest")


def test_nothing_starts_after_a_failure():
    p = JobProgress()
    p.begin("ingest")
    p.fail("ingest", "unreadable raster")
    with pytest.raises(IllegalTransition):
        p.begin("depth")
    assert p.failed.name == "ingest"


def test_progress_cannot_be_reported_on_a_phase_that_is_not_running():
    p = JobProgress()
    with pytest.raises(IllegalTransition):
        p.advance("depth", 0.5)


def test_an_unknown_phase_is_refused_by_name():
    p = JobProgress()
    with pytest.raises(IllegalTransition):
        p.begin("hallucinate")


def test_a_failure_outside_any_phase_lands_on_the_running_one():
    """The tileset builder raises outside the pipeline's stage blocks."""
    p = JobProgress()
    p.begin("ingest")
    p.complete("ingest")
    p.begin("depth")
    p.fail("error", "MemoryError: out of memory")
    assert p["depth"].status == "failed"
    assert "MemoryError" in p["depth"].error


# --------------------------------------------------------------- monotonic
def test_progress_within_a_phase_never_goes_backwards():
    p = JobProgress()
    p.begin("depth")
    p.advance("depth", 0.8)
    p.advance("depth", 0.2)
    assert p["depth"].progress == pytest.approx(0.8)


def test_out_of_range_and_nan_fractions_are_absorbed():
    """`done/total` is arithmetic on numbers the pipeline computed, not a promise."""
    p = JobProgress()
    p.begin("depth")
    p.advance("depth", float("nan"))
    assert p["depth"].progress == 0.0
    p.advance("depth", 5.0)
    assert p["depth"].progress == 1.0
    assert 0.0 <= p.overall() <= 1.0
    assert math.isfinite(p.overall())


# ------------------------------------------------------------- persistence
def test_state_survives_a_round_trip_through_json():
    import json

    p = JobProgress()
    p.skip("validation", "no reference")
    p.begin("ingest")
    p.complete("ingest", "512 x 512")
    p.begin("depth")
    p.advance("depth", 0.4, "3/8 chips")

    back = JobProgress.from_dict(json.loads(json.dumps(p.to_dict())))
    assert back.current == "depth"
    assert back["depth"].progress == pytest.approx(0.4)
    assert back["ingest"].message == "512 x 512"
    assert back["validation"].status == "skipped"
    assert back.overall() == pytest.approx(p.overall())


def test_a_job_interrupted_by_a_restart_reports_a_failure_not_a_run():
    """A job that says `running` forever is indistinguishable from a hang."""
    p = JobProgress()
    p.begin("ingest")
    p.complete("ingest")
    p.begin("depth")
    p.interrupted()
    assert p["depth"].status == "failed"
    assert "restart" in p["depth"].error
    assert p.failed.name == "depth"


def test_public_carries_what_the_screen_draws():
    p = JobProgress()
    p.begin("depth", "loading weights")
    p.advance("depth", 0.25)
    pub = p.public()
    for key in ("phase", "phase_status", "phase_progress", "message",
                "progress", "phases"):
        assert key in pub, key
    assert pub["phase"] == "depth"
    assert pub["phase_progress"] == pytest.approx(0.25)
    assert pub["phases"][0]["description"], "a phase with no description is a code word"
