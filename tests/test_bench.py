"""Throughput harness: a dead case must be a row, not the end of the sweep.

The bench documents that an out-of-memory result is recorded rather than
killing the sweep. That was only true for Python exceptions. Torch segfaults
outright (exit 139) when a second model is loaded into a warm process, which
takes the interpreter with it and, on one run, destroyed seven minutes of
completed work at the very last stage. Each case now runs in a child process.
"""
from __future__ import annotations


import pytest

from ayama.eval.bench import device_report, format_bench, sweep

# The weightless placeholder backbone has been removed along with everything
# else that produced invented pixels, so every case here loads real weights.
# That makes the sweep tests slow rather than instant - the isolation behaviour
# they check is only meaningful with a model that can actually die.
pytestmark = pytest.mark.slow
BACKBONE = "dav2-vits"


def test_device_report_describes_the_machine():
    rep = device_report()
    assert "platform" in rep and "python" in rep
    assert "cpu_count" in rep
    # torch may be absent in a bare install; the report must still be renderable.
    assert rep.get("torch") is None or isinstance(rep["torch"], str)


def test_sweep_runs_a_backbone_end_to_end():
    rep = sweep(size=256, backbones=[BACKBONE], chips=[256], batches=[1])
    assert len(rep["results"]) == 1
    r = rep["results"][0]
    assert "error" not in r, r.get("error")
    assert r["n_chips"] >= 1
    assert r["wall_s"] > 0
    assert r["s_per_chip"] > 0
    assert "environment" in rep


def test_a_failing_case_is_recorded_and_the_sweep_continues():
    """One bad cell must not cost the cells around it."""
    rep = sweep(size=256, backbones=[BACKBONE, "no-such-backbone", BACKBONE],
                chips=[256], batches=[1])
    assert len(rep["results"]) == 3
    assert "error" not in rep["results"][0]
    assert "error" in rep["results"][1], "a bogus backbone should be recorded as an error"
    assert "error" not in rep["results"][2], "the sweep stopped after a failing case"


def test_isolated_and_inline_paths_agree():
    """Isolation is a scheduling decision; it must not change the measurement."""
    iso = sweep(size=256, backbones=[BACKBONE], chips=[256], batches=[1], isolate=True)
    inline = sweep(size=256, backbones=[BACKBONE], chips=[256], batches=[1], isolate=False)
    a, b = iso["results"][0], inline["results"][0]
    assert a["n_chips"] == b["n_chips"]
    assert a["image_px"] == b["image_px"]
    assert a["backbone"] == b["backbone"]


def test_format_bench_renders_errors_without_crashing():
    rep = {"environment": device_report(), "source": "test",
           "results": [{"backbone": "x", "chip": 512, "batch_size": 8,
                        "error": "BrokenProcessPool: process died (segfault or OOM kill)"}]}
    text = format_bench(rep)
    assert "BrokenProcessPool" in text or "process died" in text


def test_repeated_model_loads_survive_in_one_sweep():
    """The exact shape of the crash: several cases, each loading a model.

    Four cases, each loading a model into the same sweep. This is the exact
    shape that used to take the interpreter down at the last stage.
    """
    rep = sweep(size=256, backbones=[BACKBONE], chips=[128, 256], batches=[1, 2])
    assert len(rep["results"]) == 4
    assert all("error" not in r for r in rep["results"]), \
        [r.get("error") for r in rep["results"]]
