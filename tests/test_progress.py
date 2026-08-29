"""Live progress reporting.

Progress output is something a person looks at rather than a number a test can
assert, so these check the properties that actually break: that quiet mode is
quiet, that a log-mode run does not emit thousands of carriage returns, that
rate and ETA appear once there is enough data to compute them, and that a task
pops off the stack even when the stage it wraps raises.
"""
from __future__ import annotations

import io
import time

import pytest

from ayama.core.progress import Live, _fmt_dur


# ── progress ────────────────────────────────────────────────────────────────
def test_quiet_mode_writes_nothing():
    buf = io.StringIO()
    live = Live(mode="none", stream=buf)
    with live.task("depth", 4, "chip") as t:
        for _ in range(4):
            t.advance(1)
    live.log("this should not appear")
    assert buf.getvalue() == ""


def test_plain_mode_does_not_spam_carriage_returns():
    """A notebook or CI log gets timestamped lines, never an in-place rewrite."""
    buf = io.StringIO()
    live = Live(mode="plain", stream=buf, plain_interval=0.0)
    with live.task("depth", 50, "chip") as t:
        for _ in range(50):
            t.advance(1)
    out = buf.getvalue()
    assert "\r" not in out, "plain mode emitted a carriage return"
    assert out.count("\n") <= 60
    assert "depth" in out


def test_rich_mode_rewrites_one_line():
    buf = io.StringIO()
    live = Live(mode="rich", stream=buf, min_interval=0.0)
    with live.task("depth", 10, "chip") as t:
        for _ in range(10):
            t.advance(1)
    out = buf.getvalue()
    assert "\r" in out, "rich mode should rewrite in place"
    assert "depth" in out


def test_nested_tasks_show_the_outer_context():
    buf = io.StringIO()
    live = Live(mode="plain", stream=buf, plain_interval=0.0)
    with live.task("study", 3, "scene") as outer:
        outer.advance(1)
        with live.task("depth", 4, "chip") as inner:
            inner.advance(1)
    out = buf.getvalue()
    assert "[study 1/3]" in out, "the inner line lost its outer context"


def test_rate_and_eta_appear_once_measurable():
    live = Live(mode="none")
    with live.task("depth", 100, "chip") as t:
        t.started = time.time() - 2.0        # pretend two seconds have passed
        t.done = 20
        assert t.rate is not None and t.rate > 0
        assert t.eta is not None and t.eta > 0


def test_a_task_pops_even_when_the_stage_raises():
    live = Live(mode="none")
    with pytest.raises(ValueError):
        with live.task("depth", 3, "chip"):
            raise ValueError("boom")
    assert live.stack == [], "a failed stage left its task on the stack"


def test_duration_formatting_is_readable():
    assert _fmt_dur(4) == "4s"
    assert _fmt_dur(125) == "2m05s"
    assert _fmt_dur(3700).endswith("m")
    assert _fmt_dur(float("nan")) == "--"


def test_the_banner_names_the_machine():
    """AYAMA runs on the CPU only, so the banner says so and never probes a card."""
    banner = Live(mode="none").banner()
    assert isinstance(banner, str) and "CPU" in banner
