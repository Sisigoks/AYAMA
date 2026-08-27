"""Live progress and the presentation figures.

Both are things a person looks at rather than a number a test can assert, so
these check the properties that actually break: that the quiet mode is quiet,
that a log-mode run does not emit thousands of carriage returns, that ETA and
rate exist once there is enough data to compute them, and that a figure which
cannot be drawn is skipped instead of taking the study down with it.
"""
from __future__ import annotations

import io
import os
import time

import pytest

from ayama.core.progress import Live, _fmt_dur, gpu_stats


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


def test_gpu_stats_is_none_without_cuda_and_never_raises():
    stats = gpu_stats()
    assert stats is None or {"name", "used_gb", "total_gb"} <= set(stats)


def test_banner_names_the_device():
    assert isinstance(Live(mode="none").banner(), str)


# ── figures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def small_study():
    """Enough of a study to draw every figure that does not need rasters."""
    return {
        "environment": {"platform": "test", "timestamp_utc": "2026-01-01T00:00:00Z"},
        "config": {"backbone": "synthetic", "size": 256, "chip": 256,
                   "seeds": [1], "gsd_m": 0.5},
        "aggregate": {
            "n_scenes": 2,
            "mae_m": {"mean": 3.3, "std": 0.1}, "rmse_m": {"mean": 5.5, "std": 0.2},
            "bias_m": {"mean": -0.6, "std": 0.1}, "median_ae_m": {"mean": 1.9, "std": 0.1},
            "pearson_r": {"mean": 0.71, "std": 0.09},
            "spearman_r": {"mean": 0.80, "std": 0.06},
            "coverage_1s": {"mean": 0.67, "std": 0.02}, "ece_m": {"mean": 2.4, "std": 0.1},
            "edge_f1": {"mean": 0.26, "std": 0.01},
            "slope_mae_deg": {"mean": 7.1, "std": 0.1},
            "baseline_mae_m": {"mean": 5.5, "std": 0.9},
            "baseline_rmse_m": {"mean": 7.8, "std": 0.8},
            "baseline_pearson_r": {"mean": 0.16, "std": 0.15},
            "dem_mae_m": {"mean": 3.5, "std": 0.05},
            "dem_rmse_m": {"mean": 5.4, "std": 0.1},
            "dem_pearson_r": {"mean": 0.71, "std": 0.09},
            "by_class_mae_m": {"road": {"mean": 1.8, "std": 0.1},
                               "building": {"mean": 12.9, "std": 0.2}},
            "timings_s": {"depth": 26.6},
        },
        "scenes": [{"seed": 1, "metrics": {"mae_m": 3.3, "rmse_m": 5.5, "coverage_1s": 0.67}}],
        "ablation": {"1": [
            {"variant": "dem_only", "n_anchors": 4021, "mae_m": 3.5, "rmse_m": 5.4,
             "pearson_r": 0.708},
            {"variant": "global_affine", "n_anchors": 4021, "mae_m": 5.5, "rmse_m": 7.8,
             "pearson_r": 0.162},
            {"variant": "agmc", "n_anchors": 4021, "mae_m": 3.3, "rmse_m": 5.5,
             "pearson_r": 0.711},
        ]},
        "sun_sweep": [
            {"sun_elevation_deg": 15, "f1": 0.29, "n_anchors": 0,
             "median_abs_height_error_m": None, "mean_anchor_weight": None},
            {"sun_elevation_deg": 50, "f1": 0.81, "n_anchors": 57,
             "median_abs_height_error_m": 1.68, "mean_anchor_weight": 0.80},
            {"sun_elevation_deg": 80, "f1": 0.38, "n_anchors": 0,
             "median_abs_height_error_m": None, "mean_anchor_weight": None},
        ],
        "lambda_sweep": [
            {"lam": None, "variant": "global_affine", "mae_m": 5.5, "pearson_r": 0.16},
            {"lam": 0.1, "variant": "agmc", "mae_m": 3.30, "pearson_r": 0.71},
            {"lam": 1.0, "variant": "agmc", "mae_m": 3.40, "pearson_r": 0.70},
            {"lam": 10.0, "variant": "agmc", "mae_m": 4.69, "pearson_r": 0.55},
        ],
        "bench": {"environment": {}, "source": "test", "results": []},
        "wall_s": 12.0,
    }


def test_figures_render_png_and_vector_pdf(small_study, tmp_path):
    from ayama.eval.figures import render_all

    out = str(tmp_path / "figures")
    written = render_all(small_study, out)
    pngs = [w for w in written if w.endswith(".png")]
    pdfs = [w for w in written if w.endswith(".pdf")]
    assert len(pngs) >= 4, "expected at least the four raster-free figures"
    assert len(pngs) == len(pdfs), "every figure must ship a vector version"
    for w in written:
        # A LaTeX table is a few hundred bytes; an empty PNG is not.
        floor = 200 if w.endswith(".tex") else 5000
        assert os.path.getsize(w) > floor, f"{w} is suspiciously small"


def test_figures_needing_rasters_are_skipped_not_fatal(small_study, tmp_path):
    """No completed run on disk means no reliability or qualitative panel."""
    from ayama.eval.figures import fig_qualitative, fig_reliability

    assert fig_reliability(small_study, str(tmp_path), scenes_dir=str(tmp_path)) == []
    assert fig_qualitative(small_study, str(tmp_path), scenes_dir=str(tmp_path)) == []


def test_latex_tables_are_wellformed(small_study, tmp_path):
    from ayama.eval.figures import write_tables

    written = write_tables(small_study, str(tmp_path))
    assert len(written) == 3
    for p in written:
        text = open(p, encoding="utf-8").read()
        assert text.count(r"\begin{tabular}") == 1
        assert text.count(r"\end{tabular}") == 1
        assert r"\toprule" in text and r"\bottomrule" in text
        # An underscore that reached LaTeX unescaped is a compile error, not a typo.
        body = text.split(r"\midrule")[1]
        assert "_" not in body or r"\_" in body


def test_headline_table_carries_the_floor_column(small_study, tmp_path):
    from ayama.eval.figures import write_tables

    text = open(write_tables(small_study, str(tmp_path))[0], encoding="utf-8").read()
    assert "DEM alone" in text, "the floor must appear in the paper table too"
    assert text.count("&") >= 3


def test_a_broken_figure_does_not_abort_the_rest(small_study, tmp_path, monkeypatch):
    from ayama.eval import figures as F

    def explode(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(F, "fig_ablation", explode)
    monkeypatch.setattr(F, "FIGURES", (("ablation", explode),) + F.FIGURES[1:])
    written = F.render_all(small_study, str(tmp_path / "f"))
    assert any(w.endswith(".tex") for w in written), "the run stopped at the bad figure"


def test_figures_survive_a_study_with_missing_sections(tmp_path):
    from ayama.eval.figures import render_all

    written = render_all({"aggregate": {}}, str(tmp_path / "empty"))
    assert all(w.endswith(".tex") for w in written)
