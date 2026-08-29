"""The site is a renderer for study.json, so its shape is a contract.

web/results.js reads specific keys. If study.py stops emitting one of them the page
degrades silently to em-dashes, which is exactly the failure mode that survives
a demo rehearsal and dies on stage. These tests run a miniature study and assert
the keys the page depends on are present and finite.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

SITE = Path(__file__).resolve().parents[1] / "web"

# Keys web/results.js reads out of the aggregate block.
AGG_KEYS = ("mae_m", "rmse_m", "bias_m", "pearson_r", "spearman_r",
            "coverage_1s", "baseline_mae_m", "baseline_rmse_m", "baseline_pearson_r",
            "by_class_mae_m", "n_scenes", "timings_s")
SCENE_KEYS = ("seed", "metrics", "tier", "anchors", "anchors_used",
              "anchors_rejected", "scene_truth", "wall_s")

# Preview images the explorer requests, from the LAYERS table in results.js.
PREVIEWS = ("rgb.jpg", "dsm_pred.png", "dsm_truth.png", "error.png",
            "sigma.png", "ndsm.png", "shadow.png")


@pytest.fixture(scope="module")
def mini_study(tmp_path_factory):
    """A real study, small enough to run in seconds on the synthetic backbone."""
    from ayama.eval import study as S

    out = str(tmp_path_factory.mktemp("study"))
    scene = S.scene_experiment(out, seed=5, size=256, backbone="synthetic",
                               chip=256, n_bootstrap=4, log=lambda m: None)
    scene_dir = f"{out}/seed5"
    return {
        "out": out,
        "environment": S.environment(),
        "config": {"backbone": "synthetic", "size": 256, "chip": 256,
                   "seeds": [5], "bootstrap": 4, "gsd_m": 0.5},
        "aggregate": S.aggregate([scene]),
        "scenes": [scene],
        "ablation": {"5": S.ablation_experiment(scene_dir, 5, backbone="synthetic",
                                                chip=256, n_bootstrap=4, log=lambda m: None)},
        "sun_sweep": S.sun_sweep(elevations=(25, 45, 65), size=256, log=lambda m: None),
        "lambda_sweep": S.lambda_sweep(scene_dir, lambdas=(0.5, 1.0, 2.0), log=lambda m: None),
        "wall_s": 1.0,
    }


def test_aggregate_carries_every_key_the_page_reads(mini_study):
    agg = mini_study["aggregate"]
    for key in AGG_KEYS:
        assert key in agg, f"aggregate is missing '{key}', which web/results.js renders"
    for key in ("mae_m", "rmse_m", "pearson_r", "baseline_mae_m"):
        assert np.isfinite(agg[key]["mean"]), f"{key} is not finite"
        assert {"mean", "std", } <= set(agg[key]) or "mean" in agg[key]


def test_scene_entries_carry_what_the_explorer_needs(mini_study):
    for scene in mini_study["scenes"]:
        for key in SCENE_KEYS:
            assert key in scene, f"scene is missing '{key}'"
        assert np.isfinite(scene["metrics"]["mae_m"])
        assert scene["scene_truth"]["sun_elevation_deg"] is not None


def test_every_preview_the_explorer_requests_exists(mini_study):
    d = Path(mini_study["out"]) / "seed5" / "preview"
    for name in PREVIEWS:
        p = d / name
        assert p.exists(), f"web/results.js requests preview/{name} and it was not written"
        assert p.stat().st_size > 500, f"{name} is suspiciously small"


def test_layer_list_in_results_js_matches_what_is_written(mini_study):
    """Catch a preview renamed on one side but not the other."""
    js = (SITE / "results.js").read_text(encoding="utf-8")
    requested = set(re.findall(r"id:\s*'([a-z_]+\.(?:png|jpg))'", js))
    assert requested, "could not parse the LAYERS table out of results.js"
    written = {p.name for p in (Path(mini_study["out"]) / "seed5" / "preview").iterdir()}
    assert requested <= written, f"results.js asks for {sorted(requested - written)}, never written"


def test_ablation_rows_have_the_columns_the_table_renders(mini_study):
    rows = mini_study["ablation"]["5"]
    assert rows and "error" not in rows[0]
    for r in rows:
        for key in ("variant", "n_anchors", "mae_m", "rmse_m", "pearson_r"):
            assert key in r, f"ablation row missing '{key}'"
    assert any(r["variant"] == "global_affine" for r in rows), "baseline row missing"


def test_sun_and_lambda_sweeps_are_plottable(mini_study):
    for r in mini_study["sun_sweep"]:
        for key in ("sun_elevation_deg", "f1", "n_anchors", "median_abs_height_error_m"):
            assert key in r
        assert 0.0 <= r["f1"] <= 1.0
    lam = mini_study["lambda_sweep"]
    assert any(r["lam"] is None for r in lam), "no baseline row in the lambda sweep"
    assert any(r["lam"] is not None for r in lam)
    for r in lam:
        assert np.isfinite(r["mae_m"])


def test_study_json_is_serialisable(mini_study, tmp_path):
    """numpy floats must not leak into the JSON the browser fetches."""
    from ayama.eval.study import save_json

    p = str(tmp_path / "study.json")
    save_json({k: v for k, v in mini_study.items() if k != "out"}, p)
    loaded = json.loads(Path(p).read_text(encoding="utf-8"))
    assert loaded["aggregate"]["n_scenes"] == 1


def test_results_json_is_strict_json_a_browser_will_parse(mini_study, tmp_path):
    """Python emits bare NaN; JSON.parse rejects it and the page renders blank.

    Metrics come back non-finite on ordinary runs — delta1 with no valid pixels,
    ECE with too few samples — so this is a normal path, not an edge case. It
    cost a fully blank deployed site once already.
    """
    from ayama.eval.study import save_json

    payload = {k: v for k, v in mini_study.items() if k != "out"}
    payload["deliberately_missing"] = float("nan")
    payload["also_missing"] = float("inf")

    p = tmp_path / "study.json"
    save_json(payload, str(p))
    text = p.read_text(encoding="utf-8")

    assert "NaN" not in text, "bare NaN token in the JSON the browser fetches"
    assert "Infinity" not in text
    loaded = json.loads(text, parse_constant=_reject_constant)
    assert loaded["deliberately_missing"] is None
    assert loaded["also_missing"] is None


def _reject_constant(name):
    raise AssertionError(f"non-finite constant {name!r} survived into the JSON")


def test_every_artifact_json_the_cli_writes_is_strict(tmp_path):
    from ayama.core.jsonio import dumps, save_json

    nested = {"a": [float("nan"), 1.0, {"b": float("-inf")}], "c": np.float32("nan")}
    text = dumps(nested)
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text, parse_constant=_reject_constant)
    assert back["a"][0] is None and back["a"][2]["b"] is None and back["c"] is None

    p = tmp_path / "x.json"
    save_json({"v": np.float64("nan")}, str(p))
    assert json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)["v"] is None
