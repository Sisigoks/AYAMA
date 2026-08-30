"""The structural scale: fitting it, choosing a model, and applying it.

This is the only part of TRAKSHA that carries anything between images, so it is
the only part where a bug can quietly make every future scene wrong in the same
direction. The tests below are mostly about refusals: refusing to prefer a
model the data does not support, refusing to apply a scale fitted under a
different frequency split, and refusing to let the bootstrap throw the scale
away - which is a real bug this suite now pins.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")

from traksha.learn.scale import (ScaleModel, Sample, fit,  # noqa: E402
                               high_band, scene_features, scene_target)


def _sample(name, target, hi_p99, floor=8.0):
    return Sample(name=name, target=target,
                  features={"hi_p99": hi_p99, "hi_std": 0.1,
                            "rd_range": 0.9, "gsd_m": 0.5},
                  ndsm_mae_at_target=5.0, floor_mae=floor)


# ------------------------------------------------------------------- target
def test_the_target_is_the_least_squares_scale():
    """a* minimises ||a*pos - truth||, so a known pair must recover its scale."""
    pos = np.abs(np.random.default_rng(0).normal(size=(64, 64))).astype(np.float32)
    truth = 37.5 * pos
    assert scene_target(pos, truth) == pytest.approx(37.5, rel=1e-6)


def test_the_target_is_finite_on_a_degenerate_band():
    assert not np.isfinite(scene_target(np.zeros((8, 8), np.float32),
                                        np.ones((8, 8), np.float32)))


def test_the_high_band_is_positive_and_suppresses_smooth_terrain():
    """A ramp is all low frequency; the band that scales structure must ignore it."""
    n = 512                                    # 256 m across, well past the 60 m split
    xx = np.mgrid[0:n, 0:n][1].astype(np.float32)
    ramp = (xx / (n - 1.0)).astype(np.float32)
    band = high_band(ramp, gsd_m=0.5, radius_m=60.0)
    assert (band >= 0).all()
    # The filter has edges, so judge the interior: that is where the claim holds.
    interior = band[64:-64, 64:-64]
    assert interior.max() < 0.02, "a pure ramp leaked into the structural band"


def test_features_need_no_ground_truth():
    """A feature that needs the answer cannot be used to predict the answer."""
    rel = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    feats = scene_features(rel, dem_m=None, gsd_m=0.5)
    assert {"hi_p99", "hi_std", "rd_range", "gsd_m"} <= set(feats)
    assert all(np.isfinite(v) for v in feats.values())
    assert "dem_relief_m" not in feats          # no DEM was supplied
    assert "dem_relief_m" in scene_features(rel, np.full((64, 64), 400.0), 0.5)


# -------------------------------------------------------------------- fitting
def test_a_constant_is_chosen_when_a_feature_model_cannot_earn_its_parameters():
    """Four scenes, no real relationship. The extra parameter must lose."""
    rng = np.random.default_rng(3)
    samples = [_sample(f"s{i}", 100.0 + rng.normal(0, 8), 0.3 + rng.normal(0, 0.02))
               for i in range(4)]
    model = fit(samples)
    assert model.kind == "constant"
    assert model.value == pytest.approx(np.mean([s.target for s in samples]))
    assert "constant won" in model.notes or "constant" in model.notes


def test_a_feature_model_is_chosen_when_the_relationship_is_real():
    """The selection must be able to go the other way, or it is not a selection."""
    samples = []
    for i, x in enumerate((0.10, 0.20, 0.30, 0.40, 0.50, 0.60)):
        samples.append(_sample(f"s{i}", 12.0 / x + 5.0, x))
    model = fit(samples)
    assert model.kind == "linear", "an exact 1/x relationship over six scenes should win"
    assert model.predict({"hi_p99": 0.25}) == pytest.approx(12.0 / 0.25 + 5.0, rel=0.02)


def test_fitting_refuses_an_empty_set():
    with pytest.raises(ValueError, match="no usable scenes"):
        fit([Sample(name="x", target=float("nan"))])


def test_leave_one_out_is_scored_in_metres_when_rasters_are_supplied():
    """Scale error and nDSM error rank models differently; only the second matters."""
    rng = np.random.default_rng(5)
    samples, rasters = [], {}
    for i in range(4):
        pos = np.abs(rng.normal(size=(32, 32))).astype(np.float32)
        a = 100.0 + i * 5.0
        truth = a * pos
        samples.append(_sample(f"s{i}", a, 0.3))
        rasters[f"s{i}"] = (pos, truth)
    model = fit(samples, rasters=rasters)
    assert np.isfinite(model.loo_mae_m)
    assert model.loo_mae_m > 0, "held-out error against a spread of scales cannot be zero"


# ------------------------------------------------------------------ model io
def test_a_model_round_trips_through_disk(tmp_path):
    m = ScaleModel(kind="constant", value=103.9, radius_m=60.0, n_scenes=4,
                   loo_mae_m=4.83, floor_mae_m=7.59, scenes=["a", "b"])
    p = str(tmp_path / "calibration.json")
    m.save(p)
    back = ScaleModel.load(p)
    assert back.value == pytest.approx(103.9)
    assert back.radius_m == 60.0 and back.n_scenes == 4
    assert back.improvement_over_floor == pytest.approx(1 - 4.83 / 7.59)
    assert "103.9" in back.describe()


def test_an_unknown_field_in_the_file_does_not_break_loading(tmp_path):
    """Forward compatibility: a newer writer must not brick an older reader."""
    import json

    p = str(tmp_path / "c.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"traksha_scale_model_version": 99, "kind": "constant",
                   "value": 50.0, "something_new": [1, 2, 3]}, fh)
    assert ScaleModel.load(p).value == 50.0


def test_the_bundled_model_is_present_and_sane():
    """It ships, so it is tested. A wrong constant makes every scene wrong."""
    from traksha.learn.scale import load_bundled

    m = load_bundled()
    assert m is not None, "no calibration.json ships - run `traksha fit`"
    assert m.kind in ("constant", "linear")
    assert 10.0 < m.predict({"hi_p99": 0.3}) < 1000.0
    assert m.n_scenes >= 1
    # It must beat the floor it was measured against, or it should not ship.
    assert m.improvement_over_floor > 0.1, m.describe()


# ------------------------------------------------------- applying it
def test_the_scale_prior_is_held_when_no_object_anchor_can_argue_with_it():
    """The heart of it: with zero object anchors the fitted scale must survive.

    Solving for `a` from terrain anchors alone is what flattens the surface
    (README §4.2). When a scale is supplied and nothing in the data speaks to
    it, it has to come out the other side unchanged.
    """
    from traksha.chhaya.agmc import solve_agmc
    from traksha.core.types import Anchor, Config, DepthField, SceneMeta, Tier

    rng = np.random.default_rng(7)
    rel = rng.random((96, 96)).astype(np.float32)
    meta = SceneMeta(gsd_m=0.5)
    depth = DepthField(relative=rel, meta=meta, backbone="test-fixture")
    anchors = [Anchor(row=int(r), col=int(c), value_m=400.0, branch="terrain", source="dem", weight=1.0)
               for r, c in rng.integers(0, 96, size=(60, 2))]

    cfg = Config(extras={"dual_branch": True})
    cal = solve_agmc(depth, anchors, cfg, tier=Tier.A, scale_prior=77.0)
    assert cal.scale_source == "fitted"
    assert np.allclose(cal.a, 77.0), "the supplied scale was solved away"


def test_without_a_prior_the_scale_is_still_solved_from_anchors():
    from traksha.chhaya.agmc import solve_agmc
    from traksha.core.types import Anchor, Config, DepthField, SceneMeta, Tier

    rng = np.random.default_rng(8)
    rel = rng.random((96, 96)).astype(np.float32)
    depth = DepthField(relative=rel, meta=SceneMeta(gsd_m=0.5), backbone="test-fixture")
    anchors = [Anchor(row=int(r), col=int(c), value_m=400.0, branch="terrain", source="dem", weight=1.0)
               for r, c in rng.integers(0, 96, size=(60, 2))]
    cal = solve_agmc(depth, anchors, Config(), tier=Tier.A)
    assert cal.scale_source == "anchors"


def test_the_bootstrap_carries_the_scale_rather_than_discarding_it():
    """The bug this argument exists to prevent.

    `bootstrap_sigma` returns a mean surface that REPLACES the delivered
    calibration's. Before it took a scale_prior it silently re-solved without
    one, so enabling the bootstrap - the default - threw the fitted scale away
    and shipped a flattened surface with a sigma computed for a different one.
    """
    from traksha.chhaya.uncertainty import bootstrap_sigma
    from traksha.core.types import Anchor, Config, DepthField, SceneMeta

    rng = np.random.default_rng(9)
    rel = rng.random((64, 64)).astype(np.float32)
    depth = DepthField(relative=rel, meta=SceneMeta(gsd_m=0.5), backbone="test-fixture")
    anchors = [Anchor(row=int(r), col=int(c), value_m=400.0, branch="terrain", source="dem", weight=1.0)
               for r, c in rng.integers(0, 64, size=(40, 2))]
    cfg = Config(extras={"dual_branch": True})

    lo, _ = bootstrap_sigma(depth, anchors, cfg, n_boot=4, scale_prior=5.0)
    hi, _ = bootstrap_sigma(depth, anchors, cfg, n_boot=4, scale_prior=200.0)
    spread_lo = float(lo.max() - lo.min())
    spread_hi = float(hi.max() - hi.min())
    assert spread_hi > spread_lo * 5, (
        "the bootstrapped surface ignored the scale prior "
        f"({spread_lo:.3f} vs {spread_hi:.3f} m of relief)")


def test_a_model_fitted_under_a_different_split_is_not_applied(tmp_path):
    """A scale in metres per unit of high-band depth only means anything under
    the split that produced it. Applying it to another is a units error."""
    from traksha.api.pipeline import _fitted_scale
    from traksha.core.types import Config, DepthField, SceneMeta

    p = str(tmp_path / "c.json")
    ScaleModel(kind="constant", value=90.0, radius_m=30.0).save(p)
    depth = DepthField(relative=np.random.default_rng(2).random((32, 32)).astype(np.float32),
                       meta=SceneMeta(gsd_m=0.5), backbone="test-fixture")

    matched = Config(extras={"scale_model": p, "hp_radius_m": 30.0})
    assert _fitted_scale(matched, depth, None) == pytest.approx(90.0)

    mismatched = Config(extras={"scale_model": p, "hp_radius_m": 60.0})
    assert _fitted_scale(mismatched, depth, None) is None


def test_the_scale_model_can_be_switched_off_and_overridden():
    from traksha.api.pipeline import _fitted_scale
    from traksha.core.types import Config, DepthField, SceneMeta

    depth = DepthField(relative=np.random.default_rng(4).random((32, 32)).astype(np.float32),
                       meta=SceneMeta(gsd_m=0.5), backbone="test-fixture")
    assert _fitted_scale(Config(extras={"scale_model": "off"}), depth, None) is None
    assert _fitted_scale(Config(extras={"scale_model": 42.0}), depth, None) == 42.0


# ------------------------------------------------- image-conditioned refinement
def test_the_refinement_cannot_move_the_calibrated_datum():
    """The one guarantee that makes refining a calibrated surface safe.

    A refinement is allowed to move height WITHIN a neighbourhood - sharpening a
    roof edge - and forbidden from moving the neighbourhood itself. If it can
    shift the local mean it has rewritten the calibration, which is the one
    thing the anchors were for.
    """
    from traksha.mesh.refine import refine_heights

    rng = np.random.default_rng(11)
    z = np.zeros((256, 256), np.float32)
    z[80:170, 80:170] = 20.0                        # a building
    z += rng.normal(0, 0.2, z.shape).astype(np.float32)
    rgb = np.stack([np.clip(z * 8, 0, 255)] * 3, -1).astype(np.uint8)

    out, dz = refine_heights(z, rgb, gsd_m=0.5, preserve_scale_m=30.0)
    assert out.shape == z.shape

    # The mean over the preserved scale must survive. Compare block means well
    # inside the field, where the box filter is not fighting the border.
    b = 60
    for r in range(b, 256 - b, 60):
        for c in range(b, 256 - b, 60):
            before = float(z[r:r + 60, c:c + 60].mean())
            after = float(out[r:r + 60, c:c + 60].mean())
            assert abs(after - before) < 0.5, (
                f"block ({r},{c}) mean moved {after - before:+.2f} m; "
                "the refinement rewrote the datum")


def test_the_refinement_is_bounded_and_keeps_ground_at_ground():
    from traksha.mesh.refine import refine_heights

    rng = np.random.default_rng(12)
    z = np.abs(rng.normal(0, 3, (128, 128))).astype(np.float32)
    rgb = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    out, dz = refine_heights(z, rgb, gsd_m=0.5, clamp_m=1.0)
    assert np.abs(dz).max() <= 1.0 + 1e-4, "the residual clamp was not applied"
    assert (out >= 0).all(), "height above ground went negative"


def test_the_guided_filter_keeps_an_edge_the_guide_has():
    """If it cannot preserve a step in the guide it is just a blur."""
    from traksha.mesh.refine import guided_filter

    step = np.zeros((64, 64), np.float32)
    step[:, 32:] = 1.0
    noisy = step + np.random.default_rng(13).normal(0, 0.05, step.shape).astype(np.float32)
    out = guided_filter(noisy, step, radius=4, eps=1e-6)

    # noise down inside the flat halves...
    assert out[:, :28].std() < noisy[:, :28].std()
    # ...and the step still a step
    assert out[:, 33:].mean() - out[:, :31].mean() > 0.9
