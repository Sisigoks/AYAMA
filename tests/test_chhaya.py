"""AGMC solver behaviour, on relations we constructed and can check exactly."""
from __future__ import annotations

import numpy as np
import pytest

from ayama.chhaya.agmc import (apply_calibration, global_affine, make_lattice,
                               solve_agmc)
from ayama.chhaya.anchors import harvest_dem, harvest_water
from ayama.core.types import (BARE_GROUND, BUILDING, WATER, Anchor, Config,
                              DepthField, SceneMeta)

META = SceneMeta(crs="EPSG:32644", transform=(0.5, 0, 0, 0, -0.5, 0), gsd_m=0.5)


def _depth(rel):
    return DepthField(relative=rel.astype(np.float32), meta=META)


def _sample(field, stride=8, weight=1.0, branch="terrain"):
    rows, cols = np.mgrid[0:field.shape[0]:stride, 0:field.shape[1]:stride]
    return [Anchor(int(r), int(c), float(field[r, c]), branch, "dem", weight)
            for r, c in zip(rows.ravel(), cols.ravel())]


def test_lattice_bilinear_weights_sum_to_one():
    lat = make_lattice((128, 128), 16)
    idx, wts = lat.weights(np.array([0, 37, 127]), np.array([0, 91, 127]))
    assert np.allclose(wts.sum(axis=1), 1.0)
    assert idx.max() < lat.n


def test_recovers_a_constant_affine_relation():
    rel = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    truth = 40.0 * rel + 300.0
    calib = solve_agmc(_depth(rel), _sample(truth), Config(lattice_stride=32))
    pred = apply_calibration(_depth(rel), calib)
    assert np.abs(pred - truth).mean() < 0.5
    assert calib.residual_rmse < 0.5


def test_spatially_varying_relation_beats_a_global_fit():
    # Scale doubles across the tile: exactly the case a global a, b cannot fit.
    rng = np.random.default_rng(1)
    rel = rng.random((128, 128)).astype(np.float32)
    xx = np.linspace(1.0, 3.0, 128)[None, :] * np.ones((128, 1))
    truth = (20.0 * xx) * rel + 300.0

    anchors = _sample(truth)
    a, b = global_affine(rel, anchors)
    global_mae = np.abs(a * rel + b - truth).mean()

    calib = solve_agmc(_depth(rel), anchors, Config(lattice_stride=16, lam_a=0.01, lam_b=0.01))
    agmc_mae = np.abs(apply_calibration(_depth(rel), calib) - truth).mean()

    # The smoothness prior deliberately stops the fields chasing every anchor,
    # so the win is large but not total. Anything under ~0.7 is the solver
    # doing its job; a regression to parity means it collapsed to a global fit.
    assert agmc_mae < 0.65 * global_mae, f"AGMC {agmc_mae:.2f} vs global {global_mae:.2f}"


def test_outliers_are_rejected_not_averaged_in():
    rng = np.random.default_rng(2)
    rel = rng.random((96, 96)).astype(np.float32)
    truth = 30.0 * rel + 250.0
    anchors = _sample(truth, stride=6)
    for k in anchors[::7]:                       # 14% gross blunders
        k.value_m += 120.0

    cfg = Config(lattice_stride=32, huber_delta=2.0, irls_iters=4)
    calib = solve_agmc(_depth(rel), anchors, cfg)
    pred = apply_calibration(_depth(rel), calib)
    assert np.abs(pred - truth).mean() < 5.0
    assert calib.n_anchors_rejected > 0


def test_relative_anchors_constrain_a_difference_not_a_datum():
    # Only relative constraints: the shape must come out right even though the
    # datum is unknowable from them.
    rel = np.zeros((64, 64), np.float32)
    rel[20:40, 20:40] = 1.0
    anchors = []
    for r in range(22, 38, 4):
        for c in range(22, 38, 4):
            anchors.append(Anchor(r, c, 25.0, "object", "shadow", 1.0, ref_row=5, ref_col=5))
    # One absolute anchor to fix the datum.
    anchors.append(Anchor(5, 5, 100.0, "absolute", "gcp", 1.0))

    calib = solve_agmc(_depth(rel), anchors, Config(lattice_stride=16, lam_a=0.05, lam_b=0.05))
    pred = apply_calibration(_depth(rel), calib)
    height = pred[25:35, 25:35].mean() - pred[0:10, 0:10].mean()
    assert height == pytest.approx(25.0, abs=3.0)


def test_no_anchors_returns_identity_and_says_so():
    rel = np.random.default_rng(3).random((32, 32)).astype(np.float32)
    calib = solve_agmc(_depth(rel), [], Config())
    assert calib.n_anchors_used == 0
    assert np.allclose(calib.a, 1.0) and np.allclose(calib.b, 0.0)
    assert not np.isfinite(calib.residual_rmse)


# ------------------------------------------------------------------ harvesters
def test_dem_anchors_are_gated_by_semantics():
    dem = np.full((64, 64), 400.0, np.float32)
    sem = np.full((64, 64), BARE_GROUND, np.uint8)
    sem[:32] = BUILDING                       # rooftops: DEM is wrong there
    anchors = harvest_dem(dem, sem, stride=4)
    assert anchors, "no anchors harvested from open ground"
    assert all(k.row >= 32 for k in anchors), "a DEM sample was taken on a rooftop"


def test_water_anchors_share_one_level():
    sem = np.full((64, 64), BARE_GROUND, np.uint8)
    sem[20:44, 20:44] = WATER
    dem = np.random.default_rng(4).normal(400.0, 3.0, (64, 64)).astype(np.float32)
    anchors = harvest_water(sem, dem_m=dem, stride=4)
    assert len(anchors) > 4
    assert len({round(k.value_m, 6) for k in anchors}) == 1, "water body is not level"


def test_water_without_a_dem_falls_back_to_relative_constraints():
    sem = np.full((64, 64), BARE_GROUND, np.uint8)
    sem[20:44, 20:44] = WATER
    anchors = harvest_water(sem, dem_m=None, stride=4)
    assert anchors and all(k.is_relative for k in anchors)
    assert all(k.value_m == 0.0 for k in anchors)


def test_dem_weight_reflects_datasheet_accuracy():
    from ayama.chhaya.anchors import dem_weight

    assert dem_weight("copernicus") > dem_weight("srtm") > dem_weight("aster")


def test_relative_anchors_stay_out_of_the_global_fit():
    """A "level with itself" water anchor must not drag the datum to zero.

    global_affine feeds the prior that AGMC is pulled toward, and it is also the
    published baseline every result is compared against. Letting a relative
    anchor in as an elevation moved the fitted scale from 30 to 472 in the case
    below, which would have made the baseline a straw man.
    """
    rel = np.linspace(0, 1, 100).reshape(1, 100).astype(np.float32)
    truth = 30.0 * rel + 400.0
    absolute = [Anchor(0, i, float(truth[0, i]), "terrain", "dem", 1.0)
                for i in range(0, 100, 5)]
    relative = [Anchor(0, i, 0.0, "terrain", "water", 0.9, ref_row=0, ref_col=10)
                for i in range(20, 40)]

    a_clean, b_clean = global_affine(rel, absolute)
    a_mixed, b_mixed = global_affine(rel, absolute + relative)
    assert a_clean == pytest.approx(30.0, abs=0.5)
    assert a_mixed == pytest.approx(a_clean, rel=0.05)
    assert b_mixed == pytest.approx(b_clean, abs=1.0)


def test_object_branch_anchors_also_stay_out_of_the_global_fit():
    rel = np.linspace(0, 1, 64).reshape(1, 64).astype(np.float32)
    truth = 20.0 * rel + 100.0
    absolute = [Anchor(0, i, float(truth[0, i]), "terrain", "dem", 1.0) for i in range(0, 64, 4)]
    shadow = [Anchor(0, i, 35.0, "object", "shadow", 1.0) for i in range(10, 30)]
    assert global_affine(rel, absolute) == pytest.approx(global_affine(rel, absolute + shadow),
                                                         rel=0.05)


def test_scale_field_stays_positive_so_structures_are_not_inverted():
    """The trap: terrain anchors that anti-correlate with the depth prior.

    Depth Anything V2 puts a ground-level perspective ramp on nadir imagery. If
    that ramp opposes the real terrain, and the anchors are almost all terrain
    samples (they are — the DEM supplies thousands, shadows supply dozens), an
    unconstrained fit picks a NEGATIVE scale: terrain matches beautifully and
    every building is rendered as a pit. Overall MAE still improves, so no
    aggregate metric catches it.
    """
    h = w = 96
    rng = np.random.default_rng(11)

    # A depth field that ranks correctly: buildings high, plus a spurious ramp
    # running the opposite way to the terrain.
    ramp = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    rel = 0.6 * ramp + 0.02 * rng.random((h, w))
    building = np.zeros((h, w), bool)
    building[30:50, 30:50] = True
    rel[building] += 0.35                       # the model gets the ordering right

    # True terrain runs the other way, so the anchors oppose the ramp.
    terrain = 400.0 - 20.0 * ramp
    anchors = [Anchor(int(r), int(c), float(terrain[r, c]), "terrain", "dem", 1.0)
               for r in range(0, h, 6) for c in range(0, w, 6) if not building[r, c]]

    cfg_free = Config(lattice_stride=24, extras={"enforce_positive_scale": False})
    cfg_ok = Config(lattice_stride=24, extras={"enforce_positive_scale": True})

    free = solve_agmc(_depth(rel), anchors, cfg_free)
    assert free.a.mean() < 0, "expected the unconstrained fit to invert; the trap has moved"

    calib = solve_agmc(_depth(rel), anchors, cfg_ok)
    assert (calib.a > 0).all(), "scale field went negative with the constraint on"

    surface = apply_calibration(_depth(rel), calib)
    ground = ~building
    assert surface[building].mean() > surface[ground].mean(), \
        "buildings came out below the ground they stand on"


def test_positive_constraint_leaves_a_well_posed_fit_alone():
    """When the data does not need the constraint, it must not change the answer."""
    rel = np.random.default_rng(12).random((96, 96)).astype(np.float32)
    truth = 25.0 * rel + 320.0
    anchors = _sample(truth, stride=6)

    free = solve_agmc(_depth(rel), anchors, Config(lattice_stride=24,
                                                   extras={"enforce_positive_scale": False}))
    held = solve_agmc(_depth(rel), anchors, Config(lattice_stride=24,
                                                   extras={"enforce_positive_scale": True}))
    assert np.allclose(free.a, held.a, atol=1e-6)
    assert np.allclose(free.b, held.b, atol=1e-6)


# ---------------------------------------------------- parallel bootstrap
def _bootstrap_inputs(size=96, seed=3):
    """A small scene with enough anchors that the bootstrap does real work."""
    import numpy as np

    from ayama.core.types import Anchor, Config, DepthField, SceneMeta

    rng = np.random.default_rng(seed)
    rel = rng.random((size, size)).astype(np.float32)
    meta = SceneMeta(crs="EPSG:32644", transform=(0.5, 0, 0, 0, -0.5, 0), gsd_m=0.5)
    depth = DepthField(relative=rel, meta=meta, backbone="test-fixture")
    anchors = [
        Anchor(int(r), int(c), float(400.0 + 8.0 * rel[r, c]), "terrain", "dem", 0.6)
        for r in range(0, size, 6) for c in range(0, size, 6)
    ]
    return depth, anchors, Config()


def test_parallel_bootstrap_is_bit_identical_to_serial():
    """Threads must not change the answer, only the wall time.

    The resample indices are drawn up front from the seeded generator and the
    results accumulated in index order rather than completion order, precisely
    so this holds. Accumulating as futures land would make sigma depend on
    thread scheduling, which is the kind of irreproducibility that is very hard
    to notice and impossible to defend.
    """
    import numpy as np

    from ayama.chhaya.uncertainty import bootstrap_sigma

    depth, anchors, cfg = _bootstrap_inputs()
    mean_s, sigma_s = bootstrap_sigma(depth, anchors, cfg, n_boot=8, workers=1)
    mean_p, sigma_p = bootstrap_sigma(depth, anchors, cfg, n_boot=8, workers=4)

    assert np.array_equal(mean_s, mean_p)
    assert np.array_equal(sigma_s, sigma_p)
    assert np.isfinite(sigma_s).all() and sigma_s.max() > 0


def test_bootstrap_worker_count_is_bounded():
    """Never more threads than solves, and never an unbounded pool."""
    from ayama.chhaya.uncertainty import _default_workers

    assert _default_workers(1, 24) == 1
    assert _default_workers(4, 24) == 4
    assert _default_workers(64, 24) == 24          # capped by the work available
    assert _default_workers(0, 2) <= 2
    assert 1 <= _default_workers(0, 24) <= 8       # auto never oversubscribes


def test_bootstrap_reports_progress_once_per_resample():
    from ayama.chhaya.uncertainty import bootstrap_sigma

    depth, anchors, cfg = _bootstrap_inputs()
    seen = []
    bootstrap_sigma(depth, anchors, cfg, n_boot=6, workers=3,
                    on_progress=lambda d, t: seen.append((d, t)))
    assert [d for d, _ in seen] == [1, 2, 3, 4, 5, 6]
    assert all(t == 6 for _, t in seen)


# ------------------------------------------------- H2: frequency separation
def test_decompose_depth_splits_by_metres_not_pixels():
    """The cutoff is physical, so the same split means the same thing at any GSD."""
    import numpy as np

    from ayama.chhaya.agmc import decompose_depth

    rng = np.random.default_rng(2)
    D = rng.random((128, 128)).astype(np.float32)

    lo_fine, hi_fine = decompose_depth(D, gsd_m=0.5, radius_m=60.0)
    lo_coarse, _ = decompose_depth(D, gsd_m=2.0, radius_m=60.0)

    # lo + hi reconstructs the input exactly
    assert np.allclose(lo_fine + hi_fine, D, atol=1e-5)
    # the high band has essentially no DC left
    assert abs(float(hi_fine.mean())) < abs(float(D.mean())) / 100
    # a coarser GSD means fewer pixels per 60 m, so less blurring
    assert lo_coarse.std() > lo_fine.std()


def test_dual_branch_releases_the_scale_field_from_its_floor():
    """The mechanism H2 exists to fix, asserted directly.

    Single-branch, a terrain-dominated anchor set drives the scale to a_min and
    pins it there - 100% of lattice nodes on the real benchmark. Routing terrain
    anchors to the offset field alone removes that pressure.
    """
    import numpy as np

    from ayama.chhaya.agmc import solve_agmc
    from ayama.core.types import Anchor, Config, DepthField, SceneMeta, Tier

    rng = np.random.default_rng(5)
    size = 96
    # A depth field whose LOW band anti-correlates with terrain, which is the
    # situation measured on the benchmark, and whose HIGH band carries structure.
    yy, xx = np.mgrid[0:size, 0:size]
    ramp = (xx / size).astype(np.float32)
    structure = np.zeros((size, size), np.float32)
    structure[30:50, 30:50] = 0.4
    D = (0.8 * (1.0 - ramp) + structure + 0.01 * rng.random((size, size))).astype(np.float32)
    terrain = 400.0 + 20.0 * ramp                    # rises where D falls

    meta = SceneMeta(crs="EPSG:32644", transform=(0.5, 0, 0, 0, -0.5, 0), gsd_m=0.5)
    depth = DepthField(relative=D, meta=meta, backbone="test-fixture")

    anchors = [Anchor(int(r), int(c), float(terrain[r, c]), "terrain", "dem", 0.6)
               for r in range(0, size, 6) for c in range(0, size, 6)
               if not (28 <= r <= 52 and 28 <= c <= 52)]
    anchors += [Anchor(40, 40, 8.0, "object", "shadow", 0.8, ref_row=25, ref_col=40)]

    cfg = Config()
    single = solve_agmc(depth, anchors, cfg, tier=Tier.A, dual_branch=False)
    dual = solve_agmc(depth, anchors, cfg, tier=Tier.A, dual_branch=True)

    floor = float(cfg.extras.get("min_scale", 0.05))
    single_pinned = float((single.a <= floor + 1e-4).mean())
    dual_pinned = float((dual.a <= floor + 1e-4).mean())
    assert single_pinned > 0.5, "the fixture no longer reproduces the collapse"
    assert dual_pinned < single_pinned, "dual branch did not release the scale field"


def test_dual_branch_calibration_is_applied_to_the_band_it_was_fitted_to():
    """The trap: applying a dual-branch field to raw depth re-adds the ramp.

    It would look plausible - a smooth surface with structure on it - and be
    wrong by exactly the low-frequency component the split exists to discard.
    """
    import numpy as np

    from ayama.chhaya.agmc import apply_calibration, decompose_depth, solve_agmc
    from ayama.core.types import Anchor, Config, DepthField, SceneMeta, Tier

    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    D = (xx / size).astype(np.float32)
    meta = SceneMeta(crs="EPSG:32644", transform=(0.5, 0, 0, 0, -0.5, 0), gsd_m=0.5)
    depth = DepthField(relative=D, meta=meta, backbone="test-fixture")
    anchors = [Anchor(int(r), int(c), 400.0, "terrain", "dem", 0.6)
               for r in range(0, size, 8) for c in range(0, size, 8)]
    anchors += [Anchor(32, 32, 5.0, "object", "shadow", 0.8, ref_row=10, ref_col=32)]

    cal = solve_agmc(depth, anchors, Config(), tier=Tier.A, dual_branch=True)
    assert cal.dual_branch is True
    assert cal.depth_high is not None

    _lo, hi = decompose_depth(D, 0.5, 60.0)
    assert np.allclose(cal.depth_high, hi, atol=1e-5)

    got = apply_calibration(depth, cal)
    assert np.allclose(got, cal.a * hi + cal.b, atol=1e-4)
    # and it is NOT what the raw field would have given
    assert not np.allclose(got, cal.a * D + cal.b, atol=1e-3)


def test_single_branch_remains_the_default():
    """H2 is a hypothesis under test, not the shipped calibration."""
    import numpy as np

    from ayama.chhaya.agmc import solve_agmc
    from ayama.core.types import Anchor, Config, DepthField, SceneMeta, Tier

    meta = SceneMeta(gsd_m=0.5)
    depth = DepthField(relative=np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32),
                       meta=meta, backbone="test-fixture")
    anchors = [Anchor(int(r), int(c), 400.0, "terrain", "dem", 0.6)
               for r in range(0, 64, 8) for c in range(0, 64, 8)]
    cal = solve_agmc(depth, anchors, Config(), tier=Tier.A)
    assert cal.dual_branch is False
    assert cal.depth_high is None
