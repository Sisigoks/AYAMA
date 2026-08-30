"""Real-dataset discovery and evaluation.

Everything else in the suite exercises the one bundled scene. These
tests cover dataset discovery: finding scenes someone else produced,
noticing what truth ships with them, and — the part that matters — comparing
against the right quantity.

US3D ships `_AGL.tif`, a height above ground, not an elevation. Silently
comparing a predicted DSM against it is a ~400 m error that would look like a
catastrophic model failure rather than a units bug, so `reference_kind` is
carried explicitly and asserted here.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("rasterio")

from traksha.data import (SceneRef, aggregate, discover,  # noqa: E402
                        discover_generic, discover_us3d)


def _write_scene(d, stem, suffixes, seed=0, size=128):
    """Write the bundled real crop out under a dataset's naming convention.

    `seed` selects a different window of the same scene, so two scenes in one
    directory are not byte-identical.
    """
    from traksha.data.sample import load_sample_scene
    from traksha.dsm.cog import write_cog, write_rgb

    sc = load_sample_scene(size=size, offset=(64 * (seed % 4), 64 * (seed % 4)))
    write_rgb(os.path.join(d, stem + suffixes["image"]), sc.rgb, sc.meta)
    if "reference" in suffixes:
        ref = (np.maximum(sc.dsm_m - sc.dtm_m, 0) if suffixes.get("_ndsm") else sc.dsm_m)
        write_cog(os.path.join(d, stem + suffixes["reference"]),
                  ref.astype(np.float32), sc.meta)
    if "semantics" in suffixes:
        write_cog(os.path.join(d, stem + suffixes["semantics"]),
                  sc.sem.astype(np.float32), sc.meta, dtype="uint8", nodata=255)
    if "dem" in suffixes:
        write_cog(os.path.join(d, stem + suffixes["dem"]),
                  sc.dtm_m.astype(np.float32), sc.meta)
    return sc


# ----------------------------------------------------------------- discovery
def test_us3d_layout_is_found_and_marked_as_height_above_ground(tmp_path):
    d = str(tmp_path)
    _write_scene(d, "JAX_004_006",
                 {"image": "_RGB.tif", "reference": "_AGL.tif",
                  "semantics": "_CLS.tif", "_ndsm": True})
    scenes = discover_us3d(d)
    assert len(scenes) == 1
    s = scenes[0]
    assert s.name == "JAX_004_006"
    assert s.reference and s.reference.endswith("_AGL.tif")
    assert s.semantics and s.semantics.endswith("_CLS.tif")
    # the assertion this module exists for
    assert s.reference_kind == "ndsm", "AGL is a height, not an elevation"


def test_generic_layout_does_not_mistake_companions_for_scenes(tmp_path):
    """`_dsm.tif` also ends in `.tif`; a naive glob finds four scenes, not one."""
    d = str(tmp_path)
    _write_scene(d, "tile_a", {"image": ".tif", "reference": "_dsm.tif",
                               "dem": "_dem.tif", "semantics": "_sem.tif"})
    scenes = discover_generic(d)
    assert [s.name for s in scenes] == ["tile_a"]
    s = scenes[0]
    assert s.reference_kind == "dsm"
    assert s.dem and s.semantics


def test_custom_suffixes_override_the_defaults(tmp_path):
    d = str(tmp_path)
    _write_scene(d, "x", {"image": "_img.tiff", "reference": "_truth.tiff"})
    scenes = discover_generic(d, suffixes={"image": "_img.tiff",
                                           "reference": "_truth.tiff"})
    assert len(scenes) == 1 and scenes[0].reference


def test_an_empty_directory_says_what_it_expected(tmp_path):
    """A dataset that silently finds nothing wastes an afternoon."""
    with pytest.raises(FileNotFoundError, match="expected images ending"):
        discover(str(tmp_path), layout="us3d")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        discover(str(tmp_path / "nope"), layout="generic")
    with pytest.raises(KeyError, match="unknown layout"):
        discover(str(tmp_path), layout="imaginary")


def test_missing_truth_is_recorded_rather_than_assumed(tmp_path):
    d = str(tmp_path)
    _write_scene(d, "bare", {"image": ".tif"})
    s = discover_generic(d)[0]
    assert s.reference is None and s.dem is None and s.semantics is None
    assert "ref" not in s.describe()


# ------------------------------------------------------------------ running
@pytest.mark.slow
def test_us3d_scene_runs_and_is_scored_against_height_above_ground(tmp_path):
    """End to end on a US3D-shaped tile, including the flat-ground floor.

    For an nDSM reference the honest floor is "predict zero everywhere". A
    method that cannot beat flat ground is not reconstructing anything, and on
    this pipeline that is a live risk rather than a hypothetical - see README
    section 4.
    """
    from traksha.core.types import Config
    from traksha.data import run_scene

    d = str(tmp_path / "ds")
    os.makedirs(d)
    _write_scene(d, "JAX_000_001",
                 {"image": "_RGB.tif", "reference": "_AGL.tif", "_ndsm": True},
                 size=192)
    ref = discover_us3d(d)[0]

    cfg = Config(backbone="dav2-vits", chip=192, n_bootstrap=3)
    rec = run_scene(ref, str(tmp_path / "out"), cfg)

    assert rec["reference_kind"] == "ndsm"
    assert np.isfinite(rec["metrics"]["mae_m"])
    assert "zero_baseline_metrics" in rec, "an nDSM run must report the flat-ground floor"
    assert np.isfinite(rec["zero_baseline_metrics"]["mae_m"])
    assert rec["tier"] in ("A", "B", "C")


def test_aggregate_reports_spread_and_the_baselines(tmp_path):
    records = [
        {"metrics": {"mae_m": 3.0, "edge_f1": 0.2},
         "dem_metrics": {"mae_m": 3.4}, "zero_baseline_metrics": {"mae_m": 5.0}},
        {"metrics": {"mae_m": 3.4, "edge_f1": 0.3},
         "dem_metrics": {"mae_m": 3.6}, "zero_baseline_metrics": {"mae_m": 5.4}},
    ]
    agg = aggregate(records)
    assert agg["n_scenes"] == 2
    assert agg["mae_m"]["mean"] == pytest.approx(3.2)
    assert agg["mae_m"]["std"] == pytest.approx(0.2)
    assert agg["dem_metrics_mae_m"]["mean"] == pytest.approx(3.5)
    assert agg["zero_baseline_metrics_mae_m"]["mean"] == pytest.approx(5.2)


def test_aggregate_survives_missing_and_non_finite_metrics():
    """delta1 comes back as None on a degenerate run; it must not poison the mean."""
    records = [
        {"metrics": {"mae_m": 3.0, "delta1": None}},
        {"metrics": {"mae_m": float("nan")}},
        {"metrics": {}},
    ]
    agg = aggregate(records)
    assert agg["n_scenes"] == 3
    assert agg["mae_m"]["mean"] == pytest.approx(3.0)   # the NaN and the gap dropped
    assert "delta1" not in agg


def test_scene_ref_describe_is_readable():
    s = SceneRef(name="t", image="t.tif", reference="t_dsm.tif", dem="t_dem.tif")
    text = s.describe()
    assert "t" in text and "ref:dsm" in text and "dem" in text


# ------------------------------------------------- the external-segmentation fix
def test_segmentation_accepts_both_bare_and_prefixed_paths(tmp_path):
    """`raster:<path>` is what the pipeline writes as provenance, so it must read.

    Passing it back in previously reached rasterio verbatim and killed the run
    at the segmentation stage with "does not exist in the file system".
    """
    from traksha.core.types import Scene
    from traksha.dsm.cog import write_cog
    from traksha.data.sample import load_sample_scene
    from traksha.semantics.segment import segment

    sc = load_sample_scene(size=96)
    p = str(tmp_path / "labels.tif")
    write_cog(p, sc.sem.astype(np.float32), sc.meta, dtype="uint8", nodata=255)
    scene = Scene(rgb=sc.rgb, meta=sc.meta)

    bare, prov_bare = segment(scene, method="raster", path=p)
    pref, prov_pref = segment(scene, method="raster", path="raster:" + p)
    assert np.array_equal(bare, pref)
    assert prov_bare == prov_pref == f"raster:{p}"


# --------------------------------------------- nDSM scoring from a DSM + DTM
def test_a_dtm_sibling_is_a_companion_not_a_scene(tmp_path):
    """`_dtm.tif` ends in `.tif`; before this it was discovered as an image.

    On the four real Swiss scenes that turned one dataset of four into a
    dataset of eight, half of them bare-earth rasters with no truth.
    """
    d = str(tmp_path)
    _write_scene(d, "zurich", {"image": ".tif", "reference": "_dsm.tif",
                               "dem": "_dem.tif"})
    _write_scene(d, "zurich_dtm", {"image": ".tif"})     # the DTM sibling
    scenes = discover_generic(d)
    assert [s.name for s in scenes] == ["zurich"]
    assert scenes[0].dtm and scenes[0].dtm.endswith("_dtm.tif")


@pytest.mark.slow
def test_a_dsm_reference_with_a_dtm_is_also_scored_as_height_above_ground(tmp_path):
    """Elevation MAE flatters this pipeline; nDSM is the claim being made.

    Most of a scene is ground and the DEM already knows the ground, so an
    elevation MAE can look respectable while no object relief is recovered at
    all. Where a bare-earth DTM ships, the height-above-ground metrics and the
    flat-ground floor are reported too - see README section 4.
    """
    from traksha.core.types import Config
    from traksha.data import run_scene

    d = str(tmp_path / "ds")
    os.makedirs(d)
    _write_scene(d, "tile", {"image": ".tif", "reference": "_dsm.tif",
                             "dem": "_dem.tif"}, size=192)
    # the DTM the generic layout looks for
    from traksha.dsm.cog import write_cog
    from traksha.data.sample import load_sample_scene
    sc = load_sample_scene(size=192)
    write_cog(os.path.join(d, "tile_dtm.tif"), sc.dtm_m.astype(np.float32), sc.meta)

    ref = discover_generic(d)[0]
    assert ref.dtm, "the DTM must be picked up for nDSM scoring to happen"
    cfg = Config(backbone="dav2-vits", chip=192, n_bootstrap=3)
    rec = run_scene(ref, str(tmp_path / "out"), cfg)

    assert np.isfinite(rec["ndsm_metrics"]["mae_m"])
    assert np.isfinite(rec["zero_baseline_metrics"]["mae_m"])
    r = rec["relief"]
    assert 0.0 <= r["object_fraction"] <= 1.0
    assert r["true_max_height_m"] > 0
    # the comparison the section 4 diagnosis turns on
    assert r["pred_mean_height_m"] is not None


def test_aggregate_reports_relief_recovery():
    records = [
        {"metrics": {"mae_m": 8.0}, "ndsm_metrics": {"mae_m": 7.5},
         "zero_baseline_metrics": {"mae_m": 7.6},
         "relief": {"true_mean_height_m": 14.0, "pred_mean_height_m": 0.1}},
        {"metrics": {"mae_m": 9.0}, "ndsm_metrics": {"mae_m": 7.7},
         "zero_baseline_metrics": {"mae_m": 7.8},
         "relief": {"true_mean_height_m": 16.0, "pred_mean_height_m": 0.3}},
    ]
    agg = aggregate(records)
    assert agg["ndsm_metrics_mae_m"]["mean"] == pytest.approx(7.6)
    assert agg["true_mean_height_m"]["mean"] == pytest.approx(15.0)
    assert agg["pred_mean_height_m"]["mean"] == pytest.approx(0.2)
