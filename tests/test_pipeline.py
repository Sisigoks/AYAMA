"""End-to-end pipeline, semantics and shadow physics.

These run on the synthetic backbone so the whole chain is exercised in seconds
with no weights. They check plumbing and invariants, never accuracy: a number
produced by the synthetic backbone is not a result.
"""
from __future__ import annotations

import numpy as np
import pytest

from ayama.core.types import (BARE_GROUND, BUILDING, Config, Scene, Tier)

rasterio = pytest.importorskip("rasterio")


@pytest.fixture(scope="module")
def written_scene(tmp_path_factory):
    """A synthetic town on disk, with its ground truth alongside."""
    from ayama.dsm.cog import write_cog, write_rgb
    from ayama.eval.synthetic_scene import make_scene

    d = tmp_path_factory.mktemp("scene")
    sc = make_scene(size=384, gsd_m=0.5, seed=21)
    paths = {
        "rgb": str(d / "scene.tif"),
        "dsm": str(d / "scene_dsm.tif"),
        "dtm": str(d / "scene_dtm.tif"),
        "out": str(d / "out"),
    }
    write_rgb(paths["rgb"], sc.rgb, sc.meta)
    write_cog(paths["dsm"], sc.dsm_m, sc.meta)
    write_cog(paths["dtm"], sc.dtm_m, sc.meta)
    return sc, paths


# ------------------------------------------------------------------- semantics
def test_heuristic_segmentation_finds_the_classes_that_exist(written_scene):
    from ayama.semantics.segment import class_fractions, segment

    sc, _ = written_scene
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    sem, provenance = segment(scene)
    assert provenance == "heuristic"
    assert sem.shape == sc.rgb.shape[:2]
    frac = class_fractions(sem)
    assert frac["bare ground"] > 0.1
    assert sum(frac.values()) == pytest.approx(1.0)


def test_segmentation_can_be_supplied_as_a_raster(written_scene, tmp_path):
    from ayama.dsm.cog import write_cog
    from ayama.semantics.segment import segment

    sc, _ = written_scene
    p = str(tmp_path / "sem.tif")
    write_cog(p, sc.sem.astype(np.float32), sc.meta, dtype="uint8", nodata=255)
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    sem, provenance = segment(scene, method="raster", path=p)
    assert provenance.startswith("raster:")
    assert np.array_equal(sem, sc.sem)


# ---------------------------------------------------------------------- shadow
def test_shadow_quality_gate_follows_the_sun():
    from ayama.semantics.shadow import quality_from_sun_elevation

    assert quality_from_sun_elevation(None) == 0.0
    assert quality_from_sun_elevation(12.0) == 0.0     # too low, shadows sprawl
    assert quality_from_sun_elevation(85.0) == 0.0     # too high, shadows vanish
    assert quality_from_sun_elevation(45.0) == pytest.approx(1.0)
    assert 0.0 < quality_from_sun_elevation(24.0) < 1.0


def test_shadow_detector_is_precise_as_well_as_complete(written_scene):
    """Recall alone is satisfied by flagging half the image; F1 is not."""
    from ayama.semantics.shadow import detect_shadow

    sc, _ = written_scene
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    mask = detect_shadow(scene, sc.sem)
    truth = sc.shadow
    if truth.sum() < 50:
        pytest.skip("scene has almost no cast shadow")
    tp = float((mask & truth).sum())
    precision = tp / max(float(mask.sum()), 1.0)
    recall = tp / float(truth.sum())
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    assert f1 > 0.7, f"shadow F1 {f1:.2f} (P {precision:.2f} R {recall:.2f})"
    assert mask.mean() < 0.35, "more than a third of the scene flagged as shadow"


def test_shadow_anchors_recover_building_heights(written_scene):
    """The physics, checked against the DSM that cast the shadows."""
    from ayama.chhaya.anchors import harvest_shadow

    sc, _ = written_scene
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    anchors = harvest_shadow(scene, sc.sem, sc.shadow)
    if not anchors:
        pytest.skip("no isolated buildings with a clean shadow in this scene")

    ndsm = sc.ndsm_m
    errors = [abs(a.value_m - float(ndsm[a.row, a.col])) for a in anchors]
    assert all(a.is_relative for a in anchors), "a shadow anchor entered as an elevation"
    assert float(np.median(errors)) < 3.0, f"median shadow height error {np.median(errors):.1f} m"


def test_shadow_anchors_are_empty_without_sun_metadata(written_scene):
    from dataclasses import replace

    from ayama.chhaya.anchors import harvest_shadow

    sc, _ = written_scene
    blind = replace(sc.meta, sun_azimuth_deg=None, sun_elevation_deg=None)
    scene = Scene(rgb=sc.rgb, meta=blind)
    assert harvest_shadow(scene, sc.sem, sc.shadow) == []


# ---------------------------------------------------------------------- ladder
def test_tier_selection_walks_the_ladder(written_scene):
    from dataclasses import replace

    from ayama.chhaya.ladder import select_tier
    from ayama.core.types import GCP

    sc, paths = written_scene
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    cfg = Config(dem_source=paths["dtm"])

    assert select_tier(scene, cfg, None, dem_available=True).tier is Tier.A
    gcps = [GCP(1, 1, 400.0), GCP(2, 2, 401.0), GCP(3, 3, 402.0)]
    assert select_tier(scene, cfg, gcps, dem_available=True).tier is Tier.B

    plain = Scene(rgb=sc.rgb, meta=replace(sc.meta, crs=None, transform=None))
    decision = select_tier(plain, cfg, None, dem_available=False)
    assert decision.tier is Tier.C and "CRS" in decision.reason


# -------------------------------------------------------------------- assembly
def test_ndsm_is_never_negative_and_dtm_sits_under_the_dsm(written_scene):
    from ayama.dsm.assemble import assemble

    sc, _ = written_scene
    surf = assemble(sc.dsm_m, sc.sem, sc.meta, tier=Tier.A)
    assert (surf.ndsm_m >= 0).all()
    assert np.isfinite(surf.dsm_m).all()
    # Buildings must come out taller than open ground.
    if (sc.sem == BUILDING).sum() > 100:
        assert surf.ndsm_m[sc.sem == BUILDING].mean() > surf.ndsm_m[sc.sem == BARE_GROUND].mean()


def test_hole_filling_removes_non_finite_pixels():
    from ayama.dsm.assemble import fill_holes

    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    a[3:5, 3:5] = np.nan
    filled = fill_holes(a)
    assert np.isfinite(filled).all()
    assert filled[0, 0] == pytest.approx(0.0)


# -------------------------------------------------------------------- pipeline
def test_full_run_produces_artifacts_and_metrics(written_scene):
    from ayama.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone="synthetic", chip=256, overlap=0.25,
                 dem_source=f"sim:{paths['dtm']}", reference=paths["dsm"],
                 n_bootstrap=4, lattice_stride=32)
    res = run(paths["rgb"], cfg, out_dir=paths["out"])

    assert res.tier is Tier.A
    assert res.anchors_used > 0
    assert res.surface is not None and np.isfinite(res.surface.dsm_m).all()
    assert res.metrics["n_px"] > 0
    assert "coverage_1s" in res.metrics
    assert res.baseline_metrics, "the global-affine baseline must always be reported"

    for key in ("dsm", "ndsm", "sigma", "sem", "shadow", "texture", "provenance"):
        assert key in res.artifacts

    with rasterio.open(res.artifacts["dsm"]) as ds:
        assert ds.crs is not None
        assert ds.tags().get("AYAMA_BACKBONE") == "synthetic"

    for stage in ("ingest", "depth", "anchors", "calibration", "assemble"):
        assert stage in res.timings_s


def test_run_without_a_dem_drops_to_tier_c(written_scene):
    from ayama.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone="synthetic", chip=256, n_bootstrap=0)
    res = run(paths["rgb"], cfg, out_dir=None, write_artifacts=False)
    assert res.tier is Tier.C
    assert "DEM" in res.tier_reason or "CRS" in res.tier_reason
    assert res.surface is not None


def test_missing_dem_file_fails_loudly(written_scene):
    """A run must never silently proceed with a DEM it could not load."""
    from ayama.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone="synthetic", chip=256, dem_source="copernicus", n_bootstrap=0)
    with pytest.raises(FileNotFoundError):
        run(paths["rgb"], cfg, out_dir=None, write_artifacts=False)


def test_ablation_variants_share_one_inference(written_scene):
    from ayama.api.pipeline import load_dem
    from ayama.core.ingest import ingest
    from ayama.depth.infer import predict_depth
    from ayama.eval.ablation import run_variants
    from ayama.semantics.segment import segment
    from ayama.semantics.shadow import detect_shadow

    sc, paths = written_scene
    scene = ingest(paths["rgb"])
    depth = predict_depth(scene, "synthetic", chip=256, overlap=0.25)
    sem, _ = segment(scene)
    shadow = detect_shadow(scene, sem)
    dem_m, _ = load_dem(f"sim:{paths['dtm']}", scene)

    cfg = Config(n_bootstrap=4, lattice_stride=32)
    rows = run_variants(scene, depth, sem, shadow, sc.dsm_m, dem_m=dem_m, cfg=cfg,
                        variants=("global_affine", "agmc", "agmc_no_shadow"))
    assert len(rows) == 3
    assert all("mae_m" in r for r in rows)
    # The gate must actually change the anchor set it is credited with.
    assert rows[1]["n_anchors"] >= rows[2]["n_anchors"]


def test_reference_is_reprojected_not_squashed(written_scene, tmp_path):
    """A reference over a larger extent must be cropped to the tile, not resized.

    Reading with out_shape alone squashes a wider lidar DSM onto the tile, and
    every metric computed against it then looks plausible and means nothing.
    """
    from ayama.api.pipeline import load_reference
    from ayama.core.ingest import ingest
    from ayama.dsm.cog import write_cog

    sc, paths = written_scene
    scene = ingest(paths["rgb"])

    # A reference covering twice the area, same CRS, same pixel size: the top-left
    # quarter of it is the scene.
    h, w = sc.dsm_m.shape
    big = np.zeros((h * 2, w * 2), np.float32)
    big[:h, :w] = sc.dsm_m
    big[h:, :] = -999.0
    big[:, w:] = -999.0
    p = str(tmp_path / "big_ref.tif")
    write_cog(p, big, sc.meta)

    ref = load_reference(p, scene)
    assert ref.shape == sc.dsm_m.shape
    ok = np.isfinite(ref)
    assert ok.mean() > 0.9, "reprojection lost most of the overlap"
    assert np.nanmax(np.abs(ref[ok] - sc.dsm_m[ok])) < 1.0, "reference was resized, not cropped"


def test_delta1_is_reported_against_reference_heights(written_scene):
    """delta1 needs heights above ground on both sides, or it is meaningless."""
    from ayama.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone="synthetic", chip=256, dem_source=f"sim:{paths['dtm']}",
                 reference=paths["dsm"], n_bootstrap=0)
    res = run(paths["rgb"], cfg, out_dir=None, write_artifacts=False)
    assert np.isfinite(res.metrics["delta1"]), "delta1 was not computed"
    assert 0.0 <= res.metrics["delta1"] <= 1.0
    assert res.metrics["delta1_n_px"] > 0


# ------------------------------------------------------- device preflight
def test_device_availability_returns_a_verdict_never_raises():
    """`preflight` must answer "is my GPU set up", not traceback into torch."""
    from ayama.cli import _device_available

    for dev in ("cpu", "cuda", "mps", "auto"):
        ok, why = _device_available(dev)
        assert isinstance(ok, bool)
        assert ok or why, f"{dev} unavailable without saying why"
    assert _device_available("cpu")[0] is True


def test_preflight_refuses_an_unavailable_device_cleanly():
    """Asking for CUDA on a CPU-only build is a verdict with an exit code.

    It previously raised `AssertionError: Torch not compiled with CUDA enabled`
    from forty frames inside torch, which is the least useful possible answer to
    the one question the command exists to answer.
    """
    import torch

    if torch.cuda.is_available():
        pytest.skip("this box has CUDA; the refusal path cannot be exercised")

    from ayama.cli import build_parser, main

    args = build_parser().parse_args(
        ["preflight", "--device", "cuda", "--backbone", "dav2-vits"])
    assert args.func.__name__ == "cmd_preflight"
    assert main(["preflight", "--device", "cuda", "--backbone", "dav2-vits"]) == 1


@pytest.mark.slow
def test_preflight_passes_end_to_end_on_this_machine():
    """The whole pipeline, on whatever device this is, with a verdict.

    Deliberately runs the real command rather than the pieces: the point is that
    one invocation proves synth -> depth -> anchors -> AGMC -> uncertainty ->
    artifacts -> tileset, which is what someone with a fresh GPU box needs.
    """
    from ayama.cli import main

    assert main(["preflight", "--device", "cpu", "--backbone", "synthetic",
                 "--size", "256", "--chip", "256", "--bootstrap", "3"]) == 0


# ------------------------------------------------------- documentation contract
def test_every_registered_backbone_is_documented():
    """A model the code offers but the README omits is an undocumented dependency.

    The README's model table is the only place a reader learns which checkpoints
    exist, which was actually run, and that none of them are trained here. Adding
    a backbone without adding a row makes that table quietly wrong, and nothing
    else in the suite would notice.
    """
    from pathlib import Path

    from ayama.depth.backbones import BACKBONES
    from ayama.depth.backbones.hf import CHECKPOINTS

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    undocumented = [b for b in BACKBONES if f"`{b}`" not in readme]
    assert not undocumented, f"backbones missing from README: {undocumented}"

    missing = [c for c in CHECKPOINTS.values() if c not in readme]
    assert not missing, f"checkpoints missing from README: {missing}"


def test_the_project_trains_nothing():
    """The claim the README makes about models, asserted against the code.

    If this ever fails it is good news - it means a trainable component landed -
    but the README says the opposite in three places and would need updating.
    """
    import ayama
    from pathlib import Path

    root = Path(ayama.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("nn.Module", "torch.save", "loss.backward", ".backward()"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"something trainable appeared: {offenders}"
