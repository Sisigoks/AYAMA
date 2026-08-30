"""End-to-end pipeline, semantics and shadow physics.

These run on the bundled real scene - a lidar crop of central Zurich - with a
real backbone. They check plumbing and invariants, never accuracy: a number
produced by a 384 px crop with four bootstrap samples is not a result. The
weightless placeholder backbone they used to run on has been removed, so the
end-to-end cases are marked `slow`.
"""
from __future__ import annotations

import numpy as np
import pytest

from traksha.core.types import (BARE_GROUND, BUILDING, Config, Scene, Tier)

rasterio = pytest.importorskip("rasterio")

# The smallest real backbone. Every case that runs inference is `slow`.
BACKBONE = "dav2-vits"


@pytest.fixture(scope="module")
def written_scene(tmp_path_factory):
    """The bundled real scene on disk, with its lidar truth alongside.

    A sun is supplied so the shadow-physics stages have something to exercise;
    swisstopo publishes none, so this angle is a test parameter, not a
    measurement.
    """
    from traksha.data.sample import load_sample_scene
    from traksha.dsm.cog import write_cog, write_rgb

    d = tmp_path_factory.mktemp("scene")
    sc = load_sample_scene(size=384, sun=(150.0, 45.0))
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
    from traksha.semantics.segment import class_fractions, segment

    sc, _ = written_scene
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    sem, provenance = segment(scene)
    assert provenance == "heuristic"
    assert sem.shape == sc.rgb.shape[:2]
    frac = class_fractions(sem)
    assert frac["bare ground"] > 0.1
    assert sum(frac.values()) == pytest.approx(1.0)


def test_segmentation_can_be_supplied_as_a_raster(written_scene, tmp_path):
    from traksha.dsm.cog import write_cog
    from traksha.semantics.segment import segment

    sc, _ = written_scene
    p = str(tmp_path / "sem.tif")
    write_cog(p, sc.sem.astype(np.float32), sc.meta, dtype="uint8", nodata=255)
    scene = Scene(rgb=sc.rgb, meta=sc.meta)
    sem, provenance = segment(scene, method="raster", path=p)
    assert provenance.startswith("raster:")
    assert np.array_equal(sem, sc.sem)


# ---------------------------------------------------------------------- shadow
def test_shadow_quality_gate_follows_the_sun():
    from traksha.semantics.shadow import quality_from_sun_elevation

    assert quality_from_sun_elevation(None) == 0.0
    assert quality_from_sun_elevation(12.0) == 0.0     # too low, shadows sprawl
    assert quality_from_sun_elevation(85.0) == 0.0     # too high, shadows vanish
    assert quality_from_sun_elevation(45.0) == pytest.approx(1.0)
    assert 0.0 < quality_from_sun_elevation(24.0) < 1.0


def test_the_shadow_detector_flags_pixels_that_are_actually_dark(written_scene):
    """What can be asserted without knowing the sun.

    Against a renderer this was an F1 test, because the renderer knew where the
    shadows were. Here nothing does: swisstopo publishes no acquisition time, so
    a geometric truth mask can only be built under an assumed sun, and scoring
    the detector against a guess measures the guess. These two properties need
    no sun at all - the mask must be non-degenerate, and it must actually be
    dark.
    """
    from traksha.semantics.shadow import detect_shadow

    sc, _ = written_scene
    mask = detect_shadow(sc.as_scene(), sc.sem)
    assert 0.02 < mask.mean() < 0.35, f"{100 * mask.mean():.0f}% flagged as shadow"

    lum = sc.rgb.astype(np.float32).mean(-1)
    assert lum[mask].mean() < 0.6 * lum[~mask].mean(),         f"'shadow' pixels are not dark: {lum[mask].mean():.0f} vs {lum[~mask].mean():.0f}"


@pytest.mark.slow
def test_what_the_shadow_detector_flags_is_in_geometric_shadow(written_scene):
    """Precision against lidar geometry, at the sun that best explains the image.

    The sun is unknown, so it is fitted rather than assumed: the azimuth whose
    ray-marched mask best matches the detector is the one the image is
    consistent with. At that azimuth most of what the detector flags really is
    occluded, which is the property the anchors depend on.

    Recall is deliberately *not* asserted. Geometric shadow includes surfaces
    that are self-shaded and shadows cast onto other roofs; a radiometric
    detector sees only the dark ones, and on this crop it recovers under a third
    of them. That gap is a real limitation, not a bug - see README section 3.4.
    """
    from traksha.eval.shadow_truth import cast_shadow_mask
    from traksha.semantics.shadow import detect_shadow

    sc, _ = written_scene
    mask = detect_shadow(sc.as_scene(), sc.sem)
    best_p, best_az = 0.0, None
    for az in (30, 120, 210, 300):
        truth = cast_shadow_mask(sc.dsm_m, az, 45.0, sc.meta.gsd_m)
        p = float((mask & truth).sum()) / max(float(mask.sum()), 1.0)
        if p > best_p:
            best_p, best_az = p, az
    assert best_p > 0.5, f"best precision {best_p:.2f} at azimuth {best_az}"
    assert mask.mean() < 0.35, "more than a third of the scene flagged as shadow"


def test_shadow_anchors_recover_building_heights(written_scene):
    """The physics, checked against the DSM that cast the shadows."""
    from traksha.chhaya.anchors import harvest_shadow

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

    from traksha.chhaya.anchors import harvest_shadow

    sc, _ = written_scene
    blind = replace(sc.meta, sun_azimuth_deg=None, sun_elevation_deg=None)
    scene = Scene(rgb=sc.rgb, meta=blind)
    assert harvest_shadow(scene, sc.sem, sc.shadow) == []


# ---------------------------------------------------------------------- ladder
def test_tier_selection_walks_the_ladder(written_scene):
    from dataclasses import replace

    from traksha.chhaya.ladder import select_tier
    from traksha.core.types import GCP

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
    from traksha.dsm.assemble import assemble

    sc, _ = written_scene
    surf = assemble(sc.dsm_m, sc.sem, sc.meta, tier=Tier.A)
    assert (surf.ndsm_m >= 0).all()
    assert np.isfinite(surf.dsm_m).all()

    # Structures must come out taller than open ground. "Structure" is taken
    # from the lidar, not from the colour heuristic: on real imagery the
    # heuristic's building class is unreliable (see the test below), and an
    # assembler test must not fail for a segmentation reason.
    tall, flat = sc.ndsm_m > 5.0, sc.ndsm_m < 0.5
    assert tall.sum() > 100 and flat.sum() > 100
    assert surf.ndsm_m[tall].mean() > surf.ndsm_m[flat].mean() + 3.0


def test_the_colour_heuristic_does_not_find_buildings_on_real_imagery(written_scene):
    """A limitation, pinned so it cannot be forgotten or silently 'fixed'.

    On rendered scenes the heuristic separated classes well, because the
    renderer painted them in separable colours. On a real orthophoto it does
    not: what it calls `building` is no taller, by lidar, than what it calls
    bare ground. Every real run in README section 3 used this heuristic and had
    no labels of its own, which is one reason the semantics-dependent anchors
    contribute so little there.
    """
    sc, _ = written_scene
    building = sc.sem == BUILDING
    ground = sc.sem == BARE_GROUND
    if building.sum() < 100 or ground.sum() < 100:
        pytest.skip("not enough of either class in this crop")
    # Not an aspiration - a record of what is true today.
    assert sc.ndsm_m[building].mean() < sc.ndsm_m[ground].mean() + 2.0,         "the heuristic now separates buildings by height - update README 3 and this test"


def test_hole_filling_removes_non_finite_pixels():
    from traksha.dsm.assemble import fill_holes

    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    a[3:5, 3:5] = np.nan
    filled = fill_holes(a)
    assert np.isfinite(filled).all()
    assert filled[0, 0] == pytest.approx(0.0)


# -------------------------------------------------------------------- pipeline
@pytest.mark.slow
def test_full_run_produces_artifacts_and_metrics(written_scene):
    from traksha.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone=BACKBONE, chip=256, overlap=0.25,
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
        assert ds.tags().get("TRAKSHA_BACKBONE") == BACKBONE

    for stage in ("ingest", "depth", "anchors", "calibration", "assemble"):
        assert stage in res.timings_s


@pytest.mark.slow
def test_run_without_a_dem_drops_to_tier_c(written_scene):
    from traksha.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone=BACKBONE, chip=256, n_bootstrap=0)
    res = run(paths["rgb"], cfg, out_dir=None, write_artifacts=False)
    assert res.tier is Tier.C
    assert "DEM" in res.tier_reason or "CRS" in res.tier_reason
    assert res.surface is not None


@pytest.mark.slow
def test_missing_dem_file_fails_loudly(written_scene):
    """A run must never silently proceed with a DEM it could not load."""
    from traksha.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone=BACKBONE, chip=256, dem_source="copernicus", n_bootstrap=0)
    with pytest.raises(FileNotFoundError):
        run(paths["rgb"], cfg, out_dir=None, write_artifacts=False)


@pytest.mark.slow
def test_ablation_variants_share_one_inference(written_scene):
    from traksha.api.pipeline import load_dem
    from traksha.core.ingest import ingest
    from traksha.depth.infer import predict_depth
    from traksha.eval.ablation import run_variants
    from traksha.semantics.segment import segment
    from traksha.semantics.shadow import detect_shadow

    sc, paths = written_scene
    scene = ingest(paths["rgb"])
    depth = predict_depth(scene, BACKBONE, chip=256, overlap=0.25)
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
    from traksha.api.pipeline import load_reference
    from traksha.core.ingest import ingest
    from traksha.dsm.cog import write_cog

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


@pytest.mark.slow
def test_delta1_is_reported_against_reference_heights(written_scene):
    """delta1 needs heights above ground on both sides, or it is meaningless."""
    from traksha.api.pipeline import run

    sc, paths = written_scene
    cfg = Config(backbone=BACKBONE, chip=256, dem_source=f"sim:{paths['dtm']}",
                 reference=paths["dsm"], n_bootstrap=0)
    res = run(paths["rgb"], cfg, out_dir=None, write_artifacts=False)
    assert np.isfinite(res.metrics["delta1"]), "delta1 was not computed"
    assert 0.0 <= res.metrics["delta1"] <= 1.0
    assert res.metrics["delta1_n_px"] > 0


@pytest.mark.slow
def test_preflight_passes_end_to_end_on_this_machine():
    """The whole pipeline, with a verdict.

    Deliberately runs the real command rather than the pieces: one invocation
    proves sample -> depth -> anchors -> AGMC -> uncertainty -> artifacts ->
    tileset, which is what someone with a fresh checkout needs to know.
    """
    from traksha.cli import main

    assert main(["preflight", "--backbone", BACKBONE,
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

    from traksha.depth.backbones import BACKBONES
    from traksha.depth.backbones.hf import CHECKPOINTS

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
    import traksha
    from pathlib import Path

    root = Path(traksha.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("nn.Module", "torch.save", "loss.backward", ".backward()"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"something trainable appeared: {offenders}"
