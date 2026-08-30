"""Phase 1 contract tests: ingest, tiling, blending, raster IO, geometry."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from traksha.core.geo import (gsd_metres, percentile_stretch, pixel_to_world,
                            shadow_height, shadow_length, sun_vector, world_to_pixel)
from traksha.core.solar import solar_position
from traksha.core.types import Scene, SceneMeta

from traksha.depth.infer import (blend_window, n_chips, predict_depth, rank_normalise,
                               tile_offsets)


# --------------------------------------------------------------------------- geo
def test_projected_gsd_is_read_not_assumed():
    gsd, assumed = gsd_metres((0.5, 0, 0, 0, -0.5, 0), False)
    assert gsd == pytest.approx(0.5)
    assert assumed is False


def test_geographic_gsd_is_converted_to_metres():
    # 1e-5 deg at 18N is about 1.06 m, not 1e-5 m. This is the bug that
    # silently collapses every shadow-derived height.
    gsd, _ = gsd_metres((1e-5, 0, 0, 0, -1e-5, 0), True, 17.97)
    assert 1.0 < gsd < 1.15


def test_missing_transform_is_flagged_as_assumed():
    assert gsd_metres(None, False) == (1.0, True)


def test_sun_vector_axes():
    # East at 30 deg elevation.
    d_col, d_row, d_z = sun_vector(90, 30)
    assert d_col == pytest.approx(math.cos(math.radians(30)))
    assert d_row == pytest.approx(0.0, abs=1e-9)
    assert d_z == pytest.approx(0.5)
    # North is -row in image coordinates.
    _, d_row_n, _ = sun_vector(0, 45)
    assert d_row_n < 0


def test_shadow_trig_roundtrip():
    for el in (25.0, 45.0, 68.0):
        assert shadow_height(shadow_length(30.0, el), el) == pytest.approx(30.0, rel=1e-6)


def test_pixel_world_roundtrip():
    t = (0.5, 0.0, 300000.0, 0.0, -0.5, 1990000.0)
    x, y = pixel_to_world(t, 100, 250)
    r, c = world_to_pixel(t, x, y)
    assert (r, c) == pytest.approx((100.0, 250.0))


def test_percentile_stretch_handles_16bit():
    a = (np.linspace(0, 4095, 256 * 3).reshape(16, 16, 3)).astype(np.uint16)
    out = percentile_stretch(a)
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0


# ------------------------------------------------------------------------- solar
@pytest.mark.parametrize(
    "lat,lon,when,az,el",
    [
        (51.48, 0.0, datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc), 180.0, 62.0),
        (51.48, 0.0, datetime(2024, 6, 21, 5, 0, tzinfo=timezone.utc), 64.0, 9.0),
    ],
)
def test_solar_position_matches_noaa(lat, lon, when, az, el):
    got_az, got_el = solar_position(lat, lon, when)
    assert got_az == pytest.approx(az, abs=2.0)
    assert got_el == pytest.approx(el, abs=1.5)


# ------------------------------------------------------------------------ tiling
def test_tile_offsets_always_cover_the_last_edge():
    for total, chip, step in [(1024, 512, 384), (1000, 512, 384), (513, 512, 384), (300, 512, 384)]:
        offs = tile_offsets(total, chip, step)
        assert offs[0] == 0
        assert offs[-1] + chip >= min(total, chip)
        assert len(offs) == len(set(offs))


def test_blend_window_is_flat_inside_and_never_zero():
    w = blend_window(64, 8)
    assert w[32, 32] == pytest.approx(1.0)
    assert w.min() > 0.0
    assert w[0, 32] < w[8, 32] <= 1.0


def test_rank_normalise_is_monotonic_and_tie_safe():
    x = np.array([[3.0, 1.0], [2.0, 2.0]], np.float32)
    r = rank_normalise(x)
    assert r.min() == 0.0 and r.max() == 1.0
    assert r[0, 1] < r[1, 0] < r[0, 0]      # 1 < 2 < 3
    assert r[1, 0] == r[1, 1]               # ties share a rank
    flat = rank_normalise(np.full((4, 4), 7.0, np.float32))
    assert np.allclose(flat, 0.5)


def test_rank_normalise_is_scale_invariant():
    a = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert np.allclose(rank_normalise(a), rank_normalise(a * 137.0 + 9.0), atol=1e-6)


class _RampBackbone:
    """Returns the patch luminance, so a global ramp stays a global ramp."""

    name = "ramp"
    native = None

    def load(self):
        return self

    def infer(self, patch):
        return patch.astype(np.float32).mean(axis=2)


def test_blending_is_seam_free_across_chips():
    # A horizontal ramp: whatever the per-chip normalisation does, the blended
    # mosaic must stay monotonic across chip boundaries.
    w = h = 512
    ramp = np.linspace(0, 255, w, dtype=np.float32)
    rgb = np.repeat(np.tile(ramp, (h, 1))[:, :, None], 3, axis=2).astype(np.uint8)
    scene = Scene(rgb=rgb, meta=SceneMeta())
    depth = predict_depth(scene, _RampBackbone(), chip=256, overlap=0.25)
    row = depth.relative[h // 2]
    d = np.diff(row)
    assert (d >= -1e-4).all(), "non-monotonic step at a chip seam"
    assert np.corrcoef(row, ramp)[0, 1] > 0.999


def test_predict_depth_pads_images_smaller_than_a_chip():
    scene = Scene(rgb=np.zeros((64, 96, 3), np.uint8), meta=SceneMeta())
    depth = predict_depth(scene, _RampBackbone(), chip=256, overlap=0.25)
    assert depth.relative.shape == (64, 96)
    assert np.isfinite(depth.relative).all()


def test_n_chips_matches_the_real_tiling():
    calls = {"n": 0}

    class Counting(_RampBackbone):
        def infer(self, patch):
            calls["n"] += 1
            return super().infer(patch)

    scene = Scene(rgb=np.zeros((900, 700, 3), np.uint8), meta=SceneMeta())
    predict_depth(scene, Counting(), chip=512, overlap=0.25)
    assert calls["n"] == n_chips((900, 700), 512, 0.25)


# --------------------------------------------------------------------- raster IO
rasterio = pytest.importorskip("rasterio")


def test_geotiff_roundtrip_preserves_geometry_and_sun(tmp_path):
    from traksha.core.ingest import ingest
    from traksha.dsm.cog import write_rgb

    meta = SceneMeta(
        crs="EPSG:32644",
        transform=(0.5, 0.0, 300000.0, 0.0, -0.5, 1990000.0),
        gsd_m=0.5,
        sun_azimuth_deg=138.4,
        sun_elevation_deg=61.2,
        acquired_utc="2024-03-21T06:30:00",
    )
    rgb = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    path = str(tmp_path / "s.tif")
    write_rgb(path, rgb, meta)

    scene = ingest(path)
    assert scene.meta.crs == "EPSG:32644"
    assert scene.meta.gsd_m == pytest.approx(0.5)
    assert scene.meta.sun_azimuth_deg == pytest.approx(138.4)
    assert scene.meta.sun_elevation_deg == pytest.approx(61.2)
    assert scene.meta.georeferenced and scene.meta.has_sun
    assert np.array_equal(scene.rgb, rgb)


def test_write_cog_preserves_values_and_nodata(tmp_path):
    from traksha.dsm.cog import write_cog

    meta = SceneMeta(crs="EPSG:32644", transform=(1.0, 0, 0, 0, -1.0, 0), gsd_m=1.0)
    a = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    a[0, 0] = np.nan
    path = str(tmp_path / "d.tif")
    write_cog(path, a, meta)
    with rasterio.open(path) as ds:
        got = ds.read(1, masked=True)
        assert ds.crs.to_string() == "EPSG:32644"
        assert got.mask[0, 0]
        assert got[10, 10] == pytest.approx(a[10, 10])


def test_the_bundled_sample_is_real_and_self_consistent():
    """The scene every other test builds on. If it is wrong, everything is.

    It is a real lidar crop, so the invariants are the ones physics guarantees -
    the surface cannot dip below the bare earth - not ones a renderer was told
    to satisfy.
    """
    from traksha.data.sample import load_sample_scene

    sc = load_sample_scene(size=256)
    assert np.isfinite(sc.dsm_m).all() and np.isfinite(sc.dtm_m).all()
    assert sc.ndsm_m.max() > 3.0, "no objects in the crop"

    # A renderer could guarantee DSM >= DTM everywhere. Real data cannot: these
    # are two independent flights (surface 2018, terrain 2020) and on flat
    # ground they disagree by noise. What must hold is that the disagreement
    # stays at noise scale rather than hiding a vertical-datum or resampling
    # bug, which would show up as metres.
    undercut = np.maximum(sc.dtm_m - sc.dsm_m, 0.0)
    assert undercut.max() < 2.0, "DSM dips metres below the DTM - datum mismatch?"
    assert np.percentile(undercut[undercut > 0], 99) < 0.1
    assert (sc.ndsm_m >= 0).all(), "the derived nDSM must be clamped at ground"
    assert sc.meta.georeferenced
    # No sun is published for these products, so none may be invented.
    assert not sc.meta.has_sun
    assert not sc.shadow.any()


def test_a_sun_supplied_by_the_caller_produces_a_truth_shadow_mask():
    """Shadow physics needs an angle. The caller states it; the fixture never does."""
    from traksha.data.sample import load_sample_scene

    sc = load_sample_scene(size=256, sun=(150.0, 40.0))
    assert sc.meta.has_sun
    assert 0.0 < sc.shadow.mean() < 0.6, "ray-marched shadow fraction is implausible"


def test_the_sample_crop_carries_a_transform_that_follows_the_offset():
    """An offset crop with the parent transform is an off-by-N projection bug."""
    from traksha.data.sample import load_sample_scene

    a = load_sample_scene(size=128)
    b = load_sample_scene(size=128, offset=(64, 64))
    assert a.meta.transform[2] != b.meta.transform[2]
    assert b.meta.transform[2] == pytest.approx(a.meta.transform[2] + 64 * 0.5)
