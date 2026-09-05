"""Bare earth, and the public DEMs that anchor it.

Two modules, one subject: where the ground is. `dsm.dtm` extracts it from the
surface, `data.dem` fetches the coarse global product the anchors are pinned to.

The DTM tests avoid asserting Bulldozer's numbers - those are its business, and
they are recorded in `dsm.dtm`'s docstring with the sweep that produced them.
What is asserted here is the contract around it: the fallback fires and says so,
NaN survives, the pixel size reaches the filter even without georeferencing, and
a run can always tell which estimator produced its terrain.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from traksha.data import dem as DEM
from traksha.dsm import dtm as DTM
from traksha.dsm.assemble import extract_dtm


class Meta:
    def __init__(self, georef=True, gsd=0.5):
        self.georeferenced = georef
        self.crs = "EPSG:2056" if georef else None
        self.transform = (gsd, 0.0, 2683000.0, 0.0, -gsd, 1247000.0) if georef else None
        self.gsd_m = gsd
        self.bounds_wgs = (8.52, 47.36, 8.54, 47.38)


def terrain_with_buildings(h=96, w=96):
    """A tilted plane with blocks on it. The blocks are what a DTM must remove."""
    yy, xx = np.mgrid[:h, :w]
    ground = 400.0 + 0.05 * xx + 0.02 * yy
    dsm = ground.astype(np.float32).copy()
    for r0, c0, r1, c1, hh in ((20, 20, 36, 36, 18.0), (55, 60, 75, 80, 25.0)):
        dsm[r0:r1, c0:c1] += hh
    return dsm, ground.astype(np.float32)


# --------------------------------------------------------------------- DTM
FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "traksha", "data", "fixture")


def _fixture_rasters():
    """The bundled Zurich crop, with survey-grade swissALTI3D truth beside it.

    Real pixels on purpose. A synthetic tilted plane with two blocks on it is
    not a fair test of this swap and in fact favours the old filter: given a
    semantic raster that calls everything ground, the morphological branch
    smooths a plane back into a plane and scores beautifully. The failure it
    actually has needs a real roofscape, a real classifier and real terrain,
    which is the same argument `tests/conftest.py` makes for the whole suite.
    """
    rasterio = pytest.importorskip("rasterio")
    paths = [os.path.join(FIXTURE, f"zurich_{k}.tif") for k in ("dsm", "dtm", "sem")]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("the bundled fixture rasters are not present")
    return [rasterio.open(p).read(1) for p in paths]


def test_the_cloth_beats_the_filter_that_believes_the_semantic_raster():
    """The whole reason for the swap, on the imagery the README quotes."""
    if not DTM.available():
        pytest.skip("bulldozer is not installed")
    dsm, truth, sem = _fixture_rasters()
    out, prov = DTM.extract(dsm, Meta(), sem=sem, workers=2)
    old = extract_dtm(dsm, sem, 0.5)
    assert prov["method"] == "bulldozer"

    new_mae = float(np.abs(out - truth)[np.isfinite(out)].mean())
    old_mae = float(np.abs(old - truth).mean())
    assert new_mae < old_mae / 3.0, f"bulldozer {new_mae:.3f} vs morphological {old_mae:.3f}"


def test_the_old_filter_sits_on_the_roofs_it_mislabelled():
    """The +6 m bias is not noise. It is the surface standing on rooftops."""
    dsm, truth, sem = _fixture_rasters()
    old = extract_dtm(dsm, sem, 0.5)
    assert float((old - truth).mean()) > 4.0


def test_the_cloth_is_close_to_unbiased_on_the_same_scene():
    if not DTM.available():
        pytest.skip("bulldozer is not installed")
    dsm, truth, sem = _fixture_rasters()
    out, _ = DTM.extract(dsm, Meta(), sem=sem, workers=2)
    assert abs(float((out - truth)[np.isfinite(out)].mean())) < 1.0


def test_the_provenance_names_the_estimator_that_ran():
    """A DTM from the fallback must never be reported as one from the simulation."""
    dsm, _ = terrain_with_buildings()
    out, prov = DTM.extract(dsm, Meta(), sem=np.zeros(dsm.shape, np.uint8), workers=2)
    assert prov["method"] in ("bulldozer", "morphological")
    if prov["method"] == "morphological":
        assert "reason" in prov


def test_requiring_bulldozer_when_it_is_absent_raises_rather_than_falling_back(monkeypatch):
    """A benchmark must not silently measure a different estimator."""
    monkeypatch.setattr(DTM, "available", lambda: False)
    dsm, _ = terrain_with_buildings()
    with pytest.raises(DTM.BulldozerUnavailable):
        DTM.extract(dsm, Meta(), required=True)


def test_the_fallback_produces_a_surface_and_records_why(monkeypatch):
    monkeypatch.setattr(DTM, "available", lambda: False)
    dsm, _ = terrain_with_buildings()
    out, prov = DTM.extract(dsm, Meta(), sem=np.zeros(dsm.shape, np.uint8))
    assert out.shape == dsm.shape
    assert prov["method"] == "morphological" and "not installed" in prov["reason"]


def test_a_scene_with_no_georeferencing_still_gets_the_right_pixel_size():
    """Every scale Bulldozer works at is in metres; an identity affine reads 1 m."""
    if not DTM.available():
        pytest.skip("bulldozer is not installed")
    dsm, _ = terrain_with_buildings()
    a, _ = DTM.extract(dsm, Meta(georef=True), workers=2)
    b, _ = DTM.extract(dsm, Meta(georef=False), workers=2)
    assert np.allclose(a, b, atol=1e-3, equal_nan=True)


def test_holes_in_the_surface_stay_holes():
    """NaN is not an elevation and zero is; the anchor harvester tests finiteness."""
    if not DTM.available():
        pytest.skip("bulldozer is not installed")
    dsm, _ = terrain_with_buildings()
    dsm[10:14, 10:14] = np.nan
    out, _ = DTM.extract(dsm, Meta(), workers=2)
    assert np.isnan(out[10:14, 10:14]).all()
    assert np.isfinite(out[40:50, 40:50]).all()


def test_the_default_object_size_is_the_measured_one():
    """15 m is the minimum of the four-scene sweep, not a guess about building width."""
    assert DTM.DEFAULT_MAX_OBJECT_SIZE_M == 15.0


# --------------------------------------------------------------------- DEM
def test_a_tile_is_named_for_its_south_west_corner():
    assert "N47_00_E008_00" in DEM.tile_name(47, 8)
    assert "N00_00_E000_00" in DEM.tile_name(0, 0)


def test_southern_and_western_tiles_use_floor_not_truncation():
    """int() truncates toward zero and would name the wrong neighbouring tile."""
    assert "S01_00" in DEM.tile_name(-1, 8)
    assert "W123_00" in DEM.tile_name(47, -123)


def test_every_tile_a_box_touches_is_listed():
    got = DEM.tiles_for_bounds((7.9, 46.9, 8.1, 47.1))
    assert set(got) == {(46, 7), (46, 8), (47, 7), (47, 8)}


def test_a_single_degree_box_needs_one_tile():
    assert DEM.tiles_for_bounds((8.52, 47.36, 8.54, 47.38)) == [(47, 8)]


def test_the_url_is_the_aws_open_data_mirror():
    url = DEM.tile_url(47, 8)
    assert url.startswith("https://copernicus-dem-30m.s3.amazonaws.com/")
    assert url.endswith("_DEM.tif")


def test_products_needing_a_login_say_so_instead_of_returning_a_403(tmp_path):
    """NASADEM is behind Earthdata; a fetcher would have to prompt or embed a secret."""
    with pytest.raises(DEM.DEMUnavailable, match="Earthdata"):
        DEM.fetch_tile(47, 8, product="nasadem", cache_dir=str(tmp_path),
                       allow_network=True)


def test_an_unknown_product_lists_the_known_ones(tmp_path):
    with pytest.raises(DEM.DEMUnavailable, match="copernicus"):
        DEM.fetch_tile(47, 8, product="invented", cache_dir=str(tmp_path))


def test_an_uncached_tile_without_network_is_an_error_not_a_silent_skip(tmp_path):
    with pytest.raises(DEM.DEMUnavailable, match="not cached"):
        DEM.fetch_tile(47, 8, cache_dir=str(tmp_path), allow_network=False)


def test_a_tile_the_product_does_not_publish_is_remembered_as_absent(tmp_path):
    """Copernicus publishes nothing over open ocean; a coastal scene touches one."""
    marker = os.path.join(str(tmp_path), DEM.tile_name(0, -140) + ".tif.missing")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("HTTP 404\n")
    assert DEM.fetch_tile(0, -140, cache_dir=str(tmp_path), allow_network=False) is None


def test_a_scene_without_georeferencing_cannot_be_given_a_global_tile():
    class Bare:
        georeferenced = False
        bounds_wgs = None

    with pytest.raises(DEM.DEMUnavailable, match="not georeferenced"):
        DEM.load_for_scene(Bare(), (16, 16))


def test_the_datasheet_travels_with_the_product():
    """The vertical datum is EGM2008 and GLO-30 is a DSM. Both matter downstream."""
    text = DEM.describe("copernicus")
    assert "EGM2008" in text and "DSM" in text
