"""OpenStreetMap as a geometric prior: projection, rasterisation, and the gate.

Nothing here touches the network. Overpass responses are small, well-specified
JSON, so the tests build one and assert on what the module does with it - which
is the part that can be wrong. The one thing a live call would test that this
does not is whether Overpass is up, and that is not a property of this code.

The claims under test are the ones the pipeline actually relies on:

* a lat/lon way lands on the right pixels, through a projected CRS;
* a filled polygon is the polygon, not a half-pixel-larger or smaller one,
  because that difference moves every IoU the regulariser gates on;
* elevated ways are excluded from the bare-earth mask, because a bridge deck is
  not the terrain;
* refining the semantic raster never overrules water, and applies footprints
  before roads.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from traksha.core.types import BARE_GROUND, BUILDING, ROAD, VEGETATION, WATER
from traksha.data import osm


class FakeMeta:
    """A north-up metre-based scene at a known place, so pixels are checkable by hand."""

    georeferenced = True
    crs = "EPSG:3857"
    gsd_m = 1.0
    bounds_wgs = (8.0, 47.0, 8.01, 47.01)

    def __init__(self, transform):
        self.transform = transform


def _meta_for(lon0, lat0, size=64, gsd=2.0):
    """A scene whose north-west corner is at (lon0, lat0), in Web Mercator."""
    from rasterio.warp import transform as warp

    xs, ys = warp("EPSG:4326", "EPSG:3857", [lon0], [lat0])
    m = FakeMeta((gsd, 0.0, xs[0], 0.0, -gsd, ys[0]))
    m.gsd_m = gsd
    return m


def _way(nodes, tags, kind="way", wid=1):
    return {"type": kind, "id": wid, "tags": tags,
            "geometry": [{"lat": la, "lon": lo} for la, lo in nodes]}


# ---------------------------------------------------------------- the query
def test_the_bounding_box_is_reordered_for_overpass():
    """Overpass takes (s, w, n, e); everything else here uses (w, s, e, n)."""
    q = osm.build_query((8.0, 47.0, 8.5, 47.5))
    assert "(47.0,8.0,47.5,8.5)" in q


def test_cache_keys_are_stable_and_distinguish_boxes():
    a = osm.cache_key((8.0, 47.0, 8.5, 47.5))
    assert a == osm.cache_key((8.0, 47.0, 8.5, 47.5))
    assert a != osm.cache_key((8.0, 47.0, 8.5, 47.6))


def test_a_missing_cache_entry_without_network_is_an_error_not_an_empty_layer(tmp_path):
    """Silently returning nothing is indistinguishable from a tile with no buildings."""
    with pytest.raises(osm.OSMUnavailable):
        osm.fetch((8.0, 47.0, 8.01, 47.01), allow_network=False,
                  cache_dir=str(tmp_path))


def test_no_bounds_is_an_error_with_a_reason(tmp_path):
    with pytest.raises(osm.OSMUnavailable, match="WGS84 bounds"):
        osm.fetch(None, allow_network=False, cache_dir=str(tmp_path))


def test_a_cached_extract_is_read_without_network(tmp_path):
    bounds = (8.0, 47.0, 8.01, 47.01)
    payload = {"elements": [], "traksha": {"fetched_utc": "2026-01-01T00:00:00Z"}}
    path = os.path.join(str(tmp_path), osm.cache_key(bounds) + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    got = osm.fetch(bounds, allow_network=False, cache_dir=str(tmp_path))
    assert got["traksha"]["fetched_utc"] == "2026-01-01T00:00:00Z"


# ----------------------------------------------------------- to pixel space
def test_a_way_projects_onto_the_pixels_it_covers():
    """The whole hop: lat/lon -> projected CRS -> affine -> (row, col)."""
    meta = _meta_for(8.0, 47.0, gsd=2.0)
    nodes = [(46.9999, 8.0001), (46.9999, 8.0004),
             (46.9996, 8.0004), (46.9996, 8.0001), (46.9999, 8.0001)]
    layer = osm.to_pixels({"elements": [_way(nodes, {"building": "yes"})]},
                          meta, (64, 64))
    assert len(layer.buildings) == 1
    ring = layer.buildings[0]
    assert ring.min() > -1 and ring.max() < 64
    assert osm.fill_polygon(ring, (64, 64)).sum() > 0


def test_going_south_increases_the_row_and_going_east_increases_the_column():
    """The invariant a misapplied affine breaks, and the one a plausible-looking
    cluster of footprints in the wrong place would not reveal on its own."""
    meta = _meta_for(8.0, 47.0, gsd=2.0)

    def place(lat, lon):
        el = _way([(lat, lon), (lat, lon + 0.0002),
                   (lat - 0.0002, lon + 0.0002), (lat - 0.0002, lon)],
                  {"building": "yes"})
        return osm.to_pixels({"elements": [el]}, meta, (64, 64)).buildings[0].mean(0)

    north, south = place(46.9998, 8.0002), place(46.9994, 8.0002)
    west, east = place(46.9998, 8.0002), place(46.9998, 8.0006)
    assert south[0] > north[0]          # +row is south
    assert east[1] > west[1]            # +col is east


def test_a_scene_without_georeferencing_cannot_place_osm():
    class Bare:
        georeferenced = False
        crs = None
        transform = None

    with pytest.raises(osm.OSMUnavailable, match="not georeferenced"):
        osm.to_pixels({"elements": []}, Bare(), (16, 16))


def test_relations_contribute_their_outer_rings_only():
    """An inner ring is a courtyard, and structural._solid cannot express a hole."""
    meta = _meta_for(8.0, 47.0)
    outer = [{"lat": 46.9999, "lon": 8.0001}, {"lat": 46.9999, "lon": 8.0005},
             {"lat": 46.9995, "lon": 8.0005}, {"lat": 46.9995, "lon": 8.0001}]
    rel = {"type": "relation", "id": 5, "tags": {"building": "yes"},
           "members": [{"type": "way", "role": "outer", "geometry": outer},
                       {"type": "way", "role": "inner", "geometry": outer}]}
    layer = osm.to_pixels({"elements": [rel]}, meta, (64, 64))
    assert len(layer.buildings) == 1


def test_road_width_is_read_from_tags_then_lanes_then_class():
    assert osm._road_width({"highway": "residential", "width": "9"}) == 9.0
    assert osm._road_width({"highway": "residential", "lanes": "4"}) == 12.0
    assert osm._road_width({"highway": "motorway"}) == osm.ROAD_WIDTH_M["motorway"]
    assert osm._road_width({"highway": "nonsense"}) == osm.DEFAULT_ROAD_WIDTH_M


# --------------------------------------------------------------- rasterising
def test_a_filled_square_is_the_square():
    """Half a pixel of drift at every edge moves every IoU the regulariser gates on."""
    ring = np.array([[10.0, 10.0], [10.0, 20.0], [20.0, 20.0], [20.0, 10.0]])
    m = osm.fill_polygon(ring, (32, 32))
    assert m[10:20, 10:20].all()
    assert m.sum() == 100


def test_a_filled_triangle_has_about_half_the_area_of_its_box():
    ring = np.array([[4.0, 4.0], [4.0, 24.0], [24.0, 4.0]])
    m = osm.fill_polygon(ring, (32, 32))
    assert 170 <= m.sum() <= 230        # 200 exactly, allowing the boundary rule


def test_a_degenerate_ring_fills_nothing():
    assert osm.fill_polygon(np.array([[1.0, 1.0], [2.0, 2.0]]), (8, 8)).sum() == 0


def test_road_width_widens_the_centreline():
    layer = osm.OSMLayer(roads=[np.array([[10.0, 2.0], [10.0, 30.0]])],
                         road_width_m=[8.0], road_elevated=[False])
    m = osm.road_mask(layer, (32, 32), gsd_m=1.0)
    # 8 m at 1 m per pixel is about 8 rows of coverage across the line.
    covered = m[:, 15].sum()
    assert 6 <= covered <= 10


def test_bridges_and_tunnels_are_excluded_from_the_bare_earth_mask():
    """A bridge deck is metres above the terrain; a DEM anchor there is confidently wrong."""
    layer = osm.OSMLayer(roads=[np.array([[10.0, 2.0], [10.0, 30.0]])],
                         road_width_m=[8.0], road_elevated=[True])
    assert not osm.road_mask(layer, (32, 32), 1.0).any()
    assert osm.road_mask(layer, (32, 32), 1.0, include_elevated=True).any()


# ------------------------------------------------------------------ the gate
def _layer_with(building=None, road=None):
    layer = osm.OSMLayer()
    if building is not None:
        layer.buildings.append(building)
    if road is not None:
        layer.roads.append(road)
        layer.road_width_m.append(6.0)
        layer.road_elevated.append(False)
    return layer


def test_footprints_demote_ground_to_building():
    """A grey roof read as bare ground is where a DEM anchor lands on a rooftop."""
    sem = np.full((32, 32), BARE_GROUND, np.uint8)
    ring = np.array([[8.0, 8.0], [8.0, 20.0], [20.0, 20.0], [20.0, 8.0]])
    out, rep = osm.refine_semantics(sem, _layer_with(building=ring), 1.0)
    assert (out[9:19, 9:19] == BUILDING).all()
    assert rep["demoted_building_px"] > 0


def test_roads_promote_vegetation_to_road():
    sem = np.full((32, 32), VEGETATION, np.uint8)
    line = np.array([[16.0, 2.0], [16.0, 30.0]])
    out, rep = osm.refine_semantics(sem, _layer_with(road=line), 1.0)
    assert (out[16, 5:25] == ROAD).all()
    assert rep["promoted_road_px"] > 0


def test_water_is_never_overruled():
    """Flat-water anchors are the most reliable constraints in the system."""
    sem = np.full((32, 32), WATER, np.uint8)
    ring = np.array([[8.0, 8.0], [8.0, 20.0], [20.0, 20.0], [20.0, 8.0]])
    line = np.array([[16.0, 2.0], [16.0, 30.0]])
    out, _ = osm.refine_semantics(sem, _layer_with(building=ring, road=line), 1.0)
    assert (out == WATER).all()


def test_a_footprint_wins_where_it_overlaps_a_road():
    """Calling a roof a road puts a DEM anchor on it; the reverse is merely wasteful."""
    sem = np.full((32, 32), BARE_GROUND, np.uint8)
    ring = np.array([[8.0, 8.0], [8.0, 24.0], [24.0, 24.0], [24.0, 8.0]])
    line = np.array([[16.0, 2.0], [16.0, 30.0]])
    out, _ = osm.refine_semantics(sem, _layer_with(building=ring, road=line), 1.0)
    assert out[16, 16] == BUILDING


def test_refining_leaves_the_input_raster_untouched():
    sem = np.full((16, 16), BARE_GROUND, np.uint8)
    before = sem.copy()
    ring = np.array([[2.0, 2.0], [2.0, 12.0], [12.0, 12.0], [12.0, 2.0]])
    osm.refine_semantics(sem, _layer_with(building=ring), 1.0)
    assert np.array_equal(sem, before)
