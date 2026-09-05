"""Facade texture coordinates, and the defect they exist to remove.

The bug is arithmetic, so the test for it is arithmetic. `structural._solid`
puts a wall's foot at the same (row, col) as its head - only z differs - and
`webmesh.write` derives UV from world x and y. Both ends of every wall therefore
receive the same texel, taken from the footprint boundary, and a footprint
boundary next to a street *is* the street.

`test_before_the_fix_a_wall_samples_the_street` is the important one here. It
asserts the defect still exists in the unfixed path, so that if someone
"fixes" `webmesh` some other way, this suite says so rather than quietly
passing on both sides.
"""
from __future__ import annotations

import numpy as np

from traksha.core.types import BARE_GROUND, BUILDING, ROAD, VEGETATION
from traksha.mesh import structural as S
from traksha.mesh import uvmap as U

GSD = 0.5


def city(h=64, w=64, ground=100.0, height=20.0, box=(20, 20, 44, 44)):
    """One building on flat ground, with a road running along its east wall."""
    dsm = np.full((h, w), ground, np.float32)
    r0, c0, r1, c1 = box
    dsm[r0:r1, c0:c1] = ground + height
    mask = np.zeros((h, w), bool)
    mask[r0:r1, c0:c1] = True

    sem = np.full((h, w), BARE_GROUND, np.uint8)
    sem[mask] = BUILDING
    # The road abuts the east wall and its rasterised width reaches the
    # footprint's own boundary column, which is the real geometry: `road_mask`
    # widens a centreline to the carriageway and a carriageway meets the
    # building line. That overlap is exactly where the defect lives.
    road = np.zeros((h, w), bool)
    road[:, c1 - 1:c1 + 6] = True
    sem[road & ~mask] = ROAD
    return dsm, mask, sem, road


def built(dsm, mask, sem):
    dtm = np.full(dsm.shape, float(dsm.min()), np.float32)
    b = S.measure(1, mask, dsm, dtm, sem)
    assert b is not None
    ndsm = (dsm - dtm).astype(np.float32)
    return S.build(dsm, ndsm, [b], GSD), [b]


# ---------------------------------------------------------------- the defect
def test_the_builder_records_where_a_roof_stops_and_a_wall_starts():
    """Nothing else in the mesh can recover it: the two share their top vertices."""
    dsm, mask, sem, _ = city()
    mesh, _ = built(dsm, mask, sem)
    spans = mesh["vertex_spans"]
    first, n_v, n_roof = spans["building_1"]
    assert 0 < n_roof < n_v


def test_before_the_fix_a_wall_samples_the_street():
    """The defect itself, asserted, so a change elsewhere cannot hide it."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    before = U.audit(mesh, dsm.shape, GSD, None, road, blds)
    assert before["wall_vertices"] > 0
    assert before["sampling_road"] > 0.05


def test_after_the_fix_no_wall_samples_the_street():
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    guv, _ = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds,
                              sem=sem, road_mask=road)
    after = U.audit(mesh, dsm.shape, GSD, guv, road, blds)
    assert after["sampling_road"] == 0.0


def test_every_wall_of_one_building_gets_the_same_texel():
    """'Single colour' is the requirement; identical UVs are what deliver it."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    guv, _ = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds,
                              sem=sem, road_mask=road)
    uv = guv["building_1"]
    first, n_v, n_roof = mesh["vertex_spans"]["building_1"]
    feet = uv[n_roof:]
    assert len(feet) > 3
    assert np.allclose(feet, feet[0])


def test_the_facade_texel_lands_inside_the_building():
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    guv, _ = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds,
                              sem=sem, road_mask=road)
    after = U.audit(mesh, dsm.shape, GSD, guv, road, blds)
    assert after["sampling_own_footprint"] == 1.0


# ------------------------------------------------------------ the conditions
def test_the_sample_avoids_a_road_that_crosses_the_footprint():
    """The conditional half: a mislabelled road pixel must not colour the wall."""
    mask = np.zeros((64, 64), bool)
    mask[20:44, 20:44] = True
    road = np.zeros((64, 64), bool)
    road[30:34, :] = True               # straight through the middle
    point, why = U.facade_sample(mask, None, road)
    assert point is not None
    assert not road[point]
    assert "OSM road network" in why


def test_the_sample_avoids_vegetation_and_says_so():
    mask = np.zeros((64, 64), bool)
    mask[20:44, 20:44] = True
    sem = np.full((64, 64), BUILDING, np.uint8)
    sem[20:36, 20:44] = VEGETATION      # a tree overhanging most of the roof
    point, why = U.facade_sample(mask, sem, None)
    assert sem[point] != VEGETATION
    assert "vegetation" in why


def test_a_footprint_with_no_admissible_pixel_is_reported_not_silently_coloured():
    mask = np.zeros((32, 32), bool)
    mask[10:20, 10:20] = True
    sem = np.full((32, 32), ROAD, np.uint8)
    point, why = U.facade_sample(mask, sem, None)
    assert point is not None
    assert "no admissible interior pixel" in why


def test_an_empty_footprint_has_no_sample():
    point, why = U.facade_sample(np.zeros((16, 16), bool))
    assert point is None and "empty" in why


# ------------------------------------------------------------------ the rim
def test_the_roof_rim_is_pulled_off_the_boundary_pixel():
    """The mixed pixel at a footprint edge is half roof and half street."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    guv, rep = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds,
                                sem=sem, road_mask=road)
    assert rep["rims_inset"] == 1
    # The roof's UV extent shrinks, because every rim sample moved inward.
    V = np.asarray(mesh["vertices"])
    F = np.asarray(mesh["triangles"], np.int64)
    name, first, count = next(g for g in mesh["groups"] if g[0] == "building_1")
    idx = np.unique(F[first:first + count])
    n_roof = mesh["vertex_spans"]["building_1"][2]
    plain = U._world_uv(V[idx][:n_roof, 0], V[idx][:n_roof, 1],
                        (dsm.shape[1] - 1) * GSD, (dsm.shape[0] - 1) * GSD)
    fixed = guv["building_1"][:n_roof]
    assert np.ptp(fixed[:, 0]) < np.ptp(plain[:, 0])


def test_the_inset_never_crosses_the_centroid():
    """A building narrower than twice the inset must not collapse to one texel."""
    dsm, mask, sem, road = city(box=(26, 26, 38, 38))       # 6 m across
    mesh, blds = built(dsm, mask, sem)
    guv, _ = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds,
                              sem=sem, road_mask=road, inset_m=50.0)
    n_roof = mesh["vertex_spans"]["building_1"][2]
    roof_uv = guv["building_1"][:n_roof]
    assert np.ptp(roof_uv[:, 0]) > 0


# ---------------------------------------------------------------- robustness
def test_overrides_are_ordered_the_way_webmesh_reads_them():
    """`write` recovers a group's vertices with np.unique and assigns positionally.
    Any other order paints the overrides onto the wrong vertices, and it renders."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    guv, _ = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds)
    F = np.asarray(mesh["triangles"], np.int64)
    for name, first, count in mesh["groups"]:
        if name in guv:
            assert len(guv[name]) == len(np.unique(F[first:first + count]))


def test_the_wall_split_survives_losing_the_recorded_spans():
    """The browser copy is decimated and renumbered; height is what still works."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    stripped = dict(mesh)
    stripped.pop("vertex_spans")
    guv, rep = U.build_group_uv(stripped, dsm.shape, GSD, buildings=blds,
                                sem=sem, road_mask=road)
    assert rep["walls_flattened"] == 1
    assert U.audit(stripped, dsm.shape, GSD, guv, road, blds)["sampling_road"] == 0.0


def test_a_mesh_with_no_groups_is_skipped_with_a_reason():
    guv, rep = U.build_group_uv(
        {"vertices": np.zeros((0, 3)), "triangles": np.zeros((0, 3), np.int64)},
        (8, 8), GSD)
    assert guv == {} and "skipped" in rep


def test_terrain_is_never_given_an_override():
    """Terrain is photographed from above; the planar projection is correct for it."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    guv, _ = U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds)
    assert "terrain" not in guv


def test_the_fix_moves_no_geometry():
    """Only where a vertex reads its colour changes. Not where it is."""
    dsm, mask, sem, road = city()
    mesh, blds = built(dsm, mask, sem)
    before = np.array(mesh["vertices"], copy=True)
    U.build_group_uv(mesh, dsm.shape, GSD, buildings=blds, sem=sem, road_mask=road)
    assert np.array_equal(mesh["vertices"], before)
