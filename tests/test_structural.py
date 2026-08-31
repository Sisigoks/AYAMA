"""The structural mesh rebuild, and the metrics that judge it.

The claim under test is narrow and checkable: buildings come out as separate
objects with vertical facades, built from calibrated heights and nothing else.
Every test here either proves a property of the geometry or proves a rejection -
because the failure mode of an extruder is not a crash, it is a confident solid
where the image says there is none.
"""
from __future__ import annotations

import numpy as np
import pytest

from traksha.core.types import BARE_GROUND, BUILDING, VEGETATION
from traksha.mesh import quality as Q
from traksha.mesh import structural as S


def scene(h=48, w=48, ground=100.0):
    dsm = np.full((h, w), ground, np.float32)
    return dsm


def block(dsm, r0, c0, r1, c1, height):
    dsm[r0:r1, c0:c1] = dsm.min() + height
    m = np.zeros(dsm.shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def ndsm_of(dsm, ground=100.0):
    return (dsm - ground).astype(np.float32)


# ------------------------------------------------------------- measurement
def test_a_tall_footprint_becomes_a_building_with_the_height_it_has():
    dsm = scene()
    m = block(dsm, 10, 10, 30, 34, 18.0)
    b = S.measure(1, m, dsm, dsm - ndsm_of(dsm))
    assert b is not None
    assert b.height_m == pytest.approx(18.0, abs=0.01)
    assert b.ground_m == pytest.approx(100.0, abs=0.01)
    assert b.roof_mad_m == pytest.approx(0.0, abs=1e-6)


def test_a_low_footprint_is_refused():
    """A kerb, a parked car, or segmentation leakage - not a building."""
    dsm = scene()
    m = block(dsm, 10, 10, 30, 34, 1.0)
    assert S.measure(1, m, dsm, dsm - ndsm_of(dsm)) is None


def test_a_tiny_footprint_is_refused():
    dsm = scene()
    m = block(dsm, 10, 10, 13, 13, 20.0)
    assert S.measure(1, m, dsm, dsm - ndsm_of(dsm)) is None


def test_a_tall_stand_of_vegetation_is_refused():
    """SAM 2 has no classes. Trees are tall, and extruding one is a fabrication."""
    dsm = scene()
    m = block(dsm, 10, 10, 30, 34, 15.0)
    sem = np.full(dsm.shape, BARE_GROUND, np.uint8)
    sem[m] = VEGETATION
    assert S.measure(1, m, dsm, dsm - ndsm_of(dsm), sem) is None


def test_a_mostly_built_footprint_with_some_greenery_is_kept_but_flagged():
    dsm = scene()
    m = block(dsm, 10, 10, 30, 34, 15.0)
    sem = np.full(dsm.shape, BUILDING, np.uint8)
    green = np.zeros(dsm.shape, bool)
    green[10:18, 10:34] = True                     # 40% of the footprint
    sem[green] = VEGETATION
    b = S.measure(1, m, dsm, dsm - ndsm_of(dsm), sem)
    assert b is not None and "vegetation" in b.note


def test_height_is_measured_against_local_ground_not_a_global_datum():
    """On a slope, a global minimum would make downhill buildings enormous."""
    h = w = 48
    ramp = np.tile(np.linspace(100.0, 140.0, w, dtype=np.float32), (h, 1))
    dsm = ramp.copy()
    m = np.zeros((h, w), bool)
    m[10:30, 30:44] = True
    dsm[m] = ramp[m] + 12.0                        # 12 m above the slope it sits on
    dtm = ramp
    b = S.measure(1, m, dsm, dtm)
    assert b is not None
    assert b.height_m == pytest.approx(12.0, abs=1.5), "ground was read globally"


def test_a_flat_roof_is_found_planar_and_a_noisy_one_is_not():
    dsm = scene()
    flat = block(dsm, 10, 10, 34, 34, 20.0)
    assert S.measure(1, flat, dsm, dsm - ndsm_of(dsm)).planar

    rough = scene()
    m = block(rough, 10, 10, 34, 34, 20.0)
    rng = np.random.default_rng(0)
    rough[m] += rng.normal(0, 3.0, int(m.sum())).astype(np.float32)
    assert not S.measure(1, m, rough, np.full(rough.shape, 100.0, np.float32)).planar


# ------------------------------------------------------------- footprints
def test_two_blobs_in_one_instance_become_two_buildings():
    """A SAM mask can span a courtyard. Two roofs are two buildings."""
    cells = np.zeros((40, 40), bool)
    cells[5:15, 5:15] = True
    cells[5:15, 25:35] = True
    parts = S.pieces(cells)
    assert len(parts) == 2
    assert all(p.sum() == 100 for p in parts)


def test_parts_touching_only_at_a_corner_are_separated():
    """A diagonal touch is a pinch, and a pinch is a non-manifold vertex."""
    cells = np.zeros((30, 30), bool)
    cells[4:12, 4:12] = True
    cells[12:20, 12:20] = True                     # corner-to-corner
    assert len(S.pieces(cells, min_cells=8)) == 2


def test_unpinch_removes_a_diagonal_pinch():
    cells = np.zeros((10, 10), bool)
    cells[2:5, 2:5] = True
    cells[5:8, 5:8] = True
    fixed = S.unpinch(cells)
    a = fixed[:-1, :-1] & fixed[1:, 1:] & ~fixed[:-1, 1:] & ~fixed[1:, :-1]
    b = fixed[:-1, 1:] & fixed[1:, :-1] & ~fixed[:-1, :-1] & ~fixed[1:, 1:]
    assert not a.any() and not b.any()
    assert fixed.sum() < cells.sum(), "nothing was removed"


def test_the_roof_is_carried_out_of_its_trusted_interior():
    """Depth blurs across an edge, so the ring inside a footprint is a blend.

    Taken at face value it makes the roof sag at its own boundary and the wall
    tops come out serrated, which is visible in any oblique view.
    """
    dsm = scene()
    m = block(dsm, 10, 10, 30, 30, 20.0)
    dsm[10, 10:30] = 108.0                         # a contaminated edge row
    inner = S.core(m)
    assert not inner[10, 15], "the edge row is not supposed to be trusted"
    filled = S._extend_from_core(dsm, m, inner)
    assert filled[10, 15] == pytest.approx(120.0), "the sag was not repaired"
    assert filled[40, 15] == pytest.approx(100.0), "ground outside was altered"


# --------------------------------------------------------------- geometry
def built_scene():
    dsm = scene()
    m = block(dsm, 10, 25, 34, 40, 20.0)
    nd = ndsm_of(dsm)
    b = S.measure(1, m, dsm, dsm - nd)
    return dsm, nd, [b]


def test_the_rebuilt_mesh_has_genuinely_vertical_facades():
    dsm, nd, bs = built_scene()
    mesh = S.build(dsm, nd, bs, gsd_m=1.0)
    v = Q.verticality(mesh["vertices"], mesh["triangles"])
    assert v["max_slope_deg"] == pytest.approx(90.0, abs=1e-6)
    assert v["wall_area_m2"] > 0


def test_the_terrain_does_not_ramp_up_to_the_building():
    """Cells share grid vertices with the footprint they border.

    Reading the DSM there would carry roof height into the terrain and put the
    ramp back one cell outside every building - which is the defect this whole
    stage exists to remove.
    """
    dsm, nd, bs = built_scene()
    mesh = S.build(dsm, nd, bs, gsd_m=1.0)
    name, first, count = mesh["groups"][0]
    assert name == "terrain"
    fn, _ = Q.face_normals(mesh["vertices"], mesh["triangles"][first:first + count])
    assert np.abs(fn[:, 2]).min() == pytest.approx(1.0, abs=1e-6)


def test_each_building_is_its_own_connected_component():
    dsm = scene(64, 64)
    m1 = block(dsm, 8, 8, 26, 26, 15.0)
    m2 = block(dsm, 8, 38, 26, 56, 22.0)
    nd = ndsm_of(dsm)
    dtm = dsm - nd
    bs = [S.measure(1, m1, dsm, dtm), S.measure(2, m2, dsm, dtm)]
    mesh = S.build(dsm, nd, bs, gsd_m=1.0)

    sep = Q.separation(mesh)
    assert sep["buildings"] == 2
    assert sep["own_component"] == 1.0
    assert sep["shared_vertices"] == 0
    assert sep["separation_score"] == 1.0


def test_the_rebuild_is_well_formed():
    dsm, nd, bs = built_scene()
    mesh = S.build(dsm, nd, bs, gsd_m=1.0)
    rep = Q.validate(mesh["vertices"], mesh["triangles"])
    assert rep["degenerate_faces"] == 0
    assert rep["non_manifold_edges"] == 0
    assert rep["unreferenced_vertices"] == 0
    assert rep["finite"]
    assert Q.normal_consistency(mesh["vertices"], mesh["triangles"]) == 1.0


def test_no_height_is_invented():
    """Every vertex sits on a calibrated value: the roof, or the ground under it."""
    dsm, nd, bs = built_scene()
    mesh = S.build(dsm, nd, bs, gsd_m=1.0)
    z = mesh["vertices"][:, 2]
    assert z.min() == pytest.approx(float(dsm.min()), abs=1e-6)
    assert z.max() == pytest.approx(float(dsm.max()), abs=1e-6)


def test_a_scene_with_no_buildings_is_just_the_terrain():
    dsm = scene()
    mesh = S.build(dsm, np.zeros_like(dsm), [], gsd_m=1.0)
    assert [g[0] for g in mesh["groups"]] == ["terrain"]
    assert Q.separation(mesh)["buildings"] == 0


# ----------------------------------------------------- against the baseline
def test_the_sheet_it_replaces_fails_the_separation_it_passes():
    """The comparison that justifies the stage.

    The height-field mesh is one connected component containing every building
    and the ground between them, and it has no vertical area at all. That is not
    a tuning difference: no threshold turns one sheet into separate solids.
    """
    from traksha.mesh.adaptive import adaptive_mesh

    dsm = scene(64, 64)
    m1 = block(dsm, 8, 8, 26, 26, 15.0)
    m2 = block(dsm, 8, 38, 26, 56, 22.0)
    nd = ndsm_of(dsm)
    dtm = dsm - nd

    sheet = adaptive_mesh(dsm.astype(np.float64), 1.0, tol_m=0.2, block=4)
    # adaptive_mesh emits 1-based indices, ready for an OBJ face line.
    base = {"vertices": sheet["vertices"], "triangles": sheet["triangles"] - 1,
            "groups": []}
    ncomp, _ = Q.components(base["vertices"], base["triangles"])
    assert ncomp == 1, "the baseline was supposed to be one sheet"
    assert Q.verticality(base["vertices"], base["triangles"])["wall_area_m2"] == 0.0

    bs = [S.measure(1, m1, dsm, dtm), S.measure(2, m2, dsm, dtm)]
    mesh = S.build(dsm, nd, bs, gsd_m=1.0)
    ncomp2, _ = Q.components(mesh["vertices"], mesh["triangles"])
    assert ncomp2 > ncomp
    assert Q.verticality(mesh["vertices"], mesh["triangles"])["wall_area_m2"] > 0.0
