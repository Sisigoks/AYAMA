"""Fitting the mesh into a triangle budget without destroying it.

Decimation is where a structural mesh is easiest to ruin: collapse across a
footprint boundary and the wall goes with it, collapse two buildings together
and the separation the whole stage exists for is gone. So these check what
survives, not just that the count came down.

PyMeshLab is optional, so every test here skips cleanly without it - and the
fallback path it skips to is itself tested, because that is what most installs
will actually run.
"""
from __future__ import annotations

import numpy as np
import pytest

from traksha.mesh import decimate as D
from traksha.mesh import quality as Q
from traksha.mesh import structural as S

needs_pymeshlab = pytest.mark.skipif(not D.available(), reason="pymeshlab not installed")


def two_buildings(n=96, ground=100.0):
    dsm = np.full((n, n), ground, np.float32)
    dsm[12:44, 12:44] = ground + 16.0
    dsm[12:44, 56:88] = ground + 24.0
    ndsm = (dsm - ground).astype(np.float32)
    dtm = dsm - ndsm
    m1 = np.zeros((n, n), bool); m1[12:44, 12:44] = True
    m2 = np.zeros((n, n), bool); m2[12:44, 56:88] = True
    bs = [S.measure(1, m1, dsm, dtm), S.measure(2, m2, dsm, dtm)]
    return S.build(dsm, ndsm, bs, gsd_m=1.0)


def test_without_pymeshlab_it_declines_rather_than_guesses():
    if D.available():
        pytest.skip("pymeshlab is installed here")
    assert D.simplify(two_buildings(), 100) is None


def test_a_mesh_already_inside_the_budget_is_left_alone():
    mesh = two_buildings()
    assert D.simplify(mesh, len(mesh["triangles"]) + 1) is None


@needs_pymeshlab
def test_it_reaches_the_budget():
    mesh = two_buildings()
    out = D.simplify(mesh, len(mesh["triangles"]) // 3)
    assert out is not None
    assert len(out["triangles"]) < len(mesh["triangles"])
    assert out["decimated"]["from"] == len(mesh["triangles"])


@needs_pymeshlab
def test_the_buildings_stay_separate():
    """The failure that matters: a collapse that welds two objects together."""
    mesh = two_buildings()
    out = D.simplify(mesh, len(mesh["triangles"]) // 3)
    before, after = Q.separation(mesh), Q.separation(out)
    assert after["buildings"] == before["buildings"]
    assert after["shared_vertices"] == 0
    assert after["own_component"] == 1.0


@needs_pymeshlab
def test_the_group_table_survives():
    """Lose it and which triangles are which building is lost with it."""
    mesh = two_buildings()
    out = D.simplify(mesh, len(mesh["triangles"]) // 3)
    assert [g[0] for g in out["groups"]] == [g[0] for g in mesh["groups"]]
    total = sum(g[2] for g in out["groups"])
    assert total == len(out["triangles"]), "the group offsets do not cover the mesh"
    for _, first, count in out["groups"]:
        assert out["triangles"][first:first + count].max() < len(out["vertices"])


@needs_pymeshlab
def test_the_facades_survive():
    """Vertical area is the signature of the rebuild; a decimator can flatten it."""
    mesh = two_buildings()
    out = D.simplify(mesh, len(mesh["triangles"]) // 3)
    before = Q.verticality(mesh["vertices"], mesh["triangles"])
    after = Q.verticality(out["vertices"], out["triangles"])
    assert after["max_slope_deg"] == pytest.approx(90.0, abs=1e-3)
    assert after["wall_area_m2"] > 0.5 * before["wall_area_m2"], "half the facades went"


@needs_pymeshlab
def test_the_result_is_still_well_formed():
    mesh = two_buildings()
    out = D.simplify(mesh, len(mesh["triangles"]) // 3)
    rep = Q.validate(out["vertices"], out["triangles"])
    assert rep["degenerate_faces"] == 0
    assert rep["finite"]
    assert Q.normal_consistency(out["vertices"], out["triangles"]) == 1.0


@needs_pymeshlab
def test_height_is_not_invented():
    """Collapse moves vertices; it must not move them outside the surface."""
    mesh = two_buildings()
    out = D.simplify(mesh, len(mesh["triangles"]) // 3)
    z0 = mesh["vertices"][:, 2]
    z1 = out["vertices"][:, 2]
    assert z1.min() >= z0.min() - 1e-3
    assert z1.max() <= z0.max() + 1e-3
