"""Handing buildings to threefiner, and the guards around doing so.

The diffusion step needs a CUDA GPU and has never run on the machine these were
written on. What is tested is everything either side of it: which buildings get
extracted, the frame they are handed over in, the round trip back, and - most
importantly - the two independent guarantees that a measured wall does not get
deformed by a generative prior.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from traksha.mesh import facades as FA
from traksha.mesh import structural as S


def two_buildings(n=64, ground=100.0):
    dsm = np.full((n, n), ground, np.float32)
    dsm[8:26, 8:26] = ground + 15.0
    dsm[8:26, 38:56] = ground + 22.0
    ndsm = (dsm - ground).astype(np.float32)
    dtm = dsm - ndsm
    m1 = np.zeros((n, n), bool); m1[8:26, 8:26] = True
    m2 = np.zeros((n, n), bool); m2[8:26, 38:56] = True
    return S.build(dsm, ndsm, [S.measure(1, m1, dsm, dtm),
                               S.measure(2, m2, dsm, dtm)], gsd_m=1.0)


# --------------------------------------------------------------- the guards
def test_geometry_training_presets_are_refused():
    """The first of two guarantees, and the one with the reason attached.

    threefiner's default modes deform vertices to satisfy a diffusion prior. On
    a surface validated against lidar in metres, that converts a measurement
    into a guess, so those presets cannot be selected at all.
    """
    with pytest.raises(FA.FacadeUnavailable, match="trains geometry"):
        FA.refine_one(np.zeros((3, 3)), np.array([[0, 1, 2]]), "/tmp/x", "b",
                      preset="sd")
    for allowed in FA.FIXED_GEOMETRY_PRESETS:
        assert "fixgeo" in allowed


def test_only_fixed_geometry_presets_are_offered():
    assert FA.DEFAULT_PRESET in FA.FIXED_GEOMETRY_PRESETS
    assert all(p in FA.PRESET_NOTES for p in FA.FIXED_GEOMETRY_PRESETS)


def test_a_result_whose_vertex_count_moved_is_refused(tmp_path):
    """The second guarantee. `fix_geo` is supposed to leave vertices alone;
    'supposed to' is not something to promise about a metric surface."""
    trimesh = pytest.importorskip("trimesh")
    path = str(tmp_path / "wrong.glb")
    trimesh.Trimesh(vertices=np.random.rand(10, 3),
                    faces=np.array([[0, 1, 2], [3, 4, 5]])).export(path)
    with pytest.raises(FA.FacadeUnavailable, match="correspondence"):
        FA.read_texture(path, expect_vertices=999)


def test_refine_refuses_without_a_gpu_unless_asked_to_dry_run(tmp_path):
    checks = FA.preflight()
    if checks["ok"]:
        pytest.skip("this machine can actually run it")
    with pytest.raises(FA.FacadeUnavailable, match="needs"):
        FA.refine(two_buildings(), str(tmp_path))


# ------------------------------------------------------------- the handover
def test_buildings_come_out_largest_first():
    groups = FA.building_groups(two_buildings())
    assert [g[0] for g in groups] == sorted(
        [g[0] for g in groups], key=lambda n: -dict(
            (x[0], x[2]) for x in FA.building_groups(two_buildings()))[n])
    assert all(n.startswith("building_") for n, _, _ in groups)


def test_terrain_is_not_offered_for_refinement():
    """It is not an object, and threefiner orbits a camera around objects."""
    assert not any(n == "terrain" for n, _, _ in FA.building_groups(two_buildings()))


def test_an_extracted_building_is_a_standalone_mesh():
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    V, F = FA.extract(mesh, first, count)
    assert len(F) == count
    assert F.min() == 0 and F.max() == len(V) - 1, "indices were not renumbered"
    assert np.isfinite(V).all()


def test_the_object_frame_is_unit_scale_and_y_up():
    """threefiner orbits at radius 2.5 around a roughly unit, Y-up object.

    A 30 m building 800 m from the scene origin is neither, and handing one over
    unchanged puts it outside the camera entirely.
    """
    mesh = two_buildings()
    V, _ = FA.extract(mesh, *FA.building_groups(mesh)[0][1:])
    local, _, frame = FA.to_object_frame(V)
    assert max(np.ptp(local, axis=0)) == pytest.approx(1.8, abs=1e-6)
    centre = (local.max(0) + local.min(0)) / 2
    assert np.allclose(centre, 0, atol=1e-6), "not centred on the origin"
    # Z-up became Y-up: the tall axis is now index 1, not 2.
    assert np.ptp(V[:, 2]) > 0 and np.ptp(local[:, 1]) > 0
    assert frame["scale"] > 0


def test_the_frame_conversion_is_exactly_invertible():
    """Positions come back in metres, in the calibrated frame, or not at all."""
    mesh = two_buildings()
    V, _ = FA.extract(mesh, *FA.building_groups(mesh)[0][1:])
    local, inverse, _ = FA.to_object_frame(V)
    assert np.abs(inverse(local) - V).max() < 1e-9


# ------------------------------------------------------------- the artifact
def test_a_dry_run_prepares_the_handover_without_a_gpu(tmp_path):
    pytest.importorskip("trimesh")
    rec = FA.refine(two_buildings(), str(tmp_path), max_buildings=2, dry_run=True)
    assert len(rec["buildings"]) == 2
    for b in rec["buildings"]:
        assert os.path.exists(b["coarse"])
        assert b["triangles"] > 0


def test_the_artifact_says_it_is_synthesised(tmp_path):
    """A reader looking at a refined facade must be told what they are looking at."""
    pytest.importorskip("trimesh")
    FA.refine(two_buildings(), str(tmp_path), max_buildings=1, dry_run=True)
    rec = json.load(open(tmp_path / "facades.json", encoding="utf-8"))
    assert rec["synthesised"] is True
    assert "not measured" in rec["what"]
    assert rec["tool"].startswith("threefiner")
    assert rec["preset"] in FA.FIXED_GEOMETRY_PRESETS


def test_preflight_names_what_is_missing():
    checks = FA.preflight()
    assert isinstance(checks["missing"], list)
    if not checks["ok"]:
        assert all(isinstance(m, str) and m for m in checks["missing"])
        assert any("CUDA" in m or "torch" in m or "threefiner" in m
                   for m in checks["missing"])
