"""The binary the viewer reads.

A format only one side can parse is not a format, so the writer is tested
against a reader that is not the renderer. What matters is that the arrays come
back in the same order and the same units they went in: a silent offset error
here draws a plausible-looking scene at the wrong scale, which is the failure
mode this whole project is organised against.
"""
from __future__ import annotations

import numpy as np
import pytest

from traksha.mesh import webmesh as W
from traksha.semantics.instances import InstanceField


def scene_with_a_block(n=48, ground=100.0, height=18.0):
    dsm = np.full((n, n), ground, np.float32)
    dsm[12:34, 12:34] = ground + height
    ndsm = (dsm - ground).astype(np.float32)
    inst = np.zeros((n, n), np.int32)
    inst[12:34, 12:34] = 1
    field = InstanceField(
        instance_map=inst, boundary=np.zeros((n, n), bool),
        confidence=np.full((n, n), 0.9, np.float32),
        records=[{"id": 1, "area_px": int((inst == 1).sum()),
                  "visible_px": int((inst == 1).sum()), "score": 0.9}],
        provenance={})
    return dsm, ndsm, field


def test_the_stride_respects_the_triangle_budget():
    assert W.choose_stride((1025, 1025), 350_000) >= 2
    assert W.choose_stride((129, 129), 350_000) == 1
    # A budget nothing can satisfy still returns a usable stride, not a crash.
    assert 1 <= W.choose_stride((4097, 4097), 10) <= 16


def test_a_written_mesh_reads_back_identically(tmp_path):
    dsm, ndsm, field = scene_with_a_block()
    mesh = W.build_web_mesh(dsm, ndsm, field, None, 1.0, max_triangles=100_000)
    path = str(tmp_path / "structural.bin")
    info = W.write(path, mesh, mesh["grid"], mesh["gsd_m"])
    back = W.read(path)

    assert info["format"] == "TKM1"
    assert len(back["positions"]) == info["vertices"] == len(mesh["vertices"])
    assert len(back["indices"]) == info["triangles"] * 3
    assert len(back["groups"]) == info["groups"]
    np.testing.assert_allclose(back["positions"], mesh["vertices"][:, :2], rtol=1e-6)
    np.testing.assert_allclose(back["heights"], mesh["vertices"][:, 2], rtol=1e-6)


def test_heights_survive_in_metres():
    """The viewer draws these directly. A unit error here is a wrong building."""
    dsm, ndsm, field = scene_with_a_block(height=18.0)
    mesh = W.build_web_mesh(dsm, ndsm, field, None, 1.0, max_triangles=100_000)
    z = mesh["vertices"][:, 2]
    assert z.min() == pytest.approx(100.0, abs=1e-3)
    assert z.max() == pytest.approx(118.0, abs=1e-3)


def test_every_vertex_carries_a_unit_normal():
    """The viewer shades from these: a wall shares its UV with the ground under
    it, so a normal map would light a facade as though it were pavement."""
    dsm, ndsm, field = scene_with_a_block()
    mesh = W.build_web_mesh(dsm, ndsm, field, None, 1.0, max_triangles=100_000)
    n = mesh["normals"]
    assert n.shape == (len(mesh["vertices"]), 3)
    lengths = np.linalg.norm(n, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_the_walls_survive_the_stride():
    """Strided, not simplified - a decimator would collapse the footprints."""
    from traksha.mesh.quality import verticality

    dsm, ndsm, field = scene_with_a_block(n=200)
    mesh = W.build_web_mesh(dsm, ndsm, field, None, 1.0, max_triangles=20_000)
    assert mesh["stride"] > 1, "this budget should have forced a coarser grid"
    v = verticality(mesh["vertices"], mesh["triangles"])
    assert v["wall_area_m2"] > 0
    assert v["max_slope_deg"] == pytest.approx(90.0, abs=1e-6)


def test_groups_name_the_terrain_and_the_buildings(tmp_path):
    dsm, ndsm, field = scene_with_a_block()
    mesh = W.build_web_mesh(dsm, ndsm, field, None, 1.0, max_triangles=100_000)
    path = str(tmp_path / "m.bin")
    W.write(path, mesh, mesh["grid"], mesh["gsd_m"])
    kinds = [g[2] for g in W.read(path)["groups"]]
    assert kinds[0] == 0, "the first group should be the terrain"
    assert 1 in kinds, "no building group was recorded"


def test_a_file_that_is_not_ours_is_refused(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"GLTF" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a TRAKSHA mesh"):
        W.read(str(bad))
