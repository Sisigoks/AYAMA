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


# --------------------------------------------------------- the final model
def fake_refined(mesh, out_dir, name, first, count):
    """What threefiner returns, stood in for: same vertices, a texture, UVs.

    The diffusion step cannot run on this machine, so the assembly is tested
    against a mesh with the shape threefiner produces rather than against
    nothing. Everything the assembler actually reads - vertex count, UVs, the
    texture image - is real.
    """
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)
    uv = np.random.default_rng(0).random((len(V), 2)).astype(np.float32)
    tm = trimesh.Trimesh(vertices=local, faces=F, process=False,
                         visual=trimesh.visual.TextureVisuals(
                             uv=uv, image=Image.new("RGB", (16, 16), (200, 100, 50))))
    path = os.path.join(out_dir, name + ".glb")
    tm.export(path)
    return {"name": name, "glb": path}


def read_obj(path):
    verts, groups, mats = [], [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            groups.append(line.split()[1])
        elif line.startswith("usemtl "):
            mats.append(line.split()[1])
    return np.asarray(verts), groups, mats


def test_the_assembled_model_keeps_the_geometry_verbatim(tmp_path):
    """The guarantee the whole stage rests on: refinement touches texture only."""
    pytest.importorskip("trimesh")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    results = [fake_refined(mesh, str(tmp_path), name, first, count)]

    info = FA.assemble(mesh, results, str(tmp_path / "refined.obj"), (64, 64), 1.0)
    V, _, _ = read_obj(info["obj"])
    assert V.shape == np.asarray(mesh["vertices"]).shape
    assert np.abs(V - np.asarray(mesh["vertices"])).max() == 0.0


def test_refined_and_measured_surfaces_get_different_materials(tmp_path):
    """Opened in any tool, which walls are invented is visible in the material."""
    pytest.importorskip("trimesh")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    info = FA.assemble(mesh, [fake_refined(mesh, str(tmp_path), name, first, count)],
                       str(tmp_path / "refined.obj"), (64, 64), 1.0)

    _, groups, mats = read_obj(info["obj"])
    assert "terrain" in groups and name in groups
    assert mats[groups.index("terrain")] == "measured_mat"
    assert mats[groups.index(name)] == "synth_" + name
    # the other building was not refined and must not claim to be
    other = [g for g in groups if g.startswith("building_") and g != name][0]
    assert mats[groups.index(other)] == "measured_mat"

    mtl = open(info["mtl"], encoding="utf-8").read()
    assert "SYNTHESISED" in mtl
    assert os.path.exists(tmp_path / (name + ".png"))


def test_assembling_with_nothing_refined_is_honest_about_it(tmp_path):
    """A CPU-only run still produces the model; it must not claim synthesis."""
    mesh = two_buildings()
    info = FA.assemble(mesh, [], str(tmp_path / "refined.obj"), (64, 64), 1.0)
    assert info["buildings_refined"] == 0
    assert info["synthesised"] is False
    _, groups, mats = read_obj(info["obj"])
    assert set(mats) == {"measured_mat"}


def test_a_refined_result_with_the_wrong_vertex_count_is_skipped(tmp_path):
    """Rather than pasting someone else's UVs onto our vertices."""
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    bad = str(tmp_path / (name + ".glb"))
    trimesh.Trimesh(vertices=np.random.rand(9, 3),
                    faces=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                    process=False,
                    visual=trimesh.visual.TextureVisuals(
                        uv=np.zeros((9, 2), np.float32),
                        image=Image.new("RGB", (8, 8)))).export(bad)

    results = [{"name": name, "glb": bad}]
    info = FA.assemble(mesh, results, str(tmp_path / "refined.obj"), (64, 64), 1.0)
    assert info["buildings_refined"] == 0
    assert "skipped" in results[0]


# ----------------------------------------------------- retiring what it replaces
def test_the_refined_model_replaces_what_it_supersedes(tmp_path):
    """`structural.obj` has identical geometry and worse texture; `surface.obj`
    is the height field the structural mesh already replaced. Shipping either
    beside the refined model offers the same surface twice."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    for n in ("surface.obj", "surface.mtl", "surface.jpg", "structural.obj",
              "structural.mtl", "structural_refined.obj"):
        (mesh_dir / n).write_text("x")

    man = {"mesh": {"obj": "mesh/surface.obj", "mtl": "mesh/surface.mtl",
                    "texture": "mesh/surface.jpg",
                    "structural": {"obj": "mesh/structural.obj",
                                   "mtl": "mesh/structural.mtl"}}}
    retired = FA.retire_superseded(str(mesh_dir), man, "mesh/structural_refined.obj")

    assert set(retired) == set(FA.SUPERSEDED)
    for n in FA.SUPERSEDED:
        assert not (mesh_dir / n).exists(), f"{n} was not removed"
    # the orthophoto stays: it is the measured texture the refined mtl references
    assert (mesh_dir / "surface.jpg").exists()
    assert (mesh_dir / "structural_refined.obj").exists()


def test_no_download_link_is_left_pointing_at_a_deleted_file(tmp_path):
    """The manifest and the disk go out of step exactly when they are updated
    in different places, so they are updated in the same call."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    for n in (*FA.SUPERSEDED, "structural_refined.obj"):
        (mesh_dir / n).write_text("x")
    man = {"mesh": {"obj": "mesh/surface.obj", "mtl": "mesh/surface.mtl",
                    "structural": {"obj": "mesh/structural.obj",
                                   "mtl": "mesh/structural.mtl"}}}
    FA.retire_superseded(str(mesh_dir), man, "mesh/structural_refined.obj")

    assert "obj" not in man["mesh"] and "mtl" not in man["mesh"]
    assert "obj" not in man["mesh"]["structural"]
    assert man["mesh"]["primary"] == "mesh/structural_refined.obj"
    # nothing the manifest still names is missing from disk
    for key, value in man["mesh"].items():
        if isinstance(value, str) and value.startswith("mesh/"):
            assert (tmp_path / value).exists(), f"{key} points at a deleted file"


def test_retiring_is_safe_when_the_files_were_never_written(tmp_path):
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    man = {}
    assert FA.retire_superseded(str(mesh_dir), man, "mesh/x.obj") == []
    assert man["mesh"]["primary"] == "mesh/x.obj"
