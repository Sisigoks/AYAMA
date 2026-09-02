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


def one_big_building(n=96, ground=100.0):
    """A building whose grid is fine relative to its size, like a real one.

    `two_buildings` is 18 cells across, so no vertex of it is within 1% of its
    own bounding-box diagonal of another and threefiner's vertex merge would do
    nothing. A 72-cell building at 0.5 m ground sampling - a normal one - has
    its whole grid inside that threshold, which is the case worth testing.
    """
    dsm = np.full((n, n), ground, np.float32)
    dsm[12:84, 12:84] = ground + 30.0
    ndsm = (dsm - ground).astype(np.float32)
    mask = np.zeros((n, n), bool)
    mask[12:84, 12:84] = True
    return S.build(dsm, ndsm, [S.measure(1, mask, dsm, dsm - ndsm)], gsd_m=1.0)


def _textured_glb(path, vertices, faces, colour=(200, 100, 50), size=32):
    """A GLB shaped the way threefiner's is: its own vertices, its own atlas.

    threefiner never returns the mesh it was given - kiui merges close vertices
    before training and unwraps a fresh atlas at export - so every stand-in here
    splits the vertices per face, which is what an unwrap does and what the
    round trip therefore has to survive.
    """
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    V = np.asarray(vertices, np.float64)[np.asarray(faces, np.int64).ravel()]
    F = np.arange(len(V), dtype=np.int64).reshape(-1, 3)
    rng = np.random.default_rng(0)
    uv = rng.random((len(V), 2))
    trimesh.Trimesh(vertices=V, faces=F, process=False,
                    visual=trimesh.visual.TextureVisuals(
                        uv=uv, image=Image.new("RGB", (size, size), colour))
                    ).export(path)
    return path


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


def test_a_result_carrying_no_texture_is_refused(tmp_path):
    """There is exactly one thing worth reading out of the GLB. If it is not
    there, the file is not a refinement of anything."""
    trimesh = pytest.importorskip("trimesh")
    path = str(tmp_path / "bare.glb")
    trimesh.Trimesh(vertices=np.random.rand(10, 3),
                    faces=np.array([[0, 1, 2], [3, 4, 5]])).export(path)
    with pytest.raises(FA.FacadeUnavailable, match="no textured geometry"):
        FA.read_refined(path)


def test_a_result_that_is_a_different_object_is_refused(tmp_path):
    """The second guarantee, and the one `fix_geo` is not trusted for.

    Vertex counts cannot carry it: threefiner merges vertices before it starts
    and splits them again unwrapping its atlas, so the mesh that comes back
    never has the count that went in. What it must still have is the *shape* -
    so that is what is checked, in space, before any colour is taken from it.
    """
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)
    # A slab, textured, in the same normalised frame - and not this building.
    other = str(tmp_path / "other.glb")
    _textured_glb(other, np.array([[-.9, -.9, -.05], [.9, -.9, -.05],
                                   [.9, .9, -.05], [-.9, .9, -.05],
                                   [-.9, -.9, .05], [.9, -.9, .05],
                                   [.9, .9, .05], [-.9, .9, .05]], float),
                  np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]))
    with pytest.raises(FA.FacadeUnavailable, match="different shape"):
        FA.bake(other, local, F)


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


def test_the_handover_carries_vertex_colour(tmp_path):
    """The regression for the failure this stage actually hit.

    threefiner initialises its texture by rendering the mesh it is given
    (`fit_tex`). kiui's renderer takes the vertex-colour branch only when `vc`
    is set, and sets `vc` only when a `v` line has six floats; otherwise it
    interpolates `vt`, which for a mesh with no UVs is `None`, and the run dies
    with `'NoneType' object has no attribute 'unsqueeze'` some minutes in.
    """
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)
    colours = np.tile([0.25, 0.5, 0.75], (len(V), 1))
    path = FA.write_handover(str(tmp_path / "b.obj"), local, F, colours)

    lines = [l.split() for l in open(path, encoding="utf-8") if l.startswith("v ")]
    assert len(lines) == len(V)
    assert all(len(l) == 7 for l in lines), "kiui reads vertex colour off v lines"
    assert [float(x) for x in lines[0][4:]] == [0.25, 0.5, 0.75]


def test_the_handover_still_has_colour_without_an_orthophoto(tmp_path):
    """A grey building is a poor starting point; a crash is a worse one."""
    mesh = two_buildings()
    V, F = FA.extract(mesh, *FA.building_groups(mesh)[0][1:])
    local, _, _ = FA.to_object_frame(V)
    path = FA.write_handover(str(tmp_path / "b.obj"), local, F, None)
    assert all(len(l.split()) == 7
               for l in open(path, encoding="utf-8") if l.startswith("v "))


def test_vertex_colour_comes_from_the_orthophoto(tmp_path):
    """North is up in the raster and +Y in the mesh, and getting that backwards
    paints the wrong end of the scene onto the building."""
    Image = pytest.importorskip("PIL.Image")
    tex = tmp_path / "surface.png"
    arr = np.zeros((64, 64, 3), np.uint8)
    arr[:32] = (255, 0, 0)          # north half red
    arr[32:] = (0, 0, 255)          # south half blue
    Image.fromarray(arr).save(tex)

    span = 63.0                      # (64 - 1) * 1 m
    V = np.array([[0.0, span, 0.0], [0.0, 0.0, 0.0]])
    got = FA.vertex_colours(V, str(tex), (64, 64), 1.0)
    assert got[0][0] > 0.9 and got[0][2] < 0.1, "the north vertex is not red"
    assert got[1][2] > 0.9 and got[1][0] < 0.1, "the south vertex is not blue"


def test_a_building_too_small_to_survive_the_cleaner_is_refused():
    """threefiner deletes connected components under 32 faces before it starts,
    and a building is one component: it would be handed an empty mesh."""
    tiny = {"vertices": np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                                  [0., 1., 1.]]),
            "triangles": np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]),
            "groups": [("building_1", 0, 4)]}
    rec = FA.refine(tiny, "/tmp/unused", max_buildings=1, dry_run=True)

    entry = rec["buildings"][0]
    assert "32 faces" in entry["skipped"]
    assert "coarse" not in entry, "a mesh that cannot be refined was still written"


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
def fake_refined(mesh, out_dir, name, first, count, colour=(200, 100, 50)):
    """What threefiner returns, stood in for: the same surface, its own topology.

    The diffusion step cannot run on this machine, so the assembly is tested
    against a mesh with the shape threefiner produces rather than against
    nothing - which specifically means a mesh whose vertices have been split by
    an unwrap, because that is the thing the old index-based round trip could
    not survive.
    """
    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)
    path = os.path.join(out_dir, name + ".glb")
    _textured_glb(path, local, F, colour=colour)
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


def obj_faces(path):
    """`f` lines as (vertex index, texture index) pairs, 1-based as written."""
    out = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("f "):
            corners = []
            for term in line.split()[1:]:
                bits = term.split("/")
                corners.append((int(bits[0]),
                                int(bits[1]) if len(bits) > 1 and bits[1] else None))
            out.append(corners)
    return out


def test_the_assembled_model_keeps_the_geometry_verbatim(tmp_path):
    """The guarantee the whole stage rests on: refinement touches texture only."""
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    results = [fake_refined(mesh, str(tmp_path), name, first, count)]

    info = FA.assemble(mesh, results, str(tmp_path / "refined.obj"), (64, 64), 1.0,
                       resolution=128)
    assert info["buildings_refined"] == 1, "nothing was painted, so nothing is proven"
    V, _, _ = read_obj(info["obj"])
    assert V.shape == np.asarray(mesh["vertices"]).shape
    assert np.abs(V - np.asarray(mesh["vertices"])).max() == 0.0
    # and every triangle still exists, refined group included
    assert len(obj_faces(info["obj"])) == len(mesh["triangles"])


def test_refined_and_measured_surfaces_get_different_materials(tmp_path):
    """Opened in any tool, which walls are invented is visible in the material."""
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    info = FA.assemble(mesh, [fake_refined(mesh, str(tmp_path), name, first, count)],
                       str(tmp_path / "refined.obj"), (64, 64), 1.0, resolution=128)

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


def test_a_refined_result_that_is_a_different_object_is_skipped(tmp_path):
    """Rather than painting someone else's walls onto this building's."""
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    bad = str(tmp_path / (name + ".glb"))
    _textured_glb(bad, np.random.default_rng(1).random((9, 3)),
                  np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))

    results = [{"name": name, "glb": bad}]
    info = FA.assemble(mesh, results, str(tmp_path / "refined.obj"), (64, 64), 1.0)
    assert info["buildings_refined"] == 0
    assert "skipped" in results[0]


# ------------------------------------------------------------------ the bake
def test_the_bake_carries_colour_onto_our_own_triangles(tmp_path):
    """The transfer that replaced the index round trip, end to end.

    The stand-in mesh has three times the vertices of ours and none of its face
    order, which is the situation threefiner actually leaves behind. What comes
    back has to be a texture indexed by *our* faces, carrying the colour that
    was on that surface.
    """
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)
    glb = str(tmp_path / "b.glb")
    _textured_glb(glb, local, F, colour=(10, 200, 40))

    got = FA.bake(glb, local, F, resolution=128)
    assert len(got["ft"]) == len(F), "the atlas is not indexed by our faces"
    assert got["ft"].max() < len(got["vt"])
    assert got["vmapping"].max() < len(V), "an atlas vertex is not one of ours"
    assert len(got["vmapping"]) == len(got["vt"])
    assert got["coverage"] > 0.05
    # the source was a single flat colour, so every painted texel is that colour
    painted = np.asarray(got["image"], np.int64).reshape(-1, 3)
    assert np.abs(painted - np.array([10, 200, 40])).max() <= 2


def test_the_baked_atlas_has_no_unpainted_texels(tmp_path):
    """A black gutter between charts shows up as a dark seam around every face."""
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)
    glb = str(tmp_path / "b.glb")
    _textured_glb(glb, local, F, colour=(200, 200, 200))

    img = np.asarray(FA.bake(glb, local, F, resolution=128)["image"])
    assert (img > 0).all(), "the gutters were left black"


def _as_threefiner_would(local, F, weld=0.03):
    """The mesh threefiner hands back, simulated: welded, then re-split.

    Both halves matter and both are what kiui does. `clean_mesh(v_pct=1)` merges
    every pair of vertices within 1% of the bounding-box diagonal before the
    first iteration - at 0.5 m ground sampling on a 40 m building that is most
    of the grid - and `auto_uv` splits vertices again along every chart seam at
    export. So the returned mesh shares neither vertex count, vertex order nor
    face order with the one that went in, which is the whole reason the texture
    is carried back through space.
    """
    keys = np.round(np.asarray(local, np.float64) / weld).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    welded = np.asarray(local, np.float64)[first]
    faces = inverse[np.asarray(F, np.int64)]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 0] != faces[:, 2])]
    return welded, faces[::-1]          # reversed: face order is not preserved


def test_the_bake_survives_the_mesh_threefiner_actually_returns(tmp_path):
    """The failure the index round trip could not see, reproduced.

    Nothing here has the same topology as the mesh handed over, and the
    transfer still has to land the right colour on the right wall.
    """
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = one_big_building()
    name, first, count = FA.building_groups(mesh)[0]
    V, F = FA.extract(mesh, first, count)
    local, _, _ = FA.to_object_frame(V)

    welded, welded_F = _as_threefiner_would(local, F)
    assert len(welded) < len(V), "the stand-in did not actually weld anything"
    glb = str(tmp_path / (name + ".glb"))
    _textured_glb(glb, welded, welded_F, colour=(30, 60, 90))

    got = FA.bake(glb, local, F, resolution=128)
    assert len(got["ft"]) == len(F)
    assert got["offset"] < FA.MAX_SURFACE_OFFSET
    assert np.abs(np.asarray(got["image"], np.int64)
                  - np.array([30, 60, 90])).max() <= 2


def test_the_refined_obj_loads_with_its_texture(tmp_path):
    """A file no loader accepts is not a deliverable. Face-varying `v/vt` and
    two materials in one OBJ are both places a writer can quietly go wrong."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    info = FA.assemble(mesh, [fake_refined(mesh, str(tmp_path), name, first, count,
                                           colour=(12, 34, 56))],
                       str(tmp_path / "refined.obj"), (64, 64), 1.0, resolution=128)

    scene = trimesh.load(info["obj"], process=False, force="scene")
    total = sum(len(g.faces) for g in scene.geometry.values())
    assert total == len(mesh["triangles"]), "triangles were lost in the OBJ"
    # the synthesised texture is on disk, is the colour that was refined, and
    # something in the loaded scene carries a UV that points into it
    from PIL import Image

    with Image.open(tmp_path / (name + ".png")) as im:
        assert np.abs(np.asarray(im.convert("RGB"), np.int64)
                      - np.array([12, 34, 56])).max() <= 2
    assert any(getattr(getattr(g, "visual", None), "uv", None) is not None
               for g in scene.geometry.values())


def test_a_refined_group_gets_face_varying_texture_indices(tmp_path):
    """A chart seam gives one vertex two texture coordinates. OBJ says that per
    corner, and the vertex index stays ours on both sides of the seam."""
    pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    mesh = two_buildings()
    name, first, count = FA.building_groups(mesh)[0]
    info = FA.assemble(mesh, [fake_refined(mesh, str(tmp_path), name, first, count)],
                       str(tmp_path / "refined.obj"), (64, 64), 1.0, resolution=128)
    assert info["buildings_refined"] == 1

    n_v = len(mesh["vertices"])
    faces = obj_faces(info["obj"])
    # Somewhere in the file a corner points at a texture coordinate that is not
    # its own vertex, and that coordinate lives in the appended atlas block.
    appended = [t for face in faces for _, t in face if t and t > n_v]
    assert appended, "no face used the baked atlas"
    assert all(v <= n_v for face in faces for v, _ in face), \
        "a face referenced a vertex that is not in the scene"


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
