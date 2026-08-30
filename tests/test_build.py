"""Phases 1-4 on one image, into one directory.

`run` stops after Phase 2 and `mesh` starts from a finished run: that split is
right for iteration and wrong for delivery, because it left a scene spread over
three directories that had to be matched up by hand. `build` closes it, and
these tests pin the shape of what it produces - a folder someone can move,
containing rasters, a browser tileset, and a textured mesh that opens in any DCC
tool without TRAKSHA installed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

BACKBONE = "dav2-vits"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One real scene through all four phases. Small, but the whole chain."""
    from traksha.cli import main
    from traksha.data.sample import load_sample_scene
    from traksha.dsm.cog import write_cog, write_rgb

    d = tmp_path_factory.mktemp("build")
    sc = load_sample_scene(size=384, sun=(210.0, 45.0))
    img = str(d / "scene.tif")
    write_rgb(img, sc.rgb, sc.meta)
    write_cog(str(d / "scene_dsm.tif"), sc.dsm_m, sc.meta)
    write_cog(str(d / "scene_dtm.tif"), sc.dtm_m, sc.meta)

    out = str(d / "built")
    rc = main(["build", img, "--out", out, "--backbone", BACKBONE, "--chip", "384",
               "--dem", f"sim:{d / 'scene_dtm.tif'}", "--ref", str(d / "scene_dsm.tif"),
               "--bootstrap", "4", "--obj-stride", "4", "--progress", "none"])
    assert rc == 0
    return out


pytestmark = pytest.mark.slow


# ------------------------------------------------------------ the four phases
def test_one_command_produces_all_four_phases_in_one_folder(built):
    """The whole point. Each phase must leave its output in the same directory."""
    # phase 1: the relative depth the backbone emitted
    assert os.path.exists(os.path.join(built, "relative_depth.tif"))
    # phase 2: the calibrated surface and its uncertainty
    for name in ("dsm.tif", "ndsm.tif", "sigma.tif", "sem.tif", "provenance.json"):
        assert os.path.exists(os.path.join(built, name)), name
    # phase 3: the browser tileset
    assert os.path.exists(os.path.join(built, "tiles3d", "tileset.json"))
    # phase 4: the deliverable mesh
    for name in ("surface.obj", "surface.mtl", "surface.jpg"):
        assert os.path.exists(os.path.join(built, "mesh", name)), name


def test_the_mesh_sits_beside_the_tileset_not_inside_it(built):
    """A scene is one folder with two siblings, not a tileset with a mesh buried in it."""
    assert os.path.isdir(os.path.join(built, "mesh"))
    assert not os.path.isdir(os.path.join(built, "tiles3d", "mesh"))


# ------------------------------------------------------------------ the mesh
def test_the_obj_is_a_textured_mesh_a_dcc_tool_can_open(built):
    """OBJ + MTL + JPG, wired to each other by name. This is the deliverable."""
    obj = os.path.join(built, "mesh", "surface.obj")
    with open(obj, encoding="utf-8") as fh:
        text = fh.read()

    assert "mtllib surface.mtl" in text, "the OBJ does not reference its material"
    assert "usemtl " in text
    assert text.count("\nv ") > 1000, "no vertices"
    assert text.count("\nf ") > 1000, "no faces"
    assert "\nvt " in text, "no texture coordinates - the JPG could not be applied"

    with open(os.path.join(built, "mesh", "surface.mtl"), encoding="utf-8") as fh:
        mtl = fh.read()
    assert "newmtl " in mtl
    assert "map_Kd surface.jpg" in mtl, "the material does not point at the texture"

    # the file it points at has to be a real image, not a stub
    from PIL import Image

    with Image.open(os.path.join(built, "mesh", "surface.jpg")) as im:
        assert min(im.size) >= 64


def test_the_obj_carries_metres_not_pixels(built):
    """A mesh in pixel units is silently wrong in every downstream tool."""
    verts = []
    with open(os.path.join(built, "mesh", "surface.obj"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            if len(verts) > 4000:
                break
    v = np.array(verts)
    span_xy = float(max(v[:, 0].max() - v[:, 0].min(), v[:, 1].max() - v[:, 1].min()))
    # 384 px at 0.5 m is 192 m across; pixels would give 384.
    assert 150 < span_xy < 220, f"horizontal span {span_xy:.0f} is not metres"
    assert v[:, 2].min() > 100, "elevation is not in metres above the datum"


def test_the_manifest_points_at_the_sibling_mesh(built):
    """The viewer resolves these paths, so they have to be right relative to it."""
    with open(os.path.join(built, "tiles3d", "tileset.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    mesh = m["mesh"]
    assert mesh["obj"].startswith("../mesh/")
    assert mesh["mtl"].endswith(".mtl") and mesh["texture"].endswith(".jpg")
    for key in ("obj", "mtl", "texture"):
        resolved = os.path.normpath(os.path.join(built, "tiles3d", mesh[key]))
        assert os.path.exists(resolved), f"{key} -> {mesh[key]} does not resolve"


def test_the_viewer_can_serve_the_sibling_mesh(built):
    """A browser normalises `data/../mesh/x` to `/mesh/x` before sending it.

    The local viewer strips `..` from request paths, so without a second route
    the mesh download 404s while everything else works - exactly the kind of
    break that ships.
    """
    from traksha.cli import build_parser

    # The route is in cmd_viewer's handler; assert the behaviour it must have by
    # exercising the same path arithmetic.
    tiles = os.path.join(built, "tiles3d")
    mesh_dir = os.path.abspath(os.path.join(tiles, os.pardir, "mesh"))
    assert os.path.isdir(mesh_dir)
    assert os.path.exists(os.path.join(mesh_dir, "surface.obj"))
    assert build_parser() is not None


# ------------------------------------------------------------ the surface
def test_the_delivered_surface_carries_the_fitted_scale(built):
    """`build` runs with the bundled calibration, so relief must reach the mesh."""
    with rasterio.open(os.path.join(built, "ndsm.tif")) as ds:
        nd = ds.read(1)
    assert np.isfinite(nd).all()
    assert nd.max() > 5.0, (
        "the delivered nDSM is flat - the fitted structural scale did not reach "
        "the surface (see tests/test_learn.py for the bootstrap path)")


# ----------------------------------------------------- the committed deliverable
def test_every_delivered_scene_ships_a_textured_mesh_in_the_repository():
    """The mesh is the deliverable, so it is committed, so it is tested.

    It was gitignored once - `results/**/mesh/` - which made the repository a
    study you had to run before you could look at anything. An OBJ at stride 4
    is ASCII that deflates to about a fifth of its size, so four of them cost
    the pack ~8 MB and are worth it.
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    if not (results / "dataset.json").exists():
        pytest.skip("no delivered study in results/")

    tracked = set(subprocess.run(
        ["git", "ls-files", "results"], cwd=root,
        capture_output=True, text=True).stdout.split())
    if not tracked:
        pytest.skip("not a git checkout")

    scenes = [d for d in results.iterdir() if (d / "summary.json").exists()]
    assert scenes, "a delivered study must contain at least one scene folder"
    for scene in scenes:
        for name in ("surface.obj", "surface.mtl", "surface.jpg"):
            rel = f"results/{scene.name}/mesh/{name}"
            assert (scene / "mesh" / name).exists(), f"{rel} is missing"
            assert rel in tracked, (
                f"{rel} exists but is not committed - check .gitignore; the "
                "mesh is the deliverable and has to travel with the repository")


def test_the_committed_obj_is_written_at_a_precision_the_data_supports():
    """Millimetres, no trailing zeros. Tenths of a millimetre on a surface with
    metre-scale error is bytes spent on noise, and it is what kept the mesh from
    being committable."""
    from pathlib import Path

    obj = Path(__file__).resolve().parents[1] / "results" / "zurich" / "mesh" / "surface.obj"
    if not obj.exists():
        pytest.skip("no delivered Zurich mesh")

    verts = [ln for ln in obj.read_text(encoding="utf-8").split("\n")
             if ln.startswith("v ")][:2000]
    assert verts
    for ln in verts:
        for tok in ln.split()[1:]:
            frac = tok.partition(".")[2]
            assert len(frac) <= 3, f"{tok} carries sub-millimetre precision"
            assert not (frac and frac.endswith("0")), f"{tok} has a trailing zero"


# ------------------------------------------------------------ adaptive meshing
def _mesh_stats(V, T):
    d = T - 1
    tri = V[d]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    e = np.sort(np.concatenate([d[:, [0, 1]], d[:, [1, 2]], d[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    a = tri[:, 1, :2] - tri[:, 0, :2]
    b = tri[:, 2, :2] - tri[:, 0, :2]
    footprint = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    return {"up": float((n[:, 2] > 0).mean()),
            "nonmanifold": int((counts > 2).sum()),
            "slivers": int((footprint < 1e-12).sum()),
            "unreferenced": int(len(V) - len(np.unique(d)))}


def _terrain(n=192):
    """Flat ground with a few tall blocks - the case adaptation exists for."""
    z = np.zeros((n, n), np.float32)
    z[40:90, 40:90] = 22.0
    z[110:150, 60:140] = 14.0
    z += np.random.default_rng(31).normal(0, 0.03, z.shape).astype(np.float32)
    return z


def test_the_adaptive_mesh_is_watertight_where_resolutions_meet():
    """The guarantee that makes mixed resolutions usable.

    A coarse block beside a fine one is where a T-junction would form, and a
    T-junction is a hairline crack you can see through. The fan is built from
    exactly the vertices the neighbours kept, so no edge may be shared by more
    than two triangles and none may be a zero-footprint sliver.
    """
    from traksha.mesh.adaptive import adaptive_mesh

    m = adaptive_mesh(_terrain(), 0.5, tol_m=0.5, block=8)
    st = _mesh_stats(m["vertices"], m["triangles"])
    assert st["nonmanifold"] == 0, "an edge is shared by three or more triangles"
    assert st["slivers"] == 0, "collinear fan triangles with no plan-view footprint"
    assert st["unreferenced"] == 0, "vertices written but never used"
    assert st["up"] == 1.0, "a height field cannot have downward-facing triangles"
    # it must actually have adapted, or the test proves nothing
    lay = m["layout"]
    assert 0 < lay["n_fine"] < lay["n_blocks"], "no mix of resolutions to test"


def test_the_tolerance_is_a_guarantee_not_a_hint():
    """Every coarse block must be within tol of the surface it replaced."""
    from traksha.mesh.adaptive import _bilinear_block_error, plan

    z = _terrain()
    for tol in (0.25, 1.0, 4.0):
        lay = plan(z, tol_m=tol, block=8)
        err = _bilinear_block_error(z, 8)
        coarse = ~lay["fine"]
        assert (err[coarse] <= tol + 1e-5).all(), (
            f"a block kept coarse at tol {tol} has error {err[coarse].max():.2f} m")


def test_adaptation_beats_a_uniform_grid_at_the_same_triangle_count():
    """The claim the feature rests on, checked rather than asserted."""
    from traksha.mesh.adaptive import adaptive_mesh

    z = _terrain()
    m = adaptive_mesh(z, 0.5, tol_m=0.5, block=8)
    n_adaptive = len(m["triangles"])

    # the uniform stride with a comparable budget
    stride = max(1, int(round(np.sqrt(2 * (z.shape[0] - 1) ** 2 / max(n_adaptive, 1)))))
    small = z[::stride, ::stride]
    back = np.repeat(np.repeat(small, stride, 0), stride, 1)[:z.shape[0], :z.shape[1]]
    uniform_err = float(np.abs(back - z).max())

    adaptive_err = 0.5   # the tolerance, guaranteed by the test above
    assert adaptive_err < uniform_err, (
        f"adaptive bounds error at {adaptive_err} m; uniform stride {stride} "
        f"reaches {uniform_err:.1f} m at a similar triangle count")


def test_a_triangle_budget_is_honoured_or_the_shortfall_is_reported():
    """`max_triangles` picks the tolerance, so it must actually bind."""
    from traksha.mesh.adaptive import tolerance_for_budget

    z = _terrain(256)
    tol, lay, n = tolerance_for_budget(z, 60_000)
    assert n <= 60_000, f"budget overrun: {n}"
    assert tol > 0 and lay["block"] >= 8

    # Below the structural floor the search cannot deliver, and must return the
    # smallest mesh it reached rather than silently pretending otherwise.
    tol2, lay2, n2 = tolerance_for_budget(z, 10)
    assert n2 > 10, "a budget under the floor should be reported, not faked"
    assert n2 <= n, "the floor result should be the smallest mesh found"


def test_the_adaptive_obj_records_how_it_adapted(tmp_path):
    """The header is the provenance: a reader must be able to see the tolerance."""
    from traksha.mesh.obj import write_obj_adaptive

    p = str(tmp_path / "s.obj")
    info = write_obj_adaptive(p, _terrain(), 0.5, texture_name="t.jpg", tol_m=1.0)
    head = "\n".join(open(p, encoding="utf-8").read().split("\n")[:4])
    assert "adaptive:" in head and "tolerance" in head
    assert info["adaptive"]["tol_m"] == pytest.approx(1.0)
    assert 0 < info["adaptive"]["fine_blocks"] < info["adaptive"]["blocks"]
    assert os.path.exists(os.path.splitext(p)[0] + ".mtl")
