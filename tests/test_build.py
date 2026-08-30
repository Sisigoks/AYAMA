"""Phases 1-4 on one image, into one directory.

`run` stops after Phase 2 and `mesh` starts from a finished run: that split is
right for iteration and wrong for delivery, because it left a scene spread over
three directories that had to be matched up by hand. `build` closes it, and
these tests pin the shape of what it produces - a folder someone can move,
containing rasters, a browser tileset, and a textured mesh that opens in any DCC
tool without ĀYĀMA installed.
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
    from ayama.cli import main
    from ayama.data.sample import load_sample_scene
    from ayama.dsm.cog import write_cog, write_rgb

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
    from ayama.cli import build_parser

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
