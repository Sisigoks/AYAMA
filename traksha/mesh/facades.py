"""Synthesised facade texture, via threefiner, on a GPU that has one.

A nadir image photographs roofs. It does not photograph walls, so the structural
mesh's facades currently sample the orthophoto at the footprint edge and stretch
that one line of pixels down the wall. The geometry of those walls is measured -
the roof height and the local ground are both calibrated values - but their
*appearance* is not, and cannot be, from this input.

threefiner (3DTopia) refines a mesh against a diffusion prior by score
distillation. Given a GPU it can paint those walls. Three constraints make that
usable here rather than corrosive.

**Geometry is fixed, and enforced twice.** Only the `*_fixgeo` presets are
accepted (`fix_geo=True, geom_mode='mesh'`); the geometry-training modes are
refused outright, because deforming a wall to satisfy a generative prior would
turn a measurement into a guess. And belt-and-braces: only the UVs and the
texture are read back. The vertex positions written out are this pipeline's own,
copied verbatim, so geometry invariance is true by construction rather than by
trusting a flag.

**The output is a separate, labelled artifact.** `structural.obj`, the tileset,
and every calibrated raster are untouched. What this writes is
`facades/<id>.glb` plus a manifest entry that says, in the artifact itself, that
the wall texture is synthesised.

**It is per building.** threefiner is an object refiner: it orbits a camera
around one normalised object. That is the wrong shape for a square kilometre of
city and exactly the right shape for one building - and the structural rebuild
has already made every building its own connected component, so the objects are
sitting there ready.

What this is not: measurement. The walls come out plausible, not photographed.
A reader looking at a refined facade is looking at what a diffusion model thinks
a building of that shape looks like, and the manifest says so.

**Not verified on hardware.** This was written and tested on a CPU-only machine:
the extraction, the frame conversion, the round trip and the guards are covered
by tests, and the diffusion step itself has never been executed here. See
`preflight()` for what a GPU box needs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

import numpy as np

from .obj import _write_block, _write_faces

# The presets that leave geometry alone. Anything else deforms vertices to
# satisfy a generative prior, which on a metric reconstruction is fabrication.
FIXED_GEOMETRY_PRESETS = ("sd_fixgeo", "if_fixgeo", "if2_fixgeo")
DEFAULT_PRESET = "sd_fixgeo"

# A T4 is 16 GB and Turing. Stable Diffusion 2 fits comfortably and is not
# gated; DeepFloyd IF-II is heavier and needs a Hugging Face token, so it is
# available but not the default.
PRESET_NOTES = {
    "sd_fixgeo": "Stable Diffusion 2, ~800 iterations. Fits a 16 GB T4.",
    "if_fixgeo": "DeepFloyd IF-I. Needs a Hugging Face token.",
    "if2_fixgeo": "DeepFloyd IF-II, finest. Heaviest; may not fit a 16 GB card.",
}

# Minutes per building on a mid-range GPU, so the default is a handful of the
# tallest rather than every building in the scene.
DEFAULT_MAX_BUILDINGS = 8

DEFAULT_PROMPT = ("a photograph of the facade of a European city building, "
                  "stone and render walls, regular rows of windows, flat roof")


class FacadeUnavailable(RuntimeError):
    """Facade refinement cannot run here. Carries what is missing."""


def _ensure_kiui_compatibility() -> None:
    """Fix kiui type annotations on Python 3.13 environments."""
    try:
        import kiui
        d = os.path.dirname(kiui.__file__)
        header = (
            "from __future__ import annotations\n"
            "from typing import Union, Optional, List, Tuple, Any\n"
            "try:\n"
            "    from torch import Tensor\n"
            "except Exception:\n"
            "    Tensor = Any\n"
            "try:\n"
            "    from numpy import ndarray\n"
            "except Exception:\n"
            "    ndarray = Any\n"
        )
        for fname in os.listdir(d):
            if fname.endswith(".py"):
                p = os.path.join(d, fname)
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        src = fh.read()
                    if "from __future__ import annotations" not in src:
                        src = header + src
                        with open(p, "w", encoding="utf-8") as fh:
                            fh.write(src)
                except Exception:
                    pass
    except Exception:
        pass


def preflight() -> dict:
    """What this needs, and which parts are present.

    Reported rather than raised so a caller can print it: the whole point is
    that this runs on a different machine from the one the pipeline usually
    runs on, and the failure a user hits is a missing dependency on that box.
    """
    _ensure_kiui_compatibility()
    out: dict = {"ok": False, "missing": [], "notes": []}

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        out["torch"] = torch.__version__
        out["cuda"] = cuda
        if cuda:
            out["device"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            out["vram_gb"] = round(total / 1e9, 1)
            if total < 14e9:
                out["notes"].append(
                    f"{total / 1e9:.0f} GB of VRAM; use the sd_fixgeo preset")
        else:
            out["missing"].append("a CUDA device (threefiner has no CPU path)")
    except ImportError:
        out["missing"].append("torch")

    for mod, why in (("threefiner", "pip install threefiner"),
                     ("nvdiffrast", "pip install "
                      "git+https://github.com/NVlabs/nvdiffrast (CUDA, source build)"),
                     ("trimesh", "pip install trimesh")):
        try:
            __import__(mod)
        except ImportError:
            out["missing"].append(f"{mod} ({why})")

    if shutil.which("threefiner") is None and "threefiner" not in str(out["missing"]):
        out["notes"].append("the `threefiner` console script is not on PATH; "
                            "the module will be invoked with -m instead")
    out["ok"] = not out["missing"]
    return out


# ----------------------------------------------------------------- geometry
def building_groups(mesh: dict):
    """Every building group in a structural mesh, largest first."""
    groups = [(name, first, count) for name, first, count in mesh.get("groups", [])
              if name.startswith("building_")]
    return sorted(groups, key=lambda g: -g[2])


def extract(mesh: dict, first: int, count: int):
    """One building as a standalone mesh, densely renumbered."""
    F = np.asarray(mesh["triangles"], np.int64)[first:first + count]
    used = np.unique(F)
    remap = np.full(int(used.max()) + 1, -1, np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(mesh["vertices"], np.float64)[used], remap[F]


def to_object_frame(V: np.ndarray):
    """Metres, Z-up, world origin -> unit-ish, Y-up, centred. Returns the inverse.

    threefiner orbits a camera at radius 2.5 around an object it assumes is
    roughly unit scale and Y-up. A building 30 m tall sitting 800 m from the
    scene origin is neither, so it is moved rather than hoped about.
    """
    centre = (V.max(0) + V.min(0)) / 2.0
    extent = float(np.max(V.max(0) - V.min(0)))
    scale = 1.8 / max(extent, 1e-6)          # into roughly [-0.9, 0.9]

    local = (V - centre) * scale
    # Z-up to Y-up: (x, y, z) -> (x, z, -y)
    out = np.stack([local[:, 0], local[:, 2], -local[:, 1]], axis=1)

    def inverse(W: np.ndarray) -> np.ndarray:
        back = np.stack([W[:, 0], -W[:, 2], W[:, 1]], axis=1)
        return back / scale + centre

    return out, inverse, {"centre": centre.tolist(), "scale": scale}


# ------------------------------------------------------------------- the run
def refine_one(vertices: np.ndarray, faces: np.ndarray, out_dir: str, name: str,
               prompt: str = DEFAULT_PROMPT, preset: str = DEFAULT_PRESET,
               iters: Optional[int] = None, timeout_s: int = 3600) -> dict:
    """Run threefiner over one building. Returns what it produced.

    The mesh is written Y-up and normalised, threefiner is invoked as a
    subprocess - it is a console script with its own argument parser, and
    importing its GUI class to drive it in-process would couple this to its
    internals - and the result is read back for its texture only.
    """
    if preset not in FIXED_GEOMETRY_PRESETS:
        raise FacadeUnavailable(
            f"'{preset}' trains geometry. Only {', '.join(FIXED_GEOMETRY_PRESETS)} "
            "are allowed here: deforming a measured wall to satisfy a generative "
            "prior would turn a measurement into a guess.")

    _ensure_kiui_compatibility()
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    local, inverse, frame = to_object_frame(np.asarray(vertices, np.float64))
    src = os.path.join(out_dir, f"{name}_coarse.obj")
    trimesh.Trimesh(vertices=local, faces=np.asarray(faces),
                    process=False).export(src)

    cmd = [shutil.which("threefiner") or "python", *([] if shutil.which("threefiner")
                                                     else ["-m", "threefiner.cli"]),
           preset, "--mesh", src, "--prompt", prompt,
           "--outdir", out_dir, "--save", f"{name}.glb",
           "--force_cuda_rast"]
    if iters:
        cmd += ["--iters", str(int(iters))]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    result = {"name": name, "preset": preset, "prompt": prompt,
              "frame": frame, "returncode": proc.returncode,
              "glb": os.path.join(out_dir, f"{name}.glb")}
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "")[-800:]
        result["glb"] = None
    return result


def read_texture(glb_path: str, expect_vertices: int):
    """Take the UVs and the texture, and nothing else.

    Vertex positions are deliberately *not* read. `fix_geo` is supposed to leave
    them alone, but "supposed to" is not a guarantee this pipeline can offer
    about a metric surface, and it does not have to: the positions it already
    has are the calibrated ones. If the vertex count moved, the correspondence
    is gone and the result is refused rather than guessed at.
    """
    import trimesh

    loaded = trimesh.load(glb_path, process=False, force="scene")
    geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    for g in geoms:
        if len(g.vertices) != expect_vertices:
            continue
        uv = getattr(getattr(g, "visual", None), "uv", None)
        image = getattr(getattr(g, "visual", None), "material", None)
        if uv is None:
            continue
        return {"uv": np.asarray(uv, np.float32),
                "image": getattr(image, "baseColorTexture", None)}
    raise FacadeUnavailable(
        f"{os.path.basename(glb_path)} has no geometry with {expect_vertices} "
        "vertices; fix_geo did not hold and the correspondence is gone")


def refine(mesh: dict, out_dir: str, max_buildings: int = DEFAULT_MAX_BUILDINGS,
           prompt: str = DEFAULT_PROMPT, preset: str = DEFAULT_PRESET,
           iters: Optional[int] = None, dry_run: bool = False) -> dict:
    """Refine the facades of the largest buildings. Writes a separate artifact.

    `dry_run` does everything except invoke the GPU: extraction, framing and the
    coarse OBJ per building. It is what makes this testable on a machine that
    cannot run the rest.
    """
    checks = preflight()
    if not dry_run and not checks["ok"]:
        raise FacadeUnavailable(
            "facade refinement needs: " + "; ".join(checks["missing"]))

    os.makedirs(out_dir, exist_ok=True)
    produced = []
    # 0 means every building. It is minutes each, so a hundred-building scene is
    # hours - which is a decision for whoever owns the GPU, not a default.
    chosen = building_groups(mesh)
    if max_buildings and max_buildings > 0:
        chosen = chosen[:max_buildings]
    for name, first, count in chosen:
        V, F = extract(mesh, first, count)
        if dry_run:
            import trimesh

            local, _, frame = to_object_frame(V)
            path = os.path.join(out_dir, f"{name}_coarse.obj")
            trimesh.Trimesh(vertices=local, faces=F, process=False).export(path)
            produced.append({"name": name, "vertices": int(len(V)),
                             "triangles": int(len(F)), "coarse": path,
                             "frame": frame, "dry_run": True})
            continue
        produced.append({**refine_one(V, F, out_dir, name, prompt, preset, iters),
                         "vertices": int(len(V)), "triangles": int(len(F))})

    record = {
        "schema": 1,
        "synthesised": True,
        "what": "Wall texture only. A nadir image does not photograph facades, "
                "so these are what a diffusion model expects a building of this "
                "shape to look like - plausible, not measured. Geometry is the "
                "pipeline's own and is unchanged.",
        "tool": "threefiner (3DTopia)",
        "preset": preset,
        "prompt": prompt,
        "buildings": produced,
        "environment": checks,
    }
    from ..core.jsonio import save_json

    save_json(record, os.path.join(out_dir, "facades.json"), indent=1)
    return record


def transfer_uv(src_V: np.ndarray, src_uv: np.ndarray, dst_V: np.ndarray):
    """Carry a painted UV from the full-resolution building onto the decimated one.

    The download is full resolution and the browser copy is quadric-decimated,
    so they do not share vertices - but they are the same building in the same
    metres, and a decimated vertex sits on (or within centimetres of) the
    original surface. Nearest neighbour is therefore the right transfer: it is
    a resampling of the parameterisation, not an interpolation of geometry, and
    it cannot move a vertex.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:                                # pragma: no cover
        return None
    if len(src_V) == 0 or len(dst_V) == 0:
        return None
    _, idx = cKDTree(np.asarray(src_V, np.float64)).query(
        np.asarray(dst_V, np.float64), k=1)
    return np.asarray(src_uv, np.float32)[idx]


# ------------------------------------------------------- the assembled model
def _world_uv(V: np.ndarray, span_x: float, span_y: float) -> np.ndarray:
    """The orthophoto UV, same convention as the tileset and the OBJ writer."""
    return np.stack([V[:, 0] / span_x, 1.0 - V[:, 1] / span_y], axis=1)


def assemble(mesh: dict, results, out_path: str, grid_shape, gsd_m: float,
             base_texture: Optional[str] = None) -> dict:
    """One model carrying threefiner's textures: the refined final .obj.

    A multi-material OBJ, which is the honest shape for this. The terrain and
    every unrefined building keep the orthophoto under `measured_mat`; each
    refined building gets a `synth_*` material pointing at its own synthesised
    texture. Opened in any tool, which surfaces are photographed and which are
    invented is visible in the material list rather than buried in a sidecar.

    **Vertices are this pipeline's, copied verbatim.** Nothing from the GLB
    reaches the geometry - only its UVs, and only for the group it came from.
    Groups share no vertices, which is what the structural rebuild guarantees,
    so one UV per vertex is unambiguous.
    """
    V = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["triangles"], np.int64)
    h, w = grid_shape
    span_x = max((w - 1) * float(gsd_m), 1e-9)
    span_y = max((h - 1) * float(gsd_m), 1e-9)

    uv = _world_uv(V, span_x, span_y)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    by_name = {r["name"]: r for r in (results or []) if r.get("glb")}
    materials: dict = {}
    refined: list = []
    for name, first, count in mesh.get("groups", []):
        rec = by_name.get(name)
        if rec is None:
            continue
        verts = np.unique(F[first:first + count])
        try:
            tex = read_texture(rec["glb"], expect_vertices=len(verts))
        except (FacadeUnavailable, OSError, ValueError) as exc:
            rec["skipped"] = str(exc)[:200]
            continue
        if tex["image"] is None:
            rec["skipped"] = "the refined mesh carried no texture image"
            continue
        png = name + ".png"
        tex["image"].save(os.path.join(out_dir, png))
        # The GLB's vertices are in the order they were handed over, which is
        # `verts`, so its UVs land on the vertices they were computed for.
        uv[verts] = tex["uv"]
        materials[name] = png
        refined.append(name)

    mtl_path = os.path.splitext(out_path)[0] + ".mtl"
    base_name = os.path.basename(base_texture) if base_texture else None
    if base_texture and os.path.exists(base_texture):
        target = os.path.join(out_dir, base_name)
        if os.path.abspath(target) != os.path.abspath(base_texture):
            shutil.copyfile(base_texture, target)

    n_buildings = len(list(building_groups(mesh)))
    with open(mtl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# TRAKSHA structural mesh, facades refined by threefiner\n")
        fh.write("# measured_mat is the orthophoto. Every synth_* material is a\n")
        fh.write("# SYNTHESISED facade: plausible, not photographed.\n")
        fh.write("newmtl measured_mat\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\nd 1\nillum 1\n")
        if base_name:
            fh.write("map_Kd " + base_name + "\n")
        for name, png in materials.items():
            fh.write("\nnewmtl synth_" + name + "\nKd 1 1 1\nKa 0 0 0\n")
            fh.write("Ks 0 0 0\nd 1\nillum 1\nmap_Kd " + png + "\n")

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# TRAKSHA structural mesh with refined facades\n")
        fh.write(f"# {len(refined)} of {n_buildings} buildings carry SYNTHESISED\n")
        fh.write("# wall texture (threefiner). Geometry is the calibrated\n")
        fh.write("# reconstruction, unchanged.\n")
        fh.write("# axes: +X east, +Y north, +Z up (metres from the SW corner)\n")
        fh.write("mtllib " + os.path.basename(mtl_path) + "\n")
        fh.write("o traksha_structural_refined\n")
        _write_block(fh, "v", V)
        _write_block(fh, "vt", uv, decimals=5)
        for name, first, count in mesh.get("groups", []):
            if count <= 0:
                continue
            fh.write("g " + name + "\n")
            mat = "synth_" + name if name in materials else "measured_mat"
            fh.write("usemtl " + mat + "\n")
            _write_faces(fh, F[first:first + count] + 1, True)

    return {
        "obj": out_path, "mtl": mtl_path,
        "uv": uv, "materials": dict(materials),
        "bytes": int(os.path.getsize(out_path)),
        "vertices": int(len(V)), "triangles": int(len(F)),
        "buildings_refined": len(refined),
        "buildings_total": n_buildings,
        "refined": refined,
        "synthesised": bool(refined),
        "textures": sorted(materials.values()),
    }

# The meshes the refined model supersedes. `structural.obj` has byte-identical
# geometry and strictly worse texture; `surface.obj` is the height field the
# structural mesh replaced. Keeping either alongside ships the same surface
# twice and leaves a reader choosing between them.
SUPERSEDED = ("surface.obj", "surface.mtl", "structural.obj", "structural.mtl")


def retire_superseded(mesh_dir: str, manifest: dict, primary_rel: str) -> list:
    """Delete what the refined model replaces, and unhook it from the manifest.

    The manifest is updated in the same call as the deletion on purpose: a
    download link pointing at a file that is no longer there is worse than one
    fewer download, and the two going out of step is exactly what happens when
    they are done in different places.

    The orthophoto is kept. It is the measured texture, the refined .mtl
    references it for every surface that was not painted, and it is the one
    image in the set that a camera actually took.
    """
    retired = []
    for name in SUPERSEDED:
        path = os.path.join(mesh_dir, name)
        if os.path.exists(path):
            os.remove(path)
            retired.append(name)

    mesh = manifest.setdefault("mesh", {})
    mesh.pop("obj", None)
    mesh.pop("mtl", None)
    mesh.setdefault("structural", {}).pop("obj", None)
    mesh["structural"].pop("mtl", None)
    mesh["retired"] = retired
    mesh["primary"] = primary_rel
    return retired
