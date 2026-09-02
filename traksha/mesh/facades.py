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
turn a measurement into a guess. And belt-and-braces: nothing from threefiner's
output reaches the geometry. Only colour does, resampled onto this pipeline's
own vertices, so geometry invariance is true by construction rather than by
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

Two facts about what threefiner actually does shape everything below, because
each of them silently breaks the obvious implementation.

*It renders the mesh you hand it before it trains.* `fit_tex` is on by default
and initialises the trainable texture by rendering the input mesh from 512
orbits. That render needs either per-vertex colour or a UV atlas with an image;
handed a bare OBJ it dereferences `vt=None` and dies. So the handover carries
per-vertex colour sampled from the orthophoto - which is also the right
initialisation, because the roof it starts from is the roof that was
photographed.

*The mesh that comes back is not the mesh that went in.* Before the first
iteration threefiner runs kiui's `clean_mesh`, which merges every pair of
vertices closer than 1% of the bounding-box diagonal - on a 40 m building at
0.5 m ground sampling, most of the grid - and at export it unwraps a fresh
atlas, which splits vertices again along every chart seam. Neither the vertex
count nor the face order survives, so the texture is carried back through space
rather than through indices: see `bake`.

**Not verified on hardware.** This was written and tested on a CPU-only machine:
the extraction, the frame conversion, the handover, the bake and the guards are
covered by tests; the diffusion step itself has never been executed here. See
`preflight()` for what a GPU box needs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import numpy as np

from .obj import _write_block, _write_faces, _write_faces_uv

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
# largest rather than every building in the scene. The range is what a T4 does
# on the sd_fixgeo preset: 512 texture-fit iterations then 800 SDS steps at
# 512 px, plus a one-off model download on the first building.
MINUTES_PER_BUILDING = (6, 12)
DEFAULT_MAX_BUILDINGS = 8

# Generous, because the first building also pays for the diffusion weights
# coming down the wire and for nvdiffrast compiling its CUDA extension.
DEFAULT_TIMEOUT_S = 5400

# threefiner deletes connected components under 32 faces before it starts
# (kiui's `clean_mesh(min_f=32)`), and a building is one connected component.
# Below that the whole object disappears and the rasteriser is handed an empty
# mesh, so those are refused here, with the reason attached, rather than
# crashing several minutes into a subprocess.
MIN_FACES_FOR_REFINEMENT = 64

# The atlas each refined building gets. 1024 px over one building is about
# 3 cm per texel on a 30 m facade, finer than the diffusion prior resolves.
DEFAULT_TEXTURE_RES = 1024

# How far the refined surface may sit from ours before the result is refused,
# as a fraction of the building's largest dimension. threefiner's vertex merge
# moves the surface by at most half its own threshold, which is 1%; past 2% it
# is a different object rather than a cleaned one.
MAX_SURFACE_OFFSET = 0.02

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


# What has to be importable, and who needs it. The last four are threefiner's
# dependencies rather than ours, and they are checked anyway because every one
# of them is imported *late*: `xatlas` and `pygltflib` only at export, `sklearn`
# only in the UV padding that runs immediately before it. A box missing one of
# those trains for ten minutes and then throws the result away.
REQUIRED_MODULES = (
    ("threefiner", "the refiner itself", "pip install threefiner"),
    ("nvdiffrast", "threefiner's rasteriser",
     "pip install git+https://github.com/NVlabs/nvdiffrast (CUDA, source build)"),
    ("trimesh", "reading the refined GLB back", "pip install trimesh"),
    ("xatlas", "unwrapping the atlas the texture is baked into",
     "pip install xatlas"),
    ("scipy", "the nearest-surface query that the bake is", "pip install scipy"),
    ("PIL", "the texture images", "pip install pillow"),
    ("pymeshlab", "threefiner's mesh cleaner", "pip install pymeshlab"),
    ("cv2", "imported by kiui.op at import time", "pip install opencv-python"),
    ("sklearn", "threefiner's UV padding, at export", "pip install scikit-learn"),
    ("pygltflib", "writing the refined GLB, at export", "pip install pygltflib"),
)


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
            _, total = torch.cuda.mem_get_info()
            out["vram_gb"] = round(total / 1e9, 1)
            if total < 14e9:
                out["notes"].append(
                    f"{total / 1e9:.0f} GB of VRAM; use the sd_fixgeo preset")
        else:
            out["missing"].append("a CUDA device (threefiner has no CPU path)")
    except ImportError:
        out["missing"].append("torch")

    for mod, why, how in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            out["missing"].append(f"{mod} - {why} ({how})")

    if shutil.which("threefiner") is None and not any(
            m.startswith("threefiner") for m in out["missing"]):
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


def _group(mesh: dict, first: int, count: int):
    """One building: which scene vertices it uses, and its own mesh.

    The scene-vertex indices come back as well because everything downstream -
    the colour handover, the baked UV - has to land on the right rows of the
    scene arrays, and deriving them twice in two places is how the two end up
    disagreeing.
    """
    F = np.asarray(mesh["triangles"], np.int64)[first:first + count]
    used = np.unique(F)
    remap = np.full(int(used.max()) + 1, -1, np.int64)
    remap[used] = np.arange(len(used))
    return used, np.asarray(mesh["vertices"], np.float64)[used], remap[F]


def extract(mesh: dict, first: int, count: int):
    """One building as a standalone mesh, densely renumbered."""
    _, V, F = _group(mesh, first, count)
    return V, F


def to_object_frame(V: np.ndarray):
    """Metres, Z-up, world origin -> unit-ish, Y-up, centred. Returns the inverse.

    threefiner orbits a camera at radius 2.5 around an object it assumes is
    roughly unit scale and Y-up. A building 30 m tall sitting 800 m from the
    scene origin is neither, so it is moved rather than hoped about.

    1.8 is not an arbitrary choice of "roughly unit": kiui re-normalises
    whatever it loads into [-0.9, 0.9], so a mesh that is already exactly that
    makes kiui's normalisation the identity and leaves the two frames
    comparable afterwards.
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


# ------------------------------------------------------------- the handover
def vertex_colours(V: np.ndarray, texture_path: Optional[str], grid_shape,
                   gsd_m: float):
    """The orthophoto sampled at every vertex. `None` if there is no orthophoto.

    This is what threefiner's texture-fit stage starts from, and it is measured
    colour: the roof is the roof that was photographed, and a wall carries the
    footprint edge it already carried in `structural.obj`. The prior repaints
    the walls; starting it from grey would also let it repaint the roof.
    """
    if not texture_path or not os.path.exists(texture_path):
        return None
    try:
        from PIL import Image
    except ImportError:                                # pragma: no cover
        return None
    with Image.open(texture_path) as im:
        arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
    th, tw = arr.shape[:2]
    h, w = grid_shape
    span_x = max((w - 1) * float(gsd_m), 1e-9)
    span_y = max((h - 1) * float(gsd_m), 1e-9)
    V = np.asarray(V, np.float64)
    # Row 0 of the texture is the north edge; +Y in the mesh points north.
    col = np.clip(V[:, 0] / span_x, 0, 1) * (tw - 1)
    row = (1.0 - np.clip(V[:, 1] / span_y, 0, 1)) * (th - 1)
    return arr[np.rint(row).astype(np.int64), np.rint(col).astype(np.int64)]


DEFAULT_COLOUR = (0.55, 0.53, 0.50)


def write_handover(path: str, local: np.ndarray, faces: np.ndarray,
                   colours: Optional[np.ndarray] = None) -> str:
    """The OBJ threefiner is given: positions plus per-vertex colour.

    The colour is not decoration. threefiner's `fit_tex` stage renders this mesh
    to initialise its trainable texture, and kiui's renderer takes the vertex
    colour branch only when `vc` is set - otherwise it interpolates `vt`, which
    on a mesh with no UVs is `None`, and the run dies with an AttributeError
    several minutes in. Six floats on a `v` line is kiui's own spelling of
    vertex colour, so writing them is both the fix and the right initialisation.
    """
    local = np.asarray(local, np.float64)
    if colours is None:
        colours = np.tile(np.asarray(DEFAULT_COLOUR, np.float64), (len(local), 1))
    colours = np.clip(np.asarray(colours, np.float64)[:, :3], 0.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# TRAKSHA building handed to threefiner\n")
        fh.write("# normalised into [-0.9, 0.9], Y-up. Six floats per vertex:\n")
        fh.write("#   x y z r g b - the colour is the orthophoto, and kiui\n")
        fh.write("#   renders vertex colour instead of dereferencing a UV\n")
        fh.write("#   that a nadir reconstruction does not have.\n")
        _write_block(fh, "v", np.concatenate([local, colours], axis=1), decimals=6)
        _write_faces(fh, np.asarray(faces, np.int64) + 1, False)
    return path


# ------------------------------------------------------------------- the run
def refine_one(vertices: np.ndarray, faces: np.ndarray, out_dir: str, name: str,
               prompt: str = DEFAULT_PROMPT, preset: str = DEFAULT_PRESET,
               iters: Optional[int] = None, timeout_s: int = DEFAULT_TIMEOUT_S,
               colours: Optional[np.ndarray] = None) -> dict:
    """Run threefiner over one building. Returns what it produced.

    The mesh is written Y-up, normalised and vertex-coloured; threefiner is
    invoked as a subprocess - it is a console script with its own argument
    parser, and importing its GUI class to drive it in-process would couple this
    to its internals - and the result is read back for its colour only.
    """
    if preset not in FIXED_GEOMETRY_PRESETS:
        raise FacadeUnavailable(
            f"'{preset}' trains geometry. Only {', '.join(FIXED_GEOMETRY_PRESETS)} "
            "are allowed here: deforming a measured wall to satisfy a generative "
            "prior would turn a measurement into a guess.")

    _ensure_kiui_compatibility()
    os.makedirs(out_dir, exist_ok=True)
    local, _, frame = to_object_frame(np.asarray(vertices, np.float64))
    src = write_handover(os.path.join(out_dir, f"{name}_coarse.obj"),
                         local, faces, colours)
    glb = os.path.join(out_dir, f"{name}.glb")

    exe = shutil.which("threefiner")
    cmd = [exe] if exe else ["python", "-m", "threefiner.cli"]
    cmd += [preset, "--mesh", src, "--prompt", prompt,
            "--outdir", out_dir, "--save", f"{name}.glb",
            # Colab and every headless box: there is no GL context to make.
            "--force_cuda_rast"]
    if iters:
        cmd += ["--iters", str(int(iters))]

    result = {"name": name, "preset": preset, "prompt": prompt, "frame": frame,
              "coarse": src, "returncode": None, "glb": None}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result["error"] = (f"no result after {timeout_s}s; raise --timeout, or "
                           "lower --iters if the GPU is a slow one")
        return result
    except OSError as exc:                             # pragma: no cover
        result["error"] = f"could not start threefiner: {exc}"
        return result

    result["returncode"] = proc.returncode
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "").strip()[-800:]
        return result
    if not os.path.exists(glb):
        result["error"] = ("threefiner exited cleanly but wrote no "
                           f"{os.path.basename(glb)}")
        return result
    result["glb"] = glb
    return result


def refine(mesh: dict, out_dir: str, max_buildings: int = DEFAULT_MAX_BUILDINGS,
           prompt: str = DEFAULT_PROMPT, preset: str = DEFAULT_PRESET,
           iters: Optional[int] = None, dry_run: bool = False,
           texture: Optional[str] = None, grid_shape=None,
           gsd_m: float = 1.0, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """Refine the facades of the largest buildings. Writes a separate artifact.

    `dry_run` does everything except invoke the GPU: extraction, framing, the
    colour lookup and the handover OBJ per building. It is what makes this
    testable on a machine that cannot run the rest, and what it writes is
    byte-for-byte the file the GPU box would be given.
    """
    checks = preflight()
    if not dry_run and not checks["ok"]:
        raise FacadeUnavailable(
            "facade refinement needs: " + "; ".join(checks["missing"]))

    os.makedirs(out_dir, exist_ok=True)
    colours = None
    if texture and grid_shape is not None:
        colours = vertex_colours(np.asarray(mesh["vertices"], np.float64),
                                 texture, grid_shape, gsd_m)

    produced = []
    # 0 means every building. It is minutes each, so a hundred-building scene is
    # hours - which is a decision for whoever owns the GPU, not a default.
    chosen = building_groups(mesh)
    if max_buildings and max_buildings > 0:
        chosen = chosen[:max_buildings]
    for name, first, count in chosen:
        used, V, F = _group(mesh, first, count)
        entry = {"name": name, "vertices": int(len(V)), "triangles": int(len(F))}
        if not np.isfinite(V).all():
            # `%.6f` on a NaN writes `nan`, which kiui parses as a float and
            # then hands to a rasteriser, minutes later, on another machine.
            entry["skipped"] = "the building has non-finite vertices"
            produced.append(entry)
            continue
        if len(F) < MIN_FACES_FOR_REFINEMENT:
            entry["skipped"] = (
                f"{len(F)} triangles: threefiner deletes connected components "
                f"under 32 faces before it starts, and this building is one "
                f"component, so there would be nothing left to paint")
            produced.append(entry)
            continue
        local, _, frame = to_object_frame(V)
        col = None if colours is None else colours[used]
        if dry_run:
            entry["coarse"] = write_handover(
                os.path.join(out_dir, f"{name}_coarse.obj"), local, F, col)
            entry["frame"] = frame
            entry["dry_run"] = True
            produced.append(entry)
            continue
        entry.update(refine_one(V, F, out_dir, name, prompt, preset, iters,
                                timeout_s=timeout_s, colours=col))
        produced.append(entry)

    record = {
        "schema": 2,
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


# ------------------------------------------------------------------ the bake
def _normalised(P: np.ndarray) -> np.ndarray:
    """Centred on its own bounding box, scaled so the largest side is 1.

    Both meshes go through this before they are compared, which makes the
    comparison independent of whatever normalisation kiui applied on the way in
    and on the way out, and gives every distance below one unit: a fraction of
    the building's largest dimension.
    """
    P = np.asarray(P, np.float64)
    lo, hi = P.min(0), P.max(0)
    return (P - (hi + lo) / 2.0) / max(float(np.max(hi - lo)), 1e-9)


def _sample_image(arr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup, in trimesh's UV convention: v = 0 is the bottom row."""
    h, w = arr.shape[:2]
    x = np.clip(np.asarray(uv, np.float64)[:, 0], 0, 1) * (w - 1)
    y = (1.0 - np.clip(np.asarray(uv, np.float64)[:, 1], 0, 1)) * (h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = arr[y0, x0] * (1 - fx) + arr[y0, x1] * fx
    bottom = arr[y1, x0] * (1 - fx) + arr[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def read_refined(glb_path: str) -> dict:
    """The refined mesh as plain arrays: positions, faces, UVs, texture.

    Vertex positions are read only to be *compared* with ours, never to replace
    them. Which is the point: the correspondence between the two meshes is
    established in space, so nothing about threefiner's topology - and it does
    change the topology - can move a calibrated vertex.
    """
    import trimesh

    loaded = trimesh.load(glb_path, process=False, force="scene")
    geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    best = None
    for g in geoms:
        visual = getattr(g, "visual", None)
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        image = getattr(material, "baseColorTexture", None) if material else None
        if uv is None or image is None or len(getattr(g, "faces", ())) == 0:
            continue
        if best is None or len(g.faces) > len(best[0].faces):
            best = (g, uv, image)
    if best is None:
        raise FacadeUnavailable(
            f"{os.path.basename(glb_path)} carries no textured geometry: "
            "there is nothing in it to take colour from")
    g, uv, image = best
    return {"vertices": np.asarray(g.vertices, np.float64),
            "faces": np.asarray(g.faces, np.int64),
            "uv": np.asarray(uv, np.float64),
            "image": np.asarray(image.convert("RGB"), np.float32) / 255.0}


def _surface_samples(V: np.ndarray, F: np.ndarray, uv: np.ndarray,
                     arr: np.ndarray, count: int, rng) -> tuple:
    """Points scattered over a textured mesh, with the colour at each one.

    Area-weighted, so a large flat wall gets as many samples per square metre as
    a fiddly roof line, and dense enough that the nearest one to any of our
    texels is closer than a texel of the source texture.
    """
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    total = float(area.sum())
    if total <= 0:
        raise FacadeUnavailable("the refined mesh has no surface area")
    pick = np.searchsorted(np.cumsum(area / total), rng.random(count))
    pick = np.clip(pick, 0, len(F) - 1)
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    bary = np.stack([1.0 - s, s * (1.0 - r2), s * r2], axis=1)[:, :, None]
    corners = F[pick]
    points = (bary * V[corners]).sum(axis=1)
    texels = (bary * uv[corners]).sum(axis=1)
    return points, _sample_image(arr, texels)


def _unwrap(V: np.ndarray, F: np.ndarray):
    """xatlas: a UV atlas for our own triangles.

    threefiner's atlas belongs to threefiner's mesh, which is not this one. Ours
    is built here so that the texture written out is indexed by our faces and
    our vertices, and `vmapping` says which of our vertices each atlas vertex is
    a copy of - the seam duplicates that make a chart possible at all.
    """
    try:
        import xatlas
    except ImportError as exc:                         # pragma: no cover
        raise FacadeUnavailable("xatlas is needed to build the facade atlas "
                                "(pip install xatlas)") from exc
    vmapping, indices, uvs = xatlas.parametrize(
        np.ascontiguousarray(V, np.float32), np.ascontiguousarray(F, np.uint32))
    if len(indices) == 0:
        raise FacadeUnavailable("xatlas produced no charts for this building")
    return (np.asarray(vmapping, np.int64), np.asarray(indices, np.int64),
            np.asarray(uvs, np.float64))


def _rasterise_atlas(vt: np.ndarray, ft: np.ndarray, P: np.ndarray, res: int):
    """Every texel an atlas triangle covers, and the 3D point it stands for.

    A software rasteriser rather than nvdiffrast, because this has to be
    runnable - and testable - on the machine the rest of the pipeline runs on,
    and one pass over a few thousand triangles is not where the time goes.
    """
    h = w = int(res)
    pos = np.zeros((h, w, 3), np.float64)
    mask = np.zeros((h, w), bool)
    px = np.asarray(vt, np.float64)[:, 0] * w - 0.5
    py = (1.0 - np.asarray(vt, np.float64)[:, 1]) * h - 0.5

    for tri in np.asarray(ft, np.int64):
        x, y = px[tri], py[tri]
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(area) < 1e-12:
            continue
        x0 = max(int(np.floor(x.min())), 0)
        x1 = min(int(np.ceil(x.max())), w - 1)
        y0 = max(int(np.floor(y.min())), 0)
        y1 = min(int(np.ceil(y.max())), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
        w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not inside.any():
            # A chart triangle smaller than a texel still has to land
            # somewhere, or the surface it covers is simply missing.
            cx = int(round(float(x.mean()))), int(round(float(y.mean())))
            if 0 <= cx[0] < w and 0 <= cx[1] < h:
                pos[cx[1], cx[0]] = P[tri].mean(axis=0)
                mask[cx[1], cx[0]] = True
            continue
        b0, b1 = w0[inside], w1[inside]
        bary = np.stack([b0, b1, 1.0 - b0 - b1], axis=1)
        pos[gy[inside], gx[inside]] = bary @ P[tri]
        mask[gy[inside], gx[inside]] = True
    return pos, mask


def _pad(image: np.ndarray, mask: np.ndarray, passes: int = 12) -> np.ndarray:
    """Bleed colour into the gutters between charts.

    A texel just outside a chart is still sampled - by mipmapping, by bilinear
    filtering at the chart edge, by any renderer that does not know where the
    charts are - and if it is black the seam shows as a dark line around every
    face of the building. The dilation reaches `passes` texels; whatever is
    still empty past that is far enough from any chart that only a very coarse
    mip level reaches it, and it gets the building's average colour rather than
    black, because black is the one value that is certainly wrong.
    """
    out = image.copy()
    filled = mask.copy()
    average = out[mask].mean(axis=0) if mask.any() else np.zeros(3)
    for _ in range(passes):
        if filled.all():
            break
        acc = np.zeros_like(out)
        count = np.zeros(filled.shape, np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                have = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
                acc += shifted * have[..., None]
                count += have
        grow = (~filled) & (count > 0)
        if not grow.any():
            break
        out[grow] = acc[grow] / count[grow][:, None]
        filled |= grow
    out[~filled] = average
    return out


def bake(glb_path: str, V_local: np.ndarray, F: np.ndarray,
         resolution: int = DEFAULT_TEXTURE_RES,
         max_offset: float = MAX_SURFACE_OFFSET, seed: int = 0) -> dict:
    """Resample the refined colour onto our own triangles. Geometry stays ours.

    Indices cannot carry the texture back. threefiner merges every pair of
    vertices closer than 1% of the bounding-box diagonal before it starts and
    unwraps a fresh atlas at export, so its vertex count, its vertex order and
    its face order are all different from the ones handed over. What the two
    meshes do share is the surface, to within the merge threshold - so the
    transfer is done there: our atlas is rasterised into 3D points, each point
    takes the colour of the nearest point on the refined surface, and the result
    is a texture indexed by our faces.

    The distance of that nearest point is also the guarantee. If threefiner
    returned something that is not this building - `fix_geo` not holding, the
    wrong file, a mesh cleaned down to nothing - the refined surface is not
    where ours is, and the median offset says so before any of it is written.
    """
    refined = read_refined(glb_path)
    ours = _normalised(V_local)
    theirs = _normalised(refined["vertices"])
    shape_gap = float(np.max(np.abs(np.ptp(ours, axis=0) - np.ptp(theirs, axis=0))))
    if shape_gap > 0.05:
        raise FacadeUnavailable(
            f"the refined mesh is a different shape ({shape_gap:.0%} of the "
            "largest side out on one axis); its texture is not this building's")

    vmapping, ft, vt = _unwrap(V_local, F)
    pos, mask = _rasterise_atlas(vt, ft, ours[vmapping], resolution)
    covered = int(mask.sum())
    if covered == 0:
        raise FacadeUnavailable("the atlas came out empty")

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    n_samples = int(min(max(4 * covered, 100_000), 1_500_000))
    points, colours = _surface_samples(theirs, refined["faces"], refined["uv"],
                                       refined["image"], n_samples, rng)
    distance, nearest = cKDTree(points).query(pos[mask], k=1)
    offset = float(np.median(distance))
    if offset > max_offset:
        raise FacadeUnavailable(
            f"the refined surface sits {offset:.1%} of the building away from "
            "the measured one; fix_geo did not hold and the colour would be "
            "painted on the wrong walls")

    image = np.zeros((resolution, resolution, 3), np.float64)
    image[mask] = colours[nearest]
    image = _pad(image, mask)

    from PIL import Image

    return {
        "vt": vt, "ft": ft, "vmapping": vmapping,
        "image": Image.fromarray(
            np.clip(image * 255.0, 0, 255).astype(np.uint8), "RGB"),
        "coverage": covered / float(resolution * resolution),
        "offset": offset,
        "max_offset_seen": float(distance.max()),
    }


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
    """The orthophoto UV, same convention as `write_obj_structural`.

    OBJ's V axis runs bottom-up and +Y in the mesh points north, so v is y
    straight: the north edge of the image is v = 1. The browser copy flips it,
    because a WebGL texture uploads with row 0 first, but that is the browser's
    convention and not this file's.
    """
    return np.stack([V[:, 0] / span_x, V[:, 1] / span_y], axis=1)


def assemble(mesh: dict, results, out_path: str, grid_shape, gsd_m: float,
             base_texture: Optional[str] = None,
             resolution: int = DEFAULT_TEXTURE_RES) -> dict:
    """One model carrying threefiner's colour: the refined final .obj.

    A multi-material OBJ, which is the honest shape for this. The terrain and
    every unrefined building keep the orthophoto under `measured_mat`; each
    refined building gets a `synth_*` material pointing at its own baked
    texture. Opened in any tool, which surfaces are photographed and which are
    invented is visible in the material list rather than buried in a sidecar.

    **Vertices are this pipeline's, copied verbatim.** Nothing from the GLB
    reaches the geometry - only colour, resampled onto an atlas built for our
    own triangles (`bake`). A refined group's faces carry `v/vt` with the two
    indices differing, because a chart seam gives one vertex two texture
    coordinates; the vertex indices are still ours, unchanged and unreordered.
    """
    V = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["triangles"], np.int64)
    h, w = grid_shape
    span_x = max((w - 1) * float(gsd_m), 1e-9)
    span_y = max((h - 1) * float(gsd_m), 1e-9)

    world_uv = _world_uv(V, span_x, span_y)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    by_name = {r["name"]: r for r in (results or []) if r.get("glb")}
    materials: dict = {}
    refined: list = []
    painted: dict = {}          # name -> (vertex indices, texture indices)
    atlases: list = []          # the vt blocks appended after the world UVs
    # The browser copy can only carry one UV per vertex, so it gets a collapsed
    # copy: a seam vertex keeps whichever chart wrote last. The OBJ does not
    # have to make that compromise and does not.
    web_uv = world_uv.copy()
    next_vt = len(V)

    for name, first, count in mesh.get("groups", []):
        rec = by_name.get(name)
        if rec is None:
            continue
        used, V_sub, F_sub = _group(mesh, first, count)
        local, _, _ = to_object_frame(V_sub)
        try:
            got = bake(rec["glb"], local, F_sub, resolution=resolution)
        except (FacadeUnavailable, OSError, ValueError, ImportError) as exc:
            rec["skipped"] = str(exc)[:300]
            continue

        png = name + ".png"
        got["image"].save(os.path.join(out_dir, png))
        painted[name] = (used[got["vmapping"][got["ft"]]], got["ft"] + next_vt)
        atlases.append(got["vt"])
        next_vt += len(got["vt"])
        web_uv[used[got["vmapping"]]] = got["vt"]
        materials[name] = png
        refined.append(name)
        rec["bake"] = {"coverage": round(got["coverage"], 4),
                       "surface_offset": round(got["offset"], 5),
                       "texture": png, "resolution": int(resolution)}

    all_uv = np.concatenate([world_uv] + atlases) if atlases else world_uv

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
        _write_block(fh, "vt", all_uv, decimals=5)
        for name, first, count in mesh.get("groups", []):
            if count <= 0:
                continue
            fh.write("g " + name + "\n")
            if name in painted:
                fh.write("usemtl synth_" + name + "\n")
                v_idx, vt_idx = painted[name]
                _write_faces_uv(fh, v_idx + 1, vt_idx + 1)
            else:
                fh.write("usemtl measured_mat\n")
                _write_faces(fh, F[first:first + count] + 1, True)

    return {
        "obj": out_path, "mtl": mtl_path,
        "uv": web_uv, "materials": dict(materials),
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
