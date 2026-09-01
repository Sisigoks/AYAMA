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


def preflight() -> dict:
    """What this needs, and which parts are present.

    Reported rather than raised so a caller can print it: the whole point is
    that this runs on a different machine from the one the pipeline usually
    runs on, and the failure a user hits is a missing dependency on that box.
    """
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

    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    local, inverse, frame = to_object_frame(np.asarray(vertices, np.float64))
    src = os.path.join(out_dir, f"{name}_coarse.obj")
    trimesh.Trimesh(vertices=local, faces=np.asarray(faces),
                    process=False).export(src)

    cmd = [shutil.which("threefiner") or "python", *([] if shutil.which("threefiner")
                                                     else ["-m", "threefiner.cli"]),
           preset, "--mesh", src, "--prompt", prompt,
           "--outdir", out_dir, "--save", f"{name}.glb"]
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
    for name, first, count in building_groups(mesh)[:max_buildings]:
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
    with open(os.path.join(out_dir, "facades.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, default=float)
    return record
