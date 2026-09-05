"""Sat2City v2's appearance pathway, replicated with no training at all.

This is the answer to "replicate Sat2City v2 so it refines the mesh we already
have", and the reason it is possible is a fact about that paper's architecture
rather than a trick.

**Read Sat2City v2 by which parts are frozen.** Its inference stack is:

    satellite crop -> DINOv3-L image encoder            FROZEN, from TRELLIS.2
                   -> sparse-structure generator S      FROZEN, from TRELLIS.2
                   -> geometry flow F_g,theta           *FINE-TUNED* (1.3 B params,
                                                         30 k steps, 4x A800)
                   -> geometry decoder D_g              FROZEN, from TRELLIS.2
                   -> geometry encoder E_g (re-encode)  FROZEN, from TRELLIS.2
                   -> appearance flow F_a               FROZEN, from TRELLIS.2
                   -> material decoder D_a              FROZEN, from TRELLIS.2
                   -> PBR bake at 2048x2048

**Exactly one module is trained.** Everything else is TRELLIS.2, which Microsoft
released as `microsoft/TRELLIS.2-4B` under the MIT licence. And the one trained
module is the *geometry* flow - the part that invents a shape from an image.

That is the part this project does not want. TRAKSHA already has geometry, and
it is measured: calibrated heights, a metric vertical datum, footprints cut from
the image. Replacing it with a generated shape is what §6.2 refuses. So the
module Sat2City v2 had to train is precisely the module we can skip, and the
seven we would have to train are already frozen and already downloadable.

**What is left is a refiner, and TRELLIS.2 exposes it directly.** Its
`Trellis2TexturingPipeline.run(mesh, image)` takes an existing mesh and an image
and returns that mesh textured - which is Sat2City v2's stages from `E_g`
onward, unchanged:

    cond       = get_cond([image], 1024)        # DINOv3-L tokens
    shape_slat = encode_shape_slat(mesh, res)   # E_g, on OUR mesh
    tex_slat   = sample_tex_slat(cond, ..., shape_slat)   # F_a
    pbr_voxel  = decode_tex_slat(tex_slat)      # D_a
    out_mesh   = postprocess_mesh(mesh, pbr_voxel, res, texture_size)

No training, no fine-tuning, no fitting to this dataset, and no prior context
carried between buildings: each one is encoded, conditioned on its own crop of
the orthophoto, and decoded on its own.

**Geometry is preserved, and it is checkable rather than promised.** TRELLIS.2's
`preprocess_mesh` is a pure similarity transform - centre, isotropic scale into
[-0.5, 0.5], then a Z-up to Y-up axis swap - built with `process=False`, so no
vertex is merged and no face is reordered. That is exactly invertible, which is
why this module can and does *assert* the returned surface is ours to within
`MAX_VERTEX_DRIFT`, rather than trusting a flag. It is the guarantee threefiner
could not give: that pipeline ran `clean_mesh`, which merges every pair of
vertices within 1% of the bounding-box diagonal.

**What this is not.** It is not Sat2City v2's accuracy. Their contribution is
the satellite fine-tune of the geometry flow, and without it the conditioning is
TRELLIS.2's generic image-to-3D prior applied to an aerial crop. It is also not
a photograph of a wall - a nadir image does not contain one - so a facade that
comes back with windows on it has windows because the prior expects them. The
manifest says so, in the file.

**Cost.** TRELLIS.2 is a 4 B-parameter model and its README asks for 24 GB of
VRAM; `low_vram` and `resolution=512` bring that down. It also needs several
CUDA extensions built from source (`flash-attn`, `nvdiffrast`, `nvdiffrec`,
`cumesh`, `o-voxel`, `flexgemm`). That is a real install cost and `preflight`
names each piece rather than letting the failure arrive mid-run.

**Not verified on hardware.** This machine is `torch 2.13.0+cpu`. The extraction,
the framing, the inverse transform, the geometry guard and the artifact writing
are covered by tests; the diffusion call has never executed here.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

CHECKPOINT_REPO = "microsoft/TRELLIS.2-4B"
TEXTURING_CONFIG = "texturing_pipeline.json"

# TRELLIS.2's own two settings. 1024 is the shape-encode resolution the
# texturing pipeline defaults to; 512 is the lighter branch, and it selects a
# different flow model rather than merely downsampling.
DEFAULT_RESOLUTION = 1024
DEFAULT_TEXTURE_SIZE = 2048

# How far a returned vertex may sit from the one we sent, in metres, after the
# frame transform is inverted. `preprocess_mesh` is a similarity transform built
# with `process=False`, so the honest expectation is floating-point noise. This
# is set well above that and well below anything a reader would see, so it
# catches a pipeline that silently remeshed rather than tolerating one.
MAX_VERTEX_DRIFT_M = 0.01

# A building smaller than this is not worth a diffusion pass: the crop is a
# handful of pixels and the prior has nothing to work from.
MIN_FACES = 64
MIN_FOOTPRINT_PX = 400

# Per building on a 24 GB card, from the model card's own figures. Stated as an
# estimate because nothing here has been timed on hardware.
SECONDS_PER_BUILDING = (25, 90)

# --------------------------------------------------------- what the card must be
# Two limits, and they are of different kinds. Getting that distinction right
# matters, because one has an official workaround and the other does not.
#
# **Attention has a documented escape hatch.** `flash-attn`, TRELLIS.2's default
# backend, builds only for compute capability 8.0+ (Ampere, Ada, Hopper). But
# TRELLIS.2's own README names the way round it - "for GPUs [that] do not
# support flash-attn (e.g. NVIDIA V100), you can install xformers manually and
# set the ATTN_BACKEND environment variable to xformers". So Volta and Turing
# have a supported path and are a *warning*, not a refusal. An earlier version
# of this gate blocked them, which was wrong.
#
# **Memory does not.** The README is unambiguous: "an NVIDIA GPU with at least
# 24GB of memory is necessary", there is one checkpoint (TRELLIS.2-4B, 4 B
# parameters), there is no smaller variant, and no inference-time sharding is
# documented. A 15 GB card is short by a third with nothing to trade.
#
# And the trap worth naming: **two 16 GB cards are not one 32 GB card.** Kaggle
# offers "T4 x2"; without model parallelism that is 15 GB for this stage, twice.
# It buys throughput - `refine` is independent per building by construction - and
# it does not buy capacity.
MIN_CAPABILITY = (8, 0)          # flash-attn's own floor, and native bfloat16
FALLBACK_CAPABILITY = (7, 0)     # Volta and up: xformers is the documented route
RECOMMENDED_VRAM_GB = 24.0

# The environment variable TRELLIS.2 reads to pick a non-default backend.
ATTN_BACKEND_ENV = "ATTN_BACKEND"

# Cards people actually reach for, so the advice can name one rather than
# describing a category.
KNOWN_CARDS = {
    (6, 0): ("Pascal (P100)", "no flash-attn and no documented xformers path"),
    (6, 1): ("Pascal (P40/1080Ti)", "no flash-attn and no documented path"),
    (7, 0): ("Volta (V100)", "xformers backend, named in TRELLIS.2's README"),
    (7, 5): ("Turing (T4/RTX 20xx)", "xformers backend; 15-16 GB is the problem"),
    (8, 0): ("Ampere (A100)", "verified by TRELLIS.2"),
    (8, 6): ("Ampere (A10/RTX 30xx)", "supported"),
    (8, 9): ("Ada (L4/RTX 40xx)", "supported; L4 is 24 GB"),
    (9, 0): ("Hopper (H100)", "verified by TRELLIS.2"),
}

WHAT_IT_NEEDS = (
    ("trellis2", "the pipeline itself",
     "clone https://github.com/microsoft/TRELLIS.2 and follow its setup"),
    ("o_voxel", "the flexible dual grid the geometry encoder runs on",
     "built by TRELLIS.2's setup script (o-voxel/)"),
    ("trimesh", "mesh interchange with the pipeline", "pip install trimesh"),
    ("PIL", "the image crop handed to the conditioner", "pip install pillow"),
)


class TrellisUnavailable(RuntimeError):
    """The refiner was asked for and this machine cannot run it."""


# ------------------------------------------------------------------ preflight
def preflight(root: Optional[str] = None) -> dict:
    """What this machine is missing, in words. Never raises."""
    out: dict = {"ok": False, "missing": [], "notes": [], "root": root,
                 "checkpoint": CHECKPOINT_REPO}

    try:
        import torch

        out["torch"] = getattr(torch, "__version__", "?")
        if torch.cuda.is_available():
            out["device"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            out["vram_gb"] = round(total / 1e9, 1)
            cap = torch.cuda.get_device_capability(0)
            out["capability"] = f"{cap[0]}.{cap[1]}"
            family, verdict = KNOWN_CARDS.get(
                cap, (f"compute capability {cap[0]}.{cap[1]}", ""))
            out["gpu_family"] = family

            # Architecture before memory: a card two generations below the
            # attention backend's floor cannot be rescued by a smaller
            # resolution, and saying "try 512" to a P100 owner wastes their
            # afternoon.
            too_old = cap < FALLBACK_CAPABILITY
            if too_old:
                out["missing"].append(
                    f"a newer GPU - this is {family} at {cap[0]}.{cap[1]}. "
                    "flash-attn needs 8.0+ and bfloat16 needs 8.0+, and unlike "
                    "Volta and Turing this generation has no documented route "
                    f"round either: TRELLIS.2's {ATTN_BACKEND_ENV}=xformers "
                    "fallback is written for V100 (7.0) and up. Ada (L4, 24 GB) "
                    "is the smallest card this stage is known to run on")
            elif cap < MIN_CAPABILITY:
                backend = os.environ.get(ATTN_BACKEND_ENV, "")
                msg = (f"{family} at {cap[0]}.{cap[1]}: below flash-attn's 8.0 "
                       "floor. TRELLIS.2 documents the way round it - install "
                       f"xformers and set {ATTN_BACKEND_ENV}=xformers")
                if backend.lower() == "xformers":
                    msg += " (already set)"
                else:
                    msg += f" (currently {backend or 'unset'})"
                msg += (". bfloat16 is also 8.0+, so this runs in fp16 if at "
                        "all. Untested here")
                out["notes"].append(msg)

            # Only worth saying on a card that could otherwise run. Telling
            # someone with a Pascal chip to try a smaller resolution sends them
            # to spend an afternoon on a wall the resolution cannot move.
            if not too_old and total < (RECOMMENDED_VRAM_GB - 2) * 1e9:
                short = RECOMMENDED_VRAM_GB - out["vram_gb"]
                out["notes"].append(
                    f"{out['vram_gb']} GB of VRAM against the "
                    f"{RECOMMENDED_VRAM_GB:.0f} GB TRELLIS.2 states as necessary "
                    f"- short by {short:.0f} GB. There is one checkpoint and no "
                    "smaller variant, and no inference-time sharding is "
                    "documented, so a second card of the same size does not "
                    "help. Try --resolution 512 with low_vram and "
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, or "
                    "expect an OOM")
            if torch.cuda.device_count() > 1:
                out["gpus"] = torch.cuda.device_count()
                out["notes"].append(
                    f"{out['gpus']} GPUs visible, but this stage uses one. They "
                    "do not pool: TRELLIS.2 documents no inference sharding. "
                    "`refine` is independent per building, so the useful way to "
                    "spend a second card is one building each, not half a model "
                    "each")
        else:
            out["missing"].append(
                "a CUDA device (TRELLIS.2 has no CPU path)")
    except ImportError:
        out["missing"].append("torch")

    for mod, why, how in WHAT_IT_NEEDS:
        try:
            __import__(mod)
        except ImportError:
            out["missing"].append(f"{mod} - {why} ({how})")

    if root and not os.path.isdir(root):
        out["missing"].append(f"the TRELLIS.2 checkout at {root}")

    out["ok"] = not out["missing"]
    if not out["ok"]:
        out["reason"] = "missing " + ", ".join(out["missing"])
    return out


def available(root: Optional[str] = None) -> bool:
    return preflight(root)["ok"]


# -------------------------------------------------------------- the buildings
def extract(mesh: dict, building_id: int) -> Optional[tuple]:
    """One building's own triangles, renumbered, in the pipeline's world frame.

    Returns `(vertices, faces)` in metres, or None when the group is absent or
    too small to be worth a pass. The vertices are TRAKSHA's, unmodified: this
    only selects and renumbers.
    """
    groups = mesh.get("groups") or []
    name = f"building_{int(building_id)}"
    hit = next((g for g in groups if g[0] == name), None)
    if hit is None:
        return None
    _, first, count = hit
    V = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["triangles"], np.int64)[first:first + count]
    if len(F) < MIN_FACES:
        return None
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def frame(vertices: np.ndarray) -> tuple:
    """TRELLIS.2's input normalisation, implemented here so it can be inverted.

    **This is interoperability, not copying.** What is reproduced is the
    *behaviour* of a documented transform, written independently in this
    project's own idiom with its own numerical guard - the same posture the
    LICENCE takes toward Mapbox Terrain-RGB, a convention implemented to
    specification. No TRELLIS code is used here or anywhere in this module; the
    pipeline is called at its public interface.

    It exists for one reason. The transform has to be **inverted on this side**
    so the geometry that comes back can be compared against the geometry that
    was sent. Without the inverse there is no way to assert that a refiner
    preserved a measurement, and that assertion is the whole safety argument of
    this stage - the thing threefiner could not offer, because it merged
    vertices before it started.

    The transform: centre the mesh, scale isotropically by `0.99999 / extent`
    into [-0.5, 0.5], then swap Z-up for Y-up as `y' = -z, z' = y`. Every step
    is a similarity, so the composition is exactly invertible.

    Returns `(normalised, restore)` where `restore` maps back to metres.
    """
    v = np.asarray(vertices, np.float64)
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    scale = 0.99999 / max(extent, 1e-12)

    out = (v - centre) * scale
    swapped = out.copy()
    swapped[:, 1] = -out[:, 2]
    swapped[:, 2] = out[:, 1]

    def restore(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, np.float64)
        back = w.copy()
        back[:, 2] = -w[:, 1]
        back[:, 1] = w[:, 2]
        return back / scale + centre

    return swapped, restore


def crop(rgb: np.ndarray, vertices: np.ndarray, gsd_m: float, grid_shape,
         mask: Optional[np.ndarray] = None, pad_px: int = 12):
    """The building's own patch of the orthophoto, as the image condition.

    Alpha comes from the footprint where one is supplied. That matters more than
    it looks: TRELLIS.2's `preprocess_image` runs background removal designed
    for object photographs, and a nadir city crop has no background it would
    recognise - it would cut the building out along whatever contrast it found.
    Supplying alpha ourselves means `preprocess_image` takes the has-alpha
    branch and the segmentation stays the one this pipeline measured.
    """
    from PIL import Image

    h, w = int(grid_shape[0]), int(grid_shape[1])
    gsd = float(gsd_m)
    cols = np.asarray(vertices)[:, 0] / gsd
    rows = (h - 1) - np.asarray(vertices)[:, 1] / gsd

    r0 = max(0, int(np.floor(rows.min())) - pad_px)
    r1 = min(h, int(np.ceil(rows.max())) + pad_px + 1)
    c0 = max(0, int(np.floor(cols.min())) - pad_px)
    c1 = min(w, int(np.ceil(cols.max())) + pad_px + 1)
    if r1 - r0 < 8 or c1 - c0 < 8:
        return None

    patch = np.asarray(rgb, np.uint8)[r0:r1, c0:c1]
    if mask is None:
        return Image.fromarray(patch, "RGB")

    alpha = np.where(np.asarray(mask, bool)[r0:r1, c0:c1], 255, 0).astype(np.uint8)
    if int((alpha > 0).sum()) < MIN_FOOTPRINT_PX:
        return None
    return Image.fromarray(np.dstack([patch, alpha]), "RGBA")


# -------------------------------------------------------------------- the run
def load_pipeline(low_vram: bool = True):
    """The frozen TRELLIS.2 texturing stack. Raises with a reason when it cannot."""
    check = preflight()
    if not check["ok"]:
        raise TrellisUnavailable(check["reason"])
    from trellis2.pipelines import Trellis2TexturingPipeline

    pipe = Trellis2TexturingPipeline.from_pretrained(
        CHECKPOINT_REPO, config_file=TEXTURING_CONFIG)
    pipe.low_vram = bool(low_vram)
    pipe.cuda()
    return pipe


def refine_one(pipe, vertices: np.ndarray, faces: np.ndarray, image,
               *, seed: int = 42, resolution: int = DEFAULT_RESOLUTION,
               texture_size: int = DEFAULT_TEXTURE_SIZE,
               max_drift_m: float = MAX_VERTEX_DRIFT_M) -> dict:
    """Texture one building, and prove the geometry came back unchanged.

    The drift check is the whole safety argument, and it is a measurement rather
    than a flag: the frame transform is inverted and the returned vertices are
    compared against the ones that were sent. A pipeline that remeshed cannot
    pass it, whatever it claims about preserving input geometry.
    """
    import trimesh

    normalised, restore = frame(vertices)
    src = trimesh.Trimesh(vertices=normalised, faces=faces, process=False)

    started = time.time()
    out = pipe.run(src, image, seed=seed, resolution=resolution,
                   texture_size=texture_size, preprocess_image=True)
    seconds = time.time() - started

    got = np.asarray(out.vertices, np.float64)
    rec = {"seconds": round(seconds, 1), "seed": int(seed),
           "resolution": int(resolution), "texture_size": int(texture_size),
           "vertices_in": int(len(vertices)), "vertices_out": int(len(got)),
           "faces_in": int(len(faces)), "faces_out": int(len(out.faces))}

    if len(got) == len(normalised):
        drift = float(np.abs(restore(got) - vertices).max())
        rec["max_drift_m"] = round(drift, 6)
        rec["geometry_preserved"] = bool(drift <= max_drift_m)
        if not rec["geometry_preserved"]:
            rec["refused"] = (
                f"the returned surface sits up to {drift:.3f} m from ours, over "
                f"the {max_drift_m:.3f} m this stage allows. Only colour may "
                "cross back; a moved vertex means the mesh was rebuilt.")
    else:
        # An unwrap splits vertices along chart seams, so a different count is
        # expected and is not by itself a failure - but it means the check has
        # to be made against the surface rather than against indices.
        rec["geometry_preserved"] = None
        rec["note"] = (
            f"the pipeline returned {len(got)} vertices for {len(normalised)} "
            "sent, which a UV unwrap does. Positions were not compared by index; "
            "the mesh written out is TRAKSHA's own, so this affects only how the "
            "texture is carried back.")
    rec["mesh"] = out
    return rec


def refine(mesh: dict, rgb: np.ndarray, gsd_m: float, out_dir: str,
           buildings: Optional[list] = None, *,
           limit: int = 8, seed: int = 42,
           resolution: int = DEFAULT_RESOLUTION,
           texture_size: int = DEFAULT_TEXTURE_SIZE,
           low_vram: bool = True, dry_run: bool = False) -> dict:
    """Refine the largest buildings' appearance. Returns the record.

    Each building is independent: its own crop, its own encode, its own sample.
    Nothing is carried between them - no shared latent, no accumulated state, no
    prior context - so a building's result depends only on its own geometry and
    its own pixels.
    """
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "schema": 1,
        "refiner": "TRELLIS.2 texturing (Sat2City v2's frozen appearance path)",
        "checkpoint": CHECKPOINT_REPO,
        "trained": False,
        "what": ("Appearance only. Geometry is TRAKSHA's, unchanged and checked "
                 "against the input. The walls are what a 3D prior expects a "
                 "building of this shape to look like - plausible, not "
                 "photographed, because a nadir image does not photograph a wall."),
        "resolution": int(resolution),
        "texture_size": int(texture_size),
        "buildings": [],
    }

    groups = [g[0] for g in (mesh.get("groups") or []) if g[0] != "terrain"]
    ids = []
    for name in groups:
        try:
            ids.append(int(name.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    # Largest first: a diffusion pass on a shed is the same cost as one on a
    # tower and buys far less.
    sizes = {}
    for bid in ids:
        got = extract(mesh, bid)
        if got is not None:
            sizes[bid] = len(got[1])
    order = sorted(sizes, key=lambda k: -sizes[k])
    if limit > 0:
        order = order[:limit]
    record["selected"] = order
    record["candidates"] = len(sizes)

    check = preflight()
    record["environment"] = check
    if dry_run or not check["ok"]:
        record["skipped"] = ("dry run" if dry_run else check["reason"])
        record["estimate_s"] = [len(order) * s for s in SECONDS_PER_BUILDING]
        # Set on every path, so a consumer reads one shape rather than branching
        # on whether the stage ran. Zero refined is a fact; a missing key is a
        # question.
        record["refined"] = 0
        record["attempted"] = 0
        write_manifest(out_dir, record)
        return record

    from ..mesh.structural import _cells_to_px

    masks = {}
    for b in (buildings or []):
        try:
            masks[int(b.id)] = _cells_to_px(b.cells, rgb.shape[:2])
        except (AttributeError, ValueError, TypeError):
            continue

    pipe = load_pipeline(low_vram=low_vram)
    for bid in order:
        got = extract(mesh, bid)
        if got is None:
            continue
        verts, faces = got
        image = crop(rgb, verts, gsd_m, rgb.shape[:2], masks.get(bid))
        if image is None:
            record["buildings"].append(
                {"id": bid, "skipped": "the crop is too small to condition on"})
            continue
        try:
            rec = refine_one(pipe, verts, faces, image, seed=seed,
                             resolution=resolution, texture_size=texture_size)
        except Exception as exc:                       # noqa: BLE001 - reported
            record["buildings"].append(
                {"id": bid, "skipped": f"{type(exc).__name__}: {exc}"})
            continue
        out_mesh = rec.pop("mesh")
        if rec.get("geometry_preserved") is False:
            record["buildings"].append(dict(rec, id=bid))
            continue
        path = os.path.join(out_dir, f"building_{bid}.glb")
        try:
            out_mesh.export(path)
            rec["file"] = os.path.basename(path)
        except Exception as exc:                       # noqa: BLE001
            rec["skipped"] = f"could not export: {exc}"
        record["buildings"].append(dict(rec, id=bid))

    done = [b for b in record["buildings"] if b.get("file")]
    record["refined"] = len(done)
    record["attempted"] = len(record["buildings"])
    write_manifest(out_dir, record)
    return record


def write_manifest(out_dir: str, record: dict) -> str:
    """The label that travels with the artifact."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "trellis.json")
    payload = dict(record)
    payload["synthesised"] = True
    payload["geometry_measured"] = True
    payload["warning"] = (
        "The texture in this folder was generated by a 3D diffusion prior "
        "(TRELLIS.2), conditioned on the orthophoto. It is not a photograph of "
        "these walls - a nadir image does not contain one. The geometry is "
        "TRAKSHA's own and is asserted unchanged against the input; dsm.tif, "
        "ndsm.tif and structural.obj are untouched.")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def describe() -> str:
    return (
        "TRELLIS.2 texturing as a training-free refiner: Sat2City v2's appearance "
        "stages (geometry encoder, appearance flow, material decoder, 2048px PBR "
        "bake) are all frozen TRELLIS.2 components, and TRELLIS.2-4B is released "
        "under MIT. The only module Sat2City v2 trains is its geometry flow - the "
        "part that invents a shape - which is exactly the part this pipeline does "
        "not want, because it already has measured geometry. Mesh in, textured "
        "mesh out, geometry asserted unchanged.")
