"""Turn a Phase 2 run directory into a tileset a browser can render.

Input is exactly what `ayama.cli run` already writes - no new inference, no new
calibration, nothing re-derived. Phase 3 is a *delivery* phase: if a number
appears in the viewer it came out of the rasters Phase 2 produced, and the
manifest records which run produced them.

The manifest is the contract between this module and `web/app.js`. It is
deliberately explicit about encodings and ranges rather than letting the
JavaScript assume defaults, because the failure mode of an implied convention
is a viewer that renders a wrong surface confidently.

It also carries `notes`: warnings *derived from the data*, not hardcoded. On the
current benchmark the honest note is that the predicted nDSM spans 0.28 m, so
the surface is flat and the viewer says so on screen. A 3D view of a flat city
that does not admit it is worse than no 3D view, because a rendered scene reads
as a finished result.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from ..core.jsonio import save_json
from .encode import (encode_linear, encode_terrain_rgb, normal_map,
                     quantisation_step)
from .obj import write_obj
from .tiles import cut, grid_size, interior, tile_specs

TILESET_VERSION = 1

# Layers the viewer knows how to draw, in the order it offers them.
LAYER_SPEC = (
    ("dsm", "dsm.tif", "terrain-rgb", "m", "Surface elevation"),
    ("ndsm", "ndsm.tif", "linear", "m", "Height above ground"),
    ("sigma", "sigma.tif", "linear", "m", "Uncertainty, 1 sigma"),
    ("error", "error.tif", "linear", "m", "Predicted minus reference"),
)


def _read_raster(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float32)
        nodata = ds.nodata
    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    return a


def _read_rgb(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), np.uint8)


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    import json

    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def load_run(run_dir: str) -> dict:
    """Read every artifact a run wrote. Missing optional layers are simply absent."""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"not a run directory: {run_dir}")
    dsm = _read_raster(os.path.join(run_dir, "dsm.tif"))
    if dsm is None:
        raise FileNotFoundError(
            f"{run_dir} has no dsm.tif - point this at a directory written by "
            "`ayama run`, not at a study root"
        )
    out = {
        "dir": os.path.abspath(run_dir),
        "dsm": dsm,
        "texture": _read_rgb(os.path.join(run_dir, "texture.jpg")),
        "provenance": _read_json(os.path.join(run_dir, "provenance.json")),
        "summary": _phase2_summary(run_dir),
        "meta": _scene_meta(os.path.join(run_dir, "dsm.tif")),
    }
    for key, fname, _enc, _units, _label in LAYER_SPEC:
        if key == "dsm":
            continue
        out[key] = _read_raster(os.path.join(run_dir, fname))
    sem = _read_raster(os.path.join(run_dir, "sem.tif"))
    out["sem"] = None if sem is None else sem.astype(np.int16)
    return out


def _scene_meta(path: str) -> dict:
    try:
        import rasterio
    except Exception:  # pragma: no cover
        return {}
    with rasterio.open(path) as ds:
        crs = str(ds.crs) if ds.crs else None
        t = ds.transform
        gsd = float(abs(t.a)) if t is not None else 1.0
        bounds = None
        if ds.crs:
            try:
                from rasterio.warp import transform_bounds

                bounds = list(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
            except Exception:
                bounds = None
        return {
            "crs": crs,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f] if t is not None else None,
            "gsd_m": gsd,
            "bounds_wgs": bounds,
            "tags": {k: v for k, v in ds.tags().items() if k.startswith("AYAMA_")},
        }


def _lod_count(shape: tuple, tile: int, min_side: int = 128) -> int:
    """Enough levels that the coarsest is still bigger than `min_side`."""
    side = min(int(shape[0]), int(shape[1]))
    n = 1
    while side // (2 ** n) >= min_side and n < 6:
        n += 1
    return n


def _save_png(rgb: np.ndarray, path: str) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb)).save(path, optimize=True)


def _save_jpg(rgb: np.ndarray, path: str, quality: int = 88) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb)).save(path, quality=quality)


def _finite_stats(a: Optional[np.ndarray]) -> dict:
    if a is None:
        return {}
    f = np.asarray(a, np.float64)
    m = np.isfinite(f)
    if not m.any():
        return {}
    v = f[m]
    return {
        "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
        "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99)),
    }


def derive_notes(run: dict, layer_ranges: dict) -> list:
    """Warnings computed from the data. Nothing here is hardcoded to a scene.

    The first check is the one that matters. Phase 2 currently emits a surface
    whose height-above-ground spans a fraction of a metre; rendered in 3D that
    is an empty plain, and a viewer that draws it without comment is presenting
    a defect as a deliverable.
    """
    notes = []
    ndsm = run.get("ndsm")
    sem = run.get("sem")
    if ndsm is not None:
        st = _finite_stats(ndsm)
        p99 = st.get("p99", 0.0)
        top = st.get("max", 0.0)
        has_buildings = bool(sem is not None and (sem == 2).sum() > 100)
        if has_buildings and p99 < 1.0:
            notes.append({
                "level": "critical",
                "id": "flat_surface",
                "text": (
                    f"Predicted height above ground reaches only {top:.2f} m "
                    f"(99th percentile {p99:.2f} m) on a scene where "
                    f"{100.0 * float((sem == 2).mean()):.1f}% of pixels are classified as "
                    "building. The calibration scale field has collapsed to its floor, so "
                    "this surface is terrain with the structures flattened. See README "
                    "section 5. Raise the vertical exaggeration to see what little relief "
                    "there is - it is a defect, not a rendering choice."
                ),
            })
        elif has_buildings and p99 < 3.0:
            notes.append({
                "level": "warning", "id": "low_relief",
                "text": f"Height above ground reaches {top:.2f} m; structures look under-built.",
            })

    sigma = run.get("sigma")
    if sigma is not None:
        s = _finite_stats(sigma).get("mean")
        for key, rng in layer_ranges.items():
            step = rng.get("step_m")
            if s and step and step > s:
                notes.append({
                    "level": "warning", "id": f"quantisation_{key}",
                    "text": (f"The {key} layer quantises at {step:.3g} m, coarser than its own "
                             f"mean uncertainty of {s:.2f} m."),
                })

    if run.get("texture") is None:
        notes.append({"level": "info", "id": "no_texture",
                      "text": "No texture.jpg in the run; the photo drape is unavailable."})

    prov = run.get("provenance") or {}
    if str(prov.get("segmentation", "")).startswith("heuristic"):
        notes.append({"level": "info", "id": "heuristic_segmentation",
                      "text": "Semantics came from the colour heuristic, not a trained model."})
    if "sim" in str(prov.get("dem", "")).lower() or "simulated" in str(prov.get("dem", "")).lower():
        notes.append({"level": "info", "id": "simulated_dem",
                      "text": f"Anchored to a simulated DEM ({prov.get('dem')}), not a real product."})
    return notes


def _phase2_summary(run_dir: str) -> dict:
    """The Phase 2 numbers for this run: metrics, tier, anchor counts.

    `ayama run` writes `summary.json` beside the rasters, but `ayama study`
    keeps every scene's metrics together in one `study.json` instead. Rather
    than make the viewer show nothing for the runs the study produced - which
    are the ones the README actually reports - the study file is consulted and
    the matching scene picked out by seed. Nothing is recomputed either way.
    """
    direct = _read_json(os.path.join(run_dir, "summary.json"))
    if direct.get("metrics"):
        return direct

    seed_dir = os.path.dirname(os.path.abspath(run_dir))
    import re

    m = re.fullmatch(r"seed(\d+)", os.path.basename(seed_dir))
    if not m:
        return direct
    study = _read_json(os.path.join(os.path.dirname(seed_dir), "study.json"))
    seeds = ((study.get("config") or {}).get("seeds")) or []
    scenes = study.get("scenes") or []
    try:
        i = list(seeds).index(int(m.group(1)))
    except ValueError:
        return direct
    if 0 <= i < len(scenes):
        out = dict(scenes[i])
        out["source"] = "study.json"
        return out
    return direct


def _sun_from_source(prov: dict) -> dict:
    """Sun angles for the viewer's light, read from the source image's tags.

    The run directory does not carry them - `provenance.json` records which
    image it came from, so the tags are read straight off that raster rather
    than re-ingesting it. Lighting the surface from the scene's own sun is not
    decoration: shadows in the draped texture then fall the same way the shading
    does, and a mismatch between the two is immediately obvious.
    """
    img = str(prov.get("image") or "")
    if not img or not os.path.exists(img):
        return {}
    try:
        import rasterio

        with rasterio.open(img) as ds:
            tags = {k.upper(): v for k, v in ds.tags().items()}
    except Exception:
        return {}
    az_keys = ("SUN_AZIMUTH", "SUNAZIMUTH", "SOLAR_AZIMUTH", "MEANSUNAZ", "SUN_AZ")
    el_keys = ("SUN_ELEVATION", "SUNELEVATION", "SOLAR_ELEVATION", "MEANSUNEL",
               "SUN_ELEV", "SUN_EL")

    def first(keys):
        for k in keys:
            if k in tags:
                try:
                    return float(tags[k])
                except (TypeError, ValueError):
                    continue
        return None

    az, el = first(az_keys), first(el_keys)
    if az is None or el is None:
        return {}
    return {
        "sun_azimuth_deg": az,
        "sun_elevation_deg": el,
        "sun": (f"{az:.1f} deg azimuth / {el:.1f} deg elevation, read from the source "
                "image. The surface is lit from the scene's own sun, so shading and the "
                "shadows in the draped texture agree."),
    }


def build_tileset(
    run_dir: str,
    out_dir: str,
    tile: int = 512,
    pad: int = 1,
    lods: Optional[int] = None,
    obj_stride: int = 2,
    write_mesh: bool = True,
    on_progress=None,
) -> dict:
    """Build `out_dir` from a Phase 2 run. Returns the manifest dict."""
    run = load_run(run_dir)
    dsm = run["dsm"]
    H, W = dsm.shape
    meta = run["meta"] or {}
    gsd = float(meta.get("gsd_m") or 1.0)
    os.makedirs(out_dir, exist_ok=True)

    n_lods = int(lods) if lods else _lod_count((H, W), tile)

    # Ranges are computed once, at full resolution, and shared by every LOD.
    # Per-LOD ranges would make the same elevation decode to different metres at
    # different zooms, which is the kind of bug that only shows up as a surface
    # that subtly breathes while you scroll.
    layer_ranges: dict = {}
    available = []
    for key, _fname, enc, units, label in LAYER_SPEC:
        a = run.get(key)
        if a is None:
            continue
        available.append(key)
        st = _finite_stats(a)
        if enc == "terrain-rgb":
            layer_ranges[key] = {
                "encoding": enc, "units": units, "label": label,
                "base_m": -10000.0, "step_m": 0.1,
                "vmin": st.get("min"), "vmax": st.get("max"),
            }
        else:
            _, vmin, vmax = encode_linear(a)
            layer_ranges[key] = {
                "encoding": enc, "units": units, "label": label,
                "vmin": vmin, "vmax": vmax, "step_m": quantisation_step(vmin, vmax),
            }
        layer_ranges[key]["stats"] = st

    texture = run.get("texture")
    if texture is not None:
        layer_ranges["texture"] = {"encoding": "jpeg", "units": None, "label": "Source imagery"}
    layer_ranges["normal"] = {"encoding": "normal-rgb", "units": None,
                              "label": "Surface normals", "note": "(n + 1) / 2, +X east +Y north +Z up"}

    lod_entries = []
    total = n_lods
    for lod in range(n_lods):
        step = 2 ** lod
        sub = {k: (run[k][::step, ::step] if run.get(k) is not None else None)
               for k in available}
        sub_h, sub_w = sub["dsm"].shape
        sub_gsd = gsd * step
        specs = tile_specs((sub_h, sub_w), tile=tile, pad=pad)
        rows, cols = grid_size((sub_h, sub_w), tile)

        tex_lod = None
        if texture is not None:
            from PIL import Image

            tex_lod = np.asarray(
                Image.fromarray(texture).resize((sub_w, sub_h), Image.BILINEAR), np.uint8)

        tiles_meta = []
        for spec in specs:
            layers = {}
            for key in available:
                padded = cut(sub[key], spec)
                if layer_ranges[key]["encoding"] == "terrain-rgb":
                    rgbp = encode_terrain_rgb(padded)
                else:
                    rgbp, _, _ = encode_linear(
                        padded, layer_ranges[key]["vmin"], layer_ranges[key]["vmax"])
                rel = f"tiles/lod{lod}/{key}_{spec.key}.png"
                _save_png(interior(rgbp, spec), os.path.join(out_dir, rel))
                layers[key] = rel

            # Normals are computed on the PADDED tile and then cropped, which is
            # the entire reason the padding exists: a gradient taken on the bare
            # interior would be one-sided at every tile edge and leave a visible
            # ridge along each seam.
            nrm = normal_map(cut(sub["dsm"], spec), sub_gsd)
            rel = f"tiles/lod{lod}/normal_{spec.key}.png"
            _save_png(interior(nrm, spec), os.path.join(out_dir, rel))
            layers["normal"] = rel

            if tex_lod is not None:
                rel = f"tiles/lod{lod}/texture_{spec.key}.jpg"
                _save_jpg(tex_lod[spec.y0:spec.y1, spec.x0:spec.x1],
                          os.path.join(out_dir, rel))
                layers["texture"] = rel

            tiles_meta.append({
                "key": spec.key, "row": spec.row, "col": spec.col,
                "x0": spec.x0, "y0": spec.y0,
                "width": spec.width, "height": spec.height,
                "layers": layers,
            })

        lod_entries.append({
            "lod": lod, "stride": step, "width": sub_w, "height": sub_h,
            "gsd_m": sub_gsd, "tile": tile, "rows": rows, "cols": cols,
            "tiles": tiles_meta,
        })
        if on_progress:
            on_progress(lod + 1, total)

    mesh_info = None
    if write_mesh:
        tex_name = None
        if texture is not None:
            tex_name = "surface.jpg"
            _save_jpg(texture, os.path.join(out_dir, "mesh", tex_name), quality=90)
        mesh_info = write_obj(
            os.path.join(out_dir, "mesh", "surface.obj"), dsm, gsd,
            texture_name=tex_name, stride=obj_stride, name="ayama_surface")
        mesh_info = {k: (os.path.relpath(v, out_dir).replace("\\", "/")
                         if k in ("obj", "mtl") else v)
                     for k, v in mesh_info.items()}
        if tex_name:
            mesh_info["texture"] = f"mesh/{tex_name}"

    manifest = {
        "ayama_tileset_version": TILESET_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run": run["dir"],
        "grid": {"width": W, "height": H, "gsd_m": gsd, "tile": tile, "pad": pad,
                 "extent_m": [W * gsd, H * gsd]},
        "crs": meta.get("crs"),
        "transform": meta.get("transform"),
        "bounds_wgs": meta.get("bounds_wgs"),
        "layers": layer_ranges,
        "default_layer": "texture" if texture is not None else "dsm",
        "lods": lod_entries,
        "mesh": mesh_info,
        "provenance": dict(run.get("provenance") or {},
                           **_sun_from_source(run.get("provenance") or {})),
        "metrics": (run.get("summary") or {}).get("metrics", {}),
        "tier": (run.get("summary") or {}).get("tier"),
        "anchors": (run.get("summary") or {}).get("anchors", {}),
        "notes": derive_notes(run, layer_ranges),
    }
    save_json(manifest, os.path.join(out_dir, "tileset.json"))
    return manifest
