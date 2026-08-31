"""Turn a Phase 2 run directory into a tileset a browser can render.

Input is exactly what `traksha.cli run` already writes - no new inference, no new
calibration, nothing re-derived. Phase 3 is a *delivery* phase: if a number
appears in the viewer it came out of the rasters Phase 2 produced, and the
manifest records which run produced them.

The manifest is the contract between this module and `web/src/renderer.js`. It is
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
from .encode import (encode_linear, encode_linear_bits, encode_terrain_rgb,
                     linear_range_for_bits, linear_step, normal_map)
from .obj import write_obj, write_obj_adaptive, write_obj_structural
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
            "`traksha run`, not at a study root"
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
            "tags": {k: v for k, v in ds.tags().items() if k.startswith("TRAKSHA_")},
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
        "p1": float(np.percentile(v, 1)), "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
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
    # Deliberately not `run["sem"]`: the relief check used to read it and that
    # is exactly what made it wrong, twice. See the comment below.
    if ndsm is not None:
        st = _finite_stats(ndsm)
        p99 = st.get("p99", 0.0)
        top = st.get("max", 0.0)
        # Two rewrites taught this check what it is actually looking for.
        #
        # First it keyed on an absolute p99 threshold, and missed a real
        # collapse: a surface can carry a couple of metres of noise-relief while
        # recovering almost none of the true structure. So it was changed to
        # compare buildings against their surroundings - which promptly produced
        # a FALSE alarm on a surface with 53 m of genuine relief, because the
        # class it was comparing came from the colour heuristic, and README
        # section 3.4 shows that heuristic does not find buildings on real
        # imagery.
        #
        # The lesson is that this check must not depend on segmentation at all.
        # It cannot use ground truth either - it runs at delivery time. What is
        # left, and what is sufficient, is the height distribution of the
        # surface being shipped: on any populated scene at this resolution,
        # something has to stand up.
        p95 = st.get("p95", p99)
        if p99 < 2.0:
            notes.append({
                "level": "critical",
                "id": "flat_surface",
                "text": (
                    f"Height above ground reaches only {top:.2f} m, and 99% of the "
                    f"surface is under {p99:.2f} m. Nothing stands up: this is "
                    "terrain with the structures flattened rather than a "
                    "reconstruction. The usual cause is a calibration that fitted "
                    "one scale to terrain and had no structural scale to apply - "
                    "see README section 4, and `traksha fit` for the fix. Raising "
                    "the vertical exaggeration will show what little relief there "
                    "is; it is a defect, not a rendering choice."
                ),
            })
        elif p99 < 8.0:
            # Deliberately a warning and not a verdict. A genuinely low-rise
            # scene - farmland, hedgerows - looks exactly like this, and the
            # tiler cannot tell the two apart without truth it does not have. So
            # it states what it measured and names the check the reader can make.
            notes.append({
                "level": "warning", "id": "low_relief",
                "text": (f"Height above ground reaches {top:.2f} m but 99% of the "
                         f"surface is under {p99:.2f} m. If this is a built-up "
                         "scene that is too little relief, and the usual cause is "
                         "a calibration with no structural scale to apply - see "
                         "README section 4 and `traksha fit`. On low-rise ground it "
                         "may be correct."),
            })
        elif p95 > 0 and top / max(p95, 1e-6) > 8.0:
            # The opposite failure: a few wild spikes over an otherwise sane
            # surface, which is what an over-large structural scale looks like.
            notes.append({
                "level": "warning", "id": "spiky_relief",
                "text": (f"Height above ground peaks at {top:.2f} m against a 95th "
                         f"percentile of {p95:.2f} m. Isolated spikes that tall are "
                         "usually the structural scale overshooting on a few "
                         "high-frequency artifacts rather than real structure."),
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

    `traksha run` writes `summary.json` beside the rasters. `traksha dataset` keeps
    every scene's metrics together in one `dataset.json` a level up, because the
    aggregate across scenes is the thing worth reading. Rather than make the
    viewer show nothing for the runs the study produced - which are the ones the
    README actually reports - the dataset file is consulted and the matching
    scene picked out by directory name. Nothing is recomputed either way.
    """
    direct = _read_json(os.path.join(run_dir, "summary.json"))
    if direct.get("metrics"):
        return direct

    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)
    dataset = _read_json(os.path.join(os.path.dirname(run_dir), "dataset.json"))
    for scene in dataset.get("scenes") or []:
        if scene.get("name") == name:
            out = dict(scene)
            out["source"] = "dataset.json"
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


def _structural_mesh(run: dict, mdir: str, out_dir: str, dsm, gsd: float,
                     tex_name: Optional[str]) -> Optional[dict]:
    """Rebuild the mesh with buildings as separate solids, if we can.

    Needs the instance segmentation and an nDSM: the instances say where the
    footprints are, and only the height above ground can say which of them are
    buildings. A run made before the segmentation stage existed simply does not
    get one, which is why this returns None rather than raising.
    """
    ndsm = run.get("ndsm")
    seg_dir = os.path.join(run.get("dir", ""), "segmentation")
    if ndsm is None or not os.path.isdir(seg_dir):
        return None

    from ..semantics import instances as inst_mod
    from . import structural as struct
    from .quality import report as mesh_report

    field = inst_mod.load(seg_dir)
    if field is None or field.count == 0:
        return None

    sem = run.get("sem")
    buildings = struct.select(field, dsm, ndsm,
                              None if sem is None else sem.astype(np.uint8))
    if not buildings:
        return None
    mesh = struct.build(dsm, ndsm, buildings, gsd)
    info = write_obj_structural(
        os.path.join(mdir, "structural.obj"), mesh, dsm.shape, gsd,
        texture_name=tex_name, name="traksha_structural")
    info = {k: (os.path.relpath(v, out_dir).replace("\\", "/")
                if k in ("obj", "mtl") else v) for k, v in info.items()}
    info["quality"] = mesh_report(mesh)
    info["buildings_detail"] = mesh.get("buildings", [])

    # And a browser-sized copy beside the tileset. Without it every structural
    # improvement is invisible on the site: the viewer draws height tiles, and
    # a height field cannot represent a wall.
    from . import webmesh

    web = webmesh.build_web_mesh(dsm, ndsm, field, run.get("sem"), gsd)
    info["web"] = webmesh.write(os.path.join(out_dir, "structural.bin"), web,
                                web["grid"], web["gsd_m"])
    return info


def build_tileset(
    run_dir: str,
    out_dir: str,
    tile: int = 512,
    pad: int = 1,
    lods: Optional[int] = None,
    obj_stride: int = 2,
    obj_tol_m: float = 2.0,
    obj_max_tris: int = 0,
    write_mesh: bool = True,
    write_structural: bool = True,
    quantise_bits: int = 24,
    mesh_dir: Optional[str] = None,
    on_progress=None,
) -> dict:
    """Build `out_dir` from a Phase 2 run. Returns the manifest dict.

    `quantise_bits` trades payload for precision on the linear layers only, and
    the viewer needs no change to read it - the decode is identical. 24 keeps
    every bit; 12 costs about a tenth of a percent of each layer's range and
    saves three quarters of its bytes, which is what the published demo uses.
    """
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
            step = linear_step(vmin, vmax, quantise_bits)
            # The encoder widens the recorded range to absorb the bit-shift, so
            # the viewer's decode stays exact and stays ignorant of quantisation.
            enc_min, enc_max = linear_range_for_bits(vmin, vmax, quantise_bits)
            layer_ranges[key] = {
                "encoding": enc, "units": units, "label": label,
                "vmin": enc_min, "vmax": enc_max,
                "bits": int(quantise_bits), "step_m": step,
                "data_range_m": [vmin, vmax],
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
                    lo, hi = layer_ranges[key]["data_range_m"]
                    rgbp = encode_linear_bits(padded, lo, hi, quantise_bits)
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
        # `mesh_dir` puts the OBJ beside the tileset rather than inside it,
        # which is what `traksha build` wants: a scene is one folder holding
        # rasters, a tiles3d/ the browser reads, and a mesh/ that opens in
        # Blender or MeshLab with no TRAKSHA installed. The manifest records the
        # path relative to itself either way, so the viewer is unaffected.
        mdir = mesh_dir or os.path.join(out_dir, "mesh")
        tex_name = None
        if texture is not None:
            tex_name = "surface.jpg"
            _save_jpg(texture, os.path.join(mdir, tex_name), quality=90)
        # Adaptive by default: the detail in this surface is concentrated on
        # the cells where a sampling step crosses a wall, and a uniform stride
        # cannot spend its triangles there. `obj_tol_m <= 0` falls back to the
        # uniform grid, which is what the small web demo wants.
        if obj_tol_m and obj_tol_m > 0:
            mesh_info = write_obj_adaptive(
                os.path.join(mdir, "surface.obj"), dsm, gsd,
                texture_name=tex_name, tol_m=obj_tol_m,
                max_triangles=obj_max_tris or None, name="traksha_surface")
        else:
            mesh_info = write_obj(
                os.path.join(mdir, "surface.obj"), dsm, gsd,
                texture_name=tex_name, stride=obj_stride, name="traksha_surface")
        mesh_info = {k: (os.path.relpath(v, out_dir).replace("\\", "/")
                         if k in ("obj", "mtl") else v)
                     for k, v in mesh_info.items()}
        if tex_name:
            mesh_info["texture"] = os.path.relpath(
                os.path.join(mdir, tex_name), out_dir).replace("\\", "/")

        # The structural rebuild, when the run carries a segmentation. It is a
        # second artifact rather than a replacement: `surface.obj` is the height
        # field this project has always delivered, and keeping both is what
        # makes the comparison in README section 6 reproducible rather than a
        # claim. Both are render-space; neither touches the calibrated rasters.
        if write_structural:
            structural = _structural_mesh(run, mdir, out_dir, dsm, gsd, tex_name)
            if structural:
                mesh_info["structural"] = structural

    manifest = {
        "traksha_tileset_version": TILESET_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run": run["dir"],
        "grid": {"width": W, "height": H, "gsd_m": gsd, "tile": tile, "pad": pad,
                 "extent_m": [W * gsd, H * gsd], "quantise_bits": int(quantise_bits)},
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
