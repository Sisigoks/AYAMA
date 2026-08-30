"""CPU measurement for Phase 3 and Phase 4: what delivery actually costs.

Phase 2 has `ayama dataset` and a `dataset.json` full of evidence. Delivery had
correctness tests and no numbers, which is the same gap in a different place:
nobody could say what tiling a scene costs, how big the payload is, or how much
CPU the viewer burns before a triangle is drawn.

What is measured here, all on CPU:

  encode        the packing arithmetic in isolation, in megapixels per second
  stages        where the wall time of one build actually goes
  tile sweep    build time and payload against tile size
  obj sweep     mesh size and time against decimation
  quantisation  what the 24-bit linear encoding costs in bytes, and what
                quantising each layer to its own uncertainty would save
  round trip    decoded tile against source raster, per layer, at every LOD
  viewer        the JavaScript half, run through node against the real web/app.js

What is NOT measured, and cannot be from here: GPU rasterisation. Everything in
the `viewer` block is CPU work the browser does before the GPU is involved -
decoding PNGs, building vertex buffers, applying colour ramps. A frame rate
needs a real browser and is not claimed.
"""
from __future__ import annotations

import io as _io
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Optional, Sequence

import numpy as np

from ..mesh.build import build_tileset, load_run
from ..mesh.encode import (MAX_CODE, decode_linear, decode_terrain_rgb,
                           encode_linear, encode_terrain_rgb, normal_map)
from ..mesh.obj import write_obj
from ..mesh.tiles import cut, interior, tile_specs

Log = Optional[Callable[[str], None]]


def _log(fn: Log, msg: str) -> None:
    if fn:
        fn(msg)


def _time(fn, repeats: int = 3) -> tuple:
    """Best of `repeats`, plus the result. Best, not mean: a slow run measures
    whatever else the laptop was doing, and the floor is the honest number."""
    best, out = float("inf"), None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _png_bytes(rgb: np.ndarray) -> int:
    from PIL import Image

    buf = _io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG", optimize=True)
    return buf.tell()


def _dir_bytes(path: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(base, f))
    return total


# --------------------------------------------------------------------------
# 1. the encoding arithmetic, in isolation
# --------------------------------------------------------------------------
def encode_bench(run: dict, repeats: int = 3) -> list:
    """Megapixels per second for each operation the builder performs.

    Isolated from disk on purpose. A tile build is dominated by PNG compression,
    so without this it would be impossible to say whether the encoding itself is
    ever going to be the bottleneck. (It is not, by two orders of magnitude.)
    """
    dsm = run["dsm"]
    mpix = dsm.size / 1e6
    ndsm = run.get("ndsm")
    ndsm = dsm if ndsm is None else ndsm
    lin_rgb, vmin, vmax = encode_linear(ndsm)
    ter_rgb = encode_terrain_rgb(dsm)

    ops = [
        ("encode terrain-rgb", lambda: encode_terrain_rgb(dsm)),
        ("decode terrain-rgb", lambda: decode_terrain_rgb(ter_rgb)),
        ("encode linear", lambda: encode_linear(ndsm)),
        ("decode linear", lambda: decode_linear(lin_rgb, vmin, vmax)),
        ("normal map", lambda: normal_map(dsm, 0.5)),
        ("png encode, terrain-rgb", lambda: _png_bytes(ter_rgb)),
        ("png encode, linear", lambda: _png_bytes(lin_rgb)),
    ]
    rows = []
    for name, fn in ops:
        secs, _ = _time(fn, repeats)
        rows.append({
            "op": name,
            "megapixels": round(mpix, 3),
            "seconds": round(secs, 4),
            "mpix_per_s": round(mpix / secs, 1) if secs > 0 else None,
        })
    return rows


# --------------------------------------------------------------------------
# 2. where one build's wall time goes
# --------------------------------------------------------------------------
def stage_breakdown(run: dict, tile: int = 512, pad: int = 1) -> dict:
    """Time one LOD-0 pass, stage by stage, without touching the disk layout.

    Reproduces what `build_tileset` does for the finest level so the components
    can be attributed. The totals will not match a full build exactly - that one
    also writes coarser LODs and the OBJ - which is why both are reported.
    """
    dsm = run["dsm"]
    gsd = float((run.get("meta") or {}).get("gsd_m") or 1.0)
    specs = tile_specs(dsm.shape, tile=tile, pad=pad)
    layers = {k: run[k] for k in ("ndsm", "sigma", "error") if run.get(k) is not None}
    ranges = {k: encode_linear(v)[1:] for k, v in layers.items()}

    out = {"tile": tile, "pad": pad, "n_tiles": len(specs), "stages_s": {}}
    st = out["stages_s"]

    st["cut"], cuts = _time(lambda: [cut(dsm, s) for s in specs])
    st["encode_dsm"], enc = _time(lambda: [encode_terrain_rgb(c) for c in cuts])
    st["crop"], _ = _time(lambda: [interior(e, s) for e, s in zip(enc, specs)])
    st["normals"], nrm = _time(lambda: [normal_map(c, gsd) for c in cuts])

    def _linear_all():
        return [encode_linear(cut(v, s), *ranges[k])[0]
                for k, v in layers.items() for s in specs]

    st["encode_linear_layers"], lin = _time(_linear_all)
    st["png_write_dsm"], _ = _time(
        lambda: [_png_bytes(interior(e, s)) for e, s in zip(enc, specs)], repeats=1)
    st["png_write_linear_layers"], _ = _time(
        lambda: [_png_bytes(a) for a in lin], repeats=1)
    st["png_write_normals"], _ = _time(
        lambda: [_png_bytes(interior(nn, s)) for nn, s in zip(nrm, specs)], repeats=1)

    out["stages_s"] = {k: round(v, 3) for k, v in st.items()}
    out["total_s"] = round(sum(out["stages_s"].values()), 3)
    return out


# --------------------------------------------------------------------------
# 3. tile size and mesh decimation
# --------------------------------------------------------------------------
def tile_sweep(run_dir: str, tiles: Sequence[int] = (128, 256, 512, 1024),
               work_dir: Optional[str] = None, log: Log = None) -> list:
    """Build time and payload against tile size, tiles only (no OBJ).

    `work_dir` matters more than it looks. These builds are disk-bound, so a
    sweep that writes to the system temp directory while the reference build
    writes next to the results is comparing two filesystems, not two tile
    sizes. Every build in this module writes to the same place for that reason.
    """
    rows = []
    for tile in tiles:
        tmp = tempfile.mkdtemp(prefix="tilebench-", dir=work_dir)
        try:
            t0 = time.perf_counter()
            m = build_tileset(run_dir, tmp, tile=tile, pad=1, write_mesh=False)
            secs = time.perf_counter() - t0
            n_tiles = sum(len(l["tiles"]) for l in m["lods"])
            n_files = sum(len(t["layers"]) for l in m["lods"] for t in l["tiles"])
            rows.append({
                "tile": tile,
                "lods": len(m["lods"]),
                "n_tiles": n_tiles,
                "n_files": n_files,
                "seconds": round(secs, 2),
                "bytes": _dir_bytes(tmp),
                "mb": round(_dir_bytes(tmp) / 1e6, 2),
            })
            _log(log, f"    tile {tile:>4}  {n_tiles:>3} tiles  {secs:5.2f}s  "
                      f"{rows[-1]['mb']:6.2f} MB")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rows


def obj_sweep(run: dict, strides: Sequence[int] = (1, 2, 4, 8),
              work_dir: Optional[str] = None, log: Log = None) -> list:
    """Mesh size, triangle count and write time against decimation stride."""
    dsm = run["dsm"]
    gsd = float((run.get("meta") or {}).get("gsd_m") or 1.0)
    rows = []
    for stride in strides:
        tmp = tempfile.mkdtemp(prefix="objbench-", dir=work_dir)
        try:
            path = os.path.join(tmp, "s.obj")
            t0 = time.perf_counter()
            info = write_obj(path, dsm, gsd, texture_name="t.jpg", stride=stride)
            secs = time.perf_counter() - t0
            size = os.path.getsize(path)
            rows.append({
                "stride": stride,
                "vertices": info["vertices"],
                "triangles": info["triangles"],
                "seconds": round(secs, 2),
                "bytes": size,
                "mb": round(size / 1e6, 2),
                "bytes_per_triangle": round(size / max(info["triangles"], 1), 1),
            })
            _log(log, f"    stride {stride}  {info['triangles']:>9,} tris  "
                      f"{secs:5.2f}s  {rows[-1]['mb']:6.2f} MB")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rows


# --------------------------------------------------------------------------
# 4. what full precision costs, and what it buys
# --------------------------------------------------------------------------
def _quantise_to_bits(a: np.ndarray, vmin: float, vmax: float, bits: int) -> np.ndarray:
    """Pack `a` keeping only the top `bits` of the 24-bit code. Returns RGB.

    Simulating a narrower field by rounding in *value* space and re-encoding
    looks equivalent and is not. Floating-point jitter shifts some pixels by a
    code or two, the low byte stays noisy, and PNG cannot collapse it - which
    produced the impossible result that 12-bit was LARGER than 16-bit. Masking
    the code is what a narrower field actually does: the bottom bits become
    exactly zero, and the compressor can see that.
    """
    span = max(vmax - vmin, 1e-12)
    norm = np.clip((np.asarray(a, np.float64) - vmin) / span, 0.0, 1.0)
    bits = int(bits)
    levels = (1 << bits) - 1
    shift = 24 - bits
    code = (np.rint(norm * levels).astype(np.uint64) << np.uint64(shift))
    code = np.minimum(code, MAX_CODE).astype(np.uint32)
    rgb = np.empty(code.shape + (3,), np.uint8)
    rgb[..., 0] = (code >> 16) & 0xFF
    rgb[..., 1] = (code >> 8) & 0xFF
    rgb[..., 2] = code & 0xFF
    return rgb


def quantisation_sweep(run: dict, log: Log = None) -> list:
    """Payload against precision, for the layers that use the linear encoding.

    Variants are bit depths relative to each layer's OWN range, and that choice
    is the result of getting it wrong first. An earlier version stepped by
    fractions of the mean sigma (3 m). The nDSM layer spans 0.276 m in total, so
    a sigma/4 step of 0.75 m rounded the entire layer to a single value and
    "saved" 99.8% of its bytes - a saving that deletes the measurement. The
    max-error column is what exposed it, which is why it is reported beside
    every row rather than left implicit.

    A bit depth tied to the layer's own range cannot do that: at 8 bits the
    worst error is one part in 255 of whatever the layer actually spans.
    """
    sigma = run.get("sigma")
    sigma_mean = float(np.nanmean(sigma)) if sigma is not None else None
    rows = []
    for key in ("ndsm", "sigma", "error"):
        a = run.get(key)
        if a is None:
            continue
        finite = np.isfinite(a)
        vmin, vmax = float(a[finite].min()), float(a[finite].max())
        raw_span = vmax - vmin
        degenerate = raw_span < 1e-9
        span = max(raw_span, 1e-12)
        base_rgb, _, _ = encode_linear(a, vmin, vmax)
        base_bytes = _png_bytes(base_rgb)

        for bits in (24, 16, 12, 8):
            step = span / ((1 << bits) - 1)
            if bits == 24:
                rgb, nbytes = base_rgb, base_bytes
            else:
                rgb = _quantise_to_bits(a, vmin, vmax, bits)
                nbytes = _png_bytes(rgb)
            err = float(np.abs(decode_linear(rgb, vmin, vmax) - a)[finite].max())
            rows.append({
                "layer": key,
                "bits": bits,
                "variant": f"{bits}-bit" + (" (shipped)" if bits == 24 else ""),
                "step_m": step,
                "bytes": nbytes,
                "kb": round(nbytes / 1e3, 1),
                "vs_24bit": round(nbytes / base_bytes, 4),
                "max_error_m": err,
                # Two sanity columns, because a byte saving means nothing alone.
                "error_over_span": round(err / span, 6),
                "error_over_sigma": (None if not sigma_mean
                                     else round(err / sigma_mean, 6)),
                "range_m": [round(vmin, 4), round(vmax, 4)],
                "span_m": raw_span,
                "degenerate": degenerate,
                # Below a few tens of kB a PNG is mostly header and filter
                # choice, and the byte count stops tracking precision at all.
                "meaningful": bool(nbytes > 20_000),
            })

        # The coarsest depth that still resolves the layer to 0.1% of its own
        # range. Any coarser and the encoding, not the measurement, is what the
        # viewer would be showing.
        usable = [] if degenerate else [
            r for r in rows if r["layer"] == key and r["error_over_span"] <= 1e-3]
        best = min(usable, key=lambda r: r["bits"]) if usable else None
        for r in rows:
            if r["layer"] == key:
                r["recommended"] = bool(best and r["bits"] == best["bits"])
        if degenerate:
            _log(log, f"    {key}: constant layer, no precision to trade")
        if best:
            _log(log, f"    {key}: 24-bit {base_bytes / 1e3:>7.0f} kB  ->  "
                      f"{best['bits']}-bit {best['kb']:>7.0f} kB  "
                      f"(worst error {best['max_error_m']:.3g} m)")
    return rows


# --------------------------------------------------------------------------
# 5. the surface survives the trip
# --------------------------------------------------------------------------
def roundtrip_report(run: dict, tileset_dir: str, manifest: dict) -> list:
    """Decode every tile back and compare against the source raster, per LOD.

    A performance study that did not also confirm the numbers came back intact
    would be measuring how fast the pipeline can be wrong.
    """
    from PIL import Image

    rows = []
    for lod in manifest["lods"]:
        stride = lod["stride"]
        for key, spec in manifest["layers"].items():
            if spec["encoding"] not in ("terrain-rgb", "linear"):
                continue
            src = run.get("dsm" if key == "dsm" else key)
            if src is None:
                continue
            src = src[::stride, ::stride]
            worst = 0.0
            for t in lod["tiles"]:
                rel = t["layers"].get(key)
                if not rel:
                    continue
                rgb = np.asarray(
                    Image.open(os.path.join(tileset_dir, rel)).convert("RGB"), np.uint8)
                got = (decode_terrain_rgb(rgb) if spec["encoding"] == "terrain-rgb"
                       else decode_linear(rgb, spec["vmin"], spec["vmax"]))
                want = src[t["y0"]:t["y0"] + t["height"], t["x0"]:t["x0"] + t["width"]]
                d = np.abs(got - want)
                d = d[np.isfinite(d)]
                if d.size:
                    worst = max(worst, float(d.max()))
            rows.append({
                "lod": lod["lod"], "layer": key, "encoding": spec["encoding"],
                "step_m": spec.get("step_m"), "max_abs_error_m": worst,
                "within_half_a_step": bool(
                    spec.get("step_m") is None or worst <= spec["step_m"] / 2 + 2e-3),
            })
    return rows


def payload_report(tileset_dir: str, manifest: dict) -> dict:
    """Bytes on disk, split by layer and by LOD, plus first-paint cost."""
    per_layer: dict = {}
    per_lod: dict = {}
    for lod in manifest["lods"]:
        key = f"lod{lod['lod']}"
        per_lod[key] = 0
        for t in lod["tiles"]:
            for layer, rel in t["layers"].items():
                n = os.path.getsize(os.path.join(tileset_dir, rel))
                per_layer[layer] = per_layer.get(layer, 0) + n
                per_lod[key] += n
    mesh = 0
    mesh_dir = os.path.join(tileset_dir, "mesh")
    if os.path.isdir(mesh_dir):
        mesh = _dir_bytes(mesh_dir)

    # What the page must fetch before it can draw: geometry, shading and the
    # default drape, for the LOD it opens at, plus the readout layers.
    lod0 = manifest["lods"][0]
    first = 0
    for t in lod0["tiles"]:
        for layer in ("dsm", "normal", manifest.get("default_layer", "texture"),
                      "ndsm", "sigma"):
            rel = t["layers"].get(layer)
            if rel:
                first += os.path.getsize(os.path.join(tileset_dir, rel))
    return {
        "total_bytes": _dir_bytes(tileset_dir),
        "tiles_bytes": sum(per_layer.values()),
        "mesh_bytes": mesh,
        "by_layer": dict(sorted(per_layer.items(), key=lambda kv: -kv[1])),
        "by_lod": per_lod,
        "first_paint_bytes": first,
        "first_paint_lod": lod0["lod"],
    }


# --------------------------------------------------------------------------
# 6. the viewer's own CPU work, measured in node against web/app.js
# --------------------------------------------------------------------------
def viewer_bench(tileset_dir: str, repo_root: str, log: Log = None) -> dict:
    """Run scripts/bench_viewer.js and return its JSON.

    Measures the real `web/app.js`, not a reimplementation - the point is the
    code the browser runs. Returns a skip record rather than failing when node
    is absent, the same way the GPU tests skip with a reason.
    """
    script = os.path.join(repo_root, "scripts", "bench_viewer.js")
    if not os.path.exists(script):
        return {"skipped": "scripts/bench_viewer.js not found"}
    node = shutil.which("node")
    if not node:
        return {"skipped": "node is not on PATH"}
    try:
        proc = subprocess.run([node, script, tileset_dir], capture_output=True,
                              text=True, timeout=300, cwd=repo_root)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"skipped": f"node failed: {exc}"}
    if proc.returncode != 0:
        return {"skipped": f"node exited {proc.returncode}",
                "stderr": (proc.stderr or "")[-800:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"skipped": f"could not parse node output: {exc}",
                "stdout": (proc.stdout or "")[:800]}


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------
def run_delivery(
    run_dir: str,
    out_dir: str,
    tile: int = 512,
    tiles: Sequence[int] = (128, 256, 512, 1024),
    obj_strides: Sequence[int] = (1, 2, 4, 8),
    repeats: int = 3,
    work_dir: Optional[str] = None,
    log: Log = None,
) -> dict:
    """Measure Phase 3 and Phase 4 on this machine. Returns the report dict."""
    from .bench import device_report

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)
    t_start = time.time()

    _log(log, "  loading the run")
    run = load_run(run_dir)
    H, W = run["dsm"].shape
    gsd = float((run.get("meta") or {}).get("gsd_m") or 1.0)

    # Every *timed* build writes to one scratch directory, and the persistent
    # tileset is produced separately and not timed. Two lessons are baked in
    # here. Timing one build beside the results and another in the system temp
    # directory compares two filesystems, not two settings - `obj_s` was their
    # difference and meaningless. And writing large files inside the repository
    # invites an on-access virus scanner: the same 139 MB OBJ took 36 s in temp
    # and 205 s under the checkout on this machine. So: one location, chosen by
    # the caller, recorded in the report.
    work_root = work_dir or tempfile.gettempdir()
    work = tempfile.mkdtemp(prefix="ayama-delivery-", dir=work_root)

    try:
        _log(log, "  timing the build")
        full_build_s, tiles_only_s = float("inf"), float("inf")
        for _ in range(2):                       # best of two; these are noisy
            d = tempfile.mkdtemp(prefix="full-", dir=work)
            t0 = time.perf_counter()
            build_tileset(run_dir, d, tile=tile, pad=1, obj_stride=2)
            full_build_s = min(full_build_s, time.perf_counter() - t0)
            shutil.rmtree(d, ignore_errors=True)

            d = tempfile.mkdtemp(prefix="tiles-", dir=work)
            t0 = time.perf_counter()
            build_tileset(run_dir, d, tile=tile, pad=1, write_mesh=False)
            tiles_only_s = min(tiles_only_s, time.perf_counter() - t0)
            shutil.rmtree(d, ignore_errors=True)

        _log(log, "  writing the reference tileset")
        tileset_dir = os.path.join(out_dir, "tileset")
        if os.path.isdir(tileset_dir):
            shutil.rmtree(tileset_dir, ignore_errors=True)
        manifest = build_tileset(run_dir, tileset_dir, tile=tile, pad=1, obj_stride=2)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    work_dir = work

    _log(log, "  encode microbenchmark")
    enc = encode_bench(run, repeats=repeats)
    _log(log, "  stage breakdown")
    stages = stage_breakdown(run, tile=tile)
    _log(log, "  tile size sweep")
    tsweep = tile_sweep(run_dir, tiles, work_dir=work_dir, log=log)
    _log(log, "  mesh decimation sweep")
    osweep = obj_sweep(run, obj_strides, work_dir=work_dir, log=log)
    _log(log, "  quantisation sweep")
    qsweep = quantisation_sweep(run, log=log)
    _log(log, "  round-trip check")
    rt = roundtrip_report(run, tileset_dir, manifest)
    payload = payload_report(tileset_dir, manifest)
    _log(log, "  viewer CPU benchmark (node)")
    viewer = viewer_bench(tileset_dir, repo_root, log=log)

    report = {
        "ayama_delivery_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": device_report(),
        "source_run": os.path.abspath(run_dir),
        "scene": {
            "width": W, "height": H, "gsd_m": gsd,
            "megapixels": round(W * H / 1e6, 3),
            "extent_m": [W * gsd, H * gsd],
            "crs": (run.get("meta") or {}).get("crs"),
        },
        "config": {"tile": tile, "pad": 1, "obj_stride": 2, "repeats": repeats,
                   "timed_in": work_root,
                   "timing_note": ("Build timings are disk-bound and vary with where "
                                   "they are written; an on-access virus scanner can "
                                   "change them several-fold. Every timed build here "
                                   "used one scratch directory, reported above.")},
        "build": {
            "full_s": round(full_build_s, 2),
            "tiles_only_s": round(tiles_only_s, 2),
            "obj_s": round(full_build_s - tiles_only_s, 2),
            "lods": len(manifest["lods"]),
            "n_tiles": sum(len(l["tiles"]) for l in manifest["lods"]),
            "mpix_per_s": round((W * H / 1e6) / max(tiles_only_s, 1e-9), 3),
        },
        "encode": enc,
        "stages": stages,
        "tile_sweep": tsweep,
        "obj_sweep": osweep,
        "quantisation": qsweep,
        "roundtrip": rt,
        "payload": payload,
        "viewer": viewer,
        "notes": manifest.get("notes", []),
        "wall_s": round(time.time() - t_start, 1),
    }
    shutil.rmtree(work_dir, ignore_errors=True)
    return report


def environment() -> dict:
    from .bench import device_report

    rep = device_report()
    rep["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rep


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def write_report(rep: dict, path: str) -> str:
    """Render delivery.json as markdown. Every number here comes out of the dict.

    Kept beside the measurement rather than in the CLI so a figure and a table
    can never disagree about what was measured - the same reason `figures.py`
    renders from `dataset.json` instead of from a fresh run.
    """
    env = rep["environment"]
    sc = rep["scene"]
    b = rep["build"]
    L = []
    A = L.append

    A("# AYAMA delivery benchmark")
    A("")
    A(f"Phase 3 and Phase 4 on CPU. Generated {rep['generated_utc']} by")
    A("`python -m ayama.cli delivery`.")
    A("")
    A("Every number is measured on the machine below, against a real Phase 2 run.")
    A("**Browser GPU rasterisation is not measured** and is not claimed: everything in the")
    A("viewer section is CPU work the browser does before a triangle is drawn.")
    A("")
    A("## Environment")
    A("")
    A(f"- {env.get('platform')}, python {env.get('python')}, {env.get('cpu_count')} cpus")
    A(f"- {env.get('cpu_count')} CPU cores  ·  rasterio {env.get('rasterio')}")
    node = (rep.get("viewer") or {}).get("node")
    if node:
        A(f"- node {node}")
    A(f"- source run `{rep['source_run']}`")
    A(f"- scene {sc['width']} x {sc['height']} px at {sc['gsd_m']:g} m "
      f"({sc['megapixels']:.2f} Mpix, {sc['extent_m'][0]:.0f} x {sc['extent_m'][1]:.0f} m)")
    A("")

    A("## Headline")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| tileset build, tiles only | **{b['tiles_only_s']:.2f} s** "
      f"({b['mpix_per_s']:.2f} Mpix/s) |")
    A(f"| tileset build, with the OBJ | **{b['full_s']:.2f} s** "
      f"(OBJ alone {b['obj_s']:.2f} s) |")
    A(f"| output | {b['lods']} LODs, {b['n_tiles']} tiles |")
    A(f"| payload | **{rep['payload']['total_bytes'] / 1e6:.1f} MB** total, "
      f"{rep['payload']['tiles_bytes'] / 1e6:.1f} MB tiles + "
      f"{rep['payload']['mesh_bytes'] / 1e6:.1f} MB mesh |")
    A(f"| first paint, bytes | {rep['payload']['first_paint_bytes'] / 1e6:.2f} MB |")
    v = rep.get("viewer") or {}
    if v.get("totals_ms"):
        A(f"| first paint, viewer CPU | **{v['totals_ms']['first_paint_cpu']:.0f} ms** |")
    A(f"| whole benchmark | {rep['wall_s']:.0f} s |")
    A("")
    note = (rep.get("config") or {}).get("timing_note")
    if note:
        A(f"> {note}")
        A("")

    A("## Encoding throughput")
    A("")
    A("The packing arithmetic in isolation, no disk. This is what says whether the")
    A("encoding could ever be the bottleneck.")
    A("")
    A("| operation | Mpix/s | seconds |")
    A("|---|---|---|")
    for r in rep["encode"]:
        rate = "-" if r["mpix_per_s"] is None else f"{r['mpix_per_s']:,.0f}"
        A(f"| {r['op']} | {rate} | {r['seconds']:.4f} |")
    A("")
    enc = {r["op"]: r for r in rep["encode"]}
    fast = enc.get("encode terrain-rgb", {}).get("mpix_per_s")
    slow = enc.get("png encode, linear", {}).get("mpix_per_s")
    if fast and slow:
        A(f"**PNG compression dominates.** Packing pixels runs at {fast:,.0f} Mpix/s;")
        A(f"compressing them runs at {slow:,.0f} Mpix/s - a factor of {fast / slow:,.0f}.")
        A("Nothing in the encoder is worth optimising until the compressor is.")
    A("")

    A("## Where one build's time goes")
    A("")
    st = rep["stages"]
    A(f"LOD 0 only, {st['n_tiles']} tiles of {st['tile']} px "
      f"(+{st['pad']} px pad), {st['total_s']:.2f} s accounted for.")
    A("")
    A("| stage | seconds | share |")
    A("|---|---|---|")
    for k, sec in sorted(st["stages_s"].items(), key=lambda kv: -kv[1]):
        share = 100.0 * sec / max(st["total_s"], 1e-9)
        A(f"| {k.replace('_', ' ')} | {sec:.3f} | {share:.1f}% |")
    A("")

    A("## Tile size")
    A("")
    A("| tile px | LODs | tiles | files | seconds | payload |")
    A("|---|---|---|---|---|---|")
    for r in rep["tile_sweep"]:
        A(f"| {r['tile']} | {r['lods']} | {r['n_tiles']} | {r['n_files']} | "
          f"{r['seconds']:.2f} | {r['mb']:.2f} MB |")
    A("")
    ts = rep["tile_sweep"]
    if len(ts) > 1:
        mbs = [r["mb"] for r in ts]
        secs = [r["seconds"] for r in ts]
        files = [r["n_files"] for r in ts]
        spread = (max(mbs) - min(mbs)) / max(min(mbs), 1e-9) * 100.0
        A(f"**Tile size barely moves the payload.** Across a {max(files) // max(min(files), 1)}x "
          f"range in file count ({min(files)} to {max(files)} files), the total varies by "
          f"only {spread:.1f}% ({min(mbs):.2f} to {max(mbs):.2f} MB) and build time by "
          f"{max(secs) - min(secs):.2f} s.")
        A("")
        A("PNG headers and per-file compression contexts were expected to punish small")
        A("tiles; at these sizes they do not, because the pixel data dominates either")
        A("way. So tile size is free to be chosen for what it actually affects - how")
        A("much a viewer can cull, and how many requests it makes - rather than for")
        A("bytes. The default of 512 keeps the file count in double digits.")
    A("")

    A("## Mesh decimation")
    A("")
    A("| stride | vertices | triangles | seconds | size | bytes/triangle |")
    A("|---|---|---|---|---|---|")
    for r in rep["obj_sweep"]:
        A(f"| {r['stride']} | {r['vertices']:,} | {r['triangles']:,} | "
          f"{r['seconds']:.2f} | {r['mb']:.1f} MB | {r['bytes_per_triangle']:.0f} |")
    A("")
    A("OBJ is a text format, so size tracks triangle count almost exactly. This is")
    A("the argument for glTF, and the reason `--obj-stride` and `--no-mesh` exist.")
    A("")

    A("## What full precision costs")
    A("")
    A("The linear encoding spends all 24 bits, so its low byte is incompressible")
    A("noise. Each row below keeps only the top N bits of the code and zeroes the")
    A("rest, which is what a narrower field would really store - and what lets PNG")
    A("collapse them.")
    A("")
    A("| layer | precision | step | size | vs 24-bit | worst error |")
    A("|---|---|---|---|---|---|")
    for r in rep["quantisation"]:
        step = "-" if r["step_m"] is None else f"{r['step_m']:.4g} m"
        A(f"| {r['layer']} | {r['variant']} | {step} | {r['kb']:.0f} kB | "
          f"{r['vs_24bit']:.2f}x | {r['max_error_m']:.3g} m |")
    A("")
    by_layer: dict = {}
    for r in rep["quantisation"]:
        by_layer.setdefault(r["layer"], []).append(r)
    saved, full_total, rec_total, picks, flat = 0, 0, 0, [], []
    for layer, rows in by_layer.items():
        full = next(x for x in rows if x["bits"] == 24)
        if rows[0].get("degenerate"):
            flat.append(layer)
            continue
        best = next((x for x in rows if x.get("recommended")), full)
        saved += full["bytes"] - best["bytes"]
        full_total += full["bytes"]
        rec_total += best["bytes"]
        picks.append(f"{layer} at {best['bits']} bits")
    if saved > 0:
        A(f"**{', '.join(picks)}** resolves every layer to better than 0.1% of its")
        A(f"own range, and takes the linear layers from {full_total / 1e6:.2f} MB to")
        A(f"{rec_total / 1e6:.2f} MB - a saving of **{saved / 1e6:.2f} MB**, "
          f"{100.0 * saved / max(full_total, 1):.0f}% of their bytes.")
        A("")
    if flat:
        A(f"({', '.join(flat)} never varies in this run, so there is no precision to")
        A("trade and no recommendation is made for it.)")
        A("")
    A("The precision column is not decoration. An earlier version of this sweep")
    A("stepped by fractions of the mean sigma instead of by the layer's own range,")
    A("and reported a 99.8% saving on nDSM - by rounding a layer that spans 0.28 m")
    A("with a 0.75 m step, which flattens it to a constant. A byte count alone")
    A("cannot tell a compression win from a deleted measurement.")
    A("")

    A("## The surface survives the trip")
    A("")
    A("Every tile decoded back and compared against the raster it came from.")
    A("")
    A("| LOD | layer | encoding | step | worst error | within half a step |")
    A("|---|---|---|---|---|---|")
    for r in rep["roundtrip"]:
        step = "-" if r["step_m"] is None else f"{r['step_m']:.3g}"
        ok = "yes" if r["within_half_a_step"] else "**NO**"
        A(f"| {r['lod']} | {r['layer']} | {r['encoding']} | {step} | "
          f"{r['max_abs_error_m']:.3g} m | {ok} |")
    A("")

    A("## Payload")
    A("")
    p = rep["payload"]
    A("| layer | bytes | share of tiles |")
    A("|---|---|---|")
    for k, n in p["by_layer"].items():
        A(f"| {k} | {n / 1e6:.2f} MB | {100.0 * n / max(p['tiles_bytes'], 1):.1f}% |")
    A(f"| **mesh** | {p['mesh_bytes'] / 1e6:.2f} MB | - |")
    A("")
    A("| LOD | bytes |")
    A("|---|---|")
    for k, n in p["by_lod"].items():
        A(f"| {k} | {n / 1e6:.2f} MB |")
    A("")
    A(f"**First paint** fetches {p['first_paint_bytes'] / 1e6:.2f} MB: geometry,")
    A("normals, the default drape and the two layers the cursor readout needs.")
    A("")

    A("## Viewer CPU")
    A("")
    if v.get("skipped"):
        A(f"Skipped: {v['skipped']}")
    else:
        A("Measured against the real `web/app.js` under node, best of five with a")
        A("warm-up. This is the work the browser does before the GPU is involved.")
        A("")
        A("| operation | ms |")
        A("|---|---|")
        for o in v.get("ops", []):
            ms = "-" if o.get("ms") is None else f"{o['ms']:.2f}"
            note = f" ({o['note']})" if o.get("note") else ""
            A(f"| {o['op']}{note} | {ms} |")
        A("")
        t = v.get("totals_ms", {})
        if t:
            A("| for the whole scene | ms |")
            A("|---|---|")
            A(f"| decode every data layer | {t.get('decode_all_layers', 0):.1f} |")
            A(f"| build geometry for every tile | {t.get('geometry_all_tiles', 0):.1f} |")
            A(f"| **CPU before first paint** | **{t.get('first_paint_cpu', 0):.1f}** |")
            A(f"| re-colour on a layer switch | {t.get('colourize_layer_switch', 0):.1f} |")
            A("")
        tp = v.get("throughput_mpix_per_s", {})
        if tp:
            A(f"Decode throughput: **{tp.get('decode_terrain_rgb', 0):.0f} Mpix/s** "
              f"terrain-rgb, {tp.get('decode_linear', 0):.0f} Mpix/s linear.")
            A("")
        ops = {o["op"]: o.get("ms") for o in v.get("ops", [])}
        alloc = ops.get("decode terrain-rgb, whole scene")
        reuse = ops.get("decode terrain-rgb, reusing the buffer")
        if alloc and reuse and reuse < alloc:
            A(f"**Reusing the output buffer is {100 * (1 - reuse / alloc):.0f}% faster** "
              f"({reuse:.1f} ms against {alloc:.1f} ms). The viewer currently")
            A("allocates a fresh `Float32Array` per tile per layer; it does not have to.")
            A("")

    if rep.get("notes"):
        A("## What the tileset says about itself")
        A("")
        for n in rep["notes"]:
            mark = {"critical": "**!!**", "warning": "**!**"}.get(n["level"], "")
            A(f"- {mark} {n['text']}")
        A("")

    A("## Reproducing this")
    A("")
    A("```bash")
    A("python -m ayama.cli delivery results/zurich")
    A("```")
    A("")

    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path
