"""TRAKSHA command line.

    python -m traksha.cli info   data/sample.tif
    python -m traksha.cli depth  data/sample.tif --out out/depth.tif
    python -m traksha.cli backbones

Phase 2 adds `run` (full pipeline with calibration and metrics).
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from typing import Optional

import numpy as np


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _fmt_meta(meta) -> str:
    lines = [
        f"  source          {meta.source}",
        f"  CRS             {meta.crs or '-'}",
        f"  GSD             {meta.gsd_m:.4g} m" + ("  (assumed)" if meta.gsd_is_assumed else ""),
        "  sun             " + (
            f"{meta.sun_azimuth_deg:.1f} deg az / {meta.sun_elevation_deg:.1f} deg el"
            if meta.has_sun else "unknown"
        ),
        f"  off-nadir       {meta.off_nadir_deg:.1f} deg",
        f"  acquired        {meta.acquired_utc or '-'}",
    ]
    if meta.bounds_wgs:
        w, s, e, n = meta.bounds_wgs
        lines.append(f"  bounds (WGS84)  {w:.5f}, {s:.5f}, {e:.5f}, {n:.5f}")
    return "\n".join(lines)


def cmd_info(args) -> int:
    from .core.ingest import ingest

    scene = ingest(args.image)
    h, w = scene.shape
    print(f"{os.path.basename(args.image)}   {w} x {h}   raw dtype {scene.raw_dtype}")
    print(_fmt_meta(scene.meta))
    if scene.meta.has_sun:
        el = scene.meta.sun_elevation_deg
        verdict = "usable for shadow anchors" if 20 <= el <= 75 else "outside the usable band (20-75 deg)"
        print(f"  shadow physics  {verdict}")
    return 0


def cmd_backbones(args) -> int:
    from .depth.backbones import BACKBONES, LABELS

    for name in BACKBONES:
        print(f"  {name:<12} {LABELS.get(name, '')}")
    return 0


def cmd_depth(args) -> int:
    from .core.ingest import ingest
    from .depth.backbones import get_backbone
    from .depth.infer import n_chips, predict_depth
    from .dsm.cog import write_cog, write_png_preview

    t0 = time.time()
    scene = ingest(args.image)
    if args.max_side and max(scene.shape) > args.max_side:
        scene = _downscale(scene, args.max_side)
    h, w = scene.shape
    print(f"ingest    {w} x {h}   {scene.meta.describe()}")

    model = get_backbone(args.backbone)
    model.load()
    total = n_chips(scene.shape, args.chip, args.overlap)
    print(f"backbone  {model.describe()}   {total} chip(s) of {args.chip} px, {int(args.overlap*100)}% overlap")

    def progress(done: int, tot: int) -> None:
        el = time.time() - t0
        sys.stdout.write(f"\r  chip {done}/{tot}   {el:5.1f}s")
        sys.stdout.flush()

    depth = predict_depth(scene, model, chip=args.chip, overlap=args.overlap,
                          on_progress=progress)
    print()

    rel = depth.relative
    finite = np.isfinite(rel)
    print(
        f"depth     range [{rel[finite].min():.4f}, {rel[finite].max():.4f}]   "
        f"mean {rel[finite].mean():.4f}   NaN {int((~finite).sum())}"
    )
    if not finite.all():
        print("  WARNING: non-finite values present in the relative depth field")

    out = write_cog(
        args.out, rel, scene.meta, description="relative depth (unitless, higher = taller)",
        tags={"TRAKSHA_STAGE": "relative_depth", "TRAKSHA_BACKBONE": depth.backbone,
              "TRAKSHA_UNITS": "none"},
    )
    print(f"wrote     {out}")
    if args.preview:
        png = os.path.splitext(args.out)[0] + ".png"
        write_png_preview(png, rel, cmap=args.cmap, vmin=0.0, vmax=1.0)
        rgb_png = os.path.splitext(args.out)[0] + "_rgb.png"
        _save_rgb(rgb_png, scene.rgb)
        print(f"wrote     {png}\nwrote     {rgb_png}")
    print(f"done      {time.time() - t0:.1f}s")
    return 0


def _downscale(scene, max_side: int):
    from PIL import Image

    from .core.types import Scene, SceneMeta

    h, w = scene.shape
    scale = max_side / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    rgb = np.asarray(Image.fromarray(scene.rgb).resize((nw, nh), Image.BILINEAR))
    m = scene.meta
    transform = None
    if m.transform:
        a, b, c, d, e, f = m.transform
        transform = (a / scale, b, c, d, e / scale, f)
    meta = SceneMeta(
        crs=m.crs, transform=transform, gsd_m=m.gsd_m / scale, bounds_wgs=m.bounds_wgs,
        sun_azimuth_deg=m.sun_azimuth_deg, sun_elevation_deg=m.sun_elevation_deg,
        off_nadir_deg=m.off_nadir_deg, acquired_utc=m.acquired_utc, source=m.source,
        gsd_is_assumed=m.gsd_is_assumed,
    )
    print(f"resize    {w} x {h} -> {nw} x {nh}   GSD {m.gsd_m:.4g} -> {meta.gsd_m:.4g} m")
    return Scene(rgb=rgb, meta=meta, path=scene.path, raw_dtype=scene.raw_dtype)


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    from PIL import Image

    im = Image.fromarray(rgb)
    if max(im.size) > 2048:
        s = 2048 / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
    im.save(path)


def cmd_sample(args) -> int:
    """Write the bundled real sample scene to disk, with its lidar truth.

    Replaces the old `synth`, which rendered a town. Nothing in this project
    runs on invented pixels any more, so the scene every quick start reaches for
    is a real one - a crop of central Zurich with airborne lidar DSM and DTM.
    See traksha/data/fixture/ATTRIBUTION.md for the swisstopo licence.
    """
    from .data.sample import load_sample_scene
    from .dsm.cog import write_cog, write_png_preview, write_rgb

    t0 = time.time()
    sun = (args.sun_az, args.sun_el) if args.sun_az is not None else None
    sc = load_sample_scene(size=args.size, sun=sun)
    stem = os.path.splitext(args.out)[0]
    write_rgb(args.out, sc.rgb, sc.meta,
              tags={"TRAKSHA_SOURCE": "swisstopo sample scene"})
    write_cog(f"{stem}_dsm.tif", sc.dsm_m, sc.meta,
              description="swissSURFACE3D lidar DSM (m)",
              tags={"TRAKSHA_STAGE": "reference_dsm", "TRAKSHA_UNITS": "m"})
    write_cog(f"{stem}_dtm.tif", sc.dtm_m, sc.meta,
              description="swissALTI3D lidar DTM (m)")
    write_cog(f"{stem}_sem.tif", sc.sem.astype(np.float32), sc.meta, dtype="uint8",
              nodata=255, description="derived labels (colour heuristic)")
    write_png_preview(f"{stem}_dsm.png", sc.dsm_m, cmap="terrain")

    nd = sc.ndsm_m
    print(f"wrote     {args.out} and {stem}_[dsm|dtm|sem].tif")
    print(f"scene     {args.size} x {args.size} @ 0.5 m   "
          f"elev {sc.dsm_m.min():.1f}-{sc.dsm_m.max():.1f} m   "
          f"max height above ground {nd.max():.1f} m")
    if sun is None:
        print("sun       none - swisstopo publishes no acquisition time for these")
        print("          products, so shadow physics will be disabled.")
        print("          Pass --sun-az/--sun-el to assume one.")
    else:
        print(f"sun       {args.sun_az:.1f} az / {args.sun_el:.1f} el   "
              f"(ASSUMED, not measured)   shadow {sc.shadow.mean()*100:.1f}%")
    print("source    swisstopo OGD; see traksha/data/fixture/ATTRIBUTION.md")
    print(f"done      {time.time() - t0:.1f}s")
    return 0


def _load_gcps(path: Optional[str]) -> list:
    """CSV with row,col,elev_m[,label] or x,y,elev_m in the scene CRS."""
    if not path:
        return []
    import csv

    from .core.types import GCP

    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            keys = {k.lower().strip(): v for k, v in rec.items() if k}
            if "row" in keys and "col" in keys:
                out.append(GCP(int(float(keys["row"])), int(float(keys["col"])),
                               float(keys.get("elev_m") or keys.get("elev") or keys.get("z")),
                               str(keys.get("label", ""))))
            else:
                raise ValueError(
                    "GCP csv needs row,col,elev_m columns "
                    "(world x,y are not converted yet)"
                )
    return out


def cmd_fit(args) -> int:
    """Fit the structural scale over a dataset and write it out.

    This is the only part of TRAKSHA that learns anything, and what it learns is
    one number: how many metres of height above ground a unit of high-frequency
    relative depth is worth. It has to be learned rather than solved because the
    anchor ladder cannot observe it - on real imagery every anchor is a ground
    anchor (README section 3.4), so the branch that scales structure gets no
    constraints at all.

    It needs a completed study, not a fresh one: the depth is read back from
    each scene's artifacts, so fitting four scenes takes about a second.
    """
    from .data import discover
    from .learn.collect import collect
    from .learn.scale import fit

    kw = {}
    for key in ("image", "reference", "semantics", "dem"):
        val = getattr(args, f"suffix_{key}", None)
        if val:
            kw.setdefault("suffixes", {})[key] = val
    scenes = discover(args.root, layout=args.layout, **kw)
    print(f"TRAKSHA fit   {args.layout}  {args.root}")
    print(f"  found        {len(scenes)} scene(s)")
    print(f"  runs         {args.runs}")

    def on_scene(name, sample, why):
        if sample is None:
            print(f"    {name:<12} skipped - {why}")
        else:
            print(f"    {name:<12} a* {sample.target:8.1f}   "
                  f"nDSM MAE at that scale {sample.ndsm_mae_at_target:5.2f} m "
                  f"(floor {sample.floor_mae:5.2f} m)")

    samples, rasters, skipped = collect(scenes, args.runs, radius_m=args.radius,
                                        on_scene=on_scene)
    if not samples:
        print("\nerror: no scene could supply a target. Height above ground needs\n"
              "       either an nDSM reference or a DSM with a bare-earth DTM.",
              file=sys.stderr)
        return 1
    if len(samples) < 3:
        print(f"\n  ! only {len(samples)} scene(s). A scale fitted on this few is a\n"
              "    guess with error bars nobody has measured. It is written anyway,\n"
              "    and the model file records how few it saw.")

    model = fit(samples, rasters=rasters, radius_m=args.radius,
                backbone=args.backbone)
    print()
    print(model.describe())
    if np.isfinite(model.loo_mae_alt_m) and model.loo_mae_alt_m != float("inf"):
        alt = "constant" if model.kind == "linear" else "one-feature"
        print(f"  the {alt} alternative scored {model.loo_mae_alt_m:.2f} m and lost")
    if skipped:
        print(f"  {len(skipped)} scene(s) skipped")

    path = model.save(args.out)
    print(f"\nwrote     {path}")
    print("          the pipeline loads this automatically; --scale-model off "
          "disables it")
    return 0


def deliver(run_dir: str, tile: int = 512, bits: int = 12, obj_stride: int = 2,
            obj_tol_m: float = 2.0, obj_max_tris: int = 500_000,
            live=None, quiet: bool = False) -> dict:
    """Phases 3 and 4 into the run's own directory. Reads, never recomputes.

    Everything a scene produces belongs together: the rasters Phase 2 wrote, the
    tileset the browser reads, and the OBJ that goes into anything else. Keeping
    the delivery layer in sibling directories meant a scene was three folders
    that had to be matched up by hand, and a tileset could outlive the run it was
    built from without anything noticing.
    """
    from .mesh.build import build_tileset

    tiles_dir = os.path.join(run_dir, "tiles3d")
    manifest = build_tileset(run_dir, tiles_dir, tile=tile, pad=1,
                             quantise_bits=bits,
                             obj_stride=obj_stride, obj_tol_m=obj_tol_m,
                             obj_max_tris=obj_max_tris, write_mesh=True,
                             mesh_dir=os.path.join(run_dir, "mesh"),
                             on_progress=(lambda d, n_: live.set(d, n_, f"lod {d - 1}"))
                             if live else None)
    if not quiet:
        g = manifest["grid"]
        n_tiles = sum(len(l["tiles"]) for l in manifest["lods"])
        print(f"  tileset      {len(manifest['lods'])} LODs, {n_tiles} tiles, "
              f"{g['width']} x {g['height']} px at {g['gsd_m']:.4g} m")
        m = manifest.get("mesh") or {}
        if m:
            ad = m.get("adaptive")
            how = (f", adaptive: {ad['fine_blocks']}/{ad['blocks']} blocks fine, "
                   f"error <= {ad['tol_m']:g} m") if ad else ""
            print(f"  mesh         {m['vertices']:,} vertices, "
                  f"{m['triangles']:,} triangles{how}")
            print(f"               {os.path.basename(m['obj'])} + "
                  f"{os.path.basename(m.get('mtl', '-'))} + "
                  f"{os.path.basename(m.get('texture', '-'))}")
        for note in manifest.get("notes", []):
            mark = {"critical": "!!", "warning": " !", "info": "  "}.get(note["level"], "  ")
            print(f"  {mark} {note['text']}")
    return manifest


def cmd_build(args) -> int:
    """Phases 1 to 4 on one image, into one directory.

    `run` stops after Phase 2 because tiling is cheap and inference is not, and
    that separation is worth keeping for iteration. It is not worth keeping for
    delivery: what someone wants from a scene is the whole scene - rasters,
    tileset, and a textured mesh - in a folder they can move.
    """
    from .api.pipeline import run as run_pipeline
    from .core.progress import Live
    from .core.types import Config
    from .eval.metrics import format_table

    cfg = Config(
        backbone=args.backbone, chip=args.chip, overlap=args.overlap,
        dem_source=args.dem, reference=args.ref, gcp_file=args.gcps,
        n_bootstrap=args.bootstrap, lattice_stride=args.stride,
        lam_a=args.lam, lam_b=args.lam,
        extras={"batch_size": args.batch, "workers": args.workers,
                "segmentation": "raster" if args.sem else "heuristic",
                "segmentation_path": args.sem,
                "instances": args.instances,
                "instance_points": args.instance_points,
                "scale_model": args.scale_model,
                "dual_branch": args.scale_model not in ("off", "none", "no")},
    )
    live = Live(mode=args.progress)
    t0 = time.time()
    print(f"TRAKSHA build   {os.path.basename(args.image)} -> {args.out}/   {live.banner()}")

    res = run_pipeline(args.image, cfg, gcps=_load_gcps(args.gcps),
                       out_dir=args.out, live=live)

    print(f"\nTier {res.tier.value} ({res.tier_reason})")
    print(f"anchors: {res.anchors_used} used / {res.anchors_rejected} rejected   "
          + "  ".join(f"{k}={v}" for k, v in res.anchor_counts.items() if k != "total"))
    if res.metrics:
        print()
        print(format_table(res.metrics, "VALIDATION"))
        if res.dem_metrics:
            print(f"\n  floor  (DEM alone)        MAE {res.dem_metrics['mae_m']:.2f} m")

    print("\ndelivery")
    with live.task("tiling", None, "lod") as t:
        deliver(args.out, tile=args.tile, bits=args.bits,
                obj_stride=args.obj_stride, obj_tol_m=args.obj_tol, live=t)

    print(f"\nwrote     {os.path.abspath(args.out)}")
    print("            rasters + tiles3d/ + mesh/surface.{obj,mtl,jpg}")
    print(f"\nview it:  python -m traksha.cli viewer {args.out}")
    print(f"total     {time.time() - t0:.1f}s")
    return 0


def cmd_refine(args) -> int:
    """Image-conditioned refinement of a delivered surface, and its verdict.

    Refining a rough mesh against the image it came from is standard practice,
    and it rests on a premise: that the coarse geometry is right and only the
    detail is missing. This command applies the refinement AND tests that
    premise, because on this pipeline the premise is false and a refinement
    shipped without the test would be a feature that does nothing.

    It reports two things:

      the delta      what the refinement did to nDSM MAE and edge F1
      the shape      how much of the error is magnitude, not placement

    If the error is mostly magnitude, refinement and rescaling can help. If it
    is mostly placement - structure predicted where there is none - then no
    edge-aware operator will fix it, because it moves height by a pixel and the
    error is tens of metres away. See README section 5.6.
    """
    import numpy as np

    from .eval.metrics import evaluate
    from .mesh.build import load_run
    from .mesh.refine import refine_run

    run = load_run(args.run)
    gsd = float(run["meta"]["gsd_m"])
    out = refine_run(run, gsd, radius_m=args.radius, eps=args.eps)
    if not out:
        print("error: the run has no nDSM, DSM and texture to refine",
              file=sys.stderr)
        return 2

    st = out["stats"]
    print(f"TRAKSHA refine   {args.run}")
    print(f"  residual     rms {st['rms_residual_m']:.3f} m, "
          f"max {st['max_abs_residual_m']:.2f} m, "
          f"{100 * st['moved_fraction']:.1f}% of pixels moved > 5 cm")

    truth = _reference_ndsm(args.ref, args.dtm, run)
    if truth is None:
        print("  no reference: pass --ref <dsm> --dtm <dtm> to score the refinement")
    else:
        rough, fine = run["ndsm"], out["ndsm"]
        a = evaluate(rough, truth, gsd=gsd, height_pred=rough, height_ref=truth)
        b = evaluate(fine, truth, gsd=gsd, height_pred=fine, height_ref=truth)
        print(f"\n  {'':<10}{'nDSM MAE':>10}{'edge F1':>10}")
        print(f"  {'rough':<10}{a['mae_m']:>10.3f}{a['edge_f1']:>10.3f}")
        print(f"  {'refined':<10}{b['mae_m']:>10.3f}{b['edge_f1']:>10.3f}")
        print(f"  {'delta':<10}{b['mae_m'] - a['mae_m']:>+10.3f}"
              f"{b['edge_f1'] - a['edge_f1']:>+10.3f}")

        # Is the error detail-shaped at all? One global rescale is the cheapest
        # possible magnitude fix; whatever it cannot remove is placement.
        obj = truth > 2.0
        if obj.any():
            r = np.asarray(rough, np.float64)
            k = float((r[obj] * truth[obj]).sum() / max((r[obj] ** 2).sum(), 1e-9))
            base = float(np.abs(r[obj] - truth[obj]).mean())
            scaled = float(np.abs(k * r[obj] - truth[obj]).mean())
            share = 100.0 * (1.0 - scaled / max(base, 1e-9))
            print(f"\n  object error {base:.2f} m; one global rescale (k = {k:.2f}) "
                  f"leaves {scaled:.2f} m")
            print(f"  -> {share:.0f}% of it is magnitude, {100 - share:.0f}% is placement")
            if share < 20:
                print("  -> the error is where structure IS, not how tall it is. "
                      "Edge-aware\n     refinement moves height by a pixel and "
                      "cannot address that.")

    if args.write:
        from .dsm.cog import write_cog

        from .core.types import SceneMeta

        m = run["meta"]
        meta = SceneMeta(crs=m.get("crs"), transform=tuple(m["transform"]),
                         gsd_m=gsd)
        write_cog(os.path.join(args.run, "ndsm_refined.tif"), out["ndsm"], meta,
                  description="height above ground, image-refined (m)")
        write_cog(os.path.join(args.run, "dsm_refined.tif"), out["dsm"], meta,
                  description="elevation, image-refined (m)")
        print(f"\nwrote     {args.run}/[ndsm|dsm]_refined.tif")
    else:
        print("\n  nothing written; pass --write to keep the refined rasters")
    return 0


def _reference_ndsm(ref: Optional[str], dtm: Optional[str], run: dict):
    """Truth height above ground from a DSM and a bare-earth DTM, or None."""
    if not ref or not dtm:
        return None
    import numpy as np
    import rasterio

    with rasterio.open(ref) as a, rasterio.open(dtm) as b:
        z = a.read(1).astype(np.float32) - b.read(1).astype(np.float32)
    if z.shape != run["ndsm"].shape:
        return None
    return np.maximum(z, 0.0)


def cmd_run(args) -> int:
    from .api.pipeline import run
    from .core.types import Config
    from .eval.metrics import format_table

    cfg = Config(
        backbone=args.backbone, chip=args.chip, overlap=args.overlap,
        dem_source=args.dem, reference=args.ref, gcp_file=args.gcps,
        n_bootstrap=args.bootstrap, lattice_stride=args.stride,
        lam_a=args.lam, lam_b=args.lam,
        extras={"batch_size": args.batch,
                "workers": args.workers,
                "segmentation": "raster" if args.sem else "heuristic",
                "segmentation_path": args.sem,
                "instances": args.instances,
                "instance_points": args.instance_points,
                "scale_model": args.scale_model,
                "dual_branch": args.scale_model not in ("off", "none", "no")},
    )
    from .core.progress import Live

    live = Live(mode=args.progress)
    t0 = time.time()
    print(f"TRAKSHA run   {os.path.basename(args.image)}   {live.banner()}")
    res = run(args.image, cfg, gcps=_load_gcps(args.gcps), out_dir=args.out, live=live)

    print(f"\nTier {res.tier.value} ({res.tier_reason})")
    print(f"anchors: {res.anchors_used} used / {res.anchors_rejected} rejected "
          f"of {res.anchor_counts.get('total', 0)}   "
          + "  ".join(f"{k}={v}" for k, v in res.anchor_counts.items() if k != "total"))
    if res.metrics:
        print()
        print(format_table(res.metrics, "VALIDATION"))
        if res.baseline_metrics:
            b = res.baseline_metrics
            print(f"\n  baseline (global affine)   MAE {b['mae_m']:.2f} m   "
                  f"RMSE {b['rmse_m']:.2f} m   r {b['pearson_r']:.3f}")
        if res.dem_metrics:
            d = res.dem_metrics
            print(f"  floor    (DEM alone)       MAE {d['mae_m']:.2f} m   "
                  f"RMSE {d['rmse_m']:.2f} m   r {d['pearson_r']:.3f}")
            if res.metrics.get("mae_m", 1e9) > d["mae_m"]:
                print("  NOTE: the pipeline did not beat the DEM it was anchored to.")
        if res.metrics_by_class:
            print("\n  error by class")
            for name, m in sorted(res.metrics_by_class.items(),
                                  key=lambda kv: kv[1]["mae_m"]):
                bar = "#" * int(min(40, m["mae_m"] * 4))
                print(f"    {name:<14}MAE {m['mae_m']:>6.2f} m  {bar}")
    if args.json:
        import json as _json

        with open(args.json, "w", encoding="utf-8") as fh:
            _json.dump(res.summary(), fh, indent=2)
        print(f"\nwrote     {args.json}")
    print(f"\ntotal     {time.time() - t0:.1f}s   " +
          "  ".join(f"{k} {v:.1f}s" for k, v in res.timings_s.items()))
    return 0


def cmd_bench(args) -> int:
    from .eval.bench import format_bench, save, sweep

    def on_case(bb, chip, batch):
        print(f"  running {bb} chip={chip} batch={batch} ...", flush=True)

    report = sweep(
        image=args.image, size=args.size,
        backbones=args.backbones.split(","),
        chips=[int(c) for c in args.chips.split(",")],
        batches=[int(b) for b in args.batches.split(",")],
        overlap=args.overlap,
        repeats=args.repeats, on_case=on_case,
    )
    print()
    print(format_bench(report))
    if args.json:
        save(report, args.json)
        print(f"\nwrote     {args.json}")
    return 0


def cmd_ablate(args) -> int:
    from .api.pipeline import load_dem, load_reference
    from .core.ingest import ingest
    from .core.types import Config
    from .depth.backbones import get_backbone
    from .depth.infer import predict_depth
    from .eval.ablation import VARIANTS, format_ablation, run_variants, save
    from .semantics.segment import segment
    from .semantics.shadow import detect_shadow

    cfg = Config(backbone=args.backbone, chip=args.chip, overlap=args.overlap,
                 dem_source=args.dem, n_bootstrap=args.bootstrap,
                 lattice_stride=args.stride, lam_a=args.lam, lam_b=args.lam,
                 extras={"batch_size": args.batch})

    t0 = time.time()
    scene = ingest(args.image)
    print(f"scene     {scene.shape[1]} x {scene.shape[0]}   {scene.meta.describe()}")

    model = get_backbone(args.backbone)
    model.load()
    print(f"backbone  {model.describe()}")
    depth = predict_depth(scene, model, chip=args.chip, overlap=args.overlap,
                          batch_size=args.batch,
                          on_progress=lambda d, t: sys.stdout.write(f"\r  chip {d}/{t}   "))
    print(f"\ndepth     {time.time() - t0:.1f}s  (inference runs once for the whole table)")

    sem, _ = segment(scene, method="raster" if args.sem else "heuristic", path=args.sem)
    shadow = detect_shadow(scene, sem)
    dem_m, dem_prov = load_dem(args.dem, scene)
    print(f"dem       {dem_prov}")

    ref = load_reference(args.ref, scene)

    variants = args.variants.split(",") if args.variants else VARIANTS
    rows = run_variants(scene, depth, sem, shadow, ref, dem_m=dem_m, cfg=cfg, variants=variants)
    print()
    print(format_ablation(rows))
    if args.json:
        save(rows, args.json)
        print(f"\nwrote     {args.json} and {os.path.splitext(args.json)[0]}.md")
    print(f"\ntotal     {time.time() - t0:.1f}s")
    return 0


def cmd_mesh(args) -> int:
    """Phase 3: a Phase 2 run directory -> a browser-ready tileset.

    Deliberately separate from `run`. Tiling is cheap and inference is not, so a
    tile size or an encoding can be changed in seconds without re-running the
    pipeline - and, more importantly, this command can only ever *read* what
    Phase 2 wrote. There is no path by which the viewer shows a number the
    calibration did not produce.
    """
    from .core.progress import Live
    from .mesh.build import build_tileset

    out = args.out or os.path.join(args.run, "tiles3d")
    live = Live(mode=args.progress)

    t0 = time.time()
    print(f"TRAKSHA mesh   {args.run} -> {out}/")
    with live.task("tiling", None, "lod") as t:
        manifest = build_tileset(
            args.run, out, tile=args.tile, pad=args.pad, lods=args.lods or None,
            obj_stride=args.obj_stride, obj_tol_m=args.obj_tol,
            write_mesh=not args.no_mesh,
            write_structural=not getattr(args, "no_structural", False),
            quantise_bits=args.bits,
            on_progress=lambda d, n: t.set(d, n, f"lod {d - 1}"),
        )

    g = manifest["grid"]
    n_tiles = sum(len(l["tiles"]) for l in manifest["lods"])
    print(f"  raster       {g['width']} x {g['height']} px at {g['gsd_m']:.4g} m "
          f"({g['extent_m'][0]:.0f} x {g['extent_m'][1]:.0f} m)")
    print(f"  levels       {len(manifest['lods'])} LODs, {n_tiles} tiles of {args.tile} px "
          f"(+{args.pad} px pad)")
    print("  layers       " + "  ".join(sorted(manifest["layers"])))
    for key, spec in sorted(manifest["layers"].items()):
        if spec.get("vmin") is None:
            continue
        print(f"    {key:<8} {spec['vmin']:9.2f} .. {spec['vmax']:9.2f} {spec.get('units') or ''}"
              f"   step {spec.get('step_m', float('nan')):.4g}")
    if manifest.get("mesh"):
        m = manifest["mesh"]
        print(f"  mesh         {m['obj']}  {m['vertices']:,} vertices, "
              f"{m['triangles']:,} triangles")
    for note in manifest.get("notes", []):
        mark = {"critical": "!!", "warning": " !", "info": "  "}.get(note["level"], "  ")
        print(f"  {mark} {note['text']}")
    print(f"done      {time.time() - t0:.1f}s   {out}/tileset.json")
    return 0


def cmd_viewer(args) -> int:
    """Phase 4: build the tileset if needed, then serve the web app locally.

    One command from a finished run to a 3D view in a browser, with no build
    step, no package manager and no network. `web/` is plain HTML, CSS and
    JavaScript for exactly that reason - the same reason every raster is a COG.
    """
    import http.server
    import socketserver
    import webbrowser

    from .mesh.build import build_tileset

    # The front end is a Vite build, so this serves web/dist. Serving the
    # sources would hand the browser a JSX entry point and produce a blank page
    # with nothing in the console to explain it.
    from .api.server import MISSING_BUILD, web_root

    web = web_root()
    if not web:
        print("error: " + MISSING_BUILD, file=sys.stderr)
        return 2

    tiles = args.tiles or os.path.join(args.run, "tiles3d")
    manifest_path = os.path.join(tiles, "tileset.json")
    if args.rebuild or not os.path.exists(manifest_path):
        print(f"building tileset -> {tiles}/")
        build_tileset(args.run, tiles, tile=args.tile, pad=1,
                      obj_stride=args.obj_stride, write_mesh=not args.no_mesh)
    else:
        print(f"using existing tileset {manifest_path}")

    # Serve web/ at the root and the tileset under /data, so the page fetches a
    # relative URL and nothing depends on where either directory happens to sit.
    web_dir, data_dir = os.path.abspath(web), os.path.abspath(tiles)
    # The OBJ lives beside the tileset, not inside it, so a scene is one folder.
    # The manifest therefore records it as `../mesh/surface.obj`, which a browser
    # normalises to `/mesh/surface.obj` before the request is ever sent - hence
    # the second route. Without it the mesh download 404s and nothing else does,
    # which is the kind of break that ships.
    mesh_dir = os.path.abspath(os.path.join(data_dir, os.pardir, "mesh"))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            clean = path.split("?", 1)[0].split("#", 1)[0]
            parts = [p for p in clean.split("/") if p not in ("", ".", "..")]
            if parts and parts[0] == "data":
                return os.path.join(data_dir, *parts[1:])
            if parts and parts[0] == "mesh":
                return os.path.join(mesh_dir, *parts[1:])
            return os.path.join(web_dir, *parts) if parts else os.path.join(web_dir, "index.html")

        def log_message(self, fmt, *a):
            if "200" not in fmt % a:
                super().log_message(fmt, *a)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        url = f"http://localhost:{args.port}/"
        print(f"  web      {web_dir}")
        print(f"  data     {data_dir}  ->  /data/tileset.json")
        print(f"\nserving {url}   (ctrl-c to stop)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def cmd_delivery(args) -> int:
    """Measure Phase 3 and Phase 4 on this machine, and write the evidence.

    The delivery counterpart to `study`. Same contract: one command, a JSON file
    holding every number, and a markdown report rendered from that JSON so the
    two can never disagree.

    Nothing here is a correctness test - `pytest tests/test_mesh.py
    tests/test_viewer.py` covers that. This answers the questions correctness
    tests do not: what does tiling cost, how big is the payload, and how much
    CPU does the viewer burn before the GPU is involved.
    """
    from .eval.delivery import run_delivery, write_report
    from .core.jsonio import save_json

    # Default the report into the run it describes. A delivery.json sitting in a
    # sibling folder outlives the run it was measured from and nothing notices.
    out = args.out or args.run
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    print(f"TRAKSHA delivery   {args.run} -> {out}/")

    tiles = tuple(int(x) for x in str(args.tiles).split(",") if x.strip())
    strides = tuple(int(x) for x in str(args.obj_strides).split(",") if x.strip())

    rep = run_delivery(
        args.run, out, tile=args.tile, tiles=tiles, obj_strides=strides,
        repeats=args.repeats, work_dir=args.work_dir, log=lambda m: print(m),
    )

    json_path = save_json(rep, os.path.join(out, "delivery.json"))
    md_path = write_report(rep, os.path.join(out, "DELIVERY.md"))

    b, p = rep["build"], rep["payload"]
    print()
    print(f"  build        {b['tiles_only_s']:.2f}s tiles, {b['full_s']:.2f}s with the OBJ"
          f"   ({b['mpix_per_s']:.2f} Mpix/s)")
    print(f"  payload      {p['total_bytes'] / 1e6:.1f} MB total, "
          f"{p['first_paint_bytes'] / 1e6:.2f} MB before first paint")
    v = rep.get("viewer") or {}
    if v.get("totals_ms"):
        print(f"  viewer CPU   {v['totals_ms']['first_paint_cpu']:.0f} ms before first paint"
              f"   (GPU not measured)")
    elif v.get("skipped"):
        print(f"  viewer CPU   skipped: {v['skipped']}")
    bad = [r for r in rep["roundtrip"] if not r["within_half_a_step"]]
    print(f"  round trip   {len(rep['roundtrip']) - len(bad)}/{len(rep['roundtrip'])}"
          f" layer-LOD pairs within half a step")
    for r in bad:
        print(f"    ! lod {r['lod']} {r['layer']}: {r['max_abs_error_m']:.3g} m")
    print(f"\ndone      {time.time() - t0:.1f}s   {json_path}   {md_path}")
    return 1 if bad else 0


def _completed_scene(ref, out_dir: str, want_delivery: bool):
    """A finished scene's record, read back off disk, or None.

    Lets a study be resumed rather than restarted. That is not a convenience:
    inference peaks at over a gigabyte and a modest machine cannot always hold
    it, so a study that cannot resume is a study that may never finish.
    """
    import json

    path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(path):
        return None
    if want_delivery and not os.path.exists(
            os.path.join(out_dir, "tiles3d", "tileset.json")):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            summ = json.load(fh)
    except Exception:
        return None
    if not summ.get("metrics"):
        return None
    rec = {"name": ref.name, "image": os.path.abspath(ref.image),
           "reference_kind": ref.reference_kind, "resumed": True}
    for key in ("tier", "tier_reason", "anchors", "timings_s", "metrics",
                "metrics_by_class", "baseline_metrics", "dem_metrics",
                "ndsm_metrics", "zero_baseline_metrics", "relief"):
        if key in summ:
            rec[key] = summ[key]
    # A resumed scene reports the artifacts it already has, or the record would
    # claim less than the folder contains.
    tileset = os.path.join(out_dir, "tiles3d", "tileset.json")
    obj = os.path.join(out_dir, "mesh", "surface.obj")
    if os.path.exists(tileset):
        rec["delivery"] = {"tileset": tileset,
                           "mesh": obj if os.path.exists(obj) else None}
    return rec


def cmd_facades(args) -> int:
    """Synthesise facade texture for a run's structural mesh, on a GPU."""
    import json

    from .mesh import facades as fa
    from .mesh.build import load_run
    from .mesh.structural import build as build_structural
    from .mesh.structural import select as select_buildings
    from .semantics import instances as inst_mod

    checks = fa.preflight()
    print("TRAKSHA facades   threefiner, fixed geometry, texture only")
    for k in ("torch", "cuda", "device", "vram_gb"):
        if k in checks:
            print(f"  {k:12s} {checks[k]}")
    for note in checks.get("notes", []):
        print(f"  note         {note}")
    if checks["missing"]:
        print("  missing:")
        for m in checks["missing"]:
            print(f"    - {m}")
        if not args.dry_run:
            print("\nNothing to do without those. `--dry-run` prepares "
                  "the per-building meshes anyway, so the handoff can "
                  "be checked on a machine with no GPU.")
            return 1

    run = load_run(args.run)
    field = inst_mod.load(os.path.join(args.run, "segmentation"))
    if field is None:
        print("error: this run has no segmentation/; re-run with instances enabled")
        return 1
    gsd = float(run["meta"].get("gsd_m") or 1.0)
    sem = run.get("sem")
    buildings = select_buildings(field, run["dsm"], run["ndsm"],
                                 None if sem is None else sem.astype("uint8"))
    mesh = build_structural(run["dsm"], run["ndsm"], buildings, gsd)
    print(f"  buildings    {len(buildings)} in the mesh, refining "
          f"{min(len(buildings), args.limit)}")

    out = args.out or os.path.join(args.run, "facades")
    rec = fa.refine(mesh, out, max_buildings=args.limit,
                    prompt=args.prompt or fa.DEFAULT_PROMPT,
                    preset=args.preset, iters=args.iters or None,
                    dry_run=args.dry_run)
    ok = sum(1 for b in rec["buildings"] if b.get("glb") or b.get("dry_run"))
    print(f"  refined      {ok}/{len(rec['buildings'])} into {out}/")

    # The assembled model: the whole scene, with threefiner's texture on the
    # buildings it refined. This is the artifact the request is really about -
    # per-building GLBs are an intermediate, not a deliverable.
    tileset = args.tileset or _guess_tileset(args.run)
    mesh_dir = os.path.join(os.path.dirname(tileset), "mesh") if tileset else out
    base_tex = os.path.join(mesh_dir, "surface.jpg")
    info = fa.assemble(mesh, rec["buildings"],
                       os.path.join(mesh_dir, "structural_refined.obj"),
                       run["dsm"].shape, gsd,
                       base_texture=base_tex if os.path.exists(base_tex) else None)
    print(f"  assembled    {os.path.relpath(info['obj'], os.getcwd())}")
    print(f"               {info['triangles']:,} triangles, "
          f"{info['buildings_refined']}/{info['buildings_total']} buildings with "
          f"synthesised walls")

    if tileset:
        # Patch the manifest the viewer reads, atomically, so a half-written
        # tileset.json cannot take the site down.
        with open(tileset, encoding="utf-8") as fh:
            man = json.load(fh)
        entry = {k: v for k, v in info.items() if k != "obj"}
        entry["obj"] = os.path.relpath(info["obj"],
                                       os.path.dirname(tileset)).replace("\\", "/")
        entry["mtl"] = os.path.relpath(info["mtl"],
                                       os.path.dirname(tileset)).replace("\\", "/")
        man.setdefault("mesh", {}).setdefault("structural", {})["refined"] = entry
        tmp = tileset + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1, default=float)
        os.replace(tmp, tileset)
        print(f"  manifest     {os.path.basename(tileset)} updated")

    rec["assembled"] = info
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=float)
    return 0


def _guess_tileset(run_dir: str):
    """Where the viewer's manifest lives for this run, if it is beside it."""
    for candidate in (os.path.join(run_dir, "tiles3d", "tileset.json"),
                      os.path.join(os.path.dirname(run_dir), "tiles", "tileset.json"),
                      os.path.join(run_dir, "..", "tiles3d", "tileset.json")):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def cmd_dataset(args) -> int:
    """Run the pipeline over a real dataset and aggregate the metrics.

    The counterpart to `study`, which generates its own scenes. Everything this
    project has measured so far came from a renderer it wrote; this is the
    command that changes that, and until it has been run against real imagery
    every number in the README carries the caveat in section 2.6.

    Downloads nothing. Point --root at data you already have.
    """
    from .core.jsonio import save_json
    from .core.progress import Live
    from .core.types import Config
    from .data import aggregate, discover, run_scene

    suffixes = {}
    for key, val in (("image", args.suffix_image),
                     ("reference", args.suffix_reference),
                     ("semantics", args.suffix_semantics),
                     ("dem", args.suffix_dem)):
        if val:
            suffixes[key] = val

    kw = {"suffixes": suffixes} if suffixes else {}
    if args.layout == "us3d" and args.dem_dir:
        kw["dem_dir"] = args.dem_dir

    try:
        scenes = discover(args.root, layout=args.layout, **kw)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.limit:
        scenes = scenes[: args.limit]

    print(f"TRAKSHA dataset   {args.layout}  {args.root}")
    print(f"  found        {len(scenes)} scenes")
    have_ref = sum(1 for s in scenes if s.reference)
    have_dem = sum(1 for s in scenes if s.dem)
    have_sem = sum(1 for s in scenes if s.semantics)
    print(f"  reference    {have_ref}/{len(scenes)}"
          f"   dem {have_dem}/{len(scenes)}   semantics {have_sem}/{len(scenes)}")
    if not have_ref:
        print("  ! no reference rasters: metrics cannot be computed, only artifacts")
    if not have_dem:
        print("  ! no DEM: scenes will drop to a lower calibration tier (see README 2.1)")

    if args.list:
        for s in scenes:
            print(f"    {s.describe()}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    live = Live(mode=args.progress)
    t0 = time.time()
    records, failures = [], []

    with live.task("dataset", len(scenes), "scene") as overall:
        for i, ref in enumerate(scenes, 1):
            cfg = Config(
                backbone=args.backbone, chip=args.chip, n_bootstrap=args.bootstrap,
                extras={"batch_size": args.batch,
                        "workers": args.workers,
                        "instances": args.instances,
                        "instance_points": args.instance_points,
                        "dual_branch": bool(args.dual_branch) or
                        args.scale_model not in ("off", "none", "no"),
                        "scale_model": args.scale_model},
            )
            out_dir = os.path.join(args.out, ref.name)
            try:
                done = _completed_scene(ref, out_dir, args.deliver)
                if args.resume and done is not None:
                    records.append(done)
                    live.log(f"  {ref.name}  already done, kept")
                    overall.set(i, len(scenes), ref.name)
                    continue
                rec = run_scene(ref, out_dir, cfg)
                # The scene's arrays are dead once the record exists, and the
                # tiler is about to read them all back off disk. Collect first:
                # a four-scene study on an 8 GB machine has no headroom to hold
                # both, and it does not need to.
                gc.collect()
                # Phases 3 and 4 into the same directory, so each scene is one
                # folder that can be opened, moved or shipped on its own.
                if args.deliver:
                    man = deliver(out_dir, tile=args.tile, bits=args.bits,
                                  obj_stride=args.obj_stride,
                                  obj_tol_m=args.obj_tol, quiet=True)
                    rec["delivery"] = {
                        "tileset": os.path.join(out_dir, "tiles3d", "tileset.json"),
                        "mesh": (man.get("mesh") or {}).get("obj"),
                        "notes": [{"level": x["level"], "id": x["id"]}
                                  for x in man.get("notes", [])],
                    }
                records.append(rec)
                m = rec.get("metrics") or {}
                mae = m.get("mae_m")
                live.log(f"  {ref.name}  tier {rec['tier']}  "
                         f"MAE {'-' if mae is None else f'{mae:.2f} m'}"
                         + ("  +3D" if args.deliver else ""))
            except Exception as exc:                    # one bad tile must not end the run
                failures.append({"name": ref.name, "error": f"{type(exc).__name__}: {exc}"})
                live.log(f"  {ref.name}  FAILED  {type(exc).__name__}: {exc}")
            overall.set(i, len(scenes), ref.name)

    payload = {
        "traksha_dataset_version": 1,
        "root": os.path.abspath(args.root),
        "layout": args.layout,
        "config": {"backbone": args.backbone, "chip": args.chip,
                   "bootstrap": args.bootstrap,
                   "dual_branch": bool(args.dual_branch)},
        "n_found": len(scenes), "n_ok": len(records), "n_failed": len(failures),
        "aggregate": aggregate(records) if records else {},
        "scenes": records, "failures": failures,
        "wall_s": round(time.time() - t0, 1),
    }
    path = save_json(payload, os.path.join(args.out, "dataset.json"))

    agg = payload["aggregate"]
    print()
    if agg.get("mae_m"):
        print(f"  MAE          {agg['mae_m']['mean']:.2f} +/- {agg['mae_m']['std']:.2f} m")
    if agg.get("edge_f1"):
        print(f"  edge F1      {agg['edge_f1']['mean']:.3f}")
    for label, key in (("DEM floor", "dem_metrics_mae_m"),
                       ("global affine", "baseline_metrics_mae_m"),
                       ("flat ground", "zero_baseline_metrics_mae_m")):
        if agg.get(key):
            print(f"  {label:<15}MAE {agg[key]['mean']:.2f} m")

    # Elevation MAE flatters this pipeline: most of a scene is ground, and the
    # DEM already knows the ground. Where a bare-earth DTM ships, print the
    # quantity actually being claimed - height above ground - beside the floor
    # that any method must clear to be reconstructing anything at all.
    if agg.get("ndsm_metrics_mae_m"):
        print("\n  nDSM (height above ground)")
        print(f"    MAE          {agg['ndsm_metrics_mae_m']['mean']:.2f} m")
        if agg.get("zero_baseline_metrics_mae_m"):
            print(f"    flat ground  {agg['zero_baseline_metrics_mae_m']['mean']:.2f} m"
                  "   <- predicting zero everywhere")
        if agg.get("true_mean_height_m") and agg.get("pred_mean_height_m"):
            t, q = agg["true_mean_height_m"]["mean"], agg["pred_mean_height_m"]["mean"]
            print(f"    relief       {q:.2f} m recovered of {t:.2f} m true"
                  f"  ({100 * q / t:.0f}%)")
    if failures:
        print(f"  ! {len(failures)} scene(s) failed")
    print(f"\ndone      {time.time() - t0:.1f}s   {path}")
    return 1 if failures and not records else 0


def in_notebook_vm() -> str:
    """"colab", "kaggle" or "" - a hosted VM whose localhost you cannot reach.

    Worth detecting because the failure is silent and misleading: the server
    starts, prints a URL, and the browser cannot open it, because 127.0.0.1 is
    inside the VM and the browser is not. Printing a working instruction beats
    printing an address that looks right.
    """
    import sys

    if "google.colab" in sys.modules or os.path.isdir("/content"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return ""


def _print_hosted_help(port: int, what: str) -> None:
    print()
    print("  ! This looks like Colab. http://127.0.0.1:%d is inside the VM and" % port)
    print("    your browser cannot reach it, so the tab will fail to open.")
    print()
    print("    Stop this cell, and run %s from a PYTHON cell instead:" % what)
    print()
    print("      import subprocess, time")
    print("      from google.colab.output import eval_js")
    print("      subprocess.Popen(['python', '-m', 'traksha.cli', 'serve',")
    print("                        '--host', '0.0.0.0', '--port', '%d'])" % port)
    print("      time.sleep(8)")
    print("      print(eval_js('google.colab.kernel.proxyPort(%d)'))" % port)
    print()
    print("    Open the URL that prints. `!python ...` cannot do this: it blocks")
    print("    the cell, and only a Python cell can ask Colab for a proxy URL.")
    print()


def cmd_serve(args) -> int:
    """Run the web service: upload a scene, watch it reconstruct, view it in 3D.

    Distinct from `viewer`, which serves one prebuilt tileset and runs no
    pipeline. This one accepts uploads and runs the whole thing per request, so
    it needs the `api` extra and it needs to be treated as a service rather than
    a script - see the warnings in traksha/api/server.py about what it does not do.
    """
    from .api.server import serve

    hosted = in_notebook_vm()
    print(f"TRAKSHA serve   http://{args.host}:{args.port}/")
    print(f"  jobs         {os.path.abspath(args.jobs)}")
    print(f"  concurrency  {args.concurrency} reconstruction"
          f"{'' if args.concurrency == 1 else 's'} at a time")
    if args.host not in ("127.0.0.1", "localhost"):
        print("  ! bound beyond localhost: there is no auth and no rate limiting here")
    if hosted:
        _print_hosted_help(args.port, "it")
    else:
        print()
    try:
        return serve(host=args.host, port=args.port, jobs_root=args.jobs,
                     max_concurrent=args.concurrency)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_preflight(args) -> int:
    """Run the whole pipeline end to end and report a verdict.

    "Does this install actually work" cannot be answered by inspection, so it is
    answered by a command: sample -> depth -> anchors -> AGMC -> uncertainty ->
    artifacts -> tileset, on the bundled real scene, with the fitted structural
    scale in play.

    Exit status is 0 only if every stage ran, every headline metric came back
    finite, and the artifacts are on disk.
    """
    import tempfile

    import numpy as np

    from .api.pipeline import run as run_pipeline
    from .core.types import Config
    from .eval.bench import device_report
    from .data.sample import load_sample_scene
    from .mesh.build import build_tileset
    from .dsm.cog import write_cog, write_rgb

    rep = device_report()
    print("TRAKSHA preflight\n")
    print(f"  torch              {rep.get('torch')}")
    print(f"  cpu                {rep.get('cpu_count')} cores, "
          f"{rep.get('threads')} thread(s)")

    failures = []
    work = tempfile.mkdtemp(prefix="traksha-preflight-")
    t0 = time.time()

    # ---- a real scene with lidar truth, so the metrics mean something ----
    sc = load_sample_scene(size=args.size)
    img = os.path.join(work, "scene.tif")
    write_rgb(img, sc.rgb, sc.meta, tags={"TRAKSHA_SOURCE": "swisstopo sample scene"})
    dsm_p = os.path.join(work, "scene_dsm.tif")
    dtm_p = os.path.join(work, "scene_dtm.tif")
    write_cog(dsm_p, sc.dsm_m, sc.meta, description="reference DSM (m)")
    write_cog(dtm_p, sc.dtm_m, sc.meta, description="reference DTM (m)")
    print(f"  scene              {args.size} x {args.size} px at 0.5 m"
          "   (bundled swisstopo sample, real lidar truth)")

    # ---- the full pipeline on the requested device -----------------------
    cfg = Config(
        backbone=args.backbone, chip=min(args.chip, args.size), reference=dsm_p,
        dem_source=f"sim:{dtm_p}", n_bootstrap=args.bootstrap,
        extras={"batch_size": args.batch,
                "workers": args.workers},
    )
    out_dir = os.path.join(work, "run")
    print("\n  running the pipeline ...")
    res = run_pipeline(img, cfg=cfg, out_dir=out_dir, write_artifacts=True)

    # ---- did the fitted scale reach the surface? -------------------------
    from .learn.scale import load_bundled

    model = load_bundled()
    print("\n  scale model        " +
          (f"{model.kind}, a = {model.value:.1f}, fitted on {model.n_scenes} scene(s)"
           if model else "none bundled - structure will not be scaled"))

    # ---- every stage must have run ---------------------------------------
    print("\n  stage timings")
    for k, v in res.timings_s.items():
        print(f"    {k:<14}{v:7.2f}s")
    missing = [s for s in ("ingest", "depth", "anchors", "calibration",
                           "uncertainty", "assemble") if s not in res.timings_s]
    if missing:
        failures.append(f"stages did not run: {missing}")

    # ---- metrics must be finite ------------------------------------------
    mets = res.metrics or {}
    print("\n  metrics")
    for k in ("mae_m", "rmse_m", "pearson_r", "coverage_1s", "edge_f1"):
        v = mets.get(k)
        ok = v is not None and np.isfinite(v)
        print(f"    {k:<14}{'-' if v is None else f'{v:7.3f}'}   {'ok' if ok else 'NOT FINITE'}")
        if not ok:
            failures.append(f"metric {k} is not finite")

    # ---- artifacts, and the delivery path -------------------------------
    for name in ("dsm", "ndsm", "sigma"):
        if not os.path.exists(os.path.join(out_dir, f"{name}.tif")):
            failures.append(f"missing artifact {name}.tif")
    try:
        man = build_tileset(out_dir, os.path.join(work, "tiles"), tile=256,
                            write_mesh=False, quantise_bits=12)
        print(f"\n  tileset            {len(man['lods'])} LODs, "
              f"{sum(len(l['tiles']) for l in man['lods'])} tiles")
    except Exception as exc:
        failures.append(f"tileset build failed: {exc}")

    # ---- verdict ---------------------------------------------------------
    dt = time.time() - t0
    print(f"\n  wall               {dt:.1f}s")
    shutil.rmtree(work, ignore_errors=True)
    if failures:
        print("\nPREFLIGHT FAILED")
        for f in failures:
            print(f"  ! {f}")
        return 1
    print("\nPREFLIGHT OK - the full pipeline runs end to end on this machine")
    return 0


def cmd_doctor(args) -> int:
    """Is this machine ready to run TRAKSHA, and how fast will it be."""
    from .eval.bench import device_report

    rep = device_report()
    print("TRAKSHA doctor\n")
    print(f"  platform        {rep['platform']}")
    print(f"  python          {rep['python']}   cpus {rep['cpu_count']}")
    print(f"  torch           {rep.get('torch') or 'NOT INSTALLED'}")
    print(f"  threads         {rep.get('threads') or '-'}"
          "   (TRAKSHA is CPU-only by design - see traksha/depth/backbones/hf.py)")
    print(f"  rasterio        {rep.get('rasterio') or 'NOT INSTALLED'}"
          + (f"   gdal {rep['gdal']}" if rep.get("rasterio") else ""))

    ok = True
    for mod in ("numpy", "scipy", "PIL", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            print(f"  MISSING         {mod}")
            ok = False
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("  transformers    NOT INSTALLED - no backbone can run")
        ok = False

    if args.load:
        from .depth.backbones import get_backbone

        for name in args.load.split(","):
            try:
                t0 = time.time()
                m = get_backbone(name).load()
                print(f"  loaded          {m.describe()}   {time.time() - t0:.1f}s   {m.stats()}")
            except Exception as exc:
                print(f"  FAILED          {name}: {type(exc).__name__}: {exc}")
                ok = False

    print("\n  " + ("ready" if ok else "not ready - see the lines above"))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("traksha", description="Metric elevation from a single image.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="print everything we can read about an image")
    pi.add_argument("image")
    pi.set_defaults(func=cmd_info)

    pb = sub.add_parser("backbones", help="list available depth backbones")
    pb.set_defaults(func=cmd_backbones)

    pd = sub.add_parser("depth", help="relative depth raster (Phase 1)")
    pd.add_argument("image")
    pd.add_argument("--out", default="out/depth.tif")
    pd.add_argument("--backbone", default="dav2-vits")
    pd.add_argument("--chip", type=int, default=1024)
    pd.add_argument("--overlap", type=float, default=0.25)
    pd.add_argument("--max-side", type=int, default=0,
                    help="downscale the longest side before inference (0 = off)")
    pd.add_argument("--preview", action="store_true", help="also write quick-look PNGs")
    pd.add_argument("--cmap", default="magma")
    pd.set_defaults(func=cmd_depth)

    psm = sub.add_parser("sample",
                         help="write the bundled real sample scene, with lidar truth")
    psm.add_argument("--out", default="data/sample.tif")
    psm.add_argument("--size", type=int, default=576)
    psm.add_argument("--sun-az", type=float, default=None,
                     help="ASSUME a sun azimuth; none is published for this scene")
    psm.add_argument("--sun-el", type=float, default=45.0)
    psm.set_defaults(func=cmd_sample)

    pfit = sub.add_parser("fit",
                          help="fit the structural scale over a dataset (the only learning step)")
    pfit.add_argument("root", help="the scene directory the study ran on")
    pfit.add_argument("--runs", required=True,
                      help="where that study wrote its per-scene artifacts")
    pfit.add_argument("--layout", default="generic", choices=["generic", "us3d"])
    pfit.add_argument("--out", default="traksha/learn/calibration.json")
    pfit.add_argument("--backbone", default="dav2-vitl",
                      help="the backbone whose runs these are; recorded in the model")
    pfit.add_argument("--radius", type=float, default=60.0,
                      help="frequency split, metres; must match the runs")
    pfit.add_argument("--suffix-image", default=None)
    pfit.add_argument("--suffix-reference", default=None)
    pfit.add_argument("--suffix-semantics", default=None)
    pfit.add_argument("--suffix-dem", default=None)
    pfit.set_defaults(func=cmd_fit)

    pb = sub.add_parser("build",
                        help="phases 1-4 on one image, into one directory")
    pb.add_argument("image")
    pb.add_argument("--out", default="out/build")
    pb.add_argument("--backbone", default="dav2-vitl")
    pb.add_argument("--chip", type=int, default=512)
    pb.add_argument("--overlap", type=float, default=0.25)
    pb.add_argument("--batch", type=int, default=1)
    pb.add_argument("--workers", type=int, default=0)
    pb.add_argument("--dem", default=None)
    pb.add_argument("--ref", default=None)
    pb.add_argument("--sem", default=None)
    pb.add_argument("--gcps", default=None)
    pb.add_argument("--bootstrap", type=int, default=12)
    pb.add_argument("--stride", type=int, default=32)
    pb.add_argument("--lam", type=float, default=1.0)
    pb.add_argument("--instances", default="auto",
                    help="structural segmentation before depth: auto, off, or a SAM 2 variant (sam2-tiny|small|base|large)")
    pb.add_argument("--instance-points", type=int, default=16,
                    help="SAM 2 prompt grid per side; cost is quadratic in this")
    pb.add_argument("--scale-model", default="auto",
                    help="fitted structural scale: auto, off, a number, or a path")
    pb.add_argument("--tile", type=int, default=512)
    pb.add_argument("--bits", type=int, default=12)
    pb.add_argument("--obj-tol", type=float, default=2.0,
                    help="adaptive mesh tolerance in metres: every block is "
                         "emitted at whatever resolution keeps it within this "
                         "of the full-resolution surface. 0 uses --obj-stride")
    pb.add_argument("--obj-stride", type=int, default=2,
                    help="uniform mesh decimation, used only when --obj-tol 0")
    pb.add_argument("--progress", default="auto", choices=["auto", "rich", "plain", "none"])
    pb.set_defaults(func=cmd_build)

    prf = sub.add_parser("refine",
                         help="image-conditioned surface refinement, and whether it helps")
    prf.add_argument("run", help="a directory written by `traksha run` or `traksha build`")
    prf.add_argument("--ref", default=None, help="reference DSM, to score the refinement")
    prf.add_argument("--dtm", default=None, help="bare-earth DTM, for height above ground")
    prf.add_argument("--radius", type=float, default=2.0, help="guide window, metres")
    prf.add_argument("--eps", type=float, default=1e-4,
                     help="guided-filter regularisation; smaller follows the image harder")
    prf.add_argument("--write", action="store_true",
                     help="keep the refined rasters (off by default: on the scenes "
                          "measured here the refinement changes nothing)")
    prf.set_defaults(func=cmd_refine)

    pr = sub.add_parser("run", help="full pipeline: depth -> anchors -> metric DSM (Phase 2)")
    pr.add_argument("image")
    pr.add_argument("--out", default="out/run", help="artifact directory")
    pr.add_argument("--backbone", default="dav2-vits")
    pr.add_argument("--batch", type=int, default=0,
                    help="chips per forward pass; 0 = pick from free VRAM")
    pr.add_argument("--workers", type=int, default=0,
                    help="threads for the uncertainty bootstrap; 0 = auto, 1 = serial. "
                         "The solves are independent and SciPy releases the GIL, so this "
                         "is ~2.3x on eight cores and bit-identical to serial")
    pr.add_argument("--chip", type=int, default=1024)
    pr.add_argument("--overlap", type=float, default=0.25)
    pr.add_argument("--dem", default=None,
                    help="bare-earth DEM GeoTIFF, or sim:<terrain.tif> for development")
    pr.add_argument("--ref", default=None, help="reference DSM for validation")
    pr.add_argument("--sem", default=None, help="segmentation raster (else heuristic)")
    pr.add_argument("--gcps", default=None, help="csv with row,col,elev_m")
    pr.add_argument("--bootstrap", type=int, default=24, help="sigma bootstrap samples; 0 = off")
    pr.add_argument("--stride", type=int, default=32, help="AGMC lattice stride in pixels")
    pr.add_argument("--lam", type=float, default=1.0,
                    help="AGMC smoothness weight; results are flat over roughly 0.25-4")
    pr.add_argument("--instances", default="auto",
                   help="structural segmentation before depth: auto, off, or a SAM 2 variant (sam2-tiny|small|base|large)")
    pr.add_argument("--instance-points", type=int, default=16,
                   help="SAM 2 prompt grid per side; cost is quadratic in this")
    pr.add_argument("--scale-model", default="auto",
                    help="fitted structural scale: auto, off, a number, or a path")
    pr.add_argument("--json", default=None, help="write the run summary here")
    pr.add_argument("--progress", default="auto", choices=["auto", "rich", "plain", "none"])
    pr.set_defaults(func=cmd_run)

    pbn = sub.add_parser("bench", help="throughput sweep over backbones, chip and batch sizes")
    pbn.add_argument("--image", default=None,
                     help="scene to bench; omit for the bundled sample (see traksha/data/sample.py)")
    pbn.add_argument("--size", type=int, default=1024, help="scene side, tiled from the sample")
    pbn.add_argument("--backbones", default="dav2-vits")
    pbn.add_argument("--chips", default="512,1024")
    pbn.add_argument("--batches", default="1")
    pbn.add_argument("--overlap", type=float, default=0.25)
    pbn.add_argument("--repeats", type=int, default=1)
    pbn.add_argument("--json", default="out/bench.json")
    pbn.set_defaults(func=cmd_bench)

    pa = sub.add_parser("ablate", help="one inference, every calibration variant, one table")
    pa.add_argument("image")
    pa.add_argument("--ref", required=True, help="reference DSM (required)")
    pa.add_argument("--dem", default=None)
    pa.add_argument("--sem", default=None)
    pa.add_argument("--backbone", default="dav2-vits")
    pa.add_argument("--batch", type=int, default=0)
    pa.add_argument("--chip", type=int, default=1024)
    pa.add_argument("--overlap", type=float, default=0.25)
    pa.add_argument("--bootstrap", type=int, default=16)
    pa.add_argument("--stride", type=int, default=32)
    pa.add_argument("--lam", type=float, default=1.0)
    pa.add_argument("--variants", default=None, help="comma-separated subset")
    pa.add_argument("--json", default="out/ablation.json")
    pa.set_defaults(func=cmd_ablate)

    pm = sub.add_parser("mesh", help="Phase 3: run directory -> browser tileset + OBJ mesh")
    pm.add_argument("run", help="a directory written by `traksha run` (contains dsm.tif)")
    pm.add_argument("--out", default=None, help="tileset directory (default: <run>/tiles3d)")
    pm.add_argument("--tile", type=int, default=512)
    pm.add_argument("--pad", type=int, default=1,
                    help="overlap pixels per side, for seam-free normals")
    pm.add_argument("--lods", type=int, default=0, help="0 = derive from the raster size")
    pm.add_argument("--obj-tol", type=float, default=2.0,
                    help="adaptive mesh tolerance, metres; 0 uses --obj-stride")
    pm.add_argument("--obj-stride", type=int, default=2,
                    help="decimation for the OBJ export; 1 is full resolution")
    pm.add_argument("--no-structural", action="store_true",

                     help="skip the structural rebuild; deliver only the "

                          "height-field surface.obj")
    pm.add_argument("--no-mesh", action="store_true", help="skip the OBJ export")
    pm.add_argument("--bits", type=int, default=24,
                    help="bits kept per linear layer; 12 saves ~76%% of tile bytes "
                         "at ~0.1%% of each layer's range (the decode is unchanged)")
    pm.add_argument("--progress", default="auto", choices=["auto", "rich", "plain", "none"])
    pm.set_defaults(func=cmd_mesh)

    pf = sub.add_parser("facades", help=(
        "synthesise facade texture with threefiner (needs a CUDA GPU). "
        "Geometry is fixed and the output is a separate, labelled artifact."))
    pf.add_argument("run", help="a run directory with segmentation/")
    pf.add_argument("--out", default=None, help="default <run>/facades")
    pf.add_argument("--limit", type=int, default=8,
                    help="how many buildings, largest first; minutes each on a GPU")
    pf.add_argument("--preset", default="sd_fixgeo",
                    choices=["sd_fixgeo", "if_fixgeo", "if2_fixgeo"],
                    help="only fixed-geometry presets; the others deform measured walls")
    pf.add_argument("--prompt", default=None)
    pf.add_argument("--iters", type=int, default=0, help="0 keeps the preset default")
    pf.add_argument("--dry-run", action="store_true",
                    help="prepare the per-building meshes without touching a GPU")
    pf.add_argument("--tileset", default=None,
                    help="tileset.json to register the refined model with; "
                         "found automatically when it sits beside the run")
    pf.add_argument("--json", default=None, help="write the record here")
    pf.set_defaults(func=cmd_facades)

    pv = sub.add_parser("viewer", help="Phase 4: build if needed, then serve the 3D web app")
    pv.add_argument("run", help="a directory written by `traksha run`")
    pv.add_argument("--tiles", default=None, help="tileset directory (default: <run>/tiles3d)")
    pv.add_argument("--port", type=int, default=8020)
    pv.add_argument("--tile", type=int, default=512)
    pv.add_argument("--obj-stride", type=int, default=2)
    pv.add_argument("--no-mesh", action="store_true")
    pv.add_argument("--rebuild", action="store_true", help="rebuild even if a tileset exists")
    pv.add_argument("--no-open", action="store_true")
    pv.set_defaults(func=cmd_viewer)

    pdel = sub.add_parser("delivery",
                          help="Phase 3/4 CPU benchmark: build cost, payload, viewer CPU")
    pdel.add_argument("run", nargs="?", default="results/zurich",
                      help="a directory written by `traksha run` or `traksha build`")
    pdel.add_argument("--out", default=None,
                      help="where delivery.json lands (default: inside the run "
                           "directory, so a scene stays one folder)")
    pdel.add_argument("--tile", type=int, default=512, help="tile size for the reference build")
    pdel.add_argument("--tiles", default="128,256,512,1024", help="tile sizes to sweep")
    pdel.add_argument("--obj-strides", default="1,2,4,8", help="mesh decimations to sweep")
    pdel.add_argument("--repeats", type=int, default=3,
                      help="timing repeats; the best is reported")
    pdel.add_argument("--work-dir", default=None,
                      help="where timed builds are written (default: the system temp "
                           "directory, which is usually not virus-scanned)")
    pdel.set_defaults(func=cmd_delivery)

    pds = sub.add_parser("dataset",
                         help="run the pipeline over a REAL dataset and aggregate metrics")
    pds.add_argument("root", help="directory you already have; nothing is downloaded")
    pds.add_argument("--layout", default="generic", choices=["generic", "us3d"])
    pds.add_argument("--out", default="results/dataset")
    pds.add_argument("--list", action="store_true", help="show what was found and stop")
    pds.add_argument("--limit", type=int, default=0, help="first N scenes only")
    pds.add_argument("--dem-dir", default=None, help="us3d: where <TILE>_DEM.tif live")
    pds.add_argument("--suffix-image", default=None)
    pds.add_argument("--suffix-reference", default=None)
    pds.add_argument("--suffix-semantics", default=None)
    pds.add_argument("--suffix-dem", default=None)
    pds.add_argument("--backbone", default="dav2-vits")
    pds.add_argument("--chip", type=int, default=512)
    pds.add_argument("--bootstrap", type=int, default=12)
    pds.add_argument("--batch", type=int, default=0)
    pds.add_argument("--workers", type=int, default=0)
    pds.add_argument("--instances", default="auto",
                    help="structural segmentation before depth: auto, off, or a SAM 2 variant (sam2-tiny|small|base|large)")
    pds.add_argument("--instance-points", type=int, default=16,
                    help="SAM 2 prompt grid per side; cost is quadratic in this")
    pds.add_argument("--scale-model", default="auto",
                     help="fitted structural scale: auto, off, a number, or a path")
    pds.add_argument("--resume", action="store_true",
                     help="keep scenes already finished in --out and skip them")
    pds.add_argument("--deliver", action="store_true",
                     help="also run phases 3 and 4 into each scene's own directory")
    pds.add_argument("--tile", type=int, default=512)
    pds.add_argument("--bits", type=int, default=12)
    pds.add_argument("--obj-tol", type=float, default=2.0,
                     help="adaptive mesh tolerance, metres; 0 uses --obj-stride")
    pds.add_argument("--obj-stride", type=int, default=2)
    pds.add_argument("--dual-branch", action="store_true",
                     help="experimental H2 calibration (README 5.5)")
    pds.add_argument("--progress", default="auto", choices=["auto", "rich", "plain", "none"])
    pds.set_defaults(func=cmd_dataset)

    psrv = sub.add_parser("serve", help="web service: upload an image, get a 3D reconstruction")
    psrv.add_argument("--host", default="127.0.0.1")
    psrv.add_argument("--port", type=int, default=8000)
    psrv.add_argument("--jobs", default="out/jobs", help="where uploads and results live")
    psrv.add_argument("--concurrency", type=int, default=1,
                      help="reconstructions at a time; each wants several cores")
    psrv.set_defaults(func=cmd_serve)

    ppre = sub.add_parser("preflight",
                          help="run the whole pipeline end to end on one device and verdict it")
    ppre.add_argument("--backbone", default="dav2-vits",
                      help="the backbone whose device path is being verified")
    ppre.add_argument("--size", type=int, default=384)
    ppre.add_argument("--chip", type=int, default=384)
    ppre.add_argument("--batch", type=int, default=0)
    ppre.add_argument("--bootstrap", type=int, default=4)
    ppre.add_argument("--workers", type=int, default=0)
    ppre.set_defaults(func=cmd_preflight)

    pdoc = sub.add_parser("doctor", help="check this machine is ready, and how fast it is")
    pdoc.add_argument("--load", default=None,
                      help="comma-separated backbones to actually load and time")
    pdoc.set_defaults(func=cmd_doctor)

    return p


def main(argv: Optional[list] = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
