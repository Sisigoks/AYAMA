"""UNNAT command line.

    python -m unnat.cli info   data/sample.tif
    python -m unnat.cli depth  data/sample.tif --out out/depth.tif
    python -m unnat.cli backbones

Phase 2 adds `run` (full pipeline with calibration and metrics).
"""
from __future__ import annotations

import argparse
import os
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

    model = get_backbone(args.backbone, device=args.device)
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
        tags={"UNNAT_STAGE": "relative_depth", "UNNAT_BACKBONE": depth.backbone,
              "UNNAT_UNITS": "none"},
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


def cmd_synth(args) -> int:
    """Write a synthetic town with a known DSM, so every stage has test data."""
    from .dsm.cog import write_cog, write_png_preview, write_rgb
    from .eval.synthetic_scene import make_scene

    t0 = time.time()
    sc = make_scene(size=args.size, gsd_m=args.gsd, seed=args.seed,
                    sun_azimuth_deg=args.sun_az, sun_elevation_deg=args.sun_el)
    stem = os.path.splitext(args.out)[0]
    write_rgb(args.out, sc.rgb, sc.meta, tags={"UNNAT_STAGE": "synthetic_rgb"})
    write_cog(f"{stem}_dsm.tif", sc.dsm_m, sc.meta, description="ground-truth DSM (m)",
              tags={"UNNAT_STAGE": "reference_dsm", "UNNAT_UNITS": "m"})
    write_cog(f"{stem}_dtm.tif", sc.dtm_m, sc.meta, description="ground-truth DTM (m)")
    write_cog(f"{stem}_sem.tif", sc.sem.astype(np.float32), sc.meta, dtype="uint8",
              nodata=255, description="semantic class ids")
    write_png_preview(f"{stem}_dsm.png", sc.dsm_m, cmap="terrain")
    write_png_preview(f"{stem}_shadow.png", sc.shadow.astype(np.float32), cmap="gray",
                      vmin=0, vmax=1)
    rel = sc.dsm_m - sc.dtm_m
    print(f"wrote     {args.out} and {stem}_[dsm|dtm|sem].tif")
    print(f"scene     {args.size} x {args.size} @ {args.gsd} m   "
          f"elev {sc.dsm_m.min():.1f}-{sc.dsm_m.max():.1f} m   "
          f"max object height {rel.max():.1f} m   shadow {sc.shadow.mean()*100:.1f}%")
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


def _stage_printer():
    """Render the pipeline as it executes. The pipeline is the technical story."""
    seen = {}

    def emit(ev):
        mark = {"running": "..", "done": "OK", "failed": "!!", "skipped": "--"}.get(ev.status, "  ")
        if ev.status == "running" and ev.stage in seen:
            sys.stdout.write(f"\r  {mark} {ev.stage:<14} {ev.detail[:60]:<60}")
            sys.stdout.flush()
            return
        if ev.status == "running":
            seen[ev.stage] = True
            sys.stdout.write(f"  {mark} {ev.stage:<14} {ev.detail[:60]}")
            sys.stdout.flush()
            return
        sys.stdout.write(f"\r  {mark} {ev.stage:<14} {ev.detail[:70]:<70}\n")
        sys.stdout.flush()

    return emit


def cmd_run(args) -> int:
    from .api.pipeline import run
    from .core.types import Config
    from .eval.metrics import format_table

    cfg = Config(
        backbone=args.backbone, chip=args.chip, overlap=args.overlap,
        dem_source=args.dem, reference=args.ref, gcp_file=args.gcps,
        n_bootstrap=args.bootstrap, lattice_stride=args.stride,
        lam_a=args.lam, lam_b=args.lam,
        extras={"device": args.device, "batch_size": args.batch,
                "segmentation": "raster" if args.sem else "heuristic",
                "segmentation_path": args.sem},
    )
    t0 = time.time()
    print(f"UNNAT run   {os.path.basename(args.image)}")
    res = run(args.image, cfg, gcps=_load_gcps(args.gcps), out_dir=args.out,
              on_event=_stage_printer())

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
        overlap=args.overlap, device=args.device, dtype=args.dtype,
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
                 extras={"device": args.device, "batch_size": args.batch})

    t0 = time.time()
    scene = ingest(args.image)
    print(f"scene     {scene.shape[1]} x {scene.shape[0]}   {scene.meta.describe()}")

    model = get_backbone(args.backbone, device=args.device)
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


def cmd_study(args) -> int:
    """Run the whole study and write reproducible results to disk."""
    from .eval import study as S
    from .eval.bench import format_bench, sweep

    out = args.out
    os.makedirs(out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    print(f"UNNAT study -> {out}/")
    env = S.environment()
    S.save_json(env, os.path.join(out, "environment.json"))
    print(f"  {env.get('gpu') or 'CPU only'}   torch {env.get('torch')}   "
          f"{env['platform']}")

    scenes, ablations = [], {}
    sun, lam = [], []

    def checkpoint(bench_block=None, note=""):
        """Write whatever is finished. A later crash then costs a stage, not the run."""
        payload = {
            "environment": env,
            "config": {"backbone": args.backbone, "size": args.size, "chip": args.chip,
                       "seeds": seeds, "bootstrap": args.bootstrap, "gsd_m": 0.5},
            "aggregate": S.aggregate(scenes) if scenes else {},
            "scenes": scenes, "ablation": ablations,
            "sun_sweep": sun, "lambda_sweep": lam,
            "bench": bench_block or {"environment": env, "source": "pending", "results": []},
            "wall_s": round(time.time() - t0, 1),
        }
        S.save_json(payload, os.path.join(out, "study.json"))
        if scenes:
            _write_study_markdown(payload, os.path.join(out, "README.md"))
        if note:
            print(f"  checkpointed {out}/study.json {note}")
        return payload

    for seed in seeds:
        s = S.scene_experiment(out, seed, size=args.size, backbone=args.backbone,
                               chip=args.chip, device=args.device, batch=args.batch,
                               n_bootstrap=args.bootstrap, log=print)
        scenes.append(s)
        ablations[str(seed)] = S.ablation_experiment(
            os.path.join(out, f"seed{seed}"), seed, backbone=args.backbone,
            chip=args.chip, n_bootstrap=args.bootstrap, log=print)
        S.release_backbone()
        checkpoint(note=f"({len(scenes)}/{len(seeds)} scenes)")

    print("\nsun elevation sweep (shadow physics, no inference)")
    sun = S.sun_sweep(size=args.sun_size, log=print)
    checkpoint()

    print("\nlambda sensitivity (cached depth)")
    lam = S.lambda_sweep(os.path.join(out, f"seed{seeds[0]}"), log=print)
    checkpoint(note="before the throughput sweep")

    print("\nthroughput")
    bench = sweep(image=os.path.join(out, f"seed{seeds[0]}", "scene.tif"),
                  backbones=args.backbone.split(","),
                  chips=[int(c) for c in args.chips.split(",")],
                  batches=[int(b) for b in args.batches.split(",")],
                  device=args.device, on_case=lambda b, c, n: print(
                      f"  {b} chip={c} batch={n} ..."))
    print()
    print(format_bench(bench))

    study = checkpoint(bench)
    agg = study["aggregate"]

    print(f"\n{'':=<70}")
    print(f"MAE      {agg['mae_m']['mean']:.2f} +/- {agg['mae_m']['std']:.2f} m "
          f"over {agg['n_scenes']} scenes")
    print(f"baseline {agg['baseline_mae_m']['mean']:.2f} +/- "
          f"{agg['baseline_mae_m']['std']:.2f} m (global affine)")
    print(f"wrote    {out}/study.json and {out}/README.md")
    print(f"total    {time.time() - t0:.0f}s")
    return 0


def _write_study_markdown(study: dict, path: str) -> None:
    a = study["aggregate"]
    env = study["environment"]
    cfg = study["config"]

    def pm(key):
        v = a.get(key)
        return "-" if not v else f"{v['mean']:.2f} ± {v['std']:.2f}"

    lines = [
        "# UNNAT results",
        "",
        f"Generated {env['timestamp_utc']} by `python -m unnat.cli study`.",
        "",
        "Every number here is measured against synthetic scenes whose exact DSM is",
        "known, using only the RGB image plus a simulated public DEM as input. They",
        "test the method end to end; they are **not** a claim about real satellite",
        "imagery, which needs a scene with a reference DSM we do not yet have.",
        "",
        "## Environment",
        "",
        f"- {env['platform']}, python {env['python']}, {env['cpu_count']} cpus",
        f"- torch {env.get('torch')} · CUDA available: {env.get('cuda_available')}"
        + (f" · {env.get('gpu')} ({env.get('vram_total_gb')} GB)" if env.get("gpu") else ""),
        f"- rasterio {env.get('rasterio')} · GDAL {env.get('gdal')}",
        f"- backbone `{cfg['backbone']}`, {cfg['size']}x{cfg['size']} px at "
        f"{cfg['gsd_m']} m, chip {cfg['chip']}, {len(cfg['seeds'])} scenes",
        "",
        "## Headline",
        "",
        "| metric | AGMC | global affine | DEM alone (floor) |",
        "|---|---|---|---|",
        f"| MAE (m) | **{pm('mae_m')}** | {pm('baseline_mae_m')} | {pm('dem_mae_m')} |",
        f"| RMSE (m) | {pm('rmse_m')} | {pm('baseline_rmse_m')} | {pm('dem_rmse_m')} |",
        f"| Pearson r | **{pm('pearson_r')}** | {pm('baseline_pearson_r')} | {pm('dem_pearson_r')} |",
        f"| Spearman rho | {pm('spearman_r')} | - | - |",
        f"| bias (m) | {pm('bias_m')} | - | - |",
        f"| 1σ coverage | {pm('coverage_1s')} | - | - |",
        f"| ECE (m) | {pm('ece_m')} | - | - |",
        f"| δ < 1.25 | {pm('delta1')} | - | - |",
        f"| edge F1 | {pm('edge_f1')} | - | - |",
        f"| slope MAE (deg) | {pm('slope_mae_deg')} | - | - |",
        "",
        "1σ coverage is the honest test of the uncertainty field: for a Gaussian it",
        "should sit near 0.68. A σ that does not predict error is decoration.",
        "",
        "**Read the third column first.** `DEM alone` is the public DEM resampled onto",
        "the image grid, with no depth model involved at all. If the method does not",
        "clear that floor by a clear margin on more than one metric, the depth model is",
        "contributing little and the other two columns are measuring a DEM interpolator.",
        "",
        "`edge F1` and `δ < 1.25` are the rows that expose this. Both describe structure:",
        "edge F1 asks whether height discontinuities land in the right place, and δ is a",
        "ratio of heights above ground. A surface that reproduces terrain and flattens every",
        "building scores well on MAE and collapses on both of those.",
        "",
        "## Error by class",
        "",
        "| class | MAE (m) |",
        "|---|---|",
    ]
    for name, v in sorted(a.get("by_class_mae_m", {}).items(), key=lambda kv: kv[1]["mean"]):
        lines.append(f"| {name} | {v['mean']:.2f} ± {v['std']:.2f} |")
    lines += [
        "",
        "Terrain is close to solved; buildings and canopy dominate the error. That is",
        "where a monocular method fails and saying so is the point of this table.",
        "",
        "## Ablation",
        "",
        "One inference per scene, every variant re-solving only the calibration, so",
        "each row sees the identical depth field.",
        "",
        "| variant | MAE (m) | RMSE (m) | r | anchors |",
        "|---|---|---|---|---|",
    ]
    by_variant: dict = {}
    for rows in study["ablation"].values():
        for r in rows:
            if "mae_m" in r:
                by_variant.setdefault(r["variant"], []).append(r)
    for variant, rows in by_variant.items():
        mae = np.mean([r["mae_m"] for r in rows])
        rmse = np.mean([r["rmse_m"] for r in rows])
        rr = np.mean([r["pearson_r"] for r in rows])
        n = int(np.mean([r["n_anchors"] for r in rows]))
        lines.append(f"| `{variant}` | {mae:.2f} | {rmse:.2f} | {rr:.3f} | {n} |")

    lines += [
        "",
        "## Shadow physics window",
        "",
        "Shadow-derived building height against sun elevation, with no depth model",
        "involved. The usable band the literature quotes is roughly 20-75 degrees.",
        "",
        "| sun elev (deg) | shadow F1 | anchors | median height error (m) | mean weight |",
        "|---|---|---|---|---|",
    ]
    def cell(v, spec=".2f"):
        """A missing measurement is None here, not NaN.

        Strict JSON has no NaN token, so anything non-finite comes back from
        study.json as null. Formatting has to accept both, or regenerating the
        report from a saved study crashes on the sun elevations where no anchor
        survived — which are exactly the rows that carry the finding.
        """
        if v is None:
            return "-"
        try:
            return "-" if not np.isfinite(float(v)) else format(float(v), spec)
        except (TypeError, ValueError):
            return "-"

    for r in study["sun_sweep"]:
        lines.append(
            f"| {cell(r.get('sun_elevation_deg'), '.0f')} | {cell(r.get('f1'))} | "
            f"{r.get('n_anchors', '-')} | {cell(r.get('median_abs_height_error_m'))} | "
            f"{cell(r.get('mean_anchor_weight'))} |"
        )

    lines += [
        "",
        "## Calibration parameter sensitivity",
        "",
        "AGMC has one free parameter, the smoothness weight. If it needed tuning per",
        "scene it would not be a method, it would be a knob.",
        "",
        "| lambda | MAE (m) | r |",
        "|---|---|---|",
    ]
    for r in study["lambda_sweep"]:
        lam = "baseline" if r["lam"] is None else f"{r['lam']}"
        lines.append(f"| {lam} | {r['mae_m']:.2f} | {r['pearson_r']:.3f} |")

    lines += [
        "",
        "## Throughput",
        "",
        "| backbone | chip | batch | chips | wall (s) | s/chip | MPix/s | peak VRAM (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in study["bench"]["results"]:
        if "error" in r:
            lines.append(f"| {r['backbone']} | {r['chip']} | {r['batch_size']} | "
                         f"{r['error']} | | | | |")
            continue
        vram = r.get("peak_vram_mb")
        lines.append(
            f"| {r['backbone']} | {r['chip']} | {r['batch_size']} | {r['n_chips']} | "
            f"{r['wall_s']:.2f} | {r['s_per_chip']:.3f} | {r['mpix_per_s']:.2f} | "
            f"{'-' if vram is None else f'{vram:.0f}'} |")

    lines += [
        "",
        "## Mean stage timings",
        "",
        "| stage | seconds |",
        "|---|---|",
    ]
    for k, v in a.get("timings_s", {}).items():
        lines.append(f"| {k} | {v:.1f} |")

    lines += [
        "",
        "## Reproducing this",
        "",
        "```bash",
        "bash scripts/setup_gpu.sh          # or scripts/setup.ps1 on Windows",
        f"python -m unnat.cli study --backbone {cfg['backbone']} "
        f"--size {cfg['size']} --seeds {','.join(str(s) for s in cfg['seeds'])} --out results",
        "```",
        "",
        f"Took {study['wall_s']:.0f}s on the machine above. Every artifact is a COG that",
        "opens in QGIS without UNNAT installed.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def cmd_doctor(args) -> int:
    """Is this machine ready to run UNNAT, and how fast will it be."""
    from .eval.bench import device_report

    rep = device_report()
    print("UNNAT doctor\n")
    print(f"  platform        {rep['platform']}")
    print(f"  python          {rep['python']}   cpus {rep['cpu_count']}")
    print(f"  torch           {rep.get('torch') or 'NOT INSTALLED'}")
    if rep.get("cuda_available"):
        print(f"  gpu             {rep['gpu']}   {rep['vram_total_gb']} GB "
              f"({rep['vram_free_gb']} GB free)   cc {rep['gpu_capability']}")
        print(f"  cuda            {rep['cuda']}")
    else:
        print("  gpu             none detected - inference will run on CPU")
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
        print("  transformers    NOT INSTALLED - only --backbone synthetic will run")
        ok = False

    if args.load:
        from .depth.backbones import get_backbone

        for name in args.load.split(","):
            try:
                t0 = time.time()
                m = get_backbone(name, device=args.device).load()
                print(f"  loaded          {m.describe()}   {time.time() - t0:.1f}s   {m.stats()}")
            except Exception as exc:
                print(f"  FAILED          {name}: {type(exc).__name__}: {exc}")
                ok = False

    print("\n  " + ("ready" if ok else "not ready - see the lines above"))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("unnat", description="Metric elevation from a single image.")
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
    pd.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pd.add_argument("--chip", type=int, default=1024)
    pd.add_argument("--overlap", type=float, default=0.25)
    pd.add_argument("--max-side", type=int, default=0,
                    help="downscale the longest side before inference (0 = off)")
    pd.add_argument("--preview", action="store_true", help="also write quick-look PNGs")
    pd.add_argument("--cmap", default="magma")
    pd.set_defaults(func=cmd_depth)

    ps = sub.add_parser("synth", help="generate a synthetic town with a known DSM")
    ps.add_argument("--out", default="data/sample.tif")
    ps.add_argument("--size", type=int, default=1024)
    ps.add_argument("--gsd", type=float, default=0.5)
    ps.add_argument("--seed", type=int, default=7)
    ps.add_argument("--sun-az", type=float, default=138.4)
    ps.add_argument("--sun-el", type=float, default=61.2)
    ps.set_defaults(func=cmd_synth)

    pr = sub.add_parser("run", help="full pipeline: depth -> anchors -> metric DSM (Phase 2)")
    pr.add_argument("image")
    pr.add_argument("--out", default="out/run", help="artifact directory")
    pr.add_argument("--backbone", default="dav2-vits")
    pr.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    pr.add_argument("--batch", type=int, default=0,
                    help="chips per forward pass; 0 = pick from free VRAM")
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
    pr.add_argument("--json", default=None, help="write the run summary here")
    pr.set_defaults(func=cmd_run)

    pbn = sub.add_parser("bench", help="throughput sweep over backbones, chip and batch sizes")
    pbn.add_argument("--image", default=None, help="real image; omit for a synthetic scene")
    pbn.add_argument("--size", type=int, default=1024, help="synthetic scene side")
    pbn.add_argument("--backbones", default="dav2-vits")
    pbn.add_argument("--chips", default="512,1024")
    pbn.add_argument("--batches", default="1")
    pbn.add_argument("--overlap", type=float, default=0.25)
    pbn.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    pbn.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16"])
    pbn.add_argument("--repeats", type=int, default=1)
    pbn.add_argument("--json", default="out/bench.json")
    pbn.set_defaults(func=cmd_bench)

    pa = sub.add_parser("ablate", help="one inference, every calibration variant, one table")
    pa.add_argument("image")
    pa.add_argument("--ref", required=True, help="reference DSM (required)")
    pa.add_argument("--dem", default=None)
    pa.add_argument("--sem", default=None)
    pa.add_argument("--backbone", default="dav2-vits")
    pa.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    pa.add_argument("--batch", type=int, default=0)
    pa.add_argument("--chip", type=int, default=1024)
    pa.add_argument("--overlap", type=float, default=0.25)
    pa.add_argument("--bootstrap", type=int, default=16)
    pa.add_argument("--stride", type=int, default=32)
    pa.add_argument("--lam", type=float, default=1.0)
    pa.add_argument("--variants", default=None, help="comma-separated subset")
    pa.add_argument("--json", default="out/ablation.json")
    pa.set_defaults(func=cmd_ablate)

    pst = sub.add_parser("study", help="run the whole study and write reproducible results")
    pst.add_argument("--out", default="results")
    pst.add_argument("--seeds", default="7,21,33", help="one independent scene per seed")
    pst.add_argument("--size", type=int, default=1024)
    pst.add_argument("--chip", type=int, default=512)
    pst.add_argument("--backbone", default="dav2-vits")
    pst.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    pst.add_argument("--batch", type=int, default=1)
    pst.add_argument("--bootstrap", type=int, default=16)
    pst.add_argument("--chips", default="512,1024", help="bench sweep chip sizes")
    pst.add_argument("--batches", default="1", help="bench sweep batch sizes")
    pst.add_argument("--sun-size", type=int, default=512,
                     help="scene size for the sun elevation sweep")
    pst.set_defaults(func=cmd_study)

    pdoc = sub.add_parser("doctor", help="check this machine is ready, and how fast it is")
    pdoc.add_argument("--load", default=None,
                      help="comma-separated backbones to actually load and time")
    pdoc.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
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
