"""Reproducible study: run the method, record what it actually did.

Everything here writes to `results/`, which is tracked in git, so the numbers on
the website are the numbers a reviewer can regenerate with one command. Each
entry carries the environment it was measured on and the provenance of every
input, because a metric without those is a rumour.

    python -m unnat.cli study --backbone dav2-vits --out results

Four experiments:

  scenes      the full pipeline on several independent synthetic towns, each
              with exact ground truth, plus the global-affine baseline
  ablation    which parts of the method earn their place, one inference per scene
  sun sweep   how shadow-derived height accuracy varies with sun elevation,
              which is the physics window the whole shadow branch depends on
  lambda      sensitivity to the one free parameter in the calibration
"""
from __future__ import annotations

import os
import platform
import time
from typing import Callable, Optional, Sequence

import numpy as np

from ..api.pipeline import load_dem, run
from ..chhaya.agmc import apply_calibration, global_affine, solve_agmc
from ..chhaya.anchors import harvest_shadow
from ..chhaya.ladder import build_anchors
from ..core.types import Config, DepthField, Scene, Tier
from ..dsm.assemble import assemble
from ..dsm.cog import write_cog, write_png_preview, write_rgb
from ..eval.ablation import run_variants
from ..eval.metrics import evaluate
from ..eval.synthetic_scene import make_scene
from ..semantics.segment import segment
from ..semantics.shadow import detect_shadow

DEFAULT_SEEDS = (7, 21, 33)


def _log(fn: Optional[Callable[[str], None]], msg: str) -> None:
    (fn or print)(msg)


# --------------------------------------------------------------------------
# 1. Full pipeline over several independent scenes
# --------------------------------------------------------------------------
def scene_experiment(
    out_dir: str,
    seed: int,
    size: int = 1024,
    gsd_m: float = 0.5,
    backbone: str = "dav2-vits",
    chip: int = 512,
    device: str = "auto",
    batch: int = 1,
    n_bootstrap: int = 16,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """One scene, generated then reconstructed from its own image alone."""
    scene_dir = os.path.join(out_dir, f"seed{seed}")
    os.makedirs(scene_dir, exist_ok=True)

    _log(log, f"  [seed {seed}] generating {size}x{size} scene")
    sc = make_scene(size=size, gsd_m=gsd_m, seed=seed)
    img = os.path.join(scene_dir, "scene.tif")
    dsm_ref = os.path.join(scene_dir, "scene_dsm.tif")
    dtm_ref = os.path.join(scene_dir, "scene_dtm.tif")
    write_rgb(img, sc.rgb, sc.meta, tags={"UNNAT_STAGE": "synthetic_rgb"})
    write_cog(dsm_ref, sc.dsm_m, sc.meta, description="ground-truth DSM (m)")
    write_cog(dtm_ref, sc.dtm_m, sc.meta, description="ground-truth DTM (m)")

    cfg = Config(backbone=backbone, chip=chip, overlap=0.25,
                 dem_source=f"sim:{dtm_ref}", reference=dsm_ref,
                 n_bootstrap=n_bootstrap, lattice_stride=32,
                 extras={"device": device, "batch_size": batch})

    _log(log, f"  [seed {seed}] running pipeline ({backbone})")
    t0 = time.time()
    res = run(img, cfg, out_dir=os.path.join(scene_dir, "run"))
    wall = time.time() - t0
    release_backbone()
    _log(log, f"  [seed {seed}] MAE {res.metrics['mae_m']:.2f} m  "
              f"baseline {res.baseline_metrics['mae_m']:.2f} m  ({wall:.0f}s)")

    _write_previews(scene_dir, sc, res)

    summary = res.summary()
    summary.update(
        seed=seed, size=size, gsd_m=gsd_m, wall_s=round(wall, 1),
        scene_truth={
            "elev_min_m": float(sc.dsm_m.min()),
            "elev_max_m": float(sc.dsm_m.max()),
            "max_object_height_m": float(sc.ndsm_m.max()),
            "shadow_fraction": float(sc.shadow.mean()),
            "sun_azimuth_deg": sc.meta.sun_azimuth_deg,
            "sun_elevation_deg": sc.meta.sun_elevation_deg,
        },
    )
    return summary


def release_backbone() -> None:
    """Drop model weights and hand the memory back before the next scene.

    On a memory-tight machine the allocator does not raise MemoryError; it
    faults, and a segfault is not catchable. Releasing between scenes is the
    difference between a study that finishes and one that dies at scene three.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_previews(scene_dir: str, sc, res, max_side: int = 768) -> None:
    """Web-sized PNGs. The site reads these directly."""
    d = os.path.join(scene_dir, "preview")
    os.makedirs(d, exist_ok=True)
    from PIL import Image

    im = Image.fromarray(sc.rgb)
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    im.convert("RGB").save(os.path.join(d, "rgb.jpg"), quality=88)

    surf = res.surface
    lo, hi = float(np.percentile(sc.dsm_m, 1)), float(np.percentile(sc.dsm_m, 99))
    write_png_preview(os.path.join(d, "dsm_pred.png"), surf.dsm_m, cmap="terrain",
                      vmin=lo, vmax=hi, max_side=max_side)
    write_png_preview(os.path.join(d, "dsm_truth.png"), sc.dsm_m, cmap="terrain",
                      vmin=lo, vmax=hi, max_side=max_side)
    write_png_preview(os.path.join(d, "ndsm.png"), surf.ndsm_m, cmap="viridis",
                      vmin=0, vmax=30, max_side=max_side)
    write_png_preview(os.path.join(d, "sigma.png"), surf.sigma_m, cmap="magma",
                      max_side=max_side)
    write_png_preview(os.path.join(d, "error.png"), surf.dsm_m - sc.dsm_m, cmap="RdBu_r",
                      vmin=-15, vmax=15, max_side=max_side)
    write_png_preview(os.path.join(d, "shadow.png"), sc.shadow.astype(np.float32),
                      cmap="gray", vmin=0, vmax=1, max_side=max_side)


# --------------------------------------------------------------------------
# 2. Ablation, sharing one inference per scene
# --------------------------------------------------------------------------
def ablation_experiment(
    scene_dir: str,
    seed: int,
    backbone: str = "dav2-vits",
    chip: int = 512,
    n_bootstrap: int = 16,
    log: Optional[Callable[[str], None]] = None,
) -> list:
    """Re-uses the relative depth the pipeline already wrote for this scene."""
    import rasterio

    from ..core.ingest import ingest

    img = os.path.join(scene_dir, "scene.tif")
    rel_path = os.path.join(scene_dir, "run", "relative_depth.tif")
    if not os.path.exists(rel_path):
        return [{"error": "no cached relative depth"}]

    scene = ingest(img)
    with rasterio.open(rel_path) as ds:
        rel = ds.read(1).astype(np.float32)
    with rasterio.open(os.path.join(scene_dir, "scene_dsm.tif")) as ds:
        ref = ds.read(1).astype(np.float32)

    depth = DepthField(relative=rel, meta=scene.meta, backbone=backbone)
    sem, _ = segment(scene)
    shadow = detect_shadow(scene, sem)
    dem_m, _ = load_dem(f"sim:{os.path.join(scene_dir, 'scene_dtm.tif')}", scene)

    cfg = Config(n_bootstrap=n_bootstrap, lattice_stride=32,
                 dem_source=f"sim:{os.path.join(scene_dir, 'scene_dtm.tif')}")
    _log(log, f"  [seed {seed}] ablation over cached depth")
    return run_variants(scene, depth, sem, shadow, ref, dem_m=dem_m, cfg=cfg)


# --------------------------------------------------------------------------
# 3. The shadow physics window
# --------------------------------------------------------------------------
def sun_sweep(
    elevations: Sequence[float] = (15, 20, 25, 30, 40, 50, 60, 70, 75, 80),
    size: int = 512,
    gsd_m: float = 0.5,
    seed: int = 21,
    log: Optional[Callable[[str], None]] = None,
) -> list:
    """Shadow-derived height accuracy against sun elevation.

    Needs no depth inference: it isolates the physics. The literature's claim is
    that shadow length is usable roughly between 20 and 75 degrees, and this
    measures whether our implementation agrees, on scenes where the true height
    of every building is known exactly.
    """
    rows = []
    for el in elevations:
        sc = make_scene(size=size, gsd_m=gsd_m, seed=seed, sun_elevation_deg=float(el))
        scene = Scene(rgb=sc.rgb, meta=sc.meta)
        sem, _ = segment(scene)
        mask = detect_shadow(scene, sem)

        truth = sc.shadow
        tp = float((mask & truth).sum())
        precision = tp / max(float(mask.sum()), 1.0)
        recall = tp / max(float(truth.sum()), 1.0)

        anchors = harvest_shadow(scene, sc.sem, mask)
        if anchors:
            err = np.array([a.value_m - float(sc.ndsm_m[a.row, a.col]) for a in anchors])
            median_abs = float(np.median(np.abs(err)))
            median_bias = float(np.median(err))
            weight = float(np.mean([a.weight for a in anchors]))
        else:
            median_abs = median_bias = weight = float("nan")

        rows.append({
            "sun_elevation_deg": float(el),
            "true_shadow_fraction": float(truth.mean()),
            "detected_fraction": float(mask.mean()),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "n_anchors": len(anchors),
            "median_abs_height_error_m": median_abs,
            "median_bias_m": median_bias,
            "mean_anchor_weight": weight,
        })
        _log(log, f"  sun {el:>4.0f} deg  F1 {rows[-1]['f1']:.2f}  "
                  f"anchors {len(anchors):>3}  medErr {median_abs:.2f} m")
    return rows


# --------------------------------------------------------------------------
# 4. Sensitivity to the one free calibration parameter
# --------------------------------------------------------------------------
def lambda_sweep(
    scene_dir: str,
    lambdas: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 50.0),
    strides: Sequence[int] = (32,),
    log: Optional[Callable[[str], None]] = None,
) -> list:
    """Is the smoothness weight a knob you have to tune, or a scale that works?"""
    import rasterio

    from ..core.ingest import ingest

    img = os.path.join(scene_dir, "scene.tif")
    rel_path = os.path.join(scene_dir, "run", "relative_depth.tif")
    if not os.path.exists(rel_path):
        return []

    scene = ingest(img)
    with rasterio.open(rel_path) as ds:
        rel = ds.read(1).astype(np.float32)
    with rasterio.open(os.path.join(scene_dir, "scene_dsm.tif")) as ds:
        ref = ds.read(1).astype(np.float32)

    depth = DepthField(relative=rel, meta=scene.meta)
    sem, _ = segment(scene)
    shadow = detect_shadow(scene, sem)
    dtm = os.path.join(scene_dir, "scene_dtm.tif")
    dem_m, _ = load_dem(f"sim:{dtm}", scene)
    base_cfg = Config(dem_source=f"sim:{dtm}")
    anchors, _ = build_anchors(scene, depth, sem, shadow, Tier.A, dem_m=dem_m, cfg=base_cfg)

    a, b = global_affine(rel, anchors)
    baseline = evaluate(a * rel + b, ref, gsd=scene.meta.gsd_m)

    rows = [{"lam": None, "stride": None, "variant": "global_affine",
             "mae_m": baseline["mae_m"], "rmse_m": baseline["rmse_m"],
             "pearson_r": baseline["pearson_r"]}]
    for stride in strides:
        for lam in lambdas:
            cfg = Config(lattice_stride=stride, lam_a=lam, lam_b=lam,
                         dem_source=f"sim:{dtm}")
            calib = solve_agmc(depth, anchors, cfg, tier=Tier.A)
            surf = assemble(apply_calibration(depth, calib), sem, scene.meta, tier=Tier.A)
            m = evaluate(surf.dsm_m, ref, gsd=scene.meta.gsd_m)
            rows.append({"lam": float(lam), "stride": int(stride), "variant": "agmc",
                         "mae_m": m["mae_m"], "rmse_m": m["rmse_m"],
                         "pearson_r": m["pearson_r"]})
            _log(log, f"  lam {lam:>6}  stride {stride}  MAE {m['mae_m']:.2f} m")
    return rows


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------
def aggregate(scenes: Sequence[dict]) -> dict:
    """Mean and spread across scenes. One scene is an anecdote."""
    def col(key, src="metrics"):
        vals = [s[src].get(key) for s in scenes if src in s and s[src].get(key) is not None]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        return vals

    out = {"n_scenes": len(scenes)}
    for key in ("mae_m", "rmse_m", "bias_m", "median_ae_m", "p90_ae_m", "pearson_r",
                "spearman_r", "coverage_1s", "ece_m", "delta1", "edge_f1",
                "slope_mae_deg"):
        vals = col(key)
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
    for key in ("mae_m", "rmse_m", "pearson_r"):
        vals = col(key, "baseline_metrics")
        if vals:
            out[f"baseline_{key}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        vals = col(key, "dem_metrics")
        if vals:
            out[f"dem_{key}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    classes = {}
    for s in scenes:
        for name, m in s.get("metrics_by_class", {}).items():
            classes.setdefault(name, []).append(m["mae_m"])
    out["by_class_mae_m"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                             for k, v in classes.items()}

    stages = {}
    for s in scenes:
        for name, t in s.get("timings_s", {}).items():
            stages.setdefault(name, []).append(t)
    out["timings_s"] = {k: float(np.mean(v)) for k, v in stages.items()}
    return out


def environment() -> dict:
    from .bench import device_report

    rep = device_report()
    rep["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rep["machine"] = platform.machine()
    return rep


def save_json(obj, path: str) -> str:
    """Strict JSON: non-finite metrics become null, not the bare `NaN` token
    that Python emits by default and no browser will parse."""
    from ..core.jsonio import save_json as _save

    return _save(obj, path)
