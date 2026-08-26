"""Ablation runner: which parts of the method actually earn their place.

The expensive stage (depth inference) runs once. Every variant then re-solves
only the calibration, which is seconds, so a full table costs one inference
instead of eight. That is not just a speed trick - it is the correct
comparison, because every variant then sees exactly the same depth field and
the difference between rows is the thing being ablated and nothing else.

Variants, in the order a reviewer will want them:

  dem_only           the public DEM resampled onto the image grid, with no
                     depth model involved at all. The floor: if the method does
                     not beat this, the depth model is contributing nothing and
                     every other row is measuring the DEM.
  global_affine      one a, b for the whole tile. The baseline everyone else
                     publishes against.
  agmc_no_gate       AGMC, but DEM anchors taken everywhere including rooftops.
                     Isolates the semantic gate.
  agmc_no_shadow     AGMC without shadow-derived height anchors.
  agmc_no_water      AGMC without the water flatness constraint.
  agmc               everything on.
  agmc_bootstrap     everything on, plus the bootstrap sigma field, so ECE and
                     coverage can be reported.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from ..chhaya.agmc import apply_calibration, global_affine, solve_agmc
from ..chhaya.anchors import harvest_dem, harvest_gcp, harvest_shadow, harvest_water
from ..chhaya.uncertainty import bootstrap_sigma, combine, reference_sigma
from ..core.types import Config, Tier
from ..dsm.assemble import assemble
from ..eval.metrics import evaluate

VARIANTS = ("dem_only", "global_affine", "agmc_no_gate", "agmc_no_shadow",
            "agmc_no_water", "agmc", "agmc_bootstrap")


def _anchors_for(variant: str, scene, depth, sem, shadow, dem_m, gcps, cfg):
    """Build the anchor set a variant is allowed to see."""
    anchors = []
    if dem_m is not None:
        if variant == "agmc_no_gate":
            # Deliberately ungated: sample the DEM everywhere, rooftops included.
            everywhere = np.zeros_like(sem)
            anchors += harvest_dem(dem_m, everywhere, source=str(cfg.dem_source or "unknown"))
        else:
            anchors += harvest_dem(dem_m, sem, source=str(cfg.dem_source or "unknown"))
        if variant != "agmc_no_water":
            anchors += harvest_water(sem, dem_m=dem_m)
    elif variant != "agmc_no_water":
        anchors += harvest_water(sem, dem_m=None)

    if gcps:
        anchors += harvest_gcp(gcps)
    if variant != "agmc_no_shadow":
        anchors += harvest_shadow(scene, sem, shadow)
    return anchors


def run_variants(
    scene,
    depth,
    sem,
    shadow,
    reference: np.ndarray,
    dem_m: Optional[np.ndarray] = None,
    gcps: Optional[Sequence] = None,
    cfg: Optional[Config] = None,
    variants: Sequence[str] = VARIANTS,
    tier: Tier = Tier.A,
    on_variant=None,
) -> list[dict]:
    cfg = cfg or Config()
    gsd = scene.meta.gsd_m
    rows = []

    for i, variant in enumerate(variants):
        if on_variant:
            on_variant(variant, i, len(variants))
        anchors = _anchors_for(variant, scene, depth, sem, shadow, dem_m, gcps, cfg)
        if not anchors and variant != "dem_only":
            rows.append({"variant": variant, "error": "no anchors"})
            continue

        sigma = None
        if variant == "dem_only":
            if dem_m is None:
                rows.append({"variant": variant, "error": "no DEM supplied"})
                continue
            surface = np.asarray(dem_m, np.float32)
            used, rejected, resid = 0, 0, float("nan")
        elif variant == "global_affine":
            a, b = global_affine(depth.relative, anchors, cfg.huber_delta)
            surface = (a * depth.relative + b).astype(np.float32)
            used, rejected, resid = len(anchors), 0, float("nan")
        elif variant == "agmc_bootstrap":
            surface, sigma_calib = bootstrap_sigma(depth, anchors, cfg,
                                                   n_boot=cfg.n_bootstrap)
            calib = solve_agmc(depth, anchors, cfg, tier=tier)
            used, rejected, resid = (calib.n_anchors_used, calib.n_anchors_rejected,
                                     calib.residual_rmse)
            from ..api.pipeline import dem_source_name

            sigma = combine(sigma_calib,
                            reference_sigma(surface.shape, dem_source_name(cfg.dem_source),
                                            tier_is_dem=dem_m is not None))
        else:
            calib = solve_agmc(depth, anchors, cfg, tier=tier)
            surface = apply_calibration(depth, calib)
            used, rejected, resid = (calib.n_anchors_used, calib.n_anchors_rejected,
                                     calib.residual_rmse)

        surf = assemble(surface, sem, scene.meta, sigma_m=sigma, tier=tier)
        m = evaluate(surf.dsm_m, reference, sigma=(surf.sigma_m if sigma is not None else None),
                     gsd=gsd, height_pred=surf.ndsm_m)
        rows.append({
            "variant": variant,
            "n_anchors": len(anchors),
            "used": used,
            "rejected": rejected,
            "calib_rmse_m": resid,
            **{k: v for k, v in m.items()},
        })
    return rows


def format_ablation(rows: Sequence[dict]) -> str:
    head = (f"  {'variant':<18}{'anchors':>9}{'MAE m':>9}{'RMSE m':>9}{'bias m':>9}"
            f"{'r':>8}{'rho':>8}{'edge F1':>9}{'1s cov':>8}")
    lines = ["UNNAT ablation", "", head, "  " + "-" * 79]
    best = min((r.get("mae_m", float("inf")) for r in rows if "mae_m" in r), default=None)
    for r in rows:
        if "error" in r:
            lines.append(f"  {r['variant']:<18}   {r['error']}")
            continue
        mark = " *" if best is not None and r.get("mae_m") == best else ""
        cov = r.get("coverage_1s")
        lines.append(
            f"  {r['variant']:<18}{r['n_anchors']:>9}{r['mae_m']:>9.2f}{r['rmse_m']:>9.2f}"
            f"{r['bias_m']:>9.2f}{r['pearson_r']:>8.3f}{r['spearman_r']:>8.3f}"
            f"{r.get('edge_f1', float('nan')):>9.3f}"
            f"{(f'{cov:.2f}' if cov is not None else '-'):>8}{mark}"
        )
    return "\n".join(lines)


def to_markdown(rows: Sequence[dict]) -> str:
    cols = ["variant", "n_anchors", "mae_m", "rmse_m", "bias_m", "pearson_r",
            "spearman_r", "edge_f1", "coverage_1s", "ece_m"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save(rows: Sequence[dict], path: str) -> str:
    from ..core.jsonio import save_json

    save_json(list(rows), path)
    md = os.path.splitext(path)[0] + ".md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(rows) + "\n")
    return path
