"""Real datasets: discovery, and running the pipeline over them.

Everything else in this repository evaluates on scenes it generated itself.
That is honest as far as it goes - the ground truth is exact and the renderer is
deterministic - but it cannot say anything about real imagery, and section 2.6
of the README lists that as the top limitation. This module is the path off
synthetic data.

It deliberately does no downloading. Datasets like DFC2019 sit behind a
registration wall, and a pipeline that silently proceeds with data it failed to
fetch is worse than one that stops. Point `--root` at a directory you already
have and this will tell you what it found - or what it expected and did not.

Two layouts are understood:

  us3d      DFC2019 Track 1 / US3D: <TILE>_RGB.tif with <TILE>_AGL.tif
            (above-ground-level, i.e. an nDSM) and optional <TILE>_CLS.tif.
            Reference is a HEIGHT ABOVE GROUND, not an elevation - which is
            exactly the quantity section 4 shows the pipeline getting wrong.

  generic   any directory of <name>.tif images with siblings named
            <name>_dsm.tif / _dem.tif / _sem.tif. Use this for national lidar
            products or anything you have assembled yourself.

Suffixes are configurable because they are the one thing likely to differ
between a dataset's documentation and the copy you actually downloaded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneRef:
    """One real scene: the image, and whatever truth ships alongside it."""

    name: str
    image: str
    reference: Optional[str] = None
    # "dsm" = absolute elevation, "ndsm" = height above ground. US3D ships nDSM,
    # and comparing one against the other silently is a 400 m error.
    reference_kind: str = "dsm"
    dem: Optional[str] = None
    # True bare earth, where a survey-grade one ships alongside the DSM. Not an
    # input - the pipeline anchors to `dem`, the degraded public one - but it is
    # what an nDSM must be measured against, so it is carried, not discarded.
    dtm: Optional[str] = None
    semantics: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [self.name]
        if self.reference:
            bits.append(f"ref:{self.reference_kind}")
        if self.dem:
            bits.append("dem")
        if self.dtm:
            bits.append("dtm")
        if self.semantics:
            bits.append("sem")
        return "  ".join(bits)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
US3D_SUFFIXES = {"image": "_RGB.tif", "reference": "_AGL.tif", "semantics": "_CLS.tif"}
GENERIC_SUFFIXES = {"image": ".tif", "reference": "_dsm.tif",
                    "dem": "_dem.tif", "dtm": "_dtm.tif", "semantics": "_sem.tif"}


def _exists(path: str) -> Optional[str]:
    return path if path and os.path.exists(path) else None


def discover_us3d(root: str, suffixes: Optional[dict] = None,
                  dem_dir: Optional[str] = None) -> list:
    """DFC2019 Track 1 / US3D tiles under `root`.

    The reference is `_AGL.tif`, a height above ground. That is recorded as
    `reference_kind="ndsm"` rather than quietly compared against elevation.
    """
    sfx = dict(US3D_SUFFIXES, **(suffixes or {}))
    out = []
    for base, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(sfx["image"]):
                continue
            stem = f[: -len(sfx["image"])]
            img = os.path.join(base, f)
            out.append(SceneRef(
                name=stem,
                image=img,
                reference=_exists(os.path.join(base, stem + sfx["reference"])),
                reference_kind="ndsm",
                semantics=_exists(os.path.join(base, stem + sfx["semantics"])),
                dem=_exists(os.path.join(dem_dir, stem + "_DEM.tif")) if dem_dir else None,
            ))
    return out


def discover_generic(root: str, suffixes: Optional[dict] = None) -> list:
    """A directory of images with conventionally-named siblings."""
    sfx = dict(GENERIC_SUFFIXES, **(suffixes or {}))
    companions = {v for k, v in sfx.items() if k != "image"}
    out = []
    for base, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(sfx["image"]):
                continue
            if any(f.endswith(c) for c in companions):
                continue                       # a companion, not a scene
            stem = f[: -len(sfx["image"])]
            out.append(SceneRef(
                name=stem,
                image=os.path.join(base, f),
                reference=_exists(os.path.join(base, stem + sfx["reference"])),
                reference_kind="dsm",
                dem=_exists(os.path.join(base, stem + sfx["dem"])),
                dtm=_exists(os.path.join(base, stem + sfx["dtm"])),
                semantics=_exists(os.path.join(base, stem + sfx["semantics"])),
            ))
    return out


LAYOUTS = {"us3d": discover_us3d, "generic": discover_generic}


def discover(root: str, layout: str = "generic", **kw) -> list:
    if layout not in LAYOUTS:
        raise KeyError(f"unknown layout '{layout}'. Known: {', '.join(LAYOUTS)}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root}")
    scenes = LAYOUTS[layout](root, **kw)
    if not scenes:
        expect = US3D_SUFFIXES if layout == "us3d" else GENERIC_SUFFIXES
        raise FileNotFoundError(
            f"no {layout} scenes under {root}.\n"
            f"  expected images ending {expect['image']}"
            f" with siblings ending {expect['reference']}\n"
            "  pass --suffix-image / --suffix-reference if your copy differs"
        )
    return scenes


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def run_scene(ref: SceneRef, out_dir: str, cfg, on_event=None) -> dict:
    """One real scene through the whole pipeline. Returns a metrics record.

    The reference handling is the part that matters. Against a DSM the pipeline
    compares elevation and derives an nDSM internally; against an AGL raster it
    compares the predicted height above ground directly, because that is the
    quantity the dataset actually measured.
    """
    import numpy as np

    from ..api.pipeline import run as run_pipeline
    from ..eval.metrics import evaluate, evaluate_by_class

    cfg.dem_source = ref.dem
    cfg.reference = ref.reference if ref.reference_kind == "dsm" else None
    if ref.semantics:
        cfg.extras["segmentation"] = "raster"
        cfg.extras["segmentation_path"] = ref.semantics

    res = run_pipeline(ref.image, cfg=cfg, out_dir=out_dir,
                       write_artifacts=True, on_event=on_event)

    record = {
        "name": ref.name, "image": os.path.abspath(ref.image),
        "reference_kind": ref.reference_kind,
        "tier": res.tier.value, "tier_reason": res.tier_reason,
        "anchors": res.anchor_counts, "timings_s": res.timings_s,
        "metrics": res.metrics, "metrics_by_class": res.metrics_by_class,
        "baseline_metrics": res.baseline_metrics, "dem_metrics": res.dem_metrics,
    }

    # An absolute-elevation reference makes MAE easy: most of a scene is ground,
    # and ground is what the DEM already knows. The quantity the method actually
    # claims - height above ground - only becomes measurable when a bare-earth
    # DTM ships alongside the DSM. Where one does, it is scored, together with
    # the flat-ground floor. See README section 4 for why that floor is not a
    # formality here.
    if ref.reference_kind == "dsm" and ref.reference and ref.dtm:
        from ..api.pipeline import load_reference
        from ..core.ingest import ingest

        scene = ingest(ref.image)
        dsm_true = load_reference(ref.reference, scene)
        dtm_true = load_reference(ref.dtm, scene)
        ndsm_true = np.maximum(dsm_true - dtm_true, 0.0)
        pred = res.surface.ndsm_m
        record["ndsm_metrics"] = evaluate(pred, ndsm_true, gsd=res.surface.meta.gsd_m,
                                          height_pred=pred, height_ref=ndsm_true)
        record["zero_baseline_metrics"] = evaluate(
            np.zeros_like(pred), ndsm_true, gsd=res.surface.meta.gsd_m)
        obj = ndsm_true > 2.0
        record["relief"] = {
            "object_fraction": float(obj.mean()),
            "true_mean_height_m": float(ndsm_true[obj].mean()) if obj.any() else None,
            "pred_mean_height_m": float(pred[obj].mean()) if obj.any() else None,
            "true_max_height_m": float(ndsm_true.max()),
            "pred_max_height_m": float(pred.max()),
        }

    if ref.reference_kind == "ndsm" and ref.reference:
        from ..api.pipeline import load_reference
        from ..core.ingest import ingest

        scene = ingest(ref.image)
        agl = load_reference(ref.reference, scene)
        pred = res.surface.ndsm_m
        record["metrics"] = evaluate(pred, agl, gsd=res.surface.meta.gsd_m,
                                     height_pred=pred, height_ref=agl)
        record["metrics_by_class"] = evaluate_by_class(
            pred, agl, np.asarray(res.surface.sem if hasattr(res.surface, "sem") else
                                  np.zeros_like(pred, np.int16)),
            gsd=res.surface.meta.gsd_m)
        # The floor baseline for a height-above-ground reference is "predict
        # zero everywhere". A method that cannot beat flat ground on nDSM is
        # not reconstructing anything, and on this pipeline that is a live risk.
        record["zero_baseline_metrics"] = evaluate(
            np.zeros_like(pred), agl, gsd=res.surface.meta.gsd_m)
    return record


def aggregate(records: list) -> dict:
    """Mean and spread across real scenes. Mirrors eval.study.aggregate."""
    import numpy as np

    out = {"n_scenes": len(records)}
    keys = ("mae_m", "rmse_m", "bias_m", "pearson_r", "spearman_r",
            "coverage_1s", "ece_m", "edge_f1", "delta1", "slope_mae_deg")
    for key in keys:
        vals = [r["metrics"].get(key) for r in records if r.get("metrics")]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
    for src in ("baseline_metrics", "dem_metrics", "zero_baseline_metrics",
                "ndsm_metrics"):
        vals = [r[src].get("mae_m") for r in records if r.get(src)]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if vals:
            out[f"{src}_mae_m"] = {"mean": float(np.mean(vals)),
                                   "std": float(np.std(vals))}
    for key in ("true_mean_height_m", "pred_mean_height_m"):
        vals = [r["relief"].get(key) for r in records if r.get("relief")]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out
