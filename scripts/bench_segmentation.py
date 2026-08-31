"""Which SAM 2 model, and which thresholds? Measured against lidar.

    python scripts/bench_segmentation.py --scenes bern geneva --variants sam2-tiny sam2-base

Total mask coverage is the wrong thing to optimise. SAM 2 is class-agnostic: it
segments roads, courtyards and shadows as readily as roofs, so a generator that
covers the whole image scores well and tells the pipeline nothing. What matters
is whether masks land on *structure*, so this scores against the lidar nDSM that
already provides ground truth for the study:

    built recall     of the pixels carrying more than 2.5 m of real structure,
                     how many does some mask cover
    precision        of the pixels the masks cover, how many carry structure

Precision near the scene's built fraction means the masks have no preference for
buildings at all, which is the expected and correct behaviour for a
class-agnostic segmenter - the building decision belongs downstream, where
height exists. Recall is the number this stage is actually responsible for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STRUCTURE_M = 2.5          # above this, a pixel carries something built


def load_scene(name: str, side: int):
    """Orthophoto plus the lidar height above ground, on one grid."""
    import rasterio

    tex = os.path.join(ROOT, "results", name, "texture.jpg")
    dsm = os.path.join(ROOT, "data", "real", name, f"{name}_dsm.tif")
    dtm = os.path.join(ROOT, "data", "real", name, f"{name}_dtm.tif")
    for p in (tex, dsm, dtm):
        if not os.path.exists(p):
            return None, None
    with rasterio.open(tex) as ds:
        rgb = np.transpose(ds.read(out_shape=(3, side, side)), (1, 2, 0)).astype(np.uint8)
    with rasterio.open(dsm) as ds:
        top = ds.read(1, out_shape=(side, side)).astype(np.float32)
    with rasterio.open(dtm) as ds:
        ground = ds.read(1, out_shape=(side, side)).astype(np.float32)
    return rgb, np.maximum(top - ground, 0.0) > STRUCTURE_M


def score(masks, built) -> dict:
    cov = np.zeros(built.shape, bool)
    for m in masks:
        cov |= m.segmentation
    hit = float((cov & built).sum())
    return {
        "masks": len(masks),
        "scene_coverage": round(100 * float(cov.mean()), 1),
        "built_recall": round(100 * hit / max(float(built.sum()), 1.0), 1),
        "precision": round(100 * hit / max(float(cov.sum()), 1.0), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["bern"])
    ap.add_argument("--variants", nargs="+", default=["sam2-tiny", "sam2-small", "sam2-base"])
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--stability", type=float, default=0.75)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from traksha.semantics.sam2 import Sam2Segmenter, Sam2Unavailable

    rows = []
    for name in args.scenes:
        rgb, built = load_scene(name, args.side)
        if rgb is None:
            print(f"skip {name}: needs results/{name}/texture.jpg and its lidar pair")
            continue
        print(f"\n{name}: {100 * built.mean():.1f}% of the scene is built")
        for variant in args.variants:
            seg = Sam2Segmenter(variant=variant, points_per_side=args.points,
                                points_per_batch=10, pred_iou_thresh=args.iou,
                                stability_score_thresh=args.stability, min_area_px=48)
            try:
                t0 = time.time()
                seg.load()
                load_s = time.time() - t0
                params = sum(p.numel() for p in seg._model.parameters()) / 1e6
                t0 = time.time()
                masks = seg.generate(rgb)
                gen_s = time.time() - t0
            except Sam2Unavailable as exc:
                print(f"  {variant:12s} unavailable: {exc}")
                continue
            finally:
                seg.unload()
            row = {"scene": name, "variant": variant, "params_M": round(params, 1),
                   "load_s": round(load_s, 1), "generate_s": round(gen_s, 1),
                   **score(masks, built)}
            rows.append(row)
            print(f"  {variant:12s} {row['params_M']:5.1f}M  gen {row['generate_s']:5.1f}s  "
                  f"masks {row['masks']:4d}  recall {row['built_recall']:5.1f}%  "
                  f"precision {row['precision']:5.1f}%")

    if rows:
        print(f"\n{'variant':12s} {'scenes':>7s} {'gen s':>7s} {'recall':>8s} {'precision':>10s}")
        for variant in args.variants:
            got = [r for r in rows if r["variant"] == variant]
            if not got:
                continue
            print(f"{variant:12s} {len(got):7d} "
                  f"{np.mean([r['generate_s'] for r in got]):7.1f} "
                  f"{np.mean([r['built_recall'] for r in got]):7.1f}% "
                  f"{np.mean([r['precision'] for r in got]):9.1f}%")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
