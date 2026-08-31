"""Does snapping footprints to image edges place the walls better? Measured.

    python scripts/bench_footprint.py --scenes bern geneva lausanne zurich

This is the diagnostic behind README section 6.2. The structural mesh puts a
wall wherever a footprint boundary is, so the boundary is the single most
consequential thing the reference image could improve - and the standard way to
improve it is guided-filter matting: filter the mask indicator with the image as
guide and re-threshold, so the boundary migrates onto the image edge.

It is scored against lidar rather than by eye, and by boundary F-score rather
than by IoU. IoU rewards getting the bulk of a footprint right and is nearly
blind to an outline that is two pixels out, which is exactly the quantity a snap
moves. The boundary F-score compares outlines directly.

Reports the delta per scene and the mean. Writes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STRUCTURE_M = 2.5


def load(run: str, scene: str):
    import rasterio

    from traksha.semantics import instances as I

    def rd(path, shape=None):
        with rasterio.open(path) as ds:
            arr = ds.read(1, out_shape=shape) if shape else ds.read(1)
            return arr.astype(np.float32)

    dsm = rd(os.path.join(run, "dsm.tif"))
    h, w = dsm.shape
    ndsm = rd(os.path.join(run, "ndsm.tif"))
    sem = rd(os.path.join(run, "sem.tif")).astype(np.uint8)
    with rasterio.open(os.path.join(run, "texture.jpg")) as ds:
        rgb = np.transpose(ds.read(out_shape=(3, h, w)), (1, 2, 0)).astype(np.uint8)

    truth_dir = os.path.join(ROOT, "data", "real", scene)
    built = (rd(os.path.join(truth_dir, f"{scene}_dsm.tif"), (h, w))
             - rd(os.path.join(truth_dir, f"{scene}_dtm.tif"), (h, w))) > STRUCTURE_M
    return dsm, ndsm, sem, rgb, built, I.load(os.path.join(run, "segmentation"))


def footprints(inst, dsm, ndsm, sem):
    """The footprints the structural builder would actually extrude."""
    from traksha.mesh import structural as S

    dtm = dsm - ndsm
    out = []
    for rec in getattr(inst, "records", []):
        if rec.get("visible_px", 0) < S.MIN_AREA_PX:
            continue
        for part in S.pieces(S.unpinch(S._to_cells(inst.mask(rec["id"])))):
            px = S._cells_to_px(part, dsm.shape)
            if S.measure(0, px, dsm, dtm, sem) is not None:
                out.append(px)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["bern", "geneva", "lausanne", "zurich"])
    ap.add_argument("--runs", nargs="*", default=[],
                    help="run directory per scene; defaults to results/<scene>")
    ap.add_argument("--bands", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--tol", type=int, default=2, help="boundary tolerance in pixels")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from traksha.mesh import footprint as FP

    rows = []
    for i, scene in enumerate(args.scenes):
        run = args.runs[i] if i < len(args.runs) else os.path.join(ROOT, "results", scene)
        if not os.path.isdir(os.path.join(run, "segmentation")):
            print(f"skip {scene}: {run} has no segmentation/ - run `traksha run` "
                  "with instances enabled first")
            continue
        dsm, ndsm, sem, rgb, built, inst = load(run, scene)
        parts = footprints(inst, dsm, ndsm, sem)
        base = np.zeros(dsm.shape, bool)
        for m in parts:
            base |= m
        before = FP.boundary_f1(base, built, args.tol)["f1"]
        print(f"\n{scene}: {len(parts)} footprints, boundary F1@{args.tol}px "
              f"{before:.4f}")
        for band in args.bands:
            cur = np.zeros(dsm.shape, bool)
            for m in parts:
                cur |= FP.snap(m, rgb, band_px=band)
            after = FP.boundary_f1(cur, built, args.tol)["f1"]
            rows.append({"scene": scene, "band": band, "before": round(before, 4),
                         "after": round(after, 4), "delta": round(after - before, 4)})
            print(f"   band {band}px -> {after:.4f}   {after - before:+.4f}")

    if rows:
        print(f"\n{'band':>6s} {'mean delta':>12s} {'helps on':>10s}")
        for band in args.bands:
            d = [r["delta"] for r in rows if r["band"] == band]
            if not d:
                continue
            print(f"{band:6d} {np.mean(d):+12.4f} {sum(x > 0 for x in d):6d}/{len(d)}")
        print("\nA delta that is flat in the band width is not a tuning problem: the "
              "\nfilter finds the same edge however much freedom it is given.")
    if args.out and rows:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
