"""Anchor harvesters, one per source.

Every anchor carries a confidence weight derived from its source and its local
conditions, and every harvester is free to return nothing. That is what lets
the tier ladder degrade instead of fail.

The semantic gate is the important part: a public DEM approximates bare earth,
so a DEM sample taken on a rooftop is not a weak anchor, it is a wrong one. It
is rejected before it enters the system rather than down-weighted inside it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.types import (BUILDING, DEM_ADMISSIBLE, WATER, Anchor,
                          GCP, Scene)

# One-sigma vertical accuracy of the public DEMs, from their datasheets. This is
# the term that honestly explains why absolute elevation is less certain than
# relative building height.
DEM_SIGMA_M = {
    "srtm": 6.0,           # ~16 m at 90% confidence
    "copernicus": 3.0,     # GLO-30, 2-4 m
    "aster": 8.5,
    "nasadem": 5.5,
    "unknown": 6.0,
}


def dem_weight(source: str) -> float:
    """Confidence in [0, 1], scaled so a 3 m DEM outweighs a 6 m one."""
    sigma = DEM_SIGMA_M.get(str(source).lower(), DEM_SIGMA_M["unknown"])
    return float(np.clip(3.0 / sigma, 0.1, 1.0))


def harvest_dem(
    dem_m: np.ndarray,
    sem: np.ndarray,
    source: str = "copernicus",
    stride: int = 16,
    weight: float = 0.6,
    slope_mask: Optional[np.ndarray] = None,
) -> list[Anchor]:
    """Sample a bare-earth DEM, but only where the scene is bare earth.

    `dem_m` must already be resampled onto the image grid.
    """
    dem_m = np.asarray(dem_m, np.float32)
    sem = np.asarray(sem)
    admissible = np.isin(sem, DEM_ADMISSIBLE) & np.isfinite(dem_m)
    if slope_mask is not None:
        # Steep ground is where a 30 m DEM posting disagrees most with a 0.5 m
        # image, so drop it rather than fight it.
        admissible &= ~np.asarray(slope_mask, bool)
    if not admissible.any():
        return []

    w = float(np.clip(weight * dem_weight(source), 0.0, 1.0))
    rows, cols = np.nonzero(admissible)
    keep = (rows % stride == 0) & (cols % stride == 0)
    rows, cols = rows[keep], cols[keep]
    return [Anchor(int(r), int(c), float(dem_m[r, c]), "terrain", "dem", w)
            for r, c in zip(rows, cols)]


def harvest_water(
    sem: np.ndarray,
    dem_m: Optional[np.ndarray] = None,
    stride: int = 24,
    weight: float = 0.9,
    min_px: int = 200,
) -> list[Anchor]:
    """Water is flat. A free accuracy win and a striking demo moment.

    Each connected water body gets anchors at one common elevation: the robust
    median of the DEM over that body, or, with no DEM, a set of equal-value
    relative constraints tying the body to its own first pixel.
    """
    sem = np.asarray(sem)
    water = sem == WATER
    if water.sum() < min_px:
        return []

    out: list[Anchor] = []
    for blob in connected_components(water, min_px=min_px):
        rows, cols = blob["rows"], blob["cols"]
        keep = (rows % stride == 0) & (cols % stride == 0)
        rr, cc = rows[keep], cols[keep]
        if rr.size < 2:
            continue
        if dem_m is not None:
            level = float(np.median(np.asarray(dem_m, np.float32)[rows, cols]))
            out += [Anchor(int(r), int(c), level, "terrain", "water", weight)
                    for r, c in zip(rr, cc)]
        else:
            # No datum available: say only that the body is level with itself.
            r0, c0 = int(rr[0]), int(cc[0])
            out += [Anchor(int(r), int(c), 0.0, "terrain", "water", weight,
                           ref_row=r0, ref_col=c0)
                    for r, c in zip(rr[1:], cc[1:])]
    return out


def harvest_gcp(gcps: Sequence[GCP], weight: float = 1.0) -> list[Anchor]:
    """Survey points pin the datum. Highest weight in the system."""
    return [Anchor(int(g.row), int(g.col), float(g.elev_m), "absolute", "gcp", weight)
            for g in gcps]


def assume_ground_plane(depth: np.ndarray, sem: np.ndarray, stride: int = 32,
                        weight: float = 0.2) -> list[Anchor]:
    """Last resort for Tier C: call the low quantile of open ground 'zero'.

    The structure this produces is trustworthy; the datum is arbitrary, and the
    UI must say so.
    """
    depth = np.asarray(depth, np.float32)
    sem = np.asarray(sem)
    ground = np.isin(sem, DEM_ADMISSIBLE)
    if not ground.any():
        ground = np.ones(sem.shape, bool)
    level = float(np.percentile(depth[ground], 20))
    rows, cols = np.nonzero(ground & (depth <= level))
    keep = (rows % stride == 0) & (cols % stride == 0)
    rows, cols = rows[keep], cols[keep]
    return [Anchor(int(r), int(c), 0.0, "terrain", "ground_plane", weight)
            for r, c in zip(rows, cols)]


def connected_components(mask: np.ndarray, min_px: int = 1) -> list[dict]:
    """Label a boolean mask; returns per-blob pixel lists and simple shape stats."""
    from scipy.ndimage import find_objects, label

    lab, n = label(np.asarray(mask, bool))
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        if sub.sum() < min_px:
            continue
        rows, cols = np.nonzero(sub)
        rows = rows + sl[0].start
        cols = cols + sl[1].start
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        out.append({
            "id": i,
            "rows": rows,
            "cols": cols,
            "mask": sub,
            "slice": sl,
            "area_px": int(sub.sum()),
            "centroid": (float(rows.mean()), float(cols.mean())),
            "bbox": (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop),
            "elongation": float(max(h, w) / max(1, min(h, w))),
        })
    return out


# --------------------------------------------------------------------------
# Shadow physics
# --------------------------------------------------------------------------
def harvest_shadow(
    scene: Scene,
    sem: np.ndarray,
    shadow_mask: np.ndarray,
    weight_scale: float = 1.0,
    min_blob_px: int = 60,
    min_height_m: float = 2.0,
    max_height_m: float = 400.0,
    max_gap_px: int = 2,
) -> list[Anchor]:
    """Per-building height from cast-shadow length.

        h = L * tan(sun elevation)

    Two decisions that matter more than the trigonometry.

    First, the anchors are RELATIVE: each one says "this roof stands h metres
    above the ground at the foot of this building", with the reference pixel
    carried alongside. A shadow measures a height, never an elevation, and
    letting it enter as an elevation is how a good height anchor silently
    becomes a bad datum anchor.

    Second, length is measured as the median of many parallel runs along the
    anti-solar direction rather than as one blob dimension. A single run is
    hostage to one occlusion; the median of forty is not.
    """
    import math

    from ..core.geo import sun_vector
    from ..semantics.shadow import quality_from_sun_elevation

    meta = scene.meta
    gate = quality_from_sun_elevation(meta.sun_elevation_deg)
    if gate <= 0.0 or meta.sun_azimuth_deg is None:
        return []

    el = float(meta.sun_elevation_deg)
    tan_el = math.tan(math.radians(el))
    gsd = float(meta.gsd_m)
    d_col, d_row, _ = sun_vector(meta.sun_azimuth_deg, el)
    # Shadows fall away from the sun.
    anti = np.array([-d_row, -d_col], np.float64)
    norm = float(np.hypot(*anti))
    if norm < 1e-6:
        return []
    anti /= norm

    sem = np.asarray(sem)
    shadow = np.asarray(shadow_mask, bool)
    buildings = sem == BUILDING
    if not buildings.any() or not shadow.any():
        return []

    H, W = sem.shape
    max_steps = int(min(max(H, W), (max_height_m / max(tan_el, 1e-6)) / max(gsd, 1e-6)))
    if max_steps < 2:
        return []

    out: list[Anchor] = []
    for blob in connected_components(buildings, min_px=min_blob_px):
        rows, cols = blob["rows"], blob["cols"]
        # Boundary pixels on the shaded side: a step along the anti-solar
        # direction leaves the building.
        nr = np.clip(np.round(rows + anti[0]).astype(int), 0, H - 1)
        nc = np.clip(np.round(cols + anti[1]).astype(int), 0, W - 1)
        edge = ~buildings[nr, nc]
        if edge.sum() < 4:
            continue
        er, ec = rows[edge], cols[edge]

        runs = []
        ref_pixels = []
        for r0, c0 in zip(er, ec):
            length = 0
            gap = 0
            rr, cc = float(r0), float(c0)
            for _ in range(max_steps):
                rr += anti[0]
                cc += anti[1]
                ri, ci = int(round(rr)), int(round(cc))
                if not (0 <= ri < H and 0 <= ci < W) or buildings[ri, ci]:
                    break
                if shadow[ri, ci]:
                    length += 1 + gap
                    gap = 0
                else:
                    gap += 1
                    if gap > max_gap_px:
                        break
            if length >= 2:
                runs.append(length)
                ref_pixels.append((int(round(r0 + anti[0] * 2)), int(round(c0 + anti[1] * 2))))
        if len(runs) < 4:
            continue

        runs_arr = np.asarray(runs, np.float64)
        length_px = float(np.median(runs_arr))
        h = length_px * gsd * tan_el
        if not (min_height_m <= h <= max_height_m):
            continue

        # Consistency of the parallel runs: a crisp isolated shadow gives forty
        # runs of nearly the same length, a contaminated one does not.
        mad = float(np.median(np.abs(runs_arr - length_px)))
        crispness = float(np.clip(1.0 - mad / max(length_px, 1e-6), 0.0, 1.0))
        isolation = blob_isolation_score(buildings, blob)
        w = float(np.clip(gate * crispness * isolation * weight_scale, 0.0, 1.0))
        if w <= 0.02:
            continue

        r_ref, c_ref = ref_pixels[len(ref_pixels) // 2]
        r_ref = int(np.clip(r_ref, 0, H - 1))
        c_ref = int(np.clip(c_ref, 0, W - 1))
        rc, cc_ = blob["centroid"]
        out.append(Anchor(int(round(rc)), int(round(cc_)), float(h), "object", "shadow", w,
                          ref_row=r_ref, ref_col=c_ref))
    return out


def blob_isolation_score(buildings: np.ndarray, blob: dict, ring_px: int = 12) -> float:
    """1.0 for a building standing alone, falling toward 0 in a dense block.

    A shadow cast into a dense block is contaminated by its neighbours' shadows
    and its length means nothing.
    """
    from scipy.ndimage import binary_dilation

    r0, c0, r1, c1 = blob["bbox"]
    H, W = buildings.shape
    rr0, cc0 = max(0, r0 - ring_px), max(0, c0 - ring_px)
    rr1, cc1 = min(H, r1 + ring_px), min(W, c1 + ring_px)
    local = np.zeros((rr1 - rr0, cc1 - cc0), bool)
    local[blob["rows"] - rr0, blob["cols"] - cc0] = True
    ring = binary_dilation(local, np.ones((2 * ring_px + 1, 2 * ring_px + 1), bool)) & ~local
    if not ring.any():
        return 1.0
    others = buildings[rr0:rr1, cc0:cc1] & ring
    return float(np.clip(1.0 - others.sum() / ring.sum(), 0.0, 1.0))
