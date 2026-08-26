"""A synthetic town with a known DSM, for building and testing before real data.

This gives every workstream something to work against on hour zero: an RGB
image with real cast shadows, the exact DSM that produced them, a semantic
mask, and correct geospatial metadata. Metrics computed against it are a test
of the plumbing, never a result to report.

The shadows are ray-marched from the DSM using the same geometry the physics
module inverts, so `harvest_shadow` can be validated against ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.geo import sun_vector
from ..core.types import BARE_GROUND, BUILDING, ROAD, VEGETATION, WATER, SceneMeta

try:
    from scipy.ndimage import gaussian_filter, map_coordinates

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


@dataclass
class SyntheticScene:
    rgb: np.ndarray        # (H, W, 3) uint8
    dsm_m: np.ndarray      # (H, W) float32, absolute elevation in metres
    dtm_m: np.ndarray      # (H, W) float32, bare-earth terrain
    sem: np.ndarray        # (H, W) uint8, class ids
    shadow: np.ndarray     # (H, W) bool, cast-shadow mask
    meta: SceneMeta

    @property
    def ndsm_m(self) -> np.ndarray:
        return (self.dsm_m - self.dtm_m).astype(np.float32)


def _fbm(shape, rng, octaves=5, base=4) -> np.ndarray:
    """Fractal value noise in [0, 1]; the terrain generator."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * (2 ** o)
        coarse = rng.random((max(2, n), max(2, n))).astype(np.float32)
        yi = np.linspace(0, coarse.shape[0] - 1, h)
        xi = np.linspace(0, coarse.shape[1] - 1, w)
        if HAVE_SCIPY:
            grid = map_coordinates(coarse, np.meshgrid(yi, xi, indexing="ij"), order=1, mode="reflect")
        else:
            grid = coarse[np.clip(yi.astype(int), 0, coarse.shape[0] - 1)][
                :, np.clip(xi.astype(int), 0, coarse.shape[1] - 1)]
        out += amp * grid
        total += amp
        amp *= 0.5
    return out / max(total, 1e-6)


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(a, sigma) if HAVE_SCIPY else a


def cast_shadow_mask(dsm: np.ndarray, sun_az: float, sun_el: float, gsd: float,
                     n_steps: int = 96) -> np.ndarray:
    """March each pixel toward the sun; shadowed if the surface blocks the ray."""
    d_col, d_row, d_z = sun_vector(sun_az, sun_el)
    h, w = dsm.shape
    rr, cc = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32),
                         indexing="ij")
    occluded = np.zeros(dsm.shape, bool)
    for i in range(1, n_steps + 1):
        t_px = float(i)                      # step one pixel at a time
        t_m = t_px * gsd
        sr = rr + d_row * t_px
        sc = cc + d_col * t_px
        if HAVE_SCIPY:
            surf = map_coordinates(dsm, [sr, sc], order=1, mode="nearest")
        else:
            surf = dsm[np.clip(sr.astype(int), 0, h - 1), np.clip(sc.astype(int), 0, w - 1)]
        ray_z = dsm + d_z * t_m
        occluded |= surf > ray_z + 1e-3
    return occluded


def make_scene(
    size: int = 1024,
    gsd_m: float = 0.5,
    seed: int = 7,
    sun_azimuth_deg: float = 138.4,
    sun_elevation_deg: float = 61.2,
    crs: str = "EPSG:32644",
    origin_xy: tuple = (300000.0, 1990000.0),
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    h = w = int(size)

    # --- terrain: gentle relief plus a valley ------------------------------
    relief = _fbm((h, w), rng, octaves=5, base=3)
    relief = _smooth(relief, size / 64.0)
    base_elev = 400.0
    dtm = base_elev + 45.0 * (relief - relief.mean())

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    river_c = 0.30 * w + 0.10 * w * np.sin(2 * math.pi * yy / (1.6 * h))
    river_d = np.abs(xx - river_c)
    river_w = 0.035 * w
    valley = np.clip(1.0 - river_d / (4 * river_w), 0, 1) ** 2
    dtm = dtm - 12.0 * valley

    sem = np.full((h, w), BARE_GROUND, np.uint8)
    dsm = dtm.copy()

    # --- water -------------------------------------------------------------
    water = river_d < river_w
    if water.any():
        # A real water surface is flat along its length; approximate with a
        # gentle downstream gradient so the flatness constraint has something
        # to bite on.
        surface = np.percentile(dtm[water], 25) - 0.5
        dsm[water] = surface
        dtm[water] = surface
        sem[water] = WATER

    # --- roads: a grid, avoiding the river ---------------------------------
    road_spacing = max(64, size // 8)
    road_half = max(2, int(round(4.0 / gsd_m)))
    road = np.zeros((h, w), bool)
    for c in range(road_spacing // 2, w, road_spacing):
        road[:, max(0, c - road_half):c + road_half] = True
    for r in range(road_spacing // 2, h, road_spacing):
        road[max(0, r - road_half):r + road_half, :] = True
    road &= ~water
    sem[road] = ROAD

    # --- buildings: blocks between the roads -------------------------------
    heights = []
    margin = road_half + 3
    for r0 in range(road_spacing // 2 + margin, h - road_spacing // 4, road_spacing):
        for c0 in range(road_spacing // 2 + margin, w - road_spacing // 4, road_spacing):
            n_per_block = int(rng.integers(1, 4))
            for _ in range(n_per_block):
                bh = int(rng.integers(int(0.15 * road_spacing), int(0.45 * road_spacing)))
                bw = int(rng.integers(int(0.15 * road_spacing), int(0.45 * road_spacing)))
                rr0 = r0 + int(rng.integers(0, max(1, road_spacing - bh - 2 * margin)))
                cc0 = c0 + int(rng.integers(0, max(1, road_spacing - bw - 2 * margin)))
                rr1, cc1 = min(h, rr0 + bh), min(w, cc0 + bw)
                if rr1 - rr0 < 3 or cc1 - cc0 < 3:
                    continue
                patch = np.s_[rr0:rr1, cc0:cc1]
                if water[patch].any() or road[patch].any() or (sem[patch] == BUILDING).any():
                    continue
                height = float(rng.choice([1, 1, 1, 2, 3], p=[.3, .25, .2, .15, .1])) * \
                    float(rng.uniform(4.0, 14.0))
                sem[patch] = BUILDING
                dsm[patch] = dtm[patch].mean() + height
                heights.append(height)

    # --- vegetation: round canopies ----------------------------------------
    n_trees = int(0.00012 * h * w)
    for _ in range(n_trees):
        r0 = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w))
        rad = float(rng.uniform(2.0, 6.0)) / gsd_m
        rr0, rr1 = max(0, int(r0 - rad)), min(h, int(r0 + rad) + 1)
        cc0, cc1 = max(0, int(c0 - rad)), min(w, int(c0 + rad) + 1)
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        sub = np.s_[rr0:rr1, cc0:cc1]
        if (sem[sub] == BUILDING).any() or water[sub].any():
            continue
        ly, lx = np.mgrid[rr0:rr1, cc0:cc1]
        d = np.hypot(ly - r0, lx - c0) / max(rad, 1e-3)
        canopy = d <= 1.0
        crown = float(rng.uniform(4.0, 12.0)) * np.sqrt(np.clip(1.0 - d ** 2, 0, 1))
        target = dtm[sub] + crown
        dsm[sub] = np.where(canopy, np.maximum(dsm[sub], target), dsm[sub])
        m = sem[sub]
        m[canopy & (m != BUILDING)] = VEGETATION
        sem[sub] = m

    # Smoothing rounds off roof edges, but it also drags the surface below the
    # terrain on steep banks, so clamp afterwards: a DSM never sits under its DTM.
    dsm = np.maximum(_smooth(dsm.astype(np.float32), 0.6), dtm.astype(np.float32))

    # --- shadows and colour -------------------------------------------------
    shadow = cast_shadow_mask(dsm, sun_azimuth_deg, sun_elevation_deg, gsd_m)
    rgb = _colourise(sem, dsm, shadow, rng, sun_azimuth_deg, sun_elevation_deg)

    meta = SceneMeta(
        crs=crs,
        transform=(gsd_m, 0.0, origin_xy[0], 0.0, -gsd_m, origin_xy[1]),
        gsd_m=gsd_m,
        bounds_wgs=None,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        off_nadir_deg=0.0,
        acquired_utc="2024-03-21T06:30:00",
        source="synthetic",
        gsd_is_assumed=False,
    )
    return SyntheticScene(rgb=rgb, dsm_m=dsm.astype(np.float32), dtm_m=dtm.astype(np.float32),
                          sem=sem, shadow=shadow, meta=meta)


_PALETTE = {
    BARE_GROUND: (168, 152, 122),
    ROAD: (105, 105, 110),
    BUILDING: (188, 176, 168),
    VEGETATION: (74, 112, 58),
    WATER: (58, 88, 122),
}


def _colourise(sem, dsm, shadow, rng, sun_az: float, sun_el: float) -> np.ndarray:
    h, w = sem.shape
    rgb = np.zeros((h, w, 3), np.float32)
    for cls, colour in _PALETTE.items():
        m = sem == cls
        if not m.any():
            continue
        rgb[m] = np.asarray(colour, np.float32)

    # Per-roof albedo variation, so buildings are not one flat grey.
    roofs = sem == BUILDING
    if roofs.any():
        tint = 0.75 + 0.5 * _fbm((h, w), rng, octaves=2, base=24)
        rgb[roofs] *= tint[roofs][:, None]

    # Texture, then diffuse shading from the surface normal.
    noise = rng.normal(0.0, 6.0, (h, w, 1)).astype(np.float32)
    rgb += noise

    d_col, d_row, d_z = sun_vector(sun_az, sun_el)
    gy, gx = np.gradient(dsm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lam = np.clip(n[..., 0] * d_col + n[..., 1] * d_row + n[..., 2] * d_z, 0, 1)
    rgb *= (0.72 + 0.38 * lam)[..., None]

    # A shadowed surface loses direct sunlight but keeps skylight, so it goes
    # dark AND blue. Darkening uniformly would render a scene in which the
    # chromatic shadow cue does not exist, and would test the detector against
    # a world that is not the one it has to work in.
    rgb[shadow] *= np.array([0.36, 0.42, 0.58], np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)
