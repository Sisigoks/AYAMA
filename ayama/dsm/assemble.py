"""Turn a calibrated surface into the delivered products: DSM, DTM, nDSM.

The decomposition is the point. DSM = DTM + nDSM, and the two branches are
anchored by different sources: a public DEM approximates bare earth and says
nothing about a 40-storey tower, while shadow trigonometry gives height above
local ground and says nothing about terrain. Keeping them apart is what stops
each source inheriting the other's error.

The DTM here is extracted, not predicted: ground-classified pixels are taken at
face value and the surface is carried under buildings and canopy from the
nearest ground. That is the classic morphological approach and it is honest
about what it can do - it will under-estimate terrain inside a very large
building footprint, because no evidence of the ground there exists in the image.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import DEM_ADMISSIBLE, ElevationSurface, SceneMeta, Tier


def fill_holes(arr: np.ndarray, mask_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Nearest-neighbour fill of non-finite pixels. Occlusion shadows, mostly."""
    from scipy.ndimage import distance_transform_edt

    a = np.asarray(arr, np.float32).copy()
    valid = np.isfinite(a) if mask_valid is None else (np.isfinite(a) & np.asarray(mask_valid, bool))
    if valid.all():
        return a
    if not valid.any():
        return np.zeros_like(a)
    _, idx = distance_transform_edt(~valid, return_indices=True)
    return a[tuple(idx)].astype(np.float32)


def extract_dtm(
    dsm_m: np.ndarray,
    sem: np.ndarray,
    gsd_m: float,
    smooth_m: float = 30.0,
) -> np.ndarray:
    """Bare-earth surface: keep ground pixels, carry them under everything else."""
    from scipy.ndimage import gaussian_filter

    dsm = np.asarray(dsm_m, np.float32)
    ground = np.isin(np.asarray(sem), DEM_ADMISSIBLE) & np.isfinite(dsm)
    if not ground.any():
        # Nothing classified as ground: fall back to a low envelope of the DSM.
        from scipy.ndimage import percentile_filter

        win = max(3, int(round(smooth_m / max(gsd_m, 1e-6))) | 1)
        return gaussian_filter(percentile_filter(dsm, 5, size=win), win / 4.0).astype(np.float32)

    carried = fill_holes(np.where(ground, dsm, np.nan).astype(np.float32))
    sigma_px = max(1.0, smooth_m / max(gsd_m, 1e-6) / 3.0)
    dtm = gaussian_filter(carried, sigma_px)
    # Never let the smoothed terrain rise above measured ground.
    return np.minimum(dtm, np.where(ground, dsm, np.inf)).astype(np.float32)


def assemble(
    surface_m: np.ndarray,
    sem: np.ndarray,
    meta: SceneMeta,
    sigma_m: Optional[np.ndarray] = None,
    tier: Tier = Tier.C,
    dtm_m: Optional[np.ndarray] = None,
) -> ElevationSurface:
    """DSM + nDSM + sigma, hole-filled, ready to write."""
    dsm = fill_holes(np.asarray(surface_m, np.float32))
    dtm = np.asarray(dtm_m, np.float32) if dtm_m is not None else extract_dtm(dsm, sem, meta.gsd_m)
    ndsm = np.maximum(dsm - dtm, 0.0).astype(np.float32)
    if sigma_m is None:
        sigma = np.zeros_like(dsm)
    else:
        sigma = fill_holes(np.asarray(sigma_m, np.float32))
    return ElevationSurface(dsm_m=dsm, ndsm_m=ndsm, sigma_m=sigma, meta=meta, tier=tier)
