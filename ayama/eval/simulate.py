"""Simulated auxiliary data, for developing Phase 2 before real DEMs arrive.

A simulated Copernicus GLO-30 is the true terrain degraded the way the real
product is degraded: coarse posting, bilinear upsampling, and correlated noise
at roughly the datasheet's one-sigma accuracy. It is a development harness, not
evidence. Every number produced against it must be labelled 'simulated'.
"""
from __future__ import annotations

import numpy as np

from ..chhaya.anchors import DEM_SIGMA_M


def simulate_public_dem(
    dtm_m: np.ndarray,
    gsd_m: float,
    posting_m: float = 30.0,
    source: str = "copernicus",
    seed: int = 0,
) -> np.ndarray:
    """Degrade a true bare-earth surface into a public-DEM lookalike."""
    from scipy.ndimage import gaussian_filter, zoom

    rng = np.random.default_rng(seed)
    a = np.asarray(dtm_m, np.float32)
    factor = max(1.0, posting_m / max(gsd_m, 1e-6))

    # Coarse posting: average down, then bilinear back up, exactly like the
    # resampling a 30 m product goes through to reach a 0.5 m grid.
    coarse = zoom(a, 1.0 / factor, order=1)
    up = zoom(coarse, np.array(a.shape) / np.array(coarse.shape), order=1)
    if up.shape != a.shape:
        up = up[:a.shape[0], :a.shape[1]]
        up = np.pad(up, ((0, a.shape[0] - up.shape[0]), (0, a.shape[1] - up.shape[1])),
                    mode="edge")

    # Vertical error in a real DEM is spatially correlated, not white.
    sigma = DEM_SIGMA_M.get(source.lower(), DEM_SIGMA_M["unknown"])
    noise = gaussian_filter(rng.normal(0.0, 1.0, a.shape).astype(np.float32), factor)
    noise *= sigma / max(float(noise.std()), 1e-6)
    return (up + noise).astype(np.float32)


def simulate_gcps(dsm_m: np.ndarray, n: int = 6, seed: int = 0, sigma_m: float = 0.05):
    """Survey points: exact elevations at scattered pixels, plus survey noise."""
    from ..core.types import GCP

    rng = np.random.default_rng(seed)
    h, w = dsm_m.shape
    margin = int(0.05 * min(h, w))
    rows = rng.integers(margin, h - margin, n)
    cols = rng.integers(margin, w - margin, n)
    return [GCP(int(r), int(c), float(dsm_m[r, c] + rng.normal(0, sigma_m)), f"gcp{i}")
            for i, (r, c) in enumerate(zip(rows, cols))]
