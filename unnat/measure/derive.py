"""Structural measurements derived from a DSM.

Everything here is a pure function of rasters already on disk, so the same code
serves the metrics module, the measure panel in the viewer, and the API.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import percentile_filter, uniform_filter

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def slope_deg(dsm: np.ndarray, gsd: float) -> np.ndarray:
    """Terrain slope in degrees. gsd in metres, so the gradient is dimensionless."""
    gy, gx = np.gradient(np.asarray(dsm, np.float32), float(gsd))
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def aspect_deg(dsm: np.ndarray, gsd: float) -> np.ndarray:
    """Downslope direction, degrees clockwise from north."""
    gy, gx = np.gradient(np.asarray(dsm, np.float32), float(gsd))
    return ((np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0).astype(np.float32)


def roughness(dsm: np.ndarray, win: int = 9) -> np.ndarray:
    """Local standard deviation of the detrended surface, in metres."""
    a = np.asarray(dsm, np.float32)
    if not HAVE_SCIPY:
        raise RuntimeError("roughness needs scipy")
    smooth = uniform_filter(a, win)
    var = uniform_filter((a - smooth) ** 2, win)
    return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def tri(dsm: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index: mean absolute difference to the 8 neighbours."""
    a = np.asarray(dsm, np.float32)
    s = np.zeros_like(a)
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        s += np.abs(a - np.roll(np.roll(a, dy, 0), dx, 1))
    return (s / 8.0).astype(np.float32)


def prominence(dsm: np.ndarray, radius_px: int = 200) -> np.ndarray:
    """How far a feature stands above the low envelope of its surroundings."""
    if not HAVE_SCIPY:
        raise RuntimeError("prominence needs scipy")
    a = np.asarray(dsm, np.float32)
    return (a - percentile_filter(a, 10, size=int(radius_px))).astype(np.float32)


def profile(dsm: np.ndarray, sigma: Optional[np.ndarray], p0, p1, gsd: float, n: int = 512):
    """Cross-section between two pixel coordinates.

    Returns (distance_m, elevation_m, sigma_m). The confidence band is what
    makes this read as a survey instrument instead of a chart.
    """
    from scipy.ndimage import map_coordinates

    rr = np.linspace(float(p0[0]), float(p1[0]), int(n))
    cc = np.linspace(float(p0[1]), float(p1[1]), int(n))
    z = map_coordinates(np.asarray(dsm, np.float32), [rr, cc], order=1, mode="nearest")
    if sigma is None:
        e = np.zeros_like(z)
    else:
        e = map_coordinates(np.asarray(sigma, np.float32), [rr, cc], order=1, mode="nearest")
    length_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    d = np.linspace(0.0, length_px * float(gsd), int(n))
    return d, z.astype(np.float32), e.astype(np.float32)
