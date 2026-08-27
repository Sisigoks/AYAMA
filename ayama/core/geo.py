"""Geospatial helpers: metric GSD, sun vectors, pixel/world transforms.

The one that matters: a degree is not a metre. If the raster CRS is geographic,
``transform.a`` is in degrees and feeding it straight into shadow trigonometry
collapses every height by a factor of ~10^5. ``gsd_metres`` is the only place
allowed to answer "how many metres is one pixel".
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# WGS84 local scale factors, good to <0.5% away from the poles.
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON_EQ = 111_320.0


def m_per_degree(lat_deg: float) -> tuple[float, float]:
    """Metres per degree of (longitude, latitude) at a given latitude."""
    return _M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg)), _M_PER_DEG_LAT


def gsd_metres(
    transform: Optional[Sequence[float]],
    crs_is_geographic: bool,
    centre_lat_deg: Optional[float] = None,
) -> tuple[float, bool]:
    """Return (gsd_m, is_assumed).

    ``transform`` is the 6-tuple affine (a, b, c, d, e, f) where |a| is the
    x pixel size and |e| the y pixel size, in CRS units.
    """
    if transform is None:
        return 1.0, True
    px = abs(float(transform[0]))
    py = abs(float(transform[4]))
    if px == 0.0 or py == 0.0:
        return 1.0, True
    if crs_is_geographic:
        lat = 0.0 if centre_lat_deg is None else centre_lat_deg
        mx, my = m_per_degree(lat)
        px, py = px * mx, py * my
    # Non-square pixels are rare in nadir products; take the geometric mean and
    # let the caller know nothing was assumed.
    return float(math.sqrt(px * py)), False


def sun_vector(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Unit vector pointing *toward* the sun, in raster axes.

    Returns ``(d_col, d_row, d_z)`` where +col is east, +row is south (image
    convention) and +z is up. Azimuth is measured clockwise from north.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    d_col = math.cos(el) * math.sin(az)     # east
    d_row = -math.cos(el) * math.cos(az)    # north is -row
    d_z = math.sin(el)
    return d_col, d_row, d_z


def shadow_height(shadow_length_m: float, sun_elevation_deg: float) -> float:
    """h = L * tan(sun elevation). The whole of the shadow physics, in one line."""
    return shadow_length_m * math.tan(math.radians(sun_elevation_deg))


def shadow_length(height_m: float, sun_elevation_deg: float) -> float:
    return height_m / max(math.tan(math.radians(sun_elevation_deg)), 1e-6)


def pixel_to_world(transform: Sequence[float], row: float, col: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform[:6]
    x = a * (col + 0.5) + b * (row + 0.5) + c
    y = d * (col + 0.5) + e * (row + 0.5) + f
    return x, y


def world_to_pixel(transform: Sequence[float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform[:6]
    det = a * e - b * d
    if abs(det) < 1e-18:
        raise ValueError("degenerate affine transform")
    dx, dy = x - c, y - f
    col = (e * dx - b * dy) / det - 0.5
    row = (-d * dx + a * dy) / det - 0.5
    return row, col


def percentile_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Map any-bit-depth imagery into uint8 without destroying contrast.

    Raw 11/12/16-bit DN values fed straight to a network trained on 8-bit sRGB
    produce garbage depth, so every non-uint8 raster goes through here.
    """
    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    if not valid.any():
        return np.zeros(a.shape, np.uint8)
    out = np.empty(a.shape, np.uint8)
    # Per-band stretch keeps colour balance sane across sensors.
    bands = a.shape[2] if a.ndim == 3 else 1
    for k in range(bands):
        band = a[..., k] if a.ndim == 3 else a
        v = band[np.isfinite(band)]
        if v.size == 0:
            lo_v, hi_v = 0.0, 1.0
        else:
            lo_v, hi_v = np.percentile(v, [lo, hi])
        if hi_v - lo_v < 1e-6:
            hi_v = lo_v + 1.0
        scaled = np.clip((band - lo_v) / (hi_v - lo_v), 0, 1) * 255.0
        if a.ndim == 3:
            out[..., k] = scaled.astype(np.uint8)
        else:
            out = scaled.astype(np.uint8)
    return out
