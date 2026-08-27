"""Pixel encodings that carry float rasters into a browser losslessly enough.

A browser cannot read a float32 GeoTIFF. It can read a PNG, so elevation has to
be packed into 8-bit channels, and the packing is the whole correctness problem
of Phase 3: get it wrong and the viewer shows a plausible surface that is not
the one the pipeline produced.

Two encodings, and picking between them is not a style choice.

``terrain_rgb``  the Mapbox convention, ``h = -10000 + v * 0.1`` over 24 bits.
                 Fixed 0.1 m step, absolute, and readable by every existing
                 terrain viewer. This is the interop format and the DSM uses it.

``linear``       ``v = (a - vmin) / (vmax - vmin)`` over the same 24 bits, with
                 the range carried in the manifest. Full precision regardless of
                 how small the range is.

**Why both.** Phase 2 on this benchmark produces an nDSM whose entire range is
0.28 m (see the README's scale-field collapse). A 0.1 m fixed step quantises
that into three levels, so a terrain-RGB nDSM layer would render as flat
terraces and the viewer would be showing a quantisation artifact rather than the
measurement. Derived layers therefore use ``linear``, which spends all 24 bits
on whatever range the layer actually has.

Both encodings are big-endian ``R * 65536 + G * 256 + B`` so the JavaScript side
is one expression and cannot disagree about byte order.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Mapbox Terrain-RGB constants. Do not change: other tools assume them.
TERRAIN_BASE_M = -10000.0
TERRAIN_STEP_M = 0.1
MAX_CODE = 256 ** 3 - 1          # 16777215


def _pack(v: np.ndarray) -> np.ndarray:
    """Split a 24-bit unsigned integer field into R, G, B planes."""
    v = np.clip(np.rint(v), 0, MAX_CODE).astype(np.uint32)
    rgb = np.empty(v.shape + (3,), np.uint8)
    rgb[..., 0] = (v >> 16) & 0xFF
    rgb[..., 1] = (v >> 8) & 0xFF
    rgb[..., 2] = v & 0xFF
    return rgb


def _unpack(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb).astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def encode_terrain_rgb(dsm_m: np.ndarray, nodata_value: float = TERRAIN_BASE_M) -> np.ndarray:
    """Elevation in metres -> (H, W, 3) uint8, Mapbox Terrain-RGB.

    Non-finite pixels are written at the base of the range, which decodes to
    -10000 m. That is not a plausible elevation anywhere on Earth, so the viewer
    can detect it as nodata instead of us inventing a separate mask channel.

    Clamping matters more than it looks: an unclamped pack would wrap a 40000 m
    value around to a small one and produce a confidently wrong surface. The
    clamp saturates instead, which is visible.
    """
    a = np.asarray(dsm_m, np.float64)
    v = (np.where(np.isfinite(a), a, nodata_value) - TERRAIN_BASE_M) / TERRAIN_STEP_M
    return _pack(v)


def decode_terrain_rgb(rgb: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_terrain_rgb`, in metres."""
    return (TERRAIN_BASE_M + _unpack(rgb).astype(np.float64) * TERRAIN_STEP_M).astype(np.float32)


def encode_linear(arr: np.ndarray, vmin: Optional[float] = None,
                  vmax: Optional[float] = None) -> tuple[np.ndarray, float, float]:
    """Any raster -> (rgb, vmin, vmax), 24-bit linear over the given range.

    Returns the range alongside the pixels because the decode is meaningless
    without it; the caller is expected to put both in the manifest. A degenerate
    range is widened rather than divided by, so a genuinely constant layer
    encodes as a flat mid-grey instead of a NaN field.
    """
    a = np.asarray(arr, np.float64)
    finite = np.isfinite(a)
    if vmin is None:
        vmin = float(a[finite].min()) if finite.any() else 0.0
    if vmax is None:
        vmax = float(a[finite].max()) if finite.any() else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    v = (np.where(finite, a, vmin) - vmin) / (vmax - vmin) * MAX_CODE
    return _pack(v), float(vmin), float(vmax)


def decode_linear(rgb: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Inverse of :func:`encode_linear`."""
    return (vmin + _unpack(rgb).astype(np.float64) / MAX_CODE * (vmax - vmin)).astype(np.float32)


def encode_linear_bits(arr: np.ndarray, vmin: float, vmax: float,
                       bits: int = 24) -> np.ndarray:
    """Linear encoding that keeps only the top `bits` of the 24-bit code.

    The decode is unchanged - `decode_linear` with the same vmin/vmax still
    reads it - so this costs the viewer nothing and is purely a payload choice.
    Zeroing the low bits is what a narrower field would really store, and it is
    what lets PNG collapse them: the delivery benchmark measures 12 bits taking
    the linear layers from 6.22 MB to 1.46 MB, a 76% saving, while still
    resolving every layer to better than 0.1% of its own range.

    Rounding in *value* space and re-encoding is not equivalent and was tried
    first: floating-point jitter keeps the low byte noisy, and 12 bits came out
    to a larger file than 16.
    """
    bits = int(max(1, min(24, bits)))
    span = max(float(vmax) - float(vmin), 1e-12)
    a = np.asarray(arr, np.float64)
    finite = np.isfinite(a)
    norm = np.clip((np.where(finite, a, vmin) - vmin) / span, 0.0, 1.0)
    if bits >= 24:
        return _pack(norm * MAX_CODE)
    levels = (1 << bits) - 1
    code = np.rint(norm * levels).astype(np.uint64) << np.uint64(24 - bits)
    return _pack(np.minimum(code, MAX_CODE).astype(np.float64))


def linear_step(vmin: float, vmax: float, bits: int = 24) -> float:
    """Metres per representable level at `bits`. Goes into the manifest."""
    bits = int(max(1, min(24, bits)))
    return float((float(vmax) - float(vmin)) / ((1 << bits) - 1))


def linear_range_for_bits(vmin: float, vmax: float, bits: int = 24) -> tuple:
    """The (vmin, vmax) to record in the manifest for a `bits`-quantised layer.

    Zeroing the low bits leaves the largest code at ``levels << shift``, which is
    slightly under the 2^24 - 1 that a plain decode divides by - so the top of
    the range would come back 0.024% low at 12 bits, a systematic compression
    toward vmin rather than a rounding error. Widening the recorded vmax by
    exactly that ratio makes the *unchanged* decoder exact again, so the only
    error left is the half-step one the caller asked for.

    This is why the range lives in the manifest at all: the encoder gets to
    choose it, and the viewer never has to know that quantisation happened.
    """
    bits = int(max(1, min(24, bits)))
    if bits >= 24:
        return float(vmin), float(vmax)
    span = float(vmax) - float(vmin)
    top = ((1 << bits) - 1) << (24 - bits)          # largest code we can emit
    return float(vmin), float(vmin) + span * MAX_CODE / top


def quantisation_step(vmin: float, vmax: float) -> float:
    """Metres per code in a linear encoding. Reported in the manifest.

    A layer whose step is coarser than its own uncertainty is being shown at a
    resolution it does not have, and the viewer should say so rather than draw
    smooth-looking terraces.
    """
    return float((vmax - vmin) / MAX_CODE)


def normal_map(dsm_m: np.ndarray, gsd_m: float, exaggeration: float = 1.0) -> np.ndarray:
    """Surface normals as (H, W, 3) uint8, the usual (n + 1) / 2 encoding.

    Precomputed here rather than derived per frame in the shader for one
    concrete reason: the viewer displaces vertices from a *decimated* height
    grid, so shading derived from those vertices would lose exactly the detail
    the LOD dropped. Normals computed at full resolution and sampled as a
    texture keep the fine structure visible at every LOD.

    `exaggeration` bakes a vertical scale in. Left at 1.0 the normals are
    physically correct; the viewer's exaggeration slider re-derives them on the
    fly, so this argument exists for static exports (OBJ, screenshots) only.
    """
    a = np.asarray(dsm_m, np.float32)
    a = np.where(np.isfinite(a), a, np.nanmin(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0)
    g = float(max(gsd_m, 1e-9))
    # np.gradient returns d/drow, d/dcol; +row is south, so the north-facing
    # component carries a sign flip to reach a right-handed (east, north, up).
    dz_drow, dz_dcol = np.gradient(a.astype(np.float64) * float(exaggeration), g)
    nx = -dz_dcol
    ny = dz_drow
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    out = np.empty(a.shape + (3,), np.uint8)
    for i, c in enumerate((nx, ny, nz)):
        out[..., i] = np.clip(np.rint((c + 1.0) * 0.5 * 255.0), 0, 255).astype(np.uint8)
    return out


def decode_normal_map(rgb: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normal_map`, as float in [-1, 1]."""
    return (np.asarray(rgb, np.float32) / 255.0) * 2.0 - 1.0
