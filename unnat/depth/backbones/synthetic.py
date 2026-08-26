"""A backbone that needs no weights, no torch, and no network.

This is NOT a depth estimator and must never appear in a results table. It
exists so that ingest -> tiling -> blending -> calibration -> mesh -> viewer can
be built, tested and demoed end to end while the real backbone is still
downloading, and so CI can run the whole pipeline in two seconds.

Anything it produces is labelled 'synthetic' in the artifact metadata.
"""
from __future__ import annotations

import numpy as np

from .base import DepthBackbone

try:
    from scipy.ndimage import gaussian_filter, uniform_filter

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def _box(a: np.ndarray, k: int) -> np.ndarray:
    """Separable box filter with edge padding, via a summed-area trick."""
    if k < 2:
        return a
    pad = k // 2
    out = a
    for axis in (0, 1):
        p = np.pad(out, [(pad, pad) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge")
        c = np.cumsum(p, axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        lo = np.take(c, range(0, out.shape[axis]), axis=axis)
        hi = np.take(c, range(k, out.shape[axis] + k), axis=axis)
        out = (hi - lo) / float(k)
    return out


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if HAVE_SCIPY:
        return gaussian_filter(a, sigma)
    # Three box passes approximate a Gaussian to within a few percent.
    k = max(1, int(round(sigma * 2)) | 1)
    return _box(_box(_box(a, k), k), k)


class SyntheticBackbone(DepthBackbone):
    name = "synthetic"
    native = None

    def infer(self, patch: np.ndarray) -> np.ndarray:
        rgb = patch.astype(np.float32) / 255.0
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

        # Structures read as locally bright and locally textured; smooth ground
        # reads as neither. Combining the two gives a surface with plausible
        # low-frequency relief and building-shaped bumps.
        base = _blur(lum, 24.0)
        detail = lum - _blur(lum, 6.0)
        if HAVE_SCIPY:
            texture = np.sqrt(np.maximum(uniform_filter(detail ** 2, 9), 0.0))
        else:
            texture = np.abs(detail)
        field = 0.55 * base + 0.45 * np.clip(texture * 6.0, 0, 1)
        return _blur(field, 2.0).astype(np.float32)

    def describe(self) -> str:
        return "synthetic (no weights; structural placeholder, not an estimator)"
