"""Cast-shadow detection from RGB.

Shadows are the only absolute-scale cue a single nadir image carries for free,
so detecting them well is worth more than it looks. The detector is chromatic
rather than a brightness threshold: a shadowed surface is dark *and* blue-
shifted, because it loses direct sunlight but keeps skylight. Dark asphalt is
dark and not blue-shifted, which is exactly the confusion a plain threshold
makes and this one does not.

Quality gate: sun elevation should sit between about 20 and 75 degrees. Low sun
elongates shadows until the length measurement is dominated by terrain slope;
high sun shortens them below the resolution of the image. Outside the band the
detector still runs but the anchors it feeds are given zero weight, which is
the honest way to say "this image cannot support shadow physics".
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import WATER, Scene

try:
    from scipy.ndimage import binary_closing, binary_opening, label

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def quality_from_sun_elevation(el_deg: Optional[float]) -> float:
    """Confidence multiplier in [0, 1] for anything derived from shadow length."""
    if el_deg is None:
        return 0.0
    el = float(el_deg)
    if el < 20.0 or el > 75.0:
        return 0.0
    rise = np.clip((el - 20.0) / 10.0, 0.0, 1.0)
    fall = np.clip((75.0 - el) / 10.0, 0.0, 1.0)
    return float(rise * fall)


def shadow_index(rgb: np.ndarray) -> np.ndarray:
    """Higher means more shadow-like. Ratio of blue to intensity, darkened."""
    a = np.asarray(rgb, np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Tsai's C3-style chromaticity: blue against the strongest of the other two.
    c3 = np.arctan2(b, np.maximum(np.maximum(r, g), 1e-6))
    c3 = (c3 - c3.min()) / max(float(np.ptp(c3)), 1e-6)
    return (c3 * (1.0 - lum)).astype(np.float32)


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(values.mean())
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom <= 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float((edges[:-1] + edges[1:])[np.argmax(sigma_b)] / 2)


def detect_shadow(
    scene: Scene,
    sem: Optional[np.ndarray] = None,
    min_px: int = 30,
    darkness_pct: float = 30.0,
    max_fraction: float = 0.35,
) -> np.ndarray:
    """Boolean cast-shadow mask.

    A shadowed surface is blue-shifted *and* dark. Thresholding the chromatic
    index alone flags every blue-leaning pixel in the scene: on the sample
    benchmark that was 42-46% of the image at a precision of 0.08-0.15. Adding
    the darkness term takes it to precision 0.95-0.97 at recall 0.83-0.86,
    F1 0.89-0.91 over five scenes, and shadow-derived building heights land
    within 1.4-2.0 m of truth.

    Provenance of those numbers, because it matters: part of that gain came from
    fixing the benchmark renderer, which used to darken shadows uniformly and so
    contained no chromatic cue to detect at all. The figures are measured against
    a renderer that now models skylight; performance on real imagery is
    unverified until we have a scene with a reference DSM.

    `max_fraction` is a backstop, not a tuning knob: no nadir scene at a usable
    sun elevation is a third shadow, so if the thresholds say otherwise they are
    wrong and the mask is tightened to the darkest pixels instead.
    """
    idx = shadow_index(scene.rgb)
    rgb = np.asarray(scene.rgb, np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    mask = (idx > _otsu(idx.ravel())) & (lum < np.percentile(lum, darkness_pct))

    if sem is not None:
        s = np.asarray(sem)
        # Water is dark and blue by nature; excluding it stops every river
        # being read as a shadow, and with it every riverside building
        # acquiring a hundred-metre height.
        mask &= s != WATER

    if mask.mean() > max_fraction:
        cut = float(np.percentile(lum, max_fraction * 100.0))
        mask &= lum < cut

    if HAVE_SCIPY:
        mask = binary_opening(mask, np.ones((3, 3), bool))
        mask = binary_closing(mask, np.ones((5, 5), bool))
        lab, n = label(mask)
        if n:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            keep = np.zeros(counts.size, bool)
            keep[counts >= min_px] = True
            mask = keep[lab]
    return mask.astype(bool)


def shadow_fraction(mask: np.ndarray) -> float:
    return float(np.asarray(mask, bool).mean())
