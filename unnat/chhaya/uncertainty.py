"""Per-pixel uncertainty, from three independent sources.

    sigma^2 = sigma_calib^2 + sigma_model^2 + sigma_ref^2

sigma_calib   bootstrap over the anchor set. Cheap, rigorous, and it captures
              the term that actually dominates: the resulting field is large
              where anchors are sparse and small where they cluster, which is
              exactly the behaviour a reviewer expects to see.
sigma_model   spread between two backbones, or MC dropout. Crude, defensible,
              nearly free.
sigma_ref     the reference DEM's own vertical accuracy, from its datasheet.
              This is the term that honestly explains why absolute elevation is
              less certain than relative building height, and saying so out
              loud reads as rigour rather than weakness.

A sigma that does not predict error is decoration. `unnat.eval.metrics` reports
ECE and one-sigma coverage precisely so this one can be shown not to be.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..core.types import Anchor, Config, DepthField
from .agmc import apply_calibration, solve_agmc
from .anchors import DEM_SIGMA_M


def bootstrap_sigma(
    depth: DepthField,
    anchors: Sequence[Anchor],
    cfg: Optional[Config] = None,
    n_boot: int = 24,
    frac: float = 0.7,
    seed: int = 0,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean surface, calibration sigma), both in metres.

    Twenty-four solves of a small sparse system take seconds, which is the whole
    reason the calibration stage was kept separate and cheap.
    """
    cfg = cfg or Config()
    anchors = list(anchors)
    n = len(anchors)
    if n < 8 or n_boot < 2:
        calib = solve_agmc(depth, anchors, cfg)
        surface = apply_calibration(depth, calib)
        return surface, np.zeros_like(surface)

    rng = np.random.default_rng(seed)
    keep_n = max(4, int(frac * n))
    mean = None
    m2 = None
    count = 0
    for i in range(n_boot):
        idx = rng.choice(n, keep_n, replace=False)
        calib = solve_agmc(depth, [anchors[j] for j in idx], cfg)
        s = apply_calibration(depth, calib)
        # Welford, so a 4k tile x 24 bootstraps never has to be held in memory.
        count += 1
        if mean is None:
            mean = s.astype(np.float64)
            m2 = np.zeros_like(mean)
        else:
            delta = s - mean
            mean += delta / count
            m2 += delta * (s - mean)
        if on_progress is not None:
            on_progress(i + 1, n_boot)

    var = m2 / max(count - 1, 1)
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def model_sigma(surfaces: Sequence[np.ndarray]) -> np.ndarray:
    """Spread between backbones. Half the absolute difference for two of them."""
    arrs = [np.asarray(s, np.float32) for s in surfaces if s is not None]
    if len(arrs) < 2:
        return np.zeros_like(arrs[0]) if arrs else np.zeros((1, 1), np.float32)
    if len(arrs) == 2:
        return (0.5 * np.abs(arrs[0] - arrs[1])).astype(np.float32)
    return np.std(np.stack(arrs), axis=0).astype(np.float32)


def reference_sigma(shape: tuple, source: Optional[str], tier_is_dem: bool = True) -> np.ndarray:
    """Constant field carrying the DEM's datasheet accuracy into the budget."""
    if not tier_is_dem or not source:
        return np.zeros(shape, np.float32)
    sigma = DEM_SIGMA_M.get(str(source).lower(), DEM_SIGMA_M["unknown"])
    return np.full(shape, float(sigma), np.float32)


def combine(*fields: Optional[np.ndarray]) -> np.ndarray:
    """Quadrature sum of independent one-sigma fields."""
    total = None
    for f in fields:
        if f is None:
            continue
        a = np.asarray(f, np.float32)
        total = a ** 2 if total is None else total + a ** 2
    if total is None:
        return np.zeros((1, 1), np.float32)
    return np.sqrt(np.maximum(total, 0.0)).astype(np.float32)
