"""Image-conditioned refinement of a calibrated height field.

The surface TRAKSHA delivers is metrically anchored but geometrically soft: its
scale and offset fields live on a 32 px lattice and the structural term is a
single constant on a Gaussian high band, so building edges arrive smeared across
several metres of ground. That is a *detail* problem, not a *datum* problem, and
the two must not be fixed with the same tool.

So this refines a residual and nothing else:

    Z_final(p) = Z_rough(p) + dZ(p),     |dZ| <= clamp,     mean(dZ | anchor) = 0

`Z_rough` keeps the metric result. `dZ` is allowed to move height *within* a
neighbourhood - sharpening a roof edge, squaring a wall - and is forbidden from
moving the neighbourhood itself. A refinement that can shift the mean has
rewritten the calibration, which is the one thing the anchors were for.

The guide is the image. A guided filter (He, Sun & Tang) transfers structure
from the orthophoto onto the height field: where the image has an edge the
filter stops averaging, so a roof boundary that was ramped over five pixels
becomes a step. This is the standard depth-refinement operator and it needs no
training, no second network and no generative prior.

**It can also transfer things that are not geometry.** A painted line, a shadow
edge and a roof edge look alike to a guide image, and the filter will happily
carve a step where there is only paint. That is why the residual is clamped, why
the guide is the low-saturation luminance rather than raw colour, and why
`traksha refine` measures edge F1 and nDSM MAE against a reference and reports
both rather than assuming the operation helped. On the four scenes in this
repository it helps; on imagery with strong texture and weak relief it may not,
and the numbers will say so.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import uniform_filter

    HAVE_SCIPY = True
except ImportError:                                   # pragma: no cover
    HAVE_SCIPY = False


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)² window, edge-replicated."""
    if HAVE_SCIPY:
        return uniform_filter(a, size=2 * r + 1, mode="nearest")
    k = 2 * r + 1
    pad = np.pad(a, r, mode="edge")
    out = np.cumsum(np.cumsum(pad, 0), 1)
    out = np.pad(out, ((1, 0), (1, 0)))
    s = (out[k:, k:] - out[:-k, k:] - out[k:, :-k] + out[:-k, :-k])
    return s / float(k * k)


def guided_filter(src: np.ndarray, guide: np.ndarray, radius: int = 4,
                  eps: float = 1e-3) -> np.ndarray:
    """He, Sun & Tang's guided filter: `src` smoothed, `guide`'s edges kept.

    Locally it fits `src ≈ a·guide + b` over each window and averages the
    coefficients. Inside a flat region the fit is dominated by `eps` and the
    output is a local mean; across an edge in the guide the fit follows the
    guide and the edge survives. That is exactly the behaviour wanted here -
    average the noise on a roof, do not average across its boundary.
    """
    s = np.asarray(src, np.float32)
    g = np.asarray(guide, np.float32)
    mean_g, mean_s = _box(g, radius), _box(s, radius)
    cov = _box(g * s, radius) - mean_g * mean_s
    var = _box(g * g, radius) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return (_box(a, radius) * g + _box(b, radius)).astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Guide channel in [0, 1].

    Luminance rather than colour: a guided filter keyed on hue will carve
    geometry out of paint. Luminance is not immune to that - a dark roof on
    light ground is both an albedo edge and a real one - but it is the channel
    where genuine geometric boundaries are most reliably present.
    """
    a = np.asarray(rgb, np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    else:
        lum = a.squeeze()
    lo, hi = np.percentile(lum, [1, 99])
    return np.clip((lum - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def refine_heights(ndsm_m: np.ndarray, rgb: np.ndarray, gsd_m: float,
                   radius_m: float = 2.0, eps: float = 4e-3,
                   clamp_m: Optional[float] = None,
                   preserve_scale_m: float = 30.0) -> tuple:
    """Sharpen height above ground against the image. Returns (refined, dZ).

    `preserve_scale_m` is the neighbourhood whose mean height is held fixed. The
    refinement may redistribute height inside it and may not change its total,
    so the calibrated surface survives at every scale the anchors constrained
    and only the detail below it moves.

    `clamp_m` bounds the residual. It defaults to a quarter of the surface's own
    99th percentile, which keeps the operator to sharpening what is already
    there rather than inventing structure the calibration never found.
    """
    z = np.asarray(ndsm_m, np.float32)
    finite = np.isfinite(z)
    if not finite.all():
        z = np.where(finite, z, 0.0).astype(np.float32)

    guide = _luminance(rgb)
    r = max(1, int(round(float(radius_m) / max(float(gsd_m), 1e-6))))
    sharp = guided_filter(z, guide, radius=r, eps=eps)

    dz = sharp - z
    # Hold the local mean: whatever is added here is taken from there.
    keep = max(1, int(round(float(preserve_scale_m) / max(float(gsd_m), 1e-6))))
    dz = dz - _box(dz, keep)

    if clamp_m is None:
        clamp_m = 0.25 * float(np.percentile(z[np.isfinite(z)], 99) or 0.0)
    if clamp_m > 0:
        dz = np.clip(dz, -clamp_m, clamp_m)

    out = np.maximum(z + dz, 0.0).astype(np.float32)    # ground is a floor
    return out, (out - z).astype(np.float32)


def refine_run(run: dict, gsd_m: float, **kw) -> dict:
    """Refine a loaded run's surface. Returns the new rasters and a report."""
    ndsm = run.get("ndsm")
    dsm = run.get("dsm")
    rgb = run.get("texture")
    if ndsm is None or dsm is None or rgb is None:
        return {}
    refined, dz = refine_heights(ndsm, rgb, gsd_m, **kw)
    return {
        "ndsm": refined,
        # The DSM moves by exactly what the nDSM moved by: the ground the
        # anchors placed is untouched, which is the whole contract.
        "dsm": (np.asarray(dsm, np.float32) + dz).astype(np.float32),
        "residual": dz,
        "stats": {
            "max_abs_residual_m": float(np.abs(dz).max()),
            "rms_residual_m": float(np.sqrt(np.mean(dz.astype(np.float64) ** 2))),
            "moved_fraction": float((np.abs(dz) > 0.05).mean()),
        },
    }
