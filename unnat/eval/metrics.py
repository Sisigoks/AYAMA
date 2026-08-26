"""Validation metrics.

Two of these carry the pitch.

`bias_m` separates a wrong datum from a wrong model. A systematic offset is
fixable in one line; random error is not. Reporting both says you understand
your own failure modes.

`coverage_1s` is the honest test of the uncertainty field. If 68% of pixels
fall inside one sigma, the error bars are real; if they do not, sigma is
decoration and should be labelled as such.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import CLASS_NAMES
from ..measure.derive import slope_deg

try:
    from scipy.stats import pearsonr, spearmanr

    HAVE_SCIPY_STATS = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY_STATS = False


def _corr(p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    if p.size < 3 or np.ptp(p) == 0 or np.ptp(g) == 0:
        return float("nan"), float("nan")
    if HAVE_SCIPY_STATS:
        return float(pearsonr(p, g)[0]), float(spearmanr(p, g)[0])
    pr = float(np.corrcoef(p, g)[0, 1])
    rp = np.argsort(np.argsort(p)).astype(np.float64)
    rg = np.argsort(np.argsort(g)).astype(np.float64)
    return pr, float(np.corrcoef(rp, rg)[0, 1])


def expected_calibration_error(err: np.ndarray, sigma: np.ndarray, n_bins: int = 10) -> float:
    """Bin by predicted sigma; compare RMS error against mean sigma in each bin.

    Returned in metres, so it is directly readable: "our error bars are off by
    0.4 m on average".
    """
    err = np.asarray(err, np.float64).ravel()
    sigma = np.asarray(sigma, np.float64).ravel()
    m = np.isfinite(err) & np.isfinite(sigma) & (sigma > 0)
    if m.sum() < n_bins * 10:
        return float("nan")
    err, sigma = err[m], sigma[m]
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    total, ece = 0, 0.0
    for i in range(n_bins):
        sel = (sigma >= edges[i]) & (sigma < edges[i + 1])
        n = int(sel.sum())
        if n < 10:
            continue
        rms = float(np.sqrt((err[sel] ** 2).mean()))
        ece += n * abs(rms - float(sigma[sel].mean()))
        total += n
    return float(ece / total) if total else float("nan")


def boundary_f1(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                gsd: float = 1.0, tol_px: int = 2, pct: float = 92.0) -> float:
    """F1 between height discontinuities, with a tolerance band.

    Building outlines are where monocular height estimation actually fails, and
    a pixelwise MAE hides that. This scores whether the edges land in the right
    place, not whether the heights are right.
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:  # pragma: no cover
        return float("nan")

    def edges(a: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(np.asarray(a, np.float32), float(gsd))
        mag = np.hypot(gx, gy)
        vals = mag[mask & np.isfinite(mag)]
        if vals.size < 100:
            return np.zeros(a.shape, bool)
        return (mag > np.percentile(vals, pct)) & mask

    ep, eg = edges(pred), edges(ref)
    if not ep.any() or not eg.any():
        return float("nan")
    struct = np.ones((2 * tol_px + 1, 2 * tol_px + 1), bool)
    prec = float((ep & binary_dilation(eg, struct)).sum()) / max(int(ep.sum()), 1)
    rec = float((eg & binary_dilation(ep, struct)).sum()) / max(int(eg.sum()), 1)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def slope_error_deg(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, gsd: float) -> float:
    sp = slope_deg(pred, gsd)
    sg = slope_deg(ref, gsd)
    d = np.abs(sp - sg)[mask & np.isfinite(sp) & np.isfinite(sg)]
    return float(d.mean()) if d.size else float("nan")


def evaluate(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    gsd: float = 1.0,
    height_pred: Optional[np.ndarray] = None,
    height_ref: Optional[np.ndarray] = None,
) -> dict:
    """Compare a predicted surface against a reference, both in metres.

    `height_pred` / `height_ref` are heights above ground (nDSM). Pass them and
    delta1 is computed on those; the ratio metric is meaningless on absolute
    elevation, where the datum is arbitrary and a 400 m offset makes every
    ratio 1.0.
    """
    pred = np.asarray(pred, np.float64)
    ref = np.asarray(ref, np.float64)
    m = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not m.any():
        return {"n_px": 0}

    p, g = pred[m], ref[m]
    d = p - g

    out = {
        "n_px": int(m.sum()),
        "mae_m": float(np.abs(d).mean()),
        "rmse_m": float(np.sqrt((d ** 2).mean())),
        "bias_m": float(d.mean()),
        "median_ae_m": float(np.median(np.abs(d))),
        "p90_ae_m": float(np.percentile(np.abs(d), 90)),
        "std_m": float(d.std()),
    }
    out["pearson_r"], out["spearman_r"] = _corr(p, g)

    hp = height_pred if height_pred is not None else None
    hg = height_ref if height_ref is not None else None
    if hp is not None and hg is not None:
        a = np.asarray(hp, np.float64)[m]
        b = np.asarray(hg, np.float64)[m]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0.5) & (b > 0.5)
        out["delta1"] = (float((np.maximum(a[ok] / b[ok], b[ok] / a[ok]) < 1.25).mean())
                         if ok.sum() > 0 else float("nan"))
        out["delta1_n_px"] = int(ok.sum())
    else:
        out["delta1"] = float("nan")

    out["slope_mae_deg"] = slope_error_deg(pred, ref, m, gsd)
    out["edge_f1"] = boundary_f1(pred, ref, m, gsd)

    if sigma is not None:
        s = np.asarray(sigma, np.float64)[m]
        out["ece_m"] = expected_calibration_error(d, s)
        out["coverage_1s"] = float((np.abs(d) <= s).mean())
        out["coverage_2s"] = float((np.abs(d) <= 2 * s).mean())
        out["mean_sigma_m"] = float(np.nanmean(s))
    return out


def evaluate_by_class(
    pred: np.ndarray,
    ref: np.ndarray,
    sem: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    gsd: float = 1.0,
) -> dict:
    """Per-class error. The panel that shows you know *where* the model fails."""
    out = {}
    sem = np.asarray(sem)
    for cls, name in CLASS_NAMES.items():
        cm = sem == cls
        if mask is not None:
            cm = cm & np.asarray(mask, bool)
        if cm.sum() < 50:
            continue
        out[name] = evaluate(pred, ref, mask=cm, sigma=sigma, gsd=gsd)
    return out


def format_table(metrics: dict, title: str = "") -> str:
    """The block that goes on the slide."""
    order = [
        ("mae_m", "MAE", "{:.2f} m"),
        ("rmse_m", "RMSE", "{:.2f} m"),
        ("bias_m", "Bias", "{:+.2f} m"),
        ("median_ae_m", "Median AE", "{:.2f} m"),
        ("p90_ae_m", "P90 AE", "{:.2f} m"),
        ("pearson_r", "Pearson r", "{:.3f}"),
        ("spearman_r", "Spearman rho", "{:.3f}"),
        ("delta1", "delta < 1.25", "{:.3f}"),
        ("slope_mae_deg", "Slope MAE", "{:.2f} deg"),
        ("edge_f1", "Edge F1", "{:.3f}"),
        ("coverage_1s", "1-sigma coverage", "{:.2f}"),
        ("ece_m", "ECE", "{:.2f} m"),
    ]
    lines = []
    if title:
        lines.append(title)
    for key, label, fmt in order:
        if key not in metrics:
            continue
        v = metrics[key]
        lines.append(f"  {label:<18}{fmt.format(v) if np.isfinite(v) else '-':>12}")
    if "n_px" in metrics:
        lines.append(f"  {'pixels':<18}{metrics['n_px']:>12,}")
    return "\n".join(lines)
