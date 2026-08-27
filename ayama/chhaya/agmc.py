"""Anchor-Graph Metric Calibration.

A global affine fit, H = a*D + b, has two unknowns for a whole tile. It is
forced to average away every local disagreement between anchor sources, and it
inherits the worst error of each. AGMC replaces the two scalars with two smooth
fields solved on a coarse lattice:

    H(x, y) = a(x, y) * D(x, y) + b(x, y)

    E(a, b) = sum_k w_k * rho( a(p_k) D(p_k) + b(p_k) - h_k )     data
            + lam_s ( ||grad a||^2 + ||grad b||^2 )               smoothness
            + lam_p ||a - a_global||^2                            prior

Two anchor kinds, because the sources measure different things:

  absolute  - "this pixel is at 412.3 m". DEM samples on bare ground, GCPs,
              ICESat-2 returns.
  relative  - "this pixel stands 34.7 m above that one". Shadow-derived
              heights, which say nothing about the datum. These enter as a
              difference of two rows, which is what keeps a shadow measurement
              from being quietly reinterpreted as an elevation.

The lattice keeps the system small (a 32 px stride on a 4k tile is 128x128
nodes, 32k unknowns, seconds on CPU) and the graph Laplacian keeps the fields
smooth between anchors. IRLS with a Huber weight gives outlier rejection
without a RANSAC loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from ..core.types import Anchor, CalibrationField, Config, DepthField, Tier


@dataclass
class Lattice:
    """Coarse control grid with bilinear interpolation onto the pixel grid."""

    h: int          # nodes down
    w: int          # nodes across
    stride: int
    shape: tuple    # (H, W) of the full raster

    @property
    def n(self) -> int:
        return self.h * self.w

    def weights(self, rows: np.ndarray, cols: np.ndarray):
        """Bilinear node indices and weights for pixel coordinates.

        Returns (idx, wts), both (N, 4). Spreading each anchor over its four
        surrounding nodes conditions the system far better than snapping to the
        nearest node, and it removes the blocky artefacts that a nearest-node
        assignment leaves in the calibration field.
        """
        gr = np.clip(np.asarray(rows, np.float64) / self.stride, 0, self.h - 1)
        gc = np.clip(np.asarray(cols, np.float64) / self.stride, 0, self.w - 1)
        r0 = np.clip(np.floor(gr).astype(int), 0, self.h - 1)
        c0 = np.clip(np.floor(gc).astype(int), 0, self.w - 1)
        r1 = np.minimum(r0 + 1, self.h - 1)
        c1 = np.minimum(c0 + 1, self.w - 1)
        fr = gr - r0
        fc = gc - c0
        idx = np.stack([r0 * self.w + c0, r0 * self.w + c1,
                        r1 * self.w + c0, r1 * self.w + c1], axis=1)
        wts = np.stack([(1 - fr) * (1 - fc), (1 - fr) * fc,
                        fr * (1 - fc), fr * fc], axis=1)
        return idx, wts

    def upsample(self, values: np.ndarray) -> np.ndarray:
        """Lattice nodes -> full raster, bilinear."""
        from scipy.ndimage import map_coordinates

        H, W = self.shape
        rr = np.clip(np.arange(H, dtype=np.float64) / self.stride, 0, self.h - 1)
        cc = np.clip(np.arange(W, dtype=np.float64) / self.stride, 0, self.w - 1)
        grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
        return map_coordinates(values.reshape(self.h, self.w), [grid_r, grid_c],
                               order=1, mode="nearest").astype(np.float32)


def make_lattice(shape: tuple, stride: int) -> Lattice:
    H, W = shape
    stride = int(max(1, stride))
    return Lattice(h=max(2, int(np.ceil(H / stride)) + 1),
                   w=max(2, int(np.ceil(W / stride)) + 1),
                   stride=stride, shape=(H, W))


def graph_laplacian(h: int, w: int) -> sp.csr_matrix:
    """5-point Laplacian on the lattice, as L^T L via the gradient operator."""
    n = h * w
    rows, cols, vals = [], [], []
    eq = 0
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if c + 1 < w:
                rows += [eq, eq]; cols += [i, i + 1]; vals += [1.0, -1.0]; eq += 1
            if r + 1 < h:
                rows += [eq, eq]; cols += [i, i + w]; vals += [1.0, -1.0]; eq += 1
    G = sp.csr_matrix((vals, (rows, cols)), shape=(eq, n))
    return (G.T @ G).tocsr()


def huber_weight(residual: np.ndarray, delta: float) -> np.ndarray:
    r = np.abs(np.asarray(residual, np.float64))
    return np.minimum(1.0, delta / np.maximum(r, 1e-9))


def global_affine(depth: np.ndarray, anchors: Sequence[Anchor],
                  huber_delta: float = 2.0, iters: int = 3) -> tuple[float, float]:
    """Robust global fit, used as the prior the spatial fields are pulled toward.

    Only anchors that state an elevation may take part. A relative anchor says
    "this point is h metres above that one"; read as an elevation it becomes
    "this point is at h metres", and a water body saying "level with itself"
    (h = 0) then drags the whole datum to zero. Excluding them here is what
    keeps the global-affine baseline an honest comparison instead of a straw man.
    """
    abs_anchors = [k for k in anchors if k.branch != "object" and not k.is_relative]
    if len(abs_anchors) < 2:
        return 1.0, 0.0
    d = np.array([depth[k.row, k.col] for k in abs_anchors], np.float64)
    h = np.array([k.value_m for k in abs_anchors], np.float64)
    w = np.array([k.weight for k in abs_anchors], np.float64)
    a, b = 1.0, float(np.average(h, weights=w) - np.average(d, weights=w))
    for _ in range(iters):
        sw = w.sum()
        if sw <= 0:
            break
        md, mh = (w * d).sum() / sw, (w * h).sum() / sw
        var = (w * (d - md) ** 2).sum() / sw
        if var < 1e-12:
            break
        a = float((w * (d - md) * (h - mh)).sum() / sw / var)
        b = float(mh - a * md)
        w = np.array([k.weight for k in abs_anchors], np.float64) * \
            huber_weight(a * d + b - h, huber_delta)
    return a, b


def solve_agmc(
    depth: DepthField | np.ndarray,
    anchors: Sequence[Anchor],
    cfg: Optional[Config] = None,
    tier: Tier = Tier.C,
    enforce_positive: Optional[bool] = None,
) -> CalibrationField:
    """Solve for the calibration fields a(x, y), b(x, y).

    `enforce_positive` keeps the scale field above `min_scale`. It defaults to
    on, and it is not a regularisation nicety - it enforces the pipeline's own
    documented convention, that relative depth increases with height.

    Why it matters: when a depth backbone carries a prior that anti-correlates
    with the terrain (Depth Anything V2 applies a ground-level perspective ramp
    to nadir imagery), and the anchor set is dominated by terrain samples, an
    unconstrained fit will happily choose a NEGATIVE scale. Terrain then matches
    beautifully and every building is turned upside down - a roof the model
    correctly ranked as higher is rendered as a pit. Measured on the benchmark:
    the scale field came out negative at every node and buildings landed 2.6 m
    *below* the ground they stand on, while the headline MAE still improved.
    A metric that cannot see an inverted city is not measuring what it claims.
    """
    cfg = cfg or Config()
    if enforce_positive is None:
        enforce_positive = bool(cfg.extras.get("enforce_positive_scale", True))
    min_scale = float(cfg.extras.get("min_scale", 0.05))
    D = depth.relative if isinstance(depth, DepthField) else np.asarray(depth, np.float32)
    H, W = D.shape
    lat = make_lattice((H, W), cfg.lattice_stride)
    n = lat.n

    anchors = [k for k in anchors if 0 <= k.row < H and 0 <= k.col < W and np.isfinite(k.value_m)]
    if not anchors:
        # Nothing to calibrate against: return an identity field and say so.
        return CalibrationField(
            a=np.ones((H, W), np.float32), b=np.zeros((H, W), np.float32),
            residual_rmse=float("nan"), n_anchors_used=0, n_anchors_rejected=0, tier=tier,
        )

    a_glob, b_glob = global_affine(D, anchors, cfg.huber_delta)

    rows = np.array([k.row for k in anchors])
    cols = np.array([k.col for k in anchors])
    idx, wts = lat.weights(rows, cols)
    dvals = D[rows, cols].astype(np.float64)
    rhs = np.array([k.value_m for k in anchors], np.float64)
    w0 = np.array([max(k.weight, 1e-6) for k in anchors], np.float64)

    # Relative anchors carry a reference pixel: the constraint is on the
    # difference between two points, not on either one's elevation.
    has_ref = np.array([k.ref_row is not None for k in anchors])
    if has_ref.any():
        ref_rows = np.array([k.ref_row if k.ref_row is not None else k.row for k in anchors])
        ref_cols = np.array([k.ref_col if k.ref_col is not None else k.col for k in anchors])
        ridx, rwts = lat.weights(ref_rows, ref_cols)
        rdvals = D[ref_rows, ref_cols].astype(np.float64)
    else:
        ridx = rwts = rdvals = None

    m = len(anchors)
    r_i, c_i, v_i = [], [], []
    for j in range(4):
        r_i.append(np.arange(m)); c_i.append(idx[:, j]);     v_i.append(wts[:, j] * dvals)
        r_i.append(np.arange(m)); c_i.append(n + idx[:, j]); v_i.append(wts[:, j])
        if ridx is not None:
            sign = np.where(has_ref, -1.0, 0.0)
            r_i.append(np.arange(m)); c_i.append(ridx[:, j])
            v_i.append(sign * rwts[:, j] * rdvals)
            r_i.append(np.arange(m)); c_i.append(n + ridx[:, j])
            v_i.append(sign * rwts[:, j])
    A = sp.csr_matrix(
        (np.concatenate(v_i), (np.concatenate(r_i), np.concatenate(c_i))),
        shape=(m, 2 * n),
    )

    L = graph_laplacian(lat.h, lat.w)
    # Balance the two terms per unknown, not per anchor. The data term sums over
    # m anchors while the smoothness term sums over n lattice nodes, so scaling
    # by m alone (with m >> n, which is the normal case) buries the anchors under
    # the prior and quietly collapses AGMC back to a global affine fit.
    lam_scale = float(max(m, 1)) / float(max(n, 1))
    R = sp.block_diag([cfg.lam_a * lam_scale * L, cfg.lam_b * lam_scale * L]).tocsr()
    # Prior: pull a toward the robust global fit. b is left free; the datum is
    # exactly what the anchors are there to determine.
    lam_p = cfg.extras.get("lam_prior", 0.05) * lam_scale
    P = sp.block_diag([lam_p * sp.eye(n), sp.csr_matrix((n, n))]).tocsr()
    prior = np.concatenate([np.full(n, a_glob), np.zeros(n)])

    x = np.concatenate([np.full(n, a_glob), np.full(n, b_glob)])
    w = w0.copy()
    resid = np.zeros(m)
    for _ in range(max(1, cfg.irls_iters)):
        Wm = sp.diags(w)
        M = (A.T @ Wm @ A + R + P).tocsr()
        rhs_vec = A.T @ (w * rhs) + P @ prior
        try:
            x = spsolve(M, rhs_vec) if M.shape[0] <= 20000 else cg(M, rhs_vec, x0=x, rtol=1e-6)[0]
        except Exception:
            x = cg(M, rhs_vec, x0=x, rtol=1e-6)[0]
        if not np.all(np.isfinite(x)):
            x = np.concatenate([np.full(n, a_glob), np.full(n, b_glob)])
            break
        if enforce_positive:
            x = _project_positive_scale(x, n, A, w, rhs, R, min_scale)
        resid = A @ x - rhs
        w = w0 * huber_weight(resid, cfg.huber_delta)

    rejected = int((w < 0.25 * w0).sum())
    used = m - rejected
    a_field = lat.upsample(x[:n])
    b_field = lat.upsample(x[n:])
    rmse = float(np.sqrt(np.mean(resid ** 2))) if m else float("nan")
    return CalibrationField(a=a_field, b=b_field, residual_rmse=rmse,
                            n_anchors_used=used, n_anchors_rejected=rejected, tier=tier)


def _project_positive_scale(x, n, A, w, rhs, R, min_scale: float = 0.05):
    """Clamp the scale field positive, then re-solve the offset field for it.

    A projected step rather than a bounded solver: clamping `a` alone would leave
    `b` fitted against the old scale and shift the whole datum, so `b` is
    re-solved with the clamped `a` held fixed. That sub-problem is linear and
    the same size as half the original, so it costs one extra solve per IRLS
    iteration.
    """
    a = x[:n]
    if (a >= min_scale).all():
        return x
    a = np.maximum(a, min_scale)

    # Residual the offset field still has to explain, with `a` fixed.
    Aa, Ab = A[:, :n], A[:, n:]
    target = rhs - Aa @ a
    Wm = sp.diags(w)
    Rb = R[n:, n:]
    M = (Ab.T @ Wm @ Ab + Rb).tocsr()
    rhs_b = Ab.T @ (w * target)
    try:
        b = spsolve(M, rhs_b) if M.shape[0] <= 20000 else cg(M, rhs_b, x0=x[n:], rtol=1e-6)[0]
    except Exception:
        b = cg(M, rhs_b, x0=x[n:], rtol=1e-6)[0]
    if not np.all(np.isfinite(b)):
        b = x[n:]
    return np.concatenate([a, b])


def apply_calibration(depth: DepthField | np.ndarray, calib: CalibrationField) -> np.ndarray:
    D = depth.relative if isinstance(depth, DepthField) else np.asarray(depth, np.float32)
    return (calib.a * D + calib.b).astype(np.float32)
