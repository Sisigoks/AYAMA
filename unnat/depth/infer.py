"""Tiled inference with seam-free blending.

Three things here are load-bearing.

1. Rank normalisation per chip. Depth Anything emits inverse relative depth with
   an arbitrary per-image scale, so two adjacent chips can disagree by a factor
   of three even where they overlap the same rooftop. Each chip is mapped onto
   its own rank-percentile in [0, 1] before blending; physical meaning is
   recovered later, once, by the calibration stage. Skip this and the output is
   a quilt.

2. Overlap harmonisation. Rank normalisation alone is not enough: it makes each
   chip internally consistent and mutually incomparable, so a scene with real
   large-scale relief comes back with each chip's own range stretched over it.
   Every chip after the first is fitted (robust affine, on the overlap band)
   onto what the mosaic already says before it is accumulated. Blending alone
   would hide the seam and keep the error.

3. A flat-top raised-cosine window. It is 1.0 across the interior of a chip and
   ramps down only inside the overlap band, so interior pixels are not
   attenuated and the weight sum never approaches zero at the image border.

Sign convention: the backbone returns higher values for surfaces closer to the
sensor; from nadir, closer means higher elevation, so relative depth maps
monotonically to height. No flip anywhere. See backbones/base.py.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.types import DepthField, Scene
from .backbones import DepthBackbone, get_backbone


def rank_normalise(chip: np.ndarray) -> np.ndarray:
    """Map values onto their rank percentile in [0, 1]. NaN-safe, tie-safe."""
    flat = chip.reshape(-1).astype(np.float32)
    finite = np.isfinite(flat)
    out = np.zeros_like(flat)
    v = flat[finite]
    if v.size == 0:
        return out.reshape(chip.shape)
    if v.size == 1 or float(np.ptp(v)) == 0.0:
        out[finite] = 0.5
        return out.reshape(chip.shape)
    # Average the ranks of tied values so flat water stays flat.
    _uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    mean_rank = starts + (counts - 1) / 2.0
    ranks = mean_rank[inv].astype(np.float32)
    out[finite] = ranks / float(v.size - 1)
    return out.reshape(chip.shape)


def blend_window(chip: int, ramp: int) -> np.ndarray:
    """Flat-top raised cosine: 1.0 inside, cosine ramp over `ramp` px at each edge."""
    ramp = int(max(1, min(ramp, chip // 2)))
    w1 = np.ones(chip, np.float32)
    t = (np.arange(ramp, dtype=np.float32) + 0.5) / ramp
    taper = 0.5 * (1.0 - np.cos(np.pi * t))
    w1[:ramp] = taper
    w1[chip - ramp:] = taper[::-1]
    w = np.outer(w1, w1)
    return np.maximum(w, 1e-3)


def robust_affine_fit(src: np.ndarray, ref: np.ndarray, w: np.ndarray,
                      iters: int = 2) -> tuple[float, float]:
    """Least-squares s, t minimising w * (s * src + t - ref)^2, Huber-reweighted.

    One IRLS pass is enough: the outliers here are occlusion edges inside the
    overlap band, not gross blunders.
    """
    src = src.astype(np.float64).ravel()
    ref = ref.astype(np.float64).ravel()
    w = w.astype(np.float64).ravel()
    if src.size < 16:
        return 1.0, 0.0
    for _ in range(max(1, iters)):
        sw = w.sum()
        if sw <= 0:
            return 1.0, 0.0
        ms = (w * src).sum() / sw
        mr = (w * ref).sum() / sw
        var = (w * (src - ms) ** 2).sum() / sw
        if var < 1e-8:
            return 1.0, float(mr - ms)
        cov = (w * (src - ms) * (ref - mr)).sum() / sw
        s = cov / var
        t = mr - s * ms
        resid = np.abs(s * src + t - ref)
        mad = np.median(resid) if resid.size else 0.0
        delta = max(1.4826 * mad, 1e-4)
        w = w * np.minimum(1.0, delta / np.maximum(resid, 1e-12))
    s = float(np.clip(s, 0.05, 20.0))
    return s, float(t)


def tile_offsets(total: int, chip: int, step: int) -> list[int]:
    """Start positions that always cover the final edge, without duplicates."""
    if total <= chip:
        return [0]
    offs = list(range(0, total - chip + 1, max(1, step)))
    if offs[-1] != total - chip:
        offs.append(total - chip)
    return offs


def predict_depth(
    scene: Scene,
    model: DepthBackbone | str = "dav2-vits",
    chip: int = 1024,
    overlap: float = 0.25,
    batch_size: int = 1,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> DepthField:
    """Tiled relative depth over a whole scene.

    `batch_size` batches the forward passes only. Accumulation stays strictly
    sequential in raster order, because each chip is harmonised against what
    the mosaic already says and that reference has to exist before the chip is
    added to it.
    """
    if isinstance(model, str):
        model = get_backbone(model)
    model.load()

    rgb = scene.rgb
    H, W = rgb.shape[:2]
    chip = int(min(chip, max(H, W))) if max(H, W) > 0 else chip
    chip = max(64, chip)
    overlap = float(np.clip(overlap, 0.0, 0.75))
    step = max(1, int(round(chip * (1.0 - overlap))))
    ramp = max(2, int(round(chip * overlap / 2.0)))

    # Pad by edge replication when the image is smaller than one chip, so the
    # model always sees a full-size input and the result is cropped back.
    pad_h = max(0, chip - H)
    pad_w = max(0, chip - W)
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    ph, pw = rgb.shape[:2]

    win = blend_window(chip, ramp)
    # Fit a chip onto the mosaic only when the overlap is big enough to be
    # informative; below that, trust the chip as-is.
    min_overlap_px = max(64, (chip * ramp) // 4)
    acc = np.zeros((ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)

    ys = tile_offsets(ph, chip, step)
    xs = tile_offsets(pw, chip, step)
    coords = [(y, x) for y in ys for x in xs]
    total = len(coords)
    if batch_size <= 0:
        batch_size = getattr(model, "suggest_batch_size", lambda _c: 1)(chip)
    batch_size = max(1, int(batch_size))

    done = 0
    for b0 in range(0, total, batch_size):
        batch = coords[b0:b0 + batch_size]
        patches = [rgb[y:y + chip, x:x + chip] for y, x in batch]
        preds = (model.infer_batch(patches) if batch_size > 1
                 else [model.infer(patches[0])])
        for (y, x), raw in zip(batch, preds):
            d = np.asarray(raw, dtype=np.float32)
            if d.shape != (chip, chip):
                raise ValueError(
                    f"backbone {getattr(model, 'name', '?')} returned {d.shape}, expected {(chip, chip)}"
                )
            d = rank_normalise(d)

            # Per-chip normalisation makes each chip internally consistent but
            # says nothing across chips: on a scene with real large-scale relief
            # two neighbours can disagree by their whole range. Before
            # accumulating, fit each chip onto what the mosaic already says in
            # the overlap band. Without this the blend hides the seam but keeps
            # the error, and no downstream calibration can undo it.
            sub_acc = acc[y:y + chip, x:x + chip]
            sub_w = wsum[y:y + chip, x:x + chip]
            seen = sub_w > 1e-3
            if seen.sum() >= min_overlap_px:
                ref = sub_acc[seen] / sub_w[seen]
                s, t = robust_affine_fit(d[seen], ref, sub_w[seen])
                d = s * d + t

            acc[y:y + chip, x:x + chip] += d * win
            wsum[y:y + chip, x:x + chip] += win
            done += 1
            if on_progress is not None:
                on_progress(done, total)

    rel = acc / np.maximum(wsum, 1e-6)
    rel = rel[:H, :W]
    # Re-normalise the mosaic once so the whole tile shares a single [0, 1] range.
    lo, hi = float(np.nanmin(rel)), float(np.nanmax(rel))
    if hi - lo > 1e-9:
        rel = (rel - lo) / (hi - lo)
    return DepthField(
        relative=rel.astype(np.float32),
        meta=scene.meta,
        backbone=getattr(model, "name", str(model)),
    )


def n_chips(shape: tuple[int, int], chip: int, overlap: float) -> int:
    H, W = shape
    chip = max(64, int(min(chip, max(H, W))))
    step = max(1, int(round(chip * (1.0 - float(np.clip(overlap, 0.0, 0.75))))))
    return len(tile_offsets(max(H, chip), chip, step)) * len(tile_offsets(max(W, chip), chip, step))
