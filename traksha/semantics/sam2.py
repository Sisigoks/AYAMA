"""SAM 2 structural segmentation, and the automatic mask generator it needs.

Why this file exists at all: the pipeline's only structural knowledge was a
colour and texture heuristic with no instances and no confidence, whose building
class the README already documents as untrustworthy (section 3.4). Nothing
downstream could tell one building from the next, because nothing upstream had
ever separated them.

**The model comes from `transformers`, not from the `sam2` package.** transformers
5.15 ships the official SAM 2 architecture as `Sam2Model` / `Sam2Processor`, so
the weights load through the same Hugging Face path the depth backbones already
use: one loading mechanism, one cache, no git checkout of a research repo pinned
to a torch version this environment does not have.

**Automatic mask generation is implemented here because transformers does not
provide it.** It exposes the promptable model only - there is no
`SAM2AutomaticMaskGenerator` - so the loop below is the published algorithm: a
regular grid of point prompts, three candidate masks per point, filtering on the
model's own predicted IoU and on a stability score, then box NMS to remove the
duplicates that a grid inevitably produces. It is a specified procedure, not a
reinvention, and the constants carry the reference values.

**What is deliberately left out.** The reference generator also re-runs the whole
grid over image crops to catch small objects. On four CPU threads that multiplies
an already dominant cost for objects smaller than a building, which is not what
this pipeline is looking for. Measured on this machine at 1024 px: 3.1 s to
encode, then roughly 0.04 s per point prompt, so a 16x16 grid is about 40 s and a
32x32 grid about three minutes. Grid size is therefore configuration, not a
constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Checkpoint per variant. Names are the pipeline's, ids are Meta's.
VARIANTS = {
    "sam2-tiny": "facebook/sam2.1-hiera-tiny",
    "sam2-small": "facebook/sam2.1-hiera-small",
    "sam2-base": "facebook/sam2.1-hiera-base-plus",
    "sam2-large": "facebook/sam2.1-hiera-large",
}
# Measured, not assumed - `scripts/bench_segmentation.py` reproduces this.
# Built recall is the fraction of pixels carrying more than 2.5 m of lidar
# structure that some mask covers, over three swisstopo city scenes at a 20x20
# grid:
#
#   variant     params   generate   built recall   precision
#   sam2-tiny    31.4M      59 s        50.9%        52.8%
#   sam2-small   38.5M      56 s        60.5%        68.4%   (bern only)
#   sam2-base    73.3M      58 s        73.0%        57.0%
#
# base wins on every scene and by 22 points on average, for no measurable extra
# generation time - the grid, not the encoder, is what this stage spends its
# time on. Only the load is slower (about 16 s against 10 s) and the resident
# model is twice the size. Taking the smallest model here would have cost a
# fifth of the structure the rest of the pipeline is meant to reconstruct.
DEFAULT_VARIANT = "sam2-base"

# Reference thresholds from the published automatic mask generator. They are
# named rather than inlined because every one of them is a quality/recall knob
# a reader may want to move, and because a bare 0.88 in a comparison is a number
# nobody can check.
PRED_IOU_THRESH = 0.75          # the model's own opinion of each mask
STABILITY_SCORE_THRESH = 0.90   # agreement under a shifted logit threshold
STABILITY_SCORE_OFFSET = 1.0
BOX_NMS_THRESH = 0.70           # a point grid produces duplicates by design
MASK_THRESHOLD = 0.0            # SAM's mask logits are thresholded at zero


class Sam2Unavailable(RuntimeError):
    """SAM 2 could not be loaded. Carries a reason a user can act on."""


@dataclass
class Mask:
    """One accepted mask, with the evidence that got it accepted."""

    segmentation: np.ndarray            # (H, W) bool
    area_px: int
    bbox: tuple                         # (x0, y0, w, h)
    predicted_iou: float
    stability_score: float
    point: tuple                        # the prompt that produced it

    @property
    def score(self) -> float:
        return float(self.predicted_iou * self.stability_score)


def _stability_score(logits, offset: float = STABILITY_SCORE_OFFSET):
    """How much a mask changes when the decision threshold moves.

    A mask whose area is the same whether the threshold is raised or lowered is
    a mask the model is confident about; one that doubles is a gradient the
    thresholding happened to cut somewhere. Computed on the low-resolution
    logits, which is what the reference implementation does and is far cheaper
    than the upsampled masks.
    """
    import torch

    hi = (logits > (MASK_THRESHOLD + offset)).flatten(-2).sum(-1, dtype=torch.int32)
    lo = (logits > (MASK_THRESHOLD - offset)).flatten(-2).sum(-1, dtype=torch.int32)
    return hi.float() / torch.clamp(lo.float(), min=1.0)


def _bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))


def _point_grid(n: int, h: int, w: int) -> np.ndarray:
    """An n x n grid of prompt points in pixel coordinates, inset from the edges."""
    offs = (np.arange(n, dtype=np.float32) + 0.5) / n
    xs, ys = np.meshgrid(offs * w, offs * h)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


@dataclass
class Sam2Segmenter:
    """Automatic mask generation over a whole image.

    `generate()` returns masks sorted largest first, deduplicated, and filtered
    by the model's predicted IoU and by stability. It holds the weights until
    `unload()`, which the pipeline calls the moment the masks exist - the same
    discipline the depth backbone follows, and for the same reason: a run that
    keeps every model it has ever used alive dies on the second scene.
    """

    variant: str = DEFAULT_VARIANT
    points_per_side: int = 16
    points_per_batch: int = 16
    pred_iou_thresh: float = PRED_IOU_THRESH
    stability_score_thresh: float = STABILITY_SCORE_THRESH
    box_nms_thresh: float = BOX_NMS_THRESH
    min_area_px: int = 64
    _model: object = field(default=None, repr=False)
    _processor: object = field(default=None, repr=False)

    # ------------------------------------------------------------- weights
    @property
    def checkpoint(self) -> str:
        try:
            return VARIANTS[self.variant]
        except KeyError:
            raise Sam2Unavailable(
                f"unknown SAM 2 variant '{self.variant}'. "
                f"Available: {', '.join(VARIANTS)}") from None

    def load(self) -> "Sam2Segmenter":
        if self._model is not None:
            return self
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise Sam2Unavailable(
                "SAM 2 needs transformers>=5.15 and torch:\n"
                "  pip install 'transformers>=5.15' torch") from exc
        try:
            self._processor = Sam2Processor.from_pretrained(self.checkpoint)
            self._model = Sam2Model.from_pretrained(self.checkpoint).eval()
        except Exception as exc:                      # network, hub, disk
            raise Sam2Unavailable(
                f"could not load {self.checkpoint}: {type(exc).__name__}: {exc}"
            ) from exc
        torch.set_grad_enabled(False)
        return self

    def unload(self) -> None:
        import gc

        self._model = None
        self._processor = None
        gc.collect()

    def describe(self) -> str:
        return f"{self.variant} ({self.checkpoint}), {self.points_per_side}x" \
               f"{self.points_per_side} point grid"

    # ---------------------------------------------------------- generation
    def generate(self, rgb: np.ndarray,
                 on_progress: Optional[Callable[[int, int], None]] = None) -> list:
        """Masks for one image. Loads the weights if they are not loaded yet."""
        self.load()
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")
        rgb = np.ascontiguousarray(rgb[..., :3])
        h, w = rgb.shape[:2]

        # Encode once. Every prompt batch reuses this; re-encoding per batch
        # would spend the whole budget on the part of the model that does not
        # depend on the prompt.
        base = self._processor(images=rgb, return_tensors="pt")
        embeddings = self._model.get_image_embeddings(base["pixel_values"])

        grid = _point_grid(self.points_per_side, h, w)
        total = len(grid)
        kept: list = []
        for start in range(0, total, self.points_per_batch):
            chunk = grid[start:start + self.points_per_batch]
            kept.extend(self._prompt_batch(rgb, chunk, embeddings, (h, w)))
            if on_progress:
                on_progress(min(start + len(chunk), total), total)

        return self._deduplicate(kept)

    def _prompt_batch(self, rgb, points, embeddings, size):
        import torch

        # The processor wants points nested [image][object][point][xy] and
        # labels one level shallower, [image][object][point]. One point per
        # object: each grid point is its own proposal, not a multi-point prompt.
        pts = [[[float(x), float(y)]] for x, y in points]
        labels = [[1] for _ in points]
        inputs = self._processor(images=rgb, input_points=[pts], input_labels=[labels],
                                 return_tensors="pt")
        out = self._model(
            input_points=inputs["input_points"], input_labels=inputs["input_labels"],
            image_embeddings=embeddings, multimask_output=True)

        logits = out.pred_masks[0]                       # (n_points, 3, h', w')
        iou = out.iou_scores[0]                          # (n_points, 3)
        stability = _stability_score(logits)             # (n_points, 3)

        good = (iou >= self.pred_iou_thresh) & (stability >= self.stability_score_thresh)
        if not bool(good.any()):
            return []

        # Only the accepted candidates are upsampled to full resolution. The
        # grid proposes three masks per point and most are rejected, so
        # upsampling first would be the dominant cost of the whole stage.
        idx = torch.nonzero(good, as_tuple=False)
        # post_process_masks wants (N, C, H, W) per image, so the accepted
        # candidates become N with a single channel each.
        chosen = logits[idx[:, 0], idx[:, 1]][:, None]
        full = self._processor.post_process_masks(
            [chosen], original_sizes=[size], binarize=True)[0]
        full = np.asarray(full).reshape(-1, *size).astype(bool)

        out_masks = []
        for k, (pi, mi) in enumerate(idx.tolist()):
            m = full[k]
            area = int(m.sum())
            if area < self.min_area_px:
                continue
            out_masks.append(Mask(
                segmentation=m, area_px=area, bbox=_bbox(m),
                predicted_iou=float(iou[pi, mi]),
                stability_score=float(stability[pi, mi]),
                point=(float(points[pi][0]), float(points[pi][1]))))
        return out_masks

    def _deduplicate(self, masks: list) -> list:
        """Box NMS. A point grid proposes the same object from several points."""
        if not masks:
            return []
        import torch
        from torchvision.ops import nms

        boxes = torch.tensor(
            [[m.bbox[0], m.bbox[1], m.bbox[0] + m.bbox[2], m.bbox[1] + m.bbox[3]]
             for m in masks], dtype=torch.float32)
        scores = torch.tensor([m.score for m in masks], dtype=torch.float32)
        keep = nms(boxes, scores, self.box_nms_thresh).tolist()
        # Largest first: the instance map is painted in this order so that a
        # small mask sitting inside a large one stays visible.
        return sorted((masks[i] for i in keep), key=lambda m: -m.area_px)
