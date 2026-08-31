"""Image-conditioned refinement of footprint boundaries.

The frameworks people reach for here - Unique3D, threefiner, Pixel2Mesh++ - all
do the same thing in principle: take a coarse mesh, take a reference image, and
deform the mesh's vertices toward evidence the image provides. None of them can
run on this data (see README section 6.2), but that *principle* transfers, and
this is the form of it that applies.

Where the evidence actually is. The mesh's structural boundaries come from SAM 2
masks, and SAM decodes its masks at 256x256 and upsamples: the outline is
right to a few metres and smooth at the scale where a roof edge is sharp. The
orthophoto is the same scene at full resolution, and a roof edge is one of the
strongest gradients in it. So the boundary is the thing the image can improve,
and the wall is the thing that boundary places.

The operator is guided-filter matting, which is the standard way to snap a
coarse segmentation onto image structure: filter the mask indicator with the
image as the guide, then re-threshold at 0.5. Inside a uniform region the fit is
dominated by the regularisation and the output is a local mean, so nothing
moves; across an image edge the fit follows the guide and the half level-set
migrates onto that edge.

**It is restricted to a narrow band around the original boundary**, and that
restriction is the whole safety argument. A guided filter given the run of the
image will happily flood a footprint across a road of similar colour. Confining
it to a few pixels either side means the refinement can sharpen a boundary and
cannot invent one, which is the same discipline README section 5.6 applied to
height refinement: it may move detail within a neighbourhood, it may not move
the neighbourhood.
"""
from __future__ import annotations

import numpy as np

# How far the boundary may move, in pixels. Two is roughly SAM's own decode
# resolution against a 1024 px scene; letting it move further stops being a
# snap and starts being a re-segmentation.
BAND_PX = 3
GUIDE_RADIUS = 4
GUIDE_EPS = 1e-4


def _band(mask: np.ndarray, width: int) -> np.ndarray:
    """The ring either side of a mask's boundary - the only place change is allowed."""
    from scipy.ndimage import binary_dilation, binary_erosion

    k = np.ones((2 * width + 1,) * 2, bool)
    return binary_dilation(mask, k) & ~binary_erosion(mask, k)


def snap(mask: np.ndarray, rgb: np.ndarray, band_px: int = BAND_PX,
         radius: int = GUIDE_RADIUS, eps: float = GUIDE_EPS) -> np.ndarray:
    """Move a footprint boundary onto the image edge it should be following.

    Returns a mask the same shape as the input, differing only within `band_px`
    of the original boundary.
    """
    from .refine import _luminance, guided_filter

    mask = np.asarray(mask, bool)
    if not mask.any() or mask.all():
        return mask

    guide = _luminance(rgb)
    soft = guided_filter(mask.astype(np.float32), guide, radius=radius, eps=eps)

    band = _band(mask, band_px)
    out = mask.copy()
    out[band] = soft[band] >= 0.5

    # A snap may not shatter the footprint: keep the largest 4-connected part,
    # and fill any hole the re-threshold punched in the interior.
    from scipy.ndimage import binary_fill_holes, label

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    lab, n = label(out, structure=cross)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        out = lab == int(sizes.argmax())
    if not out.any():
        return mask
    return binary_fill_holes(out)


# ------------------------------------------------------------------ scoring
def boundary_f1(pred: np.ndarray, truth: np.ndarray, tol_px: int = 2) -> dict:
    """How well two masks' outlines agree, within a tolerance.

    IoU rewards getting the bulk right and is nearly blind to a boundary that
    is two pixels out - which is precisely the quantity a snap is meant to
    move. The boundary F-score is not: it compares outlines directly, so it can
    say whether the walls got placed better even when the footprints barely
    changed area.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    def outline(m):
        return m & ~binary_erosion(m, np.ones((3, 3), bool))

    p, t = outline(np.asarray(pred, bool)), outline(np.asarray(truth, bool))
    if not p.any() or not t.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    dist_to_truth = distance_transform_edt(~t)
    dist_to_pred = distance_transform_edt(~p)
    precision = float((dist_to_truth[p] <= tol_px).mean())
    recall = float((dist_to_pred[t] <= tol_px).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return 0.0 if union == 0 else float((a & b).sum() / union)
