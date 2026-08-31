"""Image-conditioned footprint refinement: the operator and its scoring.

The snap is kept as a diagnostic and not shipped as a stage, because measured
over four scenes it does not help (README section 6.2). These tests pin what it
does mechanically, so the negative result stays reproducible rather than
becoming folklore, and they pin the boundary score - which is a keeper whatever
happens to the snap, because it is what makes "the walls are better placed" a
measurable claim instead of an impression.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from traksha.mesh import footprint as FP  # noqa: E402


def square(shape=(64, 64), r0=16, c0=16, r1=48, c1=48):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def image_for(mask, inside=40, outside=210):
    """A picture whose only edge is the mask's own boundary."""
    rgb = np.full((*mask.shape, 3), outside, np.uint8)
    rgb[mask] = inside
    return rgb


# ---------------------------------------------------------------- the snap
def test_a_boundary_already_on_the_edge_is_left_alone():
    m = square()
    out = FP.snap(m, image_for(m))
    assert int((m ^ out).sum()) == 0


def test_a_boundary_off_by_a_pixel_is_pulled_back_onto_the_edge():
    truth = square()
    drifted = square(r0=15, c0=15, r1=47, c1=47)      # one pixel up and left
    out = FP.snap(drifted, image_for(truth))
    assert FP.iou(out, truth) > FP.iou(drifted, truth), "the snap moved nothing useful"


def test_the_snap_cannot_move_the_boundary_further_than_its_band():
    """The safety argument: it may sharpen a boundary, it may not invent one.

    Given the run of the image a guided filter will flood a footprint across a
    road of similar colour, so the change is confined to a ring either side of
    the original outline.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    m = square()
    rgb = np.random.default_rng(0).integers(0, 255, (*m.shape, 3), dtype=np.uint8)
    out = FP.snap(m, rgb, band_px=2)
    k = np.ones((5, 5), bool)
    assert (out & ~binary_dilation(m, k)).sum() == 0, "grew past the band"
    assert (binary_erosion(m, k) & ~out).sum() == 0, "shrank past the band"


def test_the_result_stays_one_piece():
    """A re-threshold can shatter a footprint; a shattered footprint is not a building."""
    m = square()
    rgb = np.random.default_rng(1).integers(0, 255, (*m.shape, 3), dtype=np.uint8)
    out = FP.snap(m, rgb, band_px=3)
    from scipy.ndimage import label

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    _, n = label(out, structure=cross)
    assert n == 1


def test_degenerate_masks_come_back_unchanged():
    empty = np.zeros((32, 32), bool)
    full = np.ones((32, 32), bool)
    rgb = np.zeros((32, 32, 3), np.uint8)
    assert not FP.snap(empty, rgb).any()
    assert FP.snap(full, rgb).all()


# ------------------------------------------------------------- the scoring
def test_identical_outlines_score_one():
    m = square()
    assert FP.boundary_f1(m, m, tol_px=1)["f1"] == pytest.approx(1.0)
    assert FP.iou(m, m) == pytest.approx(1.0)


def test_the_boundary_score_sees_a_shift_that_iou_barely_notices():
    """Which is the whole reason it is the metric the snap is judged by."""
    # A building-sized footprint, so a three-pixel outline error is the small
    # relative change it really is - which is exactly when IoU stops noticing.
    shape = (160, 160)
    truth = square(shape, 20, 20, 140, 140)
    shifted = square(shape, 17, 17, 137, 137)         # three pixels out
    assert FP.iou(shifted, truth) > 0.8, "IoU is supposed to shrug at this"
    assert FP.boundary_f1(shifted, truth, tol_px=1)["f1"] < 0.3


def test_the_tolerance_does_what_it_says():
    truth = square()
    shifted = square(r0=14, c0=14, r1=46, c1=46)      # two pixels out
    tight = FP.boundary_f1(shifted, truth, tol_px=1)["f1"]
    loose = FP.boundary_f1(shifted, truth, tol_px=3)["f1"]
    assert loose > tight


def test_an_empty_mask_scores_zero_rather_than_raising():
    assert FP.boundary_f1(np.zeros((16, 16), bool), square((16, 16), 2, 2, 8, 8))["f1"] == 0.0
    assert FP.iou(np.zeros((16, 16), bool), np.zeros((16, 16), bool)) == 0.0
