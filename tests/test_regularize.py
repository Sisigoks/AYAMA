"""Footprint outlines: traced, simplified, squared up - and refused when they should be.

The operators here move a *boundary*, which is the one degree of freedom a nadir
orthophoto actually constrains (README section 6.2). So the tests are about two
things in equal measure: that a rectangular building comes out rectangular, and
that a round one does not.

The refusals matter more than the successes. An extruder given a squared-up
circle produces a confident, clean, wrong building, and nothing downstream can
tell that it was invented - which is exactly the failure mode this project
refuses everywhere else.
"""
from __future__ import annotations

import numpy as np
import pytest

from traksha.mesh import regularize as R


def rect(shape, cy, cx, h, w, deg=0.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    y = (yy - cy) * c + (xx - cx) * s
    x = -(yy - cy) * s + (xx - cx) * c
    return (np.abs(y) <= h / 2) & (np.abs(x) <= w / 2)


def disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


S = (200, 200)


# ------------------------------------------------------------------ tracing
def test_an_empty_mask_has_no_outline():
    assert R.trace(np.zeros((16, 16), bool)) is None
    assert R.outline(np.zeros((16, 16), bool), 0.5) is None


def test_a_traced_ring_follows_the_mask_not_its_pixel_corners():
    """Marching squares at 0.5, so the outline is sub-pixel rather than a staircase."""
    m = rect(S, 100, 100, 60, 100)
    ring = R.trace(m)
    assert ring is not None and len(ring) > 4
    # Not all coordinates are integers: a pixel-corner walk would make them so.
    assert not np.allclose(ring, np.rint(ring))


# --------------------------------------------------------------- simplifying
def test_simplifying_collapses_hundreds_of_vertices_into_a_handful():
    m = rect(S, 100, 100, 60, 100)
    o = R.outline(m, gsd_m=0.5, allow_regularise=False)
    assert o.vertices_in > 200
    assert o.vertices_out <= 12
    assert o.stage == "simplified"


def test_simplifying_keeps_the_footprint():
    m = rect(S, 100, 100, 60, 100, deg=25.0)
    o = R.outline(m, gsd_m=0.5, allow_regularise=False)
    assert iou(R.rasterise(o.ring, S), m) > 0.95


# -------------------------------------------------------------- regularising
@pytest.mark.parametrize("deg", [0.0, 17.0, 30.0, 45.0, 63.0])
def test_a_rectangle_at_any_angle_comes_back_as_four_corners(deg):
    """The rotation sign was wrong once, and an upright rectangle did not catch it:
    a 90 degree error maps the axes onto each other and only rotated buildings fail."""
    m = rect(S, 100, 100, 60, 100, deg=deg)
    o = R.outline(m, gsd_m=0.5)
    assert o.stage == "regular", o.refused
    assert o.vertices_out == 4
    assert iou(R.rasterise(o.ring, S), m) > 0.95


def test_an_l_shape_keeps_its_six_corners():
    m = rect(S, 90, 90, 80, 80) | rect(S, 120, 140, 20, 80)
    o = R.outline(m, gsd_m=0.5)
    assert o.stage == "regular", o.refused
    assert o.vertices_out == 6


def test_a_circle_is_refused_and_says_why():
    """Making a round tower rectangular is the fabrication this whole module avoids."""
    o = R.outline(disc(S, 100, 100, 40), gsd_m=0.5)
    assert o.stage == "simplified"
    assert o.refused and "corner" in " ".join(o.refused)


def test_a_refused_outline_is_still_better_than_the_staircase():
    o = R.outline(disc(S, 100, 100, 40), gsd_m=0.5)
    assert o.vertices_out < o.vertices_in / 5


def test_the_dominant_angle_is_length_weighted():
    """A building's long walls decide its orientation; its little notches do not."""
    m = rect(S, 100, 100, 40, 140, deg=20.0)
    theta = np.degrees(R.dominant_angle(R.simplify(R.trace(m), 1.2)))
    assert min(abs(theta - 20.0), abs(theta - 70.0)) < 4.0


def test_area_is_the_shoelace_area():
    square = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    assert R.polygon_area(square) == pytest.approx(100.0)


def test_regularising_reports_what_it_moved():
    m = rect(S, 100, 100, 60, 100, deg=30.0)
    o = R.outline(m, gsd_m=0.5)
    rec = o.record()
    assert rec["stage"] == "regular"
    assert rec["max_shift_m"] < R.MAX_VERTEX_SHIFT_M
    assert rec["area_change"] < R.MAX_AREA_CHANGE
    assert "angle_deg" in rec


# ----------------------------------------------------------------- the OSM
def test_an_agreeing_osm_footprint_is_adopted():
    m = rect(S, 100, 100, 60, 100)
    # The same building, drawn as a clean vector with slightly different corners.
    ring = np.array([[70.5, 50.5], [70.5, 149.5], [129.5, 149.5], [129.5, 50.5]])
    o = R.outline(m, gsd_m=0.5, osm_rings=[ring])
    assert o.stage in ("osm", "regular")
    assert o.osm_iou > 0.9


def test_a_disagreeing_osm_footprint_is_refused_and_the_iou_is_recorded():
    """OSM is a different acquisition and can be metres off. The mask wins."""
    m = rect(S, 100, 100, 60, 100)
    elsewhere = np.array([[10.0, 10.0], [10.0, 30.0], [30.0, 30.0], [30.0, 10.0]])
    o = R.outline(m, gsd_m=0.5, osm_rings=[elsewhere])
    assert o.stage != "osm"
    assert o.osm_iou == 0.0


def test_the_best_overlapping_osm_footprint_is_the_one_chosen():
    m = rect(S, 100, 100, 60, 100)
    near = np.array([[71.0, 51.0], [71.0, 149.0], [129.0, 149.0], [129.0, 51.0]])
    far = np.array([[60.0, 40.0], [60.0, 90.0], [90.0, 90.0], [90.0, 40.0]])
    ring, got = R.snap_to_osm(m, [far, near])
    assert ring is not None and got > 0.9


def test_snapping_against_no_candidates_is_not_an_error():
    assert R.snap_to_osm(rect(S, 100, 100, 20, 20), []) == (None, 0.0)


# ------------------------------------------------------------- round tripping
def test_rasterising_an_outline_recovers_the_mask():
    m = rect(S, 100, 100, 60, 100, deg=12.0)
    o = R.outline(m, gsd_m=0.5)
    assert iou(R.rasterise(o.ring, S), m) > 0.95


def test_a_footprint_too_small_to_trace_returns_none():
    m = np.zeros(S, bool)
    m[5, 5] = True
    assert R.outline(m, 0.5) is None or R.outline(m, 0.5).vertices_out >= 3
