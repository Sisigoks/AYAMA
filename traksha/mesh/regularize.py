"""Staircased footprints to straight-sided polygons.

`mesh.structural` extrudes a *cell mask*: it walks `np.nonzero(cells)` and emits
one axis-aligned wall quad per cell edge on the boundary. Every footprint in the
delivered mesh is therefore a staircase at the raster pitch, and the source of
that raster is SAM 2, which decodes its masks at 256x256 and upsamples. A
building whose real outline is six straight sides comes out with several hundred
tiny steps, each of which is a wall the renderer shades separately.

That is what makes the buildings read as blobs rather than as buildings, and no
amount of texture work fixes it, because it is the geometry.

Three operators here, applied in that order.

**Trace.** Marching squares at the 0.5 level set gives a sub-pixel outline that
follows the mask's real boundary rather than its pixel corners. That alone
removes the staircase; it leaves a polygon with one vertex per boundary pixel.

**Simplify.** Douglas-Peucker at a tolerance in metres. This is where the vertex
count collapses - a European building goes from several hundred vertices to
between four and twenty - and the tolerance is the promise: no point on the
simplified outline is further than that from the traced one.

**Regularise.** A city building's sides are parallel or perpendicular to each
other. The dominant edge direction is estimated as a length-weighted circular
mean over the edges modulo 90 degrees, the polygon is rotated into that frame,
edges are classified as along-axis or across-axis, runs of the same class are
merged, and each corner is rebuilt as the intersection of its two edges. The
result is exactly rectilinear in a rotated frame - which is what a building is.

**Every one of the three can decline.** Regularisation is accepted only if it
did not move the footprint much: `MAX_AREA_CHANGE` on the area and
`MAX_VERTEX_SHIFT_M` on any single corner. A curved building, a circular tower
or a mask that caught two roofs will fail those and keep the simplified outline,
which is still far better than the staircase. Making a round building
rectangular would be exactly the plausible-looking fabrication this project
refuses everywhere else.

**OpenStreetMap enters here, as a prior and never as an authority.** `snap_to_osm`
finds the OSM footprint that best overlaps a mask and adopts its outline *only*
where the two already agree above an IoU threshold. Where they disagree the mask
wins, because the mask is what this image observed and OSM is a different
acquisition that can be metres off-register. What is adopted is the *shape*; the
decision that a building exists at all stays with SAM 2. Measured on the bundled
Zurich fixture, OSM footprints have 0.962 precision against lidar `nDSM > 2.5 m`
and their best alignment is at zero pixel shift, so on that scene the prior is
well-registered - but the gate is what makes that a finding rather than an
assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Douglas-Peucker tolerance, in metres. Half a metre is below the width of the
# render features this affects and above the sub-pixel wobble of a traced
# 0.5 level set, so it removes the staircase without rounding real corners.
SIMPLIFY_TOL_M = 0.6

# A regularised outline may not change the footprint's area by more than this
# fraction, nor move any single corner further than this many metres. Both are
# refusals, not clamps: the polygon is discarded and the simplified one kept.
MAX_AREA_CHANGE = 0.12
MAX_VERTEX_SHIFT_M = 2.5

# Below this many vertices after merging there is no rectilinear polygon to
# build - three edges cannot alternate between two axes.
MIN_REGULAR_VERTICES = 4

# How much of a SAM footprint an OSM polygon has to cover, and be covered by,
# before its outline is adopted. 0.5 is deliberately not generous: at that
# threshold the two outlines already agree about most of the building, and the
# regularised OSM shape is a refinement of a shared answer rather than a
# replacement for a different one.
OSM_IOU_ADOPT = 0.5


@dataclass
class Outline:
    """One footprint's boundary and the account of how it got that way."""

    ring: np.ndarray                        # (N, 2) float, (row, col)
    stage: str = "traced"                   # traced | simplified | regular | osm
    vertices_in: int = 0
    vertices_out: int = 0
    area_change: float = 0.0
    max_shift_m: float = 0.0
    angle_deg: Optional[float] = None
    osm_iou: Optional[float] = None
    refused: list = field(default_factory=list)

    def record(self) -> dict:
        d = {"stage": self.stage, "vertices_in": int(self.vertices_in),
             "vertices_out": int(self.vertices_out),
             "area_change": round(float(self.area_change), 4),
             "max_shift_m": round(float(self.max_shift_m), 3)}
        if self.angle_deg is not None:
            d["angle_deg"] = round(float(self.angle_deg), 2)
        if self.osm_iou is not None:
            d["osm_iou"] = round(float(self.osm_iou), 4)
        if self.refused:
            d["refused"] = list(self.refused)
        return d


# ------------------------------------------------------------------ tracing
def trace(mask: np.ndarray) -> Optional[np.ndarray]:
    """The mask's outer boundary as a sub-pixel ring, or None if it has none.

    Marching squares at 0.5, which is the level set halfway between inside and
    outside and therefore the boundary the mask actually asserts - not the
    staircase of pixel corners that `np.nonzero` walks.

    Only the longest contour is returned. A footprint with a courtyard traces
    two rings and `structural._solid` cannot express a hole, so the inner one is
    dropped here rather than silently confusing the caller.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    try:
        from skimage.measure import find_contours
    except ImportError:                                # pragma: no cover
        return _trace_fallback(mask)

    # Pad by one so a footprint touching the raster edge still closes.
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), float)
    padded[1:-1, 1:-1] = mask.astype(float)
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    ring = max(contours, key=len) - 1.0
    return np.asarray(ring, float)


def _trace_fallback(mask: np.ndarray) -> Optional[np.ndarray]:
    """Convex hull of the mask's pixels. Crude, and only for a missing skimage.

    Recorded as a fallback rather than dressed up: a convex hull is wrong for
    any L-shaped building, so this exists to keep the module importable, not to
    produce a result anyone should ship.
    """
    rows, cols = np.nonzero(mask)
    if rows.size < 3:
        return None
    pts = np.stack([rows, cols], 1).astype(float)
    try:
        from scipy.spatial import ConvexHull
    except ImportError:                                # pragma: no cover
        return None
    return pts[ConvexHull(pts).vertices]


# --------------------------------------------------------------- simplifying
def simplify(ring: np.ndarray, tol_px: float) -> np.ndarray:
    """Douglas-Peucker. No point of the result is further than `tol_px` from the input."""
    if len(ring) < 4:
        return ring
    try:
        from skimage.measure import approximate_polygon
    except ImportError:                                # pragma: no cover
        return _dp(ring, tol_px)
    out = approximate_polygon(np.asarray(ring, float), tolerance=float(tol_px))
    return out if len(out) >= 4 else ring


def _dp(ring: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker, written out for when scikit-image is not installed."""
    pts = np.asarray(ring, float)
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = pts[j] - pts[i]
        norm = np.hypot(*seg)
        if norm < 1e-12:
            d = np.hypot(*(pts[i + 1:j] - pts[i]).T)
        else:
            d = np.abs(np.cross(seg, pts[i + 1:j] - pts[i])) / norm
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


# -------------------------------------------------------------- regularising
def _closed(ring: np.ndarray) -> np.ndarray:
    """Drop a duplicated last vertex; every routine below assumes an open ring."""
    r = np.asarray(ring, float)
    if len(r) > 1 and np.allclose(r[0], r[-1]):
        r = r[:-1]
    return r


def polygon_area(ring: np.ndarray) -> float:
    """Absolute shoelace area, in the ring's own units squared."""
    r = _closed(ring)
    if len(r) < 3:
        return 0.0
    x, y = r[:, 1], r[:, 0]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def dominant_angle(ring: np.ndarray) -> float:
    """The building's own axis, in radians, in [0, pi/2).

    Every edge is folded into the first quadrant modulo 90 degrees, because a
    rectangle's four sides lie on two perpendicular directions and both should
    vote for the same axis. The vote is taken on the doubled angle so that
    directions 1 degree either side of the fold do not cancel, and it is
    weighted by edge length so a building's long walls decide its orientation
    and its little notches do not.
    """
    r = _closed(ring)
    d = np.roll(r, -1, axis=0) - r
    length = np.hypot(d[:, 0], d[:, 1])
    live = length > 1e-9
    if not live.any():
        return 0.0
    ang = np.arctan2(d[live, 0], d[live, 1]) % (np.pi / 2.0)
    w = length[live]
    # Circular mean over a pi/2 period: map to a full turn, average, map back.
    mean = np.arctan2(float((w * np.sin(4.0 * ang)).sum()),
                      float((w * np.cos(4.0 * ang)).sum())) / 4.0
    return float(mean % (np.pi / 2.0))


def _rot(ring: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a (row, col) ring so an edge at angle `phi` comes out at `phi - theta`.

    The sign is the whole content of this function and it was wrong once. Edge
    angles here are `arctan2(drow, dcol)`, measured from +col toward +row, so
    reducing an angle by `theta` needs the matrix below and not its transpose.
    Getting it backwards rotates *away* from the building's own axis, which
    still squares up an axis-aligned footprint - a 90 degree error maps the axes
    onto each other - and fails on every rotated one. That is exactly the shape
    of bug a test on a single upright rectangle does not catch.
    """
    c, s = np.cos(theta), np.sin(theta)
    r, k = ring[:, 0], ring[:, 1]
    return np.stack([c * r - s * k, s * r + c * k], 1)


def regularise(ring: np.ndarray, tol_px: float) -> tuple:
    """Rectilinear polygon in the building's own frame, or None if it will not fit.

    Returns `(ring_or_None, angle_rad, refusal_or_None)`. The refusal is a
    sentence, not a flag, because a caller that has to explain to a reader why a
    round tower kept its curved outline needs the reason and not a boolean.
    """
    r = _closed(np.asarray(ring, float))
    if len(r) < MIN_REGULAR_VERTICES:
        return None, 0.0, "fewer than four vertices to work with"

    theta = dominant_angle(r)
    local = _rot(r, theta)

    # Classify each edge by which axis it is closer to, in the rotated frame.
    d = np.roll(local, -1, axis=0) - local
    horizontal = np.abs(d[:, 1]) >= np.abs(d[:, 0])   # runs along +col

    # Merge runs of same-class edges into one. A run of three horizontal edges
    # is a staircase the simplifier left behind; collapsing it to a single line
    # at the length-weighted mean of their rows is what squares the corner.
    n = len(local)
    runs = []
    i = 0
    while i < n:
        j = i
        while (j + 1) < n and horizontal[(j + 1) % n] == horizontal[i]:
            j += 1
        runs.append((i, j, bool(horizontal[i])))
        i = j + 1
    # The ring wraps: if the last run has the same class as the first, join them.
    if len(runs) > 1 and runs[0][2] == runs[-1][2]:
        s, e, kind = runs[-1]
        runs = [(s, runs[0][1] + n, kind)] + runs[1:-1]
    if len(runs) < MIN_REGULAR_VERTICES:
        return None, theta, (f"only {len(runs)} alternating sides after merging; "
                             "not a rectilinear outline")
    if len(runs) % 2:
        return None, theta, ("an odd number of alternating sides, so the corners "
                             "cannot close")

    # Each run becomes one line: horizontal runs fix a row, vertical runs a col.
    lines = []
    for start, end, kind in runs:
        idx = np.arange(start, end + 2) % n
        pts = local[idx]
        seg = np.hypot(*(pts[1:] - pts[:-1]).T)
        w = np.concatenate([[0.0], seg]) + np.concatenate([seg, [0.0]])
        if w.sum() <= 0:
            w = np.ones(len(pts))
        axis = 0 if kind else 1
        lines.append((kind, float(np.average(pts[:, axis], weights=w))))

    # A corner is where consecutive lines meet: one supplies the row, the
    # other the col. Alternation is guaranteed by the merge above.
    corners = []
    for k in range(len(lines)):
        a, b = lines[k], lines[(k + 1) % len(lines)]
        row = a[1] if a[0] else b[1]
        col = b[1] if a[0] else a[1]
        corners.append((row, col))
    out = _rot(np.asarray(corners, float), -theta)
    return out, theta, None


# ------------------------------------------------------------------- the OSM
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def snap_to_osm(mask: np.ndarray, osm_rings: Sequence[np.ndarray],
                iou_threshold: float = OSM_IOU_ADOPT) -> tuple:
    """The OSM footprint this mask agrees with, or None.

    Returns `(ring_or_None, iou)`. Agreement is measured by IoU against the
    mask's own rasterisation, so the comparison is like-for-like and a polygon
    that merely overlaps a corner of the mask cannot win.

    Candidates are pre-filtered by bounding-box overlap because a scene has
    hundreds of OSM footprints and rasterising each against every SAM mask is
    quadratic for no reason.
    """
    from ..data.osm import fill_polygon

    mask = np.asarray(mask, bool)
    if not mask.any() or not osm_rings:
        return None, 0.0
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()

    best, best_iou = None, 0.0
    for ring in osm_rings:
        rr0, cc0 = ring.min(0)
        rr1, cc1 = ring.max(0)
        if rr1 < r0 or rr0 > r1 or cc1 < c0 or cc0 > c1:
            continue
        other = fill_polygon(ring, mask.shape)
        v = _iou(mask, other)
        if v > best_iou:
            best, best_iou = ring, v
    if best is None or best_iou < iou_threshold:
        return None, best_iou
    return best, best_iou


# ------------------------------------------------------------------ the whole
def outline(
    mask: np.ndarray,
    gsd_m: float,
    *,
    osm_rings: Optional[Sequence[np.ndarray]] = None,
    simplify_tol_m: float = SIMPLIFY_TOL_M,
    max_area_change: float = MAX_AREA_CHANGE,
    max_shift_m: float = MAX_VERTEX_SHIFT_M,
    iou_threshold: float = OSM_IOU_ADOPT,
    allow_regularise: bool = True,
) -> Optional[Outline]:
    """Trace, simplify, optionally adopt an OSM shape, optionally square it up.

    Returns None only when the mask has no traceable boundary at all. Every
    other path returns an `Outline` that records which stage it reached and
    which stages declined, so a run can report "31 of 100 footprints were
    squared up" instead of implying it did something uniform.
    """
    gsd = max(float(gsd_m), 1e-6)
    ring = trace(mask)
    if ring is None or len(ring) < 4:
        return None

    out = Outline(ring=_closed(ring), stage="traced", vertices_in=len(ring))
    traced_area = polygon_area(out.ring)

    simplified = simplify(out.ring, simplify_tol_m / gsd)
    if len(simplified) >= 4:
        out.ring, out.stage = _closed(simplified), "simplified"

    if osm_rings:
        osm_ring, iou = snap_to_osm(mask, osm_rings, iou_threshold)
        out.osm_iou = iou
        if osm_ring is not None:
            out.ring, out.stage = _closed(osm_ring), "osm"
        elif iou > 0:
            out.refused.append(
                f"the best OSM footprint overlaps at IoU {iou:.2f}, under the "
                f"{iou_threshold:.2f} needed to adopt its shape")

    if allow_regularise:
        candidate, theta, refusal = regularise(out.ring, simplify_tol_m / gsd)
        if refusal:
            out.refused.append(refusal)
        elif candidate is not None:
            area = polygon_area(candidate)
            change = abs(area - traced_area) / max(traced_area, 1e-9)
            shift = _max_shift(out.ring, candidate) * gsd
            if change > max_area_change:
                out.refused.append(
                    f"squaring it up would change the footprint area by "
                    f"{change * 100:.1f}%, over the {max_area_change * 100:.0f}% limit")
            elif shift > max_shift_m:
                out.refused.append(
                    f"squaring it up would move a corner by {shift:.1f} m, over "
                    f"the {max_shift_m:.1f} m limit")
            else:
                out.ring, out.stage = candidate, "regular"
                out.area_change, out.max_shift_m = change, shift
                out.angle_deg = float(np.degrees(theta))

    out.vertices_out = len(out.ring)
    return out


def _max_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Furthest any point of `b` sits from the polyline `a`, in pixels.

    Point-to-*segment*, not point-to-vertex: a squared-up corner legitimately
    lands between two traced vertices, and measuring to the nearest vertex would
    report that as a large move when the outline barely shifted.
    """
    a = _closed(np.asarray(a, float))
    b = _closed(np.asarray(b, float))
    if len(a) < 2 or len(b) == 0:
        return float("inf")
    p0 = a
    p1 = np.roll(a, -1, axis=0)
    seg = p1 - p0
    denom = (seg ** 2).sum(1)
    denom[denom < 1e-12] = 1e-12
    worst = 0.0
    for pt in b:
        t = np.clip(((pt - p0) * seg).sum(1) / denom, 0.0, 1.0)
        proj = p0 + t[:, None] * seg
        worst = max(worst, float(np.hypot(*(pt - proj).T).min()))
    return worst


def rasterise(ring: np.ndarray, shape: tuple) -> np.ndarray:
    """A polygon back to a pixel mask, for the extruder to walk."""
    from ..data.osm import fill_polygon

    return fill_polygon(np.asarray(ring, float), shape)
