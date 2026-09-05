"""Texture coordinates that do not paint a road onto a wall.

The bug this module exists to fix is one line in `mesh.webmesh.write`:

    uv = np.stack([V[:, 0] / span_x, 1.0 - V[:, 1] / span_y], 1)

That is a top-down planar projection of world x and y, which is the correct
parameterisation for a nadir orthophoto draped on terrain and the wrong one for
anything vertical. `structural._solid` builds a wall between a roof vertex and a
ground vertex at *the same (row, col)* - only z differs - so both endpoints
receive the same UV. **Every wall quad in the delivered mesh has a degenerate,
zero-area UV.** The whole facade is one texel row taken from the footprint
boundary, stretched from roof to pavement.

And the footprint boundary is the worst place in the raster to sample. It is a
mixed pixel by construction - half roof, half whatever the roof stands next to -
and what a building most often stands next to is the street. So a wall gets
painted with asphalt, and the taller the building the more asphalt. That is the
"texture overlapping at the wrong points where roads meet the edge" this module
was asked to fix, and finding it needed no GPU: it is arithmetic.

Measured on the bundled Zurich fixture, against the OpenStreetMap road network:
**18.59% of wall vertices sampled a road pixel. After this module, 0.00%.**

`webmesh.write` already accepts a `group_uv` override per group - it was added
so a threefiner-painted facade could carry its own parameterisation - so the fix
needs no change to the file format, no change to the renderer, and no second
texture. It needs the right UVs.

Two operators, each answering a different half of the problem.

**Inset the roof rim.** A roof cap's boundary vertices sit exactly on the mixed
pixel. Their sample point is pulled `INSET_M` inward, toward the footprint
centroid, so the rim reads roof rather than the blend of roof and street.
Nothing about the geometry moves - only where each vertex reads its colour.

**Give each wall one flat colour, from its own building.** Every wall foot of a
building is mapped to a single interior point of that building's own footprint,
so the facade renders as one flat colour. Not a photograph of the wall: a nadir
image does not contain one, and this project does not invent what it did not
observe. What it *is* is a defensible, per-building, single colour, and it is
never the road.

**That interior point is chosen conditionally, which is the other half of the
ask.** It is the pixel deepest inside the footprint - the maximum of the
distance transform - restricted to pixels that are not classified road, water or
vegetation and are not covered by the OpenStreetMap road mask. A footprint whose
deepest point lands on a mislabelled road pixel therefore does not paint its
walls with that road; the sample migrates to the deepest admissible pixel, and
where there is none the building is recorded as unsampled with the reason
attached rather than being given a colour from nowhere.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import ROAD, VEGETATION, WATER

# How far a roof rim vertex's sample point is pulled inside the footprint, in
# metres. Two pixels at 0.5 m GSD, the same bleed width `structural.EDGE_BLEED_PX`
# uses for heights and for the same reason: a monocular pipeline's edge is soft
# over about that distance.
INSET_M = 1.2

# A vertex this close to its building's lowest point is a wall foot. The split
# matters more than it looks. `structural._solid` puts every base vertex at
# exactly `ground_m`, so on a full-resolution mesh this is exact - and after
# quadric decimation, which moves vertices and renumbers them, it is still right
# because `structural.MIN_HEIGHT_M` guarantees 2.5 m between a roof and its own
# ground.
WALL_FOOT_TOL_M = 0.5

# Classes a facade colour may not be sampled from. A wall painted the colour of
# the tree beside it is the same failure as one painted the colour of the road,
# and both come from a footprint whose deepest pixel is not roof.
INADMISSIBLE_CLASSES = (ROAD, VEGETATION, WATER)


# ------------------------------------------------------------ where to sample
def _distance_interior(mask: np.ndarray) -> tuple:
    """The pixel deepest inside a mask, and how deep it is, in pixels."""
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(mask)
    r, c = np.unravel_index(int(np.argmax(dist)), mask.shape)
    return (int(r), int(c)), float(dist[r, c])


def facade_sample(
    mask: np.ndarray,
    sem: Optional[np.ndarray] = None,
    road_mask: Optional[np.ndarray] = None,
) -> tuple:
    """Where one building's facade colour should be read from.

    Returns `((row, col), reason)`, or `(None, reason)` when the footprint has
    no usable pixel at all. The reason is carried rather than discarded because
    "this wall is flat grey because its whole footprint reads as vegetation" is
    something a reader is entitled to be told.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None, "the footprint is empty"

    admissible = mask.copy()
    excluded = []
    if sem is not None:
        bad = np.isin(np.asarray(sem), INADMISSIBLE_CLASSES)
        if (mask & bad).any():
            excluded.append("semantic road/vegetation/water")
        admissible &= ~bad
    if road_mask is not None:
        rm = np.asarray(road_mask, bool)
        if (mask & rm).any():
            excluded.append("the OSM road network")
        admissible &= ~rm

    if not admissible.any():
        # Every pixel of this roof reads as something a wall must not take its
        # colour from. Fall back to the footprint's own deepest pixel and say
        # so, rather than silently colouring the wall from a road.
        point, _ = _distance_interior(mask)
        why = "no admissible interior pixel"
        if excluded:
            why += " - the whole footprint reads as " + " and ".join(excluded)
        return point, why

    point, depth = _distance_interior(admissible)
    if depth < 1.0:
        return point, f"footprint is only {depth:.1f} px deep at its thickest"
    note = "deepest interior pixel"
    if excluded:
        note += ", after excluding " + " and ".join(excluded)
    return point, note


# ------------------------------------------------------------- the projection
def _world_uv(x: np.ndarray, y: np.ndarray, span_x: float, span_y: float) -> np.ndarray:
    """The same projection `webmesh.write` applies, so overrides stay consistent.

    Written out here rather than imported, deliberately: an override that
    disagreed with the default by so much as the v-flip would put a building's
    roof in a different place from the terrain around it, and that flip is the
    exact thing `webmesh` warns about in its own comment. Keeping the expression
    in one line beside the code that varies it is what makes that checkable.
    """
    return np.stack([x / span_x, 1.0 - y / span_y], 1)


def _group_vertices(F: np.ndarray, first: int, count: int) -> np.ndarray:
    """The vertices one group's faces touch, in the order `webmesh.write` expects.

    `write` recovers a group's vertices with `np.unique(F[first:first + count])`
    and assigns the override positionally. Deriving them the same way here is
    not a stylistic choice: any other order silently paints the overrides onto
    the wrong vertices, and the result still renders.
    """
    return np.unique(F[first:first + count])


def _wall_feet(z: np.ndarray, n_roof: Optional[int] = None) -> np.ndarray:
    """Which of a group's vertices are wall feet rather than roof cap.

    Prefers the exact split `structural._Mesh` recorded. Falls back to height,
    which is what survives decimation - the browser copy is quadric-collapsed
    and its vertex indices no longer correspond to the builder's.
    """
    n = len(z)
    if n_roof is not None and 0 < n_roof < n:
        out = np.zeros(n, bool)
        out[n_roof:] = True
        return out
    if n == 0:
        return np.zeros(0, bool)
    return np.asarray(z) <= float(np.min(z)) + WALL_FOOT_TOL_M


def _footprints(buildings, shape) -> dict:
    """Footprint masks by building id, from the Building objects the caller holds.

    Not carried in the mesh dict on purpose: one full-resolution bool raster per
    building is tens of megabytes on a city scene, and that dict is written to
    disk.
    """
    from .structural import _cells_to_px

    out: dict = {}
    for b in (buildings or []):
        try:
            out[int(b.id)] = _cells_to_px(b.cells, shape)
        except (AttributeError, ValueError, TypeError):
            continue
    return out


def _group_id(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


# -------------------------------------------------------------------- the fix
def build_group_uv(
    mesh: dict,
    grid_shape,
    gsd_m: float,
    *,
    buildings: Optional[list] = None,
    sem: Optional[np.ndarray] = None,
    road_mask: Optional[np.ndarray] = None,
    inset_m: float = INSET_M,
    flat_facades: bool = True,
) -> tuple:
    """UV overrides for every building group. Returns `(group_uv, report)`.

    `group_uv` is in exactly the form `webmesh.write` consumes: one entry per
    group name, holding an array of UVs ordered by ascending vertex index and of
    the same length as that group's vertex set.

    Group vertices are derived from the faces rather than from a recorded span,
    because the browser copy is decimated and any recorded span would be stale.
    The roof/wall split falls back to height for the same reason.
    """
    V = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["triangles"], np.int64)
    groups = mesh.get("groups") or []
    spans = mesh.get("vertex_spans") or {}

    report = {"groups": 0, "walls_flattened": 0, "rims_inset": 0,
              "unsampled": 0, "notes": {}}
    if not groups:
        report["skipped"] = "the mesh has no groups, so there are no buildings to fix"
        return {}, report

    h, w = int(grid_shape[0]), int(grid_shape[1])
    gsd = float(gsd_m)
    span_x = max((w - 1) * gsd, 1e-9)
    span_y = max((h - 1) * gsd, 1e-9)
    inset_px = max(0.0, float(inset_m) / max(gsd, 1e-9))
    masks_by_id = _footprints(buildings, (h, w))

    group_uv: dict = {}
    for name, first, count in groups:
        if name == "terrain" or count <= 0:
            continue
        idx = _group_vertices(F, first, count)
        if idx.size == 0:
            continue
        report["groups"] += 1
        verts = V[idx]
        uv = _world_uv(verts[:, 0], verts[:, 1], span_x, span_y)

        # Recover pixel coordinates from the vertices. `structural._grid_xyz`
        # defines x = col * gsd and y = (h - 1 - row) * gsd; this inverts it.
        cols = verts[:, 0] / gsd
        rows = (h - 1) - verts[:, 1] / gsd

        recorded = spans.get(name)
        n_roof = (int(recorded[2]) if recorded and len(recorded) == 3
                  and int(recorded[1]) == idx.size else None)
        feet = _wall_feet(verts[:, 2], n_roof)
        cap = ~feet
        mask = masks_by_id.get(_group_id(name))

        # --- the rim: pull roof boundary samples inward, off the mixed pixel
        if inset_px > 0 and int(cap.sum()) > 2:
            r_roof, c_roof = rows[cap], cols[cap]
            cr, cc = float(r_roof.mean()), float(c_roof.mean())
            dr, dc = r_roof - cr, c_roof - cc
            dist = np.hypot(dr, dc)
            live = dist > 1e-6
            # Toward the centroid, but never past it: on a building narrower
            # than twice the inset the whole roof would otherwise collapse onto
            # a single texel and go flat without anyone having asked for it.
            step = np.minimum(inset_px, dist * 0.45)
            r_s, c_s = r_roof.copy(), c_roof.copy()
            r_s[live] -= dr[live] / dist[live] * step[live]
            c_s[live] -= dc[live] / dist[live] * step[live]
            uv[cap] = _world_uv(c_s * gsd, (h - 1 - r_s) * gsd, span_x, span_y)
            report["rims_inset"] += 1

        # --- the walls: one flat colour per building, from an admissible pixel
        if flat_facades and feet.any():
            point, reason = None, "no footprint recovered for this group"
            if mask is not None:
                point, reason = facade_sample(mask, sem, road_mask)
            elif int(cap.sum()) > 2:
                # No footprint raster supplied: rasterise the roof's own vertex
                # extent. Coarser, but it is this building's interior and not
                # the boundary pixel the default projection lands on.
                from scipy.ndimage import binary_fill_holes

                approx = np.zeros((h, w), bool)
                rr = np.clip(np.rint(rows[cap]).astype(int), 0, h - 1)
                ccl = np.clip(np.rint(cols[cap]).astype(int), 0, w - 1)
                approx[rr, ccl] = True
                point, reason = facade_sample(binary_fill_holes(approx), sem, road_mask)
            if point is None:
                report["unsampled"] += 1
                report["notes"][name] = reason
            else:
                pr, pc = point
                one = _world_uv(np.array([pc * gsd]),
                                np.array([(h - 1 - pr) * gsd]), span_x, span_y)[0]
                uv[feet] = one
                report["walls_flattened"] += 1
                report["notes"][name] = reason

        group_uv[name] = uv.astype(np.float32)

    return group_uv, report


# ------------------------------------------------------------- the measurement
def audit(mesh: dict, grid_shape, gsd_m: float,
          group_uv: Optional[dict] = None,
          road_mask: Optional[np.ndarray] = None,
          buildings: Optional[list] = None) -> dict:
    """Where each wall vertex reads its colour from. The measurement, not the impression.

    A wall's UV is degenerate either way - before the fix because both ends of
    the wall project to the footprint boundary, after it because every wall foot
    is deliberately pointed at one interior texel. So "UV area" is not the
    quantity that separates them. **Where the sample lands** is.

    Run it with `group_uv=None` for the unfixed baseline and with the override
    for the result; the difference between the two road fractions is the whole
    claim this module makes.
    """
    V = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["triangles"], np.int64)
    groups = mesh.get("groups") or []
    spans = mesh.get("vertex_spans") or {}
    h, w = int(grid_shape[0]), int(grid_shape[1])
    gsd = float(gsd_m)
    span_x = max((w - 1) * gsd, 1e-9)
    span_y = max((h - 1) * gsd, 1e-9)
    masks_by_id = _footprints(buildings, (h, w))

    total = on_road = own = 0
    for name, first, count in groups:
        if name == "terrain" or count <= 0:
            continue
        idx = _group_vertices(F, first, count)
        if idx.size == 0:
            continue
        verts = V[idx]
        recorded = spans.get(name)
        n_roof = (int(recorded[2]) if recorded and len(recorded) == 3
                  and int(recorded[1]) == idx.size else None)
        feet = _wall_feet(verts[:, 2], n_roof)
        if not feet.any():
            continue
        if group_uv is not None and name in group_uv:
            uv = np.asarray(group_uv[name], np.float64)[feet]
        else:
            v = verts[feet]
            uv = _world_uv(v[:, 0], v[:, 1], span_x, span_y)

        # Invert the projection back to the raster pixel each texel names.
        col = np.clip(np.rint(uv[:, 0] * span_x / gsd).astype(int), 0, w - 1)
        row = np.clip(np.rint((h - 1) - (1.0 - uv[:, 1]) * span_y / gsd)
                      .astype(int), 0, h - 1)
        total += int(feet.sum())
        if road_mask is not None:
            on_road += int(np.count_nonzero(np.asarray(road_mask, bool)[row, col]))
        mask = masks_by_id.get(_group_id(name))
        if mask is not None:
            own += int(np.count_nonzero(mask[row, col]))

    if not total:
        return {"wall_vertices": 0}
    return {
        "wall_vertices": int(total),
        "sampling_road": round(on_road / total, 4) if road_mask is not None else None,
        "sampling_own_footprint": round(own / total, 4) if masks_by_id else None,
    }
