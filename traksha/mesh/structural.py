"""Mesh to better mesh: cut the sheet into terrain and separate building solids.

The delivered surface is a single connected height field. Every retained grid
node becomes a vertex and the whole grid is triangulated as one manifold, so a
twenty-metre facade is a *ramp* - one triangle spanning one ground sample
horizontally and twenty metres vertically, welded to the roof at one end and to
the pavement at the other. Two buildings either side of a four-metre alley are
joined by a continuous strip that dips into the alley and back out. Nothing in
the mesh corresponds to a building.

That is a topology defect, and it is why buildings read as bumps under a
blanket. No depth model and no calibration can fix it: the heights are already
there, and the triangulation throws the structure away.

So this stage rebuilds the mesh from the same calibrated heights:

    terrain    every cell that is not inside a building footprint
    roofs      one cap per building, at the calibrated surface
    walls      explicit vertical quads down the footprint boundary
               to the local ground

and emits each building as **its own connected component**, sharing no vertex
with the terrain or with its neighbours. Separation then becomes a property of
the mesh that can be measured rather than a look that can be argued about.

Three things this deliberately does not do.

**It does not invent height.** Every vertex z is the calibrated DSM, the local
DTM under it, or a plane fitted to calibrated values. There is no offset applied
to make buildings look raised - the height was always there, and the ramp was
hiding it.

**It does not touch the calibrated rasters.** `dsm.tif`, `ndsm.tif` and
`sigma.tif` are the scientific output and keep their values and provenance. This
is a render-space product derived from them, written beside them and labelled as
such.

**It does not force roofs flat.** A plane is fitted per roof and used only where
the roof really is planar, measured by inlier fraction and residual. Pitched and
complex roofs keep their calibrated shape, because flattening them would be
exactly the kind of plausible-looking fabrication the rest of this project
refuses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1

# A structure has to clear this much local ground to be treated as a building.
# Below it the footprint is more likely a kerb, a parked car or segmentation
# leakage, and extruding it would put walls around noise.
MIN_HEIGHT_M = 2.5
MIN_AREA_PX = 40

# Roof planarity. `PLANE_TOL_M` is how far a vertex may sit from the fitted
# plane and still count as an inlier; `PLANE_INLIER_FRAC` is how much of the
# roof has to be inliers before the plane is used at all.
PLANE_TOL_M = 0.6
PLANE_INLIER_FRAC = 0.80

# Robust spread. A roof whose calibrated heights scatter more than this, after
# MAD, is not a coherent surface - usually vegetation overhanging the mask, or a
# mask that crossed a height discontinuity - and is reported rather than trusted.
MAX_ROOF_MAD_M = 4.0


@dataclass
class Building:
    """One instance that carries enough evidence to be extruded."""

    id: int
    source_instance: int
    cells: np.ndarray                      # (H-1, W-1) bool, footprint in cell space
    area_px: int
    roof_median_m: float
    roof_mad_m: float
    ground_m: float
    height_m: float
    outlier_ratio: float
    planar: bool
    plane: Optional[tuple] = None          # (a, b, c) with z = a*col + b*row + c
    confidence: float = 0.0
    note: str = ""

    def record(self) -> dict:
        return {
            "id": self.id, "source_instance": self.source_instance,
            "area_px": self.area_px,
            "roof_median_m": round(self.roof_median_m, 3),
            "roof_mad_m": round(self.roof_mad_m, 3),
            "ground_m": round(self.ground_m, 3),
            "height_m": round(self.height_m, 3),
            "outlier_ratio": round(self.outlier_ratio, 4),
            "planar": bool(self.planar),
            "confidence": round(self.confidence, 4),
            "note": self.note,
        }


# --------------------------------------------------------------- statistics
def _mad(a: np.ndarray) -> float:
    """Median absolute deviation. Robust where a mean is not.

    A roof mask that clips a neighbouring tower, or catches a tree, moves a mean
    by metres and a median by centimetres. Every height decision below is made
    on medians for that reason.
    """
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def _fit_plane(rows: np.ndarray, cols: np.ndarray, z: np.ndarray):
    """Least squares z = a*col + b*row + c, refit once without outliers.

    Not full RANSAC: the support is a roof mask that is already mostly roof, so
    one reweighting pass on the median residual removes the chimney and the
    clipped neighbour without the cost or the randomness. Returns
    (plane, inlier_fraction, residual_rmse).
    """
    if z.size < 8:
        return None, 0.0, float("inf")
    A = np.stack([cols, rows, np.ones_like(cols)], axis=1).astype(np.float64)
    try:
        coef, *_ = np.linalg.lstsq(A, z.astype(np.float64), rcond=None)
        resid = z - A @ coef
        keep = np.abs(resid) <= max(PLANE_TOL_M, 3.0 * _mad(resid))
        if keep.sum() >= 8 and keep.sum() < z.size:
            coef, *_ = np.linalg.lstsq(A[keep], z[keep].astype(np.float64), rcond=None)
            resid = z - A @ coef
    except np.linalg.LinAlgError:
        return None, 0.0, float("inf")
    inliers = np.abs(resid) <= PLANE_TOL_M
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return tuple(float(v) for v in coef), float(inliers.mean()), rmse


def _to_cells(mask: np.ndarray) -> np.ndarray:
    """Pixel mask to cell mask: a cell belongs to the footprint only if all four
    of its corners do.

    The conservative choice on purpose. Taking any-corner would grow every
    footprint by half a cell in each direction, which closes narrow alleys - and
    preserving those gaps is most of the point.
    """
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


# How far inside a footprint the roof can be trusted. A monocular depth model
# blurs across a depth discontinuity, so the pixels just inside a roof edge are
# a blend of roof and whatever is behind it. Taking them at face value makes the
# roof sag at its own boundary and the wall tops come out serrated.
EDGE_BLEED_PX = 2


def core(mask: np.ndarray, erode_px: int = EDGE_BLEED_PX) -> np.ndarray:
    """The interior of a footprint, away from the contaminated edge ring."""
    if erode_px <= 0:
        return mask
    try:
        from scipy.ndimage import binary_erosion
    except ImportError:                                # pragma: no cover
        return mask
    inner = binary_erosion(mask, np.ones((2 * erode_px + 1,) * 2, bool))
    return inner if inner.any() else mask


def _extend_from_core(values: np.ndarray, mask: np.ndarray,
                      inner: np.ndarray) -> np.ndarray:
    """Carry reliable interior heights out to the footprint boundary.

    Nearest-neighbour, not smoothing: every height in the result is a measured
    roof height from this same roof, moved outward by a couple of pixels. It
    does not invent a value, and it does not average across the edge - which is
    what produced the sag in the first place.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:                                # pragma: no cover
        return values
    if inner.all() or not inner.any():
        return values
    _, idx = distance_transform_edt(~inner, return_indices=True)
    filled = values[tuple(idx)]
    return np.where(mask & ~inner, filled, values)


def _ring(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """The band of ground just outside a footprint, where local ground is read."""
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:                                # pragma: no cover
        return ~mask
    grown = binary_dilation(mask, np.ones((2 * width + 1, 2 * width + 1), bool))
    return grown & ~mask


def unpinch(cells: np.ndarray, max_passes: int = 4) -> np.ndarray:
    """Remove diagonal pinches from a cell mask.

    Where two cells meet at a single grid vertex and the other two cells around
    that vertex are absent, the vertex carries four boundary edges instead of
    two and the surface is non-manifold there - a renderer sees a surface that
    passes through itself, and any downstream tool that assumes manifold input
    is entitled to fail. One of the two diagonal cells is dropped, which costs
    one cell of a million and makes the mesh well-formed.

    Repeated, because removing a cell can expose another pinch, but bounded:
    this is a cleanup, not a solver.
    """
    out = cells.copy()
    for _ in range(max_passes):
        a = out[:-1, :-1]                       # cell (r-1, c-1) of vertex (r, c)
        b = out[:-1, 1:]                        # cell (r-1, c)
        c = out[1:, :-1]                        # cell (r, c-1)
        d = out[1:, 1:]                         # cell (r, c)
        pinch_ad = a & d & ~b & ~c
        pinch_bc = b & c & ~a & ~d
        if not (pinch_ad.any() or pinch_bc.any()):
            break
        out[1:, 1:][pinch_ad] = False
        out[1:, :-1][pinch_bc] = False
    return out


def _cells_to_px(cells: np.ndarray, shape) -> np.ndarray:
    """The pixels a set of cells covers - the four corners of each."""
    px = np.zeros(shape, bool)
    px[:-1, :-1] |= cells
    px[:-1, 1:] |= cells
    px[1:, :-1] |= cells
    px[1:, 1:] |= cells
    return px


def pieces(cells: np.ndarray, min_cells: int = 12) -> list:
    """Split a footprint into its 4-connected parts, **in cell space**.

    A SAM instance is not always one object: a mask can cover two roofs across a
    courtyard, or pinch to a single shared corner. Extruding such a mask as one
    solid produces two failures at once - the "building" is not one connected
    component, and a grid vertex where two parts touch only diagonally lands on
    the boundary of four facade panels, which is a non-manifold edge.

    **Cell space, not pixel space**, and that distinction is the whole point.
    The footprint is taken as the cells whose four corners are all inside the
    mask, and that test can disconnect cells whose pixels are connected - a
    one-pixel-wide neck survives in pixel space and vanishes in cell space. It
    is the cells that get triangulated, so it is the cells that have to be
    4-connected. Splitting the pixel mask instead left both failures in place.

    4-connected, not 8: diagonal touching is exactly the pinch to break.
    """
    try:
        from scipy.ndimage import label
    except ImportError:                                # pragma: no cover
        return [cells]
    lab, n = label(cells, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    return [lab == k for k in range(1, n + 1)
            if int((lab == k).sum()) >= min_cells]


def measure(instance_id: int, mask: np.ndarray, dsm: np.ndarray, dtm: np.ndarray,
            sem: Optional[np.ndarray] = None,
            confidence: float = 0.0,
            source_instance: Optional[int] = None,
            cells: Optional[np.ndarray] = None) -> Optional[Building]:
    """Turn one instance mask into a Building, or reject it with a reason.

    Rejection is the point of this function as much as measurement. SAM 2 is
    class-agnostic: its instances are roads, courtyards, shadows and trees as
    readily as roofs, and extruding one of those puts a building-shaped solid
    where the image says there is none.
    """
    from ..core.types import VEGETATION

    area = int(mask.sum())
    if area < MIN_AREA_PX:
        return None

    inner = core(mask)
    roof = dsm[inner]
    roof = roof[np.isfinite(roof)]
    if roof.size < 8:
        return None

    ring = _ring(mask)
    near = dtm[ring] if ring.any() else dtm[mask]
    near = near[np.isfinite(near)]
    if near.size == 0:
        return None

    # Ground under the footprint, read from the ring around it. The DTM already
    # carries terrain under buildings by a nearest-ground fill, but the ring is
    # what the wall actually has to meet, so that is what the wall is built to.
    ground = float(np.median(near))
    roof_med = float(np.median(roof))
    roof_mad = _mad(roof)
    height = roof_med - ground

    note = ""
    if sem is not None:
        veg = float((sem[mask] == VEGETATION).mean())
        if veg > 0.6:
            return None                    # a tree canopy, not a roof
        if veg > 0.3:
            note = f"{veg * 100:.0f}% of this footprint reads as vegetation"

    if height < MIN_HEIGHT_M:
        return None

    outliers = float((np.abs(roof - roof_med) > 3.0 * max(roof_mad, 1e-3)).mean())
    if roof_mad > MAX_ROOF_MAD_M:
        note = (note + "; " if note else "") + \
            f"roof height scatters by {roof_mad:.1f} m MAD"

    if cells is None:
        cells = _to_cells(mask)
    if not cells.any():
        return None                        # thinner than one cell: nothing to extrude

    rows, cols = np.nonzero(inner)
    plane, inlier_frac, rmse = _fit_plane(rows.astype(np.float64),
                                          cols.astype(np.float64), dsm[inner])
    planar = bool(plane is not None and inlier_frac >= PLANE_INLIER_FRAC
                  and rmse <= PLANE_TOL_M * 2)

    return Building(
        id=int(instance_id), source_instance=int(source_instance
                                                 if source_instance is not None
                                                 else instance_id),
        cells=cells, area_px=area,
        roof_median_m=roof_med, roof_mad_m=roof_mad, ground_m=ground,
        height_m=height, outlier_ratio=outliers, planar=planar,
        plane=plane if planar else None, confidence=float(confidence), note=note)


def select(instances, dsm: np.ndarray, ndsm: np.ndarray,
           sem: Optional[np.ndarray] = None) -> list:
    """Which instances are buildings. Decided on height, not on colour.

    This is why the instance stage runs before depth but the building decision
    happens here: SAM 2 supplies the outlines, and only the calibrated surface
    can say which outlines enclose something tall.
    """
    dtm = np.asarray(dsm, np.float32) - np.asarray(ndsm, np.float32)
    dsm32 = np.asarray(dsm, np.float32)
    out = []
    next_id = 1
    for rec in getattr(instances, "records", []):
        if rec.get("visible_px", rec.get("area_px", 0)) < MIN_AREA_PX:
            continue
        for part in pieces(unpinch(_to_cells(instances.mask(rec["id"])))):
            b = measure(next_id, _cells_to_px(part, dsm32.shape), dsm32, dtm, sem,
                        confidence=rec.get("score", 0.0),
                        source_instance=rec["id"], cells=part)
            if b is not None:
                out.append(b)
                next_id += 1
    return out


# ---------------------------------------------------------------- geometry
class _Mesh:
    """Accumulates vertices and triangles per named group."""

    def __init__(self):
        self.v: list = []
        self.f: list = []
        self.groups: list = []             # (name, first_face, n_faces)

    def add(self, name: str, verts: np.ndarray, tris: np.ndarray) -> None:
        if len(tris) == 0:
            return
        base = len(self.v)
        self.v.extend(verts.tolist() if isinstance(verts, np.ndarray) else verts)
        first = len(self.f)
        self.f.extend((np.asarray(tris) + base).tolist())
        self.groups.append((name, first, len(self.f) - first))

    def finish(self) -> dict:
        V = np.asarray(self.v, np.float64).reshape(-1, 3)
        F = np.asarray(self.f, np.int64).reshape(-1, 3)
        return {"vertices": V, "triangles": F, "groups": self.groups}


def _grid_xyz(rows, cols, z, h, gsd):
    """Grid indices to metres. +X east, +Y north, +Z up, from the SW corner -
    the same convention the height-field writer uses, so the two are
    interchangeable in the viewer."""
    return np.stack([cols * gsd, (h - 1 - rows) * gsd, z], axis=1)


def _sheet(z: np.ndarray, keep_cells: np.ndarray, gsd: float):
    """Triangulate the cells of a height field that `keep_cells` selects."""
    h, w = z.shape
    rr, cc = np.nonzero(keep_cells)
    if rr.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)

    # Only the vertices those cells touch, renumbered densely.
    used = np.zeros((h, w), bool)
    for dr in (0, 1):
        for dc in (0, 1):
            used[rr + dr, cc + dc] = True
    idx = np.full((h, w), -1, np.int64)
    ur, uc = np.nonzero(used)
    idx[ur, uc] = np.arange(ur.size)
    verts = _grid_xyz(ur.astype(np.float64), uc.astype(np.float64),
                      np.nan_to_num(z[ur, uc], nan=0.0).astype(np.float64), h, gsd)

    a = idx[rr, cc]
    b = idx[rr, cc + 1]
    c = idx[rr + 1, cc]
    d = idx[rr + 1, cc + 1]
    # Wound counter-clockwise seen from above, so face normals point +Z.
    tris = np.concatenate([np.stack([a, c, d], 1), np.stack([a, d, b], 1)], 0)
    return verts, tris


def _solid(cells: np.ndarray, roof_z: np.ndarray, ground: float, gsd: float,
           h: int):
    """One building: roof cap plus facades, welded into a single component.

    The roof and the walls share their top vertices, and adjacent facade panels
    share their corners, so a building comes out as one connected component
    rather than a cap floating over a ring of loose quads. That is what makes
    "these are separate objects" a countable property of the mesh instead of a
    claim about how it looks.
    """
    rr, cc = np.nonzero(cells)
    if rr.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)

    # roof vertices: every grid vertex the footprint cells touch
    used = np.zeros(roof_z.shape, bool)
    for dr in (0, 1):
        for dc in (0, 1):
            used[rr + dr, cc + dc] = True
    top = np.full(roof_z.shape, -1, np.int64)
    ur, uc = np.nonzero(used)
    top[ur, uc] = np.arange(ur.size)
    verts = list(_grid_xyz(ur.astype(np.float64), uc.astype(np.float64),
                           np.nan_to_num(roof_z[ur, uc], nan=ground).astype(np.float64),
                           h, gsd))

    a = top[rr, cc]
    b = top[rr, cc + 1]
    c = top[rr + 1, cc]
    d = top[rr + 1, cc + 1]
    tris = [np.stack([a, c, d], 1), np.stack([a, d, b], 1)]

    # One ground vertex per boundary grid vertex, created on demand and shared
    # by both facade panels that meet there.
    bottom: dict = {}

    def base_of(r, c_):
        key = (int(r), int(c_))
        if key not in bottom:
            bottom[key] = len(verts)
            verts.append(_grid_xyz(np.array([r], float), np.array([c_], float),
                                   np.array([ground]), h, gsd)[0])
        return bottom[key]

    padded = np.zeros((cells.shape[0] + 2, cells.shape[1] + 2), bool)
    padded[1:-1, 1:-1] = cells
    inner = padded[1:-1, 1:-1]
    sides = (
        (inner & ~padded[:-2, 1:-1], (0, 0), (0, 1), False),   # north edge
        (inner & ~padded[2:, 1:-1], (1, 0), (1, 1), True),     # south edge
        (inner & ~padded[1:-1, :-2], (0, 0), (1, 0), True),    # west edge
        (inner & ~padded[1:-1, 2:], (0, 1), (1, 1), False),    # east edge
    )
    panels = []
    for mask_side, off0, off1, flip in sides:
        for r0, c0 in zip(*np.nonzero(mask_side)):
            v0 = top[r0 + off0[0], c0 + off0[1]]
            v1 = top[r0 + off1[0], c0 + off1[1]]
            g0 = base_of(r0 + off0[0], c0 + off0[1])
            g1 = base_of(r0 + off1[0], c0 + off1[1])
            # Wound so every facade faces away from the solid.
            if flip:
                panels.append([[v0, g0, g1], [v0, g1, v1]])
            else:
                panels.append([[v0, v1, g1], [v0, g1, g0]])
    if panels:
        tris.append(np.asarray([t for pair in panels for t in pair], np.int64))

    return np.asarray(verts, np.float64), np.concatenate(tris, 0).astype(np.int64)


def build(dsm_m: np.ndarray, ndsm_m: np.ndarray, buildings: list, gsd_m: float,
          instances=None, use_planes: bool = True) -> dict:
    """The refined mesh: terrain with holes, plus one solid per building."""
    dsm = np.asarray(dsm_m, np.float32)
    h, w = dsm.shape
    dtm = dsm - np.asarray(ndsm_m, np.float32)

    occupied = np.zeros((h - 1, w - 1), bool)
    footprint_px = np.zeros((h, w), bool)
    for b in buildings:
        occupied |= b.cells
        footprint_px[:-1, :-1] |= b.cells
        footprint_px[:-1, 1:] |= b.cells
        footprint_px[1:, :-1] |= b.cells
        footprint_px[1:, 1:] |= b.cells

    mesh = _Mesh()

    # Terrain: every cell no building covers, read from a height field in which
    # building pixels carry the ground beneath them rather than the roof above.
    # Cells share grid vertices with the footprint they border, so without this
    # the terrain climbs to roof height at the boundary and the ramp this stage
    # exists to remove reappears one cell outside every building.
    z_terrain = np.where(footprint_px, dtm, dsm).astype(np.float32)
    # Cutting a hole for every footprint can leave the terrain pinched at a
    # vertex between two footprints that pass corner to corner, so the same
    # cleanup applies to what is left of the ground.
    tv, tt = _sheet(z_terrain, unpinch(~occupied), gsd_m)
    mesh.add("terrain", tv, tt)

    for b in buildings:
        # The roof: measured heights from the trusted interior, carried out to
        # the boundary. Without this the wall tops follow the blurred edge and
        # come out serrated - visible in any oblique view.
        px = _cells_to_px(b.cells, dsm.shape)
        roof = _extend_from_core(dsm, px, core(px)).astype(np.float32)
        if use_planes and b.planar and b.plane is not None:
            a_, b_, c_ = b.plane
            rr, cc = np.mgrid[0:h, 0:w]
            fitted = a_ * cc + b_ * rr + c_
            # Only inside this footprint, and only where the plane is close to
            # the calibrated value: a plane extrapolated past its own support is
            # a fabrication, not a fit.
            grow = np.zeros((h, w), bool)
            grow[:-1, :-1] |= b.cells
            grow[:-1, 1:] |= b.cells
            grow[1:, :-1] |= b.cells
            grow[1:, 1:] |= b.cells
            close = np.abs(fitted - roof) <= PLANE_TOL_M * 3
            roof = np.where(grow & close, fitted, roof).astype(np.float32)

        # Its own vertices, so a building shares none with the terrain or with
        # its neighbours and separation is a countable property of the mesh.
        bv, bt = _solid(b.cells, roof, b.ground_m, gsd_m, h)
        mesh.add(f"building_{b.id}", bv, bt)

    out = mesh.finish()
    out["buildings"] = [b.record() for b in buildings]
    out["schema"] = SCHEMA_VERSION
    return out
