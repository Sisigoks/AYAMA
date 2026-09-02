"""Wavefront OBJ export: the DSM as an actual textured mesh on disk.

The browser viewer is the demo; this is the deliverable. An OBJ plus MTL plus
the aligned texture opens in Blender, MeshLab, CloudCompare and every DCC tool
without a plugin, which makes the output reviewable by someone who will never
install TRAKSHA - the same reason every raster is written as a COG.

Geometry convention, stated once because a silent axis flip here produces a
mesh that looks fine and is mirrored:

    +X  east      (raster +col)
    +Y  north     (raster -row, because +row is south)
    +Z  up        (elevation in metres)

Vertices are in metres relative to the tile's south-west corner, not in CRS
coordinates. A UTM easting of 612345.0 stored as a float32 vertex has about
6 cm of representable precision left, which is coarser than the 0.1 m the
terrain encoding preserves, so the georeference is carried in the sidecar
manifest instead of being baked into vertices that cannot hold it.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def decimate(arr: np.ndarray, stride: int) -> np.ndarray:
    return np.asarray(arr)[::max(1, int(stride)), ::max(1, int(stride))]


def _write_mtl(mtl_path: str, name: str, texture_name: Optional[str]) -> None:
    """The material both writers share. `map_Kd` is what makes the JPG apply."""
    with open(mtl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"newmtl {name}_mat\n")
        fh.write("Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n")
        fh.write("d 1.0\nillum 1\n")
        if texture_name:
            fh.write(f"map_Kd {texture_name}\n")


def write_obj(
    path: str,
    dsm_m: np.ndarray,
    gsd_m: float,
    texture_name: Optional[str] = None,
    stride: int = 1,
    exaggeration: float = 1.0,
    name: str = "traksha_surface",
) -> dict:
    """Write `path` (.obj), its .mtl sidecar, and return a summary dict.

    `stride` decimates the grid; a 1024x1024 DSM at stride 1 is 1.05 M vertices
    and 2.09 M triangles, which most viewers handle but no browser should be
    asked to parse from text. Stride 2 is the sensible default for a demo.

    Non-finite pixels are dropped: a vertex is emitted for every grid point, but
    a quad is only emitted where all four corners are finite. Emitting a face
    over a NaN hole is how a single nodata pixel becomes a spike to -10000 m.
    """
    _ensure_parent(path)
    a = decimate(np.asarray(dsm_m, np.float64), stride)
    h, w = a.shape
    if h < 2 or w < 2:
        raise ValueError(f"grid too small to mesh after stride {stride}: {a.shape}")

    step = float(gsd_m) * max(1, int(stride))
    finite = np.isfinite(a)
    z = np.where(finite, a, 0.0) * float(exaggeration)

    # Vertices, row-major. +Y is north, so row 0 (the top of the raster, which
    # is the northernmost) gets the largest Y.
    cols = np.arange(w, dtype=np.float64) * step
    rows = (h - 1 - np.arange(h, dtype=np.float64)) * step
    X = np.broadcast_to(cols, (h, w))
    Y = np.broadcast_to(rows[:, None], (h, w))

    # UVs: OBJ's V axis runs bottom-up, the raster's row axis runs top-down.
    u = np.broadcast_to(np.linspace(0.0, 1.0, w), (h, w))
    v = np.broadcast_to(np.linspace(1.0, 0.0, h)[:, None], (h, w))

    quad_ok = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]
    n_quads = int(quad_ok.sum())

    mtl_path = os.path.splitext(path)[0] + ".mtl"
    mtl_name = os.path.basename(mtl_path)

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# TRAKSHA {name}\n")
        fh.write(f"# grid {h} x {w}  step {step:.6g} m  exaggeration {exaggeration:g}\n")
        fh.write("# axes: +X east, +Y north, +Z up (metres from the SW corner)\n")
        fh.write(f"mtllib {mtl_name}\n")
        fh.write(f"o {name}\n")

        _write_block(fh, "v", np.stack([X.ravel(), Y.ravel(), z.ravel()], 1))
        if texture_name:
            # A fiftieth of a texel on a 512 px tile; more is invisible.
            _write_block(fh, "vt", np.stack([u.ravel(), v.ravel()], 1), decimals=5)
        fh.write(f"usemtl {name}_mat\n")

        # OBJ indices are 1-based.
        idx = (np.arange(h * w, dtype=np.int64) + 1).reshape(h, w)
        tl, tr = idx[:-1, :-1][quad_ok], idx[:-1, 1:][quad_ok]
        bl, br = idx[1:, :-1][quad_ok], idx[1:, 1:][quad_ok]
        # Counter-clockwise seen from +Z, so face normals point up.
        _write_faces(fh, np.stack([bl, br, tr], 1), bool(texture_name))
        _write_faces(fh, np.stack([bl, tr, tl], 1), bool(texture_name))

    _write_mtl(mtl_path, name, texture_name)

    return {
        "obj": path,
        "mtl": mtl_path,
        "vertices": int(h * w),
        "triangles": int(2 * n_quads),
        "dropped_quads": int(quad_ok.size - n_quads),
        "grid": [int(h), int(w)],
        "step_m": step,
        "exaggeration": float(exaggeration),
    }


def _fmt(v: float) -> str:
    """Millimetres, with no trailing zeros. `0.0000` and `510.0000` cost bytes."""
    t = f"{v:.3f}".rstrip("0").rstrip(".")
    return t if t and t != "-0" else "0"


def _write_block(fh, tag: str, rows: np.ndarray, decimals: int = 3) -> None:
    """Write many `v`/`vt` lines without building one huge Python string.

    Precision is deliberate, not incidental. Vertices are written to the
    millimetre and texture coordinates to five decimals - about a fiftieth of a
    texel on a 512 px tile. The previous fixed `%.4f` spent five bytes writing
    `0.0000` for grid coordinates that are exact multiples of the step, and
    tenths of a millimetre on a surface whose own error is metres (README §3.2).
    Trimming that is a third off the file for nothing lost, which is the
    difference between a mesh that can live in the repository and one that
    cannot.
    """
    if decimals == 3:
        fh.writelines(tag + " " + " ".join(map(_fmt, r)) + "\n" for r in rows)
    else:
        fmt = tag + ((" %%.%df" % decimals) * rows.shape[1]) + "\n"
        fh.writelines(fmt % tuple(r) for r in rows)


def _write_faces(fh, tris: np.ndarray, with_uv: bool) -> None:
    if with_uv:
        # Vertex and texture indices are the same grid, so v/vt share an index.
        fh.writelines("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, c, c) for a, b, c in tris)
    else:
        fh.writelines("f %d %d %d\n" % (a, b, c) for a, b, c in tris)


def write_obj_adaptive(
    path: str,
    dsm_m: np.ndarray,
    gsd_m: float,
    texture_name: Optional[str] = None,
    tol_m: float = 2.0,
    block: int = 8,
    max_triangles: "int | None" = None,
    exaggeration: float = 1.0,
    name: str = "traksha_surface",
) -> dict:
    """Write an adaptively sampled OBJ: fine where the surface bends, coarse elsewhere.

    A uniform stride spends its triangles evenly on a surface whose detail is
    not distributed evenly. On the delivered Zurich scene a 2 m grid is within
    0.9 m of the full-resolution surface on average and 61 m out at worst,
    because the misses are all on the cells where a step cuts across a wall.

    `tol_m` is a guarantee rather than a setting: every block is emitted at
    whatever resolution keeps it within `tol_m` of the full-resolution surface,
    so the worst case is bounded by construction instead of being discovered
    afterwards. At 2 m it costs about the same triangles as a uniform 1 m grid
    and bounds the error twenty times tighter.

    See `traksha.mesh.adaptive` for how the levels are stitched without cracks.
    """
    from .adaptive import adaptive_mesh

    _ensure_parent(path)
    a = np.asarray(dsm_m, np.float64)
    if exaggeration != 1.0:
        a = a * float(exaggeration)
    mesh = adaptive_mesh(a, gsd_m, tol_m=tol_m, block=block,
                         max_triangles=max_triangles)
    tol_m = mesh["tol_m"]
    V, UV, T = mesh["vertices"], mesh["uv"], mesh["triangles"]
    lay = mesh["layout"]

    mtl_path = os.path.splitext(path)[0] + ".mtl"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# TRAKSHA {name}\n")
        fh.write(f"# adaptive: {lay['n_fine']}/{lay['n_blocks']} blocks at full "
                 f"resolution, tolerance {tol_m:g} m, block {block} px\n")
        fh.write("# axes: +X east, +Y north, +Z up (metres from the SW corner)\n")
        fh.write(f"mtllib {os.path.basename(mtl_path)}\n")
        fh.write(f"o {name}\n")
        _write_block(fh, "v", V)
        if texture_name:
            _write_block(fh, "vt", UV, decimals=5)
        fh.write(f"usemtl {name}_mat\n")
        _write_faces(fh, T, bool(texture_name))

    _write_mtl(mtl_path, name, texture_name)
    return {
        "obj": path,
        "mtl": mtl_path,
        "vertices": int(len(V)),
        "triangles": int(len(T)),
        "dropped_quads": 0,
        "grid": [int(a.shape[0]), int(a.shape[1])],
        "step_m": float(gsd_m),
        "exaggeration": float(exaggeration),
        "adaptive": {"tol_m": float(tol_m), "block": int(block),
                     "fine_blocks": int(lay["n_fine"]),
                     "blocks": int(lay["n_blocks"])},
    }

def write_obj_structural(
    path: str,
    mesh: dict,
    grid_shape: tuple,
    gsd_m: float,
    texture_name: Optional[str] = None,
    name: str = "traksha_structural",
) -> dict:
    """Write the structurally rebuilt mesh: terrain plus one group per building.

    Groups are not decoration. Each building is its own connected component with
    its own vertices, so `g building_7` selects a whole object in any tool that
    reads OBJ - which is the difference between a scene of buildings and a sheet
    that happens to be bumpy.

    UVs come from world position rather than from the grid, because this mesh
    has vertices the grid does not: a facade's top and bottom sit at the same
    easting and northing, so both sample the orthophoto at the footprint edge
    and the texture runs down the wall. That is the honest thing a nadir image
    can offer - it photographed the roof, not the facade.
    """
    from .quality import validate

    _ensure_parent(path)
    V = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["triangles"], np.int64)
    h, w = grid_shape
    span_x = max((w - 1) * float(gsd_m), 1e-9)
    span_y = max((h - 1) * float(gsd_m), 1e-9)
    UV = np.stack([V[:, 0] / span_x, V[:, 1] / span_y], axis=1)

    mtl_path = os.path.splitext(path)[0] + ".mtl"
    groups = mesh.get("groups") or [("surface", 0, len(F))]
    stats = validate(V, F)

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# TRAKSHA {name}\n")
        fh.write(f"# {len(groups)} groups, {stats['components']} connected components\n")
        fh.write("# render-space product: rebuilt from the calibrated DSM and the\n")
        fh.write("#   instance segmentation. The calibrated rasters are unchanged.\n")
        fh.write("# axes: +X east, +Y north, +Z up (metres from the SW corner)\n")
        fh.write(f"mtllib {os.path.basename(mtl_path)}\n")
        fh.write(f"o {name}\n")
        _write_block(fh, "v", V)
        if texture_name:
            _write_block(fh, "vt", UV, decimals=5)
        for gname, first, count in groups:
            if count <= 0:
                continue
            fh.write(f"g {gname}\n")
            fh.write(f"usemtl {name}_mat\n")
            # OBJ indices are 1-based; the builder works 0-based throughout.
            _write_faces(fh, F[first:first + count] + 1, bool(texture_name))

    _write_mtl(mtl_path, name, texture_name)
    return {
        "obj": path, "mtl": mtl_path,
        # Recorded because this file is two orders of magnitude larger than the
        # tileset, and the download link says so before anyone clicks it.
        "bytes": int(os.path.getsize(path)),
        "vertices": int(len(V)), "triangles": int(len(F)),
        "groups": int(len(groups)),
        "buildings": int(len(mesh.get("buildings", []))),
        "components": int(stats["components"]),
        "degenerate_faces": int(stats["degenerate_faces"]),
        "non_manifold_edges": int(stats["non_manifold_edges"]),
    }


def _write_faces_uv(fh, tris: np.ndarray, uvs: np.ndarray) -> None:
    """`f v/vt` where the two index streams are not the same stream.

    Every grid-derived mesh here has one texture coordinate per vertex, so
    `_write_faces` writes `v/v` and is done. A synthesised facade does not: it
    carries its own UV atlas, and a vertex sitting on a chart seam has one
    position and two texture coordinates. OBJ says that per face corner, which
    is what this writes.
    """
    fh.writelines("f %d/%d %d/%d %d/%d\n" % (t[0], u[0], t[1], u[1], t[2], u[2])
                  for t, u in zip(tris, uvs))
