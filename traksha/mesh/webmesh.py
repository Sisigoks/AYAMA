"""The structural mesh in a form a browser can draw.

The viewer renders height *tiles*: a grid of vertices whose z comes from a
24-bit PNG. That representation cannot express a wall - a height field has one
z per (x, y) by definition - so every structural improvement made to the mesh
has been invisible on the site. This is the bridge.

`structural.obj` is the detailed download at roughly 135 MB, which is not a
thing to hand a browser. So the same builder runs on a strided grid chosen to
hit a triangle budget, and the result is written as one compact binary rather
than as OBJ text: positions, heights, UVs, per-vertex normals and indices as
typed arrays the renderer uploads straight into GL buffers with no parsing.

**Normals travel with the mesh, and they have to.** The viewer shades tiles from
a normal *map* - a texture indexed by (u, v) - which is exactly the assumption a
wall breaks: the roof and the pavement below it share a UV, so a facade would be
shaded as though it were the ground. Per-vertex normals are the fix, and they are
computed from the geometry rather than resampled from anything.

Strided, not decimated. Choosing every n-th row and column keeps the builder's
own logic - footprints, walls, the terrain hole - intact at a coarser grid,
where a mesh simplifier would collapse exactly the boundaries the stage exists
to create.
"""
from __future__ import annotations

import os
import struct

import numpy as np

MAGIC = b"TKM1"
FORMAT_VERSION = 1

# What a browser should be asked to hold. The height-field surface.obj is
# budgeted to about the same, so the two are comparable when a reader switches
# between them.
DEFAULT_MAX_TRIANGLES = 350_000


def choose_stride(shape, max_triangles: int) -> int:
    """The coarsest grid that is still finer than the budget requires.

    Two triangles per cell is the upper bound - the terrain hole under every
    footprint means the real count is lower - so this errs toward more detail
    rather than less.
    """
    h, w = shape
    for stride in range(1, 17):
        cells = ((h - 1) // stride) * ((w - 1) // stride)
        if cells * 2 <= max_triangles:
            return stride
    return 16


def build_web_mesh(dsm: np.ndarray, ndsm: np.ndarray, instances, sem,
                   gsd_m: float, max_triangles: int = DEFAULT_MAX_TRIANGLES,
                   source: "dict | None" = None):
    """Structural mesh at a browser-sized triangle budget, with normals.

    Two ways to get inside the budget, and they are not close. Given the
    full-resolution mesh (`source`) and PyMeshLab, the triangles are removed by
    per-group quadric edge collapse, which takes them from where the surface is
    flat. Without either, the mesh is rebuilt on a strided grid, which takes
    *resolution* uniformly - including from the roof edges and from any building
    smaller than a few grid steps, which stops existing altogether.

    Measured on the delivered Bern scene at a 250 000 triangle budget, against
    the full-resolution mesh:

        stride       RMS 1.003 m   max 13.234 m    99 components,  75 buildings
        collapse     RMS 0.059 m   max  2.993 m   203 components, 100 buildings

    The reference has 203 components and 100 buildings, so striding was deleting
    a quarter of them.
    """
    from ..semantics.instances import InstanceField
    from . import decimate as D
    from . import structural as S
    from .quality import vertex_normals

    if source is not None and D.available():
        simplified = D.simplify(source, max_triangles)
        mesh = simplified if simplified is not None else dict(source)
        mesh["normals"] = vertex_normals(mesh["vertices"], mesh["triangles"])
        mesh.setdefault("decimated", {"method": "already within budget"})
        mesh["stride"] = 1
        mesh["gsd_m"] = float(gsd_m)
        mesh["grid"] = [int(dsm.shape[0]), int(dsm.shape[1])]
        return mesh

    stride = choose_stride(dsm.shape, max_triangles)
    sl = (slice(None, None, stride), slice(None, None, stride))
    d = np.ascontiguousarray(dsm[sl])
    nd = np.ascontiguousarray(ndsm[sl])
    sm = None if sem is None else np.ascontiguousarray(sem[sl]).astype(np.uint8)
    gsd = float(gsd_m) * stride

    small = InstanceField(
        instance_map=np.ascontiguousarray(instances.instance_map[sl]),
        boundary=np.ascontiguousarray(instances.boundary[sl]),
        confidence=np.ascontiguousarray(instances.confidence[sl]),
        records=instances.records, provenance=instances.provenance)

    buildings = S.select(small, d, nd, sm)
    mesh = S.build(d, nd, buildings, gsd)
    mesh["normals"] = vertex_normals(mesh["vertices"], mesh["triangles"])
    mesh["decimated"] = {"method": "uniform stride (PyMeshLab unavailable)"}
    mesh["stride"] = stride
    mesh["gsd_m"] = gsd
    mesh["grid"] = [int(d.shape[0]), int(d.shape[1])]
    return mesh


def write(path: str, mesh: dict, grid_shape, gsd_m: float) -> dict:
    """Write the binary the renderer reads. Returns what the manifest records."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    V = np.asarray(mesh["vertices"], np.float32)
    F = np.asarray(mesh["triangles"], np.uint32)
    N = np.asarray(mesh.get("normals"), np.float32)
    h, w = grid_shape
    span_x = max((w - 1) * float(gsd_m), 1e-9)
    span_y = max((h - 1) * float(gsd_m), 1e-9)

    # Positions stay absolute metres from the south-west corner, the same frame
    # the OBJ uses. The viewer subtracts its own scene centre, so the file does
    # not have to agree with it about where the middle is.
    pos = np.ascontiguousarray(V[:, :2])
    height = np.ascontiguousarray(V[:, 2])
    uv = np.ascontiguousarray(np.stack([V[:, 0] / span_x, V[:, 1] / span_y], 1))

    groups = mesh.get("groups") or [("surface", 0, len(F))]
    table = []
    for name, first, count in groups:
        kind = 0 if name == "terrain" else 1
        try:
            ident = int(name.rsplit("_", 1)[-1])
        except ValueError:
            ident = 0
        table.append((first, count, kind, ident))

    header = struct.pack(
        "<4sIIIIfffff", MAGIC, FORMAT_VERSION, len(V), int(F.size), len(table),
        float(gsd_m), float(span_x), float(span_y),
        float(height.min()) if len(V) else 0.0,
        float(height.max()) if len(V) else 0.0)

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(pos.astype("<f4").tobytes())
        fh.write(height.astype("<f4").tobytes())
        fh.write(uv.astype("<f4").tobytes())
        fh.write(N.astype("<f4").tobytes() if N.size else
                 np.zeros((len(V), 3), "<f4").tobytes())
        fh.write(F.ravel().astype("<u4").tobytes())
        for first, count, kind, ident in table:
            fh.write(struct.pack("<IIII", first, count, kind, ident))

    return {
        "path": os.path.basename(path),
        "format": "TKM1",
        "vertices": int(len(V)),
        "triangles": int(len(F)),
        "groups": len(table),
        "buildings": int(sum(1 for _, _, k, _ in table if k == 1)),
        "stride": int(mesh.get("stride", 1)),
        "gsd_m": round(float(mesh.get("gsd_m", gsd_m)), 4),
        "reduction": mesh.get("decimated", {}),
        "bytes": int(os.path.getsize(path)),
    }


def read(path: str) -> dict:
    """Read a TKM1 back. Used by the tests, so the writer has a reader that
    is not the renderer - a format only one side can parse is not a format."""
    with open(path, "rb") as fh:
        blob = fh.read()
    magic, version, nv, ni, ng, gsd, sx, sy, zmin, zmax = struct.unpack_from(
        "<4sIIIIfffff", blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a TRAKSHA mesh: {magic!r}")
    off = struct.calcsize("<4sIIIIfffff")

    def take(count, dtype, itemsize):
        nonlocal off
        arr = np.frombuffer(blob, dtype, count, off)
        off += count * itemsize
        return arr

    pos = take(nv * 2, "<f4", 4).reshape(nv, 2)
    height = take(nv, "<f4", 4)
    uv = take(nv * 2, "<f4", 4).reshape(nv, 2)
    nrm = take(nv * 3, "<f4", 4).reshape(nv, 3)
    idx = take(ni, "<u4", 4)
    groups = [struct.unpack_from("<IIII", blob, off + 16 * i) for i in range(ng)]
    return {"version": version, "gsd_m": gsd, "extent_m": (sx, sy),
            "z_range": (zmin, zmax), "positions": pos, "heights": height,
            "uv": uv, "normals": nrm, "indices": idx, "groups": groups}
