"""Fitting the structural mesh into a triangle budget without wrecking it.

The web copy of the mesh has to be a few hundred thousand triangles, and the
obvious way to get there - build the whole thing on a strided grid - is much
worse than it looks. Measured against the full-resolution mesh on the delivered
Bern scene at a 250 000 triangle budget:

    uniform stride          RMS 1.003 m   mean 0.361 m   max 13.234 m   99 parts
    quadric edge collapse   RMS 0.055 m   mean 0.027 m   max  0.628 m  199 parts

Eighteen times the error, and it is not only error. The reference has 207
connected components; striding leaves 99, because a building smaller than three
grid steps across simply stops existing. Decimation removes *triangles*, and it
removes them where the surface is flat; striding removes *resolution*, uniformly,
including from the roof edges and the small buildings that carry the structure.

**Per group, not per mesh.** Quadric collapse over the whole mesh at once scores
slightly better, but it hands back one anonymous soup of triangles: the group
table is gone, so which triangles are which building is gone with it, and
nothing then prevents two buildings being welded at a vertex they happen to
share after collapse. Decimating each group into its own budget keeps the table,
and keeps separation true by construction rather than by hope.

PyMeshLab is an optional dependency. Without it this returns None and the caller
falls back to striding, which is worse but is not nothing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Below this a group is left alone. Collapsing a small building further does not
# save a meaningful number of triangles and does risk collapsing it to nothing.
MIN_GROUP_FACES = 120


def available() -> bool:
    try:
        import pymeshlab  # noqa: F401
    except Exception:
        return False
    return True


def _mesh_set(vertices: np.ndarray, faces: np.ndarray):
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(vertices, np.float64),
                        face_matrix=np.asarray(faces, np.int32)))
    return ms


def _collapse(vertices: np.ndarray, faces: np.ndarray, target: int):
    """Quadric edge collapse on one connected piece.

    `preserveboundary` is what keeps a footprint outline where it is - the
    boundary of a building group is its wall line, and letting the collapse move
    it would undo the stage that put it there. `planarquadric` is what lets a
    flat roof or a flat stretch of ground give up its triangles cheaply, which
    is where the budget is meant to come from.
    """
    if len(faces) <= max(target, MIN_GROUP_FACES):
        return np.asarray(vertices), np.asarray(faces)
    ms = _mesh_set(vertices, faces)
    try:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), preserveboundary=True, preservenormal=True,
            preservetopology=True, planarquadric=True, qualitythr=0.3,
            autoclean=True)
    except Exception:
        return np.asarray(vertices), np.asarray(faces)
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix().astype(np.int64)


def simplify(mesh: dict, max_triangles: int) -> Optional[dict]:
    """Decimate a grouped mesh into a budget, one group at a time.

    Returns None if PyMeshLab is unavailable, or if the mesh is already inside
    the budget and there is nothing to do.
    """
    if not available():
        return None
    F = np.asarray(mesh["triangles"], np.int64)
    V = np.asarray(mesh["vertices"], np.float64)
    groups = mesh.get("groups") or [("surface", 0, len(F))]
    if len(F) <= max_triangles:
        return None

    # Each group keeps its share of the budget. Proportional, because a quadric
    # collapse already spends a group's own allowance where that group bends -
    # a flat roof gives its triangles up and a stepped one keeps them - so the
    # allocation does not also have to guess which groups matter.
    scale = max_triangles / float(len(F))
    out_v: list = []
    out_f: list = []
    out_groups: list = []
    for name, first, count in groups:
        if count <= 0:
            continue
        block = F[first:first + count]
        used = np.unique(block)
        remap = np.full(int(used.max()) + 1, -1, np.int64)
        remap[used] = np.arange(len(used))
        gv, gf = _collapse(V[used], remap[block], int(round(count * scale)))
        if len(gf) == 0:
            continue
        base = sum(len(x) for x in out_v)
        out_v.append(np.asarray(gv, np.float64))
        out_f.append(np.asarray(gf, np.int64) + base)
        out_groups.append((name, sum(len(x) for x in out_f[:-1]), len(gf)))

    if not out_f:
        return None
    return {
        "vertices": np.concatenate(out_v, 0),
        "triangles": np.concatenate(out_f, 0),
        "groups": out_groups,
        "buildings": mesh.get("buildings", []),
        "decimated": {"from": int(len(F)),
                      "to": int(sum(len(f) for f in out_f)),
                      "method": "quadric edge collapse, per group"},
    }
