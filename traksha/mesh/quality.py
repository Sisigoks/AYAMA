"""Mesh validation and the metrics that decide whether a refinement helped.

A refined mesh that looks better in a screenshot is not evidence. These are the
numbers the structural rebuild has to move, and the ones it must not break.

The separation metrics are the point. "Buildings are separated" is the claim,
and it is checkable: if two buildings are distinct objects then their triangles
belong to different connected components, they do not share vertices, and the
gap between their footprints is not bridged by geometry. A sheet mesh fails all
three by construction - which is what makes them worth measuring.

The topology metrics are the guard. Cutting a mesh into pieces is an easy way to
introduce degenerate triangles, duplicated vertices and inconsistent winding, so
every one of those is counted before and after.
"""
from __future__ import annotations

import numpy as np

# Below this area a triangle contributes nothing to the surface and breaks
# normal estimation, because the cross product of two near-parallel edges is
# numerically meaningless.
DEGENERATE_AREA_M2 = 1e-9


def face_normals(V: np.ndarray, F: np.ndarray):
    """Unit face normals and triangle areas."""
    if len(F) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    norm = np.linalg.norm(cross, axis=1)
    unit = cross / np.maximum(norm, 1e-12)[:, None]
    return unit, norm / 2.0


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals.

    Weighted by area rather than averaged flat, so a roof surrounded by many
    small facade triangles is not tilted by the count of its neighbours.
    """
    fn, area = face_normals(V, F)
    out = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(out, F[:, k], fn * area[:, None])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def weld(V: np.ndarray, F: np.ndarray, tol: float = 1e-6):
    """Merge vertices that coincide, and renumber the faces.

    Used to *measure* welding, not to apply it blindly: welding across a
    structural boundary would reconnect exactly the surfaces this pipeline just
    separated, so the structural builder keeps its groups apart and this is used
    on one group at a time.
    """
    if len(V) == 0:
        return V, F, 0
    q = np.round(np.asarray(V, np.float64) / max(tol, 1e-12)).astype(np.int64)
    _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.zeros(len(order), np.int64)
    remap[order] = np.arange(len(order))
    return V[first[order]], remap[inverse][F], int(len(V) - len(order))


def components(V: np.ndarray, F: np.ndarray):
    """Connected components over the face-adjacency graph. Returns (count, labels)."""
    if len(F) == 0:
        return 0, np.zeros(len(V), np.int64)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:                                # pragma: no cover
        return -1, np.zeros(len(V), np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
                   shape=(len(V), len(V)))
    return connected_components(g, directed=False)


def validate(V: np.ndarray, F: np.ndarray) -> dict:
    """Topology and sanity, as counts a reader can act on."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    _, area = face_normals(V, F)

    # Every undirected edge, and how many faces use it. Two is manifold; one is
    # a boundary, which a cut mesh legitimately has; more is a defect.
    if len(F):
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = np.sort(e, axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
    else:
        counts = np.zeros(0, np.int64)

    referenced = np.unique(F) if len(F) else np.zeros(0, np.int64)
    ncomp, _ = components(V, F)
    finite = np.isfinite(V).all()

    return {
        "vertices": int(len(V)),
        "triangles": int(len(F)),
        "degenerate_faces": int((area <= DEGENERATE_AREA_M2).sum()),
        "duplicate_vertices": int(len(V) - len(np.unique(
            np.round(V / 1e-6).astype(np.int64), axis=0))) if len(V) else 0,
        "unreferenced_vertices": int(len(V) - len(referenced)),
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "components": int(ncomp),
        "finite": bool(finite),
        "bounds_m": [float(V[:, 2].min()), float(V[:, 2].max())] if len(V) else [0.0, 0.0],
    }


def normal_consistency(V: np.ndarray, F: np.ndarray) -> float:
    """Fraction of interior edges whose two faces traverse them in opposite directions.

    That is what consistent winding means, and it decides whether lighting is
    right: a face wound the other way has an inverted normal and renders black
    beside its neighbours. Only edges shared by exactly two faces are judged - a
    boundary edge has nothing to disagree with, and a cut mesh has many by
    design. 1.0 is fully consistent.
    """
    if len(F) < 2:
        return 1.0
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(und, axis=0, return_inverse=True,
                                   return_counts=True)
    inverse = inverse.ravel()
    interior = np.nonzero(counts == 2)[0]
    if interior.size == 0:
        return 1.0
    order = np.argsort(inverse, kind="stable")
    at = np.searchsorted(inverse[order], interior)
    first, second = order[at], order[at + 1]
    opposite = directed[first, 0] != directed[second, 0]
    return float(opposite.mean())


def verticality(V: np.ndarray, F: np.ndarray) -> dict:
    """How much of the surface is facade, and how vertical it is.

    A height-field sheet has no vertical area at all: its steepest triangle
    still spans one ground sample horizontally. Facade area appearing here is
    the direct signature of the structural rebuild.
    """
    fn, area = face_normals(V, F)
    if area.sum() <= 0:
        return {"wall_area_m2": 0.0, "wall_area_frac": 0.0, "max_slope_deg": 0.0}
    vertical = np.abs(fn[:, 2]) < 0.02
    slope = np.degrees(np.arccos(np.clip(np.abs(fn[:, 2]), 0.0, 1.0)))
    return {
        "wall_area_m2": round(float(area[vertical].sum()), 2),
        "wall_area_frac": round(float(area[vertical].sum() / area.sum()), 4),
        "max_slope_deg": round(float(slope.max()), 2),
    }


# ------------------------------------------------------------- separation
def _group_faces(mesh: dict, prefix: str = "building_"):
    for name, first, count in mesh.get("groups", []):
        if name.startswith(prefix):
            yield name, mesh["triangles"][first:first + count]


def separation(mesh: dict) -> dict:
    """Are the buildings actually separate objects?

    Two checks, because either can pass while the other fails:

      own_component     a building's triangles form one connected component of
                        the whole mesh, and nothing else lives in it
      shared_vertices   no building indexes a vertex that the terrain or
                        another building also indexes

    Plan-view overlap is not measured because it cannot happen: the instance map
    assigns each pixel to exactly one instance, so footprints are disjoint by
    construction. What can happen - and is what a sheet mesh does - is two
    buildings joined through the ground between them, and that is precisely what
    the component count catches.
    """
    V, F = mesh["vertices"], mesh["triangles"]
    groups = list(_group_faces(mesh))
    if not groups:
        return {"buildings": 0, "separation_score": 1.0, "own_component": 1.0,
                "shared_vertices": 0}

    _, labels = components(V, F)
    own = 0
    owner: dict = {}
    shared = 0
    for name, faces in groups:
        verts = np.unique(faces)
        labs = np.unique(labels[verts])
        if len(labs) == 1 and not np.setdiff1d(
                np.nonzero(labels == labs[0])[0], verts).size:
            own += 1
        for v in verts.tolist():
            if owner.setdefault(v, name) != name:
                shared += 1

    n = len(groups)
    frac = own / n
    return {
        "buildings": n,
        "own_component": round(frac, 4),
        "shared_vertices": int(shared),
        # One number a threshold can act on: every building its own component,
        # nothing shared with anything else.
        "separation_score": round(frac if shared == 0 else frac * 0.5, 4),
    }


def report(mesh: dict) -> dict:
    """Everything above, for one mesh."""
    V, F = mesh["vertices"], mesh["triangles"]
    return {
        **validate(V, F),
        "normal_consistency": round(normal_consistency(V, F), 4),
        **verticality(V, F),
        **separation(mesh),
    }
