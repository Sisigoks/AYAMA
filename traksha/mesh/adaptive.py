"""Adaptive triangulation: full resolution where the surface bends, coarse where it does not.

A uniform stride is the wrong instrument for this surface. Decimating the
delivered Zürich DSM to a 2 m grid loses 0.9 m of height on average and **61 m
at worst**, because the error is not spread out - it is concentrated on the
handful of cells where a sampling step cuts across a twenty-metre wall. Paying
for that with a uniformly finer grid buys detail everywhere, and it is only
needed in a few places.

So the grid is chosen per block. Each block of `block` cells is measured against
what a flat bilinear patch through its corners would give; blocks the patch
already explains are emitted coarse, and the rest are emitted at full
resolution.

**The seam is the part worth reading.** Mixing resolutions creates T-junctions -
a coarse triangle's edge passing through a fine neighbour's vertex - and a
T-junction is a hairline crack you can see through in any renderer. It is
avoided by construction: a coarse block is not two big triangles but a fan from
its centre out to *every* fine vertex on its perimeter. Its boundary therefore
uses exactly the vertices its neighbours use, whatever level they chose, and the
mesh closes regardless of how the levels fall.
"""
from __future__ import annotations

import numpy as np


def _blocks(n: int, block: int):
    """Start indices of vertex blocks covering 0..n-1, last one ragged.

    Vertices, not cells: a block from r0 spans r0..r0+block inclusive, so the
    last block is clamped to the final row rather than running past it. Getting
    this wrong is how the first version indexed column 1024 of a 1024-wide grid.
    """
    starts = list(range(0, n - 1, block))
    return starts or [0]


def _bilinear_block_error(z: np.ndarray, block: int) -> np.ndarray:
    """Per-block max |z - bilinear(corners)|.

    A block the corner patch already explains carries no information a finer
    grid would keep, so it is exactly the thing worth measuring. Written as a
    loop over blocks rather than a vectorised stencil because the last block on
    each axis is a different size, and a clever stencil that silently drops it
    would leave a strip of the scene unmeshed.
    """
    h, w = z.shape
    rs, cs = _blocks(h, block), _blocks(w, block)
    err = np.zeros((len(rs), len(cs)), np.float32)
    for bi, r0 in enumerate(rs):
        r1 = min(r0 + block, h - 1)
        for bj, c0 in enumerate(cs):
            c1 = min(c0 + block, w - 1)
            sub = z[r0:r1 + 1, c0:c1 + 1]
            nh, nw = sub.shape
            if nh < 2 or nw < 2:
                continue
            wy = np.linspace(0.0, 1.0, nh, dtype=np.float32)[:, None]
            wx = np.linspace(0.0, 1.0, nw, dtype=np.float32)[None, :]
            plane = (sub[0, 0] * (1 - wy) * (1 - wx)
                     + sub[0, -1] * (1 - wy) * wx
                     + sub[-1, 0] * wy * (1 - wx)
                     + sub[-1, -1] * wy * wx)
            err[bi, bj] = float(np.nanmax(np.abs(sub - plane)))
    return err


def plan(z: np.ndarray, tol_m: float = 0.35, block: int = 8,
         err: "np.ndarray | None" = None) -> dict:
    """Decide which blocks are fine and which are coarse, and which vertices live.

    Returns the block mask, a boolean vertex-used mask over the full grid, and
    the counts, so a caller can report what the adaptation bought before writing
    anything.
    """
    z = np.asarray(z, np.float32)
    h, w = z.shape
    if h < block + 1 or w < block + 1:
        raise ValueError(f"grid {z.shape} is smaller than one {block} px block")

    if err is None:
        err = _bilinear_block_error(z, block)
    fine = err > float(tol_m)
    used = np.zeros((h, w), bool)
    rs, cs = _blocks(h, block), _blocks(w, block)

    for bi, r0 in enumerate(rs):
        r1 = min(r0 + block, h - 1)
        for bj, c0 in enumerate(cs):
            c1 = min(c0 + block, w - 1)
            if fine[bi, bj]:
                used[r0:r1 + 1, c0:c1 + 1] = True
            else:
                # Corners always. Intermediate vertices on an edge only where
                # the neighbour across it is fine and will use them - between
                # two coarse blocks the edge is a single segment, which is where
                # nearly all of the saving comes from.
                for r in (r0, r1):
                    for c in (c0, c1):
                        used[r, c] = True
                # The fan radiates from the centre, not from a corner. Fanning
                # from a corner puts three collinear ring vertices in one
                # triangle wherever an edge carries intermediates for a fine
                # neighbour, and a triangle with no footprint in plan view is a
                # vertical sliver - 3.6% of them, before this line.
                used[(r0 + r1) // 2, (c0 + c1) // 2] = True
                if bi > 0 and fine[bi - 1, bj]:
                    used[r0, c0:c1 + 1] = True
                if bi + 1 < len(rs) and fine[bi + 1, bj]:
                    used[r1, c0:c1 + 1] = True
                if bj > 0 and fine[bi, bj - 1]:
                    used[r0:r1 + 1, c0] = True
                if bj + 1 < len(cs) and fine[bi, bj + 1]:
                    used[r0:r1 + 1, c1] = True

    return {"fine": fine, "used": used, "block": int(block),
            "err": err, "tol_m": float(tol_m),
            "n_fine": int(fine.sum()), "n_blocks": int(fine.size),
            "n_vertices": int(used.sum())}


def triangles(z: np.ndarray, layout: dict) -> np.ndarray:
    """Triangle indices into the *used* vertices, row-major. Returns (T, 3) int64."""
    h, w = z.shape
    block = layout["block"]
    fine, used = layout["fine"], layout["used"]

    # dense index -> compacted index, 1-based for OBJ
    idx = np.full((h, w), -1, np.int64)
    idx[used] = np.arange(int(used.sum()), dtype=np.int64) + 1

    rs, cs = _blocks(h, block), _blocks(w, block)
    tris = []
    for bi, r0 in enumerate(rs):
        r1 = min(r0 + block, h - 1)
        for bj, c0 in enumerate(cs):
            c1 = min(c0 + block, w - 1)
            if fine[bi, bj]:
                sub = idx[r0:r1 + 1, c0:c1 + 1]
                tl, tr = sub[:-1, :-1], sub[:-1, 1:]
                bl, br = sub[1:, :-1], sub[1:, 1:]
                ok = (tl > 0) & (tr > 0) & (bl > 0) & (br > 0)
                tris.append(np.stack([bl[ok], br[ok], tr[ok]], 1))
                tris.append(np.stack([bl[ok], tr[ok], tl[ok]], 1))
            else:
                # Counter-clockwise seen from +Z, keeping only vertices that
                # survived `plan` - which is exactly the set the neighbours use,
                # so the boundary matches and no T-junction can form.
                ring = ([(r1, c) for c in range(c0, c1)]
                        + [(r, c1) for r in range(r1, r0, -1)]
                        + [(r0, c) for c in range(c1, c0, -1)]
                        + [(r, c0) for r in range(r0, r1)])
                ring = [q for q in ring if idx[q] > 0]
                centre = idx[(r0 + r1) // 2, (c0 + c1) // 2]
                if len(ring) < 3 or centre <= 0:
                    continue
                a = np.array([idx[q] for q in ring], np.int64)
                b = np.roll(a, -1)
                tris.append(np.stack(
                    [a, b, np.full(len(a), centre, np.int64)], 1))

    if not tris:
        return np.zeros((0, 3), np.int64)
    return np.concatenate([t for t in tris if len(t)], 0)


def tolerance_for_budget(z: np.ndarray, max_triangles: int, block: int = 8,
                         floor_m: float = 0.25) -> tuple:
    """The tightest tolerance whose mesh fits in `max_triangles`.

    A tolerance is a quality guarantee and a triangle count is a file size, and
    only one of them can be chosen freely. Exposing the tolerance alone meant a
    dense scene silently produced a 58 MB OBJ - past the size at which a host
    starts warning - while a flat one produced a small a one at the same
    nominal quality. Choosing the budget and reporting the tolerance that fits
    is the way round that keeps both predictable.

    The block error field is computed once per block size and reused across
    tolerances, so this costs one pass plus a few cheap counts.

    The block size is escalated when even the loosest tolerance overruns. It
    has to be: a coarse block still costs a handful of triangles, so `block`
    sets a floor of roughly `4 * (H/block) * (W/block)` that no tolerance can go
    under. At 8 px on a 1024 px grid that floor is 80k triangles, which is why a
    20k budget quietly came back at 80k until this loop existed.
    """
    z = np.asarray(z, np.float32)
    fallback = None

    for blk in (block, block * 2, block * 4, block * 8):
        if min(z.shape) < blk + 1:
            break
        err = _bilinear_block_error(z, blk)

        def at(tol, _blk=blk, _err=err):
            layout = plan(z, tol_m=tol, block=_blk, err=_err)
            return layout, len(triangles(z, layout))

        lo = None                  # tightest tolerance known to overrun
        for tol in (floor_m, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0):
            if tol < floor_m:
                continue
            layout, n = at(tol)
            if n <= max_triangles:
                if lo is None:
                    return float(tol), layout, n
                # Bisect between the last overrun and this fit: the ladder is
                # coarse, and landing at 377k when 500k was allowed throws away
                # quality that was paid for.
                best = (float(tol), layout, n)
                a, b = lo, float(tol)
                for _ in range(4):
                    mid = 0.5 * (a + b)
                    lay_m, n_m = at(mid)
                    if n_m <= max_triangles:
                        best, b = (mid, lay_m, n_m), mid
                    else:
                        a = mid
                return best
            lo = float(tol)
            # Keep the SMALLEST overrun seen, not the last. A bigger block
            # lowers the coarse floor but makes every fine block cost
            # quadratically more, so escalating is not monotonic and the last
            # attempt is often the worst one.
            if fallback is None or n < fallback[2]:
                fallback = (float(tol), layout, n)

    # Every block size overran. Return the smallest mesh actually reached and
    # let the caller see the count, rather than pretending the budget was met:
    # this scheme has a floor of a few triangles per block, and a budget below
    # it is not achievable by choosing a tolerance.
    return fallback if fallback else (20.0, plan(z, 20.0, block), 0)


def adaptive_mesh(z: np.ndarray, gsd_m: float, tol_m: float = 0.35,
                  block: int = 8, max_triangles: "int | None" = None) -> dict:
    """Vertices, UVs and triangles for an adaptively sampled height field.

    Vertices are metres from the south-west corner, +X east, +Y north, +Z up -
    the same convention as the regular writer, so the two are interchangeable.
    """
    z = np.asarray(z, np.float32)
    h, w = z.shape
    if max_triangles:
        tol_m, layout, _ = tolerance_for_budget(z, int(max_triangles), block)
        block = layout["block"]
    else:
        layout = plan(z, tol_m=tol_m, block=block)
    used = layout["used"]

    rows, cols = np.nonzero(used)
    step = float(gsd_m)
    xyz = np.stack([cols * step,
                    (h - 1 - rows) * step,
                    np.where(np.isfinite(z[rows, cols]), z[rows, cols], 0.0)], 1)
    uv = np.stack([cols / max(w - 1, 1), 1.0 - rows / max(h - 1, 1)], 1)

    tris = triangles(z, layout)
    return {"vertices": xyz.astype(np.float64), "uv": uv.astype(np.float64),
            "triangles": tris, "layout": layout, "tol_m": float(tol_m)}
