"""Tiling with overlap padding, so per-tile derivatives match at the seams.

This is the same problem [`traksha/depth/infer.py`](../depth/infer.py) solves for
inference chips, reappearing at delivery resolution and with a different fix.
Inference chips overlap and are *blended*, because neighbouring chips genuinely
disagree. Delivery tiles are cut from one already-consistent raster, so there is
nothing to blend - the failure mode is narrower and sharper:

  a normal, a slope or a mesh vertex normal computed at the last row of a tile
  needs the first row of its neighbour. Without it the gradient is one-sided,
  every tile boundary gets a faint ridge, and on a 3D surface those ridges read
  as real terrain features.

So each tile carries `pad` extra pixels of its neighbours on every side. The
padded band is used for derivatives and then discarded; only the interior is
ever displayed. At the raster border, where there is no neighbour, the edge row
is replicated - which makes the border gradient zero rather than wrong.

`pad = 1` is enough for a 3x3 gradient stencil and is the default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """One tile's place in the grid.

    Interior bounds are the pixels this tile owns; padded bounds are what it
    must read to compute derivatives over that interior. `inset` is where the
    interior starts inside the padded array, which is what the viewer needs to
    crop back to the owned region.
    """

    row: int
    col: int
    y0: int
    x0: int
    y1: int
    x1: int
    pad: int
    shape: tuple           # (H, W) of the full raster

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def padded_bounds(self) -> tuple:
        """(py0, px0, py1, px1), clipped to the raster."""
        H, W = self.shape
        return (max(0, self.y0 - self.pad), max(0, self.x0 - self.pad),
                min(H, self.y1 + self.pad), min(W, self.x1 + self.pad))

    @property
    def inset(self) -> tuple:
        """(top, left) offset of the interior inside the array `cut` returns.

        Always ``(pad, pad)``. `cut` replicates wherever the raster ran out, so
        every tile comes back with a full pad band on all four sides regardless
        of whether it sits at a border - which is what lets `interior` crop with
        one rule instead of four cases. The *unpadded* read window, which does
        vary at the borders, is `padded_bounds`.
        """
        return (self.pad, self.pad)

    @property
    def key(self) -> str:
        return f"{self.row}_{self.col}"


def tile_specs(shape: tuple, tile: int = 512, pad: int = 1) -> list[TileSpec]:
    """Cover a raster with non-overlapping interiors of at most `tile` px.

    The last row and column are short rather than overhanging, so the union of
    the interiors is exactly the raster and no pixel is owned twice. A viewer
    that assumes uniform tile sizes would tear here, which is why the manifest
    carries every tile's real size.
    """
    H, W = int(shape[0]), int(shape[1])
    tile = int(max(1, tile))
    pad = int(max(0, pad))
    out: list[TileSpec] = []
    for r, y0 in enumerate(range(0, H, tile)):
        for c, x0 in enumerate(range(0, W, tile)):
            out.append(TileSpec(row=r, col=c, y0=y0, x0=x0,
                                y1=min(y0 + tile, H), x1=min(x0 + tile, W),
                                pad=pad, shape=(H, W)))
    return out


def grid_size(shape: tuple, tile: int = 512) -> tuple:
    """(rows, cols) of the tile grid."""
    H, W = int(shape[0]), int(shape[1])
    tile = int(max(1, tile))
    return ((H + tile - 1) // tile, (W + tile - 1) // tile)


def cut(arr: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Extract a tile with its padding, replicating at the raster border."""
    a = np.asarray(arr)
    py0, px0, py1, px1 = spec.padded_bounds
    sub = a[py0:py1, px0:px1]
    # Replicate where the raster ran out, so every tile has the full pad band.
    top = spec.pad - (spec.y0 - py0)
    left = spec.pad - (spec.x0 - px0)
    bottom = spec.pad - (py1 - spec.y1)
    right = spec.pad - (px1 - spec.x1)
    if any(v > 0 for v in (top, left, bottom, right)):
        width = [(max(0, top), max(0, bottom)), (max(0, left), max(0, right))]
        width += [(0, 0)] * (sub.ndim - 2)
        sub = np.pad(sub, width, mode="edge")
    return sub


def interior(padded: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Crop a padded tile back to the pixels it owns."""
    p = spec.pad
    if p == 0:
        return padded
    return padded[p:p + spec.height, p:p + spec.width]


def reassemble(tiles: dict, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Put interiors back together. The inverse of tiling, for tests.

    `tiles` maps `spec.key` to an already-cropped interior array.
    """
    H, W = int(shape[0]), int(shape[1])
    first = next(iter(tiles.values()))
    trailing = first.shape[2:]
    out = np.zeros((H, W) + trailing, dtype)
    for spec in tile_specs(shape, tile=_infer_tile(tiles, shape)):
        t = tiles.get(spec.key)
        if t is None:
            continue
        out[spec.y0:spec.y1, spec.x0:spec.x1] = t
    return out


def _infer_tile(tiles: dict, shape: tuple) -> int:
    """Largest interior dimension present, which is the nominal tile size."""
    return max(max(t.shape[0], t.shape[1]) for t in tiles.values())


def pyramid(arr: np.ndarray, levels: int = 3) -> Iterator[tuple]:
    """Yield (lod, decimated array) for lod = 0 .. levels - 1, stride 2**lod.

    Plain decimation, not averaging, and the choice is deliberate: averaging a
    DSM mixes rooftops with the ground beside them and invents elevations that
    exist nowhere on the surface. A decimated DSM is a real subset of measured
    heights, which is the right trade for a viewer whose job is to show what was
    measured. Averaging is correct for imagery, so the texture is resampled
    separately by PIL.
    """
    a = np.asarray(arr)
    for lod in range(max(1, levels)):
        step = 2 ** lod
        if lod and (a.shape[0] // step < 2 or a.shape[1] // step < 2):
            break
        yield lod, a[::step, ::step]
