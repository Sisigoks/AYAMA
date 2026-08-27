"""Phase 3: turn a calibrated surface into something a browser can draw.

    encode.py   float raster -> 8-bit channels, losslessly enough (and the
                arithmetic that decides what "enough" means)
    tiles.py    tiling with overlap padding, so seams have no ridges
    obj.py      Wavefront OBJ + MTL, the offline textured mesh deliverable
    build.py    a Phase 2 run directory -> a tileset + manifest

Nothing here re-derives elevation. Phase 3 is delivery: every number it shows
came out of Phase 2, and `tileset.json` records which run produced it.
"""
from .build import build_tileset, load_run
from .encode import (decode_linear, decode_terrain_rgb, encode_linear,
                     encode_terrain_rgb, normal_map)
from .obj import write_obj
from .tiles import cut, interior, tile_specs

__all__ = [
    "build_tileset", "load_run",
    "encode_terrain_rgb", "decode_terrain_rgb", "encode_linear", "decode_linear",
    "normal_map", "write_obj", "tile_specs", "cut", "interior",
]
