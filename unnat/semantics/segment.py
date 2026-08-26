"""Five-class segmentation: bare ground, road, building, vegetation, water.

Two implementations behind one interface.

`raster`     - load a segmentation someone else produced. This is the path a
               real deployment takes, and the one to use whenever a trained
               model is available.
`heuristic`  - colour, texture and (optionally) height cues, no weights. Honest
               about what it is: it exists so the anchor harvester, the
               semantic gate and the whole Phase 2 path can be built, tested and
               benchmarked before a segmentation model is trained. Its output
               is labelled `heuristic` in every artifact it touches.

The gate it feeds is not cosmetic. A public DEM approximates bare earth, so a
DEM sample on a rooftop is not a weak anchor, it is a wrong one, and the gate
is what keeps it out.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import (BARE_GROUND, BUILDING, ROAD, VEGETATION, WATER, Scene)

try:
    from scipy.ndimage import (binary_closing, binary_opening, median_filter,
                               uniform_filter)

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def _channels(rgb: np.ndarray):
    a = np.asarray(rgb, np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    total = np.maximum(r + g + b, 1e-6)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    exg = (2 * g - r - b) / total          # excess green: vegetation index without NIR
    exb = (2 * b - r - g) / total          # excess blue: water and shadow lean blue
    return r, g, b, lum, sat, exg, exb


def _texture(lum: np.ndarray, win: int = 7) -> np.ndarray:
    if not HAVE_SCIPY:
        return np.zeros_like(lum)
    smooth = uniform_filter(lum, win)
    var = uniform_filter((lum - smooth) ** 2, win)
    return np.sqrt(np.maximum(var, 0.0))


def segment_heuristic(
    scene: Scene,
    ndsm_m: Optional[np.ndarray] = None,
    shadow: Optional[np.ndarray] = None,
    building_height_m: float = 2.5,
) -> np.ndarray:
    """Colour/texture classifier. Returns (H, W) uint8 of class ids.

    When an nDSM is available it dominates the building decision, because
    height is a far better building cue than colour. Without one, the classifier
    falls back to brightness and texture and will confuse bright bare ground
    with rooftops; that error is exactly why the DEM anchors are also filtered
    by slope.
    """
    r, g, b, lum, sat, exg, exb = _channels(scene.rgb)
    tex = _texture(lum)

    sem = np.full(lum.shape, BARE_GROUND, np.uint8)

    # Vegetation: the one class colour genuinely separates.
    veg = exg > 0.06
    sem[veg] = VEGETATION

    # Water: blue-leaning, smooth, and not vegetation. Smoothness is the
    # discriminating cue; blue roofs and blue tarpaulins are not smooth.
    water = (exb > 0.05) & (tex < np.percentile(tex, 35)) & ~veg
    if HAVE_SCIPY and water.any():
        water = binary_opening(water, np.ones((5, 5), bool))
        water = binary_closing(water, np.ones((9, 9), bool))
    sem[water] = WATER

    # Road: low saturation, mid-to-dark grey, smooth.
    open_ground = ~veg & ~water
    road = open_ground & (sat < 0.20) & (lum < np.percentile(lum[open_ground], 55)) \
        & (tex < np.percentile(tex, 45))
    if HAVE_SCIPY and road.any():
        road = binary_opening(road, np.ones((3, 3), bool))
    sem[road] = ROAD

    # Building: height if we have it, brightness plus texture if we do not.
    if ndsm_m is not None:
        tall = np.asarray(ndsm_m, np.float32) > building_height_m
        build = tall & ~water
        # Tall and green is a tree, not a building.
        build &= ~veg
    else:
        bright = lum > np.percentile(lum[open_ground], 78)
        build = open_ground & bright & (tex > np.percentile(tex, 55))
    if HAVE_SCIPY and build.any():
        build = binary_opening(build, np.ones((3, 3), bool))
        build = binary_closing(build, np.ones((5, 5), bool))
    sem[build] = BUILDING

    # A shadow is not a class. Pixels in shadow keep whatever class their
    # neighbourhood says, otherwise every shadow becomes 'road'.
    if shadow is not None and HAVE_SCIPY:
        sh = np.asarray(shadow, bool)
        if sh.any():
            filled = median_filter(sem, size=9)
            sem = np.where(sh, filled, sem).astype(np.uint8)

    return sem


def segment_from_raster(path: str, shape: tuple) -> np.ndarray:
    """Load a segmentation someone else produced, resampled onto the image grid."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        arr = ds.read(1, out_shape=shape, resampling=Resampling.nearest)
    return np.asarray(arr, np.uint8)


def segment(
    scene: Scene,
    method: str = "heuristic",
    path: Optional[str] = None,
    ndsm_m: Optional[np.ndarray] = None,
    shadow: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, str]:
    """Returns (class raster, provenance string). Provenance goes into the artifacts."""
    if method == "raster" or path:
        if not path:
            raise ValueError("segmentation method 'raster' needs a path")
        return segment_from_raster(path, scene.shape), f"raster:{path}"
    return segment_heuristic(scene, ndsm_m=ndsm_m, shadow=shadow), "heuristic"


def class_fractions(sem: np.ndarray) -> dict:
    from ..core.types import CLASS_NAMES

    total = float(sem.size)
    return {name: float((sem == cls).sum()) / total for cls, name in CLASS_NAMES.items()}
