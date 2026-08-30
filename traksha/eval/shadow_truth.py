"""Ground-truth cast shadows, ray-marched from a measured surface model.

`detect_shadow` finds shadows in an image by their radiometry. This does the
opposite and independent thing: given a surface model and a sun direction, it
computes geometrically which pixels *must* be in shadow. Run against an airborne
lidar DSM it is a truth mask, so the detector and the shadow-height inversion in
`traksha.chhaya` can be scored rather than eyeballed.

It marches the same ray the physics module inverts, which is the point: an error
in the geometry convention shows up as a disagreement here rather than as a
quiet bias in the anchors.

Note that a sun direction is required and never invented. Where acquisition time
is unknown - as it is for every swisstopo product this repository uses - the
caller must supply an angle and own the assumption.
"""
from __future__ import annotations

import numpy as np

from ..core.geo import sun_vector

try:
    from scipy.ndimage import map_coordinates

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def cast_shadow_mask(dsm: np.ndarray, sun_az: float, sun_el: float, gsd: float,
                     n_steps: int = 96) -> np.ndarray:
    """March each pixel toward the sun; shadowed if the surface blocks the ray."""
    d_col, d_row, d_z = sun_vector(sun_az, sun_el)
    h, w = dsm.shape
    rr, cc = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32),
                         indexing="ij")
    occluded = np.zeros(dsm.shape, bool)
    for i in range(1, n_steps + 1):
        t_px = float(i)                      # step one pixel at a time
        t_m = t_px * gsd
        sr = rr + d_row * t_px
        sc = cc + d_col * t_px
        if HAVE_SCIPY:
            surf = map_coordinates(dsm, [sr, sc], order=1, mode="nearest")
        else:
            surf = dsm[np.clip(sr.astype(int), 0, h - 1), np.clip(sc.astype(int), 0, w - 1)]
        ray_z = dsm + d_z * t_m
        occluded |= surf > ray_z + 1e-3
    return occluded
