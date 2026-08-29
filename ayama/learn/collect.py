"""Turn a completed dataset run into fitting samples.

The inputs are the two things a study already produces: the relative depth each
scene's backbone emitted, and the lidar truth that came with the scene. Nothing
is re-inferred - fitting the scale over four scenes takes about a second,
because the expensive part happened when the study ran.

Truth handling is the part worth reading. The target is height above ground, so
a DSM reference is only usable when a bare-earth DTM ships with it; a scene with
an elevation reference and no DTM is skipped with a reason rather than being
fitted against the wrong quantity.
"""
from __future__ import annotations

import os

import numpy as np

from .scale import Sample, high_band, scene_features, scene_target


def _read(path: str) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float32)


def collect_scene(ref, run_dir: str, radius_m: float = 60.0):
    """One (Sample, positive-high-band, true nDSM) triple, or (None, reason).

    `ref` is a `ayama.data.SceneRef`; `run_dir` is where that scene's pipeline
    artifacts were written.
    """
    rel_path = os.path.join(run_dir, "relative_depth.tif")
    if not os.path.exists(rel_path):
        return None, "no relative_depth.tif - was the run written with artifacts?"
    if not ref.reference:
        return None, "no reference raster"

    if ref.reference_kind == "ndsm":
        ndsm = _read(ref.reference)
    elif ref.dtm:
        ndsm = np.maximum(_read(ref.reference) - _read(ref.dtm), 0.0)
    else:
        return None, ("reference is an elevation and no bare-earth DTM ships "
                      "with it, so height above ground cannot be formed")

    rel = _read(rel_path)
    if rel.shape != ndsm.shape:
        return None, f"grid mismatch: depth {rel.shape} vs truth {ndsm.shape}"

    import rasterio

    with rasterio.open(rel_path) as ds:
        gsd = abs(ds.transform.a) or 1.0

    dem = _read(ref.dem) if ref.dem and os.path.exists(ref.dem) else None
    pos = high_band(rel, gsd, radius_m)
    a_star = scene_target(pos, ndsm)
    if not np.isfinite(a_star):
        return None, "the high band is empty; no scale can be fitted"

    sample = Sample(
        name=ref.name,
        target=a_star,
        features=scene_features(rel, dem, gsd, radius_m),
        ndsm_mae_at_target=float(np.abs(np.maximum(a_star * pos, 0) - ndsm).mean()),
        floor_mae=float(np.abs(ndsm).mean()),
    )
    return (sample, pos, ndsm), None


def collect(scenes, runs_root: str, radius_m: float = 60.0,
            on_scene=None) -> tuple:
    """Samples and rasters for every scene that can supply a target.

    Returns `(samples, rasters, skipped)`. `rasters` feeds the leave-one-out
    comparison so it can be made in metres of nDSM error rather than in error on
    the fitted scale - those rank models differently, and only the first is what
    anyone cares about.
    """
    samples, rasters, skipped = [], {}, []
    for ref in scenes:
        run_dir = os.path.join(runs_root, ref.name)
        got, why = collect_scene(ref, run_dir, radius_m)
        if got is None:
            skipped.append((ref.name, why))
            if on_scene:
                on_scene(ref.name, None, why)
            continue
        sample, pos, ndsm = got
        samples.append(sample)
        rasters[sample.name] = (pos, ndsm)
        if on_scene:
            on_scene(ref.name, sample, None)
    return samples, rasters, skipped
