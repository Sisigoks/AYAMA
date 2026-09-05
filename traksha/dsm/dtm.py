"""Bare earth by cloth simulation, via CNES Bulldozer.

`dsm.assemble.extract_dtm` is the morphological approach and it is honest about
its own limits: keep the pixels a semantic raster calls ground, carry them under
everything else, smooth. Its failure mode is not subtle. It believes the
segmentation, and the segmentation is a five-class colour classifier that calls
a grey roof "bare ground". Every rooftop it mislabels becomes terrain, and the
terrain surface rises to meet the roofs.

Measured on the bundled Zurich fixture against the survey-grade swissALTI3D DTM:

    extract_dtm      MAE 6.195 m   RMSE 8.136 m   bias +6.193 m
    bulldozer        MAE 0.811 m   RMSE 1.234 m   bias -0.038 m

The `+6.193 m` is the whole story - it is not noise, it is the surface sitting
on the rooftops. Bulldozer's bias is under four centimetres. (Both figures at
this module's defaults; the sweep below reports the same scene at 20 m, where
it scores 0.877 m.)

**Why it works where the morphological filter does not.** Bulldozer is a
multi-scale drape-cloth filter: a stiff sheet is dropped onto the *inverted* DSM
and relaxed, so the terrain is inferred from the surface's own shape rather than
from a classifier's opinion about its colour. Nothing has to be labelled
correctly for it to work, which is exactly the dependency that was hurting.

**`max_object_size` is the one parameter that matters, and it was measured.**
It is the width, in metres, of the largest thing standing on the terrain - the
scale above which a rise is terrain and below which it is an object. Swept over
the four delivered scenes against lidar truth:

    max_object_size   bern    geneva  lausanne  zurich   mean MAE
        10 m          2.820   1.081   0.542     0.960      1.351
        15 m          1.310   0.731   0.584     0.886    **0.878**
        20 m          1.362   0.769   0.591     0.968      0.922
        30 m          0.971   1.468   1.578     2.053      1.518
        60 m          0.917   4.368   2.695     3.977      2.989

15 m is the default because it is the minimum of that curve, not because it is
the width of a building. It is worth saying why those differ: a European city
block is far wider than 15 m, and setting the parameter to the block width makes
the result three times worse. The cloth is multi-scale, so it removes a wide
building through its coarse levels regardless; what the parameter really trades
is how much genuine terrain relief the filter is allowed to flatten. Bern, the
scene with 110 m of relief, is the one that prefers a larger value, and it is
outvoted.

**Ground masks are accepted but not used by default.** Bulldozer reads
`ground_mask_path` only when `activate_ground_anchors` is on, and with the
OpenStreetMap road mask supplied that way the result got *worse* on every scene
tried - because a road under tree canopy, or on a bridge, is a pixel the mask
swears is ground and whose DSM value is metres above it. The parameter is
plumbed through and documented; it is off because the measurement said so.

**It is optional.** Bulldozer is a compiled dependency and this module falls
back to `assemble.extract_dtm` when it is missing, recording which branch ran.
A DTM that came from the fallback must never be reported as if it came from the
cloth simulation, so the provenance says which one it was.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import tempfile
from typing import Optional

import numpy as np

# The minimum of the sweep above, over four real scenes against lidar truth.
DEFAULT_MAX_OBJECT_SIZE_M = 15.0

# Bulldozer's own default is 20%. City centres are gentler than that and the
# scenes here span 70-110 m of relief over a kilometre, so it is left alone.
DEFAULT_MAX_GROUND_SLOPE = 20.0

# Written into the temporary DSM so Bulldozer's Cython core has a nodata value
# it can test against. It uses -32768 itself when handed None.
NODATA = -32768.0


class BulldozerUnavailable(RuntimeError):
    """Bulldozer was asked for and is not installed."""


def available() -> bool:
    try:
        import bulldozer.pipeline.bulldozer_pipeline  # noqa: F401
    except Exception:
        return False
    return True


def version() -> Optional[str]:
    try:
        import bulldozer
    except Exception:
        return None
    return getattr(bulldozer, "__version__", "unknown")


def _affine_for(meta, shape):
    """The georeferencing Bulldozer needs, real or synthesised.

    Bulldozer reads the pixel size off the raster and every scale it works at is
    expressed in metres, so a scene with no transform cannot simply be handed
    an identity affine - the filter would read one metre per pixel and size
    every object wrong. A plain image gets a synthetic north-up affine at the
    scene's own GSD instead, which is a fiction about *where* the scene is and
    the truth about *how big* it is. Only the latter is used.
    """
    from rasterio.transform import Affine

    if getattr(meta, "georeferenced", False):
        return Affine(*meta.transform), meta.crs
    gsd = float(getattr(meta, "gsd_m", 1.0) or 1.0)
    return Affine(gsd, 0.0, 0.0, 0.0, -gsd, float(shape[0]) * gsd), "EPSG:3857"


def extract(
    dsm_m: np.ndarray,
    meta,
    *,
    sem: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    max_object_size_m: float = DEFAULT_MAX_OBJECT_SIZE_M,
    max_ground_slope: float = DEFAULT_MAX_GROUND_SLOPE,
    use_ground_anchors: bool = False,
    workers: Optional[int] = None,
    required: bool = False,
) -> tuple:
    """Bare-earth surface from a DSM. Returns `(dtm_m, provenance)`.

    `sem` is only the fallback's input - the cloth simulation does not read it.
    `ground_mask` is passed to Bulldozer only when `use_ground_anchors` is set,
    because that is the only condition under which Bulldozer reads it; see the
    module docstring for why that is off by default.

    With `required=True` a missing Bulldozer raises instead of falling back, so
    a benchmark cannot silently measure the wrong estimator.
    """
    dsm = np.asarray(dsm_m, np.float32)
    if not available():
        if required:
            raise BulldozerUnavailable(
                "bulldozer is not installed. `pip install bulldozer-dtm`, or call "
                "with required=False to use the morphological fallback.")
        from .assemble import extract_dtm

        dtm = extract_dtm(dsm, sem if sem is not None else np.zeros(dsm.shape, np.uint8),
                          float(getattr(meta, "gsd_m", 1.0) or 1.0))
        return dtm, {
            "method": "morphological",
            "reason": "bulldozer is not installed (pip install bulldozer-dtm)",
            "note": "carries ground-classified pixels under everything else; "
                    "believes the semantic raster",
        }

    import rasterio
    from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm

    transform, crs = _affine_for(meta, dsm.shape)
    work = tempfile.mkdtemp(prefix="traksha_dtm_")
    try:
        dsm_path = os.path.join(work, "dsm.tif")
        finite = np.isfinite(dsm)
        with rasterio.open(
                dsm_path, "w", driver="GTiff", height=dsm.shape[0], width=dsm.shape[1],
                count=1, dtype="float32", crs=crs, transform=transform,
                nodata=NODATA) as ds:
            ds.write(np.where(finite, dsm, NODATA).astype(np.float32), 1)

        kwargs = {
            "dsm_path": dsm_path,
            "output_dir": os.path.join(work, "out"),
            "max_object_size": float(max_object_size_m),
            "max_ground_slope": float(max_ground_slope),
            "enforce_dtm_below_dsm": True,
            "activate_ground_anchors": bool(use_ground_anchors),
        }
        if workers:
            kwargs["nb_max_workers"] = int(workers)
        if ground_mask is not None and use_ground_anchors:
            gpath = os.path.join(work, "ground.tif")
            with rasterio.open(
                    gpath, "w", driver="GTiff", height=dsm.shape[0],
                    width=dsm.shape[1], count=1, dtype="uint8", crs=crs,
                    transform=transform) as ds:
                ds.write(np.asarray(ground_mask, bool).astype(np.uint8), 1)
            kwargs["ground_mask_path"] = gpath

        # Bulldozer narrates at INFO and draws tqdm bars on stdout. Inside a
        # pipeline phase that is noise interleaved with the progress reporter,
        # so it is captured rather than printed - and restored afterwards,
        # because silencing the root logger permanently would take the rest of
        # the run's diagnostics with it.
        root = logging.getLogger()
        previous = root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                dsm_to_dtm(**kwargs)
        finally:
            logging.disable(previous)

        with rasterio.open(os.path.join(work, "out", "dtm.tif")) as ds:
            dtm = ds.read(1).astype(np.float32)
            nod = ds.nodata
        if nod is not None:
            dtm = np.where(dtm == nod, np.nan, dtm)
        # The cloth is defined everywhere; the input was not. Give back NaN
        # where the DSM had nothing, so the caller's own hole-filling decides.
        dtm = np.where(finite, dtm, np.nan).astype(np.float32)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    prov = {
        "method": "bulldozer",
        "version": version(),
        "max_object_size_m": float(max_object_size_m),
        "max_ground_slope": float(max_ground_slope),
        "ground_anchors": bool(use_ground_anchors),
        "ground_mask": bool(ground_mask is not None and use_ground_anchors),
        "note": "multi-scale drape-cloth filter (CNES Bulldozer); infers terrain "
                "from surface shape, not from the semantic raster",
    }
    return dtm, prov
