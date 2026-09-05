"""Public global DEMs, actually fetched.

Until now `api.pipeline.load_dem` refused every source it could not open as a
local file - "Network DEM fetching is not wired up" - and every published result
in this repository was anchored to `sim:`, a survey-grade Swiss DTM artificially
degraded to look like a global product. That is a defensible way to *develop*
the method and it is not a way to run it on an arbitrary scene, because the
degradation model is an assumption about what the real product would have looked
like. This module removes the assumption.

**Copernicus GLO-30 is the default, and the reason is measured.** Across the
published vertical-accuracy assessments GLO-30 ranks first in urban/industrial
and low-relief classes - 0.82 m mean error and 2.34 m RMSE over urban Cape Town -
while NASADEM only takes the lead on steep terrain. City centres are low-relief,
which is the regime GLO-30 wins. That ordering is already encoded in
`chhaya.anchors.DEM_SIGMA_M` (copernicus 3.0 m against nasadem 5.5 m) and it is
what the anchor weights key off, so choosing the other one would make every
anchor in the run less confident for no gain.

**NASADEM is supported as a file, not as a fetch, and that is not an oversight.**
Its distribution is behind NASA Earthdata authentication. A fetcher would have to
either prompt for credentials mid-run or embed them, and this pipeline's rule is
that a run must never silently proceed with data it could not load. Download the
tile yourself and pass the path; `dem_source_name` already recognises the name
and applies NASADEM's own sigma.

**Two caveats that belong in the code, not only in a paper.**

*GLO-30 is a DSM, not a DTM.* It is an X-band radar surface: over a city its
30 m postings sit somewhere between the street and the rooftops. The pipeline
already defends against this - `chhaya.anchors.harvest_dem` admits a sample only
where the scene is bare earth - and `data.osm`'s road mask makes that gate far
sharper than the five-class colour segmentation could. Anchoring to GLO-30
without a ground mask is the failure this note exists to prevent.

*Its vertical datum is EGM2008.* A national product is very often orthometric
against a local datum instead, and mixing the two is a metres-scale bias applied
uniformly to every absolute anchor. The offset is recorded in the provenance
rather than corrected, because correcting it needs a geoid model this repository
does not ship.
"""
from __future__ import annotations

import math
import os
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

import numpy as np

USER_AGENT = "traksha/0.1 (+https://github.com/traksha; research use)"

# The AWS Open Data mirror. Public, anonymous, no request-payer: a plain HTTPS
# GET works, which is why this needs no boto3 and no credentials.
GLO30_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
GLO90_BUCKET = "https://copernicus-dem-90m.s3.amazonaws.com"

# One degree square each, ~41 MB at 30 m. A city scene touches one, or two
# across a degree line, or four at a corner.
TILE_DEG = 1

DEFAULT_TIMEOUT_S = 300

# What each product's datasheet says about itself, carried into the provenance
# so a reader can see which numbers the uncertainty budget was built on.
PRODUCTS = {
    "copernicus": {
        "name": "Copernicus DEM GLO-30",
        "bucket": GLO30_BUCKET,
        "posting_m": 30.0,
        "datum": "EGM2008",
        "kind": "DSM",
        "note": "best of the free global DEMs in urban and low-relief terrain",
    },
    "copernicus90": {
        "name": "Copernicus DEM GLO-90",
        "bucket": GLO90_BUCKET,
        "posting_m": 90.0,
        "datum": "EGM2008",
        "kind": "DSM",
        "note": "the 90 m sibling; use when GLO-30 has no coverage",
    },
}
DEFAULT_PRODUCT = "copernicus"

# Fetched by hand, not by this module. Listed so an unsupported request gets an
# explanation instead of a 403 from a bucket that was never going to answer.
MANUAL_PRODUCTS = {
    "nasadem": ("NASADEM is distributed through NASA Earthdata, which requires "
                "a login. Download the tile and pass its path with --dem."),
    "srtm": ("SRTM is distributed through NASA Earthdata, which requires a "
             "login. Download the tile and pass its path with --dem."),
    "aster": ("ASTER GDEM is distributed through NASA Earthdata or METI, both "
              "of which require a login. Download the tile and pass its path."),
}


class DEMUnavailable(RuntimeError):
    """A DEM was asked for and could not be obtained. Never returned as zeros."""


def default_cache_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "data", "_cache", "dem")


# ------------------------------------------------------------------ tiling
def tile_name(lat: int, lon: int, product: str = DEFAULT_PRODUCT) -> str:
    """Copernicus names a tile by the integer degree of its **south-west** corner.

    So the tile covering 46.5 N is N46, and the one covering 0.5 S is S01 -
    `floor` then absolute value, not `int()`, which truncates toward zero and
    would name the southern-hemisphere tile after its neighbour.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    res = "10" if product != "copernicus90" else "30"
    return f"Copernicus_DSM_COG_{res}_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tiles_for_bounds(bounds_wgs: Sequence[float]) -> list:
    """Every one-degree tile the box touches, as (lat, lon) SW corners."""
    w, s, e, n = (float(v) for v in bounds_wgs)
    lats = range(int(math.floor(s)), int(math.floor(n)) + 1, TILE_DEG)
    lons = range(int(math.floor(w)), int(math.floor(e)) + 1, TILE_DEG)
    return [(la, lo) for la in lats for lo in lons]


def tile_url(lat: int, lon: int, product: str = DEFAULT_PRODUCT) -> str:
    name = tile_name(lat, lon, product)
    return f"{PRODUCTS[product]['bucket']}/{name}/{name}.tif"


# ---------------------------------------------------------------- fetching
def fetch_tile(lat: int, lon: int, *, product: str = DEFAULT_PRODUCT,
               cache_dir: Optional[str] = None, allow_network: bool = False,
               timeout_s: int = DEFAULT_TIMEOUT_S) -> Optional[str]:
    """One tile on disk, or None where the product genuinely has no such tile.

    None is reserved for a 403/404 - Copernicus publishes no tile over open
    ocean, and a scene on a coast legitimately touches one. Every other failure
    raises, because a network error that silently drops a tile would leave a
    hole in the mosaic that anchors would then be harvested around.
    """
    if product in MANUAL_PRODUCTS:
        raise DEMUnavailable(MANUAL_PRODUCTS[product])
    if product not in PRODUCTS:
        raise DEMUnavailable(
            f"unknown DEM product '{product}'. Fetchable: {', '.join(PRODUCTS)}. "
            f"By hand: {', '.join(MANUAL_PRODUCTS)}.")

    cache_dir = cache_dir or default_cache_dir()
    name = tile_name(lat, lon, product)
    path = os.path.join(cache_dir, name + ".tif")
    missing = path + ".missing"
    if os.path.exists(path):
        return path
    if os.path.exists(missing):
        return None
    if not allow_network:
        raise DEMUnavailable(
            f"tile {name} is not cached and network access was not requested. "
            "Pass --fetch-dem to allow the download, or supply a local GeoTIFF "
            "with --dem <path>.")

    os.makedirs(cache_dir, exist_ok=True)
    url = tile_url(lat, lon, product)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, \
                open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        if exc.code in (403, 404):
            # Remembered, so a scene on a coastline does not re-request a tile
            # that does not exist on every run.
            with open(missing, "w", encoding="utf-8") as fh:
                fh.write(f"{url}\nHTTP {exc.code} at "
                         f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            return None
        raise DEMUnavailable(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise DEMUnavailable(f"could not fetch {url}: {exc}") from exc

    os.replace(tmp, path)
    return path


# --------------------------------------------------------------- assembling
def load_for_scene(meta, shape: tuple, *, product: str = DEFAULT_PRODUCT,
                   allow_network: bool = False, cache_dir: Optional[str] = None,
                   timeout_s: int = DEFAULT_TIMEOUT_S) -> tuple:
    """A public DEM resampled onto the scene grid, plus its provenance.

    Returns `(dem_m, provenance)`. Every tile is reprojected into the same
    destination and combined by filling holes rather than by overwriting, so a
    scene straddling a degree line gets a seamless surface and a tile that does
    not exist leaves NaN rather than zero. Zero is an elevation; NaN is not, and
    the anchor harvester tests for finiteness.
    """
    if not getattr(meta, "georeferenced", False):
        raise DEMUnavailable(
            "the scene is not georeferenced, so no global DEM tile can be placed "
            "on it. Supply a DEM already on the image grid, or georeference the image.")
    bounds = getattr(meta, "bounds_wgs", None)
    if not bounds:
        raise DEMUnavailable("the scene has no WGS84 bounds, so no tile can be chosen")

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    H, W = int(shape[0]), int(shape[1])
    out = np.full((H, W), np.nan, np.float32)
    used, absent = [], []

    for lat, lon in tiles_for_bounds(bounds):
        path = fetch_tile(lat, lon, product=product, cache_dir=cache_dir,
                          allow_network=allow_network, timeout_s=timeout_s)
        if path is None:
            absent.append(tile_name(lat, lon, product))
            continue
        tmp = np.full((H, W), np.nan, np.float32)
        with rasterio.open(path) as ds:
            reproject(
                source=rasterio.band(ds, 1),
                destination=tmp,
                dst_transform=Affine(*meta.transform),
                dst_crs=meta.crs,
                src_nodata=ds.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        out = np.where(np.isfinite(out), out, tmp)
        used.append(os.path.basename(path))

    if not np.isfinite(out).any():
        raise DEMUnavailable(
            f"no {PRODUCTS[product]['name']} coverage over this scene "
            f"(tiles tried: {', '.join(absent) or 'none'})")

    spec = PRODUCTS[product]
    prov = {
        "product": spec["name"],
        "source": product,
        "kind": spec["kind"],
        "posting_m": spec["posting_m"],
        "vertical_datum": spec["datum"],
        "tiles": used,
        "tiles_absent": absent,
        "coverage": round(float(np.isfinite(out).mean()), 4),
        "note": spec["note"],
    }
    return out, prov


def describe(product: str = DEFAULT_PRODUCT) -> str:
    if product in PRODUCTS:
        s = PRODUCTS[product]
        return (f"{s['name']}: {s['kind']} at {s['posting_m']:.0f} m posting, "
                f"{s['datum']} vertical datum - {s['note']}")
    if product in MANUAL_PRODUCTS:
        return MANUAL_PRODUCTS[product]
    return f"unknown DEM product '{product}'"
