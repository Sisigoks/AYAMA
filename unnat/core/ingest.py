"""Read an image into a Scene: pixels plus everything we can learn about geometry.

Two paths. A GeoTIFF carries CRS, affine and usually the sun angles as tags. A
plain PNG/JPG carries at best EXIF GPS and a timestamp, from which the sun
angles are computed. Anything missing is left as None so downstream stages can
degrade the calibration tier honestly instead of inventing numbers.
"""
from __future__ import annotations

import os
from datetime import timezone
from typing import Any, Optional

import numpy as np

from .geo import gsd_metres, percentile_stretch
from .solar import parse_utc, solar_position
from .types import Scene, SceneMeta

try:  # rasterio is optional so Phase 1 runs on a bare numpy install
    import rasterio
    from rasterio.warp import transform_bounds

    HAVE_RASTERIO = True
except Exception:  # pragma: no cover - only on installs without rasterio
    rasterio = None
    HAVE_RASTERIO = False

GEOTIFF_EXT = (".tif", ".tiff", ".gtiff")

# Tag spellings seen across Maxar, Planet, Airbus, ISRO and QGIS exports.
_SUN_AZ_KEYS = ("SUN_AZIMUTH", "SUNAZIMUTH", "SOLAR_AZIMUTH", "MEANSUNAZ", "SUN_AZ")
_SUN_EL_KEYS = ("SUN_ELEVATION", "SUNELEVATION", "SOLAR_ELEVATION", "MEANSUNEL",
                "SUN_ELEV", "SUN_EL")
_OFF_NADIR_KEYS = ("OFF_NADIR", "OFFNADIR", "MEANOFFNADIRVIEWANGLE",
                   "MEAN_OFF_NADIR_VIEW_ANGLE", "VIEW_ANGLE")
_DATE_KEYS = ("ACQUISITION_DATE", "ACQUISITIONDATETIME", "ACQUIRED", "DATE_ACQUIRED",
              "TIFFTAG_DATETIME", "EARLIESTACQTIME", "DATETIME")


def _all_tags(ds) -> dict:
    tags = {}
    try:
        tags.update(ds.tags())
    except Exception:
        pass
    try:
        namespaces = ds.tag_namespaces()
    except Exception:
        namespaces = []
    for ns in namespaces:
        try:
            tags.update(ds.tags(ns=ns))
        except Exception:
            continue
    return {str(k).upper().replace(" ", "_"): v for k, v in tags.items()}


def _pick(tags: dict, keys: tuple) -> Optional[str]:
    for k in keys:
        if k in tags and str(tags[k]).strip():
            return str(tags[k]).strip()
    # Fall back to a contains-match, which catches vendor prefixes like NITF_.
    for k in keys:
        for tk, tv in tags.items():
            if k in tk and str(tv).strip():
                return str(tv).strip()
    return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(str(v).split()[0].rstrip("dDeg°"))
    except (TypeError, ValueError):
        return None


def _sun_elevation_from_zenith(tags: dict) -> Optional[float]:
    z = _as_float(_pick(tags, ("SUN_ZENITH", "SOLAR_ZENITH", "SUN_ZENITH_ANGLE")))
    return None if z is None else 90.0 - z


def _to_uint8_rgb(arr: np.ndarray, dtype_name: str) -> np.ndarray:
    """(H, W, C) of any dtype -> (H, W, 3) uint8 for a pretrained encoder."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)
    if dtype_name == "uint8" and np.isfinite(arr).all():
        return np.ascontiguousarray(arr.astype(np.uint8))
    return percentile_stretch(arr)


def _ingest_geotiff(path: str) -> Scene:
    with rasterio.open(path) as ds:
        idx = list(range(1, min(ds.count, 3) + 1)) or [1]
        arr = ds.read(idx).transpose(1, 2, 0)
        raw_dtype = str(ds.dtypes[0])
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr.astype(np.float32))
        rgb = _to_uint8_rgb(arr, raw_dtype)

        tags = _all_tags(ds)
        crs = str(ds.crs) if ds.crs else None
        transform = tuple(ds.transform)[:6] if ds.transform else None

        bounds_wgs = None
        centre_lat = None
        if ds.crs:
            try:
                bounds_wgs = tuple(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
                centre_lat = 0.5 * (bounds_wgs[1] + bounds_wgs[3])
            except Exception:
                bounds_wgs = None
        gsd, assumed = gsd_metres(
            transform, bool(ds.crs and ds.crs.is_geographic), centre_lat
        )

        sun_az = _as_float(_pick(tags, _SUN_AZ_KEYS))
        sun_el = _as_float(_pick(tags, _SUN_EL_KEYS))
        if sun_el is None:
            sun_el = _sun_elevation_from_zenith(tags)
        acquired = _pick(tags, _DATE_KEYS)

        if (sun_az is None or sun_el is None) and bounds_wgs and acquired:
            when = parse_utc(acquired)
            if when is not None:
                lon = 0.5 * (bounds_wgs[0] + bounds_wgs[2])
                lat = 0.5 * (bounds_wgs[1] + bounds_wgs[3])
                sun_az, sun_el = solar_position(lat, lon, when)

        meta = SceneMeta(
            crs=crs,
            transform=transform,
            gsd_m=gsd,
            bounds_wgs=bounds_wgs,
            sun_azimuth_deg=sun_az,
            sun_elevation_deg=sun_el,
            off_nadir_deg=_as_float(_pick(tags, _OFF_NADIR_KEYS)) or 0.0,
            acquired_utc=acquired,
            source="geotiff",
            gsd_is_assumed=assumed,
        )
    return Scene(rgb=rgb, meta=meta, path=path, raw_dtype=raw_dtype)


def _exif_gps(exif) -> Optional[tuple]:
    from PIL.ExifTags import GPSTAGS

    try:
        gps_raw = exif.get_ifd(0x8825)
    except Exception:
        gps_raw = None
    if not gps_raw:
        return None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

    def _dms(val, ref, negative_ref):
        if val is None:
            return None
        try:
            d, m, s = (float(x) for x in val)
        except (TypeError, ValueError):
            return None
        deg = d + m / 60.0 + s / 3600.0
        return -deg if str(ref).upper().startswith(negative_ref) else deg

    lat = _dms(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"), "S")
    lon = _dms(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"), "W")
    if lat is None or lon is None:
        return None
    return lat, lon


def _ingest_image(path: str) -> Scene:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        raw_dtype = "uint8" if im.mode in ("RGB", "L", "P", "RGBA") else "uint16"
        try:
            exif = im.getexif()
        except Exception:
            exif = None
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    sun_az = sun_el = None
    acquired = None
    if exif:
        from PIL.ExifTags import TAGS

        flat = {TAGS.get(k, k): v for k, v in exif.items()}
        acquired = flat.get("DateTimeOriginal") or flat.get("DateTime")
        latlon = _exif_gps(exif)
        when = parse_utc(acquired)
        if latlon and when is not None:
            # EXIF timestamps are local-time-without-zone; treat as UTC and say so.
            when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
            sun_az, sun_el = solar_position(latlon[0], latlon[1], when)

    meta = SceneMeta(
        crs=None,
        transform=None,
        gsd_m=1.0,
        bounds_wgs=None,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        off_nadir_deg=0.0,
        acquired_utc=str(acquired) if acquired else None,
        source="image+exif" if sun_az is not None else "image",
        gsd_is_assumed=True,
    )
    return Scene(rgb=rgb, meta=meta, path=path, raw_dtype=raw_dtype)


def ingest(path: str) -> Scene:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(GEOTIFF_EXT):
        if not HAVE_RASTERIO:
            raise RuntimeError(
                "rasterio is required to read GeoTIFFs. "
                "pip install rasterio, or pass a PNG/JPG."
            )
        return _ingest_geotiff(path)
    return _ingest_image(path)
