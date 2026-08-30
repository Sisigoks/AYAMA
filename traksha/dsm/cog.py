"""Raster output. Everything the pipeline produces lands on disk in a format
QGIS can open without TRAKSHA installed, because that is the fallback demo when
anything downstream breaks.

Cloud Optimised GeoTIFF when GDAL supports the COG driver, a tiled GeoTIFF with
internal overviews otherwise, and a plain .npy plus PNG if rasterio is absent.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

from ..core.types import SceneMeta

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    HAVE_RASTERIO = True
except Exception:  # pragma: no cover
    rasterio = None
    HAVE_RASTERIO = False

NODATA = -9999.0


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_cog(
    path: str,
    arr: np.ndarray,
    meta: Optional[SceneMeta] = None,
    dtype: str = "float32",
    nodata: float = NODATA,
    description: str = "",
    tags: Optional[dict] = None,
) -> str:
    """Write a single-band raster. Returns the path actually written."""
    _ensure_parent(path)
    a = np.asarray(arr, dtype=np.float32)
    out = np.where(np.isfinite(a), a, nodata).astype(dtype)

    if not HAVE_RASTERIO:
        npy = os.path.splitext(path)[0] + ".npy"
        np.save(npy, a)
        side = os.path.splitext(path)[0] + ".json"
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(_meta_dict(meta, a.shape), fh, indent=2)
        return npy

    h, w = a.shape
    transform = Affine(*meta.transform) if (meta and meta.transform) else Affine.identity()
    crs = meta.crs if (meta and meta.crs) else None

    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=2 if dtype.startswith("float") else 1,
        BIGTIFF="IF_SAFER",
    )
    if _cog_driver_available():
        profile.update(driver="COG", overview_resampling="average")
        for k in ("tiled", "blockxsize", "blockysize", "BIGTIFF"):
            profile.pop(k, None)

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(out, 1)
        if description:
            ds.set_band_description(1, description)
        ds.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        if profile["driver"] == "GTiff":
            factors = [f for f in (2, 4, 8, 16) if min(h, w) // f >= 256]
            if factors:
                ds.build_overviews(factors, Resampling.average)
                ds.update_tags(ns="rio_overview", resampling="average")
    return path


def write_rgb(path: str, rgb: np.ndarray, meta: Optional[SceneMeta] = None,
              tags: Optional[dict] = None) -> str:
    """Write a 3-band uint8 raster (the aligned texture)."""
    _ensure_parent(path)
    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if not HAVE_RASTERIO or not path.lower().endswith((".tif", ".tiff")):
        from PIL import Image

        Image.fromarray(a).save(path, quality=92)
        return path
    h, w = a.shape[:2]
    transform = Affine(*meta.transform) if (meta and meta.transform) else Affine.identity()
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs=(meta.crs if meta else None), transform=transform, photometric="RGB",
        tiled=True, blockxsize=512, blockysize=512, compress="deflate",
    ) as ds:
        ds.write(a.transpose(2, 0, 1))
        if meta is not None:
            auto = {}
            if meta.sun_azimuth_deg is not None:
                auto["SUN_AZIMUTH"] = f"{meta.sun_azimuth_deg:.4f}"
            if meta.sun_elevation_deg is not None:
                auto["SUN_ELEVATION"] = f"{meta.sun_elevation_deg:.4f}"
            if meta.off_nadir_deg:
                auto["OFF_NADIR"] = f"{meta.off_nadir_deg:.4f}"
            if meta.acquired_utc:
                auto["ACQUISITION_DATE"] = str(meta.acquired_utc)
            auto.update(tags or {})
            ds.update_tags(**{k: str(v) for k, v in auto.items()})
        elif tags:
            ds.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def write_png_preview(
    path: str,
    arr: np.ndarray,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    max_side: int = 2048,
) -> str:
    """Quick-look PNG. Fixed vmin/vmax keeps consecutive runs comparable."""
    _ensure_parent(path)
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if vmin is None:
        vmin = float(np.percentile(a[finite], 1)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(a[finite], 99)) if finite.any() else 1.0
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0

    norm = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    norm = np.where(finite, norm, 0.0)

    rgba = _apply_cmap(norm, cmap)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)

    from PIL import Image

    im = Image.fromarray(rgba, mode="RGBA")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.BILINEAR)
    im.save(path)
    return path


def _apply_cmap(norm: np.ndarray, cmap: str) -> np.ndarray:
    try:
        import matplotlib

        lut = (matplotlib.colormaps[cmap](np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    except Exception:  # pragma: no cover - matplotlib-free fallback
        lut = _fallback_lut(cmap)
    idx = (norm * 255).astype(np.uint8)
    return lut[idx]


def _fallback_lut(cmap: str) -> np.ndarray:
    anchors = {
        "gray": [(0, 0, 0), (255, 255, 255)],
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
        "magma": [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)],
        "terrain": [(51, 51, 153), (0, 153, 102), (243, 226, 137), (140, 92, 61), (255, 255, 255)],
        "RdBu_r": [(5, 48, 97), (146, 197, 222), (247, 247, 247), (244, 165, 130), (103, 0, 31)],
    }.get(cmap, [(0, 0, 0), (255, 255, 255)])
    stops = np.linspace(0, 255, len(anchors))
    lut = np.zeros((256, 4), np.uint8)
    lut[:, 3] = 255
    grid = np.arange(256)
    for c in range(3):
        lut[:, c] = np.interp(grid, stops, [a[c] for a in anchors]).astype(np.uint8)
    return lut


def _cog_driver_available() -> bool:
    if not HAVE_RASTERIO:
        return False
    try:
        from rasterio.env import Env

        with Env() as env:
            return "COG" in env.drivers()
    except Exception:
        return False


def _meta_dict(meta: Optional[SceneMeta], shape: Sequence[int]) -> dict:
    d = {"height": int(shape[0]), "width": int(shape[1])}
    if meta is not None:
        d.update(
            crs=meta.crs,
            transform=list(meta.transform) if meta.transform else None,
            gsd_m=meta.gsd_m,
            sun_azimuth_deg=meta.sun_azimuth_deg,
            sun_elevation_deg=meta.sun_elevation_deg,
            off_nadir_deg=meta.off_nadir_deg,
            acquired_utc=meta.acquired_utc,
        )
    return d
