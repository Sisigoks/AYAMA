"""Fetch a real scene with real elevation truth. No registration, no account.

    python scripts/fetch_swisstopo.py --out data/real/zurich

Everything measured in README sections 3 to 5 came from a renderer this project
wrote. This is the smallest honest step off that: a real orthophoto over a real
city, with an airborne-lidar DSM and DTM for the same ground, all published by
swisstopo under an Open Government Data licence that permits commercial use.

Why swisstopo and not a satellite benchmark. The satellite datasets that ship
co-registered height truth - DFC2019/US3D above all - sit behind a registration
wall, so a script cannot fetch them and a claim built on them cannot be
reproduced by running one command. swisstopo is the best source that is simply
downloadable. The cost is honest and worth stating: this is AERIAL imagery, so
it tests the calibration against real surfaces and real building geometry, but
not against satellite viewing geometry or satellite radiometry.

What it builds:

    scene.tif        the orthophoto, resampled to the elevation grid
    scene_dsm.tif    swissSURFACE3D  - lidar DSM, the reference
    scene_dtm.tif    swissALTI3D     - lidar DTM, the true bare earth
    scene_dem.tif    that DTM degraded to a public DEM's posting and noise

The last one matters. swissALTI3D is survey-grade at 0.5 m; anchoring to it
directly would be a far easier problem than the one the method claims to solve.
It is degraded to Copernicus GLO-30's 30 m posting and 3 m correlated noise by
`simulate_public_dem`, so the anchors are no better than a free global DEM
would give and the imagery and the truth are the only real advantages.

Sun angles are computed from the acquisition date and the tile centre with
`ayama.core.solar`, and written into the GeoTIFF. Read the caveat that prints
with them: an orthophoto is a mosaic of many frames, so one sun vector is an
approximation in a way it is not for a single satellite acquisition.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.request
from datetime import datetime

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections"
COLLECTIONS = {
    "ortho": "ch.swisstopo.swissimage-dop10",
    "dsm": "ch.swisstopo.swisssurface3d-raster",
    "dtm": "ch.swisstopo.swissalti3d",
}
UA = {"User-Agent": "ayama-fetch/0.1"}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return json.load(r)


def find_assets(bbox: str, tile: str = "") -> dict:
    """Pick one tile present in all three collections, preferring nearby years."""
    found: dict = {}
    for kind, coll in COLLECTIONS.items():
        items = _get_json(f"{STAC}/{coll}/items?bbox={bbox}&limit=50").get("features", [])
        for f in items:
            if tile and tile not in f["id"]:
                continue
            key = f["id"].split("_")[-1]           # e.g. 2682-1246
            for name, a in f["assets"].items():
                if not name.endswith(".tif"):
                    continue
                found.setdefault(key, {}).setdefault(kind, []).append(
                    {"name": name, "href": a["href"], "datetime": f.get("properties", {}).get("datetime")})

    complete = {k: v for k, v in found.items() if len(v) == 3}
    if not complete:
        raise SystemExit(
            f"no tile under bbox {bbox} carries ortho + dsm + dtm.\n"
            "  try a different --bbox, or --tile to name one explicitly")
    key = tile if tile in complete else sorted(complete)[0]
    return {"tile": key, "assets": complete[key]}


def _pick(cands: list, resolution: str) -> dict:
    """Prefer the requested resolution, and the earliest year among matches."""
    at_res = [c for c in cands if f"_{resolution}_" in c["name"]]
    pool = at_res or cands
    return sorted(pool, key=lambda c: c["name"])[0]


def download(href: str, dest: str) -> str:
    if os.path.exists(dest):
        print(f"    have {os.path.basename(dest)}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(urllib.request.Request(href, headers=UA)) as r, \
            open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                pct = 100 * got / total
                print(f"\r    {os.path.basename(dest)}  {got/1e6:6.1f}/{total/1e6:.1f} MB "
                      f"({pct:3.0f}%)", end="", flush=True)
    os.replace(tmp, dest)
    print()
    return dest


def build_scene(raw: dict, out: str, size: int, gsd: float, when: str,
                stem: str = "scene") -> None:
    """Crop and resample the three rasters onto one common grid."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from ayama.core.solar import solar_position
    from ayama.core.types import SceneMeta
    from ayama.dsm.cog import write_cog, write_rgb
    from ayama.eval.simulate import simulate_public_dem

    # The DSM defines the grid: it is the reference, and resampling truth is
    # worse than resampling the inputs onto it.
    with rasterio.open(raw["dsm"]) as ds:
        crs, dsm_res = ds.crs, abs(ds.transform.a)
        left, top = ds.bounds.left, ds.bounds.top
        scale = gsd / dsm_res
        from rasterio.transform import Affine

        tr = Affine(gsd, 0, left, 0, -gsd, top)
        dsm = ds.read(
            1, out_shape=(int(ds.height / scale), int(ds.width / scale)),
            resampling=Resampling.bilinear)[:size, :size].astype("float32")

    def onto_grid(path, resampling=Resampling.bilinear, bands=1):
        with rasterio.open(path) as src:
            dst = np.zeros((bands, size, size), "float32")
            reproject(source=rasterio.band(src, list(range(1, bands + 1))),
                      destination=dst, dst_transform=tr, dst_crs=crs,
                      resampling=resampling)
        return dst

    dtm = onto_grid(raw["dtm"])[0]
    rgb = onto_grid(raw["ortho"], Resampling.average, bands=3)
    rgb = np.clip(rgb, 0, 255).astype("uint8").transpose(1, 2, 0)

    # sun position from the acquisition date and the tile centre
    cx, cy = left + size * gsd / 2, top - size * gsd / 2
    from rasterio.warp import transform as warp_pts

    lon, lat = warp_pts(crs, "EPSG:4326", [cx], [cy])
    lon, lat = lon[0], lat[0]
    dt = datetime.fromisoformat(when.replace("Z", "+00:00")) if when else None
    az, el = (solar_position(lat, lon, dt) if dt else (None, None))

    # swisstopo's STAC datetime for these products is a nominal year marker
    # (2019-01-01T00:00Z), not an acquisition instant. Fed to a solar model it
    # puts the sun 65 degrees BELOW the horizon, and writing that into the file
    # would be a fabricated number dressed as metadata. If the result is not a
    # daylight sun the tags are omitted: the pipeline then reports
    # `has_sun = False` and disables shadow physics, which is the correct
    # behaviour for imagery whose acquisition time we genuinely do not know.
    if el is None or el <= 5.0:
        if el is not None:
            print(f"\n  ! sun from the STAC datetime ({dt.isoformat()}) is {el:.1f} deg -")
            print("    that timestamp is a year marker, not an acquisition time.")
            print("    Writing no sun angles. Pass --when 2019-07-01T11:00:00Z if you")
            print("    know the real flight time and want shadow physics enabled.")
        az = el = None

    meta = SceneMeta(crs=str(crs), transform=(tr.a, tr.b, tr.c, tr.d, tr.e, tr.f),
                     gsd_m=gsd,
                     sun_azimuth_deg=None if az is None else round(az, 2),
                     sun_elevation_deg=None if el is None else round(el, 2),
                     acquired_utc=dt.isoformat() if dt else None, source="swisstopo")

    os.makedirs(out, exist_ok=True)
    tags = {"AYAMA_SOURCE": "swisstopo swissimage-dop10"}
    if az is not None:
        tags.update({"SUN_AZIMUTH": f"{az:.4f}", "SUN_ELEVATION": f"{el:.4f}"})
    write_rgb(os.path.join(out, stem + ".tif"), rgb, meta, tags=tags)
    write_cog(os.path.join(out, stem + "_dsm.tif"), dsm, meta,
              description="swissSURFACE3D lidar DSM (m)")
    write_cog(os.path.join(out, stem + "_dtm.tif"), dtm, meta,
              description="swissALTI3D lidar DTM (m)")
    write_cog(os.path.join(out, stem + "_dem.tif"),
              simulate_public_dem(dtm, gsd, source="copernicus"), meta,
              description="swissALTI3D degraded to Copernicus GLO-30 posting and noise")

    ndsm = np.maximum(dsm - dtm, 0)
    print(f"\n  scene       {size} x {size} px at {gsd} m  ({crs})")
    print(f"  centre      {lat:.4f} N  {lon:.4f} E")
    if az is None:
        print("  sun         none written - acquisition time unknown")
        print("              shadow physics is disabled for this scene")
    else:
        print(f"  sun         {az:.1f} deg az / {el:.1f} deg el   (from {dt.date()})")
        print("              NOTE: an ortho is a mosaic of many frames, so one sun")
        print("              vector is an approximation. A satellite acquisition is")
        print("              a single instant and does not carry this caveat.")
    print(f"  elevation   {float(dsm.min()):.1f} .. {float(dsm.max()):.1f} m")
    print(f"  true nDSM   max {float(ndsm.max()):.1f} m, "
          f"mean over >2 m {float(ndsm[ndsm > 2].mean()):.1f} m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/real/zurich")
    ap.add_argument("--bbox", default="8.530,47.365,8.545,47.375", help="lon,lat,lon,lat")
    ap.add_argument("--tile", default="", help="swisstopo tile key, e.g. 2682-1246")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--ortho-res", default="0.1", choices=["0.1", "2"])
    ap.add_argument("--cache", default="data/_cache",
                    help="raw downloads. Keep it OUTSIDE the dataset root, or "
                         "scene discovery treats every raw tile as a scene")
    ap.add_argument("--when", default="",
                    help="acquisition time, e.g. 2019-07-01T11:00:00Z; enables shadow physics")
    args = ap.parse_args()

    socket.setdefaulttimeout(120)
    print("AYAMA fetch   swisstopo open government data")
    sel = find_assets(args.bbox, args.tile)
    print(f"  tile         {sel['tile']}")

    raw, when = {}, None
    for kind in ("ortho", "dsm", "dtm"):
        pick = _pick(sel["assets"][kind], args.ortho_res if kind == "ortho" else "0.5")
        when = when or (pick["datetime"] if kind == "ortho" else None)
        raw[kind] = download(pick["href"], os.path.join(args.cache, pick["name"]))

    stem = os.path.basename(os.path.normpath(args.out)) or "scene"
    build_scene(raw, args.out, args.size, args.gsd, args.when or when or "", stem)
    print(f"\n  written to   {os.path.abspath(args.out)}")
    print("\n  run it:")
    print(f"    python -m ayama.cli run {args.out}/{stem}.tif --out out/real \\")
    print(f"        --dem {args.out}/{stem}_dem.tif --ref {args.out}/{stem}_dsm.tif")
    print("\n  or the whole collection at once:")
    print(f"    python -m ayama.cli dataset {os.path.dirname(args.out)} --layout generic \\")
    print("        --backbone dav2-vitl --out results/real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
