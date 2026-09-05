"""OpenStreetMap footprints and road centrelines, as a geometric prior.

README section 2.8 rejects OpenStreetMap and gives a good reason for it -
building *heights* there are sparse, inconsistently attributed and tied to no
particular acquisition. That judgement is about heights and it still stands.
Nothing in this module reads a height.

What it reads is **shape** and **where the ground is**, and those are different
claims with different evidence behind them.

**Shape.** A footprint in this pipeline comes from SAM 2, which decodes its
masks at 256x256 and upsamples: the outline is right to a few metres and it is
*staircased*, because `mesh.structural` walks raster cells and emits an
axis-aligned wall quad per cell edge. A European city building is a polygon with
a handful of straight sides meeting at right angles, and OSM has that polygon,
surveyed, as vectors. Using it to regularise an outline is not importing a
measurement - it is importing a *shape prior*, and the pipeline still decides
from the image which pixels are building.

**Where the ground is.** The road network is the stronger half. Chhaya's DEM
anchor gate (`chhaya.anchors.harvest_dem`) admits a DEM sample only where the
scene is bare earth, and it decides that from a five-class colour segmentation
that confuses asphalt with roof, shadow with water, and a gravel yard with
anything. A road centreline from OSM is a surveyed assertion that the ground is
at that location, and a 30 m DEM posting sampled on a road is the closest a
public DEM gets to being right. That is what makes this worth the network call.

**Registration is not assumed.** OSM and the orthophoto are different
acquisitions and can disagree by metres. So nothing here is authoritative:
`mesh.regularize` adopts an OSM polygon only where it already agrees with the
SAM 2 mask, and reports the offset it measured rather than trusting it. This
module's job ends at handing over geometry in pixel coordinates, with the
provenance attached.

**The network call is opt-in and cached.** A run must never silently proceed
with data it failed to fetch - the same discipline `api.pipeline.load_dem`
applies to DEMs. `fetch` without `allow_network=True` reads the cache or raises;
it does not quietly return nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Overpass asks for a User-Agent that identifies the client. Without one the
# public instance answers 406 Not Acceptable, which is easy to misread as a
# malformed query - it is not, it is a missing header.
USER_AGENT = "traksha/0.1 (+https://github.com/traksha; research use)"

# Mirrors in preference order. The public instance rate-limits aggressively and
# Kumi is the usual second choice; a run tries each once rather than hammering
# the first.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

DEFAULT_TIMEOUT_S = 180

# Carriageway width by highway class, metres, used only when the way carries no
# `width` tag. These are the width of the *paved surface*, which is what the
# orthophoto sees and what a DEM posting lands on - not the legal right of way,
# which is much wider and would swallow the buildings either side.
ROAD_WIDTH_M = {
    "motorway": 14.0, "motorway_link": 8.0,
    "trunk": 12.0, "trunk_link": 8.0,
    "primary": 12.0, "primary_link": 7.0,
    "secondary": 10.0, "secondary_link": 6.0,
    "tertiary": 8.0, "tertiary_link": 6.0,
    "unclassified": 6.0, "residential": 6.0, "living_street": 5.0,
    "service": 4.0, "track": 3.5, "busway": 7.0,
    "pedestrian": 5.0, "footway": 2.5, "cycleway": 2.5,
    "path": 2.0, "steps": 1.5, "corridor": 2.0,
}
DEFAULT_ROAD_WIDTH_M = 6.0

# Ways tagged with these are not a surface a DEM posting can be trusted on:
# a bridge deck is metres above the terrain and a tunnel is metres below it.
# Both would enter the anchor system as a confident, wrong elevation.
ELEVATED_TAGS = ("bridge", "tunnel")


class OSMUnavailable(RuntimeError):
    """OSM data was asked for and could not be obtained. Never returned as empty."""


# --------------------------------------------------------------------- cache
def cache_key(bounds_wgs: Sequence[float]) -> str:
    """A stable name for one bounding box, rounded so jitter does not miss.

    Six decimal places is about 0.1 m at the equator - finer than any scene
    boundary this pipeline sees - so two runs over the same tile hit the same
    cache entry, and two runs over different tiles never do.
    """
    w, s, e, n = (round(float(v), 6) for v in bounds_wgs)
    raw = f"{w},{s},{e},{n}"
    return f"osm_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def default_cache_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "data", "_cache", "osm")


# ------------------------------------------------------------------- fetching
def build_query(bounds_wgs: Sequence[float], timeout_s: int = DEFAULT_TIMEOUT_S) -> str:
    """The Overpass QL for every building and every highway in the box.

    Overpass takes its bounding box as (south, west, north, east), which is the
    opposite order to the (west, south, east, north) that `SceneMeta.bounds_wgs`
    and every rasterio call use. Converting it in exactly one place is the only
    defence against a query that returns a plausible number of buildings from
    the wrong continent.
    """
    w, s, e, n = (float(v) for v in bounds_wgs)
    box = f"{s},{w},{n},{e}"
    return (
        f"[out:json][timeout:{int(timeout_s)}];"
        "("
        f'way["building"]({box});'
        f'relation["building"]({box});'
        f'way["highway"]({box});'
        ");"
        "out geom;"
    )


def fetch(
    bounds_wgs: Optional[Sequence[float]],
    *,
    allow_network: bool = False,
    cache_dir: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    refresh: bool = False,
) -> dict:
    """Raw Overpass JSON for a bounding box, from cache or from the network.

    Raises `OSMUnavailable` rather than returning an empty result, because an
    empty result is indistinguishable from "this tile genuinely has no buildings"
    and a caller that cannot tell those apart will report the wrong thing.
    """
    if not bounds_wgs:
        raise OSMUnavailable(
            "the scene has no WGS84 bounds, so there is no box to query. "
            "OSM needs a georeferenced image (SceneMeta.crs and .transform)."
        )
    cache_dir = cache_dir or default_cache_dir()
    path = os.path.join(cache_dir, cache_key(bounds_wgs) + ".json")

    if os.path.exists(path) and not refresh:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    if not allow_network:
        raise OSMUnavailable(
            f"no cached OSM extract for this scene ({os.path.basename(path)}) and "
            "network access was not requested. Pass --osm to allow the Overpass "
            "call, or place a cached extract at that path."
        )

    query = build_query(bounds_wgs, timeout_s)
    data = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for url in OVERPASS_MIRRORS:
        req = urllib.request.Request(
            url, data=data,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s + 30) as resp:
                payload = json.load(resp)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError) as exc:
            errors.append(f"{urllib.parse.urlparse(url).netloc}: {exc}")
            time.sleep(1.0)
    else:
        raise OSMUnavailable(
            "every Overpass mirror refused the query - " + "; ".join(errors))

    os.makedirs(cache_dir, exist_ok=True)
    payload.setdefault("traksha", {})
    payload["traksha"] = {"bounds_wgs": [float(v) for v in bounds_wgs],
                          "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime())}
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    return payload


# ------------------------------------------------------------- to pixel space
@dataclass
class OSMLayer:
    """OSM geometry expressed in this scene's pixel grid.

    Rings and lines are (N, 2) float arrays of (row, col). Float, not int:
    a footprint corner sits where it sits, and rounding it to the raster before
    the regulariser has had a look is throwing away the sub-pixel precision that
    is the whole reason to prefer a vector outline over a mask.
    """

    buildings: list = field(default_factory=list)      # list[(N,2) float array]
    roads: list = field(default_factory=list)          # list[(N,2) float array]
    road_width_m: list = field(default_factory=list)   # one per road
    road_elevated: list = field(default_factory=list)  # bridge/tunnel flag
    provenance: dict = field(default_factory=dict)

    @property
    def counts(self) -> dict:
        return {"buildings": len(self.buildings), "roads": len(self.roads),
                "roads_elevated": int(sum(self.road_elevated))}

    def summary(self) -> dict:
        d = dict(self.counts)
        d.update(self.provenance)
        return d


def _way_geometry(el: dict) -> Optional[np.ndarray]:
    geom = el.get("geometry")
    if not geom:
        return None
    return np.array([[g["lat"], g["lon"]] for g in geom], float)


def _relation_outers(el: dict) -> list:
    """The outer rings of a multipolygon relation.

    Only outer roles are taken. An inner ring is a courtyard, and a courtyard
    is not something this pipeline's footprints can express - `structural._solid`
    extrudes a simply-connected cell mask - so importing one would produce a
    ring of walls around a hole that the mesh then fills in anyway.
    """
    out = []
    for m in el.get("members") or []:
        if m.get("role") != "outer" or m.get("type") != "way":
            continue
        geom = m.get("geometry")
        if geom and len(geom) >= 4:
            out.append(np.array([[g["lat"], g["lon"]] for g in geom], float))
    return out


def _road_width(tags: dict) -> float:
    raw = tags.get("width") or tags.get("est_width")
    if raw:
        try:
            # OSM widths are metres unless suffixed; "12 m" and "12" both occur.
            return float(str(raw).split()[0].replace(",", "."))
        except (ValueError, IndexError):
            pass
    lanes = tags.get("lanes")
    if lanes:
        try:
            # 3.0 m of carriageway per lane is the European design figure.
            return max(3.0, float(lanes) * 3.0)
        except ValueError:
            pass
    return ROAD_WIDTH_M.get(str(tags.get("highway", "")), DEFAULT_ROAD_WIDTH_M)


def to_pixels(payload: dict, meta, shape: tuple) -> OSMLayer:
    """Project an Overpass response onto the scene grid.

    Two hops, and both are needed. OSM is WGS84 lat/lon; the scene is very often
    a projected CRS (EPSG:2056 for the Swiss tiles here), so the coordinates go
    through PROJ before they go through the affine. Skipping the reprojection
    and treating degrees as metres is the same class of error `core.geo`'s
    `gsd_metres` exists to prevent, and it fails silently - the footprints land
    in a plausible-looking cluster a few pixels across.
    """
    from ..core.geo import world_to_pixel

    if meta is None or not meta.georeferenced:
        raise OSMUnavailable(
            "the scene is not georeferenced, so OSM geometry cannot be placed on it")

    elements = payload.get("elements") or []
    lat_lon, kinds, tag_list = [], [], []
    for el in elements:
        tags = el.get("tags") or {}
        if el.get("type") == "way" and "building" in tags:
            g = _way_geometry(el)
            if g is not None and len(g) >= 4:
                lat_lon.append(g); kinds.append("building"); tag_list.append(tags)
        elif el.get("type") == "relation" and "building" in tags:
            for g in _relation_outers(el):
                lat_lon.append(g); kinds.append("building"); tag_list.append(tags)
        elif el.get("type") == "way" and "highway" in tags:
            g = _way_geometry(el)
            if g is not None and len(g) >= 2:
                lat_lon.append(g); kinds.append("road"); tag_list.append(tags)

    layer = OSMLayer(provenance={
        "source": "overpass",
        "elements": len(elements),
        "crs": meta.crs,
        "fetched_utc": (payload.get("traksha") or {}).get("fetched_utc"),
        # OpenStreetMap data is ODbL 1.0, and the licence's attribution and
        # share-alike terms attach to anything derived from it. A run that used
        # OSM therefore has to say so in its own artifacts, not only in this
        # repository's LICENCE - so the notice travels in the provenance and
        # ends up in the run's osm.json rather than depending on whoever
        # publishes the output remembering.
        "licence": "ODbL 1.0",
        "attribution": "© OpenStreetMap contributors",
        "attribution_url": "https://www.openstreetmap.org/copyright",
    })
    if not lat_lon:
        return layer

    # One PROJ call for every vertex in the scene, not one per way: the
    # transform has per-call setup cost that dominates on a few thousand
    # short ways.
    from rasterio.warp import transform as warp_transform

    sizes = [len(g) for g in lat_lon]
    flat = np.concatenate(lat_lon, 0)
    xs, ys = warp_transform("EPSG:4326", meta.crs,
                            flat[:, 1].tolist(), flat[:, 0].tolist())
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    rows = np.empty(xs.shape, float)
    cols = np.empty(xs.shape, float)
    for i in range(xs.size):
        rows[i], cols[i] = world_to_pixel(meta.transform, xs[i], ys[i])
    px = np.stack([rows, cols], 1)

    h, w = int(shape[0]), int(shape[1])
    off = 0
    for size, kind, tags in zip(sizes, kinds, tag_list):
        ring = px[off:off + size]
        off += size
        # Drop anything that misses the raster entirely. Overpass returns every
        # way that *intersects* the box, including ones whose geometry is
        # mostly outside it, and a clip is not needed - the rasteriser handles
        # partial coverage - but a way with no vertex anywhere near the scene is
        # just weight in every downstream loop.
        if not _touches(ring, h, w):
            continue
        if kind == "building":
            layer.buildings.append(ring)
        else:
            layer.roads.append(ring)
            layer.road_width_m.append(_road_width(tags))
            layer.road_elevated.append(
                any(str(tags.get(t, "no")).lower() not in ("no", "")
                    for t in ELEVATED_TAGS))
    return layer


def _touches(ring: np.ndarray, h: int, w: int, pad: float = 64.0) -> bool:
    r0, c0 = ring.min(0)
    r1, c1 = ring.max(0)
    return not (r1 < -pad or c1 < -pad or r0 > h + pad or c0 > w + pad)


# ------------------------------------------------------------- rasterisation
def fill_polygon(ring: np.ndarray, shape: tuple, out: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """Even-odd scanline fill of one (row, col) ring.

    Written out rather than delegated to rasterio.features or cv2, because
    neither is a dependency this stage should add - rasterio is present but its
    `features` module wants a GDAL geometry, and OpenCV is not installed. A
    scanline fill over the ring's own bounding box is a dozen lines and is
    exact, which matters: this mask is compared against a SAM mask by IoU, and
    a rasteriser that disagrees by half a pixel at every edge moves that number.

    Pixel centres are the sample points, matching `core.geo.pixel_to_world`.
    """
    h, w = int(shape[0]), int(shape[1])
    if out is None:
        out = np.zeros((h, w), bool)
    if len(ring) < 3:
        return out

    r = np.asarray(ring[:, 0], float)
    c = np.asarray(ring[:, 1], float)
    # Close the ring if the source did not.
    if r[0] != r[-1] or c[0] != c[-1]:
        r = np.append(r, r[0]); c = np.append(c, c[0])

    r0 = max(0, int(np.floor(r.min())))
    r1 = min(h - 1, int(np.ceil(r.max())))
    if r1 < r0:
        return out

    ra, rb = r[:-1], r[1:]
    ca, cb = c[:-1], c[1:]
    # Horizontal edges contribute no crossings and would divide by zero.
    live = ra != rb
    ra, rb, ca, cb = ra[live], rb[live], ca[live], cb[live]
    if ra.size == 0:
        return out
    lo = np.minimum(ra, rb)
    hi = np.maximum(ra, rb)
    slope = (cb - ca) / (rb - ra)

    for row in range(r0, r1 + 1):
        y = row + 0.5
        # Half-open in y so a vertex shared by two edges is counted once, which
        # is what stops a single-pixel hole appearing at every ring corner.
        hit = (lo <= y) & (y < hi)
        if not hit.any():
            continue
        xs = ca[hit] + (y - ra[hit]) * slope[hit]
        xs.sort()
        for i in range(0, xs.size - 1, 2):
            a = int(np.ceil(xs[i] - 0.5))
            b = int(np.floor(xs[i + 1] - 0.5))
            if b < a:
                continue
            a = max(a, 0); b = min(b, w - 1)
            if b >= a:
                out[row, a:b + 1] = True
    return out


def _draw_line(out: np.ndarray, p0, p1) -> None:
    """Rasterise one segment by supersampling along it.

    Bresenham would do, but the endpoints here are floats and the segments are
    short; stepping at half-pixel intervals is simpler, has no special cases,
    and cannot leave a gap.
    """
    h, w = out.shape
    d = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(2, int(np.ceil(d * 2.0)) + 1)
    t = np.linspace(0.0, 1.0, n)
    rr = np.rint(p0[0] + t * (p1[0] - p0[0])).astype(int)
    cc = np.rint(p0[1] + t * (p1[1] - p0[1])).astype(int)
    keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    out[rr[keep], cc[keep]] = True


def road_mask(layer: OSMLayer, shape: tuple, gsd_m: float,
              include_elevated: bool = False,
              max_width_m: float = 30.0) -> np.ndarray:
    """Bare-earth mask from the road network: centrelines widened to carriageway.

    `include_elevated` is False by design. A bridge deck is metres above the
    terrain it crosses and a tunnel portal is metres below it, so a DEM sample
    taken on one is not a weak terrain anchor - it is a confidently wrong one,
    the same failure mode `chhaya.anchors.harvest_dem`'s semantic gate exists to
    prevent for rooftops.

    Widening is done per width class rather than per way so the dilation runs a
    handful of times instead of once per road.
    """
    from scipy.ndimage import binary_dilation

    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), bool)
    if not layer.roads:
        return out

    gsd = max(float(gsd_m), 1e-6)
    by_radius: dict = {}
    for ring, width_m, elevated in zip(layer.roads, layer.road_width_m,
                                       layer.road_elevated):
        if elevated and not include_elevated:
            continue
        width_m = min(float(width_m), float(max_width_m))
        # Half-width in pixels, minus the half pixel the centreline itself
        # already covers. Below zero the line alone is wider than the road.
        radius = int(round(max(0.0, (width_m / gsd) / 2.0 - 0.5)))
        by_radius.setdefault(radius, []).append(ring)

    for radius, rings in by_radius.items():
        band = np.zeros((h, w), bool)
        for ring in rings:
            for i in range(len(ring) - 1):
                _draw_line(band, ring[i], ring[i + 1])
        if radius > 0:
            # A disc, not a square: a square structuring element widens a
            # diagonal road by sqrt(2) and puts its edge inside the buildings.
            k = 2 * radius + 1
            yy, xx = np.mgrid[:k, :k] - radius
            band = binary_dilation(band, (yy ** 2 + xx ** 2) <= radius ** 2 + 1e-9)
        out |= band
    return out


def building_mask(layer: OSMLayer, shape: tuple) -> np.ndarray:
    """Every OSM footprint in the scene, as one raster mask."""
    out = np.zeros((int(shape[0]), int(shape[1])), bool)
    for ring in layer.buildings:
        fill_polygon(ring, shape, out)
    return out


def load(meta, shape: tuple, *, allow_network: bool = False,
         cache_dir: Optional[str] = None, refresh: bool = False) -> OSMLayer:
    """Fetch (or read from cache) and project, in one call. The usual entry point."""
    payload = fetch(getattr(meta, "bounds_wgs", None), allow_network=allow_network,
                    cache_dir=cache_dir, refresh=refresh)
    return to_pixels(payload, meta, shape)


# -------------------------------------------------------- sharpening the gate
def refine_semantics(sem: np.ndarray, layer: OSMLayer, gsd_m: float,
                     promote_roads: bool = True,
                     demote_buildings: bool = True) -> tuple:
    """Correct the five-class raster where OSM knows better. Returns `(sem, report)`.

    This is the change that makes OSM worth fetching for *elevation* rather than
    only for geometry, and the mechanism is `chhaya.anchors.harvest_dem`. That
    harvester samples a public DEM only where the scene is `DEM_ADMISSIBLE` -
    bare ground, road or water - and it decides that from a five-class colour
    classifier. The classifier's errors are not random. A grey roof reads as
    bare ground, and a DEM sample taken there is not a weak anchor, it is a
    confidently wrong one: it asserts that the terrain is at roof height.

    Two corrections, in this order, and each rests on different evidence.

    **Demote OSM building footprints to BUILDING.** Measured on the bundled
    Zurich fixture against airborne lidar, an OSM footprint has **0.962
    precision** for `nDSM > 2.5 m` - 96 pixels in 100 inside one really do carry
    height - and the best alignment between the two is at **zero pixel shift**.
    That is a strong enough prior to overrule a colour classifier about whether
    a pixel is bare ground.

    **Promote the road network to ROAD, outside those footprints.** A road
    centreline is a surveyed assertion that the ground is there, and the same
    fixture puts the median lidar `nDSM` under the OSM road mask at **-0.00 m**.
    Roads are the surface a 30 m DEM posting is most nearly right about, so
    admitting more of them is the cheapest accuracy there is here.

    Buildings are applied first and roads are masked against them, because where
    the two overlap - an arcade, a building over a passage - the footprint is
    the safer claim: wrongly calling a roof a road puts a DEM anchor on it.

    Elevated ways are already excluded upstream by `road_mask`: a bridge deck is
    not the terrain and neither is a tunnel portal.
    """
    from ..core.types import BUILDING, ROAD, WATER

    out = np.array(sem, copy=True)
    report = {"promoted_road_px": 0, "demoted_building_px": 0}
    shape = out.shape

    bmask = building_mask(layer, shape) if demote_buildings else np.zeros(shape, bool)
    if demote_buildings and bmask.any():
        # Water is never overruled. A footprint drawn over a canal - a boathouse,
        # a bridge-side building - would otherwise remove a flat-water anchor,
        # and those are the most reliable constraints in the whole system.
        change = bmask & (out != BUILDING) & (out != WATER)
        out[change] = BUILDING
        report["demoted_building_px"] = int(change.sum())

    if promote_roads:
        rmask = road_mask(layer, shape, gsd_m) & ~bmask
        change = rmask & (out != ROAD) & (out != WATER)
        out[change] = ROAD
        report["promoted_road_px"] = int(change.sum())

    total = float(out.size)
    report["changed_fraction"] = round(
        (report["promoted_road_px"] + report["demoted_building_px"]) / total, 4)
    return out, report
