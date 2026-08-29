"""Cut the committed test fixture out of a real swisstopo scene.

    python scripts/fetch_swisstopo.py --out data/real/zurich
    python scripts/make_test_fixture.py

The suite used to build its rasters with a renderer. Every published number now
comes from real imagery, and a test suite that exercises the pipeline on
invented pixels is testing a different pipeline from the one that ships - so the
fixture is real too: a 576 x 576 crop of central Zurich with the airborne lidar
DSM and DTM for the same ground.

It is committed (about 2 MB) so that a fresh clone can run the whole suite with
no network and no 350 MB download. swisstopo publishes these products as Open
Government Data permitting redistribution with attribution; see the licence note
in ayama/data/fixture/ATTRIBUTION.md, which this script writes.

The crop is chosen for building density rather than for looking nice: the
calibration is most interesting where there is relief to recover.
"""
from __future__ import annotations

import os
import sys

SIZE = 576
SRC = "data/real/zurich"
DST = "ayama/data/fixture"

ATTRIBUTION = """# Test fixture provenance

These rasters are a 576 x 576 px crop (0.5 m GSD, EPSG:2056) of central Zurich,
cut from Swiss federal open geodata by `scripts/make_test_fixture.py`.

| file | source product | what it is |
|---|---|---|
| `zurich_rgb.tif` | SWISSIMAGE 10 cm | airborne orthophoto, averaged to 0.5 m |
| `zurich_dsm.tif` | swissSURFACE3D Raster | airborne lidar digital surface model |
| `zurich_dtm.tif` | swissALTI3D | airborne lidar bare-earth terrain model |
| `zurich_sem.tif` | derived | the repository's own colour heuristic, baked at fixture-build time so tests do not re-run it |

**Source:** Federal Office of Topography swisstopo, <https://www.swisstopo.admin.ch>.

**Licence:** Swiss Open Government Data. Free use, including commercial use,
with attribution required: *"Source: Federal Office of Topography swisstopo"*.
See <https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices>.

**No sun angles are recorded.** swisstopo publishes a nominal year for these
products, not an acquisition instant, so the fixture carries no solar metadata.
Tests that exercise shadow physics choose a sun explicitly and say so; that
angle is a test parameter, never a measurement.
"""


def main() -> int:
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    from ayama.core.types import Scene, SceneMeta
    from ayama.dsm.cog import write_cog, write_rgb
    from ayama.semantics.segment import segment

    if not os.path.isdir(SRC):
        print(f"error: {SRC} not found. Run scripts/fetch_swisstopo.py first.",
              file=sys.stderr)
        return 1

    with rasterio.open(os.path.join(SRC, "zurich_dsm.tif")) as ds:
        full = ds.read(1)
        crs, tr = ds.crs, ds.transform

    # Pick the SIZE-wide window with the most relief above local ground: the
    # calibration is only interesting where there is height to recover.
    with rasterio.open(os.path.join(SRC, "zurich_dtm.tif")) as ds:
        dtm_full = ds.read(1)
    ndsm = np.maximum(full - dtm_full, 0)
    best, best_score = (0, 0), -1.0
    step = 64
    for r in range(0, full.shape[0] - SIZE + 1, step):
        for c in range(0, full.shape[1] - SIZE + 1, step):
            score = float((ndsm[r:r + SIZE, c:c + SIZE] > 3.0).mean())
            if score > best_score:
                best, best_score = (r, c), score
    r0, c0 = best
    print(f"  window     row {r0} col {c0}   {100 * best_score:.0f}% of pixels above 3 m")

    meta = SceneMeta(
        crs=str(crs), gsd_m=0.5,
        transform=(tr.a, tr.b, tr.c + c0 * tr.a, tr.d, tr.e, tr.f + r0 * tr.e),
        source="swisstopo (SWISSIMAGE / swissSURFACE3D / swissALTI3D)")
    _ = Affine  # transform is carried as a tuple on SceneMeta

    os.makedirs(DST, exist_ok=True)
    sl = (slice(r0, r0 + SIZE), slice(c0, c0 + SIZE))

    with rasterio.open(os.path.join(SRC, "zurich.tif")) as ds:
        rgb = np.stack([ds.read(b)[sl] for b in (1, 2, 3)], axis=-1).astype("uint8")
    write_rgb(os.path.join(DST, "zurich_rgb.tif"), rgb, meta,
              tags={"AYAMA_SOURCE": "swisstopo SWISSIMAGE 10cm"})
    write_cog(os.path.join(DST, "zurich_dsm.tif"), full[sl].astype("float32"), meta,
              description="swissSURFACE3D lidar DSM (m)")
    write_cog(os.path.join(DST, "zurich_dtm.tif"), dtm_full[sl].astype("float32"), meta,
              description="swissALTI3D lidar DTM (m)")

    # Bake the heuristic labels so tests get a stable semantic raster without
    # re-deriving one. This is a derived product, not a measurement.
    sem, prov = segment(Scene(rgb=rgb, meta=meta), method="heuristic")
    write_cog(os.path.join(DST, "zurich_sem.tif"), sem.astype("float32"), meta,
              dtype="uint8", nodata=255, description=f"derived labels ({prov})")

    with open(os.path.join(DST, "ATTRIBUTION.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(ATTRIBUTION)

    total = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST))
    nd = ndsm[sl]
    print(f"  elevation  {full[sl].min():.1f} .. {full[sl].max():.1f} m")
    print(f"  nDSM       max {nd.max():.1f} m, mean over >2 m {nd[nd > 2].mean():.1f} m")
    print(f"  written    {DST}/  ({total / 1e6:.1f} MB, committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
