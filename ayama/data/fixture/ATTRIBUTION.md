# Test fixture provenance

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
