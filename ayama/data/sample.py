"""The bundled sample scene: a real crop with real lidar truth.

Several things need *a* scene with known heights - the test suite, `preflight`,
`bench` - and for a long time this project generated one with a renderer. It no
longer does. Every number it publishes comes from real imagery, and a pipeline
smoke-tested on invented pixels is not the pipeline that ships; real data also
fails in ways a renderer does not, which is where the interesting bugs live.

So the sample is real: a 576 x 576 px crop of central Zurich at 0.5 m, with the
airborne lidar DSM and the bare-earth DTM for the same ground. It is bundled
with the package (about 2.5 MB) so `ayama preflight` works on a fresh install
with no network. `scripts/make_test_fixture.py` regenerates it, and
`ayama/data/fixture/ATTRIBUTION.md` carries the swisstopo licence and the
attribution it requires.

**No sun angles are recorded**, because swisstopo publishes a nominal year for
these products rather than an acquisition instant. Callers that need shadow
physics pass `sun=(azimuth, elevation)`, which both sets the metadata and
ray-marches a truth mask from the lidar surface - and which makes the assumption
explicit at the call site, where it belongs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

FIXTURE = os.path.join(os.path.dirname(__file__), "fixture")
FIXTURE_SIZE = 576


@dataclass
class SampleScene:
    """A real crop, presented the way the pipeline expects a scene."""

    rgb: np.ndarray          # (H, W, 3) uint8, orthophoto
    dsm_m: np.ndarray        # (H, W) float32, lidar surface model
    dtm_m: np.ndarray        # (H, W) float32, lidar bare earth
    sem: np.ndarray          # (H, W) uint8, derived labels
    shadow: np.ndarray       # (H, W) bool, empty unless a sun was supplied
    meta: object

    @property
    def ndsm_m(self) -> np.ndarray:
        return np.maximum(self.dsm_m - self.dtm_m, 0.0).astype(np.float32)

    def as_scene(self):
        from ..core.types import Scene

        return Scene(rgb=self.rgb, meta=self.meta, raw_dtype="uint8")


def available() -> bool:
    return os.path.exists(os.path.join(FIXTURE, "zurich_dsm.tif"))


def load_sample_scene(size: int = 384, sun: Optional[tuple] = None,
                      offset: tuple = (0, 0)) -> SampleScene:
    """The bundled Zurich crop, cropped or reflected to `size`.

    `sun` is `(azimuth_deg, elevation_deg)`. Supplying it sets the scene
    metadata and ray-marches a ground-truth shadow mask from the lidar DSM,
    which is how a caller states "assume this sun" - none is published.
    """
    import rasterio

    from ..core.types import SceneMeta

    if not available():
        raise FileNotFoundError(
            f"the bundled sample scene is missing from {FIXTURE}.\n"
            "  regenerate it with:\n"
            "    python scripts/fetch_swisstopo.py --out data/real/zurich\n"
            "    python scripts/make_test_fixture.py")

    def read(name, bands=1):
        with rasterio.open(os.path.join(FIXTURE, name)) as ds:
            tr = ds.transform
            a = (ds.read(1) if bands == 1
                 else np.stack([ds.read(b) for b in range(1, bands + 1)], -1))
            return a, str(ds.crs), (tr.a, tr.b, tr.c, tr.d, tr.e, tr.f)

    rgb, crs, tr = read("zurich_rgb.tif", bands=3)
    dsm, _, _ = read("zurich_dsm.tif")
    dtm, _, _ = read("zurich_dtm.tif")
    sem, _, _ = read("zurich_sem.tif")

    r0, c0 = offset

    def fit(a):
        """Crop to `size`, reflecting the crop if `size` exceeds the fixture."""
        if size <= FIXTURE_SIZE and r0 + size <= FIXTURE_SIZE and c0 + size <= FIXTURE_SIZE:
            return np.ascontiguousarray(a[r0:r0 + size, c0:c0 + size])
        reps = int(np.ceil(size / FIXTURE_SIZE))
        pad = [(0, (reps - 1) * FIXTURE_SIZE)] * 2 + [(0, 0)] * (a.ndim - 2)
        return np.ascontiguousarray(np.pad(a, pad, mode="reflect")[:size, :size])

    rgb, dsm, dtm, sem = fit(rgb), fit(dsm), fit(dtm), fit(sem)
    # The transform must follow the crop, or every georeferenced assertion is
    # off by the offset and the error looks like a projection bug.
    a, b, c, d, e, f = tr
    meta = SceneMeta(
        crs=crs, transform=(a, b, c + c0 * a, d, e, f + r0 * e), gsd_m=0.5,
        sun_azimuth_deg=None if sun is None else float(sun[0]),
        sun_elevation_deg=None if sun is None else float(sun[1]),
        source="swisstopo sample scene (see ayama/data/fixture/ATTRIBUTION.md)")

    if sun is None:
        shadow = np.zeros(dsm.shape, bool)
    else:
        from ..eval.shadow_truth import cast_shadow_mask

        shadow = cast_shadow_mask(dsm, float(sun[0]), float(sun[1]), 0.5)

    return SampleScene(rgb=rgb.astype(np.uint8), dsm_m=dsm.astype(np.float32),
                       dtm_m=dtm.astype(np.float32), sem=sem.astype(np.uint8),
                       shadow=shadow, meta=meta)
