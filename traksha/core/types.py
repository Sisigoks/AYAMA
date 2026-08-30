"""TRAKSHA data contracts.

Every pipeline stage is a pure function ``stage(input_contract) -> output_contract``.
These dataclasses are the interface between the six parallel workstreams; they are
defined on day zero and must not change without telling the whole team.

Pure python only: no torch, no rasterio imports in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np

AnchorSource = Literal["dem", "shadow", "icesat", "gcp", "water", "ground_plane"]
Branch = Literal["terrain", "object", "absolute"]

# Semantic class ids. Kept as plain ints so masks stay uint8 rasters.
BARE_GROUND = 0
ROAD = 1
BUILDING = 2
VEGETATION = 3
WATER = 4
CLASS_NAMES = {
    BARE_GROUND: "bare ground",
    ROAD: "road",
    BUILDING: "building",
    VEGETATION: "vegetation",
    WATER: "water",
}
# Classes where a public bare-earth DEM sample is a legitimate terrain anchor.
DEM_ADMISSIBLE = (BARE_GROUND, ROAD, WATER)


class Tier(str, Enum):
    """Calibration ladder rung actually used for a run (see spec 2.1)."""

    A = "A"  # automatic DEM calibration
    B = "B"  # GCP-assisted
    C = "C"  # manual / physics-only

    @property
    def label(self) -> str:
        return {
            Tier.A: "Automatic (DEM)",
            Tier.B: "GCP-assisted",
            Tier.C: "Manual / physics-only",
        }[self]


@dataclass(frozen=True)
class SceneMeta:
    crs: Optional[str] = None            # "EPSG:32644", or None for a plain image
    transform: Optional[tuple] = None    # affine, 6 floats (a,b,c,d,e,f)
    gsd_m: float = 1.0                   # metres per pixel, always metres
    bounds_wgs: Optional[tuple] = None   # (w, s, e, n) in EPSG:4326
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    off_nadir_deg: float = 0.0
    acquired_utc: Optional[str] = None
    source: str = "unknown"              # "geotiff" | "image+exif" | "image"
    gsd_is_assumed: bool = False         # True when 1.0 m was assumed, not read

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None

    @property
    def has_sun(self) -> bool:
        return self.sun_azimuth_deg is not None and self.sun_elevation_deg is not None

    def describe(self) -> str:
        bits = []
        if self.georeferenced:
            bits.append(self.crs or "?")
        else:
            bits.append("not georeferenced")
        bits.append(f"GSD {self.gsd_m:.3g} m" + (" (assumed)" if self.gsd_is_assumed else ""))
        if self.has_sun:
            bits.append(
                f"sun {self.sun_azimuth_deg:.1f}° az / {self.sun_elevation_deg:.1f}° el"
            )
        else:
            bits.append("sun unknown")
        if self.off_nadir_deg:
            bits.append(f"off-nadir {self.off_nadir_deg:.1f}°")
        return "   ".join(bits)


@dataclass
class Scene:
    rgb: np.ndarray            # (H, W, 3) uint8, already stretched to display range
    meta: SceneMeta
    path: Optional[str] = None
    raw_dtype: str = "uint8"   # dtype of the source raster, before stretch

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[0], self.rgb.shape[1]


@dataclass
class DepthField:
    """Unitless relative surface. Higher value = taller (see traksha.depth.infer)."""

    relative: np.ndarray                  # (H, W) float32 in [0, 1]
    meta: SceneMeta
    terrain: Optional[np.ndarray] = None  # low-frequency branch, filled by dual head
    objects: Optional[np.ndarray] = None  # high-frequency branch
    backbone: str = "unknown"

    @property
    def has_branches(self) -> bool:
        return self.terrain is not None and self.objects is not None


@dataclass
class Anchor:
    """One metric constraint on the surface.

    Absolute: "pixel (row, col) is at value_m metres".
    Relative: with ref_row/ref_col set, "(row, col) stands value_m metres above
    (ref_row, ref_col)". Shadow measurements are relative and must stay that
    way; reinterpreting one as an elevation is how a good height anchor becomes
    a bad datum anchor.
    """

    row: int
    col: int
    value_m: float          # metres; meaning depends on `branch` and ref_*
    branch: Branch
    source: AnchorSource
    weight: float = 1.0     # 0..1 confidence
    ref_row: Optional[int] = None
    ref_col: Optional[int] = None

    @property
    def is_relative(self) -> bool:
        return self.ref_row is not None and self.ref_col is not None


@dataclass
class CalibrationField:
    a: np.ndarray                  # (H, W) scale field
    b: np.ndarray                  # (H, W) offset field
    residual_rmse: float = float("nan")
    n_anchors_used: int = 0
    n_anchors_rejected: int = 0
    tier: Tier = Tier.C
    # Which band `a` multiplies. A dual-branch solve fits the scale against the
    # high-frequency depth only, and carries that band with it: applying the
    # field to raw depth instead would reintroduce the low-frequency ramp the
    # split exists to discard, and the result would look plausible and be wrong.
    dual_branch: bool = False
    depth_high: Optional[np.ndarray] = None
    # Where the structural scale came from: solved from anchors, used as a
    # prior, or held at a value fitted offline against lidar (`traksha fit`).
    # Recorded because a number that was measured elsewhere must never be
    # mistaken for one this scene's anchors produced.
    scale_source: str = "anchors"


@dataclass
class ElevationSurface:
    dsm_m: np.ndarray
    ndsm_m: np.ndarray
    sigma_m: np.ndarray            # per-pixel 1 sigma, metres
    meta: SceneMeta
    tier: Tier = Tier.C


@dataclass
class Building:
    id: int
    centroid: tuple[float, float]
    height_m: float
    sigma_m: float
    area_m2: float


@dataclass
class Cliff:
    id: int
    drop_m: float
    length_m: float
    mean_slope_deg: float


@dataclass
class GCP:
    row: int
    col: int
    elev_m: float
    label: str = ""


@dataclass
class Config:
    """Run configuration. Hydra yaml maps straight onto this."""

    backbone: str = "dav2-vits"
    chip: int = 1024
    overlap: float = 0.25
    dem_source: Optional[str] = None      # "copernicus" | "srtm" | path to a GeoTIFF
    gcp_file: Optional[str] = None
    reference: Optional[str] = None       # ground-truth DSM for validation
    use_icesat: bool = False
    test_time_refinement: bool = False
    n_bootstrap: int = 24
    lattice_stride: int = 32
    lam_a: float = 1.0
    lam_b: float = 1.0
    huber_delta: float = 2.0
    irls_iters: int = 3
    extras: dict = field(default_factory=dict)


@dataclass
class StageEvent:
    """One line of the processing screen; also what the SSE endpoint emits."""

    stage: str
    status: Literal["pending", "running", "done", "failed", "skipped"]
    detail: str = ""
    pct: float = 0.0


__all__ = [n for n in dir() if not n.startswith("_")]
