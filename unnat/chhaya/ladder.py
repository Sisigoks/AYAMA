"""The calibration ladder: three strategies, tried in order, each standalone.

    TIER A  automatic DEM      needs CRS + a public DEM
    TIER B  GCP-assisted       needs >= 3 points of known elevation
    TIER C  manual / physics    needs nothing but the image

The system degrades instead of failing, and the tier it actually used is
reported in the UI with the reason. That badge is a differentiator on its own,
because it tells a planner exactly how much to trust the output: Tier A is good
terrain with weaker absolute building heights, Tier B pins the datum, Tier C
gives trustworthy relative structure on an arbitrary datum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.types import Anchor, Config, DepthField, GCP, Scene, Tier
from .anchors import (assume_ground_plane, harvest_dem, harvest_gcp,
                      harvest_shadow, harvest_water)


@dataclass
class TierDecision:
    tier: Tier
    reason: str

    def __str__(self) -> str:
        return f"Tier {self.tier.value} ({self.tier.label}) - {self.reason}"


def select_tier(scene: Scene, cfg: Config, gcps: Optional[Sequence[GCP]] = None,
                dem_available: bool = False) -> TierDecision:
    if gcps and len(gcps) >= 3:
        return TierDecision(Tier.B, f"{len(gcps)} ground control points supplied")
    if scene.meta.georeferenced and (dem_available or cfg.dem_source):
        return TierDecision(Tier.A, f"georeferenced ({scene.meta.crs}) with a public DEM")
    if not scene.meta.georeferenced:
        return TierDecision(Tier.C, "no CRS: a public DEM cannot be located")
    return TierDecision(Tier.C, "no DEM source configured")


def build_anchors(
    scene: Scene,
    depth: DepthField,
    sem: np.ndarray,
    shadow: np.ndarray,
    tier: Tier,
    dem_m: Optional[np.ndarray] = None,
    gcps: Optional[Sequence[GCP]] = None,
    cfg: Optional[Config] = None,
    slope_mask: Optional[np.ndarray] = None,
) -> tuple[list[Anchor], dict]:
    """Harvest every anchor the current tier allows. Returns (anchors, counts)."""
    cfg = cfg or Config()
    anchors: list[Anchor] = []
    counts: dict = {}

    if tier in (Tier.A, Tier.B) and dem_m is not None:
        dem = harvest_dem(dem_m, sem, source=str(cfg.dem_source or "unknown"),
                          weight=0.6, slope_mask=slope_mask)
        water = harvest_water(sem, dem_m=dem_m, weight=0.9)
        anchors += dem + water
        counts["dem"] = len(dem)
        counts["water"] = len(water)
    else:
        water = harvest_water(sem, dem_m=None, weight=0.9)
        anchors += water
        counts["water"] = len(water)

    if tier is Tier.B and gcps:
        g = harvest_gcp(gcps, weight=1.0)
        anchors += g
        counts["gcp"] = len(g)

    # Shadow physics runs on every tier: it needs nothing but the image and the
    # sun angles, and it is the only absolute-scale cue in Tier C.
    sh = harvest_shadow(scene, sem, shadow, weight_scale=1.0)
    anchors += sh
    counts["shadow"] = len(sh)

    if not anchors:
        gp = assume_ground_plane(depth.relative, sem)
        anchors += gp
        counts["ground_plane"] = len(gp)

    counts["total"] = len(anchors)
    return anchors, counts
