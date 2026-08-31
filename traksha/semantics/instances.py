"""The structural segmentation artifact: instances, boundaries and confidence.

A list of masks is not a contract. This turns one into the rasters the rest of
the pipeline can index - an instance id per pixel, a boundary map, a confidence
map - plus a record per instance carrying the evidence that produced it. Every
downstream stage reads this, so it is versioned and it is written to disk in one
directory that can be inspected on its own.

Two decisions worth stating.

**Smaller masks are painted last.** SAM proposes nested masks: a roof section
inside a roof inside a block. Painting largest-first means the finer structure
survives, which is the structure a building-scale reconstruction needs. Painting
the other way round would silently erase every small instance behind the first
large one that contained it.

**Nothing here decides what a building is.** SAM 2 is class-agnostic: these
instances are shadows, courtyards, roads and trees as much as roofs. Deciding
which are buildings needs height, and height does not exist until Chhaya has
run. That decision therefore lives downstream, and this module carries no
`building` field for anyone to mistake for one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1
UNASSIGNED = 0          # instance id 0 means "no instance covers this pixel"


@dataclass
class InstanceField:
    """Per-pixel instance ids plus the per-instance record behind them."""

    instance_map: np.ndarray                  # (H, W) int32, 0 = unassigned
    boundary: np.ndarray                      # (H, W) bool
    confidence: np.ndarray                    # (H, W) float32 in 0..1
    records: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.instance_map.shape

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def coverage(self) -> float:
        """Fraction of the image any instance covers."""
        return float((self.instance_map != UNASSIGNED).mean())

    def mask(self, instance_id: int) -> np.ndarray:
        return self.instance_map == int(instance_id)

    def summary(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "instances": self.count,
            "coverage": round(self.coverage, 4),
            "mean_confidence": round(float(self.confidence[
                self.instance_map != UNASSIGNED].mean()) if self.count else 0.0, 4),
            "provenance": self.provenance,
        }

    # ------------------------------------------------------------- on disk
    def save(self, out_dir: str, meta=None) -> dict:
        """Write the artifact. Rasters as GeoTIFF when there is georeferencing."""
        os.makedirs(out_dir, exist_ok=True)
        art = {}
        if meta is not None:
            from ..dsm.cog import write_cog

            art["instance_map"] = write_cog(
                os.path.join(out_dir, "instances.tif"),
                self.instance_map.astype(np.float32), meta, dtype="int32",
                description="SAM 2 instance ids (0 = unassigned)")
            art["boundary"] = write_cog(
                os.path.join(out_dir, "boundary.tif"),
                self.boundary.astype(np.float32), meta, dtype="uint8", nodata=255,
                description="instance boundaries")
            art["confidence"] = write_cog(
                os.path.join(out_dir, "confidence.tif"), self.confidence, meta,
                description="per-pixel instance confidence")
        else:
            np.save(os.path.join(out_dir, "instances.npy"), self.instance_map)
            art["instance_map"] = os.path.join(out_dir, "instances.npy")

        path = os.path.join(out_dir, "metadata.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({**self.summary(), "records": self.records}, fh, indent=1)
        art["metadata"] = path
        return art


def _boundaries(instance_map: np.ndarray) -> np.ndarray:
    """Pixels where the instance id changes. The structural edges, exactly."""
    b = np.zeros(instance_map.shape, bool)
    b[:-1, :] |= instance_map[:-1, :] != instance_map[1:, :]
    b[1:, :] |= instance_map[:-1, :] != instance_map[1:, :]
    b[:, :-1] |= instance_map[:, :-1] != instance_map[:, 1:]
    b[:, 1:] |= instance_map[:, :-1] != instance_map[:, 1:]
    return b


def from_masks(masks, shape, provenance: Optional[dict] = None,
               min_area_px: int = 0) -> InstanceField:
    """Build the artifact from accepted masks, largest painted first.

    Ids are assigned in paint order starting at 1, so id order is area order and
    a reader can rely on that without consulting the records.
    """
    h, w = shape
    inst = np.zeros((h, w), np.int32)
    conf = np.zeros((h, w), np.float32)
    records = []

    ordered = sorted(masks, key=lambda m: -int(m.area_px))
    for m in ordered:
        seg = np.asarray(m.segmentation, bool)
        if seg.shape != (h, w) or int(seg.sum()) < min_area_px:
            continue
        instance_id = len(records) + 1
        inst[seg] = instance_id
        # The confidence of whichever instance ended up owning the pixel.
        conf[seg] = float(m.score)
        records.append({
            "id": instance_id,
            "area_px": int(m.area_px),
            "bbox": [int(v) for v in m.bbox],
            "predicted_iou": round(float(m.predicted_iou), 4),
            "stability_score": round(float(m.stability_score), 4),
            "score": round(float(m.score), 4),
            "prompt_point": [round(float(v), 1) for v in m.point],
        })

    # Painting overwrites, so an instance fully covered by later, smaller ones
    # can end up with no pixels. Recording it would promise a mask that cannot
    # be indexed, so its visible area is recorded and the caller can see it.
    present, counts = np.unique(inst, return_counts=True)
    visible = dict(zip(present.tolist(), counts.tolist()))
    for r in records:
        r["visible_px"] = int(visible.get(r["id"], 0))

    return InstanceField(
        instance_map=inst, boundary=_boundaries(inst), confidence=conf,
        records=records,
        provenance={**(provenance or {}), "written_utc":
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def empty(shape, reason: str) -> InstanceField:
    """The artifact a run produces when segmentation did not happen.

    A stage that is skipped must still produce something downstream can read,
    or every consumer grows a `None` branch and the reason it was skipped is
    lost by the second one.
    """
    h, w = shape
    return InstanceField(
        instance_map=np.zeros((h, w), np.int32),
        boundary=np.zeros((h, w), bool),
        confidence=np.zeros((h, w), np.float32),
        records=[], provenance={"skipped": reason})
