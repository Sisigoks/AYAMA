"""Backbone adapter interface.

One adapter per model, all behind the same two-method surface, so swapping
Depth Anything V2 for DPT is a config flag and not a code change. That is what
makes the ablation table cheap to produce.

Convention, stated once and never flipped again:

    infer() returns HIGHER values for surfaces CLOSER to the sensor.

For a nadir sensor, closer means higher elevation, so the returned field maps
monotonically to height and no sign flip is needed anywhere in the pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class DepthBackbone:
    name: str = "base"
    #: Native input side length the model prefers; chips are resized to it.
    native: Optional[int] = None

    def load(self) -> "DepthBackbone":
        """Materialise weights. Called once, lazily, before the first infer()."""
        return self

    def infer(self, patch: np.ndarray) -> np.ndarray:
        """(h, w, 3) uint8 -> (h, w) float32, higher = closer to the sensor."""
        raise NotImplementedError

    def infer_batch(self, patches: list) -> list:
        """Several chips at once. Overridden by adapters that can use a GPU.

        The default is a loop, so a backbone that cannot batch still works and
        the caller never has to care which kind it has.
        """
        return [self.infer(p) for p in patches]

    def describe(self) -> str:
        return self.name

    def stats(self) -> dict:
        """Whatever the adapter knows about itself: device, dtype, parameters."""
        return {"name": self.name, "device": "cpu"}
