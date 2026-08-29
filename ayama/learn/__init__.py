"""Learning the one thing the anchor graph cannot observe.

Everything else in ĀYĀMA solves a system per image. This package is the only
part that carries something *between* images: the metric scale of structure.

The reason it has to exist is measured, not assumed. On real imagery every
anchor the ladder finds is a ground anchor - shadows yield nothing when no
acquisition time is published - so the calibration learns terrain and flattens
buildings (README §3.4, §4.2). The depth field still contains the structure;
what is missing is the number of metres per unit of relative depth, and that
number is not observable from ground anchors at all.

So it is fitted once, offline, against lidar truth, and shipped. That is the
whole idea: not weights, a calibration constant - the empirical answer to "how
tall is a unit of relative depth" - supplied to the branch that has no anchors.

The fitter chooses its own model by leave-one-out (`fit`), and on the four
scenes available it chooses a constant, because a feature model fitted on three
points loses to one. That decision is data's to make and is recorded in the
model file, so adding scenes can change it without anyone editing code.
"""
from .scale import (ScaleModel, Sample, fit, load_bundled,  # noqa: F401
                    scene_features, scene_target)

__all__ = ["ScaleModel", "Sample", "fit", "load_bundled",
           "scene_features", "scene_target"]
