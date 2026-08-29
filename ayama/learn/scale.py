"""Fit, store and apply the structural scale.

The quantity being learned is a single number per scene:

    a* = argmin_a || a * max(D_hi, 0) - nDSM_true ||^2

where `D_hi` is the high-frequency band of the relative depth field. It is the
metres of height above ground carried by one unit of high-band depth. Solving
that least-squares problem in closed form is one line; the work here is in
deciding what to do with the answers across scenes, and in refusing to claim
more than the data supports.

Two model families are considered:

  constant  a is the mean of the fitted scales. One parameter.
  linear    a = coef / feature + intercept, on one inference-time feature.
            Two parameters, and it needs enough scenes to earn them.

`fit` picks between them by leave-one-out and records the comparison in the
model file, so the choice is auditable and changes on its own as scenes are
added. On the four scenes in this repository the constant wins, which is what
gets shipped.

Nothing here is a neural network and nothing needs a GPU. The fit is a handful
of dot products over scene-level summaries.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

BUNDLED = os.path.join(os.path.dirname(__file__), "calibration.json")

# The candidate feature. Relative depth is normalised per scene, so a scene
# whose high band already spans a lot of range needs fewer metres per unit -
# hence the reciprocal. It is a candidate, not a finding: with four scenes it
# loses to a constant, and `fit` says so.
FEATURE = "hi_p99"


@dataclass
class Sample:
    """One scene's fitted scale and the features that might explain it."""

    name: str
    target: float                 # a*, metres per unit of high-band depth
    features: dict = field(default_factory=dict)
    ndsm_mae_at_target: float = float("nan")
    floor_mae: float = float("nan")


# ---------------------------------------------------------------------------
# features and target
# ---------------------------------------------------------------------------
def high_band(relative: np.ndarray, gsd_m: float, radius_m: float = 60.0) -> np.ndarray:
    """The positive high-frequency band: what carries structure.

    Clipped at zero because a negative excursion below local terrain is not a
    building, and letting it through would let the scale be fitted against
    something the model is not being asked to predict.
    """
    from ..chhaya.agmc import decompose_depth

    _lo, hi = decompose_depth(np.asarray(relative, np.float32), gsd_m, radius_m)
    return np.maximum(hi, 0.0).astype(np.float32)


def scene_features(relative: np.ndarray, dem_m: Optional[np.ndarray],
                   gsd_m: float, radius_m: float = 60.0) -> dict:
    """Scene-level summaries computable at inference, with no ground truth.

    That constraint is the whole point: a feature that needs the answer cannot
    be used to predict the answer.
    """
    rel = np.asarray(relative, np.float64)
    pos = high_band(rel, gsd_m, radius_m).astype(np.float64)
    feats = {
        "hi_p99": float(np.percentile(pos, 99)),
        "hi_std": float(pos.std()),
        "rd_range": float(np.percentile(rel, 99) - np.percentile(rel, 1)),
        "gsd_m": float(gsd_m),
    }
    if dem_m is not None:
        d = np.asarray(dem_m, np.float64)
        finite = d[np.isfinite(d)]
        if finite.size:
            feats["dem_relief_m"] = float(np.percentile(finite, 99)
                                          - np.percentile(finite, 1))
    return feats


def scene_target(pos_high: np.ndarray, ndsm_true: np.ndarray) -> float:
    """The least-squares scale for one scene. Closed form, no iteration."""
    p = np.asarray(pos_high, np.float64)
    t = np.asarray(ndsm_true, np.float64)
    ok = np.isfinite(p) & np.isfinite(t)
    denom = float((p[ok] * p[ok]).sum())
    if denom <= 0:
        return float("nan")
    return float((p[ok] * t[ok]).sum() / denom)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass
class ScaleModel:
    """What gets shipped: a rule for turning high-band depth into metres."""

    # Which depth model produced the fields this was fitted against. Relative
    # depth is normalised per scene, but two backbones do not distribute it the
    # same way, so a constant in metres per unit of high-band depth is only
    # meaningful for the backbone that produced that band. Recorded, checked,
    # and warned about rather than silently applied.
    backbone: str = ""
    kind: str = "constant"            # "constant" | "linear"
    value: float = 0.0                # constant: the scale itself
    feature: Optional[str] = None     # linear: which feature
    coef: float = 0.0                 # linear: a = coef / feature + intercept
    intercept: float = 0.0
    radius_m: float = 60.0            # the frequency split it was fitted under

    n_scenes: int = 0
    loo_mae_m: float = float("nan")   # held-out nDSM MAE of the chosen model
    loo_mae_alt_m: float = float("nan")   # ... of the model not chosen
    floor_mae_m: float = float("nan")     # predict-zero-everywhere, same scenes
    scenes: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    fitted_utc: str = ""
    notes: str = ""

    def predict(self, features: Optional[dict] = None) -> float:
        if self.kind == "linear" and self.feature:
            x = float((features or {}).get(self.feature, 0.0))
            if x > 0:
                return self.coef / x + self.intercept
        return float(self.value)

    @property
    def improvement_over_floor(self) -> float:
        """Fraction by which held-out nDSM MAE beats predicting zero."""
        if not np.isfinite(self.loo_mae_m) or not self.floor_mae_m:
            return float("nan")
        return 1.0 - self.loo_mae_m / self.floor_mae_m

    def describe(self) -> str:
        head_bb = f" [{self.backbone}]" if self.backbone else ""
        if self.kind == "linear":
            head = f"linear on 1/{self.feature}: a = {self.coef:.3f}/x + {self.intercept:.1f}"
        else:
            head = f"constant a = {self.value:.1f} m per unit of high-band depth"
        return (f"{head}{head_bb}\n"
                f"  fitted on {self.n_scenes} scene(s), held-out nDSM MAE "
                f"{self.loo_mae_m:.2f} m vs a flat-ground floor of "
                f"{self.floor_mae_m:.2f} m "
                f"({100 * self.improvement_over_floor:.0f}% better)")

    # -- io ----------------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"ayama_scale_model_version": 1, **asdict(self)}, fh, indent=2)
            fh.write("\n")
        return path

    @classmethod
    def load(cls, path: str) -> "ScaleModel":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d.pop("ayama_scale_model_version", None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_bundled(backbone: str = "") -> Optional[ScaleModel]:
    """The calibration shipped for `backbone`, or None if none applies.

    One file per backbone, because the constant is only meaningful for the high
    band the backbone that produced it emits. `calibration.json` is the primary
    (dav2-vitl); others sit beside it as `calibration_<backbone>.json`. Asking
    for a backbone with no fitted scale returns None rather than the wrong one.
    """
    candidates = []
    if backbone:
        candidates.append(os.path.join(os.path.dirname(__file__),
                                       f"calibration_{backbone}.json"))
    candidates.append(BUNDLED)
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            model = ScaleModel.load(path)
        except Exception:
            continue
        if backbone and model.backbone and model.backbone != backbone:
            continue
        return model
    return None


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def _fit_constant(samples: Sequence[Sample]) -> float:
    return float(np.mean([s.target for s in samples]))


def _fit_linear(samples: Sequence[Sample], feature: str):
    """a = coef / feature + intercept. Returns None if it cannot be fitted."""
    xs, ys = [], []
    for s in samples:
        v = s.features.get(feature)
        if v and np.isfinite(v) and v > 0:
            xs.append(1.0 / float(v))
            ys.append(s.target)
    if len(xs) < 3 or float(np.std(xs)) <= 0:
        return None
    coef, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(coef), float(intercept)


def _held_out_mae(a_hat: float, pos: np.ndarray, ndsm_true: np.ndarray) -> float:
    return float(np.abs(np.maximum(a_hat * pos, 0.0) - ndsm_true).mean())


def fit(samples: Sequence[Sample], rasters: Optional[dict] = None,
        radius_m: float = 60.0, feature: str = FEATURE,
        backbone: str = "") -> ScaleModel:
    """Fit the scale, and let leave-one-out choose the model family.

    `rasters` maps scene name -> (positive high band, true nDSM). When it is
    supplied the leave-one-out comparison is made in the units that matter -
    nDSM MAE in metres - rather than in error on the fitted scale, which is not
    the same ranking. Without it the comparison falls back to scale error.

    A model is never selected because it is more sophisticated. With three
    training scenes a two-parameter fit loses, and this says so.
    """
    samples = [s for s in samples if np.isfinite(s.target)]
    if not samples:
        raise ValueError("no usable scenes: every fitted scale was non-finite")

    const_all = _fit_constant(samples)
    lin_all = _fit_linear(samples, feature)

    def loo(kind: str) -> float:
        errs = []
        for i, held in enumerate(samples):
            train = samples[:i] + samples[i + 1:]
            if not train:
                continue
            if kind == "constant":
                a_hat = _fit_constant(train)
            else:
                got = _fit_linear(train, feature)
                x = held.features.get(feature)
                if not got or not x or x <= 0:
                    return float("inf")
                a_hat = got[0] / float(x) + got[1]
            if rasters and held.name in rasters:
                pos, truth = rasters[held.name]
                errs.append(_held_out_mae(a_hat, pos, truth))
            else:
                errs.append(abs(a_hat - held.target))
        return float(np.mean(errs)) if errs else float("inf")

    loo_const = loo("constant") if len(samples) > 1 else float("nan")
    loo_lin = loo("linear") if (lin_all and len(samples) > 3) else float("inf")

    use_linear = bool(lin_all) and np.isfinite(loo_lin) and loo_lin < loo_const
    floor = float(np.mean([s.floor_mae for s in samples
                           if np.isfinite(s.floor_mae)] or [float("nan")]))

    model = ScaleModel(
        backbone=backbone,
        kind="linear" if use_linear else "constant",
        value=const_all,
        feature=feature if use_linear else None,
        coef=lin_all[0] if use_linear else 0.0,
        intercept=lin_all[1] if use_linear else 0.0,
        radius_m=float(radius_m),
        n_scenes=len(samples),
        loo_mae_m=loo_lin if use_linear else loo_const,
        loo_mae_alt_m=loo_const if use_linear else loo_lin,
        floor_mae_m=floor,
        scenes=[s.name for s in samples],
        targets=[round(s.target, 3) for s in samples],
        fitted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=("Selected by leave-one-out. "
               + ("A one-feature model won." if use_linear else
                  "A constant won; a one-feature model fitted on the remaining "
                  "scenes did not beat it. Add scenes and refit - this choice is "
                  "made by the data, not in code.")),
    )
    return model
