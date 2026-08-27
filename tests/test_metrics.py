"""Metric behaviour, checked against surfaces whose error we constructed."""
from __future__ import annotations

import numpy as np
import pytest

from ayama.eval.metrics import (evaluate, evaluate_by_class,
                                expected_calibration_error, format_table)


@pytest.fixture
def ref():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    surface = 400.0 + 0.05 * yy + 8.0 * np.sin(xx / 20.0)
    surface[40:60, 40:60] += 25.0        # a building
    return surface + rng.normal(0, 0.05, surface.shape).astype(np.float32)


def test_perfect_prediction_scores_zero_error(ref):
    m = evaluate(ref, ref, gsd=0.5)
    assert m["mae_m"] == pytest.approx(0.0)
    assert m["rmse_m"] == pytest.approx(0.0)
    assert m["bias_m"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)


def test_bias_separates_a_wrong_datum_from_a_wrong_model(ref):
    shifted = evaluate(ref + 3.0, ref, gsd=0.5)
    assert shifted["bias_m"] == pytest.approx(3.0, abs=1e-6)
    assert shifted["mae_m"] == pytest.approx(3.0, abs=1e-6)
    assert shifted["std_m"] == pytest.approx(0.0, abs=1e-6)

    rng = np.random.default_rng(1)
    noisy = evaluate(ref + rng.normal(0, 3.0, ref.shape), ref, gsd=0.5)
    assert abs(noisy["bias_m"]) < 0.2          # no systematic offset
    assert noisy["std_m"] == pytest.approx(3.0, rel=0.15)


def test_sigma_coverage_hits_the_gaussian_expectation(ref):
    rng = np.random.default_rng(2)
    sigma = np.full(ref.shape, 2.0, np.float32)
    pred = ref + rng.normal(0, 2.0, ref.shape)
    m = evaluate(pred, ref, sigma=sigma, gsd=0.5)
    assert m["coverage_1s"] == pytest.approx(0.68, abs=0.03)
    assert m["coverage_2s"] == pytest.approx(0.95, abs=0.02)
    assert m["ece_m"] < 0.15                    # sigma predicts the error


def test_ece_catches_an_overconfident_sigma():
    rng = np.random.default_rng(3)
    err = rng.normal(0, 5.0, 20000)
    honest = expected_calibration_error(err, np.full(err.shape, 5.0))
    overconfident = expected_calibration_error(err, np.full(err.shape, 1.0))
    assert honest < 0.3
    assert overconfident > 3.0


def test_delta1_needs_heights_not_absolute_elevation(ref):
    # On absolute elevation the datum swamps the ratio, so delta1 is only
    # computed when heights above ground are supplied.
    m = evaluate(ref + 2.0, ref, gsd=0.5)
    assert np.isnan(m["delta1"])

    h_ref = np.full(ref.shape, 20.0)
    m2 = evaluate(ref + 2.0, ref, gsd=0.5, height_pred=h_ref + 2.0, height_ref=h_ref)
    assert m2["delta1"] == pytest.approx(1.0)
    m3 = evaluate(ref, ref, gsd=0.5, height_pred=h_ref * 1.5, height_ref=h_ref)
    assert m3["delta1"] == pytest.approx(0.0)


def test_edge_f1_drops_when_structures_move(ref):
    aligned = evaluate(ref, ref, gsd=0.5)["edge_f1"]
    shifted_pred = np.roll(ref, 12, axis=1)
    moved = evaluate(shifted_pred, ref, gsd=0.5)["edge_f1"]
    assert aligned == pytest.approx(1.0, abs=1e-6)
    assert moved < aligned


def test_per_class_breakdown_isolates_the_hard_class():
    from ayama.core.types import BARE_GROUND, BUILDING

    ref = np.full((64, 64), 400.0)
    sem = np.full((64, 64), BARE_GROUND, np.uint8)
    sem[16:48, 16:48] = BUILDING
    ref[16:48, 16:48] = 425.0

    pred = ref.copy()
    pred[sem == BUILDING] += 6.0               # buildings wrong, ground right

    by_class = evaluate_by_class(pred, ref, sem, gsd=0.5)
    assert by_class["bare ground"]["mae_m"] == pytest.approx(0.0)
    assert by_class["building"]["mae_m"] == pytest.approx(6.0)


def test_format_table_renders_only_present_metrics(ref):
    txt = format_table(evaluate(ref + 1.0, ref, gsd=0.5), title="Validation")
    assert "MAE" in txt and "Bias" in txt
    assert "1-sigma coverage" not in txt        # no sigma was supplied
