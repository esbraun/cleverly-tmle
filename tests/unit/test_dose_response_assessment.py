"""The assessment surface of a dose-response fit (X104): support, scores, the outcome model.

Every reading restates a number the fit stored. The witnesses check that each reads the right
number: the support rows against the stored pair records, the score threshold against the
scaled standard errors, and the nuisance record against an independent out-of-fold fit.
"""

from __future__ import annotations

import warnings
from dataclasses import astuple, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.assessment import AssessmentStatus, assessment_capabilities
from cleverly.dose_response import DoseResponse
from cleverly.estimators._nuisance import fit_nuisances
from cleverly.estimators.dose_response import DoseResponseTMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.inference.sparse import influence_report
from cleverly.learners.crossfit import Folds
from cleverly.learners.density import default_density_bins
from cleverly.validation.score import score_threshold

#: The assumed range of the Gaussian outcome: 8 units wide, so a score read against the
#: unscaled standard errors would sit eight times too loose.
Q_BOUNDS = (-3.0, 5.0)


def _binary(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.normal(2.5 + 0.6 * w, 1.2)
    y = rng.binomial(1, expit(-1.0 + 0.5 * a - 0.08 * a * a + 0.4 * w))
    return pd.DataFrame({"Y": y, "A": a, "W": w})


def _gaussian(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.normal(2.5 + 0.6 * w, 1.2)
    y = np.clip(0.2 + 0.4 * a - 0.05 * a * a + 0.6 * w + rng.normal(0.0, 0.8, n), -2.9, 4.9)
    return pd.DataFrame({"Y": y, "A": a, "W": w})


def _curve(floor: float) -> DoseResponse:
    return DoseResponse(
        doses=(1.0, 2.5, 4.0), bandwidths=(0.75,), support=(0.5, 4.5), support_floor=floor
    )


def _estimator(curve: DoseResponse, frame: pd.DataFrame, **settings: Any) -> DoseResponseTMLE:
    gaussian = frame["Y"].nunique() > 2
    options: dict[str, Any] = {
        "outcome_learner": LinearRegression() if gaussian else LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "random_state": 0,
        "n_folds": 2,
    }
    if gaussian:
        options.update(q_bounds=Q_BOUNDS, fluctuation="linear")
    options.update(settings)
    return DoseResponseTMLE(curve, **options)


def _fit(curve: DoseResponse, frame: pd.DataFrame, **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(curve, frame, **settings).fit(
            frame, outcome="Y", treatment="A", covariates=["W"]
        )


@pytest.fixture(scope="module")
def binary() -> Any:
    return _fit(_curve(1.0), _binary())


@pytest.fixture(scope="module")
def gaussian() -> Any:
    return _fit(_curve(1.0), _gaussian())


def test_the_rows_read_the_fit_and_refuse_what_it_lacks(binary: Any) -> None:
    rows = {row.operation: row for row in binary.diagnostics.capabilities}
    assert rows.keys() == {row.operation for row in assessment_capabilities(binary)}
    assert all(rows[name].available for name in ("support", "score_equations", "nuisance_models"))
    for name in ("corrections", "truncation_curve"):
        assert rows[name].status is AssessmentStatus.NOT_APPLICABLE
    assert rows["refute"].refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert rows["refute"].refusal.roadmap == "X82"
    for name in ("corrections", "truncation_curve", "refute"):
        with pytest.raises(CapabilityError) as caught:
            getattr(binary.diagnostics, name)()
        assert caught.value.refusal.reason is rows[name].refusal.reason
    assert binary.replayability.unreconstructible == ("estimator configuration",)


def test_every_sensitivity_row_refuses_with_its_record(binary: Any) -> None:
    expected = {
        "omitted_confounding": ("X82", AssessmentStatus.UNAVAILABLE),
        "robustness_value": ("X82", AssessmentStatus.UNAVAILABLE),
        "elements": ("X82", AssessmentStatus.UNAVAILABLE),
        "contour": ("X82", AssessmentStatus.UNAVAILABLE),
        "benchmark": ("X82", AssessmentStatus.UNAVAILABLE),
        "marginal_bounds": ("X82", AssessmentStatus.UNAVAILABLE),
        "simulated_confounding": ("X66", AssessmentStatus.UNAVAILABLE),
        "evalue": (None, AssessmentStatus.NOT_APPLICABLE),
        "missingness": (None, AssessmentStatus.NOT_APPLICABLE),
        "tipping_gamma": (None, AssessmentStatus.NOT_APPLICABLE),
    }
    sensitivity = binary.sensitivity
    assert {row.operation for row in sensitivity.capabilities} == set(expected)
    for operation, (roadmap, status) in expected.items():
        row = sensitivity.capability(operation)
        assert row.status is status and row.refusal.roadmap == roadmap, operation
        with pytest.raises(CapabilityError) as caught:
            getattr(sensitivity, operation)()
        assert row.refusal.condition in str(caught.value)


def test_the_support_report_restates_the_pair_records(binary: Any) -> None:
    report = binary.diagnostics.support()
    assert report.pairs == binary.pair_support
    frame = report.to_frame()
    assert list(frame["name"]) == [pair.name for pair in binary.pair_support]
    np.testing.assert_array_equal(frame["kish"], [pair.kish for pair in binary.pair_support])
    assert report.quadrature_shift == binary.quadrature_shift


def test_the_support_row_warns_below_the_floor_and_never_passes(binary: Any) -> None:
    """Completed when no pair is below the floor, a warning naming the count when some are.

    The floor sits between the pairs' Kish sizes, which a fit with any floor reproduces, so
    the same pairs are read both ways.
    """
    clear = binary.validate()["support"]
    assert clear.status is AssessmentStatus.COMPLETED
    assert "0 of 3 pair(s) below the floor" in clear.detail
    kish = sorted(pair.kish for pair in binary.pair_support)
    floor = 0.5 * (kish[0] + kish[-1])
    split = _fit(_curve(floor), _binary())
    below = sum(pair.below_floor for pair in split.pair_support)
    assert 0 < below < 3
    item = split.validate()["support"]
    assert item.status is AssessmentStatus.WARNING
    assert f"{below} pair(s) carry the experimental low_dose_support_plugin" in item.detail
    assert "provisional (X95)" in item.detail
    assert AssessmentStatus.PASSED not in {clear.status, item.status}


def test_the_score_threshold_reads_the_scaled_standard_errors(gaussian: Any) -> None:
    """A stored score of half the threshold passes and of twice the threshold fails.

    The threshold uses each pair's plug-in SE divided by the scaler's range of 8. A report
    that read the unscaled SE would pass twice the threshold, and one that multiplied would
    fail half of it.
    """
    record = gaussian.nuisance_record
    assert record.scaler.range == pytest.approx(8.0)
    scaled = [gaussian.estimates[name].plugin_std_error / 8.0 for name in gaussian.scores]
    threshold = score_threshold(scaled, record.predictions.size)
    report = gaussian.diagnostics.score_equations()
    assert report.threshold == pytest.approx(threshold, rel=1e-12)
    assert report.passed
    for factor, passed in ((0.5, True), (2.0, False)):
        scores = dict.fromkeys(gaussian.scores, factor * threshold)
        moved = replace(gaussian, scores=scores)
        assert moved.diagnostics.score_equations().passed is passed
        expected = AssessmentStatus.PASSED if passed else AssessmentStatus.FAILED
        assert moved.validate()["score_equations"].status is expected


def test_the_nuisance_record_is_the_independent_out_of_fold_fit(gaussian: Any) -> None:
    """The record equals ``nuisance.outcome.observed`` of a separate fit on the same folds."""
    record = gaussian.nuisance_record
    frame = _gaussian()
    estimator = _estimator(_curve(1.0), frame)
    engine = estimator._engine()
    data = engine._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta=None,
        weights=None,
        id=None,
        intermediate=None,
        strata=None,
        treatment_kind="continuous",
    )
    nuisance = fit_nuisances(
        data,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=None,
        intermediate_learner=None,
        folds=Folds(record.folds, int(record.folds.max()) + 1),
        scaler=record.scaler,
        density_bins=default_density_bins(data.n),
        dose_nodes=(2.5,),
    )
    np.testing.assert_allclose(record.predictions, nuisance.outcome.observed, atol=1e-12)
    assert record.cross_fitted and len(np.unique(record.folds)) == 2
    for name in ("predictions", "outcome", "weights", "folds"):
        assert np.asarray(getattr(record, name)).shape == (data.n,), name


def test_a_held_out_fold_never_sees_its_own_outcomes(gaussian: Any) -> None:
    """Reflect the outcomes of fold 0 and refit: its predictions do not move, the others do.

    The outcome range is fixed by ``q_bounds``, so the scaling cannot move with the outcomes.
    """
    record = gaussian.nuisance_record
    frame = _gaussian()
    held = record.folds == 0
    frame.loc[held, "Y"] = sum(Q_BOUNDS) - frame.loc[held, "Y"]
    flipped = _fit(_curve(1.0), frame).nuisance_record
    np.testing.assert_array_equal(flipped.folds, record.folds)
    np.testing.assert_allclose(flipped.predictions[held], record.predictions[held], atol=1e-12)
    assert np.max(np.abs(flipped.predictions[~held] - record.predictions[~held])) > 1e-3


def test_the_nuisance_report_names_its_evaluation(gaussian: Any) -> None:
    """Out of fold on a cross-fitted fit, in sample on ``cross_fit=False``."""
    report = gaussian.diagnostics.nuisance_models()
    assert [model.name for model in report.models] == ["outcome"]
    assert report.evaluation == "out_of_fold" == report.models[0].evaluation
    assert report.models[0].metrics["mse"] == pytest.approx(
        float(
            np.mean(
                (
                    gaussian.nuisance_record.scaler.scale(_gaussian()["Y"])
                    - (gaussian.nuisance_record.predictions)
                )
                ** 2
            )
        )
    )
    in_sample = _fit(_curve(1.0), _binary(), cross_fit=False)
    report = in_sample.diagnostics.nuisance_models()
    assert not in_sample.nuisance_record.cross_fitted
    assert report.evaluation == "in_sample" == report.models[0].evaluation
    detail = in_sample.validate()["nuisance_models"].detail
    assert "held-out" not in detail and "out-of-fold" not in detail


def test_influence_reads_the_stored_curves_at_the_fit_alpha() -> None:
    """The report is the estimates' own, at the fit's alpha (0.1 here, not the 0.05 default)."""
    result = _fit(_curve(1.0), _binary(), alpha=0.1)
    report = result.diagnostics.influence()
    assert report.alpha == 0.1
    expected = influence_report(result.estimates, alpha=0.1, backend=result.backend)
    assert [row.name for row in report.rows] == [row.name for row in expected.rows]
    for got, want in zip(report.rows, expected.rows, strict=True):
        np.testing.assert_equal(astuple(got), astuple(want))
