"""Retained evidence for each longitudinal nuisance role.

Seed 41 is fixed here because the checks compare stored arrays and exact weighted
arithmetic. They do not make a repeated-sampling claim.
"""

from __future__ import annotations

import pickle
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.sequential import Mechanism
from cleverly.msm import MSM
from cleverly.validation.longitudinal import (
    LONGITUDINAL_CENSORING_NOT_FITTED,
    LONGITUDINAL_MECHANISM_PREDICTIONS_MISSING,
    LongitudinalNuisanceDiagnostics,
    LongitudinalNuisanceRow,
    _longitudinal_nuisances,
)
from cleverly.validation.nuisance import NuisanceModelReport
from tests.conftest import FAST_KWARGS
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

PARAMETRIC = {
    "outcome_learner": sklearn.linear_model.LinearRegression(),
    "pseudo_learner": sklearn.linear_model.LinearRegression(),
    "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
    "censoring_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
    "n_folds": 2,
    "learner_folds": 2,
    "random_state": 41,
    "simultaneous": False,
}


@pytest.fixture(scope="module")
def weighted_result():  # type: ignore[no-untyped-def]
    frame, _ = make_longitudinal(n=220, seed=41)
    frame["analysis_weight"] = np.linspace(0.5, 2.0, len(frame))
    return LTMLE({"always": 1, "never": 0}, **PARAMETRIC).fit(
        frame,
        **COLUMNS,
        weights="analysis_weight",
    )


def _row(report, role: str, *, time: int | None = None):  # type: ignore[no-untyped-def]
    return next(
        row for row in report.rows if row.role == role and (time is None or row.time == time)
    )


def test_longitudinal_rows_canonicalise_separately_constructed_nan_sentinels(tmp_path) -> None:
    left = LongitudinalNuisanceRow("always", None, None, 1, 10, float("nan"))
    right = LongitudinalNuisanceRow("always", None, None, 1, 10, float("nan"))
    assert left.mse is not right.mse
    assert left == right
    assert hash(left) == hash(right)

    report = LongitudinalNuisanceDiagnostics((left,), backend="pandas")
    path = tmp_path / "sentinel-report.joblib"
    joblib.dump(report, path)
    assert joblib.load(path) == report


def test_longitudinal_row_equality_keeps_nested_model_details() -> None:
    left_model = NuisanceModelReport(
        name="treatment",
        kind="classification",
        metrics={"log_loss": 0.2, "auc": float("nan")},
        calibration={
            "mean_prediction": [0.1, float("nan")],
            "observed_rate": [0.2, float("nan")],
        },
        learner_weights={"logistic": 1.0},
        learner_risks={"logistic": float("nan")},
    )
    right_model = NuisanceModelReport(
        name="treatment",
        kind="classification",
        metrics={"auc": float("nan"), "log_loss": 0.2},
        calibration={
            "mean_prediction": [0.1, float("nan")],
            "observed_rate": [0.2, float("nan")],
        },
        learner_weights={"logistic": 1.0},
        learner_risks={"logistic": float("nan")},
    )
    left = LongitudinalNuisanceRow("always", None, None, 1, 10, float("nan"), model=left_model)
    right = LongitudinalNuisanceRow("always", None, None, 1, 10, float("nan"), model=right_model)

    assert left_model.metrics["auc"] is not right_model.metrics["auc"]
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1
    assert pickle.loads(pickle.dumps(left)) == left

    changed_model = replace(right_model, metrics={"auc": float("nan"), "log_loss": 0.3})
    assert replace(right, model=changed_model) != left


def test_to_frame_preserves_the_legacy_column_prefix(weighted_result) -> None:  # type: ignore[no-untyped-def]
    columns = list(weighted_result.diagnostics.nuisance_models().to_frame().columns)
    assert columns[:6] == ["regimen", "cause", "horizon", "time", "n", "mse"]


def test_each_role_reports_its_stored_weighted_loss(weighted_result) -> None:  # type: ignore[no-untyped-def]
    report = weighted_result.diagnostics.nuisance_models()
    assert {row.role for row in report.rows} == {
        "treatment",
        "censoring",
        "outcome",
        "pseudo_outcome",
    }
    assert {row.evaluation for row in report.rows} == {"out_of_fold"}

    data = weighted_result.data
    at_risk = data.regimen_masks(data.treatment).at_risk(1)
    arm = np.nan_to_num(data.treatment[:, 0], nan=0.0).astype(int)
    probabilities = weighted_result.mechanism.treatment_observed[0]
    observed = probabilities[np.arange(data.n), arm]
    treatment = _row(report, "treatment", time=1)
    assert treatment.loss_name == "log_loss"
    assert treatment.loss == pytest.approx(
        -np.average(np.log(observed[at_risk]), weights=data.weights[at_risk])
    )

    censoring = _row(report, "censoring", time=1)
    retained = weighted_result.mechanism.censoring_observed[0]
    actual = data.uncensored[:, 0].astype(float)
    expected = -np.average(
        actual[at_risk] * np.log(retained[at_risk])
        + (1.0 - actual[at_risk]) * np.log(1.0 - retained[at_risk]),
        weights=data.weights[at_risk],
    )
    assert censoring.loss == pytest.approx(expected)

    for role in ("outcome", "pseudo_outcome"):
        row = _row(report, role)
        fit = next(
            fit
            for fit in weighted_result.fits.values()
            if fit.regimen.label == row.regimen and fit.cause == row.cause
        )
        step = next(step for step in fit.steps if step.time == row.time)
        expected = np.average(
            np.square(step.pseudo_outcome[step.trained_on] - step.initial[step.trained_on]),
            weights=data.weights[step.trained_on],
        )
        assert row.loss == pytest.approx(expected)
        assert row.model is not None
        assert row.model.calibration


def test_role_specific_prediction_mutations_move_only_the_bound_loss(weighted_result) -> None:  # type: ignore[no-untyped-def]
    baseline = weighted_result.diagnostics.nuisance_models()
    mechanism = weighted_result.mechanism
    changed_treatment = list(mechanism.treatment_observed)
    changed_treatment[0] = np.full_like(changed_treatment[0], 0.5)
    mutated = replace(
        weighted_result,
        mechanism=replace(mechanism, treatment_observed=tuple(changed_treatment)),
    )
    changed = _longitudinal_nuisances(mutated)
    assert _row(changed, "treatment", time=1).loss != _row(baseline, "treatment", time=1).loss
    assert _row(changed, "censoring", time=1).loss == _row(baseline, "censoring", time=1).loss

    changed_censoring = list(mechanism.censoring_observed)
    changed_censoring[0] = np.full_like(changed_censoring[0], 0.5)
    mutated = replace(
        weighted_result,
        mechanism=replace(mechanism, censoring_observed=tuple(changed_censoring)),
    )
    changed = _longitudinal_nuisances(mutated)
    assert _row(changed, "censoring", time=1).loss != _row(baseline, "censoring", time=1).loss
    assert _row(changed, "treatment", time=1).loss == _row(baseline, "treatment", time=1).loss

    for role in ("outcome", "pseudo_outcome"):
        wanted = _row(baseline, role)
        key, fit = next(
            (key, fit)
            for key, fit in weighted_result.fits.items()
            if fit.regimen.label == wanted.regimen and fit.cause == wanted.cause
        )
        steps = tuple(
            replace(step, initial=np.full_like(step.initial, 0.5))
            if step.time == wanted.time
            else step
            for step in fit.steps
        )
        mutated = replace(
            weighted_result, fits={**weighted_result.fits, key: replace(fit, steps=steps)}
        )
        changed = _longitudinal_nuisances(mutated)
        assert _row(changed, role).loss != wanted.loss
        other = "pseudo_outcome" if role == "outcome" else "outcome"
        assert _row(changed, other).loss == _row(baseline, other).loss


def test_later_mechanism_node_uses_only_its_nontrivial_at_risk_rows(weighted_result) -> None:  # type: ignore[no-untyped-def]
    result = weighted_result
    data = result.data
    mechanism = result.mechanism
    baseline = result.diagnostics.nuisance_models()
    at_risk = data.regimen_masks(data.treatment).at_risk(2)
    assert 0 < int(at_risk.sum()) < data.n

    arm = np.nan_to_num(data.treatment[:, 1], nan=0.0).astype(int)
    probabilities = mechanism.treatment_observed[1]
    observed = probabilities[np.arange(data.n), arm]
    treatment = _row(baseline, "treatment", time=2)
    assert treatment.n == int(at_risk.sum())
    assert treatment.loss == pytest.approx(
        -np.average(np.log(observed[at_risk]), weights=data.weights[at_risk])
    )

    retained = mechanism.censoring_observed[1]
    actual = data.uncensored[:, 1].astype(float)
    censoring = _row(baseline, "censoring", time=2)
    expected = -np.average(
        actual[at_risk] * np.log(retained[at_risk])
        + (1.0 - actual[at_risk]) * np.log(1.0 - retained[at_risk]),
        weights=data.weights[at_risk],
    )
    assert censoring.n == int(at_risk.sum())
    assert censoring.loss == pytest.approx(expected)

    treatment_outside = list(mechanism.treatment_observed)
    treatment_outside[1] = treatment_outside[1].copy()
    treatment_outside[1][~at_risk] = 0.5
    outside = _longitudinal_nuisances(
        replace(result, mechanism=replace(mechanism, treatment_observed=tuple(treatment_outside)))
    )
    assert _row(outside, "treatment", time=2).loss == treatment.loss
    assert _row(outside, "treatment", time=2).n == treatment.n

    treatment_inside = list(mechanism.treatment_observed)
    treatment_inside[1] = treatment_inside[1].copy()
    treatment_inside[1][at_risk] = 0.5
    inside = _longitudinal_nuisances(
        replace(result, mechanism=replace(mechanism, treatment_observed=tuple(treatment_inside)))
    )
    assert _row(inside, "treatment", time=2).loss != treatment.loss
    assert _row(inside, "treatment", time=2).n == treatment.n

    censoring_outside = list(mechanism.censoring_observed)
    censoring_outside[1] = censoring_outside[1].copy()
    censoring_outside[1][~at_risk] = 0.5
    outside = _longitudinal_nuisances(
        replace(result, mechanism=replace(mechanism, censoring_observed=tuple(censoring_outside)))
    )
    assert _row(outside, "censoring", time=2).loss == censoring.loss
    assert _row(outside, "censoring", time=2).n == censoring.n

    censoring_inside = list(mechanism.censoring_observed)
    censoring_inside[1] = censoring_inside[1].copy()
    censoring_inside[1][at_risk] = 0.5
    inside = _longitudinal_nuisances(
        replace(result, mechanism=replace(mechanism, censoring_observed=tuple(censoring_inside)))
    )
    assert _row(inside, "censoring", time=2).loss != censoring.loss
    assert _row(inside, "censoring", time=2).n == censoring.n


def test_complete_data_and_legacy_mechanisms_record_omissions() -> None:
    frame, _ = make_longitudinal(n=140, seed=41, censoring=False)
    complete = LTMLE({"always": 1}, **PARAMETRIC).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
    )
    report = complete.diagnostics.nuisance_models()
    assert not any(row.role == "censoring" for row in report.rows)
    assert [(item.role, item.time, item.reason) for item in report.omissions] == [
        ("censoring", None, LONGITUDINAL_CENSORING_NOT_FITTED)
    ]

    source = complete.mechanism
    legacy = Mechanism(
        source.treatment,
        source.censoring,
        source.treatment_by_fold,
        source.censoring_by_fold,
    )
    legacy_report = _longitudinal_nuisances(replace(complete, mechanism=legacy))
    assert not any(row.role == "treatment" for row in legacy_report.rows)
    assert {item.reason for item in legacy_report.omissions} == {
        LONGITUDINAL_CENSORING_NOT_FITTED,
        LONGITUDINAL_MECHANISM_PREDICTIONS_MISSING,
    }
    assert any(row.role == "outcome" for row in legacy_report.rows)


def test_categorical_treatment_is_one_model_row_with_multinomial_log_loss() -> None:
    frame = multivalue_panel(n=220, seed=41)
    result = LTMLE({"third": (2, 0)}, **PARAMETRIC).fit(frame, **COLUMNS)
    report = result.diagnostics.nuisance_models()
    treatment = [row for row in report.rows if row.role == "treatment"]
    assert len(treatment) == result.data.n_times
    first = treatment[0]
    mask = result.data.regimen_masks(result.data.treatment).at_risk(1)
    actual = result.data.treatment[:, 0].astype(int)
    probabilities = result.mechanism.treatment_observed[0]
    expected = -np.average(
        np.log(probabilities[np.arange(result.data.n), actual][mask]),
        weights=result.data.weights[mask],
    )
    assert first.loss == pytest.approx(expected)
    assert first.model is not None
    assert first.model.kind == "multinomial probability"
    assert first.model.calibration == {}


def test_binary_outcome_reports_weighted_brier_loss() -> None:
    frame, _ = make_longitudinal_survival(n=180, seed=41)
    frame["analysis_weight"] = np.linspace(0.5, 2.0, len(frame))
    result = LTMLE({"always": 1}, **PARAMETRIC).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        weights="analysis_weight",
    )
    report = result.diagnostics.nuisance_models()
    row = _row(report, "outcome")
    fit = next(
        fit
        for fit in result.fits.values()
        if fit.regimen.label == row.regimen and fit.horizon == row.horizon
    )
    step = next(step for step in fit.steps if step.time == row.time)
    expected = np.average(
        np.square(step.pseudo_outcome[step.trained_on] - step.initial[step.trained_on]),
        weights=result.data.weights[step.trained_on],
    )
    assert row.loss_name == "brier"
    assert row.loss == pytest.approx(expected)


def test_report_persistence_keeps_rows_models_and_omissions(weighted_result, tmp_path) -> None:  # type: ignore[no-untyped-def]
    report = weighted_result.diagnostics.nuisance_models()
    path = tmp_path / "longitudinal.joblib"
    joblib.dump(weighted_result, path)
    restored = joblib.load(path)
    restored_report = restored.diagnostics.nuisance_models()
    assert restored_report == report
    assert [row.reported_loss for row in restored_report.rows] == [
        row.reported_loss for row in report.rows
    ]
    assert restored_report.omissions == report.omissions
    for expected, actual in zip(report.rows, restored_report.rows, strict=True):
        assert expected.model is not None and actual.model is not None
        assert actual.model.name == expected.model.name
        assert actual.model.kind == expected.model.kind
        assert actual.model.metrics == pytest.approx(expected.model.metrics, nan_ok=True)
        assert actual.model.calibration.keys() == expected.model.calibration.keys()
        for name, values in expected.model.calibration.items():
            np.testing.assert_array_equal(actual.model.calibration[name], values)
        assert actual.model.learner_weights == pytest.approx(expected.model.learner_weights)
        assert actual.model.learner_risks == pytest.approx(expected.model.learner_risks)

    first = report.rows[0]
    assert first.model is not None
    changed_model = replace(first.model, metrics={**first.model.metrics, "log_loss": 123.0})
    assert replace(first, model=changed_model) != first


def test_same_fit_through_pandas_and_polars_keeps_the_complete_report() -> None:
    results = {}
    for backend in ("pandas", "polars"):
        frame, _ = make_longitudinal(n=220, seed=41, backend=backend)
        results[backend] = LTMLE({"always": 1, "never": 0}, **PARAMETRIC).fit(frame, **COLUMNS)
        assert results[backend].data.backend == backend

    reports = {name: result.diagnostics.nuisance_models() for name, result in results.items()}
    pandas_report = reports["pandas"]
    polars_report = reports["polars"]
    assert pandas_report.omissions == polars_report.omissions
    assert len(pandas_report.rows) == len(polars_report.rows)
    for left, right in zip(pandas_report.rows, polars_report.rows, strict=True):
        assert (
            left.regimen,
            left.cause,
            left.horizon,
            left.time,
            left.n,
            left.role,
            left.evaluation,
            left.loss_name,
        ) == (
            right.regimen,
            right.cause,
            right.horizon,
            right.time,
            right.n,
            right.role,
            right.evaluation,
            right.loss_name,
        )
        assert right.reported_loss == pytest.approx(left.reported_loss)
        assert left.model is not None and right.model is not None
        assert (right.model.name, right.model.kind) == (left.model.name, left.model.kind)
        assert right.model.metrics == pytest.approx(left.model.metrics, nan_ok=True)
        assert right.model.learner_weights == pytest.approx(left.model.learner_weights)
        assert right.model.learner_risks == pytest.approx(left.model.learner_risks)
        assert right.model.calibration.keys() == left.model.calibration.keys()
        for name, values in left.model.calibration.items():
            np.testing.assert_allclose(right.model.calibration[name], values, equal_nan=True)

    pandas_frame = pandas_report.to_frame()
    polars_frame = polars_report.to_frame()
    pd.testing.assert_frame_equal(pandas_frame, polars_frame.to_pandas(), check_dtype=False)


@pytest.mark.parametrize("n_folds, msm", [(1, None), (2, None), (1, 1)])
def test_every_recursion_constructor_retains_super_learner_details(n_folds, msm) -> None:
    frame, _ = make_longitudinal(n=300, seed=41)
    settings = {
        **FAST_KWARGS,
        "pseudo_learner": FAST_KWARGS["outcome_learner"],
        "n_folds": n_folds,
        "simultaneous": False,
    }
    working_model = MSM(
        design=lambda label, _horizon, data: np.column_stack(
            [np.ones(len(data)), np.full(len(data), label == "always")]
        ),
        terms=("intercept", "always"),
    )
    result = LTMLE(
        {"always": 1, "never": 0},
        **settings,
        **({"msm": working_model} if msm is not None else {}),
    ).fit(frame, **COLUMNS)
    report = result.diagnostics.nuisance_models()
    regression_rows = [row for row in report.rows if row.role in {"outcome", "pseudo_outcome"}]
    assert regression_rows
    assert all(row.model is not None and row.model.learner_weights for row in regression_rows)
    expected = "out_of_fold" if n_folds > 1 else "in_sample"
    assert {row.evaluation for row in report.rows} == {expected}


def test_observed_law_predictions_are_additive_to_fitted_outputs(monkeypatch) -> None:
    """The extra prediction design cannot move regimen predictions or fold companions."""
    from cleverly.longitudinal import sequential

    original = sequential.cross_fit_companion

    def old_path(*args, **kwargs):  # type: ignore[no-untyped-def]
        internal = [key for key in kwargs["predict_designs"] if key.startswith("__cleverly_")]
        reduced_kwargs = {
            **kwargs,
            "predict_designs": {
                key: value
                for key, value in kwargs["predict_designs"].items()
                if key not in internal
            },
        }
        predictions, companion, diagnostics = original(*args, **reduced_kwargs)
        # Production discards these injected values. They only satisfy the new return
        # schema while the fitted path exercises the former prediction-design set.
        exemplar = next(iter(predictions.values()))
        predictions.update({key: np.zeros_like(exemplar) for key in internal})
        return predictions, companion, diagnostics

    frame, _ = make_longitudinal(n=120, seed=41)
    monkeypatch.setattr(sequential, "cross_fit_companion", old_path)
    baseline = LTMLE({"always": 1, "never": 0}, **PARAMETRIC).fit(frame, **COLUMNS)
    monkeypatch.setattr(sequential, "cross_fit_companion", original)
    enhanced = LTMLE({"always": 1, "never": 0}, **PARAMETRIC).fit(frame, **COLUMNS)

    for time in range(enhanced.data.n_times):
        for label in ("always", "never"):
            np.testing.assert_array_equal(
                baseline.mechanism.treatment[time][label],
                enhanced.mechanism.treatment[time][label],
            )
            np.testing.assert_array_equal(
                baseline.mechanism.censoring[time][label],
                enhanced.mechanism.censoring[time][label],
            )
    for key, estimate in enhanced.estimates.items():
        assert estimate.psi == baseline.estimates[key].psi
        np.testing.assert_array_equal(
            estimate.influence_curve,
            baseline.estimates[key].influence_curve,
        )
    for key, fit in enhanced.fits.items():
        for old_step, new_step in zip(baseline.fits[key].steps, fit.steps, strict=True):
            np.testing.assert_array_equal(old_step.initial, new_step.initial)
            np.testing.assert_array_equal(old_step.targeted, new_step.targeted)


def test_internal_observed_prediction_keys_cannot_collide_with_regimen_labels() -> None:
    frame, _ = make_longitudinal(n=120, seed=41)
    labels = ("__cleverly_observed_treatment__", "__cleverly_observed_censoring__")
    result = LTMLE(dict.fromkeys(labels, 1), **PARAMETRIC).fit(frame, **COLUMNS)
    assert set(result.mechanism.treatment[0]) == set(labels)
    assert set(result.mechanism.censoring[0]) == set(labels)
    assert len(result.mechanism.treatment_observed) == result.data.n_times
    assert len(result.mechanism.censoring_observed) == result.data.n_times
