"""Assessment of a continuous-time survival fit (X101): every row runs or refuses by name.

The support, nuisance-model and score-equation rows and ``diagnostics.influence()`` read
the fit's record and its influence curves. Every refutation, the truncation curve and every
sensitivity analysis refuses with a ``CapabilityError`` naming its roadmap item, and the
capability row declares the same record, so ``assess()`` reports each refusal and never
raises.

The witnesses guard components that could vanish without any exact check noticing: the
arm axis of the stored censoring survival, the orientation of the stored treatment
probability, the score gate, the support gate, and the free functions' own refusals, which
still name the item when a capability row is lost.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.assessment import AssessmentStatus
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.inference.sparse import influence_report
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from cleverly.survival.diagnostics import (
    ContinuousSurvivalNuisances,
    ContinuousSurvivalScores,
    ContinuousSurvivalSupport,
)
from tests.unit._capability_sweep_support import KINDS, ran, sweep, sweep_arguments


@pytest.fixture
def result() -> Any:
    """A fresh copy of the sweep's continuous-time kind, with an empty report cache."""
    return KINDS["continuous_survival"].build()


#: Each refused diagnostic and the roadmap item its refusal names.
DIAGNOSTIC_REFUSALS = {"refute": "X101", "truncation_curve": "X101"}

#: Each refused sensitivity analysis and the roadmap item its refusal names.
SENSITIVITY_REFUSALS = {
    "omitted_confounding": "F16",
    "robustness_value": "F16",
    "elements": "F16",
    "contour": "F16",
    "benchmark": "F16",
    "marginal_bounds": "F16",
    "missingness": "F16",
    "tipping_gamma": "F16",
    "simulated_confounding": "X101",
    "evalue": "X6",
}


# ----------------------------------------------------------------- the combined report


def test_assess_runs_every_row_or_names_its_refusal(result: Any) -> None:
    report = result.assess(include_refits=True, include_retargets=True, random_state=0)
    statuses = {
        item.name: item.status
        for item in (*report.validation.items, *report.diagnostics.items, *report.sensitivity.items)
    }
    assert statuses["score_equations"] is AssessmentStatus.PASSED
    assert statuses["support"] is AssessmentStatus.COMPLETED
    assert statuses["nuisance_models"] is AssessmentStatus.COMPLETED
    assert statuses["corrections"] is AssessmentStatus.NOT_APPLICABLE
    for name in (*DIAGNOSTIC_REFUSALS, *SENSITIVITY_REFUSALS):
        assert statuses[name] is AssessmentStatus.UNAVAILABLE, name
    for item in (*report.diagnostics.items, *report.sensitivity.items):
        if item.name in DIAGNOSTIC_REFUSALS or item.name in SENSITIVITY_REFUSALS:
            expected = {**DIAGNOSTIC_REFUSALS, **SENSITIVITY_REFUSALS}[item.name]
            assert item.refusal is not None and item.refusal.roadmap == expected, item.name


def test_the_capability_sweep_finds_no_disagreement(result: Any) -> None:
    assert sweep(result) == []
    answered = ran(KINDS["continuous_survival"].build(), sweep_arguments(result))
    assert {"support", "nuisance_models", "score_equations"} <= answered


def test_validate_passes_on_the_record(result: Any) -> None:
    assert result.validate().passed


# ----------------------------------------------------------------- the rows that run


def test_support_reads_the_treatment_and_censoring_fits(result: Any) -> None:
    report = result.diagnostics.support()
    record = result.record
    assert isinstance(report, ContinuousSurvivalSupport)
    assert report.n == result.n == record.p1.shape[0]
    assert set(report.propensity_quantiles) == {"overall", *result.arms}
    overall = report.propensity_quantiles["overall"]
    assert overall[0.0] == pytest.approx(record.p1.min())
    assert overall[1.0] == pytest.approx(record.p1.max())
    assert set(report.censoring_quantiles) == set(result.horizons)
    received = record.retention_before[record.arm, np.arange(result.n)]
    for k, time in enumerate(result.horizons):
        assert report.censoring_quantiles[time][0.0] == pytest.approx(received[:, k].min())
    # Both arms, every row, every target time; uncensored up to a later time is no likelier.
    assert record.retention_before.shape == (2, result.n, len(result.horizons))
    assert np.all(np.diff(record.retention_before, axis=-1) <= 0)
    assert np.all((record.retention_before > 0) & (record.retention_before <= 1))
    for label in result.arms:
        ratio = report.effective_sample_size[label]["ratio"]
        assert 0.0 < ratio <= 1.0
    assert report.weight_floor == record.weight_floor > 0
    assert report.bounded_share == record.bounded_share
    assert report.severity == "adequate"


def test_score_equations_report_the_walk_against_its_cutoff(result: Any) -> None:
    report = result.diagnostics.score_equations()
    record = result.record
    assert isinstance(report, ContinuousSurvivalScores)
    # Every risk of both arms, each cause and each time is one target of the walk.
    assert len(report.names) == len(report.standardized_score) == 2 * 2 * 2
    assert set(report.names) <= set(result.estimates)
    assert report.cutoff == record.cutoff
    assert report.passed and report.converged and bool(record.converged)
    assert report.steps == record.steps
    assert report.worst_ratio == pytest.approx(max(record.standardized_score) / record.cutoff)


def test_score_equations_refuse_a_caller_tolerance(result: Any) -> None:
    with pytest.raises(CapabilityError, match="read the report's cutoff") as caught:
        result.diagnostics.score_equations(tolerance=0.1)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_nuisance_models_list_each_selection(result: Any) -> None:
    report = result.diagnostics.nuisance_models()
    assert isinstance(report, ContinuousSurvivalNuisances)
    roles = [row.role for row in report.rows]
    assert roles == ["hazard of cause 1", "hazard of cause 2", "censoring hazard", "treatment"]
    assert report.rows[0].chosen == "CoxHazard('main')"
    assert report.rows[-1].chosen in report.rows[-1].members
    assert not report.cross_fitted and report.n_folds == 1
    assert report.grid_size == result.record.grid_size


def test_influence_reads_each_estimates_curve(result: Any) -> None:
    report = result.diagnostics.influence()
    alpha = next(iter(result.estimates.values())).alpha
    direct = influence_report(result.estimates, alpha=alpha)
    assert [row.name for row in report.rows] == list(result.estimates)
    for mine, theirs in zip(report.rows, direct.rows, strict=True):
        np.testing.assert_array_equal(
            dataclasses.astuple(mine)[1:-1], dataclasses.astuple(theirs)[1:-1]
        )
    for row in report.rows:
        assert row.n == result.n


# ----------------------------------------------------------------- the rows that refuse


@pytest.mark.parametrize(("operation", "roadmap"), DIAGNOSTIC_REFUSALS.items())
def test_each_refused_diagnostic_names_its_item(result: Any, operation: str, roadmap: str) -> None:
    row = result.diagnostics.capability(operation)
    assert not row.available and row.status is AssessmentStatus.UNAVAILABLE
    assert row.refusal.roadmap == roadmap
    with pytest.raises(CapabilityError, match=rf"{roadmap} in docs/roadmap\.md") as caught:
        getattr(result.diagnostics, operation)()
    assert caught.value.refusal.roadmap == roadmap
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP


@pytest.mark.parametrize(("operation", "roadmap"), SENSITIVITY_REFUSALS.items())
def test_each_refused_sensitivity_analysis_names_its_item(
    result: Any, operation: str, roadmap: str
) -> None:
    row = result.sensitivity.capability(operation)
    assert not row.available and row.status is AssessmentStatus.UNAVAILABLE
    assert row.refusal.roadmap == roadmap
    with pytest.raises(CapabilityError, match=rf"{roadmap} in docs/roadmap\.md") as caught:
        getattr(result.sensitivity, operation)()
    assert caught.value.refusal.roadmap == roadmap


def test_an_explicit_evalue_estimand_lifts_no_refusal(result: Any) -> None:
    name = next(key for key in result.estimates if key.startswith("rr_regimen"))
    with pytest.raises(CapabilityError, match=r"X6 in docs/roadmap\.md"):
        result.sensitivity.evalue(name)


def test_corrections_do_not_apply(result: Any) -> None:
    """The row and the free function carry one refusal record, which the facade raises."""
    from cleverly.validation import correction_check

    row = result.diagnostics.capability("corrections")
    assert row.status is AssessmentStatus.NOT_APPLICABLE
    with pytest.raises(CapabilityError, match="correction system") as facade:
        result.diagnostics.corrections()
    with pytest.raises(CapabilityError) as free:
        correction_check(result, tolerance=1e-3)
    assert row.refusal is not None
    assert free.value.refusal == row.refusal
    assert row.refusal.reason is RefusalReason.INVALID_DECLARATION
    # The facade prefixes the row's status, as it does for every row it gates.
    assert facade.value.refusal.reason is row.refusal.reason
    assert facade.value.refusal.condition.endswith(row.refusal.condition)


def test_nothing_replays_from_the_stored_fit(result: Any) -> None:
    replay = result.replayability
    assert replay.summarize_existing_artifacts
    assert not (replay.retarget_cached_nuisances or replay.refit_nuisances)
    assert not replay.rerun_configuration


# ----------------------------------------------------------------- witnesses


def _free_calls() -> dict[str, tuple[Any, dict[str, Any], str]]:
    from cleverly import sensitivity, validation
    from tests.unit._simulated_confounding_support import _GRID

    return {
        "refute": (validation.refute, {}, "X101"),
        "truncation_curve": (sensitivity.truncation_curve, {}, "X101"),
        "simulated_confounding": (sensitivity.simulated_confounding, {"grid": _GRID}, "X101"),
        "omitted_variable_bounds": (sensitivity.omitted_variable_bounds, {}, "F16"),
        "robustness_value": (sensitivity.robustness_value, {}, "F16"),
        "sensitivity_elements": (sensitivity.sensitivity_elements, {}, "F16"),
        "contour_data": (sensitivity.contour_data, {}, "F16"),
        "benchmark": (sensitivity.benchmark, {"covariates": ["W"]}, "F16"),
        "marginal_sensitivity_bounds": (sensitivity.marginal_sensitivity_bounds, {}, "F16"),
        "missingness_tilt": (sensitivity.missingness_tilt, {}, "F16"),
        "tipping_gamma": (sensitivity.tipping_gamma, {}, "F16"),
        "evalue": (sensitivity.evalue, {}, "X6"),
        "positivity_report": (sensitivity.positivity_report, {}, None),
        "nuisance_diagnostics": (validation.nuisance_diagnostics, {}, None),
        "score_check": (validation.score_check, {}, None),
        "correction_check": (validation.correction_check, {"tolerance": 1e-3}, None),
    }


@pytest.mark.parametrize("name", list(_free_calls()))
def test_a_free_function_refuses_by_name_without_the_facade(result: Any, name: str) -> None:
    """Off the facade, each function refuses before it reads a point-treatment attribute.

    The witness of a lost capability row: no row gates these calls, and each one still
    raises the record its row declares, rather than an ``AttributeError`` from the
    point-treatment state the result does not hold. A point-treatment report redirects to
    the facade, and the correction check refuses as its ``corrections`` row does.
    """
    function, kwargs, roadmap = _free_calls()[name]
    with pytest.raises(CapabilityError) as caught:
        function(result, **kwargs)
    assert caught.value.refusal.roadmap == roadmap
    if name == "correction_check":
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    elif roadmap is None:
        assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_a_failed_walk_fails_the_score_row(result: Any) -> None:
    """The score gate is not vacuous: a target above the cutoff fails the row and validation."""
    record = result.record
    scores = (*record.standardized_score[:-1], 3.0 * record.cutoff)
    failed = dataclasses.replace(
        result,
        record=dataclasses.replace(record, standardized_score=scores, converged=False),
    )
    item = failed.validate()["score_equations"]
    assert item.status is AssessmentStatus.FAILED
    assert "3 x the cutoff" in item.detail
    assert not failed.validate().passed


def test_a_binding_floor_warns_on_the_support_row(result: Any) -> None:
    """The support gate reads the share of cells the floor raised."""
    record = dataclasses.replace(result.record, bounded_share=0.2)
    item = dataclasses.replace(result, record=record).validate()["support"]
    assert item.status is AssessmentStatus.WARNING
    assert "20.0% of the followed cells" in item.detail


def _arm_censored(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Two causes, with censoring three times as fast in the treated arm."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    h1 = np.exp(-1.5 + 0.5 * w - a)
    h2 = np.exp(-2.0 - 0.5 * w - 0.5 * a)
    t = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    c = rng.exponential(1 / (0.1 * np.exp(np.log(3.0) * a)))
    return pd.DataFrame(
        {"time": np.minimum(t, c), "event": np.where(t <= c, cause, 0), "A": a, "W": w}
    )


def test_the_stored_probabilities_are_the_received_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonzero witness of the arm each stored array is read under.

    The treatment probability is a known function, so a swap of the arms of ``pi`` moves it
    exactly. Censoring depends on the arm, so the stored censoring survival of each arm
    differs from the other arm's by far more than rounding, and a swap of the arm axis fails.
    A target on an observed censoring time separates the left limit S_c(t_k-) from S_c(t_k),
    so a lookup that includes the target's own jump fails too.
    """
    from cleverly.survival import engine

    frame = _arm_censored()
    known = 1 / (1 + np.exp(-0.8 * frame["W"].to_numpy()))

    def fixed(learner, L, A, train, predict, folds, random_state):  # type: ignore[no-untyped-def]
        return known[predict], None

    monkeypatch.setattr(engine, "_fit_propensity", fixed)
    # One target sits on an observed censoring time, where S_c(t_k-) and S_c(t_k) differ.
    time = frame["time"].to_numpy()
    censored = np.sort(time[(frame["event"].to_numpy() == 0) & (time > 0.5) & (time < 1.0)])
    on_censoring = float(censored[censored.size // 2])
    horizons = [0.5, on_censoring, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ContinuousSurvivalTMLE(
            horizons,
            cross_fit=False,
            hazard_library=[CoxHazard("main")],
            censoring_library=[CoxHazard("main")],
            random_state=0,
        ).fit(frame, time="time", event="event", treatment="A", baseline=["W"])
    record = fit.record
    np.testing.assert_array_equal(record.p1, known)
    np.testing.assert_array_equal(record.arm, frame["A"].to_numpy())

    # S_c(t-) recomputed from the stored censoring model as the product, over the observed
    # times strictly before t, of one minus each censoring hazard jump, under each arm.
    grid = np.unique(time[time <= horizons[-1]])

    def survival(arm: int, strictly_before: bool) -> np.ndarray:
        X = np.column_stack([np.full(time.size, float(arm)), frame["W"].to_numpy()])
        cumulative = record.censoring.model.cumulative_hazard(X, grid)
        jumps = np.diff(cumulative, axis=1, prepend=0.0)
        steps = 1.0 - np.clip(jumps, 0.0, 1.0)
        columns = []
        for target in horizons:
            kept = grid < target if strictly_before else grid <= target
            columns.append(np.prod(steps[:, kept], axis=1))
        return np.column_stack(columns)

    for arm in (0, 1):
        np.testing.assert_allclose(
            record.retention_before[arm], survival(arm, strictly_before=True), rtol=1e-12
        )
        # At a censoring time the left limit excludes that time's own jump.
        at = survival(arm, strictly_before=False)[:, 1]
        assert np.any(record.retention_before[arm, :, 1] > at)
    # The arms differ, so a swap of axis 0 cannot pass the check above.
    assert np.max(np.abs(record.retention_before[1] - record.retention_before[0])) > 0.05
    # The floor is the resolved default the weights applied, 5 / (sqrt(n) log n).
    n = len(frame)
    assert record.weight_floor == 5.0 / (np.sqrt(n) * np.log(n))


def test_the_stored_treatment_probability_is_out_of_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under cross-fitting each row carries the propensity of the model its fold did not train.

    The patched fit returns a value that encodes its training rows, so a row filled from an
    in-sample fit, or from another fold's model, carries the wrong value.
    """
    from cleverly.survival import engine

    frame = _arm_censored(n=300)
    n = len(frame)
    calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def encoding(learner, L, A, train, predict, folds, random_state):  # type: ignore[no-untyped-def]
        rows = np.arange(n)
        trained, predicted = rows[train], rows[predict]
        value = 0.2 + 0.6 * float(np.mean(trained)) / n
        calls.append((trained, predicted, value))
        return np.full(predicted.size, value), None

    monkeypatch.setattr(engine, "_fit_propensity", encoding)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ContinuousSurvivalTMLE(
            [0.5, 1.0],
            cross_fit=True,
            n_folds=3,
            hazard_library=[CoxHazard("main")],
            censoring_library=[CoxHazard("main")],
            random_state=0,
        ).fit(frame, time="time", event="event", treatment="A", baseline=["W"])
    p1 = fit.record.p1
    assert len(calls) == 3
    np.testing.assert_array_equal(np.sort(np.concatenate([c[1] for c in calls])), np.arange(n))
    for trained, predicted, value in calls:
        assert np.intersect1d(trained, predicted).size == 0
        np.testing.assert_array_equal(p1[predicted], value)
    # Each fold's value differs, so a row read from another fold's model fails the check above.
    assert len({value for _, _, value in calls}) == 3
