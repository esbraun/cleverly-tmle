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
    row = result.diagnostics.capability("corrections")
    assert row.status is AssessmentStatus.NOT_APPLICABLE
    with pytest.raises(CapabilityError, match="correction system"):
        result.diagnostics.corrections()


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
    }


@pytest.mark.parametrize("name", list(_free_calls()))
def test_a_free_function_refuses_by_name_without_the_facade(result: Any, name: str) -> None:
    """Off the facade, each function refuses before it reads a point-treatment attribute.

    The witness of a lost capability row: no row gates these calls, and each one still
    raises the record its row declares, rather than an ``AttributeError`` from the
    point-treatment state the result does not hold. A point-treatment report redirects to
    the facade.
    """
    function, kwargs, roadmap = _free_calls()[name]
    with pytest.raises(CapabilityError) as caught:
        function(result, **kwargs)
    assert caught.value.refusal.roadmap == roadmap
    if roadmap is None:
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
    """
    from cleverly.survival import engine

    frame = _arm_censored()
    known = 1 / (1 + np.exp(-0.8 * frame["W"].to_numpy()))

    def fixed(learner, L, A, train, predict, folds, random_state):  # type: ignore[no-untyped-def]
        return known[predict], None

    monkeypatch.setattr(engine, "_fit_propensity", fixed)
    horizons = [0.5, 1.0]
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

    # S_c(t_k-) recomputed from the stored censoring model, over the observed times before
    # each target, under each arm.
    time = frame["time"].to_numpy()
    grid = np.unique(time[time <= horizons[-1]])
    before = np.searchsorted(grid, horizons, side="left")

    def survival(arm: int) -> np.ndarray:
        X = np.column_stack([np.full(time.size, float(arm)), frame["W"].to_numpy()])
        cumulative = record.censoring.model.cumulative_hazard(X, grid)
        jumps = np.diff(np.concatenate([np.zeros((X.shape[0], 1)), cumulative], axis=1), axis=1)
        steps = np.concatenate([np.ones((X.shape[0], 1)), 1.0 - np.clip(jumps, 0.0, 1.0)], axis=1)
        return np.cumprod(steps, axis=1)[:, before]

    for arm in (0, 1):
        np.testing.assert_allclose(record.retention_before[arm], survival(arm), rtol=1e-12)
    # The arms differ, so a swap of axis 0 cannot pass the check above.
    assert np.max(np.abs(record.retention_before[1] - record.retention_before[0])) > 0.05
    # The floor is the resolved default the weights applied, 5 / (sqrt(n) log n).
    n = len(frame)
    assert record.weight_floor == 5.0 / (np.sqrt(n) * np.log(n))
