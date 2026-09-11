"""Exact, full-recursion evidence for longitudinal cumulative-bound curves."""

from __future__ import annotations

import inspect
import json
import pickle
from dataclasses import replace
from typing import Any

import narwhals as nw
import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    LongitudinalTreatment,
    RegimeContrast,
)
from cleverly._assessment_cache import _CACHE_GENERATIONS, _pack_cached
from cleverly.assessment import _truncation_item, replayability
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
)
from cleverly.learners import SuperLearner
from cleverly.longitudinal import LTMLE, LongitudinalResult
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal import sequential as longitudinal_sequential
from cleverly.longitudinal.estimator import (
    LONGITUDINAL_REPLAY_FITTED_BOUND_MISMATCH,
    LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE,
    LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER,
    LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED,
    LONGITUDINAL_REPLAY_RECIPE_MISSING,
    _estimate_fit_contributors,
    _exact_replay_equal,
    _replay_random_state_omissions,
    _replay_recipe,
    _score_cell_truncation_counts,
)
from cleverly.longitudinal.regimen import Plan, resolve_regimens
from cleverly.longitudinal.sequential import Mechanism, fit_regimen
from cleverly.msm import MSM
from cleverly.utils.frames import available_backends
from cleverly.validation.longitudinal import _longitudinal_scores
from tests.unit.test_sequential_design import multivalue_panel

COLUMNS = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}

SETTINGS = {
    "reference": "never",
    "outcome_learner": LinearRegression(),
    "pseudo_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "censoring_learner": LogisticRegression(max_iter=1000),
    "n_folds": 2,
    "learner_folds": 2,
    "random_state": 17,
    "simultaneous": False,
}


def _payload(frame: Any) -> dict[str, list[Any]]:
    return nw.from_native(frame, eager_only=True).to_dict(as_series=False)


def _assert_active_curve(result: Any, bound: float = 0.3) -> None:
    """Pin movement and every alias's score-cell count to its contributing fits."""

    payload = _payload(result.diagnostics.truncation_curve([bound]))
    recipe = result.replay_recipe
    assert recipe is not None
    estimates, fits, msm_fits = longitudinal_estimator._refit_bound(result, recipe, (bound, 1.0))
    for index, name in enumerate(payload["estimand"]):
        expected = _score_cell_truncation_counts(_estimate_fit_contributors(result, name, fits))
        assert (
            payload["truncated_score_cells"][index],
            payload["evaluated_score_cells"][index],
        ) == expected
    assert any(value > 0 for value in payload["truncated_score_cells"])
    assert any(value != 0.0 for value in payload["delta_from_fitted"])

    active = replace(result, estimates=estimates, fits=fits, msm_fits=msm_fits)
    score_rows = [
        row
        for row in _longitudinal_scores(active, tolerance=recipe.tol).rows
        if row.kind == "solver"
    ]
    assert score_rows
    assert all(row.converged and row.passed for row in score_rows)
    assert all(row.relative_score <= recipe.tol for row in score_rows)


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    frame, _ = make_longitudinal(n=140, seed=17)
    return LTMLE({"always": 1, "never": 0}, **SETTINGS).fit(frame, **COLUMNS)


def test_recipe_precedes_replay_and_retains_only_unfitted_inputs(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    assert recipe.omissions == ()
    assert [plan.label for plan in recipe.plans] == ["always", "never"]
    assert all(not plan.values.flags.writeable for plan in recipe.plans)
    assert all(not np.shares_memory(plan.values, result.data.treatment) for plan in recipe.plans)
    assert recipe.outcome_learner is not None
    assert recipe.pseudo_learner is not None
    # A cloneable specification is retained, never any fitted node-level Q learner.
    assert not hasattr(recipe.outcome_learner, "models_")
    assert not hasattr(recipe.pseudo_learner, "models_")
    assert not any("fit" in field for field in vars(recipe))


def test_replay_recipe_is_the_only_appended_constructor_slot() -> None:
    parameters = tuple(inspect.signature(LongitudinalResult).parameters)
    assert parameters[-1] == "replay_recipe"


def test_deterministic_recursive_learners_need_no_fit_level_seed() -> None:
    frame, _ = make_longitudinal(n=140, seed=17)
    result = LTMLE(
        {"always": 1, "never": 0},
        **{**SETTINGS, "random_state": None},
    ).fit(frame, **COLUMNS)

    assert result.replay_recipe is not None
    assert result.replay_recipe.omissions == ()
    assert replayability(result).refit_nuisances
    payload = _payload(result.diagnostics.truncation_curve([result.config.g_bounds]))
    assert payload["psi"] == payload["fitted_psi"]
    assert all(value == 0.0 for value in payload["delta_from_fitted"])


def test_explicit_grid_preserves_order_pairs_backend_and_result_subset(result) -> None:  # type: ignore[no-untyped-def]
    requested = tuple(result.estimates)[1::-1]
    curve = result.diagnostics.truncation_curve([(0.01, 0.9), 0.2], estimands=requested)
    payload = _payload(curve)

    assert list(payload) == [
        "lower_bound",
        "upper_bound",
        "estimand",
        "psi",
        "is_fitted_bound",
        "fitted_lower_bound",
        "fitted_upper_bound",
        "fitted_psi",
        "delta_from_fitted",
        "truncated_score_cells",
        "evaluated_score_cells",
    ]
    assert payload["estimand"] == list(requested) * 2
    assert list(zip(payload["lower_bound"], payload["upper_bound"], strict=True)) == [
        (0.01, 0.9),
        (0.01, 0.9),
        (0.2, 1.0),
        (0.2, 1.0),
    ]
    assert not {"std_err", "ci_lower", "ci_upper"} & payload.keys()
    assert result.data.backend in type(curve).__module__


def test_fitted_bound_reproduces_every_retained_fit_artifact_exactly(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    estimates, fits, msm_fits = longitudinal_estimator._refit_bound(
        result, recipe, result.config.g_bounds
    )

    assert _exact_replay_equal(estimates, result.estimates)
    assert _exact_replay_equal(fits, result.fits)
    assert _exact_replay_equal(msm_fits, result.msm_fits)
    curve = _payload(result.diagnostics.truncation_curve([result.config.g_bounds]))
    assert all(curve["is_fitted_bound"])
    assert all(value == 0.0 for value in curve["delta_from_fitted"])
    assert curve["psi"] == curve["fitted_psi"]


def test_fitted_bound_gate_refuses_a_nonpoint_artifact_mismatch(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = longitudinal_estimator._refit_bound

    def mutated(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        estimates, fits, msm_fits = real(*args, **kwargs)
        key = next(iter(fits))
        fit = fits[key]
        changed = fit.cumulative.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
        return estimates, {**fits, key: replace(fit, cumulative=changed)}, msm_fits

    monkeypatch.setattr(longitudinal_estimator, "_refit_bound", mutated)
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_FITTED_BOUND_MISMATCH):
        longitudinal_estimator.longitudinal_truncation_curve(result, [0.25])


def test_crossfit_active_replay_rejects_a_stitched_oof_slab_mutation(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    active = (0.25, 1.0)
    _, correct, _ = longitudinal_estimator._refit_bound(result, recipe, active)
    real = Mechanism.cumulative_with_unbounded

    def stitched(self: Mechanism, data: Any, plan: Any, bounds: Any, *, fold: int | None = None):  # type: ignore[no-untyped-def]
        return real(
            self,
            data,
            plan,
            bounds,
            fold=None if tuple(bounds) == active else fold,
        )

    monkeypatch.setattr(Mechanism, "cumulative_with_unbounded", stitched)
    _, mutated, _ = longitudinal_estimator._refit_bound(result, recipe, active)
    with pytest.raises(AssertionError):
        assert _exact_replay_equal(mutated, correct)


def test_changed_bound_rebuilds_earlier_pseudo_outcomes_and_moves_estimates(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    estimates, fits, _ = longitudinal_estimator._refit_bound(result, recipe, (0.25, 1.0))

    assert any(estimates[name].psi != result.estimates[name].psi for name in estimates)
    for key, fit in fits.items():
        original = result.fits[key]
        assert fit.steps[0].pseudo_outcome.shape == original.steps[0].pseudo_outcome.shape
        assert np.any(fit.steps[0].pseudo_outcome != original.steps[0].pseudo_outcome)


def test_active_replay_exactly_matches_a_direct_fresh_recursion(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    assert recipe.outcome_learner is not None
    assert recipe.pseudo_learner is not None
    _, replayed, _ = longitudinal_estimator._refit_bound(result, recipe, (0.25, 1.0))
    plan = recipe.plans[0]
    fresh = fit_regimen(
        result.data,
        plan,
        result.mechanism,
        outcome_learner=clone(recipe.outcome_learner),
        pseudo_learner=clone(recipe.pseudo_learner),
        folds=result.folds,
        scaler=result.scaler,
        g_bounds=(0.25, 1.0),
        horizon=result.config.horizons[0],
        cause=None,
        alpha=recipe.alpha,
        max_iter=recipe.max_iter,
        tol=recipe.tol,
        n_jobs=recipe.n_jobs,
    )

    assert _exact_replay_equal(replayed[plan.label], fresh)


def test_reusing_one_earlier_outcome_prediction_breaks_active_replay(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    correct = _payload(longitudinal_estimator.longitudinal_truncation_curve(result, [0.25]))
    real = longitudinal_sequential.prepare_node

    def reused(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        node = real(*args, **kwargs)
        plan = args[1]
        cumulative = args[2]
        time = args[4]
        fitted = result.fits[plan.label]
        _, fitted_slab = result.mechanism.cumulative_with_unbounded(
            result.data,
            plan,
            result.config.g_bounds,
            fold=kwargs["outer_fold"],
        )
        if time == 1 and not np.array_equal(cumulative, fitted_slab):
            stale = fitted.steps[0].initial
            return replace(node, initial=stale)
        return node

    monkeypatch.setattr(longitudinal_sequential, "prepare_node", reused)
    mutated = _payload(longitudinal_estimator.longitudinal_truncation_curve(result, [0.25]))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(mutated["psi"], correct["psi"])


def test_score_cell_counts_are_alias_specific_and_use_only_scored_rows(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    _, fits, _ = longitudinal_estimator._refit_bound(result, recipe, (0.25, 1.0))
    curve = _payload(longitudinal_estimator.longitudinal_truncation_curve(result, [0.25]))

    for index, name in enumerate(curve["estimand"]):
        contributing = _estimate_fit_contributors(result, name, fits)
        expected = _score_cell_truncation_counts(contributing)
        assert (
            curve["truncated_score_cells"][index],
            curve["evaluated_score_cells"][index],
        ) == expected
    level_counts = curve["evaluated_score_cells"][:2]
    contrast_count = curve["evaluated_score_cells"][2]
    assert contrast_count == sum(level_counts)


def test_recipe_and_curve_survive_persistence(result) -> None:  # type: ignore[no-untyped-def]
    restored = pickle.loads(pickle.dumps(result))
    expected = _payload(result.diagnostics.truncation_curve([0.2]))
    actual = _payload(restored.diagnostics.truncation_curve([0.2]))

    assert restored.replay_recipe is not None
    assert all(not plan.values.flags.writeable for plan in restored.replay_recipe.plans)
    assert actual == expected


def test_workflow_and_direct_entries_have_exact_fit_and_curve_parity() -> None:
    frame, _ = make_longitudinal(n=140, seed=17)
    direct = LTMLE({"always": 1, "never": 0}, **SETTINGS).fit(frame, **COLUMNS)
    workflow = (
        CausalStudy(frame, design=LongitudinalTreatment(**COLUMNS))
        .identify(RegimeContrast({"always": 1, "never": 0}, reference="never"))
        .estimate(**{name: value for name, value in SETTINGS.items() if name != "reference"})
    )
    requested = tuple(workflow.estimates)
    direct_payload = _payload(
        direct.diagnostics.truncation_curve([direct.config.g_bounds, 0.2], estimands=requested)
    )
    workflow_payload = _payload(
        workflow.diagnostics.truncation_curve([workflow.config.g_bounds, 0.2])
    )

    assert len(requested) == 1
    assert _exact_replay_equal(direct.estimates[requested[0]], workflow.estimates[requested[0]])
    assert _exact_replay_equal(direct.fits, workflow.fits)
    assert _exact_replay_equal(direct.mechanism, workflow.mechanism)
    assert _exact_replay_equal(direct.folds, workflow.folds)
    assert direct.replay_recipe is not None and workflow.replay_recipe is not None
    for left, right in zip(direct.replay_recipe.plans, workflow.replay_recipe.plans, strict=True):
        assert left.label == right.label
        np.testing.assert_array_equal(left.values, right.values)
    assert direct.replay_recipe.outcome_learner is not None
    assert workflow.replay_recipe.outcome_learner is not None
    assert direct.replay_recipe.outcome_learner.get_params(
        deep=True
    ) == workflow.replay_recipe.outcome_learner.get_params(deep=True)
    assert direct_payload == workflow_payload


def test_survival_curve_replays_every_horizon_in_report_order() -> None:
    frame, _ = make_longitudinal_survival(n=180, seed=31)
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": LogisticRegression(max_iter=1000, random_state=31),
            "n_folds": 1,
        },
    ).fit(
        frame,
        outcome=("Y1", "Y2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )
    payload = _payload(result.diagnostics.truncation_curve([result.config.g_bounds, 0.2]))

    assert payload["estimand"] == list(result.estimates) * 2
    assert len(set(payload["estimand"])) == len(result.config.horizons)
    assert all(payload["is_fitted_bound"][: len(result.estimates)])
    _assert_active_curve(result)


def test_msm_curve_replays_projection_artifacts_and_reports_only_terms() -> None:
    frame, _ = make_longitudinal(n=180, seed=37)
    duration = {"always": 2.0, "never": 0.0, "late": 1.0}

    def design(label: Any, horizon: int, baseline: Any) -> np.ndarray:
        del horizon
        return np.column_stack([np.ones(len(baseline)), np.full(len(baseline), duration[label])])

    result = LTMLE(
        {"always": 1, "never": 0, "late": (0, 1)},
        msm=MSM(design=design, terms=("intercept", "duration")),
        **{name: value for name, value in SETTINGS.items() if name not in {"reference", "n_folds"}},
        n_folds=1,
    ).fit(frame, **COLUMNS)
    recipe = result.replay_recipe
    assert recipe is not None
    estimates, fits, msm_fits = longitudinal_estimator._refit_bound(
        result, recipe, result.config.g_bounds
    )

    assert _exact_replay_equal(estimates, result.estimates)
    assert _exact_replay_equal(fits, result.fits)
    assert _exact_replay_equal(msm_fits, result.msm_fits)
    payload = _payload(result.diagnostics.truncation_curve([0.2]))
    assert payload["estimand"] == list(result.estimates)
    assert all(name.startswith("msm_regimen[") for name in payload["estimand"])
    _assert_active_curve(result)


def test_competing_risk_curve_preserves_cause_horizon_structure() -> None:
    frame, _ = make_longitudinal_competing(n=220, seed=41)
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": LogisticRegression(max_iter=1000, random_state=41),
            "n_folds": 1,
        },
    ).fit(
        frame,
        outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")},
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )
    payload = _payload(result.diagnostics.truncation_curve([result.config.g_bounds]))

    assert payload["estimand"] == list(result.estimates)
    assert len(payload["estimand"]) == 4
    assert all("relapse" in name or "death" in name for name in payload["estimand"])
    _assert_active_curve(result)


def test_dynamic_categorical_weighted_clustered_recipe_replays_exactly() -> None:
    frame = multivalue_panel(n=300, seed=43).assign(
        analysis_weight=np.linspace(0.5, 1.5, 300),
        cluster=np.repeat(np.arange(150), 2),
    )
    result = LTMLE(
        {
            "never": 0,
            "dynamic": (2, lambda history: (history["L2"] > 0).astype(float)),
        },
        reference="never",
        outcome_learner=LogisticRegression(max_iter=1000, random_state=43),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=1,
        learner_folds=2,
        random_state=43,
        simultaneous=False,
    ).fit(
        frame,
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1",),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
        weights="analysis_weight",
        id="cluster",
    )

    assert result.data.cluster is not None
    assert not np.all(result.data.weights == 1.0)
    result.diagnostics.truncation_curve([result.config.g_bounds])


def test_pandas_and_polars_curves_have_exact_payload_parity() -> None:
    required = ("pandas", "polars")
    assert set(required) <= set(available_backends())
    payloads = {}
    for backend in required:
        frame, _ = make_longitudinal(n=80, seed=23, backend=backend)
        result = LTMLE(
            {"always": 1},
            **{**SETTINGS, "reference": "always", "n_folds": 1},
        ).fit(frame, **COLUMNS)
        curve = result.diagnostics.truncation_curve([result.config.g_bounds, 0.2])
        assert backend in type(curve).__module__
        payloads[backend] = _payload(curve)

    assert payloads["pandas"] == payloads["polars"]


def test_serial_and_parallel_replays_are_bit_identical(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    serial = replace(result, replay_recipe=replace(recipe, n_jobs=1))
    parallel = replace(result, replay_recipe=replace(recipe, n_jobs=2))
    serial.assessment_cache.clear()
    parallel.assessment_cache.clear()
    assert serial.assessment_cache is not parallel.assessment_cache
    calls: list[int] = []
    real = longitudinal_estimator._refit_bound

    def tracked(target: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        assert target.replay_recipe is not None
        calls.append(target.replay_recipe.n_jobs)
        return real(target, *args, **kwargs)

    monkeypatch.setattr(longitudinal_estimator, "_refit_bound", tracked)
    serial_curve = _payload(serial.diagnostics.truncation_curve([0.2]))
    parallel_curve = _payload(parallel.diagnostics.truncation_curve([0.2]))

    assert calls == [1, 1, 2, 2]
    assert serial_curve == parallel_curve


def test_seeded_stochastic_templates_are_accepted_and_replay_exactly() -> None:
    frame, _ = make_longitudinal(n=90, seed=29)
    forest = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=29)
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": forest,
            "pseudo_learner": forest,
            "n_folds": 1,
        },
    ).fit(frame, **COLUMNS)

    assert replayability(result).refit_nuisances
    result.diagnostics.truncation_curve([result.config.g_bounds])


def test_stable_replay_omissions_cover_legacy_rng_and_unclonable_cases(result) -> None:  # type: ignore[no-untyped-def]
    legacy = replace(result, replay_recipe=None)
    assert replayability(legacy).unreconstructible == (LONGITUDINAL_REPLAY_RECIPE_MISSING,)
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_RECIPE_MISSING):
        legacy.diagnostics.truncation_curve([0.2])

    learner = RandomForestRegressor(n_estimators=2)
    assert _replay_random_state_omissions([learner]) == (LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED,)
    assert _replay_random_state_omissions(
        [replace_random_state(learner, np.random.RandomState(1))]
    ) == (LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER,)

    class Unclonable(BaseEstimator):
        def __sklearn_clone__(self):  # type: ignore[no-untyped-def]
            raise TypeError("deliberately unclonable")

    regimen = resolve_regimens({"always": 1}, 2)[0]
    recipe = _replay_recipe(
        [Plan(regimen, np.ones((3, 2)))],
        Unclonable(),
        LinearRegression(),
        alpha=0.0,
        max_iter=1,
        tol=1e-8,
        random_state=1,
        n_jobs=1,
    )
    assert recipe.omissions == (LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE,)


def test_nested_super_learner_random_states_are_audited() -> None:
    seeded = SuperLearner(
        library=[RandomForestRegressor(n_estimators=2, random_state=7)],
        n_folds=2,
        random_state=7,
    )
    unseeded_member = SuperLearner(
        library=[RandomForestRegressor(n_estimators=2, random_state=None)],
        n_folds=2,
        random_state=7,
    )

    assert _replay_random_state_omissions([seeded]) == ()
    assert _replay_random_state_omissions([unseeded_member]) == (
        LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED,
    )
    regimen = resolve_regimens({"always": 1}, 2)[0]
    recipe = _replay_recipe(
        [Plan(regimen, np.ones((3, 2)))],
        unseeded_member,
        LinearRegression(),
        alpha=0.0,
        max_iter=1,
        tol=1e-8,
        random_state=7,
        n_jobs=1,
    )
    assert recipe.omissions == (LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED,)

    frame, _ = make_longitudinal(n=90, seed=47)
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": unseeded_member,
            "pseudo_learner": unseeded_member,
            "n_folds": 1,
            "random_state": 7,
        },
    ).fit(frame, **COLUMNS)
    assert replayability(result).unreconstructible == (LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED,)
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_RANDOM_STATE_UNSEEDED):
        result.diagnostics.truncation_curve([0.2])


def replace_random_state(learner: RandomForestRegressor, state: Any) -> RandomForestRegressor:
    """Clone the tiny test estimator's constructor settings with another RNG object."""

    return RandomForestRegressor(n_estimators=learner.n_estimators, random_state=state)


def test_combined_interpreter_names_distinct_cumulative_bound_pairs(result) -> None:  # type: ignore[no-untyped-def]
    curve = result.diagnostics.truncation_curve([(0.01, 0.9), (0.01, 1.0)])
    detail = _truncation_item(curve, result).detail
    assert "evaluated cumulative bound pairs [(0.01, 0.9), (0.01, 1)]" in detail


def test_combined_interpreter_preserves_point_lower_bound_wording() -> None:
    report = nw.from_dict(
        {
            "bound": [0.01, 0.2],
            "upper_bound": [0.99, 0.8],
            "estimand": ["ate", "ate"],
            "psi": [1.0, 1.1],
            "delta_from_fitted": [0.0, 0.1],
        },
        backend=__import__("pandas"),
    ).to_native()
    detail = _truncation_item(report, object()).detail
    assert "evaluated lower bounds [0.01, 0.2]" in detail
    assert "cumulative bound pairs" not in detail


def test_longitudinal_grid_is_required_and_mechanism_axis_is_refused(result) -> None:  # type: ignore[no-untyped-def]
    capability = result.diagnostics.capability("truncation_curve")
    assert capability.available
    assert capability.execution == "refit"
    assert capability.cost == "expensive"
    assert capability.requires_arguments == ("bounds",)
    assert capability.requires_replay == "refit_nuisances"
    with pytest.raises(CapabilityError, match="explicit bounds grid"):
        result.diagnostics.truncation_curve()
    with pytest.raises(CapabilityError, match="point-treatment option"):
        result.diagnostics.truncation_curve([0.2], mechanism=True)


def test_combined_report_requires_both_bounds_and_the_refit_opt_in(result) -> None:  # type: ignore[no-untyped-def]
    missing = result.diagnostics.run_all()["truncation_curve"]
    unpaid = result.diagnostics.run_all(arguments={"truncation_curve": {"bounds": [0.2]}})[
        "truncation_curve"
    ]
    wrong_flag = result.diagnostics.run_all(
        include_retargets=True,
        arguments={"truncation_curve": {"bounds": [0.2]}},
    )["truncation_curve"]
    completed = result.diagnostics.run_all(
        include_refits=True,
        arguments={"truncation_curve": {"bounds": [0.2]}},
    )["truncation_curve"]

    assert missing.status is AssessmentStatus.DEFERRED
    assert "bounds" in missing.detail
    assert unpaid.status is AssessmentStatus.DEFERRED
    assert "include_refits=True" in unpaid.detail
    assert wrong_flag.status is AssessmentStatus.DEFERRED
    assert "include_refits=True" in wrong_flag.detail
    assert completed.status is AssessmentStatus.COMPLETED
    assert completed._report is not None


def test_persisted_pre_rm4_combined_unavailable_row_is_not_reused(result) -> None:  # type: ignore[no-untyped-def]
    artifact = pickle.loads(pickle.dumps(result))
    artifact.assessment_cache.clear()
    current = artifact.diagnostics.run_all()
    current_key = next(
        key for key in artifact.assessment_cache if key.startswith("diagnostics.run_all:")
    )
    prefix, encoded = current_key.split(":", 1)
    normalized = json.loads(encoded)
    assert normalized["cache_generation"] == _CACHE_GENERATIONS[prefix] == 8
    normalized["cache_generation"] -= 1
    stale_key = f"{prefix}:{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"
    stale = replace(
        current,
        items=tuple(
            replace(
                item,
                status=AssessmentStatus.UNAVAILABLE,
                detail="pre-RM4 longitudinal truncation was unavailable",
            )
            if item.name == "truncation_curve"
            else item
            for item in current.items
        ),
    )
    artifact.assessment_cache.clear()
    artifact.assessment_cache[stale_key] = _pack_cached(stale, artifact.data.backend)

    restored = pickle.loads(pickle.dumps(artifact))
    refreshed = restored.diagnostics.run_all()

    assert refreshed["truncation_curve"].status is AssessmentStatus.DEFERRED
    assert "bounds" in refreshed["truncation_curve"].detail
    assert "pre-RM4" not in refreshed["truncation_curve"].detail
    assert stale_key in restored.assessment_cache
    assert current_key in restored.assessment_cache
