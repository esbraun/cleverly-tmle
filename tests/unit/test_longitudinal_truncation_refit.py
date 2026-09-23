"""Exact, full-recursion evidence for longitudinal cumulative-bound curves."""

from __future__ import annotations

import inspect
import json
import pickle
from dataclasses import replace
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
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
    _consumed_prefixes,
    _exact_replay_equal,
    _replay_random_state_omissions,
    _replay_recipe,
    _score_cell_truncation_counts,
)
from cleverly.longitudinal.regimen import Plan, resolve_regimens
from cleverly.longitudinal.sequential import Mechanism
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


def _assert_active_curve(result: Any, bound: float = 0.3, cells: Any = None) -> None:
    """Pin movement and every alias's score-cell count to its contributing fits.

    ``cells`` names the ``(regimen, cause, horizon)`` triples each reported name owes, and
    is what makes the count check evidence about the *contributor set*.  Without it both
    sides of the comparison are production helpers: the reported pair is checked against
    what :func:`_score_cell_truncation_counts` returns for the fits the production code
    filed under that name, which no wrong contributor set can fail.  With it the same pair
    is also checked against a census of the cells
    ``docs/technical-reference/validation-methods.md`` says that name reads.
    """

    payload = _payload(result.diagnostics.truncation_curve([bound]))
    recipe = result.replay_recipe
    assert recipe is not None
    replay = longitudinal_estimator._refit_bound(result, recipe, (bound, 1.0))
    estimates, fits, msm_fits = replay.estimates, replay.fits, replay.msm_fits
    consumed = _consumed_prefixes(result.data, result.mechanism, recipe.plans, (bound, 1.0))
    if cells is not None:
        assert set(payload["estimand"]) == set(cells)
    for index, name in enumerate(payload["estimand"]):
        expected = _score_cell_truncation_counts(replay.contributors[name], consumed)
        assert (
            payload["truncated_score_cells"][index],
            payload["evaluated_score_cells"][index],
        ) == expected
        if cells is not None:
            assert expected == _independent_cell_census(result, cells[name], (bound, 1.0), fits)
    assert any(value > 0 for value in payload["truncated_score_cells"])
    assert any(value != 0.0 for value in payload["delta_from_fitted"])

    active = replace(result, estimates=estimates, fits=fits, msm_fits=msm_fits)
    rows = _longitudinal_scores(active, tolerance=recipe.tol).rows
    score_rows = [row for row in rows if row.kind == "solver"]
    assert score_rows
    assert all(row.converged and row.passed for row in score_rows)
    assert all(row.relative_score <= recipe.tol for row in score_rows)

    # A cross-fitted replay solves one pooled fluctuation per node over every follower, as
    # a single-fold one does, so each solver row is the node's own equation and no replay
    # emits a stitching row.  The solver rows are recomputed from the replayed arrays here,
    # so a replay that reported a score its arrays do not pose fails as well.  A working
    # model pools each node's score across its cells, so no single cell's arrays pose it.
    assert [row for row in rows if row.kind == "stitching"] == []
    for fit in fits.values() if result.msm is None else ():
        weights = np.asarray(fit.obs_weights, dtype=float)
        for step in fit.steps:
            assert step.fluctuation.folds == ()
            multiplier = weights * step.clever
            score = float(np.mean(multiplier * (step.pseudo_outcome - step.targeted)))
            assert abs(score) <= recipe.tol * float(np.mean(np.abs(multiplier)))


def _independent_cell_census(
    result: Any, cells: Any, bound: tuple[float, float], fits: Any
) -> tuple[int, int]:
    """Census the truncated and evaluated cells without any production counting helper.

    Rebuilt here from the stored out-of-fold mechanism factors, by ``np.cumprod`` and
    ``np.clip`` rather than through the production product, and from each step's own
    ``trained_on`` mask.  A count read off the wrong mechanism, or off ``K`` fold slabs as
    the fold-fluctuated construction did, fails this.

    ``cells`` is a sequence of ``(regimen, cause, horizon)`` triples, which is the index a
    fit carries.  A reported name owes the cells its caller writes out rather than the ones
    the production contributor map files under it: a level owes its own cell, a contrast its
    two, and an MSM coefficient every regimen and horizon cell for its cause.  Each triple
    selects exactly one fit, which is asserted rather than assumed.
    """

    truncated = 0
    evaluated = 0
    for label, cause, horizon in cells:
        contributing = [
            fit
            for fit in fits.values()
            if (fit.regimen.label, fit.cause, fit.horizon) == (label, cause, horizon)
        ]
        assert len(contributing) == 1
        unbounded, bounded = _longhand_cumulative(result, label, bound)
        for step in contributing[0].steps:
            rows = step.trained_on
            column = step.time - 1
            truncated += int(np.count_nonzero(unbounded[rows, column] != bounded[rows, column]))
            evaluated += int(np.count_nonzero(rows))
    return truncated, evaluated


def _longhand_cumulative(
    result: Any, label: str, bound: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The raw and bounded out-of-fold cumulative mechanism, from the stored factors."""

    data = result.data
    factors = []
    for time in range(data.n_times):
        factors.append(result.mechanism.treatment[time][label])
        factors.append(
            result.mechanism.censoring[time][label] if data.censoring_names else np.ones(data.n)
        )
    raw = np.cumprod(np.column_stack(factors), axis=1)[:, 1::2]
    return raw, np.clip(raw, *bound)


def _clip_directions(result: Any, bound: tuple[float, float]) -> tuple[int, int]:
    """How many out-of-fold cumulative cells each endpoint of ``bound`` moved.

    A raised cell sat below the lower endpoint and a lowered one above the upper endpoint,
    so the two counts say which endpoints of a pair actually bind on this sample.  A census
    whose truncated counts come only from one of them is no witness for the other.
    """

    raised = 0
    lowered = 0
    for plan in result.replay_recipe.plans:
        unbounded, bounded = _longhand_cumulative(result, plan.label, bound)
        raised += int(np.count_nonzero(unbounded < bounded))
        lowered += int(np.count_nonzero(unbounded > bounded))
    return raised, lowered


class Unclonable(BaseEstimator):
    """An estimator whose ``fit`` and ``predict`` work and whose clone raises.

    Used twice: by the injected-recipe witness, and by the real fit that reaches the same
    omission through a learner slot the recursion never fits.
    """

    def __sklearn_clone__(self) -> Unclonable:
        raise TypeError("deliberately unclonable")

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Unclonable:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> Any:
        return np.full(len(X), self.mean_)


def _one_node_panel(n: int = 200, seed: int = 5) -> Any:
    """One treatment node, one censoring node, and an end-of-study outcome.

    A single node is also the horizon, so :func:`~cleverly.longitudinal.sequential.prepare_node`
    takes ``outcome_learner`` there and no pseudo-outcome regression ever runs.  ``C1 == 1``
    is still under observation and ``C1 == 0`` is censored, which is why ``Y`` is missing
    there.  About a tenth are censored, so the censoring regression sees both classes.
    """

    rng = np.random.default_rng(seed)
    baseline_one = rng.normal(size=n)
    baseline_two = rng.normal(size=n)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * baseline_one))).astype(float)
    observed = (rng.random(n) > 0.1).astype(float)
    outcome = 0.5 * baseline_one + 0.3 * treatment + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {
            "W1": baseline_one,
            "W2": baseline_two,
            "A1": treatment,
            "C1": observed,
            "Y": np.where(observed == 1.0, outcome, np.nan),
        }
    )


#: ``{"outcome": ..., ...}`` for :func:`_one_node_panel`.
ONE_NODE_COLUMNS = {
    "outcome": "Y",
    "treatment": ("A1",),
    "baseline": ("W1", "W2"),
    "time_varying": ((),),
    "censoring": ("C1",),
}


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    frame, _ = make_longitudinal(n=140, seed=17)
    return LTMLE({"always": 1, "never": 0}, **SETTINGS).fit(frame, **COLUMNS)


@pytest.fixture(scope="module")
def single_fold_result():  # type: ignore[no-untyped-def]
    """The same sample and settings at one outer fold, so the slab census has a control."""

    frame, _ = make_longitudinal(n=140, seed=17)
    return LTMLE({"always": 1, "never": 0}, **{**SETTINGS, "n_folds": 1}).fit(frame, **COLUMNS)


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
    # The fold draw is unseeded, so the sample must keep both classes of every binary
    # target in each training fold. A fold that holds all of one class fails the fit. At
    # n=140 that happens on 0.7% of draws, mostly through the 8 censorings at time 2. At
    # n=600 the exact hypergeometric union bound over the six targets is below 5e-11.
    frame, _ = make_longitudinal(n=600, seed=17)
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


def test_grid_order_is_first_appearance_and_not_sorted_order(result) -> None:  # type: ignore[no-untyped-def]
    """A requested grid whose order is not its sorted order, which every other grid here is.

    Deduplicating the resolved pairs has to keep the order they were first asked in.  Sorting
    the distinct pairs instead leaves every column of every row correct and moves the rows, so
    only a request that is already out of order can fail it.  ``0.3`` is also asked twice, so
    the one pair it denotes is still reported once.
    """

    requested = [0.3, 0.1, 0.3]
    assert sorted(set(requested)) != list(dict.fromkeys(requested))
    payload = _payload(result.diagnostics.truncation_curve(requested))
    names = list(result.estimates)

    assert payload["estimand"] == names * 2
    assert list(zip(payload["lower_bound"], payload["upper_bound"], strict=True)) == [
        (0.3, 1.0)
    ] * len(names) + [(0.1, 1.0)] * len(names)


def test_fitted_bound_reproduces_every_retained_fit_artifact_exactly(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    replay = longitudinal_estimator._refit_bound(result, recipe, result.config.g_bounds)
    estimates, fits, msm_fits = replay.estimates, replay.fits, replay.msm_fits

    assert _exact_replay_equal(estimates, result.estimates)
    assert _exact_replay_equal(fits, result.fits)
    assert _exact_replay_equal(msm_fits, result.msm_fits)
    curve = _payload(result.diagnostics.truncation_curve([result.config.g_bounds]))
    assert all(curve["is_fitted_bound"])
    assert all(value == 0.0 for value in curve["delta_from_fitted"])
    assert curve["psi"] == curve["fitted_psi"]


def test_delta_from_fitted_points_from_the_fitted_estimate_to_the_replayed_one(result) -> None:  # type: ignore[no-untyped-def]
    """A rising estimate reports a positive delta, so the subtraction cannot be reversed."""

    payload = _payload(result.diagnostics.truncation_curve([0.3]))
    row = payload["estimand"].index("ey_regimen[always]")

    # Pinned rather than recomputed: ``estimate - fitted`` and ``fitted - estimate`` agree on
    # the magnitude and differ only here, so only the direction and the literal are evidence.
    # The literals are the pooled cross-fitted construction's.  An independent longhand of
    # it -- weighted least-squares fold recursions, a hand Newton solve per node against the
    # out-of-fold mechanism by ``np.cumprod`` -- gives 0.5630412924858 and 0.7403837710467,
    # within 4e-13 of the fit.  The node-2 ``epsilon`` is -1.70 on these 140 rows.
    assert payload["fitted_psi"][row] == pytest.approx(0.56304129, rel=1e-6)
    assert payload["psi"][row] == pytest.approx(0.74038377, rel=1e-6)
    assert payload["psi"][row] > payload["fitted_psi"][row]
    assert payload["delta_from_fitted"][row] > 0.0
    assert payload["delta_from_fitted"][row] == pytest.approx(0.17734248, rel=1e-6)


def test_delta_from_fitted_is_negative_where_the_replayed_estimate_falls(  # type: ignore[no-untyped-def]
    single_fold_result,
) -> None:
    """The companion direction, so no absolute value or reversal passes both witnesses."""

    payload = _payload(single_fold_result.diagnostics.truncation_curve([0.12]))
    row = payload["estimand"].index("ey_regimen[never]")

    assert payload["psi"][row] < payload["fitted_psi"][row]
    assert payload["delta_from_fitted"][row] < 0.0
    assert payload["delta_from_fitted"][row] == pytest.approx(-0.00404959, rel=1e-6)


def test_fitted_bound_reference_columns_restate_the_fits_own_cumulative_pair(result) -> None:  # type: ignore[no-untyped-def]
    """The reference columns carry ``config.g_bounds``, and the flag names that pair only."""

    fitted = result.config.g_bounds
    assert fitted == (0.01, 1.0)
    # ``(0.01, 0.9)`` shares the fitted lower bound and not the pair, so a flag read off the
    # lower bound alone reports it as the fitted row.
    grid = [(0.01, 0.9), 0.12, fitted]
    payload = _payload(result.diagnostics.truncation_curve(grid))
    pairs = list(zip(payload["lower_bound"], payload["upper_bound"], strict=True))

    assert set(payload["fitted_lower_bound"]) == {fitted[0]}
    assert set(payload["fitted_upper_bound"]) == {fitted[1]}
    assert [pair for pair, flag in zip(pairs, payload["is_fitted_bound"], strict=True) if flag] == [
        fitted
    ] * 3
    assert sum(payload["is_fitted_bound"]) == 3
    assert len(payload["is_fitted_bound"]) == 9
    for pair, fitted_psi, psi, flag in zip(
        pairs, payload["fitted_psi"], payload["psi"], payload["is_fitted_bound"], strict=True
    ):
        assert flag == (pair == fitted)
        # Only one direction: a pair that is not the fitted one can still leave the estimate
        # where it was, because the upper bound ``0.9`` clips no cell on this sample.
        if flag:
            assert psi == fitted_psi


def test_fitted_bound_gate_refuses_a_nonpoint_artifact_mismatch(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = longitudinal_estimator._refit_bound

    def mutated(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        replay = real(*args, **kwargs)
        key = next(iter(replay.fits))
        fit = replay.fits[key]
        changed = fit.cumulative.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
        return replace(replay, fits={**replay.fits, key: replace(fit, cumulative=changed)})

    monkeypatch.setattr(longitudinal_estimator, "_refit_bound", mutated)
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_FITTED_BOUND_MISMATCH):
        longitudinal_estimator.longitudinal_truncation_curve(result, [0.25])


def test_crossfit_active_replay_rejects_a_fold_slab_mutation(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A replay that divided by fold 0's slab instead of the out-of-fold pair is caught."""
    recipe = result.replay_recipe
    assert recipe is not None
    active = (0.25, 1.0)
    correct = longitudinal_estimator._refit_bound(result, recipe, active).fits
    real = Mechanism.cumulative_with_unbounded

    def slab(self: Mechanism, data: Any, plan: Any, bounds: Any, *, fold: int | None = None):  # type: ignore[no-untyped-def]
        return real(
            self,
            data,
            plan,
            bounds,
            fold=0 if tuple(bounds) == active and fold is None else fold,
        )

    monkeypatch.setattr(Mechanism, "cumulative_with_unbounded", slab)
    mutated = longitudinal_estimator._refit_bound(result, recipe, active).fits
    with pytest.raises(AssertionError):
        assert _exact_replay_equal(mutated, correct)


def test_changed_bound_rebuilds_earlier_pseudo_outcomes_and_moves_estimates(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    replay = longitudinal_estimator._refit_bound(result, recipe, (0.25, 1.0))
    estimates, fits = replay.estimates, replay.fits

    assert any(estimates[name].psi != result.estimates[name].psi for name in estimates)
    for key, fit in fits.items():
        original = result.fits[key]
        assert fit.steps[0].pseudo_outcome.shape == original.steps[0].pseudo_outcome.shape
        assert np.any(fit.steps[0].pseudo_outcome != original.steps[0].pseudo_outcome)


def test_active_replay_exactly_matches_an_independent_top_level_fit(result) -> None:  # type: ignore[no-untyped-def]
    """A replay at a bound equals a fit that was *configured* at that bound, bit for bit.

    The comparison is against the public entry point rather than against the arguments
    ``_refit_bound`` hands the recursion, so a shared mistake in those arguments cannot
    satisfy both sides.  It also pins the premise the replay rests on: the mechanism and the
    outer folds do not depend on ``g_bounds``, so reusing the fitted ones is exact.
    """

    frame, _ = make_longitudinal(n=140, seed=17)
    active = (0.25, 1.0)
    fresh = LTMLE({"always": 1, "never": 0}, **SETTINGS, g_bounds=active).fit(frame, **COLUMNS)
    recipe = result.replay_recipe
    assert recipe is not None
    replayed = longitudinal_estimator._refit_bound(result, recipe, active)

    assert fresh.config.g_bounds == active
    assert result.config.g_bounds != active
    assert _exact_replay_equal(result.mechanism, fresh.mechanism)
    assert _exact_replay_equal(result.folds, fresh.folds)
    assert _exact_replay_equal(replayed.estimates, fresh.estimates)
    assert _exact_replay_equal(replayed.fits, fresh.fits)
    assert _exact_replay_equal(replayed.msm_fits, fresh.msm_fits)
    # A float comparison as well, because ``_exact_replay_equal`` would also accept two
    # dictionaries that agreed by both being empty.
    assert [replayed.estimates[name].psi for name in fresh.estimates] == [
        fresh.estimates[name].psi for name in fresh.estimates
    ]
    assert len(fresh.estimates) == 3


def test_end_of_study_crossfit_active_replay_solves_every_node_and_stitched_score(result) -> None:  # type: ignore[no-untyped-def]
    """The score witness for the end-of-study target, on the only cross-fitted fixture."""

    assert result.folds.n_folds == 2
    assert result.msm is None
    assert not result.data.is_survival
    _assert_active_curve(result, cells=CONTRIBUTING_CELLS)


def test_reusing_one_earlier_outcome_prediction_breaks_active_replay(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An active replay must rebuild node 1's pooled outcome from node 2's new targeted fit.

    Under the pooled construction the untargeted fold recursions read no mechanism, so node
    1's ``initial`` is the same at every bound and reusing it is harmless: the pooled pass
    reads it as an offset.  The bound-dependent array is the pooled outcome, the targeted
    prediction of node 2.  The mutation hands node 1's fluctuation the fitted bound's stale
    pooled outcome, and the replayed estimate has to move.
    """
    correct = _payload(longitudinal_estimator.longitudinal_truncation_curve(result, [0.25]))
    real = longitudinal_sequential._fluctuate_node
    replaying = {"active": False}
    real_refit = longitudinal_estimator._refit_bound

    def tracked(target: Any, recipe: Any, bounds: Any):  # type: ignore[no-untyped-def]
        replaying["active"] = tuple(bounds) != result.config.g_bounds
        return real_refit(target, recipe, bounds)

    def reused(pseudo_outcome: Any, initial: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if replaying["active"] and kwargs["time"] == 1:
            stale = result.fits[kwargs["label"]].steps[0].pseudo_outcome
            assert not np.array_equal(stale, pseudo_outcome)
            pseudo_outcome = stale
        return real(pseudo_outcome, initial, *args, **kwargs)

    monkeypatch.setattr(longitudinal_estimator, "_refit_bound", tracked)
    monkeypatch.setattr(longitudinal_sequential, "_fluctuate_node", reused)
    mutated = _payload(longitudinal_estimator.longitudinal_truncation_curve(result, [0.25]))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(mutated["psi"], correct["psi"])


def test_score_cell_counts_are_alias_specific_and_use_only_scored_rows(result) -> None:  # type: ignore[no-untyped-def]
    recipe = result.replay_recipe
    assert recipe is not None
    replay = longitudinal_estimator._refit_bound(result, recipe, (0.25, 1.0))
    curve = _payload(longitudinal_estimator.longitudinal_truncation_curve(result, [0.25]))
    consumed = _consumed_prefixes(result.data, result.mechanism, recipe.plans, (0.25, 1.0))

    for index, name in enumerate(curve["estimand"]):
        expected = _score_cell_truncation_counts(replay.contributors[name], consumed)
        assert (
            curve["truncated_score_cells"][index],
            curve["evaluated_score_cells"][index],
        ) == expected
    level_counts = curve["evaluated_score_cells"][:2]
    contrast_count = curve["evaluated_score_cells"][2]
    assert contrast_count == sum(level_counts)


#: ``(regimen, cause, horizon)`` behind each reported name on the two-arm end-of-study
#: fixture.  Written out rather than read from the production contributor map, because the
#: composition of the contrast is part of what the census below is evidence for.
CONTRIBUTING_CELLS = {
    "ey_regimen[always]": (("always", None, 2),),
    "ey_regimen[never]": (("never", None, 2),),
    "ate_regimen[always vs never]": (("always", None, 2), ("never", None, 2)),
}

#: The same rule on the survival fixture, where one regimen has one cell per horizon.  A
#: horizon-1 fit runs one node and a horizon-2 fit runs two, so a name that read both
#: horizons would report a denominator this census does not allow.
SURVIVAL_CONTRIBUTING_CELLS = {
    "risk_regimen[always @ t=1]": (("always", None, 1),),
    "risk_regimen[always @ t=2]": (("always", None, 2),),
}

#: The same rule on the competing-risk fixture, where the cell is indexed by cause as well.
#: The two causes share a regimen, a plan and a risk set, so a name that read both would
#: double its own counts rather than report a visibly different mechanism.
COMPETING_CONTRIBUTING_CELLS = {
    "cif_regimen[always, relapse @ t=1]": (("always", "relapse", 1),),
    "cif_regimen[always, relapse @ t=2]": (("always", "relapse", 2),),
    "cif_regimen[always, death @ t=1]": (("always", "death", 1),),
    "cif_regimen[always, death @ t=2]": (("always", "death", 2),),
}

#: Every regimen and horizon cell of the coefficient's cause, which on this fixture is all
#: three regimens at the one horizon.  Written once per reported name rather than shared,
#: because "the same cells for every term" is the claim: the projection is pooled, so no
#: coefficient is a statement about one regimen.
MSM_CONTRIBUTING_CELLS = {
    "msm_regimen[intercept]": (("always", None, 2), ("never", None, 2), ("late", None, 2)),
    "msm_regimen[duration]": (("always", None, 2), ("never", None, 2), ("late", None, 2)),
}

#: ``(pair, estimand)`` to the ``(truncated, evaluated)`` cell counts a two-fold fit owes.
#: A cross-fitted fit solves one pooled fluctuation per node with the out-of-fold mechanism,
#: so its denominators are the single-fold ones below: one pass over each node's followers.
#: The numerators count truncations in the out-of-fold pair, which differs from the
#: single-fold fit's in-sample mechanism, so they differ from the single-fold numerators.
#: Every value was recounted by :func:`_independent_cell_census`, which rebuilds the pair
#: with ``np.cumprod`` over the stored factors, and agrees with the production count.
#: The fold-fluctuated construction's census counted ``K`` fold slabs and reported twice
#: these denominators.
#:
#: Every pair here keeps the upper endpoint at ``1.0``, so every truncated cell was *raised*.
#: ``UPPER_BOUND_CELL_CENSUS`` and ``TWO_SIDED_CELL_CENSUS`` carry the other direction.
CROSSFIT_CELL_CENSUS = {
    ((0.01, 1.0), "ey_regimen[always]"): (0, 112),
    ((0.01, 1.0), "ey_regimen[never]"): (0, 80),
    ((0.01, 1.0), "ate_regimen[always vs never]"): (0, 192),
    ((0.12, 1.0), "ey_regimen[always]"): (3, 112),
    ((0.12, 1.0), "ey_regimen[never]"): (8, 80),
    ((0.12, 1.0), "ate_regimen[always vs never]"): (11, 192),
    ((0.3, 1.0), "ey_regimen[always]"): (22, 112),
    ((0.3, 1.0), "ey_regimen[never]"): (35, 80),
    ((0.3, 1.0), "ate_regimen[always vs never]"): (57, 192),
}

#: The same census at one outer fold, where the one mechanism is the in-sample fit.
SINGLE_FOLD_CELL_CENSUS = {
    ((0.12, 1.0), "ey_regimen[always]"): (1, 112),
    ((0.12, 1.0), "ey_regimen[never]"): (3, 80),
    ((0.12, 1.0), "ate_regimen[always vs never]"): (4, 192),
    ((0.3, 1.0), "ey_regimen[always]"): (17, 112),
    ((0.3, 1.0), "ey_regimen[never]"): (28, 80),
    ((0.3, 1.0), "ate_regimen[always vs never]"): (45, 192),
}

#: The two-fold census of a pair whose *upper* endpoint is the one that binds.  ``0.01`` is
#: the fitted lower endpoint and raises nothing on this sample, so all ``46`` of the
#: contrast's truncated cells were lowered to ``0.5``.
UPPER_BOUND_CELL_CENSUS = {
    ((0.01, 0.5), "ey_regimen[always]"): (43, 112),
    ((0.01, 0.5), "ey_regimen[never]"): (3, 80),
    ((0.01, 0.5), "ate_regimen[always vs never]"): (46, 192),
}

#: The two-fold census of a pair whose endpoints both bind.  ``25`` of the contrast's ``51``
#: truncated cells were raised to ``0.2`` and the other ``26`` were lowered to ``0.6``.  No
#: cell is both, because a prefix cannot sit below ``0.2`` and above ``0.6`` at once.
TWO_SIDED_CELL_CENSUS = {
    ((0.2, 0.6), "ey_regimen[always]"): (36, 112),
    ((0.2, 0.6), "ey_regimen[never]"): (15, 80),
    ((0.2, 0.6), "ate_regimen[always vs never]"): (51, 192),
}


def _assert_cell_census(result: Any, census: Any) -> None:
    """Check the reported counts against pinned values and an independent recount."""

    pairs = sorted({pair for pair, _ in census})
    payload = _payload(result.diagnostics.truncation_curve(pairs))
    reported = {
        ((payload["lower_bound"][index], payload["upper_bound"][index]), name): (
            payload["truncated_score_cells"][index],
            payload["evaluated_score_cells"][index],
        )
        for index, name in enumerate(payload["estimand"])
    }
    assert reported == census

    recipe = result.replay_recipe
    assert recipe is not None
    for pair in pairs:
        fits = longitudinal_estimator._refit_bound(result, recipe, pair).fits
        for name, cells in CONTRIBUTING_CELLS.items():
            if (pair, name) not in census:
                continue
            assert _independent_cell_census(result, cells, pair, fits) == census[(pair, name)]


def test_crossfit_score_cell_counts_census_the_out_of_fold_mechanism(result) -> None:  # type: ignore[no-untyped-def]
    """Two folds, and one pass over the followers with the out-of-fold mechanism.

    The fold recursions read no mechanism, and the pooled fluctuation reads the out-of-fold
    pair once per node.  So the denominators equal the single-fold fixture's, and a census
    of the ``K`` fold slabs would report twice them.  The witness that the slabs would also
    count differently is that they differ from the out-of-fold pair where a fold trained.
    """

    assert result.folds.n_folds == 2
    _assert_cell_census(result, CROSSFIT_CELL_CENSUS)
    assert {
        name: counts[1] for (pair, name), counts in CROSSFIT_CELL_CENSUS.items() if pair[0] == 0.12
    } == {name: counts[1] for (_, name), counts in SINGLE_FOLD_CELL_CENSUS.items()}
    moved = 0.0
    for plan in result.replay_recipe.plans:
        _, pair = result.mechanism.cumulative_with_unbounded(result.data, plan, (0.12, 1.0))
        for fold, (train, _) in enumerate(result.folds):
            _, slab = result.mechanism.cumulative_with_unbounded(
                result.data, plan, (0.12, 1.0), fold=fold
            )
            moved = max(moved, float(np.max(np.abs(slab[train] - pair[train]))))
    assert moved > 1e-6


def test_single_fold_score_cell_counts_census_the_one_consumed_slab(single_fold_result) -> None:  # type: ignore[no-untyped-def]
    """One fold, so the denominators are a single pass and the contrast is still additive."""

    assert single_fold_result.folds.n_folds == 1
    _assert_cell_census(single_fold_result, SINGLE_FOLD_CELL_CENSUS)


def test_upper_bound_truncation_is_counted_on_a_pair_that_binds_only_above(result) -> None:  # type: ignore[no-untyped-def]
    """A truncated cell can be one the bound *lowered*, which no census above can show.

    Every pinned pair above keeps the upper endpoint at ``1.0``, and the one pair in this
    module with a real upper endpoint, ``(0.01, 0.9)``, moves no cell on this sample.  So a
    comparison that only reported a raised prefix satisfied all of them.  Here the lower
    endpoint is the fitted one and raises nothing, and the counts are entirely the cells the
    upper endpoint lowered.
    """

    assert result.folds.n_folds == 2
    assert result.config.g_bounds == (0.01, 1.0)
    raised, lowered = _clip_directions(result, (0.01, 0.5))
    assert raised == 0
    assert lowered > 0
    _assert_cell_census(result, UPPER_BOUND_CELL_CENSUS)


def test_both_endpoints_binding_counts_the_cells_each_one_moved(result) -> None:  # type: ignore[no-untyped-def]
    """One pair whose two endpoints both bind, so neither one-sided comparison passes it.

    ``(0.2, 0.6)`` raises some cumulative prefixes and lowers others on this sample.  A
    count of either direction alone is short of the pinned total by the other direction's
    share, so this row fails whichever half a comparison drops.
    """

    assert result.folds.n_folds == 2
    raised, lowered = _clip_directions(result, (0.2, 0.6))
    assert raised > 0
    assert lowered > 0
    _assert_cell_census(result, TWO_SIDED_CELL_CENSUS)


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
    _assert_active_curve(result, cells=SURVIVAL_CONTRIBUTING_CELLS)


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
    replay = longitudinal_estimator._refit_bound(result, recipe, result.config.g_bounds)
    estimates, fits, msm_fits = replay.estimates, replay.fits, replay.msm_fits

    assert _exact_replay_equal(estimates, result.estimates)
    assert _exact_replay_equal(fits, result.fits)
    assert _exact_replay_equal(msm_fits, result.msm_fits)
    payload = _payload(result.diagnostics.truncation_curve([0.2]))
    assert payload["estimand"] == list(result.estimates)
    assert all(name.startswith("msm_regimen[") for name in payload["estimand"])
    _assert_active_curve(result, cells=MSM_CONTRIBUTING_CELLS)


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
    _assert_active_curve(result, cells=COMPETING_CONTRIBUTING_CELLS)


def test_crossfit_survival_curve_replays_every_horizon_on_three_folds() -> None:
    """Cross-fitting crossed with the survival target, which the single-fold fixture omits."""

    frame, _ = make_longitudinal_survival(n=180, seed=31)
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": LogisticRegression(max_iter=1000, random_state=31),
            "n_folds": 3,
        },
    ).fit(
        frame,
        outcome=("Y1", "Y2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )
    payload = _payload(result.diagnostics.truncation_curve([result.config.g_bounds]))

    assert result.folds.n_folds == 3
    assert payload["estimand"] == list(result.estimates)
    assert len(set(payload["estimand"])) == len(result.config.horizons) == 2
    assert all(payload["is_fitted_bound"])
    # Three folds, so each reported horizon is three untargeted fold recursions and one
    # pooled fluctuation per node, whose score is recomputed from the replayed arrays.
    _assert_active_curve(result, cells=SURVIVAL_CONTRIBUTING_CELLS)


def test_crossfit_competing_risk_curve_replays_every_cause_on_two_folds() -> None:
    """Cross-fitting crossed with competing risks, which the single-fold fixture omits."""

    frame, _ = make_longitudinal_competing(n=400, seed=7)
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": LogisticRegression(max_iter=1000, random_state=7),
            "n_folds": 2,
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

    assert result.folds.n_folds == 2
    assert payload["estimand"] == list(result.estimates)
    assert len(payload["estimand"]) == 4
    assert all("relapse" in name or "death" in name for name in payload["estimand"])
    _assert_active_curve(result, cells=COMPETING_CONTRIBUTING_CELLS)


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
    payload = _payload(result.diagnostics.truncation_curve([result.config.g_bounds, 0.3]))
    names = [
        "ey_regimen[never]",
        "ey_regimen[dynamic]",
        "ate_regimen[dynamic vs never]",
    ]

    # A categorical arm names the level it assigns, and a dynamic arm names the rule's label.
    # The curve reports those same keys, so a reader can join a row back to the parameter.
    assert payload["estimand"] == names * 2
    assert list(result.estimates) == names
    fitted = payload["delta_from_fitted"][:3]
    active = payload["delta_from_fitted"][3:]
    assert all(value == 0.0 for value in fitted)
    assert all(value != 0.0 for value in active)
    assert payload["truncated_score_cells"][:3] == [0, 0, 0]
    assert all(value > 0 for value in payload["truncated_score_cells"][3:])
    _assert_active_curve(result)


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

    regimen = resolve_regimens({"always": 1}, 2)[0]
    recipe = _replay_recipe(
        [Plan(regimen, np.ones((3, 2)))],
        Unclonable(),
        LinearRegression(),
        alpha=0.0,
        max_iter=1,
        tol=1e-8,
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


def test_empty_grid_and_unknown_estimand_are_capability_refusals(result) -> None:  # type: ignore[no-untyped-def]
    """Both are caller-input refusals, which one report row can carry without aborting."""

    with pytest.raises(CapabilityError, match="the requested grid is empty"):
        result.diagnostics.truncation_curve([])
    with pytest.raises(CapabilityError, match=r"cannot report the requested estimands"):
        result.diagnostics.truncation_curve([0.2], estimands=["ey_regimen[sometimes]"])
    # The refusal names what this fit does report, so the caller can correct the request.
    with pytest.raises(CapabilityError, match=r"ey_regimen\[always\]"):
        result.diagnostics.truncation_curve([0.2], estimands=["ey_regimen[sometimes]"])


def test_an_empty_grid_costs_one_combined_row_and_not_the_battery(result) -> None:  # type: ignore[no-untyped-def]
    """A bare exception here aborted every later row; a refusal marks this row only."""

    paid = {"include_refits": True, "arguments": {"truncation_curve": {"bounds": [0.2]}}}
    complete = result.diagnostics.run_all(**paid)
    empty = result.diagnostics.run_all(
        include_refits=True, arguments={"truncation_curve": {"bounds": []}}
    )

    assert empty["truncation_curve"].status is AssessmentStatus.UNAVAILABLE
    assert "the requested grid is empty" in empty["truncation_curve"].detail
    assert [item.name for item in empty.items] == [item.name for item in complete.items]
    assert [item.status for item in empty.items if item.name != "truncation_curve"] == [
        item.status for item in complete.items if item.name != "truncation_curve"
    ]
    assert complete["truncation_curve"].status is AssessmentStatus.COMPLETED


def test_duplicate_grid_entries_replay_and_report_one_bound_pair(result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``0.2`` and ``(0.2, 1.0)`` are one pair, and a replay is minutes of work."""

    calls: list[tuple[float, float]] = []
    real = longitudinal_estimator._refit_bound

    def tracked(target: Any, recipe: Any, bounds: Any):  # type: ignore[no-untyped-def]
        calls.append(tuple(bounds))
        return real(target, recipe, bounds)

    monkeypatch.setattr(longitudinal_estimator, "_refit_bound", tracked)
    payload = _payload(
        longitudinal_estimator.longitudinal_truncation_curve(result, [0.2, 0.2, (0.2, 1.0)])
    )

    # One proof of the fitted bound, then one replay of the single distinct requested pair.
    assert calls == [result.config.g_bounds, (0.2, 1.0)]
    assert (
        list(zip(payload["lower_bound"], payload["upper_bound"], strict=True)) == [(0.2, 1.0)] * 3
    )
    assert payload["estimand"] == list(result.estimates)


def test_object_seeded_learner_refuses_the_curve_end_to_end() -> None:
    """A generator object survives a real fit, so this refusal is reachable and reached."""

    frame, _ = make_longitudinal(n=90, seed=29)
    forest = RandomForestRegressor(
        n_estimators=3, max_depth=2, random_state=np.random.RandomState(1)
    )
    result = LTMLE(
        {"always": 1},
        **{
            **SETTINGS,
            "reference": "always",
            "outcome_learner": forest,
            "n_folds": 1,
        },
    ).fit(frame, **COLUMNS)

    assert result.replay_recipe is not None
    assert result.replay_recipe.omissions == (LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER,)
    assert replayability(result).unreconstructible == (
        LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER,
    )
    assert not replayability(result).refit_nuisances
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_RANDOM_STATE_NON_INTEGER):
        result.diagnostics.truncation_curve([0.2])
    row = result.diagnostics.run_all(
        include_refits=True, arguments={"truncation_curve": {"bounds": [0.2]}}
    )["truncation_curve"]
    assert row.status is AssessmentStatus.UNAVAILABLE
    assert "integer seed" in row.detail


def test_an_unused_unclonable_pseudo_learner_fits_and_then_refuses_the_curve() -> None:
    """The whole omission chain, from a fit that really ran to the refusal it causes.

    ``pseudo_learner`` is resolved and recorded whether or not the recursion ever fits it,
    and on a one-node panel it never does: the only node is the horizon, so that node takes
    ``outcome_learner``.  A learner that refuses to be cloned therefore reaches the recipe
    without reaching a nuisance regression, the fit succeeds, and the replay is
    unreconstructible.  Every other witness for this omission injects it into a recipe.
    """

    result = LTMLE(
        {"always": 1},
        reference="always",
        outcome_learner=LinearRegression(),
        pseudo_learner=Unclonable(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=1,
        learner_folds=2,
        random_state=11,
        simultaneous=False,
    ).fit(_one_node_panel(), **ONE_NODE_COLUMNS)

    # The premise: one node, which is the horizon, so no pseudo-outcome regression ran.
    assert result.config.n_times == 1
    assert [step.time for step in result.fits["always"].steps] == [1]
    assert list(result.estimates) == ["ey_regimen[always]"]

    assert result.replay_recipe is not None
    assert result.replay_recipe.outcome_learner is not None
    assert result.replay_recipe.pseudo_learner is None
    assert result.replay_recipe.omissions == (LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE,)
    assert replayability(result).unreconstructible == (LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE,)
    assert not replayability(result).refit_nuisances
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE):
        result.diagnostics.truncation_curve([0.2])


def test_an_unclonable_learner_omission_refuses_the_curve_on_a_real_result(result) -> None:  # type: ignore[no-untyped-def]
    """The same refusal on the cross-fitted fixture the rest of this module uses.

    The omission is injected here rather than produced, so that the refusal is checked on a
    result with two regimens and two folds.  It is reachable from a real fit as well, which
    ``test_an_unused_unclonable_pseudo_learner_fits_and_then_refuses_the_curve`` shows.
    """

    recipe = result.replay_recipe
    assert recipe is not None
    blocked = replace(
        result,
        replay_recipe=replace(
            recipe,
            outcome_learner=None,
            omissions=(LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE,),
        ),
    )

    assert replayability(blocked).unreconstructible == (LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE,)
    assert not replayability(blocked).refit_nuisances
    with pytest.raises(CapabilityError, match=LONGITUDINAL_REPLAY_LEARNER_UNCLONABLE):
        blocked.diagnostics.truncation_curve([0.2])


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
    # ``result`` is module scoped and every earlier test's report is filed on it, so the
    # clone starts with a populated cache.  Clearing the clone's own dictionary cannot reach
    # the fixture's: ``assessment_cache`` is ``field(init=False, default_factory=dict)``, so
    # both ``replace`` and an unpickle build a fresh one.
    artifact.assessment_cache.clear()
    assert artifact.assessment_cache is not result.assessment_cache
    current = artifact.diagnostics.run_all()
    combined = [key for key in artifact.assessment_cache if key.startswith("diagnostics.run_all:")]
    # One, and said so rather than taken off the front: the key below is rebuilt by editing
    # this one, and a second combined key here would make that edit describe the wrong call.
    assert len(combined) == 1
    current_key = combined[0]
    prefix, encoded = current_key.split(":", 1)
    normalized = json.loads(encoded)
    # Read from the table rather than pinned to a literal.  Every result-changing fix to a
    # reported column bumps this generation, so a literal here fails on the bump it is meant
    # to accompany.  ``tests/unit/test_assessment_contract.py`` derives its seed the same way.
    assert normalized["cache_generation"] == _CACHE_GENERATIONS[prefix]
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
