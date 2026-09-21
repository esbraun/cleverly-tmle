"""Cross-fitted longitudinal TMLE runs untargeted folds, then one pooled epsilon per node.

Díaz, Williams, Hoffman and Schenck (2023, *JASA* 118(542), Section 5.2, Steps 1-4) run an
untargeted sequential regression in every outer training fold, stitch the held-out
predictions, and then solve one fluctuation per node over every follower with the
out-of-fold mechanism.  The proof of their Theorem 3 holds each fold's regressions fixed
given its training rows, which covers that construction.  It does not cover the one the
package shipped before: a training-fold fluctuation whose targeted prediction is carried
into the fold's next regression.  ``lmtp`` 1.5.4 has that shape, with out-of-fold density
ratios on the training rows.  Mutation M3 below is the previous ``cleverly`` longhand,
which read fold ``k``'s own mechanism slab instead.

Every check here reads a fit whose outcome learners are deliberately misspecified, so each
node's ``epsilon`` is far from zero.  An exact-law check is blind to a term that vanishes at
the truth, and a fluctuation that barely moves cannot tell a correct score from a wrong one.
:func:`_assert_epsilon_witness` states that premise on every fixture before a claim reads it.

=====  ======================================================================================
claim  what fails it
=====  ======================================================================================
T1     each node's pooled score, read from the step's own arrays, is solved; mutations M1
       (epsilon fitted on fold 0's training rows only) and M1b (loss weights dropped) break it
T2     ``clever`` is exactly ``1 / cumulative`` from a longhand out-of-fold product; mutation
       M2 (fold 0's slab read as the out-of-fold pair) breaks it
T3     ``initial`` is the longhand stitch of untargeted fold recursions on end-of-study,
       survival, competing-risk and categorical dynamic-rule fits; the previous
       fold-targeted carry (M3) differs from it; ``initial`` ignores ``g_bounds``
T4     a misplaced fold fails T3's equality while the pooled solver row still passes
T5     the nuisance diagnostic scores ``initial`` against ``regression_target``
T6     one fold still carries targeted values and never enters the cross-fit path
T7     ``warn_on_fold_convergence`` still reports the engine-level fold solves it serves
=====  ======================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
    make_longitudinal_weighted,
)
from cleverly.exceptions import ConvergenceWarning, PositivityWarning
from cleverly.longitudinal import LTMLE, LongitudinalResult, sequential
from cleverly.longitudinal.sequential import Mechanism, RegimenFit
from cleverly.validation.longitudinal import _longitudinal_nuisances, _longitudinal_scores
from tests.unit.test_sequential_design import multivalue_panel

#: The columns every ``make_longitudinal*`` frame shares.
SHARED = {
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}

#: The pooled fluctuation's own convergence tolerance, which every fit here keeps.
TOL = 1e-10


def _misspecified(**overrides: Any) -> dict[str, Any]:
    """Constant outcome regressions and a logistic mechanism.

    A constant cannot follow the ``tanh`` outcome surface, so every node's initial fit is
    wrong and its ``epsilon`` is not small.  The mechanism stays logistic, so the clever
    covariate is a real function of the history rather than a constant.
    """
    return {
        "outcome_learner": DummyRegressor(),
        "pseudo_learner": DummyRegressor(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "n_folds": 3,
        "learner_folds": 2,
        "random_state": 0,
        "simultaneous": False,
        "tol": TOL,
        **overrides,
    }


def fit_end_of_study(**overrides: Any) -> LongitudinalResult:
    frame, _ = make_longitudinal(n=600, seed=3)
    return LTMLE({"always": 1, "never": 0}, **_misspecified(**overrides)).fit(
        frame, outcome="Y", **SHARED
    )


def fit_survival(**overrides: Any) -> LongitudinalResult:
    frame, _ = make_longitudinal_survival(n=800, seed=4)
    return LTMLE({"always": 1, "never": 0}, **_misspecified(**overrides)).fit(
        frame, outcome=("Y1", "Y2"), **SHARED
    )


def fit_competing(**overrides: Any) -> LongitudinalResult:
    frame, _ = make_longitudinal_competing(n=900, seed=5)
    return LTMLE({"always": 1, "never": 0}, **_misspecified(**overrides)).fit(
        frame, outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")}, **SHARED
    )


def fit_weighted(**overrides: Any) -> LongitudinalResult:
    frame, _ = make_longitudinal_weighted(n=1400, seed=6)
    return LTMLE({"always": 1, "never": 0}, **_misspecified(**overrides)).fit(
        frame, outcome="Y", weights="w", **SHARED
    )


def fit_categorical(**overrides: Any) -> LongitudinalResult:
    """A three-level first node, a static plan and a dynamic rule."""
    frame = multivalue_panel(n=600, seed=8)
    return LTMLE(
        {
            "never": 0,
            "dynamic": (2, lambda history: (history["L2"] > 0).astype(float)),
        },
        **_misspecified(reference="never", **overrides),
    ).fit(
        frame,
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1",),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )


BUILDERS = {
    "end_of_study": fit_end_of_study,
    "survival": fit_survival,
    "competing": fit_competing,
    "weighted": fit_weighted,
    "categorical_dynamic": fit_categorical,
}


@pytest.fixture(scope="module", params=sorted(BUILDERS))
def pooled(request: pytest.FixtureRequest) -> LongitudinalResult:
    """Each outcome kind, fitted at three outer folds."""
    return BUILDERS[request.param]()


def _assert_epsilon_witness(result: LongitudinalResult) -> None:
    """Every node moved: the premise that makes a solved score evidence."""
    epsilons = [
        abs(float(np.ravel(step.fluctuation.epsilon)[0]))
        for fit in result.fits.values()
        for step in fit.steps
    ]
    assert epsilons
    assert min(epsilons) > 1e-4, epsilons


def _relative_scores(result: LongitudinalResult) -> list[float]:
    """Each node's pooled score over its scale, computed from the step's own arrays."""
    scores = []
    for fit in result.fits.values():
        weights = np.asarray(fit.obs_weights, dtype=float)
        for step in fit.steps:
            multiplier = weights * step.clever
            score = float(np.mean(multiplier * (step.pseudo_outcome - step.targeted)))
            scale = float(np.mean(np.abs(multiplier)))
            scores.append(abs(score) / scale)
    return scores


def _longhand_cumulative(
    result: LongitudinalResult, fit: RegimenFit, *, fold: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The raw and bounded cumulative mechanism, from the stored factors alone.

    ``np.cumprod`` over the interleaved factors rather than the production loop, and
    ``np.clip`` rather than :func:`~cleverly.utils.bounds.bound`.  The multiplication
    order is the same, so the two agree to the last bit.
    """
    mechanism = result.mechanism
    label = fit.regimen.label
    factors = []
    for time in range(result.data.n_times):
        if fold is None:
            treatment = mechanism.treatment[time][label]
            censoring = mechanism.censoring[time][label]
        else:
            treatment = mechanism.treatment_by_fold[time][label][fold]
            censoring = mechanism.censoring_by_fold[time][label][fold]
        factors.append(treatment)
        factors.append(censoring if result.data.censoring_names else np.ones(result.data.n))
    raw = np.cumprod(np.column_stack(factors), axis=1)[:, 1::2]
    lower, upper = result.config.g_bounds
    return raw, np.clip(raw, lower, upper)


# --------------------------------------------------------------------------------------
# T1: the pooled score is solved, and the two mutations that should break it do.
# --------------------------------------------------------------------------------------


def test_every_node_moves_on_every_outcome_kind(pooled: LongitudinalResult) -> None:
    assert pooled.folds.n_folds == 3
    _assert_epsilon_witness(pooled)


def test_each_pooled_node_solves_its_score_over_every_follower(
    pooled: LongitudinalResult,
) -> None:
    """T1: the equation a node solved is the equation its arrays pose, on every row."""
    _assert_epsilon_witness(pooled)
    assert max(_relative_scores(pooled)) < 1e-9
    for fit in pooled.fits.values():
        for step in fit.steps:
            # One solve over every follower, so no fold records and no stitched residual.
            assert step.fluctuation.folds == ()
            assert step.fluctuation.n_solver_calls == 1
            assert step.fluctuation.converged
        curve = np.asarray(fit.influence_curve_scaled, dtype=float)
        assert abs(float(np.mean(curve))) < 1e-9 * (1.0 + float(np.std(curve)))


def test_scores_read_as_one_solver_row_per_node(pooled: LongitudinalResult) -> None:
    rows = _longitudinal_scores(pooled, tolerance=1e-8).rows
    nodes = sum(len(fit.steps) for fit in pooled.fits.values())
    assert [row.kind for row in rows] == ["solver"] * nodes
    assert all(row.passed for row in rows)


def _training_rows_of_fold_zero() -> np.ndarray:
    return np.asarray(fit_end_of_study().folds.assignment != 0)


def test_m1_an_epsilon_fitted_on_one_folds_training_rows_leaves_the_score_unsolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1: a training-fold fluctuation's row mask, applied to the pooled pass, fails T1."""
    training = _training_rows_of_fold_zero()
    real = sequential.solve_fluctuation

    def narrowed(outcome: Any, initial: Any, submodel: Any, weights: Any, observed: Any, **kw: Any):  # type: ignore[no-untyped-def]
        return real(outcome, initial, submodel, weights, observed & training, **kw)

    monkeypatch.setattr(sequential, "solve_fluctuation", narrowed)
    mutated = fit_end_of_study()
    assert np.array_equal(mutated.folds.assignment != 0, training)
    assert max(_relative_scores(mutated)) > 1e-4


def test_m1b_dropped_loss_weights_leave_the_score_unsolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1b: an unweighted fluctuation solves a different equation from the clever one."""
    real = sequential.solve_fluctuation

    def unweighted(
        outcome: Any, initial: Any, submodel: Any, weights: Any, observed: Any, **kw: Any
    ):  # type: ignore[no-untyped-def]
        return real(outcome, initial, submodel, np.where(weights > 0.0, 1.0, 0.0), observed, **kw)

    monkeypatch.setattr(sequential, "solve_fluctuation", unweighted)
    assert max(_relative_scores(fit_end_of_study())) > 1e-4


def test_m1b_dropped_observation_weights_leave_the_weighted_score_unsolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1b on the weighted fit: the tilt must reach the loss, not only the plug-in."""
    reference = fit_weighted()
    observation = np.asarray(reference.data.weights, dtype=float)
    assert np.ptp(observation) > 0.5
    real = sequential.solve_fluctuation

    def untilted(outcome: Any, initial: Any, submodel: Any, weights: Any, observed: Any, **kw: Any):  # type: ignore[no-untyped-def]
        return real(outcome, initial, submodel, weights / observation, observed, **kw)

    monkeypatch.setattr(sequential, "solve_fluctuation", untilted)
    assert max(_relative_scores(fit_weighted())) > 1e-4


# --------------------------------------------------------------------------------------
# T2: the clever covariate is the out-of-fold mechanism, exactly.
# --------------------------------------------------------------------------------------


def _assert_clever_is_the_out_of_fold_reciprocal(result: LongitudinalResult) -> None:
    for fit in result.fits.values():
        raw, bounded = _longhand_cumulative(result, fit)
        np.testing.assert_array_equal(fit.cumulative_unbounded, raw)
        np.testing.assert_array_equal(fit.cumulative, bounded)
        for step in fit.steps:
            expected = np.where(
                step.trained_on,
                1.0 / np.where(step.trained_on, bounded[:, step.time - 1], 1.0),
                0.0,
            )
            np.testing.assert_array_equal(step.clever, expected)


def _fold_slabs_differ_on_training_rows(result: LongitudinalResult) -> float:
    """How far a fold's own slab sits from the out-of-fold pair on its training rows."""
    largest = 0.0
    for fit in result.fits.values():
        _, bounded = _longhand_cumulative(result, fit)
        for fold, (train, _) in enumerate(result.folds):
            _, slab = _longhand_cumulative(result, fit, fold=fold)
            for step in fit.steps:
                rows = train[step.trained_on[train]]
                column = step.time - 1
                if rows.size:
                    largest = max(
                        largest, float(np.max(np.abs(slab[rows, column] - bounded[rows, column])))
                    )
    return largest


def test_clever_is_the_reciprocal_of_the_out_of_fold_cumulative(
    pooled: LongitudinalResult,
) -> None:
    """T2, with the witness that a fold slab would have given a different covariate."""
    _assert_clever_is_the_out_of_fold_reciprocal(pooled)
    assert _fold_slabs_differ_on_training_rows(pooled) > 1e-6


def test_clever_is_the_out_of_fold_reciprocal_under_an_active_bound() -> None:
    """T2 where the lower bound replaces cells the score reads."""
    with pytest.warns(PositivityWarning, match="truncation"):
        result = fit_end_of_study(g_bounds=(0.2, 1.0))
    _assert_epsilon_witness(result)
    clipped = 0
    for fit in result.fits.values():
        for step in fit.steps:
            column = step.time - 1
            clipped += int(
                np.count_nonzero(
                    fit.cumulative_unbounded[step.trained_on, column]
                    != fit.cumulative[step.trained_on, column]
                )
            )
    assert clipped > 0
    _assert_clever_is_the_out_of_fold_reciprocal(result)
    assert max(_relative_scores(result)) < 1e-9


def test_m2_a_fold_slab_read_as_the_out_of_fold_pair_fails_t2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2: dividing by fold 0's slab everywhere is what T2 exists to catch."""
    real = Mechanism.cumulative_with_unbounded

    def slab(self: Mechanism, data: Any, plan: Any, bounds: Any, *, fold: int | None = None):  # type: ignore[no-untyped-def]
        return real(self, data, plan, bounds, fold=0 if fold is None else fold)

    monkeypatch.setattr(Mechanism, "cumulative_with_unbounded", slab)
    mutated = fit_end_of_study()
    monkeypatch.undo()
    with pytest.raises(AssertionError):
        _assert_clever_is_the_out_of_fold_reciprocal(mutated)


# --------------------------------------------------------------------------------------
# T3 and T4: the stitched initial fit is the untargeted fold recursion.
# --------------------------------------------------------------------------------------


def fit_linear(n_folds: int, **overrides: Any) -> LongitudinalResult:
    """Linear sequential regressions, so the longhand is a weighted least-squares solve."""
    frame, _ = make_longitudinal(n=900, seed=12)
    settings = {
        **_misspecified(n_folds=n_folds),
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        **overrides,
    }
    return LTMLE({"always": 1, "never": 0}, **settings).fit(frame, outcome="Y", **SHARED)


def _least_squares(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """A weighted linear fit on ``rows``, predicted everywhere and clipped to ``[0, 1]``."""
    full = np.column_stack([np.ones(len(design)), design])
    root = np.sqrt(weights[rows])
    beta, *_ = np.linalg.lstsq(full[rows] * root[:, None], target[rows] * root, rcond=None)
    return np.clip(full @ beta, 0.0, 1.0)


def _intercept_fluctuation(
    target: np.ndarray, initial: np.ndarray, weights: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """A plain Newton solve of one intercept, for the previous fold-local update (M3)."""
    offset = logit(np.clip(initial, 0.0005, 0.9995))
    epsilon = 0.0
    for _ in range(100):
        fitted = expit(offset[rows] + epsilon)
        gradient = float(np.sum(weights[rows] * (target[rows] - fitted)))
        curvature = float(np.sum(weights[rows] * fitted * (1.0 - fitted)))
        step = gradient / curvature
        epsilon += step
        if abs(step) < 1e-14:
            break
    return np.asarray(expit(offset + epsilon))


def _longhand_masks(
    result: LongitudinalResult, fit: RegimenFit, time: int
) -> tuple[np.ndarray, np.ndarray]:
    """Node ``time``'s ``at_risk`` and ``following`` rows, from the stored arrays alone.

    Prefix ``all`` over the raw censoring, treatment and cumulative event arrays, rather
    than the package's mask scans.  ``at_risk`` reads the prefix through ``time - 1`` and
    ``following`` through ``time``.  Both are event-free through ``time - 1``: a unit that
    has the event at ``time`` is in that node's regression.
    """
    data = result.data
    plan = np.broadcast_to(np.asarray(fit.assignment, dtype=float), (data.n, data.n_times))
    matches = data.treatment == plan
    uncensored = np.asarray(data.uncensored, dtype=bool)
    event_free = (
        np.ones(data.n, dtype=bool)
        if data.event is None or time == 1
        else ~np.asarray(data.event, dtype=bool)[:, time - 2]
    )
    entering = uncensored[:, : time - 1].all(axis=1) & matches[:, : time - 1].all(axis=1)
    leaving = uncensored[:, :time].all(axis=1) & matches[:, :time].all(axis=1)
    return entering & event_free, leaving & event_free


def _longhand_target(
    result: LongitudinalResult, fit: RegimenFit, carried: np.ndarray, time: int
) -> np.ndarray:
    """What node ``time`` regresses, composed longhand.

    End of study: the carried value.  Survival: ``Y_t + (1 - Y_t) * carried``.  Competing
    risks: the cause's own event as the numerator, and the **all-cause** survival factor,
    both read straight from the stored cumulative event arrays.
    """
    data = result.data
    if data.event is None:
        return carried
    any_cause = np.asarray(data.event, dtype=float)[:, time - 1]
    if fit.cause is None:
        numerator = any_cause
    else:
        assert data.cause_event is not None
        column = data.cause_labels.index(fit.cause)
        numerator = np.asarray(data.cause_event, dtype=float)[:, time - 1, column]
    return np.asarray(numerator + (1.0 - any_cause) * carried)


def _longhand_seed(result: LongitudinalResult) -> np.ndarray:
    """``Q_{k+1}``: zero on a survival fit, the observed outcome at the end of study."""
    data = result.data
    if data.event is not None:
        return np.zeros(data.n)
    observed = np.asarray(data.uncensored, dtype=bool).all(axis=1)
    return np.where(observed, np.nan_to_num(data.outcome, nan=0.0), 0.5)


def _longhand_recursion(
    result: LongitudinalResult, fit: RegimenFit, *, fold_targeted: bool = False
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Stitched initial predictions and regression targets, recomputed from the data.

    ``fold_targeted=False`` is the Section 5.2 recursion: fold ``k`` carries its own
    untargeted prediction to the earlier node.  ``fold_targeted=True`` is the previous
    ``cleverly`` construction, mutation M3: fold ``k`` first fluctuates on its training
    followers with its own mechanism slab, and carries that targeted prediction.
    """
    data = result.data
    weights = np.asarray(data.weights, dtype=float)
    initial = {time: np.full(data.n, np.nan) for time in range(1, fit.horizon + 1)}
    target = {time: np.full(data.n, np.nan) for time in range(1, fit.horizon + 1)}
    for fold in range(result.folds.n_folds):
        held_out = result.folds.assignment == fold
        training = ~held_out
        _, slab = _longhand_cumulative(result, fit, fold=fold)
        carried = _longhand_seed(result)
        for time in range(fit.horizon, 0, -1):
            at_risk, following = _longhand_masks(result, fit, time)
            outcome = _longhand_target(result, fit, carried, time)
            rows = following & training
            prediction = np.where(
                at_risk,
                _least_squares(data.covariate_history(time), outcome, weights, rows),
                0.5,
            )
            initial[time][held_out] = prediction[held_out]
            target[time][held_out] = outcome[held_out]
            if fold_targeted:
                loss = np.where(at_risk, weights / np.where(at_risk, slab[:, time - 1], 1.0), 0.0)
                prediction = np.where(
                    at_risk, _intercept_fluctuation(outcome, prediction, loss, rows), 0.5
                )
            carried = np.where(at_risk, prediction, 0.5)
    return initial, target


def _assert_stitched_longhand(result: LongitudinalResult) -> None:
    """T3's claim, with the witness that the pooled update moved every node."""
    for fit in result.fits.values():
        initial, target = _longhand_recursion(result, fit)
        for step in fit.steps:
            at_risk, following = _longhand_masks(result, fit, step.time)
            np.testing.assert_array_equal(step.at_risk, at_risk)
            np.testing.assert_array_equal(step.trained_on, following)
            np.testing.assert_allclose(step.initial, initial[step.time], rtol=0.0, atol=1e-12)
            assert step.regression_target is not None
            np.testing.assert_allclose(
                step.regression_target, target[step.time], rtol=0.0, atol=1e-12
            )
            # The witness: an initial fit equal to the targeted one would pass the
            # equalities above whether the pooled pass ran or not.
            moved = np.abs(step.targeted - step.initial)[step.at_risk]
            assert float(np.max(moved)) > 1e-6, (fit.regimen.label, fit.cause, step.time)


@pytest.fixture(scope="module", params=[3, 5], ids=["K=3", "K=5"])
def linear(request: pytest.FixtureRequest) -> LongitudinalResult:
    return fit_linear(request.param)


def test_stitched_initial_is_the_untargeted_fold_recursion(linear: LongitudinalResult) -> None:
    """T3: every node's ``initial`` is the longhand stitch, on every row."""
    _assert_epsilon_witness(linear)
    _assert_stitched_longhand(linear)


def _linear_settings(**overrides: Any) -> dict[str, Any]:
    return {
        **_misspecified(),
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        **overrides,
    }


def fit_linear_survival() -> LongitudinalResult:
    frame, _ = make_longitudinal_survival(n=800, seed=4)
    return LTMLE({"always": 1, "never": 0}, **_linear_settings()).fit(
        frame, outcome=("Y1", "Y2"), **SHARED
    )


def fit_linear_competing() -> LongitudinalResult:
    frame, _ = make_longitudinal_competing(n=900, seed=5)
    return LTMLE({"always": 1, "never": 0}, **_linear_settings()).fit(
        frame, outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")}, **SHARED
    )


def fit_linear_categorical() -> LongitudinalResult:
    frame = multivalue_panel(n=600, seed=8)
    return LTMLE(
        {
            "never": 0,
            "dynamic": (2, lambda history: (history["L2"] > 0).astype(float)),
        },
        **_linear_settings(reference="never"),
    ).fit(
        frame,
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1",),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )


LINEAR_KINDS = {
    "survival": fit_linear_survival,
    "competing": fit_linear_competing,
    "categorical_dynamic": fit_linear_categorical,
}


@pytest.fixture(scope="module", params=sorted(LINEAR_KINDS))
def linear_kind(request: pytest.FixtureRequest) -> LongitudinalResult:
    return LINEAR_KINDS[request.param]()


def test_stitched_initial_is_the_untargeted_fold_recursion_on_every_outcome_kind(
    linear_kind: LongitudinalResult,
) -> None:
    """T3 beyond the end of study: event composition, causes and a dynamic rule.

    Each kind changes what a fold regresses or which rows it regresses on.  A survival
    fit composes ``Y_t + (1 - Y_t) * carried``.  A competing-risk fit reads the cause's
    event in the numerator and all-cause survival in the factor.  A dynamic rule gives
    each row its own arm at node 2.  The longhand recomputes each from the raw arrays.
    """
    data = linear_kind.data
    assert linear_kind.folds.n_folds == 3
    if data.is_competing:
        # The witness that the numerator is the cause's own event: a cause's cumulative
        # indicator differs from the all-cause one, so the two compositions differ.
        assert data.cause_event is not None
        assert data.event is not None
        first = np.asarray(data.cause_event, dtype=float)[:, :, 0]
        assert np.any(first != np.asarray(data.event, dtype=float))
        assert {fit.cause for fit in linear_kind.fits.values()} == set(data.cause_labels)
    elif data.is_survival:
        assert {fit.horizon for fit in linear_kind.fits.values()} == {1, 2}
    else:
        # The witness that the rule is dynamic: node 2's arm varies across rows.
        dynamic = np.asarray(linear_kind.fits["dynamic"].assignment, dtype=float)
        assert dynamic.ndim == 2
        assert np.unique(dynamic[:, 1]).size == 2
    _assert_stitched_longhand(linear_kind)


def test_m3_the_previous_fold_targeted_carry_is_a_different_initial_fit(
    linear: LongitudinalResult,
) -> None:
    """M3: carrying fold-targeted predictions moves node 1's stitched fit off T3's."""
    for fit in linear.fits.values():
        fold_targeted, _ = _longhand_recursion(linear, fit, fold_targeted=True)
        assert float(np.max(np.abs(fit.steps[0].initial - fold_targeted[1]))) > 1e-6


def test_the_initial_fit_does_not_read_the_bounds() -> None:
    """T3: the fold recursions read no mechanism, so only the targeted fit moves."""
    loose = fit_linear(3)
    with pytest.warns(PositivityWarning, match="truncation"):
        tight = fit_linear(3, g_bounds=(0.25, 1.0))
    moved = 0.0
    for label, fit in loose.fits.items():
        for left, right in zip(fit.steps, tight.fits[label].steps, strict=True):
            np.testing.assert_array_equal(left.initial, right.initial)
            np.testing.assert_array_equal(left.regression_target, right.regression_target)
            moved = max(moved, float(np.max(np.abs(left.targeted - right.targeted))))
    assert moved > 1e-6


def test_a_misplaced_fold_fails_the_longhand_but_not_the_solver_row() -> None:
    """T4: why T3 exists.  The pooled solver row cannot see a rotated fold.

    Rotating node 1's held-out predictions across folds leaves a perfectly solvable node:
    re-solving the pooled fluctuation on the rotated offset reaches its root, and the
    production solver row passes.  Only the comparison with the untargeted fold recursion
    fails, which replaces the stitching gate of the fold-fluctuated construction.
    """
    result = fit_linear(3)
    fit = result.fits["always"]
    step = fit.steps[0]
    order = np.argsort(result.folds.assignment, kind="stable")
    rotated = step.initial.copy()
    rotated[order] = np.roll(step.initial[order], result.n // result.folds.n_folds)
    initial, _ = _longhand_recursion(result, fit)
    assert float(np.max(np.abs(rotated - initial[1]))) > 1e-3

    weights = np.asarray(result.data.weights, dtype=float)
    counterfactual, clever = sequential._clever_covariate(
        step.at_risk, step.trained_on, fit.cumulative, 1
    )
    np.testing.assert_array_equal(clever, step.clever)
    fluctuation = sequential._fluctuate_node(
        step.pseudo_outcome,
        rotated,
        weights * counterfactual,
        step.trained_on,
        label="always",
        time=1,
        alpha=0.9995,
        max_iter=20,
        tol=TOL,
    )
    targeted = fluctuation.targeted.arms[0.0]
    damaged = replace(step, initial=rotated, targeted=targeted, fluctuation=fluctuation)
    broken = replace(
        result,
        fits={**result.fits, "always": replace(fit, steps=(damaged, *fit.steps[1:]))},
    )
    rows = [
        row
        for row in _longitudinal_scores(broken, tolerance=1e-8).rows
        if row.regimen == "always" and row.time == 1
    ]
    assert [row.kind for row in rows] == ["solver"]
    assert rows[0].passed
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(damaged.initial, initial[1], rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------------------
# T5: the nuisance diagnostic scores the regression against what it was fitted to.
# --------------------------------------------------------------------------------------


def test_the_nuisance_loss_reads_the_regression_target_when_cross_fitted() -> None:
    result = fit_linear(3)
    report = _longitudinal_nuisances(result)
    weights = np.asarray(result.data.weights, dtype=float)
    fit = result.fits["always"]
    step = fit.steps[0]
    assert step.regression_target is not None
    rows = step.trained_on
    # The witness: the fold target and the pooled pseudo-outcome differ where it is read.
    assert float(np.max(np.abs(step.regression_target[rows] - step.pseudo_outcome[rows]))) > 1e-6
    row = next(
        item
        for item in report.rows
        if item.role == "pseudo_outcome" and item.regimen == "always" and item.time == 1
    )
    expected = np.average(
        np.square(step.regression_target[rows] - step.initial[rows]), weights=weights[rows]
    )
    wrong = np.average(
        np.square(step.pseudo_outcome[rows] - step.initial[rows]), weights=weights[rows]
    )
    assert row.evaluation == "out_of_fold"
    assert row.loss == pytest.approx(expected, rel=1e-12)
    assert abs(expected - wrong) > 1e-8


def test_one_fold_regresses_the_pseudo_outcome_it_reports() -> None:
    """T5 at ``K = 1``: no separate target, because the regression target is the pseudo-outcome."""
    result = fit_linear(1)
    weights = np.asarray(result.data.weights, dtype=float)
    for fit in result.fits.values():
        masks = result.data.regimen_masks(fit.assignment)
        for step in fit.steps:
            assert step.regression_target is None
            prediction = _least_squares(
                result.data.covariate_history(step.time),
                step.pseudo_outcome,
                weights,
                masks.following(step.time),
            )
            np.testing.assert_allclose(
                step.initial, np.where(step.at_risk, prediction, 0.5), rtol=0.0, atol=1e-12
            )


# --------------------------------------------------------------------------------------
# T6: the single-fold fit is the canonical recursion it always was.
# --------------------------------------------------------------------------------------


def _assert_pseudo_outcome_carries_targeted(result: LongitudinalResult) -> None:
    for fit in result.fits.values():
        for later, earlier in zip(fit.steps[1:], fit.steps[:-1], strict=True):
            carried = np.where(later.at_risk, later.targeted, 0.5)
            np.testing.assert_array_equal(earlier.pseudo_outcome, carried)
            assert float(np.max(np.abs(later.targeted - later.initial)[later.at_risk])) > 1e-6


def test_one_fold_carries_targeted_values_and_skips_the_cross_fit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a single-fold fit entered the cross-fitted construction")

    monkeypatch.setattr(sequential, "_fit_regimen_crossfit", refuse)
    # Linear rather than constant regressions: at one fold a constant carried back is
    # predicted exactly at the earlier node, which would leave its epsilon at zero.
    result = fit_linear(1)
    # The witness this claim needs is the one the helper asserts: at every later node the
    # targeted fit sits away from the initial one, so carrying either is distinguishable.
    _assert_pseudo_outcome_carries_targeted(result)
    assert max(_relative_scores(result)) < 1e-9


def test_the_pooled_pass_carries_its_own_targeted_values() -> None:
    """Step 3 of the Section 5.2 construction: the pooled outcome is the later targeted fit."""
    result = fit_end_of_study()
    _assert_pseudo_outcome_carries_targeted(result)


# --------------------------------------------------------------------------------------
# T7: the fold-convergence warning the engine-level working model still relies on.
# --------------------------------------------------------------------------------------


def _node(converged_folds: list[bool], failure: str | None = None) -> Any:
    records = tuple(SimpleNamespace(converged=flag) for flag in converged_folds)
    return SimpleNamespace(
        folds=records,
        converged=all(converged_folds),
        failure=None if all(converged_folds) else failure,
    )


def test_fold_convergence_is_silent_when_every_fold_converged() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sequential.warn_on_fold_convergence([(1, _node([True, True])), (2, _node([True]))], "msm")


def test_fold_convergence_failures_are_reported_once_with_count_nodes_and_modes() -> None:
    nodes = [
        (2, _node([False, True, False], "max_iter_reached")),
        (1, _node([True, False], "diverged")),
        (3, _node([True, True])),
    ]
    with pytest.warns(ConvergenceWarning) as caught:
        sequential.warn_on_fold_convergence(nodes, "msm")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "3 outer-fold targeting solve(s) did not converge for 'msm'" in message
    assert "at node(s) [1, 2]" in message
    assert "(diverged, max_iter_reached)" in message


def test_a_pooled_fit_hands_the_fold_warning_nothing() -> None:
    """The per-regimen path hands the warning nothing, so it cannot speak for that path."""
    result = fit_end_of_study()
    nodes = [(step.time, step.fluctuation) for fit in result.fits.values() for step in fit.steps]
    assert all(fluctuation.folds == () for _, fluctuation in nodes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sequential.warn_on_fold_convergence(nodes, "always")
