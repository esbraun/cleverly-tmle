"""Cross-fitted policy fits: untargeted fold recursions that carry the policy mean, then one
pooled update per node that moves every arm.

The per-regimen construction is the one ``tests/unit/test_pooled_longitudinal_targeting.py``
pins for deterministic plans (Díaz, Williams, Hoffman and Schenck 2023, Section 5.2).  A
policy node adds three things, and the longhand stitch and two production mutations check them:

* each fold's untargeted recursion carries the fold's policy-weighted mean of its per-arm
  predictions to the earlier node, and the parent stitches the held-out per-arm predictions;
* the pooled fluctuation at a policy node moves every stitched per-arm prediction by the one
  pooled coefficient, and the node carries their policy-weighted mean;
* the pooled score at every node is solved over every row that remains on the plan, with
  the ratio in the loss weight.

The outcome learners are misspecified, so every ``epsilon`` is far from zero and a wrong
score cannot pass by vanishing.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

from cleverly.estimators._nuisance import fit_on_rows
from cleverly.learners._fitting import predict_mean
from cleverly.longitudinal import LTMLE, sequential
from cleverly.longitudinal.sequential import outcome_design

from .. import discrete_law_longitudinal_multivalue as law
from .. import longitudinal_policies as policies
from ..studies.canonical_ltmle import QuasiBinomialGLM

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}
FILLER = 0.5
CAUGHT = 1e-4


def _fit(**overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": DummyRegressor(),
        "treatment_learner": law.CellProbabilities(),
        "n_folds": 3,
        "random_state": 0,
        "g_bounds": (1e-8, 1.0 - 1e-8),
        "simultaneous": False,
        "reference": "low",
        "tol": 1e-12,
        **overrides,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(policies.regimens(("low", "mix")), **settings).fit(
            law.sample(law.PROBS, 900, 11), **COLUMNS
        )


@pytest.fixture(scope="module")
def result() -> Any:
    return _fit()


def _longhand_stitch(result: Any) -> dict[int, np.ndarray]:
    """Each fold's untargeted recursion, refitted here from the plan and the fold rows."""
    data = result.data
    fit = result.fits["mix"]
    plan = next(item for item in result.replay_recipe.plans if item.label == "mix")
    masks = plan.masks(data)
    stitched = {time: np.full((data.n, 3), FILLER) for time in (1, 2)}
    for train, test in result.folds:
        rows = np.zeros(data.n, dtype=bool)
        rows[train] = True
        carried = np.where(data.uncensored_through(2), np.nan_to_num(data.outcome), FILLER)
        for time, learner, task in (
            (2, QuasiBinomialGLM(), "classification"),
            (1, DummyRegressor(), "regression"),
        ):
            fitted_on = masks.following(time) & rows
            model = fit_on_rows(
                clone(learner),
                outcome_design(data, plan, time),
                carried,
                data.weights,
                np.flatnonzero(fitted_on),
                task,
                None,
            )
            by_arm = np.column_stack(
                [
                    predict_mean(model, outcome_design(data, plan, time, arm=code), task)
                    for code in range(3)
                ]
            )
            by_arm = np.clip(by_arm, 0.0, 1.0)
            at_risk = masks.at_risk(time)
            stitched[time][test] = np.where(at_risk[:, None], by_arm, FILLER)[test]
            carried = np.where(at_risk, np.sum(fit.policy[time - 1] * by_arm, axis=1), FILLER)
    return stitched


def test_every_epsilon_is_far_from_zero(result: Any) -> None:
    for step in result.fits["mix"].steps:
        assert abs(float(step.fluctuation.epsilon[0])) > 0.01


def test_the_stitched_per_arm_predictions_are_the_untargeted_fold_recursions(result: Any) -> None:
    longhand = _longhand_stitch(result)
    for step in result.fits["mix"].steps:
        np.testing.assert_allclose(step.initial_by_arm, longhand[step.time], atol=1e-10, rtol=0)
        codes = result.fits["mix"].assignment[:, step.time - 1].astype(int)
        selected = step.initial_by_arm[np.arange(result.data.n), codes]
        assert np.array_equal(step.initial[step.at_risk], selected[step.at_risk])


def _score(result: Any) -> float:
    worst = 0.0
    weights = result.data.weights
    for step in result.fits["mix"].steps:
        rows = step.trained_on
        residual = (weights * step.clever * (step.pseudo_outcome - step.targeted))[rows]
        worst = max(worst, abs(float(np.sum(residual))) / result.data.n)
    return worst


def test_each_node_solves_its_ratio_score_over_every_supported_row(result: Any) -> None:
    assert _score(result) < 1e-10


def test_each_node_carries_the_policy_mean_of_its_targeted_arms(result: Any) -> None:
    fit = result.fits["mix"]
    first, second = fit.steps
    expected = np.sum(fit.policy[1] * second.targeted_by_arm, axis=1)
    np.testing.assert_allclose(second.value[second.at_risk], expected[second.at_risk], atol=0)
    np.testing.assert_array_equal(
        first.pseudo_outcome[first.trained_on], second.value[first.trained_on]
    )


def test_a_per_fold_fluctuation_leaves_the_pooled_score_unsolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: each node's fluctuation solves over one fold's rows only."""
    original = sequential._fluctuate_node

    def one_fold(pseudo: Any, initial: Any, weights: Any, fitted_on: Any, **kwargs: Any) -> Any:
        half = fitted_on.copy()
        half[len(half) // 3 :] = False
        return original(pseudo, initial, weights, half, **kwargs)

    monkeypatch.setattr(sequential, "_fluctuate_node", one_fold)
    assert _score(_fit()) > CAUGHT


def _stitch_gap(result: Any) -> float:
    """How far the fit's stitched per-arm predictions sit from the independent longhand."""
    longhand = _longhand_stitch(result)
    return max(
        float(np.max(np.abs(step.initial_by_arm - longhand[step.time])))
        for step in result.fits["mix"].steps
    )


def test_a_fold_that_carries_the_observed_arm_fails_the_longhand_stitch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: a fold's untargeted recursion carries the observed-arm prediction, not the mean.

    The policy carry inside the fold recursion is the step this file exists for.  The longhand
    stitch rebuilds every fold's recursion from the data and the policy, so it reads nothing the
    mutation writes.
    """
    original = sequential._fit_node_regression

    def observed_carry(*args: Any, **kwargs: Any) -> Any:
        node = original(*args, **kwargs)
        if node.initial_marginal is None or kwargs.get("fit_rows") is None:
            return node
        return dataclasses.replace(node, initial_marginal=node.initial)

    monkeypatch.setattr(sequential, "_fit_node_regression", observed_carry)
    assert _stitch_gap(_fit()) > CAUGHT


def test_a_per_arm_block_from_other_rows_fails_the_longhand_stitch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: each row's stitched per-arm block is another fold's row, kept self-consistent.

    The observed-arm column is re-selected from the moved block, so the stitch agrees with
    itself and the pooled score is still solved.  Only the independent longhand sees it.
    """
    original = sequential.untargeted_fold_recursions

    def moved(data: Any, *args: Any, **kwargs: Any) -> Any:
        stitched = original(data, *args, **kwargs)
        shift = data.n // 3
        for item in stitched:
            for time, block in item.initial_by_arm.items():
                rolled = np.roll(block, shift, axis=0)
                item.initial_by_arm[time] = rolled
                codes = np.nan_to_num(data.treatment[:, time - 1], nan=0.0).astype(int)
                item.initial[time] = rolled[np.arange(data.n), codes]
        return stitched

    monkeypatch.setattr(sequential, "untargeted_fold_recursions", moved)
    mutated = _fit()
    assert _score(mutated) < 1e-10
    assert _stitch_gap(mutated) > CAUGHT
