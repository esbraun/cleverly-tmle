"""The fits, the limits and the controls of the ``learned-rule-ep`` study (P1 part (c)).

The study measures the learned-rule value when each fold's rule is the sign of the EP curve of
the effect given the declared modifiers, ``LearnedRule(loss="ep", modifiers=MODIFIERS)``, with
the default sieve (additive cosines of ``W1`` and the indicator of ``W2``, its size chosen by
cross-validation over the inner folds with the nuisances refitted inside each).  It reuses the
RM30 laws ``non_exceptional``, ``misspecified_limit`` and ``exceptional``, the main-terms outcome
learner and the logistic treatment learner of :mod:`tests.studies._learned_rule_law` unchanged
(P1 plan section 5.5, Q12: the RM30 GLMs, not the ensemble).  The modifiers are both adjustment
columns, so the value of the optimal rule (R12) is admitted.

The truth of each replication is rule L4 of the RM30 studies: the harness refits each fold's
rule through the package's ``fit_fold_rule`` with the fold's recorded inner seed, the fit's
resolved ``g_bounds`` and its ``alpha``, requires the fit's rule on every validation row, and
integrates each rule's value on the RM30 grid.  The refit reproduces the record through the
package's code.  The construction rests on the unit witnesses W-c1 to W-c3 of
``tests/unit/test_learned_rule_ep.py``.

The fold-locality control is :class:`~tests.studies._learned_rule_law.ValidationRowRule`: fold
``v``'s EP curve is fitted on fold ``v``'s validation rows from the fit's outer cross-fitted
nuisances (:func:`~tests.studies._learned_rule_law.validation_ep_rule`), so the rows that
evaluate the rule also chose it.

The population limit (:func:`limit_scores`).  With either inner nuisance consistent, the EP
risk is doubly robust and the fold curves converge to the projection of the conditional
effect ``tau(W) = Qbar0(1, W) - Qbar0(0, W)`` onto the closure of the sieve's span as its size
grows: the additive functions ``f(W1) + h(W2)``.  ``W1`` and ``W2`` are independent, so the
projection is ``E[tau | W1] + E[tau | W2] - E[tau]``.  The inner treatment learner is the
correct logistic model, so the DR branch through ``g`` holds whatever the main-terms outcome
learner converges to.  ``tau`` is not additive on the probability scale, so the limit rule
differs from ``d0`` on a band near the threshold; its regret is about ``2.5e-5`` at
``non_exceptional``, 0.002 SE at n = 4,000 (:func:`optimal_regret`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from cleverly.bases import Modifiers
from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_optimal_law as optimal_law
from tests.studies.evidence.schema import reported_inference

#: The two gated laws; ``exceptional`` is a reporting law (C3 fails).
LAWS = ("non_exceptional", "misspecified_limit")
EXCEPTIONAL_LAW = "exceptional"

#: The declared modifiers: both adjustment columns, with the law's domain and levels.
MODIFIERS = Modifiers(["W1", "W2"], domains={"W1": (-1.0, 1.0)}, levels={"W2": (0.0, 1.0)})

#: The reported parameter of a learned-rule fit, and of an optimal-value fit.
ESTIMAND = base.ESTIMAND
OPTIMAL_ESTIMAND = optimal_law.ESTIMAND

#: The descriptive columns of the paired comparison with the plug-in rule.
VALUE_GAIN_COLUMNS = ("plug_in_estimate", "plug_in_std_error", "plug_in_truth")


def rule() -> LearnedRule:
    """The declared rule: the EP curve given ``(W1, W2)`` with the default sieve."""
    return LearnedRule(loss="ep", modifiers=MODIFIERS)


# ------------------------------------------------------------------------------ the fits


@dataclass(frozen=True)
class Measured:
    """One sample's declared fit and its plug-in comparison.

    Parameters
    ----------
    result : object
        The declared fit.
    truth : tests.studies._learned_rule_law.Truth
        Its harness truth.
    warnings : int
        The solver warnings of the declared fit and its refits.
    plug_in_estimate : float
        The plug-in fit's estimate on the same sample and outer folds.
    plug_in_std_error : float
        The plug-in fit's reported standard error.
    plug_in_truth : float
        The plug-in fit's harness truth.
    """

    result: Any
    truth: base.Truth
    warnings: int
    plug_in_estimate: float
    plug_in_std_error: float
    plug_in_truth: float


def _same_folds(first: Any, second: Any) -> bool:
    pairs = zip(first.nuisance.folds, second.nuisance.folds, strict=True)
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        and np.array_equal(np.asarray(c), np.asarray(d))
        for (a, c), (b, d) in pairs
    )


def measured(frame: pd.DataFrame, law: str, fold_seed: int) -> Measured:
    """Fit the declared EP rule and the plug-in rule on one sample and fold seed.

    Both fits draw the same outer folds; a fit whose folds differ stops the replication.
    Rule L4's stops of :func:`tests.studies._learned_rule_law.measured` apply to each fit.
    """
    result, truth, warnings = base.measured(
        frame, base.outcome_learner(), law, fold_seed, rule=rule()
    )
    plug_in, plug_truth, _ = base.measured(frame, base.outcome_learner(), law, fold_seed)
    if not _same_folds(result, plug_in):
        raise RuntimeError("the plug-in fit drew other outer folds than the declared fit")
    return Measured(
        result=result,
        truth=truth,
        warnings=warnings,
        plug_in_estimate=float(plug_in.estimates[ESTIMAND].psi),
        plug_in_std_error=reported_inference(plug_in.estimates[ESTIMAND])[0],
        plug_in_truth=plug_truth.truth,
    )


def value_gain_columns(fitted: Measured) -> dict[str, float]:
    """The descriptive :data:`VALUE_GAIN_COLUMNS` of one measured sample."""
    return {
        "plug_in_estimate": fitted.plug_in_estimate,
        "plug_in_std_error": fitted.plug_in_std_error,
        "plug_in_truth": fitted.plug_in_truth,
    }


def control(frame: pd.DataFrame, law: str, fold_seed: int) -> tuple[Any, base.Truth, int]:
    """The fold-locality control: each fold's EP curve fitted on its validation rows."""
    return base.measured(frame, base.outcome_learner(), law, fold_seed, mutation=True, rule=rule())


def optimal_estimator(fold_seed: int) -> TMLE:
    """The declared optimal-value fit: :func:`base.estimator`'s settings, ``ey_optimal_rule``."""
    return TMLE(
        learned_rule=rule(),
        estimands=[optimal_law.TARGET],
        n_folds=base.N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=base.outcome_learner(),
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


@dataclass(frozen=True)
class OptimalMeasured:
    """One optimal-value fit with the known mechanism, and its refit fold rules.

    Parameters
    ----------
    result : object
        The fit.
    truth : float
        :math:`\\Psi_{d_0}(P_0)`, the fixed target.
    refit : tests.studies._learned_rule_optimal_law.Refit
        The value of the fit's fold rules and their oracle SE (rule L4).
    warnings : int
        The solver warnings of the fit and its refits.
    """

    result: Any
    truth: float
    refit: Any
    warnings: int


def optimal_measured(frame: pd.DataFrame, fold_seed: int) -> OptimalMeasured:
    """Fit the value of the optimal rule at ``non_exceptional`` with ``g0`` supplied.

    The harness is :func:`tests.studies._learned_rule_optimal_law.harness`, which refits each
    fold's EP rule through ``fit_fold_rule`` and requires it on every validation row.
    """
    with base.solver_warning_count() as count:
        result = (
            optimal_estimator(fold_seed)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=list(base.COVARIATES),
                treatment_probabilities=optimal_law.known_probabilities(frame),
            )
            .single()
        )
        refit = optimal_law.harness(result, base.outcome_learner(), "non_exceptional")
    if tuple(result.estimates) != (OPTIMAL_ESTIMAND,):
        raise base.HarnessMismatch(f"the fit reports {tuple(result.estimates)}")
    if "known_mechanism" not in result.config.support_key.qualifiers:
        raise base.HarnessMismatch("the optimal-value fit does not key known_mechanism")
    estimate = result.estimates[OPTIMAL_ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or SE (rule L4): {estimate}")
    truth = optimal_law.law_limit("non_exceptional")["truth"]
    return OptimalMeasured(result=result, truth=truth, refit=refit, warnings=count[0])


# ------------------------------------------------------------------------- the limits


def _effect(law: str) -> dict[float, np.ndarray]:
    """``tau(W1, w2)`` on the grid, for each ``w2``."""
    return {
        w2: base.qbar0(1.0, base.GRID, w2, law) - base.qbar0(0.0, base.GRID, w2, law)
        for w2 in base.W2_LEVELS
    }


def _mass() -> np.ndarray:
    grid = base.GRID
    mass = np.full(grid.size, grid[1] - grid[0])
    mass[[0, -1]] /= 2.0
    return mass / mass.sum()


@cache
def limit_scores(law: str) -> dict[float, np.ndarray]:
    """The EP limit curve on the grid: the additive projection of ``tau``, for each ``w2``."""
    tau = _effect(law)
    mass = _mass()
    given_w1 = np.mean([tau[w2] for w2 in base.W2_LEVELS], axis=0)
    overall = float(np.sum(mass * given_w1))
    return {w2: given_w1 + float(np.sum(mass * tau[w2])) - overall for w2 in base.W2_LEVELS}


def limit_rule(law: str) -> dict[float, np.ndarray]:
    """The limit rule ``1{theta_1 > 0}`` on the grid."""
    return {w2: score > 0.0 for w2, score in limit_scores(law).items()}


def limit_value(law: str) -> tuple[float, float]:
    """The value and the row-curve variance of the limit rule at ``law``."""
    return base.rule_value(limit_rule(law), law)


def optimal_regret(law: str) -> float:
    """``Psi_{d0} - Psi_{d1}``: the value the additive limit rule gives up against ``d0``."""
    optimal = {w2: effect > 0.0 for w2, effect in _effect(law).items()}
    return base.rule_value(optimal, law)[0] - limit_value(law)[0]


def disagreement(law: str) -> float:
    """The population share on which the limit rule and ``d0`` differ."""
    mass = _mass()
    return float(
        np.mean(
            [
                np.sum(mass * ((limit_scores(law)[w2] > 0.0) != (_effect(law)[w2] > 0.0)))
                for w2 in base.W2_LEVELS
            ]
        )
    )


def near_threshold(law: str, epsilon: float) -> float:
    """The population share whose limit score lies within ``epsilon`` of 0 (C3)."""
    mass = _mass()
    return float(
        np.mean([np.sum(mass * (np.abs(score) < epsilon)) for score in limit_scores(law).values()])
    )


def thresholds(law: str) -> dict[float, tuple[float, ...]]:
    """The values of ``W1`` at which the limit score changes sign, for each ``W2``."""
    out = {}
    for w2, half in limit_scores(law).items():
        crossings = np.flatnonzero(np.diff(np.sign(half)) != 0)
        out[w2] = tuple(
            float(
                base.GRID[i] - half[i] * (base.GRID[i + 1] - base.GRID[i]) / (half[i + 1] - half[i])
            )
            for i in crossings
        )
    return out
