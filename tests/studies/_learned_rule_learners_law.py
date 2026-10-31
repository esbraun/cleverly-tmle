"""The fits, the limits and the controls of the ``learned-rule-learners`` study (X11 PR C).

The study measures the learned-rule value when each fold's rule comes from a rule learner,
``LearnedRule(loss="blip")`` or ``LearnedRule(loss="weighted_classification")`` (X11 part (g)),
rather than from the plug-in of the outcome regression. It reuses the RM30 law
``misspecified_limit``, the main-terms outcome learner and the logistic treatment learner of
:mod:`tests.studies._learned_rule_law` unchanged. Its two scenarios are the two losses:

``blip``
    ``LearnedRule(loss="blip", learner=Pipeline(PolynomialFeatures(2), LinearRegression()))``.
``weighted_classification``
    ``LearnedRule(loss="weighted_classification", learner=Pipeline(PolynomialFeatures(2),
    LogisticRegression(C=1e6, max_iter=5000)))``.

The basis of both rule learners holds ``W1**2``, which the outcome learner lacks, so their limit
rules differ from the plug-in limit rule (W-g1 of ``tests/unit/test_learned_rule_learners.py``).

The truth of each replication is rule L4 of the RM30 studies: the harness refits each fold's
rule through the package's ``fit_fold_rule`` with the fold's recorded inner seed and the fit's
resolved ``g_bounds``, requires the fit's rule on every validation row, and integrates each
rule's value on the RM30 grid. For a rule learner that refit is a reproduction of the record
through the package's code, not an independent check of the construction. The target is the
value of the realized rule, whatever its construction, and the construction's correctness rests
on the unit witnesses W-g2 and W-g4.

Two quantities sit beside the declared fit. The fold-locality control is the deliberate mutation
:class:`~tests.studies._learned_rule_law.ValidationRowRule`: fold ``v``'s rule learner is fitted
on fold ``v``'s *validation* rows, to the pseudo-outcome of the fit's outer cross-fitted
nuisances, so the rows that evaluate the rule also chose it, and the estimate is optimistic. The control
resolving below its band shows that the instrument sees a gross fold-locality violation. It is no
evidence about the subtle hazard of pseudo-outcome nuisances read from the outer cross-fitted
predictions at the rows of ``T_v``, whose leak is of order 1/n and invisible to any calibration
cell; that hazard rests on the unit witnesses W-g3 and W-g4, and the declared fit passing its
band is the evidence that it is fold-local.
The value gain is the descriptive paired comparison with the plug-in rule: the plug-in fit of the
same sample with the same fold seed, which draws the same outer folds, and its own harness truth.

The population limits (:func:`limit_scores`) are the projections that the rule learners
converge to when the inner nuisances converge to their limits: the logistic main-terms
projection ``Qbar*`` of the outcome learner and the true mechanism ``g0``, which the correctly
specified treatment learner reaches. ``g_bounds="auto"`` resolves to about 0.0147 at n = 2,000,
and ``g0`` lies in about [0.378, 0.574], so the truncation does not bind in the limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_law as base
from tests.studies.evidence.schema import reported_inference

#: The RM30 law every draw comes from.
LAW = "misspecified_limit"

#: The two losses, which are also the scenario names.
LOSSES = ("blip", "weighted_classification")

#: The reported parameter of every fit.
ESTIMAND = base.ESTIMAND

#: The inverse penalty of the weighted classifier. The sample objective is
#: ``sum_i |D_i| logloss_i + ||beta||^2 / (2 C)``, so a fixed ``C`` vanishes against the sum as
#: ``n`` grows and the limit is the unpenalized weighted projection. At ``C = 1e6`` the penalty
#: is also negligible at n = 2,000, as in the unit witnesses.
CLASSIFIER_C = 1e6

#: The descriptive columns of the paired comparison with the plug-in rule, carried by every
#: primary row, every calibration-positive property row and every row of the diagnostic cells:
#: the plug-in fit's estimate, its reported standard error and its own harness truth, on the same
#: sample and the same outer folds. The standard error lets a page read the plug-in's SE ratio
#: on the same draws as the rule learner's. No rule reads any of them.
VALUE_GAIN_COLUMNS = ("plug_in_estimate", "plug_in_std_error", "plug_in_truth")

#: The range of ``g0(1 | W)`` on the support, ``expit(-0.5)`` to ``expit(0.3)``. Every
#: ``g0(a | W)`` is at least ``expit(-0.5)``, far above the ``g_bounds="auto"`` floor
#: ``5 / (sqrt(n) log n)``, about 0.0147 at n = 2,000, so the truncation does not bind in the
#: limit. :func:`propensity_floor` reads the fitted mechanism of one fit for the pilot.
G0_RANGE = (0.3775406687981454, 0.5744425168116589)


# ------------------------------------------------------------------------- the learners


def blip_learner() -> Pipeline:
    """The declared blip learner: a linear regression on every term of degree 2 in ``W``."""
    return Pipeline([("basis", PolynomialFeatures(2)), ("fit", LinearRegression())])


def classifier() -> Pipeline:
    """The declared weighted classifier, on the same basis, at :data:`CLASSIFIER_C`."""
    return Pipeline(
        [
            ("basis", PolynomialFeatures(2)),
            ("fit", LogisticRegression(C=CLASSIFIER_C, max_iter=5000)),
        ]
    )


def rule(loss: str) -> LearnedRule:
    """The declared rule of ``loss``."""
    if loss == "blip":
        return LearnedRule(loss="blip", learner=blip_learner())
    if loss == "weighted_classification":
        return LearnedRule(loss="weighted_classification", learner=classifier())
    raise KeyError(f"no declared rule learner for loss {loss!r}")


# ------------------------------------------------------------------------------ the fits


@dataclass(frozen=True)
class Measured:
    """One sample's declared fit, its control, and its plug-in comparison.

    Parameters
    ----------
    result : object
        The declared fit.
    truth : tests.studies._learned_rule_law.Truth
        Its harness truth.
    warnings : int
        The solver warnings of the declared fit and its refits.
    control : object or None
        The fold-locality control fit, when asked for.
    control_truth : tests.studies._learned_rule_law.Truth or None
        Its harness truth, the value of the rules it learned on the validation rows.
    control_warnings : int
        The solver warnings of the control and its refits.
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
    control: Any
    control_truth: base.Truth | None
    control_warnings: int
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


def measured(frame: pd.DataFrame, loss: str, fold_seed: int, *, control: bool = False) -> Measured:
    """Fit the declared rule learner, its plug-in comparison, and optionally its control.

    The three fits share the sample and the fold seed, so they share the outer folds; a fit
    whose folds differ stops the replication, because the comparison would then not be paired.
    Rule L4's stops of :func:`tests.studies._learned_rule_law.measured` apply to each fit.
    """
    learner = base.outcome_learner()
    result, truth, warnings = base.measured(frame, learner, LAW, fold_seed, rule=rule(loss))
    plug_in, plug_truth, _ = base.measured(frame, base.outcome_learner(), LAW, fold_seed)
    fits = [plug_in]
    mutated, mutated_truth, mutated_warnings = None, None, 0
    if control:
        mutated, mutated_truth, mutated_warnings = base.measured(
            frame, base.outcome_learner(), LAW, fold_seed, mutation=True, rule=rule(loss)
        )
        fits.append(mutated)
    for other in fits:
        if not _same_folds(result, other):
            raise RuntimeError("a paired fit drew other outer folds than the declared fit")
    return Measured(
        result=result,
        truth=truth,
        warnings=warnings,
        control=mutated,
        control_truth=mutated_truth,
        control_warnings=mutated_warnings,
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


def propensity_floor(result: Any) -> float:
    """The smallest untruncated fitted ``g(a | W)`` of the outer fit, over the rows and arms."""
    return float(np.min(result.nuisance.bounded_propensity((0.0, 1.0))))


# ------------------------------------------------------------------------- the limits


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    """The covariate grid, ``W2 = 0`` then ``W2 = 1``, and the probability mass of each point."""
    grid = base.GRID
    mass = np.full(grid.size, grid[1] - grid[0])
    mass[[0, -1]] /= 2.0
    covariates = np.vstack(
        [np.column_stack([grid, np.full_like(grid, w2)]) for w2 in base.W2_LEVELS]
    )
    return covariates, np.concatenate([mass * 0.25, mass * 0.25])


@cache
def outcome_projection() -> Pipeline:
    """The limit ``Qbar*`` of the outcome learner: its logistic projection under ``P0``.

    Each grid point of ``(A, W)`` carries its mass, its arm probability and each outcome its
    probability, so the weighted log-likelihood is the population one. The ridge is removed.
    """
    covariates, mass = _quadrature()
    design, outcome, weight = [], [], []
    for a in (0.0, 1.0):
        treated = base.g0(covariates[:, 0], covariates[:, 1])
        q = base.qbar0(a, covariates[:, 0], covariates[:, 1], LAW)
        arm = treated if a else 1.0 - treated
        for y, p_y in ((1.0, q), (0.0, 1.0 - q)):
            design.append(np.column_stack([np.full(len(q), a), covariates]))
            outcome.append(np.full(len(q), y))
            weight.append(mass * arm * p_y)
    model = base.outcome_learner().set_params(
        model__C=1e12, model__tol=1e-13, model__max_iter=20_000
    )
    weights = np.concatenate(weight)
    model.fit(
        np.vstack(design), np.concatenate(outcome), model__sample_weight=weights / weights.mean()
    )
    return model


def _pseudo_outcome_law() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The law of the limit pseudo-outcome on the grid.

    Returns the row's grid index, the value ``D = (2a - 1)/g0(a | W) (y - Qbar*(a, W)) +
    Qbar*(1, W) - Qbar*(0, W)`` and its probability mass, for each grid point and each
    ``(a, y)``.
    """
    covariates, mass = _quadrature()
    model = outcome_projection()
    ones = np.ones((covariates.shape[0], 1))
    q1 = np.clip(model.predict_proba(np.hstack([ones, covariates]))[:, 1], 0, 1)
    q0 = np.clip(model.predict_proba(np.hstack([0.0 * ones, covariates]))[:, 1], 0, 1)
    treated = base.g0(covariates[:, 0], covariates[:, 1])
    index, value, weight = [], [], []
    for a in (0.0, 1.0):
        arm = treated if a else 1.0 - treated
        fitted = q1 if a else q0
        true = base.qbar0(a, covariates[:, 0], covariates[:, 1], LAW)
        for y, p_y in ((1.0, true), (0.0, 1.0 - true)):
            index.append(np.arange(covariates.shape[0]))
            value.append((2.0 * a - 1.0) / arm * (y - fitted) + q1 - q0)
            weight.append(mass * arm * p_y)
    return np.concatenate(index), np.concatenate(value), np.concatenate(weight)


@cache
def limit_scores() -> dict[str, np.ndarray]:
    """The limit score minus its threshold of each rule on the grid, and the grid mass.

    ``plug_in``: the blip ``Qbar*(1, W) - Qbar*(0, W)`` of the outcome learner's limit.
    ``blip``: the least-squares projection of ``E[D | W]`` on the basis. With ``g = g0``,
    ``E[D | W] = Qbar0(1, W) - Qbar0(0, W)`` whatever ``Qbar*`` is, so this is the projection
    of the true blip; it is computed here from the law of ``D``, which checks that identity.
    ``weighted_classification``: ``P(D > 0 | W) - 1/2`` of the weighted logistic projection of
    ``1{D > 0}`` on the basis with weights ``|D|``, integrated over ``(A, Y)`` given ``W``
    under the inner limits. Its rule ``1{score > 0}`` is the rule of the fitted probability
    above one half. ``optimal``: the true blip.
    """
    covariates, mass = _quadrature()
    model = outcome_projection()
    ones = np.ones((covariates.shape[0], 1))
    plug_in = (
        model.predict_proba(np.hstack([ones, covariates]))[:, 1]
        - model.predict_proba(np.hstack([0.0 * ones, covariates]))[:, 1]
    )
    index, value, weight = _pseudo_outcome_law()
    rows = covariates[index]
    blip = (
        blip_learner()
        .fit(rows, value, fit__sample_weight=weight / weight.mean())
        .predict(covariates)
    )
    labels = (value > 0.0).astype(float)
    size = weight * np.abs(value)
    projection = classifier().set_params(fit__C=1e12, fit__tol=1e-13, fit__max_iter=50_000)
    projection.fit(rows, labels, fit__sample_weight=size / size.mean())
    classification = projection.predict_proba(covariates)[:, 1] - 0.5
    optimal = base.qbar0(1.0, covariates[:, 0], covariates[:, 1], LAW) - base.qbar0(
        0.0, covariates[:, 0], covariates[:, 1], LAW
    )
    return {
        "plug_in": np.asarray(plug_in, dtype=float),
        "blip": np.asarray(blip, dtype=float),
        "weighted_classification": np.asarray(classification, dtype=float),
        "optimal": np.asarray(optimal, dtype=float),
        "mass": mass,
    }


def limit_rule(name: str) -> dict[float, np.ndarray]:
    """The limit rule ``name`` on the grid, in the shape :func:`base.rule_value` reads."""
    halves = np.split(limit_scores()[name] > 0.0, 2)
    return dict(zip(base.W2_LEVELS, halves, strict=True))


def limit_value(name: str) -> tuple[float, float]:
    """The value and the row-curve variance of the limit rule ``name``."""
    return base.rule_value(limit_rule(name), LAW)


def disagreement(first: str, second: str) -> float:
    """The population share of ``W`` on which two limit rules differ."""
    scores = limit_scores()
    return float(np.sum(scores["mass"] * ((scores[first] > 0.0) != (scores[second] > 0.0))))


def near_threshold(name: str, epsilon: float) -> float:
    """The population share of ``W`` whose limit score lies within ``epsilon`` of its threshold.

    C3 asks that the limit rule have no mass at its threshold; a share of order ``epsilon``
    (a crossing with a nonzero slope) satisfies it.
    """
    scores = limit_scores()
    return float(np.sum(scores["mass"] * (np.abs(scores[name]) < epsilon)))


def thresholds(name: str) -> dict[float, tuple[float, ...]]:
    """The values of ``W1`` at which the limit score of ``name`` changes sign, for each ``W2``."""
    out = {}
    for w2, half in zip(base.W2_LEVELS, np.split(limit_scores()[name], 2), strict=True):
        crossings = np.flatnonzero(np.diff(np.sign(half)) != 0)
        out[w2] = tuple(
            float(
                base.GRID[i] - half[i] * (base.GRID[i + 1] - base.GRID[i]) / (half[i + 1] - half[i])
            )
            for i in crossings
        )
    return out
