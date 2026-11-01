r"""The law, the learners, the fit and the truth harness of ``learned-rule-arms`` (X11 PR H, (d)).

The study measures the cross-fitted learned-rule value over three arms: the plug-in rule
:math:`d_{nv}(w) = \arg\max_a \bar Q_{nv}(a, w)` of each fold's outcome model, a tie going to the
lowest arm code, evaluated on the fold's validation rows and averaged with weight ``1/V``.

**The law** (:data:`LAW`). ``W1 ~ U(-1, 1)`` and ``W2 ~ Bernoulli(0.5)``. The treatment has the
three arms 0, 1 and 2, with the multinomial logit mechanism

.. math::

    g_0(a \mid W) \propto \exp\{\eta_a(W)\},\quad \eta_0 = 0,\quad
    \eta_1 = 0.7 W_1 + 0.2 W_2,\quad \eta_2 = -0.4 W_1 + 0.2 W_2,

whose smallest arm probability on the support is 0.166 (:func:`g0`). The outcome is binary,
``logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + b_a(W)``, with the arm blips

.. math::

    b_0 = 0,\quad b_1 = -1 + 6 W_1,\quad
    b_2 = -1.75 - 6 W_1 - W_2 - 4 (W_1^2 - 1/3).

Arm 2 is best at low ``W1``, arm 1 at high ``W1`` and arm 0 between them. The quadratic term of
arm 2's blip is the term the ``missing_quadratic`` learner omits.

**The learners.** :class:`ArmLogistic` is a logistic regression on a fixed set of products of the
outcome design ``(A1, A2, W1, W2)``, where ``A1`` and ``A2`` are the package's arm indicators.
It declares itself parametric to :func:`cleverly._support.learner_family`.

``declared``
    Every arm's intercept and slopes in ``W1`` and ``W2``, and ``A2 W1^2``: a correct model, so
    its population limit is ``Qbar0`` and its limit rule is the optimal rule.
``missing_quadratic``
    The same without ``A2 W1^2``. Its population limit (:func:`population_coefficients`, by
    the population score equations on the quadrature grid) is not ``Qbar0``, and its limit
    rule differs from the optimal rule on 7.0% of the population.
``drop_w1``
    The arm indicators, ``W2`` and their products, without ``W1``: saturated in ``(A, W2)``, so
    its limit is the six cell means ``E[Y | A, W2]``, which ``W1`` confounds.

The treatment learner is ``LogisticRegression(C=numpy.inf, max_iter=5000)`` on ``(W1, W2)``, the
covariates of ``g0``'s linear scores: scikit-learn's multinomial model fitted by maximum
likelihood, a correct model. ``C=numpy.inf`` is the unpenalized fit; scikit-learn 1.8
deprecated ``penalty=None`` for it.

**The target** of a fit is data-adaptive: ``V^-1 sum_v Psi_{d_v}(P0)`` with
``Psi_d(P0) = E Qbar0(d(W), W)``. :func:`harness` refits each fold's outcome learner on the fit's
training rows through the package's own ``fit_on_rows``, takes the argmax of its predictions at
the three arms, and requires it to equal the fit's realized rule on every validation row and the
fit's stored fold rule (``fold_rules``) on every row, in every fold (rule L4). It then integrates
``Qbar0(d_v(W), W)`` by the trapezoid rule on the RM30 grid of 4,001 points of ``W1`` for each
``W2``. The same quadrature gives the oracle SE and the limit rules of :func:`limit_rule`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.interventions import LearnedRule
from cleverly.interventions.learned import fold_rules, realized_arms
from cleverly.learners._fitting import predict_mean
from tests.studies import _learned_rule_law as base

#: The one law of the study.
LAW = "three_arms"

#: The arm codes, in the package's sorted code order: the internal arm index is the code.
ARMS = (0, 1, 2)

#: The outer folds of every fit, RM30's ten.
N_FOLDS = base.N_FOLDS

#: The reported parameter of every fit.
ESTIMAND = base.ESTIMAND

#: The covariates, and the quadrature grid of ``W1`` with the two levels of ``W2``.
COVARIATES = base.COVARIATES
GRID = base.GRID
W2_LEVELS = base.W2_LEVELS
CRITICAL = base.CRITICAL

#: The linear scores of ``g0``, as ``(intercept, W1, W2)`` for arms 1 and 2; arm 0's is zero.
G_SCORES = {1: (0.0, 0.7, 0.2), 2: (0.0, -0.4, 0.2)}

#: The outcome's shared logit ``0.2 + 0.5 W1 - 0.3 W2``, and each arm's blip as
#: ``(intercept, W1, W2, quadratic)``, the quadratic term being ``-q (W1^2 - 1/3)``.
OUTCOME_BASE = (0.2, 0.5, -0.3)
BLIPS: dict[int, tuple[float, float, float, float]] = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (-1.0, 6.0, 0.0, 0.0),
    2: (-1.75, -6.0, -1.0, 4.0),
}

#: The columns each row carries beside the shared schema (RM30's): the descriptive oracle SE,
#: the validation rows whose refit rule equals the fit's, and the solver warnings.
HARNESS_COLUMNS = base.HARNESS_COLUMNS

SOLVER_WARNINGS = base.SOLVER_WARNINGS
HarnessMismatch = base.HarnessMismatch


# --------------------------------------------------------------------------- the law


def g0(w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``P(A = a | W)`` for each arm, as a ``(3, m)`` array in arm-code order."""
    w1 = np.asarray(w1, dtype=float)
    w2 = np.broadcast_to(np.asarray(w2, dtype=float), w1.shape)
    scores = [np.zeros_like(w1)]
    for arm in (1, 2):
        intercept, slope, level = G_SCORES[arm]
        scores.append(intercept + slope * w1 + level * w2)
    return np.asarray(softmax(np.vstack(scores), axis=0), dtype=float)


def blip(arm: int, w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """Arm ``arm``'s blip ``b_a(W)`` on the logit scale."""
    intercept, slope, level, quadratic = BLIPS[int(arm)]
    w1 = np.asarray(w1, dtype=float)
    return np.asarray(
        intercept + slope * w1 + level * np.asarray(w2) - quadratic * (w1**2 - 1.0 / 3.0),
        dtype=float,
    )


def qbar0(arm: int, w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``P(Y = 1 | A = arm, W)``."""
    w1 = np.asarray(w1, dtype=float)
    intercept, slope, level = OUTCOME_BASE
    return np.asarray(
        expit(intercept + slope * w1 + level * np.asarray(w2) + blip(arm, w1, w2)), dtype=float
    )


def qbar0_rows(arms: np.ndarray, w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``Qbar0(arms[i], W_i)`` at each point, for a per-point arm code."""
    arms = np.asarray(arms, dtype=int)
    stacked = np.vstack([qbar0(arm, w1, w2) for arm in ARMS])
    return np.asarray(np.take_along_axis(stacked, arms[None, :], axis=0)[0], dtype=float)


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows from :data:`LAW`, with columns ``W1, W2, A, Y``.

    The arm is the first ``a`` whose cumulative probability exceeds one uniform draw.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    cumulative = np.cumsum(g0(w1, w2), axis=0)
    u = rng.uniform(0.0, 1.0, n)
    a = np.sum(u[None, :] >= cumulative[:-1], axis=0).astype(float)
    y = rng.binomial(1, qbar0_rows(a, w1, w2)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


# ------------------------------------------------------------------------- the learners

#: The products of the outcome design ``(A1, A2, W1, W2)`` each learner fits, by name.
TERMS = {
    "declared": ("A1", "A2", "W1", "W2", "A1W1", "A1W2", "A2W1", "A2W2", "A2W1^2"),
    "missing_quadratic": ("A1", "A2", "W1", "W2", "A1W1", "A1W2", "A2W1", "A2W2"),
    "drop_w1": ("A1", "A2", "W2", "A1W2", "A2W2"),
}


def features(design: np.ndarray, terms: tuple[str, ...]) -> np.ndarray:
    """The columns ``terms`` of the outcome design ``(A1, A2, W1, W2)``."""
    design = np.asarray(design, dtype=float)
    a1, a2, w1, w2 = design.T
    columns = {
        "A1": a1,
        "A2": a2,
        "W1": w1,
        "W2": w2,
        "A1W1": a1 * w1,
        "A1W2": a1 * w2,
        "A2W1": a2 * w1,
        "A2W2": a2 * w2,
        "A2W1^2": a2 * w1**2,
    }
    return np.column_stack([columns[term] for term in terms])


class ArmLogistic(ClassifierMixin, BaseEstimator):
    """A logistic outcome model on a fixed set of products of ``(A1, A2, W1, W2)``.

    Parameters
    ----------
    terms : str
        The key of :data:`TERMS`: ``"declared"``, ``"missing_quadratic"`` or ``"drop_w1"``.

    Notes
    -----
    The model is fitted by maximum likelihood (``C=numpy.inf``, ``tol=1e-10``). Its form is
    fixed before the data are seen, so it declares itself parametric to
    :func:`cleverly._support.learner_family`, which would read a ``Pipeline`` as adaptive.
    """

    cleverly_learner_family: ClassVar[str] = "parametric"

    def __init__(self, terms: str = "declared") -> None:
        self.terms = terms

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> ArmLogistic:
        """Fit the logistic model to the product columns of ``X``."""
        self.model_ = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10)
        self.model_.fit(features(X, TERMS[self.terms]), y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """The class probabilities at the product columns of ``X``."""
        return np.asarray(self.model_.predict_proba(features(X, TERMS[self.terms])), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The more probable class at each row."""
        return np.asarray(self.model_.predict(features(X, TERMS[self.terms])))


def outcome_learner(terms: str = "declared") -> ArmLogistic:
    """The outcome learner of one cell."""
    if terms not in TERMS:
        raise KeyError(f"no outcome learner {terms!r}")
    return ArmLogistic(terms)


def treatment_learner() -> LogisticRegression:
    """The declared treatment learner on ``(W1, W2)``: the correct multinomial model, by MLE."""
    return LogisticRegression(C=np.inf, max_iter=5000)


# ------------------------------------------------------------------------------ the fit


def estimator(learner: Any, fold_seed: int) -> TMLE:
    """The declared estimator: RM30's settings on three arms, with the fold seed of rule L2."""
    return TMLE(
        learned_rule=LearnedRule(),
        n_folds=N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=treatment_learner(),
        random_state=fold_seed,
    )


def fit(frame: pd.DataFrame, learner: Any, fold_seed: int) -> Any:
    """Fit :func:`estimator` to one sample and return the single result."""
    return (
        estimator(learner, fold_seed)
        .fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES))
        .single()
    )


def arm_design(arm: int, covariates: np.ndarray) -> np.ndarray:
    """The outcome design ``(A1, A2, W1, W2)`` with every row set to ``arm``."""
    covariates = np.asarray(covariates, dtype=float)
    ones = np.ones((covariates.shape[0], 1))
    return np.hstack([float(arm == 1) * ones, float(arm == 2) * ones, covariates])


def argmax_rule(model: Any, covariates: np.ndarray) -> np.ndarray:
    """The plug-in rule of a fitted outcome model at ``covariates``: lowest code on a tie.

    The predictions are the package's ``predict_mean``, clipped to the unit interval as the fit
    clips them, and ``numpy.argmax`` returns the first maximum.
    """
    predictions = np.vstack(
        [
            np.clip(predict_mean(model, arm_design(arm, covariates), "classification"), 0.0, 1.0)
            for arm in ARMS
        ]
    )
    return np.asarray(np.argmax(predictions, axis=0), dtype=np.int8)


def untargeted_estimate(result: Any) -> float:
    """The fit without its fluctuation: the ``1/V`` fold average of the initial plug-ins.

    Each fold's plug-in is the mean of ``Qbar_v(d_i, W_i)`` over its validation rows, read off
    the out-of-fold outcome regression at each row's realized arm.
    """
    nuisance = result.nuisance
    scaler = nuisance.scaler
    arms = np.vstack(
        [
            np.asarray(scaler.unscale_levels(nuisance.outcome.arms[level]), dtype=float)
            for level in nuisance.outcome.levels
        ]
    )
    rule = realized_arms(nuisance.regimes).astype(int)
    plug_in = np.take_along_axis(arms, rule[None, :], axis=0)[0]
    value = float(np.mean([np.mean(plug_in[np.asarray(test)]) for _, test in nuisance.folds]))
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate (rule L4): {value}")
    return value


# -------------------------------------------------------------------------- the harness


def rule_value(rule: Mapping[float, np.ndarray]) -> tuple[float, float]:
    """``Psi_d(P0)`` and ``Var_P0 D*(d, Qbar0, g0)`` of a rule given on the grid.

    ``rule[w2]`` holds the arm code at each grid point of ``W1``. With ``Q = Qbar0(d(W), W)`` and
    ``g = g0(d(W) | W)``, the variance is ``E[Q (1 - Q) / g] + E[Q^2] - Psi^2``; both integrals
    use the trapezoid rule on :data:`GRID`, with the density ``1/2`` of ``W1`` and the
    probability ``1/2`` of each ``W2``.
    """
    value = 0.0
    second = 0.0
    for w2 in W2_LEVELS:
        arms = np.asarray(rule[w2], dtype=int)
        q = qbar0_rows(arms, GRID, w2)
        g = np.take_along_axis(g0(GRID, w2), arms[None, :], axis=0)[0]
        value += 0.5 * np.trapezoid(q, GRID) / 2.0
        second += 0.5 * np.trapezoid(q * (1.0 - q) / g + q**2, GRID) / 2.0
    return float(value), float(second - value**2)


def grid_rule(model: Any) -> dict[float, np.ndarray]:
    """A fitted model's plug-in rule on the grid, for each ``W2``."""
    return {
        w2: argmax_rule(model, np.column_stack([GRID, np.full_like(GRID, w2)])) for w2 in W2_LEVELS
    }


@dataclass(frozen=True)
class Truth:
    """What the harness computes for one fit.

    Parameters
    ----------
    truth : float
        ``(1/V) sum_v Psi_{d_v}(P0)``, the value of the rules the fit used.
    oracle_se : float
        ``sqrt(V^-2 sum_v Var_P0 D*(d_v, Qbar0, g0) / n_v)``, by the same quadrature.
    fold_values : tuple of float
        ``Psi_{d_v}(P0)`` for each fold.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    """

    truth: float
    oracle_se: float
    fold_values: tuple[float, ...]
    rows_checked: int


def harness(result: Any, learner: Any) -> Truth:
    """Refit each fold's rule, check it against the fit's rules, and integrate its value.

    The refit is the package's ``fit_on_rows`` on the fold's training rows with the fit's own
    design, outcome and weights, so a deterministic learner reproduces the fitted model, and
    :func:`argmax_rule` takes its argmax. The design :func:`arm_design` builds is checked against
    the data's own ``counterfactual_design`` at every arm.

    Raises
    ------
    HarnessMismatch
        When a refit rule differs from the fit's realized rule on a validation row, or from the
        fit's stored rule of that fold on any row (rule L4), or when the designs differ.
    """
    data = result.data
    nuisance = result.nuisance
    covariates = np.asarray(data.covariates, dtype=float)
    for arm in ARMS:
        if not np.array_equal(arm_design(arm, covariates), data.counterfactual_design(arm)):
            raise HarnessMismatch(f"the study's design at arm {arm} is not the package's")
    realized = realized_arms(nuisance.regimes)
    stored = fold_rules(nuisance)
    if stored is None:
        raise HarnessMismatch("the fit stores no fold rules")
    design = np.asarray(data.treatment_design(), dtype=float)
    outcome = np.asarray(data.outcome, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    values: list[float] = []
    variances: list[float] = []
    sizes: list[int] = []
    checked = 0
    for fold, (train, test) in enumerate(nuisance.folds):
        test = np.asarray(test)
        model = fit_on_rows(
            learner, design, outcome, weights, np.asarray(train), "classification", None
        )
        refit = argmax_rule(model, covariates)
        disagree = int(np.sum(refit[test] != realized[test]))
        if disagree:
            raise HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        elsewhere = int(np.sum(refit != stored[:, fold]))
        if elsewhere:
            raise HarnessMismatch(
                f"fold {fold}: the refit rule differs from the stored fold rule on {elsewhere} "
                f"of {len(refit)} rows"
            )
        checked += len(test)
        value, variance = rule_value(grid_rule(model))
        values.append(value)
        variances.append(variance)
        sizes.append(len(test))
    folds = len(values)
    oracle = float(np.sqrt(sum(v / m for v, m in zip(variances, sizes, strict=True))) / folds)
    return Truth(
        truth=float(np.mean(values)),
        oracle_se=oracle,
        fold_values=tuple(values),
        rows_checked=checked,
    )


# ------------------------------------------------------------------- the limit rules


@cache
def population_coefficients(terms: str, points: int = GRID.size) -> np.ndarray:
    """The population limit of the learner ``terms``: the root of its population score equations.

    Each grid point of ``(W1, W2)`` carries its trapezoid mass, each arm its probability under
    ``g0`` and the outcome its mean ``Qbar0``, so ``E[X (Y - expit(X beta))] = 0`` is solved
    exactly for the logistic model by Newton's method (the score is linear in ``Y``, so the
    outcome's mean replaces it). The coefficients follow scikit-learn's layout: the intercept
    first, then the columns of :data:`TERMS` ``[terms]``.
    """
    grid = np.linspace(-1.0, 1.0, points)
    mass = np.full(points, 2.0 / (points - 1))
    mass[[0, -1]] /= 2.0
    mass *= 0.25  # the density 1/2 of W1, and the probability 1/2 of each W2
    rows, targets, weights = [], [], []
    for w2 in W2_LEVELS:
        covariates = np.column_stack([grid, np.full_like(grid, w2)])
        probabilities = g0(grid, w2)
        for arm in ARMS:
            columns = features(arm_design(arm, covariates), TERMS[terms])
            rows.append(np.column_stack([np.ones(points), columns]))
            targets.append(qbar0(arm, grid, w2))
            weights.append(mass * probabilities[arm])
    x = np.vstack(rows)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        mean = expit(x @ beta)
        score = x.T @ (weight * (target - mean))
        information = (x * (weight * mean * (1.0 - mean))[:, None]).T @ x
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-13:
            beta.setflags(write=False)
            return beta
    raise RuntimeError(f"the population score equations of {terms} did not converge")


def limit_predictions(terms: str, w1: np.ndarray, w2: float) -> np.ndarray:
    """The learner's population limit ``Qbar_1(a, W)`` at each arm, as a ``(3, m)`` array."""
    beta = population_coefficients(terms)
    covariates = np.column_stack([np.asarray(w1, dtype=float), np.full(np.size(w1), w2)])
    return np.vstack(
        [
            expit(beta[0] + features(arm_design(arm, covariates), TERMS[terms]) @ beta[1:])
            for arm in ARMS
        ]
    )


def limit_rule(terms: str) -> dict[float, np.ndarray]:
    """The argmax rule of the learner's population limit on the grid, lowest code on a tie.

    For ``declared`` the limit is ``Qbar0``, and the rule is the optimal one.
    """
    return {
        w2: np.asarray(np.argmax(limit_predictions(terms, GRID, w2), axis=0), dtype=np.int8)
        for w2 in W2_LEVELS
    }


def optimal_rule() -> dict[float, np.ndarray]:
    """``argmax_a Qbar0(a, W)`` on the grid."""
    return {
        w2: np.asarray(
            np.argmax(np.vstack([qbar0(arm, GRID, w2) for arm in ARMS]), axis=0), dtype=np.int8
        )
        for w2 in W2_LEVELS
    }


def limit_rule_at(terms: str, covariates: np.ndarray) -> np.ndarray:
    """The limit rule of the learner ``terms`` at the rows ``covariates``, lowest code on a tie."""
    covariates = np.asarray(covariates, dtype=float)
    out = np.empty(covariates.shape[0], dtype=np.int8)
    for w2 in W2_LEVELS:
        here = covariates[:, 1] == w2
        out[here] = np.argmax(limit_predictions(terms, covariates[here, 0], w2), axis=0)
    return out


# ------------------------------------------------------------------- one measured fit


@dataclass(frozen=True)
class Measured:
    """One fit with its truth, its margin diagnostics and its solver-warning count.

    Parameters
    ----------
    result : Any
        The single result of the fit.
    truth : Truth
        The harness of the fit.
    solver_warnings : int
        The solver warnings of the fit and of its refits.
    """

    result: Any
    truth: Truth
    solver_warnings: int


def measured(frame: pd.DataFrame, learner: Any, fold_seed: int) -> Measured:
    """Fit one sample, run its harness, and count the solver warnings of both.

    Rule L4 stops the run on a non-finite estimate or standard error and on a harness mismatch:
    each raises an error that the replication loops do not record.
    """
    with base.solver_warning_count() as count:
        result = fit(frame, learner, fold_seed)
        truth = harness(result, learner)
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    return Measured(result=result, truth=truth, solver_warnings=count[0])


#: The near-tie threshold of the margin diagnostic, on the probability scale of the outcome
#: regression: the threshold of the declared gate on ``P(gap < c)`` (the design record).
NEAR_TIE = 0.01


def margin_columns(result: Any, terms: str) -> dict[str, float]:
    """The margin diagnostic of one fit against the limit rule of its learner ``terms``.

    - ``limit_disagreement``: the share of rows whose realized rule is not the limit rule there;
    - ``near_tie_share``: the share of rows whose own fold's top-two gap of the outcome
      regression is below :data:`NEAR_TIE`, the gap the record's ``gap_quantiles`` summarize;
    - ``min_arm_share``: the smallest share of rows the realized rule assigns one arm.
    """
    nuisance = result.nuisance
    realized = realized_arms(nuisance.regimes)
    covariates = np.asarray(result.data.covariates, dtype=float)
    predictions = np.sort(
        np.vstack(
            [
                np.asarray(nuisance.outcome.arms[level], dtype=float)
                for level in nuisance.outcome.levels
            ]
        ),
        axis=0,
    )
    gap = predictions[-1] - predictions[-2]
    shares = np.bincount(realized, minlength=len(ARMS)) / realized.size
    return {
        "limit_disagreement": float(np.mean(realized != limit_rule_at(terms, covariates))),
        "near_tie_share": float(np.mean(gap < NEAR_TIE)),
        "min_arm_share": float(np.min(shares)),
    }
