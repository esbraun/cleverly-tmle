r"""The law, the learners, the fit and the harness of ``learned-rule-intermediate`` (X11 (j)).

The study measures the learned rule at the treatment with a binary intermediate ``Z`` held at
each level ``z``: ``TMLE(learned_rule=LearnedRule()).fit(..., intermediate="Z")`` reports one
result per level, and each level learns its own fold rules, the plug-in
:math:`d^z_{nv}(w) = 1\{\bar Q_{nv}(1, z, w) > \bar Q_{nv}(0, z, w)\}` (a tie to arm 0). The
target of a level is the data-adaptive
:math:`\tilde\psi^z_{0n} = V^{-1}\sum_v E_0\bar Q_0(d^z_{nv}(W), z, W)`, under the shipped
controlled-direct-effect identification. Beside it the study fits the known rule
``1{W1 > 0}`` through the same fold-evaluated controlled-direct-effect path, the path that
witness W-j1 (``tests/unit/test_learned_rule_intermediate.py``) reduces the learned fit to.

**The law** (:data:`LAW`). RM30's covariates and treatment, ``W1 ~ U(-1, 1)``,
``W2 ~ Bernoulli(0.5)``, ``logit g0(1 | W) = 0.3 W1 - 0.2 W2``, then a binary intermediate after
the treatment and a binary outcome:

.. math::

    \operatorname{logit} P(Z = 1 \mid A, W) = 0.2 + 0.6 A - 1.2 W_1, \qquad
    \operatorname{logit}\bar Q_0(a, z, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + 0.3 z
        + a (0.1 + W_1 - 0.6 z).

At ``z = 0`` the outcome regression is RM30's ``non_exceptional`` one, blip ``0.1 + W1``; at
``z = 1`` the blip is ``-0.5 + W1``. The optimal rules are ``1{W1 > -0.1}`` (55% treated) and
``1{W1 > 0.5}`` (25%), so the two levels' rules differ on 30% of the population, and a fit
that ignored ``Z`` would carry one level's rule to the other. The smallest
``g0(a | W) q0(z | a, W)`` on the support is 0.045, at ``a = 1``, ``z = 0``, ``W1 = -1``.

**The learners.** The ``declared`` outcome learner is RM30's: every pairwise product of the
outcome design ``(A, W1, W2, Z)``, logistic, fitted by maximum likelihood. It holds ``A W1`` and
``A Z``, so it is a correct model and its limit rule at each level is the optimal rule there.
The ``drop_w1`` learner keeps ``(A, W2, Z)`` and every product of them, saturated, so its limit
is the eight cell means :math:`E[Y \mid A, Z, W_2]`, which ``W1`` confounds through both ``g0``
and ``q0``: the untargeted plug-in of its rule is biased at both levels (``targeting_necessity``).
The treatment learner is RM30's; the intermediate learner is a main-terms logistic model of
``(A, W1, W2)``, a correct model.

**The harness** (:func:`harness`) refits each fold's outcome learner on the fit's training rows
through the package's own ``fit_on_rows``, takes its rule at each level, and requires it to equal
the fit's realized rule on every validation row and the fit's stored fold rule of that level on
every row (rule L4). It integrates :math:`\bar Q_0(d(W), z, W)` by the trapezoid rule on RM30's
4,001 points of ``W1`` for each ``W2``. The same quadrature gives the oracle SE, from
:math:`\operatorname{Var} D^*_z(d) = E_W[\bar Q_0(1 - \bar Q_0)/(g_0(d \mid W) q_0(z \mid d, W))]
+ \operatorname{Var}_W \bar Q_0(d(W), z, W)`. The harness also rebuilds each level's curve
longhand and requires its variance to equal the reported one (:data:`EXACTNESS_TOLERANCE`), and
then rebuilds it without the factor :math:`1\{Z = z\}/q` of the clever covariate, the curve of
the ``term_dropped_control`` (witness W-j3's mutation).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.interventions import LearnedRule, Rule
from cleverly.interventions.learned import fold_rules, realized_arms
from cleverly.learners._fitting import predict_mean
from tests.studies import _learned_rule_law as base

#: The one law of the study.
LAW = "intermediate"

#: The levels of the intermediate, in the order the fit reports them.
LEVELS = (0.0, 1.0)

#: The reported parameter of the learned-rule fit at every level.
ESTIMAND = base.ESTIMAND

#: The known rule of the known-rule cell, and its reported parameter.
KNOWN_RULE = "w1>0"
KNOWN_ESTIMAND = f"ey_regime[{KNOWN_RULE}]"

#: The outer folds, the covariates, the quadrature grid and the multiplier: RM30's.
N_FOLDS = base.N_FOLDS
COVARIATES = base.COVARIATES
GRID = base.GRID
W2_LEVELS = base.W2_LEVELS
CRITICAL = base.CRITICAL

#: ``logit P(Z = 1 | A, W) = intercept + a_coefficient A + w1_coefficient W1``.
Z_SCORE = (0.2, 0.6, -1.2)

#: ``logit Qbar0(a, z, W) = 0.2 + 0.5 W1 - 0.3 W2 + Z_MAIN z + a (0.1 + W1 - BLIP_SHIFT z)``.
Z_MAIN = 0.3
BLIP_SHIFT = 0.6

#: The relative tolerance of the longhand variance of each level's curve against the reported
#: one. A draw outside it stops the run (rule L4).
EXACTNESS_TOLERANCE = 1e-8

#: The columns each row carries beside the shared schema. ``oracle_se`` is descriptive.
#: ``rule_rows_checked`` counts the validation rows whose refit rule equals the fit's rule, which
#: is every row of a completed draw, and ``solver_warnings`` the warnings of the fit and its
#: refits (rule L4). ``dropped_std_error`` is the SE of the curve without ``1{Z = z} / q``.
HARNESS_COLUMNS = ("oracle_se", "rule_rows_checked", "solver_warnings", "dropped_std_error")

SOLVER_WARNINGS = base.SOLVER_WARNINGS
HarnessMismatch = base.HarnessMismatch


# --------------------------------------------------------------------------- the law


def g0(w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``P(A = 1 | W)``, RM30's."""
    return base.g0(w1, w2)


def q0(z: float, a: np.ndarray | float, w1: np.ndarray) -> np.ndarray:
    """``P(Z = z | A = a, W)``."""
    intercept, slope_a, slope_w1 = Z_SCORE
    one = expit(intercept + slope_a * np.asarray(a, dtype=float) + slope_w1 * np.asarray(w1))
    return np.asarray(one if z else 1.0 - one, dtype=float)


def blip(z: float, w1: np.ndarray) -> np.ndarray:
    """The blip of ``A`` at level ``z``, on the logit scale."""
    return np.asarray(0.1 + np.asarray(w1, dtype=float) - BLIP_SHIFT * z, dtype=float)


def qbar0(a: np.ndarray | float, z: float, w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``P(Y = 1 | A = a, Z = z, W)``."""
    w1 = np.asarray(w1, dtype=float)
    return np.asarray(
        expit(0.2 + 0.5 * w1 - 0.3 * np.asarray(w2) + Z_MAIN * z + np.asarray(a) * blip(z, w1)),
        dtype=float,
    )


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows, drawn in the order ``W1, W2, A, Z, Y``.

    The first four draws are RM30's (:func:`tests.studies._learned_rule_law.draw`).
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, g0(w1, w2)).astype(float)
    z = rng.binomial(1, q0(1.0, a, w1)).astype(float)
    p = np.where(z > 0.5, qbar0(a, 1.0, w1, w2), qbar0(a, 0.0, w1, w2))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Z": z, "Y": y})


# ------------------------------------------------------------------------- the learners


def declared_learner() -> Pipeline:
    """RM30's learner on the outcome design ``(A, W1, W2, Z)``: a correct model of the law."""
    return base.outcome_learner()


def drop_w1_learner() -> Pipeline:
    """The ``targeting_necessity`` learner: ``(A, W2, Z)`` and every product, saturated."""
    return Pipeline(
        [
            ("keep", ColumnTransformer([("keep", "passthrough", [0, 2, 3])])),
            ("features", PolynomialFeatures(3, interaction_only=True, include_bias=False)),
            ("model", LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)),
        ]
    )


#: The outcome learner of each cell, by name.
LEARNERS: dict[str, Callable[[], Any]] = {
    "declared": declared_learner,
    "drop_w1": drop_w1_learner,
}


def intermediate_learner() -> LogisticRegression:
    """The intermediate learner on ``(A, W1, W2)``: a correct main-terms model."""
    return LogisticRegression(C=1e6, max_iter=1000)


def known_rule_values(frame: pd.DataFrame) -> np.ndarray:
    """The known rule ``1{W1 > 0}`` at the rows of a covariate frame."""
    return np.asarray((frame["W1"] > 0.0).astype(float))


def known_rule() -> Rule:
    """The known rule of the known-rule cell."""
    return Rule(known_rule_values, name=KNOWN_RULE, rule_kind="known")


# ------------------------------------------------------------------------------ the fit


def estimator(learner: Any, fold_seed: int, *, known: bool = False) -> TMLE:
    """The declared estimator: RM30's settings with an intermediate, and rule L2's fold seed.

    ``known=True`` fits the known rule ``1{W1 > 0}`` through the same fold-evaluated path in place
    of the learned rule.
    """
    return TMLE(
        learned_rule=None if known else LearnedRule(),
        interventions=[known_rule()] if known else None,
        n_folds=N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        intermediate_learner=intermediate_learner(),
        random_state=fold_seed,
    )


def fit(frame: pd.DataFrame, learner: Any, fold_seed: int, *, known: bool = False) -> Any:
    """Fit :func:`estimator` to one sample and return the result set, keyed by level."""
    return estimator(learner, fold_seed, known=known).fit(
        frame, outcome="Y", treatment="A", covariates=list(COVARIATES), intermediate="Z"
    )


def untargeted_estimate(result: Any) -> float:
    """One level's fit without its fluctuation: the ``1/V`` fold average of the plug-ins.

    The result of a level holds the outcome regression at that level, so this is RM30's
    :func:`~tests.studies._learned_rule_law.untargeted_estimate`.
    """
    return base.untargeted_estimate(result)


# ------------------------------------------------------------------------ quadrature

#: The trapezoid mass of each grid point of ``W1`` for one level of ``W2``, times the density
#: 1/2 of ``W1`` and the probability 1/2 of the level.
MASS = np.full(GRID.size, GRID[1] - GRID[0])
MASS[[0, -1]] /= 2.0
MASS *= 0.25


def rule_value(rule: Mapping[float, np.ndarray], z: float) -> tuple[float, float]:
    """``Psi_z(d) = E Qbar0(d(W), z, W)`` and ``Var D*_z(d)`` of a rule given on the grid.

    ``rule[w2]`` is the rule at each grid point of ``W1`` for that ``W2``. With
    ``Q = Qbar0(d(W), z, W)``, ``g = g0(d(W) | W)`` and ``q = q0(z | d(W), W)``, the variance is
    ``E[Q (1 - Q) / (g q)] + E[Q^2] - Psi^2``.
    """
    value = second = 0.0
    for w2 in W2_LEVELS:
        treat = np.asarray(rule[w2], dtype=float)
        q = qbar0(treat, z, GRID, w2)
        treated = g0(GRID, w2)
        g = np.where(treat > 0.5, treated, 1.0 - treated)
        density = q0(z, treat, GRID)
        value += float(np.sum(MASS * q))
        second += float(np.sum(MASS * (q * (1.0 - q) / (g * density) + q**2)))
    return value, second - value**2


def optimal_rule(z: float) -> dict[float, np.ndarray]:
    """The optimal rule at level ``z`` on the grid: treat where the blip is positive."""
    return {w2: blip(z, GRID) > 0.0 for w2 in W2_LEVELS}


def known_grid() -> dict[float, np.ndarray]:
    """The known rule ``1{W1 > 0}`` on the grid."""
    return dict.fromkeys(W2_LEVELS, GRID > 0.0)


@cache
def level_limit(z: float) -> dict[str, float]:
    """The declared learner's limit at level ``z``, by quadrature.

    ``truth`` and ``bound``: the value of the optimal rule (the correct learner's limit rule) and
    the SD of its curve. ``treated``: its treated share. ``known_truth`` and ``known_bound``: the
    same for the known rule ``1{W1 > 0}``. ``disagreement``: the mass on which the optimal rules
    of the two levels differ.
    """
    truth, variance = rule_value(optimal_rule(z), z)
    known, known_variance = rule_value(known_grid(), z)
    rules = [optimal_rule(level) for level in LEVELS]
    return {
        "truth": truth,
        "bound": float(np.sqrt(variance)),
        "treated": float(sum(np.sum(MASS * rules[int(z)][w2]) for w2 in W2_LEVELS)),
        "known_truth": known,
        "known_bound": float(np.sqrt(known_variance)),
        "disagreement": float(
            sum(np.sum(MASS * (rules[0][w2] != rules[1][w2])) for w2 in W2_LEVELS)
        ),
    }


def smallest_weight_product() -> float:
    """The smallest ``g0(a | W) q0(z | a, W)`` over the grid, the arms and the levels."""
    out = np.inf
    for w2 in W2_LEVELS:
        for a in (0.0, 1.0):
            g = g0(GRID, w2) if a else 1.0 - g0(GRID, w2)
            for z in LEVELS:
                out = min(out, float(np.min(g * q0(z, a, GRID))))
    return out


@cache
def drop_w1_limit(z: float) -> dict[str, float]:
    """The ``drop_w1`` learner's limit at level ``z``: the cell means of ``(A, Z, W2)``.

    ``rule_value`` is the value of its limit rule, ``plug_in`` the plug-in it converges to, and
    ``bias`` their difference, the untargeted bias that ``targeting_necessity`` reads. The cell
    mean is ``E[g0 q0 Qbar0] / E[g0 q0]`` over ``W1`` at each ``(a, z, W2)``.
    """
    plug_in = value = 0.0
    treated = 0.0
    for w2 in W2_LEVELS:
        means = {}
        for a in (0.0, 1.0):
            g = g0(GRID, w2) if a else 1.0 - g0(GRID, w2)
            weight = MASS * g * q0(z, a, GRID)
            means[a] = float(np.sum(weight * qbar0(a, z, GRID, w2)) / np.sum(weight))
        rule = 1.0 if means[1.0] > means[0.0] else 0.0
        treated += 0.5 * rule
        plug_in += 0.5 * means[rule]
        value += float(np.sum(MASS * qbar0(rule, z, GRID, w2)))
    _, variance = rule_value(dict.fromkeys(W2_LEVELS, np.full(GRID.size, treated > 0.0)), z)
    return {
        "plug_in": plug_in,
        "rule_value": value,
        "bias": plug_in - value,
        "treated": treated,
        "bound": float(np.sqrt(variance)),
    }


# -------------------------------------------------------------------------- the harness


def level_design(arm: float, z: float, covariates: np.ndarray) -> np.ndarray:
    """The outcome design ``(A, W1, W2, Z)`` with every row set to ``arm`` and ``z``."""
    covariates = np.asarray(covariates, dtype=float)
    rows = covariates.shape[0]
    return np.hstack([np.full((rows, 1), arm), covariates, np.full((rows, 1), z)])


def level_blip(model: Any, z: float, covariates: np.ndarray) -> np.ndarray:
    """``Qbar(1, z, W) - Qbar(0, z, W)`` of a fitted outcome model, clipped as the fit clips."""
    treated = np.clip(predict_mean(model, level_design(1.0, z, covariates), "classification"), 0, 1)
    control = np.clip(predict_mean(model, level_design(0.0, z, covariates), "classification"), 0, 1)
    return np.asarray(treated - control, dtype=float)


@dataclass(frozen=True)
class Truth:
    """What the harness computes for one level of one fit.

    Parameters
    ----------
    truth : float
        ``(1/V) sum_v Psi_z(d_v)``, the value of the rules the fit used at the level.
    oracle_se : float
        ``sqrt(V^-2 sum_v Var D*_z(d_v) / n_v)``, by the same quadrature.
    fold_values : tuple of float
        ``Psi_z(d_v)`` for each fold.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    dropped_std_error : float
        The SE of the level's curve rebuilt without the factor ``1{Z = z} / q``.
    """

    truth: float
    oracle_se: float
    fold_values: tuple[float, ...]
    rows_checked: int
    dropped_std_error: float


def _fold_variance(curve: np.ndarray, folds: list[np.ndarray]) -> float:
    """``V^-2 sum_v n_v^-2 sum_{i in v} D_i^2``, the fold-evaluated variance of a curve."""
    return float(sum(np.sum(curve[test] ** 2) / test.size**2 for test in folds) / len(folds) ** 2)


def curve_variances(result: Any, z: float) -> tuple[float, float]:
    """The longhand variance of a level's curve with and without ``1{Z = z} / q``.

    ``H = 1{A = d} 1{Z = z} / (g(A | W) q(z | A, W))`` at the fit's bounds, and the curve is
    ``H (Y - Qbar*) + Qbar*(d, z, W) - psi_v`` on each validation fold, with ``psi_v`` the fold's
    mean of ``Qbar*(d, z, W)``. The mutation drops ``1{Z = z} / q`` from ``H`` (W-j3).
    """
    data = result.data
    nuisance = result.nuisance
    targeted = result.fluctuations["regime"].targeted
    lower, upper = result.config.g_bounds
    a = np.asarray(data.treatment, dtype=float)
    rule = realized_arms(nuisance.regimes).astype(float)
    g1 = np.clip(np.asarray(nuisance.propensity.arm(1.0), dtype=float), lower, upper)
    g = np.where(a == 1.0, g1, 1.0 - g1)
    density = np.asarray(nuisance.intermediate_density(z, result.config.missingness_bound))
    q = np.where(a == 1.0, density[:, 1], density[:, 0])
    at_level = (np.asarray(data.intermediate) == z).astype(float)
    on_rule = (a == rule).astype(float)
    y = np.asarray(nuisance.scaler.scale(data.outcome), dtype=float)
    residual = y - np.asarray(targeted.observed, dtype=float)
    q_rule = np.where(rule == 1.0, targeted.arms[1.0], targeted.arms[0.0])
    folds = [np.asarray(test) for _, test in nuisance.folds]
    centred = np.empty(data.n)
    for test in folds:
        centred[test] = q_rule[test] - np.mean(q_rule[test])
    scale = float(nuisance.scaler.upper - nuisance.scaler.lower)
    with_factor = _fold_variance(on_rule * at_level / (g * q) * residual + centred, folds)
    without = _fold_variance(on_rule / g * residual + centred, folds)
    return with_factor * scale**2, without * scale**2


def harness(results: Any, learner: Any, *, known: bool = False) -> dict[float, Truth]:
    """Refit each fold's rule, check it at each level, and integrate each level's value.

    The refit is the package's ``fit_on_rows`` on the fold's training rows with the fit's own
    design ``(A, W1, W2, Z)``, outcome and weights, so the deterministic learner reproduces the
    fitted model. The design :func:`level_design` builds is checked against the data's own
    ``counterfactual_design`` at every arm and level. A known-rule fit refits nothing: its rule is
    ``1{W1 > 0}`` at both levels, which the harness checks against the fit's rule.

    Raises
    ------
    HarnessMismatch
        When a refit rule differs from the fit's realized rule on a validation row or from the
        stored fold rule of its level on any row (rule L4), when the designs differ, or when a
        level's longhand variance differs from the reported one by more than
        :data:`EXACTNESS_TOLERANCE`.
    """
    first = results[LEVELS[0]]
    data = first.data
    covariates = np.asarray(data.covariates, dtype=float)
    for arm in (0.0, 1.0):
        for z in LEVELS:
            expected = data.counterfactual_design(arm, intermediate_value=z)
            if not np.array_equal(level_design(arm, z, covariates), expected):
                raise HarnessMismatch(f"the study's design at ({arm}, {z}) is not the package's")
    folds = list(first.nuisance.folds)
    models = []
    if not known:
        design = np.asarray(data.treatment_design(include_intermediate=True), dtype=float)
        outcome = np.asarray(data.outcome, dtype=float)
        weights = np.asarray(data.weights, dtype=float)
        for train, _ in folds:
            models.append(
                fit_on_rows(
                    learner, design, outcome, weights, np.asarray(train), "classification", None
                )
            )
    grid = {w2: np.column_stack([GRID, np.full_like(GRID, w2)]) for w2 in W2_LEVELS}
    out = {}
    for z in LEVELS:
        result = results[z]
        realized = realized_arms(result.nuisance.regimes).astype(float)
        stored = None if known else fold_rules(result.nuisance)
        if not known and stored is None:
            raise HarnessMismatch(f"the fit stores no fold rules at level {z}")
        values: list[float] = []
        variances: list[float] = []
        sizes: list[int] = []
        checked = 0
        for fold, (_, test) in enumerate(folds):
            test = np.asarray(test)
            if known:
                on_rows = (covariates[:, 0] > 0.0).astype(float)
                on_grid = known_grid()
            else:
                on_rows = (level_blip(models[fold], z, covariates) > 0.0).astype(float)
                assert stored is not None
                if not np.array_equal(on_rows, stored[:, fold].astype(float)):
                    raise HarnessMismatch(f"fold {fold}: the refit rule at z = {z} is not stored")
                on_grid = {w2: level_blip(models[fold], z, p) > 0.0 for w2, p in grid.items()}
            disagree = int(np.sum(on_rows[test] != realized[test]))
            if disagree:
                raise HarnessMismatch(
                    f"fold {fold}: the refit rule at z = {z} differs from the fit's rule on "
                    f"{disagree} of {len(test)} validation rows"
                )
            checked += len(test)
            value, variance = rule_value(on_grid, z)
            values.append(value)
            variances.append(variance)
            sizes.append(len(test))
        reported = float(result.estimates[KNOWN_ESTIMAND if known else ESTIMAND].variance)
        with_factor, without = curve_variances(result, z)
        if abs(with_factor - reported) > EXACTNESS_TOLERANCE * reported:
            raise HarnessMismatch(
                f"the longhand variance at z = {z} is {with_factor!r}, the fit's {reported!r}"
            )
        oracle = float(np.sqrt(sum(v / m for v, m in zip(variances, sizes, strict=True))))
        out[z] = Truth(
            truth=float(np.mean(values)),
            oracle_se=oracle / len(values),
            fold_values=tuple(values),
            rows_checked=checked,
            dropped_std_error=float(np.sqrt(without)),
        )
    return out


@dataclass(frozen=True)
class Measured:
    """One fit at both levels, its harness and its solver warnings.

    Parameters
    ----------
    results : TMLEResultSet
        The fit, keyed by level.
    truths : dict
        The harness of each level.
    solver_warnings : int
        The solver warnings of the fit and its refits.
    known : bool
        Whether the fit is of the known rule.
    """

    results: Any
    truths: dict[float, Truth]
    solver_warnings: int
    known: bool

    def estimate(self, z: float) -> Any:
        """The level's reported estimate."""
        return self.results[z].estimates[KNOWN_ESTIMAND if self.known else ESTIMAND]


def measured(
    frame: pd.DataFrame, learner_name: str, fold_seed: int, *, known: bool = False
) -> Measured:
    """Fit one sample at both levels, run its harness, and count the solver warnings.

    Rule L4 stops the run on a non-finite estimate or SE, on a harness mismatch, on a fit that
    reports another parameter or other levels, and on a key that does not read the intermediate.
    """
    learner = LEARNERS[learner_name]()
    with base.solver_warning_count() as count:
        results = fit(frame, learner, fold_seed, known=known)
        truths = harness(results, learner, known=known)
    if tuple(results) != LEVELS:
        raise HarnessMismatch(f"the fit reports the levels {tuple(results)}, not {LEVELS}")
    name = KNOWN_ESTIMAND if known else ESTIMAND
    for z in LEVELS:
        result = results[z]
        if tuple(result.estimates) != (name,):
            raise HarnessMismatch(f"the fit reports {tuple(result.estimates)}, not {name}")
        if result.config.support_key.missing != ("intermediate",):
            raise HarnessMismatch(f"the key reads {result.config.support_key.missing}")
        estimate = result.estimates[name]
        if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
            raise RuntimeError(f"a non-finite estimate or SE (rule L4): {estimate}")
    return Measured(results=results, truths=truths, solver_warnings=count[0], known=known)


def harness_columns(fitted: Measured, z: float) -> dict[str, Any]:
    """The harness columns of one level's row."""
    truth = fitted.truths[z]
    return {
        "oracle_se": truth.oracle_se,
        "rule_rows_checked": truth.rows_checked,
        "solver_warnings": fitted.solver_warnings,
        "dropped_std_error": truth.dropped_std_error,
    }
