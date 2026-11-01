r"""The laws, the fit, the truth and the quadrature of the value of the optimal rule (X11 (f)).

The ``learned-rule-optimal-value`` study (:mod:`tests.studies.learned_rule_optimal`) reads the
CV-TMLE of ``ey_optimal_rule[learned rule]``, whose target is the fixed parameter
:math:`\Psi_{d_0}(P_0) = E_0\bar Q_0(d_0(W), W)` with
:math:`d_0(w) = 1\{\bar Q_0(1, w) > \bar Q_0(0, w)\}`, a tie going to arm 0. Every law keeps the
RM30 covariates, treatment mechanism and outcome base of :mod:`tests.studies._learned_rule_law`
(``W1 ~ U(-1, 1)``, ``W2 ~ Bernoulli(0.5)``, ``logit g0 = 0.3 W1 - 0.2 W2``,
``logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a b(W)``) and differs in the blip ``b``:

``non_exceptional``
    RM30's ``0.1 + W1``. The declared interaction learner is correct, so its limit rule is
    :math:`d_0` (condition C3f), no unit ties (C7), and the blip has a bounded density at 0, so
    the margin exponent is 1 (C8).
``wrong_limit``
    The even blip ``3 (W1^2 - 0.5)``. The declared learner's limit blip is a negative constant
    at each ``W2``, so its limit rule treats nobody while :math:`d_0` treats ``|W1| > 0.707``:
    the cross-validated target converges to :math:`\Psi_{d_1} < \Psi_{d_0}`. The gap is 0.0452,
    3.29 standard errors of :math:`d_1`'s curve at n = 2,500 (W-f2).
``partially_exceptional``
    The hinge ``(W1 - 0.2)_+``, zero on ``W1 <= 0.2``, a set of mass 0.6, so C7 fails there.
    Its outcome learner adds the product ``A (W1 - 0.2)_+`` to the declared products, so the
    model is correct and the fit's rule converges to :math:`d_0` off the tie set.

The truth :math:`\Psi_{d_0}` is one number per law, by the trapezoid rule on RM30's 4,001
points of ``W1`` for each ``W2``. The harness also refits each fold's rule (rule L4) and
integrates its value, so each draw carries the data-adaptive
:math:`\tilde\psi_{0n} = V^{-1}\sum_v \Psi_{d_{nv}}(P_0)` and the regret
:math:`\Psi_{d_0} - \tilde\psi_{0n} \ge 0` beside the fixed truth.

The population limit of a logistic outcome learner solves its population score equations; the
quadrature rows of :func:`limit_blip` carry the trapezoid weight, the probability of ``W2``,
``g0(a | W)`` and ``P(Y = y | a, W)``, so a weighted fit on them is that limit up to the
quadrature. ``tests/unit/test_learned_rule_optimal.py`` (W-f2) and
``tests/unit/test_learned_rule_optimal_design.py`` both read these functions, so the declared
control and the witness share one computation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_law as base

#: The reported parameter of every fit.
ESTIMAND = "ey_optimal_rule[learned rule]"

#: The target the estimator is asked for.
TARGET = "ey_optimal_rule"

#: The amplitude of the ``wrong_limit`` blip ``c (W1^2 - 0.5)``: W-f2's declared ``c = 3``.
WRONG_LIMIT_AMPLITUDE = 3.0

#: The knot and amplitude of the ``partially_exceptional`` blip ``c (W1 - knot)_+``: the plan's
#: example, with ``c = 1`` declared before any draw (impl review, study-design answer 4).
HINGE_KNOT = 0.2
HINGE_AMPLITUDE = 1.0

#: The blip ``b(W)`` of each law on the logit scale, as a function of ``W1``.
BLIPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "non_exceptional": base.BLIPS["non_exceptional"],
    "wrong_limit": lambda w1: WRONG_LIMIT_AMPLITUDE * (w1**2 - 0.5),
    "partially_exceptional": lambda w1: HINGE_AMPLITUDE * np.maximum(w1 - HINGE_KNOT, 0.0),
}

#: The columns each row carries beside the shared schema. ``oracle_se`` (of the fold rules'
#: curves), ``rule_value`` (the data-adaptive target) and ``regret`` are descriptive, and so is
#: ``rule_disagreement``, the record's share of rows on which the ten fold rules do not all
#: agree. ``rule_rows_checked`` counts the validation rows whose refit rule equals the fit's
#: rule, and ``solver_warnings`` the solver warnings of the fit and its refits (rule L4).
HARNESS_COLUMNS = (
    "oracle_se",
    "rule_rows_checked",
    "solver_warnings",
    "rule_value",
    "regret",
    "rule_disagreement",
)


# --------------------------------------------------------------------------- the laws


def blip(law: str, w1: np.ndarray) -> np.ndarray:
    """The law's blip ``b(W)`` on the logit scale."""
    return np.asarray(BLIPS[law](np.asarray(w1, dtype=float)), dtype=float)


def qbar0(a: Any, w1: np.ndarray, w2: Any, law: str) -> np.ndarray:
    """``P(Y = 1 | A = a, W)`` of the named law."""
    w1 = np.asarray(w1, dtype=float)
    return np.asarray(
        expit(0.2 + 0.5 * w1 - 0.3 * np.asarray(w2) + np.asarray(a) * blip(law, w1)), dtype=float
    )


def law_qbar(law: str) -> Callable[[Any, np.ndarray, float], np.ndarray]:
    """:func:`qbar0` of ``law`` as the ``qbar(a, w1, w2)`` the quadrature reads."""
    return lambda a, w1, w2: qbar0(a, w1, w2, law)


def wrong_limit_qbar(a: Any, w1: np.ndarray, w2: float, amplitude: float) -> np.ndarray:
    """``P(Y = 1 | A = a, W)`` with the even blip ``amplitude (W1^2 - 0.5)``."""
    even = amplitude * (np.asarray(w1, dtype=float) ** 2 - 0.5)
    return np.asarray(expit(0.2 + 0.5 * w1 - 0.3 * w2 + np.asarray(a) * even), dtype=float)


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows, with the draw order of :func:`base.draw` (``W1, W2, A, Y``).

    For ``non_exceptional`` this is :func:`tests.studies._learned_rule_law.draw` exactly.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, base.g0(w1, w2)).astype(float)
    y = rng.binomial(1, qbar0(a, w1, w2, law)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def known_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """The known treatment mechanism of a draw, ``[g0(0 | W), g0(1 | W)]`` for each row."""
    treated = base.g0(frame["W1"].to_numpy(dtype=float), frame["W2"].to_numpy(dtype=float))
    return np.column_stack([1.0 - treated, treated])


# ------------------------------------------------------------------------- the learners


def hinge_features(design: np.ndarray) -> np.ndarray:
    """RM30's six products of ``(A, W1, W2)`` and ``A (W1 - knot)_+``.

    The design is the package's ``(A, W1, W2)``.
    """
    design = np.asarray(design, dtype=float)
    a, w1, w2 = design[:, 0], design[:, 1], design[:, 2]
    hinge = np.maximum(w1 - HINGE_KNOT, 0.0)
    return np.column_stack([a, w1, w2, a * w1, a * w2, w1 * w2, a * hinge])


def hinge_learner() -> Pipeline:
    """The ``partially_exceptional`` outcome learner: a correct logistic model of that law."""
    return Pipeline(
        [
            ("features", FunctionTransformer(hinge_features)),
            ("model", LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)),
        ]
    )


#: The outcome learner of each law: the declared RM30 learner, except on the hinge law.
LEARNERS: dict[str, Callable[[], Any]] = {
    "non_exceptional": base.outcome_learner,
    "wrong_limit": base.outcome_learner,
    "partially_exceptional": hinge_learner,
}


def rm30_features(design: np.ndarray) -> np.ndarray:
    """The declared learner's products of ``(A, W1, W2)``."""
    return np.asarray(
        PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(design),
        dtype=float,
    )


#: The feature map of each law's learner, for :func:`limit_blip`.
FEATURES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "non_exceptional": rm30_features,
    "wrong_limit": rm30_features,
    "partially_exceptional": hinge_features,
}


# ------------------------------------------------------------------------ quadrature


def rule_moments(
    rule: Mapping[float, np.ndarray], qbar: Callable[[Any, np.ndarray, float], np.ndarray]
) -> tuple[float, float]:
    """``Psi_d(P0)`` and ``Var D*(d, Qbar0, g0)`` of a rule given on the grid.

    ``rule[w2]`` is the rule at each point of :data:`base.GRID` for that ``W2``. This is
    :func:`tests.studies._learned_rule_law.rule_value` with the outcome regression passed in.
    """
    value = second = 0.0
    for w2 in base.W2_LEVELS:
        treat = np.asarray(rule[w2], dtype=float)
        q = qbar(treat, base.GRID, w2)
        treated = base.g0(base.GRID, w2)
        g = np.where(treat > 0.5, treated, 1.0 - treated)
        value += 0.5 * np.trapezoid(q, base.GRID) / 2.0
        second += 0.5 * np.trapezoid(q * (1.0 - q) / g + q**2, base.GRID) / 2.0
    return float(value), float(second - value**2)


def limit_blip(
    qbar: Callable[[Any, np.ndarray, float], np.ndarray],
    features: Callable[[np.ndarray], np.ndarray] = rm30_features,
) -> dict[float, np.ndarray]:
    """The blip ``Qbar1(1, W) - Qbar1(0, W)`` of a logistic learner's population limit.

    ``features`` maps the design ``(A, W1, W2)`` to the learner's columns. The limit solves the
    population score equations, which the weighted fit on the quadrature rows solves up to the
    quadrature.
    """
    grid = base.GRID
    step = np.full(grid.size, grid[1] - grid[0])
    step[[0, -1]] /= 2.0
    designs, outcomes, weights = [], [], []
    for w2 in base.W2_LEVELS:
        treated = base.g0(grid, w2)
        for a in (0.0, 1.0):
            g = treated if a else 1.0 - treated
            q = qbar(a, grid, w2)
            for y in (0.0, 1.0):
                designs.append(
                    np.column_stack([np.full(grid.size, a), grid, np.full(grid.size, w2)])
                )
                outcomes.append(np.full(grid.size, y))
                weights.append(step * 0.25 * g * (q if y else 1.0 - q))
    design = features(np.vstack(designs))
    weight = np.concatenate(weights)
    model = LogisticRegression(C=1e6, max_iter=10_000, tol=1e-12).fit(
        design, np.concatenate(outcomes), sample_weight=weight * weight.size / weight.sum()
    )
    out = {}
    for w2 in base.W2_LEVELS:
        rows = np.column_stack([np.ones(grid.size), grid, np.full(grid.size, w2)])
        control = rows.copy()
        control[:, 0] = 0.0
        out[w2] = (
            model.predict_proba(features(rows))[:, 1] - model.predict_proba(features(control))[:, 1]
        )
    return out


def optimal_rule(qbar: Callable[[Any, np.ndarray, float], np.ndarray]) -> dict[float, np.ndarray]:
    """:math:`d_0` on the grid: treat where ``Qbar0(1, W) > Qbar0(0, W)``, a tie to arm 0."""
    return {w2: qbar(1.0, base.GRID, w2) > qbar(0.0, base.GRID, w2) for w2 in base.W2_LEVELS}


@cache
def target_gap(amplitude: float | None) -> tuple[float, float, float]:
    """W-f2: the gap ``Psi_d0 - Psi_d1``, the variance of ``d1``'s curve and ``d1``'s share.

    ``d1`` is the declared learner's limit rule on the even-blip law of ``amplitude``;
    ``amplitude=None`` reads the ``non_exceptional`` law, where the learner is correct.
    """

    def qbar(a: Any, w1: np.ndarray, w2: float) -> np.ndarray:
        if amplitude is None:
            return qbar0(a, w1, w2, "non_exceptional")
        return wrong_limit_qbar(a, w1, w2, amplitude)

    limit = limit_blip(qbar)
    d1 = {w2: limit[w2] > 0.0 for w2 in base.W2_LEVELS}
    optimal, _ = rule_moments(optimal_rule(qbar), qbar)
    limit_value, variance = rule_moments(d1, qbar)
    share = float(np.mean([np.mean(d1[w2]) for w2 in base.W2_LEVELS]))
    return optimal - limit_value, variance, share


#: The trapezoid mass of each grid point of ``W1`` for one level of ``W2``, times the density
#: 1/2 of ``W1`` and the probability 1/2 of the level.
MASS = np.full(base.GRID.size, base.GRID[1] - base.GRID[0])
MASS[[0, -1]] /= 2.0
MASS *= 0.25


@cache
def law_limit(law: str) -> dict[str, float]:
    """The conditions of :mod:`tests.studies.learned_rule_optimal`'s gate at ``law``.

    ``truth``, ``variance``: :math:`\\Psi_{d_0}` and the variance of :math:`D^*(d_0)`.
    ``treated``: :math:`d_0`'s treated share. ``tie_mass``: :math:`P_0(b_0(W) = 0)` (C7).
    ``disagreement``: the mass on which the law's learner's limit rule differs from
    :math:`d_0` off the tie set (C3f). ``limit_value``: that limit rule's value. ``margin``:
    the largest :math:`P_0(0 < |b_0(W)| \\le t) / t` over :data:`MARGIN_LEVELS`, on the
    probability scale (C8 with exponent 1).
    """
    qbar = law_qbar(law)
    d0 = optimal_rule(qbar)
    truth, variance = rule_moments(d0, qbar)
    limit = limit_blip(qbar, FEATURES[law])
    d1 = {w2: limit[w2] > 0.0 for w2 in base.W2_LEVELS}
    limit_value, _ = rule_moments(d1, qbar)
    tie = disagreement = treated = 0.0
    gaps, masses = [], []
    for w2 in base.W2_LEVELS:
        gap = qbar(1.0, base.GRID, w2) - qbar(0.0, base.GRID, w2)
        zero = blip(law, base.GRID) == 0.0
        tie += float(np.sum(MASS * zero))
        disagreement += float(np.sum(MASS * (~zero) * (d1[w2] != d0[w2])))
        treated += float(np.sum(MASS * d0[w2]))
        gaps.append(np.abs(gap[~zero]))
        masses.append(MASS[~zero])
    gap, mass = np.concatenate(gaps), np.concatenate(masses)
    margin = max(float(np.sum(mass * (gap <= t))) / t for t in MARGIN_LEVELS)
    return {
        "truth": truth,
        "variance": variance,
        "treated": treated,
        "tie_mass": tie,
        "disagreement": disagreement,
        "limit_value": limit_value,
        "margin": margin,
    }


#: The levels ``t`` of the margin reading ``P0(0 < |b0| <= t) / t`` (probability scale).
MARGIN_LEVELS = (0.005, 0.01, 0.02, 0.05)


# ------------------------------------------------------------------------------ the fit


def estimator(learner: Any, fold_seed: int) -> TMLE:
    """The declared estimator of the value of the optimal rule, with the fold seed of rule L2.

    The settings are those of :func:`tests.studies._learned_rule_law.estimator` with
    ``estimands=["ey_optimal_rule"]``; every other setting is the constructor's default.
    """
    return TMLE(
        learned_rule=LearnedRule(),
        estimands=["ey_optimal_rule"],
        n_folds=base.N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


def fit(frame: pd.DataFrame, learner: Any, fold_seed: int, *, known_g: bool) -> Any:
    """Fit :func:`estimator` to one sample, with ``g0`` supplied when ``known_g``."""
    known = known_probabilities(frame) if known_g else None
    return (
        estimator(learner, fold_seed)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(base.COVARIATES),
            treatment_probabilities=known,
        )
        .single()
    )


@dataclass(frozen=True)
class Refit:
    """The refit fold rules of one fit.

    Parameters
    ----------
    rule_value : float
        :math:`\\tilde\\psi_{0n} = V^{-1}\\sum_v \\Psi_{d_v}(P_0)`, the value of the fit's rules.
    oracle_se : float
        ``sqrt(V^-2 sum_v Var D*(d_v) / n_v)``, by the same quadrature.
    fold_values : tuple of float
        :math:`\\Psi_{d_v}(P_0)` for each fold.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    """

    rule_value: float
    oracle_se: float
    fold_values: tuple[float, ...]
    rows_checked: int


def harness(result: Any, learner: Any, law: str) -> Refit:
    """Refit each fold's rule, check it against the fit's rule, and integrate its value.

    The refit is :func:`tests.studies._learned_rule_law.fold_rule`, the package's own
    ``fit_on_rows`` on the fold's training rows, so the deterministic learner reproduces the
    fitted rule exactly.

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When the refit rule differs from the fit's rule on any validation row (rule L4).
    """
    covariates = np.asarray(result.data.covariates, dtype=float)
    fitted = base.rule_of(result)
    grid = {w2: np.column_stack([base.GRID, np.full_like(base.GRID, w2)]) for w2 in base.W2_LEVELS}
    qbar = law_qbar(law)
    values, variances, sizes = [], [], []
    checked = 0
    for fold, (_, test) in enumerate(result.nuisance.folds):
        rule = base.fold_rule(result, learner, fold)
        disagree = int(np.sum(rule(covariates[test]) != fitted[test]))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        checked += len(test)
        value, variance = rule_moments({w2: rule(points) for w2, points in grid.items()}, qbar)
        values.append(value)
        variances.append(variance)
        sizes.append(len(test))
    folds = len(values)
    oracle = float(np.sqrt(sum(v / m for v, m in zip(variances, sizes, strict=True))) / folds)
    return Refit(
        rule_value=float(np.mean(values)),
        oracle_se=oracle,
        fold_values=tuple(values),
        rows_checked=checked,
    )


@dataclass(frozen=True)
class Measured:
    """One fit, its refit fold rules and its solver warnings.

    Parameters
    ----------
    result : TMLEResult
        The fit.
    refit : Refit
        Its fold rules' values.
    solver_warnings : int
        The solver warnings of the fit and its refits.
    """

    result: Any
    refit: Refit
    solver_warnings: int


def measured(
    frame: pd.DataFrame, law: str, learner: Any, fold_seed: int, *, known_g: bool
) -> Measured:
    """Fit one sample, run its harness, and count the solver warnings of both.

    Rule L4 stops the run on a non-finite estimate or SE, on a harness mismatch, on a fit that
    reports another parameter, and on a known-g fit whose key does not read ``known_mechanism``:
    each raises an error that the replication loops do not record.
    """
    with base.solver_warning_count() as count:
        result = fit(frame, learner, fold_seed, known_g=known_g)
        refit = harness(result, learner, law)
    if tuple(result.estimates) != (ESTIMAND,):
        raise base.HarnessMismatch(f"the fit reports {tuple(result.estimates)}, not {ESTIMAND}")
    known = "known_mechanism" in result.config.support_key.qualifiers
    if known != known_g:
        raise base.HarnessMismatch(f"a fit with known_g={known_g} keys known_mechanism={known}")
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or SE (rule L4): {estimate}")
    return Measured(result=result, refit=refit, solver_warnings=count[0])


def rule_disagreement(result: Any) -> float:
    """The record's share of rows on which the fold rules do not all agree."""
    (split,) = result.extra["learned_rule"].splits
    return float(split.rule_disagreement)


def harness_columns(fitted: Measured, law: str) -> dict[str, Any]:
    """The harness columns of one fit's row."""
    truth = law_limit(law)["truth"]
    return {
        "oracle_se": fitted.refit.oracle_se,
        "rule_rows_checked": fitted.refit.rows_checked,
        "solver_warnings": fitted.solver_warnings,
        "rule_value": fitted.refit.rule_value,
        "regret": truth - fitted.refit.rule_value,
        "rule_disagreement": rule_disagreement(fitted.result),
    }
