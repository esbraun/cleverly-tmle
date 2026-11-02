r"""The law, the learners, the fit and the harness of ``learned-regimen-ltmle`` (X11 (b)).

The study measures ``LTMLE(regimens=LearnedRegimen())``: two decision nodes, a binary treatment
at each, censoring after each, an end-of-study binary outcome. Each training fold ``v`` learns
the rule pair :math:`d_{nv} = (d_{nv,1}, d_{nv,2})` by backward plug-in Q-learning, and the fit
reports the stacked value ``ey_regimen[learned regimen]``, whose target is the data-adaptive
:math:`\tilde\psi_{0n} = \sum_v (n_v / n) \Psi_{d_{nv}}(P_0)`.

**The law** (:data:`LAW`). ``W1 ~ U(-1, 1)``, ``W2 ~ Bernoulli(0.5)``, then

.. math::

    \operatorname{logit} g_1(1 \mid W) &= 0.8 W_1 - 0.3 W_2, \qquad
    \operatorname{logit} P(C_1 = 1 \mid A_1, W) = 2.5 + 0.3 W_1, \\
    L_2 \mid A_1, W &\sim N(0.5 W_1 + 0.2 W_2 - 0.8 A_1, 1), \\
    \operatorname{logit} g_2(1 \mid L_2, A_1, W) &= -0.2 + 0.3 L_2 + 0.4 A_1, \qquad
    \operatorname{logit} P(C_2 = 1 \mid \ldots) = 2.5 - 0.2 L_2, \\
    \operatorname{logit} \bar Q_0(a_1, a_2, L_2, W) &= -0.2 + 0.6 W_1 - 0.2 W_2 + 0.3 L_2
        + a_2 (-0.1 + 1.2 L_2 - 0.6 a_1) + a_1 (0.3 + 1.2 W_1).

``L2`` is affected by ``A1``, which lowers it by 0.8. In each stratum ``(a1, a2)`` the outcome
logit is linear in ``(W1, W2, L2)``, so the declared node-2 learner, a main-terms logistic model
fitted per stratum, is a correct model. The optimal node-2 rule treats where
``L2 > (0.1 + 0.6 a1) / 1.2``, and its blip has slope 1.2 in ``L2``, which has a density at the
boundary: a margin at that boundary. The optimal node-1 rule is a threshold in ``W1`` for each
``W2``, where the value of the two node-1 arms crosses with a nonzero slope: a margin at that
boundary too. :func:`margins` reads the mass near each boundary, the weak-margin diagnostic.

**The learners.** Node 2: ``LogisticRegression(C=1e6)`` per stratum on ``(W1, W2, L2)``, a
correct model. Node 1 (``declared``): a cubic polynomial in ``(W1, W2)`` fitted by least squares
to the carried node-2 value, which approximates the node-1 regression, an integral of a logistic
curve over ``L2``, closely. Two misspecified node-1 learners: ``drop_w1``, least squares on
``W2`` alone, which drops the confounder ``W1`` of ``A1`` (``targeting_necessity``), and
``squared_w1``, least squares on ``(W1^2, W2)``, which keeps ``W1`` only through its square
(``conservative_coverage``). The treatment and censoring learners are main-terms logistic models,
correct at both nodes.

**The harness** (:func:`harness`) refits each fold's backward Q-learning on the fit's training
rows with scikit-learn alone, requires both rules to equal the fit's stored fold rules on every
row (rule L4), and integrates each fold's rule pair by quadrature: the trapezoid rule on 1,001
points of ``W1`` for each ``W2``, and for ``L2`` given ``(A1, W)`` the Gauss-Legendre rule on the
probability scale, split at the fold's node-2 threshold, which is exact for a stratum-wise
logistic model because the sign of its blip is affine in ``L2``. The same quadrature gives the
oracle SE at the fold rules. The harness also rebuilds the curve from the fit's stored node steps
and requires its variance to equal the reported one (:data:`EXACTNESS_TOLERANCE`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from cleverly.longitudinal import LTMLE, LearnedRegimen
from tests.studies import _learned_rule_law as base

#: The one law of the study.
LAW = "two_node"

#: The reported parameter.
ESTIMAND = "ey_regimen[learned regimen]"

#: The outer folds: the ``LTMLE`` default.
N_FOLDS = 10

#: The panel columns of :func:`draw`, as ``LTMLE.fit`` reads them.
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}

#: The quadrature grid of ``W1`` (1,001 points, a quarter of RM30's, because each draw integrates
#: ten fold rule pairs) and the levels of ``W2``. Against 16,001 points and 64 nodes the value of
#: a fold rule pair moves by about 1e-5, 0.05% of an SE at n = 2,000
#: (``test_learned_regimen_design``).
GRID = np.linspace(-1.0, 1.0, 1001)
W2_LEVELS = base.W2_LEVELS
CRITICAL = base.CRITICAL

#: Gauss-Legendre nodes on each side of the node-2 threshold, on the probability scale of ``L2``.
LEGENDRE = 24

#: ``logit g1(1 | W) = G1[0] W1 + G1[1] W2``.
G1 = (0.8, -0.3)
#: ``logit P(C1 = 1 | W) = C1[0] + C1[1] W1``.
C1 = (2.5, 0.3)
#: ``E[L2 | A1, W] = L2_MEAN[0] W1 + L2_MEAN[1] W2 + L2_MEAN[2] A1``, with unit SD.
L2_MEAN = (0.5, 0.2, -0.8)
#: ``logit g2(1 | L2, A1) = G2[0] + G2[1] L2 + G2[2] A1``.
G2 = (-0.2, 0.3, 0.4)
#: ``logit P(C2 = 1 | L2) = C2[0] + C2[1] L2``.
C2 = (2.5, -0.2)
#: The outcome logit's main terms ``(1, W1, W2, L2)``, the node-2 blip ``(1, L2, A1)`` and the
#: node-1 effect ``(1, W1)``.
Y_MAIN = (-0.2, 0.6, -0.2, 0.3)
Y_BLIP2 = (-0.1, 1.2, -0.6)
Y_BLIP1 = (0.3, 1.2)

#: The relative tolerance of the variance rebuilt from the stored node steps against the
#: reported one. A draw outside it stops the run (rule L4).
EXACTNESS_TOLERANCE = 1e-8

#: The columns each row carries beside the shared schema. ``oracle_se`` is descriptive.
#: ``rule_rows_checked`` counts the rows whose refit rule pair equals the fit's stored one in
#: every fold, which is every row of a completed draw, and ``solver_warnings`` the warnings of
#: the fit and its refits (rule L4).
HARNESS_COLUMNS = ("oracle_se", "rule_rows_checked", "solver_warnings")

SOLVER_WARNINGS = base.SOLVER_WARNINGS
HarnessMismatch = base.HarnessMismatch


# --------------------------------------------------------------------------- the law


def g1(w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``P(A1 = 1 | W)``."""
    return np.asarray(expit(G1[0] * np.asarray(w1) + G1[1] * np.asarray(w2)), dtype=float)


def c1(w1: np.ndarray) -> np.ndarray:
    """``P(C1 = 1 | W)``, the probability of staying uncensored after node 1."""
    return np.asarray(expit(C1[0] + C1[1] * np.asarray(w1)), dtype=float)


def l2_mean(a1: np.ndarray | float, w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``E[L2 | A1 = a1, W]``."""
    return np.asarray(
        L2_MEAN[0] * np.asarray(w1) + L2_MEAN[1] * np.asarray(w2) + L2_MEAN[2] * np.asarray(a1),
        dtype=float,
    )


def g2(l2: np.ndarray, a1: np.ndarray | float) -> np.ndarray:
    """``P(A2 = 1 | L2, A1, W)``."""
    return np.asarray(expit(G2[0] + G2[1] * np.asarray(l2) + G2[2] * np.asarray(a1)), dtype=float)


def c2(l2: np.ndarray) -> np.ndarray:
    """``P(C2 = 1 | ...)``, the probability of staying uncensored after node 2."""
    return np.asarray(expit(C2[0] + C2[1] * np.asarray(l2)), dtype=float)


def blip2(a1: np.ndarray | float, l2: np.ndarray) -> np.ndarray:
    """The node-2 blip on the logit scale."""
    return np.asarray(
        Y_BLIP2[0] + Y_BLIP2[1] * np.asarray(l2) + Y_BLIP2[2] * np.asarray(a1), dtype=float
    )


def qbar0(
    a1: np.ndarray | float,
    a2: np.ndarray | float,
    l2: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray | float,
) -> np.ndarray:
    """``P(Y = 1 | A1 = a1, A2 = a2, L2, W)``."""
    w1, l2 = np.asarray(w1, dtype=float), np.asarray(l2, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    logit = (
        Y_MAIN[0]
        + Y_MAIN[1] * w1
        + Y_MAIN[2] * np.asarray(w2)
        + Y_MAIN[3] * l2
        + np.asarray(a2) * blip2(a1, l2)
        + a1 * (Y_BLIP1[0] + Y_BLIP1[1] * w1)
    )
    return np.asarray(expit(logit), dtype=float)


def draw(n: int, seed: int) -> pd.DataFrame:
    """One panel of ``n`` rows, drawn in the order ``W1, W2, A1, C1, L2, A2, C2, Y``.

    A row censored after node 1 has no ``L2``, ``A2``, ``C2`` or ``Y``; a row censored after
    node 2 has no ``Y``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a1 = rng.binomial(1, g1(w1, w2)).astype(float)
    s1 = rng.binomial(1, c1(w1)).astype(float)
    l2 = l2_mean(a1, w1, w2) + rng.standard_normal(n)
    a2 = rng.binomial(1, g2(l2, a1)).astype(float)
    s2 = rng.binomial(1, c2(l2)).astype(float)
    y = rng.binomial(1, qbar0(a1, a2, l2, w1, w2)).astype(float)
    frame = pd.DataFrame(
        {"W1": w1, "W2": w2, "A1": a1, "C1": s1, "L2": l2, "A2": a2, "C2": s2, "Y": y}
    )
    frame.loc[s1 == 0, ["L2", "A2", "C2", "Y"]] = np.nan
    frame.loc[(s1 == 1) & (s2 == 0), "Y"] = np.nan
    return frame


# ------------------------------------------------------------------------- the learners


def outcome_learner() -> LogisticRegression:
    """The node-2 learner: main-terms logistic, fitted per stratum, a correct model."""
    return LogisticRegression(C=1e6, max_iter=1000)


def declared_node1() -> Pipeline:
    """The declared node-1 learner: a cubic polynomial in ``(W1, W2)``, least squares."""
    return Pipeline(
        [
            ("features", PolynomialFeatures(3, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )


def drop_w1_node1() -> Pipeline:
    """A misspecified node-1 learner: least squares on ``W2`` alone, without ``W1``."""
    return Pipeline(
        [
            ("keep", ColumnTransformer([("keep", "passthrough", [1])])),
            ("model", LinearRegression()),
        ]
    )


def squared_w1_node1() -> Pipeline:
    """A misspecified node-1 learner: least squares on ``(W1^2, W2)``."""
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("square", FunctionTransformer(np.square), [0]),
                        ("keep", "passthrough", [1]),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )


def nuisance_learner() -> LogisticRegression:
    """The treatment and censoring learner: main-terms logistic, correct at both nodes."""
    return LogisticRegression(C=1e6, max_iter=1000)


#: The node-1 learner of each cell, by name.
NODE1_LEARNERS: dict[str, Callable[[], Any]] = {
    "declared": declared_node1,
    "drop_w1": drop_w1_node1,
    "squared_w1": squared_w1_node1,
}


# ------------------------------------------------------------------------------ the fit


def _validation_fold_rules(
    data: Any,
    plan: Any,
    *,
    folds: Any,
    scaler: Any,
    outcome_learner: Any,
    pseudo_learner: Any,
    n_jobs: int = 1,
    bounds: Any = None,
) -> Any:
    """The ``fold_locality`` mutation: learn fold ``v``'s rule pair on fold ``v``'s validation
    rows, the rows that then evaluate it. Everything after the rules is the fit's own."""
    from cleverly.longitudinal import learned

    outcomes = []
    for fold, (_, test) in enumerate(folds):
        rows = np.zeros(data.n, dtype=bool)
        rows[test] = True
        outcomes.append(
            learned._learn_in_fold(
                data,
                plan,
                scaler=scaler,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                train=rows,
                fold=fold,
                bounds=bounds,
            )
        )
    return learned.LearnedRules(
        assignments=tuple(out[0] for out in outcomes),
        blips=tuple(out[1] for out in outcomes),
        second_node_blips=tuple(out[2] for out in outcomes),
        folds=folds,
    )


@contextlib.contextmanager
def validation_rows() -> Iterator[None]:
    """Run the block with each fold's rules learned on its validation rows (study only).

    ``LTMLE`` reads ``learn_fold_rules`` from its module at fit time, so the block swaps that
    name and restores it. A study worker runs one fit at a time, so no other fit sees the swap.
    """
    from cleverly.longitudinal import estimator as module

    original = module.learn_fold_rules
    module.learn_fold_rules = _validation_fold_rules
    try:
        yield
    finally:
        module.learn_fold_rules = original


def estimator(node1: str, fold_seed: int) -> LTMLE:
    """The declared estimator, with the fold seed of rule L2."""
    return LTMLE(
        LearnedRegimen(),
        n_folds=N_FOLDS,
        random_state=fold_seed,
        outcome_learner=outcome_learner(),
        pseudo_learner=NODE1_LEARNERS[node1](),
        treatment_learner=nuisance_learner(),
        censoring_learner=nuisance_learner(),
    )


def fit(frame: pd.DataFrame, node1: str, fold_seed: int, *, mutation: bool = False) -> Any:
    """Fit the declared estimator to one panel and return the result."""
    context = validation_rows() if mutation else contextlib.nullcontext()
    with context:
        return estimator(node1, fold_seed).fit(frame, **COLUMNS)


def untargeted_estimate(result: Any) -> float:
    """The fit without its fluctuation: the mean of the stitched node-1 initial values.

    Each row's initial value is its own fold's untargeted recursion under that fold's rule pair,
    so the mean weights each fold by ``n_v / n``, as the stacked target does.
    """
    (regimen_fit,) = result.fits.values()
    scaler = result.scaler
    mean = float(np.mean(np.asarray(regimen_fit.steps[0].initial, dtype=float)))
    value = float(scaler.lower + (scaler.upper - scaler.lower) * mean)
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate (rule L4): {value}")
    return value


# ------------------------------------------------------------------------ quadrature

#: The trapezoid mass of each grid point of ``W1`` for one level of ``W2``, times the density
#: 1/2 of ``W1`` and the probability 1/2 of the level.
MASS = np.full(GRID.size, GRID[1] - GRID[0])
MASS[[0, -1]] /= 2.0
MASS *= 0.25

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(LEGENDRE)


def _pieces(cut: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legendre nodes and weights on ``[0, cut]`` and ``[cut, 1]`` of the probability scale.

    Returns the probability of each node, its weight, and whether it lies above the cut, each of
    shape ``(len(cut), 2 * LEGENDRE)``.
    """
    cut = np.asarray(cut, dtype=float)[:, None]
    half = (_NODES[None, :] + 1.0) / 2.0
    low = cut * half
    high = cut + (1.0 - cut) * half
    probability = np.concatenate([low, high], axis=1)
    weight = np.concatenate(
        [cut * _WEIGHTS[None, :] / 2.0, (1.0 - cut) * _WEIGHTS[None, :] / 2.0], axis=1
    )
    above = np.concatenate([np.zeros_like(low, dtype=bool), np.ones_like(high, dtype=bool)], axis=1)
    return probability, weight, above


@dataclass(frozen=True)
class RulePair:
    """A rule pair on the quadrature grid.

    Parameters
    ----------
    first : dict
        ``first[w2]`` is the node-1 arm at each grid point of ``W1``.
    slope : dict
        ``slope[w2][a1]``: the coefficient of ``L2`` in the node-2 blip's sign, by arm.
    offset : dict
        ``offset[w2][a1]``: its value at ``L2 = 0`` at each grid point of ``W1``, by arm. Node 2
        treats where ``offset + slope L2 > 0``.
    """

    first: dict[float, np.ndarray]
    slope: dict[float, dict[float, float]]
    offset: dict[float, dict[float, np.ndarray]]


def _inner(pair: RulePair, w2: float) -> dict[str, np.ndarray]:
    r"""The conditional moments given ``W`` at each grid point of ``W1``, under ``pair``.

    ``value``: :math:`E[\bar Q_0(d_1, d_2, L_2, W) \mid A_1 = d_1, W]`; ``square``: the mean of
    its square; ``outcome``: the mean of :math:`\bar Q_0(1 - \bar Q_0) / (g_2 c_2)` at ``d``.
    """
    a1 = np.asarray(pair.first[w2], dtype=float)
    slope = np.where(a1 > 0.5, pair.slope[w2][1.0], pair.slope[w2][0.0])
    offset = np.where(a1 > 0.5, pair.offset[w2][1.0], pair.offset[w2][0.0])
    mean = l2_mean(a1, GRID, w2)
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = np.where(slope != 0.0, -offset / slope, 0.0)
    cut = np.where(slope != 0.0, norm.cdf(threshold - mean), 0.5)
    probability, weight, above = _pieces(cut)
    # A threshold far in a tail leaves one piece of zero width; its nodes carry no weight, and
    # the clip keeps their L2 finite.
    l2 = mean[:, None] + norm.ppf(np.clip(probability, 1e-15, 1.0 - 1e-15))
    # Above the threshold the blip has the sign of the slope; with no slope, of the offset.
    treat_above = np.where(slope != 0.0, slope > 0.0, offset > 0.0)[:, None]
    treat_below = np.where(slope != 0.0, slope < 0.0, offset > 0.0)[:, None]
    a2 = np.where(above, treat_above, treat_below).astype(float)
    q = qbar0(a1[:, None], a2, l2, GRID[:, None], w2)
    g = np.where(a2 > 0.5, g2(l2, a1[:, None]), 1.0 - g2(l2, a1[:, None]))
    return {
        "value": np.sum(weight * q, axis=1),
        "square": np.sum(weight * q**2, axis=1),
        "outcome": np.sum(weight * q * (1.0 - q) / (g * c2(l2)), axis=1),
    }


def rule_value(pair: RulePair) -> tuple[float, float]:
    """``Psi(d)`` and ``Var D*(d)`` of a rule pair, by quadrature.

    With ``H1 = 1{A1 = d1, C1 = 1} / (g1 c1)`` and ``H2 = H1 1{A2 = d2, C2 = 1} / (g2 c2)``, the
    efficient influence function is ``H2 (Y - Q2) + H1 (Q2(d) - Q1) + Q1(d) - Psi``, whose three
    terms are uncorrelated, so its variance is the sum of their variances.
    """
    value = second = 0.0
    for w2 in W2_LEVELS:
        moments = _inner(pair, w2)
        a1 = np.asarray(pair.first[w2], dtype=float)
        treated = g1(GRID, w2)
        weight = 1.0 / (np.where(a1 > 0.5, treated, 1.0 - treated) * c1(GRID))
        conditional = moments["square"] - moments["value"] ** 2
        value += float(np.sum(MASS * moments["value"]))
        second += float(
            np.sum(MASS * (weight * (moments["outcome"] + conditional) + moments["value"] ** 2))
        )
    return value, second - value**2


def _second_node(a1: float, w2: float) -> tuple[float, np.ndarray]:
    """The optimal node-2 rule's slope and offset at arm ``a1``: the law's blip."""
    return Y_BLIP2[1], np.full(GRID.size, Y_BLIP2[0] + Y_BLIP2[2] * a1)


def _arm_values(slope: dict, offset: dict, w2: float) -> dict[float, np.ndarray]:
    """``E[Qbar0(a1, d2, L2, W) | A1 = a1, W]`` at each arm, under a node-2 rule."""
    out = {}
    for a1 in (0.0, 1.0):
        pair = RulePair(
            first={w2: np.full(GRID.size, a1)},
            slope={w2: slope},
            offset={w2: offset},
        )
        out[a1] = _inner(pair, w2)["value"]
    return out


@cache
def optimal_pair() -> RulePair:
    """The optimal rule pair on the grid: the law's node-2 blip, then the better node-1 arm."""
    first: dict[float, np.ndarray] = {}
    slope: dict[float, dict[float, float]] = {}
    offset: dict[float, dict[float, np.ndarray]] = {}
    for w2 in W2_LEVELS:
        slope[w2], offset[w2] = {}, {}
        for a1 in (0.0, 1.0):
            slope[w2][a1], offset[w2][a1] = _second_node(a1, w2)
        values = _arm_values(slope[w2], offset[w2], w2)
        first[w2] = (values[1.0] > values[0.0]).astype(float)
    return RulePair(first=first, slope=slope, offset=offset)


@cache
def law_limit() -> dict[str, float]:
    """The optimal rule pair's value, the SD of its curve and its treated shares.

    ``treated_1``: the node-1 share; ``treated_2``: the node-2 share among the rows that follow
    the node-1 rule (uncensored, by the law of ``L2`` given the rule's arm).
    """
    pair = optimal_pair()
    truth, variance = rule_value(pair)
    treated_2 = 0.0
    for w2 in W2_LEVELS:
        a1 = pair.first[w2]
        threshold = (-Y_BLIP2[0] - Y_BLIP2[2] * a1) / Y_BLIP2[1]
        treated_2 += float(np.sum(MASS * norm.sf(threshold - l2_mean(a1, GRID, w2))))
    return {
        "truth": truth,
        "bound": float(np.sqrt(variance)),
        "treated_1": float(sum(np.sum(MASS * pair.first[w2]) for w2 in W2_LEVELS)),
        "treated_2": treated_2,
    }


@cache
def static_values() -> dict[tuple[float, float], float]:
    """The value of each static regimen ``(a1, a2)``, for the gap of the optimal pair."""
    out = {}
    for a1 in (0.0, 1.0):
        for a2 in (0.0, 1.0):
            pair = RulePair(
                first=dict.fromkeys(W2_LEVELS, np.full(GRID.size, a1)),
                slope={w2: {0.0: 0.0, 1.0: 0.0} for w2 in W2_LEVELS},
                offset={
                    w2: {0.0: np.full(GRID.size, a2 - 0.5), 1.0: np.full(GRID.size, a2 - 0.5)}
                    for w2 in W2_LEVELS
                },
            )
            out[a1, a2] = rule_value(pair)[0]
    return out


@cache
def margins() -> dict[str, float]:
    """The weak-margin diagnostic of the law, at the optimal rule pair.

    ``node_2``: the probability, among rows that follow the node-1 rule, that the node-2 blip is
    within 0.1 of zero on the logit scale; ``node_1``: the probability that the two node-1 arms'
    values differ by less than 0.01.
    """
    pair = optimal_pair()
    node_2 = node_1 = 0.0
    half = 0.1 / Y_BLIP2[1]
    for w2 in W2_LEVELS:
        a1 = pair.first[w2]
        threshold = (-Y_BLIP2[0] - Y_BLIP2[2] * a1) / Y_BLIP2[1]
        mean = l2_mean(a1, GRID, w2)
        node_2 += float(
            np.sum(MASS * (norm.cdf(threshold + half - mean) - norm.cdf(threshold - half - mean)))
        )
        values = _arm_values(pair.slope[w2], pair.offset[w2], w2)
        node_1 += float(np.sum(MASS * (np.abs(values[1.0] - values[0.0]) < 0.01)))
    return {"node_2": node_2, "node_1": node_1}


#: The features of each misspecified node-1 learner on the grid, at one level of ``W2``.
NODE1_FEATURES: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "drop_w1": lambda w1, w2: np.column_stack([np.ones_like(w1), np.full_like(w1, w2)]),
    "squared_w1": lambda w1, w2: np.column_stack([np.ones_like(w1), w1**2, np.full_like(w1, w2)]),
}


@cache
def node1_limit(name: str) -> dict[str, Any]:
    """A misspecified node-1 learner's limit, with the correct node-2 rule.

    Least squares in each node-1 branch ``a1`` converges to the projection of the carried node-2
    value ``V(a1, W)`` on the learner's features, weighted by the density of the uncensored rows
    that took the arm, ``g1(a1 | W) c1(W)``. ``plug_in`` is the plug-in it converges to,
    ``rule_value`` the value of its limit rule pair, ``bias`` their difference (the untargeted
    bias that ``targeting_necessity`` reads), ``gap`` the limit's arm contrast at each grid point
    by ``W2`` level, and ``near`` the share of ``W`` where that contrast is below 0.01.
    """
    optimal = optimal_pair()
    features = NODE1_FEATURES[name]
    coefficients = {}
    for a1 in (0.0, 1.0):
        gram, moment = 0.0, 0.0
        for w2 in W2_LEVELS:
            x = features(GRID, w2)
            g = g1(GRID, w2) if a1 else 1.0 - g1(GRID, w2)
            weight = MASS * g * c1(GRID)
            value = _arm_values(optimal.slope[w2], optimal.offset[w2], w2)[a1]
            gram = gram + x.T @ (x * weight[:, None])
            moment = moment + x.T @ (weight * value)
        coefficients[a1] = np.linalg.solve(gram, moment)
    first: dict[float, np.ndarray] = {}
    gap: dict[float, np.ndarray] = {}
    plug_in = near = 0.0
    for w2 in W2_LEVELS:
        x = features(GRID, w2)
        arms = {a1: np.clip(x @ coefficients[a1], 0.0, 1.0) for a1 in (0.0, 1.0)}
        gap[w2] = arms[1.0] - arms[0.0]
        first[w2] = (gap[w2] > 0.0).astype(float)
        plug_in += float(np.sum(MASS * np.where(first[w2] > 0.5, arms[1.0], arms[0.0])))
        near += float(np.sum(MASS * (np.abs(gap[w2]) < 0.01)))
    pair = RulePair(first=first, slope=optimal.slope, offset=optimal.offset)
    value, variance = rule_value(pair)
    return {
        "plug_in": plug_in,
        "rule_value": value,
        "bias": plug_in - value,
        "treated_1": float(sum(np.sum(MASS * first[w2]) for w2 in W2_LEVELS)),
        "bound": float(np.sqrt(variance)),
        "gap": gap,
        "near": near,
    }


# -------------------------------------------------------------------------- the harness


@dataclass(frozen=True)
class FoldModels:
    """One fold's backward Q-learning, refitted with scikit-learn on its learning rows.

    Parameters
    ----------
    second : dict
        ``second[a1, a2]``: the node-2 logistic model of stratum ``(a1, a2)`` on
        ``(W1, W2, L2)``.
    first : dict
        ``first[a1]``: the node-1 model of branch ``a1`` on ``(W1, W2)``.
    rules : numpy.ndarray
        The ``(n, 2)`` rule pair at every row of the panel.
    """

    second: dict[tuple[float, float], Any]
    first: dict[float, Any]
    rules: np.ndarray


def _panel(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "a1": frame["A1"].to_numpy(dtype=float),
        "a2": np.nan_to_num(frame["A2"].to_numpy(dtype=float)),
        "observed": (frame["C1"] == 1).to_numpy(),
        "complete": ((frame["C1"] == 1) & (frame["C2"] == 1)).to_numpy(),
        "x2": np.nan_to_num(frame[["W1", "W2", "L2"]].to_numpy(dtype=float)),
        "x1": frame[["W1", "W2"]].to_numpy(dtype=float),
        "y": np.nan_to_num(frame["Y"].to_numpy(dtype=float)),
    }


def refit_fold(frame: pd.DataFrame, node1: str, rows: np.ndarray) -> FoldModels:
    """Backward plug-in Q-learning on ``rows`` with scikit-learn alone (rule L4).

    Node 2: one logistic model per stratum ``(a1, a2)`` on the uncensored ``rows`` of the
    stratum, the blip ``Q(a1, 1) - Q(a1, 0)`` at every row uncensored after node 1, and the rule
    ``1{blip > 0}``. Node 1: in each branch ``a1``, the node-1 learner on the ``rows`` uncensored
    after node 1 that took ``a1``, fitted to the node-2 prediction at that branch's rule, clipped
    to ``[0, 1]``; the rule takes the larger branch, with a tie to arm 0.
    """
    panel = _panel(frame)
    n = len(frame)
    second_models: dict[tuple[float, float], Any] = {}
    first_models: dict[float, Any] = {}
    blips = np.full((n, 2), np.nan)
    branch = np.zeros((n, 2))
    for arm in (0.0, 1.0):
        predicted = []
        for later in (0.0, 1.0):
            fit_rows = rows & panel["complete"] & (panel["a1"] == arm) & (panel["a2"] == later)
            model = outcome_learner().fit(panel["x2"][fit_rows], panel["y"][fit_rows])
            second_models[arm, later] = model
            predicted.append(np.clip(model.predict_proba(panel["x2"])[:, 1], 0.0, 1.0))
        blip = np.where(panel["observed"], predicted[1] - predicted[0], np.nan)
        blips[:, int(arm)] = blip
        rule = np.nan_to_num(blip) > 0.0
        target = np.where(rule, predicted[1], predicted[0])
        fit_rows = rows & panel["observed"] & (panel["a1"] == arm)
        model = NODE1_LEARNERS[node1]().fit(panel["x1"][fit_rows], target[fit_rows])
        first_models[arm] = model
        branch[:, int(arm)] = np.clip(model.predict(panel["x1"]), 0.0, 1.0)
    first_rule = (branch[:, 1] - branch[:, 0] > 0.0).astype(int)
    chosen = blips[np.arange(n), first_rule]
    second_rule = (np.nan_to_num(chosen) > 0.0).astype(int)
    return FoldModels(
        second=second_models,
        first=first_models,
        rules=np.column_stack([first_rule, second_rule]),
    )


def grid_pair(models: FoldModels) -> RulePair:
    """A fold's rule pair on the quadrature grid, from its refitted models.

    The node-2 rule is the sign of ``logit Q(a1, 1) - logit Q(a1, 0)``, the sign of the blip,
    which is affine in ``L2`` for a stratum-wise main-terms logistic model: its slope is the
    difference of the two models' ``L2`` coefficients.
    """
    first: dict[float, np.ndarray] = {}
    slope: dict[float, dict[float, float]] = {}
    offset: dict[float, dict[float, np.ndarray]] = {}
    for w2 in W2_LEVELS:
        x1 = np.column_stack([GRID, np.full(GRID.size, w2)])
        arms = {arm: np.clip(models.first[arm].predict(x1), 0.0, 1.0) for arm in (0.0, 1.0)}
        first[w2] = (arms[1.0] - arms[0.0] > 0.0).astype(float)
        slope[w2], offset[w2] = {}, {}
        for arm in (0.0, 1.0):
            treated, control = models.second[arm, 1.0], models.second[arm, 0.0]
            coefficient = treated.coef_[0] - control.coef_[0]
            intercept = float(treated.intercept_[0] - control.intercept_[0])
            slope[w2][arm] = float(coefficient[2])
            offset[w2][arm] = intercept + coefficient[0] * GRID + coefficient[1] * w2
    return RulePair(first=first, slope=slope, offset=offset)


@dataclass(frozen=True)
class Truth:
    """What the harness computes for one fit.

    Parameters
    ----------
    truth : float
        ``sum_v (n_v / n) Psi(d_v)``, the stacked value of the rule pairs the fit used.
    oracle_se : float
        ``sqrt(sum_v n_v Var D*(d_v)) / n``, by the same quadrature.
    fold_values : tuple of float
        ``Psi(d_v)`` for each fold.
    rows_checked : int
        The rows whose refit rule pair equals the fit's stored one in every fold.
    """

    truth: float
    oracle_se: float
    fold_values: tuple[float, ...]
    rows_checked: int


def curve_variance(result: Any) -> float:
    """The variance rebuilt from the stored node steps: the stacked curve centred at the pooled
    estimate, ``sum_i (Z_i - psi)^2 / (n (n - 1))``, on the outcome's own scale."""
    (regimen_fit,) = result.fits.values()
    steps = regimen_fit.steps
    z = np.asarray(steps[0].targeted, dtype=float).copy()
    for step in steps:
        z = z + np.where(
            step.trained_on,
            np.asarray(step.clever) * (np.asarray(step.pseudo_outcome) - step.targeted),
            0.0,
        )
    n = z.size
    centred = z - float(np.mean(steps[0].targeted))
    scale = float(result.scaler.upper - result.scaler.lower)
    return float(np.sum(centred**2) / (n * (n - 1))) * scale**2


def harness(result: Any, frame: pd.DataFrame, node1: str, *, mutation: bool = False) -> Truth:
    """Refit each fold's rule pair, check it, and integrate its value.

    ``mutation=True`` refits each fold on its validation rows, as the ``fold_locality`` control
    learns them.

    Raises
    ------
    HarnessMismatch
        When a refit rule pair differs from the fit's stored fold rules on any row (rule L4), or
        when the variance rebuilt from the node steps differs from the reported one by more than
        :data:`EXACTNESS_TOLERANCE`.
    """
    record = result.learned_regimen
    if record is None:
        raise HarnessMismatch("the fit stores no learned-regimen record")
    folds = result.folds
    n = len(frame)
    stored = np.asarray(record.fold_rules, dtype=int)
    values, variances, sizes = [], [], []
    for fold, (train, test) in enumerate(folds):
        rows = np.zeros(n, dtype=bool)
        rows[np.asarray(test if mutation else train)] = True
        models = refit_fold(frame, node1, rows)
        differ = int(np.sum(np.any(models.rules != stored[fold], axis=1)))
        if differ:
            raise HarnessMismatch(
                f"fold {fold}: the refit rule pair differs from the stored one on {differ} rows"
            )
        value, variance = rule_value(grid_pair(models))
        values.append(value)
        variances.append(variance)
        sizes.append(len(test))
    reported = float(result.estimates[ESTIMAND].std_error) ** 2
    rebuilt = curve_variance(result)
    if abs(rebuilt - reported) > EXACTNESS_TOLERANCE * reported:
        raise HarnessMismatch(f"the rebuilt variance is {rebuilt!r}, the fit's {reported!r}")
    weights = np.asarray(sizes, dtype=float) / n
    return Truth(
        truth=float(np.dot(weights, values)),
        oracle_se=float(np.sqrt(np.dot(sizes, variances)) / n),
        fold_values=tuple(values),
        rows_checked=n,
    )


@dataclass(frozen=True)
class Measured:
    """One fit, its harness and its solver warnings.

    Parameters
    ----------
    result : LongitudinalResult
        The fit.
    truth : Truth
        Its harness.
    solver_warnings : int
        The solver warnings of the fit and its refits.
    """

    result: Any
    truth: Truth
    solver_warnings: int

    @property
    def estimate(self) -> Any:
        """The reported estimate."""
        return self.result.estimates[ESTIMAND]


def measured(
    frame: pd.DataFrame, node1: str, fold_seed: int, *, mutation: bool = False
) -> Measured:
    """Fit one panel, run its harness, and count the solver warnings.

    Rule L4 stops the run on a non-finite estimate or SE, on a harness mismatch, and on a fit
    that reports another parameter or does not key the regimen kind ``learned``.
    """
    with base.solver_warning_count() as count:
        result = fit(frame, node1, fold_seed, mutation=mutation)
        truth = harness(result, frame, node1, mutation=mutation)
    if tuple(result.estimates) != (ESTIMAND,):
        raise HarnessMismatch(f"the fit reports {tuple(result.estimates)}, not {ESTIMAND}")
    if "learned" not in result.config.support_key.family:
        raise HarnessMismatch(f"the key reads {sorted(result.config.support_key.family)}")
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or SE (rule L4): {estimate}")
    return Measured(result=result, truth=truth, solver_warnings=count[0])


def harness_columns(fitted: Measured) -> dict[str, Any]:
    """The harness columns of one row."""
    return {
        "oracle_se": fitted.truth.oracle_se,
        "rule_rows_checked": fitted.truth.rows_checked,
        "solver_warnings": fitted.solver_warnings,
    }
