"""The laws, fits and truth harness of the ``learned-rule-compositions`` study (X11 PR A).

The study measures the learned-rule value under three settings of the fold-evaluated engine
that X11 parts (h), (k) and (l) lift: fold targeting, known observation weights and clusters.
It reuses the RM30 laws and learners of :mod:`tests.studies._learned_rule_law` and adds what
each setting needs.

``fold_targeted``
    The ``non_exceptional`` law, fitted with ``targeting_scheme="fold"``.
``weighted``
    The ``misspecified_limit`` law with the known weight ``w = 1 + 1.5 * 1{W1 > 0}``. The
    declared outcome learner has no ``W1^2``, so the weighted and the unweighted rules have
    different limits (``tests/unit/test_learned_rule_compositions.py``). The target is the
    fold-average value in the weighted population, ``E[w Qbar0(d(W), W)] / E[w]``.
``clustered``
    The ``non_exceptional`` linear predictor plus a cluster effect ``u ~ N(0, SIGMA_U^2)``
    shared by the rows of each cluster, in clusters of equal size. The fit declares ``id=``. The
    row-level outcome regression is the marginal ``Qbar0(a, W) = E_u expit(eta(a, W) + u)``,
    computed by Gauss-Hermite quadrature with :data:`HERMITE_NODES` nodes.

The truth of each replication is the value of the rules its fit learned (rule L4 of the RM30
studies): the harness refits each fold's outcome learner on the fit's own training rows with
the fit's own weights, requires the fit's rule on every validation row, and integrates the rule's
value by the trapezoid rule on the RM30 grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_law as base

#: The outer folds of every fit, as in the RM30 studies.
N_FOLDS = base.N_FOLDS

#: The reported parameter of every fit.
ESTIMAND = base.ESTIMAND

#: Each scenario's RM30 law, which sets its blip.
LAWS: dict[str, str] = {
    "fold_targeted": "non_exceptional",
    "weighted": "misspecified_limit",
    "clustered": "non_exceptional",
    "clustered_few": "non_exceptional",
}

#: The scenarios of the primary phase; ``clustered_few`` is a property cell only.
PRIMARY_SCENARIOS = ("fold_targeted", "weighted", "clustered")

#: The cluster effect's standard deviation on the logit scale.
SIGMA_U = 1.5

#: The rows of every cluster: 200 clusters of 10 at n = 2,000, and 30 clusters of 34 at
#: n = 1,020 for the few-cluster cell, whose fit takes the t reference.
CLUSTER_SIZES = {"clustered": 10, "clustered_few": 34}

#: The Gauss-Hermite nodes of the marginal outcome regression of the ``clustered`` scenario.
HERMITE_NODES = 64


def weight(w1: np.ndarray) -> np.ndarray:
    """The known observation weight of the ``weighted`` scenario, ``1 + 1.5 * 1{W1 > 0}``."""
    return 1.0 + 1.5 * (np.asarray(w1, dtype=float) > 0.0)


@cache
def _hermite() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
    return nodes, weights / weights.sum()


def qbar0(scenario: str, a: Any, w1: np.ndarray, w2: Any) -> np.ndarray:
    """The row-level ``P(Y = 1 | A = a, W)`` of ``scenario``.

    For ``clustered`` it is the marginal over the cluster effect; for the others the RM30 law's
    own regression.
    """
    law = LAWS[scenario]
    if scenario not in CLUSTER_SIZES:
        return base.qbar0(a, w1, w2, law)
    w1 = np.asarray(w1, dtype=float)
    eta = 0.2 + 0.5 * w1 - 0.3 * np.asarray(w2) + np.asarray(a) * base.blip(law, w1)
    nodes, masses = _hermite()
    return np.asarray(
        np.tensordot(expit(eta[..., None] + SIGMA_U * nodes), masses, axes=([-1], [0])),
        dtype=float,
    )


def draw(scenario: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows of ``scenario``.

    ``weighted`` adds the weight column ``w``; ``clustered`` adds ``cluster`` and needs ``n`` to
    be a multiple of its size in :data:`CLUSTER_SIZES`.
    """
    rng = np.random.default_rng(seed)
    law = LAWS[scenario]
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, base.g0(w1, w2)).astype(float)
    if scenario in CLUSTER_SIZES:
        size = CLUSTER_SIZES[scenario]
        if n % size:
            raise ValueError(f"n = {n} is not a multiple of the cluster size {size}")
        cluster = np.repeat(np.arange(n // size), size)
        u = rng.normal(0.0, SIGMA_U, n // size)[cluster]
        eta = 0.2 + 0.5 * w1 - 0.3 * w2 + a * base.blip(law, w1) + u
        y = rng.binomial(1, expit(eta)).astype(float)
        return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "cluster": cluster})
    y = rng.binomial(1, base.qbar0(a, w1, w2, law)).astype(float)
    frame = pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})
    if scenario == "weighted":
        frame["w"] = weight(w1)
    return frame


def estimator(scenario: str, learner: Any, fold_seed: int) -> TMLE:
    """The declared estimator of ``scenario``, with the fold seed of rule L2."""
    return TMLE(
        learned_rule=LearnedRule(),
        n_folds=N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="fold" if scenario == "fold_targeted" else "pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


#: The fit arguments of each scenario beside the outcome, treatment and covariates.
FIT_ARGUMENTS: dict[str, dict[str, str]] = {
    "fold_targeted": {},
    "weighted": {"weights": "w"},
    "clustered": {"id": "cluster"},
    "clustered_few": {"id": "cluster"},
}


def fit(frame: pd.DataFrame, scenario: str, learner: Any, fold_seed: int) -> Any:
    """Fit the declared estimator of ``scenario`` to one sample and return the single result."""
    return (
        estimator(scenario, learner, fold_seed)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(base.COVARIATES),
            **FIT_ARGUMENTS[scenario],
        )
        .single()
    )


def rule_value(treat: Mapping[float, np.ndarray], scenario: str) -> tuple[float, float]:
    """``Psi_d`` and the variance of its row-level curve, for a rule given on the grid.

    The value is ``E[w Qbar0(d(W), W)] / E[w]`` with ``w = 1`` outside ``weighted``. The
    variance is ``E[(w / E w)^2 {Q (1 - Q) / g + (Q - Psi)^2}]``, the variance of the weighted
    row-level curve. For ``clustered`` it ignores the cluster correlation, so it is a row-level
    reference only (descriptive; no rule reads it).
    """
    grid = base.GRID
    mass = 0.0
    total = 0.0
    pieces = []
    for w2 in base.W2_LEVELS:
        rule = np.asarray(treat[w2], dtype=float)
        q = qbar0(scenario, rule, grid, w2)
        treated = base.g0(grid, w2)
        g = np.where(rule > 0.5, treated, 1.0 - treated)
        w = weight(grid) if scenario == "weighted" else np.ones_like(grid)
        mass += 0.5 * np.trapezoid(w, grid) / 2.0
        total += 0.5 * np.trapezoid(w * q, grid) / 2.0
        pieces.append((w, q, g))
    value = total / mass
    second = 0.0
    for w, q, g in pieces:
        ratio = w / mass
        second += 0.5 * np.trapezoid(ratio**2 * (q * (1.0 - q) / g + (q - value) ** 2), grid) / 2.0
    return float(value), float(second)


def population_rule(scenario: str, points: int = 4_001) -> dict[float, np.ndarray]:
    """The rule of the declared learner's population projection on ``scenario``, on the grid.

    Each grid point of ``(W1, W2)`` carries its trapezoid mass, each arm its propensity, each
    outcome its probability under :func:`qbar0`, and each row its known weight, so the weighted
    log-likelihood the learner maximises is the population one. The learner's ridge is removed.
    """
    grid = np.linspace(-1.0, 1.0, points)
    mass = np.full(points, 2.0 / (points - 1))
    mass[[0, -1]] /= 2.0
    design, sample_weight, outcome = [], [], []
    for w2 in base.W2_LEVELS:
        treated = base.g0(grid, w2)
        w = weight(grid) if scenario == "weighted" else np.ones(points)
        for a in (0.0, 1.0):
            q = qbar0(scenario, a, grid, w2)
            for y, p_y in ((1.0, q), (0.0, 1.0 - q)):
                design.append(np.column_stack([np.full(points, a), grid, np.full(points, w2)]))
                sample_weight.append(mass * (treated if a else 1.0 - treated) * p_y * w)
                outcome.append(np.full(points, y))
    model = base.outcome_learner().set_params(
        model__C=1e12, model__tol=1e-13, model__max_iter=20_000
    )
    weights = np.concatenate(sample_weight)
    model.fit(
        np.vstack(design), np.concatenate(outcome), model__sample_weight=weights / weights.mean()
    )
    return {
        w2: base._blip(model, np.column_stack([base.GRID, np.full_like(base.GRID, w2)])) > 0.0
        for w2 in base.W2_LEVELS
    }


def cluster_covariance(treat: Mapping[float, np.ndarray], scenario: str) -> float:
    """The covariance of the row curves of two rows of one cluster, at the rule ``treat``.

    Two rows of a cluster share only ``u``, so the covariance is ``E_u[m(u)^2]`` with
    ``m(u) = E_W[expit(eta(d(W), W) + u) - Qbar0(d(W), W)]``: the residual term
    ``H (Y - Qbar0)`` averaged over ``A`` given ``W``, whose mean given ``u`` is that difference
    at the rule's arm. The terms in ``Qbar0(d, W) - Psi`` are independent across rows.
    """
    nodes, masses = _hermite()
    law = LAWS[scenario]
    shift = np.zeros_like(nodes)
    for w2 in base.W2_LEVELS:
        rule = np.asarray(treat[w2], dtype=float)
        eta = 0.2 + 0.5 * base.GRID - 0.3 * w2 + rule * base.blip(law, base.GRID)
        q = qbar0(scenario, rule, base.GRID, w2)
        conditional = expit(eta[:, None] + SIGMA_U * nodes) - q[:, None]
        shift += 0.5 * np.trapezoid(conditional, base.GRID, axis=0) / 2.0
    return float(np.sum(masses * shift**2))


@dataclass(frozen=True)
class Truth:
    """The truth of one fit; the fields of :class:`tests.studies._learned_rule_law.Truth`."""

    truth: float
    oracle_se: float
    fold_values: tuple[float, ...]
    rows_checked: int


def harness(result: Any, learner: Any, scenario: str) -> Truth:
    """Refit each fold's rule on the fit's training rows, check it, and integrate its value.

    The refit goes through the package's ``fit_on_rows`` with the fit's own (normalised)
    weights and clusters, so a deterministic learner reproduces the fitted model exactly.

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When the refit rule differs from the fit's rule on any validation row.
    """
    data = result.data
    design = np.asarray(data.treatment_design(), dtype=float)
    covariates = np.asarray(data.covariates, dtype=float)
    outcome = np.asarray(data.outcome, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    fitted = base.rule_of(result)
    grid = {w2: np.column_stack([base.GRID, np.full_like(base.GRID, w2)]) for w2 in base.W2_LEVELS}
    values: list[float] = []
    variances: list[float] = []
    sizes: list[int] = []
    checked = 0
    for fold, (train, test) in enumerate(result.nuisance.folds):
        model = fit_on_rows(
            learner, design, outcome, weights, np.asarray(train), "classification", data.cluster
        )
        refit = base._blip(model, covariates[test]) > 0.0
        disagree = int(np.sum(refit != fitted[test]))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        checked += len(test)
        value, variance = rule_value(
            {w2: base._blip(model, points) > 0.0 for w2, points in grid.items()}, scenario
        )
        values.append(value)
        variances.append(variance)
        sizes.append(len(test))
    folds = len(values)
    oracle = float(np.sqrt(sum(v / m for v, m in zip(variances, sizes, strict=True))) / folds)
    return Truth(float(np.mean(values)), oracle, tuple(values), checked)


def measured(
    frame: pd.DataFrame, scenario: str, learner: Any, fold_seed: int
) -> tuple[Any, Truth, int]:
    """Fit one sample, run its harness, and count the solver warnings of both (rule L4)."""
    with base.solver_warning_count() as count:
        result = fit(frame, scenario, learner, fold_seed)
        truth = harness(result, learner, scenario)
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    return result, truth, count[0]


def row_variance_se(result: Any) -> float:
    """The standard error the fit would report if it ignored its clusters.

    The fold-evaluated row form ``V^-2 sum_v n_v^-2 sum_{i in v} D_i^2`` with the fit's own
    targeted curve. The ``clustered`` scenario's ``row_variance_control`` reports it beside the
    fit's estimate, so it must under-cover when the cluster effect is strong.
    """
    data = result.data
    nuisance = result.nuisance
    fluctuation = result.fluctuations["regime"]
    treat = base.rule_of(result).astype(float)
    treatment = np.asarray(data.treatment, dtype=float)
    lower, upper = result.config.g_bounds
    g1 = np.clip(np.asarray(nuisance.propensity.arm(1.0), dtype=float), lower, upper)
    g_observed = np.where(treatment == 1.0, g1, 1.0 - g1)
    targeted = fluctuation.targeted
    q_rule = np.where(treat == 1.0, targeted.arms[1.0], targeted.arms[0.0])
    y = np.asarray(nuisance.scaler.scale(data.outcome), dtype=float)
    core = (treatment == treat) / g_observed * (y - np.asarray(targeted.observed, dtype=float))
    variance = 0.0
    tests = [np.asarray(test) for _, test in nuisance.folds]
    for test in tests:
        curve = core[test] + q_rule[test] - np.mean(q_rule[test])
        variance += float(np.sum(curve**2)) / test.size**2
    return float(np.sqrt(variance) / len(tests) * nuisance.scaler.range)
