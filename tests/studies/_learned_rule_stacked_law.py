r"""The laws, fits and truth harness of the ``learned-rule-stacked`` study (X11 PR F1, part (e)).

The study measures the stacked report of the learned-rule value, ``cv_evaluation=False``: one
pooled fluctuation over every validation row, the estimate
:math:`\hat\psi = \sum_v (W_v/W)\psi^*_{nv}`, and the target
:math:`\sum_v (W_v/W)\Psi_{d_{nv}}(P_0)`, where :math:`W_v` is fold :math:`v`'s weight mass
(:math:`n_v` without weights). It reads the RM30 laws and learners of
:mod:`tests.studies._learned_rule_law` and the weighted law of
:mod:`tests.studies._learned_rule_compositions_law` without editing either, and adds one law.

``rows``
    The RM30 ``non_exceptional`` law, drawn by RM30's ``draw``. At n = 2,000 the ten folds hold
    200 rows each, so :math:`W_v/W = 1/V` and the cell measures the pooled centring alone.
``weighted``
    The ``weighted`` law of ``learned-rule-compositions``, drawn by its ``draw``: the
    ``misspecified_limit`` blip with the known weight ``1 + 1.5 * 1{W1 > 0}``. Equal row counts
    carry unequal weight masses, so :math:`W_v/W \ne 1/V`.
``clustered``
    The ``non_exceptional`` linear predictor plus a cluster effect :math:`\sigma_u z_j`,
    :math:`z_j \sim N(0, 1)`, shared by the rows of cluster :math:`j`, with a cluster size that
    increases in :math:`z_j`: :math:`N_j = 5 + \mathrm{Binomial}(10, \operatorname{expit}(\kappa
    z_j))`, so sizes run from 5 to 15 with mean 10. A draw of nominal ``n`` holds ``n // 10``
    clusters. Every row of a cluster shares :math:`z_j`, and :math:`W_1, W_2, A` are drawn per
    row from the RM30 law.

Cluster sizes that depend on the outcome make the row population size-biased: a row's cluster
effect has density proportional to :math:`E(N \mid z)\,\phi(z)`. The row-level outcome
regression is therefore
:math:`\bar Q_0(a, W) = \int \operatorname{expit}(\eta(a, W) + \sigma_u z)\,E(N\mid z)\phi(z)\,dz
/ E(N)`, by Gauss-Hermite quadrature on :data:`HERMITE_NODES` nodes.

The truth of each replication is the value of the rules its fit learned, weighted by the fit's
own fold masses (rule L4 of the RM30 studies): the harness refits each fold's outcome learner on
the fit's training rows with the fit's weights and clusters, requires the fit's rule on every
validation row, integrates each rule's value by the trapezoid rule on the RM30 grid, and weights
fold :math:`v` by :math:`W_v/W` computed from the fit's data and folds. It checks those weights
against the record's ``fold_weights``. A fold-evaluated fit of the same draw (the reporting row
of the clustered cell) is weighted by :math:`1/V`.
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
from cleverly.inference import influence_variance
from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_compositions_law as compositions
from tests.studies import _learned_rule_law as base

#: The outer folds of every fit, as in the RM30 studies.
N_FOLDS = base.N_FOLDS

#: The reported parameter of every fit.
ESTIMAND = base.ESTIMAND

#: The scenarios, each a calibration cell and a primary scenario.
SCENARIOS = ("rows", "weighted", "clustered")

#: Each scenario's RM30 law, which sets its blip.
LAWS: dict[str, str] = {
    "rows": "non_exceptional",
    "weighted": "misspecified_limit",
    "clustered": "non_exceptional",
}

#: The cluster effect's standard deviation on the logit scale.
SIGMA_U = 1.0

#: The slope of the size law on the standardized cluster effect: ``P(trial) = expit(KAPPA z)``.
KAPPA = 1.5

#: The size law ``N = SIZE_FLOOR + Binomial(SIZE_TRIALS, expit(KAPPA z))``: 5 to 15, mean 10.
SIZE_FLOOR = 5
SIZE_TRIALS = 10
MEAN_SIZE = SIZE_FLOOR + SIZE_TRIALS // 2

#: The Gauss-Hermite nodes of every integral over the cluster effect.
HERMITE_NODES = 64

#: The columns a row carries beside the shared schema. ``oracle_se`` is descriptive.
#: ``rule_rows_checked`` counts the validation rows whose refit rule equals the fit's rule.
#: ``solver_warnings`` counts the warnings of the fit and of its truth refits. ``cross_term`` and
#: ``between_term`` are the two extra terms of the stacked variance identity, each over the
#: fold-evaluated second moment (:func:`variance_terms`). ``rows`` is the realized row count;
#: a clustered draw publishes the nominal ``n``. No rule reads any of them.
HARNESS_COLUMNS = (
    "oracle_se",
    "rule_rows_checked",
    "solver_warnings",
    "cross_term",
    "between_term",
    "rows",
)


# --------------------------------------------------------------------------- the law


@cache
def _hermite() -> tuple[np.ndarray, np.ndarray]:
    """Standard normal nodes and masses."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
    return nodes, weights / weights.sum()


def size_probability(z: np.ndarray) -> np.ndarray:
    """The per-trial probability of the size law at the standardized cluster effect ``z``."""
    return np.asarray(expit(KAPPA * np.asarray(z, dtype=float)), dtype=float)


def mean_size(z: np.ndarray) -> np.ndarray:
    """``E(N | z)``."""
    return SIZE_FLOOR + SIZE_TRIALS * size_probability(z)


def second_size_moment(z: np.ndarray) -> np.ndarray:
    """``E(N^2 | z)``."""
    p = size_probability(z)
    return SIZE_TRIALS * p * (1.0 - p) + mean_size(z) ** 2


def row_masses() -> np.ndarray:
    """The size-biased masses of the cluster effect at the Hermite nodes: a row's law of ``z``."""
    nodes, masses = _hermite()
    biased = masses * mean_size(nodes)
    return np.asarray(biased / biased.sum(), dtype=float)


def linear_predictor(scenario: str, a: Any, w1: np.ndarray, w2: Any) -> np.ndarray:
    """``0.2 + 0.5 W1 - 0.3 W2 + a b(W)``, the RM30 logit of ``scenario``'s law."""
    w1 = np.asarray(w1, dtype=float)
    law = LAWS[scenario]
    return 0.2 + 0.5 * w1 - 0.3 * np.asarray(w2) + np.asarray(a) * base.blip(law, w1)


def conditional_q(a: Any, w1: np.ndarray, w2: Any) -> np.ndarray:
    """``P(Y = 1 | A = a, W, z)`` of the clustered law, with the nodes on the last axis."""
    nodes, _ = _hermite()
    eta = linear_predictor("clustered", a, w1, w2)
    return np.asarray(expit(eta[..., None] + SIGMA_U * nodes), dtype=float)


def qbar0(scenario: str, a: Any, w1: np.ndarray, w2: Any) -> np.ndarray:
    """The row-level ``P(Y = 1 | A = a, W)`` of ``scenario``.

    For ``clustered`` it integrates the cluster effect over a row's size-biased law; for the
    others it is the RM30 law's own regression.
    """
    if scenario != "clustered":
        return base.qbar0(a, w1, w2, LAWS[scenario])
    return np.asarray(
        np.tensordot(conditional_q(a, w1, w2), row_masses(), axes=([-1], [0])), dtype=float
    )


def weight(scenario: str, w1: np.ndarray) -> np.ndarray:
    """The known observation weight: the compositions weight in ``weighted``, 1 otherwise."""
    if scenario == "weighted":
        return compositions.weight(w1)
    return np.ones_like(np.asarray(w1, dtype=float))


def draw_clustered(n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n // MEAN_SIZE`` clusters of the clustered law.

    The order of the draws is the cluster effects, the sizes, then the rows' ``W1``, ``W2``,
    ``A`` and ``Y``. The frame holds ``W1, W2, A, Y, cluster``.
    """
    if n % MEAN_SIZE:
        raise ValueError(f"n = {n} is not a multiple of the mean cluster size {MEAN_SIZE}")
    rng = np.random.default_rng(seed)
    clusters = n // MEAN_SIZE
    z = rng.normal(0.0, 1.0, clusters)
    sizes = SIZE_FLOOR + rng.binomial(SIZE_TRIALS, size_probability(z))
    cluster = np.repeat(np.arange(clusters), sizes)
    rows = cluster.size
    w1 = rng.uniform(-1.0, 1.0, rows)
    w2 = rng.binomial(1, 0.5, rows).astype(float)
    a = rng.binomial(1, base.g0(w1, w2)).astype(float)
    eta = linear_predictor("clustered", a, w1, w2) + SIGMA_U * z[cluster]
    y = rng.binomial(1, expit(eta)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "cluster": cluster})


def draw(scenario: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``scenario``: RM30's draw, the compositions draw, or the clustered one."""
    if scenario == "rows":
        return base.draw(LAWS["rows"], n, seed)
    if scenario == "weighted":
        return compositions.draw("weighted", n, seed)
    if scenario == "clustered":
        return draw_clustered(n, seed)
    raise KeyError(f"unknown scenario {scenario!r}")


# ------------------------------------------------------------------------------ the fit


def estimator(learner: Any, fold_seed: int, *, cv_evaluation: bool = False) -> TMLE:
    """The declared estimator with the fold seed of rule L2: the stacked report by default.

    ``cv_evaluation=True`` gives the fold-evaluated report of the clustered reporting row.
    """
    return TMLE(
        learned_rule=LearnedRule(),
        n_folds=N_FOLDS,
        cv_evaluation=cv_evaluation,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


#: The fit arguments of each scenario beside the outcome, treatment and covariates.
FIT_ARGUMENTS: dict[str, dict[str, str]] = {
    "rows": {},
    "weighted": {"weights": "w"},
    "clustered": {"id": "cluster"},
}


def fit(
    frame: pd.DataFrame,
    scenario: str,
    learner: Any,
    fold_seed: int,
    *,
    cv_evaluation: bool = False,
) -> Any:
    """Fit the declared estimator of ``scenario`` to one sample and return the single result."""
    return (
        estimator(learner, fold_seed, cv_evaluation=cv_evaluation)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(base.COVARIATES),
            **FIT_ARGUMENTS[scenario],
        )
        .single()
    )


# ---------------------------------------------------------------- moments at a rule


@dataclass(frozen=True)
class ClusterMoments:
    r"""The moments of the row curve of the clustered law at one rule.

    Parameters
    ----------
    value : float
        :math:`\Psi_d = E_{\text{row}}\{\bar Q_0(d(W), W)\}`.
    cluster_means : ndarray
        :math:`c(z) = E_W\{\operatorname{expit}(\eta(d(W), W) + \sigma_u z)\}` at each node,
        the mean of a row's curve given its cluster effect.
    within_variances : ndarray
        :math:`v(z)`, the variance of a row's curve given its cluster effect, at each node.
    row_variance : float
        The variance of one row's curve under the size-biased row law, what a row rule reads.
    cluster_variance : float
        :math:`E(T^2)/E(N)`, with :math:`T` the cluster total of the centred curve: ``n`` times
        the asymptotic variance of the estimate.
    """

    value: float
    cluster_means: np.ndarray
    within_variances: np.ndarray
    row_variance: float
    cluster_variance: float

    @property
    def curve_icc(self) -> float:
        """The intra-cluster correlation of the row curve under the row law."""
        between = float(np.sum(row_masses() * (self.cluster_means - self.value) ** 2))
        return between / self.row_variance

    @property
    def size_curve_correlation(self) -> float:
        """The correlation of the cluster size with the cluster mean of the curve, by cluster."""
        nodes, masses = _hermite()
        size = mean_size(nodes)
        mean_c = float(np.sum(masses * self.cluster_means))
        covariance = float(np.sum(masses * (size - MEAN_SIZE) * (self.cluster_means - mean_c)))
        size_variance = float(np.sum(masses * (second_size_moment(nodes) - MEAN_SIZE**2)))
        curve_variance = float(np.sum(masses * (self.cluster_means - mean_c) ** 2))
        return covariance / np.sqrt(size_variance * curve_variance)


def cluster_moments(treat: Mapping[float, np.ndarray]) -> ClusterMoments:
    """The moments of the clustered law's row curve at the rule ``treat`` given on the grid.

    Given ``z`` and ``W`` the curve ``Z = H (Y - Qbar0) + Qbar0(d, W)`` has mean
    ``Q_z = expit(eta(d, W) + sigma_u z)`` and variance ``{Q_z (1 - Q_z) + (Q_z - Qbar0)^2} / g
    - (Q_z - Qbar0)^2``, with ``g = g0(d(W) | W)``. Every ``W`` integral is the trapezoid rule on
    the RM30 grid, and every ``z`` integral the Hermite rule.
    """
    grid = base.GRID
    nodes, masses = _hermite()
    means = np.zeros_like(nodes)
    second = np.zeros_like(nodes)
    conditional_variance = np.zeros_like(nodes)
    for w2 in base.W2_LEVELS:
        rule = np.asarray(treat[w2], dtype=float)
        q_z = conditional_q(rule, grid, w2)
        q_bar = qbar0("clustered", rule, grid, w2)[:, None]
        treated = base.g0(grid, w2)
        g = np.where(rule > 0.5, treated, 1.0 - treated)[:, None]
        gap = q_z - q_bar
        within = (q_z * (1.0 - q_z) + gap**2) / g - gap**2
        means += 0.5 * np.trapezoid(q_z, grid, axis=0) / 2.0
        second += 0.5 * np.trapezoid(q_z**2, grid, axis=0) / 2.0
        conditional_variance += 0.5 * np.trapezoid(within, grid, axis=0) / 2.0
    within_variances = conditional_variance + second - means**2
    row = row_masses()
    value = float(np.sum(row * means))
    row_variance = float(np.sum(row * (within_variances + (means - value) ** 2)))
    total = np.sum(
        masses
        * (mean_size(nodes) * within_variances + second_size_moment(nodes) * (means - value) ** 2)
    )
    cluster_variance = float(total / np.sum(masses * mean_size(nodes)))
    return ClusterMoments(value, means, within_variances, row_variance, cluster_variance)


def rule_value(treat: Mapping[float, np.ndarray], scenario: str) -> tuple[float, float]:
    """``Psi_d`` and the variance of the row curve at the rule ``treat`` given on the grid.

    ``rows`` is RM30's ``rule_value``, ``weighted`` the compositions one, and ``clustered`` reads
    :func:`cluster_moments`; for it the variance is row-level, a descriptive reference.
    """
    if scenario == "rows":
        return base.rule_value(treat, LAWS["rows"])
    if scenario == "weighted":
        return compositions.rule_value(treat, "weighted")
    moments = cluster_moments(treat)
    return moments.value, moments.row_variance


def population_rule(scenario: str, points: int = 4_001) -> dict[float, np.ndarray]:
    """The rule of the declared learner's population projection on ``scenario``, on the grid.

    ``rows`` and ``weighted`` are the compositions projections of the same laws
    (``fold_targeted`` reads ``non_exceptional`` unweighted). ``clustered`` projects the
    size-biased row regression the same way.
    """
    if scenario == "rows":
        return compositions.population_rule("fold_targeted", points)
    if scenario == "weighted":
        return compositions.population_rule("weighted", points)
    grid = np.linspace(-1.0, 1.0, points)
    mass = np.full(points, 2.0 / (points - 1))
    mass[[0, -1]] /= 2.0
    design, sample_weight, outcome = [], [], []
    for w2 in base.W2_LEVELS:
        treated = base.g0(grid, w2)
        for a in (0.0, 1.0):
            q = qbar0("clustered", a, grid, w2)
            for y, p_y in ((1.0, q), (0.0, 1.0 - q)):
                design.append(np.column_stack([np.full(points, a), grid, np.full(points, w2)]))
                sample_weight.append(mass * (treated if a else 1.0 - treated) * p_y)
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


def efficiency_bound(scenario: str) -> float:
    """The SD of ``sqrt(n)`` times the estimate at the scenario's limit rule.

    The row curve's SD for ``rows`` and ``weighted``; ``sqrt(E(T^2) / E(N))`` for
    ``clustered``. They size the noise controls only.
    """
    rule = population_rule(scenario)
    if scenario == "clustered":
        return float(np.sqrt(cluster_moments(rule).cluster_variance))
    _, variance = rule_value(rule, scenario)
    return float(np.sqrt(variance))


# -------------------------------------------------------------------------- the harness


@dataclass(frozen=True)
class Truth:
    """The truth of one fit.

    Parameters
    ----------
    truth : float
        ``sum_v omega_v Psi_{d_v}(P0)`` with the fit's scheme weights ``omega_v``.
    oracle_se : float
        ``sqrt(sum_v omega_v^2 Var D_v / n_v)`` at each fold's rule, row-level; descriptive.
    fold_values : tuple of float
        ``Psi_{d_v}(P0)`` for each fold.
    fold_weights : tuple of float
        ``W_v / W`` from the fit's weights and folds for the stacked report, ``1 / V`` for the
        fold-evaluated one.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    """

    truth: float
    oracle_se: float
    fold_values: tuple[float, ...]
    fold_weights: tuple[float, ...]
    rows_checked: int


def fold_masses(result: Any) -> np.ndarray:
    """Each fold's weight mass, from the fit's own (mean-one) weights and validation folds."""
    weights = np.asarray(result.data.weights, dtype=float)
    return np.array([float(np.sum(weights[np.asarray(test)])) for _, test in result.nuisance.folds])


def scheme_weights(result: Any) -> np.ndarray:
    """The fold weights of the fit's report, computed from its data, and checked on its record.

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When the record's ``fold_weights`` differ from the weights computed here.
    """
    (split,) = result.extra["learned_rule"].splits
    if split.scheme == "stacked":
        masses = fold_masses(result)
        weights = masses / masses.sum()
    else:
        weights = np.full(len(result.nuisance.folds), 1.0 / len(result.nuisance.folds))
    if not np.allclose(split.fold_weights, weights, rtol=0.0, atol=1e-12):
        raise base.HarnessMismatch(
            f"the record's {split.scheme} fold weights {split.fold_weights} are not {weights}"
        )
    return weights


def harness(result: Any, learner: Any, scenario: str) -> Truth:
    """Refit each fold's rule on the fit's training rows, check it, and weight its value.

    The refit goes through the package's ``fit_on_rows`` with the fit's own weights and
    clusters, so a deterministic learner reproduces the fitted model exactly.

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When the refit rule differs from the fit's rule on any validation row, or the record's
        fold weights from the fit's own.
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
    omega = scheme_weights(result)
    oracle = float(
        np.sqrt(sum(w**2 * v / m for w, v, m in zip(omega, variances, sizes, strict=True)))
    )
    return Truth(
        truth=float(np.dot(omega, values)),
        oracle_se=oracle,
        fold_values=tuple(values),
        fold_weights=tuple(float(w) for w in omega),
        rows_checked=checked,
    )


def measured(
    frame: pd.DataFrame,
    scenario: str,
    learner: Any,
    fold_seed: int,
    *,
    cv_evaluation: bool = False,
) -> tuple[Any, Truth, int]:
    """Fit one sample, run its harness, and count the solver warnings of both (rule L4)."""
    with base.solver_warning_count() as count:
        result = fit(frame, scenario, learner, fold_seed, cv_evaluation=cv_evaluation)
        truth = harness(result, learner, scenario)
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    return result, truth, count[0]


# ------------------------------------------------------------------ the diagnostics


def variance_terms(result: Any) -> tuple[float, float]:
    r"""The cross and between terms of the stacked variance, each over the within term.

    With :math:`\phi_i = w_i (Z_i - \hat\psi)` the stored curve, :math:`\psi_v` fold
    :math:`v`'s recorded plug-in and :math:`D_i = Z_i - \psi_{v(i)}`,

    .. math::

        \sum_i \phi_i^2 = \sum_i w_i^2 D_i^2
        + 2 \sum_v (\psi_v - \hat\psi) \sum_{i \in v} w_i^2 D_i
        + \sum_v (\psi_v - \hat\psi)^2 \sum_{i \in v} w_i^2 .

    The first term is the fold-evaluated second moment. Returns the second and the third over
    it. Without weights this is the identity of ``point-treatment-tmle.md``.
    """
    estimate = result.estimates[ESTIMAND]
    weights = np.asarray(result.data.weights, dtype=float)
    centred = np.asarray(estimate.influence_curve, dtype=float) / weights
    (split,) = result.extra["learned_rule"].splits
    within = cross = between = 0.0
    for (_, test), plug_in in zip(result.nuisance.folds, split.fold_estimates, strict=True):
        rows = np.asarray(test)
        gap = float(plug_in) - float(estimate.psi)
        d = centred[rows] - gap
        w2 = weights[rows] ** 2
        within += float(np.sum(w2 * d**2))
        cross += 2.0 * gap * float(np.sum(w2 * d))
        between += gap**2 * float(np.sum(w2))
    return cross / within, between / within


def row_variance_se(result: Any) -> float:
    """The standard error the stacked fit would report if it ignored its clusters.

    The centred row rule ``{n (n - 1)}^-1 sum_i (phi_i - phibar)^2`` of the fit's own curve.
    The ``clustered`` cell's ``row_variance_control`` reports it beside the fit's estimate, so
    it must under-cover when the curve's intra-cluster correlation is material.
    """
    curve = np.asarray(result.estimates[ESTIMAND].influence_curve, dtype=float)
    return float(np.sqrt(influence_variance(curve)))


def same_partition(first: Any, second: Any) -> bool:
    """Whether two fits drew the same validation folds and used the same rule on every row."""
    folds = [np.asarray(test) for _, test in first.nuisance.folds]
    others = [np.asarray(test) for _, test in second.nuisance.folds]
    return (
        len(folds) == len(others)
        and all(np.array_equal(a, b) for a, b in zip(folds, others, strict=True))
        and bool(np.array_equal(base.rule_of(first), base.rule_of(second)))
    )


def diagnostic_columns(result: Any, truth: Truth, solver_warnings: int) -> dict[str, Any]:
    """The harness columns of one fit."""
    cross, between = variance_terms(result)
    return {
        "oracle_se": truth.oracle_se,
        "rule_rows_checked": truth.rows_checked,
        "solver_warnings": solver_warnings,
        "cross_term": cross,
        "between_term": between,
        "rows": int(result.data.n),
    }


# ------------------------------------------------------------------ the ratio bias


def cluster_mean_curve(treat: Mapping[float, np.ndarray], z: np.ndarray) -> np.ndarray:
    """``c(z) = E_W[expit(eta(d(W), W) + sigma_u z)]`` at the rule ``treat``, for each ``z``."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    for w2 in base.W2_LEVELS:
        rule = np.asarray(treat[w2], dtype=float)
        eta = linear_predictor("clustered", rule, base.GRID, w2)
        out += 0.5 * np.trapezoid(expit(eta[:, None] + SIGMA_U * z[None, :]), base.GRID, axis=0)
    return out / 2.0


def ratio_bias_oracle(
    draws: int, seed: int, *, clusters: int = 200, folds: int = N_FOLDS, equal_sizes: bool = False
) -> dict[str, float]:
    """The ratio bias of each report at a fixed rule, from the law alone, in stacked SDs.

    The method of ``tests/unit/test_fold_evaluated_ratio_bias.py``: no fit. Each cluster
    carries its size and the exact mean of its rows' curve given its effect, ``c(z_j)``, at the
    limit rule. The stacked report is the pooled row mean ``sum_j N_j c_j / sum_j N_j``; the
    fold-evaluated report is the ``1/V`` average of the same ratio over each fold's
    ``clusters / folds`` clusters, which are exchangeable, so dealing them in index order is a
    random split. The bias of each is against ``Psi_d = E(N c) / E(N)``, in units of the stacked
    SD ``sqrt(E(T^2) / E(N) / n)`` at ``n = clusters * 10``. ``equal_sizes`` fixes every size at
    10, the control under which the fold-evaluated bias vanishes.
    """
    rule = population_rule("clustered")
    moments = cluster_moments(rule)
    fine = np.linspace(-7.0, 7.0, 2_801)
    curve = cluster_mean_curve(rule, fine)
    sd = float(np.sqrt(moments.cluster_variance / (clusters * MEAN_SIZE)))
    psi = moments.value if not equal_sizes else float(np.sum(_hermite()[1] * moments.cluster_means))
    rng = np.random.default_rng(seed)
    stacked, evaluated = [], []
    block = 10_000
    for start in range(0, draws, block):
        size = min(block, draws - start)
        z = rng.normal(size=(size, clusters))
        if equal_sizes:
            n_j = np.full((size, clusters), MEAN_SIZE)
        else:
            n_j = SIZE_FLOOR + rng.binomial(SIZE_TRIALS, size_probability(z))
        total = n_j * np.interp(z, fine, curve)
        stacked.append(total.sum(axis=1) / n_j.sum(axis=1))
        shape = (size, folds, clusters // folds)
        evaluated.append(
            (total.reshape(shape).sum(axis=2) / n_j.reshape(shape).sum(axis=2)).mean(axis=1)
        )
    pooled = np.concatenate(stacked)
    by_fold = np.concatenate(evaluated)
    root = np.sqrt(draws)
    return {
        "stacked": float((pooled.mean() - psi) / sd),
        "stacked_se": float(pooled.std(ddof=1) / root / sd),
        "fold_evaluated": float((by_fold.mean() - psi) / sd),
        "fold_evaluated_se": float(by_fold.std(ddof=1) / root / sd),
    }
