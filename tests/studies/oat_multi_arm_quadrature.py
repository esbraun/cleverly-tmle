"""Exact-law quadrature for the per-arm OAT stack with ``K`` arms and observation weights.

The ``K``-arm and weighted counterpart of :mod:`tests.studies.oat_parametric_quadrature`
(X58 plan, 13.2 and 13.4). Each covariate node ``W = (W1, W2, W3)`` is expanded into the
``2 K`` cells ``(A, Y)`` with mass ``g_0(A | W) Qbar_0(A, W)^Y (1 - Qbar_0(A, W))^(1 - Y)``.
A weighted law multiplies each cell by its weight ``w(W)``, scaled to mean one under ``P``.
The production learners fit with the tilted masses as sample weights, and
:class:`~cleverly.estimators._oat_stack.StackProblem` takes them as its measure, so the grid's
Jacobian and curve rows are the production code's own at the population fit.

The reported curve of a weighted fit is ``w_i (-A_w^{-1} U_i)``: the stack's rows times the
weight, as :func:`~cleverly.inference.influence.counterfactual_means` multiplies the
corrected curve by ``w``. The asymptotic covariance of ``sqrt(n)`` times the estimates is the
second moment of those curves under ``P``.

A weight that jumps at ``W1 = 0`` takes the half-line grid of
:func:`half_line_nodes`, so no node straddles the jump.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, logit

from cleverly.estimators._oat_stack import ArmDesign, StackProblem, corrected_curves
from cleverly.learners import ParametricMechanism, ParametricOutcome
from cleverly.learners.parametric import OutcomeFeatures
from tests.studies import oat_per_arm_laws as laws
from tests.studies.oat_parametric_quadrature import _solve_fluctuation

ALPHA = 0.9995
"""cleverly's default shrinkage bound."""

Mechanism = Callable[[np.ndarray], np.ndarray]
Outcome = Callable[[int, np.ndarray], np.ndarray]
Weight = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ArmLaw:
    """A ``K``-arm law on three standard-normal covariates.

    Parameters
    ----------
    name : str
        The law's name.
    k : int
        The number of arms.
    mechanism : callable
        ``W -> g_0(. | W)``, shape ``(m, K)``.
    outcome : callable
        ``(a, W) -> Qbar_0(a, W)``.
    weight : callable or None
        ``W -> w(W)``, or ``None`` for an unweighted law.
    """

    name: str
    k: int
    mechanism: Mechanism = field(compare=False)
    outcome: Outcome = field(compare=False)
    weight: Weight | None = field(default=None, compare=False)


def from_oat_law(law: laws.OatLaw, *, weighted: bool | None = None) -> ArmLaw:
    """The quadrature form of a study law of :mod:`tests.studies.oat_per_arm_laws`.

    Parameters
    ----------
    law : OatLaw
        The study law.
    weighted : bool or None
        Whether to carry the declared weight; ``None`` follows ``law.weighted``.

    Returns
    -------
    ArmLaw
        The same law on the quadrature grid.
    """
    tilt = law.weighted if weighted is None else weighted
    return ArmLaw(
        law.name,
        law.k,
        lambda w: law.mechanism(w[:, 0], w[:, 2]),
        lambda a, w: law.outcome_probability(a, w[:, 0], w[:, 1]),
        (lambda w: laws.weight(w[:, 0])) if tilt else None,
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    expo = np.exp(scores)
    return np.asarray(expo / expo.sum(axis=1, keepdims=True))


THREE_ARM_ACTIVE = from_oat_law(laws.THREE_ARM_ACTIVE)
"""The three-arm law of ``ctmle-oat-per-arm``."""

THREE_ARM_SHARED = ArmLaw(
    "three_arm_shared",
    3,
    THREE_ARM_ACTIVE.mechanism,
    lambda a, w: np.asarray(expit((-0.4, 0.0, 0.4)[a] + 3.5 * w[:, 0] + 0.8 * w[:, 1])),
)
"""``three_arm_active``'s mechanism with a strong shared prognosis (X58 plan, 13.2): the arm
curves correlate at about 0.25."""

THREE_ARM_U_SHAPED = ArmLaw(
    "three_arm_u_shaped",
    3,
    lambda w: _softmax(
        np.column_stack(
            [
                np.zeros(w.shape[0]),
                -1.5 + 1.8 * w[:, 0] ** 2 + 0.5 * w[:, 2],
                0.5 * w[:, 0] - 0.4 * w[:, 2],
            ]
        )
    ),
    lambda a, w: np.asarray(
        expit((-0.2, 0.2, 0.4)[a] + (0.9, 1.2, -0.7)[a] * w[:, 0] + 0.4 * w[:, 1])
    ),
)
"""Arm 1's mechanism is U-shaped in ``W1``, so no arm's mechanism is a function of its own
``Qbar``: the cubic-logit mechanisms miss the projection, and arm 0's ``kappa`` is about 1.10
(X58 plan, 13.2, KW4)."""

THREE_ARM_STEEP = ArmLaw(
    "three_arm_steep",
    3,
    lambda w: _softmax(
        np.column_stack(
            [np.zeros(w.shape[0]), 2.5 * w[:, 0] + 0.8 * w[:, 2], 0.3 * w[:, 0] - 0.3 * w[:, 2]]
        )
    ),
    lambda a, w: np.asarray(expit((-0.2, 0.3, 0.1)[a] + 1.5 * w[:, 0] + 0.5 * w[:, 1])),
)
"""A steep mechanism on arm 1: ``g_bounds = (0.05, 0.95)`` binds on its rows (KW1)."""

BINARY_ACTIVE_WEIGHTED = from_oat_law(laws.BINARY_WEIGHTED)
"""``binary_active`` under the declared weight ``w = 0.5 + 1{W1 > 0}``."""

BINARY_ACTIVE = from_oat_law(laws.BINARY_ACTIVE)
"""The primary binary law."""


@dataclass(frozen=True)
class Grid:
    """The expanded quadrature grid.

    Parameters
    ----------
    covariates : ndarray
        ``(m, 3)``, ``W`` of each cell.
    treatment : ndarray
        ``(m,)``, the arm code ``A`` of each cell, in ``0..K-1``.
    outcome : ndarray
        ``(m,)``, ``Y`` of each cell.
    mass : ndarray
        ``(m,)``, the probability of each cell under ``P``; the masses sum to 1.
    weight : ndarray
        ``(m,)``, ``w`` of each cell, with mean one under ``P``. Ones for an unweighted law.
    """

    covariates: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    mass: np.ndarray
    weight: np.ndarray

    @property
    def tilt(self) -> np.ndarray:
        """``mass * weight``: the measure ``P_w`` up to its total."""
        return self.mass * self.weight


@lru_cache(maxsize=8)
def hermite_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """A tensor Gauss-Hermite rule of ``nodes`` points per axis on three standard normals.

    Parameters
    ----------
    nodes : int
        The points per axis.

    Returns
    -------
    tuple of ndarray
        ``(m, 3)`` nodes and ``(m,)`` masses summing to 1.
    """
    points, weights = hermegauss(nodes)
    weights = weights / weights.sum()
    mesh = np.meshgrid(points, points, points, indexing="ij")
    covariates = np.column_stack([axis.ravel() for axis in mesh])
    mass = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).ravel()
    return covariates, mass


@lru_cache(maxsize=8)
def half_line_nodes(legendre: int, hermite: int) -> tuple[np.ndarray, np.ndarray]:
    """A rule with ``W1`` split at zero: Gauss-Legendre on ``[-9, 0]`` and ``[0, 9]``.

    ``W2`` and ``W3`` take Gauss-Hermite nodes. A weight that jumps at ``W1 = 0`` is smooth on
    each half, so the rule converges for it.

    Parameters
    ----------
    legendre : int
        The Legendre points on each half of the ``W1`` axis.
    hermite : int
        The Hermite points on each other axis.

    Returns
    -------
    tuple of ndarray
        ``(m, 3)`` nodes and ``(m,)`` masses summing to 1.
    """
    x, wx = leggauss(legendre)
    half = 4.5
    positive = half * (x + 1.0)
    w1 = np.concatenate([-positive[::-1], positive])
    m1 = np.concatenate([wx[::-1], wx]) * half * np.exp(-0.5 * w1**2) / np.sqrt(2.0 * np.pi)
    points, weights = hermegauss(hermite)
    weights = weights / weights.sum()
    a, b, c = np.meshgrid(w1, points, points, indexing="ij")
    covariates = np.column_stack([a.ravel(), b.ravel(), c.ravel()])
    mass = (m1[:, None, None] * weights[None, :, None] * weights[None, None, :]).ravel()
    return covariates, mass / mass.sum()


def grid(law: ArmLaw, covariates: np.ndarray, mass: np.ndarray) -> Grid:
    """The law's ``2 K`` cells at each node.

    Parameters
    ----------
    law : ArmLaw
        The law.
    covariates : ndarray
        ``(m, 3)`` nodes.
    mass : ndarray
        ``(m,)`` node masses.

    Returns
    -------
    Grid
        The cells, with the weight at mean one under ``P``.
    """
    g = law.mechanism(covariates)
    parts = []
    for arm in range(law.k):
        q = law.outcome(arm, covariates)
        for y in (0, 1):
            cell = mass * g[:, arm] * (q if y else 1.0 - q)
            parts.append(
                (covariates, np.full(mass.size, float(arm)), np.full(mass.size, float(y)), cell)
            )
    cells = np.concatenate([part[3] for part in parts])
    cells = cells / cells.sum()
    w = np.concatenate([part[0] for part in parts])
    weight = np.ones(cells.size) if law.weight is None else np.asarray(law.weight(w), dtype=float)
    return Grid(
        covariates=w,
        treatment=np.concatenate([part[1] for part in parts]),
        outcome=np.concatenate([part[2] for part in parts]),
        mass=cells,
        weight=weight / float(cells @ weight),
    )


def treatment_block(codes: np.ndarray, k: int) -> np.ndarray:
    """The drop-first arm block of :meth:`~cleverly.data.CausalData.treatment_block`.

    Parameters
    ----------
    codes : ndarray
        ``(m,)`` arm codes in ``0..K-1``.
    k : int
        The number of arms.

    Returns
    -------
    ndarray
        ``(m, 1)`` codes at two arms, ``(m, K - 1)`` indicators of arms ``1..K-1`` otherwise.
    """
    if k == 2:
        return codes.reshape(-1, 1).astype(float)
    return np.column_stack([(codes == arm).astype(float) for arm in range(1, k)])


@dataclass(frozen=True)
class PopulationFit:
    """The population per-arm OAT fit and its stack on a grid.

    Parameters
    ----------
    law : ArmLaw
        The law.
    grid : Grid
        The cells.
    problem : StackProblem
        The stack on the grid's tilted measure ``P_w``.
    theta : ndarray
        The population ``theta*``.
    psi_star : ndarray
        ``(K,)``, the limit of the arm means under ``P_w``.
    psi_obs : float
        ``E_w[Y]``, the natural-course mean under ``P_w``.
    epsilons : ndarray
        ``(K,)``, ``eps*`` of each arm.
    clipped : ndarray
        ``(K,)``, the ``P_w`` mass on which ``g_bounds`` binds, per arm.
    """

    law: ArmLaw
    grid: Grid
    problem: StackProblem
    theta: np.ndarray
    psi_star: np.ndarray
    psi_obs: float
    epsilons: np.ndarray
    clipped: np.ndarray


def population_fit(
    law: ArmLaw,
    covariates: np.ndarray,
    mass: np.ndarray,
    *,
    features: OutcomeFeatures = "arm_interactions",
    bounds: tuple[float, float] = laws.G_BOUNDS,
    alpha: float = ALPHA,
    degree: int = 3,
) -> PopulationFit:
    """The population per-arm OAT fit of ``law`` on the given nodes.

    Parameters
    ----------
    law : ArmLaw
        The law.
    covariates : ndarray
        ``(m, 3)`` nodes.
    mass : ndarray
        ``(m,)`` node masses.
    features : str
        The outcome feature map of :class:`~cleverly.learners.ParametricOutcome`.
    bounds : tuple of float
        ``g_bounds``.
    alpha : float
        The shrinkage bound.
    degree : int
        The degree of the mechanism basis in ``logit q``.

    Returns
    -------
    PopulationFit
        The fit and its stack.
    """
    cells = grid(law, covariates, mass)
    tilt = cells.tilt
    share = tilt / tilt.sum()
    block = treatment_block(cells.treatment, law.k)
    outcome_model = ParametricOutcome(features, arm_columns=block.shape[1]).fit(
        np.column_stack([block, cells.covariates]), cells.outcome, sample_weight=tilt
    )
    beta = outcome_model.coef_
    template = ParametricMechanism(degree=degree)
    designs: list[ArmDesign] = []
    gammas: list[np.ndarray] = []
    epsilons: list[float] = []
    psis: list[float] = []
    clipped: list[float] = []
    for arm in range(law.k):
        at_arm = treatment_block(np.full(cells.outcome.size, float(arm)), law.k)
        x_arm = outcome_model.design_matrix(np.column_stack([at_arm, cells.covariates]))
        q = expit(x_arm @ beta)
        indicator = (cells.treatment == arm).astype(float)
        mechanism = ParametricMechanism(degree=degree).fit(
            q[:, None], indicator, sample_weight=tilt
        )
        g = mechanism.predict(q[:, None])
        clever = 1.0 / np.clip(g, *bounds)
        level = logit(np.clip(q, 1.0 - alpha, alpha))
        eps = _solve_fluctuation(indicator, cells.outcome, level, clever, share, alpha)
        targeted = np.clip(expit(level + eps * clever), 1.0 - alpha, alpha)
        designs.append(
            ArmDesign(
                features=x_arm,
                indicator=indicator,
                basis=template.design_matrix,
                basis_derivative=template.derivative,
                bounds=bounds,
            )
        )
        gammas.append(mechanism.coef_)
        epsilons.append(eps)
        psis.append(float(share @ targeted))
        clipped.append(float(share @ ((g <= bounds[0]) | (g >= bounds[1]))))
    problem = StackProblem(
        outcome_model.design_matrix(np.column_stack([block, cells.covariates])),
        cells.outcome,
        designs,
        alpha,
        mass=tilt,
    )
    theta = problem.pack([beta], [[gamma] for gamma in gammas], epsilons, psis)
    return PopulationFit(
        law=law,
        grid=cells,
        problem=problem,
        theta=theta,
        psi_star=np.array(psis),
        psi_obs=float(share @ cells.outcome),
        epsilons=np.array(epsilons),
        clipped=np.array(clipped),
    )


def arm_curves(fit: PopulationFit) -> np.ndarray:
    """``(m, K)``, the reported stacked curves ``w_i (-A_w^{-1} U_i)`` of the arm means."""
    return corrected_curves(fit.problem, fit.theta).curves * fit.grid.weight[:, None]


def shipped_curves(fit: PopulationFit) -> np.ndarray:
    """``(m, K)``, the shipped curves ``w (R H (Y - Q*) + Q* - psi)`` of the arm means."""
    problem, theta = fit.problem, fit.theta
    targeted = problem.targeted(theta)
    beta = theta[problem.slices["beta"]]
    out = np.empty_like(targeted)
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ beta)
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        out[:, j] = (
            arm.indicator * clever * (problem.outcome - targeted[:, j])
            + targeted[:, j]
            - fit.psi_star[j]
        )
    return out * fit.grid.weight[:, None]


def observed_curve(fit: PopulationFit) -> np.ndarray:
    """``(m,)``, the ``ey_obs`` curve ``w (Y - psi_obs)``."""
    return np.asarray(fit.grid.weight * (fit.grid.outcome - fit.psi_obs))


def sd(fit: PopulationFit, curve: np.ndarray) -> float:
    """The asymptotic standard deviation of ``sqrt(n)`` times an estimate with ``curve``."""
    return float(np.sqrt(fit.grid.mass @ curve**2))


def covariance(fit: PopulationFit, curves: np.ndarray) -> np.ndarray:
    """``(k, k)``, the asymptotic covariance of ``sqrt(n)`` times estimates with ``curves``."""
    return np.asarray((curves * fit.grid.mass[:, None]).T @ curves)


def contrast_curves(fit: PopulationFit, arms: np.ndarray, reference: int) -> dict[str, np.ndarray]:
    """The arm means, the ATEs, the log risk ratios, ``par`` and ``paf`` against ``reference``.

    Parameters
    ----------
    fit : PopulationFit
        The fit.
    arms : ndarray
        ``(m, K)`` arm-mean curves, stacked or shipped.
    reference : int
        The reference arm.

    Returns
    -------
    dict of str to ndarray
        The curves, keyed ``ey[a]``, ``ate[a vs r]``, ``log_rr[a vs r]``, ``par[r]``,
        ``paf[r]``.
    """
    psi = fit.psi_star
    obs = observed_curve(fit)
    out = {f"ey[{a}]": arms[:, a] for a in range(fit.law.k)}
    for a in range(fit.law.k):
        if a == reference:
            continue
        out[f"ate[{a} vs {reference}]"] = arms[:, a] - arms[:, reference]
        out[f"log_rr[{a} vs {reference}]"] = (
            arms[:, a] / psi[a] - arms[:, reference] / psi[reference]
        )
    out[f"par[{reference}]"] = obs - arms[:, reference]
    out[f"paf[{reference}]"] = (
        -arms[:, reference] / fit.psi_obs + psi[reference] * obs / fit.psi_obs**2
    )
    return out


def kappa(fit: PopulationFit) -> np.ndarray:
    """``(K,)``, ``E[v H] / E[R v H^2]`` per arm at the population fit (X58 plan, 2.3)."""
    problem, theta, share = fit.problem, fit.theta, fit.problem.mass
    targeted = problem.targeted(theta)
    beta = theta[problem.slices["beta"]]
    out = []
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ beta)
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        v = targeted[:, j] * (1.0 - targeted[:, j])
        out.append(float((share @ (v * clever)) / (share @ (arm.indicator * v * clever**2))))
    return np.array(out)


def closed_form_curves(fit: PopulationFit, kappas: np.ndarray | None = None) -> np.ndarray:
    """``(m, K)``, ``(q - psi) + kappa R H (Y - q) + (d - kappa e)' IF_beta`` per arm.

    The closed form of the X58 plan, 2.3, with ``K - 1`` arm indicators in ``x_a``. It holds
    when the outcome model is correct, so that ``eps* = 0`` and ``Q*_a = q_a``. The rows are
    the stack's, before the weight factor.

    Parameters
    ----------
    fit : PopulationFit
        The fit.
    kappas : ndarray or None
        ``(K,)``; ``None`` reads :func:`kappa`.

    Returns
    -------
    ndarray
        The closed-form rows.
    """
    problem, theta, share = fit.problem, fit.theta, fit.problem.mass
    beta = theta[problem.slices["beta"]]
    x, y = problem.features, problem.outcome
    fitted = expit(x @ beta)
    information = (x * (share * fitted * (1.0 - fitted))[:, None]).T @ x
    if_beta = np.linalg.solve(information, (x * (y - fitted)[:, None]).T).T
    kappas = kappa(fit) if kappas is None else kappas
    out = np.empty((y.size, len(problem.arms)))
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ beta)
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        v = q * (1.0 - q)
        d = (share * v) @ arm.features
        e = (share * arm.indicator * clever * v) @ arm.features
        out[:, j] = (
            (q - fit.psi_star[j])
            + kappas[j] * arm.indicator * clever * (y - q)
            + if_beta @ (d - kappas[j] * e)
        )
    return out


def critical_value(cov: np.ndarray, *, alpha: float = 0.05, draws: int = 1_000_000) -> float:
    """The ``1 - alpha`` quantile of ``max_j |Z_j| / sd_j`` for ``Z ~ N(0, cov)``.

    Parameters
    ----------
    cov : ndarray
        ``(k, k)`` covariance.
    alpha : float
        The joint level.
    draws : int
        The Gaussian draws, from a fixed seed.

    Returns
    -------
    float
        The critical value ``c*`` of the exact-law band.
    """
    sds = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(sds, sds)
    rng = np.random.default_rng(20261016)
    root = np.linalg.cholesky(correlation + 1e-14 * np.eye(len(sds)))
    stat = np.max(np.abs(rng.standard_normal((draws, len(sds))) @ root.T), axis=1)
    return float(np.quantile(stat, 1.0 - alpha))
