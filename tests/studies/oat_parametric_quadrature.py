"""Exact-law quadrature for the per-arm OAT stack with parametric nuisances.

Each tensor Gauss-Hermite node of ``W = (W1, W2, W3)``, three independent standard normals,
is expanded into the four ``(A, Y)`` cells with mass
``g_0(A | W) Qbar_0(A, W)^Y (1 - Qbar_0(A, W))^(1 - Y)``. A population M-estimator is then a
weighted fit on that grid. The production learners fit with the masses as sample weights,
and :class:`cleverly.estimators._oat_stack.StackProblem` takes them as its measure, so the
grid's Jacobian ``A`` and its curve rows are the production code's own at the population
fit. The asymptotic covariance of ``sqrt(n)`` times the arm means is ``E[IF IF']``.

A fit whose outcome model is wrong returns the pseudo-true ``psi*`` (X58 plan, 2.8). The
fluctuation coefficient is solved on the grid by a one-dimensional Newton step on the
clipped ``Q*``. The outer re-shrink binds only at far tail nodes; :class:`PopulationFit`
records their mass.

Convergence in the node count is pinned per law by ``tests/unit/test_oat_stack.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logit

from cleverly.estimators._oat_stack import ArmDesign, StackProblem, corrected_curves
from cleverly.learners import ParametricMechanism, ParametricOutcome
from cleverly.learners.parametric import OutcomeFeatures
from tests.studies import oat_per_arm_laws as laws

ALPHA = 0.9995
"""cleverly's default shrinkage bound."""

Mechanism = Callable[[np.ndarray], np.ndarray]
Outcome = Callable[[int, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class QuadratureLaw:
    """A binary-treatment law on three standard-normal covariates.

    Parameters
    ----------
    name : str
        The law's name.
    treated : callable
        ``W -> g_0(1 | W)``.
    outcome : callable
        ``(a, W) -> Qbar_0(a, W)``.
    """

    name: str
    treated: Mechanism = field(compare=False)
    outcome: Outcome = field(compare=False)


def _binary_active_treated(w: np.ndarray) -> np.ndarray:
    return np.asarray(laws.BINARY_ACTIVE.mechanism(w[:, 0], w[:, 2])[:, 1])


def _binary_active_outcome(a: int, w: np.ndarray) -> np.ndarray:
    return laws.BINARY_ACTIVE.outcome_probability(a, w[:, 0], w[:, 1])


BINARY_ACTIVE = QuadratureLaw("binary_active", _binary_active_treated, _binary_active_outcome)
"""The primary law of ``ctmle-oat-per-arm``."""

U_SHAPED = QuadratureLaw(
    "u_shaped",
    lambda w: np.asarray(expit(-1.2 + 1.2 * w[:, 0] ** 2 + 0.5 * w[:, 2])),
    lambda a, w: np.asarray(expit(-0.2 + 0.4 * a + (0.9 + 0.3 * a) * w[:, 0] + 0.4 * w[:, 1])),
)
"""A mechanism U-shaped in ``W1``: the cubic-logit mechanism misses it, so ``kappa != 1``."""

STEEP = QuadratureLaw(
    "steep",
    lambda w: np.asarray(expit(2.5 * w[:, 0] + 0.8 * w[:, 2])),
    lambda a, w: np.asarray(expit(-0.2 + 0.5 * a + 1.5 * w[:, 0] + 0.5 * w[:, 1])),
)
"""A steep mechanism: ``g_bounds = (0.05, 0.95)`` clips about 12% of each arm's rows."""

SHARED_PROGNOSIS = QuadratureLaw(
    "shared_prognosis",
    _binary_active_treated,
    lambda a, w: np.asarray(expit((0.3 if a else -0.3) + 3.5 * w[:, 0] + 0.8 * w[:, 1])),
)
"""``binary_active``'s mechanism with a strong shared prognosis: the arm curves correlate."""


@dataclass(frozen=True)
class Grid:
    """The expanded quadrature grid.

    Parameters
    ----------
    covariates : ndarray
        ``(m, 3)``, ``W`` of each cell.
    treatment : ndarray
        ``(m,)``, ``A`` of each cell.
    outcome : ndarray
        ``(m,)``, ``Y`` of each cell.
    mass : ndarray
        ``(m,)``, the probability of each cell; the masses sum to 1.
    """

    covariates: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    mass: np.ndarray


@lru_cache(maxsize=16)
def _nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = hermegauss(nodes)
    weights = weights / weights.sum()
    mesh = np.meshgrid(points, points, points, indexing="ij")
    covariates = np.column_stack([axis.ravel() for axis in mesh])
    mass = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).ravel()
    return covariates, mass


def grid(law: QuadratureLaw, nodes: int = 30) -> Grid:
    """The law's cells at a tensor Gauss-Hermite rule of ``nodes`` points per axis.

    Parameters
    ----------
    law : QuadratureLaw
        The law.
    nodes : int
        The points per covariate axis.

    Returns
    -------
    Grid
        Four cells per node, ``(A, Y)`` in ``{0, 1}^2``.
    """
    covariates, mass = _nodes(nodes)
    treated = law.treated(covariates)
    parts = []
    for arm in (0, 1):
        g = treated if arm else 1.0 - treated
        q = law.outcome(arm, covariates)
        for y in (0, 1):
            cell = mass * g * (q if y else 1.0 - q)
            parts.append(
                (covariates, np.full(mass.size, float(arm)), np.full(mass.size, float(y)), cell)
            )
    total = np.concatenate([part[3] for part in parts])
    return Grid(
        covariates=np.concatenate([part[0] for part in parts]),
        treatment=np.concatenate([part[1] for part in parts]),
        outcome=np.concatenate([part[2] for part in parts]),
        mass=total / total.sum(),
    )


@dataclass(frozen=True)
class PopulationFit:
    """The population fit of the per-arm OAT estimator on a grid.

    Parameters
    ----------
    grid : Grid
        The cells.
    problem : StackProblem
        The stack on the grid's measure.
    theta : ndarray
        The population ``theta*``.
    psi_star : ndarray
        ``(2,)``, the limit of the arm means.
    psi_0 : ndarray
        ``(2,)``, the true arm means ``E[Qbar_0(a, W)]``.
    epsilons : ndarray
        ``(2,)``, ``eps*`` of each arm.
    clipped : ndarray
        ``(2,)``, the mass on which ``g_bounds`` binds, per arm.
    outer_binding : ndarray
        ``(2,)``, the mass on which the outer re-shrink of ``Q*`` binds, per arm.
    """

    grid: Grid
    problem: StackProblem
    theta: np.ndarray
    psi_star: np.ndarray
    psi_0: np.ndarray
    epsilons: np.ndarray
    clipped: np.ndarray
    outer_binding: np.ndarray


def _solve_fluctuation(
    indicator: np.ndarray,
    outcome: np.ndarray,
    level: np.ndarray,
    clever: np.ndarray,
    mass: np.ndarray,
    alpha: float,
) -> float:
    eps = 0.0
    for _ in range(200):
        raw = expit(level + eps * clever)
        inside = (raw > 1.0 - alpha) & (raw < alpha)
        targeted = np.clip(raw, 1.0 - alpha, alpha)
        score = np.sum(mass * indicator * clever * (outcome - targeted))
        information = np.sum(mass * indicator * inside * clever**2 * targeted * (1.0 - targeted))
        step = score / information
        eps += step
        if abs(step) < 1e-15:
            return float(eps)
    raise AssertionError("the population fluctuation did not converge")


def population_fit(
    law: QuadratureLaw,
    *,
    nodes: int = 30,
    features: OutcomeFeatures = "arm_interactions",
    covariates: Sequence[int] = (0, 1, 2),
    degree: int = 3,
    bounds: tuple[float, float] = laws.G_BOUNDS,
    alpha: float = ALPHA,
) -> PopulationFit:
    """The population per-arm OAT fit and its stack on the law's grid.

    Parameters
    ----------
    law : QuadratureLaw
        The law.
    nodes : int
        The points per covariate axis.
    features : str
        The outcome feature map of :class:`~cleverly.learners.ParametricOutcome`.
    covariates : sequence of int
        The columns of ``W`` the outcome design reads. Dropping ``W1`` gives a wrong
        outcome model.
    degree : int
        The degree of the mechanism basis in ``logit q``.
    bounds : tuple of float
        ``g_bounds``.
    alpha : float
        The shrinkage bound.

    Returns
    -------
    PopulationFit
        The fit, its stack and its limits.
    """
    cells = grid(law, nodes)
    w = cells.covariates[:, list(covariates)]
    mass = cells.mass
    outcome_model = ParametricOutcome(features).fit(
        np.column_stack([cells.treatment, w]), cells.outcome, sample_weight=mass
    )
    beta = outcome_model.coef_
    mechanism_template = ParametricMechanism(degree=degree)
    designs: list[ArmDesign] = []
    gammas: list[np.ndarray] = []
    epsilons: list[float] = []
    psis: list[float] = []
    clipped: list[float] = []
    binding: list[float] = []
    edge = 1.0 - alpha
    for arm in (0, 1):
        x_arm = outcome_model.design_matrix(np.column_stack([np.full(mass.size, float(arm)), w]))
        q = expit(x_arm @ beta)
        indicator = (cells.treatment == arm).astype(float)
        mechanism = ParametricMechanism(degree=degree).fit(
            q[:, None], indicator, sample_weight=mass
        )
        g = mechanism.predict(q[:, None])
        clever = 1.0 / np.clip(g, *bounds)
        level = logit(np.clip(q, edge, alpha))
        eps = _solve_fluctuation(indicator, cells.outcome, level, clever, mass, alpha)
        raw = expit(level + eps * clever)
        binding.append(float(mass @ ((raw <= edge) | (raw >= alpha))))
        targeted = np.clip(raw, edge, alpha)
        designs.append(
            ArmDesign(
                features=x_arm,
                indicator=indicator,
                basis=mechanism_template.design_matrix,
                basis_derivative=mechanism_template.derivative,
                bounds=bounds,
            )
        )
        gammas.append(mechanism.coef_)
        epsilons.append(eps)
        psis.append(float(mass @ targeted))
        clipped.append(float(mass @ ((g <= bounds[0]) | (g >= bounds[1]))))
    problem = StackProblem(
        outcome_model.design_matrix(np.column_stack([cells.treatment, w])),
        cells.outcome,
        designs,
        alpha,
        mass=mass,
    )
    theta = problem.pack(beta, gammas, epsilons, psis)
    truth = [float(mass @ law.outcome(arm, cells.covariates)) for arm in (0, 1)]
    return PopulationFit(
        grid=cells,
        problem=problem,
        theta=theta,
        psi_star=np.array(psis),
        psi_0=np.array(truth),
        epsilons=np.array(epsilons),
        clipped=np.array(clipped),
        outer_binding=np.array(binding),
    )


def arm_curves(fit: PopulationFit) -> np.ndarray:
    """``(m, 2)``, the corrected curve rows of the arm means on the grid."""
    return corrected_curves(fit.problem, fit.theta).curves


def covariance(fit: PopulationFit) -> np.ndarray:
    """``(2, 2)``, the asymptotic covariance of ``sqrt(n) (psi_0, psi_1)``."""
    curves = arm_curves(fit)
    return np.asarray((curves * fit.grid.mass[:, None]).T @ curves)


def ate_sd(fit: PopulationFit) -> float:
    """The asymptotic standard deviation of ``sqrt(n)`` times the ATE."""
    cov = covariance(fit)
    return float(np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]))


def shipped_curves(fit: PopulationFit) -> np.ndarray:
    """``(m, 2)``, the shipped adaptive-propensity curve ``R H (Y - Q*) + Q* - psi``."""
    problem, theta = fit.problem, fit.theta
    targeted = problem.targeted(theta)
    out = np.empty_like(targeted)
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ theta[problem.slices["beta"]])
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        out[:, j] = (
            arm.indicator * clever * (problem.outcome - targeted[:, j])
            + targeted[:, j]
            - fit.psi_star[j]
        )
    return out


def kappa(fit: PopulationFit) -> np.ndarray:
    """``(2,)``, ``E[v H] / E[R v H^2]`` per arm at the population fit (X58 plan, 2.3)."""
    problem, theta, mass = fit.problem, fit.theta, fit.grid.mass
    targeted = problem.targeted(theta)
    out = []
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ theta[problem.slices["beta"]])
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        v = targeted[:, j] * (1.0 - targeted[:, j])
        out.append(float((mass @ (v * clever)) / (mass @ (arm.indicator * v * clever**2))))
    return np.array(out)


def closed_form_curves(fit: PopulationFit) -> np.ndarray:
    """``(m, 2)``, ``(q - psi) + kappa R H (Y - q) + (d - kappa e)' IF_beta`` per arm.

    The closed form of the X58 plan, 2.3, which holds when the outcome model is correct, so
    that ``eps* = 0`` and ``Q*_a = q_a``.
    """
    problem, theta, mass = fit.problem, fit.theta, fit.grid.mass
    beta = theta[problem.slices["beta"]]
    x, y = problem.features, problem.outcome
    fitted = expit(x @ beta)
    information = (x * (mass * fitted * (1.0 - fitted))[:, None]).T @ x
    if_beta = np.linalg.solve(information, (x * (y - fitted)[:, None]).T).T
    kappas = kappa(fit)
    out = np.empty((y.size, len(problem.arms)))
    for j, arm in enumerate(problem.arms):
        gamma = theta[problem.slices[f"gamma[{j}]"]]
        q = expit(arm.features @ beta)
        clever = 1.0 / np.clip(expit(arm.basis(q) @ gamma), *arm.bounds)
        v = q * (1.0 - q)
        d = (mass * v) @ arm.features
        e = (mass * arm.indicator * clever * v) @ arm.features
        out[:, j] = (
            (q - fit.psi_star[j])
            + kappas[j] * arm.indicator * clever * (y - q)
            + if_beta @ (d - kappas[j] * e)
        )
    return out


def fixed_outcome_curves(fit: PopulationFit) -> np.ndarray:
    """``(m, 2)``, the arm-mean rows of ``-A^{-1} U`` with the ``beta`` block removed.

    The stack of a pinned outcome model: ``beta`` is a constant, so its rows and columns
    leave the system.
    """
    problem = fit.problem
    stack = problem.blocks(fit.theta)
    keep = np.ones(problem.dimension, dtype=bool)
    keep[problem.slices["beta"]] = False
    jac = stack.jacobian[np.ix_(keep, keep)]
    solved = -np.linalg.solve(jac, stack.rows[:, keep].T).T
    return solved[:, -len(problem.arms) :]


def sample(
    law: QuadratureLaw, n: int, seed: int, *, bounded: tuple[float, float] | None = None
) -> pd.DataFrame:
    """``n`` rows of the law: ``W1``, ``W2``, ``W3``, ``A`` and ``Y``.

    Parameters
    ----------
    law : QuadratureLaw
        The law.
    n : int
        The rows.
    seed : int
        The random seed.
    bounded : tuple of float, optional
        ``(lower, upper)``: draw ``Y = lower + (upper - lower) B`` with
        ``B ~ Beta(4 Qbar_0, 4 (1 - Qbar_0))``, so the scaled outcome has mean ``Qbar_0``.
        ``None`` draws a binary ``Y``.

    Returns
    -------
    DataFrame
        The sample.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    a = rng.binomial(1, law.treated(w))
    q = np.where(a == 1, law.outcome(1, w), law.outcome(0, w))
    if bounded is None:
        y = rng.binomial(1, q).astype(float)
    else:
        lower, upper = bounded
        y = lower + (upper - lower) * rng.beta(4.0 * q, 4.0 * (1.0 - q))
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2], "A": a, "Y": y})
