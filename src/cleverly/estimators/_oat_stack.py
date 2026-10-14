"""The stacked estimating equation of the per-arm outcome-adaptive C-TMLE.

With a parametric outcome GLM and a parametric mechanism per arm, the fitted per-arm OAT
estimator solves one square system ``P_n U(theta) = 0`` in

``theta = (beta, gamma_1, eps_1, ..., gamma_K, eps_K, psi_1, ..., psi_K)``.

For one row the estimating functions are

* outcome: ``x (Y - expit(x' beta))``;
* mechanism of arm ``a``: ``phi(q_a) (R_a - G_a)``, with ``q_a = expit(x_a' beta)``,
  ``R_a = 1{A = a}`` and ``G_a = expit(phi(q_a)' gamma_a)``;
* fluctuation of arm ``a``: ``R_a H_a (Y - Q*_a)``, with ``H_a = 1 / clip(G_a, lo, hi)``
  and ``Q*_a = clip(expit(logit(clip(q_a, 1 - alpha, alpha)) + eps_a H_a), 1 - alpha,
  alpha)``;
* arm mean: ``Q*_a - psi_a``.

The influence curve of the estimate is ``-A_n^{-1} U_i(theta_n)`` with the analytic Jacobian
``A_n = P_n dU / dtheta'`` (Stefanski and Boos 2002; Saul and Hudgens 2020, eq. 2). The
Jacobian is block lower triangular in the order above, so the curve is a forward
substitution through the diagonal blocks. Each clip enters the Jacobian as the indicator of
its unclipped side.

The module reads arrays and coefficients only. It does not fit anything. The theory, the
conditions and the witnesses are in the X58 plan (plans repository, ``X58-plan.md``,
Sections 2 and 4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit, logit

from .._typing import FloatArray

__all__ = [
    "ArmDesign",
    "FittedStack",
    "StackBlocks",
    "StackCurves",
    "StackProblem",
    "corrected_curves",
    "fitted_stack",
]

Basis = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class ArmDesign:
    """The fixed inputs of one arm's mechanism, fluctuation and mean.

    Parameters
    ----------
    features : ndarray
        ``(n, p)``, the outcome features ``x_a`` with the treatment set to the arm.
    indicator : ndarray
        ``(n,)``, ``R_a = 1{A = a}``.
    basis : callable
        ``q -> phi(q)``, ``(n, r)``.
    basis_derivative : callable
        ``q -> d phi / d logit(q)``, ``(n, r)``, zero where the basis clip binds.
    bounds : tuple of float
        ``(lo, hi)``, the bounds on the mechanism.
    """

    features: FloatArray
    indicator: FloatArray
    basis: Basis
    basis_derivative: Basis
    bounds: tuple[float, float]


@dataclass(frozen=True)
class StackBlocks:
    """The row estimating functions and the sample Jacobian at one ``theta``.

    Parameters
    ----------
    rows : ndarray
        ``(n, k)``, ``U_i(theta)`` in the parameter order of :class:`StackProblem`.
    jacobian : ndarray
        ``(k, k)``, ``P_n dU / dtheta'``.
    """

    rows: FloatArray
    jacobian: FloatArray


@dataclass(frozen=True)
class StackCurves:
    """The corrected curves of the arm means and what the solve recorded.

    Parameters
    ----------
    curves : ndarray
        ``(n, K)``, the ``psi`` rows of ``-A_n^{-1} U_i``, on the scale of the fitted
        outcome.
    reciprocal_conditions : dict
        The equilibrated reciprocal condition number of each diagonal block, keyed by the
        block's name (``"beta"``, ``"gamma[j]"``, ``"eps[j]"``).
    outer_binding : tuple of int
        Per arm, the rows on which the outer re-shrink of ``Q*`` binds.
    clipped_share : dict
        Per clip (``"g_bounds"``, ``"alpha"``, ``"basis"``), the share of rows clipped in
        each arm.
    """

    curves: FloatArray
    reciprocal_conditions: dict[str, float]
    outer_binding: tuple[int, ...]
    clipped_share: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class _ArmState:
    """One arm's fitted quantities at a ``theta``, and their derivatives."""

    q: FloatArray
    phi: FloatArray
    dphi: FloatArray
    mechanism: FloatArray
    inside_bounds: FloatArray
    clever: FloatArray
    inside_alpha: FloatArray
    level: FloatArray
    targeted: FloatArray
    inside_outer: FloatArray


class StackProblem:
    """The per-arm OAT system for one sample.

    Parameters
    ----------
    features : ndarray
        ``(n, p)``, the outcome features ``x`` at the observed treatment.
    outcome : ndarray
        ``(n,)``, the outcome on the fitted ``[0, 1]`` scale.
    arms : sequence of ArmDesign
        One design per arm, in the order of the arm means.
    alpha : float
        The shrinkage bound of the fluctuation: predictions are kept in
        ``[1 - alpha, alpha]``.
    mass : ndarray, optional
        ``(n,)``, nonnegative row masses of the measure that replaces ``P_n``. ``None`` is
        the empirical measure. An exact-law grid passes its quadrature masses. The curves
        stay ``-A^{-1} U_i``, unweighted by the mass.
    """

    def __init__(
        self,
        features: FloatArray,
        outcome: FloatArray,
        arms: Sequence[ArmDesign],
        alpha: float,
        mass: FloatArray | None = None,
    ) -> None:
        self.features = np.asarray(features, dtype=float)
        self.outcome = np.asarray(outcome, dtype=float).reshape(-1)
        self.arms = tuple(arms)
        self.alpha = float(alpha)
        self.n, self.p = self.features.shape
        if mass is None:
            self.mass = np.full(self.n, 1.0 / self.n)
        else:
            weights = np.asarray(mass, dtype=float).reshape(-1)
            self.mass = weights / np.sum(weights)
        self.r = tuple(int(arm.basis(np.full(1, 0.5)).shape[1]) for arm in self.arms)
        sizes = [("beta", self.p)]
        for j, width in enumerate(self.r):
            sizes += [(f"gamma[{j}]", width), (f"eps[{j}]", 1)]
        sizes += [(f"psi[{j}]", 1) for j in range(len(self.arms))]
        offsets = np.cumsum([0] + [size for _, size in sizes])
        self.slices = {
            name: slice(int(offsets[i]), int(offsets[i + 1])) for i, (name, _) in enumerate(sizes)
        }
        self.dimension = int(offsets[-1])

    def pack(
        self,
        beta: FloatArray,
        gammas: Sequence[FloatArray],
        epsilons: Sequence[float],
        psis: Sequence[float],
    ) -> FloatArray:
        """``theta`` from its parts.

        Parameters
        ----------
        beta : ndarray
            The outcome coefficients.
        gammas : sequence of ndarray
            Each arm's mechanism coefficients.
        epsilons : sequence of float
            Each arm's fluctuation coefficient.
        psis : sequence of float
            Each arm's mean.

        Returns
        -------
        ndarray
            ``(k,)``, the stacked parameter.
        """
        theta = np.empty(self.dimension)
        theta[self.slices["beta"]] = beta
        for j in range(len(self.arms)):
            theta[self.slices[f"gamma[{j}]"]] = gammas[j]
            theta[self.slices[f"eps[{j}]"]] = epsilons[j]
            theta[self.slices[f"psi[{j}]"]] = psis[j]
        return theta

    def _mean(self, values: FloatArray) -> FloatArray:
        """The mass-weighted mean over rows."""
        return np.asarray(self.mass @ values)

    def _arm_state(self, theta: FloatArray, j: int) -> _ArmState:
        arm = self.arms[j]
        beta = theta[self.slices["beta"]]
        gamma = theta[self.slices[f"gamma[{j}]"]]
        eps = float(theta[self.slices[f"eps[{j}]"]][0])
        q = expit(arm.features @ beta)
        phi = np.asarray(arm.basis(q), dtype=float)
        dphi = np.asarray(arm.basis_derivative(q), dtype=float)
        mechanism = expit(phi @ gamma)
        lo, hi = arm.bounds
        inside_bounds = ((mechanism > lo) & (mechanism < hi)).astype(float)
        clever = 1.0 / np.clip(mechanism, lo, hi)
        edge = 1.0 - self.alpha
        inside_alpha = ((q > edge) & (q < self.alpha)).astype(float)
        level = logit(np.clip(q, edge, self.alpha))
        raw = expit(level + eps * clever)
        inside_outer = ((raw > edge) & (raw < self.alpha)).astype(float)
        targeted = np.clip(raw, edge, self.alpha)
        return _ArmState(
            q=q,
            phi=phi,
            dphi=dphi,
            mechanism=mechanism,
            inside_bounds=inside_bounds,
            clever=clever,
            inside_alpha=inside_alpha,
            level=level,
            targeted=targeted,
            inside_outer=inside_outer,
        )

    def targeted(self, theta: FloatArray) -> FloatArray:
        """``Q*_a`` of every arm at ``theta``.

        Parameters
        ----------
        theta : ndarray
            The stacked parameter.

        Returns
        -------
        ndarray
            ``(n, K)``, the targeted predictions.
        """
        return np.column_stack([self._arm_state(theta, j).targeted for j in range(len(self.arms))])

    def rows(self, theta: FloatArray) -> FloatArray:
        """``U_i(theta)`` for every row.

        Parameters
        ----------
        theta : ndarray
            The stacked parameter.

        Returns
        -------
        ndarray
            ``(n, k)``, the estimating functions.
        """
        return self.blocks(theta, jacobian=False).rows

    def blocks(self, theta: FloatArray, *, jacobian: bool = True) -> StackBlocks:
        """The estimating functions and, optionally, the analytic sample Jacobian.

        Parameters
        ----------
        theta : ndarray
            The stacked parameter.
        jacobian : bool
            Whether to build the Jacobian. Without it the Jacobian is an empty matrix.

        Returns
        -------
        StackBlocks
            ``U_i(theta)`` and ``P_n dU / dtheta'``.
        """
        n, x, y = self.n, self.features, self.outcome
        theta = np.asarray(theta, dtype=float)
        rows = np.zeros((n, self.dimension))
        jac = np.zeros((self.dimension, self.dimension)) if jacobian else np.zeros((0, 0))
        ib = self.slices["beta"]
        beta = theta[ib]
        fitted = expit(x @ beta)
        rows[:, ib] = x * (y - fitted)[:, None]
        if jacobian:
            jac[ib, ib] = -(x * (self.mass * fitted * (1.0 - fitted))[:, None]).T @ x
        for j, arm in enumerate(self.arms):
            state = self._arm_state(theta, j)
            ig = self.slices[f"gamma[{j}]"]
            ie = self.slices[f"eps[{j}]"]
            ip = self.slices[f"psi[{j}]"]
            eps = float(theta[ie][0])
            indicator = arm.indicator
            residual = y - state.targeted
            rows[:, ig] = state.phi * (indicator - state.mechanism)[:, None]
            rows[:, ie] = (indicator * state.clever * residual)[:, None]
            rows[:, ip] = (state.targeted - theta[ip][0])[:, None]
            if not jacobian:
                continue
            xa = arm.features
            g = state.mechanism
            gv = g * (1.0 - g)
            # q = expit(x_a' beta), so d logit(q) / d beta = x_a.
            dphi_db = state.dphi[:, :, None] * xa[:, None, :]
            dg_dgamma = gv[:, None] * state.phi
            dg_db = (gv * (state.dphi @ theta[ig]))[:, None] * xa
            # H = 1 / clip(G, lo, hi): derivative -1 / G~^2 inside the bounds, 0 outside.
            dh = -state.inside_bounds * state.clever**2
            dh_dgamma = dh[:, None] * dg_dgamma
            dh_db = dh[:, None] * dg_db
            # Q* = clip(expit(level + eps H)): its derivative carries the outer indicator.
            slope = state.targeted * (1.0 - state.targeted) * state.inside_outer
            dlevel_db = state.inside_alpha[:, None] * xa
            dq_db = slope[:, None] * (dlevel_db + eps * dh_db)
            dq_dgamma = (slope * eps)[:, None] * dh_dgamma
            dq_deps = slope * state.clever
            jac[ig, ig] = -(state.phi * (self.mass * gv)[:, None]).T @ state.phi
            jac[ig, ib] = np.einsum(
                "nrp,n->rp", dphi_db, self.mass * (indicator - g)
            ) - state.phi.T @ (self.mass[:, None] * dg_db)
            weight = indicator * residual
            carrier = indicator * state.clever
            jac[ie, ie] = -self._mean(carrier * dq_deps)
            jac[ie, ig] = self._mean(weight[:, None] * dh_dgamma - carrier[:, None] * dq_dgamma)
            jac[ie, ib] = self._mean(weight[:, None] * dh_db - carrier[:, None] * dq_db)
            jac[ip, ip] = -1.0
            jac[ip, ie] = self._mean(dq_deps)
            jac[ip, ig] = self._mean(dq_dgamma)
            jac[ip, ib] = self._mean(dq_db)
        return StackBlocks(rows=rows, jacobian=jac)

    def clip_summary(
        self, theta: FloatArray
    ) -> tuple[tuple[int, ...], dict[str, tuple[float, ...]]]:
        """Per arm, the rows where the outer re-shrink binds and each clip's share.

        Parameters
        ----------
        theta : ndarray
            The stacked parameter.

        Returns
        -------
        tuple
            The outer-binding row counts, and the clipped share of each clip per arm.
        """
        outer: list[int] = []
        shares: dict[str, list[float]] = {"g_bounds": [], "alpha": [], "basis": []}
        for j in range(len(self.arms)):
            state = self._arm_state(theta, j)
            outer.append(int(np.sum(state.inside_outer == 0.0)))
            shares["g_bounds"].append(float(self._mean((state.inside_bounds == 0.0).astype(float))))
            shares["alpha"].append(float(self._mean((state.inside_alpha == 0.0).astype(float))))
            basis_inside = np.any(state.dphi[:, 1:] != 0.0, axis=1)
            shares["basis"].append(float(self._mean((~basis_inside).astype(float))))
        return tuple(outer), {name: tuple(values) for name, values in shares.items()}


def _equilibrated_rcond(block: FloatArray) -> float:
    """The reciprocal condition number of ``block`` after symmetric diagonal scaling."""
    if not np.all(np.isfinite(block)):
        return 0.0
    diagonal = np.abs(np.diag(block))
    if np.any(diagonal <= 0.0):
        return 0.0
    scale = 1.0 / np.sqrt(diagonal)
    scaled = block * scale[:, None] * scale[None, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[0] <= 0.0:
        return 0.0
    return float(singular[-1] / singular[0])


def corrected_curves(problem: StackProblem, theta: FloatArray) -> StackCurves:
    """The arm-mean rows of ``-A_n^{-1} U_i(theta)``, by forward substitution.

    Parameters
    ----------
    problem : StackProblem
        The system of one sample.
    theta : ndarray
        The fitted stacked parameter.

    Returns
    -------
    StackCurves
        The curves on the fitted ``[0, 1]`` scale, each diagonal block's reciprocal condition
        number, and the clip counts.
    """
    stack = problem.blocks(theta)
    jac, rows = stack.jacobian, stack.rows
    solved = np.zeros_like(rows)
    rconds: dict[str, float] = {}
    order = ["beta"]
    for j in range(len(problem.arms)):
        order += [f"gamma[{j}]", f"eps[{j}]"]
    order += [f"psi[{j}]" for j in range(len(problem.arms))]
    done: list[slice] = []
    for name in order:
        block = problem.slices[name]
        rhs = -rows[:, block]
        for earlier in done:
            rhs = rhs - solved[:, earlier] @ jac[block, earlier].T
        diagonal = jac[block, block]
        if not name.startswith("psi"):
            rconds[name] = _equilibrated_rcond(diagonal)
        if rconds.get(name, 1.0) == 0.0:
            solved[:, block] = np.nan
        else:
            solved[:, block] = np.linalg.solve(diagonal, rhs.T).T
        done.append(block)
    curves = np.column_stack(
        [solved[:, problem.slices[f"psi[{j}]"]][:, 0] for j in range(len(problem.arms))]
    )
    outer, shares = problem.clip_summary(theta)
    return StackCurves(
        curves=curves,
        reciprocal_conditions=rconds,
        outer_binding=outer,
        clipped_share=shares,
    )


@dataclass(frozen=True)
class FittedStack:
    """The stack of one in-sample per-arm OAT fit, read from its result.

    Parameters
    ----------
    problem : StackProblem
        The system on the empirical measure.
    theta : ndarray
        The fitted ``theta_n``.
    scale : float
        The width of the outcome scaler: curves on the fitted scale times ``scale`` are on
        the outcome scale.
    targeted_mismatch : float
        The largest absolute difference between ``Q*_a`` recomputed from ``theta_n`` and the
        targeted predictions the fit reports. It is at rounding level when the fluctuation's
        solver path never met the outer re-shrink.
    """

    problem: StackProblem
    theta: FloatArray
    scale: float
    targeted_mismatch: float


def fitted_stack(result: Any) -> FittedStack:
    """Read the stack of a fitted in-sample per-arm OAT estimate.

    The outcome and treatment learners must be
    :class:`~cleverly.learners.ParametricOutcome` and
    :class:`~cleverly.learners.ParametricMechanism`. The coefficients come from their
    ``diagnostics_`` entries, the fluctuation coefficients and targeted predictions from the
    ``"mean"`` fluctuation. Nothing is refitted.

    Parameters
    ----------
    result : Result
        A ``CTMLE(strategy="oat", oat_design="per_arm", cross_fit=False)`` result.

    Returns
    -------
    FittedStack
        The system, ``theta_n``, the outcome scale and the targeted-prediction check.

    Raises
    ------
    ValueError
        If the result is not an in-sample per-arm fit with the parametric learners.
    """
    from ..learners.parametric import ParametricMechanism, ParametricOutcome

    estimator = result.estimator
    outcome_learner = estimator.outcome_learner
    treatment_learner = estimator.treatment_learner
    if type(outcome_learner) is not ParametricOutcome or type(treatment_learner) is not (
        ParametricMechanism
    ):
        raise ValueError("the stack needs ParametricOutcome and ParametricMechanism learners")
    nuisance = result.nuisance
    if not nuisance.folds.is_single:
        raise ValueError("the stack reads an in-sample fit; this fit is cross-fitted")
    data = result.data
    arms = tuple(float(arm) for arm in data.arm_codes)
    diagnostics = nuisance.diagnostics
    (outcome_entry,) = diagnostics["outcome"]
    beta = np.asarray(outcome_entry.coef, dtype=float)
    gammas = []
    for entries in diagnostics["propensity"]:
        (entry,) = entries
        gammas.append(np.asarray(entry.coef, dtype=float))
    fluctuation = result.fluctuations["mean"]
    epsilons = [float(value) for value in fluctuation.epsilon]
    targeted = np.column_stack([fluctuation.targeted.arms[arm] for arm in arms])
    bounds = tuple(float(value) for value in result.config.g_bounds)
    designs = [
        ArmDesign(
            features=outcome_learner.design_matrix(data.counterfactual_design(arm)),
            indicator=(np.asarray(data.treatment) == arm).astype(float),
            basis=treatment_learner.design_matrix,
            basis_derivative=treatment_learner.derivative,
            bounds=(bounds[0], bounds[1]),
        )
        for arm in arms
    ]
    problem = StackProblem(
        outcome_learner.design_matrix(data.treatment_design()),
        nuisance.scaler.scale(data.outcome),
        designs,
        float(estimator.alpha),
    )
    theta = problem.pack(beta, gammas, epsilons, list(np.mean(targeted, axis=0)))
    mismatch = float(np.max(np.abs(problem.targeted(theta) - targeted)))
    scaler = nuisance.scaler
    return FittedStack(
        problem=problem,
        theta=theta,
        scale=float(scaler.upper - scaler.lower),
        targeted_mismatch=mismatch,
    )
