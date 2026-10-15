"""The stacked estimating equation of the per-arm outcome-adaptive C-TMLE.

With a parametric outcome GLM and a parametric mechanism per arm, the fitted per-arm OAT
estimator solves one square system ``P_n U(theta) = 0``. In sample the parameter is

``theta = (beta, gamma_1, eps_1, ..., gamma_K, eps_K, psi_1, ..., psi_K)``.

For one row the estimating functions are

* outcome: ``x (Y - expit(x' beta))``;
* mechanism of arm ``a``: ``phi(q_a) (R_a - G_a)``, with ``q_a = expit(x_a' beta)``,
  ``R_a = 1{A = a}`` and ``G_a = expit(phi(q_a)' gamma_a)``;
* fluctuation of arm ``a``: ``R_a H_a (Y - Q*_a)``, with ``H_a = 1 / clip(G_a, lo, hi)``
  and ``Q*_a = clip(expit(logit(clip(q_a, 1 - alpha, alpha)) + eps_a H_a), 1 - alpha,
  alpha)``;
* arm mean: ``Q*_a - psi_a``.

A fit cross-fitted over ``V`` folds solves the exact ``V``-fold system (X58 plan, 13.3):

``theta = (beta_1, ..., beta_V, gamma_{1,1..V}, eps_1, ..., gamma_{K,1..V}, eps_K, psi_1..K)``.

Fold ``k``'s outcome and mechanism equations read the training rows ``v(i) != k`` only, and
``gamma_{a,k}`` reads ``beta_k``. The fluctuation and mean equations read every row at the
fits of the row's own held-out fold ``v(i)``, with one pooled ``eps_a`` per arm. The in-sample
system is the ``V = 1`` case, with training and held-out rows both the whole sample.

The influence curve of the estimate is ``-A_n^{-1} U_i(theta_n)`` with the analytic Jacobian
``A_n = P_n dU / dtheta'`` (Stefanski and Boos 2002; Saul and Hudgens 2020, eq. 2). The
Jacobian is block lower triangular in the order above, so the curve is a forward
substitution through the diagonal blocks. Each clip enters the Jacobian as the indicator of
its unclipped side.

The module reads arrays and coefficients only. It does not fit anything. The theory, the
conditions and the witnesses are in the X58 plan (plans repository, ``X58-plan.md``,
Sections 2, 4 and 13).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit, logit

from .._typing import FloatArray, IntArray

__all__ = [
    "MAX_TARGETED_GAP",
    "MIN_BLOCK_RCOND",
    "ArmDesign",
    "FittedStack",
    "FoldCoefficients",
    "FoldStackProblem",
    "StackBlocks",
    "StackCurves",
    "StackProblem",
    "StackReport",
    "corrected_curves",
    "fitted_stack",
    "stack_from_fit",
    "stack_report",
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
        ``(n, k)``, ``U_i(theta)`` in the parameter order of :class:`FoldStackProblem`.
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
        block's name (``"beta"``, ``"gamma[j]"``, ``"eps[j]"`` in sample; ``"beta[k]"``,
        ``"gamma[j,k]"``, ``"eps[j]"`` over ``V`` folds).
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
    """One arm's fitted quantities at one fold's fits, and their derivatives."""

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


class FoldStackProblem:
    """The per-arm OAT system of one sample, in sample or cross-fitted over ``V`` folds.

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
    folds : ndarray of int, optional
        ``(n,)``, the held-out fold ``v(i)`` of each row, in ``0..V-1``, each fold
        nonempty. ``None`` is the in-sample fit, ``V = 1``, whose training and held-out
        rows are both the whole sample.
    mass : ndarray, optional
        ``(n,)``, nonnegative row masses of the measure that replaces ``P_n``. ``None`` is
        the empirical measure. An exact-law grid passes its quadrature masses. The curves
        stay ``-A^{-1} U_i``, unweighted by the mass.

    Raises
    ------
    ValueError
        If ``folds`` does not label every row with a fold in ``0..V-1``, or leaves a fold
        empty.
    """

    def __init__(
        self,
        features: FloatArray,
        outcome: FloatArray,
        arms: Sequence[ArmDesign],
        alpha: float,
        folds: IntArray | None = None,
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
        if folds is None:
            self.labels = np.zeros(self.n, dtype=int)
            self.n_folds = 1
            every = np.ones((1, self.n))
            self.training, self.held = every, every
        else:
            labels = np.asarray(folds).reshape(-1)
            if labels.shape != (self.n,) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("folds must hold one integer fold label per row")
            n_folds = int(labels.max()) + 1 if labels.size else 0
            if labels.min() < 0 or np.unique(labels).size != n_folds:
                raise ValueError(
                    f"folds must label each row with a fold in 0..{n_folds - 1}, every "
                    "fold nonempty"
                )
            self.labels = labels.astype(int)
            self.n_folds = n_folds
            held = np.vstack([(self.labels == k).astype(float) for k in range(n_folds)])
            # One fold holds out every row and trains on every row: the in-sample system.
            self.held = held
            self.training = held if n_folds == 1 else 1.0 - held
        self.r = tuple(int(arm.basis(np.full(1, 0.5)).shape[1]) for arm in self.arms)
        sizes = [(self.beta_name(k), self.p) for k in range(self.n_folds)]
        for j, width in enumerate(self.r):
            sizes += [(self.gamma_name(j, k), width) for k in range(self.n_folds)]
            sizes.append((f"eps[{j}]", 1))
        sizes += [(f"psi[{j}]", 1) for j in range(len(self.arms))]
        offsets = np.cumsum([0] + [size for _, size in sizes])
        self.slices = {
            name: slice(int(offsets[i]), int(offsets[i + 1])) for i, (name, _) in enumerate(sizes)
        }
        self.dimension = int(offsets[-1])

    def beta_name(self, k: int) -> str:
        """The name of fold ``k``'s outcome block.

        Parameters
        ----------
        k : int
            The fold.

        Returns
        -------
        str
            ``"beta"`` in sample, ``"beta[k]"`` over ``V`` folds.
        """
        return "beta" if self.n_folds == 1 else f"beta[{k}]"

    def gamma_name(self, j: int, k: int) -> str:
        """The name of fold ``k``'s mechanism block of arm ``j``.

        Parameters
        ----------
        j : int
            The arm's position.
        k : int
            The fold.

        Returns
        -------
        str
            ``"gamma[j]"`` in sample, ``"gamma[j,k]"`` over ``V`` folds.
        """
        return f"gamma[{j}]" if self.n_folds == 1 else f"gamma[{j},{k}]"

    def arm_blocks(self, j: int) -> tuple[str, ...]:
        """The diagonal blocks whose solve arm ``j``'s mean reads.

        Parameters
        ----------
        j : int
            The arm's position.

        Returns
        -------
        tuple of str
            Every fold's outcome block, every fold's mechanism block of the arm, and the
            arm's fluctuation block.
        """
        return (
            *(self.beta_name(k) for k in range(self.n_folds)),
            *(self.gamma_name(j, k) for k in range(self.n_folds)),
            f"eps[{j}]",
        )

    def pack(
        self,
        betas: Sequence[FloatArray],
        gammas: Sequence[Sequence[FloatArray]],
        epsilons: Sequence[float],
        psis: Sequence[float],
    ) -> FloatArray:
        """``theta`` from its parts.

        Parameters
        ----------
        betas : sequence of ndarray
            Each fold's outcome coefficients; one entry in sample.
        gammas : sequence of sequence of ndarray
            Per arm, each fold's mechanism coefficients.
        epsilons : sequence of float
            Each arm's fluctuation coefficient.
        psis : sequence of float
            Each arm's mean.

        Returns
        -------
        ndarray
            ``(k,)``, the stacked parameter.

        Raises
        ------
        ValueError
            If a fold's coefficients are missing.
        """
        if len(betas) != self.n_folds or any(len(arm) != self.n_folds for arm in gammas):
            raise ValueError(
                f"the stack has {self.n_folds} folds; got {len(betas)} outcome and "
                f"{[len(arm) for arm in gammas]} mechanism coefficient sets"
            )
        theta = np.empty(self.dimension)
        for k in range(self.n_folds):
            theta[self.slices[self.beta_name(k)]] = betas[k]
        for j in range(len(self.arms)):
            for k in range(self.n_folds):
                theta[self.slices[self.gamma_name(j, k)]] = gammas[j][k]
            theta[self.slices[f"eps[{j}]"]] = epsilons[j]
            theta[self.slices[f"psi[{j}]"]] = psis[j]
        return theta

    def _fold_masses(self, k: int) -> tuple[FloatArray, FloatArray]:
        """Fold ``k``'s training and held-out masks times the row masses, for the Jacobian."""
        return self.mass * self.training[k], self.mass * self.held[k]

    def _mean(self, values: FloatArray) -> FloatArray:
        """The mass-weighted mean over rows."""
        return np.asarray(self.mass @ values)

    def _arm_state(self, theta: FloatArray, j: int, k: int = 0) -> _ArmState:
        arm = self.arms[j]
        beta = theta[self.slices[self.beta_name(k)]]
        gamma = theta[self.slices[self.gamma_name(j, k)]]
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

    def _own(self, states: Sequence[_ArmState], field: str) -> FloatArray:
        """One field of each row's own held-out fold state."""
        return self._assemble([np.asarray(getattr(state, field)) for state in states])

    def _assemble(self, per_fold: Sequence[FloatArray]) -> FloatArray:
        """Per row, the value of the row's own held-out fold, from one array per fold."""
        if self.n_folds == 1:
            return per_fold[0]
        out = np.zeros(self.n)
        for k, values in enumerate(per_fold):
            out = out + self.held[k] * values
        return out

    def targeted(self, theta: FloatArray) -> FloatArray:
        """``Q*_a`` of every arm at ``theta``, each row at its own held-out fold's fits.

        Parameters
        ----------
        theta : ndarray
            The stacked parameter.

        Returns
        -------
        ndarray
            ``(n, K)``, the targeted predictions.
        """
        return np.column_stack(
            [
                self._own([self._arm_state(theta, j, k) for k in range(self.n_folds)], "targeted")
                for j in range(len(self.arms))
            ]
        )

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
        for k in range(self.n_folds):
            ib = self.slices[self.beta_name(k)]
            fitted = expit(x @ theta[ib])
            rows[:, ib] = x * (self.training[k] * (y - fitted))[:, None]
            if jacobian:
                trained, _ = self._fold_masses(k)
                jac[ib, ib] = -(x * (trained * fitted * (1.0 - fitted))[:, None]).T @ x
        for j, arm in enumerate(self.arms):
            states = [self._arm_state(theta, j, k) for k in range(self.n_folds)]
            ie = self.slices[f"eps[{j}]"]
            ip = self.slices[f"psi[{j}]"]
            eps = float(theta[ie][0])
            indicator = arm.indicator
            targeted = self._own(states, "targeted")
            residual = y - targeted
            rows[:, ie] = (indicator * self._own(states, "clever") * residual)[:, None]
            rows[:, ip] = (targeted - theta[ip][0])[:, None]
            xa = arm.features
            for k, state in enumerate(states):
                ig = self.slices[self.gamma_name(j, k)]
                ib = self.slices[self.beta_name(k)]
                g = state.mechanism
                rows[:, ig] = state.phi * (self.training[k] * (indicator - g))[:, None]
                if not jacobian:
                    continue
                gv = g * (1.0 - g)
                # q = expit(x_a' beta_k), so d logit(q) / d beta_k = x_a.
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
                # Fold k's mechanism equation reads its training rows; the fluctuation and
                # mean equations read fold k's fits on the rows fold k holds out.
                trained, own = self._fold_masses(k)
                jac[ig, ig] = -(state.phi * (trained * gv)[:, None]).T @ state.phi
                jac[ig, ib] = np.einsum(
                    "nrp,n->rp", dphi_db, trained * (indicator - g)
                ) - state.phi.T @ (trained[:, None] * dg_db)
                weight = indicator * (y - state.targeted)
                carrier = indicator * state.clever
                jac[ie, ie] -= own @ (carrier * dq_deps)
                jac[ie, ig] = own @ (weight[:, None] * dh_dgamma - carrier[:, None] * dq_dgamma)
                jac[ie, ib] = own @ (weight[:, None] * dh_db - carrier[:, None] * dq_db)
                jac[ip, ie] += own @ dq_deps
                jac[ip, ig] = own @ dq_dgamma
                jac[ip, ib] = own @ dq_db
            if jacobian:
                jac[ip, ip] = -1.0
        return StackBlocks(rows=rows, jacobian=jac)

    def clip_summary(
        self, theta: FloatArray
    ) -> tuple[tuple[int, ...], dict[str, tuple[float, ...]]]:
        """Per arm, the rows where the outer re-shrink binds and each clip's share.

        Each row reads its own held-out fold's fits.

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
            states = [self._arm_state(theta, j, k) for k in range(self.n_folds)]
            outer.append(int(np.sum(self._own(states, "inside_outer") == 0.0)))
            inside_bounds = self._own(states, "inside_bounds")
            inside_alpha = self._own(states, "inside_alpha")
            shares["g_bounds"].append(float(self._mean((inside_bounds == 0.0).astype(float))))
            shares["alpha"].append(float(self._mean((inside_alpha == 0.0).astype(float))))
            basis_inside = self._assemble(
                [np.any(state.dphi[:, 1:] != 0.0, axis=1).astype(float) for state in states]
            )
            shares["basis"].append(float(self._mean((basis_inside == 0.0).astype(float))))
        return tuple(outer), {name: tuple(values) for name, values in shares.items()}


class StackProblem(FoldStackProblem):
    """The in-sample per-arm OAT system: :class:`FoldStackProblem` with ``V = 1``.

    Parameters
    ----------
    features : ndarray
        ``(n, p)``, the outcome features ``x`` at the observed treatment.
    outcome : ndarray
        ``(n,)``, the outcome on the fitted ``[0, 1]`` scale.
    arms : sequence of ArmDesign
        One design per arm, in the order of the arm means.
    alpha : float
        The shrinkage bound of the fluctuation.
    mass : ndarray, optional
        ``(n,)``, nonnegative row masses of the measure that replaces ``P_n``. ``None`` is
        the empirical measure.
    """

    def __init__(
        self,
        features: FloatArray,
        outcome: FloatArray,
        arms: Sequence[ArmDesign],
        alpha: float,
        mass: FloatArray | None = None,
    ) -> None:
        super().__init__(features, outcome, arms, alpha, folds=None, mass=mass)


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


def corrected_curves(problem: FoldStackProblem, theta: FloatArray) -> StackCurves:
    """The arm-mean rows of ``-A_n^{-1} U_i(theta)``, by forward substitution.

    Parameters
    ----------
    problem : FoldStackProblem
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
    # The slices are laid out in the solve order, so every earlier block is a prefix.
    for name, block in problem.slices.items():
        rhs = -rows[:, block] - solved[:, : block.start] @ jac[block, : block.start].T
        diagonal = jac[block, block]
        if not name.startswith("psi"):
            rconds[name] = _equilibrated_rcond(diagonal)
        if rconds.get(name, 1.0) == 0.0:
            # Zeros, not NaN: NaN times a structurally zero cross-arm entry of ``A_n`` is NaN,
            # which would reach the arm whose stack held. The arm that reads this block fails.
            solved[:, block] = 0.0
        else:
            solved[:, block] = np.linalg.solve(diagonal, rhs.T).T
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


#: The reciprocal condition number below which a diagonal block of ``A_n`` is singular.
MIN_BLOCK_RCOND = 1e-12

#: The largest gap between ``Q*_a`` recomputed from ``eps_a`` and the fit's targeted
#: predictions before the fluctuation counts as path dependent (X58 plan, 11).
MAX_TARGETED_GAP = 1e-10


@dataclass(frozen=True)
class FoldCoefficients:
    """The fitted coefficients of each fold's outcome model and each arm's mechanisms.

    The per-arm fit writes this record into ``nuisance.diagnostics["stack"]``, keyed by
    fold index. An in-sample fit holds one fold. Construction refuses a missing fold rather
    than shifting the later folds into its place.

    Parameters
    ----------
    outcome : tuple of tuple of float
        ``beta_k`` of each fold ``k``, in fold order.
    mechanism : tuple of tuple of tuple of float
        Per arm, ``gamma_{a,k}`` of each fold ``k``, in fold order.
    folds : int
        ``V``, the number of folds.

    Raises
    ------
    ValueError
        If a fold has no outcome or mechanism coefficients.
    """

    outcome: tuple[tuple[float, ...], ...]
    mechanism: tuple[tuple[tuple[float, ...], ...], ...]
    folds: int

    def __post_init__(self) -> None:
        """Refuse a record that does not hold one entry per fold."""
        counts = [len(self.outcome), *(len(arm) for arm in self.mechanism)]
        if any(count != self.folds for count in counts):
            raise ValueError(
                f"FoldCoefficients needs one entry per fold for {self.folds} folds; got "
                f"{len(self.outcome)} outcome entries and "
                f"{[len(arm) for arm in self.mechanism]} mechanism entries per arm"
            )

    @classmethod
    def from_entries(
        cls,
        outcome: Mapping[int, Any],
        mechanism: Mapping[int, Sequence[Any]],
        *,
        folds: int,
        arms: int,
    ) -> FoldCoefficients:
        """The record from each fold's learner diagnostics, keyed by fold index.

        Parameters
        ----------
        outcome : mapping of int to diagnostics
            Fold ``k``'s outcome learner diagnostics, with a ``coef`` attribute.
        mechanism : mapping of int to sequence of diagnostics
            Fold ``k``'s mechanism diagnostics, one per arm in arm order.
        folds : int
            ``V``.
        arms : int
            ``K``.

        Returns
        -------
        FoldCoefficients
            The coefficients in fold order.

        Raises
        ------
        ValueError
            If a fold lacks an outcome entry, or an arm of a fold lacks its mechanism
            entry. A positional record would shift every later fold into its place.
        """
        betas: list[tuple[float, ...]] = []
        gammas: list[list[tuple[float, ...]]] = [[] for _ in range(arms)]
        for k in range(folds):
            entry = outcome.get(k)
            if entry is None:
                raise ValueError(
                    f"fold {k} of {folds} has no outcome coefficients; the stack reads one "
                    "ParametricOutcome fit per fold"
                )
            betas.append(tuple(float(value) for value in entry.coef))
            per_arm = mechanism.get(k)
            if per_arm is None or len(per_arm) != arms or any(e is None for e in per_arm):
                raise ValueError(
                    f"fold {k} of {folds} lacks a mechanism fit for some arm; the stack reads "
                    f"one ParametricMechanism fit per arm and fold, {arms} arms"
                )
            for j, arm_entry in enumerate(per_arm):
                gammas[j].append(tuple(float(value) for value in arm_entry.coef))
        return cls(
            outcome=tuple(betas),
            mechanism=tuple(tuple(arm) for arm in gammas),
            folds=folds,
        )


@dataclass(frozen=True)
class FittedStack:
    """The stack of one per-arm OAT fit, read from the fit's parts.

    Parameters
    ----------
    problem : FoldStackProblem
        The system on the empirical measure.
    theta : ndarray
        The fitted ``theta_n``.
    scale : float
        The width of the outcome scaler: curves on the fitted scale times ``scale`` are on
        the outcome scale.
    targeted_mismatch : float
        The largest absolute difference between ``Q*_a`` recomputed from ``theta_n`` and the
        targeted predictions the fit reports, over every arm.
    arm_mismatch : tuple of float
        The same gap per arm.
    converged : bool
        Whether the fluctuation met its convergence test.
    shipped : ndarray or None
        ``(n, K)``, the fit's own uncorrected curve per arm on the fitted scale, before the
        data weights: ``h_a (Y - Q*) + Q*_a - psi_a`` with the fit's clever covariate. It is
        read from the fit's arrays, so it equals the curve
        :func:`~cleverly.inference.influence.counterfactual_means` subtracts the correction
        from. ``None`` on a stack built without a fit.
    """

    problem: FoldStackProblem
    theta: FloatArray
    scale: float
    targeted_mismatch: float
    arm_mismatch: tuple[float, ...] = ()
    converged: bool = True
    shipped: FloatArray | None = None


@dataclass(frozen=True)
class StackReport:
    """The corrected curves of one fit, as the estimate builder reads them.

    Parameters
    ----------
    corrections : dict of float to ndarray
        Per arm code, the shipped curve minus the corrected curve on the fitted scale. The
        estimate builder subtracts it from the shipped curve. A failed arm holds zeros.
    failures : dict of float to str
        Per failed arm code, the condition that failed.
    shipped_std_error : dict of float to float
        Per arm code, the standard error of the shipped curve on the fitted scale.
    curves : StackCurves or None
        The solve's record; ``None`` when the stack could not be read.
    """

    corrections: dict[float, FloatArray]
    failures: dict[float, str]
    shipped_std_error: dict[float, float]
    curves: StackCurves | None


def stack_from_fit(
    outcome_learner: Any,
    treatment_learner: Any,
    data: Any,
    nuisance: Any,
    fluctuation: Any,
    clever: FloatArray,
    bounds: tuple[float, float],
    alpha: float,
) -> FittedStack:
    """Read the stack of a per-arm OAT fit from its parts. Nothing is refitted.

    The coefficients come from the :class:`FoldCoefficients` record in
    ``nuisance.diagnostics["stack"]``, the fold labels from ``nuisance.folds``, and the
    fluctuation coefficients and targeted predictions from the ``"mean"`` fluctuation. The
    shipped curve reads the fit's clever covariate and targeted predictions, so on a
    cross-fitted fit each row reads its own fold's mechanism.

    Parameters
    ----------
    outcome_learner : ParametricOutcome
        The outcome learner template; its feature map builds ``x``.
    treatment_learner : ParametricMechanism
        The treatment learner template; its basis builds ``phi``.
    data : CausalData
        The prepared data.
    nuisance : NuisanceEstimates
        The fit's nuisances, with the fold coefficient record.
    fluctuation : Fluctuation
        The ``"mean"`` fluctuation.
    clever : ndarray
        ``(n, K)``, the fit's clever covariate of each arm, ``R_a / G~_a`` from the bounded
        mechanism the targeting read.
    bounds : tuple of float
        The resolved ``g_bounds``.
    alpha : float
        The shrinkage bound.

    Returns
    -------
    FittedStack
        The system, ``theta_n``, the targeted-prediction check and the shipped curves.

    Raises
    ------
    KeyError
        If the nuisances carry no fold coefficient record.
    ValueError
        If the record's fold count differs from the fit's folds.
    """
    arms = tuple(float(arm) for arm in data.arm_codes)
    record = nuisance.diagnostics["stack"]
    if not isinstance(record, FoldCoefficients):
        raise TypeError(f"the stack record is a {type(record).__name__}, not FoldCoefficients")
    folds = nuisance.folds
    if record.folds != folds.n_folds:
        raise ValueError(
            f"the coefficient record holds {record.folds} folds and the fit {folds.n_folds}"
        )
    epsilons = [float(value) for value in fluctuation.epsilon]
    targeted = np.column_stack([fluctuation.targeted.arms[arm] for arm in arms])
    designs = [
        ArmDesign(
            features=outcome_learner.design_matrix(data.counterfactual_design(arm)),
            indicator=(np.asarray(data.treatment) == arm).astype(float),
            basis=treatment_learner.design_matrix,
            basis_derivative=treatment_learner.derivative,
            bounds=(float(bounds[0]), float(bounds[1])),
        )
        for arm in arms
    ]
    scaled = nuisance.scaler.scale(data.outcome)
    problem = FoldStackProblem(
        outcome_learner.design_matrix(data.treatment_design()),
        scaled,
        designs,
        float(alpha),
        folds=None if folds.is_single else np.asarray(folds.assignment),
    )
    theta = problem.pack(
        [np.asarray(beta, dtype=float) for beta in record.outcome],
        [[np.asarray(gamma, dtype=float) for gamma in arm] for arm in record.mechanism],
        epsilons,
        list(np.mean(targeted, axis=0)),
    )
    gaps = np.max(np.abs(problem.targeted(theta) - targeted), axis=0)
    # The bracket of ``counterfactual_means``, summed in its association, so the shipped
    # curve here equals the one the estimate builder subtracts the correction from.
    residual = np.asarray(scaled, dtype=float) - fluctuation.targeted.observed
    weights = np.asarray(data.weights, dtype=float)
    shipped = np.column_stack(
        [
            np.asarray(clever, dtype=float)[:, j] * residual
            + targeted[:, j]
            - float(np.average(targeted[:, j], weights=weights))
            for j in range(len(arms))
        ]
    )
    scaler = nuisance.scaler
    return FittedStack(
        problem=problem,
        theta=theta,
        scale=float(scaler.upper - scaler.lower),
        targeted_mismatch=float(np.max(gaps)),
        arm_mismatch=tuple(float(gap) for gap in gaps),
        converged=bool(fluctuation.converged),
        shipped=shipped,
    )


def stack_report(stack: FittedStack, arms: Sequence[float]) -> StackReport:
    """The corrections the estimate builder subtracts, and the arms whose stack failed.

    An arm fails when the fluctuation did not converge, when a fold's outcome block or the
    arm's mechanism or fluctuation block is numerically singular
    (:data:`MIN_BLOCK_RCOND`), or when the arm's recomputed ``Q*`` differs from the fit's
    (:data:`MAX_TARGETED_GAP`). A failed arm keeps the shipped curve.

    Parameters
    ----------
    stack : FittedStack
        The fit's stack, with its shipped curves.
    arms : sequence of float
        The arm codes, in the stack's order.

    Returns
    -------
    StackReport
        The corrections, the failures, the shipped standard errors and the solve's record.

    Raises
    ------
    ValueError
        If the stack carries no shipped curves.
    """
    if stack.shipped is None:
        raise ValueError("stack_report reads the fit's shipped curves; the stack holds none")
    problem, theta = stack.problem, stack.theta
    solved = corrected_curves(problem, theta)
    rconds = solved.reciprocal_conditions
    corrections: dict[float, FloatArray] = {}
    failures: dict[float, str] = {}
    shipped_se: dict[float, float] = {}
    n = problem.n
    for j, arm in enumerate(arms):
        shipped = stack.shipped[:, j]
        shipped_se[arm] = float(np.std(shipped, ddof=1) / np.sqrt(n))
        reason = _arm_failure(stack, rconds, j)
        if reason is None:
            corrections[arm] = shipped - solved.curves[:, j]
        else:
            failures[arm] = reason
            corrections[arm] = np.zeros(n)
    return StackReport(
        corrections=corrections,
        failures=failures,
        shipped_std_error=shipped_se,
        curves=solved,
    )


def _arm_failure(stack: FittedStack, rconds: dict[str, float], j: int) -> str | None:
    if not stack.converged:
        return "the fluctuation did not meet its convergence test"
    for name in stack.problem.arm_blocks(j):
        value = rconds[name]
        if not value >= MIN_BLOCK_RCOND:
            return (
                f"the {name} block of the stacked Jacobian has reciprocal condition number "
                f"{value:.3g}, below {MIN_BLOCK_RCOND:g}"
            )
    gap = stack.arm_mismatch[j] if stack.arm_mismatch else stack.targeted_mismatch
    if gap > MAX_TARGETED_GAP:
        return (
            f"the targeted predictions differ by {gap:.3g} from those recomputed from the "
            f"fluctuation coefficient, above {MAX_TARGETED_GAP:g}: the solver path met the "
            "outer re-shrink, or a retarget moved g_bounds away from the declared bounds"
        )
    return None


def fitted_stack(result: Any, draw: int | None = None) -> FittedStack:
    """Read the stack of a fitted per-arm OAT estimate, in sample or cross-fitted.

    Parameters
    ----------
    result : Result
        A ``CTMLE(strategy="oat", oat_design="per_arm")`` result with
        :class:`~cleverly.learners.ParametricOutcome` and
        :class:`~cleverly.learners.ParametricMechanism` learners.
    draw : int, optional
        The cross-fitting repeat to read, ``result.repeats[draw]``. ``None`` reads the
        first.

    Returns
    -------
    FittedStack
        The system, ``theta_n``, the outcome scale, the targeted-prediction check and the
        shipped curves.

    Raises
    ------
    ValueError
        If the result is not a per-arm fit with the parametric learners.
    """
    from ..learners.parametric import ParametricMechanism, ParametricOutcome
    from .targeting import build_submodel

    estimator = result.estimator
    outcome_learner = estimator.outcome_learner
    treatment_learner = estimator.treatment_learner
    if type(outcome_learner) is not ParametricOutcome or type(treatment_learner) is not (
        ParametricMechanism
    ):
        raise ValueError("the stack needs ParametricOutcome and ParametricMechanism learners")
    repeat = result.repeats[0 if draw is None else draw]
    bounds = tuple(float(value) for value in result.config.g_bounds)
    submodel = build_submodel(
        result.data,
        repeat.nuisance,
        "mean",
        bounds=(bounds[0], bounds[1]),
        nuisance_bound=float(estimator.nuisance_bound),
    )
    arms = tuple(float(arm) for arm in result.data.arm_codes)
    return stack_from_fit(
        outcome_learner,
        treatment_learner,
        result.data,
        repeat.nuisance,
        repeat.fluctuations["mean"],
        np.column_stack([submodel.column_for(arm) for arm in arms]),
        (bounds[0], bounds[1]),
        float(estimator.alpha),
    )
