"""The dense continuous-time walk, transcribed from the engine before blocks (test oracle).

The engine is X6 part A's at ``d2b949e5``: its clever covariate's quotient is the stable
conditional risk, and its switched variance reads the predictable variation of the targeted
fit.  The block engine (``cleverly.survival._chunked``) must return what this dense walk returns from
the same initial fit: every unit's ``(2, J, M)`` hazards on the full grid, evaluated, tilted and
accepted at once.  The walk, the evaluation and the off-grid diagnostic below are the engine's
own code before blocks, kept verbatim apart from imports; only :func:`dense_target` is glue that
reads the nuisance fits from the block engine's initial fit.  The tests anchor this module to
numbers the dense engine itself produced, so a change here that drifts from it fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cleverly._typing import FloatArray
from cleverly.survival import _targeting
from cleverly.survival._targeting import (
    MIN_STEP,
    SIGMA_FLOOR,
    Norm,
    Problem,
    point_weights,
)


def curves(dlam: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Survival just after each time and the cause incidences, from hazard jumps.

    Parameters
    ----------
    dlam : ndarray
        ``(2, J, n, M)`` hazard jumps.

    Returns
    -------
    S : ndarray
        ``(2, n, M)`` :math:`S(u_m)`, the product limit.
    F : ndarray
        ``(2, J, n, M)`` :math:`F_j(u_m) = \\sum_{u \\le u_m} S(u^-)\\, d\\Lambda_j(u)`.
    """
    survival = np.cumprod(1.0 - dlam.sum(axis=1), axis=-1)
    incidence = _before(survival)[:, None] * dlam
    np.cumsum(incidence, axis=-1, out=incidence)
    return survival, incidence


def _before(survival: FloatArray) -> FloatArray:
    """``S(u_m^-)``: the survival just before each grid time, from ``S(u_m)``."""
    return np.concatenate([np.ones((*survival.shape[:-1], 1)), survival[..., :-1]], axis=-1)


def _after(x: FloatArray) -> FloatArray:
    r""":math:`\sum_{s > k} x_s` at each grid position :math:`k`, along the last axis."""
    out = np.zeros_like(x)
    out[..., :-1] = np.cumsum(x[..., :0:-1], axis=-1)[..., ::-1]
    return out


def conditional_risk(survival: FloatArray, dlam_cause: FloatArray, tail: FloatArray) -> FloatArray:
    r"""The quotient :math:`H_j(u_k)` of the clever covariate, without cancellation.

    Parameters
    ----------
    survival : ndarray
        ``(..., M)`` :math:`S(u_m)`.
    dlam_cause : ndarray
        ``(..., M)`` :math:`d\Lambda_j(u_m)`, broadcastable against ``survival``.
    tail : ndarray
        ``(..., M)`` :math:`W(u_m) = \sum_{m' \ge m} \omega_{m'}`, broadcastable.

    Returns
    -------
    ndarray
        :math:`\sum_{s > k} W(u_s) S(u_{s-1}) d\Lambda_j(u_s) / S(u_k)`, which equals
        :math:`\{\sum_{m \ge k} \omega_m F_j(u_m) - W(u_k) F_j(u_k)\} / S(u_k)` (module
        docstring).
    """
    return _after(tail * _before(survival) * dlam_cause) / survival


@dataclass(frozen=True)
class Evaluation:
    """The plug-in, the score and the influence curve at one set of hazards.

    Parameters
    ----------
    psi : ndarray
        ``(P,)`` plug-in estimates.
    score : ndarray
        ``(P,)`` :math:`P_n` of the score part of each efficient influence function.
    influence : ndarray
        ``(n, P)`` efficient influence curves.
    survival : ndarray
        ``(2, n, M)`` survival.
    incidence : ndarray
        ``(2, J, n, M)`` cause incidences.
    """

    psi: FloatArray
    score: FloatArray
    influence: FloatArray
    survival: FloatArray
    incidence: FloatArray

    @property
    def sigma(self) -> FloatArray:
        """``(P,)`` :math:`\\hat\\sigma_p = \\sqrt{P_n D_p^2}`."""
        return np.sqrt(np.mean(self.influence**2, axis=0))


def evaluate(problem: Problem, dlam: FloatArray) -> Evaluation:
    """Evaluate the plug-in and the efficient influence curves (design section 4.1).

    Parameters
    ----------
    problem : Problem
        The fixed inputs.
    dlam : ndarray
        ``(2, J, n, M)`` hazard jumps.

    Returns
    -------
    Evaluation
        The plug-in, score and influence curves.
    """
    survival, incidence = curves(dlam)
    J, Q = problem.n_causes, problem.n_rows
    n = problem.n
    rows = problem.target_weights.T  # (M, Q)
    parts = np.zeros((n, 2 * J * Q))
    plugins = np.zeros((n, 2 * J * Q))
    for a in (0, 1):
        mask = (problem.arm == a).astype(float)[:, None]
        # In place, since each (J, n, M) temporary is hundreds of MB at n = 4,000.
        weighted = problem.at_risk[None] * dlam[a]
        np.subtract(problem.events, weighted, out=weighted)
        weighted *= problem.weight[a][None]
        weighted *= mask[None]  # (J, n, M)
        # R(u_s^-) = sum_{k < s} r(u_k), the residual ratio cumulated strictly before u_s.
        ratio = weighted.sum(axis=0) / survival[a]
        ratio_before = np.zeros_like(ratio)
        np.cumsum(ratio[:, :-1], axis=1, out=ratio_before[:, 1:])
        before = _before(survival[a])
        for j in range(J):
            # Each target's score part is sum_k {W(u_k) w (dN_j - Y dLam_j)(u_k) - r(u_k)
            # H_j(u_k)}, which the swap of the module docstring writes as the weighted sum over
            # the grid of the cumulated sum_{s <= u_m} {w (dN_j - Y dLam_j)(u_s) - S(u_s^-)
            # dLam_j(u_s) R(u_s^-)}; a unit row picks one grid position exactly.
            increment = before * dlam[a, j]
            increment *= ratio_before
            np.subtract(weighted[j], increment, out=increment)
            cum_own = np.cumsum(increment, axis=1)
            block = slice((a * J + j) * Q, (a * J + j + 1) * Q)
            parts[:, block] = cum_own @ rows
            plugins[:, block] = incidence[a, j] @ rows
        # Freed before the next arm allocates its own, so two arms' temporaries never coexist.
        del weighted, ratio, ratio_before, before, increment, cum_own
    psi = plugins.mean(axis=0)
    influence = parts + plugins - psi
    return Evaluation(psi, parts.mean(axis=0), influence, survival, incidence)


def _direction(
    problem: Problem, evaluation: Evaluation, v: FloatArray, dlam: FloatArray
) -> FloatArray:
    r"""``(2, J, n, M)`` the combined clever covariate :math:`\sum_p v_p c_p(u, a, L_i)`.

    A target with grid weights :math:`\omega_m` has the clever covariate
    :math:`1\{l = j\} W(u) - \{\sum_{u_m \ge u} \omega_m F_j(u_m) - W(u) F_j(u)\} / S(u)`,
    :math:`W(u) = \sum_{u_m \ge u} \omega_m`, divided by :math:`\pi S_c`.  The quotient is
    :func:`conditional_risk` of the combined weights, a reverse cumulative sum over time, so a
    step costs :math:`O(JnM)` for any weights.  ``dlam`` holds the hazard jumps ``evaluation``
    was computed at.
    """
    J, Q = problem.n_causes, problem.n_rows
    n, M = problem.at_risk.shape
    out = np.zeros((2, J, n, M))
    for a in (0, 1):
        # The weight each grid point carries for cause j: sum_q v_(a,j,q) w_(q,m).
        own = v[a * J * Q : (a + 1) * J * Q].reshape(J, Q) @ problem.target_weights
        own = np.cumsum(own[:, ::-1], axis=1)[:, ::-1]  # W(u)
        survival = evaluation.survival[a]
        shared = np.zeros((n, M))
        for j in range(J):
            shared += conditional_risk(survival, dlam[a, j], own[j][None, :])
        for l in range(J):
            out[a, l] = problem.weight[a] * (own[l][None, :] - shared)
    return out


def tilt_jumps(dlam: FloatArray, increment: FloatArray) -> FloatArray:
    r"""Tilt the cause hazard jumps of every cell along a multinomial logistic path.

    Parameters
    ----------
    dlam : ndarray
        ``(2, J, n, M)`` hazard jumps, each cell's sum below one.
    increment : ndarray
        ``(2, J, n, M)`` step times direction.

    Returns
    -------
    ndarray
        :math:`d\Lambda_l e^{x_l} / (1 - \sum_k d\Lambda_k + \sum_k d\Lambda_k e^{x_k})`.

    Notes
    -----
    In continuous time the jumps are infinitesimal and this is the exponential tilt
    :math:`d\Lambda_l e^{x_l}` of the source's submodel.  On observed jump times it keeps every
    cell's sum below one, so the product limit stays a probability, and its derivative at zero is
    the same score :math:`\sum_l c_l (dN_l - Y\, d\Lambda_l)` under the multinomial
    likelihood of the jumps (design rev 2.1, item 5).
    """
    tilted = np.exp(increment)
    tilted *= dlam
    denominator = tilted.sum(axis=1, keepdims=True)
    denominator += 1.0
    denominator -= dlam.sum(axis=1, keepdims=True)
    tilted /= denominator
    return tilted


def _metric(sigma: FloatArray, norm: Norm) -> FloatArray:
    P = sigma.shape[0]
    if norm == "euclidean":
        return np.eye(P)
    diagonal = np.maximum(np.diag(sigma), SIGMA_FLOOR * max(float(np.max(np.diag(sigma))), 1e-300))
    if norm == "variance":
        return np.diag(1.0 / diagonal)
    held = sigma.copy()
    held[np.diag_indices(P)] = diagonal
    ridge = 1e-10 * float(np.trace(held)) / P
    return np.linalg.inv(held + ridge * np.eye(P))


@dataclass
class WalkResult:
    """The outcome of the targeting walk.

    Parameters
    ----------
    dlam : ndarray
        ``(2, J, n, M)`` targeted hazard jumps.
    evaluation : Evaluation
        The plug-in, score and influence curves at the targeted hazards.
    initial : Evaluation
        The same at the initial hazards, without its survival and incidence curves.
    steps : int
        Accepted steps.
    halvings : int
        Rejected steps.
    converged : bool
        Whether the stopping rule was met.
    cutoff : float
        The bound on :math:`\\max_p |P_n D_p| / \\hat\\sigma_p`.
    final_step : float
        The step size at the end.
    frozen_share : float
        The share of (unit, arm) pairs held for at least one trial step, accepted or not,
        because that step would have taken their survival at the last target time below the
        floor.
    history : list of float
        The stopping statistic after each accepted step, the initial fit's first.
    """

    dlam: FloatArray
    evaluation: Evaluation
    initial: Evaluation
    steps: int
    halvings: int
    converged: bool
    cutoff: float
    final_step: float
    frozen_share: float = 0.0
    history: list[float] = field(default_factory=list)

    @property
    def standardized_score(self) -> FloatArray:
        """``(P,)`` :math:`|P_n D_p| / \\hat\\sigma_p` at the end."""
        return np.abs(self.evaluation.score) / self.evaluation.sigma


def standardized(evaluation: Evaluation) -> float:
    """The largest :math:`|P_n D_p| / \\hat\\sigma_p`.

    Parameters
    ----------
    evaluation : Evaluation
        An evaluation.

    Returns
    -------
    float
        The stopping statistic.
    """
    sigma = evaluation.sigma
    return float(np.max(np.abs(evaluation.score) / sigma))


_EMPTY = np.zeros(0)


def walk(
    problem: Problem,
    dlam: FloatArray,
    *,
    cutoff: float,
    norm: Norm = "covariance",
    step: float = 0.1,
    max_steps: int = 500,
    max_halvings: int | None = None,
    tilt: Literal["all", "factual"] = "all",
) -> WalkResult:
    """Walk the universal least favourable submodel until every score is small.

    Parameters
    ----------
    problem : Problem
        The fixed inputs.
    dlam : ndarray
        ``(2, J, n, M)`` initial hazard jumps.
    cutoff : float
        Stop when :math:`\\max_p |P_n D_p| / \\hat\\sigma_p` is at or below it.
    norm : {"covariance", "variance", "euclidean"}
        The inner product, with the covariance held at the initial fit.
    step : float
        The first step size; halved on each rejected step and doubled, up to its first
        value, after each accepted one.
    max_steps : int
        The cap on accepted steps.
    max_halvings : int or None
        The cap on rejected steps, ``2 * max_steps`` by default.
    tilt : {"all", "factual"}
        ``"all"`` tilts each arm's hazard for every unit.  ``"factual"`` tilts only units
        whose received arm is the target's arm, as ``concrete`` 1.0.5 does; it exists for the
        mutation control that shows that form is not doubly robust.

    Returns
    -------
    WalkResult
        The targeted hazards and the walk's record.
    """
    halving_cap = 2 * max_steps if max_halvings is None else max_halvings
    initial_step = step
    # Never written in place: each accepted step replaces it, so no copy is held.
    current = np.asarray(dlam, dtype=float)
    del dlam
    initial = evaluate(problem, current)
    metric = _metric(
        np.cov(initial.influence, rowvar=False, bias=True).reshape(
            problem.n_targets, problem.n_targets
        ),
        norm,
    )
    # A trial is rejected below the floor, or below the initial fit's own smallest survival
    # when that is lower, so an accepted initial fit is never rejected as a starting point.
    survival_floor = min(_targeting.WALK_SURVIVAL_FLOOR, float(np.min(initial.survival[..., -1])))
    light = Evaluation(initial.psi, initial.score, initial.influence, _EMPTY, _EMPTY)
    evaluation = initial
    del initial
    # Pairs held on the step being tried, and pairs held on any step, for the record.
    frozen = np.zeros((2, problem.n), dtype=bool)
    ever = np.zeros((2, problem.n), dtype=bool)
    steps = halvings = 0
    history = [standardized(evaluation)]
    converged = history[-1] <= cutoff
    if tilt == "factual":
        factual = np.stack([(problem.arm == a).astype(float) for a in (0, 1)])[:, None, :, None]
    while not converged and steps < max_steps and halvings < halving_cap and step >= MIN_STEP:
        x = evaluation.score
        weighted = metric @ x
        length = float(np.sqrt(x @ weighted))
        if length == 0.0:
            converged = True
            break
        direction = _direction(problem, evaluation, weighted / length, current)
        if tilt == "factual":
            direction = direction * factual
        if frozen.any():
            direction *= ~frozen[:, None, :, None]
        direction *= step
        candidate = tilt_jumps(current, direction)
        del direction
        # A trial step can take a survival to zero, where the score divides 0 by 0; the check
        # below rejects such a trial, so its warning carries nothing.
        with np.errstate(divide="ignore", invalid="ignore"):
            trial = evaluate(problem, candidate)
        low = trial.survival[..., -1] < survival_floor
        if low.any():
            del candidate, trial
            new = low & ~frozen
            if new.any():
                # Pairs the step takes below the floor are held for this step and the same step
                # is tried again; they move again on the next step, whose direction is
                # recomputed.
                frozen |= new
                ever |= new
                continue
            # Only held pairs read below the floor (a tilt of zero can round a held pair's
            # survival down by an ulp), so holding more cannot help: halve as for a rejection.
            step /= 2
            halvings += 1
            frozen[:] = False
            continue
        trial_length = float(np.sqrt(trial.score @ metric @ trial.score))
        if not np.isfinite(trial_length) or trial_length >= length:
            # Freed now rather than on the next trial, which would hold two of each.
            del candidate, trial
            step /= 2
            halvings += 1
            frozen[:] = False
            continue
        current, evaluation = candidate, trial
        frozen[:] = False
        step = min(2.0 * step, initial_step)
        steps += 1
        history.append(standardized(evaluation))
        converged = history[-1] <= cutoff
    return WalkResult(
        current,
        evaluation,
        light,
        steps,
        halvings,
        converged,
        cutoff,
        step,
        float(ever.mean()),
        history,
    )


OFF_GRID_TIMES = 100


@dataclass
class OffGrid:
    """The dense off-grid diagnostic's share and the vectors it reads."""

    share: float
    score: FloatArray | None = None
    psi: FloatArray | None = None
    sigma: FloatArray | None = None


def off_grid_share(
    problem: Problem,
    result: WalkResult,
    grid: FloatArray,
    rng: np.random.Generator,
    cutoff: float,
) -> OffGrid:
    """Item 10 Table 4: the share of random times at which the score equation holds."""
    draws = np.sort(rng.uniform(0.0, float(grid[-1]), OFF_GRID_TIMES))
    index = np.searchsorted(grid, draws, side="right") - 1
    keep = index >= 0
    if not keep.any():
        return OffGrid(1.0)
    shifted = Problem(
        problem.arm,
        problem.at_risk,
        problem.events,
        problem.weight,
        point_weights(index[keep], grid.size),
    )
    evaluation = evaluate(shifted, result.dlam)
    sigma = evaluation.sigma
    informative = sigma > 0
    if not informative.any():
        return OffGrid(1.0, evaluation.score, evaluation.psi, sigma)
    ok = np.abs(evaluation.score[informative]) <= cutoff * sigma[informative]
    return OffGrid(float(np.mean(ok)), evaluation.score, evaluation.psi, sigma)


@dataclass
class Reference:
    """What the dense engine's ``_target`` read from its walk, before building estimates."""

    result: WalkResult
    off_grid: OffGrid
    bounded_share: float
    last_time_floor_share: float
    predictable: FloatArray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def dense_target(
    initial: Any,
    floor: float,
    *,
    cutoff: float,
    norm: Norm = "covariance",
    step: float = 0.1,
    max_steps: int = 500,
    tilt: Literal["all", "factual"] = "all",
    adjust_weight: Callable[[FloatArray], FloatArray] | None = None,
) -> Reference:
    """Run the dense walk and diagnostic from a block engine's initial fit.

    The initial hazards, censoring retention and propensity are read on the full grid from
    the fit's hazard sources, and the bounded product and indicators are built as the dense
    engine built them.

    Parameters
    ----------
    initial : cleverly.survival.engine._Initial
        The block engine's initial fit.
    floor : float
        The weight floor.
    cutoff : float
        The walk's cutoff.
    norm : {"covariance", "variance", "euclidean"}
        The walk's inner product.
    step : float
        The first step size.
    max_steps : int
        The cap on accepted steps.
    tilt : {"all", "factual"}
        The tilt.
    adjust_weight : callable or None
        Changes the ``(2, n, M)`` weights before the walk, as a test patches the engine's.

    Returns
    -------
    Reference
        The walk and its diagnostics.
    """
    n, A, T, D, grid = initial.n, initial.A, initial.T, initial.D, initial.grid
    dlam = initial.hazards.jumps(0, n)
    dgam = initial.hazards.censoring_jumps(0, n)
    retention = np.cumprod(1.0 - np.clip(dgam, 0.0, 1.0), axis=-1)
    retention_before = np.concatenate([np.ones((2, n, 1)), retention[..., :-1]], axis=-1)
    pi = np.stack([1.0 - initial.p1, initial.p1])
    product = pi[:, :, None] * retention_before
    weight = 1.0 / np.maximum(product, floor)
    if adjust_weight is not None:
        weight = adjust_weight(weight)
    factual = A[None, :, None] == np.arange(2)[:, None, None]
    reach = T[None, :, None] >= grid[None, None, :]
    relevant = factual & reach
    bounded_share = float(np.mean(product[relevant] < floor)) if relevant.any() else 0.0
    last_time_floor_share = float(np.mean(product[..., -1] < floor))
    at_risk = T[:, None] >= grid[None, :]
    events = np.stack(
        [(T[:, None] == grid[None, :]) & (D[:, None] == code) for code in initial.causes]
    )
    problem = Problem(A, at_risk, events, weight, initial.target_weights)
    result = walk(
        problem, dlam, cutoff=cutoff, norm=norm, step=step, max_steps=max_steps, tilt=tilt
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = initial.rng_state
    off_grid = off_grid_share(problem, result, grid, rng, cutoff)
    # The switched variance's predictable variation, as the dense engine's _target formed it.
    floored = product / floor
    np.minimum(floored, 1.0, out=floored)
    predictable = _targeting.predictable_covariance(
        problem, result.evaluation, result.dlam, floored
    )
    return Reference(result, off_grid, bounded_share, last_time_floor_share, predictable)
