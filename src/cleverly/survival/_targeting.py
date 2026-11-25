r"""The one-step TMLE of cause-specific absolute risks in continuous time.

Rytgaard, H. C. W. and van der Laan, M. J. (2024).  One-step targeted maximum likelihood
estimation for targeting cause-specific absolute risks and survival curves.  *Biometrika*
111(1), 129-145 (read as arXiv:2107.01537v2).

Every array is indexed on the sorted distinct observed times :math:`u_1 < \dots < u_M` up to the
last time a target reads.  Hazards are jumps there, held for both arms of every unit,
``dlam[a, l, i, m]`` :math:`= d\Lambda_l(u_m \mid a, L_i)`, so the update tilts the hazard
*function* of each arm for every unit (design section 4.2).  The cause incidences are the
Aalen-Johansen sums :math:`F_j(t) = \sum_{u \le t} S(u^-)\, d\Lambda_j(u)`, with :math:`S` the
product limit, so the incidences and the survival add to one at every step.

Each target is a weighted sum of one arm's cause-:math:`j` incidence over the grid,
:math:`\sum_m \omega_{q,m} F_j(u_m \mid a, L)`: a risk at a target time is the unit vector of
its grid position, and a restricted mean time lost up to :math:`\tau` has the weights
:math:`\Delta_m(\tau) = \min(u_{m+1}, \tau) - u_m` for :math:`u_m \le \tau` (X6 part (c)).
Targets are ordered arm-major, then cause, then weight row: ``p = (a * J + j) * Q + q``.

The clever covariate of a target with weights :math:`\omega` holds the conditional quotient

.. math::

    H_j(u_k) = \frac{\sum_{m \ge k} \omega_m \{F_j(u_m) - F_j(u_k)\}}{S(u_k)}
    = \sum_{s > k} W(u_s)\, \frac{S(u_{s-1})\, d\Lambda_j(u_s)}{S(u_k)},
    \qquad W(u_s) = \sum_{m \ge s} \omega_m,

the second form by :math:`F_j(u_m) - F_j(u_k) = \sum_{k < s \le m} S(u_{s-1})\,
d\Lambda_j(u_s)` and a swap of the sums.  The first form differences two cumulative sums near
one and divides by a survival that can be :math:`10^{-20}` where a fitted hazard extrapolates,
so it loses every digit; the second sums the incidence increments themselves after
:math:`u_k`, each carrying the factor :math:`S(u_{s-1}) \le S(u_k)`, so numerator and survival
share one scale and nothing is differenced (``_after``).  With :math:`\omega \ge 0` every
term is non-negative and :math:`0 \le H_j \le \sum_m \omega_m`, a conditional risk.  The
score sums :math:`\sum_k r(u_k) H_j(u_k)` over the residual ratio :math:`r = \sum_l w
(dN_l - Y d\Lambda_l) / S`, which the same swap gives as :math:`\sum_s W(u_s)\, S(u_{s-1})
d\Lambda_j(u_s) \sum_{k < s} r(u_k)`: every product there pairs a :math:`1 / S(u_k)` with a
:math:`S(u_{s-1}) \le S(u_k)`.  Both forms need :math:`S` in the normal range of double
precision, which the engine checks (``engine.SURVIVAL_UNDERFLOW``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

from .._typing import FloatArray, IntArray

Norm = Literal["covariance", "variance", "euclidean"]

#: Each diagonal entry of the held covariance is floored at this share of the largest one, so
#: a target with a tiny influence-curve variance cannot dominate the direction.
SIGMA_FLOOR: Final = 1e-8
#: A step that takes some unit's survival at the last target time below this, or below the
#: initial fit's own smallest survival when that is lower, is rejected, so the walk never drives
#: a survival towards zero that the initial fit kept away from it.
WALK_SURVIVAL_FLOOR: Final = 1e-8
#: The smallest step before the walk stops unconverged.
MIN_STEP: Final = 1e-8


@dataclass(frozen=True)
class Problem:
    """The fixed inputs of the walk.

    Parameters
    ----------
    arm : ndarray
        ``(n,)`` received arm, 0 or 1.
    at_risk : ndarray
        ``(n, M)`` indicator, boolean or float, that the unit's follow-up reaches :math:`u_m`.
    events : ndarray
        ``(J, n, M)`` indicator of an event of each cause at :math:`u_m`.
    weight : ndarray
        ``(2, n, M)`` the bounded :math:`1 / (\\pi(a \\mid L)\\, S_c(u_m^- \\mid a, L))`.
    target_weights : ndarray
        ``(Q, M)`` the grid weights :math:`\\omega_{q,m}` of each target of one arm and cause.
        A risk at :math:`t_k` is the unit vector of the last :math:`m` with
        :math:`u_m \\le t_k` (:func:`point_weights`).  With ``expand`` set, the weights are
        on the full grid of ``expand.size`` times.
    expand : ndarray or None
        On a grid reduced to some of the full grid's times, ``(M_full,)`` the position on the
        reduced grid of the last kept time at or before each full-grid time, ``-1`` before the
        first.  The curves are constant between kept times, so a reduced curve read through
        ``expand`` is the full-grid curve.  ``None`` when ``target_weights`` is on the arrays'
        own grid.
    kept : ndarray or None
        ``(M,)`` the full-grid position of each reduced time, set with ``expand``.
    """

    arm: IntArray
    at_risk: np.ndarray
    events: np.ndarray
    weight: FloatArray
    target_weights: FloatArray
    expand: IntArray | None = None
    kept: IntArray | None = None

    @property
    def n(self) -> int:
        """The number of units."""
        return int(self.arm.size)

    @property
    def n_causes(self) -> int:
        """The number of causes."""
        return int(self.events.shape[0])

    @property
    def n_rows(self) -> int:
        """The number of weight rows, the targets of one arm and cause."""
        return int(self.target_weights.shape[0])

    @property
    def n_targets(self) -> int:
        """The number of targets, two arms by every cause and weight row."""
        return 2 * self.n_causes * self.n_rows


_EMPTY = np.zeros(0)


def point_weights(index: IntArray, size: int) -> FloatArray:
    """``(K, M)`` the unit weight rows of risks at the grid positions ``index``.

    Parameters
    ----------
    index : ndarray
        ``(K,)`` grid positions, each in ``range(size)``.
    size : int
        The grid size :math:`M`.

    Returns
    -------
    ndarray
        Row ``k`` is one at ``index[k]`` and zero elsewhere.
    """
    out = np.zeros((len(index), size))
    out[np.arange(len(index)), np.asarray(index, dtype=int)] = 1.0
    return out


def restricted_weights(grid: FloatArray, tau: float) -> FloatArray:
    r"""``(M,)`` the grid weights of :math:`\int_0^\tau F(u)\, du` for a step function on the grid.

    With :math:`F` right-continuous, zero before :math:`u_1` and constant between grid points,
    :math:`\int_0^\tau F = \sum_{u_m \le \tau} F(u_m) \{\min(u_{m+1}, \tau) - u_m\}` exactly,
    with :math:`u_{M+1} = \infty`.

    Parameters
    ----------
    grid : ndarray
        ``(M,)`` the sorted grid.
    tau : float
        The restriction time.

    Returns
    -------
    ndarray
        The weight of each grid point, zero past :math:`\tau`.
    """
    upper = np.minimum(np.append(grid[1:], np.inf), tau)
    return np.where(grid <= tau, upper - grid, 0.0)


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


def _filled(values: FloatArray, expand: IntArray) -> FloatArray:
    """``values`` on a reduced grid read at every full-grid time, zero before the first kept."""
    padded = np.concatenate([np.zeros((*values.shape[:-1], 1)), values], axis=-1)
    # np.take keeps the result C-ordered; an index on the last axis would not, and a matrix
    # product on another memory order sums in another order.
    return np.take(padded, expand + 1, axis=-1)


def evaluate_rows(
    problem: Problem, dlam: FloatArray, survival: FloatArray, incidence: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Each unit's score parts and plug-ins of every target, at hazards with these curves.

    Every operation is per unit, so the rows of a subset of units are those of the whole.

    Parameters
    ----------
    problem : Problem
        The fixed inputs.
    dlam : ndarray
        ``(2, J, n, M)`` hazard jumps.
    survival : ndarray
        ``(2, n, M)`` their survival, from :func:`curves`.
    incidence : ndarray
        ``(2, J, n, M)`` their cause incidences, from :func:`curves`.

    Returns
    -------
    parts : ndarray
        ``(n, P)`` each unit's score part of each target's efficient influence function.
    plugins : ndarray
        ``(n, P)`` each unit's plug-in of each target.
    """
    J, Q = problem.n_causes, problem.n_rows
    n = problem.n
    rows = problem.target_weights.T  # (M, Q)
    expand = problem.expand
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
            if expand is None:
                parts[:, block] = cum_own @ rows
                plugins[:, block] = incidence[a, j] @ rows
            else:
                parts[:, block] = _filled(cum_own, expand) @ rows
                plugins[:, block] = _filled(incidence[a, j], expand) @ rows
        # Freed before the next arm allocates its own, so two arms' temporaries never coexist.
        del weighted, ratio, ratio_before, before, increment, cum_own
    return parts, plugins


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
    parts, plugins = evaluate_rows(problem, dlam, survival, incidence)
    return summarize(parts, plugins, survival, incidence)


def summarize(
    parts: FloatArray,
    plugins: FloatArray,
    survival: FloatArray | None = None,
    incidence: FloatArray | None = None,
) -> Evaluation:
    """The evaluation of every unit's score parts and plug-ins: the means over units.

    Parameters
    ----------
    parts : ndarray
        ``(n, P)`` score parts, from :func:`evaluate_rows`.
    plugins : ndarray
        ``(n, P)`` plug-ins.
    survival : ndarray or None
        The curves to carry; ``None`` carries an empty array.
    incidence : ndarray or None
        The curves to carry; ``None`` carries an empty array.

    Returns
    -------
    Evaluation
        The plug-in, score and influence curves.
    """
    psi = plugins.mean(axis=0)
    influence = parts + plugins - psi
    return Evaluation(
        psi,
        parts.mean(axis=0),
        influence,
        _EMPTY if survival is None else survival,
        _EMPTY if incidence is None else incidence,
    )


def _direction(
    problem: Problem, evaluation: Evaluation, v: FloatArray, dlam: FloatArray
) -> FloatArray:
    r"""``(2, J, n, M)`` the combined clever covariate :math:`\sum_p v_p c_p(u, a, L_i)`.

    A target with grid weights :math:`\omega_m` has the clever covariate
    :math:`1\{l = j\} W(u) - \{\sum_{u_m \ge u} \omega_m F_j(u_m) - W(u) F_j(u)\} / S(u)`,
    :math:`W(u) = \sum_{u_m \ge u} \omega_m`, divided by :math:`\pi S_c`.  The quotient is
    :func:`conditional_risk` of the combined weights, a reverse cumulative sum over time, so a
    step costs :math:`O(JnM)` for any weights.  ``dlam`` holds the hazard jumps ``evaluation``
    was computed at, on the arrays' own grid.  On a reduced grid (``problem.expand`` set)
    :math:`W` runs over the full grid and is read at the kept times; the quotient sums only
    increments :math:`S(u_{s-1})\, d\Lambda_j(u_s)`, which are zero at every dropped time, so
    its sum over the kept times is the full grid's term for term.
    """
    J, Q = problem.n_causes, problem.n_rows
    n, M = problem.at_risk.shape
    expand, kept = problem.expand, problem.kept
    out = np.zeros((2, J, n, M))
    for a in (0, 1):
        # The weight each grid point carries for cause j: sum_q v_(a,j,q) w_(q,m).
        own = v[a * J * Q : (a + 1) * J * Q].reshape(J, Q) @ problem.target_weights
        own = np.cumsum(own[:, ::-1], axis=1)[:, ::-1]  # W(u)
        if expand is not None:
            assert kept is not None
            own = np.take(own, kept, axis=-1)
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
    from ._chunked import DensePlan, run_walk

    plan = DensePlan(problem, dlam)
    del dlam
    return run_walk(
        plan,
        cutoff=cutoff,
        norm=norm,
        step=step,
        max_steps=max_steps,
        max_halvings=max_halvings,
        tilt=tilt,
    ).result


def default_cutoff(n: int) -> float:
    """Item 10's stopping bound :math:`1 / (\\sqrt n \\log n)`.

    Parameters
    ----------
    n : int
        The number of units.

    Returns
    -------
    float
        The bound.
    """
    return 1.0 / (np.sqrt(n) * np.log(n))


#: Units per block of the predictable-variation pass, so its ``(J, P, units, M)`` temporaries
#: stay near 50 MB at the largest grids the engine meets.
_PV_BLOCK_CELLS: Final = 6_000_000


def _multinomial_moment(clever: FloatArray, dlam: FloatArray, rho: FloatArray) -> FloatArray:
    r"""``(P, P)`` :math:`\sum_{i,m} \rho\, V(u_m)_{pq}` for one arm and one block of units.

    :math:`V_{pq} = \sum_l c_{l,p} c_{l,q}\, d\Lambda_l - (\sum_l c_{l,p}\, d\Lambda_l)
    (\sum_l c_{l,q}\, d\Lambda_l)`, the conditional covariance of the multinomial jumps of the
    causes given survival to :math:`u_m`.

    Parameters
    ----------
    clever : ndarray
        ``(J, P, b, M)`` :math:`c_{l,p}(u_m, L_i)`.
    dlam : ndarray
        ``(J, b, M)`` the arm's hazard jumps.
    rho : ndarray
        ``(b, M)`` the nonnegative weight :math:`\hat S(u^-)\, \hat\pi \hat S_c(u^-)\, \hat w^2`.

    Returns
    -------
    ndarray
        The block's contribution.
    """
    J, P = clever.shape[:2]
    out = np.zeros((P, P))
    first = np.zeros(clever.shape[1:])
    for l in range(J):
        scaled = clever[l] * np.sqrt(rho * dlam[l])[None]
        flat = scaled.reshape(P, -1)
        out += flat @ flat.T
        first += clever[l] * dlam[l][None]
    first *= np.sqrt(rho)[None]
    flat = first.reshape(P, -1)
    out -= flat @ flat.T
    return out


def predictable_rows(causes: int, rows: int, times: int) -> int:
    """Units per block of the predictable-variation pass, at least one.

    Parameters
    ----------
    causes : int
        Causes ``J``.
    rows : int
        Weight rows ``Q``.
    times : int
        Grid size ``M``.

    Returns
    -------
    int
        Units whose ``(J, J Q, units, M)`` clever covariates hold about ``_PV_BLOCK_CELLS``
        cells.
    """
    return max(1, _PV_BLOCK_CELLS // max(1, causes * causes * rows * times))


def _martingale_block(
    problem: Problem,
    evaluation: Evaluation,
    dlam: FloatArray,
    reach: FloatArray | None,
    a: int,
) -> FloatArray:
    r"""``(JQ, JQ)`` :math:`\sum_{i,m} \rho_{im} V_a(u_m \mid L_i)` over the targets of arm ``a``.

    Every unit contributes for both arms, whatever it received: the term is
    :math:`E[M_a^2 \mid L]`, a function of :math:`L` alone.
    """
    J, Q = problem.n_causes, problem.n_rows
    n, M = problem.at_risk.shape
    survival = evaluation.survival[a]
    # W_q(u_m) = sum over m' >= m of the row's weights, the same for every unit.
    tail = np.cumsum(problem.target_weights[:, ::-1], axis=1)[:, ::-1]  # (Q, M)
    out = np.zeros((J * Q, J * Q))
    block = predictable_rows(J, Q, M)
    for start in range(0, n, block):
        rows = slice(start, min(n, start + block))
        weight = problem.weight[a][rows]
        rho = _before(survival[rows]) * weight
        rho = rho * (1.0 if reach is None else reach[a][rows])
        clever = np.empty((J, J * Q, rho.shape[0], M))
        for j in range(J):
            # G_q(u) = {sum_{m' >= m} w_q F_j(u_m') - W_q(u) F_j(u)} / S(u), (Q, b, M)
            shared = conditional_risk(
                survival[rows][None], dlam[a, j][rows][None], tail[:, None, :]
            )
            cols = slice(j * Q, (j + 1) * Q)
            for l in range(J):
                clever[l, cols] = -shared
                if l == j:
                    clever[l, cols] += tail[:, None, :]
            # Released before the moment, which then holds only the covariates.
            del shared
        out += _multinomial_moment(clever, dlam[a][:, rows], rho)
        del clever, rho
    return out


class PredictableVariation:
    r"""The predictable-variation second moments, summed over consecutive blocks of units.

    :meth:`add` takes the units of one block, in row order, and :meth:`matrix` returns
    :func:`predictable_covariance` of every unit added: each unit's plug-ins are kept, so the
    plug-in covariance is centred once at the end, and the martingale term of each arm is
    carried as a running sum.  A block of at most :func:`predictable_rows` units adds its term
    as one block of :func:`predictable_covariance` does, so blocks of that size, cut where it
    cuts them, give its matrix bit for bit.

    Parameters
    ----------
    n : int
        The units that will be added.
    causes : int
        Causes ``J``.
    rows : int
        Weight rows ``Q``.
    """

    def __init__(self, n: int, causes: int, rows: int) -> None:
        self.n = int(n)
        self.causes, self.rows = int(causes), int(rows)
        self.plugins = np.zeros((self.n, 2 * self.causes * self.rows))
        self.moments = np.zeros((2, self.causes * self.rows, self.causes * self.rows))
        self.added = 0

    def add(
        self,
        problem: Problem,
        evaluation: Evaluation,
        dlam: FloatArray,
        reach: FloatArray | None = None,
    ) -> None:
        r"""Add the next units in row order.

        Parameters
        ----------
        problem : Problem
            Their fixed inputs; the weight and the target weights are read, and the
            indicators' shapes only.
        evaluation : Evaluation
            Their survival and incidences at ``dlam``.
        dlam : ndarray
            ``(2, J, b, M)`` their hazard jumps.
        reach : ndarray or None
            ``(2, b, M)`` :math:`\min(1, \hat\pi \hat S_c(u^-) / g_{\min})`, or ``None`` to
            read one everywhere.
        """
        J, Q = self.causes, self.rows
        start, stop = self.added, self.added + problem.n
        rows = problem.target_weights.T
        for a in (0, 1):
            for j in range(J):
                cols = slice((a * J + j) * Q, (a * J + j + 1) * Q)
                self.plugins[start:stop, cols] = evaluation.incidence[a, j] @ rows
        # The martingale term reads the survival only; a caller that passes the evaluation
        # without keeping it lets its incidences go here.
        survival = Evaluation(_EMPTY, _EMPTY, _EMPTY, evaluation.survival, _EMPTY)
        del evaluation
        for a in (0, 1):
            self.moments[a] += _martingale_block(problem, survival, dlam, reach, a)
        self.added = stop

    def matrix(self) -> FloatArray:
        """``(P, P)`` the second moments of every unit added, divided by :math:`n`.

        Returns
        -------
        ndarray
            The matrix of :func:`predictable_covariance`.
        """
        assert self.added == self.n, "every unit is added before the matrix is read"
        n, JQ = self.n, self.causes * self.rows
        centred = self.plugins - self.plugins.mean(axis=0)
        out = centred.T @ centred / n
        for a in (0, 1):
            block = slice(a * JQ, (a + 1) * JQ)
            out[block, block] += self.moments[a] / n
        return out


def predictable_covariance(
    problem: Problem,
    evaluation: Evaluation,
    dlam: FloatArray,
    reach: FloatArray | None = None,
) -> FloatArray:
    r"""``(P, P)`` the predictable-variation second moments of the influence curves.

    .. math::

        \hat\Sigma_{pv} = \widehat{\mathrm{Cov}}_n(\hat F) + \sum_a P_n \sum_{u_m}
        \hat S(u_m^-)\, \hat\pi \hat S_c(u_m^-)\, \hat w_a^2\, V_a(u_m),

    with :math:`V_a` the multinomial covariance of :func:`_multinomial_moment` for the targets of
    arm :math:`a` and zero across arms (:math:`1\{A = 1\} 1\{A = 0\} = 0`), and
    :math:`\hat F` each unit's plug-in of every target.  A target with grid weights
    :math:`\omega` has :math:`c_l(u) = 1\{l = j\} W(u) - \{\sum_{u_m \ge u} \omega_m F_j(u_m) -
    W(u) F_j(u)\} / S(u)`, the clever covariate of :func:`_direction` without the weight.  Where
    the positivity floor binds, :math:`\hat\pi \hat S_c \hat w^2 = \hat w \min(1, \hat\pi \hat
    S_c / g_{\min})`, so the floored weight enters squared, as it does in :math:`P_n \hat D^2`.

    At the true nuisances on a sample that is its own law, the matrix equals the population
    covariance of the influence curves: the conditional covariance of the martingale part
    integrates to its second moment, and its cross term with :math:`\hat F` is zero (X6 plan,
    2.1.3, a derivation).  Its diagonal is the predictable variance of each target.  The
    engine adds its units block by block through :class:`PredictableVariation`.

    Parameters
    ----------
    problem : Problem
        The fixed inputs.
    evaluation : Evaluation
        The evaluation at ``dlam``: its survival and incidences.
    dlam : ndarray
        ``(2, J, n, M)`` the hazard jumps, the targeted ones for a fit.
    reach : ndarray or None
        ``(2, n, M)`` :math:`\min(1, \hat\pi \hat S_c(u^-) / g_{\min})`, or ``None`` to read
        one everywhere.

    Returns
    -------
    ndarray
        Population second moments (divided by :math:`n`), in the order of ``evaluation``'s
        influence columns.
    """
    variation = PredictableVariation(problem.n, problem.n_causes, problem.n_rows)
    variation.add(problem, evaluation, dlam, reach)
    return variation.matrix()
