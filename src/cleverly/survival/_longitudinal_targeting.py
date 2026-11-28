r"""One node of the continuous-time longitudinal TMLE: its curves, scores and walk (X6 part (e)).

Fix a regime and a node :math:`k`: the rows are the units that reached visit :math:`k`
following the regime through visit :math:`k - 1`.  Every array is on the pooled grid of
observed times :math:`u_1 < \dots < u_M` up to the last target time, and a row's segment runs
on the times after its visit :math:`s_k` (its jumps are zero before).

**Exits.**  At :math:`u` a row exits by a death of cause :math:`l` or by its next visit, one
multinomial cell: :math:`dE_{d_l}(u) = d\Lambda^{d_l}(u)` and
:math:`dE_v(u) = (1 - \sum_l d\Lambda^{d_l}(u))\, d\Lambda^v(u)`, all evaluated with
:math:`A(s_k)` at the regime's value (the slot).  Segment survival is
:math:`S_k(u) = \prod (1 - \sum_e dE_e)`.

**Values.**  Target :math:`p = (j, \tau)` gives each exit a value,
:math:`v^p_{d_l}(u) = 1\{l = j\} 1\{u \le \tau\}` and :math:`v^p_v(u) = 1\{u \le \tau\}
F^\nu_j(\tau \mid u)`, the cause-:math:`j` incidence by :math:`\tau` of the continuation started
at :math:`u`.  The value after :math:`t` with no exit yet is

.. math::

    Z^p_k(t) = \sum_{t < u \le \tau} \frac{S_k(u^-)}{S_k(t)} \sum_e v^p_e(u)\, dE_e(u),

summed without cancellation as :func:`~cleverly.survival._targeting.conditional_risk` sums it.
Where :math:`S_k(u) = 0` (a certain visit) the "none" outcome has probability zero and the
engine sets :math:`Z_k(u) = v^p_v(u)` (the guard), so the visit term is zero there.  The clever
covariate of exit :math:`e` is :math:`c^p_e(u) = v^p_e(u) - Z^p_k(u)`.

**Scores.**  A row's part of :math:`P_n D^p` at node :math:`k` is

.. math::

    \sum_u h_k(u) \sum_e c^p_e(u) \{dN_e(u) - Y_k(u)\, dE_e(u)\}
    + 1\{\text{visit } k + 1 \text{ at } u\}\, h_k(u)\, \{Z^p_{k+1}(u) - F^\nu_j(\tau \mid u)\},

with :math:`h_k = 1\{A(s_k) = d_r\} \tilde h_k` the score weight and :math:`\tilde h_k` the slot
weight (Rytgaard, Gerds and van der Laan 2022, display (19), with one marked visit process and
the death terms split by cause).

**The walk** tilts the exits of every row with direction :math:`\sum_p \omega_p \tilde h_k(u)
c^p_e(u)` (:func:`~cleverly.survival._targeting.tilt_jumps`), and the continuation's bin hazards
at every visit-grid time of every row with :math:`\sum_p \omega_p \tilde h_k(u) \kappa^p(b, l
\mid u)`, the bins' clever covariate.  The slot weight, free of the observed :math:`A(s_k)`,
makes the tilted slot curve a function of the slot design for every row (rev 7, B1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .._typing import FloatArray, IntArray
from ._chunked import for_blocks
from ._targeting import MIN_STEP, _after, _before, _metric, tilt_jumps

__all__ = ["NodeProblem", "NodeWalk", "node_rows", "soft_outcome", "walk_node"]

_LOG = logging.getLogger(__name__)

#: The (target, row, time) cells of one block's temporaries.
NODE_CELLS: Final = 2**21


@dataclass(frozen=True)
class NodeProblem:
    """The fixed inputs of one node of one regime.

    Parameters
    ----------
    rows : ndarray
        ``(c,)`` the unit of each row.
    unit_weight : ndarray
        ``(c,)`` each row's unit weight (one, or the private seam's).
    total_weight : float
        The sum of every unit's weight, the divisor of :math:`P_n`.
    at_risk : ndarray
        ``(c, M)`` :math:`Y_k(u)`.
    events : ndarray
        ``(E, c, M)`` the observed exit of each row: deaths by cause, then the visit.
    follow : ndarray
        ``(c,)`` whether the row follows the regime at visit :math:`k`.
    slot_weight : ndarray
        ``(c, M)`` the bounded :math:`\\tilde h_k(u)`, zero before the row's visit.
    cause : ndarray
        ``(P,)`` each target's cause, as an index.
    tau : ndarray
        ``(P,)`` each target's time, as a grid index.
    tau_bin : ndarray
        ``(P,)`` the bin each target's time ends.
    bin_of : ndarray
        ``(M,)`` the bin of each grid time.
    n_bins : int
        The number of bins, one per target time.
    visit_grid : ndarray
        ``(U,)`` the grid positions where a visit can fall; empty at a node with no visit exit.
    visit_at : ndarray
        ``(c,)`` the position in ``visit_grid`` of the row's observed next visit, ``-1`` without.
    next_value : ndarray
        ``(c, P)`` :math:`Z^p_{k+1}` at the observed next visit, from the next node's final fit.
    """

    rows: IntArray
    unit_weight: FloatArray
    total_weight: float
    at_risk: np.ndarray
    events: np.ndarray
    follow: np.ndarray
    slot_weight: FloatArray
    cause: IntArray
    tau: IntArray
    tau_bin: IntArray
    bin_of: IntArray
    n_bins: int
    visit_grid: IntArray
    visit_at: IntArray
    next_value: FloatArray

    @property
    def size(self) -> int:
        """The number of rows."""
        return int(self.rows.size)

    @property
    def has_visit(self) -> bool:
        """Whether the visit is an exit of this node."""
        return bool(self.visit_grid.size)

    @property
    def n_targets(self) -> int:
        """The number of targets."""
        return int(self.cause.size)

    def take(self, lo: int, hi: int) -> NodeProblem:
        """The rows ``lo:hi``.

        Parameters
        ----------
        lo : int
            First row.
        hi : int
            One past the last row.

        Returns
        -------
        NodeProblem
            The block.
        """
        rows = slice(lo, hi)
        return NodeProblem(
            self.rows[rows],
            self.unit_weight[rows],
            self.total_weight,
            self.at_risk[rows],
            self.events[:, rows],
            self.follow[rows],
            self.slot_weight[rows],
            self.cause,
            self.tau,
            self.tau_bin,
            self.bin_of,
            self.n_bins,
            self.visit_grid,
            self.visit_at[rows],
            self.next_value[rows],
        )


@dataclass
class NodeState:
    """A node's current fit: the exit cells and the continuation's bin hazards.

    Parameters
    ----------
    exits : ndarray
        ``(E, c, M)`` the multinomial exit cells.
    continuation : ndarray or None
        ``(J, c, U, B)`` the bin hazards after a visit at each visit-grid time, ``None`` at a
        node with no visit exit.
    """

    exits: FloatArray
    continuation: FloatArray | None


def continuation_curves(lam: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The survival after each bin and the bin probabilities of the binned law.

    Parameters
    ----------
    lam : ndarray
        ``(J, ..., B)`` bin hazards.

    Returns
    -------
    survival : ndarray
        ``(..., B)`` the probability of no event through each bin.
    nu : ndarray
        ``(J, ..., B)`` :math:`\\nu(b, l)`.
    """
    survival = np.cumprod(1.0 - lam.sum(axis=0), axis=-1)
    return survival, _before(survival)[None] * lam


def _incidence(nu: FloatArray, cause: IntArray, tau_bin: IntArray) -> FloatArray:
    """``(P, ...)`` each target's incidence by its time: the bins up to its bin, of its cause."""
    cumulative = np.cumsum(nu, axis=-1)
    return np.stack([cumulative[cause[p], ..., tau_bin[p]] for p in range(cause.size)])


@dataclass
class _Parts:
    """One block's curves at one state, what the score, the direction and the outcome read."""

    survival: FloatArray  # (c, M)
    values: FloatArray  # (P, c, M) Z^p_k(u), with the guard
    visit_value: FloatArray  # (P, c, M) v^p_v(u)
    start: FloatArray  # (c, P) Z^p_k(s_k)
    incidence: FloatArray | None  # (P, c, U) F^nu_p(u)
    nu: FloatArray | None  # (J, c, U, B)
    bin_survival: FloatArray | None  # (c, U, B)


def _mask(problem: NodeProblem, M: int) -> FloatArray:
    """``(P, M)`` the indicator :math:`1\\{u \\le \\tau_p\\}`."""
    return (np.arange(M)[None, :] <= problem.tau[:, None]).astype(float)


def node_curves(problem: NodeProblem, state: NodeState) -> _Parts:
    """The values of every target on the block's rows at ``state``.

    Parameters
    ----------
    problem : NodeProblem
        The block.
    state : NodeState
        Its fit.

    Returns
    -------
    _Parts
        The curves.
    """
    exits = state.exits
    _E, c, M = exits.shape
    P = problem.n_targets
    mask = _mask(problem, M)
    survival = np.cumprod(1.0 - exits.sum(axis=0), axis=-1)
    before = _before(survival)
    visit_value = np.zeros((P, c, M))
    incidence = nu = bin_survival = None
    if problem.has_visit:
        assert state.continuation is not None
        bin_survival, nu = continuation_curves(state.continuation)
        incidence = _incidence(nu, problem.cause, problem.tau_bin)  # (P, c, U)
        visit_value[:, :, problem.visit_grid] = incidence * mask[:, None, problem.visit_grid]
    values = np.empty((P, c, M))
    start = np.empty((c, P))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in range(P):
            gain = exits[problem.cause[p]] * mask[p][None, :]
            if problem.has_visit:
                gain = gain + visit_value[p] * exits[-1]
            weighted = before * gain
            start[:, p] = weighted.sum(axis=1)
            z = _after(weighted) / survival
            # The guard: where the segment's survival is zero, Z = v_v.
            values[p] = np.where(survival > 0, z, visit_value[p])
    return _Parts(survival, values, visit_value, start, incidence, nu, bin_survival)


def node_scores(problem: NodeProblem, state: NodeState, parts: _Parts | None = None) -> FloatArray:
    """``(c, P)`` each row's part of :math:`P_n D^p` at node :math:`k`, unweighted.

    Parameters
    ----------
    problem : NodeProblem
        The block.
    state : NodeState
        Its fit.
    parts : _Parts or None
        Its curves, when already computed.

    Returns
    -------
    ndarray
        The exit term plus the covariate term.
    """
    parts = node_curves(problem, state) if parts is None else parts
    exits = state.exits
    _E, c, M = exits.shape
    P = problem.n_targets
    mask = _mask(problem, M)
    Y = problem.at_risk
    dN = problem.events.astype(float)
    residual = dN - Y[None] * exits  # (E, c, M)
    total_residual = residual.sum(axis=0)
    h = problem.slot_weight * problem.follow[:, None]
    out = np.empty((c, P))
    for p in range(P):
        own = residual[problem.cause[p]] * mask[p][None, :]
        if problem.has_visit:
            own = own + parts.visit_value[p] * residual[-1]
        out[:, p] = (h * (own - parts.values[p] * total_residual)).sum(axis=1)
    if problem.has_visit:
        assert parts.incidence is not None
        seen = np.flatnonzero(problem.visit_at >= 0)
        if seen.size:
            at = problem.visit_at[seen]
            weight = h[seen, problem.visit_grid[at]]
            predicted = parts.incidence[:, seen, at].T  # (s, P)
            out[seen] += weight[:, None] * (problem.next_value[seen] - predicted)
    return out


def node_direction(
    problem: NodeProblem, state: NodeState, v: FloatArray, parts: _Parts | None = None
) -> tuple[FloatArray, FloatArray | None]:
    """The walk's direction in the exits and in the continuation at ``state``.

    Parameters
    ----------
    problem : NodeProblem
        The block.
    state : NodeState
        Its fit.
    v : ndarray
        ``(P,)`` the direction over targets.
    parts : _Parts or None
        Its curves, when already computed.

    Returns
    -------
    exits : ndarray
        ``(E, c, M)`` :math:`\\sum_p v_p \\tilde h_k c^p_e`.
    continuation : ndarray or None
        ``(J, c, U, B)`` :math:`\\sum_p v_p \\tilde h_k \\kappa^p`.
    """
    parts = node_curves(problem, state) if parts is None else parts
    exits = state.exits
    E, c, M = exits.shape
    P = problem.n_targets
    mask = _mask(problem, M)
    weight = problem.slot_weight
    shared = np.tensordot(v, parts.values, axes=1)  # (c, M)
    out = np.empty_like(exits)
    for e in range(E):
        own = np.zeros((c, M))
        if problem.has_visit and e == E - 1:
            own = np.tensordot(v, parts.visit_value, axes=1)
        else:
            for p in range(P):
                if problem.cause[p] == e:
                    own = own + v[p] * mask[p][None, :]
        out[e] = weight * (own - shared)
    if not problem.has_visit:
        return out, None
    assert state.continuation is not None and parts.nu is not None
    assert parts.bin_survival is not None
    lam = state.continuation
    _J, _, _U, B = lam.shape
    h = weight[:, problem.visit_grid][:, :, None]  # (c, U, 1)
    before = _before(parts.bin_survival)
    direction = np.zeros_like(lam)
    bins = np.arange(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in range(P):
            inside = (bins <= problem.tau_bin[p]).astype(float)
            later = _after(before * lam[problem.cause[p]] * inside[None, None, :])
            rest = np.where(parts.bin_survival > 0, later / parts.bin_survival, 0.0)
            direction -= v[p] * (h * rest)[None]
            direction[problem.cause[p]] += v[p] * h * inside[None, None, :]
    return out, direction


def tilt(
    state: NodeState, exits: FloatArray, continuation: FloatArray | None, step: float
) -> NodeState:
    """The state one step along a direction.

    Parameters
    ----------
    state : NodeState
        The current fit.
    exits : ndarray
        The exits' direction.
    continuation : ndarray or None
        The continuation's direction.
    step : float
        The step size.

    Returns
    -------
    NodeState
        The tilted fit.
    """
    tilted = tilt_jumps(state.exits[None], step * exits[None])[0]
    lam = state.continuation
    if lam is not None:
        assert continuation is not None
        lam = tilt_jumps(lam[None], step * continuation[None])[0]
    return NodeState(tilted, lam)


def soft_outcome(problem: NodeProblem, state: NodeState) -> FloatArray:
    """``(c, B, J)`` the binned sub-distribution of the event time after the row's visit.

    The death of cause :math:`l` in bin :math:`b` directly, or a visit at :math:`u` followed by
    the continuation's bin :math:`b`: the analogue of X70's ``pseudo_distribution``.

    Parameters
    ----------
    problem : NodeProblem
        The node.
    state : NodeState
        Its final fit.

    Returns
    -------
    ndarray
        Each row's soft outcome, for the previous node's continuation.
    """
    exits = state.exits
    E, _c, M = exits.shape
    B = problem.n_bins
    J = E - 1 if problem.has_visit else E
    survival = np.cumprod(1.0 - exits.sum(axis=0), axis=-1)
    before = _before(survival)
    onehot = np.zeros((M, B))
    inside = problem.bin_of < B
    onehot[np.flatnonzero(inside), problem.bin_of[inside]] = 1.0
    out = np.einsum("lcm,mb->cbl", before[None] * exits[:J], onehot)
    if problem.has_visit:
        assert state.continuation is not None
        _, nu = continuation_curves(state.continuation)
        mass = (before * exits[-1])[:, problem.visit_grid]  # (c, U)
        out += np.einsum("cu,lcub->cbl", mass, nu)
    return out


def node_rows(problem: NodeProblem, state: NodeState) -> tuple[FloatArray, FloatArray]:
    """``(c, P)`` each row's score part and its value at its visit, :math:`Z^p_k(s_k)`.

    Parameters
    ----------
    problem : NodeProblem
        The node.
    state : NodeState
        Its fit.

    Returns
    -------
    tuple of ndarray
        The parts and the values.
    """
    parts = node_curves(problem, state)
    return node_scores(problem, state, parts), parts.start


def _blocks(problem: NodeProblem, state: NodeState) -> list[tuple[int, int]]:
    E, c, M = state.exits.shape
    width = problem.n_targets * M * max(E, 1)
    if state.continuation is not None:
        J, _, U, B = state.continuation.shape
        width = max(width, problem.n_targets * U * B * J)
    size = max(1, NODE_CELLS // max(width, 1))
    return [(lo, min(lo + size, c)) for lo in range(0, c, size)]


def _block_state(state: NodeState, lo: int, hi: int) -> NodeState:
    lam = None if state.continuation is None else state.continuation[:, lo:hi]
    return NodeState(state.exits[:, lo:hi], lam)


@dataclass
class NodeWalk:
    """The record of one node's walk.

    Parameters
    ----------
    state : NodeState
        The final fit.
    parts : ndarray
        ``(c, P)`` each row's score part at the final fit.
    start : ndarray
        ``(c, P)`` each row's :math:`Z^p_k(s_k)` at the final fit.
    steps : int
        Accepted steps.
    halvings : int
        Rejected steps.
    converged : bool
        Whether every target's standardized node score reached the cutoff.
    standardized : ndarray
        ``(P,)`` each target's final :math:`|P_n D^p_k| / \\hat\\sigma_p`.
    history : list of float
        The largest standardized node score after each accepted step.
    """

    state: NodeState
    parts: FloatArray
    start: FloatArray
    steps: int = 0
    halvings: int = 0
    converged: bool = True
    standardized: FloatArray = field(default_factory=lambda: np.zeros(0))
    history: list[float] = field(default_factory=list)


def _evaluate(problem: NodeProblem, state: NodeState, n_jobs: int) -> tuple[FloatArray, FloatArray]:
    c, P = problem.size, problem.n_targets
    parts = np.empty((c, P))
    start = np.empty((c, P))
    bounds = _blocks(problem, state)

    def work(b: int) -> tuple[FloatArray, FloatArray]:
        lo, hi = bounds[b]
        return node_rows(problem.take(lo, hi), _block_state(state, lo, hi))

    def commit(b: int, found: tuple[FloatArray, FloatArray]) -> None:
        lo, hi = bounds[b]
        parts[lo:hi], start[lo:hi] = found

    for_blocks(len(bounds), work, commit, n_jobs)
    return parts, start


def _direction(
    problem: NodeProblem, state: NodeState, v: FloatArray, n_jobs: int
) -> tuple[FloatArray, FloatArray | None]:
    exits = np.empty_like(state.exits)
    lam = None if state.continuation is None else np.empty_like(state.continuation)
    bounds = _blocks(problem, state)

    def work(b: int) -> tuple[FloatArray, FloatArray | None]:
        lo, hi = bounds[b]
        return node_direction(problem.take(lo, hi), _block_state(state, lo, hi), v)

    def commit(b: int, found: tuple[FloatArray, FloatArray | None]) -> None:
        lo, hi = bounds[b]
        exits[:, lo:hi] = found[0]
        if lam is not None:
            assert found[1] is not None
            lam[:, lo:hi] = found[1]

    for_blocks(len(bounds), work, commit, n_jobs)
    return exits, lam


def score_of(problem: NodeProblem, parts: FloatArray) -> FloatArray:
    """``(P,)`` the node's part of :math:`P_n D^p`.

    Parameters
    ----------
    problem : NodeProblem
        The node.
    parts : ndarray
        ``(c, P)`` its rows' parts.

    Returns
    -------
    ndarray
        The weighted sum over the rows divided by the total weight of every unit.
    """
    return (problem.unit_weight @ parts) / problem.total_weight


def walk_node(
    problem: NodeProblem,
    initial: NodeState,
    *,
    sigma: FloatArray,
    covariance: FloatArray,
    cutoff: float,
    norm: str = "covariance",
    step: float = 0.1,
    max_steps: int = 500,
    walk: bool = True,
    n_jobs: int = 1,
) -> NodeWalk:
    """Walk one node's universal submodel until each target's node score is below the cutoff.

    The stopping statistic is :math:`\\max_p |P_n D^p_k| / \\hat\\sigma_p`, with
    :math:`\\hat\\sigma_p` held at the initial pass's whole-curve value, so the cutoffs of the
    nodes add up to one bound on :math:`P_n D^p`.  Steps follow the baseline walk: a trial is
    accepted when its score is shorter in the metric held at the initial fit, the step halves
    on a rejection and doubles, up to its first value, after an acceptance.

    Parameters
    ----------
    problem : NodeProblem
        The node.
    initial : NodeState
        The initial fit.
    sigma : ndarray
        ``(P,)`` the initial whole-curve standard deviations.
    covariance : ndarray
        ``(P, P)`` the initial whole-curve covariance, which fixes the metric.
    cutoff : float
        The node's cutoff, :math:`s_n / N`.
    norm : {"covariance", "variance", "euclidean"}
        The metric.
    step : float
        The first step size.
    max_steps : int
        The cap on accepted steps.
    walk : bool
        ``False`` records the initial fit without a step.
    n_jobs : int
        Threads.

    Returns
    -------
    NodeWalk
        The final fit and the record.
    """
    sigma = np.where(sigma > 0, sigma, 1.0)
    metric = _metric(covariance, norm)  # type: ignore[arg-type]
    state = initial
    parts, start = _evaluate(problem, state, n_jobs)
    score = score_of(problem, parts)
    statistic = float(np.max(np.abs(score) / sigma)) if score.size else 0.0
    history = [statistic]
    converged = statistic <= cutoff
    if not walk:
        return NodeWalk(state, parts, start, 0, 0, converged, np.abs(score) / sigma, history)
    steps = halvings = 0
    initial_step = step
    while not converged and steps < max_steps and halvings < 2 * max_steps and step >= MIN_STEP:
        weighted = metric @ score
        length = float(np.sqrt(score @ weighted))
        if length == 0.0:
            converged = True
            break
        v = weighted / length
        exits_dir, lam_dir = _direction(problem, state, v, n_jobs)
        while step >= MIN_STEP and halvings < 2 * max_steps:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                candidate = tilt(state, exits_dir, lam_dir, step)
                trial_parts, trial_start = _evaluate(problem, candidate, n_jobs)
            trial = score_of(problem, trial_parts)
            trial_length = float(np.sqrt(trial @ metric @ trial))
            if np.isfinite(trial_length) and trial_length < length:
                state, parts, start, score = candidate, trial_parts, trial_start, trial
                steps += 1
                step = min(2.0 * step, initial_step)
                break
            step /= 2.0
            halvings += 1
        else:
            break
        statistic = float(np.max(np.abs(score) / sigma))
        history.append(statistic)
        converged = statistic <= cutoff
    return NodeWalk(state, parts, start, steps, halvings, converged, np.abs(score) / sigma, history)
