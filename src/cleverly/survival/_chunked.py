r"""The targeting walk of the continuous-time engine in passes over blocks of units.

The walk of :mod:`cleverly.survival._targeting` reads each unit only through that unit's own
rows (its follow-up, its initial hazards and its weights) and a few global quantities: the
direction vector :math:`v_s` and step of each accepted step, the pairs held on it, and the means
over units that fix the metric, the score and the stopping statistic.  So the walk runs over
contiguous blocks of units ("chunks"), one block at a time, and holds every unit at every grid
time only for the blocks it keeps ("resident" blocks).  A block that is not kept is rebuilt
from the fitted hazard models and the record of accepted steps whenever a pass needs it, with the
same operations in the same order, so the rebuilt hazards are bit for bit the kept ones.

Each pass writes every block's rows of the score parts and plug-ins into full ``(n, P)``
arrays, and every mean over units is taken on those arrays with the calls the dense walk uses.
For targets whose grid weights are unit vectors (risks) and the shipped hazard learners, every
number is therefore bit for bit the dense walk's, whatever the block size, the kept blocks or
the number of trial steps a pass evaluates.

A pass with some block not kept evaluates a short ladder of step sizes, the step and its halves,
so one rebuild serves the rejections a walk usually meets; :func:`ladder_decision` consumes the
ladder exactly as the dense loop would take its trials one at a time.

X106 part C2 (bounded-memory continuous-time survival).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

import numpy as np

from .._typing import FloatArray, IntArray
from . import _targeting as T

_LOG = logging.getLogger(__name__)

#: The (unit, grid time) cells of one cause in one block: a ``(J, c, M)`` block of floats is
#: ``8 * CHUNK_CELLS`` bytes, 2 MiB.
CHUNK_CELLS: Final = 2**18
#: The trial step sizes one pass evaluates when some block is rebuilt: the step and its halves.
LADDER: Final = 4
#: The share of the memory available that kept blocks may take under the default budget.
RESIDENT_SHARE: Final = 0.75
#: A bound on a step's increment at a time the reduced grid drops, above which the increment is
#: computed exactly on the full grid: ``exp`` overflows above about 709.78.
OVERFLOW_BOUND: Final = 700.0
#: Whether the walk runs on the event-support grid (a private switch for the benchmark).
REDUCE_GRID: Final = True


def chunk_size(n: int, causes: int, times: int) -> int:
    """The units of one block: ``CHUNK_CELLS // (causes * times)``, at least one, at most ``n``.

    It depends on the dimensions only, never on the fitted learners, the budget or the
    machine, so every fit of the same data runs the same blocks.

    Parameters
    ----------
    n : int
        Units.
    causes : int
        Causes ``J``.
    times : int
        Grid size ``M``.

    Returns
    -------
    int
        The block size ``c``.
    """
    return int(min(max(CHUNK_CELLS // max(int(causes) * int(times), 1), 1), max(int(n), 1)))


def blocks(n: int, size: int) -> list[tuple[int, int]]:
    """The contiguous row ranges ``[lo, hi)`` of ``size`` units that cover ``range(n)``.

    Parameters
    ----------
    n : int
        Units.
    size : int
        Units per block.

    Returns
    -------
    list of tuple of int
        The ranges in row order.
    """
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


# ------------------------------------------------------------------- the decision


@dataclass(frozen=True)
class Outcome:
    """What one trial step size of a pass showed, over every unit.

    Parameters
    ----------
    low : ndarray or None
        ``(2, n)`` the pairs whose survival at the grid's end fell below the walk's floor on
        the trial, or ``None`` when none did.
    still_low : bool
        With ``low`` set: whether a held pair still read below the floor when the same step
        was retried with the low pairs held.
    length : float
        The length of the trial's score in the walk's metric: the retried trial's when
        ``low`` is set.
    """

    low: np.ndarray | None
    still_low: bool
    length: float


@dataclass(frozen=True)
class Decision:
    """How the walk consumed a ladder of trial steps.

    Parameters
    ----------
    consumed : int
        The entries the walk took, in order.
    accepted : int or None
        The accepted entry, the last consumed, or ``None``.
    steps : int
        Accepted steps after the ladder.
    halvings : int
        Rejected steps after the ladder.
    step : float
        The step size after the ladder.
    marked : tuple of int
        The consumed entries whose low pairs join the pairs ever held.
    """

    consumed: int
    accepted: int | None
    steps: int
    halvings: int
    step: float
    marked: tuple[int, ...]


def ladder_decision(
    outcomes: Sequence[Outcome],
    *,
    length: float,
    steps: int,
    halvings: int,
    step: float,
    initial_step: float,
    max_steps: int,
    halving_cap: int,
    min_step: float,
) -> Decision:
    """Take a ladder's trials one at a time, as the dense walk's loop takes them.

    Entry ``e`` of ``outcomes`` is the trial at the step size ``step / 2**e``.  The dense
    loop tries a step; if some pairs fall below the floor it holds them and retries the same
    step, and halves when a held pair still reads low; otherwise it halves when the trial's
    score is not shorter (or not finite) and accepts it when it is.  Before each trial it
    checks its loop condition, so the ladder stops where the dense loop would stop.

    Parameters
    ----------
    outcomes : sequence of Outcome
        The trials of the ladder, largest step first.
    length : float
        The current score's length, the bar a trial must pass.
    steps : int
        Accepted steps so far.
    halvings : int
        Rejected steps so far.
    step : float
        The step size of entry 0.
    initial_step : float
        The walk's first step size, the cap on the step after an acceptance.
    max_steps : int
        The cap on accepted steps.
    halving_cap : int
        The cap on rejected steps.
    min_step : float
        The smallest step size the loop tries.

    Returns
    -------
    Decision
        The consumed entries, the accepted one and the counters after them.
    """
    marked: list[int] = []
    for entry, outcome in enumerate(outcomes):
        if not (steps < max_steps and halvings < halving_cap and step >= min_step):
            return Decision(entry, None, steps, halvings, step, tuple(marked))
        if outcome.low is not None:
            marked.append(entry)
            if outcome.still_low:
                step /= 2
                halvings += 1
                continue
        if not np.isfinite(outcome.length) or outcome.length >= length:
            step /= 2
            halvings += 1
            continue
        return Decision(
            entry + 1, entry, steps + 1, halvings, min(2.0 * step, initial_step), tuple(marked)
        )
    return Decision(len(outcomes), None, steps, halvings, step, tuple(marked))


# ------------------------------------------------------------------- the plan


class Plan(Protocol):
    """The blocks of one walk and how to build each one's inputs."""

    n: int
    n_targets: int

    @property
    def bounds(self) -> list[tuple[int, int]]:
        """The row range of each block."""
        ...  # pragma: no cover - protocol

    def problem(self, k: int) -> T.Problem:
        """The fixed inputs of block ``k`` on the walk's grid."""
        ...  # pragma: no cover - protocol

    def initial(self, k: int) -> FloatArray:
        """``(2, J, c, M)`` the initial hazard jumps of block ``k`` on the walk's grid."""
        ...  # pragma: no cover - protocol

    def overflow(
        self, k: int, units: IntArray, state: _State, v: FloatArray, step: float
    ) -> np.ndarray | None:
        """``(2, len(units))`` the pairs whose increment overflows at a dropped time, or ``None``.

        ``None`` when the walk's grid drops no time.  Called only for units whose bound
        passes :data:`OVERFLOW_BOUND`.
        """
        ...  # pragma: no cover - protocol

    def dropped_weight(self, k: int) -> FloatArray | None:
        """``(2, c)`` each pair's largest weight at a dropped time, or ``None`` when none."""
        ...  # pragma: no cover - protocol


@dataclass
class _State:
    """A block's hazards after ``applied`` accepted steps, with their curves."""

    jumps: FloatArray
    survival: FloatArray
    incidence: FloatArray
    applied: int
    problem: T.Problem


@dataclass(frozen=True)
class Accepted:
    """One accepted step of the walk, enough to replay it on any block.

    Parameters
    ----------
    v : ndarray
        ``(P,)`` the unit direction in target space.
    step : float
        The step size.
    held : ndarray or None
        Flat indices into ``(2, n)`` of the pairs held on the step, or ``None``.
    """

    v: FloatArray
    step: float
    held: IntArray | None


@dataclass
class WalkRun:
    """What :func:`run_walk` returns besides the dense walk's record.

    Parameters
    ----------
    result : WalkResult
        The walk's record, with ``dlam`` and the curves filled only when every unit sits in
        one kept block.
    accepted : list of Accepted
        The accepted steps.
    passes : int
        Passes over the blocks, the initial and the final ones included.
    replayed : int
        Accepted steps replayed on blocks that were not kept, summed over blocks.
    """

    result: T.WalkResult
    accepted: list[Accepted] = field(default_factory=list)
    passes: int = 0
    replayed: int = 0


def _light(survival: FloatArray, incidence: FloatArray) -> T.Evaluation:
    return T.Evaluation(T._EMPTY, T._EMPTY, T._EMPTY, survival, incidence)


class _Walker:
    """The block loop: states, replay and one trial on one block."""

    def __init__(self, plan: Plan, *, resident: int, tilt: Literal["all", "factual"]) -> None:
        self.plan = plan
        self.bounds = plan.bounds
        self.resident = int(min(max(resident, 0), len(self.bounds)))
        self.tilt = tilt
        self.slots: list[_State | None] = [None] * len(self.bounds)
        self.accepted: list[Accepted] = []
        self.replayed = 0

    def fresh(self, k: int) -> _State:
        jumps = self.plan.initial(k)
        survival, incidence = T.curves(jumps)
        return _State(jumps, survival, incidence, 0, self.plan.problem(k))

    def keep(self, k: int, state: _State) -> None:
        if k < self.resident:
            self.slots[k] = state

    def state(self, k: int) -> _State:
        """Block ``k`` after every accepted step, kept or rebuilt."""
        slot = self.slots[k]
        state = slot if slot is not None else self.fresh(k)
        target = len(self.accepted)
        if state.applied < target and slot is None:
            self.replayed += target - state.applied
        lo, hi = self.bounds[k]
        while state.applied < target:
            step = self.accepted[state.applied]
            direction = self.direction(k, state, step.v)
            if step.held is not None:
                held = self.local(step.held, lo, hi)
                if held.any():
                    direction *= ~held[:, None, :, None]
            direction *= step.step
            jumps = T.tilt_jumps(state.jumps, direction)
            del direction
            survival, incidence = T.curves(jumps)
            state = _State(jumps, survival, incidence, state.applied + 1, state.problem)
        self.keep(k, state)
        return state

    def local(self, flat: IntArray, lo: int, hi: int) -> np.ndarray:
        """The ``(2, hi - lo)`` mask of the flat ``(2, n)`` indices that fall in rows ``lo:hi``."""
        n = self.plan.n
        arm, unit = np.divmod(flat, n)
        inside = (unit >= lo) & (unit < hi)
        mask = np.zeros((2, hi - lo), dtype=bool)
        mask[arm[inside], unit[inside] - lo] = True
        return mask

    def direction(self, k: int, state: _State, v: FloatArray) -> FloatArray:
        direction = T._direction(state.problem, _light(state.survival, state.incidence), v)
        if self.tilt == "factual":
            arm = state.problem.arm
            factual = np.stack([(arm == a).astype(float) for a in (0, 1)])[:, None, :, None]
            direction = direction * factual
        return direction

    def trial(
        self,
        k: int,
        state: _State,
        direction: FloatArray,
        step: float,
        v: FloatArray,
        floor: float,
        held: np.ndarray | None,
    ) -> tuple[FloatArray, FloatArray, np.ndarray]:
        """One trial on block ``k``: its parts, plug-ins and low pairs."""
        increment = direction if held is None else direction * ~held[:, None, :, None]
        increment = increment * step
        candidate = T.tilt_jumps(state.jumps, increment)
        del increment
        # A trial step can take a survival to zero, where the score divides 0 by 0; the walk
        # rejects such a trial, so its warning carries nothing.
        with np.errstate(divide="ignore", invalid="ignore"):
            survival, incidence = T.curves(candidate)
            parts, plugins = T.evaluate_rows(state.problem, candidate, survival, incidence)
        low = survival[..., -1] < floor
        del candidate, survival, incidence
        bound = self.plan.dropped_weight(k)
        if bound is not None:
            nan = self.overflowing(k, state, direction, step, v, held, bound)
            if nan is not None:
                # The dense walk's tilt of a zero jump at a dropped time is inf * 0 there: the
                # pair's jumps, survival and parts turn NaN, so its survival is not low and the
                # trial's score is not finite.
                units = np.flatnonzero(nan.any(axis=0))
                parts[units] = np.nan
                plugins[units] = np.nan
                low &= ~nan
        return parts, plugins, low

    def overflowing(
        self,
        k: int,
        state: _State,
        direction: FloatArray,
        step: float,
        v: FloatArray,
        held: np.ndarray | None,
        dropped: FloatArray,
    ) -> np.ndarray | None:
        """``(2, c)`` the pairs whose dense increment overflows at a dropped time, or ``None``."""
        problem = state.problem
        J, Q = problem.n_causes, problem.n_rows
        weights = problem.target_weights
        with np.errstate(invalid="ignore", over="ignore"):
            # |W_l| and the incidence sums are at most the row's absolute weight mass times the
            # largest absolute incidence, and the survival at a dropped time is the last kept
            # one's (or one), so this bounds |step * direction| at every dropped time.
            mass = np.stack(
                [
                    np.abs(v[a * J * Q : (a + 1) * J * Q].reshape(J, Q) @ weights).sum(axis=1)
                    for a in (0, 1)
                ]
            )  # (2, J)
            largest = np.nan_to_num(
                np.abs(state.incidence).max(axis=(1, 3)), nan=np.inf
            )  # (2, c)
            lowest = np.minimum(np.abs(state.survival).min(axis=-1), 1.0)  # (2, c)
            lowest = np.where(np.isnan(lowest), 0.0, lowest)
            shared = 2.0 * largest * mass.sum(axis=1)[:, None] / lowest
            bound = step * dropped * (mass.max(axis=1)[:, None] + shared)
        bound = np.where(np.isnan(bound), np.inf, bound)
        if self.tilt == "factual":
            bound = np.where(problem.arm[None, :] == np.arange(2)[:, None], bound, 0.0)
        if held is not None:
            bound = np.where(held, 0.0, bound)
        flagged = bound > OVERFLOW_BOUND
        if not flagged.any():
            return None
        units = np.flatnonzero(flagged.any(axis=0))
        exact = self.plan.overflow(k, units, state, v, step)
        if exact is None:
            return None
        out = np.zeros_like(flagged)
        out[:, units] = exact & flagged[:, units]
        return out if out.any() else None


def run_walk(
    plan: Plan,
    *,
    cutoff: float,
    norm: T.Norm = "covariance",
    step: float = 0.1,
    max_steps: int = 500,
    max_halvings: int | None = None,
    tilt: Literal["all", "factual"] = "all",
    resident: int | None = None,
    ladder: int | None = None,
    final: Callable[[int, int, int, Any], None] | None = None,
) -> WalkRun:
    """Walk the universal least favourable submodel over the blocks of ``plan``.

    The walk, its stopping rule, its step control and its record are those of
    :func:`cleverly.survival._targeting.walk`; only the order in which units are visited
    differs.

    Parameters
    ----------
    plan : Plan
        The blocks.
    cutoff : float
        Stop when :math:`\\max_p |P_n D_p| / \\hat\\sigma_p` is at or below it.
    norm : {"covariance", "variance", "euclidean"}
        The inner product, with the covariance held at the initial fit.
    step : float
        The first step size.
    max_steps : int
        The cap on accepted steps.
    max_halvings : int or None
        The cap on rejected steps, ``2 * max_steps`` by default.
    tilt : {"all", "factual"}
        As :func:`~cleverly.survival._targeting.walk` takes it.
    resident : int or None
        The blocks kept between passes, the first ones in row order; ``None`` keeps all.
    ladder : int or None
        The trial step sizes a pass evaluates when some block is not kept; ``None`` takes
        :data:`LADDER`.  A pass with every block kept evaluates one.
    final : callable or None
        Called once per block after the walk as ``final(k, lo, hi, state)`` with the block's
        targeted state.

    Returns
    -------
    WalkRun
        The record and the accepted steps.
    """
    halving_cap = 2 * max_steps if max_halvings is None else max_halvings
    initial_step = step
    bounds = plan.bounds
    n, P = plan.n, plan.n_targets
    walker = _Walker(plan, resident=len(bounds) if resident is None else resident, tilt=tilt)

    all_kept = walker.resident == len(bounds)
    rungs = 1 if all_kept else int(LADDER if ladder is None else ladder)
    passes = 0

    # Pass 0: the initial fit.
    started = time.perf_counter()
    parts = np.empty((n, P))
    plugins = np.empty((n, P))
    lowest, undefined = np.inf, False
    for k, (lo, hi) in enumerate(bounds):
        state = walker.fresh(k)
        parts[lo:hi], plugins[lo:hi] = T.evaluate_rows(
            state.problem, state.jumps, state.survival, state.incidence
        )
        last = state.survival[..., -1]
        if np.isnan(last).any():
            undefined = True
        elif last.size:
            lowest = min(lowest, float(np.min(last)))
        walker.keep(k, state)
        del state
    passes += 1
    _log_pass("initial", 0, started, walker, passes)
    initial = T.summarize(parts, plugins)
    del parts, plugins
    metric = T._metric(np.cov(initial.influence, rowvar=False, bias=True).reshape(P, P), norm)
    # A trial is rejected below the floor, or below the initial fit's own smallest survival
    # when that is lower, so an accepted initial fit is never rejected as a starting point.
    # np.min over every unit is NaN as soon as one survival is, and min() then keeps the floor.
    survival_floor = min(T.WALK_SURVIVAL_FLOOR, np.nan if undefined else lowest)
    light = T.Evaluation(initial.psi, initial.score, initial.influence, T._EMPTY, T._EMPTY)
    evaluation = initial
    del initial
    ever = np.zeros((2, n), dtype=bool)
    steps = halvings = 0
    history = [T.standardized(evaluation)]
    converged = history[-1] <= cutoff
    while not converged and steps < max_steps and halvings < halving_cap and step >= T.MIN_STEP:
        x = evaluation.score
        weighted = metric @ x
        length = float(np.sqrt(x @ weighted))
        if length == 0.0:
            converged = True
            break
        v = weighted / length
        sizes = [step]
        while (
            len(sizes) < rungs
            and halvings + len(sizes) < halving_cap
            and sizes[-1] / 2 >= T.MIN_STEP
        ):
            sizes.append(sizes[-1] / 2)
        started = time.perf_counter()
        tried = [(np.empty((n, P)), np.empty((n, P))) for _ in sizes]
        retried = [(np.empty((n, P)), np.empty((n, P))) for _ in sizes]
        lows = [np.zeros((2, n), dtype=bool) for _ in sizes]
        still = [False for _ in sizes]
        for k, (lo, hi) in enumerate(bounds):
            state = walker.state(k)
            direction = walker.direction(k, state, v)
            for e, size in enumerate(sizes):
                p, q, low = walker.trial(k, state, direction, size, v, survival_floor, None)
                tried[e][0][lo:hi], tried[e][1][lo:hi] = p, q
                if low.any():
                    lows[e][:, lo:hi] = low
                    p, q, again = walker.trial(k, state, direction, size, v, survival_floor, low)
                    still[e] = still[e] or bool(again.any())
                retried[e][0][lo:hi], retried[e][1][lo:hi] = p, q
            del state, direction
        passes += 1
        outcomes = []
        for e in range(len(sizes)):
            if lows[e].any():
                judged = retried[e][0]
                outcomes.append(Outcome(lows[e], still[e], _length(judged, metric)))
            else:
                outcomes.append(Outcome(None, False, _length(tried[e][0], metric)))
        decision = ladder_decision(
            outcomes,
            length=length,
            steps=steps,
            halvings=halvings,
            step=step,
            initial_step=initial_step,
            max_steps=max_steps,
            halving_cap=halving_cap,
            min_step=T.MIN_STEP,
        )
        for e in decision.marked:
            ever |= lows[e]
        steps, halvings, step = decision.steps, decision.halvings, decision.step
        if decision.accepted is not None:
            e = decision.accepted
            held = lows[e].any()
            chosen = retried[e] if held else tried[e]
            evaluation = T.summarize(*chosen)
            walker.accepted.append(
                Accepted(v, sizes[e], np.flatnonzero(lows[e].ravel()) if held else None)
            )
            history.append(T.standardized(evaluation))
            converged = history[-1] <= cutoff
        del tried, retried, lows
        _log_pass("step", steps, started, walker, passes)
    dlam: FloatArray = T._EMPTY
    survival = incidence = None
    if final is not None or all_kept:
        started = time.perf_counter()
        for k, (lo, hi) in enumerate(bounds):
            state = walker.state(k)
            if final is not None:
                final(k, lo, hi, state)
            if len(bounds) == 1 and all_kept:
                dlam, survival, incidence = state.jumps, state.survival, state.incidence
            del state
        passes += 1
        _log_pass("final", steps, started, walker, passes)
    targeted = T.Evaluation(
        evaluation.psi,
        evaluation.score,
        evaluation.influence,
        T._EMPTY if survival is None else survival,
        T._EMPTY if incidence is None else incidence,
    )
    result = T.WalkResult(
        dlam,
        targeted,
        light,
        steps,
        halvings,
        converged,
        cutoff,
        step,
        float(ever.mean()),
        history,
    )
    return WalkRun(result, walker.accepted, passes, walker.replayed)


def _length(parts: FloatArray, metric: FloatArray) -> float:
    """The trial's score length in the metric, as the dense walk computes it."""
    score = parts.mean(axis=0)
    return float(np.sqrt(score @ metric @ score))


def _log_pass(kind: str, step: int, started: float, walker: _Walker, passes: int) -> None:
    if _LOG.isEnabledFor(logging.INFO):
        _LOG.info(
            "continuous-time walk pass %d (%s, after %d accepted steps): %.3f s, %d steps "
            "replayed so far, %d of %d blocks kept",
            passes,
            kind,
            step,
            time.perf_counter() - started,
            walker.replayed,
            walker.resident,
            len(walker.bounds),
        )


# ------------------------------------------------------------------- dense plan


class DensePlan:
    """Every unit in one block, on the problem's own grid: the dense walk.

    Parameters
    ----------
    problem : Problem
        The fixed inputs.
    dlam : ndarray
        ``(2, J, n, M)`` initial hazard jumps.
    """

    def __init__(self, problem: T.Problem, dlam: FloatArray) -> None:
        self._problem = problem
        self._dlam = [np.asarray(dlam, dtype=float)]
        self.n = problem.n
        self.n_targets = problem.n_targets

    @property
    def bounds(self) -> list[tuple[int, int]]:
        """One block of every unit."""
        return [(0, self.n)]

    def problem(self, k: int) -> T.Problem:  # numpy ignore=PR01
        """The problem."""
        return self._problem

    def initial(self, k: int) -> FloatArray:  # numpy ignore=PR01
        """The initial jumps, handed over once so no name here keeps them."""
        return self._dlam.pop() if self._dlam else np.empty(0)

    def overflow(  # numpy ignore=PR01
        self, k: int, units: IntArray, state: _State, v: FloatArray, step: float
    ) -> np.ndarray | None:
        """No time is dropped."""
        return None

    def dropped_weight(self, k: int) -> FloatArray | None:  # numpy ignore=PR01
        """No time is dropped."""
        return None


# ------------------------------------------------------------------- hazard sources


def _jumps(cumulative: FloatArray) -> FloatArray:
    """``(c, M)`` the jumps of cumulative hazards on the grid."""
    return np.diff(np.concatenate([np.zeros((cumulative.shape[0], 1)), cumulative], axis=1), axis=1)


@dataclass
class _FoldEntry:
    """How one fold's model predicts its units: by factors, by cells or by calling it."""

    kind: Literal["cox", "cells", "model"]
    model: Any
    base: FloatArray | None = None
    curves: tuple[FloatArray, FloatArray] | None = None


class HazardSource:
    """The initial cumulative hazards of one cause, or of censoring, for both arms of every unit.

    Each fold's fitted model predicts that fold's units.  A :class:`~.cox.CoxHazard` keeps
    one scale per unit and arm and one baseline per fold, computed on the fold's units in one
    call as the dense engine computed them; a :class:`~.cox.StratifiedNelsonAalen` keeps each
    unit's cell and each cell's curve while the cells are at most a block's units, so the
    hazards of any block are rebuilt bit for bit.  Any other learner, and a stratified learner
    with more cells, predicts a block's units when the block is built.

    Parameters
    ----------
    X : ndarray
        ``(n, p)`` the design, treatment first.
    folds : ndarray
        ``(n,)`` each unit's prediction fold.
    grid : ndarray
        ``(M,)`` the grid.
    block : int
        Units per block, the most cells a stratified learner keeps.
    """

    def __init__(self, X: FloatArray, folds: IntArray, grid: FloatArray, block: int) -> None:
        self._X = X
        self._folds = folds
        self._grid = grid
        self._block = int(block)
        n = folds.size
        self._scale = np.zeros((2, n))
        self._index = np.zeros((2, n), dtype=int)
        self._entries: dict[int, _FoldEntry] = {}

    def add(self, fold: int, predict: np.ndarray, model: Any) -> None:
        """Take the fitted ``model`` of ``fold``, which predicts the units ``predict``.

        Parameters
        ----------
        fold : int
            The fold label.
        predict : ndarray
            ``(n,)`` the fold's units.
        model : object
            The fitted hazard learner.
        """
        from .cox import CoxHazard, StratifiedNelsonAalen

        rows = np.flatnonzero(predict)
        # Exact types only: the library's clone turns a subclass back into these, and a user
        # subclass may predict otherwise.
        if type(model) is CoxHazard:
            entry = _FoldEntry("cox", model)
            for a in (0, 1):
                Xa = self._X[predict].copy()
                Xa[:, 0] = a
                scale, base = model.hazard_factors(Xa, self._grid)
                self._scale[a, rows] = scale
                entry.base = base
        elif type(model) is StratifiedNelsonAalen:
            tables = []
            for a in (0, 1):
                Xa = self._X[predict].copy()
                Xa[:, 0] = a
                index, curves = model.hazard_cells(Xa, self._grid)
                self._index[a, rows] = index
                tables.append(curves)
            if max(table.shape[0] for table in tables) <= self._block:
                entry = _FoldEntry("cells", model, curves=(tables[0], tables[1]))
            else:
                entry = _FoldEntry("model", model)
        else:
            entry = _FoldEntry("model", model)
        self._entries[int(fold)] = entry

    @property
    def calls_model(self) -> bool:
        """Whether some fold's model is called when a block is built."""
        return any(entry.kind == "model" for entry in self._entries.values())

    def cumulative(self, a: int, lo: int, hi: int) -> FloatArray:
        """``(hi - lo, M)`` the cumulative hazards of units ``lo:hi`` at arm ``a``.

        Parameters
        ----------
        a : int
            The arm.
        lo : int
            The first unit.
        hi : int
            One past the last unit.

        Returns
        -------
        ndarray
            The cumulative hazards on the grid.
        """
        labels = self._folds[lo:hi]
        out = np.empty((hi - lo, self._grid.size))
        for fold in np.unique(labels):
            entry = self._entries[int(fold)]
            within = labels == fold
            sel: slice | np.ndarray = slice(None) if within.all() else within
            if entry.kind == "cox":
                assert entry.base is not None
                out[sel] = self._scale[a, lo:hi][sel][:, None] * entry.base[None, :]
            elif entry.kind == "cells":
                assert entry.curves is not None
                out[sel] = entry.curves[a][self._index[a, lo:hi][sel]]
            else:
                Xa = self._X[lo:hi][sel].copy()
                Xa[:, 0] = a
                out[sel] = entry.model.cumulative_hazard(Xa, self._grid)
        return out

    def jumps(self, a: int, lo: int, hi: int) -> FloatArray:
        """``(hi - lo, M)`` the hazard jumps of units ``lo:hi`` at arm ``a``.

        Parameters
        ----------
        a : int
            The arm.
        lo : int
            The first unit.
        hi : int
            One past the last unit.

        Returns
        -------
        ndarray
            The jumps on the grid.
        """
        return _jumps(self.cumulative(a, lo, hi))


@dataclass
class InitialHazards:
    """The initial hazards of every cause and of censoring, rebuilt one block at a time.

    Parameters
    ----------
    causes : tuple of HazardSource
        One source per cause, in cause order.
    censoring : HazardSource
        The censoring hazard's source.
    """

    causes: tuple[HazardSource, ...]
    censoring: HazardSource

    def jumps(self, lo: int, hi: int) -> FloatArray:
        """``(2, J, hi - lo, M)`` the cause hazard jumps of units ``lo:hi``.

        Parameters
        ----------
        lo : int
            The first unit.
        hi : int
            One past the last unit.

        Returns
        -------
        ndarray
            The jumps on the full grid.
        """
        first = self.causes[0].jumps(0, lo, hi)
        out = np.empty((2, len(self.causes), *first.shape))
        out[0, 0] = first
        del first
        for a in (0, 1):
            for j, source in enumerate(self.causes):
                if a or j:
                    out[a, j] = source.jumps(a, lo, hi)
        return out

    def censoring_jumps(self, lo: int, hi: int) -> FloatArray:
        """``(2, hi - lo, M)`` the censoring hazard jumps of units ``lo:hi``.

        Parameters
        ----------
        lo : int
            The first unit.
        hi : int
            One past the last unit.

        Returns
        -------
        ndarray
            The jumps on the full grid.
        """
        return np.stack([self.censoring.jumps(a, lo, hi) for a in (0, 1)])

    @property
    def calls_model(self) -> bool:
        """Whether some source calls a model when a block is built."""
        return any(source.calls_model for source in (*self.causes, self.censoring))
