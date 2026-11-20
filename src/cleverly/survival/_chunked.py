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

With ``n_jobs`` above one a pass hands its blocks to a pool of threads (:func:`for_blocks`).
Each block writes only its own rows and its own kept slot; what a block adds to a sum over
blocks, and any warning it raises, is committed in row order, so a result never depends on
``n_jobs``.  A user learner is called under a lock, and the threads leave the BLAS thread
settings alone.

X106 part C2 (bounded-memory continuous-time survival).
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final, Literal, Protocol

import numpy as np

from .._typing import FloatArray, IntArray
from .._warnings import merge, recording
from ..utils.parallel import map_parallel
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
#: Where a unit's cause hazard jumps at one arm and time sum to at least ``1 - JUMP_BOUND``,
#: :func:`bound_jumps` scales them to sum to ``1 - JUMP_BOUND``, each cause keeping its share,
#: so the initial survival there falls to about ``JUMP_BOUND`` rather than to zero or below.
#: The value is the walk's survival floor's, :data:`~._targeting.WALK_SURVIVAL_FLOOR`, but the
#: constant is its own, so a test that moves the walk's floor leaves the bound alone.
JUMP_BOUND: Final = 1e-8
#: Whether the walk runs on the event-support grid (a private switch for the benchmark).
REDUCE_GRID: Final = True
#: Held while a hazard learner other than the shipped ones predicts, so that no learner is
#: entered by two threads at once; reentrant, so a learner that itself fits on one thread
#: does not wait on itself.
_LEARNER_LOCK: Final = threading.RLock()


def for_blocks(
    count: int,
    work: Callable[[int], Any],
    commit: Callable[[int, Any], None],
    n_jobs: int = 1,
) -> None:
    """Run ``work(k)`` for every block ``k`` and ``commit(k, result)`` in block order.

    With ``n_jobs`` above one the work runs on that many threads through
    :func:`cleverly.utils.parallel.map_parallel`, ``2 * n_jobs`` blocks at a time, each block in
    a copy of the caller's context (so NumPy's error state reaches it) under a warning recorder
    of its own; the batch's warnings and results are merged and committed in block order once
    all of them have finished.  Commits always run on the calling
    thread.  The error of the earliest failing block, in block order, is raised here once its
    batch has finished.

    Parameters
    ----------
    count : int
        The blocks.
    work : callable
        Builds block ``k``'s result; it writes only that block's rows.
    commit : callable
        Takes each result in block order.
    n_jobs : int
        The threads.
    """
    if n_jobs <= 1 or count <= 1:
        for k in range(count):
            commit(k, work(k))
        return
    base = contextvars.copy_context()

    def run(k: int) -> tuple[bool, Any, tuple[Any, ...]]:
        # The block runs in the caller's context, so its package warnings would reach the
        # caller's recorder from several threads at once; a recorder of its own keeps them,
        # and they are merged in block order below.
        def recorded() -> tuple[bool, Any, tuple[Any, ...]]:
            with recording() as records:
                try:
                    value = work(k)
                except Exception as error:  # raised in block order below
                    return False, error, tuple(records)
            return True, value, tuple(records)

        return base.copy().run(recorded)

    depth = 2 * n_jobs
    for start in range(0, count, depth):
        batch = map_parallel(
            run,
            list(range(start, min(start + depth, count))),
            n_jobs=n_jobs,
            # Threads always: the blocks write into the walk's shared arrays, so a process
            # backend set around the fit (joblib.parallel_config) must not take them.
            require="sharedmem",
        )
        for k, (ok, value, records) in enumerate(batch, start):
            merge(records)
            if not ok:
                raise value
            commit(k, value)
        del batch, value, records


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

    def gap_ends(self) -> IntArray:
        """The walk-grid times that a run of dropped times precedes, in order."""
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
        self._count = threading.Lock()

    def fresh(self, k: int) -> _State:
        jumps = self.plan.initial(k)
        survival, incidence = T.curves(jumps)
        return _State(jumps, survival, incidence, 0, self.plan.problem(k))

    def keep(self, k: int, state: _State) -> _State:
        """Keep block ``k``'s state when it is a kept block; return the state to work on."""
        if k >= self.resident:
            return state
        slot = self.slots[k]
        if slot is None or slot is state or len(self.bounds) == 1:
            # The dense walk (one block) starts from the caller's own jumps, which must not be
            # written.
            self.slots[k] = state
            return state
        # Each step is written into the kept arrays.  New arrays would come from the malloc
        # arena of whichever thread replayed the block (glibc keeps an arena per thread), so
        # with threads the freed kept arrays would pile up in other threads' arenas and the
        # resident memory would creep pass after pass.
        np.copyto(slot.jumps, state.jumps)
        np.copyto(slot.survival, state.survival)
        np.copyto(slot.incidence, state.incidence)
        slot.applied = state.applied
        return slot

    def state(self, k: int) -> _State:
        """Block ``k`` after every accepted step, kept or rebuilt."""
        slot = self.slots[k]
        state = slot if slot is not None else self.fresh(k)
        target = len(self.accepted)
        if state.applied < target and slot is None:
            with self._count:
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
        return self.keep(k, state)

    def local(self, flat: IntArray, lo: int, hi: int) -> np.ndarray:
        """The ``(2, hi - lo)`` mask of the flat ``(2, n)`` indices that fall in rows ``lo:hi``."""
        n = self.plan.n
        arm, unit = np.divmod(flat, n)
        inside = (unit >= lo) & (unit < hi)
        mask = np.zeros((2, hi - lo), dtype=bool)
        mask[arm[inside], unit[inside] - lo] = True
        return mask

    def direction(self, k: int, state: _State, v: FloatArray) -> FloatArray:
        direction = T._direction(
            state.problem, _light(state.survival, state.incidence), v, state.jumps
        )
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
        floor: float | FloatArray,
        held: np.ndarray | None,
        reach: FloatArray | None,
    ) -> tuple[FloatArray, FloatArray, np.ndarray, bool]:
        """One trial on block ``k``: its parts, plug-ins, low pairs and whether it overflowed.

        ``reach`` bounds each pair's ``|direction|`` at the times the walk's grid drops
        (:meth:`reach`), ``None`` when it drops none.  The last value says the dense walk's
        ``exp`` overflowed at a dropped time, which :func:`overflow_probe` answers for when
        the block is committed.
        """
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
        overflowed = False
        if reach is not None:
            bound = step * reach
            if held is not None:
                bound = np.where(held, 0.0, bound)
            flagged = bound > OVERFLOW_BOUND
            if flagged.any():
                units = np.flatnonzero(flagged.any(axis=0))
                exact = self.plan.overflow(k, units, state, v, step)
                if exact is not None:
                    nan = np.zeros_like(flagged)
                    nan[:, units] = exact & flagged[:, units]
                    # The dense walk's tilt of a zero jump at a dropped time is inf * 0 there:
                    # the pair's jumps, survival and parts turn NaN, so its survival is not low
                    # and the trial's score is not finite.
                    rows = np.flatnonzero(nan.any(axis=0))
                    overflowed = bool(rows.size)
                    parts[rows] = np.nan
                    plugins[rows] = np.nan
                    low &= ~nan
        return parts, plugins, low, overflowed

    def reach(self, k: int, state: _State, v: FloatArray) -> FloatArray | None:
        r"""``(2, c)`` a bound on each pair's direction at the dropped times, or ``None``.

        At a dropped time :math:`u` every hazard jump is zero, so the dense direction there is
        the pair's weight times :math:`d_l(u) = W_l(u) - \sum_j H_j(u)`, with :math:`W` the
        reverse sum of the target weights and :math:`H_j` the conditional risk of
        :func:`~cleverly.survival._targeting.conditional_risk`.  :math:`H_j(u)` sums the
        increments :math:`W(u_s) S(u_{s-1}) d\Lambda_j(u_s)` after :math:`u`, which are zero
        at dropped times, and divides by :math:`S(u)`, which a dropped time repeats from the
        last kept one.  So over a run of dropped times that ends before the kept time
        :math:`g`, :math:`H_j` is one number: the reverse sum of the increments from :math:`g`
        on, over :math:`S(u_g^-)`, computed here with the dense walk's operations in its
        order (adding the dropped times' zeros changes no bit), and zero on a trailing run.
        Only :math:`W_l` moves within a run, and only for a restricted mean; a risk's is its
        value at :math:`g`.  The bound is the pair's largest dropped weight times the largest
        positive :math:`\max_{l, u} W_l(u) - \sum_j H_j` over the runs; rounding is monotone,
        so it bounds the dense increment's every term.  Only a positive increment can
        overflow.  A pair masked out of the step (the factual tilt) reads zero, and an
        undefined :math:`d` reads infinity.
        """
        dropped = self.plan.dropped_weight(k)
        if dropped is None:
            return None
        problem = state.problem
        J, Q = problem.n_causes, problem.n_rows
        c = problem.n
        expand, kept = problem.expand, problem.kept
        largest = np.zeros((2, c))
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            for a in (0, 1):
                own = v[a * J * Q : (a + 1) * J * Q].reshape(J, Q) @ problem.target_weights
                revs = np.cumsum(own[:, ::-1], axis=1)[:, ::-1]  # W, (J, M)
                trailing: float | None = None
                if expand is None:
                    # A risk's weights sit on kept times: W over a run is its value at the end.
                    ends = self.plan.gap_ends()
                    tops = revs[:, ends].max(axis=0) if ends.size else np.empty(0)
                    tail = revs
                else:
                    assert kept is not None
                    G = kept.size
                    mask = np.ones(expand.size, dtype=bool)
                    mask[kept] = False
                    # The kept time each dropped time precedes; G for the trailing run.
                    run = expand[mask] + 1
                    peaks = np.full(G + 1, -np.inf)
                    np.maximum.at(peaks, run, revs[:, mask].max(axis=0))
                    present = np.unique(run)
                    ends = present[present < G]
                    tops = peaks[ends]
                    if present.size and present[-1] == G:
                        trailing = float(peaks[G])
                    tail = np.take(revs, kept, axis=-1)
                if ends.size == 0 and trailing is None:
                    continue
                before = T._before(state.survival[a])
                shared = np.zeros((c, ends.size))
                for j in range(J):
                    # conditional_risk's increments, as the dense direction forms them.
                    x = tail[j][None, :] * before * state.jumps[a, j]
                    total = np.cumsum(x[:, ::-1], axis=1)[:, ::-1]
                    del x
                    shared += total[:, ends] / before[:, ends]
                    del total
                del before
                np.subtract(tops[None, :], shared, out=shared)
                factor = shared  # (c, runs)
                factor[np.isnan(factor)] = np.inf
                best = factor.max(axis=1) if ends.size else np.full(c, -np.inf)
                if trailing is not None:
                    # Past the last kept time nothing follows, so H is zero there.
                    best = np.maximum(best, trailing)
                largest[a] = np.maximum(best, 0.0)
                del shared, factor
            bound = dropped * largest
        bound = np.where(np.isnan(bound), np.inf, bound)
        if self.tilt == "factual":
            bound = np.where(problem.arm[None, :] == np.arange(2)[:, None], bound, 0.0)
        return bound


def overflow_probe() -> None:
    """Raise the dense walk's floating-point signals for a trial that overflowed.

    The dense walk's ``exp`` overflowed at a dropped time and its product with the zero jump
    there was invalid; NumPy's error state answers here as it did there, on the thread that
    commits the block, in block order.
    """
    probe = np.exp(np.full(1, 2.0 * OVERFLOW_BOUND))
    probe *= 0.0


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
    final: Callable[[int, int, int, Any], Callable[[], None] | None] | None = None,
    n_jobs: int = 1,
    own_floor: np.ndarray | None = None,
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
        targeted state.  It writes only the block's rows and may return a callable, which
        is called in block order on the calling thread (to add the block to a sum).
    n_jobs : int
        The threads a pass runs its blocks on (:func:`for_blocks`); it never changes a result.
    own_floor : ndarray or None
        ``(2, n)`` the (unit, arm) pairs whose initial jumps the engine bounded
        (:func:`bound_jumps`), or ``None`` when it bounded none.  A bounded pair's initial
        survival at the grid's end is about :data:`JUMP_BOUND` or less.  If it entered the
        walk's floor, the lower of ``WALK_SURVIVAL_FLOOR`` and the lowest initial survival, it
        would lower the floor of every pair.  Each such pair instead takes its own floor, the
        lower of the others' floor and its own initial survival there, and the others' floor
        reads the other pairs only.  ``None`` runs the walk with one floor for every pair.

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
    lasts: list[tuple[bool, float, FloatArray | None]] = []
    for_blocks(
        len(bounds),
        partial(_initial_rows, walker, parts, plugins, own_floor),
        partial(_collect, lasts),
        n_jobs,
    )
    undefined = any(nan for nan, _, _ in lasts)
    lowest = min((smallest for _, smallest, _ in lasts), default=np.inf)
    passes += 1
    _log_pass("initial", 0, started, walker, passes)
    initial = T.summarize(parts, plugins)
    del parts, plugins
    metric = T._metric(np.cov(initial.influence, rowvar=False, bias=True).reshape(P, P), norm)
    # A trial is rejected below the floor, or below the initial fit's own smallest survival
    # when that is lower, so an accepted initial fit is never rejected as a starting point.
    # np.min over every unit is NaN as soon as one survival is, and min() then keeps the floor.
    survival_floor: float | FloatArray = min(T.WALK_SURVIVAL_FLOOR, np.nan if undefined else lowest)
    if own_floor is not None:
        # A pair the engine bounded keeps its own floor (``own_floor`` above), so the bound
        # moves no other pair's walk; its NaN survival reads the others' floor.
        own_lasts = np.concatenate([last for _, _, last in lasts if last is not None], axis=1)
        survival_floor = np.where(
            own_floor, np.fmin(survival_floor, own_lasts), float(survival_floor)
        )
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
        for_blocks(
            len(bounds),
            partial(_trial_rows, walker, v, sizes, survival_floor, tried, retried, lows),
            partial(_trial_commit, still),
            n_jobs,
        )
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
        whole: list[_State] = []
        for_blocks(
            len(bounds),
            partial(_final_rows, walker, final, len(bounds) == 1 and all_kept),
            partial(_final_commit, whole),
            n_jobs,
        )
        if whole:
            dlam, survival, incidence = whole[0].jumps, whole[0].survival, whole[0].incidence
        del whole
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


def _collect(into: list[Any], k: int, found: Any) -> None:
    """Keep block ``k``'s result."""
    into.append(found)


def _initial_rows(
    walker: _Walker,
    parts: FloatArray,
    plugins: FloatArray,
    own_floor: np.ndarray | None,
    k: int,
) -> tuple[bool, float, FloatArray | None]:
    """Pass 0 on block ``k``: its rows of the initial fit, and whether and how low its
    survival at the grid's end is (NaN, and the smallest value otherwise), over the pairs
    without their own floor, with every pair's survival there when some pair has one."""
    lo, hi = walker.bounds[k]
    state = walker.fresh(k)
    parts[lo:hi], plugins[lo:hi] = T.evaluate_rows(
        state.problem, state.jumps, state.survival, state.incidence
    )
    last = state.survival[..., -1]
    every: FloatArray | None = None
    if own_floor is not None:
        every = last.copy()
        last = last[~own_floor[:, lo:hi]]
    nan = bool(np.isnan(last).any())
    smallest = float(np.min(last)) if last.size and not nan else np.inf
    walker.keep(k, state)
    # ``last`` is a view: deleted, so it keeps no block's survival through the walk.
    del state, last
    return nan, smallest, every


def _trial_rows(
    walker: _Walker,
    v: FloatArray,
    sizes: list[float],
    floor: float | FloatArray,
    tried: list[tuple[FloatArray, FloatArray]],
    retried: list[tuple[FloatArray, FloatArray]],
    lows: list[np.ndarray],
    k: int,
) -> tuple[list[bool], int]:
    """A step pass on block ``k``: every trial size's rows, with the retry holding low pairs.

    Returns whether each size's retry still had low pairs, and how many trials overflowed.
    """
    lo, hi = walker.bounds[k]
    if not isinstance(floor, float):
        floor = floor[:, lo:hi]
    state = walker.state(k)
    direction = walker.direction(k, state, v)
    reach = walker.reach(k, state, v)
    again_any = [False for _ in sizes]
    probes = 0
    for e, size in enumerate(sizes):
        p, q, low, over = walker.trial(k, state, direction, size, v, floor, None, reach)
        probes += over
        tried[e][0][lo:hi], tried[e][1][lo:hi] = p, q
        if low.any():
            lows[e][:, lo:hi] = low
            p, q, again, over = walker.trial(k, state, direction, size, v, floor, low, reach)
            probes += over
            again_any[e] = bool(again.any())
        retried[e][0][lo:hi], retried[e][1][lo:hi] = p, q
    del state, direction, reach
    return again_any, probes


def _trial_commit(still: list[bool], k: int, found: tuple[list[bool], int]) -> None:
    """Take block ``k``'s step pass: its overflow signals, then its still-low flags."""
    for _ in range(found[1]):
        overflow_probe()
    for e, again in enumerate(found[0]):
        still[e] = still[e] or again


def _final_rows(
    walker: _Walker,
    final: Callable[[int, int, int, Any], Callable[[], None] | None] | None,
    whole: bool,
    k: int,
) -> tuple[_State | None, Callable[[], None] | None]:
    """The final pass on block ``k``: the caller's rows, and the state when it is the walk's
    only block."""
    lo, hi = walker.bounds[k]
    state = walker.state(k)
    then = None if final is None else final(k, lo, hi, state)
    return (state if whole else None), then


def _final_commit(
    whole: list[_State], k: int, found: tuple[_State | None, Callable[[], None] | None]
) -> None:
    """Take block ``k``'s final pass in block order."""
    state, then = found
    if then is not None:
        then()
    if state is not None:
        whole.append(state)


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

    def gap_ends(self) -> IntArray:
        """No time is dropped."""
        return np.empty(0, dtype=int)


# ------------------------------------------------------------------- hazard sources


def _jumps(cumulative: FloatArray) -> FloatArray:
    """``(c, M)`` the jumps of cumulative hazards on the grid."""
    return np.diff(np.concatenate([np.zeros((cumulative.shape[0], 1)), cumulative], axis=1), axis=1)


def bound_jumps(dlam: FloatArray) -> np.ndarray | None:
    r"""Bound each cell's combined cause jump below one, in place, and say where it acted.

    Where the cause jumps of a unit at one arm and time sum to :math:`s \ge 1 - \delta`, with
    :math:`\delta` = :data:`JUMP_BOUND`, each is multiplied by :math:`(1 - \delta) / s`, so
    the cell sums to :math:`1 - \delta` and each cause keeps its share :math:`d\Lambda_j / s`.
    A Breslow jump is :math:`s_i d / \sum_{R} s_k`, below one for a unit in the risk set at
    its received arm when one event falls at the time, so on data with distinct times the bound
    acts only on extrapolated cells: a unit read at its counterfactual arm, or after it left
    follow-up.  Tied events (a bootstrap resample's copies) multiply the baseline jump.  The
    continuous-time hazard has no atoms, so at a fixed law the probability that the bound acts
    tends to zero, and the estimand, the score equation and the influence curve are unchanged.

    A cell whose sum is not finite is left as it is: the engine refuses an infinite sum, and a
    NaN jump is walked as the dense walk walks it.  When no cell reaches the bound, ``dlam`` is
    not written to.

    Parameters
    ----------
    dlam : ndarray
        ``(2, J, c, M)`` the cause hazard jumps of ``c`` units, written in place.

    Returns
    -------
    ndarray or None
        ``(2, c, M)`` the cells the bound scaled, or ``None`` when it scaled none.
    """
    bounded: np.ndarray | None = None
    for a in range(dlam.shape[0]):
        total = dlam[a].sum(axis=0)
        with np.errstate(invalid="ignore"):
            over = (total >= 1.0 - JUMP_BOUND) & np.isfinite(total)
        if not over.any():
            continue
        if bounded is None:
            bounded = np.zeros((dlam.shape[0], *total.shape), dtype=bool)
        bounded[a] = over
        dlam[a][:, over] *= (1.0 - JUMP_BOUND) / total[over]
    return bounded


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
        # Filled by the folds whose model keeps per-unit factors, allocated on first use.
        self._scale: FloatArray | None = None
        self._index: IntArray | None = None
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
            if self._scale is None:
                self._scale = np.zeros((2, self._folds.size))
            for a in (0, 1):
                Xa = self._X[predict].copy()
                Xa[:, 0] = a
                scale, base = model.hazard_factors(Xa, self._grid)
                self._scale[a, rows] = scale
                entry.base = base
        elif type(model) is StratifiedNelsonAalen:
            arms = []
            for a in (0, 1):
                Xa = self._X[predict].copy()
                Xa[:, 0] = a
                arms.append(Xa)
            # Each arm's table holds one row per cell its units fall in; a fold with more cells
            # than a block's units is predicted from the model block by block instead, so no
            # table outgrows a block's array.
            if max(len(set(model._keys(Xa))) for Xa in arms) <= self._block:
                if self._index is None:
                    self._index = np.zeros((2, self._folds.size), dtype=int)
                tables = []
                for a, Xa in enumerate(arms):
                    index, curves = model.hazard_cells(Xa, self._grid)
                    self._index[a, rows] = index
                    tables.append(curves)
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
                assert entry.base is not None and self._scale is not None
                out[sel] = self._scale[a, lo:hi][sel][:, None] * entry.base[None, :]
            elif entry.kind == "cells":
                assert entry.curves is not None and self._index is not None
                out[sel] = entry.curves[a][self._index[a, lo:hi][sel]]
            else:
                Xa = self._X[lo:hi][sel].copy()
                Xa[:, 0] = a
                with _LEARNER_LOCK:
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
        """``(2, J, hi - lo, M)`` the cause hazard jumps of units ``lo:hi``, bounded.

        Every reader of the initial hazards (the engine's checks, its replay check and the
        walk's blocks) takes them from here, so all read the same bounded jumps
        (:func:`bound_jumps`).

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
        return self.bounded_jumps(lo, hi)[0]

    def bounded_jumps(self, lo: int, hi: int) -> tuple[FloatArray, np.ndarray | None]:
        """The bounded cause hazard jumps of units ``lo:hi`` and the cells the bound scaled.

        Parameters
        ----------
        lo : int
            The first unit.
        hi : int
            One past the last unit.

        Returns
        -------
        tuple of ndarray
            ``(2, J, hi - lo, M)`` the jumps on the full grid, as :meth:`jumps` returns them,
            and ``(2, hi - lo, M)`` the scaled cells, or ``None`` when the bound scaled none.
        """
        out = self.raw_jumps(lo, hi)
        return out, bound_jumps(out)

    def raw_jumps(self, lo: int, hi: int) -> FloatArray:
        """``(2, J, hi - lo, M)`` the cause hazard jumps of units ``lo:hi`` as the models give them.

        Parameters
        ----------
        lo : int
            The first unit.
        hi : int
            One past the last unit.

        Returns
        -------
        ndarray
            The jumps on the full grid, before :func:`bound_jumps`.
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
