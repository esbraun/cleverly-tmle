"""Wall-clock timing of a fit's phases, off by default.

Where a longitudinal fit's time goes was previously answered by running it under
``cProfile`` and bucketing the lines by *filename* -- anything under ``sklearn``,
``lightgbm``, ``joblib``, ``scipy`` or ``threadpoolctl`` counted as "the learner", and the
remainder as "the package".  That is a crude instrument and it was measurably wrong in
both directions: it charges a profiler's per-call overhead to whichever code makes the
most calls, which is exactly the code being investigated, and it cannot separate two
package-owned phases from each other at all.

So the phases say what they are.  ``phase("mask_construction")`` around the scan,
``phase("outcome_learner_fit")`` around the cross-fitting, and the shares are read off
rather than inferred from a call graph.

**Disabled by default and cheap when it is.**  With no collector installed, ``phase``
returns a shared do-nothing object -- one module-global read and a pair of empty
``__enter__``/``__exit__`` calls, which is nanoseconds against a node's milliseconds.  No
timing is taken, no name is interned, nothing is appended.

**Nesting is explicit.**  A phase inside another is timed inside it, so the inclusive
totals overlap and do not sum to the fit.  :attr:`PhaseProfile.exclusive` subtracts what
the children took, and it is that column which sums to the whole; both are reported,
because "the recursion is 60% of the fit" and "the recursion's own arithmetic is 4% of the
fit" are different sentences and a report that carries one of them invites the other.

**One thread's worth, plus what the workers report.**  The stack is thread-local and the
collector times the thread that installed it, so what a joblib worker does shows up in the
parent as the wall time the worker was waited on -- which is what a *share* of a fit means.
That is the right denominator and the wrong breakdown: a fit that runs its whole recursion
in ``K`` worker processes would report the fan-out as one opaque phase and nothing inside
it.  So a worker may open its own :func:`collect_phases` and hand the profile back, and the
parent merges it into :attr:`PhaseProfile.workers` with :func:`merge_worker_phases`.

**The merge is a child and not an addition.**  ``K`` folds running at once accumulate more
processor time than the parent spent waiting for them, so adding worker time to the
parent's own totals would put ``sum(exclusive)`` above ``total_seconds`` and turn every
``share`` into a mixture of wall time and processor time.  ``workers`` is therefore a
second profile, read beside the parent's rather than inside it, and its ``total_seconds``
is the sum of the workers' spans rather than any elapsed interval.
"""

from __future__ import annotations

import contextlib
import threading
import time as _time
from collections.abc import Iterator
from dataclasses import dataclass, field

__all__ = [
    "PhaseProfile",
    "collect_phases",
    "merge_worker_phases",
    "phase",
    "profile_phases",
    "profiling",
]


class _Disabled:
    """The object ``phase`` returns when nothing is collecting.

    A class with two empty methods rather than a ``@contextmanager`` generator: entering a
    generator-based context manager allocates a generator and runs it to its first yield,
    which is roughly a microsecond, and this is entered once per node per regimen per
    horizon.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


_DISABLED = _Disabled()


@dataclass
class PhaseProfile:
    """What each phase cost, and how often it ran.

    ``inclusive`` counts a phase's whole span including any phase nested inside it;
    ``exclusive`` counts the part that was not inside a child.  ``counts`` is how many
    times each phase was entered, which is what tells a per-node cost from a per-fit one.
    """

    inclusive: dict[str, float] = field(default_factory=dict)
    exclusive: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    total_seconds: float = 0.0
    #: What the parallel workers reported, merged.  ``None`` until one does.  Its times are
    #: processor time across the workers and its ``total_seconds`` is their sum, so it is
    #: not comparable with this profile's wall clock and is never added into it.
    workers: PhaseProfile | None = None

    def record(self, name: str, elapsed: float, children: float) -> None:
        self.inclusive[name] = self.inclusive.get(name, 0.0) + elapsed
        self.exclusive[name] = self.exclusive.get(name, 0.0) + (elapsed - children)
        self.counts[name] = self.counts.get(name, 0) + 1

    def merge(self, other: PhaseProfile) -> None:
        """Add ``other``'s phases into this one, name by name.

        Used to fold one worker's profile into the accumulated ``workers`` child.  Not for
        folding a worker into a parent: see the module docstring for why that would break
        the wall-clock denominator every ``share`` is taken against.
        """
        for name, value in other.inclusive.items():
            self.inclusive[name] = self.inclusive.get(name, 0.0) + value
        for name, value in other.exclusive.items():
            self.exclusive[name] = self.exclusive.get(name, 0.0) + value
        for name, count in other.counts.items():
            self.counts[name] = self.counts.get(name, 0) + count
        self.total_seconds += other.total_seconds

    @property
    def total_counts(self) -> dict[str, int]:
        """How often each phase ran, here and in any parallel worker.

        The *counts* are the one column that means the same thing on both sides of the
        worker boundary -- a phase entered is a phase entered, whichever process did it --
        so they are the column a caller can sum.  The times are not, and there is
        deliberately no ``total_seconds`` beside this.
        """
        merged = dict(self.counts)
        if self.workers is not None:
            for name, count in self.workers.counts.items():
                merged[name] = merged.get(name, 0) + count
        return merged

    def share(self, name: str) -> float:
        """``name``'s exclusive time as a fraction of the whole profiled region."""
        if not self.total_seconds:
            return float("nan")
        return self.exclusive.get(name, 0.0) / self.total_seconds

    def summary(self) -> str:
        """A table, ordered by exclusive cost.  Deliberately plain text.

        This is a diagnostic for a benchmark and a developer, not a result object: giving
        it a dataframe would put it on the narwhals contract the package makes about
        *reported* quantities, and a timing is not one.
        """
        lines = [
            f"{'phase':<28} {'calls':>7} {'exclusive s':>12} {'share':>7} {'inclusive s':>12}",
            "-" * 70,
        ]
        for name in sorted(self.exclusive, key=lambda key: -self.exclusive[key]):
            lines.append(
                f"{name:<28} {self.counts[name]:>7d} {self.exclusive[name]:>12.4f} "
                f"{self.share(name):>6.1%} {self.inclusive[name]:>12.4f}"
            )
        accounted = sum(self.exclusive.values())
        lines.append("-" * 70)
        lines.append(
            f"{'accounted for':<28} {'':>7} {accounted:>12.4f} "
            f"{accounted / self.total_seconds if self.total_seconds else float('nan'):>6.1%} "
            f"{self.total_seconds:>12.4f}"
        )
        workers = self.workers
        if workers is not None:
            # Indented and headed rather than appended to the table above, because these
            # are processor seconds across the workers and those are wall seconds in the
            # parent. One table would invite the reader to add them up.
            lines.append("")
            lines.append("in parallel workers (processor time, not wall time):")
            for name in sorted(workers.exclusive, key=lambda key: -workers.exclusive[key]):
                lines.append(
                    f"  {name:<26} {workers.counts[name]:>7d} "
                    f"{workers.exclusive[name]:>12.4f} {'':>7} "
                    f"{workers.inclusive[name]:>12.4f}"
                )
        return "\n".join(lines)


class _State(threading.local):
    collector: PhaseProfile | None = None
    stack: list[float] | None = None


_STATE = _State()


class _Span:
    """One entered phase.  Holds its start and what its children consumed."""

    __slots__ = ("children", "name", "started")

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = 0.0
        self.children = 0.0

    def __enter__(self) -> _Span:
        stack = _STATE.stack
        assert stack is not None
        stack.append(0.0)
        self.started = _time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed = _time.perf_counter() - self.started
        stack = _STATE.stack
        assert stack is not None
        self.children = stack.pop()
        if stack:
            stack[-1] += elapsed
        collector = _STATE.collector
        if collector is not None:
            collector.record(self.name, elapsed, self.children)


def phase(name: str) -> _Span | _Disabled:
    """Time ``name`` for the duration of the block, if anything is collecting."""
    if _STATE.collector is None:
        return _DISABLED
    return _Span(name)


def profiling() -> bool:
    """Whether anything is collecting phases on this thread.

    Read by a caller about to fan work out to workers, so it can decide whether the
    workers should collect at all.  A worker that profiles when nobody asked pays the
    timing cost for a profile nothing reads.
    """
    return _STATE.collector is not None


@contextlib.contextmanager
def collect_phases(enabled: bool) -> Iterator[PhaseProfile | None]:
    """Collect this worker's phases, and yield the profile to hand back to the parent.

    ``enabled`` is the parent's :func:`profiling` answer, read *before* the fan-out and
    carried into the worker.  A worker process starts with no collector of its own, so it
    cannot tell whether anybody asked; without the flag it would time every phase of every
    fold on every fit and hand the result to nobody.

    Yields ``None`` when disabled, and also when a collector is already installed here --
    which means the caller ran us inline rather than in a worker, so its own collector
    already sees these phases and a second one would steal them.  Unlike
    :func:`profile_phases` this does not *refuse* to nest, because a worker's collector is
    its own process's and the parent's is untouched.

    Returning the profile rather than writing to a shared one is the whole point: with the
    loky backend there is no shared anything, which is why the phases inside a fold
    recursion used to vanish whenever ``n_jobs`` rose above one.
    """
    if not enabled or _STATE.collector is not None:
        yield None
        return
    profile = PhaseProfile()
    _STATE.collector = profile
    _STATE.stack = []
    started = _time.perf_counter()
    try:
        yield profile
    finally:
        profile.total_seconds = _time.perf_counter() - started
        _STATE.collector = None
        _STATE.stack = None


def merge_worker_phases(profile: PhaseProfile | None) -> None:
    """Fold one worker's returned profile into the collecting parent's ``workers`` child.

    A no-op when ``profile`` is ``None`` or nothing is collecting here, so a caller does
    not have to branch on either.
    """
    collector = _STATE.collector
    if profile is None or collector is None:
        return
    if collector.workers is None:
        collector.workers = PhaseProfile()
    collector.workers.merge(profile)


@contextlib.contextmanager
def profile_phases() -> Iterator[PhaseProfile]:
    """Collect phase timings for the duration of the block.

    Nesting two of these is refused rather than silently merged: the inner one would
    steal the outer one's phases, and a share computed against the wrong denominator is
    worse than no share.
    """
    if _STATE.collector is not None:
        raise RuntimeError("a phase profile is already being collected on this thread")
    profile = PhaseProfile()
    _STATE.collector = profile
    _STATE.stack = []
    started = _time.perf_counter()
    try:
        yield profile
    finally:
        profile.total_seconds = _time.perf_counter() - started
        _STATE.collector = None
        _STATE.stack = None
