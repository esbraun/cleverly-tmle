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

**One thread's worth.**  The stack is thread-local and the collector times the thread that
installed it.  A joblib worker -- process or thread -- does not report; what shows up in
the parent is the wall time the worker was waited on, which is what a share of a fit
means anyway.
"""

from __future__ import annotations

import contextlib
import threading
import time as _time
from collections.abc import Iterator
from dataclasses import dataclass, field

__all__ = ["PhaseProfile", "phase", "profile_phases"]


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

    def record(self, name: str, elapsed: float, children: float) -> None:
        self.inclusive[name] = self.inclusive.get(name, 0.0) + elapsed
        self.exclusive[name] = self.exclusive.get(name, 0.0) + (elapsed - children)
        self.counts[name] = self.counts.get(name, 0) + 1

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
