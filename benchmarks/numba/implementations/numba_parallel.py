"""``njit(parallel=True)``, and the one thing a parallel reduction must not do.

**Thread-local accumulation, never atomics.**  Every parallel kernel here that accumulates
into a shared array does so by giving each thread its own block of the accumulator and
reducing at the end.  The alternative -- one shared array with atomic adds -- is correct
but contends: with a skewed cluster design most rows target a handful of slots, and the
threads serialise on exactly those.  The cost is memory: ``threads x n_clusters x
n_estimands`` doubles.  Where that is too much the kernel says so and falls back to a
partition over clusters, and the benchmark reports both.

**A parallel reduction reassociates, so the tolerance is looser and stated.**  Summing
``n`` floats in ``p`` blocks and adding the blocks is not the same rounding as summing them
in order, and the difference grows with ``n``.  That is not an error to be tuned away; it
is a property of the algorithm, and each kernel's validator declares the tolerance it
needs rather than every kernel sharing one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["PARALLEL_AVAILABLE", "effective_threads", "pjit", "prange"]

F = TypeVar("F", bound=Callable[..., Any])

try:
    import numba as _numba
    from numba import prange  # noqa: F401  (re-exported for the kernels)

    PARALLEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    _numba = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]
    PARALLEL_AVAILABLE = False


def pjit(**options: Any) -> Callable[[F], F]:
    """``numba.njit(parallel=True)`` with this package's flags."""
    defaults: dict[str, Any] = {
        "cache": False,
        "fastmath": False,
        "nogil": True,
        "boundscheck": False,
        "parallel": True,
    }
    defaults.update(options)

    def decorate(function: F) -> F:
        if not PARALLEL_AVAILABLE:
            return function
        return _numba.njit(**defaults)(function)  # type: ignore[no-any-return,union-attr]

    return decorate


def effective_threads() -> int:
    """The thread count a ``prange`` kernel will actually use right now.

    Reported alongside the *requested* count in every result row.  numba silently caps a
    request at ``NUMBA_NUM_THREADS``, so a run that asked for eight on a four-core box
    would otherwise be filed as an eight-core measurement with a suspiciously poor
    efficiency.
    """
    if not PARALLEL_AVAILABLE:
        return 1
    return int(_numba.get_num_threads())  # type: ignore[union-attr]
