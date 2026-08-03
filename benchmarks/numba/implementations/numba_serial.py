"""``njit`` with the flags this package measures under, and a no-numba fallback.

Two decisions are worth stating, because both change what the numbers mean.

**``cache=False``.**  A cached kernel loads from disk on its second process instead of
compiling, which is the right production setting and the wrong benchmark setting: the
compile time is one of the things being measured, and a cache turns it into a number that
depends on whether an earlier run happened to leave a file behind.  The cold measurement
builds a *fresh* function object each time (see :func:`fresh`), so the compilation is real.

**``fastmath=False``.**  ``fastmath=True`` lets LLVM reassociate floating-point sums, which
usually buys a genuine speed-up on a reduction and always breaks the correctness gate's
tighter tolerances -- and, worse, breaks them *by an amount that depends on the thread
count*, so a parallel run and a serial run of the same kernel would disagree for a reason
that has nothing to do with the parallelism.  Where a kernel would plausibly benefit, the
benchmark carries a separate ``fastmath`` variant rather than turning it on globally, so
the trade is visible instead of assumed.

Without numba installed every decorator here becomes the identity, the kernels still run
as plain Python, and the runner marks those rows unavailable rather than reporting a
Python loop as though it were a compiled one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["NUMBA_AVAILABLE", "fresh", "njit"]

F = TypeVar("F", bound=Callable[..., Any])

try:
    import numba as _numba

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    _numba = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


def njit(**options: Any) -> Callable[[F], F]:
    """``numba.njit`` with this package's flags, or the identity without numba."""
    defaults: dict[str, Any] = {
        "cache": False,
        "fastmath": False,
        "nogil": True,
        "boundscheck": False,
    }
    defaults.update(options)

    def decorate(function: F) -> F:
        if not NUMBA_AVAILABLE:
            return function
        return _numba.njit(**defaults)(function)  # type: ignore[no-any-return,union-attr]

    return decorate


def fresh(builder: Callable[[], Callable[..., Any]]) -> Callable[[], Callable[..., Any]]:
    """Wrap a kernel *builder* so each call returns a not-yet-compiled function.

    A cold-compile measurement has to time a compilation that actually happens.  Calling
    an already-jitted function a second time with the same signature returns immediately
    from numba's in-memory cache, so timing "the first call" of a module-level kernel
    measures the first call of the *process* and nothing after it.  A builder that defines
    the function inside itself produces a new dispatcher each time, which compiles again.
    """
    return builder
