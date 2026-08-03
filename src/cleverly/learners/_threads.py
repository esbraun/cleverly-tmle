"""Thread limits for nuisance-model fits.

Cross-fitting and the Super Learner already provide parallelism at the level that
matters: independent fits across folds and candidates, dispatched with joblib.  The
native thread pools inside individual learners (OpenMP in LightGBM and scikit-learn's
histogram gradient boosting, BLAS in the linear models) then add a *second* layer of
parallelism over the same cores.

Nested parallelism is redundant at best.  Each joblib worker spawns as many native
threads as there are visible cores, so with ``n_jobs=4`` on four cores the process asks
for sixteen threads' worth of work; they contend rather than progress.  The nuisance
models TMLE fits are also small -- a few thousand rows -- which is the regime where
thread startup and synchronisation cost more than the arithmetic saved.  A single
gradient-boosting fit on 600 rows measured 0.24s with default threading and 0.16s
limited to one thread here: the native threads were a net loss even before any
oversubscription.

The default is therefore to run each individual fit single-threaded and to parallelise
across fits.  Call :func:`set_thread_limit` to change it -- ``None`` leaves the native
pools untouched, which is the right choice if you are fitting one very large model
rather than many small ones.

The controller is built once, and that is a performance fix
-----------------------------------------------------------

:func:`threadpoolctl.threadpool_limits` constructs a fresh ``ThreadpoolController`` on
every call, and constructing one walks **every shared object the process has loaded**
(``dl_iterate_phdr``, then a prefix match against each library's path).  That is a fixed
cost paid per entry, and this context manager is entered once per fit *and* once per
predict -- thousands of times in a DR-TMLE ``retarget``, which measured **57%** of its
runtime inside ``threadpoolctl`` before this change, against 40% of an LTMLE fit.

Reusing one controller makes the walk a once-per-process cost.  Measured on the
four-core container this repository's cloud sessions run in, with OpenBLAS and OpenMP
loaded: **0.688 ms per entry building a controller each time, 0.013 ms reusing one**
-- 53x, and the gap widens with the number of loaded libraries (1.44 ms per entry was
measured in a process that had also imported LightGBM).

What that trades away is the ability to see a pool that was loaded *after* the cached
controller was built, and this package has exactly one such case: LightGBM is imported
lazily, inside the function that builds the learner, so a controller cached at the first
fit can predate the OpenMP pool it is meant to limit.  That is handled by invalidating
explicitly at the one place that imports a backend
(:func:`cleverly.learners.library.has_lightgbm`) rather than by a heuristic that tries to
notice.  :func:`refresh_thread_pools` is the same lever for a caller who loads a native
library itself.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from typing import Any

__all__ = [
    "get_thread_limit",
    "refresh_thread_pools",
    "set_thread_limit",
    "thread_limit",
]

_LIMIT: int | None = 1

try:  # pragma: no cover - threadpoolctl ships with scikit-learn
    from threadpoolctl import ThreadpoolController as _ThreadpoolController
except ImportError:  # pragma: no cover - defensive
    _ThreadpoolController = None

#: The process-wide controller, built on first use.  Guarded by a lock because a
#: thread-parallel joblib backend enters :func:`thread_limit` from several threads at
#: once, and two threads racing to build one would pay the walk twice -- which is
#: harmless but is the cost this exists to avoid.
_CONTROLLER: Any = None
_CONTROLLER_LOCK = threading.Lock()


def set_thread_limit(limit: int | None) -> None:
    """Set the native thread limit applied around each nuisance-model fit.

    Parameters
    ----------
    limit:
        Threads per fit.  ``1`` (the default) keeps each fit single-threaded so the
        parallelism happens across folds and candidates instead.  ``None`` disables the
        limiting entirely and leaves each library's own defaults in place.
    """
    global _LIMIT
    if limit is not None and limit < 1:
        raise ValueError(f"thread limit must be a positive integer or None; got {limit}")
    _LIMIT = limit


def get_thread_limit() -> int | None:
    """The thread limit currently applied around each fit."""
    return _LIMIT


def refresh_thread_pools() -> None:
    """Discard the cached controller, so the next limit re-scans the loaded libraries.

    Call this after loading a native library whose thread pool should be limited --
    importing a learner backend, or ``dlopen``-ing something directly.  The package calls
    it itself where it imports LightGBM.  It is cheap: the cost is one deferred walk, on
    the next :func:`thread_limit` entry rather than on this call.
    """
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None


def _controller() -> Any:
    """The process-wide controller, built on first use.  ``None`` without threadpoolctl."""
    global _CONTROLLER
    if _ThreadpoolController is None:  # pragma: no cover - defensive
        return None
    if _CONTROLLER is None:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is None:
                _CONTROLLER = _ThreadpoolController()
    return _CONTROLLER


# A forked child inherits the parent's mappings, so the cached controller's handles stay
# valid there and a refresh would only pay for the walk again.  A *spawned* child
# re-imports this module and starts with no cache at all.  Both are correct without a
# hook; this one exists for the third case -- a child that goes on to load a library the
# parent never had -- where the cache is stale in the child alone and rebuilding it is
# one walk.
if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX
    os.register_at_fork(after_in_child=refresh_thread_pools)


@contextlib.contextmanager
def thread_limit(limit: int | None = -1) -> Iterator[None]:
    """Limit native thread pools for the duration of the block.

    ``limit=-1`` means "use the configured default"; pass an explicit value to override
    it for one call.

    The controller doing the limiting is process-wide and built once; see the module
    docstring for why, and :func:`refresh_thread_pools` for when that has to be undone.
    """
    resolved = _LIMIT if limit == -1 else limit
    controller = _controller()
    if resolved is None or controller is None:
        yield
        return
    with controller.limit(limits=resolved):
        yield
