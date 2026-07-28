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
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

__all__ = ["get_thread_limit", "set_thread_limit", "thread_limit"]

_LIMIT: int | None = 1

try:  # pragma: no cover - threadpoolctl ships with scikit-learn
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - defensive
    _threadpool_limits = None


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


@contextlib.contextmanager
def thread_limit(limit: int | None = -1) -> Iterator[None]:
    """Limit native thread pools for the duration of the block.

    ``limit=-1`` means "use the configured default"; pass an explicit value to override
    it for one call.
    """
    resolved = _LIMIT if limit == -1 else limit
    if resolved is None or _threadpool_limits is None:
        yield
        return
    with _threadpool_limits(limits=resolved):
        yield
