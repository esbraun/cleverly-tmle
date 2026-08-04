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

Overlapping blocks: every one applies, and only the last one out restores
-------------------------------------------------------------------------

A ``threadpoolctl`` limiter snapshots whatever is in force when it is built and writes that
back when it is released.  That is exactly right for one thread nesting blocks, and wrong
the moment two blocks overlap::

    A enters limit(1)          snapshots 4, applies 1
    B enters limit(1)          snapshots 1, applies 1
    A exits                    writes back 4   <- B is still inside, and is no longer limited
    B exits                    writes back 1   <- the process is limited for good

Both halves are wrong: B silently loses the limit it asked for, and the setting the process
started with never comes back.  This is **not** something the controller cache introduced --
a fresh ``threadpool_limits`` per call has the identical save/restore race, and did -- and
nothing inside this package reaches it, because every
:func:`cleverly.utils.parallel.map_parallel` call leaves ``prefer=None``, so joblib
dispatches to loky *processes* and each worker has pools of its own.  But
:func:`thread_limit` is public API, and an ambient ``joblib.parallel_backend("threading")``
or a caller's own threads reach it in one step.

So the *snapshots* are refcounted rather than the applies.  ``_STACK`` is the limits
currently in force, innermost last; exactly one limiter is kept -- ``_ROOT``, built by the
outermost block, whose snapshot is the process's own setting -- and it is what the last
block out restores.  An inner exit re-*applies* the limit that is now outermost rather than
restoring its own snapshot, because under overlap a snapshot is some other block's setting
rather than the enclosing one's.

**Every block applies, including one asking for a limit already in force**, and that is the
part it is tempting to optimise away.  It cannot be, because the limits are not uniformly
process-global -- which is the half of "threadpoolctl is process-global" that is false, and
the half that decides this design.  Measured here with numpy, scikit-learn and their pools
loaded, applying ``limits=1`` in the main thread and reading the pools from another::

    openblas   1     <- process-global: the other thread sees the limit
    openblas   1
    openmp     4     <- thread-local: it does not

OpenMP's thread count is an ICV that ``omp_set_num_threads`` sets for the *calling* thread,
so a second thread that took a reference on the first thread's block instead of applying
would run its OpenMP regions unlimited -- silently, and only for the backends that use
OpenMP, which here is LightGBM and scikit-learn's histogram gradient boosting.  It is also
why "apply one outer limit from the coordinating thread" is not the fix it sounds like: the
coordinating thread's ICV is not the workers'.  Re-applying is cheap in any case -- the
walk is what the cache removed, and setting the counts on already-discovered libraries is
the 0.013 ms half.

What none of this does is make two *different* simultaneous limits both hold.  There is one
set of pools, so the alternatives were to raise (which would break the single-thread nesting
that is a legitimate pattern) or to quietly take the minimum (so a caller asking for four
gets one and is never told).  A later block's limit wins while it holds it, the enclosing
one comes back when it leaves, and the process's own setting comes back when the last block
does -- in whatever order the exits arrive.  If you need two different limits at once, you
need two processes.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
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

#: The process-wide controller, built on first use.  ``_CONTROLLER_LOCK`` guards its
#: *construction* and nothing else: two threads racing to build one would each pay the
#: walk, which is harmless but is precisely the cost this cache exists to avoid.  What
#: makes overlapping :func:`thread_limit` blocks correct is the single kept snapshot
#: below, not this lock -- see the module docstring for why the two are different problems.
_CONTROLLER: Any = None
_CONTROLLER_LOCK = threading.Lock()


@dataclass(eq=False)
class _Entry:
    """One open block, and the limit it put in force.

    ``eq=False`` so that :meth:`list.remove` takes out the block that is actually leaving:
    two blocks at the same limit would otherwise compare equal and the wrong one would go.
    """

    limit: int


#: The open blocks, innermost last.  Guarded by ``_STATE_LOCK``, which is held across the
#: bookkeeping *and* the apply -- deciding whether this is the outermost block and taking
#: its snapshot has to be atomic -- and never across the ``yield``.  Holding it for the
#: duration of a limited block would serialise every nuisance fit in the process, which is
#: the whole thing this module exists to avoid; do not "simplify" it into a lock around
#: the context manager.
_STACK: list[_Entry] = []
#: The one limiter kept, built by the outermost block.  Its snapshot is the process's own
#: setting, so it -- and nothing else -- is what the last block out restores.
_ROOT: Any = None
_STATE_LOCK = threading.Lock()


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

    It deliberately leaves ``_STACK`` and ``_ROOT`` alone.  Blocks that are already open
    still have to unwind, and the root limiter holds the process's own setting on its own
    reference rather than through the cache.  What a refresh cannot do is retroactively
    limit a pool discovered part-way through an open block -- which is what "the cost is
    one deferred walk, on the next entry" means.
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


def _before_fork() -> None:
    """Take both locks so the child cannot inherit one held by a thread it does not have."""
    _CONTROLLER_LOCK.acquire()
    _STATE_LOCK.acquire()


def _after_fork_in_parent() -> None:
    _STATE_LOCK.release()
    _CONTROLLER_LOCK.release()


def _after_fork_in_child() -> None:
    """Start the child with no open blocks, and with the locks free.

    The child inherits the *applied* limits, so from where it stands those are the
    process's own setting and its first block should snapshot from there.  Carrying the
    parent's stack across instead would record blocks no frame in the child will ever
    exit, and the limit would never be restored.
    """
    global _STACK, _ROOT
    _STACK, _ROOT = [], None
    _STATE_LOCK.release()
    _CONTROLLER_LOCK.release()
    refresh_thread_pools()


# A forked child inherits the parent's mappings, so the cached controller's handles stay
# valid there and a refresh would only pay for the walk again.  A *spawned* child
# re-imports this module and starts with no cache at all.  Both are correct without a
# hook; this one exists for the third case -- a child that goes on to load a library the
# parent never had -- where the cache is stale in the child alone and rebuilding it is
# one walk.  The ``before``/``after_in_parent`` halves are not about the cache at all:
# a fork taken while another thread held either lock would give the child a lock nobody
# can release, and the child deadlocks on its first entry.
if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_in_parent,
        after_in_child=_after_fork_in_child,
    )


def _acquire(controller: Any, resolved: int) -> _Entry:
    """Put ``resolved`` in force from *this* thread, and record the open block."""
    global _ROOT
    with _STATE_LOCK:
        # `controller.limit(...)` applies on construction rather than on `__enter__`, so
        # this is the apply -- and it happens for every block, since OpenMP's count is
        # thread-local and a block that skipped it would run unlimited.  The returned
        # limiter is kept only for the outermost one: every other snapshot records some
        # inner state rather than the process's own, and restoring from one of those is
        # the bug the module docstring opens with.
        limiter = controller.limit(limits=resolved)
        if not _STACK:
            _ROOT = limiter
        entry = _Entry(resolved)
        _STACK.append(entry)
        return entry


def _release(controller: Any, entry: _Entry) -> None:
    """Close the block, and put back whatever it was covering."""
    global _ROOT
    with _STATE_LOCK:
        if entry not in _STACK:
            # The stack was reset under this frame, which is what a forked child looks
            # like: it inherits the frame but not the blocks, and the reset already left
            # its view of the limits at what it inherited.  Nothing to put back.
            return
        _STACK.remove(entry)
        if _STACK:
            # Re-*apply* the limit that is now innermost rather than restore this block's
            # snapshot: with overlapping blocks that snapshot is another one's setting.
            controller.limit(limits=_STACK[-1].limit)
        elif _ROOT is not None:
            _ROOT.restore_original_limits()
            _ROOT = None


@contextlib.contextmanager
def thread_limit(limit: int | None = -1) -> Iterator[None]:
    """Limit native thread pools for the duration of the block.

    ``limit=-1`` means "use the configured default"; pass an explicit value to override
    it for one call.

    The controller doing the limiting is process-wide and built once; see the module
    docstring for why, for what refcounting the blocks buys when two of them overlap, and
    :func:`refresh_thread_pools` for when the cache has to be discarded.
    """
    resolved = _LIMIT if limit == -1 else limit
    controller = _controller()
    if resolved is None or controller is None:
        yield
        return
    entry = _acquire(controller, resolved)
    try:
        yield
    finally:
        _release(controller, entry)
