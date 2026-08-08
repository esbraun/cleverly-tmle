"""Thread-pool control, core counting and memory measurement.

Every number this package reports is a number *at a stated core count*, and that is only
meaningful if the core count is enforced rather than hoped for.  Three separate pools can
claim cores in one process here:

* **numba**'s thread pool, sized by ``NUMBA_NUM_THREADS`` at import and adjustable
  downward afterwards by :func:`numba.set_num_threads`;
* **BLAS**, reached through ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` /
  ``OPENBLAS_NUM_THREADS`` / ``BLIS_NUM_THREADS`` / ``VECLIB_MAXIMUM_THREADS`` at load
  time, and by :mod:`threadpoolctl` afterwards;
* **joblib / multiprocessing workers**, which the task-parallel modes use.

Left alone, all three default to "every visible core", so a four-worker task-parallel run
on four cores asks for sixteen threads' worth of work.  That measures the scheduler.
:class:`ThreadPlan` states one intended assignment and :func:`applied` enforces it.

**The environment variables have to be set before numba and numpy are imported.**  A
process that has already imported them can lower the *effective* count at runtime but
cannot raise it past what was read at load, so :func:`bootstrap_environment` is called
from ``cli`` before any of this package's other imports, and a run at a core count above
the one the process booted with is refused rather than silently capped.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import resource
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ThreadPlan",
    "applied",
    "bootstrap_environment",
    "cpu_seconds",
    "environment_record",
    "logical_cores",
    "numba_available",
    "peak_rss_bytes",
    "physical_cores",
]

#: Every variable a numerical library on this stack reads for its thread count.
#: ``NUMBA_NUM_THREADS`` is separated out because it is the only one that must be set
#: *high* at boot and lowered per measurement -- the rest are ceilings we only ever want
#: at one.
_BLAS_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def bootstrap_environment(max_threads: int) -> None:
    """Set the load-time thread ceilings, before numpy or numba is imported.

    ``max_threads`` is the largest core count any measurement in this run will ask for.
    numba's pool is fixed at import, so it is set to that maximum here and lowered per
    measurement with :func:`numba.set_num_threads`; BLAS is left at the maximum too, so
    that the *threaded-BLAS* mode has something to thread with, and pinned to one for
    every other mode through :mod:`threadpoolctl`.

    Raising a ceiling after the import has happened does not work, which is why this is a
    separate function called from ``__main__`` rather than a context manager: by the time
    a context manager could run, the pools exist.
    """
    os.environ.setdefault("NUMBA_NUM_THREADS", str(max_threads))
    for name in _BLAS_VARIABLES:
        os.environ.setdefault(name, str(max_threads))


def logical_cores() -> int:
    """Cores this process may run on, honouring an **affinity mask**.

    Not a cgroup quota, which this docstring used to claim and the code never checked.
    ``sched_getaffinity`` reports which CPUs the scheduler will place the process on; a
    container's CFS quota caps how much of them it may consume, and leaves the mask alone.
    So inside a runner limited to two cores' worth of a sixteen-core host this returns
    sixteen, and a benchmark that sized itself from it would be reporting a thread count it
    never had.

    The number is right for the *benchmark* case it is used in, where the sweep pins its own
    thread plan and records what it asked for -- and the environment record carries the
    request beside the result precisely so a reader can tell.  For sizing a *test* run,
    :func:`tests.parallel.available_cores` is the one to use: it goes through joblib, which
    goes through loky, which reads the quota.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        return os.cpu_count() or 1


def physical_cores() -> int:
    """Distinct physical cores, or the logical count when that cannot be established.

    Reported *beside* the logical count and never in place of it.  A speed-up curve that
    flattens at the physical-core count and one that flattens at the logical count are
    different findings -- the first is the machine saturating, the second is hyper-threads
    sharing an execution unit -- and a report that knows only one number cannot tell them
    apart.
    """
    try:
        with open("/proc/cpuinfo") as handle:
            text = handle.read()
    except OSError:  # pragma: no cover - not Linux
        return logical_cores()
    pairs = set(
        zip(
            re.findall(r"physical id\s*:\s*(\d+)", text),
            re.findall(r"core id\s*:\s*(\d+)", text),
            strict=False,
        )
    )
    return len(pairs) or logical_cores()


def numba_available() -> bool:
    """Whether ``import numba`` succeeds, without paying for it twice."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class ThreadPlan:
    """One measurement's intended assignment of cores to pools.

    Attributes
    ----------
    numba_threads:
        Threads inside a ``parallel=True`` kernel.  ``1`` for every serial mode, which is
        enforced rather than assumed: a ``prange`` kernel compiled once keeps its pool,
        so a serial measurement taken after a parallel one is only serial if the count is
        set back.
    blas_threads:
        Threads inside BLAS.  ``1`` everywhere except the deliberate *threaded-BLAS*
        control mode, because a numpy reference silently using four cores would be
        compared against a numba kernel using one and reported as numba losing.
    workers:
        Process or thread workers over independent tasks -- folds, candidates, regimens,
        bootstrap chunks.  ``1`` unless the mode is task-parallel.
    """

    numba_threads: int = 1
    blas_threads: int = 1
    workers: int = 1

    @property
    def requested_cores(self) -> int:
        """Cores this plan asks the machine for: workers times threads per worker."""
        return self.workers * max(self.numba_threads, self.blas_threads)

    def oversubscribed(self, available: int) -> bool:
        return self.requested_cores > available


@contextlib.contextmanager
def applied(plan: ThreadPlan) -> Iterator[None]:
    """Enforce ``plan`` for the duration of the block.

    BLAS is limited through :mod:`threadpoolctl`, which reaches whichever library numpy
    and scikit-learn actually loaded.  numba's count is set through its own API and
    restored afterwards, because it is process-global state and a measurement that left
    it changed would silently retune every measurement after it.
    """
    from threadpoolctl import threadpool_limits

    restore: int | None = None
    if plan.numba_threads != 1 or _numba_pool_touched():
        restore = _set_numba_threads(plan.numba_threads)
    try:
        with threadpool_limits(limits=plan.blas_threads):
            yield
    finally:
        if restore is not None:
            _set_numba_threads(restore)


_NUMBA_TOUCHED = False


def _numba_pool_touched() -> bool:
    return _NUMBA_TOUCHED


def _set_numba_threads(count: int) -> int:
    """Set numba's thread count, returning the previous one.  A no-op without numba."""
    global _NUMBA_TOUCHED
    try:
        import numba
    except ImportError:
        return count
    previous = int(numba.get_num_threads())
    ceiling = int(numba.config.NUMBA_NUM_THREADS)
    if count > ceiling:
        raise ValueError(
            f"asked for {count} numba threads but the process booted with a ceiling of "
            f"{ceiling}. NUMBA_NUM_THREADS is read at import and cannot be raised "
            "afterwards; re-run with --num-cores including the larger value so the CLI "
            "sets the ceiling before importing numba"
        )
    numba.set_num_threads(count)
    _NUMBA_TOUCHED = True
    return previous


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    A high-water mark rather than a current reading, so it survives an allocation that
    has already been freed by the time the measurement ends -- which is exactly the
    allocation worth reporting: the multiplier bootstrap's ``(chunk, n)`` array exists
    only inside its loop and is the largest thing the process ever holds.

    It is also monotone over the life of the process, so a *difference* across a timed
    block is what a caller wants; :func:`peak_rss_delta` in :mod:`.timing` takes it.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(usage) * (1 if sys.platform == "darwin" else 1024)


def cpu_seconds() -> float:
    """User plus system CPU time for this process and its waited-for children.

    Reported beside wall-clock so a parallel mode that is no faster can be told apart
    from one that is no faster *and* burning four cores to do it.
    """
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime)


@dataclass(frozen=True)
class Environment:
    """Everything about the box a result must be read against."""

    git_sha: str
    git_dirty: bool
    python_version: str
    platform: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    numpy_version: str
    scipy_version: str
    numba_version: str | None
    llvmlite_version: str | None
    blas_backend: str
    blas_threading_layer: str
    threadpool_info: list[dict[str, Any]] = field(default_factory=list)
    env_variables: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_record() -> Environment:
    """Collect the environment record written to ``environment.json``.

    Not optional decoration.  Comparing a run from one machine against a run from another
    as though they were the same measurement is the commonest way a benchmark suite tells
    a lie, and the only defence is that every row carries the box it came from.
    """
    import numpy
    import scipy
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    blas = next((p for p in pools if p.get("user_api") == "blas"), {})
    try:
        import llvmlite

        import numba

        numba_version: str | None = numba.__version__
        llvmlite_version: str | None = llvmlite.__version__
    except ImportError:
        numba_version = llvmlite_version = None

    return Environment(
        git_sha=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_model=_cpu_model(),
        physical_cores=physical_cores(),
        logical_cores=logical_cores(),
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
        numba_version=numba_version,
        llvmlite_version=llvmlite_version,
        blas_backend=str(blas.get("internal_api", "unknown")),
        blas_threading_layer=str(blas.get("threading_layer", "unknown")),
        threadpool_info=[dict(p) for p in pools],
        env_variables={
            name: os.environ[name]
            for name in (*_BLAS_VARIABLES, "NUMBA_NUM_THREADS")
            if name in os.environ
        },
    )


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover - not Linux
        pass
    return platform.processor() or platform.machine()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return ""
    return out.stdout.strip()
