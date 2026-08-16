"""How many cores this test run may actually use.

``os.cpu_count()`` answers a different question from the one every parallel setting in this
suite is asking.  It reports the cores the *machine* has; what a tier needs to know is how
many the *process* may use, which is smaller whenever there is a CPU affinity mask (a
``taskset``, a CI runner pinning a job) or a cgroup CFS quota (every container -- the
GitHub runners, and the sandbox this repository is developed in).  A run that sizes itself
from the first number asks a quota it cannot exceed for four times the parallelism it will
get, and then spends the difference on context switches.

That mistake is not hypothetical here.  ``pytest-xdist``'s ``-n auto`` prefers
``psutil.cpu_count`` and falls back to ``os.cpu_count`` -- ``psutil`` is not a dependency of
this project, so ``-n auto`` is the host count, quota or no quota.

**The detector is ``joblib.cpu_count()`` and it is not reimplemented here.**  joblib is
already a dependency, it delegates to loky, and loky is where the cgroup and affinity
handling actually lives -- its own documentation says it "takes into account additional
constraints such as Linux CFS scheduler quotas (typically set by container runtimes such as
docker) and CPU affinity".  Hand-rolling a ``/sys/fs/cgroup`` reader beside it would be a
second implementation of a thing that is hard to get right and silently wrong when it is
not: cgroup v1 and v2 differ, a quota of ``-1`` means unlimited, and a fractional quota has
to round in the direction that does not oversubscribe.

``benchmarks/numba/resources.logical_cores()`` is the older answer to the same question and
it is *narrower* than its docstring claimed -- ``os.sched_getaffinity`` is the affinity mask
and not the quota.  Its docstring now says so.  The two are kept separate because they
answer different questions rather than because one cannot be imported: the benchmark wants
the mask, since it pins its own thread plan and records what it asked for, and a test run
wants the quota, since it is the quota that decides how many workers actually fit.
"""

from __future__ import annotations

import os

import joblib

__all__ = ["CORES_ENV", "available_cores", "describe_cores"]

#: Pins the budget without pinning the process.  Above ``LOKY_MAX_CPU_COUNT`` deliberately:
#: that one also constrains joblib's own pools at runtime, so a developer who wants the test
#: harness to *think* it has two cores while leaving everything else alone needs a knob that
#: is only about the harness.
CORES_ENV = "CLEVERLY_TEST_CORES"


def available_cores() -> int:
    """Cores this process may use: the environment override, else joblib's count.

    Never zero and never negative -- a budget of zero would make every ``min()`` downstream
    silently serialise, which is the failure that looks like the code working.
    """
    override = os.environ.get(CORES_ENV)
    if override is not None:
        try:
            requested = int(override)
        except ValueError as error:
            raise ValueError(
                f"{CORES_ENV}={override!r} is not an integer; unset it or give a core count"
            ) from error
        if requested < 1:
            raise ValueError(f"{CORES_ENV}={requested} must be at least 1")
        return requested
    return max(1, int(joblib.cpu_count()))


def describe_cores() -> str:
    """One line for the header, so a surprising run says what it thought it had.

    Reports the host count beside the usable one rather than instead of it: the two being
    different is the normal state in a container and is exactly what a reader chasing an
    unexpected runtime needs to see.
    """
    usable = available_cores()
    host = os.cpu_count() or 1
    override = os.environ.get(CORES_ENV)
    source = f"{CORES_ENV}={override}" if override else "joblib.cpu_count()"
    return f"usable cores: {usable} (from {source}; host reports {host})"


#: How much the *outer* layer may exceed the core budget.  One, and the reason it is not
#: more is that this layer already balances: the fast tier is 3,491 mostly-short tests and
#: xdist keeps every worker fed, so an extra worker buys context switches.
#:
#: The *inner* layer is where deliberate oversubscription lives, and it is measured rather
#: than chosen -- see ``docs/architecture-invariants.md``. For ``n_jobs=2`` on the
#: simulation studies, three
#: paired runs on four cores, 35% faster than ``n_jobs=1``, because xdist cannot split the
#: one 6-13s study that is the critical path and the inner pool halves it.  That number is
#: for four cores and this plan does not rescale it by assumption.
OUTER_OVERSUBSCRIPTION = 1.0

#: The measured inner setting on the simulation studies, kept as a *floor* rather than a
#: formula.  Raising it with the core count would be editing a measured decision on a guess.
STUDY_JOBS = 2


def worker_count() -> int:
    """Workers for the outer (xdist) layer.

    Exported into ``PYTEST_XDIST_AUTO_NUM_WORKERS`` by the CI jobs and the noxfile so that
    ``-n auto`` stops meaning ``os.cpu_count()``.  It has to be overridden rather than left
    alone: ``pytest-xdist`` prefers ``psutil.cpu_count`` and falls back to ``os.cpu_count``,
    and ``psutil`` is not a dependency here -- so ``auto`` is the *host* count, and inside a
    quota-limited container that is several times what the job can actually use.
    """
    return max(1, int(available_cores() * OUTER_OVERSUBSCRIPTION))


if __name__ == "__main__":  # pragma: no cover - a shell entry point, not a test
    import sys

    print(worker_count() if "--workers" in sys.argv else describe_cores())
    sys.exit(0)
