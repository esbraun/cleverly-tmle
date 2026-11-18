"""Parallel execution helpers.

Cross-fitting, bootstrap replicates and simulation studies are all
embarrassingly parallel; joblib handles them and keeps the ``n_jobs``
convention users already know from scikit-learn.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from joblib import Parallel, delayed

from .._warnings import merge, run_recorded
from .random import note_parallel

__all__ = ["available_cpus", "map_parallel", "resolve_n_jobs"]

T = TypeVar("T")


#: This process's control groups and the mount of the control group file system.
CGROUP = "/proc/self/cgroup"
CGROUP_ROOT = "/sys/fs/cgroup"


def _groups(cgroup: str, root: str) -> list[tuple[int, str]]:
    """The CPU control group of this process and its ancestors, with their versions.

    As :func:`cleverly.utils.memory.available_memory` reads the memory limit: a group whose
    path does not exist under the mount (a container's own group mounted as the root) is read
    at the mount root.
    """
    try:
        with open(cgroup, encoding="ascii") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    found: list[tuple[int, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            version, mount = 2, root
        elif "cpu" in controllers.split(","):
            version, mount = 1, os.path.join(root, "cpu")
            if not os.path.isdir(mount):
                mount = os.path.join(root, controllers)
        else:
            continue
        relative = path.strip("/")
        if not os.path.isdir(os.path.join(mount, relative)):
            relative = ""
        while True:
            found.append((version, os.path.join(mount, relative) if relative else mount))
            if not relative:
                break
            relative = os.path.dirname(relative)
    return found


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _cgroup_cpus(cgroup: str = CGROUP, root: str = CGROUP_ROOT) -> float | None:
    """The smallest CPU quota among this process's control groups, or ``None`` without one.

    cgroup v2 writes ``cpu.max`` as "quota period" ("max period" without a quota); cgroup v1
    writes ``cpu.cfs_quota_us`` (``-1`` without one) and ``cpu.cfs_period_us``.
    """
    smallest: float | None = None
    for version, directory in _groups(cgroup, root):
        quota: float | None = None
        try:
            if version == 2:
                text = _read(os.path.join(directory, "cpu.max"))
                if text:
                    value, period = text.split()[:2]
                    if value != "max" and int(period) > 0:
                        quota = int(value) / int(period)
            else:
                value_text = _read(os.path.join(directory, "cpu.cfs_quota_us"))
                period_text = _read(os.path.join(directory, "cpu.cfs_period_us"))
                if value_text and period_text and int(value_text) > 0 and int(period_text) > 0:
                    quota = int(value_text) / int(period_text)
        except ValueError:
            quota = None
        if quota is not None and quota > 0:
            smallest = quota if smallest is None else min(smallest, quota)
    return smallest


def available_cpus() -> int:
    """The CPUs this process may use: its affinity, capped by a control group quota.

    Returns
    -------
    int
        At least one.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpus(CGROUP, CGROUP_ROOT)
    if quota is not None:
        cpus = min(cpus, max(1, math.ceil(quota)))
    return max(1, cpus)


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalise a scikit-learn style ``n_jobs`` to an int joblib accepts."""
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs must be a positive integer, -1, or None")
    return int(n_jobs)


def map_parallel(
    func: Callable[..., T],
    payloads: Iterable[Any],
    *,
    n_jobs: int | None = 1,
    prefer: str | None = None,
) -> list[T]:
    """Apply ``func`` to each payload, in parallel when ``n_jobs != 1``.

    Payloads that are tuples are splatted as positional arguments; anything
    else is passed as a single argument.

    Each task runs under its own warning recorder and returns its records with its
    value, because a worker's warnings never reach the parent. The parent merges them
    in task order, and the serial path takes the same route, so a fit records the same
    warnings at any ``n_jobs`` (:mod:`cleverly._warnings`).
    """
    # Serial calls too: whether a call runs in parallel is a setting, not a property of the work.
    note_parallel()
    items = list(payloads)
    jobs = resolve_n_jobs(n_jobs)
    if jobs == 1 or len(items) <= 1:
        outcomes = [
            run_recorded(func, *item) if isinstance(item, tuple) else run_recorded(func, item)
            for item in items
        ]
    else:
        runner = Parallel(n_jobs=jobs, prefer=prefer)
        outcomes = list(
            runner(
                delayed(run_recorded)(func, *item)
                if isinstance(item, tuple)
                else delayed(run_recorded)(func, item)
                for item in items
            )
        )
    values = []
    for value, records in outcomes:
        merge(records)
        values.append(value)
    return values
