"""Parallel execution helpers.

Cross-fitting, bootstrap replicates and simulation studies are all
embarrassingly parallel; joblib handles them and keeps the ``n_jobs``
convention users already know from scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from joblib import Parallel, delayed

__all__ = ["map_parallel", "resolve_n_jobs"]

T = TypeVar("T")


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
    """
    items = list(payloads)
    jobs = resolve_n_jobs(n_jobs)
    if jobs == 1 or len(items) <= 1:
        return [func(*item) if isinstance(item, tuple) else func(item) for item in items]

    runner = Parallel(n_jobs=jobs, prefer=prefer)
    return list(
        runner(
            delayed(func)(*item) if isinstance(item, tuple) else delayed(func)(item)
            for item in items
        )
    )
