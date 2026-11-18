"""A cache of the arrays one longitudinal fit reads more than once, alive for that fit only.

The backward recursion reads the same observed-history designs, regimen masks, cumulative
mechanism products and clever covariates at every horizon, cause and fold of one fit.  Built
afresh at each read they cost time, and the fit's retained records then hold one copy per
horizon of arrays that are equal by construction.  Inside :func:`fit_cache` each such array is
built once, marked read-only, and handed to every reader, so the records share it.

The cache is keyed by the identity of the objects an array is computed from (the panel, the
plan, the mechanism, an earlier cached array) and holds those objects, so an identity cannot
be reused while the cache lives.  It lives in a context variable for the duration of one
draw's fit, and is dropped when the fit returns.  Outside it, or in a worker thread that does
not inherit the context, every reader computes as before, so the cache can change when an
array is computed and which object holds it, never its value.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

import numpy as np

__all__ = ["cached", "fit_cache", "frozen"]

_T = TypeVar("_T")


class _FitCache:
    """The entries of one fit: key to ``(sources, value)``, the sources held alive."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: dict[Hashable, tuple[tuple[Any, ...], Any]] = {}


_ACTIVE: ContextVar[_FitCache | None] = ContextVar("cleverly_longitudinal_fit_cache", default=None)


@contextmanager
def fit_cache() -> Iterator[None]:
    """Share the arrays a fit reads more than once until the block exits.

    A nested block reuses the outer cache, so a fit called inside another fit's block shares
    its entries only when it reads the same objects.
    """
    if _ACTIVE.get() is not None:
        yield
        return
    token = _ACTIVE.set(_FitCache())
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def frozen(value: _T) -> _T:
    """``value`` with every array in it, or in a tuple or list of it, marked read-only."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, (tuple, list)):
        for item in value:
            frozen(item)
    return value


def cached(
    name: str, sources: tuple[Any, ...], extra: Hashable, compute: Callable[[], _T]
) -> _T:
    """``compute()``, built once per active fit for the same sources and ``extra``.

    Parameters
    ----------
    name : str
        What is cached, which keeps two readers of the same sources apart.
    sources : tuple
        The objects the value is computed from, compared by identity and held by the cache.
    extra : hashable
        The plain arguments of the computation (a node, the bounds).
    compute : callable
        Builds the value.  Its arrays are marked read-only before they are shared.

    Returns
    -------
    object
        The value.
    """
    cache = _ACTIVE.get()
    if cache is None:
        return compute()
    key = (name, tuple(id(source) for source in sources), extra)
    hit = cache.entries.get(key)
    if hit is not None:
        return hit[1]  # type: ignore[no-any-return]
    value = frozen(compute())
    cache.entries[key] = (sources, value)
    return value
