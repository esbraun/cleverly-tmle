"""Arrays one longitudinal fit reads more than once, shared while some reader holds them.

The backward recursion reads the same regimen masks, cumulative mechanism products, clever
covariates and observed-history designs at every horizon, cause and fold of one fit.  Built
afresh at each read they cost time, and the fit's retained records then hold one copy per
horizon of arrays that are equal by construction.  Inside :func:`fit_cache` such an array is
built once, marked read-only, and handed to every reader that asks for it while an earlier
reader still holds it, so the records share one copy.

An entry holds its value weakly.  It lives exactly as long as some record or reader holds the
value, and is dropped when the last one lets go, so no cached value outlives its last reader.
An entry is keyed by the identity of the objects the value is computed from (the panel, the
plan, the mechanism, an earlier cached value) and holds those objects while its value lives, so
an identity cannot be reused under a live entry.  Two things can therefore live longer than
without the cache: the cleaned covariate blocks of each panel object the draw reads
(:func:`held`), held until the draw returns at the size of that panel's own covariates, and
the sources of a value a record keeps (a regimen's masks behind the node masks its records
hold), which is what lets later horizons share that value.  The held blocks are dropped when
the draw returns, so they never add to what a result retains; while the draw runs they can
raise its peak by up to their size where nothing else is shared.

The cache lives in a context variable for the duration of one draw's fit and is emptied when
the fit returns.  Outside it every reader computes as before, so the cache can change when an
array is computed and which object holds it, never its value.  A thread or process that does
not see the context computes as before; one that shares it shares the cache, where a race only
computes a value twice.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import numpy as np

__all__ = ["active", "cached", "fit_cache", "frozen", "held", "in_fit_cache"]

_T = TypeVar("_T")
_P = ParamSpec("_P")


class _Entry:
    """One shared value: its sources, held, and weak references to its parts."""

    __slots__ = ("refs", "sources", "split")

    def __init__(
        self, sources: tuple[Any, ...], refs: tuple[weakref.ref[Any], ...], split: bool
    ) -> None:
        self.sources = sources
        self.refs = refs
        #: Whether the value is a tuple whose parts are referenced one by one.
        self.split = split

    def value(self) -> Any:
        """The value while every part lives, else ``None``."""
        parts = tuple(ref() for ref in self.refs)
        if any(part is None for part in parts):
            return None
        return parts if self.split else parts[0]


class _FitCache:
    """The entries of one fit: weak entries by key, and the few values held strongly."""

    __slots__ = ("__weakref__", "entries", "strong")

    def __init__(self) -> None:
        self.entries: dict[Hashable, _Entry] = {}
        self.strong: dict[Hashable, tuple[tuple[Any, ...], Any]] = {}


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
    cache = _FitCache()
    token = _ACTIVE.set(cache)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        cache.entries.clear()
        cache.strong.clear()


def in_fit_cache(method: Callable[_P, _T]) -> Callable[_P, _T]:
    """``method`` run inside :func:`fit_cache`, with its signature kept."""

    @wraps(method)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with fit_cache():
            return method(*args, **kwargs)

    return run


def active() -> bool:
    """Whether a fit's cache is active in this context."""
    return _ACTIVE.get() is not None


def frozen(value: _T) -> _T:
    """``value`` with every array in it marked read-only.

    Arrays inside a tuple or list, and the array fields of a dataclass, are marked too.
    """
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, (tuple, list)):
        for item in value:
            frozen(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            field_value = getattr(value, item.name)
            if isinstance(field_value, np.ndarray):
                field_value.flags.writeable = False
    return value


def _key(name: str, sources: tuple[Any, ...], extra: Hashable) -> Hashable:
    return (name, tuple(id(source) for source in sources), extra)


def _drop(cache_ref: weakref.ref[_FitCache], key: Hashable, ref: weakref.ref[Any]) -> None:
    """Remove ``key`` when its value dies, unless a newer entry has replaced it."""
    cache = cache_ref()
    if cache is None:
        return
    entry = cache.entries.get(key)
    if entry is not None and any(item is ref for item in entry.refs):
        del cache.entries[key]


def cached(name: str, sources: tuple[Any, ...], extra: Hashable, compute: Callable[[], _T]) -> _T:
    """``compute()``, shared within an active fit while some reader holds the value.

    Parameters
    ----------
    name : str
        What is cached, which keeps two readers of the same sources apart.
    sources : tuple
        The objects the value is computed from, compared by identity and held by the cache
        while the value lives.
    extra : hashable
        The plain arguments of the computation (a node, the bounds).
    compute : callable
        Builds the value: an array, a dataclass, or a tuple of arrays.  Its arrays are marked
        read-only before they are shared.

    Returns
    -------
    object
        The value.
    """
    cache = _ACTIVE.get()
    if cache is None:
        return compute()
    key = _key(name, sources, extra)
    entry = cache.entries.get(key)
    if entry is not None:
        hit = entry.value()
        if hit is not None:
            # A tuple value comes back as a new tuple of the same arrays.
            return hit
    value = frozen(compute())
    split = isinstance(value, tuple)
    parts: tuple[Any, ...] = value if isinstance(value, tuple) else (value,)
    cache_ref = weakref.ref(cache)

    def callback(dead: weakref.ref[Any]) -> None:
        _drop(cache_ref, key, dead)

    try:
        refs = tuple(weakref.ref(part, callback) for part in parts)
    except TypeError:
        return value  # a value that cannot be referenced weakly is not shared
    cache.entries[key] = _Entry(sources, refs, split)
    return value


def held(name: str, sources: tuple[Any, ...], extra: Hashable, compute: Callable[[], _T]) -> _T:
    """``compute()``, built once per active fit and held until the fit returns.

    Only for values whose size is bounded by the panel's own (its cleaned covariate blocks,
    one copy per panel object read in the draw), since the cache keeps them alive whether or
    not a reader holds them.

    Parameters
    ----------
    name : str
        What is cached.
    sources : tuple
        The objects the value is computed from, compared by identity and held by the cache.
    extra : hashable
        The plain arguments of the computation.
    compute : callable
        Builds the value.  Its arrays are marked read-only.

    Returns
    -------
    object
        The value.
    """
    cache = _ACTIVE.get()
    if cache is None:
        return compute()
    key = _key(name, sources, extra)
    hit = cache.strong.get(key)
    if hit is not None:
        return hit[1]
    value = frozen(compute())
    cache.strong[key] = (sources, value)
    return value
