"""Content-addressed storage for post-fit assessment answers.

An assessment operation is a pure function of the fitted artifacts and the arguments it
was given, so the answer is filed under a key built from the operation name and the
normalized arguments.  These primitives live below :mod:`cleverly.assessment` because
:mod:`cleverly.sensitivity` stores its derived estimates in the same cache and must not
import the routing layer to do it.

A dataframe is packed into :class:`_CachedFrame` before it is stored.  A stored pandas or
Polars object would pin the backend's own memory for the life of the result and would not
survive persistence; the packed form holds Python scalars and rebuilds the frame on
demand.

The cache keeps every answer unless its owner sets a byte budget (:class:`AssessmentCache`).
Under a budget it drops only answers that it can compute again bit for bit from the result's
stored artifacts.  An answer that refits, draws fresh entropy, fits a learner or runs a parallel
section is kept, because computing it again could give different numbers.
"""

from __future__ import annotations

import copy as _copy
import hashlib
import itertools
import json
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from .utils.frames import emit_frame
from .utils.memory import check_max_memory_bytes
from .utils.random import recording_marks


def _frame_payload(frame: Any) -> dict[str, Any]:
    # A sensitivity result that keeps its rows as Python values (``MissingnessTilt``,
    # ``SensitivityContour``) gives them directly.
    rows = getattr(frame, "_payload", None)
    if callable(rows):
        return dict(rows())
    if isinstance(frame, _CachedFrame):
        return dict(zip(frame.columns, frame.values, strict=True))
    if type(frame).__module__.startswith("polars"):
        return frame.to_dict(as_series=False)
    return frame.to_dict(orient="list")


@dataclass(frozen=True)
class _CachedFrame:
    columns: tuple[str, ...]
    values: tuple[tuple[Any, ...], ...]
    backend: str | None

    @classmethod
    def from_frame(cls, frame: Any, backend: str | None) -> _CachedFrame:
        payload = _frame_payload(frame)
        columns = tuple(str(column) for column in payload)
        values = tuple(
            tuple(_python_scalar(value) for value in payload[column]) for column in columns
        )
        return cls(columns, values, backend)

    def materialize(self) -> Any:
        return emit_frame(dict(zip(self.columns, self.values, strict=True)), backend=self.backend)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _pack_cached(value: Any, backend: str | None) -> Any:
    module = type(value).__module__
    if module.startswith("pandas") or module.startswith("polars"):
        return _CachedFrame.from_frame(value, backend)
    return value


def _unpack_cached(value: Any) -> Any:
    return value.materialize() if isinstance(value, _CachedFrame) else value


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return {"array": [list(value.shape), str(value.dtype), digest]}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    # A fitted object's repr is a rounded one-line summary, so two different results can
    # share it. Such an object keys by the content its ``_cache_token`` names instead.
    token = getattr(value, "_cache_token", None)
    if callable(token):
        return {"token": _normalize(token()), "type": type(value).__qualname__}
    return {"object": repr(value), "type": type(value).__qualname__}


def _cache_key(operation: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    normalized = {"args": _normalize(tuple(args)), "kwargs": _normalize(kwargs)}
    return f"{operation}:{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"


_RETAIN_PACKED: ContextVar[bool] = ContextVar("assessment_retain_packed", default=False)


def _result_backend(result: Any) -> str | None:
    """The dataframe backend a result's reports come back in.

    A result with a data container reads its backend there; a continuous-time survival
    result's container records the same backend as the result itself.
    """
    data = getattr(result, "data", None)
    if data is not None:
        return getattr(data, "backend", None)
    return getattr(result, "backend", None)


#: Decides whether a request's answer may be dropped and computed again, given the result, the
#: operation, its positional and keyword arguments.  :mod:`cleverly.assessment` installs the
#: rule, which reads the capability rows; until it does, every answer is kept.
_EVICTION_RULE: list[Callable[[Any, str, Sequence[Any], Mapping[str, Any]], bool]] = []


def _may_evict(result: Any, operation: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> bool:
    if not _EVICTION_RULE:
        return False
    try:
        return bool(_EVICTION_RULE[0](result, operation, args, kwargs))
    except Exception:
        # A request whose row cannot be resolved is kept: keeping is always correct.
        return False


def _classify(
    result: Any,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    marks: set[str],
    evictable: bool | None,
) -> bool:
    """Whether a freshly computed answer may be dropped: the rule allows it and nothing marked."""
    if marks:
        return False
    return _may_evict(result, operation, args, kwargs) if evictable is None else evictable


def _cached(
    result: Any,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    compute: Callable[[], Any],
    *,
    evictable: bool | None = None,
) -> Any:
    """Return the filed answer to a request, computing and filing it on a miss.

    ``evictable`` overrides the installed rule for a caller that knows its own answer; ``None``
    asks the rule.  Either way, an answer whose computation marked nondeterministic work in its
    own code (:mod:`cleverly.utils.random`) is never evictable.
    """
    cache = result.assessment_cache
    key = _cache_key(operation, args, kwargs)
    if key in cache:
        _touch(cache, key)
        stored = cache[key]
    else:
        with recording_marks() as marks:
            value = compute()
        allowed = _classify(result, operation, args, kwargs, marks=marks, evictable=evictable)
        stored = _pack_cached(value, _result_backend(result))
        _file(cache, key, stored, evictable=allowed)
    return stored if _RETAIN_PACKED.get() else _unpack_cached(stored)


# ------------------------------------------------------------------------------- budget


@dataclass(frozen=True)
class CacheEntry:
    """One filed answer of an :class:`AssessmentCache`.

    Parameters
    ----------
    key : str
        The key: the operation and its normalized arguments.
    operation : str
        The operation, the key's text before its first colon.
    nbytes : int
        The bytes the answer is counted at (:attr:`AssessmentCache.nbytes`).
    evictable : bool
        Whether a budget or :meth:`AssessmentCache.evict` may drop it.
    """

    key: str
    operation: str
    nbytes: int
    evictable: bool


class AssessmentCache(dict[str, Any]):
    """The assessment answers a result keeps, with an optional byte budget.

    A plain mapping from a key (the operation and its normalized arguments) to the answer.
    With no budget, the default, every answer is kept for the life of the result and saved
    with it, as a mapping without this class would.

    Some answers can be computed again exactly from the result's stored artifacts: the
    support, nuisance and score diagnostics, the stored-artifact validation, the
    omitted-confounder elements and bounds, the E-value and the combined reports that hold
    them.  Only those are *evictable*.  An answer that refits nuisance models, draws a fresh
    random seed, fits a learner or runs a parallel section is *pinned*, because computing it
    again could give different numbers; neither a budget nor :meth:`evict` removes it.  A
    dropped answer is computed again, bit for bit the same, the next time it is asked for.

    Parameters
    ----------
    *args : Any
        Passed to :class:`dict`: a mapping or an iterable of key and answer pairs.
    **kwargs : Any
        Passed to :class:`dict`: answers by key.

    Attributes
    ----------
    max_bytes : int or None
    nbytes : int
    pinned_nbytes : int

    Notes
    -----
    The byte count walks each answer: arrays by their buffers, Python objects by their sizes.
    It stops at fitted learners, estimators, data containers and results, and counts an array
    an answer shares with the fitted result, so it can exceed what dropping the answer frees;
    the budget then drops answers early rather than keep more than it says.  ``clear()``,
    ``del`` and ``pop`` remove pinned answers too: a repeated unseeded refit then draws a new
    seed, as it always has after its answer is removed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_state()

    def _reset_state(self) -> None:
        self._max_bytes: int | None = None
        #: Each evictable key, with the value it was filed with.  Any other key, or a key whose
        #: value has been replaced since, is pinned.  The value itself is held, not its id, so a
        #: freed address that a later write reuses cannot pass for the filed value.
        self._evictable: dict[str, Any] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._used: dict[str, int] = {}
        self._clock = itertools.count()

    # Persistence and copies carry the entries, the budget and the evictable keys.  Sizes and
    # recency are recomputed; a restored cache's recency starts at insertion order.

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (), self.__getstate__(), None, iter(self.items()))

    def __getstate__(self) -> tuple[int | None, tuple[str, ...]]:
        return (self._max_bytes, tuple(key for key in self._evictable if self._is_evictable(key)))

    def __setstate__(self, state: tuple[int | None, tuple[str, ...]]) -> None:
        # Pickle restores the items before the state, and the copies below build the items
        # first too, so each evictable key binds to its value at once.
        self._reset_state()
        max_bytes, evictable = state
        self._max_bytes = max_bytes
        self._evictable = {key: dict.__getitem__(self, key) for key in evictable if key in self}

    def __copy__(self) -> AssessmentCache:
        other = type(self)(self)
        other.__setstate__(self.__getstate__())
        return other

    def __deepcopy__(self, memo: dict[int, Any]) -> AssessmentCache:
        other = type(self)()
        memo[id(self)] = other
        for key, value in self.items():
            dict.__setitem__(other, key, _copy.deepcopy(value, memo))
        other.__setstate__(self.__getstate__())
        return other

    def copy(self) -> AssessmentCache:
        """Return a copy with the same entries, budget and evictable keys.

        Returns
        -------
        AssessmentCache
            The copy, sharing the answers and no bookkeeping.
        """
        return self.__copy__()

    # ------------------------------------------------------------------ bookkeeping

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._forget(key)

    def pop(self, key: str, *default: Any) -> Any:
        """Remove ``key`` and return its answer, as :meth:`dict.pop` does.

        Parameters
        ----------
        key : str
            The key to remove.
        *default : object
            The value to return when ``key`` is absent.

        Returns
        -------
        object
            The removed answer, or the default.
        """
        value = super().pop(key, *default)
        self._forget(key)
        return value

    def popitem(self) -> tuple[str, Any]:
        """Remove and return the last filed answer, as :meth:`dict.popitem` does.

        Returns
        -------
        tuple
            The key and its answer.
        """
        key, value = super().popitem()
        self._forget(key)
        return key, value

    def clear(self) -> None:
        """Remove every answer, pinned ones included; the budget stays."""
        super().clear()
        self._evictable.clear()
        self._sizes.clear()
        self._used.clear()

    def _forget(self, key: str) -> None:
        self._evictable.pop(key, None)
        self._sizes.pop(key, None)
        self._used.pop(key, None)

    def _is_evictable(self, key: str) -> bool:
        if key not in self or key not in self._evictable:
            return False
        return self._evictable[key] is dict.__getitem__(self, key)

    def _size(self, key: str) -> int:
        value = dict.__getitem__(self, key)
        recorded = self._sizes.get(key)
        if recorded is not None and recorded[0] == id(value):
            return recorded[1]
        size = _payload_nbytes(value)
        self._sizes[key] = (id(value), size)
        return size

    def _prune(self) -> None:
        for side in (self._evictable, self._sizes, self._used):
            for key in [key for key in side if key not in self]:
                del side[key]

    @property
    def max_bytes(self) -> int | None:
        """The budget on the whole cache, in bytes, or ``None`` for no budget (the default).

        A positive integer bounds :attr:`nbytes`, pinned answers included: after each filing,
        and at once when it is set, evictable answers are dropped, least recently used first,
        until the total fits or no evictable answer is left.  Pinned answers alone can exceed
        it.  It is not a memory guard: ``None`` means no budget, not the memory available now.
        """
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int | None) -> None:
        self._max_bytes = check_max_memory_bytes(value, name="max_bytes")
        self._enforce()

    @property
    def nbytes(self) -> int:
        """The bytes the filed answers are counted at."""
        self._prune()
        return sum(self._size(key) for key in self)

    @property
    def pinned_nbytes(self) -> int:
        """The part of :attr:`nbytes` in answers that are never dropped."""
        self._prune()
        return sum(self._size(key) for key in self if not self._is_evictable(key))

    def info(self) -> tuple[CacheEntry, ...]:
        """Describe each filed answer, in the order the answers were filed.

        Returns
        -------
        tuple of CacheEntry
            One record per answer: its key, operation, counted bytes and whether a budget
            may drop it.
        """
        self._prune()
        return tuple(
            CacheEntry(key, key.split(":", 1)[0], self._size(key), self._is_evictable(key))
            for key in self
        )

    def evict(self) -> int:
        """Drop every evictable answer, whatever the budget.

        Returns
        -------
        int
            The counted bytes of the dropped answers.
        """
        self._prune()
        freed = 0
        for key in [key for key in self if self._is_evictable(key)]:
            freed += self._size(key)
            del self[key]
        self._prune()
        return freed

    def _enforce(self) -> None:
        if self._max_bytes is None:
            return
        self._prune()
        total = sum(self._size(key) for key in self)
        if total <= self._max_bytes:
            return
        candidates = sorted(
            (key for key in self if self._is_evictable(key)),
            key=lambda key: self._used.get(key, -1),
        )
        for key in candidates:
            if total <= self._max_bytes:
                break
            total -= self._size(key)
            del self[key]
        self._prune()


def _touch(cache: Mapping[str, Any], key: str) -> None:
    if isinstance(cache, AssessmentCache):
        cache._used[key] = next(cache._clock)


def _is_evictable(cache: Mapping[str, Any], key: str) -> bool:
    return isinstance(cache, AssessmentCache) and cache._is_evictable(key)


def _file(cache: Any, key: str, value: Any, *, evictable: bool) -> None:
    """File ``value`` under ``key``, and keep the cache within its budget.

    An evictable answer larger than the whole budget is not kept; the caller still returns it.
    """
    cache[key] = value
    if not isinstance(cache, AssessmentCache):
        return
    if evictable:
        cache._evictable[key] = value
    else:
        cache._evictable.pop(key, None)
    cache._sizes.pop(key, None)
    _touch(cache, key)
    if cache._max_bytes is None:
        return
    if evictable and cache._size(key) > cache._max_bytes:
        # Dropping it alone keeps every older answer that fits.
        del cache[key]
        return
    cache._enforce()


# ---------------------------------------------------------------------------- sizing


def _is_stop(value: Any) -> bool:
    """An object the byte count does not enter: it belongs to the fit, not to an answer."""
    kind = type(value)
    if isinstance(value, (type, types.ModuleType, types.FunctionType, types.MethodType, Enum)):
        return True
    if callable(getattr(kind, "fit", None)):
        # Learners, estimators and anything else that fits.
        return True
    if "assessment_cache" in getattr(kind, "__dataclass_fields__", {}):
        return True
    module = kind.__module__
    return module.startswith("cleverly.data") and kind.__name__ == "CausalData"


def _payload_nbytes(value: Any) -> int:
    """The bytes one answer holds, each object counted once (see :class:`AssessmentCache`)."""
    seen: set[int] = set()
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            total += item.nbytes if item.base is None else sys.getsizeof(item)
            if item.base is not None:
                stack.append(item.base)
            continue
        if _is_stop(item):
            total += sys.getsizeof(item)
            continue
        module = type(item).__module__
        if module.startswith("pandas") and callable(getattr(item, "memory_usage", None)):
            usage = item.memory_usage(deep=True)
            total += int(usage.sum() if hasattr(usage, "sum") else usage)
            continue
        if module.startswith("polars") and callable(getattr(item, "estimated_size", None)):
            total += int(item.estimated_size())
            continue
        if module.startswith(("pandas", "polars")):
            # Scalars and dtypes (``pd.NA``, ``Timestamp``, ``CategoricalDtype``, Polars dtypes)
            # carry no buffer to report and are counted at their shallow size.
            total += sys.getsizeof(item)
            continue
        total += sys.getsizeof(item)
        if isinstance(item, (str, bytes, bytearray, int, float, complex, bool, type(None))):
            continue
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            stack.extend(item)
        else:
            if is_dataclass(item):
                stack.extend(getattr(item, f.name, None) for f in fields(item))
            state = getattr(item, "__dict__", None)
            if isinstance(state, dict):
                stack.append(state)
            for slot in getattr(type(item), "__slots__", ()) or ():
                if isinstance(slot, str) and hasattr(item, slot):
                    stack.append(getattr(item, slot))
    return total
