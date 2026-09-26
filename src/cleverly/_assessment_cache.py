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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils.frames import emit_frame


def _frame_payload(frame: Any) -> dict[str, Any]:
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
    return {"object": repr(value), "type": type(value).__qualname__}


def _cache_key(operation: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    normalized = {"args": _normalize(tuple(args)), "kwargs": _normalize(kwargs)}
    return f"{operation}:{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"


_RETAIN_PACKED: ContextVar[bool] = ContextVar("assessment_retain_packed", default=False)


def _cached(
    result: Any,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    compute: Callable[[], Any],
) -> Any:
    cache = result.assessment_cache
    key = _cache_key(operation, args, kwargs)
    if key not in cache:
        value = compute()
        cache[key] = _pack_cached(value, getattr(result.data, "backend", None))
    return cache[key] if _RETAIN_PACKED.get() else _unpack_cached(cache[key])
