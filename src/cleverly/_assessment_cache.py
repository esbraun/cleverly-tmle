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


#: Generations for cached reports whose interpretation or retained schema changed.  Keep
#: the operation name itself stable so cache inspection and persisted-artifact tooling can
#: still group keys by their prefix.  The generation lives in the normalized payload,
#: which makes an older entry a harmless cache miss instead of serving a stale report.
#:
#: ``assess()`` is composed from ``validate`` and the two ``run_all`` surfaces rather than
#: cached under a fourth key.  Versioning those three keys is therefore what invalidates
#: both a direct call and the corresponding part of ``assess()``.  Every operation that
#: caches a report carrying assessment items needs a row here, which
#: ``tests/unit/test_assessment_contract.py`` checks against the cache a real fit writes.
#:
#: ``diagnostics.run_all`` moved to 10 with the pooled cross-fitted longitudinal
#: construction.  Its truncation row now counts cells of the one out-of-fold mechanism, and
#: a battery cached on an artifact from the fold-fluctuated construction reports a curve
#: that this version's fitted-bound replay gate refuses.
#:
#: ``sensitivity.run_all`` moved to 3 with RM21. Every E-value request on a fit with an
#: intermediate variable now refuses, and a battery cached on a controlled-direct-effect
#: result before that change carries a completed E-value row. The single ``evalue`` entry
#: needs no bump, because the facade selects the branch before it reads the cache.
#:
#: RM22 moved ``sensitivity.run_all`` to 4 and gave three single entries their first
#: generation, 2. The curve of an ATT or ATC ``nu^2`` gained the conditioning-share term, so a
#: cached bound, robustness value or set of elements holds the older limits and curve. A
#: plug-in bound now refuses its limits, and a cached one would serve them. ``contour`` and
#: ``benchmark`` read no curve, so their entries stay unversioned.
#:
#: A later boundary correction changes zero-variance elements and bounds, unreachable
#: robustness values, and the report rows that describe them. Their four entries move
#: again so a saved fit recomputes those values rather than serving old NaNs or 0.9999.
#:
#: RM23 moved ``diagnostics.run_all`` to 11 and ``sensitivity.run_all`` to 6. Each row now
#: reads the predicate its call raises from, so a battery cached before RM23 can carry a
#: row that says "the operation declined this request". The truncation row of an
#: incremental fit, the refute row of a fit given ``split_plan=``, of the natural-course
#: mean, or of a generated-outcome or measurement-error request the call refuses before any
#: refit, and both tilt rows of a shift, incremental, regime, MSM or ratio-only fit with
#: missing outcomes are such rows. A restored result whose refit this version refuses now
#: reads its refit rows unavailable. The ``benchmark`` row of a fit with one covariate now
#: reads unavailable, where it asked for ``covariates=``. The ``simulated_confounding`` row
#: reads unavailable for a request that names a categorical or constant benchmark
#: covariate, a natural-course mean or a zero-delta policy mean, and so does the bare row
#: of a natural-course fit. The five omitted-variable rows of an arm-indexed fit that
#: reports no mean and no linear contrast now read unavailable. No single entry moves.
#: Each call checks the row of its request before it reads the cache, so an entry for a
#: request that now refuses is never served, and a request that still runs computes what
#: it computed before. A benchmark that names every covariate raised ``DataError`` before
#: RM23, and each refused ``simulated_confounding`` request and each omitted-variable call
#: on such a fit raised ``CapabilityError``, so no entry exists for any of them.
#: ``validate`` reads no row that RM23 changed.
_CACHE_GENERATIONS: dict[str, int] = {
    "diagnostics.support": 4,
    "diagnostics.nuisance_models": 2,
    "diagnostics.run_all": 11,
    "sensitivity.elements": 3,
    "sensitivity.omitted_confounding": 3,
    "sensitivity.robustness_value": 3,
    "sensitivity.run_all": 6,
    "validate": 5,
}


def _cache_key(operation: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    normalized = {"args": _normalize(tuple(args)), "kwargs": _normalize(kwargs)}
    generation = _CACHE_GENERATIONS.get(operation)
    if generation is not None:
        normalized["cache_generation"] = generation
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
