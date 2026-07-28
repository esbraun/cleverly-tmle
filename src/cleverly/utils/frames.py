"""Backend-agnostic dataframe helpers.

Everything user-facing goes through :mod:`narwhals`, so pandas and polars are
both first-class without a single branch in the estimator code.  The rule the
library follows: *results come back in the backend the caller handed in*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import narwhals as nw
import numpy as np

from .._typing import Backend, FloatArray

__all__ = [
    "DEFAULT_BACKEND",
    "as_frame",
    "available_backends",
    "column_array",
    "frame_from_dict",
    "is_dataframe",
    "matrix_from_columns",
    "namespace_of",
    "resolve_backend",
]


def available_backends() -> tuple[str, ...]:
    """Which dataframe backends are importable in this environment."""
    found = []
    for name in ("pandas", "polars"):
        try:  # pragma: no cover - depends on the installed extras
            __import__(name)
        except ImportError:
            continue
        found.append(name)
    return tuple(found)


def _default_backend() -> str:
    backends = available_backends()
    if not backends:
        raise ImportError(
            "no dataframe backend found; install cleverly[pandas] or cleverly[polars]"
        )
    return backends[0]


DEFAULT_BACKEND = "pandas"


def resolve_backend(backend: Backend | str | None) -> str:
    """Pick a backend module name, falling back to whatever is installed."""
    if backend is None:
        if "pandas" in available_backends():
            return "pandas"
        return _default_backend()
    if backend not in ("pandas", "polars"):
        raise ValueError(f"backend must be 'pandas' or 'polars'; got {backend!r}")
    if backend not in available_backends():
        raise ImportError(f"backend {backend!r} requested but {backend} is not installed")
    return backend


def is_dataframe(obj: Any) -> bool:
    """True when ``obj`` is a dataframe narwhals can wrap."""
    if obj is None or isinstance(obj, (np.ndarray, Mapping)):
        return False
    try:
        nw.from_native(obj, eager_only=True)
    except (TypeError, ValueError):
        return False
    return True


def as_frame(data: Any) -> nw.DataFrame[Any]:
    """Wrap a native dataframe in a narwhals eager frame."""
    frame = nw.from_native(data, eager_only=True)
    return frame


def namespace_of(frame: nw.DataFrame[Any] | None) -> Any:
    """The native namespace (``pandas``/``polars`` module) behind ``frame``."""
    if frame is None:
        return __import__(resolve_backend(None))
    return nw.get_native_namespace(frame)


def frame_from_dict(
    data: Mapping[str, Any],
    *,
    like: nw.DataFrame[Any] | None = None,
    backend: Backend | str | None = None,
) -> Any:
    """Build a native dataframe from columns.

    Pass ``like`` to mirror an input frame's backend (the common case inside
    result objects), or ``backend`` to name one explicitly.
    """
    namespace = namespace_of(like) if like is not None else __import__(resolve_backend(backend))
    payload = {
        key: np.asarray(value) if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    return nw.from_dict(payload, backend=namespace).to_native()


def column_array(frame: nw.DataFrame[Any], name: str, *, dtype: Any = float) -> FloatArray:
    """Extract one column as a 1-d numpy array."""
    if name not in frame.columns:
        raise KeyError(f"column {name!r} not found; available columns: {list(frame.columns)}")
    values = frame[name].to_numpy()
    return np.asarray(values, dtype=dtype)


def matrix_from_columns(
    frame: nw.DataFrame[Any], names: Sequence[str], *, dtype: Any = float
) -> FloatArray:
    """Extract several columns as a 2-d ``(n, len(names))`` array."""
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"columns {missing} not found; available columns: {list(frame.columns)}")
    if not names:
        return np.empty((frame.shape[0], 0), dtype=float)
    columns = [np.asarray(frame[name].to_numpy(), dtype=dtype).reshape(-1) for name in names]
    return np.column_stack(columns)
