"""Backend-agnostic dataframe helpers.

Everything user-facing goes through :mod:`narwhals`, so pandas and polars are
both first-class without a single branch in the estimator code.  The rule the
library follows: *results come back in the backend the caller handed in*.

**That promise is about the library, not about the dtype backend.**  A frame read
with ``dtype_backend="pyarrow"`` comes back as pandas, but as *numpy-backed*
pandas: :func:`frame_from_dict` builds from numpy arrays through
``narwhals.from_dict``, which has no ``dtype_backend`` knob.  Every result this
package emits is a dense float column with no nulls in it, so there is nothing an
arrow dtype would carry that a float64 one does not -- but a caller who hands in
``ArrowDtype`` and expects it back would otherwise be surprised silently.

Arrow-backed *input* is a first-class, tested configuration all the same; what
makes it one is that the numeric roles are cast inside narwhals on the way in
(:func:`column_array`) rather than left to whichever numpy dtype the backend
happens to produce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import narwhals as nw
import numpy as np

from .._typing import Backend, FloatArray

__all__ = [
    "as_frame",
    "available_backends",
    "backend_of",
    "column_array",
    "emit_frame",
    "frame_from_dict",
    "has_nulls",
    "is_dataframe",
    "matrix_from_columns",
    "resolve_backend",
]

#: Every eager backend narwhals can both read and build, in the order a caller who
#: named none of them gets one.  ``pyarrow`` is here because :func:`is_dataframe`
#: already admits a ``pyarrow.Table`` -- narwhals takes it, and results were already
#: coming back as tables -- so the choice is between declaring that and leaving an
#: undeclared third backend half-working.
_BACKENDS: tuple[str, ...] = ("pandas", "polars", "pyarrow")


def available_backends() -> tuple[str, ...]:
    """Which dataframe backends are importable in this environment."""
    found = []
    for name in _BACKENDS:
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


def resolve_backend(backend: Backend | str | None) -> str:
    """Pick a backend module name, falling back to whatever is installed."""
    if backend is None:
        if "pandas" in available_backends():
            return "pandas"
        return _default_backend()
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}; got {backend!r}")
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


def backend_of(frame: nw.DataFrame[Any]) -> str:
    """The name of the backend behind ``frame`` -- ``"pandas"``, ``"polars"``, ...

    A *name* rather than the module, because this is what the containers keep.  Holding
    the input frame itself would pin it in memory for the life of every result derived
    from the fit, and holding the module would not survive :func:`cleverly.load`.
    """
    return str(nw.get_native_namespace(frame).__name__)


def frame_from_dict(
    data: Mapping[str, Any],
    *,
    backend: Backend | str | None = None,
) -> Any:
    """Build a native dataframe from columns, in ``backend`` or the default one."""
    namespace = __import__(resolve_backend(backend))
    return nw.from_dict(dict(data), backend=namespace).to_native()


def emit_frame(
    payload: Mapping[str, Any],
    data: Any = None,
    *,
    backend: Backend | str | None = None,
) -> Any:
    """A result table, in the backend the fit's data arrived in.

    The tail every ``to_frame(data=None)`` in the package ends with.  Two ways to say
    which backend that is, and a report should carry the second: ``data`` is a
    container exposing ``frame_like`` -- a :class:`~cleverly.data.CausalData` or a
    :class:`~cleverly.longitudinal.LongitudinalData` -- while ``backend`` is the name
    one recorded, which is what every report class now keeps for itself.

    Keeping the name is what makes "pandas in, pandas out" hold for a diagnostic and
    not only for an estimate.  It used to hold for neither: ``data`` defaulted to
    ``None`` and *nothing inside the package ever passed it*, so a polars fit's
    ``validation.nuisance().to_frame()`` came back as pandas -- while
    :meth:`cleverly.interventions.RegimeSupport.to_frame` documented the opposite.

    The ``hasattr`` is deliberate rather than defensive: several of these methods are
    reachable with a plain result object in ``data``, and one of the seven call sites
    this replaces already guarded for it while the other six would have raised.
    """
    if data is not None and hasattr(data, "frame_like"):
        return data.frame_like(payload)
    return frame_from_dict(payload, backend=backend)


def has_nulls(frame: nw.DataFrame[Any], name: str) -> bool:
    """Whether column ``name`` holds a null, asked through narwhals.

    Asked *before* the column becomes numpy, because that is the only place the
    question has one answer.  A null survives into numpy as ``nan`` from a float
    column, as ``pd.NA`` inside an ``object`` array from a nullable or arrow-backed
    one, and as ``None`` from an object column -- and only the first is something a
    downstream ``np.isfinite`` check can see.
    """
    return bool(frame[name].is_null().any())


def _require(frame: nw.DataFrame[Any], names: Sequence[str]) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"columns {missing} not found; available columns: {list(frame.columns)}")


def column_array(frame: nw.DataFrame[Any], name: str, *, dtype: Any = float) -> FloatArray:
    """Extract one column as a 1-d numpy array.

    A float ``dtype`` is applied as a narwhals ``cast`` rather than as a numpy one.
    That is what makes an arrow-backed or nullable column correct by construction:
    ``Series.to_numpy`` picks its dtype from the values a column happens to hold --
    ``object`` carrying ``pd.NA`` as soon as one of them is null -- whereas a declared
    cast to ``Float64`` maps a null to ``nan``, which is exactly what the numpy-backed
    path has always produced and what the validation layer knows how to reject.
    """
    _require(frame, [name])
    series = frame[name]
    if dtype is float or dtype is np.float64:
        return np.asarray(series.cast(nw.Float64).to_numpy(), dtype=float)
    return np.asarray(series.to_numpy(), dtype=dtype)


def matrix_from_columns(
    frame: nw.DataFrame[Any], names: Sequence[str], *, dtype: Any = float
) -> FloatArray:
    """Extract several columns as a 2-d ``(n, len(names))`` array."""
    _require(frame, names)
    if not names:
        return np.empty((frame.shape[0], 0), dtype=float)
    columns = [column_array(frame, name, dtype=dtype).reshape(-1) for name in names]
    return np.column_stack(columns)
