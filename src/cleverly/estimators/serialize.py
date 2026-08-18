"""Whole-result persistence backed by joblib.

The stored object includes the fitted result, its cached arrays, method configuration,
and the unfitted nuisance-estimator templates retained by the estimator. Consequently a
loaded result has the same refit capabilities as the object that was saved.

Joblib uses pickle internally. Loading a file can execute arbitrary code and is safe only
for artifacts from a trusted source produced in a compatible Python environment.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib

__all__ = ["dumps", "load", "loads", "save"]

_LEGACY_ZIP_MAGIC = b"PK\x03\x04"
_COMPRESSION = 3


def _check_result(result: Any) -> None:
    from ..longitudinal import LongitudinalResult
    from .base import TMLEResult

    if not isinstance(result, (TMLEResult, LongitudinalResult)):
        raise TypeError(f"save expects a fitted causal result; got {type(result).__name__}")


def _legacy_error() -> ValueError:
    return ValueError(
        "this is a legacy cleverly .npz result. Load it with the cleverly version that "
        "created it, then refit and save the result in the current .joblib format"
    )


def save(result: Any, path: str | Path) -> Path:
    """Serialize a complete fitted result to one trusted ``.joblib`` artifact."""
    _check_result(result)
    destination = Path(path)
    try:
        joblib.dump(result, destination, compress=_COMPRESSION)
    except Exception as error:
        raise TypeError(
            "the fitted result is not joblib-serializable; nuisance estimators and custom "
            "callables must be importable and pickle-compatible"
        ) from error
    return destination


def load(path: str | Path) -> Any:
    """Load a complete result from a trusted joblib artifact.

    Never load a file from an untrusted source: joblib deserialization can execute
    arbitrary Python code.
    """
    source = Path(path)
    with source.open("rb") as handle:
        if handle.read(4) == _LEGACY_ZIP_MAGIC:
            raise _legacy_error()
    result = joblib.load(source)
    _check_result(result)
    return result


def dumps(result: Any) -> bytes:
    """Serialize a complete fitted result to joblib bytes."""
    _check_result(result)
    buffer = io.BytesIO()
    try:
        joblib.dump(result, buffer, compress=_COMPRESSION)
    except Exception as error:
        raise TypeError(
            "the fitted result is not joblib-serializable; nuisance estimators and custom "
            "callables must be importable and pickle-compatible"
        ) from error
    return buffer.getvalue()


def loads(blob: bytes) -> Any:
    """Load a complete result from trusted joblib bytes."""
    if blob.startswith(_LEGACY_ZIP_MAGIC):
        raise _legacy_error()
    result = joblib.load(io.BytesIO(blob))
    _check_result(result)
    return result
