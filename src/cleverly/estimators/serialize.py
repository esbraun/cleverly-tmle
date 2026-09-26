"""Whole-result persistence backed by joblib.

The stored object includes the fitted result, its cached arrays, method configuration,
and the unfitted nuisance-estimator templates retained by the estimator. Consequently a
loaded result has the same refit capabilities as the object that was saved.

An artifact is an envelope that records :data:`cleverly.__version__` beside the pickled
result. A load by a different version emits
:class:`~cleverly.exceptions.VersionMismatchWarning` and then loads the result as saved,
with no migration. Only :func:`save` and :func:`load`, and :func:`dumps` and :func:`loads`,
record and compare the version. A direct pickle or joblib dump of a result gets no check.

Joblib uses pickle internally. Loading a file can execute arbitrary code and is safe only
for artifacts from a trusted source produced in a compatible Python environment.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Any

import joblib

from .._saved_version import warn_on_version_mismatch
from .._version import __version__

__all__ = ["dumps", "load", "loads", "save"]

_COMPRESSION = 3
_FORMAT = "cleverly.result"


def _check_result(result: Any) -> None:
    from ..longitudinal import LongitudinalResult
    from .base import TMLEResult

    if not isinstance(result, (TMLEResult, LongitudinalResult)):
        raise TypeError(f"save expects a fitted causal result; got {type(result).__name__}")


def _write(result: Any, destination: Path | IO[bytes]) -> None:
    _check_result(result)
    payload = io.BytesIO()
    try:
        joblib.dump(result, payload)
    except Exception as error:
        raise TypeError(
            "the fitted result is not joblib-serializable; nuisance estimators and custom "
            "callables must be importable and pickle-compatible"
        ) from error
    envelope = {"format": _FORMAT, "cleverly_version": __version__, "payload": payload.getvalue()}
    joblib.dump(envelope, destination, compress=_COMPRESSION)


def _read(source: Path | IO[bytes]) -> Any:
    outer = joblib.load(source)
    if not (isinstance(outer, dict) and outer.get("format") == _FORMAT):
        # An artifact written before the envelope existed holds the result bare.
        _check_result(outer)
        warn_on_version_mismatch(None, stacklevel=4)
        return outer
    saved = outer.get("cleverly_version")
    warn_on_version_mismatch(saved, stacklevel=4)
    try:
        result = joblib.load(io.BytesIO(outer["payload"]))
    except Exception as error:
        error.add_note(f"saved by cleverly {saved}; this is cleverly {__version__}")
        raise
    _check_result(result)
    return result


def save(result: Any, path: str | Path) -> Path:
    """Serialize a complete fitted result to one trusted ``.joblib`` artifact.

    The artifact records :data:`cleverly.__version__`, and :func:`load` compares it.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        The fitted result to write.
    path : str or Path
        Destination file.

    Returns
    -------
    Path
        The destination path.

    Raises
    ------
    TypeError
        When ``result`` is not a fitted causal result, or when joblib cannot serialize it.
    """
    destination = Path(path)
    _write(result, destination)
    return destination


def load(path: str | Path) -> Any:
    """Load a complete result from a trusted joblib artifact.

    Never load a file from an untrusted source: joblib deserialization can execute
    arbitrary Python code.

    Parameters
    ----------
    path : str or Path
        File written by :meth:`~cleverly.estimators.TMLEResult.save`.

    Returns
    -------
    TMLEResult or LongitudinalResult
        The stored result, with the artifacts the file carried.

    Warns
    -----
    VersionMismatchWarning
        When a different cleverly version wrote the file, or when the file records no
        version. The result then loads as saved, with no migration.
    """
    return _read(Path(path))


def dumps(result: Any) -> bytes:
    """Serialize a complete fitted result to joblib bytes.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        The fitted result to write.

    Returns
    -------
    bytes
        The artifact, which records :data:`cleverly.__version__`.

    Raises
    ------
    TypeError
        When ``result`` is not a fitted causal result, or when joblib cannot serialize it.
    """
    buffer = io.BytesIO()
    _write(result, buffer)
    return buffer.getvalue()


def loads(blob: bytes) -> Any:
    """Load a complete result from trusted joblib bytes.

    Parameters
    ----------
    blob : bytes
        Bytes written by :func:`dumps`.

    Returns
    -------
    TMLEResult or LongitudinalResult
        The stored result.

    Warns
    -----
    VersionMismatchWarning
        When a different cleverly version wrote the bytes, or when they record no version.
        The result then loads as saved, with no migration.
    """
    return _read(io.BytesIO(blob))
