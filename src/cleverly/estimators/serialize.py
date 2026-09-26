"""Whole-result persistence backed by joblib.

The stored object includes the fitted result, its cached arrays, method configuration,
and the unfitted nuisance-estimator templates retained by the estimator. Consequently a
loaded result has the same refit capabilities as the object that was saved.

An artifact is one gzip stream at compression level 3 that holds two joblib pickles in
sequence. The first is a small header that records :data:`cleverly.__version__`. The
second is the result. Both pickles stream through the compressor, so an artifact never
holds a second uncompressed copy of the result in memory. :func:`load` reads the header
first. On a different version it emits
:class:`~cleverly.exceptions.VersionMismatchWarning` before it unpickles the result, and
then it loads the result as saved, with no migration. It then reads the stream to its end,
so gzip checks the stream's CRC and length, and it refuses a stream with data after the
result. :func:`save` writes a temporary file beside the destination and replaces the
destination only when the write succeeds. Only :func:`save` and :func:`load`,
and :func:`dumps` and :func:`loads`, record and compare the version. A direct pickle or
joblib dump of a result gets no check.

Joblib uses pickle internally. Loading a file can execute arbitrary code and is safe only
for artifacts from a trusted source produced in a compatible Python environment.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import os
import uuid
from pathlib import Path
from typing import IO, Any

import joblib

from .._saved_version import warn_on_version_mismatch
from .._version import __version__

__all__ = ["dumps", "load", "loads", "save"]

_COMPRESSION = 3
_FORMAT = "cleverly.result"
_GZIP_MAGIC = b"\x1f\x8b"


def _check_result(result: Any, *, operation: str) -> None:
    from ..longitudinal import LongitudinalResult
    from .base import TMLEResult

    if not isinstance(result, (TMLEResult, LongitudinalResult)):
        raise TypeError(f"{operation} expects a fitted causal result; got {type(result).__name__}")


def _write(result: Any, destination: IO[bytes]) -> None:
    """Stream the version header and then ``result`` into one gzip stream."""
    header = {"format": _FORMAT, "cleverly_version": __version__}
    # ``filename=""`` and ``mtime=0`` keep the path and the clock out of the gzip header.
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=destination, compresslevel=_COMPRESSION, mtime=0
    ) as stream:
        joblib.dump(header, stream)
        try:
            joblib.dump(result, stream)
        except Exception as error:
            raise TypeError(
                "the fitted result is not joblib-serializable; nuisance estimators and custom "
                "callables must be importable and pickle-compatible"
            ) from error


def _is_header(value: Any) -> bool:
    return isinstance(value, dict) and value.get("format") == _FORMAT


def _drain(stream: Any) -> None:
    """Read ``stream`` to its end, which makes gzip check the CRC and length trailer.

    joblib stops at the pickle's last opcode, before the trailer, so without this read a
    flipped trailer byte, a cut trailer, or a second gzip member would load silently.
    """
    if stream.read(1):
        raise ValueError("the artifact holds data after the result")


def _payload_note(saved: Any) -> str:
    """The note a failed read of the result carries, after a readable header."""
    if saved == __version__:
        return (
            f"the artifact records this cleverly version ({__version__}), so it can be "
            "truncated or corrupt, or a development snapshot that shares the version string "
            "can have written it"
        )
    return f"saved by cleverly {saved}; this is cleverly {__version__}"


def _read(source: IO[bytes], *, operation: str) -> Any:
    """Read the header, warn on a version mismatch, and only then unpickle the result.

    ``stacklevel=4`` skips :func:`warn_on_version_mismatch`, this function, and
    :func:`load` or :func:`loads`, so the warning names the caller's load call.
    """
    compressed = source.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC
    source.seek(0)
    with contextlib.ExitStack() as stack:
        # joblib reads a gzip stream one pickle at a time, so the header load leaves the
        # stream at the start of the result. Any other file is a bare joblib dump, and
        # joblib opens its own decompressor for it.
        stream: Any = (
            stack.enter_context(gzip.GzipFile(mode="rb", fileobj=source)) if compressed else source
        )
        try:
            first = joblib.load(stream)
            header = compressed and _is_header(first)
            if compressed and not header:
                _drain(stream)
        except Exception as error:
            error.add_note(
                "cleverly could not read a version header from this artifact. It can predate "
                "version recording, be a direct joblib or pickle dump, or be truncated or "
                f"corrupt. This is cleverly {__version__}"
            )
            raise
        if not header:
            # A bare artifact holds the result with no header.
            _check_result(first, operation=operation)
            warn_on_version_mismatch(None, stacklevel=4)
            return first
        saved = first.get("cleverly_version")
        warn_on_version_mismatch(saved, stacklevel=4)
        try:
            result = joblib.load(stream)
            _drain(stream)
        except Exception as error:
            error.add_note(_payload_note(saved))
            raise
    _check_result(result, operation=operation)
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
    _check_result(result, operation="save")
    destination = Path(path)
    # The temporary file sits beside the destination, so ``os.replace`` is one rename on one
    # file system, and a failed write leaves an earlier artifact at ``path`` as it was.
    # ``open(..., "xb")`` creates it with the process umask, as a direct write would.
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        with open(partial, "xb") as handle:
            _write(result, handle)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
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

    Raises
    ------
    TypeError
        When the file holds no fitted causal result.
    ValueError
        When the file holds data after the result.
    """
    with Path(path).open("rb") as handle:
        return _read(handle, operation="load")


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
    _check_result(result, operation="dumps")
    buffer = io.BytesIO()
    _write(result, buffer)
    # In CPython ``getvalue`` returns the buffer's own bytes object, not a copy of it.
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

    Raises
    ------
    TypeError
        When the bytes hold no fitted causal result.
    ValueError
        When the bytes hold data after the result.
    """
    return _read(io.BytesIO(blob), operation="loads")
