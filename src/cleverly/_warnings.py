"""Warnings the package raises, recorded on the result of the fit that raised them.

A warning printed during a fit is gone by the time anyone reads the result, and a summary
printed later would never mention it. Every warning in the package is raised through
:func:`warn`, which records it on the active recorder and then calls
:func:`warnings.warn` exactly as the call site did, so what a user sees and where it
points do not change. A fit entry point decorated with :func:`records_fit` runs under its
own recorder and attaches what it caught to its result as ``fit_warnings``, and the
result's summary prints each one as a note.

Why at the call site rather than a ``showwarning`` hook: the user's filters and the
once-per-location registry hide repeats from a hook, a hook is process-global, and a
warning raised in a joblib worker never reaches the parent's hook.
:func:`~cleverly.utils.parallel.map_parallel` instead returns each task's records with its
value (:func:`run_recorded`), and the parent merges them in task order, so ``n_jobs=1``
and ``n_jobs=2`` record the same set.

:func:`suppressed` is the one way the package silences its own warnings, at the places
that refit on resamples or perturbed data: it silences and stops recording together, so
a bootstrap replicate's warning does not become a note on the main fit.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, ParamSpec, TypeVar

__all__ = [
    "FitWarning",
    "attach",
    "merge",
    "recording",
    "records_data",
    "records_fit",
    "run_recorded",
    "suppressed",
    "warn",
]

T = TypeVar("T")
P = ParamSpec("P")


@dataclass(frozen=True)
class FitWarning:
    """One warning a fit raised.

    Parameters
    ----------
    category : str
        The warning class's name, such as ``"PositivityWarning"``. A name rather than the
        class, so a saved result loads without importing it.
    message : str
        The message, as the user saw it.
    kind : type of Warning or None
        The class itself, which lets a parent drop a worker's record of a category it
        suppresses. Not compared, so two records of one message are one record.
    count : int
        How many times the fit raised this message. Not compared.
    """

    category: str
    message: str
    kind: type[Warning] | None = field(default=None, compare=False, repr=False)
    count: int = field(default=1, compare=False)


def _combined(
    held: Iterable[FitWarning], new: Iterable[FitWarning], *, add_counts: bool
) -> tuple[FitWarning, ...]:
    """``held`` then the records of ``new`` it lacks, in first-seen order.

    With ``add_counts`` a record in both keeps the sum of the two counts, for records of
    separate runs. Without, it keeps the held count, for a record seen twice.
    """
    out = list(held)
    for record in new:
        if record in out:
            index = out.index(record)
            if add_counts:
                out[index] = replace(out[index], count=out[index].count + record.count)
        else:
            out.append(record)
    return tuple(out)


class _Recorder:
    """The warnings caught under one :func:`recording` block, in first-seen order."""

    def __init__(self) -> None:
        self.records: list[FitWarning] = []

    def add(self, record: FitWarning) -> None:
        self.records[:] = _combined(self.records, (record,), add_counts=True)


_RECORDER: ContextVar[_Recorder | None] = ContextVar("cleverly_fit_warnings", default=None)
#: Categories :func:`suppressed` stops recording. A tuple so the check is one ``issubclass``.
_SUPPRESSED: ContextVar[tuple[type[Warning], ...]] = ContextVar(
    "cleverly_suppressed_warnings", default=()
)


def _is_suppressed(category: type[Warning]) -> bool:
    suppressed = _SUPPRESSED.get()
    return bool(suppressed) and issubclass(category, suppressed)


def warn(
    message: str,
    category: type[Warning] = UserWarning,
    *,
    stacklevel: int = 1,
    record: bool = True,
) -> None:
    """Raise a warning, and record it on the active fit recorder.

    Parameters
    ----------
    message : str
        The message.
    category : type of Warning
        The warning class.
    stacklevel : int
        As :func:`warnings.warn` reads it at the call site. This function adds its own
        frame, so the location a user sees is the one the call site always reported.
    record : bool
        ``False`` for a warning that is not about a fit, such as a loaded artifact's
        version warning.
    """
    if record and not _is_suppressed(category):
        recorder = _RECORDER.get()
        if recorder is not None:
            recorder.add(FitWarning(category.__name__, str(message), category))
    warnings.warn(message, category, stacklevel=stacklevel + 1)


@contextmanager
def recording() -> Iterator[list[FitWarning]]:
    """Record every package warning raised in the block.

    The block gets a fresh recorder, so a nested fit keeps its own records and does not
    add them to the enclosing fit's.

    Yields
    ------
    list of FitWarning
        The records, filled as the block runs.
    """
    recorder = _Recorder()
    token = _RECORDER.set(recorder)
    try:
        yield recorder.records
    finally:
        _RECORDER.reset(token)


def merge(records: Iterable[FitWarning]) -> None:
    """Add records made elsewhere, such as in a worker process, to the active recorder.

    A category the current block suppresses is dropped, as it would have been had the
    warning been raised here.

    Parameters
    ----------
    records : iterable of FitWarning
        The records.
    """
    recorder = _RECORDER.get()
    if recorder is None:
        return
    for record in records:
        if record.kind is not None and _is_suppressed(record.kind):
            continue
        recorder.add(record)


def run_recorded(func: Callable[..., T], *args: Any) -> tuple[T, tuple[FitWarning, ...]]:
    """Run ``func(*args)`` under its own recorder and return its value with the records.

    The unit of work :func:`~cleverly.utils.parallel.map_parallel` sends to a worker, so
    the records travel back to the parent as data.

    Parameters
    ----------
    func : callable
        The task.
    *args : Any
        Its arguments.

    Returns
    -------
    tuple
        The task's value and the warnings it recorded.
    """
    with recording() as records:
        value = func(*args)
    return value, tuple(records)


@contextmanager
def suppressed(category: type[Warning] = Warning) -> Iterator[None]:
    """Silence warnings of ``category`` and stop recording them, inside the block.

    Parameters
    ----------
    category : type of Warning
        The class to silence, with its subclasses.

    Yields
    ------
    None
        Nothing.
    """
    token = _SUPPRESSED.set((*_SUPPRESSED.get(), category))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category)
            yield
    finally:
        _SUPPRESSED.reset(token)


def attach(result: Any, records: Iterable[FitWarning]) -> Any:
    """Store ``records`` on a fit's result as ``fit_warnings``, after any it holds.

    A result set (one result per level of an intermediate variable) gets the records on
    each of its results. A subclass's ``fit`` that calls its parent's returns the parent's
    result, which already holds the parent's records; they are kept.

    Parameters
    ----------
    result : Any
        The fit's result.
    records : iterable of FitWarning
        The records.

    Returns
    -------
    Any
        ``result``.
    """
    records = _plain(records)
    members = getattr(result, "results", None)
    targets = list(members.values()) if isinstance(members, dict) else [result]
    for target in targets:
        if not hasattr(target, "fit_warnings"):
            continue
        # The data's own warnings came first, when the data was built, possibly before
        # any fit. A record already held is not counted again.
        data = getattr(target, "data", None)
        held = _combined(
            getattr(data, "validation_warnings", ()), target.fit_warnings, add_counts=False
        )
        object.__setattr__(target, "fit_warnings", _combined(held, records, add_counts=True))
    return result


def _plain(records: Iterable[FitWarning]) -> tuple[FitWarning, ...]:
    """The records without their class, so a saved object names its warnings and loads
    without importing them."""
    return tuple(
        FitWarning(record.category, record.message, count=record.count) for record in records
    )


def records_data(func: Callable[P, T]) -> Callable[P, T]:
    """Decorate a data constructor so the data records the warnings its checks raised.

    A fit reads them from its data (:func:`attach`), so a warning raised when the data was
    built, before any fit ran, still reaches the fit's summary.

    Parameters
    ----------
    func : callable
        The constructor, which returns an object with a ``validation_warnings`` field.

    Returns
    -------
    callable
        The same constructor, run under :func:`recording`.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with recording() as records:
            data = func(*args, **kwargs)
        if hasattr(data, "validation_warnings"):
            held = tuple(data.validation_warnings)
            object.__setattr__(
                data,
                "validation_warnings",
                _combined(held, _plain(records), add_counts=False),
            )
        return data

    return wrapper


def records_fit(func: Callable[P, T]) -> Callable[P, T]:
    """Decorate a fit entry point so its result records the warnings the fit raised.

    Decorate only a function that returns a result (or a set of results), never a learner's
    ``fit``: a nested recorder keeps what it catches, so a learner's warnings would never
    reach the result.

    Parameters
    ----------
    func : callable
        The fit method.

    Returns
    -------
    callable
        The same method, run under :func:`recording`.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with recording() as records:
            result = func(*args, **kwargs)
        attach(result, records)
        return result

    return wrapper
