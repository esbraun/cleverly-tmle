"""A smoke of a study's property phase through the shared driver's own property step.

The shared driver skips the property phase on a smoke run (:mod:`tests.canonical.regenerate`:
a ``--replicates`` other than the declared budget writes schema-bearing placeholders), so a
defect in a study's summary step or its failure accounting would first show in the declared
run.  :func:`run` calls the driver's property step, ``_property_artifacts``, with the study's
property module bound to a few draws of a stream disjoint from the declared one: the rows, the
failure records and their reconciliation, ``summarize_properties``, the failure accounting
(``property_accounting`` with the module's ``PROPERTY_RULES``) and ``correct_properties`` all
run as they would in the declared run.

One replicate of the first group is made to raise a recordable ``DataError`` (the injected
failure), so the accounting rules run on a real record rather than being skipped.  The smoke
runs in one process, so that injection and the per-group timings see every fit; each group's
mean seconds per fit go to the timings file that the study's cost gate reads.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd

from cleverly.exceptions import DataError
from tests.canonical.regenerate import _property_artifacts
from tests.studies.evidence.manifest import write_csv

#: The message of the injected failure, which the smoke's checks look for.
INJECTED = "injected smoke failure: exercises the failure accounting"


@contextlib.contextmanager
def instrumented(
    module: Any,
    worker: str,
    *,
    group_of: Callable[[tuple[Any, ...]], str],
    fail: tuple[Any, ...] | None,
) -> Iterator[dict[str, list[float]]]:
    """Time every call of ``module.<worker>`` by group, and raise once at ``fail``.

    ``group_of`` maps the worker's positional arguments (without the drawn sample) to the
    group whose cost the call counts toward; ``fail`` is the arguments of the call that raises
    :data:`INJECTED` as a :class:`~cleverly.exceptions.DataError`.
    """
    original = getattr(module, worker)
    timings: dict[str, list[float]] = defaultdict(list)

    def timed(*args: Any) -> Any:
        key = tuple(args[:-1])
        if fail is not None and key == tuple(fail):
            raise DataError(INJECTED)
        start = time.perf_counter()
        try:
            return original(*args)
        finally:
            timings[group_of(key)].append(time.perf_counter() - start)

    setattr(module, worker, timed)
    try:
        yield timings
    finally:
        setattr(module, worker, original)


def run(
    properties: ModuleType,
    record: Any,
    *,
    generate: Callable[..., pd.DataFrame],
    output: Path,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The driver's property step on ``generate``'s rows; returns the summary and failure files.

    ``generate(n_jobs=...)`` replaces the module's ``generate_property_rows`` (a few draws of
    the smoke stream).  The artifacts go to ``output``, outside the repository.
    """
    names = {
        name: getattr(properties, name) for name in dir(properties) if not name.startswith("__")
    }
    proxy = SimpleNamespace(**names)
    proxy.__name__ = properties.__name__
    proxy.generate_property_rows = generate
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output / name
        for name in ("properties.csv", "property-replicates.parquet", *record.property_artifacts)
    }
    arguments = argparse.Namespace(skip_properties=False, jobs=1)
    summary, failure_files = _property_artifacts(
        proxy,  # type: ignore[arg-type]
        arguments,
        output,
        paths,
        record,
    )
    assert summary is not None
    return summary, failure_files


def timing_frame(timings: dict[str, list[float]], order: Sequence[str]) -> pd.DataFrame:
    """One row per group: the fits the smoke timed and their mean seconds."""
    return pd.DataFrame(
        [
            {
                "group": label,
                "fits": len(timings[label]),
                "seconds_per_fit": sum(timings[label]) / len(timings[label]),
            }
            for label in order
            if timings.get(label)
        ],
        columns=["group", "fits", "seconds_per_fit"],
    )


def write_timings(frame: pd.DataFrame, path: Path) -> None:
    """Write the smoke timings that a study's cost gate reads."""
    write_csv(frame, path, float_format="%.4g")
