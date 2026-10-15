"""Check that a change moves no byte of any recomputed ``properties.csv``.

RM39b2 rev 3 commit 7c moves each statistic a property verdict computes inline into a function
that the verdict and the failure accounting both call (D3'c).  The move must change no
published byte.  This script reads each registered study's committed
``property-replicates.csv.gz``, recomputes the summary through the module's own
``summarize_properties``, applies the committed property failure accounting as the
regeneration driver does, and writes the result through
:func:`tests.studies.evidence.manifest.write_csv`.  It is not the fast tier's recompute test,
which compares values at a relative tolerance.

It compares the bytes two ways.

| comparison | what it shows |
| --- | --- |
| with the committed ``properties.csv`` | whether this machine reproduces the published bytes. A study that differs is reported with its largest relative difference, because the floating-point summation of another machine can move the last digit |
| with ``--against DIR``, the bytes ``--dump DIR`` wrote from a reference checkout on this machine | whether the change moved any byte. This comparison is exact |

Usage::

    git stash; python -m tests.canonical.property_summary_identity --dump /tmp/before
    git stash pop; python -m tests.canonical.property_summary_identity --against /tmp/before
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies.evidence import property_failures
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import registered


def recomputed_bytes(slug: str) -> bytes:
    """The ``properties.csv`` bytes the study's own code writes from its committed rows.

    Parameters
    ----------
    slug : str
        A registered study.

    Returns
    -------
    bytes
        The recomputed file.
    """
    (record,) = [record for record in registered() if record.slug == slug]
    module = record.properties()
    # ``round_trip`` reads back exactly the floats ``write_csv`` wrote.
    rows = pd.read_csv(record.artifact("property-replicates.csv.gz"), float_precision="round_trip")
    summary = module.summarize_properties(rows)
    path = record.artifact(property_failures.PROPERTY_ACCOUNTING)
    if path.exists():
        accounting = pd.read_csv(path, keep_default_na=False, na_values=[""])
        summary = property_failures.correct_properties(summary, accounting, module=module)
    buffer = io.StringIO()
    write_csv(summary, buffer)  # type: ignore[arg-type]
    return buffer.getvalue().encode("utf-8")


def largest_relative_difference(left: bytes, right: bytes) -> float:
    """The largest relative difference between two tables' numeric cells.

    ``inf`` when the tables differ in shape, columns, or a non-numeric cell.

    Parameters
    ----------
    left : bytes
        One CSV file.
    right : bytes
        The other.

    Returns
    -------
    float
        The largest ``|a - b| / max(|a|, |b|)`` over cells where the two differ.
    """
    first = pd.read_csv(io.BytesIO(left), dtype=str, keep_default_na=False)
    second = pd.read_csv(io.BytesIO(right), dtype=str, keep_default_na=False)
    if first.shape != second.shape or list(first.columns) != list(second.columns):
        return math.inf
    largest = 0.0
    for column in first.columns:
        for a, b in zip(first[column], second[column], strict=True):
            if a == b:
                continue
            try:
                x, y = float(a), float(b)
            except ValueError:
                return math.inf
            if not (np.isfinite(x) and np.isfinite(y)):
                return math.inf
            largest = max(largest, abs(x - y) / max(abs(x), abs(y)))
    return largest


def compare(task: tuple[str, Path | None, Path | None]) -> tuple[str, str, str, float]:
    """``(slug, committed, reference, seconds)`` for one study."""
    slug, dump, against = task
    (record,) = [record for record in registered() if record.slug == slug]
    started = time.perf_counter()
    try:
        produced = recomputed_bytes(slug)
    except Exception as error:
        message = f"error: {type(error).__name__}: {error}"
        return slug, message, message, time.perf_counter() - started
    committed = record.artifact("properties.csv").read_bytes()
    if produced == committed:
        versus_committed = "identical"
    else:
        versus_committed = f"differs (largest relative difference {largest_relative_difference(produced, committed):.3g})"
    versus_reference = "-"
    if dump is not None:
        (dump / f"{slug}.csv").write_bytes(produced)
    if against is not None:
        reference = against / f"{slug}.csv"
        if not reference.exists():
            versus_reference = "no reference"
        else:
            versus_reference = "identical" if reference.read_bytes() == produced else "differs"
    return slug, versus_committed, versus_reference, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Compare every study with committed property artifacts.

    The exit status is non-zero when a study differs from ``--against``, or, without it,
    when a study errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", nargs="*", help="the studies to check (default: all)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dump", type=Path, help="write each recomputed file here")
    parser.add_argument("--against", type=Path, help="compare with the files --dump wrote")
    parser.add_argument("--log", type=Path, help="append the report here as well")
    arguments = parser.parse_args(argv)
    if arguments.dump is not None:
        arguments.dump.mkdir(parents=True, exist_ok=True)
    slugs = [
        record.slug
        for record in registered()
        if record.artifact("properties.csv").exists()
        and record.artifact("property-replicates.csv.gz").exists()
        and (not arguments.slugs or record.slug in arguments.slugs)
    ]
    tasks = [(slug, arguments.dump, arguments.against) for slug in slugs]
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        results = sorted(pool.map(compare, tasks))
    lines = [
        f"{slug}\tcommitted: {committed}\treference: {reference}\t{seconds:.1f}s"
        for slug, committed, reference, seconds in results
    ]
    same = sum(committed == "identical" for _, committed, _, _ in results)
    lines.append(f"committed bytes: {same} of {len(results)} identical")
    failed = [slug for slug, committed, _, _ in results if committed.startswith("error")]
    if arguments.against is not None:
        matched = sum(reference == "identical" for _, _, reference, _ in results)
        lines.append(f"reference bytes: {matched} of {len(results)} identical")
        failed += [slug for slug, _, reference, _ in results if reference != "identical"]
    report = "\n".join(lines) + "\n"
    sys.stdout.write(report)
    if arguments.log is not None:
        with arguments.log.open("a", encoding="utf-8") as handle:
            handle.write(report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
