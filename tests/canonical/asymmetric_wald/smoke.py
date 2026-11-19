"""A pre-declaration smoke of asymmetric-wald's post-fit path, on non-held-out groups only.

Fix-3 PR 3 review, finding M1. ``regenerate.py --replicates 3`` exercises the driver up to
the manifest, but a smoke run skips the property phase. This script drives the shared
driver's own property phase (:func:`tests.canonical.regenerate._property_artifacts`) on
:data:`~tests.studies.asymmetric_wald.SMOKE_GROUPS`, which hold no held-out law, with a capped
budget per group: the fits, the failure records, the stamp, the reconciliation, the failure
accounting over the declared clauses, the corrected summary and the falsifier hook. By
default one ``ho_hurdle`` replicate among the first 2,000 is made to raise, so the failure
path runs even when no H2 sample leaves a design cell empty.

It never fits D1, D2, C1, C2 or C3: it refuses a group list that names one. It writes only to
``--output``, which must lie outside the study directory, and it is not a declared run; its
files are disposable.

Run it from the repository root::

    python -m tests.canonical.asymmetric_wald.smoke --output <scratch> --jobs 4
"""

from __future__ import annotations

import argparse
import functools
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.exceptions import DataError
from tests.canonical import regenerate as driver
from tests.parallel import available_cores
from tests.studies import asymmetric_wald as study
from tests.studies import asymmetric_wald_properties as props
from tests.studies.evidence import property_failures
from tests.studies.small_sample_laws import EMPTY_DESIGN_CELL

#: The replicates of each group the smoke fits.
BUDGET = 400

#: The replicate made to raise: a group with a gated cell, among the first 2,000.
INJECTED = ("ho_hurdle", 1)

_REPLICATE = props._replicate


def _injected_replicate(label: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """:func:`~tests.studies.asymmetric_wald_properties._replicate`, raising on :data:`INJECTED`.

    It raises :class:`~cleverly.exceptions.DataError`, the class H2's
    :class:`~tests.studies.small_sample_laws.CellMeans` raises on an empty design cell, which
    :func:`cleverly.validation._failures.recordable` records as a failed replicate. A bare
    ``RuntimeError`` raised from test code is not recordable and stops the run, by design.
    """
    if (label, replicate) == INJECTED:
        raise DataError(f"smoke: injected failure ({EMPTY_DESIGN_CELL} stand-in)")
    return _REPLICATE(label, replicate, frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument(
        "--no-injection", action="store_true", help="fit every replicate as the study does"
    )
    arguments = parser.parse_args(argv)
    groups = tuple(study.SMOKE_GROUPS)
    named = {study.GROUP_BY_LABEL[label].law for label in groups}
    if named & set(study.HELDOUT_LAWS):
        raise SystemExit(f"refused: the smoke groups name held-out laws {sorted(named)}")
    out = arguments.output.resolve()
    if out == study.STUDY.artifacts.resolve() or study.STUDY.artifacts.resolve() in out.parents:
        raise SystemExit("refused: --output must lie outside the study directory")
    out.mkdir(parents=True, exist_ok=True)

    if not arguments.no_injection:
        props._replicate = _injected_replicate
    shim = types.SimpleNamespace(
        **{name: getattr(props, name) for name in dir(props) if not name.startswith("__")}
    )
    shim.generate_property_artifacts = functools.partial(
        props.generate_property_artifacts, budget=arguments.budget, groups=groups
    )
    names = ("property-replicates.parquet", "properties.csv", *study.STUDY.property_artifacts)
    paths = {name: out / name for name in names}
    summary, failure_files = driver._property_artifacts(
        shim,  # type: ignore[arg-type]
        argparse.Namespace(skip_properties=False, jobs=arguments.jobs),
        study.STUDY.artifacts,
        paths,
        study.STUDY,
    )

    problems = []
    if summary is None:
        problems.append("the property phase returned no summary")
    for name in names:
        if not (out / name).exists():
            problems.append(f"no {name}")
    if not arguments.no_injection:
        if tuple(failure_files) != driver.PROPERTY_FAILURE_FILES:
            problems.append(f"failure files {failure_files}, expected all three")
        records_path = out / property_failures.PROPERTY_FAILURES
        if records_path.exists():
            records = pd.read_csv(records_path)
            if not records["message"].astype(str).str.contains("injected failure").any():
                problems.append("the injected failure is not among the records")
    falsifiers = pd.read_csv(out / "falsifiers.csv")
    if summary is not None:
        cells = sorted(summary["cell"].astype(str).unique())
        print(f"summary rows {len(summary)}, cells {len(cells)}", flush=True)
        gated = summary.loc[summary["property"].isin(study.COVERAGE_FAMILIES)]
        columns = [
            "property", "cell", "failed_replicates", "lower_leg_ci_lower",
            "leg_shown_ci_upper", "below_count", "above_count", "passed",
        ]  # fmt: skip
        print(gated[[c for c in columns if c in gated.columns]].to_string(index=False))
    print(f"falsifier rows {len(falsifiers)} (held-out cells absent, so none fired)")
    print(f"failure files: {list(failure_files)}")
    for problem in problems:
        print(f"SMOKE FAILED: {problem}", file=sys.stderr, flush=True)
    if not problems:
        print(f"smoke passed; files in {out}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
