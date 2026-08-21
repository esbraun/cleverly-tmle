"""Regenerate fold-evaluated CV-TMLE evidence artifacts."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from tests.parallel import available_cores
from tests.studies.evidence.comparison import empty_equivalence
from tests.studies.evidence.manifest import write_csv, write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.schema import validate_replicates
from tests.studies.fold_cvtmle_properties import generate_property_rows, summarize_properties
from tests.studies.fold_evaluated_cvtmle import (
    CONFIGURATION,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    STUDY,
    draw_and_fit,
)

HERE = Path(__file__).resolve().parent
ARTIFACT_NAMES = (
    "replicates.csv.gz",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.csv.gz",
    "properties.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=PRIMARY_REPLICATES)
    parser.add_argument("--n", type=int, default=PRIMARY_N)
    parser.add_argument("--skip-properties", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--jobs", type=int, default=available_cores())
    args = parser.parse_args()
    if args.replicates < 2 or args.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    record = dataclasses.replace(STUDY, replicates=args.replicates, n=args.n, artifacts=out)

    rows = draw_and_fit(replicates=args.replicates, n=args.n, n_jobs=args.jobs)
    validate_replicates(rows, record=record)
    paths = {name: out / name for name in ARTIFACT_NAMES}
    write_csv(rows, paths["replicates.csv.gz"], compression={"method": "gzip", "mtime": 0})
    summaries = summarize(rows)
    write_csv(summaries, paths["summary.csv"])
    performance = independent_performance_tests(rows, record=record, n_jobs=args.jobs)
    write_csv(performance, paths["performance-tests.csv"])
    write_csv(empty_equivalence(), paths["equivalence.csv"])

    if args.skip_properties:
        for name in ("property-replicates.csv.gz", "properties.csv"):
            committed = HERE / name
            if not committed.exists():
                raise RuntimeError(f"--skip-properties needs a committed {name}")
            paths[name].write_bytes(committed.read_bytes())
    else:
        property_rows = generate_property_rows(n_jobs=args.jobs)
        write_csv(
            property_rows,
            paths["property-replicates.csv.gz"],
            compression={"method": "gzip", "mtime": 0},
        )
        write_csv(summarize_properties(property_rows), paths["properties.csv"])

    write_manifest(
        out / "manifest.json",
        record,
        [paths[name] for name in ARTIFACT_NAMES],
        configuration=CONFIGURATION,
    )
    failures = {
        "independent performance": performance.loc[~performance["passed"]],
        "statistical property": summarize_properties(property_rows).query("not passed")
        if not args.skip_properties
        else None,
    }
    reported = {
        name: frame for name, frame in failures.items() if frame is not None and not frame.empty
    }
    if reported and not args.allow_failures:
        raise RuntimeError(
            "\n\n".join(
                f"{name} gates failed:\n{frame.to_string(index=False)}"
                for name, frame in reported.items()
            )
        )


if __name__ == "__main__":
    main()
