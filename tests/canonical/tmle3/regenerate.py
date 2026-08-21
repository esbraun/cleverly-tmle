"""Regenerate the frozen same-sample cleverly/R tmle3 evidence artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from tests.parallel import available_cores
from tests.studies.canonical_properties import generate_property_rows, summarize_properties
from tests.studies.canonical_tmle import (
    CONFIGURATION,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    REFERENCE_METADATA,
    STUDY,
    draw_and_fit,
)
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.manifest import write_csv, write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.schema import REPLICATE_COLUMNS, validate_replicates

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-tmle3-reference:ed72f8a"

ARTIFACT_NAMES = (
    "replicates.csv.gz",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.csv.gz",
    "properties.csv",
)


def run_r(samples: Path, truths: Path, output: Path, *, cores: int) -> None:
    """Fit the R reference on the same samples, using the whole core budget.

    Run after the Python side finishes rather than beside it: the two are the same work on
    the same cores, and overlapping them would leave both contending for a machine neither
    can have.
    """
    subprocess.run(["docker", "build", "-t", IMAGE, str(HERE)], check=True)
    mount = str(samples.parent.resolve())
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            f"CLEVERLY_R_CORES={cores}",
            "-v",
            f"{mount}:/work",
            IMAGE,
            f"/work/{samples.name}",
            f"/work/{truths.name}",
            f"/work/{output.name}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=PRIMARY_REPLICATES)
    parser.add_argument("--n", type=int, default=PRIMARY_N)
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--skip-properties", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE,
        help="where to write the artifacts; point it elsewhere for a timing probe",
    )
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument(
        "--cache",
        type=Path,
        help=(
            "keep the drawn samples and fitted cleverly rows here and reuse them when they "
            "match; the Python phase is the long half and a failed R phase should not cost it "
            "twice"
        ),
    )
    args = parser.parse_args()
    if args.replicates < 2 or args.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    record = dataclasses.replace(STUDY, replicates=args.replicates, n=args.n, artifacts=out)
    print(
        f"cleverly: {args.replicates} replications x {len(record.scenarios)} laws on {args.jobs} cores"
    )

    with tempfile.TemporaryDirectory(prefix="cleverly-tmle3-") as raw:
        scratch = Path(args.cache) if args.cache else Path(raw)
        scratch.mkdir(parents=True, exist_ok=True)
        samples_path = scratch / "samples.csv.gz"
        truth_path = scratch / "truth.csv"
        python_path = scratch / "python-rows.csv.gz"
        r_path = scratch / "r-results.csv"
        cached = args.cache and all(
            path.exists() for path in (samples_path, truth_path, python_path)
        )
        if cached:
            python = pd.read_csv(python_path)
            usable = len(python) and python["replicate"].max() + 1 == args.replicates
            if not usable:
                raise RuntimeError(f"{python_path} does not hold {args.replicates} replications")
            print(f"reusing the cached python phase in {scratch}")
        else:
            samples, truths, python = draw_and_fit(
                replicates=args.replicates, n=args.n, n_jobs=args.jobs
            )
            write_csv(samples, samples_path, compression="gzip")
            write_csv(truths, truth_path)
            write_csv(python, python_path, compression="gzip")
        if args.skip_r:
            committed = pd.read_csv(HERE / "replicates.csv.gz")
            r = committed.loc[
                (committed["implementation"] == "tmle3")
                & (committed["replicate"] < args.replicates)
                & (committed["n"] == args.n)
            ]
            if r.empty:
                raise RuntimeError("no compatible committed R rows for --skip-r")
        else:
            run_r(samples_path, truth_path, r_path, cores=args.jobs)
            r = pd.read_csv(r_path)

    for column in REPLICATE_COLUMNS:
        if column not in ("implementation", "scenario", "estimand", "inference_scale"):
            r[column] = pd.to_numeric(r[column], errors="raise")

    rows = pd.concat([python, r], ignore_index=True)
    rows = rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["scenario", "replicate", "estimand", "implementation"], ignore_index=True
    )
    validate_replicates(rows, record=record)

    paths = {name: out / name for name in ARTIFACT_NAMES}
    write_csv(rows, paths["replicates.csv.gz"], compression={"method": "gzip", "mtime": 0})
    summaries = summarize(rows)
    write_csv(summaries, paths["summary.csv"])
    performance = independent_performance_tests(rows, record=record, n_jobs=args.jobs)
    write_csv(performance, paths["performance-tests.csv"])
    equivalent = equivalence(rows, summaries, performance, record=record, n_jobs=args.jobs)
    write_csv(equivalent, paths["equivalence.csv"])

    if args.skip_properties:
        for name in ("property-replicates.csv.gz", "properties.csv"):
            if paths[name].exists():
                continue
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
        reference_files=[HERE / "Dockerfile", HERE / "run_tmle3.R"],
        reference_metadata=REFERENCE_METADATA,
        configuration=CONFIGURATION,
    )

    failures = {
        "independent performance": performance.loc[~performance["passed"]],
        "paired similarity and non-inferiority": equivalent.loc[~equivalent["passed"]],
        "reference validity": equivalent.loc[~equivalent["reference_valid"]],
        "statistical property": pd.read_csv(paths["properties.csv"]).query("not passed"),
    }
    reported = {name: frame for name, frame in failures.items() if not frame.empty}
    if reported and not args.allow_failures:
        raise RuntimeError(
            "\n\n".join(
                f"{name} gates failed:\n{frame.to_string(index=False)}"
                for name, frame in reported.items()
            )
        )


if __name__ == "__main__":
    main()
