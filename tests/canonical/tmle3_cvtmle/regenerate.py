"""Regenerate the same-sample, same-fold cleverly/R stacked CV-TMLE evidence."""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from tests.parallel import available_cores
from tests.studies.canonical_cvtmle import (
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
from tests.studies.stacked_cvtmle_properties import (
    generate_property_rows,
    summarize_properties,
)

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-tmle3-cvtmle-reference:ed72f8a"
ARTIFACT_NAMES = (
    "replicates.csv.gz",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.csv.gz",
    "properties.csv",
)


def run_r(samples: Path, truths: Path, output: Path, *, cores: int) -> None:
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
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.replicates < 2 or args.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    record = dataclasses.replace(STUDY, replicates=args.replicates, n=args.n, artifacts=out)

    with tempfile.TemporaryDirectory(prefix="cleverly-tmle3-cvtmle-") as raw:
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
            if python.empty or python["replicate"].max() + 1 != args.replicates:
                raise RuntimeError("cached Python rows have the wrong replication count")
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
                (committed["implementation"] == STUDY.reference)
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
    rows = pd.concat([python, r], ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]
    rows = rows.sort_values(
        ["scenario", "replicate", "estimand", "implementation"], ignore_index=True
    )
    validate_replicates(rows, record=record)

    paths = {name: out / name for name in ARTIFACT_NAMES}
    write_csv(rows, paths["replicates.csv.gz"], compression={"method": "gzip", "mtime": 0})
    summaries = summarize(rows)
    write_csv(summaries, paths["summary.csv"])
    performance = independent_performance_tests(rows, record=record, n_jobs=args.jobs)
    write_csv(performance, paths["performance-tests.csv"])
    paired = equivalence(rows, summaries, performance, record=record, n_jobs=args.jobs)
    write_csv(paired, paths["equivalence.csv"])

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
        reference_files=[HERE / "Dockerfile", HERE / "run_tmle3_cvtmle.R"],
        reference_metadata=REFERENCE_METADATA,
        configuration=CONFIGURATION,
    )
    failures = {
        "independent performance": performance.loc[~performance["passed"]],
        "paired comparison": paired.loc[~paired["passed"]],
        "reference validity": paired.loc[~paired["reference_valid"]],
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
