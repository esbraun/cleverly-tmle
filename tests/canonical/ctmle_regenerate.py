"""Shared regeneration driver for the two C-TMLE evidence studies."""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

import pandas as pd

from tests.parallel import available_cores
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.manifest import write_csv, write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.schema import REPLICATE_COLUMNS, validate_replicates

ARTIFACT_NAMES = (
    "replicates.csv.gz",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.csv.gz",
    "properties.csv",
)


def _run_r(
    here: Path, image: str, runner: str, samples: Path, truths: Path, output: Path, *, cores: int
) -> None:
    subprocess.run(["docker", "build", "-t", image, str(here)], check=True)
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
            image,
            f"/work/{samples.name}",
            f"/work/{truths.name}",
            f"/work/{output.name}",
        ],
        check=True,
    )


def main(study: ModuleType, properties: ModuleType, *, here: Path, image: str, runner: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=study.PRIMARY_REPLICATES)
    parser.add_argument("--n", type=int, default=study.PRIMARY_N)
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument("--skip-properties", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="run the paired primary study only; intended for disposable feasibility probes",
    )
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--jobs", type=int, default=available_cores())
    arguments = parser.parse_args()
    if arguments.replicates < 2 or arguments.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    record = dataclasses.replace(
        study.STUDY,
        replicates=arguments.replicates,
        n=arguments.n,
        artifacts=out,
    )
    with tempfile.TemporaryDirectory(prefix=f"cleverly-{study.STUDY.slug}-") as raw:
        scratch = Path(raw)
        samples_path = scratch / "samples.csv.gz"
        truth_path = scratch / "truth.csv"
        r_path = scratch / "r-results.csv"
        samples, truths, python = study.draw_and_fit(
            replicates=arguments.replicates,
            n=arguments.n,
            n_jobs=arguments.jobs,
        )
        write_csv(samples, samples_path, compression="gzip")
        write_csv(truths, truth_path)
        if arguments.skip_r:
            committed = pd.read_csv(here / "replicates.csv.gz")
            r = committed.loc[
                (committed["implementation"] == study.STUDY.reference)
                & (committed["replicate"] < arguments.replicates)
                & (committed["n"] == arguments.n)
            ]
            if r.empty:
                raise RuntimeError("no compatible committed R rows for --skip-r")
        else:
            _run_r(
                here,
                image,
                runner,
                samples_path,
                truth_path,
                r_path,
                cores=arguments.jobs,
            )
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
    performance = independent_performance_tests(rows, record=record, n_jobs=arguments.jobs)
    write_csv(performance, paths["performance-tests.csv"])
    paired = equivalence(rows, summaries, performance, record=record, n_jobs=arguments.jobs)
    write_csv(paired, paths["equivalence.csv"])

    if arguments.primary_only:
        print(performance.to_string(index=False))
        print(paired.to_string(index=False))
        return

    if arguments.skip_properties:
        for name in ("property-replicates.csv.gz", "properties.csv"):
            committed = here / name
            if not committed.exists():
                raise RuntimeError(f"--skip-properties needs a committed {name}")
            paths[name].write_bytes(committed.read_bytes())
    else:
        property_rows = properties.generate_property_rows(n_jobs=arguments.jobs)
        write_csv(
            property_rows,
            paths["property-replicates.csv.gz"],
            compression={"method": "gzip", "mtime": 0},
        )
        write_csv(properties.summarize_properties(property_rows), paths["properties.csv"])

    write_manifest(
        out / "manifest.json",
        record,
        [paths[name] for name in ARTIFACT_NAMES],
        reference_files=[here / "Dockerfile", here / runner],
        reference_metadata=study.REFERENCE_METADATA,
        configuration=study.CONFIGURATION,
    )
    failures = {
        "independent performance": performance.loc[~performance["passed"]],
        "paired comparison": paired.loc[~paired["passed"]],
        "reference validity": paired.loc[~paired["reference_valid"]],
        "statistical property": pd.read_csv(paths["properties.csv"]).query(
            "not passed or not property_passed"
        ),
    }
    reported = {name: frame for name, frame in failures.items() if not frame.empty}
    if reported and not arguments.allow_failures:
        raise RuntimeError(
            "\n\n".join(
                f"{name} gates failed:\n{frame.to_string(index=False)}"
                for name, frame in reported.items()
            )
        )
