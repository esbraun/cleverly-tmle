"""Regenerate the frozen same-sample cleverly/R tmle3 evidence artifacts."""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies.canonical_properties import generate_property_rows, summarize_properties
from tests.studies.canonical_tmle import (
    ARTIFACT_COLUMNS,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    SEED,
    cleverly_rows,
    draw_scenario,
    equivalence,
    independent_performance_tests,
    summarize,
    write_manifest,
)

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-tmle3-reference:ed72f8a"


def samples_and_python(replicates: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_values = np.random.SeedSequence(SEED).generate_state(2 * replicates)
    sample_frames: list[pd.DataFrame] = []
    python_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(("continuous", "binary")):
        for replicate in range(replicates):
            seed = int(seed_values[scenario_index * replicates + replicate])
            frame, truth = draw_scenario(scenario, n, seed)
            payload = frame.copy()
            payload.insert(0, "replicate", replicate)
            payload.insert(0, "scenario", scenario)
            for name, value in truth.items():
                payload[f"truth_{name}"] = value
            sample_frames.append(payload)
            python_rows.extend(cleverly_rows(frame, truth, scenario, replicate))
    return pd.concat(sample_frames, ignore_index=True), pd.DataFrame(python_rows)


def run_r(samples: Path, output: Path) -> None:
    subprocess.run(["docker", "build", "-t", IMAGE, str(HERE)], check=True)
    mount = str(samples.parent.resolve())
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{mount}:/work",
            IMAGE,
            f"/work/{samples.name}",
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
    args = parser.parse_args()
    if args.replicates < 2 or args.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")

    with tempfile.TemporaryDirectory(prefix="cleverly-tmle3-") as raw:
        scratch = Path(raw)
        samples_path = scratch / "samples.csv.gz"
        r_path = scratch / "r-results.csv"
        samples, python = samples_and_python(args.replicates, args.n)
        samples.to_csv(samples_path, index=False, compression="gzip")
        if args.skip_r:
            existing = HERE / "replicates.csv.gz"
            r = pd.read_csv(existing).query("implementation == 'tmle3'")
            r = r.query("replicate < @args.replicates and n == @args.n")
            if r.empty:
                raise RuntimeError("no compatible committed R rows for --skip-r")
        else:
            run_r(samples_path, r_path)
            r = pd.read_csv(r_path)

    numeric = [
        "replicate",
        "n",
        "truth",
        "estimate",
        "inference_estimate",
        "std_error",
        "ci_lower",
        "ci_upper",
        "covered",
        "initial_estimate",
    ]
    for column in numeric:
        r[column] = pd.to_numeric(r[column], errors="raise")

    rows = pd.concat([python, r], ignore_index=True)
    rows = rows.loc[:, ARTIFACT_COLUMNS].sort_values(
        ["scenario", "replicate", "estimand", "implementation"]
    )
    if not math.isclose(float(rows["covered"].min()), 0.0) and not math.isclose(
        float(rows["covered"].max()), 1.0
    ):
        raise RuntimeError("coverage column was not populated")
    replicate_path = HERE / "replicates.csv.gz"
    summary_path = HERE / "summary.csv"
    equivalence_path = HERE / "equivalence.csv"
    performance_path = HERE / "performance-tests.csv"
    property_replicate_path = HERE / "property-replicates.csv.gz"
    property_summary_path = HERE / "properties.csv"
    rows.to_csv(replicate_path, index=False, compression={"method": "gzip", "mtime": 0})
    summaries = summarize(rows)
    summaries.to_csv(summary_path, index=False)
    equivalent = equivalence(rows, summaries)
    equivalent.to_csv(equivalence_path, index=False)
    performance = independent_performance_tests(rows)
    performance.to_csv(performance_path, index=False)
    if args.skip_properties:
        if not property_replicate_path.exists() or not property_summary_path.exists():
            raise RuntimeError("--skip-properties needs committed property artifacts")
    else:
        property_rows = generate_property_rows()
        property_rows.to_csv(
            property_replicate_path,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        property_summary = summarize_properties(property_rows)
        property_summary.to_csv(property_summary_path, index=False)
    write_manifest(
        HERE / "manifest.json",
        [
            replicate_path,
            summary_path,
            equivalence_path,
            performance_path,
            property_replicate_path,
            property_summary_path,
        ],
        replicates=args.replicates,
        n=args.n,
        reference_files=[HERE / "Dockerfile", HERE / "run_tmle3.R"],
    )
    failures = equivalent.loc[~equivalent["passed"]]
    performance_failures = performance.loc[~performance["passed"]]
    property_failures = pd.read_csv(property_summary_path).query("not passed")
    if not property_failures.empty and not args.allow_failures:
        raise RuntimeError(
            f"statistical property gates failed:\n{property_failures.to_string(index=False)}"
        )
    if not failures.empty and not args.allow_failures:
        raise RuntimeError(f"equivalence gates failed:\n{failures.to_string(index=False)}")
    if not performance_failures.empty and not args.allow_failures:
        raise RuntimeError(
            f"independent performance gates failed:\n{performance_failures.to_string(index=False)}"
        )


if __name__ == "__main__":
    main()
