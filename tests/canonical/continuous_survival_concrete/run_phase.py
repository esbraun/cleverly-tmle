"""Run the ``concrete`` comparator phase of ``continuous-time-survival`` (roadmap X6 part f).

The phase is declared in :mod:`tests.studies.continuous_survival_concrete`: three paired cells,
500 draws each at n = 1,000, against R ``concrete`` 1.0.5 pinned by its CRAN archive tarball.
It runs on a machine with Docker, never in CI, and it never blocks a merge.  The registered
record's artifacts are not touched.

Usage, from the repository root::

    python -m tests.canonical.continuous_survival_concrete.run_phase --output SCRATCH
    python -m tests.canonical.continuous_survival_concrete.run_phase --output SCRATCH --publish

``SCRATCH`` is a directory outside the repository; a rerun with the same directory resumes.
The run writes the draws, fits the cleverly side (``--jobs`` processes, each fit
single-threaded), builds ``cleverly-concrete:1.0.5`` from ``tests/canonical/concrete`` and runs
``run_concrete.R`` in it (``--r-cores`` workers; the harness checkpoints each replicate under
``SCRATCH/r-checkpoints``), then writes the tables and ``manifest.json`` under
``SCRATCH/artifacts``.  ``--publish`` copies them into this directory, where
``tests/unit/test_continuous_survival_concrete.py`` replays them.  ``--replicates`` below the
declared count is a smoke run, which ``--publish`` refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tests.canonical.regenerate import Reference, failures_path, image_content
from tests.parallel import available_cores
from tests.studies import continuous_survival_concrete as phase
from tests.studies.evidence.manifest import (
    NEWLINE,
    hashes,
    provenance,
    reference_hashes,
    write_csv,
    write_rows,
)
from tests.studies.evidence.registry import ROOT

# Each cleverly fit runs single-threaded; the processes are the parallelism.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

HERE = Path(__file__).resolve().parent
CANONICAL = ROOT / "tests" / "canonical"

REFERENCE = Reference(
    image=phase.IMAGE,
    runner=phase.RUNNER,
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=CANONICAL / "concrete",
    runner_root=CANONICAL,
)


#: The run configuration a scratch directory holds, written by its first run.
CONFIGURATION = "run.json"


def configuration(replicates: int) -> dict[str, Any]:
    """What a scratch directory's draws, seeds and declaration are, for the resume check."""

    def digest(values: Any) -> str:
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()

    return {
        "replicates": replicates,
        "sample_seeds_sha256": digest(
            [phase.sample_seed(cell, r) for r in range(replicates) for cell in phase.CELLS]
        ),
        "group_seeds_sha256": digest([phase.group_seed(r) for r in range(replicates)]),
        "declaration_sha256": digest(phase.declaration()),
    }


def claim(scratch: Path, replicates: int) -> None:
    """Record the run in ``scratch``, or refuse a scratch written by another configuration."""
    path = scratch / CONFIGURATION
    wanted = configuration(replicates)
    if path.exists():
        held = json.loads(path.read_text(encoding="utf-8"))
        if held != wanted:
            differing = sorted(key for key in wanted if held.get(key) != wanted[key])
            raise SystemExit(
                f"{scratch} holds a run with another configuration ({', '.join(differing)}); "
                "use a new --output directory"
            )
        return
    if any(scratch.iterdir()):
        raise SystemExit(f"{scratch} holds files but no {CONFIGURATION}; use an empty directory")
    path.write_text(json.dumps(wanted, indent=2, sort_keys=True) + NEWLINE, encoding="utf-8")


def coverage_gaps(rows: pd.DataFrame, failures: pd.DataFrame, replicates: int) -> list[str]:
    """Each (implementation, cell, replicate) the run declared that neither rows nor failures hold."""
    gaps = []
    fitted = set(zip(rows["implementation"], rows["scenario"], rows["replicate"], strict=True))
    failed: set[tuple[str, str, int]] = set()
    for record in failures.to_dict("records") if len(failures) else []:
        implementation = str(record["implementation"])
        if implementation == phase.REFERENCE:
            # A harness record names its group's scenarios and replicates, ';'-joined.
            for cell in str(record["scenarios"]).split(";"):
                for r in str(record["replicates"]).split(";"):
                    failed.add((phase.REFERENCE, cell, int(r)))
        else:
            for name, _ in phase.FITS[str(record["scenario"])]:
                failed.add((name, str(record["scenario"]), int(record["replicate"])))
    for cell in phase.CELLS:
        names = [name for name, _ in phase.FITS[cell]] + [phase.REFERENCE]
        for r in range(replicates):
            for name in names:
                if (name, cell, r) not in fitted and (name, cell, r) not in failed:
                    gaps.append(f"{name} {cell} {r}")
    return gaps


def write_inputs(scratch: Path, replicates: int) -> None:
    """The draws, the truths and the R seeds, written once per scratch directory."""
    samples = scratch / "samples.csv.gz"
    if not samples.exists():
        frames = [
            phase.sample_rows(cell, r, phase.draw(cell, r))
            for r in range(replicates)
            for cell in phase.CELLS
        ]
        # Replicate order, as the harness's stream reader requires; float repr round-trips.
        pd.concat(frames, ignore_index=True).to_csv(
            samples, index=False, compression="gzip", lineterminator=NEWLINE
        )
    write_csv(phase.truth_rows(phase.CELLS, replicates), scratch / "truth.csv")
    write_csv(phase.seed_rows(replicates), scratch / "seeds.csv")


def _fit_one(cell: str, replicate: int, directory: str) -> None:
    path = Path(directory) / f"{cell}-{replicate:04d}.csv"
    if path.exists():
        return
    try:
        rows = pd.DataFrame(phase.fit_replicate(cell, replicate), columns=list(phase.PHASE_COLUMNS))
    except Exception as error:  # a failed fit costs its replicate, and is recorded
        record = {"implementation": "cleverly", "scenario": cell, "replicate": replicate}
        record["message"] = f"{type(error).__name__}: {error}".replace("\n", " ")
        pd.DataFrame([record]).to_csv(path.with_suffix(".failure"), index=False)
        rows = pd.DataFrame(columns=list(phase.PHASE_COLUMNS))
    temporary = path.with_suffix(".tmp")
    write_csv(rows, temporary)
    temporary.replace(path)


def fit_python(scratch: Path, replicates: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every cleverly fit, checkpointed per cell and replicate."""
    directory = scratch / "python"
    directory.mkdir(exist_ok=True)
    keys = [(cell, r) for r in range(replicates) for cell in phase.CELLS]
    joblib.Parallel(n_jobs=jobs, verbose=5)(
        joblib.delayed(_fit_one)(cell, r, str(directory)) for cell, r in keys
    )
    frames = [
        pd.read_csv(directory / f"{cell}-{r:04d}.csv", float_precision="round_trip")
        for cell, r in keys
    ]
    rows = pd.concat([frame for frame in frames if len(frame)], ignore_index=True)
    failed = sorted(directory.glob("*.failure"))
    failures = pd.concat([pd.read_csv(path) for path in failed]) if failed else pd.DataFrame()
    return rows.loc[:, list(phase.PHASE_COLUMNS)], failures


def run_reference(scratch: Path, cores: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``concrete`` on the same draws, in the pinned image."""
    output = scratch / "concrete.csv"
    if not output.exists():
        (scratch / "r-checkpoints").mkdir(exist_ok=True)
        REFERENCE.run(
            HERE,
            scratch / "samples.csv.gz",
            scratch / "truth.csv",
            output,
            cores=cores,
            env={
                "CLEVERLY_R_SEEDS": "/work/seeds.csv",
                "CLEVERLY_CHECKPOINT": "/work/r-checkpoints",
            },
        )
    rows = phase.read_reference(output)
    recorded = failures_path(output)
    failures = pd.read_csv(recorded) if recorded.exists() else pd.DataFrame()
    if len(failures):
        failures.insert(0, "implementation", phase.REFERENCE)
    return rows, failures


def publish(scratch: Path, rows: pd.DataFrame, failures: pd.DataFrame, cores: int) -> Path:
    """The phase's artifacts and manifest, under ``SCRATCH/artifacts``."""
    artifacts = scratch / "artifacts"
    shutil.rmtree(artifacts, ignore_errors=True)
    artifacts.mkdir()
    written = [artifacts / "replicates.parquet"]
    write_rows(rows, written[0])
    for name, table in phase.tables(rows).items():
        write_csv(table, artifacts / name)
        written.append(artifacts / name)
    if len(failures):
        write_csv(failures, artifacts / "failures.csv")
        written.append(artifacts / "failures.csv")
    versions_file = scratch / "concrete-versions.txt"
    manifest: dict[str, Any] = {
        "declaration": dict(phase.declaration()),
        "generated_with": {
            "subject": {"implementation": phase.SUBJECT, **provenance()},
            "reference": {
                **phase.declaration()["reference"],
                "image_content": image_content(phase.IMAGE),
                "versions": (
                    versions_file.read_text(encoding="utf-8").splitlines()
                    if versions_file.exists()
                    else []
                ),
                "r_cores": cores,
            },
        },
        "reference_sha256": reference_hashes(ROOT / name for name in phase.REFERENCE_SOURCES),
        "study_module_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in phase.MODULES
        },
        "sha256": hashes(written),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + NEWLINE
    (artifacts / "manifest.json").write_bytes(text.encode("utf-8"))
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=max(1, available_cores() - 1))
    parser.add_argument("--r-cores", type=int, default=max(1, available_cores() - 1))
    parser.add_argument("--replicates", type=int, default=phase.REPLICATES)
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args(argv)
    scratch = options.output.resolve()
    if scratch.is_relative_to(ROOT.resolve()):
        parser.error("--output is a scratch directory outside the repository")
    if options.publish and options.replicates != phase.REPLICATES:
        parser.error(f"a published run fits the declared {phase.REPLICATES} replicates")
    scratch.mkdir(parents=True, exist_ok=True)
    claim(scratch, options.replicates)
    write_inputs(scratch, options.replicates)
    python_rows, python_failures = fit_python(scratch, options.replicates, options.jobs)
    reference_rows, reference_failures = run_reference(scratch, options.r_cores)
    rows = pd.concat([python_rows, reference_rows], ignore_index=True)
    rows = rows.sort_values(["scenario", "replicate", "implementation", "estimand"], kind="stable")
    failures = pd.concat([python_failures, reference_failures], ignore_index=True)
    gaps = coverage_gaps(rows, failures, options.replicates)
    if gaps:
        raise SystemExit(f"{len(gaps)} declared fits have neither rows nor a failure: {gaps[:10]}")
    artifacts = publish(scratch, rows.reset_index(drop=True), failures, options.r_cores)
    print(phase.tables(rows)["verdicts.csv"].to_string(index=False))
    if options.publish:
        # A file an earlier publish left (a failures.csv, say) must not outlive this one.
        for path in HERE.iterdir():
            if path.is_file() and path.suffix not in (".py", ".R"):
                path.unlink()
        for name in (*phase.TABLES, "replicates.parquet", "failures.csv", "manifest.json"):
            if (artifacts / name).exists():
                shutil.copy2(artifacts / name, HERE / name)
        print(f"copied the artifacts into {HERE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
