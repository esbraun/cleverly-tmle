"""Run the hte3 comparison of the EP curve (P1 plan, PR 4b), or its smoke.

The cells, the draws, the readings and the sizing rule are those of :mod:`tests.studies.ep_hte3`.
For each cell the driver runs two phases.  The Python phase re-draws the parent group's leading
replicates, refits ``cleverly``'s EP as the parent did, and exports the nuisances
(:func:`tests.studies.ep_hte3.python_phase`).  The R phase writes them as an inputs archive and
runs ``hte3_ep_runner.R`` in the image ``cleverly-hte3:eac64673`` on chunks of
:data:`CHUNK` replicates.  Each replicate is then reduced to one row of ``replicates.csv.gz``,
with the numpy emulation's agreement with the R curve beside it (the two gates of
:func:`tests.studies.ep_hte3.emulation_agreement` and the largest per-point gap).  The
mutation controls M1 and M2 reuse the first 20 exports of E8 and C6 with ``pi.hat`` set to 0.5.

**A declared run** passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``.  It refuses to start while
:data:`tests.studies.ep_hte3.REPLICATES` is empty (the comparison is pending its declaration),
and it refuses under the shared run guard (:func:`tests.diagnostics.rm18_shared.refusals`: a
clean tree, ``HEAD`` equal to its pushed upstream, ``cleverly`` from this tree) and unless every
thread variable is 1.  It writes :data:`tests.studies.ep_hte3.RUN_FILES` (and ``failures.csv``
when a replicate was lost) into ``--output`` and copies them into this directory.  The
prediction files are not kept; ``manifest.json`` records a SHA-256 of each cell's predictions.

**Checkpoints** live under ``~/.cache/cleverly/runs/hte3-ep-<resume key>/``, which the driver
prints: one per Python replicate (:func:`tests.studies.evidence.replication.fit_replicates`) and
one per cell and R chunk.  A stopped run resumes from them when it is started again with a new
empty ``--output``; ``--fresh`` discards them first.

**A smoke** passes ``--smoke N --output <scratch>``.  It draws ``N`` replicates of every reading
cell on the ``hte3_smoke`` stream, which no parent and no declared run draws, and the mutation
controls on the first ``N`` of E8 and C6.  It needs no clean tree.  It writes
:data:`tests.studies.ep_hte3.SMOKE_FILES`: per replicate the R-against-emulation agreement over
all 20,000 points (both gates, the largest per-point gap, the same gates for the emulation
without hte3's truncation, which is the mutation witness on C6), hte3's truncation cutoffs
against the emulation's, and the warnings and glmnet fits of the EP step; the first
:data:`tests.studies.ep_hte3.SMOKE_POINTS` points of both curves; Sieve's index matrix per
cell; seconds per fit and the re-costing against the pilot
(:func:`tests.studies.ep_hte3.recost`); and the R session.  The local runner commits them under
``tests/canonical/hte3_ep/smoke/``, which the design test reads.

A red reading is published red; :mod:`tests.studies.ep_hte3` holds the route it takes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.canonical.declared_run import outside, thread_refusals
from tests.canonical.regenerate import Reference, digest, image_content
from tests.diagnostics.rm18_shared import refusals, run_log, write_table
from tests.studies import ep_hte3 as study
from tests.studies import ep_hte3_pilot as emulation
from tests.studies.evidence import replication
from tests.studies.evidence.manifest import provenance, reference_hashes
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent

#: The container: the runner and the harness it sources are mounted from ``tests/canonical``.
REFERENCE = Reference(
    image=study.IMAGE,
    runner="hte3_ep/hte3_ep_runner.R",
    mount_runner=True,
    runner_root=CANONICAL,
    build_context=HERE,
)

IMPLEMENTATION = "cleverly-ep-hte3"

#: Replicates per R container.
CHUNK = 25

#: The modules whose bytes produced the rows, recorded in the manifest.
MODULES = (
    "tests/studies/ep_hte3.py",
    "tests/studies/ep_hte3_pilot.py",
    "tests/studies/ep_cate.py",
    "tests/studies/ep_cate_properties.py",
    "tests/studies/ep_crr.py",
    "tests/studies/ep_crr_properties.py",
    "tests/studies/conditional_projection.py",
    "tests/canonical/hte3_ep/regenerate.py",
)

#: The reference sources the fast tier gates against the tree.
SOURCES = (study.DOCKERFILE, study.RUNNER, study.HARNESS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resume_key(smoke: int | None) -> str:
    """What a resumed run must share with the run whose checkpoints it reuses."""
    return digest(
        {
            "commit": provenance()["cleverly_commit"],
            "smoke": smoke,
            "replicates": dict(study.REPLICATES),
            "sources": reference_hashes(SOURCES),
            "modules": reference_hashes(ROOT / name for name in MODULES),
        }
    )[:16]


def checkpoint_root(smoke: int | None, *, fresh: bool) -> Path:
    root = Path.home() / ".cache" / "cleverly" / "runs" / f"hte3-ep-{resume_key(smoke)}"
    if fresh and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints: {root}", flush=True)
    return root


# ---------------------------------------------------------------- the two phases


def python_phase(
    cell: study.Cell, replicates: Sequence[int], stream: str, *, jobs: int, root: Path
) -> tuple[list[study.Export], pd.DataFrame]:
    """The Python exports of ``cell`` and the records of the replicates that failed."""
    keys = [(cell.label, int(r), stream) for r in replicates]
    outcomes = replication.fit_replicates(
        keys,
        study.export_key,
        n_jobs=jobs,
        label=f"{cell.label}-{stream}",
        implementation=IMPLEMENTATION,
        cell_of=lambda key: str(key[0]),
        replicate_of=lambda key: int(key[1]),
        draw=study.draw_key,
        seed_of=lambda key: study.draw_seed(study.CELL_BY_LABEL[key[0]], key[1], key[2]),
        checkpoint=root / "python",
    )
    return list(outcomes.values), outcomes.failures


def _theta_by_replicate(path: Path) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    out = {}
    for replicate, rows in frame.groupby("replicate", sort=True):
        rows = rows.sort_values("point")
        out[int(replicate)] = rows["theta"].to_numpy(dtype=float)
    return out


def r_chunk(
    cell: study.Cell, exports: Sequence[study.Export], work: Path, *, jobs: int
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, dict[str, Any]], pd.DataFrame | None]:
    """Run hte3 on one chunk.

    Returns the curve, the seconds and the runner's per-replicate record (truncation extremes,
    warnings, glmnet fits) of each fitted replicate, and any failure records.
    """
    work.mkdir(parents=True, exist_ok=True)
    name = f"{cell.label}-{exports[0].replicate:03d}-{exports[-1].replicate:03d}"
    inputs, points, output = (
        work / f"{name}.inputs.csv.gz",
        work / f"{cell.label}.points.csv.gz",
        work / f"{name}.theta.csv",
    )
    study.write_inputs(study.input_frame(cell, exports), inputs)
    if not points.exists():
        study.write_inputs(study.points_frame(cell), points)
    REFERENCE.run(HERE, inputs, points, output, cores=jobs)
    frame = pd.read_csv(output)
    seconds = frame.groupby("replicate")["seconds"].first().to_dict()
    failures_file = output.with_name(f"{output.stem}.failures.csv")
    failures = pd.read_csv(failures_file) if failures_file.exists() else None
    fits: dict[int, dict[str, Any]] = {}
    for path in sorted(output.with_name(f"{output.stem}.fits").glob("*.csv")):
        record = pd.read_csv(path).iloc[0].to_dict()
        fits[int(record["replicate"])] = record
    theta = _theta_by_replicate(output)
    return theta, {int(k): float(v) for k, v in seconds.items()}, fits, failures


EXTREMES = ("pi1_low", "pi1_high", "pi0_low", "pi0_high")


def agreement_row(
    cell: study.Cell,
    export: study.Export,
    curve: np.ndarray,
    emulated: np.ndarray,
    fit: dict[str, Any],
    target: np.ndarray,
    points: np.ndarray,
) -> dict[str, Any]:
    """One replicate's R-against-emulation record (the smoke's ``agreement.csv``)."""
    gates = study.emulation_agreement(target, curve, emulated, export.ise_oracle)
    witness = study.emulation_agreement(
        target, curve, emulation.emulate(cell, export, points, truncate=False), export.ise_oracle
    )
    extremes = emulation.truncation_extremes(export)
    return {
        "cell": cell.label,
        "replicate": export.replicate,
        "points": int(curve.size),
        "log_ise_gap": gates["log_ise_gap"],
        "rms_ratio": gates["rms_ratio"],
        "max_abs_difference": gates["max_gap"],
        "max_abs_theta": float(np.max(np.abs(curve))),
        "untruncated_log_ise_gap": witness["log_ise_gap"],
        "untruncated_rms_ratio": witness["rms_ratio"],
        "cutoff_gap": max(abs(float(fit[name]) - extremes[name]) for name in EXTREMES),
        **{f"r_{name}": float(fit[name]) for name in EXTREMES},
        **{f"emulation_{name}": extremes[name] for name in EXTREMES},
        "warnings": int(fit["warnings"]),
        "glmnet_warnings": int(fit["glmnet_warnings"]),
        "glmnet_fits": int(fit["glmnet_fits"]),
        "glmnet_npasses": float(fit["glmnet_npasses"]),
    }


def run_cell(
    cell: study.Cell,
    exports: Sequence[study.Export],
    *,
    jobs: int,
    root: Path,
    work: Path,
    committed: study.Committed | None,
    keep_points: int = 0,
) -> dict[str, Any]:
    """The R phase and the reduction of one cell, checkpointed per chunk."""
    rows: list[dict[str, Any]] = []
    digests: list[str] = []
    timings: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []
    kept: list[pd.DataFrame] = []
    failures: list[pd.DataFrame] = []
    points = study.ise_points(cell)
    target = study.truth(cell, points)
    for start in range(0, len(exports), CHUNK):
        chunk = list(exports[start : start + CHUNK])
        name = f"{cell.label}-{chunk[0].replicate:03d}-{chunk[-1].replicate:03d}"
        saved = root / "r" / f"{name}.json"
        if saved.exists():
            record = json.loads(saved.read_text(encoding="utf-8"))
            print(f"{name}: reusing its checkpoint", flush=True)
        else:
            theta, seconds, fits, lost = r_chunk(cell, chunk, work / cell.label, jobs=jobs)
            record = {"rows": [], "digests": [], "timings": [], "agreement": [], "points": []}
            record["failures"] = None if lost is None else lost.to_dict(orient="records")
            for export in chunk:
                if export.replicate not in theta:
                    continue
                curve = theta[export.replicate]
                emulated = emulation.emulate(cell, export, points)
                record["rows"].append(
                    study.reduce(cell, export, curve, committed=committed, emulated=emulated)
                )
                record["digests"].append(study.prediction_digest(curve))
                record["timings"].append(
                    {
                        "cell": cell.label,
                        "replicate": export.replicate,
                        "python_seconds": export.seconds,
                        "r_seconds": seconds[export.replicate],
                    }
                )
                if export.replicate not in fits:
                    raise RuntimeError(
                        f"{name}: replicate {export.replicate} has a curve and no runner record"
                    )
                record["agreement"].append(
                    agreement_row(
                        cell, export, curve, emulated, fits[export.replicate], target, points
                    )
                )
                if keep_points:
                    record["points"].append(
                        {
                            "replicate": export.replicate,
                            "theta_r": curve[:keep_points].tolist(),
                            "theta_emulation": emulated[:keep_points].tolist(),
                        }
                    )
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(json.dumps(record), encoding="utf-8")
        rows += record["rows"]
        digests += record["digests"]
        timings += record["timings"]
        agreement += record["agreement"]
        if record["failures"]:
            failures.append(pd.DataFrame(record["failures"]).assign(implementation="hte3"))
        for kept_row in record.get("points", []):
            kept.append(
                pd.DataFrame(
                    {
                        "cell": cell.label,
                        "replicate": kept_row["replicate"],
                        "point": np.arange(1, len(kept_row["theta_r"]) + 1),
                        "theta_r": kept_row["theta_r"],
                        "theta_emulation": kept_row["theta_emulation"],
                    }
                )
            )
    return {
        "rows": rows,
        "digest": study.cell_digest(digests),
        "timings": timings,
        "agreement": agreement,
        "points": kept,
        "failures": failures,
    }


def sidecars(work: Path) -> tuple[str, pd.DataFrame]:
    """The R session text of the first chunk, and every chunk's Sieve index matrices."""
    sessions = sorted(work.rglob("*.theta.session.txt"))
    session = sessions[0].read_text(encoding="utf-8") if sessions else ""
    frames = [pd.read_csv(path) for path in sorted(work.rglob("*.theta.index.csv"))]
    index = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
    return session, index


# ---------------------------------------------------------------- the run


def run(
    *,
    smoke: int | None,
    output: Path,
    jobs: int,
    fresh: bool,
) -> dict[str, Any]:
    """Every cell's two phases; returns the assembled tables."""
    root = checkpoint_root(smoke, fresh=fresh)
    work = output / "_work"
    stream = study.PROPERTY_STREAM if smoke is None else study.SMOKE_STREAM
    committed = study.Committed.load() if smoke is None else None
    results: dict[str, dict[str, Any]] = {}
    python_failures: list[pd.DataFrame] = []
    started = time.time()
    for cell in study.READING_CELLS:
        count = study.REPLICATES[cell.label] if smoke is None else smoke
        exports, lost = python_phase(cell, range(count), stream, jobs=jobs, root=root)
        if len(lost):
            python_failures.append(lost)
        results[cell.label] = run_cell(
            cell,
            exports,
            jobs=jobs,
            root=root,
            work=work,
            committed=committed,
            keep_points=study.SMOKE_POINTS if smoke is not None else 0,
        )
        for mutation in study.MUTATION_CELLS:
            if mutation.mutation_of != cell.label:
                continue
            count = study.REPLICATES[mutation.label] if smoke is None else smoke
            mutated = [export.mutated(mutation.label) for export in exports[:count]]
            results[mutation.label] = run_cell(
                mutation,
                mutated,
                jobs=jobs,
                root=root,
                work=work,
                committed=None,
                keep_points=study.SMOKE_POINTS if smoke is not None else 0,
            )
        print(f"{cell.label}: done at {time.time() - started:.0f} s", flush=True)
    rows = pd.DataFrame(
        [row for cell in study.CELLS for row in results[cell.label]["rows"]],
        columns=list(study.REPLICATE_COLUMNS),
    )
    failures = [frame for result in results.values() for frame in result["failures"]]
    failures += python_failures
    session, index = sidecars(work)
    return {
        "rows": rows,
        "digests": {label: result["digest"] for label, result in results.items()},
        "timings": pd.DataFrame([t for result in results.values() for t in result["timings"]]),
        "agreement": pd.DataFrame([a for result in results.values() for a in result["agreement"]]),
        "points": pd.concat(
            [frame for result in results.values() for frame in result["points"]]
            or [pd.DataFrame()],
            ignore_index=True,
        ),
        "failures": pd.concat(failures, ignore_index=True) if failures else None,
        "session": session,
        "index": index,
    }


def manifest(result: dict[str, Any], *, smoke: int | None) -> dict[str, Any]:
    session = result["session"].splitlines()
    return {
        "comparison": "ep-hte3",
        "kind": "different estimator",
        "status": "smoke" if smoke is not None else "run",
        "commits": {
            "hte3": study.HTE3_COMMIT,
            "sl3": study.SL3_COMMIT,
            "tmle3": study.TMLE3_COMMIT,
        },
        "image": study.IMAGE,
        "image_content": image_content(study.IMAGE),
        "reference_sha256": reference_hashes(SOURCES),
        "module_sha256": reference_hashes(ROOT / name for name in MODULES),
        "r_versions": session[:7],
        "provenance": provenance(),
        "replicates": dict(study.REPLICATES) if smoke is None else smoke,
        "reported": sorted(study.REPORTED),
        "prediction_sha256": result["digests"],
        "lost_replicates": 0 if result["failures"] is None else len(result["failures"]),
    }


def write_run(result: dict[str, Any], output: Path) -> list[str]:
    write_table(result["rows"], output / "replicates.csv.gz")
    # Lost replicates are counted against the declared ones, never refilled; a cell that lost
    # more than 5% is reported (``study.readings``).
    readings = study.readings(result["rows"], declared=study.REPLICATES)
    write_table(readings, output / "comparison.csv")
    written = ["replicates.csv.gz", "comparison.csv"]
    if result["failures"] is not None:
        write_table(result["failures"], output / "failures.csv")
        written.append("failures.csv")
    text = json.dumps(manifest(result, smoke=None), indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(text, encoding="utf-8")
    return [*written, "manifest.json"]


def write_smoke(result: dict[str, Any], output: Path) -> None:
    agreement = result["agreement"]
    recost = study.recost(result["timings"])
    write_table(agreement, output / "agreement.csv")
    write_table(result["points"], output / "points.csv")
    write_table(result["index"], output / "sieve-index.csv")
    write_table(result["timings"], output / "timings.csv")
    write_table(recost, output / "recost.csv")
    (output / "session.txt").write_text(result["session"], encoding="utf-8")
    write_table(result["rows"], output / "replicates.csv")
    for column, gate in (
        ("log_ise_gap", study.EMULATION_LOG_ISE_GATE),
        ("rms_ratio", study.EMULATION_RMS_GATE),
        ("cutoff_gap", study.CUTOFF_TOLERANCE),
    ):
        worst = float(agreement[column].max())
        verdict = "within" if worst <= gate else "OUTSIDE"
        share = worst / gate
        note = "" if share <= study.GATE_USE_LIMIT else "; above the 10% share: re-declare the gate"
        print(f"{column}: worst {worst:.3e}, {verdict} {gate:g} ({share:.1%} of it{note})")
    print(f"largest per-point gap (reported): {agreement['max_abs_difference'].max():.3e}")
    c6 = agreement.loc[agreement["cell"] == "C6"]
    moved = (c6["untruncated_log_ise_gap"] > study.EMULATION_LOG_ISE_GATE) | (
        c6["untruncated_rms_ratio"] > study.EMULATION_RMS_GATE
    )
    print(f"C6 without the truncation exceeds a gate in {int(moved.sum())} of {len(c6)} replicates")
    print(
        f"EP-step warnings: {int(agreement['warnings'].sum())}, naming glmnet or convergence: "
        f"{int(agreement['glmnet_warnings'].sum())}; glmnet fits per replicate: "
        f"{sorted(agreement['glmnet_fits'].unique().tolist())}"
    )
    print(recost.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    ratio = float(recost["ratio"].max())
    stop = ratio > study.PILOT_TIME_FACTOR_LIMIT
    print(
        f"Python seconds per replicate: at most {ratio:.2f} times the pilot's"
        + ("; STOP: above 3, report before any declared run" if stop else ""),
        flush=True,
    )
    for cell in study.MUTATION_CELLS:
        rows = result["rows"].loc[result["rows"]["cell"] == cell.label]
        ratio = rows["log_ise_hte3"] - rows["log_ise_oracle"]
        print(f"{cell.label}: hte3 oracle comparison {ratio.round(3).tolist()}", flush=True)
    if result["failures"] is not None:
        print(f"lost replicates:\n{result['failures'].to_string()}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    kind = "declared" if arguments.smoke is None else "smoke"
    output = outside(arguments.output, kind)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refused: the scratch --output {output} is not empty")
    if arguments.smoke is None:
        if not study.REPLICATES:
            raise SystemExit(
                "refused: the comparison is pending its declaration; tests/studies/ep_hte3.py "
                "sets REPLICATES and REPORTED from design_numbers() in the declaration commit"
            )
        refused = refusals(False) + thread_refusals()
        if refused:
            raise SystemExit("refused:\n  " + "\n  ".join(refused))
    elif arguments.smoke < 1 or arguments.smoke > study.MUTATION_REPLICATES:
        raise SystemExit(f"refused: --smoke takes 1 to {study.MUTATION_REPLICATES} replicates")
    output.mkdir(parents=True, exist_ok=True)
    title = "ep-hte3 declared run" if arguments.smoke is None else "ep-hte3 smoke"
    with run_log(output, title):
        result = run(
            smoke=arguments.smoke, output=output, jobs=arguments.jobs, fresh=arguments.fresh
        )
        shutil.rmtree(output / "_work", ignore_errors=True)
        if arguments.smoke is not None:
            write_smoke(result, output)
            return
        written = write_run(result, output)
    if "failures.csv" not in written:
        (HERE / "failures.csv").unlink(missing_ok=True)
    for name in [*written, "run.log"]:
        shutil.copyfile(output / name, HERE / name)
    print(f"copied {len(written) + 1} files from {output} to {HERE}", flush=True)
    readings = pd.read_csv(HERE / "comparison.csv")
    print(readings.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
