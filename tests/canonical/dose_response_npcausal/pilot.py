"""The local R pilot of ``dose-response-npcausal`` (X93 S2): real ``ctseff`` on the pilot draws.

It needs Docker and nothing else: the image is built from
``tests/canonical/npcausal_incremental`` on first use (``Reference.run``), and every fit goes
through the declared runner ``run_study.R`` and the shared harness, the code path of the
declared run.  Each ``ctseff`` call is its own harness group: the pilot gives replicate
``4 r + k`` the three doses of candidate bandwidth ``k`` of pilot draw ``r``, so a call that
errors inside ``study_stage`` is one failure record with R's own message.  The draws are the
Python pilot's (:func:`tests.studies.dose_response_npcausal_pilot.draw`), on seed streams no
declared cell reads.

Two commands, in order, each with a scratch directory outside the repository::

    python -m tests.canonical.dose_response_npcausal.pilot probe --scratch <dir> --jobs <n>
    python -m tests.canonical.dose_response_npcausal.pilot reference --scratch <dir> --jobs <n>

``probe`` runs smoke steps S1, S2, S4, S6 and S7 of the design review on constructed draws and
the first :data:`PROBE_DRAWS` pilot draws, prints a verdict per step, writes
``<scratch>/probe/report.json`` and prints the S8 command.  It writes nothing in the
repository.  ``reference`` fits all 1,000 pilot draws at every candidate bandwidth in resumable
chunks of :data:`CHUNK` draws (a rerun skips each finished chunk), then commits
``tests/studies/dose_response_npcausal_pilot_r.csv.gz`` (one row per draw and estimand),
``..._pilot_r_failures.csv`` (one row per failed call, with R's message) and
``..._pilot_r.json`` (image, source hashes, cores, wall time), and prints the declaration
:func:`~tests.studies.dose_response_npcausal_pilot.declaration_from_pilot` reads off them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.canonical.dose_response_npcausal.regenerate import REFERENCE
from tests.canonical.regenerate import build_image, failures_path, git_state, image_content
from tests.studies import dose_response_npcausal as study
from tests.studies import dose_response_npcausal_pilot as pilot
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import reference_seed

HERE = Path(__file__).resolve().parent
CANDIDATES = study.CANDIDATE_BANDWIDTHS
#: Pilot draws a container run fits; a finished chunk is never refitted.
CHUNK = 50
#: Pilot draws the probe's agreement step (S4) fits.
PROBE_DRAWS = 20
#: The probe's agreement criteria (review smoke step S4), in units of the R standard error.
AGREEMENT_ESTIMATE = 0.05
AGREEMENT_SE = 0.01
#: How far the probe's constructed draws isolate the lowest or the highest dose.
ISOLATION = 2.5
SAMPLE_COLUMNS = ["scenario", "replicate", *study.COVARIATES, "A", "Y"]
ROW_COLUMNS = [
    "replicate",
    "implementation",
    "estimand",
    "dose",
    "bandwidth",
    "truth",
    "estimate",
    "std_error",
    "ci_lower",
    "ci_upper",
    "covered",
]
FAILURE_COLUMNS = ["replicate", "bandwidth", "group_id", "class", "stage", "message"]


def _identifier(draw: int, k: int) -> int:
    return draw * len(CANDIDATES) + k


def _names(h: float) -> list[str]:
    return [name for name, (_, width, _) in study.PILOT_PAIRS.items() if width == h]


def write_inputs(directory: Path, frames: dict[int, pd.DataFrame]) -> None:
    """The runner's three inputs for ``frames`` (pilot draw -> sample), every candidate each."""
    directory.mkdir(parents=True, exist_ok=True)
    samples, truths, seeds = [], [], []
    for draw_id, frame in sorted(frames.items()):
        for k, h in enumerate(CANDIDATES):
            identifier = _identifier(draw_id, k)
            sample = frame.copy()
            sample.insert(0, "replicate", identifier)
            sample.insert(0, "scenario", study.SCENARIO)
            samples.append(sample)
            truths += [
                {
                    "scenario": study.SCENARIO,
                    "replicate": identifier,
                    "estimand": name,
                    "truth": study.TRUTH[name],
                }
                for name in _names(h)
            ]
            seeds += [
                {
                    "scenario": scenario,
                    "replicate": identifier,
                    "seed": reference_seed(study.STUDY, "run_study-pilot", scenario, identifier),
                }
                for scenario in (study.SCENARIO, "*")
            ]
    pd.concat(samples, ignore_index=True)[SAMPLE_COLUMNS].to_csv(
        directory / "samples.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    pd.DataFrame(truths).to_csv(directory / "truth.csv", index=False)
    pd.DataFrame(seeds).to_csv(directory / "seeds.csv", index=False)


def run_container(directory: Path, cores: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run ``run_study.R`` on ``directory``'s inputs: the decoded rows, failures and wall time."""
    output = directory / "results.csv"
    start = time.perf_counter()
    REFERENCE.run(
        HERE,
        directory / "samples.csv.gz",
        directory / "truth.csv",
        output,
        cores=cores,
        env={"CLEVERLY_R_SEEDS": "/work/seeds.csv"},
    )
    wall = time.perf_counter() - start
    return decode_rows(output), decode_failures(failures_path(output)), wall


def decode_rows(path: Path) -> pd.DataFrame:
    """The runner's rows with the pilot draw and the triweight bandwidth restored."""
    rows = pd.read_csv(path, float_precision="round_trip")
    k = rows["replicate"] % len(CANDIDATES)
    rows["bandwidth"] = [CANDIDATES[int(index)] for index in k]
    rows["replicate"] = rows["replicate"] // len(CANDIDATES)
    rows["dose"] = [study.PILOT_PAIRS[name][0] for name in rows["estimand"]]
    return rows[ROW_COLUMNS].sort_values(["replicate", "bandwidth", "dose"], ignore_index=True)


def decode_failures(path: Path) -> pd.DataFrame:
    """The harness's failure records, one per failed call, with the draw and bandwidth."""
    if not path.exists():
        return pd.DataFrame(columns=FAILURE_COLUMNS)
    failures = pd.read_csv(path, dtype={"replicates": str})
    identifier = failures["replicates"].astype(int)
    failures["bandwidth"] = [CANDIDATES[int(index) % len(CANDIDATES)] for index in identifier]
    failures["replicate"] = identifier // len(CANDIDATES)
    return failures[FAILURE_COLUMNS].sort_values(["replicate", "bandwidth"], ignore_index=True)


def _sketch(frame: pd.DataFrame) -> dict[float, pd.DataFrame | str]:
    """The transcription's outcome per candidate: its rows, or its refusal text."""
    out: dict[float, pd.DataFrame | str] = {}
    try:
        inputs = pilot.ctseff_inputs(frame)
    except pilot.CtseffRefusal as error:
        return dict.fromkeys(CANDIDATES, str(error))
    for h in CANDIDATES:
        try:
            out[h] = pilot.ctseff_fit(inputs, study.NPCAUSAL_BANDWIDTHS[h], study.DOSES)
        except pilot.CtseffRefusal as error:
            out[h] = str(error)
    return out


def _reread(directory: Path) -> dict[int, pd.DataFrame]:
    """The samples exactly as written, so the transcription fits the bytes R reads."""
    samples = pd.read_csv(directory / "samples.csv.gz", float_precision="round_trip")
    first = samples.loc[samples["replicate"] % len(CANDIDATES) == 0]
    return {
        int(identifier) // len(CANDIDATES): group[[*study.COVARIATES, "A", "Y"]].reset_index(
            drop=True
        )
        for identifier, group in first.groupby("replicate")
    }


def compare(
    frames: dict[int, pd.DataFrame], rows: pd.DataFrame, failures: pd.DataFrame
) -> pd.DataFrame:
    """Per call: R's outcome, the transcription's, and their agreement in R standard errors."""
    records = []
    for draw_id, frame in frames.items():
        sketch = _sketch(frame)
        for h in CANDIDATES:
            failed = failures.loc[(failures["replicate"] == draw_id) & (failures["bandwidth"] == h)]
            r_rows = rows.loc[(rows["replicate"] == draw_id) & (rows["bandwidth"] == h)]
            outcome = sketch[h]
            record: dict[str, Any] = {
                "draw": draw_id,
                "bandwidth": h,
                "r_failure": str(failed["message"].iloc[0]) if len(failed) else "",
                "sketch_failure": outcome if isinstance(outcome, str) else "",
                "estimate_gap": math.nan,
                "se_ratio_gap": math.nan,
            }
            if len(r_rows) and not isinstance(outcome, str):
                merged = r_rows.merge(outcome, on="dose")
                record["estimate_gap"] = float(
                    (
                        (merged["estimate_x"] - merged["estimate_y"]).abs() / merged["std_error"]
                    ).max()
                )
                record["se_ratio_gap"] = float(
                    (merged["std_error"] / merged["se"] - 1.0).abs().max()
                )
            records.append(record)
    return pd.DataFrame(records)


def _rscript(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "Rscript", study.NPCAUSAL_IMAGE, "-e", code],
        capture_output=True,
        text=True,
    )


def _constructed() -> dict[int, pd.DataFrame]:
    """Pilot draw 0 unchanged, then with its lowest and its highest dose isolated by 2.5."""
    frame = pilot.draw(0)
    a = frame["A"].to_numpy()
    order = np.argsort(a)
    low, high = frame.copy(), frame.copy()
    low.loc[order[0], "A"] = a[order[1]] - ISOLATION
    high.loc[order[-1], "A"] = a[order[-2]] + ISOLATION
    return {0: frame, 1: low, 2: high}


def probe(scratch: Path, cores: int) -> int:
    """Smoke steps S1, S2, S4, S6, S7 and S8; returns the exit status."""
    root = scratch / "probe"
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    verdicts: dict[str, bool] = {}

    print("S1 image and packages", flush=True)
    build_image(study.NPCAUSAL_IMAGE, REFERENCE.build_context or HERE)
    check = _rscript(
        'for (p in c("SuperLearner", "earth", "gam", "ranger", "KernSmooth", "nnls")) '
        "stopifnot(requireNamespace(p, quietly = TRUE)); "
        'cat(as.character(packageVersion("npcausal")), R.version.string, sep = "\\n")'
    )
    report["S1"] = {"stdout": check.stdout, "stderr": check.stderr[-2000:]}
    verdicts["S1"] = check.returncode == 0 and check.stdout.startswith(study.NPCAUSAL_VERSION)
    print(check.stdout.strip(), flush=True)

    print("S2 the B1 mechanism", flush=True)
    mechanism = _rscript(
        "library(KernSmooth); set.seed(1); a <- c(rnorm(4000, 2.5, 1.64), -8); "
        "y <- a + rnorm(4001); r <- locpoly(a, y, bandwidth = 0.3457676162028392); "
        'cat("end node NaN:", is.nan(r$y[1]), "\\n"); '
        'cat("approx at min(a) NA:", is.na(approx(r, xout = min(a))$y), "\\n"); '
        "h <- 0.5[which.min(NA_real_)]; "
        'cat("zero-length bandwidth:", tryCatch({locpoly(a, y, bandwidth = h); "no error"}, '
        'error = function(e) conditionMessage(e)), "\\n")'
    )
    print(mechanism.stdout.strip(), mechanism.stderr.strip()[-500:], flush=True)
    constructed = root / "constructed"
    write_inputs(constructed, _constructed())
    rows, failures, _ = run_container(constructed, cores)
    calls = compare(_reread(constructed), rows, failures)
    calls["draw"] = calls["draw"].map({0: "unchanged", 1: "lowest isolated", 2: "highest isolated"})
    print(calls[["draw", "bandwidth", "r_failure", "sketch_failure"]].to_string(), flush=True)
    report["S2"] = {"mechanism": mechanism.stdout, "calls": calls.to_dict("records")}
    verdicts["S2 R fails where the transcription fails"] = bool(
        ((calls["r_failure"] != "") == (calls["sketch_failure"] != "")).all()
    )
    report["S2"]["b1_confirmed"] = bool((calls["r_failure"] != "").any())

    print(f"S4 agreement on pilot draws 0-{PROBE_DRAWS - 1}", flush=True)
    agreement = root / "agreement"
    write_inputs(agreement, {r: pilot.draw(r) for r in range(PROBE_DRAWS)})
    rows, failures, wall = run_container(agreement, cores)
    calls = compare(_reread(agreement), rows, failures)
    both = calls.dropna(subset=["estimate_gap"])
    report["S4"] = {"calls": calls.to_dict("records")}
    print(
        f"calls {len(calls)}, R failures {(calls['r_failure'] != '').sum()}, transcription "
        f"failures {(calls['sketch_failure'] != '').sum()}, largest estimate gap "
        f"{both['estimate_gap'].max():.4f} SE, largest SE-ratio gap {both['se_ratio_gap'].max():.4f}",
        flush=True,
    )
    verdicts["S4 failures agree"] = bool(
        ((calls["r_failure"] != "") == (calls["sketch_failure"] != "")).all()
    )
    verdicts["S4 estimates agree"] = bool((both["estimate_gap"] <= AGREEMENT_ESTIMATE).all())
    verdicts["S4 standard errors agree"] = bool((both["se_ratio_gap"] <= AGREEMENT_SE).all())

    print("S6 determinism: draws 0 and 1 again, from fresh inputs", flush=True)
    again = root / "determinism"
    write_inputs(again, {r: pilot.draw(r) for r in range(2)})
    run_container(again, cores)
    first = pd.read_csv(agreement / "results.csv", dtype=str)
    second = pd.read_csv(again / "results.csv", dtype=str)
    first = first.loc[first["replicate"].astype(int) < 2 * len(CANDIDATES)].reset_index(drop=True)
    verdicts["S6 identical rows"] = bool(first.equals(second.reset_index(drop=True)))

    seconds = wall * cores / len(calls)
    report["S7"] = {
        "wall_seconds": wall,
        "cores": cores,
        "calls": len(calls),
        "cpu_seconds_per_call": seconds,
        "pilot_cpu_hours": seconds * pilot.DRAWS * len(CANDIDATES) / 3600.0,
        "declared_reference_cpu_hours": seconds * 2 * study.PRIMARY_REPLICATES / 3600.0,
    }
    print(
        f"S7 {seconds:.1f} CPU-s per call; the R pilot needs about "
        f"{report['S7']['pilot_cpu_hours']:.1f} CPU-hours, the declared reference about "
        f"{report['S7']['declared_reference_cpu_hours']:.1f} at {study.PRIMARY_REPLICATES:,} "
        "replicates",
        flush=True,
    )
    print(
        "S8 after the declaration is final: python -m "
        "tests.canonical.dose_response_npcausal.regenerate --replicates 20 --output <scratch> "
        "--allow-failures --jobs <n>",
        flush=True,
    )
    report["verdicts"] = verdicts
    (root / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, held in verdicts.items():
        print(f"{'pass' if held else 'FAIL'}  {name}", flush=True)
    print(f"report: {root / 'report.json'}", flush=True)
    return 0 if all(verdicts.values()) else 1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(scratch: Path, cores: int, chunk: int, draws: int) -> int:
    """Fit every pilot draw at every candidate in resumable chunks, then commit the tables."""
    chunks = []
    for start in range(0, draws, chunk):
        directory = scratch / f"chunk-{start // chunk:03d}"
        done = directory / "done.json"
        if not done.exists():
            stop = min(start + chunk, draws)
            print(f"draws {start}-{stop - 1}", flush=True)
            write_inputs(directory, {r: pilot.draw(r) for r in range(start, stop)})
            _, _, wall = run_container(directory, cores)
            done.write_text(
                json.dumps({"wall_seconds": wall, "cores": cores, "draws": stop - start}),
                encoding="utf-8",
            )
        chunks.append(directory)
    rows = pd.concat([decode_rows(d / "results.csv") for d in chunks], ignore_index=True)
    failures = pd.concat(
        [decode_failures(failures_path(d / "results.csv")) for d in chunks], ignore_index=True
    )
    timing = [json.loads((d / "done.json").read_text(encoding="utf-8")) for d in chunks]
    cpu = sum(entry["wall_seconds"] * entry["cores"] for entry in timing)
    calls = draws * len(CANDIDATES)
    metadata = {
        "draws": draws,
        "candidate_bandwidths": list(CANDIDATES),
        "npcausal_bandwidths": {f"{h:g}": study.NPCAUSAL_BANDWIDTHS[h] for h in CANDIDATES},
        "image": study.NPCAUSAL_IMAGE,
        "image_content": image_content(study.NPCAUSAL_IMAGE),
        "dockerfile_sha256": _sha(REFERENCE.build_context / "Dockerfile")
        if REFERENCE.build_context
        else None,
        "runner_sha256": _sha(HERE / "run_study.R"),
        "harness_sha256": _sha(HERE.parent / "study_harness.R"),
        "git": git_state(),
        "cores": sorted({entry["cores"] for entry in timing}),
        "wall_seconds": sum(entry["wall_seconds"] for entry in timing),
        "cpu_seconds": cpu,
        "cpu_seconds_per_call": cpu / calls,
        "failed_calls": {f"{h:g}": int((failures["bandwidth"] == h).sum()) for h in CANDIDATES},
    }
    print(json.dumps({key: metadata[key] for key in ("failed_calls", "cpu_seconds_per_call")}))
    if draws != pilot.DRAWS:
        print(f"{draws} draws is not the declared pilot; nothing is committed", flush=True)
        return 0
    rows.to_csv(pilot.R_OUTPUT, index=False, compression={"method": "gzip", "mtime": 0})
    failures.to_csv(pilot.R_FAILURES, index=False)
    pilot.R_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    declaration = pilot.declaration_from_pilot()
    print(f"wrote {pilot.R_OUTPUT}, {pilot.R_FAILURES} and {pilot.R_METADATA}")
    print(f"declaration read off the R pilot: {declaration}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("probe", "reference"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--draws", type=int, default=pilot.DRAWS)
    arguments = parser.parse_args(argv)
    scratch = arguments.scratch.resolve()
    if scratch.is_relative_to(ROOT):
        parser.error("--scratch must lie outside the repository")
    scratch.mkdir(parents=True, exist_ok=True)
    if arguments.command == "probe":
        return probe(scratch, arguments.jobs)
    return reference(scratch, arguments.jobs, arguments.chunk, arguments.draws)


if __name__ == "__main__":
    sys.exit(main())
