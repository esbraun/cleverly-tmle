"""The sizing pilot of ``point-shift-learned-density`` (X121 part b2).

A pilot, not a result. Each draw of the law at n = 3,000 fits the cross-fitted learned-density
arm once and records each contrast's estimate, reported SE and interval. :func:`summarize` writes
``point_shift_learned_density_pilot.csv``: per parameter (the three contrasts, which size the
calibration count, and the four policy means, whose bias legs the reading gates), the draws, the SE ratio (mean reported SE
over the SD of the estimate; the truth is constant), its bootstrap SD and the spread constant
``c = SD * sqrt(2 P)`` that implies, the SD of the estimate in efficient-influence-curve units
(``k``, which the noise control's power reads), the coverage, the standardized bias and the mean
CPU seconds of one draw. ``test_point_shift_learned_density_design`` reads the CSV and
applies the per-cell predictive rule
(:func:`~tests.studies.mtp_study_tools.per_arm_replicates`) to set the calibration count, which
is raised to that value before the declared run, up to
:data:`~tests.studies.point_shift_learned_density_properties.CALIBRATION_REPLICATES_CAP`.

Every draw comes from ``stream_seed(PILOT_RECORD, "se_ratio_pilot", "crossfit_learned", r)`` and
its fit seed from the same tuple under ``"se_ratio_pilot_fit"``. :data:`PILOT_RECORD` is the
study's record at :data:`PILOT_SEED`; the design test checks its seeds against the declared ones.
The run appends one JSON line per draw, so a stopped run resumes; a line cut short is skipped and
its draw is run again::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.point_shift_learned_density_pilot --raw <scratch dir> \\
        --jobs <cores> [--first R --last R] [--summarize | --count]

``--first`` and ``--last`` select a range of draws, so the pilot can be split across machines;
``--summarize`` reads every ``*.jsonl`` file in ``--raw``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import _point_shift_learned_density_law as law
from tests.studies import point_shift_learned_density as study
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

#: The pilot's seed, distinct from the record's own seeds.
PILOT_SEED = 20261213
PILOT_RECORD = replace(study.STUDY, seed=PILOT_SEED, resampling_seed=PILOT_SEED + 1)
STREAM = "se_ratio_pilot"
FIT_STREAM = "se_ratio_pilot_fit"
ARM = study.GATED_ARM

#: The pilot's draws. At 3,200 the per-cell rule's count at a true SE ratio of 1 is 4,800, and
#: a larger pilot buys little (``test_point_shift_learned_density_design``).
PILOT_DRAWS = 3_200

PILOT_CSV = Path(__file__).with_name("point_shift_learned_density_pilot.csv")
#: The bootstrap of each pilot SE ratio.
BOOTSTRAP_REPLICATES = 2_000
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def sample_seed(replicate: int) -> int:
    """The pilot's sample seed of one draw."""
    return stream_seed(PILOT_RECORD, STREAM, ARM, replicate)


def fit_seed(replicate: int) -> int:
    """The pilot's fit seed of one draw."""
    return stream_seed(PILOT_RECORD, FIT_STREAM, ARM, replicate)


def one(replicate: int) -> dict[str, Any]:
    """Fit one pilot draw and return each contrast's readings."""
    frame = law.draw(law.N, sample_seed(replicate))
    start = time.process_time()
    result = law.fit(ARM, frame, fit_seed(replicate))
    seconds = time.process_time() - start
    readings = {}
    for tag, name in law.PARAMETERS.items():
        estimate = result.estimates[name]
        std_error, low, high = reported_inference(estimate)
        readings[tag] = {
            "estimate": float(estimate.psi),
            "std_error": std_error,
            "covered": int(low <= law.TRUTH[tag] <= high),
        }
    return {"replicate": replicate, "seconds": seconds, "readings": readings}


def read_raw(raw: Path) -> dict[int, dict[str, Any]]:
    """Every complete line of every ``*.jsonl`` file under ``raw``, by replicate."""
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(record["replicate"])] = record
    return out


def run(raw: Path, jobs: int, first: int, last: int) -> None:
    """Fit every draw in ``[first, last]`` not yet in ``raw``, appending as each finishes."""
    for variable in THREAD_VARIABLES:
        if os.environ.get(variable) != "1":
            raise SystemExit(f"set {variable}=1 before the pilot")
    raw.mkdir(parents=True, exist_ok=True)
    done = read_raw(raw)
    todo = [r for r in range(first, last + 1) if r not in done]
    target = raw / f"pilot_{first}_{last}.jsonl"
    with Pool(jobs) as pool, target.open("a") as handle:
        for record in pool.imap_unordered(one, todo):
            handle.write(json.dumps(record) + "\n")
            handle.flush()


def summarize(raw: Path) -> pd.DataFrame:
    """The pilot table, written to :data:`PILOT_CSV`."""
    records = read_raw(raw)
    rows = []
    for tag in law.PARAMETERS:
        estimates = np.array([r["readings"][tag]["estimate"] for r in records.values()])
        errors = np.array([r["readings"][tag]["std_error"] for r in records.values()])
        covered = np.array([r["readings"][tag]["covered"] for r in records.values()])
        spread = float(np.std(estimates, ddof=1))
        ratio = float(np.mean(errors)) / spread
        rng = np.random.default_rng(stream_seed(PILOT_RECORD, "pilot_bootstrap", tag))
        draws = len(records)
        boot = np.empty(BOOTSTRAP_REPLICATES)
        for b in range(BOOTSTRAP_REPLICATES):
            pick = rng.integers(0, draws, size=draws)
            boot[b] = float(np.mean(errors[pick])) / float(np.std(estimates[pick], ddof=1))
        ratio_sd = float(np.std(boot, ddof=1))
        rows.append(
            {
                "contrast": tag,
                "draws": draws,
                "se_ratio": ratio,
                "se_ratio_bootstrap_sd": ratio_sd,
                "implied_c": ratio_sd * float(np.sqrt(2.0 * draws)),
                "estimate_sd": spread,
                "efficiency_k": (
                    spread * np.sqrt(law.N) / study.TAG_BOUNDS[tag]
                    if tag in study.TAG_BOUNDS
                    else float("nan")
                ),
                "coverage": float(np.mean(covered)),
                "standardized_bias": (float(np.mean(estimates)) - law.TRUTH[tag]) / spread,
                "mean_seconds": float(np.mean([r["seconds"] for r in records.values()])),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(PILOT_CSV, index=False, float_format="%.6f")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=PILOT_DRAWS - 1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--count", action="store_true")
    arguments = parser.parse_args()
    if arguments.count:
        print(len(read_raw(arguments.raw)))
    elif arguments.summarize:
        print(summarize(arguments.raw).to_string(index=False))
    else:
        run(arguments.raw, arguments.jobs, arguments.first, arguments.last)


if __name__ == "__main__":
    main()
