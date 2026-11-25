"""The sizing pilot of ``learned-rule-contrast-comparators`` (T58).

A pilot, not a result. Each draw of ``non_exceptional`` at n = 2,000 fits fit C and fit N under
both reports on one fold seed, refits the fold rules once (rule L4), and records each tag's
estimate, reported SE and truth. :func:`summarize` writes
``learned_rule_contrast_comparators_pilot.csv``: per report and tag, the draws, the SE ratio (the
mean reported SE over the SD of the error ``estimate - truth``), the coverage, the standardized
bias, the excess kurtosis of the error, the solver warnings and the mean CPU seconds of one draw.
``tests/unit/test_learned_rule_contrast_comparators_design.py`` reads the CSV and applies the
arm-pooled predictive rule to the calibration count, which is raised before the run if the rule
asks for more.

Every draw comes from ``stream_seed(PILOT_RECORD, "se_ratio_pilot", law, r)`` and its fold seed
from ``stream_seed(PILOT_RECORD, "se_ratio_pilot_folds", law, r)``. :data:`PILOT_RECORD` is the
study's record at :data:`PILOT_SEED`, and the design test checks the pilot's seeds against every
declared learned-rule seed. The run appends one JSON line per draw, so a stopped run resumes::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.learned_rule_contrast_comparators_pilot --raw <scratch dir> \\
        --jobs <cores> [--summarize]
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
from scipy.stats import kurtosis

from tests.studies import _learned_rule_contrast_comparators_law as law
from tests.studies import learned_rule_contrast_comparators as study
from tests.studies.evidence.seeds import stream_seed

#: The pilot's seed, distinct from the record's own seeds.
PILOT_SEED = 20265811
PILOT_RECORD = replace(study.STUDY, seed=PILOT_SEED, resampling_seed=PILOT_SEED + 1)

STREAM = "se_ratio_pilot"
FOLD_STREAM = "se_ratio_pilot_folds"

#: The pilot's law and draws.
PILOT_LAW = study.PRIMARY_LAW
PILOT_DRAWS = 4_000

PILOT_CSV = Path(__file__).with_name("learned_rule_contrast_comparators_pilot.csv")
RAW_NAME = f"{PILOT_LAW}.jsonl"

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

CRITICAL = law.base.CRITICAL


def sample_seed(replicate: int) -> int:
    """The pilot's sample seed of one draw."""
    return stream_seed(PILOT_RECORD, STREAM, PILOT_LAW, replicate)


def fold_seed(replicate: int) -> int:
    """The pilot's fold seed of one draw."""
    return stream_seed(PILOT_RECORD, FOLD_STREAM, PILOT_LAW, replicate)


def one(replicate: int) -> dict[str, Any]:
    """Fit one pilot draw and return each report's and tag's readings."""
    frame = law.base.draw(PILOT_LAW, study.PRIMARY_N, sample_seed(replicate))
    start = time.process_time()
    fitted = law.measured(frame, PILOT_LAW, law.base.outcome_learner(), fold_seed(replicate))
    seconds = time.process_time() - start
    readings: dict[str, dict[str, float]] = {}
    for report in law.REPORTS:
        truths = law.truths(fitted.result("value", report), fitted.refit, PILOT_LAW)
        for tag, (_, name) in law.TAGS.items():
            estimate = fitted.result(tag, report).estimates[name]
            readings[f"{report}_{tag}"] = {
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "truth": truths[tag][0],
            }
    return {
        "replicate": replicate,
        "labels": readings,
        "solver_warnings": int(fitted.solver_warnings),
        "seconds": seconds,
    }


def records(path: Path) -> list[dict[str, Any]]:
    """The complete JSON lines of ``path``; a line cut short by a stopped run is skipped."""
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarize(raw: Path) -> pd.DataFrame:
    """One row per pilot label, from the raw JSON lines."""
    records_ = records(raw / RAW_NAME)
    if len({r["replicate"] for r in records_}) != len(records_):
        raise RuntimeError("the raw pilot file holds a draw twice")
    out = []
    for report in law.REPORTS:
        for tag in law.TAGS:
            label = f"{report}_{tag}"
            frame = pd.DataFrame([record["labels"][label] for record in records_])
            error = frame["estimate"] - frame["truth"]
            spread = float(error.std(ddof=1))
            out.append(
                {
                    "label": label,
                    "report": report,
                    "tag": tag,
                    "n": study.PRIMARY_N,
                    "replicates": len(frame),
                    "se_ratio": float(frame["std_error"].mean()) / spread,
                    "coverage": float((error.abs() <= CRITICAL * frame["std_error"]).mean()),
                    "standardized_bias": float(error.mean()) / spread,
                    "excess_kurtosis": float(kurtosis(error, fisher=True, bias=False)),
                    "solver_warnings": int(sum(r["solver_warnings"] for r in records_)),
                    "seconds": float(np.mean([r["seconds"] for r in records_])),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    """Run (resumably) or summarize the pilot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    if arguments.summarize:
        summarize(arguments.raw).to_csv(PILOT_CSV, index=False, float_format="%.6g")
        return
    unset = [name for name in THREAD_VARIABLES if os.environ.get(name) != "1"]
    if unset:
        raise SystemExit(f"set {', '.join(unset)} to 1: the pilot times one thread per fit")
    path = arguments.raw / RAW_NAME
    done = set()
    if path.exists():
        done = {record["replicate"] for record in records(path)}
    count = PILOT_DRAWS if arguments.cap is None else min(arguments.cap, PILOT_DRAWS)
    tasks = [r for r in range(count) if r not in done]
    with Pool(arguments.jobs, maxtasksperchild=50) as pool:
        for record in pool.imap_unordered(one, tasks, chunksize=4):
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
