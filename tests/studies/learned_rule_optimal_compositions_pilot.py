"""The sizing pilot of ``learned-rule-optimal-compositions`` (X11 plan rev 2, section 5).

A pilot, not a result. Each setting fits the declared estimator of its calibration draw at
n = 2,000 and records, for each of its labels, the estimate, the reported SE, the fixed truth,
the value of the fit's own fold rules (rule L4's refit) and the CPU seconds of the fit and its
refit. :func:`summarize` writes ``learned_rule_optimal_compositions_pilot.csv``: per label, the
draws, the SE ratio (the mean reported SE over the SD of the error ``estimate - truth``), the
coverage of the truth, the standardized bias, the same three against the fold rules' value, the
mean regret, the solver warnings and the mean CPU seconds of one draw of the setting.
``tests/unit/test_learned_rule_optimal_compositions_design.py`` reads the CSV and applies the
arm-pooled predictive rule to set the calibration count.

Every draw comes from ``stream_seed(PILOT_RECORD, "se_ratio_pilot", setting, r)`` and its fold
seed from ``stream_seed(PILOT_RECORD, "se_ratio_pilot_folds", setting, r)``, where
:data:`PILOT_RECORD` is the study's record at :data:`PILOT_SEED`, labels no declared stream
reads; the design test checks the pilot's seeds against every declared learned-rule seed::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.learned_rule_optimal_compositions_pilot --raw <scratch dir> \\
        --jobs 4 [--summarize]
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

from tests.studies import _learned_rule_optimal_compositions_law as law
from tests.studies import learned_rule_optimal_compositions as study
from tests.studies.evidence.seeds import stream_seed

#: The pilot's seed, chosen before the record's own seed and distinct from it.
PILOT_SEED = 20271125
PILOT_RECORD = replace(study.STUDY, seed=PILOT_SEED, resampling_seed=PILOT_SEED + 1)

STREAM = "se_ratio_pilot"
FOLD_STREAM = "se_ratio_pilot_folds"

#: The draws of each setting: the plan's pilot of at least 2,000 per cell.
PILOT_DRAWS = 2_000

PILOT_CSV = Path(__file__).with_name("learned_rule_optimal_compositions_pilot.csv")

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

CRITICAL = law.base.CRITICAL


def sample_seed(setting: str, replicate: int) -> int:
    """The pilot's sample seed of one draw."""
    return stream_seed(PILOT_RECORD, STREAM, setting, replicate)


def one(task: tuple[str, int]) -> dict[str, Any]:
    """Fit one pilot draw of ``setting`` and return its labels' readings."""
    setting, replicate = task
    frame = law.draw(setting, study.PRIMARY_N, sample_seed(setting, replicate))
    fold_seed = stream_seed(PILOT_RECORD, FOLD_STREAM, setting, replicate)
    start = time.process_time()
    fitted = law.measured(frame, setting, fold_seed)
    seconds = time.process_time() - start
    labels = {}
    for label in law.PARAMETERS[setting]:
        estimate = fitted.estimate(label)
        labels[label] = {
            "estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "truth": law.truth(label),
            "rule_value": float(fitted.rule_values[label]),
        }
    return {
        "setting": setting,
        "replicate": replicate,
        "labels": labels,
        "solver_warnings": int(fitted.solver_warnings),
        "seconds": seconds,
    }


def _readings(error: pd.Series, std_error: pd.Series) -> tuple[float, float, float]:
    spread = float(error.std(ddof=1))
    return (
        float(std_error.mean()) / spread,
        float((error.abs() <= CRITICAL * std_error).mean()),
        float(error.mean()) / spread,
    )


def summarize(raw: Path) -> pd.DataFrame:
    """One row per pilot label, from the raw ``<setting>.jsonl`` records."""
    out = []
    for setting in law.SETTINGS:
        path = raw / f"{setting}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for label in law.PARAMETERS[setting]:
            frame = pd.DataFrame([record["labels"][label] for record in records])
            se_ratio, coverage, bias = _readings(
                frame["estimate"] - frame["truth"], frame["std_error"]
            )
            rule_ratio, rule_coverage, rule_bias = _readings(
                frame["estimate"] - frame["rule_value"], frame["std_error"]
            )
            out.append(
                {
                    "label": label,
                    "setting": setting,
                    "n": study.PRIMARY_N,
                    "replicates": len(frame),
                    "se_ratio": se_ratio,
                    "coverage": coverage,
                    "standardized_bias": bias,
                    "rule_value_se_ratio": rule_ratio,
                    "rule_value_coverage": rule_coverage,
                    "rule_value_standardized_bias": rule_bias,
                    "mean_regret": float((frame["truth"] - frame["rule_value"]).mean()),
                    "solver_warnings": int(sum(r["solver_warnings"] for r in records)),
                    "seconds": float(np.mean([r["seconds"] for r in records])),
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
    tasks = []
    for setting in law.SETTINGS:
        path = arguments.raw / f"{setting}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["replicate"] for line in path.read_text().splitlines()}
        count = PILOT_DRAWS if arguments.cap is None else min(arguments.cap, PILOT_DRAWS)
        tasks += [(setting, r) for r in range(count) if r not in done]
    with Pool(arguments.jobs, maxtasksperchild=50) as pool:
        for record in pool.imap_unordered(one, tasks, chunksize=4):
            with (arguments.raw / f"{record['setting']}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
