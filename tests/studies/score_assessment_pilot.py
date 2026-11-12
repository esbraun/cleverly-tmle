"""The SE-ratio pilot of the ``score-assessment`` study (X97 plan, section 5.1).

A pilot, not a result: it fixes the SE ratios the calibration replication count is computed
at, before the declaration.  Each calibration group is fitted :data:`REPLICATES` times on draws
from the named stream ``stream_seed(STUDY, "se_ratio_pilot", label, r)``, which is disjoint
from the run's ``property_sample`` streams and from its primary ``SEED``.  For every reported
parameter the pilot records the ratio of the mean reported SE to the empirical SD of the
estimates with its bootstrap standard error, the SE's coefficient of variation, the estimates'
kurtosis, the Wald coverage and the standardized bias; on the paired row it records the two SE
controls' ratios too; and for every group the count of fits that settled only on a reduced span
and the single-thread seconds per fit.  A second table
holds the correlation of every two estimates of a group, which the sizing rule reads.

The committed summaries are ``score_assessment_pilot.csv`` and
``score_assessment_pilot_correlations.csv`` beside this module.  They are written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 python -m tests.studies.score_assessment_pilot \\
        --raw <scratch dir> --jobs 4 [--groups ...] [--replicates R] [--summarize]

Each group's replicate records go to ``<scratch dir>/<label>.json``; ``--summarize`` reads
every record there and writes the two CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.studies import score_assessment as study
from tests.studies.evidence.seeds import stream_seed

#: The stream the pilot draws from.
STREAM = "se_ratio_pilot"

#: Replications of each calibration group (plan 5.1, rev 2.2: every group at one length).
REPLICATES = 2_400

#: Bootstrap resamples of each ratio's standard error.
BOOTSTRAP = 400

COLUMNS = (
    "group",
    "label",
    "replicates",
    "failed",
    "se_ratio",
    "se_ratio_se",
    "se_cv",
    "kurtosis",
    "coverage",
    "standardized_bias",
    "unpaired_se_ratio",
    "no_residual_se_ratio",
    "reduced",
    "seconds",
)


def _one(key: tuple[str, int]) -> dict[str, Any]:
    label, replicate = key
    group = study.PILOT_GROUP_BY_LABEL[label]
    seed = stream_seed(study.STUDY, STREAM, label, replicate)
    frame, truths = study.draw_from_seed(label, group.n, seed)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = study.fit_group(group, frame, random_state=replicate)
    except Exception as error:
        return {"replicate": replicate, "error": repr(error)[:300]}
    seconds = time.perf_counter() - start
    estimates = fitted["estimates"]
    return {
        "replicate": replicate,
        "seconds": seconds,
        "truth": {k: float(truths[k]) for k in estimates},
        "estimate": {k: float(v.psi) for k, v in estimates.items()},
        "se": {k: float(v.std_error) for k, v in estimates.items()},
        "unpaired_se": fitted["unpaired_se"],
        "no_residual_se": fitted["no_residual_se"],
        "reduced": fitted["reduced"],
    }


def run(labels: tuple[str, ...], raw: Path, *, jobs: int, replicates: int) -> None:
    """Fit replicates ``0`` to ``replicates - 1`` of each group's pilot into ``raw``."""
    raw.mkdir(parents=True, exist_ok=True)
    with Pool(jobs) as pool:
        for label in labels:
            records = pool.map(_one, [(label, r) for r in range(replicates)], chunksize=4)
            (raw / f"{label}.json").write_text(json.dumps(records))
            failed = sum("error" in record for record in records)
            print(f"{label}: {replicates} replicates, {failed} failed", flush=True)


def _ratio(numerator: np.ndarray, sd: float) -> float:
    return float(np.mean(numerator) / sd)


def summarize(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-parameter pilot summary and the within-group correlations of every record."""
    rows, correlations = [], []
    for label in study.PILOT_GROUPS:
        path = raw / f"{label}.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        good = [record for record in records if "error" not in record]
        seconds = float(np.mean([record["seconds"] for record in good]))
        names = list(good[0]["estimate"])
        matrix = np.array([[record["estimate"][name] for name in names] for record in good])
        for name in names:
            estimate = np.array([record["estimate"][name] for record in good])
            se = np.array([record["se"][name] for record in good])
            truth = good[0]["truth"][name]
            sd = float(estimate.std(ddof=1))
            rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, "bootstrap", label, name))
            index = rng.integers(0, len(good), (BOOTSTRAP, len(good)))
            resampled = se[index].mean(axis=1) / estimate[index].std(axis=1, ddof=1)
            paired = name == "paired"
            rows.append(
                {
                    "group": label,
                    "label": name,
                    "replicates": len(good),
                    "failed": len(records) - len(good),
                    "se_ratio": float(se.mean() / sd),
                    "se_ratio_se": float(resampled.std(ddof=1)),
                    "se_cv": float(se.std(ddof=1) / se.mean()),
                    "kurtosis": float(stats.kurtosis(estimate, fisher=False)),
                    "coverage": float(np.mean(np.abs(estimate - truth) <= study.CRITICAL * se)),
                    "standardized_bias": float((estimate.mean() - truth) / sd),
                    "unpaired_se_ratio": _ratio(np.array([r["unpaired_se"] for r in good]), sd)
                    if paired
                    else math.nan,
                    "no_residual_se_ratio": _ratio(
                        np.array([r["no_residual_se"] for r in good]), sd
                    )
                    if paired
                    else math.nan,
                    "reduced": sum(record.get("reduced", 0) > 0 for record in good),
                    "seconds": seconds,
                }
            )
        correlation = np.corrcoef(matrix, rowvar=False)
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                correlations.append(
                    {"group": label, "a": a, "b": names[j], "correlation": correlation[i, j]}
                )
    return pd.DataFrame(rows, columns=list(COLUMNS)), pd.DataFrame(correlations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args()
    if not arguments.summarize:
        labels = tuple(arguments.groups) if arguments.groups else study.PILOT_GROUPS
        run(labels, arguments.raw, jobs=arguments.jobs, replicates=arguments.replicates)
        return
    summary, correlations = summarize(arguments.raw)
    summary.to_csv(study.PILOT_CSV, index=False, float_format="%.6g")
    correlations.to_csv(study.PILOT_CORRELATIONS_CSV, index=False, float_format="%.6g")
    print(f"wrote {study.PILOT_CSV} and {study.PILOT_CORRELATIONS_CSV}", flush=True)


if __name__ == "__main__":
    main()
