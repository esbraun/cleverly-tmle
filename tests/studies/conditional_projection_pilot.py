"""The SE-ratio pilot of the ``conditional-projection`` study (re-review RB1, RB3).

A pilot, not a result: it fixes the SE ratios the declaration's pass probabilities are
computed at, before the run.  Each calibration group is fitted ``R`` times on draws from the
named stream ``stream_seed(STUDY, "se_ratio_pilot", label, r)``, which is disjoint from the
run's ``property_sample`` streams and from its primary ``SEED``.  For every reported
coefficient the pilot records the ratio of the mean reported SE to the empirical SD of the
estimates with its bootstrap standard error, the SE's coefficient of variation and the estimates' kurtosis (the two inputs of
:func:`tests.studies.conditional_projection.calibration_row_pass`), the Wald coverage, and on
``clustered_4k`` the ratio of the mean row-level SE to the same SD.

The committed summary is ``conditional_projection_pilot.csv`` beside this module.  It is
written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 python -m tests.studies.conditional_projection_pilot \\
        --raw <scratch dir> --jobs 4 [--groups ...] [--replicates R] [--start S] [--summarize]

Each group's replicate records go to ``<scratch dir>/<label>.json``; ``--summarize`` reads
every record there and writes the CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.studies import conditional_projection as study
from tests.studies.evidence.seeds import stream_seed

#: The stream the pilot draws from.
STREAM = "se_ratio_pilot"

#: Replications of each GLM calibration group, and of the ensemble group.
REPLICATES = 2_400
ENSEMBLE_REPLICATES = 960

#: Bootstrap resamples of each ratio's standard error.
BOOTSTRAP = 400

#: The committed summary.
PILOT_CSV = Path(__file__).with_name("conditional_projection_pilot.csv")

COLUMNS = (
    "group",
    "name",
    "replicates",
    "failed",
    "se_ratio",
    "se_ratio_se",
    "se_cv",
    "kurtosis",
    "coverage",
    "standardized_bias",
    "iid_se_ratio",
)


def pilot_groups() -> tuple[str, ...]:
    """The groups with calibration rows, in declaration order."""
    return tuple(
        group.label for group in study.GROUPS if group.calibration or group.law == "assess"
    )


def _one(key: tuple[str, int]) -> dict[str, Any]:
    label, replicate = key
    group = study.GROUP_BY_LABEL[label]
    seed = stream_seed(study.STUDY, STREAM, label, replicate)
    frame, truths = study.draw_from_seed(label, group.n, seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = study.fit_group(group, frame, random_state=replicate)
    except Exception as error:
        return {"replicate": replicate, "error": repr(error)[:300]}
    estimates = {k: v for k, v in fitted["estimates"].items() if k in truths}
    record: dict[str, Any] = {
        "replicate": replicate,
        "truth": {k: float(truths[k]) for k in estimates},
        "estimate": {k: float(v.psi) for k, v in estimates.items()},
        "se": {k: float(v.std_error) for k, v in estimates.items()},
    }
    if "iid_se" in fitted["controls"]:
        stem = study.stem(group)
        record["iid_se"] = {
            f"{stem}[{term}]": float(value) for term, value in fitted["controls"]["iid_se"].items()
        }
    return record


def run(
    labels: tuple[str, ...], raw: Path, *, jobs: int, replicates: int | None, start: int = 0
) -> None:
    """Fit replicates ``start`` to ``start + R - 1`` of each group's pilot.

    The records go to ``raw/<label>.json``.  With ``start > 0`` they are appended to the
    records already there, which must be exactly replicates ``0`` to ``start - 1``, so an
    extension continues the stream without overlapping it.
    """
    raw.mkdir(parents=True, exist_ok=True)
    with Pool(jobs) as pool:
        for label in labels:
            size = replicates or (
                ENSEMBLE_REPLICATES
                if study.GROUP_BY_LABEL[label].learner == "ensemble"
                else REPLICATES
            )
            path = raw / f"{label}.json"
            earlier = json.loads(path.read_text()) if start else []
            if [record["replicate"] for record in earlier] != list(range(start)):
                raise RuntimeError(f"{path} does not hold exactly replicates 0 to {start - 1}")
            records = pool.map(_one, [(label, r) for r in range(start, start + size)], chunksize=4)
            path.write_text(json.dumps(earlier + records))
            failed = sum("error" in record for record in records)
            print(f"{label}: replicates {start} to {start + size - 1}, {failed} failed", flush=True)


def summarize(raw: Path) -> pd.DataFrame:
    """The per-coefficient pilot summary of every record in ``raw``."""
    rows = []
    for label in pilot_groups():
        path = raw / f"{label}.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        good = [record for record in records if "error" not in record]
        for name in good[0]["estimate"]:
            estimate = np.array([record["estimate"][name] for record in good])
            se = np.array([record["se"][name] for record in good])
            truth = good[0]["truth"][name]
            sd = float(estimate.std(ddof=1))
            rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, "bootstrap", label, name))
            index = rng.integers(0, len(good), (BOOTSTRAP, len(good)))
            resampled = se[index].mean(axis=1) / estimate[index].std(axis=1, ddof=1)
            iid = (
                float(np.mean([record["iid_se"][name] for record in good]) / sd)
                if "iid_se" in good[0]
                else math.nan
            )
            rows.append(
                {
                    "group": label,
                    "name": name,
                    "replicates": len(good),
                    "failed": len(records) - len(good),
                    "se_ratio": float(se.mean() / sd),
                    "se_ratio_se": float(resampled.std(ddof=1)),
                    "se_cv": float(se.std(ddof=1) / se.mean()),
                    "kurtosis": float(stats.kurtosis(estimate, fisher=False)),
                    "coverage": float(np.mean(np.abs(estimate - truth) <= study.CRITICAL * se)),
                    "standardized_bias": float((estimate.mean() - truth) / sd),
                    "iid_se_ratio": iid,
                }
            )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args()
    if not arguments.summarize:
        labels = tuple(arguments.groups) if arguments.groups else pilot_groups()
        run(
            labels,
            arguments.raw,
            jobs=arguments.jobs,
            replicates=arguments.replicates,
            start=arguments.start,
        )
    else:
        summarize(arguments.raw).to_csv(PILOT_CSV, index=False, float_format="%.6g")
        print(f"wrote {PILOT_CSV}", flush=True)


if __name__ == "__main__":
    main()
