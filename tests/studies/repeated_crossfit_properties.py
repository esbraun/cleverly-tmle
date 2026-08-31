"""Independent properties for repeated point-treatment cross-fitting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleverly.learners.crossfit import CrossFitPlan
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import cv_fit
from tests.studies.cvtmle_properties import generate, summarize
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    paired_spread_ratio_interval,
    replicate_row,
    require_complete,
)
from tests.studies.evidence.property_verdicts import finish
from tests.studies.evidence.seeds import stream_seed
from tests.studies.repeated_crossfit import (
    FOLD_SEED_TRIALS,
    MAX_REPEAT_SPREAD_RATIO,
    N_FOLDS,
    REPEAT_STABILITY_N,
    REPEATS,
    STUDY,
    draw_from_seed,
)

REPEAT_STABILITY_COLUMNS = (
    "spread_ratio",
    "spread_ratio_ci_lower",
    "spread_ratio_ci_upper",
    "spread_ratio_boundary",
)


def _first_repeat_seed(base_seed: int) -> int:
    """The first fold seed that the estimator derives from a repeated fit's seed."""
    plan = CrossFitPlan(n_folds=N_FOLDS, random_state=base_seed, repeats=REPEATS)
    seed = plan.seeds()[0]
    if seed is None:
        raise RuntimeError("the labelled repeat-stability seed unexpectedly derived None")
    return seed


def _stability_fit(frame: pd.DataFrame, *, repeats: int, random_state: int) -> Any:
    return cv_fit(
        frame,
        binary=True,
        estimands=("ate",),
        n_folds=N_FOLDS,
        repeats=repeats,
        cv_evaluation=False,
        random_state=random_state,
    )


def _stability_trial(
    frame: pd.DataFrame,
    truth: float,
    replicate: int,
    requested: int,
    base_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    repeated = _stability_fit(frame, repeats=REPEATS, random_state=base_seed)
    first_seed = _first_repeat_seed(base_seed)
    control = _stability_fit(frame, repeats=1, random_state=first_seed)

    repeated_first = repeated.repeats[0].nuisance.folds.assignment
    control_folds = control.nuisance.folds.assignment
    if not np.array_equal(repeated_first, control_folds):
        raise RuntimeError("the one-repeat control is not the repeated fit's first fold draw")
    if repeated.repeats[0].psi["ate"] != control.psi("ate"):
        raise RuntimeError("the one-repeat control does not reproduce the first draw's ATE")

    return (
        replicate_row(
            property_name="repeat_stability",
            cell="three_repeats",
            role="positive",
            replicate=replicate,
            n=REPEAT_STABILITY_N,
            requested=requested,
            truth=truth,
            estimate=repeated["ate"],
            alpha=STUDY.margins.alpha,
        ),
        replicate_row(
            property_name="repeat_stability",
            cell="one_repeat_control",
            role="control",
            replicate=replicate,
            n=REPEAT_STABILITY_N,
            requested=requested,
            truth=truth,
            estimate=control["ate"],
            alpha=STUDY.margins.alpha,
        ),
    )


def generate_repeat_stability_rows(
    *, trials: int = FOLD_SEED_TRIALS, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    """Fit paired one- and three-repeat reports on one fixed binary sample."""
    sample_seed = stream_seed(STUDY, "repeat_stability", "sample")
    frame, truth = draw_from_seed("binary", REPEAT_STABILITY_N, sample_seed)
    payloads = [
        (
            frame,
            float(truth["ate"]),
            replicate,
            trials,
            stream_seed(STUDY, "repeat_stability", "fold_seed", replicate),
        )
        for replicate in range(trials)
    ]
    paired = map_parallel(_stability_trial, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for pair in paired for row in pair])
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def _summarize_repeat_stability(rows: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Build fixed-sample stability rows without population-sampling endpoints."""
    require_complete(rows)
    repeated = rows.loc[rows["cell"] == "three_repeats"]
    control = rows.loc[rows["cell"] == "one_repeat_control"]
    result = paired_spread_ratio_interval(
        repeated,
        control,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "repeat_stability", "bootstrap"),
    )
    passed = result.interval.high < MAX_REPEAT_SPREAD_RATIO
    records: list[dict[str, object]] = []
    for cell, group in rows.groupby("cell", sort=True):
        record = dict.fromkeys(columns, np.nan)
        record.update(
            {
                "property": "repeat_stability",
                "cell": cell,
                "role": group["role"].iloc[0],
                "n": int(group["n"].iloc[0]),
                "replicates": len(group),
                "failed_replicates": int(group["failed_replicates"].iloc[0]),
                "spread_ratio": result.ratio,
                "spread_ratio_ci_lower": result.interval.low,
                "spread_ratio_ci_upper": result.interval.high,
                "spread_ratio_boundary": MAX_REPEAT_SPREAD_RATIO,
                "passed": passed,
                "property_passed": passed,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    inherited = generate(
        "repeated",
        repeats=REPEATS,
        n_folds=N_FOLDS,
        include_overfitting=False,
        n_jobs=n_jobs,
    )
    stability = generate_repeat_stability_rows(n_jobs=n_jobs)
    return pd.concat([inherited, stability], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    stability_mask = rows["property"] == "repeat_stability"
    inherited = rows.loc[~stability_mask]
    stability = rows.loc[stability_mask]
    summary, rates = summarize(
        inherited,
        STUDY,
        "repeated",
        include_overfitting=False,
        extra_columns=REPEAT_STABILITY_COLUMNS,
        return_parts=True,
    )
    summary = pd.concat(
        [summary, _summarize_repeat_stability(stability, summary.columns)], ignore_index=True
    )
    return finish(summary, rates)
