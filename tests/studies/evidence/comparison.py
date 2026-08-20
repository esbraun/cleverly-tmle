"""Comparing the implementation under study with a canonical reference.

Two claims, and they are not the same shape.  *Similarity* is symmetric: the paired mean
difference must be small in either direction, because a large difference either way means
the two are not computing the same thing.  *Non-inferiority* is one-sided: the study exists
to catch the subject being worse, and a subject that is better is a result, not a failure.

Whether each implementation is any good on its own is a third question and belongs to
:mod:`tests.studies.evidence.performance`.  It is carried here as two separate columns
rather than folded into one verdict, so a reference that degrades is reported against the
reference instead of turning the subject's row red.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.inference import (
    bootstrap,
    lower_bound,
    student_interval,
    upper_bound,
)
from tests.studies.evidence.pairing import paired_wide
from tests.studies.evidence.registry import StudyRecord

PAIRED_COLUMNS = ("estimate", "covered", "inference_estimate", "std_error")

#: Beyond the declared non-inferiority margins, reported and never gated.  A subject that
#: beats the reference by more than a symmetric screen is a result; folding the screen into
#: the verdict would fail the row for being better.
RMSE_SCREEN = 1.25
COVERAGE_SHORTFALL_SCREEN = 0.05


def _bounds(payload: tuple[StudyRecord, pd.DataFrame, str, float, int]) -> dict[str, Any]:
    """One cell's one-sided bootstrap bounds, as a picklable unit of work."""
    record, group, estimand, truth, seed = payload
    subject, reference = record.implementation, str(record.reference)
    paired = paired_wide(group, PAIRED_COLUMNS, implementations=(subject, reference))
    arrays = paired.arrays(PAIRED_COLUMNS, (subject, reference))
    margins = record.margins

    def rmse_ratio(draw: dict[str, np.ndarray]) -> np.ndarray:
        subject_error = draw[f"estimate_{subject}"] - truth
        reference_error = draw[f"estimate_{reference}"] - truth
        return np.sqrt(np.mean(subject_error**2, axis=1) / np.mean(reference_error**2, axis=1))

    def coverage_difference(draw: dict[str, np.ndarray]) -> np.ndarray:
        return np.mean(draw[f"covered_{subject}"] - draw[f"covered_{reference}"], axis=1)

    def calibration_excess(draw: dict[str, np.ndarray]) -> np.ndarray:
        subject_ratio = draw[f"std_error_{subject}"].mean(axis=1) / draw[
            f"inference_estimate_{subject}"
        ].std(axis=1, ddof=1)
        reference_ratio = draw[f"std_error_{reference}"].mean(axis=1) / draw[
            f"inference_estimate_{reference}"
        ].std(axis=1, ddof=1)
        return np.abs(subject_ratio - 1.0) - np.abs(reference_ratio - 1.0)

    samples = bootstrap(
        arrays,
        {
            "rmse_ratio": rmse_ratio,
            "coverage_difference": coverage_difference,
            "calibration_excess": calibration_excess,
        },
        replicates=margins.bootstrap_replicates,
        seed=seed,
    )
    comparable = estimand not in record.incomparable_se
    return {
        "rmse_ratio_upper": upper_bound(
            samples["rmse_ratio"], confidence_level=margins.confidence_level
        ),
        "coverage_difference_lower": lower_bound(
            samples["coverage_difference"], confidence_level=margins.confidence_level
        ),
        "calibration_excess_upper": (
            upper_bound(samples["calibration_excess"], confidence_level=margins.confidence_level)
            if comparable
            else math.nan
        ),
        "paired_replicates": len(paired),
        "dropped_replications": paired.dropped,
    }


def equivalence(
    rows: pd.DataFrame,
    summaries: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    record: StudyRecord,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """Paired similarity and one-sided non-inferiority, per scenario-estimand cell."""
    if record.reference is None:
        raise ValueError(f"{record.slug} declares no reference implementation to compare against")
    subject, reference = record.implementation, record.reference
    margins = record.margins
    verdicts = performance.set_index(["implementation", "scenario", "estimand"])["passed"]
    cells = list(rows.groupby(["scenario", "estimand"], sort=True))
    bounds = map_parallel(
        _bounds,
        [
            (
                (
                    record,
                    group,
                    str(key[1]),
                    float(group["truth"].iloc[0]),
                    record.seed + 20_000 + index,
                ),
            )
            for index, (key, group) in enumerate(cells)
        ],
        n_jobs=n_jobs,
    )

    records: list[dict[str, Any]] = []
    for (key, group), bound in zip(cells, bounds, strict=True):
        scenario, estimand = key
        paired = paired_wide(group, ("estimate",), implementations=(subject, reference))
        difference = paired.column("estimate", subject) - paired.column("estimate", reference)
        interval = student_interval(difference, confidence_level=margins.confidence_level)
        pooled_sd = float(
            np.sqrt(
                0.5
                * (
                    np.var(paired.column("estimate", subject), ddof=1)
                    + np.var(paired.column("estimate", reference), ddof=1)
                )
            )
        )
        mean_margin = margins.paired_difference * pooled_sd

        cell = summaries.query("scenario == @scenario and estimand == @estimand").set_index(
            "implementation"
        )
        rmse_ratio = float(cell.loc[subject, "rmse"] / cell.loc[reference, "rmse"])
        coverage_difference = float(cell.loc[subject, "coverage"] - cell.loc[reference, "coverage"])
        se_comparable = estimand not in record.incomparable_se
        calibration_excess = float(
            max(
                0.0,
                abs(cell.loc[subject, "se_ratio"] - 1.0)
                - abs(cell.loc[reference, "se_ratio"] - 1.0),
            )
        )
        not_inferior = bool(
            bound["rmse_ratio_upper"] <= margins.rmse_noninferiority
            and bound["coverage_difference_lower"] >= margins.coverage_noninferiority
            and (
                not se_comparable
                or bound["calibration_excess_upper"] <= margins.calibration_noninferiority
            )
        )
        similar = interval.within(-mean_margin, mean_margin)
        records.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "subject": subject,
                "reference": reference,
                "paired_replicates": bound["paired_replicates"],
                "dropped_replications": bound["dropped_replications"],
                "confidence_level": margins.confidence_level,
                "mean_difference": float(np.mean(difference)),
                "paired_se": float(np.std(difference, ddof=1) / math.sqrt(len(difference))),
                "paired_ci_lower": interval.low,
                "paired_ci_upper": interval.high,
                "mean_margin": mean_margin,
                "margin_utilization": abs(float(np.mean(difference))) / mean_margin,
                "paired_similarity": similar,
                "rmse_ratio": rmse_ratio,
                "rmse_ratio_upper": bound["rmse_ratio_upper"],
                "rmse_noninferiority_margin": margins.rmse_noninferiority,
                "coverage_difference": coverage_difference,
                "subject_coverage_shortfall": max(0.0, -coverage_difference),
                "coverage_difference_lower": bound["coverage_difference_lower"],
                "coverage_noninferiority_margin": margins.coverage_noninferiority,
                "se_comparable": se_comparable,
                "se_ratio_difference": (
                    float(abs(cell.loc[subject, "se_ratio"] - cell.loc[reference, "se_ratio"]))
                    if se_comparable
                    else math.nan
                ),
                "subject_calibration_excess": calibration_excess if se_comparable else math.nan,
                "calibration_excess_upper": bound["calibration_excess_upper"],
                "calibration_noninferiority_margin": (
                    margins.calibration_noninferiority if se_comparable else math.nan
                ),
                "rmse_screen": bool(rmse_ratio <= RMSE_SCREEN),
                "coverage_shortfall_screen": bool(
                    max(0.0, -coverage_difference) <= COVERAGE_SHORTFALL_SCREEN
                ),
                "subject_valid": bool(verdicts.loc[(subject, scenario, estimand)]),
                "reference_valid": bool(verdicts.loc[(reference, scenario, estimand)]),
                "subject_not_inferior": not_inferior,
                "passed": bool(similar and not_inferior),
            }
        )
    return pd.DataFrame.from_records(records)
