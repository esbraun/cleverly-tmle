"""Descriptive summaries, and each implementation's own verdict against known truth.

These verdicts are what stops agreement between two poor implementations from counting as
evidence: each is measured against the law, not against the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from cleverly.validation import ReplicationRecord, summarize_replications
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.inference import (
    Interval,
    bootstrap,
    clopper_pearson,
    percentile_interval,
    standardized_bias_verdict,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import truth_on_inference_scale

CELL_KEYS = ["implementation", "scenario", "estimand"]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute the published performance table from the per-replication rows."""
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(CELL_KEYS, sort=True):
        implementation, scenario, estimand = key
        inference = group["inference_estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        truth_inference = truth_on_inference_scale(
            str(estimand), truth, str(group["inference_scale"].iloc[0])
        )
        replicates = len(group)
        summary = summarize_replications(
            tuple(
                ReplicationRecord(
                    replicate=index,
                    seed=-1,
                    estimand=str(estimand),
                    truth=float(row.truth),
                    estimate=float(row.estimate),
                    std_error=float(row.std_error),
                    covered=bool(row.covered),
                    rejected=False,
                    inference_estimate=float(row.inference_estimate),
                    alpha=0.05,
                )
                for index, row in enumerate(group.itertuples(index=False))
            ),
            estimand=str(estimand),
            n=int(group["n"].iloc[0]),
        )
        empirical_se = float(np.std(inference, ddof=1))
        inference_bias = float(np.mean(inference) - truth_inference)
        records.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "estimand": estimand,
                "n": int(group["n"].iloc[0]),
                "replicates": replicates,
                "truth": truth,
                "mean_estimate": summary.mean_estimate,
                "bias": summary.bias,
                "bias_se": summary.bias_se,
                "root_n_bias": summary.root_n_bias,
                "rmse": summary.rmse,
                "empirical_se": empirical_se,
                "mean_std_error": summary.mean_std_error,
                "se_ratio": summary.se_ratio,
                "coverage": summary.coverage,
                "coverage_se": summary.coverage_se,
                "inference_bias": inference_bias,
                "standardized_bias": inference_bias / empirical_se,
            }
        )
    return pd.DataFrame.from_records(records)


def se_ratio_interval(
    inference: np.ndarray,
    std_errors: np.ndarray,
    *,
    record: StudyRecord,
    seed: int,
) -> Interval:
    """Bootstrap interval for mean reported SE over empirical SD."""
    samples = bootstrap(
        {"inference": inference, "std_error": std_errors},
        {
            "ratio": lambda draw: (
                draw["std_error"].mean(axis=1) / draw["inference"].std(axis=1, ddof=1)
            )
        },
        replicates=record.margins.bootstrap_replicates,
        seed=seed,
    )["ratio"]
    return percentile_interval(samples, confidence_level=record.margins.confidence_level)


def _cell(payload: tuple[StudyRecord, pd.DataFrame, tuple[str, str, str], int]) -> dict[str, Any]:
    """One implementation-scenario-estimand verdict, as a picklable unit of work."""
    record, group, key, seed = payload
    implementation, scenario, estimand = key
    margins = record.margins
    inference = group["inference_estimate"].to_numpy(dtype=float)
    std_errors = group["std_error"].to_numpy(dtype=float)
    truth = float(group["truth"].iloc[0])
    scale = str(group["inference_scale"].iloc[0])
    errors = inference - truth_on_inference_scale(estimand, truth, scale)
    replicates = len(group)

    bias = standardized_bias_verdict(
        errors,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
    )
    covered = int(group["covered"].sum())
    coverage = clopper_pearson(covered, replicates, confidence_level=margins.confidence_level)
    ratio = se_ratio_interval(inference, std_errors, record=record, seed=seed)
    se_calibrated = ratio.within(*margins.se_ratio_sanity)
    coverage_valid = bool(coverage.low >= margins.coverage_floor)
    return {
        "implementation": implementation,
        "scenario": scenario,
        "estimand": estimand,
        "replicates": replicates,
        "confidence_level": margins.confidence_level,
        "inference_scale": scale,
        "bias": bias.bias,
        "bias_ci_lower": bias.interval.low,
        "bias_ci_upper": bias.interval.high,
        "bias_margin": bias.margin,
        "standardized_bias": bias.standardized,
        "bias_equivalent": bias.equivalent,
        "coverage": covered / replicates,
        "coverage_ci_lower": coverage.low,
        "coverage_ci_upper": coverage.high,
        "coverage_floor": margins.coverage_floor,
        "coverage_valid": coverage_valid,
        "over_covered": bool(coverage.low > margins.over_coverage_ceiling),
        "se_ratio": float(np.mean(std_errors) / np.std(inference, ddof=1)),
        "se_ratio_ci_lower": ratio.low,
        "se_ratio_ci_upper": ratio.high,
        "se_ratio_resolution": ratio.resolution(1.0),
        "se_ratio_margin_lower": margins.se_ratio_sanity[0],
        "se_ratio_margin_upper": margins.se_ratio_sanity[1],
        "se_calibrated": se_calibrated,
        "passed": bool(bias.equivalent and coverage_valid and se_calibrated),
    }


def independent_performance_tests(
    rows: pd.DataFrame,
    *,
    record: StudyRecord,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """Test each implementation against known truth with pre-declared margins.

    Three claims, each bounded by a margin rather than tested against zero:

    *bias* -- the Student interval for the error on the reported scale lies within
    ``standardized_bias`` empirical standard deviations of zero;

    *coverage* -- the exact interval's lower endpoint clears ``coverage_floor``.  One-sided
    on purpose.  Asking the coverage interval to *contain* the nominal rate is a test of
    exact nominal coverage, which no estimator with a finite-sample remainder satisfies, so
    it starts failing as soon as the Monte Carlo error drops below the real shortfall --
    that is, it punishes the study for adding replications.  Whether the interval is valid
    is the question; whether it is valid to the third decimal is not;

    *reported standard error* -- a two-sided sanity band, no tighter than the coverage floor
    implies, with the resolution it achieved recorded beside it so the band cannot be read as
    a tighter calibration claim than the replication count supports.
    """
    cells = list(rows.groupby(CELL_KEYS, sort=True))
    payloads = [
        (
            (
                record,
                group,
                (str(key[0]), str(key[1]), str(key[2])),
                record.seed + 10_000 + index,
            ),
        )
        for index, (key, group) in enumerate(cells)
    ]
    return pd.DataFrame.from_records(map_parallel(_cell, payloads, n_jobs=n_jobs))
