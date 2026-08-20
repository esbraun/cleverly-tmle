"""Shared repeated-sampling claims for the two literature-backed CV-TMLE reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_properties
from tests.studies.evidence.inference import percentile_interval
from tests.studies.evidence.properties import PropertyCell, rate, run_cells, summarize_cells
from tests.studies.evidence.registry import StudyRecord

OVERFIT_REPLICATES = 400
OVERFIT_N = 500
OVERFIT_SE_FLOOR = 0.85
OVERFIT_SE_CONTROL_CEILING = 0.75
OVERFIT_COVERAGE_GAIN = 0.15


def cells(variant: str) -> tuple[PropertyCell, ...]:
    """The ordinary TMLE claims plus the overfitting experiment CV-TMLE exists for."""
    inherited = tuple(replace(cell) for cell in canonical_properties.cells())
    dgp = nonlinear_dgp()
    overfit = (
        PropertyCell(
            property="crossfit_overfitting",
            cell=f"{variant}_cvtmle",
            dgp=dgp,
            outcome_learner=lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            treatment_learner=canonical_properties.LogisticRegression,
            n=OVERFIT_N,
            replicates=OVERFIT_REPLICATES,
            seed=10_100,
        ),
        PropertyCell(
            property="crossfit_overfitting",
            cell="in_sample_control",
            dgp=dgp,
            outcome_learner=lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            treatment_learner=canonical_properties.LogisticRegression,
            n=OVERFIT_N,
            replicates=OVERFIT_REPLICATES,
            seed=10_100,
        ),
    )
    return (*inherited, *overfit)


def estimator(record: StudyRecord, variant: str) -> Callable[[PropertyCell], Callable[[], Any]]:
    def factory(cell: PropertyCell) -> Callable[[], Any]:
        control = cell.property == "crossfit_overfitting" and cell.cell == "in_sample_control"
        return lambda: TMLE(
            outcome_learner=cell.outcome_learner(),
            treatment_learner=cell.treatment_learner(),
            cross_fit=not control,
            n_folds=10,
            targeting_scheme="pooled",
            cv_evaluation=variant == "fold_evaluated" and not control,
            estimands=cell.estimand,
            simultaneous=False,
            g_bounds=(0.025, 0.975),
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )

    return factory


def generate(record: StudyRecord, variant: str, *, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(variant), estimator(record, variant), n_jobs=n_jobs)


def _se_ratio_interval(
    group: pd.DataFrame, *, replicates: int, confidence_level: float, seed: int
) -> tuple[float, float]:
    values = group[["estimate", "std_error"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[picks]
    ratios = draws[:, :, 1].mean(axis=1) / draws[:, :, 0].std(axis=1, ddof=1)
    interval = percentile_interval(ratios, confidence_level=confidence_level)
    return interval.low, interval.high


def _coverage_gain_interval(
    positive: pd.DataFrame,
    control: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    paired = positive[["replicate", "covered"]].merge(
        control[["replicate", "covered"]], on="replicate", suffixes=("_positive", "_control")
    )
    differences = paired["covered_positive"].to_numpy(dtype=float) - paired[
        "covered_control"
    ].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(differences), size=(replicates, len(differences)))
    interval = percentile_interval(
        differences[picks].mean(axis=1), confidence_level=confidence_level
    )
    return interval.low, interval.high


def summarize(rows: pd.DataFrame, record: StudyRecord, variant: str) -> pd.DataFrame:
    """Summarize shared cells and make the overfitting control load-bearing."""
    margins = record.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    summary["slope"] = np.nan
    summary["slope_ci_lower"] = np.nan
    summary["slope_ci_upper"] = np.nan
    summary["se_ratio_ci_lower"] = np.nan
    summary["se_ratio_ci_upper"] = np.nan
    summary["coverage_gain_ci_lower"] = np.nan
    summary["coverage_gain_ci_upper"] = np.nan
    summary["passed"] = False

    positive = (summary["property"] == "double_robustness") & (summary["cell"] != "both_wrong")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    control = (summary["property"] == "double_robustness") & (summary["cell"] == "both_wrong")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    efficiency = summary["property"] == "root_n_and_efficiency"
    summary.loc[efficiency, "passed"] = (
        (summary.loc[efficiency, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[efficiency, "se_ratio"].between(*margins.se_ratio_sanity)
        & summary.loc[efficiency, "bias_equivalent"]
    )

    null = summary["property"] == "type_i_error"
    summary.loc[null, "passed"] = (
        summary.loc[null, "rejection_ci_upper"] <= margins.alpha + margins.type_i_margin
    ) & (summary.loc[null, "coverage_ci_lower"] >= margins.coverage_floor)
    power = summary["property"] == "power"
    summary.loc[power, "passed"] = (
        summary.loc[power, "rejection_ci_lower"] >= canonical_properties.MINIMUM_POWER
    )

    rates: list[dict[str, Any]] = []
    for statistic, cell in (("spread", "empirical_sd"), ("reported", "reported_se")):
        fitted = rate(
            rows,
            property_name="root_n_and_efficiency",
            statistic=statistic,
            bootstrap_replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=record.seed + 30_000 + len(rates),
        )
        if statistic == "spread":
            rate_passed = fitted.consistent_with(canonical_properties.ROOT_N_SLOPE)
        else:
            # The reported SE rate is largely an arithmetic consequence of the
            # influence-curve scaling.  Use a practical root-n equivalence band
            # instead of requiring an increasingly precise CI to contain -0.5.
            rate_passed = fitted.interval.low >= -0.55 and fitted.interval.high <= -0.45
        row: dict[str, Any] = dict.fromkeys(summary.columns, np.nan)
        row.update(
            {
                "property": "root_n_rate",
                "cell": cell,
                "n": max(canonical_properties.RATE_SIZES),
                "replicates": canonical_properties.RATE_REPLICATES
                * len(canonical_properties.RATE_SIZES),
                "failed_replicates": 0,
                "slope": fitted.slope,
                "slope_ci_lower": fitted.interval.low,
                "slope_ci_upper": fitted.interval.high,
                "passed": bool(
                    rate_passed and fitted.excludes(canonical_properties.EXCLUDED_SLOPE)
                ),
            }
        )
        rates.append(row)

    overfit_rows = rows.loc[rows["property"] == "crossfit_overfitting"]
    positive_name = f"{variant}_cvtmle"
    positive_rows = overfit_rows.loc[overfit_rows["cell"] == positive_name]
    control_rows = overfit_rows.loc[overfit_rows["cell"] == "in_sample_control"]
    positive_se = _se_ratio_interval(
        positive_rows,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=record.seed + 40_000,
    )
    control_se = _se_ratio_interval(
        control_rows,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=record.seed + 40_001,
    )
    gain = _coverage_gain_interval(
        positive_rows,
        control_rows,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=record.seed + 40_002,
    )
    for cell, interval in ((positive_name, positive_se), ("in_sample_control", control_se)):
        mask = (summary["property"] == "crossfit_overfitting") & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval[0]
        summary.loc[mask, "se_ratio_ci_upper"] = interval[1]
        summary.loc[mask, "coverage_gain_ci_lower"] = gain[0]
        summary.loc[mask, "coverage_gain_ci_upper"] = gain[1]
    overfit_passed = bool(
        positive_se[0] >= OVERFIT_SE_FLOOR
        and positive_se[1] <= margins.se_ratio_sanity[1]
        and control_se[1] <= OVERFIT_SE_CONTROL_CEILING
        and gain[0] >= OVERFIT_COVERAGE_GAIN
    )
    summary.loc[summary["property"] == "crossfit_overfitting", "passed"] = overfit_passed

    summary = pd.concat([summary, pd.DataFrame(rates)], ignore_index=True)
    summary["passed"] = summary["passed"].astype(bool)
    return summary.sort_values(["property", "cell"], ignore_index=True)
