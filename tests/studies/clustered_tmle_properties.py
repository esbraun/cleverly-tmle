"""Cluster-robust inference property for point-treatment CV-TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.inference import influence_variance
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_clustered_tmle import (
    PRIMARY_N,
    PROPERTY_REPLICATES,
    SCENARIO,
    STUDY,
    draw_from_seed,
    fit_cleverly,
)
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    control_row,
    coverage_gain_interval,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.seeds import stream_seed

CONTROL_SE_RATIO_CEILING = 0.80
COVERAGE_GAIN = 0.03
TARGET = "ate"
FAMILY = "clustered_inference"
POSITIVE = "cluster_robust"
CONTROL = "iid_control"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def _fit_replication(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, seed = payload
    frame, truth = draw_from_seed(SCENARIO, PRIMARY_N, seed)
    result = fit_cleverly(frame)
    estimate = result[TARGET]
    robust = replicate_row(
        property_name=FAMILY,
        cell=POSITIVE,
        role="positive",
        replicate=replicate,
        n=PRIMARY_N,
        requested=PROPERTY_REPLICATES,
        truth=truth[TARGET],
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    iid_standard_error = float(np.sqrt(influence_variance(estimate.influence_curve)))
    iid = control_row(
        property_name=FAMILY,
        cell=CONTROL,
        replicate=replicate,
        n=PRIMARY_N,
        requested=PROPERTY_REPLICATES,
        truth=truth[TARGET],
        estimate=float(estimate.psi),
        standard_error=iid_standard_error,
        critical=CRITICAL,
    )
    if robust["estimate"] != iid["estimate"]:
        raise AssertionError("the IID control changed the point estimate")
    return [robust, iid]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit each draw once and derive both variance cells from its ATE influence curve."""
    payloads = [
        (
            (
                replicate,
                stream_seed(STUDY, "property_sample", FAMILY, "paired", replicate),
            ),
        )
        for replicate in range(PROPERTY_REPLICATES)
    ]
    outcomes = map_parallel(_fit_replication, payloads, n_jobs=n_jobs)
    return pd.DataFrame([row for rows in outcomes for row in rows])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared calibration, IID-control, and paired-gain rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        rate_labels=(),
    )
    margins = STUDY.margins
    selected = rows.loc[rows["property"] == FAMILY]
    positive = selected.loc[selected["cell"] == POSITIVE]
    control = selected.loc[selected["cell"] == CONTROL]
    positive_se = se_ratio_interval(
        positive,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(STUDY, FAMILY, POSITIVE),
    )
    control_se = se_ratio_interval(
        control,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(STUDY, FAMILY, CONTROL),
    )
    gain = coverage_gain_interval(
        positive,
        control,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(STUDY, FAMILY, "coverage_gain"),
    )

    positive_row = summary.loc[
        (summary["property"] == FAMILY) & (summary["cell"] == POSITIVE)
    ].iloc[0]
    coverage = Interval(
        float(positive_row["coverage_ci_lower"]),
        float(positive_row["coverage_ci_upper"]),
    )
    verdicts = {
        POSITIVE: bool(
            positive_se.within(*margins.calibration_se_ratio)
            and coverage.within(*margins.calibration_coverage)
        ),
        CONTROL: bool(control_se.high <= CONTROL_SE_RATIO_CEILING),
    }
    joint = bool(all(verdicts.values()) and gain[0] >= COVERAGE_GAIN)
    for cell, interval in ((POSITIVE, positive_se), (CONTROL, control_se)):
        mask = (summary["property"] == FAMILY) & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "coverage_gain_ci_lower"] = gain[0]
        summary.loc[mask, "coverage_gain_ci_upper"] = gain[1]
        summary.loc[mask, "passed"] = verdicts[cell]
        summary.loc[mask, "property_passed"] = joint
    return finish(summary, rates)
