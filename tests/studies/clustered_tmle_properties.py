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
from tests.studies.evidence.properties import control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    CLUSTER_ROBUST_CONTROL_SE_CEILING,
    CLUSTERED_COVERAGE_GAIN,
    apply_shared_verdicts,
    clustered_inference_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

#: Named here as well because the two thresholds are this study's declared margins, and
#: ``test_the_registered_design_matches_the_declared_plan`` reads them off the study module.
#: Bound to the shared declaration rather than retyped, so one literal states each number.
CONTROL_SE_RATIO_CEILING = CLUSTER_ROBUST_CONTROL_SE_CEILING
COVERAGE_GAIN = CLUSTERED_COVERAGE_GAIN
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
    clustered_inference_verdicts(summary, rows, STUDY, positive_cell=POSITIVE, control_cell=CONTROL)
    return finish(summary, rates)
