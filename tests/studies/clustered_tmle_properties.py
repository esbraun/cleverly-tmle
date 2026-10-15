"""Cluster-robust inference property for point-treatment CV-TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.inference import influence_variance
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_clustered_tmle import (
    PRIMARY_N,
    PROPERTY_REPLICATES,
    SCENARIO,
    STUDY,
    draw_from_seed,
    fit_cleverly,
    law,
    outcome_learner,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import PropertyCell, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    CLUSTER_ROBUST_CONTROL_SE_CEILING,
    CLUSTERED_COVERAGE_GAIN,
    apply_shared_verdicts,
    clustered_inference_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
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


def declared_cells() -> tuple[PropertyCell, ...]:
    """Both arms of this study's one property family, as declarations of the law they read.

    The two arms are one fit, not two: :func:`_fit_replication` draws once, fits once, and
    derives the cluster-robust and the i.i.d. variance from the same influence curve.  The
    declaration still names both, because the committed rows publish both and the truth
    each one carries has to be read back against the law that produced it.

    ``seed`` names the per-replication stream :func:`generate_property_rows` draws from
    rather than a single sample's seed, and nothing here is passed to
    :func:`~tests.studies.evidence.properties.run_cells`.

    Returns
    -------
    tuple of PropertyCell
        The cluster-robust positive arm, then its i.i.d. control.
    """
    dgp = law()
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=cell,
            dgp=dgp,
            outcome_learner=outcome_learner,
            treatment_learner=lambda: OracleTreatment(law()),
            n=PRIMARY_N,
            replicates=PROPERTY_REPLICATES,
            seed=stream_seed(STUDY, "property_sample", FAMILY, "paired", 0),
            role=role,
            estimand=TARGET,
        )
        for cell, role in ((POSITIVE, "positive"), (CONTROL, "control"))
    )


def _draw_replication(payload: tuple[int, int]) -> Any:
    return draw_from_seed(SCENARIO, PRIMARY_N, payload[1])


def _fit_replication(payload: tuple[int, int]) -> list[dict[str, Any]]:
    return _fit_drawn(payload, _draw_replication(payload))


def _fit_drawn(payload: tuple[int, int], drawn: Any) -> list[dict[str, Any]]:
    replicate, _ = payload
    frame, truth = drawn
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


def _cells_of(key: tuple[tuple[int, int]]) -> list[str]:
    return [f"{FAMILY}/{POSITIVE}", f"{FAMILY}/{CONTROL}"]


def _cell_of(key: tuple[tuple[int, int]]) -> str:
    return f"{FAMILY}/{POSITIVE}"


def _replicate_of(key: tuple[tuple[int, int]]) -> int:
    return key[0][0]


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


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
    outcomes = fit_properties(
        payloads,
        splat_drawn(_fit_drawn),
        cells_of=_cells_of,
        draw=splat_draw(_draw_replication),
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
    )
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
