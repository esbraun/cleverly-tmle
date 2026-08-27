"""Repeated-sampling properties for outcome-adaptive multi-arm C-TMLE."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common, multi_arm_properties
from tests.studies.canonical_multi_arm_ctmle_oat import STUDY
from tests.studies.evidence.properties import (
    PropertyCell,
    run_cells,
    se_ratio_deficit_interval,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.seeds import stream_seed

GENERATED_DESIGN_DEFICIT = 0.01


def cells() -> tuple[PropertyCell, ...]:
    asymptotic = tuple(
        replace(
            cell,
            outcome_learner=multi_arm_properties.oracle_outcome(cell.dgp.effect),
        )
        for cell in multi_arm_properties.asymptotic_cells(seed=23_100)
    )
    return (
        *multi_arm_properties.oat_robustness_cells(seed=23_100),
        *asymptotic,
        *multi_arm_properties.generated_design_cells(seed=23_500),
    )


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    return lambda: CTMLE(
        strategy="oat",
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=True,
        n_folds=5,
        estimands="ate",
        reference=multi_arm_common.REFERENCE,
        simultaneous=False,
        g_bounds=multi_arm_common.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("se_ratio_deficit_lower", "se_ratio_deficit_upper"),
    )
    robustness = summary["property"] == "robustness_contract"
    positive = robustness & (summary["role"] == "positive")
    control = robustness & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    generated = rows.loc[rows["property"] == "generated_design"]
    oracle = generated.loc[generated["cell"] == "oracle_design"].sort_values("replicate")
    estimated = generated.loc[generated["cell"] == "estimated"].sort_values("replicate")
    oracle_interval = se_ratio_interval(
        oracle,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", "oracle_design"),
    )
    deficit = se_ratio_deficit_interval(
        estimated,
        oracle,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", "deficit"),
    )
    estimated_interval = se_ratio_interval(
        estimated,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", "estimated"),
    )
    verdicts = {
        "oracle_design": oracle_interval.within(*STUDY.margins.calibration_se_ratio),
        "estimated": deficit.high <= -GENERATED_DESIGN_DEFICIT,
    }
    joint = bool(all(verdicts.values()))
    for cell, interval in (
        ("oracle_design", oracle_interval),
        ("estimated", estimated_interval),
    ):
        mask = (summary["property"] == "generated_design") & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "se_ratio_deficit_lower"] = deficit.low
        summary.loc[mask, "se_ratio_deficit_upper"] = deficit.high
        summary.loc[mask, "passed"] = verdicts[cell]
        summary.loc[mask, "property_passed"] = joint
    return finish(summary, rates)
