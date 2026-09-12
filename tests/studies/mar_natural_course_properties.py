"""Repeated-sampling properties for the MAR natural-course mean."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_mar_natural_course import (
    EFFICIENCY_SD,
    ESTIMANDS,
    NUISANCE_BOUND,
    STUDY,
    initial_estimate,
)
from tests.studies.evidence.properties import (
    ReplicationSpec,
    control_row,
    property_role,
    replicate_row,
    replication_payloads,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.missing_outcome_study_helpers import (
    FailTreatment,
    NaturalCourseLaw,
    sample_discrete,
)

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
MISSINGNESS_DISPLACEMENT = 0.25
TARGET = "ey_obs"
SCENARIO = "binary_mar_natural_course"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

WRONG_Q = 1.0 - mar.Q
WRONG_PI = np.array([[0.80, 0.75], [0.55, 0.45], [0.30, 0.25]])
TRUTH = float(mar.functional(mar.PROBS, TARGET))


def _law(configuration: str) -> NaturalCourseLaw:
    q = mar.Q if configuration in {"both_correct", "outcome_correct"} else WRONG_Q
    pi = mar.PI if configuration in {"both_correct", "response_correct"} else WRONG_PI
    return NaturalCourseLaw(q=q, pi=pi)


def _fit(frame: pd.DataFrame, configuration: str, *, delta: bool = True) -> Any:
    law = _law(configuration)
    return (
        TMLE(
            estimands=ESTIMANDS,
            outcome_learner=OracleOutcome(law),
            treatment_learner=FailTreatment() if delta else OracleTreatment(law),
            missingness_learner=OracleMissingness(law) if delta else None,
            cross_fit=False,
            fluctuation="logistic",
            targeting="iterative",
            target_weights=False,
            simultaneous=False,
            nuisance_bound=NUISANCE_BOUND,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            **({"delta": "Delta"} if delta else {}),
        )
        .single()
    )


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    frame = sample_discrete(mar.PROBS, n, seed)
    result = _fit(frame, configuration)
    role = property_role(
        configuration,
        controls={"both_wrong"},
        property_name=property_name,
        n=n,
        rate_sizes=RATE_SIZES,
    )
    rows = [
        replicate_row(
            property_name=property_name,
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=TRUTH,
            estimate=result[TARGET],
            alpha=STUDY.margins.alpha,
        )
    ]
    if property_name == "targeting_necessity":
        rows[0]["cell"] = "ey_obs__targeted"
        rows.append(
            control_row(
                property_name=property_name,
                cell="ey_obs__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=TRUTH,
                estimate=initial_estimate(result),
                standard_error=float(result[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "missingness_necessity":
        rows[0]["cell"] = "ey_obs__declared"
        complete = frame.loc[frame["Delta"] == 1.0, ["W", "A", "Y"]].reset_index(drop=True)
        ignored = _fit(complete, configuration, delta=False)
        rows.append(
            control_row(
                property_name=property_name,
                cell="ey_obs__complete_case_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=TRUTH,
                estimate=float(ignored[TARGET].psi),
                standard_error=float(ignored[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs = [
        ReplicationSpec(
            "double_robustness",
            configuration,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
        for configuration in ("both_correct", "outcome_correct", "response_correct", "both_wrong")
    ]
    specs.extend(
        ReplicationSpec("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct")
        for size in RATE_SIZES
    )
    specs.extend(
        [
            ReplicationSpec(
                "interval_calibration",
                "ey_obs__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ReplicationSpec(
                "targeting_necessity",
                "targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "response_correct",
            ),
            ReplicationSpec(
                "missingness_necessity",
                "declared",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "response_correct",
            ),
        ]
    )
    return replication_payloads(STUDY, specs)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    controls = calibration_controls(
        rows,
        STUDY,
        labels=(TARGET,),
        efficiency_bounds={TARGET: EFFICIENCY_SD},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "missingness_displacement"),
        efficiency_bounds={TARGET: EFFICIENCY_SD},
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=(TARGET,),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="missingness_necessity",
        labels=(TARGET,),
        arms=("declared", "complete_case_control"),
        column="missingness_displacement",
        threshold=MISSINGNESS_DISPLACEMENT,
    )
    return finish(summary, rates)
