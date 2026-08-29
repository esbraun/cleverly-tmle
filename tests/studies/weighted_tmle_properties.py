"""Independent repeated-sampling properties for weighted point-treatment TMLE."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly._typing import EstimandName
from cleverly.utils.parallel import map_parallel
from tests.conftest import OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_weighted_tmle import STUDY, fit_cleverly
from tests.studies.evidence.properties import control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    alternative_target_necessity_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.point_study_helpers import initial_estimates
from tests.studies.weighted_point_common import (
    FinitePointLaw,
    G,
    Q,
    population_truth,
    sample_selected,
    selected_truth,
    weighted_ate_efficiency_sd,
)

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 1_000
#: The power cell is a paired size for the type-I cell, so it is sized identically.  Bound
#: to those constants rather than retyped, and named separately so the cell that reads a
#: budget reads a constant that names it.
POWER_REPLICATES = NULL_REPLICATES
POWER_N = NULL_N
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
WEIGHT_DISPLACEMENT = 0.50
ALTERNATIVE_EFFECT = 0.50
TARGET: EstimandName = "ate"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: Deliberately misspecified nuisances. Each stays bounded away from zero and one, and the
#: joint misspecification moves the ATE enough for the registered negative control to resolve.
WRONG_Q = np.array([[0.55, 0.65], [0.45, 0.35], [0.55, 0.25]])
WRONG_G = np.array([0.70, 0.30, 0.70])
NULL_Q = np.column_stack([Q[:, 0], Q[:, 0]])
ALTERNATIVE_Q = np.column_stack([Q[:, 0], Q[:, 0] + ALTERNATIVE_EFFECT])
TRUTH = float(population_truth(Q)[TARGET])
SELECTED_TRUTH = float(selected_truth(Q)[TARGET])
EFFICIENCY_SD = weighted_ate_efficiency_sd(Q)


def _learners(configuration: str, q: np.ndarray) -> tuple[Any, Any]:
    """Return the declared exact or deliberately wrong nuisance pair."""
    outcome_q = q if configuration in {"both_correct", "outcome_correct"} else WRONG_Q
    treatment_g = G if configuration in {"both_correct", "treatment_correct"} else WRONG_G
    return OracleOutcome(FinitePointLaw(q=outcome_q)), OracleTreatment(
        FinitePointLaw(q=q, g=treatment_g)
    )


def _law_for(property_name: str) -> np.ndarray:
    if property_name == "type_i_error":
        return NULL_Q
    if property_name == "power":
        return ALTERNATIVE_Q
    return Q


def fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    """Run one property replication and return the rows its cell publishes.

    Public because two fast-tier controls call it directly.  Each fits one declared
    payload and checks that the paired arms separate in the direction the study claims,
    which is a statement about this function rather than about a private helper.

    Parameters
    ----------
    payload : tuple
        The property name, cell, replicate index, sample size, requested replication
        count, seed, and nuisance configuration, in that order.

    Returns
    -------
    list of dict
        One row for the cell, plus the paired control row where the family declares one.
    """
    property_name, cell, replicate, n, requested, seed, configuration = payload
    q = _law_for(property_name)
    frame = sample_selected(q, n, seed)
    truth = float(population_truth(q)[TARGET])
    outcome, treatment = _learners(configuration, q)
    fit = partial(
        fit_cleverly,
        estimands=(TARGET,),
        outcome_learner=outcome,
        treatment_learner=treatment,
    )
    result = fit(frame)
    role = "control" if cell == "both_wrong" else "positive"
    rows = [
        replicate_row(
            property_name=property_name,
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth,
            estimate=result[TARGET],
            alpha=STUDY.margins.alpha,
        )
    ]
    if property_name == "targeting_necessity":
        rows[0]["cell"] = "ate__targeted"
        rows.append(
            control_row(
                property_name=property_name,
                cell="ate__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=initial_estimates(result)[TARGET],
                standard_error=float(result[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "weight_necessity":
        rows[0]["cell"] = "ate__weighted"
        omitted = fit(frame, use_weights=False)
        rows.append(
            control_row(
                property_name=property_name,
                cell="ate__omitted_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(omitted[TARGET].psi),
                standard_error=float(omitted[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str, str]] = []
    for configuration in ("both_correct", "outcome_correct", "treatment_correct", "both_wrong"):
        specs.append(
            (
                "double_robustness",
                configuration,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
                "paired",
            )
        )
    for size in RATE_SIZES:
        cell = f"n_{size}"
        specs.append(("root_n_and_efficiency", cell, size, RATE_REPLICATES, "both_correct", cell))
    specs.extend(
        [
            (
                "interval_calibration",
                "ate__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
                "correctly_specified",
            ),
            (
                "type_i_error",
                "sharp_null",
                NULL_N,
                NULL_REPLICATES,
                "both_correct",
                "sharp_null",
            ),
            (
                "power",
                "alternative",
                POWER_N,
                POWER_REPLICATES,
                "both_correct",
                "alternative",
            ),
            (
                "targeting_necessity",
                "targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "treatment_correct",
                "paired",
            ),
            (
                "weight_necessity",
                "weighted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "both_correct",
                "paired",
            ),
        ]
    )
    payloads: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, requested, configuration, stream_cell in specs:
        for replicate in range(requested):
            seed = stream_seed(
                STUDY,
                "property_sample",
                property_name,
                stream_cell,
                replicate,
            )
            payloads.append(((property_name, cell, replicate, n, requested, seed, configuration),))
    return payloads


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Run the declared cells and derive paired calibration controls from the same draws."""
    outcomes = map_parallel(fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
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
    """Apply the shared verdicts plus targeting and weight-necessity rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "necessity_displacement",
            "alternative_truth",
            "alternative_bias_ci_lower",
            "alternative_bias_ci_upper",
            "alternative_bias_margin",
            "alternative_bias_equivalent",
        ),
        efficiency_bounds={TARGET: EFFICIENCY_SD},
    )
    calibration_verdicts(
        summary,
        margins=STUDY.margins,
        efficiency_band=EFFICIENCY_RATIO_BAND,
    )
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=(TARGET,),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    alternative_target_necessity_verdicts(
        summary,
        rows,
        STUDY,
        family="weight_necessity",
        labels=(TARGET,),
        arms=("weighted", "omitted_control"),
        alternative_truths={TARGET: SELECTED_TRUTH},
        column="necessity_displacement",
        threshold=WEIGHT_DISPLACEMENT,
    )
    return finish(summary, rates)
