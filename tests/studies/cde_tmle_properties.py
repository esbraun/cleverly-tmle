"""Independent repeated-sampling properties for controlled direct-effect TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from tests import discrete_law_cde as cde
from tests.conftest import (
    OracleDirectOutcome,
    OracleIntermediate,
    OracleMissingness,
    OracleTreatment,
)
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cde_tmle import (
    EFFICIENCY_SD,
    G_BOUNDS,
    NUISANCE_BOUND,
    STUDY,
)
from tests.studies.cde_study_helpers import probabilities, sample_discrete, truths
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    ReplicationSpec,
    control_row,
    payload_cell,
    payload_replicate,
    property_role,
    replicate_row,
    replication_payloads,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    robustness_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.point_study_helpers import initial_estimates

ROBUSTNESS_REPLICATES = 1_200
ROBUSTNESS_N = 2_000
RATE_REPLICATES = 2_400
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 12_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
EXACT_SEPARATION_FLOOR = 2.2
TARGET = "ate"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

# Deterministically selected on the exact finite law. Each single-mechanism control has
# an asymptotic standardized separation above EXACT_SEPARATION_FLOOR at n=2,000 for both
# intermediate levels.
WRONG_QBAR = np.array(
    [
        [[0.28, 0.28], [0.12, 0.88]],
        [[0.28, 0.28], [0.28, 0.12]],
        [[0.88, 0.88], [0.12, 0.88]],
    ]
)
WRONG_G = np.array([0.62, 0.38, 0.88])
WRONG_QZ = np.array([[0.45, 0.90], [0.50, 0.60], [0.65, 0.55]])
WRONG_PI = np.array([[0.72, 0.88], [0.28, 0.72], [0.88, 0.28]])
WRONG_PROBS = probabilities(WRONG_QBAR, g=WRONG_G, qz=WRONG_QZ, pi=WRONG_PI)

NULL_QBAR = np.array(
    [
        [[0.25, 0.75], [0.25, 0.75]],
        [[0.50, 0.25], [0.50, 0.25]],
        [[0.75, 0.50], [0.75, 0.50]],
    ]
)
NULL_PROBS = probabilities(NULL_QBAR)
BASE_TRUTHS = {level: truths(cde.PROBS, (TARGET,), level)[TARGET] for level in cde.LEVELS}
NULL_TRUTHS = {level: truths(NULL_PROBS, (TARGET,), level)[TARGET] for level in cde.LEVELS}


def _learners(configuration: str, probs: np.ndarray) -> tuple[Any, Any, Any, Any]:
    correct = cde.DiscreteLaw(probs)
    wrong = cde.DiscreteLaw(WRONG_PROBS)
    outcome = correct if configuration in {"all_correct", "outcome_correct"} else wrong
    treatment = (
        correct
        if configuration
        in {"all_correct", "mechanisms_correct", "intermediate_wrong", "observation_wrong"}
        else wrong
    )
    intermediate = (
        correct
        if configuration
        in {"all_correct", "mechanisms_correct", "treatment_wrong", "observation_wrong"}
        else wrong
    )
    observation = (
        correct
        if configuration
        in {"all_correct", "mechanisms_correct", "treatment_wrong", "intermediate_wrong"}
        else wrong
    )
    return (
        OracleDirectOutcome(outcome),
        OracleTreatment(treatment),
        OracleIntermediate(intermediate),
        OracleMissingness(observation),
    )


def _fit(frame: pd.DataFrame, configuration: str, probs: np.ndarray) -> Any:
    outcome, treatment, intermediate, missingness = _learners(configuration, probs)
    return TMLE(
        estimands=(TARGET,),
        outcome_learner=outcome,
        treatment_learner=treatment,
        intermediate_learner=intermediate,
        missingness_learner=missingness,
        cross_fit=False,
        simultaneous=False,
        g_bounds=G_BOUNDS,
        nuisance_bound=NUISANCE_BOUND,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        intermediate="Z",
        delta="Delta",
    )


#: Each calibration label's positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived(("z0", "z1"))

_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, which is a function of the key alone."""
    property_name, _, _, n, _, seed, _ = key[0]
    probs = NULL_PROBS if property_name == "type_i_error" else cde.PROBS
    return sample_discrete(probs, n, seed)


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    property_name, cell = key[0][:2]
    names: list[str] = []
    for level in cde.LEVELS:
        label = f"z{level}"
        if property_name == "targeting_necessity":
            names += [f"{property_name}/{label}__targeted", f"{property_name}/{label}__untargeted"]
        else:
            names.append(f"{property_name}/{label}__{cell}")
    for name in list(names):
        names += DERIVED_CELLS.get(name, ())
    return names


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    (payload,) = key
    property_name, cell, replicate, n, requested, _seed, configuration = payload
    probs = NULL_PROBS if property_name == "type_i_error" else cde.PROBS
    result = _fit(frame, configuration, probs)
    rows: list[dict[str, Any]] = []
    for level in cde.LEVELS:
        label = f"z{level}"
        truth = (NULL_TRUTHS if property_name == "type_i_error" else BASE_TRUTHS)[level]
        role = property_role(
            configuration,
            controls={"treatment_wrong", "intermediate_wrong", "observation_wrong"},
            property_name=property_name,
            n=n,
            rate_sizes=RATE_SIZES,
        )
        estimate = result[float(level)][TARGET]
        row = replicate_row(
            property_name=property_name,
            cell=f"{label}__{cell}",
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth,
            estimate=estimate,
            alpha=STUDY.margins.alpha,
        )
        rows.append(row)
        if property_name == "targeting_necessity":
            rows[-1]["cell"] = f"{label}__targeted"
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=initial_estimates(result[float(level)])[TARGET],
                    standard_error=float(estimate.std_error),
                    critical=CRITICAL,
                )
            )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[ReplicationSpec] = []
    for configuration in (
        "all_correct",
        "outcome_correct",
        "mechanisms_correct",
        "treatment_wrong",
        "intermediate_wrong",
        "observation_wrong",
    ):
        specs.append(
            ReplicationSpec(
                "cde_robustness",
                configuration,
                ROBUSTNESS_N,
                ROBUSTNESS_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(
            ReplicationSpec(
                "root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "all_correct"
            )
        )
    specs.extend(
        [
            ReplicationSpec(
                "interval_calibration",
                "correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "all_correct",
            ),
            ReplicationSpec("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "all_correct"),
            ReplicationSpec("power", "alternative", NULL_N, NULL_REPLICATES, "all_correct"),
            ReplicationSpec(
                "targeting_necessity",
                "targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "mechanisms_correct",
            ),
        ]
    )
    return replication_payloads(STUDY, specs)


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = fit_properties(
        _payloads(),
        _fit_replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    controls = calibration_controls(
        rows,
        STUDY,
        labels=("z0", "z1"),
        efficiency_bounds=EFFICIENCY_SD,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement",),
        rate_labels=("z0", "z1"),
        efficiency_bounds=EFFICIENCY_SD,
    )
    robustness_verdicts(summary, family="cde_robustness")
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("z0", "z1"),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)
