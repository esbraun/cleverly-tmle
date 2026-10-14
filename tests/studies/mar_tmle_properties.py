"""Independent repeated-sampling properties for ordinary MAR TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_mar_tmle import G_BOUNDS, NUISANCE_BOUND, STUDY
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
from tests.studies.missing_outcome_study_helpers import (
    WRONG_PI,
    WRONG_Q,
    efficiency_sd,
    probabilities,
    sample_discrete,
)
from tests.studies.point_study_helpers import initial_estimates

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
MISSINGNESS_DISPLACEMENT = 0.25
TARGET = "ate"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

WRONG_G = np.array([0.70, 0.30, 0.70])
WRONG_PROBS = probabilities(WRONG_Q, g=WRONG_G, pi=WRONG_PI)
NULL_Q = np.column_stack([np.array([0.30, 0.50, 0.70])] * 2)
NULL_PROBS = probabilities(NULL_Q)
TRUTH = float(mar.functional(mar.PROBS, TARGET))
NULL_TRUTH = float(mar.functional(NULL_PROBS, TARGET))
EFFICIENCY_SD = efficiency_sd(mar.PROBS, TARGET)


def _learners(configuration: str, probs: np.ndarray) -> tuple[Any, Any, Any]:
    correct = mar.DiscreteLaw(probs)
    wrong = mar.DiscreteLaw(WRONG_PROBS)
    q = correct if configuration in {"both_correct", "outcome_correct"} else wrong
    g = (
        correct
        if configuration in {"both_correct", "mechanisms_correct", "observation_wrong"}
        else wrong
    )
    pi = (
        correct
        if configuration in {"both_correct", "mechanisms_correct", "treatment_wrong"}
        else wrong
    )
    return OracleOutcome(q), OracleTreatment(g), OracleMissingness(pi)


def _fit(
    frame: pd.DataFrame,
    configuration: str,
    probs: np.ndarray,
    *,
    delta: bool = True,
) -> Any:
    outcome, treatment, missingness = _learners(configuration, probs)
    return (
        TMLE(
            estimands=(TARGET,),
            outcome_learner=outcome,
            treatment_learner=treatment,
            missingness_learner=missingness if delta else None,
            cross_fit=False,
            simultaneous=False,
            g_bounds=G_BOUNDS,
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


#: The calibration positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived(("ate",))

_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, which is a function of the key alone."""
    property_name, _, _, n, _, seed, _ = key[0]
    probs = NULL_PROBS if property_name == "type_i_error" else mar.PROBS
    return sample_discrete(probs, n, seed)


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    property_name, cell = key[0][:2]
    if property_name == "targeting_necessity":
        return [f"{property_name}/ate__targeted", f"{property_name}/ate__untargeted"]
    if property_name == "missingness_necessity":
        return [f"{property_name}/ate__declared", f"{property_name}/ate__complete_case_control"]
    name = f"{property_name}/{cell}"
    return [name, *DERIVED_CELLS.get(name, ())]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    (payload,) = key
    property_name, cell, replicate, n, requested, _seed, configuration = payload
    probs = NULL_PROBS if property_name == "type_i_error" else mar.PROBS
    result = _fit(frame, configuration, probs)
    truth = NULL_TRUTH if property_name == "type_i_error" else TRUTH
    role = property_role(
        configuration,
        controls={"treatment_wrong", "observation_wrong"},
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
    if property_name == "missingness_necessity":
        rows[0]["cell"] = "ate__declared"
        complete = frame.loc[frame["Delta"] == 1.0, ["W", "A", "Y"]].reset_index(drop=True)
        ignored = _fit(complete, configuration, probs, delta=False)
        rows.append(
            control_row(
                property_name=property_name,
                cell="ate__complete_case_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(ignored[TARGET].psi),
                standard_error=float(ignored[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[ReplicationSpec] = []
    for configuration in (
        "both_correct",
        "outcome_correct",
        "mechanisms_correct",
        "treatment_wrong",
        "observation_wrong",
    ):
        specs.append(
            ReplicationSpec(
                "mar_robustness",
                configuration,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(
            ReplicationSpec(
                "root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct"
            )
        )
    specs.extend(
        [
            ReplicationSpec(
                "interval_calibration",
                "ate__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ReplicationSpec("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "both_correct"),
            ReplicationSpec("power", "alternative", NULL_N, NULL_REPLICATES, "both_correct"),
            ReplicationSpec(
                "targeting_necessity",
                "targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "mechanisms_correct",
            ),
            ReplicationSpec(
                "missingness_necessity",
                "declared",
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

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


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
        labels=("ate",),
        efficiency_bounds={"ate": EFFICIENCY_SD},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.
TARGETING_NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=("ate",),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
)

#: The ``missingness_necessity`` family's declaration, read by its verdict and its rule.
MISSINGNESS_NECESSITY = property_verdicts.Necessity(
    family="missingness_necessity",
    labels=("ate",),
    arms=("declared", "complete_case_control"),
    column="missingness_displacement",
    threshold=MISSINGNESS_DISPLACEMENT,
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "missingness_displacement"),
        efficiency_bounds={"ate": EFFICIENCY_SD},
    )
    robustness_verdicts(summary, family="mar_robustness")
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    necessity_verdicts(summary, rows, MISSINGNESS_NECESSITY)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY, MISSINGNESS_NECESSITY),
)
