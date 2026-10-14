"""Independent repeated-sampling properties for randomized MAR DR-TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_mar_drtmle import (
    CONFIGURATION,
    G_BOUNDS,
    MAX_OUTER,
    NUISANCE_BOUND,
    PROBS,
    RANDOMIZATION,
    STUDY,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    robustness_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.missing_outcome_study_helpers import (
    efficiency_sd,
    probabilities,
    sample_discrete,
)

ROBUSTNESS_REPLICATES = 800
ROBUSTNESS_N = 2_000
#: Larger than the ladder budget the ordinary MAR study runs, and the reason is the rule the
#: smallest rung answers to.  That rung is a control, so it must *resolve*: its exact coverage
#: interval has to land clear of nominal on one side or the other, and an interval straddling
#: nominal establishes nothing in either direction.  At 400 replications the 99% interval ran
#: 0.8996 to 0.9646 around a point estimate of 0.9375 and straddled.  Resolution is bought with
#: replications and nothing else -- the floor it must clear does not move -- so the budget is
#: set from the width the verdict needs: at 1,200 the lower endpoint clears 0.90 for any true
#: coverage at or above 0.93, which the other two rungs put this estimator well inside.
RATE_REPLICATES = 1_200
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
CORRECTION_REPLICATES = 1_200
CORRECTION_N = 2_000
CORRECTION_SCORE_RATIO = 0.01
UNCORRECTED_SCORE_FLOOR = 1e-3
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGET = "ate"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

WRONG_Q = np.full_like(mar.Q, 0.5)
WRONG_PI = np.full_like(mar.PI, 0.6)
WRONG_PROBS = probabilities(WRONG_Q, g=RANDOMIZATION, pi=WRONG_PI)
TRUTH = float(mar.functional(PROBS, TARGET))
EFFICIENCY_SD = efficiency_sd(PROBS, TARGET)


def _fit(frame: pd.DataFrame, configuration: str) -> Any:
    correct = mar.DiscreteLaw(PROBS)
    wrong = mar.DiscreteLaw(WRONG_PROBS)
    q = correct if configuration in {"both_correct", "observation_drift"} else wrong
    pi = correct if configuration in {"both_correct", "outcome_drift"} else wrong
    return (
        DRTMLE(
            randomized=True,
            cross_fit=False,
            outcome_learner=OracleOutcome(q),
            treatment_learner=OracleTreatment(correct),
            missingness_learner=OracleMissingness(pi),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2_000),
            estimands=(TARGET,),
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            guard=tuple(CONFIGURATION["guard"]),
            reduction=str(CONFIGURATION["reduction"]),
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=np.full(len(frame), RANDOMIZATION[0]),
        )
        .single()
    )


#: The calibration positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived(("ate",))

_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, _, _, n, _, seed, _ = key[0]
    return sample_discrete(PROBS, n, seed)


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    property_name, cell = key[0][:2]
    if property_name == "correction_necessity":
        return [
            f"{property_name}/five_reduction_cycle__closed_score",
            f"{property_name}/five_reduction_cycle__initial_score_control",
        ]
    name = f"{property_name}/{cell}"
    return [name, *DERIVED_CELLS.get(name, ())]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, _seed, configuration = key[0]
    result = _fit(frame, configuration)
    role = "control" if configuration == "both_wrong" else "positive"
    if property_name == "root_n_and_efficiency" and n == min(RATE_SIZES):
        role = "control"
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
    if property_name == "correction_necessity":
        reduction = result.repeats[0].fluctuations["mean"].reduction
        paired = []
        for score_cell, score_role, score in (
            ("five_reduction_cycle__closed_score", "positive", reduction.score),
            ("five_reduction_cycle__initial_score_control", "control", reduction.score_initial),
        ):
            paired.append(
                control_row(
                    property_name=property_name,
                    cell=score_cell,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=0.0,
                    estimate=float(np.max(np.abs(score))),
                    # No scale to report: the pair is one score read at two points of the
                    # cycle, and the verdict is a ratio of its own bias endpoints.
                    standard_error=1.0,
                    critical=1.0,
                    role=score_role,
                )
            )
        rows = paired
    return rows


def _payloads() -> list[_Key]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_drift", "observation_drift", "both_wrong"):
        specs.append(
            (
                "corrected_mar_inference",
                configuration,
                ROBUSTNESS_N,
                ROBUSTNESS_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct"))
    specs.extend(
        [
            (
                "interval_calibration",
                "ate__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            (
                "correction_necessity",
                "corrected",
                CORRECTION_N,
                CORRECTION_REPLICATES,
                "outcome_drift",
            ),
        ]
    )
    out: list[_Key] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            out.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return out


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
        labels=("ate",),
        efficiency_bounds={"ate": EFFICIENCY_SD},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        efficiency_bounds={"ate": EFFICIENCY_SD},
    )
    robustness_verdicts(summary, family="corrected_mar_inference")
    robustness = summary["property"] == "corrected_mar_inference"
    positive = robustness & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = (
        summary.loc[positive, "passed"]
        & (summary.loc[positive, "coverage_ci_lower"] >= STUDY.margins.coverage_floor)
        & summary.loc[positive, "se_ratio"].between(*STUDY.margins.se_ratio_sanity)
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)

    correction = summary["property"] == "correction_necessity"
    initial = summary.loc[
        correction & summary["cell"].str.endswith("__initial_score_control")
    ].iloc[0]
    for index in summary.index[correction]:
        cell = str(summary.loc[index, "cell"])
        if cell.endswith("__closed_score"):
            passed = float(summary.loc[index, "bias_ci_upper"]) <= (
                CORRECTION_SCORE_RATIO * float(initial["bias_ci_lower"])
            )
        else:
            passed = float(summary.loc[index, "bias_ci_lower"]) >= UNCORRECTED_SCORE_FLOOR
        summary.loc[index, "passed"] = bool(passed)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)
