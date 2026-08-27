"""Independent repeated-sampling properties for incremental odds interventions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests import incrementals
from tests.conftest import OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_incremental_interventions import STUDY
from tests.studies.evidence.properties import (
    control_row,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.intervention_study_helpers import (
    INTERVENTION_CALIBRATION_REPLICATES,
    efficiency_sd,
    incremental_estimates,
    probabilities,
    sample_discrete,
)
from tests.studies.regime_property_helpers import NULL_PROBS, WRONG_PROBS

MECHANISM_REPLICATES = 1_200
MECHANISM_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = INTERVENTION_CALIBRATION_REPLICATES
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SCORE_REPLICATES = 1_600
SCORE_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
NECESSITY_DISPLACEMENT = 0.25
TARGETING_DISPLACEMENT = NECESSITY_DISPLACEMENT
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGET = "ate_ipsi[odds x2 vs natural course]"
SCORE_TARGET = "ey_ipsi[odds x2]"
TRUTH = float(law.functional(law.PROBS, TARGET))
NULL_TRUTH = float(law.functional(NULL_PROBS, TARGET))
EFFICIENCY_SD = efficiency_sd(law.PROBS, TARGET)
SCORE_Q = np.array([[0.05, 0.95], [0.10, 0.90], [0.20, 0.80]])
SCORE_PROBS = probabilities(SCORE_Q)
SCORE_TRUTH = float(law.functional(SCORE_PROBS, SCORE_TARGET))
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def fit(frame: pd.DataFrame, probs: np.ndarray, configuration: str) -> Any:
    """Fit the declared curve under a correct or deliberately wrong nuisance pair."""
    q_correct = configuration in {"both_correct", "mechanism_wrong"}
    g_correct = configuration in {"both_correct", "outcome_wrong"}
    outcome_law = law.DiscreteLaw(probs if q_correct else WRONG_PROBS)
    mechanism_law = law.DiscreteLaw(probs if g_correct else WRONG_PROBS)
    return (
        TMLE(
            incremental=incrementals.interventions(),
            outcome_learner=OracleOutcome(outcome_law),
            treatment_learner=OracleTreatment(mechanism_law),
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def _score_control_standard_error(result: Any) -> float:
    """SE after deleting only the treatment-mechanism derivative from the EIF."""
    fluctuation = result.fluctuations["ipsi"]
    if fluctuation.mechanism is None:  # pragma: no cover - a study contract guard
        raise AssertionError("an incremental fit did not retain mechanism targeting")
    tilts = result.nuisance.incremental.at(fluctuation.mechanism.propensity)
    code = float(tilts.names.index("odds x2"))
    blip = fluctuation.targeted.arms[1.0] - fluctuation.targeted.arms[0.0]
    treatment = np.asarray(result.data.treatment, dtype=float)
    extra = tilts.derivative[:, int(code)] * blip * (treatment - fluctuation.mechanism.propensity)
    control_curve = np.asarray(result[SCORE_TARGET].influence_curve, dtype=float) - extra
    return float(np.std(control_curve, ddof=1) / np.sqrt(result.data.n))


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    if property_name == "type_i_error":
        probs = NULL_PROBS
    elif property_name == "treatment_score_necessity":
        probs = SCORE_PROBS
    else:
        probs = law.PROBS
    frame = sample_discrete(probs, n, seed)
    result = fit(frame, probs, configuration)
    target = SCORE_TARGET if property_name == "treatment_score_necessity" else TARGET
    truth = float(law.functional(probs, target))
    role = (
        "control"
        if cell == "mechanism_wrong"
        or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
        else "positive"
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
            estimate=result[target],
            alpha=STUDY.margins.alpha,
        )
    ]
    if property_name == "mechanism_requirement" and cell == "mechanism_wrong":
        frozen = incremental_estimates(
            result,
            targeted_outcome=True,
            targeted_mechanism=False,
        )[TARGET]
        rows[0] = control_row(
            property_name=property_name,
            cell=cell,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth,
            estimate=frozen,
            standard_error=float(result[TARGET].std_error),
            critical=CRITICAL,
        )
    if property_name == "targeting_necessity":
        arm = "outcome" if cell == "outcome_targeted" else "mechanism"
        rows[0]["cell"] = f"{arm}__targeted"
        untargeted = incremental_estimates(
            result,
            targeted_outcome=arm == "mechanism",
            targeted_mechanism=arm == "outcome",
        )[TARGET]
        rows.append(
            control_row(
                property_name=property_name,
                cell=f"{arm}__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=untargeted,
                standard_error=float(result[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "treatment_score_necessity":
        rows[0]["cell"] = "odds_x2__full_eif"
        rows.append(
            control_row(
                property_name=property_name,
                cell="odds_x2__regime_curve_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(result[SCORE_TARGET].psi),
                standard_error=_score_control_standard_error(result),
                critical=CRITICAL,
            )
        )
    if property_name == "natural_course_identity":
        estimate = result["ey_ipsi[natural course]"]
        sample_mean = float(np.mean(np.asarray(frame["Y"], dtype=float)))
        rows = [
            replicate_row(
                property_name=property_name,
                cell="natural__ipsi",
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=sample_mean,
                estimate=estimate,
                alpha=STUDY.margins.alpha,
            ),
            control_row(
                property_name=property_name,
                cell="natural__mean",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=sample_mean,
                estimate=sample_mean,
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
            ),
        ]
        rows[1]["role"] = "positive"
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str]] = [
        (
            "mechanism_requirement",
            "both_correct",
            MECHANISM_N,
            MECHANISM_REPLICATES,
            "both_correct",
        ),
        (
            "mechanism_requirement",
            "outcome_wrong",
            MECHANISM_N,
            MECHANISM_REPLICATES,
            "outcome_wrong",
        ),
        (
            "mechanism_requirement",
            "mechanism_wrong",
            MECHANISM_N,
            MECHANISM_REPLICATES,
            "mechanism_wrong",
        ),
    ]
    for size in RATE_SIZES:
        specs.append(
            (
                "root_n_and_efficiency",
                f"contrast__n_{size}",
                size,
                RATE_REPLICATES,
                "both_correct",
            )
        )
    specs.extend(
        [
            (
                "interval_calibration",
                "contrast__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "both_correct"),
            ("power", "alternative", NULL_N, NULL_REPLICATES, "both_correct"),
            (
                "targeting_necessity",
                "outcome_targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "outcome_wrong",
            ),
            (
                "targeting_necessity",
                "mechanism_targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "both_wrong",
            ),
            (
                "treatment_score_necessity",
                "full_eif",
                SCORE_N,
                SCORE_REPLICATES,
                "both_correct",
            ),
            (
                "natural_course_identity",
                "identity",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "both_correct",
            ),
        ]
    )
    out: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            out.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    controls = calibration_controls(
        rows,
        STUDY,
        labels=("contrast",),
        efficiency_bounds={"contrast": EFFICIENCY_SD},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        rate_labels=("contrast",),
        efficiency_bounds={"contrast": EFFICIENCY_SD},
        extra_columns=("targeting_displacement", "maximum_identity_difference"),
    )
    calibration_verdicts(
        summary,
        margins=STUDY.margins,
        efficiency_band=EFFICIENCY_RATIO_BAND,
    )

    mechanism = summary["property"] == "mechanism_requirement"
    positive = mechanism & (summary["role"] == "positive")
    control = mechanism & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]
    summary.loc[mechanism, "property_passed"] = bool(summary.loc[mechanism, "passed"].all())

    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("outcome", "mechanism"),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )

    score = summary["property"] == "treatment_score_necessity"
    score_passes: list[bool] = []
    for index in summary.index[score]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == "treatment_score_necessity") & (rows["cell"] == cell)]
        interval = ratio_intervals(
            group,
            replicates=STUDY.margins.bootstrap_replicates,
            confidence_level=STUDY.margins.confidence_level,
            seed=stream_seed(STUDY, "treatment_score_necessity", cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = interval.low
        summary.loc[index, "se_ratio_ci_upper"] = interval.high
        if cell.endswith("__full_eif"):
            passed = interval.within(*EFFICIENCY_RATIO_BAND)
        else:
            passed = interval.high < STUDY.margins.calibration_se_ratio[0]
        summary.loc[index, "passed"] = passed
        score_passes.append(bool(passed))
    summary.loc[score, "property_passed"] = bool(all(score_passes))

    identity = summary["property"] == "natural_course_identity"
    source = rows.loc[rows["property"] == "natural_course_identity"]
    wide = source.pivot(index="replicate", columns="cell", values="estimate")
    difference = float(np.max(np.abs(wide["natural__ipsi"] - wide["natural__mean"])))
    summary.loc[identity, "maximum_identity_difference"] = difference
    summary.loc[identity, "passed"] = difference < 1e-8
    summary.loc[identity, "property_passed"] = bool(difference < 1e-8)
    return finish(summary, rates)
