"""Independent repeated-sampling properties for continuous shift policies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from cleverly.datasets import ShiftDGP
from cleverly.estimators import TMLE
from cleverly.learners.density import bin_edges
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_shift_policies import (
    DENSITY_BINS,
    ORACLE_DENSITY_BINS,
    POLICIES,
    STUDY,
    OracleShiftDensity,
    OracleShiftOutcome,
    fit_shift_estimator,
    initial_estimates,
    reversed_ratio_control,
    shifts,
)
from tests.studies.evidence.properties import control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.intervention_study_helpers import INTERVENTION_CALIBRATION_REPLICATES

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = INTERVENTION_CALIBRATION_REPLICATES
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
NECESSITY_DISPLACEMENT = 0.25
TARGETING_DISPLACEMENT = NECESSITY_DISPLACEMENT
IDENTITY_TOLERANCE = 1e-8
PROPERTY_CURVATURE = 0.15
TARGET = "ate_shift[+0.5 capped vs natural course]"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def policy_dgp(*, effect: float = 0.5, curvature: float = 0.25) -> ShiftDGP:
    """A conditional-normal dose law with a controllable shift effect."""

    def dose_mean(w: np.ndarray) -> np.ndarray:
        return 2.0 + 0.7 * w[:, 0] - 0.3 * w[:, 1]

    def outcome_mean(w: np.ndarray, a: np.ndarray) -> np.ndarray:
        return 1.0 + effect * a + curvature * a**2 + w[:, 0] - 0.5 * w[:, 1] + 0.2 * w[:, 2]

    return ShiftDGP(
        name=f"shift_property_{effect:g}_{curvature:g}",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        dose_mean=dose_mean,
        outcome_mean=outcome_mean,
        dose_scale=1.0,
        noise_scale=0.8,
    )


def fit(
    frame: pd.DataFrame,
    dgp: ShiftDGP,
    configuration: str,
    *,
    capped: bool = True,
) -> Any:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    density_correct = configuration in {"both_correct", "density_correct"}
    outcome = OracleShiftOutcome(dgp) if q_correct else LinearRegression()
    density_bins = ORACLE_DENSITY_BINS if density_correct and not q_correct else DENSITY_BINS
    edges = tuple(float(value) for value in bin_edges(np.asarray(frame["A"]), density_bins))
    density_dgp = dgp
    if not density_correct:
        density_dgp = replace(
            dgp,
            dose_mean=lambda w: 1.7 + 0.3 * w[:, 0],
        )
    density = OracleShiftDensity(density_dgp, edges)
    estimator = TMLE(
        shifts=shifts(capped=capped),
        outcome_learner=outcome,
        treatment_learner=density,
        cross_fit=False,
        simultaneous=False,
        density_bins=density_bins,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )
    return fit_shift_estimator(estimator, frame, dgp)


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    dgp = (
        policy_dgp(effect=0.0, curvature=0.0)
        if property_name == "type_i_error"
        else policy_dgp(curvature=PROPERTY_CURVATURE)
    )
    frame, truth_map = dgp.sample(n, shifts=POLICIES, seed=seed, backend="pandas")
    truth = float(truth_map[TARGET])
    result = fit(frame, dgp, configuration)
    role = (
        "control"
        if cell == "both_wrong"
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
            estimate=result[TARGET],
            alpha=STUDY.margins.alpha,
        )
    ]
    if property_name == "targeting_necessity":
        rows[0]["cell"] = "shift__targeted"
        rows.append(
            control_row(
                property_name=property_name,
                cell="shift__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=initial_estimates(result)[TARGET],
                standard_error=float(result[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "ratio_necessity":
        rows[0]["cell"] = "shift__declared"
        wrong = reversed_ratio_control(result)
        rows.append(
            control_row(
                property_name=property_name,
                cell="shift__reversed_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(wrong[TARGET].psi),
                standard_error=float(wrong[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "cap_necessity":
        rows[0]["cell"] = "shift__declared_cap"
        wrong = fit(frame, dgp, configuration, capped=False)
        rows.append(
            control_row(
                property_name=property_name,
                cell="shift__uncapped_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(wrong[TARGET].psi),
                standard_error=float(wrong[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "natural_course_identity":
        estimate = result["ey_shift[natural course]"]
        natural_truth = float(np.mean(np.asarray(frame["Y"], dtype=float)))
        rows = [
            replicate_row(
                property_name=property_name,
                cell="natural__shift",
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=natural_truth,
                estimate=estimate,
                alpha=STUDY.margins.alpha,
            ),
            control_row(
                property_name=property_name,
                cell="natural__mean",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=natural_truth,
                estimate=natural_truth,
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
            ),
        ]
        rows[1]["role"] = "positive"
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_correct", "density_correct", "both_wrong"):
        specs.append(
            (
                "double_robustness",
                configuration,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct"))
    specs.extend(
        [
            (
                "interval_calibration",
                "correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "both_correct"),
            ("power", "alternative", NULL_N, NULL_REPLICATES, "both_correct"),
            (
                "targeting_necessity",
                "targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "density_correct",
            ),
            ("ratio_necessity", "declared", NECESSITY_N, NECESSITY_REPLICATES, "density_correct"),
            ("cap_necessity", "declared_cap", NECESSITY_N, NECESSITY_REPLICATES, "both_correct"),
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
    source = rows.loc[
        (rows["property"] == "interval_calibration") & (rows["cell"] == "correctly_specified")
    ]
    shrunken = source.copy()
    shrunken["cell"] = "shrunken_se_control"
    shrunken["role"] = "control"
    shrunken["std_error"] *= SHRUNKEN_SE_FACTOR
    half = CRITICAL * shrunken["std_error"]
    shrunken["covered"] = (
        (shrunken["estimate"] - half <= shrunken["truth"])
        & (shrunken["truth"] <= shrunken["estimate"] + half)
    ).astype(int)
    shrunken["rejected"] = (np.abs(shrunken["estimate"] / shrunken["std_error"]) > CRITICAL).astype(
        int
    )
    return pd.concat([rows, shrunken], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "necessity_displacement",
            "maximum_identity_difference",
        ),
    )
    calibration = summary["property"] == "interval_calibration"
    positive = calibration & (summary["role"] == "positive")
    control = calibration & (summary["role"] == "control")
    summary.loc[positive, "passed"] = (
        (summary.loc[positive, "se_ratio_ci_lower"] >= STUDY.margins.calibration_se_ratio[0])
        & (summary.loc[positive, "se_ratio_ci_upper"] <= STUDY.margins.calibration_se_ratio[1])
        & (summary.loc[positive, "coverage_ci_lower"] >= STUDY.margins.calibration_coverage[0])
        & (summary.loc[positive, "coverage_ci_upper"] <= STUDY.margins.calibration_coverage[1])
    )
    summary.loc[control, "passed"] = (
        summary.loc[control, "se_ratio_ci_upper"] < STUDY.margins.calibration_se_ratio[0]
    )
    for family, arms, column, threshold in (
        (
            "targeting_necessity",
            ("targeted", "untargeted"),
            "targeting_displacement",
            TARGETING_DISPLACEMENT,
        ),
        (
            "ratio_necessity",
            ("declared", "reversed_control"),
            "necessity_displacement",
            NECESSITY_DISPLACEMENT,
        ),
        (
            "cap_necessity",
            ("declared_cap", "uncapped_control"),
            "necessity_displacement",
            NECESSITY_DISPLACEMENT,
        ),
    ):
        necessity_verdicts(
            summary,
            rows,
            family=family,
            labels=("shift",),
            arms=arms,
            column=column,
            threshold=threshold,
        )
    identity = summary["property"] == "natural_course_identity"
    source = rows.loc[rows["property"] == "natural_course_identity"]
    wide = source.pivot(index="replicate", columns="cell", values="estimate")
    difference = float(np.max(np.abs(wide["natural__shift"] - wide["natural__mean"])))
    summary.loc[identity, "maximum_identity_difference"] = difference
    summary.loc[identity, "passed"] = difference < IDENTITY_TOLERANCE
    summary.loc[identity, "property_passed"] = bool(difference < IDENTITY_TOLERANCE)
    return finish(summary, rates)
