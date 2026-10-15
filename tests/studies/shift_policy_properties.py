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
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
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
TARGET = "ate_policy[+0.5 capped vs natural course]"
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
        policies=shifts(capped=capped),
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


def _law(property_name: str) -> ShiftDGP:
    return (
        policy_dgp(effect=0.0, curvature=0.0)
        if property_name == "type_i_error"
        else policy_dgp(curvature=PROPERTY_CURVATURE)
    )


def _draw_replication(payload: tuple[str, str, int, int, int, int, str]) -> Any:
    property_name, _, _, n, _, seed, _ = payload
    return _law(property_name).sample(n, policies=POLICIES, seed=seed, backend="pandas")


def _fit_replication(
    payload: tuple[str, str, int, int, int, int, str], drawn: Any
) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, _, configuration = payload
    dgp = _law(property_name)
    frame, truth_map = drawn
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
        estimate = result["ey_policy[natural course]"]
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


#: The calibration positive arm to the one control the generator derives from it.
DERIVED_CELLS = {
    "interval_calibration/correctly_specified": ("interval_calibration/shrunken_se_control",)
}

_PAIRS = {
    "targeting_necessity": ("shift__targeted", "shift__untargeted"),
    "ratio_necessity": ("shift__declared", "shift__reversed_control"),
    "cap_necessity": ("shift__declared_cap", "shift__uncapped_control"),
    "natural_course_identity": ("natural__shift", "natural__mean"),
}


def _cells_of(key: tuple[tuple[str, str, int, int, int, int, str]]) -> list[str]:
    """The ``property/cell`` names the key's rows and derived control fill."""
    property_name, cell = key[0][:2]
    if property_name in _PAIRS:
        return [f"{property_name}/{name}" for name in _PAIRS[property_name]]
    name = f"{property_name}/{cell}"
    return [name, *DERIVED_CELLS.get(name, ())]


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


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = fit_properties(
        _payloads(),
        splat_drawn(_fit_replication),
        draw=splat_draw(_draw_replication),
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        derived=DERIVED_CELLS,
    )
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


#: The three necessity declarations, read by the verdicts and the rules.
NECESSITIES = (
    property_verdicts.Necessity(
        family="targeting_necessity",
        labels=("shift",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    ),
    property_verdicts.Necessity(
        family="ratio_necessity",
        labels=("shift",),
        arms=("declared", "reversed_control"),
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    ),
    property_verdicts.Necessity(
        family="cap_necessity",
        labels=("shift",),
        arms=("declared_cap", "uncapped_control"),
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    ),
)

#: The cells the ``natural_course_identity`` supremum compares.
IDENTITY_CELLS = ("natural__shift", "natural__mean")


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
    for necessity in NECESSITIES:
        necessity_verdicts(summary, rows, necessity)
    identity = summary["property"] == "natural_course_identity"
    source = rows.loc[rows["property"] == "natural_course_identity"]
    wide = source.pivot(index="replicate", columns="cell", values="estimate")
    shifted, natural = IDENTITY_CELLS
    difference = float(np.max(np.abs(wide[shifted] - wide[natural])))
    summary.loc[identity, "maximum_identity_difference"] = difference
    summary.loc[identity, "passed"] = difference < IDENTITY_TOLERANCE
    summary.loc[identity, "property_passed"] = bool(difference < IDENTITY_TOLERANCE)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration verdict is
#: this module's own code, declared with the shared rule's clauses and no efficiency band.  The
#: natural-course identity is a sup-norm with a fixed tolerance and is not accountable (R3-1).
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(*NECESSITIES),
    property_verdicts.identity_rule(
        "natural_course_identity",
        column="maximum_identity_difference",
        cells=IDENTITY_CELLS,
        tolerance=IDENTITY_TOLERANCE,
        shared="shift_policy_properties.summarize_properties",
    ),
)
