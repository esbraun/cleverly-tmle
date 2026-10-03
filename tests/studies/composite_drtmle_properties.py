"""Independent repeated-sampling properties for the composite-indicator missing-data DR-TMLE.

Every fit uses law-table primaries and the GLM reductions of
:mod:`tests.studies.canonical_composite_drtmle`.  The families follow
:mod:`tests.studies.multi_arm_mar_drtmle_properties`, across the three scenarios of the
record, and add two of their own:

* ``ordinary_targeting``: the composite TMLE (``guard=()``) on the two missing-treatment
  scenarios, which R ``drtmle``'s ``out$tmle`` pairs with in the primary rows;
* ``treatment_complete_case``: a control that drops the rows with an unrecorded treatment and
  fits the rest.  The recording depends on the treatment, so this targets another parameter,
  and its bias must resolve away from zero.

Declared before any run:

* the simultaneous cells read the fits of their scenario at n = 2,000.  The binary
  missing-treatment scenario shares them with ``interval_calibration``; the other two
  scenarios fit a band of their own.  A band passes when its 99% joint-coverage interval
  lies inside ``[0.92, 0.98]``, and its pointwise control when the 99% upper endpoint of
  joint pointwise coverage is below 0.95.  Design, from the exact inference-scale covariance
  (``tests/unit/test_simultaneous_cell_design.py``): ``p0`` 0.8832, 0.8835 and 0.8220,
  oracle critical values 2.326, 2.324 and 2.508, control power 1.0 at 2,400 replications;
* the power cell runs at n = 1,000.  At the plan's n = 500 the planned power of the
  two-sided 5% test of ``ate`` is 0.738 (EIF SD 1.808, ``ate`` 0.21), below the 0.95 the
  plan requires; at n = 1,000 it is 0.957;
* the both-wrong floors and the drift tables are below, with the large-sample limits they
  were set from.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.canonical_composite_drtmle import (
    BINARY,
    LAW_KEYS,
    LAWS,
    OBSERVATIONAL,
    STUDY,
    THREE_ARM,
    fit_cleverly,
)
from tests.studies.evidence.properties import control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    robustness_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows

ROBUSTNESS_REPLICATES = 800
ROBUSTNESS_N = 2_000
ORDINARY_REPLICATES = 800
ORDINARY_N = 2_000
RATE_REPLICATES = 1_200
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
NULL_REPLICATES = 1_200
POWER_REPLICATES = 1_200
TEST_N = 2_000
POWER_N = 1_000
CORRECTION_REPLICATES = 1_200
CORRECTION_N = 2_000
COMPLETE_CASE_REPLICATES = 800
COMPLETE_CASE_N = 2_000
CORRECTION_SCORE_RATIO = 0.01
UNCORRECTED_SCORE_FLOOR = 1e-3
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
#: The ``ordinary_targeting`` positive rule beyond the shared bias rule: the 99% coverage
#: lower bound and the SE-ratio band of the plan.
ORDINARY_COVERAGE_FLOOR = 0.90
ORDINARY_SE_BAND = (0.80, 1.20)
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def _keyed(scenario: str) -> laws.Law:
    """The scenario's law under its law-module key, which the name helpers read."""
    return dataclasses.replace(LAWS[scenario], key=LAW_KEYS[scenario])


#: Each scenario's reported names and their truths.
NAMES = {scenario: laws.ESTIMANDS[LAW_KEYS[scenario]] for scenario in LAWS}
TRUTHS = {scenario: laws.truths(_keyed(scenario)) for scenario in LAWS}
TARGET = "ate"
THREE_ARM_TARGET = "ate[mid vs high]"
EFFICIENCY_SD = laws.efficiency_sd(_keyed(BINARY), TARGET)

#: The contrasts of ``corrected_mar_inference``, by cell prefix.
CONTRASTS: dict[str, tuple[str, str]] = {
    "composite_observational_ate": (OBSERVATIONAL, "ate"),
    "composite_binary_ate": (BINARY, "ate"),
    "composite_three_arm_ate_low": (THREE_ARM, "ate[low vs high]"),
    "composite_three_arm_ate_mid": (THREE_ARM, "ate[mid vs high]"),
}
CONFIGURATIONS = ("both_correct", "outcome_drift", "mechanism_drift", "both_wrong")
#: Every cell of the family: each contrast at each configuration, except the both-wrong control
#: of the three-arm ``low`` contrast, which :data:`BOTH_WRONG_BIAS_FLOOR` says cannot
#: discriminate.
CORRECTED_CELLS = tuple(
    f"{prefix}__{configuration}"
    for prefix in CONTRASTS
    for configuration in CONFIGURATIONS
    if (prefix, configuration) != ("composite_three_arm_ate_low", "both_wrong")
)

#: The drifts.  The outcome drift is the constant regression ``Q = 0.5``, the coarsened drift of
#: :mod:`tests.studies.multi_arm_mar_drtmle_properties`.  The mechanism drift is a wrong
#: ``P(A = a | Delta_A = 1, W)`` and, where the treatment can be missing, a wrong
#: ``P(Delta_A = 1 | W)``.  Every drift cell uses these tables, so the both-wrong control is the
#: composition of the two one-wrong cells.
#:
#: Declared before the run, and not ``Q -> 1 - Q``.  At the three-arm law that drift left the
#: both-wrong limit of ``ate[mid vs high]`` near 0.004 on a frame of 40,000 rows, inside the
#: shared discrimination margin.  A search over three outcome drifts (``1 - Q``, ``0.5``, a
#: column roll) and three treatment-factor drifts (``WRONG_G_THREE``, uniform, a column roll),
#: with and without a wrong recording vector, found no pair that moves both three-arm contrasts
#: by 0.02.  The constant regression moves ``ate[mid vs high]`` most.
WRONG_MU = 0.5
WRONG_RECORDED = np.array([0.70, 0.95, 0.60])
WRONG_G = {
    OBSERVATIONAL: laws._two_arm(laws.WRONG_G_TWO),
    BINARY: laws._two_arm(laws.WRONG_G_TWO),
    THREE_ARM: laws.WRONG_G_THREE,
}

#: The both-wrong control must resolve its bias away from zero by at least this much: the
#: lower end of the 99% interval of ``|bias|`` must clear it.  About half the large-sample
#: limit of the both-wrong bias, measured before the run as the difference from the
#: both-correct fit on one frame of 100,000 rows, at two seeds:
#:
#: ================================  ===============  ===============  =====
#: contrast                          seed 7           seed 8           floor
#: ================================  ===============  ===============  =====
#: ``composite_observational_ate``   -0.069           -0.082           0.035
#: ``composite_binary_ate``          -0.051           -0.052           0.025
#: ``composite_three_arm_ate_mid``   -0.093           -0.079           0.040
#: ``composite_three_arm_ate_low``   +0.002           +0.010           none
#: ================================  ===============  ===============  =====
#:
#: The one-wrong cells moved by at most 0.002 at both seeds.  The three-arm ``low`` contrast
#: has no both-wrong control: its limit is inside the shared discrimination margin (about
#: 0.012 at n = 2,000), so the cell could not discriminate.  Its positive cells stay.
BOTH_WRONG_BIAS_FLOOR: dict[str, float] = {
    "composite_observational_ate": 0.035,
    "composite_binary_ate": 0.025,
    "composite_three_arm_ate_mid": 0.040,
}

#: The sharp null for ``ate``: the arm-1 column of the outcome table is the arm-0 column.
_NULL_MU = LAWS[BINARY].mu.copy()
_NULL_MU[:, 1] = LAWS[BINARY].mu[:, 0]
NULL_LAW = dataclasses.replace(
    LAWS[BINARY], mu=_NULL_MU, scenario="binary_mar_outcome_and_treatment_sharp_null", key="l1"
)
NULL_TRUTHS = laws.truths(NULL_LAW)

#: The complete-case control's expected bias: the plug-in of the complete-case law,
#: ``sum_w P(w | Delta_A = 1) E(Y | A = a, W = w)``, minus the truth.
_RECORDED_W = LAWS[BINARY].p_w * LAWS[BINARY].recorded
_COMPLETE_CASE_MEANS = (_RECORDED_W / _RECORDED_W.sum()) @ LAWS[BINARY].mu
COMPLETE_CASE_BIAS = float(
    (_COMPLETE_CASE_MEANS[1] - _COMPLETE_CASE_MEANS[0]) - TRUTHS[BINARY][TARGET]
)


def _tables(scenario: str, configuration: str) -> dict[str, Any]:
    law = LAWS[scenario]
    tables: dict[str, Any] = {}
    if configuration in {"outcome_drift", "both_wrong"}:
        tables["mu"] = np.full_like(law.mu, WRONG_MU)
    if configuration in {"mechanism_drift", "both_wrong"}:
        tables["g"] = WRONG_G[scenario]
        if law.pi_treatment is not None:
            tables["recorded"] = WRONG_RECORDED
    return tables


def _row(
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: Any,
) -> dict[str, Any]:
    return replicate_row(
        property_name=property_name,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def _necessity_rows(
    result: Any, property_name: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    """The largest absolute extra-score column, closed and initial, over every arm."""
    reduction = result.repeats[0].fluctuations["mean"].reduction
    mechanism = result.repeats[0].fluctuations["mean"].mechanism
    closed = np.concatenate([np.abs(reduction.score), np.abs(mechanism.score)])
    initial = np.abs(reduction.score_initial)
    rows = []
    for cell, role, score in (
        ("composite_cycle__closed_score", "positive", closed),
        ("composite_cycle__initial_score_control", "control", initial),
    ):
        rows.append(
            control_row(
                property_name=property_name,
                cell=cell,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=0.0,
                estimate=float(np.max(score)),
                standard_error=1.0,
                critical=1.0,
                role=role,
            )
        )
    return rows


#: Each ``simultaneous_coverage`` label's scenario.
BAND_LABELS = {
    "composite_observational": OBSERVATIONAL,
    "composite_binary": BINARY,
    "composite_three_arm": THREE_ARM,
}


def _band_rows(
    result: Any, scenario: str, label: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    return joint_coverage_rows(
        result,
        TRUTHS[scenario],
        NAMES[scenario],
        label=label,
        replicate=replicate,
        n=n,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


Payload = tuple[str, str, int, int, int, int, str]


def _fit_replication(payload: Payload) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    if property_name == "type_i_error":
        frame = laws.sample(NULL_LAW, n, seed)
        result = fit_cleverly(frame, NULL_LAW, request=("ate",), simultaneous=False)
        return [
            _row(
                property_name,
                cell,
                "positive",
                replicate,
                n,
                requested,
                NULL_TRUTHS[TARGET],
                result[TARGET],
            )
        ]
    if property_name == "simultaneous_coverage":
        scenario = BAND_LABELS[cell]
        frame = laws.sample(LAWS[scenario], n, seed)
        result = fit_cleverly(frame, LAWS[scenario], simultaneous=True)
        return _band_rows(result, scenario, cell, replicate, n, requested)
    if property_name == "corrected_mar_inference":
        # One fit per scenario and configuration; a scenario reports each of its contrasts.
        scenario = cell
        frame = laws.sample(LAWS[scenario], n, seed)
        result = fit_cleverly(
            frame, LAWS[scenario], simultaneous=False, **_tables(scenario, configuration)
        )
        role = "control" if configuration == "both_wrong" else "positive"
        return [
            _row(
                property_name,
                f"{prefix}__{configuration}",
                role,
                replicate,
                n,
                requested,
                TRUTHS[scenario][name],
                result[name],
            )
            for prefix, (owner, name) in CONTRASTS.items()
            if owner == scenario and f"{prefix}__{configuration}" in CORRECTED_CELLS
        ]
    if property_name == "ordinary_targeting":
        scenario = BINARY if cell == "binary" else THREE_ARM
        name = TARGET if cell == "binary" else THREE_ARM_TARGET
        frame = laws.sample(LAWS[scenario], n, seed)
        result = fit_cleverly(frame, LAWS[scenario], guard=(), request=("ate",), simultaneous=False)
        return [
            _row(
                property_name,
                cell,
                "positive",
                replicate,
                n,
                requested,
                TRUTHS[scenario][name],
                result[name],
            )
        ]
    if property_name == "treatment_complete_case":
        law = LAWS[BINARY]
        frame = laws.sample(law, n, seed)
        recorded = frame.loc[frame["DeltaA"] == 1.0].drop(columns="DeltaA")
        complete_law = dataclasses.replace(law, pi_treatment=None)
        result = fit_cleverly(
            recorded.reset_index(drop=True), complete_law, request=("ate",), simultaneous=False
        )
        return [
            _row(
                property_name,
                cell,
                "control",
                replicate,
                n,
                requested,
                TRUTHS[BINARY][TARGET],
                result[TARGET],
            )
        ]
    law = LAWS[BINARY]
    frame = laws.sample(law, n, seed)
    if property_name == "interval_calibration":
        result = fit_cleverly(frame, law, simultaneous=True)
        return [
            _row(
                property_name,
                cell,
                "positive",
                replicate,
                n,
                requested,
                TRUTHS[BINARY][TARGET],
                result[TARGET],
            ),
            *_band_rows(result, BINARY, "composite_binary", replicate, n, requested),
        ]
    result = fit_cleverly(
        frame, law, request=("ate",), simultaneous=False, **_tables(BINARY, configuration)
    )
    if property_name == "correction_necessity":
        return _necessity_rows(result, property_name, replicate, n, requested)
    role = "positive"
    if property_name == "root_n_and_efficiency" and n == min(RATE_SIZES):
        role = "control"
    return [
        _row(
            property_name,
            cell,
            role,
            replicate,
            n,
            requested,
            TRUTHS[BINARY][TARGET],
            result[TARGET],
        )
    ]


def _specs() -> list[tuple[str, str, int, int, str]]:
    specs: list[tuple[str, str, int, int, str]] = [
        ("corrected_mar_inference", scenario, ROBUSTNESS_N, ROBUSTNESS_REPLICATES, configuration)
        for scenario in LAWS
        for configuration in CONFIGURATIONS
    ]
    specs += [
        ("ordinary_targeting", cell, ORDINARY_N, ORDINARY_REPLICATES, "both_correct")
        for cell in ("binary", "three_arm")
    ]
    specs += [
        ("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct")
        for size in RATE_SIZES
    ]
    specs += [
        (
            "interval_calibration",
            "ate__correctly_specified",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "both_correct",
        ),
        # The binary scenario's band reads the calibration fits; these two fit their own.
        (
            "simultaneous_coverage",
            "composite_observational",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "both_correct",
        ),
        (
            "simultaneous_coverage",
            "composite_three_arm",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "both_correct",
        ),
        ("type_i_error", "sharp_null", TEST_N, NULL_REPLICATES, "both_correct"),
        ("power", "ate", POWER_N, POWER_REPLICATES, "both_correct"),
        (
            "correction_necessity",
            "composite_cycle",
            CORRECTION_N,
            CORRECTION_REPLICATES,
            "outcome_drift",
        ),
        (
            "treatment_complete_case",
            "drop_unrecorded__control",
            COMPLETE_CASE_N,
            COMPLETE_CASE_REPLICATES,
            "both_correct",
        ),
    ]
    return specs


def _payloads(budget: int | None = None) -> list[tuple[Payload]]:
    """Every replication's payload.  ``budget`` caps each cell, for a smoke run only."""
    out: list[tuple[Payload]] = []
    for property_name, cell, n, replicates, configuration in _specs():
        count = replicates if budget is None else min(budget, replicates)
        stream = cell if property_name != "corrected_mar_inference" else f"{cell}__{configuration}"
        for replicate in range(count):
            seed = stream_seed(STUDY, "property_sample", property_name, stream, replicate)
            out.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(budget), n_jobs=n_jobs)
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
    summary, rates = apply_shared_verdicts(rows, STUDY, efficiency_bounds={"ate": EFFICIENCY_SD})
    robustness_verdicts(summary, family="corrected_mar_inference")
    robustness = summary["property"] == "corrected_mar_inference"
    positive = robustness & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = (
        summary.loc[positive, "passed"]
        & (summary.loc[positive, "coverage_ci_lower"] >= STUDY.margins.coverage_floor)
        & summary.loc[positive, "se_ratio"].between(*STUDY.margins.se_ratio_sanity)
    )
    control = robustness & (summary["role"] == "control")
    for index in summary.index[control]:
        prefix = str(summary.loc[index, "cell"]).split("__", 1)[0]
        low = float(summary.loc[index, "bias_ci_lower"])
        high = float(summary.loc[index, "bias_ci_upper"])
        # The interval of |bias| clears the floor only when the bias interval lies wholly on
        # one side of zero, at least the floor away from it.
        cleared = max(low, -high) >= BOTH_WRONG_BIAS_FLOOR[prefix]
        summary.loc[index, "passed"] = bool(summary.loc[index, "passed"]) and cleared

    ordinary = summary["property"] == "ordinary_targeting"
    summary.loc[ordinary, "passed"] = (
        summary.loc[ordinary, "bias_equivalent"]
        & (summary.loc[ordinary, "coverage_ci_lower"] >= ORDINARY_COVERAGE_FLOOR)
        & summary.loc[ordinary, "se_ratio"].between(*ORDINARY_SE_BAND)
    )
    complete_case = summary["property"] == "treatment_complete_case"
    summary.loc[complete_case, "passed"] = summary.loc[complete_case, "bias_discriminated"]

    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)

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
