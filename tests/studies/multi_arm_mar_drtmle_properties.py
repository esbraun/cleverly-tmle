"""Independent repeated-sampling properties for randomized multi-arm MAR DR-TMLE.

Every fit uses law-table primaries and the binary study's GLM reductions. The law is L3 of
:mod:`tests.studies.mar_arm_indexed_laws`. Each family mirrors
:mod:`tests.studies.mar_drtmle_properties` at three arms, and adds a size and a power cell for
the contrast ``ate[mid vs high]``.

The simultaneous-band cells read the shared joint-coverage helper of
``tests/studies/evidence/simultaneous.py``. They are added when that module is on the branch.
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
from tests.studies.canonical_multi_arm_mar_drtmle import LAW, STUDY, fit_cleverly
from tests.studies.evidence.properties import control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    robustness_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

ROBUSTNESS_REPLICATES = 800
ROBUSTNESS_N = 2_000
RATE_REPLICATES = 1_200
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
NULL_REPLICATES = 1_200
POWER_REPLICATES = 1_200
TEST_N = 2_000
CORRECTION_REPLICATES = 1_200
CORRECTION_N = 2_000
CORRECTION_SCORE_RATIO = 0.01
UNCORRECTED_SCORE_FLOOR = 1e-3
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
#: The both-wrong control must resolve its bias away from zero by at least this much: the
#: lower end of the 99% interval of ``|bias|`` must clear it. Declared before the run, at half
#: the both-wrong limit measured on the rounded L3 realization (+0.0166 and -0.0260; see
#: :data:`WRONG`), so the control's power is stated rather than inferred.
#:
#: Which gate binds. The control passes only when the shared ``bias_discriminated`` rule (the
#: bias interval lies outside 0.25 empirical SD, about 0.0088 at n = 2,000) and this floor
#: both hold. For ``l3_ate_low`` the shared margin is the larger, so it is the binding gate
#: and the floor adds nothing. For ``l3_ate_mid`` the floor (0.013) binds. Measured at
#: n = 2,000 on 80 off-stream replications (the F4 implementation review, seeds from 7.7e6):
#: bias +0.0193 and -0.0212, empirical SD 0.035, so at 800 replications the interval lower
#: ends sit near 0.016 and 0.018, above both gates. The floors are kept as declared.
BOTH_WRONG_BIAS_FLOOR: dict[str, float] = {"l3_ate_low": 0.008, "l3_ate_mid": 0.013}
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The two contrasts the corrected-inference family reports, by cell prefix.
CONTRASTS = {laws.label(LAW, name): name for name in ("ate[low vs high]", "ate[mid vs high]")}
TARGET = "ate[low vs high]"
TEST_TARGET = "ate[mid vs high]"
CONFIGURATIONS = ("both_correct", "outcome_drift", "observation_drift", "both_wrong")
#: The coarsened drifts, as in :mod:`tests.studies.mar_drtmle_properties`: a constant outcome
#: regression and a constant observation mechanism. Every drift cell uses these two tables, so
#: the both-wrong control is the composition of the two one-wrong cells.
#:
#: Declared before the run, and not ``mar_arm_indexed_laws.wrong_tables``. The F4 plan named
#: those tables. Their both-wrong limit, measured on a rounded realization of L3 at N = 100,000,
#: is +0.0026 and -0.0049 for the two contrasts. That is inside the 99% half-width of about
#: 0.0032 at 800 replications and well inside the standardized-bias margin, so with them the
#: control could not discriminate. The coarsened tables give +0.0166 and -0.0260 on the same
#: realization, and each one-wrong cell stays at 0.0000 to 0.0001.
WRONG = {"mu": np.full_like(LAW.mu, 0.5), "pi": np.full_like(LAW.pi, 0.6)}
TRUTHS = laws.truths(LAW)
EFFICIENCY_SD = laws.efficiency_sd(LAW, TARGET)

#: The sharp null for ``ate[mid vs high]``: the ``mid`` column of the outcome table is the
#: ``high`` column, so the contrast is zero and ``rr[mid vs high]`` is one.
_NULL_MU = LAW.mu.copy()
_NULL_MU[:, LAW.labels.index("mid")] = LAW.mu[:, LAW.labels.index("high")]
NULL_LAW = dataclasses.replace(LAW, mu=_NULL_MU, scenario="three_arm_mar_sharp_null")
NULL_TRUTHS = laws.truths(NULL_LAW)


def _tables(configuration: str) -> dict[str, np.ndarray | None]:
    return {
        "mu": WRONG["mu"] if configuration in {"outcome_drift", "both_wrong"} else None,
        "pi": WRONG["pi"] if configuration in {"observation_drift", "both_wrong"} else None,
    }


def _fit(frame: pd.DataFrame, configuration: str, *, law: laws.Law = LAW) -> Any:
    return fit_cleverly(frame, law=law, simultaneous=False, **_tables(configuration))


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
    """The largest absolute extra-score column, closed and initial, over all K arms."""
    reduction = result.repeats[0].fluctuations["mean"].reduction
    rows = []
    for cell, role, score in (
        ("five_reduction_cycle__closed_score", "positive", reduction.score),
        ("five_reduction_cycle__initial_score_control", "control", reduction.score_initial),
    ):
        rows.append(
            control_row(
                property_name=property_name,
                cell=cell,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=0.0,
                estimate=float(np.max(np.abs(score))),
                standard_error=1.0,
                critical=1.0,
                role=role,
            )
        )
    return rows


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    if property_name == "type_i_error":
        frame = laws.sample(NULL_LAW, n, seed)
        result = _fit(frame, configuration, law=NULL_LAW)
        return [
            _row(
                property_name,
                cell,
                "positive",
                replicate,
                n,
                requested,
                NULL_TRUTHS[TEST_TARGET],
                result[TEST_TARGET],
            )
        ]
    frame = laws.sample(LAW, n, seed)
    result = _fit(frame, configuration)
    if property_name == "corrected_mar_inference":
        role = "control" if configuration == "both_wrong" else "positive"
        return [
            _row(
                property_name,
                f"{prefix}__{configuration}",
                role,
                replicate,
                n,
                requested,
                TRUTHS[name],
                result[name],
            )
            for prefix, name in CONTRASTS.items()
        ]
    if property_name == "correction_necessity":
        return _necessity_rows(result, property_name, replicate, n, requested)
    target = TEST_TARGET if property_name == "power" else TARGET
    role = "positive"
    if property_name == "root_n_and_efficiency" and n == min(RATE_SIZES):
        role = "control"
    return [
        _row(property_name, cell, role, replicate, n, requested, TRUTHS[target], result[target])
    ]


def _specs() -> list[tuple[str, str, int, int, str]]:
    specs: list[tuple[str, str, int, int, str]] = [
        (
            "corrected_mar_inference",
            configuration,
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            configuration,
        )
        for configuration in CONFIGURATIONS
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
        (
            "type_i_error",
            "l3_ate_mid__randomized_sharp_null",
            TEST_N,
            NULL_REPLICATES,
            "both_correct",
        ),
        (
            "power",
            "l3_ate_mid__randomized_alternative",
            TEST_N,
            POWER_REPLICATES,
            "both_correct",
        ),
        (
            "correction_necessity",
            "five_reduction_cycle",
            CORRECTION_N,
            CORRECTION_REPLICATES,
            "outcome_drift",
        ),
    ]
    return specs


def _payloads(
    budget: int | None = None,
) -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    """Every replication's payload. ``budget`` caps each cell, for a smoke run only."""
    out: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in _specs():
        count = replicates if budget is None else min(budget, replicates)
        for replicate in range(count):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            out.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
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
