"""Shared repeated-sampling properties for categorical longitudinal TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_multivalue as law
from tests.parallel import STUDY_JOBS
from tests.studies import categorical_longitudinal_common as common
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
TARGETING_REPLICATES = 1_200
TARGETING_N = 2_000
MUTATION_REPLICATES = 1_200
MUTATION_N = 2_000
OVERFIT_REPLICATES = 40_000
OVERFIT_N = 1_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.10
CATEGORICAL_PROBABILITY_DISPLACEMENT = 0.10
RULE_DISPLACEMENT = 0.10
#: Shared documentation vocabulary for :func:`rule_necessity` families.
NECESSITY_DISPLACEMENT = RULE_DISPLACEMENT

EFFICIENCY_SD = {
    label: float(np.sqrt(np.sum(law.PROBS * law.eif_at(law.PROBS, name) ** 2)))
    for label, name in common.CONTRASTS.items()
}


def _null_outcome() -> np.ndarray:
    """A no-effect law that still makes the intermediate history load bearing."""
    outcome = np.empty_like(law.Q)
    for w in range(2):
        for a1 in range(3):
            for l2 in range(2):
                value = 0.5 + 0.2 * (l2 - law.P_L2[w, a1])
                outcome[w, a1, l2, :] = value
    return outcome


NULL_OUTCOME = _null_outcome()
NULL_PROBS = law.probabilities(NULL_OUTCOME)
NULL_TRUTH = float(law.functional(NULL_PROBS, common.DYNAMIC_NAME))


def _replicate(
    record: StudyRecord,
    cross_fit: bool,
    property_name: str,
    cell_suffix: str,
    replicate: int,
    n: int,
    requested: int,
    seed: int,
    configuration: str,
) -> list[dict[str, Any]]:
    if property_name == "crossfit_overfitting":
        frame = law.sample(law.PROBS, n, seed, noise=True)
        result = common.fit(frame, cross_fit=cross_fit, configuration=configuration)
        estimate = result[common.DYNAMIC_NAME]
        return [
            replicate_row(
                property_name=property_name,
                cell=cell_suffix,
                role="control" if configuration == "overfit_control" else "positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=float(law.TRUTH[common.DYNAMIC_NAME]),
                estimate=estimate,
                alpha=record.margins.alpha,
            )
        ]

    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    frame = law.sample(probs, n, seed)

    if property_name == "categorical_probability_necessity":
        positive = common.fit(frame, cross_fit=cross_fit, configuration="mechanism_correct")
        control = common.fit(frame, cross_fit=cross_fit, configuration="binary_complement")
        truth = float(law.TRUTH[common.STATIC_NAME])
        return [
            replicate_row(
                property_name=property_name,
                cell="third_arm__assigned_probability",
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=positive[common.STATIC_NAME],
                alpha=record.margins.alpha,
            ),
            replicate_row(
                property_name=property_name,
                cell="third_arm__binary_complement",
                role="control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=control[common.STATIC_NAME],
                alpha=record.margins.alpha,
            ),
        ]

    if property_name == "rule_necessity":
        positive = common.fit(frame, cross_fit=cross_fit, configuration="both_correct")
        control = common.fit(
            frame, cross_fit=cross_fit, configuration="both_correct", mutate_rule=True
        )
        truth = float(law.TRUTH[common.DYNAMIC_NAME])
        return [
            replicate_row(
                property_name=property_name,
                cell="dynamic__declared_rule",
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=positive[common.DYNAMIC_NAME],
                alpha=record.margins.alpha,
            ),
            replicate_row(
                property_name=property_name,
                cell="dynamic__reversed_rule",
                role="control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=control[common.DYNAMIC_NAME],
                alpha=record.margins.alpha,
            ),
        ]

    result = common.fit(frame, cross_fit=cross_fit, configuration=configuration)
    labels = ("dynamic",) if property_name in {"type_i_error", "power"} else tuple(common.CONTRASTS)
    rows: list[dict[str, Any]] = []
    for label in labels:
        name = common.CONTRASTS[label]
        truth = NULL_TRUTH if property_name == "type_i_error" else float(law.TRUTH[name])
        role = (
            "control"
            if cell_suffix == "both_wrong"
            or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
            else "positive"
        )
        rows.append(
            replicate_row(
                property_name=property_name,
                cell=f"{label}__{cell_suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
                alpha=record.margins.alpha,
            )
        )
        if property_name == "targeting_necessity":
            left, right = name[len("ate_regimen[") : -1].split(" vs ")
            plug_in = common.untargeted(
                frame, left, configuration, result.folds
            ) - common.untargeted(frame, right, configuration, result.folds)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=plug_in,
                    standard_error=float(result[name].std_error),
                    critical=float(norm.ppf(1.0 - record.margins.alpha / 2.0)),
                )
            )
    return rows


def _payloads(
    record: StudyRecord, cross_fit: bool
) -> list[tuple[StudyRecord, bool, str, str, int, int, int, int, str]]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
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
                TARGETING_N,
                TARGETING_REPLICATES,
                "mechanism_correct",
            ),
            (
                "categorical_probability_necessity",
                "paired",
                MUTATION_N,
                MUTATION_REPLICATES,
                "paired",
            ),
            ("rule_necessity", "paired", MUTATION_N, MUTATION_REPLICATES, "paired"),
        ]
    )
    if cross_fit:
        specs.extend(
            [
                (
                    "crossfit_overfitting",
                    "cross_fitted_categorical_ltmle",
                    OVERFIT_N,
                    OVERFIT_REPLICATES,
                    "overfit_crossfit",
                ),
                (
                    "crossfit_overfitting",
                    "in_sample_control",
                    OVERFIT_N,
                    OVERFIT_REPLICATES,
                    "overfit_control",
                ),
            ]
        )
    payloads = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            paired_cell = (
                "paired"
                if property_name
                in {
                    "categorical_probability_necessity",
                    "rule_necessity",
                    "crossfit_overfitting",
                }
                else cell
            )
            seed = stream_seed(record, "property_sample", property_name, paired_cell, replicate)
            payloads.append(
                (
                    record,
                    cross_fit,
                    property_name,
                    cell,
                    replicate,
                    n,
                    replicates,
                    seed,
                    configuration,
                )
            )
    return payloads


def generate_property_rows(
    record: StudyRecord, *, cross_fit: bool, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    outcomes = map_parallel(_replicate, _payloads(record, cross_fit), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                record,
                labels=tuple(common.CONTRASTS),
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=float(norm.ppf(1.0 - record.margins.alpha / 2.0)),
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(
    rows: pd.DataFrame, record: StudyRecord, *, cross_fit: bool
) -> pd.DataFrame:
    extra = [
        "targeting_displacement",
        "categorical_probability_displacement",
        "rule_displacement",
    ]
    if cross_fit:
        extra.extend(["coverage_gain_ci_lower", "coverage_gain_ci_upper"])
    summary, rates = apply_shared_verdicts(
        rows,
        record,
        extra_columns=tuple(extra),
        rate_labels=tuple(common.CONTRASTS),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=record.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=tuple(common.CONTRASTS),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="categorical_probability_necessity",
        labels=("third_arm",),
        arms=("assigned_probability", "binary_complement"),
        column="categorical_probability_displacement",
        threshold=CATEGORICAL_PROBABILITY_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="rule_necessity",
        labels=("dynamic",),
        arms=("declared_rule", "reversed_rule"),
        column="rule_displacement",
        threshold=RULE_DISPLACEMENT,
    )
    if cross_fit:
        crossfit_overfitting_verdicts(
            summary,
            rows,
            record,
            positive_cell="cross_fitted_categorical_ltmle",
        )
    return finish(summary, rates)
