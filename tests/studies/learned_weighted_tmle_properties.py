"""Repeated-sampling properties for learned weighted point-treatment TMLE."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies.canonical_learned_weighted_tmle import STUDY, fit_cleverly
from tests.studies.evidence.properties import (
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    alternative_target_necessity_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_weighted_point_common import (
    SELECTED_ATE,
    TARGET_ATE,
    sample_selected,
    weighted_ate_efficiency_sd,
)
from tests.studies.point_study_helpers import initial_estimates

RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 2_000
POWER_REPLICATES = NULL_REPLICATES
POWER_N = NULL_N
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
# The exact bound supplies one independent noise unit. The study does not claim the estimator
# attains it because its outcome regression deliberately omits treatment-effect modification.
CALIBRATION_NOISE_SD = weighted_ate_efficiency_sd()
LEARNER_WEIGHT_DISPLACEMENT = 0.50
ALTERNATIVE_EFFECT = 0.50
TARGET = "ate"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def _row(
    result: Any,
    *,
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
) -> dict[str, Any]:
    return replicate_row(
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


def fit_replication(payload: tuple[str, str, int, int, int, int]) -> list[dict[str, Any]]:
    """Fit one declared property replication and return its shared-schema rows."""
    property_name, cell, replicate, n, requested, seed = payload
    effect = (
        0.0
        if property_name == "type_i_error"
        else (ALTERNATIVE_EFFECT if property_name == "power" else TARGET_ATE)
    )
    frame = sample_selected(n, seed, effect=effect)
    result = fit_cleverly(frame, estimands=(TARGET,))
    if property_name != "learner_weight_necessity":
        return [
            _row(
                result,
                property_name=property_name,
                cell=cell,
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=effect,
            )
        ]

    control = fit_cleverly(frame, estimands=(TARGET,), learner_weights=False)
    weighted_plugin = initial_estimates(result, (TARGET,))[TARGET]
    unweighted_plugin = initial_estimates(control, (TARGET,))[TARGET]
    weighted_se = float(result[TARGET].std_error)
    control_se = float(control[TARGET].std_error)
    return [
        _row(
            result,
            property_name=property_name,
            cell="ate__weighted_targeted",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=TARGET_ATE,
        ),
        _row(
            control,
            property_name=property_name,
            cell="ate__unweighted_targeted",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=TARGET_ATE,
        ),
        control_row(
            property_name=property_name,
            cell="ate__weighted_plugin",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=TARGET_ATE,
            estimate=weighted_plugin,
            standard_error=weighted_se,
            critical=CRITICAL,
        ),
        control_row(
            property_name=property_name,
            cell="ate__unweighted_plugin_control",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=TARGET_ATE,
            estimate=unweighted_plugin,
            standard_error=control_se,
            critical=CRITICAL,
        ),
    ]


#: The calibration positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived((TARGET,), positive_suffix="treatment_correct")

_Key = tuple[tuple[str, str, int, int, int, int]]


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    property_name, cell = key[0][:2]
    if property_name == "learner_weight_necessity":
        return [
            f"{property_name}/ate__{arm}"
            for arm in (
                "weighted_targeted",
                "unweighted_targeted",
                "weighted_plugin",
                "unweighted_plugin_control",
            )
        ]
    name = f"{property_name}/{cell}"
    return [name, *DERIVED_CELLS.get(name, ())]


def _payloads() -> list[_Key]:
    specs = [
        *[
            ("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, f"n_{size}")
            for size in RATE_SIZES
        ],
        (
            "interval_calibration",
            "ate__treatment_correct",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "correctly_specified",
        ),
        ("type_i_error", "target_null", NULL_N, NULL_REPLICATES, "paired_test"),
        ("power", "alternative", POWER_N, POWER_REPLICATES, "paired_test"),
        (
            "learner_weight_necessity",
            "paired",
            NECESSITY_N,
            NECESSITY_REPLICATES,
            "paired",
        ),
    ]
    payloads: list[_Key] = []
    for property_name, cell, n, requested, stream_cell in specs:
        for replicate in range(requested):
            seed = stream_seed(
                STUDY,
                "property_sample",
                property_name,
                stream_cell,
                replicate,
            )
            payloads.append(((property_name, cell, replicate, n, requested, seed),))
    return payloads


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Generate every predeclared cell and paired calibration control."""
    outcomes = fit_properties(
        _payloads(),
        splat(fit_replication),
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    controls = calibration_controls(
        rows,
        STUDY,
        labels=(TARGET,),
        efficiency_bounds={TARGET: CALIBRATION_NOISE_SD},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
        positive_suffix="treatment_correct",
    )
    return pd.concat([rows, controls], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply shared repeated-sampling rules, the calibration rules and the weight witness.

    The positive calibration arm is named ``treatment_correct`` rather than
    ``correctly_specified``, because this study fits a main-effects outcome regression that
    omits the treatment-effect modification.  The study publishes no efficiency ratio for the
    same reason, so :func:`calibration_verdicts` reads both control arms on the SE ratio.
    """
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "necessity_displacement",
            "alternative_truth",
            "alternative_bias_ci_lower",
            "alternative_bias_ci_upper",
            "alternative_bias_margin",
            "alternative_bias_equivalent",
        ),
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix="treatment_correct")

    alternative_target_necessity_verdicts(
        summary,
        rows,
        STUDY,
        family="learner_weight_necessity",
        labels=(TARGET,),
        arms=("weighted_plugin", "unweighted_plugin_control"),
        alternative_truths={TARGET: SELECTED_ATE},
        column="necessity_displacement",
        threshold=LEARNER_WEIGHT_DISPLACEMENT,
    )
    return finish(summary, rates)
