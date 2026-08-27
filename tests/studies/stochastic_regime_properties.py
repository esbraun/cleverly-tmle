"""Independent repeated-sampling properties for known stochastic regimes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests.parallel import STUDY_JOBS
from tests.studies import regime_property_helpers as shared_regime
from tests.studies.canonical_stochastic_regimes import G_BOUNDS, STUDY, interventions
from tests.studies.evidence.properties import control_row, replicate_row
from tests.studies.evidence.seeds import stream_seed
from tests.studies.intervention_study_helpers import (
    efficiency_sd,
    initial_regime_estimates,
    sample_discrete,
)
from tests.studies.regime_property_helpers import (
    CALIBRATION_N,
    CALIBRATION_REPLICATES,
    DOUBLE_ROBUST_N,
    DOUBLE_ROBUST_REPLICATES,
    NECESSITY_N,
    NECESSITY_REPLICATES,
    NULL_N,
    NULL_PROBS,
    NULL_REPLICATES,
    RATE_REPLICATES,
    RATE_SIZES,
    add_calibration_controls,
    fit_regimes,
    summarize,
)

# Public aliases consumed by the shared evidence-document renderer. Their values live in
# one place so the deterministic and stochastic studies cannot drift apart.
EFFICIENCY_RATIO_BAND = shared_regime.EFFICIENCY_RATIO_BAND
NECESSITY_DISPLACEMENT = shared_regime.NECESSITY_DISPLACEMENT
SHRUNKEN_SE_FACTOR = shared_regime.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = shared_regime.TARGETING_DISPLACEMENT

TARGET = "ate_regime[tilt vs never]"
LABEL = "tilt"
TRUTH = float(law.functional(law.PROBS, TARGET))
NULL_TRUTH = float(law.functional(NULL_PROBS, TARGET))
EFFICIENCY_SD = efficiency_sd(law.PROBS, TARGET)
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    frame = sample_discrete(probs, n, seed)
    result = fit_regimes(frame, probs, configuration, interventions(), g_bounds=G_BOUNDS)
    truth = NULL_TRUTH if property_name == "type_i_error" else TRUTH
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
        rows[0]["cell"] = f"{LABEL}__targeted"
        initial = initial_regime_estimates(result)[TARGET]
        rows.append(
            control_row(
                property_name=property_name,
                cell=f"{LABEL}__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=initial,
                standard_error=float(result[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "density_necessity":
        rows[0]["cell"] = f"{LABEL}__declared"
        control = fit_regimes(
            frame,
            probs,
            configuration,
            interventions(uniform=True),
            g_bounds=G_BOUNDS,
        )
        rows.append(
            control_row(
                property_name=property_name,
                cell=f"{LABEL}__uniform_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(control[TARGET].psi),
                standard_error=float(control[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_correct", "treatment_correct", "both_wrong"):
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
                f"{LABEL}__correctly_specified",
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
                "treatment_correct",
            ),
            (
                "density_necessity",
                "declared",
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
    return add_calibration_controls(rows, STUDY, label=LABEL, efficiency_sd=EFFICIENCY_SD)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        rows,
        STUDY,
        label=LABEL,
        efficiency_sd=EFFICIENCY_SD,
        necessity_family="density_necessity",
        necessity_arms=("declared", "uniform_control"),
        include_static_reduction=False,
    )
