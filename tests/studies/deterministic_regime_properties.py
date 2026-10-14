"""Independent repeated-sampling properties for deterministic regimes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from cleverly.interventions import Static
from tests import discrete_law as law
from tests.parallel import STUDY_JOBS
from tests.studies import regime_property_helpers as shared_regime
from tests.studies.canonical_deterministic_regimes import G_BOUNDS, STUDY, interventions
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import calibration_derived
from tests.studies.evidence.replication import fit_properties, splat
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
    learners,
    summarize,
)

# Public aliases consumed by the shared evidence-document renderer. Their values live in
# one place so the deterministic and stochastic studies cannot drift apart.
EFFICIENCY_RATIO_BAND = shared_regime.EFFICIENCY_RATIO_BAND
NECESSITY_DISPLACEMENT = shared_regime.NECESSITY_DISPLACEMENT
SHRUNKEN_SE_FACTOR = shared_regime.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = shared_regime.TARGETING_DISPLACEMENT

TARGET = "ate_regime[rule vs never]"
LABEL = "rule"
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
    if property_name == "rule_necessity":
        rows[0]["cell"] = f"{LABEL}__declared"
        control = fit_regimes(
            frame,
            probs,
            configuration,
            (Static(0.0, name="never"), Static(1.0, name="rule")),
            g_bounds=G_BOUNDS,
        )
        rows.append(
            control_row(
                property_name=property_name,
                cell=f"{LABEL}__static_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=float(control[TARGET].psi),
                standard_error=float(control[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "static_reduction":
        outcome, treatment = learners(probs, configuration)
        regime = (
            TMLE(
                interventions=(Static(0.0), Static(1.0)),
                outcome_learner=outcome,
                treatment_learner=treatment,
                cross_fit=False,
                simultaneous=False,
                g_bounds=G_BOUNDS,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
        outcome, treatment = learners(probs, configuration)
        arm = (
            TMLE(
                estimands=("ey0", "ey1", "ate"),
                outcome_learner=outcome,
                treatment_learner=treatment,
                cross_fit=False,
                simultaneous=False,
                g_bounds=G_BOUNDS,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
        rows = []
        for cell_name, estimate in (
            ("never__regime", regime["ey_regime[always 0.0]"]),
            ("never__arm", arm["ey0"]),
        ):
            rows.append(
                replicate_row(
                    property_name=property_name,
                    cell=cell_name,
                    role="positive",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=float(law.functional(probs, "ey0")),
                    estimate=estimate,
                    alpha=STUDY.margins.alpha,
                )
            )
    return rows


#: The calibration positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived((LABEL,))


def _cells_of(key: tuple[tuple[str, str, int, int, int, int, str]]) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    property_name, cell = key[0][:2]
    if property_name == "targeting_necessity":
        return [f"{property_name}/{LABEL}__targeted", f"{property_name}/{LABEL}__untargeted"]
    if property_name == "rule_necessity":
        return [f"{property_name}/{LABEL}__declared", f"{property_name}/{LABEL}__static_control"]
    if property_name == "static_reduction":
        return [f"{property_name}/never__regime", f"{property_name}/never__arm"]
    name = f"{property_name}/{cell}"
    return [name, *DERIVED_CELLS.get(name, ())]


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
            ("rule_necessity", "declared", NECESSITY_N, NECESSITY_REPLICATES, "both_correct"),
            ("static_reduction", "reduction", NECESSITY_N, NECESSITY_REPLICATES, "both_correct"),
        ]
    )
    out: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            out.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return out


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = fit_properties(
        _payloads(),
        splat(_fit_replication),
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return add_calibration_controls(rows, STUDY, label=LABEL, efficiency_sd=EFFICIENCY_SD)


#: The study's two necessity declarations, read by its verdicts and its rules.
NECESSITIES = shared_regime.necessities(LABEL, "rule_necessity", ("declared", "static_control"))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        rows,
        STUDY,
        label=LABEL,
        efficiency_sd=EFFICIENCY_SD,
        declared=NECESSITIES,
        include_static_reduction=True,
    )


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).
PROPERTY_RULES = shared_regime.property_rules(NECESSITIES, include_static_reduction=True)
