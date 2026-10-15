"""Shared verdict and nuisance helpers for the two regime studies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.property_verdicts import (
    Necessity,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.intervention_study_helpers import (
    INTERVENTION_CALIBRATION_REPLICATES,
    probabilities,
)

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
EFFICIENCY_RATIO_BAND = (0.90, 1.10)

WRONG_Q = 1.0 - law.Q
WRONG_G = np.array([0.75, 0.25, 0.75])
WRONG_PROBS = probabilities(WRONG_Q, g=WRONG_G)
NULL_Q = np.column_stack([np.array([0.25, 0.45, 0.65])] * 2)
NULL_PROBS = probabilities(NULL_Q)


def learners(probs: np.ndarray, configuration: str) -> tuple[Any, Any]:
    """Return fixed correct or deliberately wrong finite-law nuisance learners."""
    dgp = law.DiscreteLaw(probs)
    wrong = law.DiscreteLaw(WRONG_PROBS)
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "treatment_correct"}
    outcome = OracleOutcome(dgp) if q_correct else OracleOutcome(wrong)
    treatment = OracleTreatment(dgp) if g_correct else OracleTreatment(wrong)
    return outcome, treatment


def fit_regimes(
    frame: pd.DataFrame,
    probs: np.ndarray,
    configuration: str,
    interventions: tuple[Any, ...],
    *,
    g_bounds: tuple[float, float],
) -> Any:
    """Fit one ordinary regime TMLE under a declared nuisance configuration."""
    outcome, treatment = learners(probs, configuration)
    return (
        TMLE(
            interventions=interventions,
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=False,
            simultaneous=False,
            g_bounds=g_bounds,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def add_calibration_controls(
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    label: str,
    efficiency_sd: float,
) -> pd.DataFrame:
    """Append the shared shrunken-SE and noise calibration controls."""
    controls = calibration_controls(
        rows,
        record,
        labels=(label,),
        efficiency_bounds={label: efficiency_sd},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=float(norm.ppf(1.0 - record.margins.alpha / 2.0)),
    )
    return pd.concat([rows, controls], ignore_index=True)


#: The cells the ``static_reduction`` identity compares, and the column it is published in.
STATIC_REDUCTION_CELLS = ("never__regime", "never__arm")
STATIC_REDUCTION_COLUMN = "maximum_static_difference"


def necessities(
    label: str, necessity_family: str, necessity_arms: tuple[str, str]
) -> tuple[Necessity, Necessity]:
    """The two necessity declarations of a regime study: targeting, and its own step.

    Parameters
    ----------
    label : str
        The study's parameter label.
    necessity_family : str
        The family of the study's own necessity control.
    necessity_arms : tuple of str
        That family's positive and control suffixes.

    Returns
    -------
    tuple of Necessity
        The ``targeting_necessity`` declaration and the study's own.
    """
    return (
        Necessity(
            family="targeting_necessity",
            labels=(label,),
            arms=("targeted", "untargeted"),
            column="targeting_displacement",
            threshold=TARGETING_DISPLACEMENT,
        ),
        Necessity(
            family=necessity_family,
            labels=(label,),
            arms=necessity_arms,
            column="necessity_displacement",
            threshold=NECESSITY_DISPLACEMENT,
        ),
    )


def property_rules(
    declared: tuple[Necessity, ...], *, include_static_reduction: bool
) -> tuple[pf.FamilyRule, ...]:
    """The declared rules of a regime study (RM39b2 rev 3, D3'a).

    Parameters
    ----------
    declared : tuple of Necessity
        The study's :func:`necessities`.
    include_static_reduction : bool
        Whether the study publishes ``static_reduction``, which is not accountable (R3-1).

    Returns
    -------
    tuple of FamilyRule
        The calibration rule with the efficiency band, the necessity rules, and the identity.
    """
    rules = [
        property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
        *property_verdicts.necessity_rules(*declared),
    ]
    if include_static_reduction:
        rules.append(
            property_verdicts.identity_rule(
                "static_reduction",
                column=STATIC_REDUCTION_COLUMN,
                cells=STATIC_REDUCTION_CELLS,
                tolerance=None,
                shared="regime_property_helpers.summarize",
            )
        )
    return tuple(rules)


def summarize(
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    label: str,
    efficiency_sd: float,
    declared: tuple[Necessity, ...],
    include_static_reduction: bool,
) -> pd.DataFrame:
    """Apply shared verdicts plus regime-specific necessity controls."""
    extra = ["targeting_displacement", "necessity_displacement"]
    if include_static_reduction:
        extra.append(STATIC_REDUCTION_COLUMN)
    summary, rates = apply_shared_verdicts(
        rows,
        record,
        extra_columns=tuple(extra),
        efficiency_bounds={label: efficiency_sd},
    )
    calibration_verdicts(
        summary,
        margins=record.margins,
        efficiency_band=EFFICIENCY_RATIO_BAND,
    )
    for necessity in declared:
        necessity_verdicts(summary, rows, necessity)
    if include_static_reduction:
        mask = summary["property"] == "static_reduction"
        source = rows.loc[rows["property"] == "static_reduction"]
        wide = source.pivot(index="replicate", columns="cell", values="estimate")
        regime, arm = STATIC_REDUCTION_CELLS
        difference = float(np.max(np.abs(wide[regime] - wide[arm])))
        summary.loc[mask, STATIC_REDUCTION_COLUMN] = difference
        summary.loc[mask, "passed"] = difference == 0.0
        summary.loc[mask, "property_passed"] = bool(difference == 0.0)
    return finish(summary, rates)
