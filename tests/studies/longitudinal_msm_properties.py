"""Independent properties for the ordinary longitudinal MSM projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.datasets import RULE_LABEL
from cleverly.longitudinal import LTMLE
from tests import discrete_law_longitudinal as law
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_properties as end_study_properties
from tests.studies.canonical_longitudinal_msm import DURATION as study_durations
from tests.studies.canonical_longitudinal_msm import (
    G_BOUNDS,
    STUDY,
    TERMS,
    declared_msm,
)
from tests.studies.canonical_longitudinal_msm import (
    PROJECTION_WEIGHT as study_projection_weights,
)
from tests.studies.canonical_ltmle import declared_regimens
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_000
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 700
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N
PROJECTION_REPLICATES = DOUBLE_ROBUST_REPLICATES
PROJECTION_N = DOUBLE_ROBUST_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25
PROJECTION_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

LABELS = ("never", "always", "early", "treat_if_l2")
REGIMENS = declared_regimens({label: law.REGIMEN_SPEC[label] for label in LABELS})

# The two laws spell the same dynamic plan differently: ``cleverly.datasets`` calls it
# ``RULE_LABEL`` and the finite-support law calls it ``treat_if_l2``.  Only the spelling
# differs, so the declared measure is *relabelled* from the paired study rather than typed
# out a second time.  A projection coefficient is defined by its measure, so a property study
# that drifted onto a different one would be reporting bias against a parameter the paired
# study never estimated -- and both halves would still be internally consistent.
_RELABEL = {RULE_LABEL: "treat_if_l2"}


def _relabelled(declared: Mapping[str, float]) -> dict[str, float]:
    return {_RELABEL.get(label, label): float(value) for label, value in declared.items()}


DURATION = _relabelled(study_durations)
PROJECTION_WEIGHT = _relabelled(study_projection_weights)
if tuple(DURATION) != LABELS or tuple(PROJECTION_WEIGHT) != LABELS:  # pragma: no cover
    raise AssertionError(
        "the paired study reordered or renamed its plans; the projection operator below "
        f"reads them positionally, so {LABELS} must still be the declared order"
    )
NAMES = {term: f"msm_regimen[{term}]" for term in TERMS}
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def projection_operator(weights: dict[str, float] = PROJECTION_WEIGHT) -> np.ndarray:
    design = np.column_stack([np.ones(len(LABELS)), [DURATION[label] for label in LABELS]])
    mass = np.array([weights[label] for label in LABELS])
    return np.linalg.solve(design.T @ (mass[:, None] * design), design.T * mass)


def coefficients(probs: np.ndarray, weights: dict[str, float] = PROJECTION_WEIGHT) -> np.ndarray:
    means = np.array([law.functional(probs, f"ey_regimen[{label}]") for label in LABELS])
    return projection_operator(weights) @ means


def influence_curves() -> np.ndarray:
    curves = np.column_stack([law.eif(f"ey_regimen[{label}]") for label in LABELS])
    return curves @ projection_operator().T


TRUTH = dict(zip(NAMES.values(), coefficients(law.PROBS), strict=True))
EFFICIENCY_SD = {
    term: float(np.sqrt(np.sum(law.PROBS * influence_curves()[:, index] ** 2)))
    for index, term in enumerate(TERMS)
}

NULL_OUTCOME = np.array(end_study_properties.NULL_OUTCOME, copy=True)
NULL_OUTCOME[:, 1, :, 0] = 0.5
NULL_PROBS = law.probabilities(NULL_OUTCOME)
NULL_TRUTH = float(coefficients(NULL_PROBS)[TERMS.index("duration")])


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    return end_study_properties.sample(probs, n, seed)


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        law.CellMeans() if g_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if g_correct else DummyClassifier(strategy="prior"),
    )


def fit(
    frame: pd.DataFrame,
    configuration: str = "both_correct",
    *,
    uniform: bool = False,
) -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration)
    weights = dict.fromkeys(LABELS, 1.0) if uniform else PROJECTION_WEIGHT
    return LTMLE(
        REGIMENS,
        msm=declared_msm(DURATION, weights),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **COLUMNS)


#: The calibration positive arm of each term to the controls derived from it.
DERIVED_CELLS = calibration_derived(TERMS)

_Key = tuple[tuple[str, str, int, int, int, int, str]]

_SINGLE_TERM = {"type_i_error", "power", "targeting_necessity", "projection_necessity"}


def _terms(property_name: str) -> tuple[str, ...]:
    return ("duration",) if property_name in _SINGLE_TERM else TERMS


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    property_name, _, _, n, _, seed, _ = key[0]
    return sample(NULL_PROBS if property_name == "type_i_error" else law.PROBS, n, seed)


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    property_name, cell_suffix = key[0][:2]
    names = []
    for term in _terms(property_name):
        name = f"{property_name}/{term}__{cell_suffix}"
        names.append(name)
        names.extend(DERIVED_CELLS.get(name, ()))
        if property_name == "targeting_necessity":
            names.append(f"{property_name}/{term}__untargeted")
        if property_name == "projection_necessity":
            names.append(f"{property_name}/{term}__uniform_weights")
    return names


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    property_name, cell_suffix, replicate, n, requested, _seed, configuration = key[0]
    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    result = fit(frame, configuration)
    terms = _terms(property_name)
    truths = coefficients(probs)
    rows: list[dict[str, Any]] = []
    for term in terms:
        name = NAMES[term]
        index = TERMS.index(term)
        truth = float(truths[index])
        role = (
            "control"
            if cell_suffix == "both_wrong"
            or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
            else "positive"
        )
        rows.append(
            replicate_row(
                property_name=property_name,
                cell=f"{term}__{cell_suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
                alpha=STUDY.margins.alpha,
            )
        )
        if property_name == "targeting_necessity":
            plug_in_means = {
                label: end_study_properties.untargeted(frame, label, configuration)
                for label in LABELS
            }
            plug_in = projection_operator() @ np.array([plug_in_means[label] for label in LABELS])
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{term}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=float(plug_in[index]),
                    standard_error=float(result[name].std_error),
                    critical=CRITICAL,
                )
            )
        if property_name == "projection_necessity":
            wrong = fit(frame, configuration, uniform=True)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{term}__uniform_weights",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=float(wrong[name].psi),
                    standard_error=float(wrong[name].std_error),
                    critical=CRITICAL,
                )
            )
    return rows


def _payloads() -> list[_Key]:
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
                "projection_necessity",
                "declared_weights",
                PROJECTION_N,
                PROJECTION_REPLICATES,
                "both_correct",
            ),
        ]
    )
    payloads: list[_Key] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            payloads.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return payloads


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = fit_properties(
        _payloads(),
        _fit_replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=TERMS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "projection_displacement"),
        rate_labels=TERMS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("duration",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="projection_necessity",
        labels=("duration",),
        arms=("declared_weights", "uniform_weights"),
        column="projection_displacement",
        threshold=PROJECTION_DISPLACEMENT,
    )
    return finish(summary, rates)
