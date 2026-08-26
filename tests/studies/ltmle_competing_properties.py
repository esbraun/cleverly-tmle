"""Independent repeated-sampling properties for competing-risk LTMLE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_competing as law
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ltmle_competing as canonical
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    paired_displacement,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 4_000
RATE_REPLICATES = 1_600
RATE_SIZES = (4_000, 8_000, 32_000)
CALIBRATION_REPLICATES = 9_600
CALIBRATION_N = 32_000
NULL_REPLICATES = 1_600
NULL_N = 4_000
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N
RECURSION_REPLICATES = DOUBLE_ROBUST_REPLICATES
RECURSION_N = DOUBLE_ROBUST_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25
RECURSION_DISPLACEMENT = 0.25

STUDY = canonical.STUDY
REGIMENS = law.REGIMEN_SPEC
REFERENCE = law.REGIMEN_REFERENCE
PROPERTY_LABELS = canonical.PROPERTY_LABELS
CONTRASTS = {
    "relapse_dynamic_t2": "ate_regimen[continue_if_l2 vs never, relapse @ t=2]",
    "death_static_t2": "ate_regimen[always vs never, death @ t=2]",
}
EFFICIENCY_SD = {
    label: float(np.sqrt(np.sum(law.PROBS * law.eif(name) ** 2)))
    for label, name in CONTRASTS.items()
}

# The null retains baseline confounding and two positive causes.  Treatment does not enter
# either hazard, and the second-node hazards do not depend on the treatment-affected L2.
NULL_H1 = np.array(
    [
        [[0.25, 0.25], [0.50, 0.50]],
        [[0.25, 0.25], [0.25, 0.25]],
    ]
)
NULL_H2 = np.full_like(law.H2, 0.25)
# At W=1 the two cause hazards remain distinct while their sum stays below one.
NULL_H2[0, 1, :, :, :] = 0.50
NULL_PROBS = law.probabilities(NULL_H1, NULL_H2)
POWER_PROBS = law.PROBS
NULL_TRUTH = {label: float(law.functional(NULL_PROBS, name)) for label, name in CONTRASTS.items()}
POWER_TRUTH = {label: float(law.functional(POWER_PROBS, name)) for label, name in CONTRASTS.items()}

_SUPPORT_FRAME = law.frame().iloc[law.first_row_of()].reset_index(drop=True)


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    cells = np.random.default_rng(seed).choice(len(law.SUPPORT), size=n, p=probs)
    return _SUPPORT_FRAME.iloc[cells].reset_index(drop=True)


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        canonical.KnownCompetingMechanism("treatment")
        if g_correct
        else DummyClassifier(strategy="prior"),
        canonical.KnownCompetingMechanism("censoring")
        if g_correct
        else DummyClassifier(strategy="prior"),
    )


def fit(frame: pd.DataFrame, configuration: str = "both_correct", *, n_folds: int = 1) -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration)
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        horizons=(2,),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=n_folds,
        learner_folds=5,
        g_bounds=canonical.G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _plan_arms(frame: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    first, second = law.REGIMEN_ARMS[label]
    w = frame["W"].to_numpy().astype(int)
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    node1 = np.full(len(frame), float(first)) if np.ndim(first) == 0 else np.asarray(first)[w]
    node2 = (
        np.full(len(frame), float(second)) if np.ndim(second) == 0 else np.asarray(second)[w, l2]
    )
    return np.asarray(node1), np.asarray(node2)


def _fold_masks(folds: Folds, held_out: int) -> tuple[np.ndarray, np.ndarray]:
    evaluated = folds.assignment == held_out
    training = np.ones_like(evaluated) if folds.n_folds == 1 else ~evaluated
    return training, evaluated


def untargeted(
    frame: pd.DataFrame,
    label: str,
    cause: str,
    horizon: int,
    configuration: str,
    folds: Folds,
    *,
    cause_specific_survival: bool = False,
) -> float:
    """Fold-specific unfluctuated recursion for one cause and plan."""
    outcome, pseudo, _, _ = _learners(configuration)
    first, second = _plan_arms(frame, label)
    target1, target2 = law.outcome_columns()[cause]
    other = next(value for value in law.CAUSES if value != cause)
    other1 = law.outcome_columns()[other][0]
    y1 = np.nan_to_num(frame[target1].to_numpy())
    d1 = np.nan_to_num(frame[other1].to_numpy())
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    baseline = frame[["W"]].to_numpy(dtype=float)
    stitched = np.empty(len(frame), dtype=float)

    if horizon == 1:
        for held_out in range(folds.n_folds):
            training, evaluated = _fold_masks(folds, held_out)
            model = clone(outcome).fit(
                baseline[training & followed_one], y1[training & followed_one]
            )
            stitched[evaluated] = model.predict_proba(baseline[evaluated])[:, 1]
        return float(np.mean(stitched))

    history = np.column_stack([baseline[:, 0], np.nan_to_num(frame["L2"].to_numpy())])
    event_free = (y1 == 0.0) & (d1 == 0.0)
    followed_two = (
        followed_one
        & event_free
        & (frame["C2"].to_numpy() == 1.0)
        & (frame["A2"].to_numpy() == second)
    )
    events = frame[target2].to_numpy()
    for held_out in range(folds.n_folds):
        training, evaluated = _fold_masks(folds, held_out)
        later = clone(outcome).fit(
            history[training & followed_two], events[training & followed_two]
        )
        carried = np.asarray(later.predict_proba(history))[:, 1]
        survival = 1.0 - y1 if cause_specific_survival else event_free.astype(float)
        pseudo_outcome = y1 + survival * carried
        earlier = clone(pseudo).fit(
            baseline[training & followed_one], pseudo_outcome[training & followed_one]
        )
        stitched[evaluated] = earlier.predict(baseline[evaluated])
    return float(np.mean(stitched))


def _fit_replication(
    payload: tuple[str, str, int, int, int, int, str],
    *,
    study: Any,
    fit_fn: Callable[[pd.DataFrame, str], Any],
) -> list[dict[str, Any]]:
    property_name, suffix, replicate, n, requested, seed, configuration = payload
    probs = (
        NULL_PROBS
        if property_name == "type_i_error"
        else POWER_PROBS
        if property_name == "power"
        else law.PROBS
    )
    frame = sample(probs, n, seed)
    result = fit_fn(frame, configuration)
    rows: list[dict[str, Any]] = []
    for label, name in CONTRASTS.items():
        truth = (
            NULL_TRUTH[label]
            if property_name == "type_i_error"
            else POWER_TRUTH[label]
            if property_name == "power"
            else float(law.TRUTH[name])
        )
        role = "control" if suffix == "both_wrong" else "positive"
        rows.append(
            replicate_row(
                property_name=property_name,
                cell=f"{label}__{suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
                alpha=study.margins.alpha,
            )
        )
        if property_name == "targeting_necessity":
            inside = name[len("ate_regimen[") : -1]
            comparison_cause, horizon_text = inside.rsplit(" @ t=", 1)
            comparison, cause = comparison_cause.rsplit(", ", 1)
            left, right = comparison.split(" vs ")
            horizon = int(horizon_text)
            unfluctuated = untargeted(
                frame, left, cause, horizon, configuration, result.folds
            ) - untargeted(frame, right, cause, horizon, configuration, result.folds)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=unfluctuated,
                    standard_error=float(result[name].std_error),
                    critical=float(norm.ppf(1.0 - study.margins.alpha / 2.0)),
                )
            )
    return rows


def _recursion_replication(
    payload: tuple[int, int, int, int],
    *,
    study: Any,
    fit_fn: Callable[[pd.DataFrame, str], Any],
) -> list[dict[str, Any]]:
    replicate, n, requested, seed = payload
    frame = sample(law.PROBS, n, seed)
    result = fit_fn(frame, "both_correct")
    rows: list[dict[str, Any]] = []
    for cause in law.CAUSES:
        name = f"cif_regimen[always, {cause} @ t=2]"
        estimate = result[name]
        truth = float(law.TRUTH[name])
        rows.append(
            replicate_row(
                property_name="competing_risk_recursion_necessity",
                cell=f"{cause}_always_t2__all_cause",
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=estimate,
                alpha=study.margins.alpha,
            )
        )
        rows.append(
            control_row(
                property_name="competing_risk_recursion_necessity",
                cell=f"{cause}_always_t2__cause_specific_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=untargeted(
                    frame,
                    "always",
                    cause,
                    2,
                    "both_correct",
                    result.folds,
                    cause_specific_survival=True,
                ),
                standard_error=float(estimate.std_error),
                critical=float(norm.ppf(1.0 - study.margins.alpha / 2.0)),
            )
        )
    return rows


def _payloads(study: Any) -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
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
        ]
    )
    return [
        (
            (
                property_name,
                cell,
                replicate,
                n,
                replicates,
                stream_seed(study, "property_sample", property_name, cell, replicate),
                configuration,
            ),
        )
        for property_name, cell, n, replicates, configuration in specs
        for replicate in range(replicates)
    ]


def generate_for(
    study: Any,
    fit_fn: Callable[[pd.DataFrame, str], Any],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    def runner(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
        return _fit_replication(payload, study=study, fit_fn=fit_fn)

    outcomes = map_parallel(runner, _payloads(study), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    recursion_payloads = [
        (
            (
                replicate,
                RECURSION_N,
                RECURSION_REPLICATES,
                stream_seed(study, "competing_recursion", replicate),
            ),
        )
        for replicate in range(RECURSION_REPLICATES)
    ]

    def recursion_runner(payload: tuple[int, int, int, int]) -> list[dict[str, Any]]:
        return _recursion_replication(payload, study=study, fit_fn=fit_fn)

    recursion = map_parallel(recursion_runner, recursion_payloads, n_jobs=n_jobs)
    rows = pd.concat(
        [
            rows,
            pd.DataFrame([row for result in recursion for row in result]),
            calibration_controls(
                rows,
                study,
                labels=PROPERTY_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=float(norm.ppf(1.0 - study.margins.alpha / 2.0)),
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate_for(STUDY, lambda frame, config: fit(frame, config), n_jobs=n_jobs)


def summarize_for(
    rows: pd.DataFrame, study: Any, *, crossfit_positive_cell: str | None = None
) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        study,
        extra_columns=("targeting_displacement", "recursion_displacement"),
        rate_labels=PROPERTY_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=study.margins, efficiency_band=EFFICIENCY_RATIO_BAND)

    targeting = summary["property"] == "targeting_necessity"
    summary.loc[targeting & (summary["role"] == "positive"), "passed"] = summary.loc[
        targeting & (summary["role"] == "positive"), "bias_equivalent"
    ]
    summary.loc[targeting & (summary["role"] == "control"), "passed"] = summary.loc[
        targeting & (summary["role"] == "control"), "bias_discriminated"
    ]
    targeting_displacement = min(
        paired_displacement(
            rows, "targeting_necessity", f"{label}__targeted", f"{label}__untargeted"
        )
        for label in PROPERTY_LABELS
    )
    summary.loc[targeting, "targeting_displacement"] = targeting_displacement
    summary.loc[targeting, "property_passed"] = bool(
        summary.loc[targeting, "passed"].all() and targeting_displacement >= TARGETING_DISPLACEMENT
    )

    recursion = summary["property"] == "competing_risk_recursion_necessity"
    summary.loc[recursion & (summary["role"] == "positive"), "passed"] = summary.loc[
        recursion & (summary["role"] == "positive"), "bias_equivalent"
    ]
    summary.loc[recursion & (summary["role"] == "control"), "passed"] = summary.loc[
        recursion & (summary["role"] == "control"), "bias_discriminated"
    ]
    recursion_displacement = min(
        paired_displacement(
            rows,
            "competing_risk_recursion_necessity",
            f"{cause}_always_t2__all_cause",
            f"{cause}_always_t2__cause_specific_control",
        )
        for cause in law.CAUSES
    )
    summary.loc[recursion, "recursion_displacement"] = recursion_displacement
    summary.loc[recursion, "property_passed"] = bool(
        summary.loc[recursion, "passed"].all() and recursion_displacement >= RECURSION_DISPLACEMENT
    )
    if crossfit_positive_cell is not None:
        crossfit_overfitting_verdicts(summary, rows, study, positive_cell=crossfit_positive_cell)
    return finish(summary, rates)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize_for(rows, STUDY)
