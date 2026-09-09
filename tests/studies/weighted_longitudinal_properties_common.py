"""Shared property machinery for weighted longitudinal evidence studies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.datasets import WEIGHTED_SELECTION_PROBABILITIES
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as law
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    alternative_target_necessity_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed
from tests.studies.ltmle_crossfit_properties import KnownDiscreteMechanism, _plan_arms

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
NECESSITY_DISPLACEMENT = 0.25
TARGETING_DISPLACEMENT = NECESSITY_DISPLACEMENT
WEIGHT_DISPLACEMENT = NECESSITY_DISPLACEMENT
LEARNER_WEIGHT_DISPLACEMENT = NECESSITY_DISPLACEMENT
G_BOUNDS = (1e-8, 1.0)

REGIMENS = {key: law.REGIMEN_SPEC[key] for key in ("never", "always", "treat_if_l2")}
REFERENCE = "never"
CONTRASTS = {
    "static": "ate_regimen[always vs never]",
    "dynamic": "ate_regimen[treat_if_l2 vs never]",
}
WEIGHT_MEANS = {
    "static": "ey_regimen[always]",
    "dynamic": "ey_regimen[treat_if_l2]",
}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Return the common predeclared property-cell inventory."""
    return {
        "double_robustness": tuple(
            f"{label}__{configuration}"
            for label in CONTRASTS
            for configuration in (
                "both_correct",
                "outcome_correct",
                "mechanism_correct",
                "both_wrong",
            )
        ),
        "root_n_and_efficiency": tuple(
            f"{label}__n_{size}" for label in CONTRASTS for size in RATE_SIZES
        ),
        "root_n_rate": tuple(
            f"{label}__{statistic}"
            for label in CONTRASTS
            for statistic in ("empirical_sd", "reported_se")
        ),
        "interval_calibration": tuple(
            f"{label}__{cell}"
            for label in CONTRASTS
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
        "type_i_error": ("static__sharp_null",),
        "power": ("static__alternative",),
        "targeting_necessity": tuple(
            f"{label}__{arm}" for label in CONTRASTS for arm in ("targeted", "untargeted")
        ),
        "weight_necessity": tuple(
            f"{label}__{arm}"
            for label in WEIGHT_MEANS
            for arm in ("weighted", "omitted_weight_control")
        ),
        "learner_weight_necessity": tuple(
            f"static__{arm}" for arm in ("weighted_learners", "discarded_learner_weight_control")
        ),
    }


SELECTION_LOW, SELECTION_HIGH = WEIGHTED_SELECTION_PROBABILITIES


def _selection(point: tuple[Any, ...]) -> float:
    return SELECTION_LOW if float(point[0]) > 0.0 else SELECTION_HIGH


SELECTION = np.array([_selection(point) for point in law.SUPPORT], dtype=float)
SELECTION_RATE = float(np.sum(law.PROBS * SELECTION))
SELECTED_PROBS = law.PROBS * SELECTION / SELECTION_RATE
OBS_WEIGHTS = 1.0 / SELECTION


def _history_weight(point: tuple[Any, ...]) -> float:
    """A load-bearing weight that changes every longitudinal nuisance family."""
    _, a1, c1, l2, _, _, y = point
    observed_l2 = 0.0 if l2 is None else float(l2)
    observed_y = 0.0 if y is None else float(y)
    return 1.0 + 0.5 * float(a1) + 0.4 * (1.0 - float(c1)) + 0.3 * observed_l2 + 0.8 * observed_y


LEARNER_OBS_WEIGHTS = np.array([_history_weight(point) for point in law.SUPPORT], dtype=float)
LEARNER_SELECTION = 1.0 / LEARNER_OBS_WEIGHTS
LEARNER_SELECTED_PROBS = law.PROBS * LEARNER_SELECTION
LEARNER_SELECTED_PROBS /= LEARNER_SELECTED_PROBS.sum()


def efficiency_sd(name: str) -> float:
    """Return the exact selected-law SD of the weighted target-population curve."""
    curve = law.weighted_eif(name, OBS_WEIGHTS, base=SELECTED_PROBS)
    return float(np.sqrt(np.sum(SELECTED_PROBS * np.square(curve))))


EFFICIENCY_SD = {label: efficiency_sd(name) for label, name in CONTRASTS.items()}


def sample(
    probs: np.ndarray,
    n: int,
    seed: int,
    *,
    selection: np.ndarray = SELECTION,
) -> pd.DataFrame:
    """Sample exactly ``n`` selected finite-support rows and attach fixed weights."""
    selected = np.asarray(probs, dtype=float) * selection
    selected /= selected.sum()
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(law.SUPPORT), size=n, p=selected)
    frame = pd.DataFrame(
        {
            name: np.array(
                [
                    np.nan if point[position] is None else float(point[position])
                    for point in law.SUPPORT
                ]
            )[cells]
            for position, name in enumerate(("W", "A1", "C1", "L2", "A2", "C2", "Y"))
        }
    )
    frame["obs_weight"] = 1.0 / selection[cells]
    return frame


class DiscardSampleWeight(BaseEstimator):
    """Clone an estimator but deliberately drop ``sample_weight`` at fit time."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> DiscardSampleWeight:
        del sample_weight
        self.estimator_ = clone(self.estimator).fit(X, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.estimator_.predict_proba(X)


def _learners(configuration: str, *, cross_fit: bool) -> tuple[Any, Any, Any, Any]:
    if configuration in {"weighted_learners", "discard_learner_weights"}:
        # This deliberate control needs learners whose fitted values depend on sample
        # weights. Cross-fitting can leave a training complement without a support cell,
        # but an exact mechanism would make discarding learner weights observationally inert.
        learners: tuple[Any, Any, Any, Any] = (
            law.CellMeans(),
            law.CellMeans(),
            law.CellMeans(),
            law.CellMeans(),
        )
        if configuration == "discard_learner_weights":
            return tuple(DiscardSampleWeight(learner) for learner in learners)  # type: ignore[return-value]
        return learners
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    outcome: Any = law.CellMeans() if q_correct else DummyClassifier(strategy="prior")
    pseudo: Any = law.CellMeans() if q_correct else DummyRegressor(strategy="mean")
    if g_correct and cross_fit:
        # A training complement can miss a finite-support cell.  Use the law's exact
        # mechanism so "mechanism correct" stays correct on every realized split.
        treatment: Any = KnownDiscreteMechanism("treatment")
        censoring: Any = KnownDiscreteMechanism("censoring")
    else:
        treatment = law.CellMeans() if g_correct else DummyClassifier(strategy="prior")
        censoring = law.CellMeans() if g_correct else DummyClassifier(strategy="prior")
    return outcome, pseudo, treatment, censoring


def fit(
    frame: pd.DataFrame,
    configuration: str = "both_correct",
    *,
    cross_fit: bool,
    use_weights: bool = True,
) -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration, cross_fit=cross_fit)
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=5 if cross_fit else 1,
        learner_folds=2 if cross_fit else 5,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        weights="obs_weight" if use_weights else None,
    )


def untargeted(
    frame: pd.DataFrame,
    label: str,
    configuration: str,
    *,
    cross_fit: bool,
    folds: Any,
) -> float:
    """Return the matched observation-weighted plug-in without fluctuation."""
    outcome, pseudo, _, _ = _learners(configuration, cross_fit=cross_fit)
    first, second = _plan_arms(frame, label)
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    followed_two = (
        followed_one & (frame["C2"].to_numpy() == 1.0) & (frame["A2"].to_numpy() == second)
    )
    weights = frame["obs_weight"].to_numpy(dtype=float)
    baseline = frame[["W"]].to_numpy(dtype=float)
    history = np.column_stack([baseline[:, 0], l2])
    outcomes = frame["Y"].to_numpy()
    stitched = np.empty(len(frame), dtype=float)
    assignments = np.asarray(folds.assignment)
    for held_out in range(folds.n_folds):
        training = np.ones(len(frame), dtype=bool) if not cross_fit else assignments != held_out
        later = clone(outcome).fit(
            history[training & followed_two],
            outcomes[training & followed_two],
            sample_weight=weights[training & followed_two],
        )
        carried = np.asarray(later.predict_proba(history))[:, 1]
        earlier = clone(pseudo).fit(
            baseline[training & followed_one],
            carried[training & followed_one],
            sample_weight=weights[training & followed_one],
        )
        evaluated = np.ones(len(frame), dtype=bool) if not cross_fit else assignments == held_out
        stitched[evaluated] = earlier.predict(baseline[evaluated])
        if not cross_fit:
            break
    return float(np.average(stitched, weights=weights))


def _fit_replication(
    record: StudyRecord,
    cross_fit: bool,
    payload: tuple[str, str, int, int, int, int, str],
) -> list[dict[str, Any]]:
    property_name, suffix, replicate, n, requested, seed, configuration = payload
    probs = law.PROBS
    truth_source = law.TRUTH
    if property_name == "type_i_error":
        from tests.studies.ltmle_properties import NULL_PROBS, NULL_TRUTH

        probs = NULL_PROBS
        truth_source = {CONTRASTS["static"]: NULL_TRUTH}
    selection = LEARNER_SELECTION if property_name == "learner_weight_necessity" else SELECTION
    frame = sample(probs, n, seed, selection=selection)
    use_weights = property_name != "weight_necessity" or suffix == "weighted"
    fit_configuration = (
        "discard_learner_weights"
        if property_name == "learner_weight_necessity"
        and suffix == "discarded_learner_weight_control"
        else configuration
    )
    result = fit(frame, fit_configuration, cross_fit=cross_fit, use_weights=use_weights)
    names = WEIGHT_MEANS if property_name == "weight_necessity" else CONTRASTS
    labels = (
        ("static",)
        if property_name in {"type_i_error", "power", "learner_weight_necessity"}
        else tuple(names)
    )
    rows: list[dict[str, Any]] = []
    for label in labels:
        name = names[label]
        truth = float(truth_source[name])
        is_control = suffix in {
            "both_wrong",
            "omitted_weight_control",
            "discarded_learner_weight_control",
        } or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
        if property_name == "learner_weight_necessity":
            left, right = name[len("ate_regimen[") : -1].split(" vs ")
            plug_in = untargeted(
                frame,
                left,
                fit_configuration,
                cross_fit=cross_fit,
                folds=result.folds,
            ) - untargeted(
                frame,
                right,
                fit_configuration,
                cross_fit=cross_fit,
                folds=result.folds,
            )
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{label}__{suffix}",
                    role="control" if is_control else "positive",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=plug_in,
                    standard_error=float(result[name].std_error),
                    critical=float(norm.ppf(1.0 - record.margins.alpha / 2.0)),
                )
            )
        else:
            rows.append(
                replicate_row(
                    property_name=property_name,
                    cell=f"{label}__{suffix}",
                    role="control" if is_control else "positive",
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
            plug_in = untargeted(
                frame,
                left,
                configuration,
                cross_fit=cross_fit,
                folds=result.folds,
            ) - untargeted(
                frame,
                right,
                configuration,
                cross_fit=cross_fit,
                folds=result.folds,
            )
            critical = float(norm.ppf(1.0 - record.margins.alpha / 2.0))
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
                    critical=critical,
                )
            )
    return rows


def _payloads(record: StudyRecord) -> list[tuple[str, str, int, int, int, int, str]]:
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
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "mechanism_correct",
            ),
            ("weight_necessity", "weighted", NECESSITY_N, NECESSITY_REPLICATES, "both_correct"),
            (
                "weight_necessity",
                "omitted_weight_control",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "both_correct",
            ),
            (
                "learner_weight_necessity",
                "weighted_learners",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "weighted_learners",
            ),
            (
                "learner_weight_necessity",
                "discarded_learner_weight_control",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "weighted_learners",
            ),
        ]
    )
    return [
        (
            property_name,
            suffix,
            replicate,
            n,
            replicates,
            stream_seed(
                record,
                "property_sample",
                property_name,
                "paired"
                if property_name in {"weight_necessity", "learner_weight_necessity"}
                else suffix,
                replicate,
            ),
            configuration,
        )
        for property_name, suffix, n, replicates, configuration in specs
        for replicate in range(replicates)
    ]


def generate_property_rows(
    record: StudyRecord, *, cross_fit: bool, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    calls = [(record, cross_fit, payload) for payload in _payloads(record)]
    outcomes = map_parallel(_fit_replication, calls, n_jobs=n_jobs)
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    critical = float(norm.ppf(1.0 - record.margins.alpha / 2.0))
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                record,
                labels=tuple(CONTRASTS),
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=critical,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame, record: StudyRecord) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        record,
        extra_columns=(
            "targeting_displacement",
            "weight_displacement",
            "learner_weight_displacement",
            "alternative_truth",
            "alternative_bias_ci_lower",
            "alternative_bias_ci_upper",
            "alternative_bias_margin",
            "alternative_bias_equivalent",
        ),
        rate_labels=tuple(CONTRASTS),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=record.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=tuple(CONTRASTS),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    alternative_target_necessity_verdicts(
        summary,
        rows,
        record,
        family="weight_necessity",
        labels=tuple(WEIGHT_MEANS),
        arms=("weighted", "omitted_weight_control"),
        alternative_truths={
            label: float(law.functional(SELECTED_PROBS, name))
            for label, name in WEIGHT_MEANS.items()
        },
        column="weight_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    alternative_target_necessity_verdicts(
        summary,
        rows,
        record,
        family="learner_weight_necessity",
        labels=("static",),
        arms=("weighted_learners", "discarded_learner_weight_control"),
        alternative_truths={
            "static": float(law.functional(LEARNER_SELECTED_PROBS, CONTRASTS["static"]))
        },
        column="learner_weight_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    return finish(summary, rates)
