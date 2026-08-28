"""Shared law, fits, rows, and controls for categorical longitudinal evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as binary_law
from tests import discrete_law_longitudinal_multivalue as law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import QuasiBinomialGLM
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

G_BOUNDS = (1e-8, 1.0)
N_FOLDS = 5
SCENARIO = "categorical_end_of_study"
REGIMENS = law.REGIMEN_SPEC
REFERENCE = law.REFERENCE
ESTIMANDS = law.NAMES
LEVELS = tuple(sorted(law.ARM_LABELS))

STATIC_NAME = "ate_regimen[standard vs low]"
DYNAMIC_NAME = "ate_regimen[respond vs low]"
CONTRASTS = {"static": STATIC_NAME, "dynamic": DYNAMIC_NAME}

MUTATED_REGIMENS = {
    **REGIMENS,
    "respond": (
        "standard",
        lambda history: np.where(history["L2"] == 1, "low", "high"),
    ),
}


def property_cells(*, cross_fit: bool) -> dict[str, tuple[str, ...]]:
    """The shared property declaration, plus the cross-fit-specific pair."""
    cells = {
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
            f"{label}__n_{size}" for label in CONTRASTS for size in (500, 2000, 8000)
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
        "type_i_error": ("dynamic__sharp_null",),
        "power": ("dynamic__alternative",),
        "targeting_necessity": tuple(
            f"{label}__{arm}" for label in CONTRASTS for arm in ("targeted", "untargeted")
        ),
        "categorical_probability_necessity": (
            "third_arm__assigned_probability",
            "third_arm__binary_complement",
        ),
        "rule_necessity": ("dynamic__declared_rule", "dynamic__reversed_rule"),
    }
    if cross_fit:
        cells["crossfit_overfitting"] = (
            "cross_fitted_categorical_ltmle",
            "in_sample_control",
        )
    return cells


class BinaryComplementMechanism(BaseEstimator):
    """A deliberate mutation that replaces the third arm by a binary complement."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> BinaryComplementMechanism:
        self.base_ = KnownCategoricalMechanism().fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.base_.classes_, dtype=float)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        probabilities = np.asarray(self.base_.predict_proba(X), dtype=float)
        high = LEVELS.index("high")
        standard = LEVELS.index("standard")
        probabilities[:, standard] = 1.0 - probabilities[:, high]
        return probabilities


class KnownCategoricalMechanism(BaseEstimator):
    """The exact three-arm mechanism, used when continuous noise makes cells unique."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownCategoricalMechanism:
        del X, y, sample_weight
        self.classes_ = np.arange(len(LEVELS), dtype=float)
        return self

    @staticmethod
    def _in_report_order(raw: np.ndarray) -> np.ndarray:
        return np.column_stack([raw[:, law.ARM_LABELS.index(label)] for label in LEVELS])

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w = matrix[:, 0].astype(int)
        if matrix.shape[1] in (1, 2):
            return self._in_report_order(law.G1[w])
        if matrix.shape[1] not in (4, 5):
            raise ValueError(f"unexpected categorical mechanism design {matrix.shape}")
        offset = 1 if matrix.shape[1] == 5 else 0
        l2 = matrix[:, 1 + offset].astype(int)
        indicators = matrix[:, 2 + offset : 4 + offset]
        previous = np.where(indicators[:, 0] == 1.0, 1, np.where(indicators[:, 1] == 1.0, 2, 0))
        raw_previous = np.array(
            [law.ARM_LABELS.index(LEVELS[int(code)]) for code in previous], dtype=int
        )
        return self._in_report_order(law.G2[w, raw_previous, l2])


def _learners(configuration: str) -> tuple[Any, Any, Any]:
    if configuration in {"overfit_crossfit", "overfit_control"}:
        return (
            DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
            DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            KnownCategoricalMechanism(),
        )
    if configuration == "primary":
        return QuasiBinomialGLM(), QuasiBinomialGLM(), KnownCategoricalMechanism()
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    outcome = binary_law.CellMeans() if q_correct else DummyClassifier(strategy="prior")
    pseudo = binary_law.CellMeans() if q_correct else DummyRegressor(strategy="mean")
    if configuration == "binary_complement":
        mechanism: Any = BinaryComplementMechanism()
    else:
        mechanism = KnownCategoricalMechanism() if g_correct else DummyClassifier(strategy="prior")
    return outcome, pseudo, mechanism


def fit(
    frame: pd.DataFrame,
    *,
    cross_fit: bool,
    configuration: str = "both_correct",
    mutate_rule: bool = False,
) -> Any:
    """Fit either registered construction through one result-identical adapter."""
    outcome, pseudo, treatment = _learners(configuration)
    use_cross_fit = cross_fit and configuration != "overfit_control"
    baseline = ["W", *(("U",) if "U" in frame else ())]
    return LTMLE(
        MUTATED_REGIMENS if mutate_rule else REGIMENS,
        reference=REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        n_folds=N_FOLDS if use_cross_fit else 1,
        learner_folds=2,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=baseline,
        time_varying=[[], ["L2"]],
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return law.sample(law.PROBS, n, seed), dict(law.TRUTH)


def draw_for(
    record: StudyRecord, scenario: str, n: int, replicate: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(record, scenario, replicate))


def result_rows(
    record: StudyRecord,
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    initials = {
        f"ey_regimen[{label}]": float(np.mean(result.fits[label].steps[0].initial))
        for label in REGIMENS
    }
    for label in REGIMENS:
        if label == REFERENCE:
            continue
        initials[f"ate_regimen[{label} vs {REFERENCE}]"] = (
            initials[f"ey_regimen[{label}]"] - initials[f"ey_regimen[{REFERENCE}]"]
        )
    rows = []
    for name in ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": record.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": initials[name],
            }
        )
    return rows


def _replicate(
    record: StudyRecord, cross_fit: bool, scenario: str, replicate: int, n: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frame, truth = draw_for(record, scenario, n, replicate)
    result = fit(frame, cross_fit=cross_fit, configuration="primary")
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, result_rows(record, result, truth, scenario, replicate)


def draw_and_fit(
    record: StudyRecord,
    *,
    cross_fit: bool,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        (record, cross_fit, scenario, replicate, n)
        for scenario in record.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]


def _plan_labels(frame: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    node1, node2 = law.REGIMEN_ARMS[label]
    l2 = frame["L2"].to_numpy(dtype=int)
    first_code = np.full(len(frame), int(node1), dtype=int)
    second_code = (
        np.full(len(frame), int(node2), dtype=int)
        if np.ndim(node2) == 0
        else np.asarray(node2, dtype=int)[l2]
    )
    labels = np.asarray(law.ARM_LABELS, dtype=object)
    return labels[first_code], labels[second_code]


def _predict(learner: Any, design: np.ndarray, *, classification: bool) -> np.ndarray:
    if classification:
        probabilities = np.asarray(learner.predict_proba(design), dtype=float)
        classes = np.asarray(learner.classes_)
        matches = np.flatnonzero(classes == 1)
        if not matches.size:
            return np.zeros(len(design), dtype=float)
        return probabilities[:, int(matches[0])]
    return np.asarray(learner.predict(design), dtype=float)


def untargeted(
    frame: pd.DataFrame,
    label: str,
    configuration: str,
    folds: Any,
) -> float:
    """The same fold-specific sequential plug-in with every targeting update removed."""
    outcome, pseudo, _ = _learners(configuration)
    first, second = _plan_labels(frame, label)
    followed_one = frame["A1"].to_numpy() == first
    followed_two = followed_one & (frame["A2"].to_numpy() == second)
    baseline_names = ["W", *(("U",) if "U" in frame else ())]
    baseline = frame[baseline_names].to_numpy(dtype=float)
    history = np.column_stack([baseline, frame["L2"].to_numpy(dtype=float)])
    stitched = np.empty(len(frame), dtype=float)
    for held_out, (training_rows, evaluated_rows) in enumerate(folds):
        del held_out
        training = np.zeros(len(frame), dtype=bool)
        training[training_rows] = True
        if not training_rows.size:
            training[:] = True
        later = clone(outcome).fit(
            history[training & followed_two], frame["Y"].to_numpy()[training & followed_two]
        )
        carried = _predict(later, history, classification=True)
        earlier = clone(pseudo).fit(
            baseline[training & followed_one], carried[training & followed_one]
        )
        stitched[evaluated_rows] = _predict(earlier, baseline[evaluated_rows], classification=False)
    return float(np.mean(stitched))


def empty_initial() -> float:
    """Schema value for controls that are not complete estimator fits."""
    return math.nan
