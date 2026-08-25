r"""Registered protocol for the canonical DR-TMLE comparison.

The law is the complete-data binary law in Benkeser et al. (2017).  The two
implementations receive the same realized rows and fold assignment.  Each side fits the
same declared GLM nuisance class; the comparison is about the corrected construction, not
about two unrelated learner libraries.

This module is intentionally importable without R or Docker.  The container is used only by
``tests/canonical/drtmle/regenerate.py`` when the registered evidence is regenerated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cleverly.data import CausalData
from cleverly.estimators import DRTMLE
from cleverly.learners.crossfit import Folds, check_integrity
from cleverly.utils.bounds import expit
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import replicate_seed

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:"
    "fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 3000
SEED = 20260824
G_BOUNDS = (0.01, 0.99)
N_FOLDS = 10
#: Stricter than the canonical package's ``1 / n`` default at the registered n = 3000,
#: while remaining well below the theorem's n^-1/2 scale.
SCORE_AUDIT_TOLERANCE = 1e-4
SCENARIOS = ("outcome_correct", "treatment_correct", "both_correct")
ESTIMANDS = ("ey0", "ey1", "ate")
FIT_DIAGNOSTIC_COLUMNS = (
    "implementation",
    "scenario",
    "replicate",
    "n",
    "score_max",
    "score_passed",
    "solver_passed",
    "bound_active",
)

STUDY = StudyRecord(
    name="DR-TMLE for binary complete data",
    slug="canonical-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "drtmle",
    document="docs/technical-reference/method-evidence.md",
    anchor="canonical-dr-tmle",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260826,
    margins=Margins(),
    implementation="cleverly",
    reference="drtmle-r",
    publication_policy="reporting",
    extra_artifacts=("fit-diagnostics.csv",),
    modules=(
        "tests/studies/canonical_drtmle.py",
        "tests/studies/drtmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_drtmle",
    properties_module="tests.studies.drtmle_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "treatment_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_1500", "n_4500"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
    },
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "source_law": "Benkeser et al. (2017), Section 4",
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "targeting_scheme": "pooled",
    "reduced_crossfit": "pooled",
    "reduction": "univariate",
    "guard": ["Q", "g"],
    "update_order": "drtmle",
    "qsteps": 2,
    "max_iter": 100,
    "score_audit_tolerance": SCORE_AUDIT_TOLERANCE,
    "g_bounds": list(G_BOUNDS),
    "nuisance_models": {
        "correct": "unpenalized logistic GLM with W1:W2",
        "misspecified": "unpenalized main-effects logistic GLM",
        "reduced_Q": "Gaussian GLM",
        "reduced_g": "binomial GLM",
    },
}


class ColumnLogistic(BaseEstimator, ClassifierMixin):
    """Unpenalized logistic regression on a fixed subset of design columns."""

    def __init__(self, columns: Sequence[int] | None = None) -> None:
        self.columns = columns

    def _select(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if self.columns is None:
            return values
        return values[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ColumnLogistic:
        self.model_ = LogisticRegression(
            C=np.inf,
            max_iter=5000,
            solver="newton-cholesky",
            tol=1e-10,
            random_state=0,
        ).fit(self._select(design), target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._select(design)), dtype=float)


class FixedFoldDRTMLE(DRTMLE):
    """Study-only DR-TMLE whose outer fold assignment is supplied by the sample."""

    def __init__(self, assignment: np.ndarray, **kwargs: Any) -> None:
        self._assignment = np.asarray(assignment, dtype=np.int64)
        super().__init__(**kwargs)

    def _folds(self, data: CausalData, seed: int | None = None) -> Folds:
        del seed
        if data.n != self._assignment.size:
            raise ValueError(
                f"fold assignment has {self._assignment.size} rows for a {data.n}-row fit"
            )
        folds = Folds(self._assignment.copy(), int(self._assignment.max()) + 1)
        check_integrity(folds, cluster=data.cluster)
        return folds


def _linear_predictor(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return -w1 + 2.0 * w1 * w2


def truth() -> dict[str, float]:
    """Independent quadrature truth for the paper law."""

    def arm_mean(arm: float) -> float:
        def integrate(w1: float) -> float:
            return 0.5 * (
                float(expit(0.2 * arm - w1))
                + float(expit(0.2 * arm + w1))
            )

        value, error = quad(integrate, -2.0, 2.0, epsabs=1e-13, epsrel=1e-13, limit=200)
        if error > 1e-11:
            raise RuntimeError(f"paper-law quadrature error {error:g} exceeds its audit bar")
        return value / 4.0

    ey0, ey1 = arm_mean(0.0), arm_mean(1.0)
    return {"ey0": ey0, "ey1": ey1, "ate": ey1 - ey0}


def fixed_folds(treatment: np.ndarray, seed: int) -> np.ndarray:
    """The exact zero-based fold vector shared with R."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    assignment = np.empty(len(treatment), dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(treatment)), treatment)):
        assignment[test] = fold
    return assignment


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in (*SCENARIOS, "both_wrong"):
        raise KeyError(scenario)
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-2.0, 2.0, size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    linear = _linear_predictor(w1, w2)
    a = rng.binomial(1, expit(linear)).astype(float)
    y = rng.binomial(1, expit(0.2 * a + linear)).astype(float)
    folds = fixed_folds(a, seed + 1)
    return (
        pd.DataFrame(
            {
                "Y": y,
                "A": a,
                "W1": w1,
                "W2": w2,
                "W12": w1 * w2,
                "fold": folds,
            }
        ),
        truth(),
    )


def draw_for(
    record: StudyRecord, scenario: str, n: int, replicate: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(record, scenario, replicate))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_for(STUDY, scenario, n, replicate)


def _learners(scenario: str) -> tuple[ColumnLogistic, ColumnLogistic]:
    outcome_correct = scenario in {"outcome_correct", "both_correct"}
    treatment_correct = scenario in {"treatment_correct", "both_correct"}
    # Outcome design is [A, W1, W2, W12]; treatment design is [W1, W2, W12].
    outcome_columns = None if outcome_correct else (0, 1, 2)
    treatment_columns = None if treatment_correct else (0, 1)
    return ColumnLogistic(outcome_columns), ColumnLogistic(treatment_columns)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    outcome, treatment = _learners(scenario)
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    result = (
        FixedFoldDRTMLE(
            assignment,
            outcome_learner=outcome,
            treatment_learner=treatment,
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=ColumnLogistic(),
            cross_fit=True,
            n_folds=N_FOLDS,
            estimands=ESTIMANDS,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            guard=("Q", "g"),
            reduction="univariate",
            reduced_crossfit="pooled",
            update_order="drtmle",
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W12"],
        )
        .single()
    )
    score_check = result.diagnostics.score_equations()
    worst_score = max(abs(float(row.score)) for row in score_check.rows)
    if not np.isfinite(worst_score):
        raise RuntimeError("DR-TMLE empirical score audit is non-finite")
    raw = np.asarray(result.repeats[0].nuisance.propensity.values, dtype=float)
    bounded = result.repeats[0].nuisance.bounded_propensity(G_BOUNDS)
    if not np.array_equal(raw, bounded):
        raise RuntimeError("the propensity bound activated in the canonical comparison")
    return result


def cleverly_rows(
    frame: pd.DataFrame,
    reference_truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    *,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario) if result is None else result
    score_check = result.diagnostics.score_equations()
    score_max = max(abs(float(row.score)) for row in score_check.rows)
    initial = result.repeats[0].nuisance.outcome.arms
    initial_values = {
        "ey0": float(np.mean(initial[0.0])),
        "ey1": float(np.mean(initial[1.0])),
        "ate": float(np.mean(initial[1.0] - initial[0.0])),
    }
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS:
        estimate = result.estimates[name]
        target = float(reference_truth[name])
        low, high = estimate.ci
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": target,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= target <= high),
                "initial_estimate": initial_values[name],
                "score_max": score_max,
                "score_passed": score_max <= SCORE_AUDIT_TOLERANCE,
                "solver_passed": _solver_passed(score_check),
                "bound_active": False,
            }
        )
    return rows


def _solver_passed(score_check: Any) -> bool:
    """Whether every actual fluctuation row completed without a recorded failure."""
    return all(not row.failure for row in score_check.rows if row.kind == "fluctuation")


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference_truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    payload_frame = frame.copy()
    nuisance = result.repeats[0].nuisance
    payload_frame["qn0"] = nuisance.outcome.arms[0.0]
    payload_frame["qn1"] = nuisance.outcome.arms[1.0]
    payload_frame["gn1"] = nuisance.propensity.arm(1.0)
    payload_frame.insert(0, "replicate", replicate)
    payload_frame.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in reference_truth.items()},
    }
    return payload_frame, truth_row, cleverly_rows(
        frame, reference_truth, scenario, replicate, result=result
    )


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in SCENARIOS
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([frame for frame, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([item for _, item, _ in outcomes])
    rows = pd.DataFrame([row for _, _, fitted in outcomes for row in fitted])
    return samples, truths, rows


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One fit-health row per implementation, scenario, and replication."""
    diagnostics = (
        rows.loc[:, list(FIT_DIAGNOSTIC_COLUMNS)]
        .drop_duplicates()
        .sort_values(["scenario", "replicate", "implementation"], ignore_index=True)
    )
    # drtmle exposes no convergence flag.  Native completion with finite output is its
    # solver-success signal; normalize old checkpoint rows that predate that distinction.
    diagnostics.loc[
        diagnostics["implementation"].eq(STUDY.reference), "solver_passed"
    ] = True
    counts = diagnostics.groupby(["implementation", "scenario", "replicate"]).size()
    if not counts.eq(1).all():
        raise ValueError("fit diagnostics are not unique by implementation/scenario/replicate")
    return {"fit-diagnostics.csv": diagnostics}


def scientific_failures(
    artifacts: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    diagnostics = artifacts["fit-diagnostics.csv"]
    failed = diagnostics.loc[
        ~diagnostics["score_passed"].astype(bool)
        | ~diagnostics["solver_passed"].astype(bool)
    ]
    return {"fit health": failed}
