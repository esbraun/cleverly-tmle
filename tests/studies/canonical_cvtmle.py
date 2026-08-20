"""Canonical stacked CV-TMLE evidence against pinned R ``tmle3``.

The ordinary-TMLE study supplies the laws and exact truths.  This study changes the
estimator construction: both implementations use the same ten outer folds, out-of-fold
GLM nuisance predictions, one common update over the stacked validation rows, and a
whole-sample plug-in evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_tmle import SCENARIO_ESTIMANDS, draw_for
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1000
SEED = 20240820
N_FOLDS = 10
G_BOUNDS = (0.025, 0.975)

PROPERTY_CELLS = {
    "double_robustness": (
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
    ),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "interval_calibration": ("correctly_specified",),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
    "crossfit_overfitting": ("stacked_cvtmle", "in_sample_control"),
}

STUDY = StudyRecord(
    name="stacked point-treatment CV-TMLE",
    slug="canonical-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle3_cvtmle",
    document="docs/technical-reference/method-evidence.md",
    anchor="stacked-point-treatment-cv-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-stacked-cvtmle",
    reference="tmle3-cvtmle",
    incomparable_se=frozenset({"paf"}),
    modules=(
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/stacked_cvtmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_cvtmle",
    properties_module="tests.studies.stacked_cvtmle_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "targeting_scheme": "pooled",
    "cv_evaluation": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "q_bounds": "sample outcome range",
    "folds": "identical treatment-stratified assignments supplied to both implementations",
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from *this* study's declared seed.

    The laws come from the ordinary-TMLE study; the samples do not.  This row is separate
    evidence, and it would not be if it re-used another study's draws.
    """
    return draw_for(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """The public stacked-validation construction matched to R ``cvtmle=TRUE``."""
    binary = set(frame["Y"].dropna().unique()).issubset({0, 1})
    outcome = (
        LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs") if binary else LinearRegression()
    )
    treatment = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    scenario = "binary" if binary else "continuous"
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=True,
            n_folds=N_FOLDS,
            targeting_scheme="pooled",
            cv_evaluation=False,
            estimands=SCENARIO_ESTIMANDS[scenario],
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def _rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, estimate in result.estimates.items():
        reference = float(truth[name])
        low, high = estimate.ci
        inference_estimate = (
            float(estimate.log_psi)
            if estimate.scale == "ratio" and estimate.log_psi is not None
            else float(estimate.psi)
        )
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": inference_estimate,
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if estimate.scale == "ratio" else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return _rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    payload_frame = frame.copy()
    payload_frame.insert(0, "fold", result.nuisance.folds.assignment)
    payload_frame.insert(0, "replicate", replicate)
    payload_frame.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return payload_frame, truth_row, _rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([frame for frame, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, records in outcomes for row in records])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]
