"""Registered evidence for repeated point-treatment cross-fitted TMLE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import G_BOUNDS, rows_from_result
from tests.studies.canonical_tmle import SCENARIO_ESTIMANDS, draw_for
from tests.studies.canonical_tmle import draw_from_seed as canonical_tmle_draw_from_seed
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS

PRIMARY_REPLICATES = 800
PRIMARY_N = 1000
SEED = 20260924
N_FOLDS = 5
REPEATS = 3

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
    "crossfit_overfitting": ("repeated_cvtmle", "in_sample_control"),
    "fold_repeat_stability": (
        "rowwise_three_draw_average",
        "one_fixed_split",
        "equal_fold_average",
    ),
    "repeat_aggregation": ("oracle_mean", "oracle_median", "stress_mean", "stress_median"),
    "repeat_variance": (
        "oracle_averaged_ic",
        "oracle_dml_mean",
        "stress_averaged_ic",
        "stress_dml_mean",
    ),
}

STUDY = StudyRecord(
    name="repeated point-treatment cross-fitted TMLE",
    slug="repeated-crossfit-tmle",
    artifacts=ROOT / "tests" / "canonical" / "repeated_crossfit",
    document="docs/technical-reference/method-evidence/repeated-cross-fitting.md",
    anchor="repeated-point-treatment-cross-fitted-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261025,
    margins=Margins(),
    implementation="cleverly-repeated-cvtmle",
    reference=None,
    modules=(
        "tests/studies/repeated_crossfit.py",
        "tests/studies/repeated_crossfit_properties.py",
        "tests/conftest.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.repeated_crossfit",
    properties_module="tests.studies.repeated_crossfit_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "repeats": REPEATS,
    "targeting_scheme": "pooled",
    "cv_evaluation": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "q_bounds": "sample outcome range",
    "repeat_aggregation": (
        "mean point estimate and row-aligned influence curve; ratios average on the log scale"
    ),
    "decision_study": {
        "replicates": 400,
        "n": 1000,
        "n_folds": 5,
        "repeats": 5,
        "seed": 20260925,
        "tail_probability": 0.04,
        "tail_treatment_probability": 0.12,
        "other_treatment_probability": 0.50,
        "tail_effect_increment": 8.0,
    },
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_for(STUDY, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    return canonical_tmle_draw_from_seed(scenario, n, seed)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """Fit the declared five-fold, three-draw stacked CV-TMLE."""
    outcome = (
        LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
        if scenario == "binary"
        else LinearRegression()
    )
    treatment = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=True,
            n_folds=N_FOLDS,
            repeats=REPEATS,
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


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema."""
    return rows_from_result(STUDY, fit_cleverly(frame, scenario), truth, scenario, replicate)


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every declared primary replication."""
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for records in outcomes for row in records])
    return rows.loc[:, list(REPLICATE_COLUMNS)]
