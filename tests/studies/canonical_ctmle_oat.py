"""Outcome-adaptive C-TMLE evidence against archived tlverse ``ctmle3``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._typing import EstimandName
from cleverly.estimators import CTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_tmle import draw_for
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS

CTMLE3_COMMIT = "a4ea77b07747dfee9b2eecb9cbca88262e0559ea"
TMLE3_COMMIT = "3a610058cd89c17bb417c15fc891254388787f33"
SL3_COMMIT = "821ca890cb8701fdb59f823e28c6356e50d092bc"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 800
PRIMARY_N = 1000
SEED = 20240823
G_BOUNDS = (0.025, 0.975)

SCENARIO_ESTIMANDS: Mapping[str, tuple[EstimandName, ...]] = {
    "binary": ("ey0", "ey1", "ate", "rr", "or"),
}

PROPERTY_CELLS = {
    "robustness_contract": ("outcome_correct", "outcome_wrong"),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "interval_calibration": ("correctly_specified",),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
    "crossfit_overfitting": ("cross_fitted_oat", "in_sample_control"),
    "generated_design": ("estimated", "oracle_design"),
}

STUDY = StudyRecord(
    name="outcome-adaptive point-treatment C-TMLE",
    slug="canonical-ctmle-oat",
    artifacts=ROOT / "tests" / "canonical" / "ctmle3_oat",
    document="docs/technical-reference/method-evidence/outcome-adaptive-point-treatment-c-tmle.md",
    anchor="outcome-adaptive-point-treatment-c-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-ctmle-oat",
    reference="tlverse-ctmle3-oat",
    modules=(
        "tests/studies/canonical_ctmle_oat.py",
        "tests/studies/ctmle_oat_properties.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ctmle_oat",
    properties_module="tests.studies.ctmle_oat_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "ctmle3_commit": CTMLE3_COMMIT,
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "strategy": "oat",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "comparison_scope": "binary two-arm treatment-specific means and derived contrasts",
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_for(STUDY, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    from tests.studies.canonical_tmle import sample_continuous, scenario_dgp, truth_for

    dgp = scenario_dgp(scenario)
    if scenario == "continuous":
        return sample_continuous(dgp, n, seed)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    binary = scenario == "binary"
    outcome = (
        LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs") if binary else LinearRegression()
    )
    return (
        CTMLE(
            strategy="oat",
            outcome_learner=outcome,
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            cross_fit=False,
            estimands=SCENARIO_ESTIMANDS[scenario],
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario)
    rows: list[dict[str, Any]] = []
    for name in SCENARIO_ESTIMANDS[scenario]:
        estimate = result[name]
        reference = float(truth[name])
        low, high = estimate.ci
        ratio = estimate.scale == "ratio"
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": (
                    float(estimate.log_psi)
                    if ratio and estimate.log_psi is not None
                    else float(estimate.psi)
                ),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if ratio else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return sample, truth_row, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, records in outcomes for row in records])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]
