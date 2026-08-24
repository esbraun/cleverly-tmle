"""Canonical selector-based C-TMLE evidence against R ``ctmle``.

The comparison is deliberately bounded to the common unpenalized construction.  Cleverly's
published penalty is validated independently because it follows the paper equation rather than
the implementation-specific adjustment in R ``ctmle``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly._typing import EstimandName
from cleverly.estimators import CTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

CTMLE_COMMIT = "18de559f47dc1286617350a0668391e80e1dbf7c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 800
PRIMARY_N = 2000
SEED = 20240822
G_BOUNDS = (0.025, 0.975)

SCENARIO_ESTIMANDS: Mapping[str, tuple[EstimandName, ...]] = {
    "binary_greedy": ("ate",),
    "binary_ordered": ("ate",),
    "binary_discrete": ("ate",),
}

PROPERTY_CELLS = {
    "double_robustness": (
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
    ),
    "selector_necessity": ("collaborative", "empty_control"),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "interval_calibration": ("correctly_specified",),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
}

STUDY = StudyRecord(
    name="selector-based point-treatment C-TMLE",
    slug="canonical-ctmle-selector",
    artifacts=ROOT / "tests" / "canonical" / "ctmle_selector",
    document="docs/technical-reference/method-evidence.md",
    anchor="selector-based-point-treatment-c-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-ctmle-selector",
    reference="r-ctmle",
    modules=(
        "tests/studies/canonical_ctmle_selector.py",
        "tests/studies/ctmle_selector_properties.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ctmle_selector",
    properties_module="tests.studies.ctmle_selector_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "ctmle_commit": CTMLE_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "cross_fit": False,
    "simultaneous_intervals": False,
    "selection_folds": 5,
    "selection_inner_folds": 2,
    "penalty": False,
    "g_bounds": list(G_BOUNDS),
    "comparison_scope": "binary ATE; continuous outcomes are assessed independently",
}


#: All three scenarios are the same binary law.  What differs between them is the selector
#: strategy, not the process -- the distinct names exist so each strategy draws its own
#: ``replicate_seed`` stream rather than three verdicts being one draw reported three times.
LAW = "binary"


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw directly from a supplied seed for the manifest-seed audit."""
    from tests.studies.canonical_tmle import scenario_dgp, truth_for

    del scenario
    dgp = scenario_dgp(LAW)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


def _strategy(scenario: str) -> tuple[str, dict[str, Any]]:
    covariates = ("W1", "W2", "W3")
    if scenario.endswith("ordered"):
        return "ordered", {"ordering": covariates}
    if scenario.endswith("discrete"):
        return "discrete", {"candidates": ((), ("W1",), ("W1", "W2"), covariates)}
    return "greedy", {}


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    strategy, options = _strategy(scenario)
    return (
        CTMLE(
            strategy=strategy,
            outcome_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            cross_fit=False,
            selection_folds=5,
            selection_inner_folds=2,
            penalty=False,
            estimands=("ate",),
            ctmle_estimand="ate",
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


def _rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    n: int,
) -> list[dict[str, Any]]:
    estimate = result["ate"]
    reference = float(truth["ate"])
    low, high = estimate.ci
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": "ate",
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
        }
    ]


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return _rows_from_result(fit_cleverly(frame, scenario), truth, scenario, replicate, len(frame))


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    sample = frame.copy()
    # The split this fit actually scored its candidates over, taken off the fit rather
    # than rebuilt here from the same rule.  R selects against the same partition, so a
    # reconstruction that silently stopped matching would move the reference's answer
    # while every gate in the study kept passing.
    sample.insert(0, "selection_fold", result.extra["ctmle"].folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return sample, truth_row, _rows_from_result(result, truth, scenario, replicate, len(frame))


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
