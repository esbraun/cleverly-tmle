"""Registered multi-arm selector C-TMLE evidence with an explicit empty R comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord

PRIMARY_REPLICATES = 800
PRIMARY_N = 1500
SEED = 20260829
SCENARIOS = {
    "multi_arm_selector_greedy": multi_arm_common.ALL_ESTIMANDS,
    "multi_arm_selector_ordered": multi_arm_common.ALL_ESTIMANDS,
    "multi_arm_selector_discrete": multi_arm_common.ALL_ESTIMANDS,
}

STUDY = StudyRecord(
    name="selector-based multi-arm C-TMLE",
    slug="canonical-multi-arm-ctmle-selector",
    artifacts=ROOT / "tests" / "canonical" / "multi_arm_ctmle_selector",
    document="docs/technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md",
    anchor="selector-based-multi-arm-c-tmle",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261003,
    margins=Margins(),
    implementation="cleverly-multi-arm-ctmle-selector",
    reference=None,
    publication_policy="reporting",
    modules=(
        "tests/studies/multi_arm_common.py",
        "tests/studies/canonical_multi_arm_ctmle_selector.py",
        "tests/studies/multi_arm_ctmle_selector_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_multi_arm_ctmle_selector",
    properties_module="tests.studies.multi_arm_ctmle_selector_properties",
    property_cells={
        "selector_necessity": ("collaborative", "empty_control"),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
    },
)

REFERENCE_METADATA: dict[str, str] = {}
CONFIGURATION = {
    "strategies": ["greedy", "ordered", "discrete"],
    "outcome_family": "binomial",
    "treatment_levels": list(multi_arm_common.LABELS),
    "reference": multi_arm_common.REFERENCE,
    "cross_fit": False,
    "selection_folds": 5,
    "selection_inner_folds": 2,
    "penalty": False,
    "comparison_scope": "none; R ctmle 0.1.2 is binary-treatment only",
}


def draw_from_seed(scenario: str, n: int, seed: int):  # type: ignore[no-untyped-def]
    return multi_arm_common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int):  # type: ignore[no-untyped-def]
    return multi_arm_common.draw_for(STUDY, scenario, n, replicate)


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
            estimands=("ey", "ate", "rr", "or"),
            ctmle_estimand="ate",
            reference=multi_arm_common.REFERENCE,
            simultaneous=False,
            g_bounds=multi_arm_common.G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: dict[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return multi_arm_common.cleverly_rows(STUDY, fit_cleverly, frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    rows = multi_arm_common.draw_and_fit(
        STUDY,
        fit_cleverly,
        replicates=replicates,
        n=n,
        n_jobs=n_jobs,
        include_samples=False,
    )
    assert isinstance(rows, pd.DataFrame)
    return rows
