"""Registered ordinary multi-arm TMLE comparison against pinned R ``tmle3``."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 1500
SEED = 20260827
SCENARIO = "multi_arm_binary"

STUDY = StudyRecord(
    name="ordinary multi-arm point-treatment TMLE",
    slug="canonical-multi-arm-tmle",
    artifacts=ROOT / "tests" / "canonical" / "multi_arm_tmle3",
    document="docs/technical-reference/method-evidence/ordinary-multi-arm-tmle.md",
    anchor="ordinary-multi-arm-tmle",
    scenarios={SCENARIO: multi_arm_common.ALL_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261001,
    margins=Margins(),
    implementation="cleverly-multi-arm-tmle",
    reference="tmle3-multi-arm",
    modules=(
        "tests/studies/multi_arm_common.py",
        "tests/studies/canonical_multi_arm_tmle.py",
        "tests/studies/multi_arm_tmle_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_multi_arm_tmle",
    properties_module="tests.studies.multi_arm_tmle_properties",
    property_cells={
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
    },
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}
CONFIGURATION = {
    "outcome_family": "binomial",
    "treatment_levels": list(multi_arm_common.LABELS),
    "reference": multi_arm_common.REFERENCE,
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(multi_arm_common.G_BOUNDS),
    "outcome_model": "intercept-only (matched to sl3 Lrnr_mean)",
    "treatment_model": "multinomial logistic regression (matched to pinned nnet adapter)",
}


def draw_from_seed(scenario: str, n: int, seed: int):  # type: ignore[no-untyped-def]
    return multi_arm_common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int):  # type: ignore[no-untyped-def]
    return multi_arm_common.draw_for(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    del scenario
    return (
        TMLE(
            outcome_learner=DummyClassifier(strategy="prior"),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            cross_fit=False,
            estimands=("ey", "ate", "rr", "or"),
            reference=multi_arm_common.REFERENCE,
            simultaneous=False,
            g_bounds=multi_arm_common.G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
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


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS):
    return multi_arm_common.draw_and_fit(
        STUDY, fit_cleverly, replicates=replicates, n=n, n_jobs=n_jobs
    )
