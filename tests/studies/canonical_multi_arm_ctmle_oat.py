"""Registered multi-arm outcome-adaptive C-TMLE comparison against R ``ctmle3``.

The primary fit is not cross-fitted and ``strategy="oat"`` draws no selection folds, so no
split is drawn here at all.  ``stratify_folds="none"`` is still passed, because a
declaration the study makes is what a reader can check against the manifest, and because
the property cells reach the same keyword through a cross-fitted fit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

CTMLE3_COMMIT = "a4ea77b07747dfee9b2eecb9cbca88262e0559ea"
TMLE3_COMMIT = "3a610058cd89c17bb417c15fc891254388787f33"
SL3_COMMIT = "821ca890cb8701fdb59f823e28c6356e50d092bc"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 1500
SEED = 20260828
SCENARIO = "multi_arm_binary_oat"

#: What every split this study draws is held to.  ``"none"`` is passed explicitly rather
#: than left to the estimator's default, so this study's declaration does not move when that
#: default does.
STRATIFY_FOLDS = "none"

STUDY = StudyRecord(
    name="outcome-adaptive multi-arm C-TMLE",
    slug="canonical-multi-arm-ctmle-oat",
    artifacts=ROOT / "tests" / "canonical" / "multi_arm_ctmle3_oat",
    document="docs/technical-reference/method-evidence/outcome-adaptive-multi-arm-c-tmle.md",
    anchor="outcome-adaptive-multi-arm-c-tmle",
    scenarios={SCENARIO: multi_arm_common.ALL_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261002,
    margins=Margins(),
    implementation="cleverly-multi-arm-ctmle-oat",
    reference="ctmle3-multi-arm-oat",
    publication_policy="reporting",
    modules=(
        "tests/studies/multi_arm_common.py",
        "tests/studies/canonical_multi_arm_ctmle_oat.py",
        "tests/studies/multi_arm_ctmle_oat_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_multi_arm_ctmle_oat",
    properties_module="tests.studies.multi_arm_ctmle_oat_properties",
    property_cells={
        "robustness_contract": ("outcome_correct", "outcome_wrong"),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "generated_design": ("oracle_design", "estimated"),
    },
)

REFERENCE_METADATA = {
    "ctmle3_commit": CTMLE3_COMMIT,
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}
CONFIGURATION = {
    "strategy": "oat",
    "outcome_family": "binomial",
    "treatment_levels": list(multi_arm_common.LABELS),
    "reference": multi_arm_common.REFERENCE,
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(multi_arm_common.G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "q_bounds": (
        "none anywhere in this row; the outcome is binary on the primary law and on every "
        "property law, so the scaler is already the identity"
    ),
    "folds": (
        "none on the primary: it is not cross-fitted and the outcome-adaptive strategy "
        "draws no selection folds. The property cells draw unstratified five-fold outer "
        "assignments"
    ),
    "initial_treatment_model": "empirical arm probabilities (matched to sl3 Lrnr_mean)",
}


def draw_from_seed(scenario: str, n: int, seed: int):  # type: ignore[no-untyped-def]
    return multi_arm_common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int):  # type: ignore[no-untyped-def]
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    del scenario
    return (
        CTMLE(
            strategy="oat",
            outcome_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            treatment_learner=DummyClassifier(strategy="prior"),
            cross_fit=False,
            estimands=("ey", "ate", "rr", "or"),
            reference=multi_arm_common.REFERENCE,
            simultaneous=False,
            g_bounds=multi_arm_common.G_BOUNDS,
            # The outcome is binary here, so no ``q_bounds``: the scaler is already the
            # identity and ``TMLE._scaler`` refuses a second declaration of it.  The
            # split keyword is inert on this fit, which draws none, and is passed so the
            # study declares one rule rather than two.
            stratify_folds=STRATIFY_FOLDS,
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
