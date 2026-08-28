"""Registered cross-fitted categorical longitudinal TMLE evidence study."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import categorical_longitudinal_common as common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord

LMTP_VERSION = "1.5.4"
LMTP_SOURCE_COMMIT = "f04a2b47f46debc515ce4ae778e05ebfde922c44"
LMTP_TARBALL_SHA256 = "fd49d9f291d4ddabb78c36d152b25aaa234a7204b645b9921f998c152e3d2ba5"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 4_000
PRIMARY_N = 2_000
SEED = 20260828
RESAMPLING_SEED = 2026082801

STUDY = StudyRecord(
    name="cross-fitted categorical longitudinal TMLE",
    slug="canonical-categorical-ltmle-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "categorical_ltmle_crossfit",
    document=(
        "docs/technical-reference/method-evidence/cross-fitted-categorical-longitudinal-tmle.md"
    ),
    anchor="cross-fitted-categorical-longitudinal-tmle",
    scenarios={common.SCENARIO: common.ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-categorical-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_categorical_ltmle_crossfit.py",
        "tests/studies/categorical_ltmle_crossfit_properties.py",
        "tests/studies/categorical_longitudinal_properties.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/categorical_ltmle_runner.R",
    ),
    runner_module="tests.studies.canonical_categorical_ltmle_crossfit",
    properties_module="tests.studies.categorical_ltmle_crossfit_properties",
    property_cells=common.property_cells(cross_fit=True),
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "cross_fitted",
    "outcome_kind": "end_of_study",
    "cross_fit": True,
    "n_folds": common.N_FOLDS,
    "learner_folds": 2,
    "fold_evaluation": "training-fold recursion evaluated on its held-out rows",
    "treatment_levels": list(common.LEVELS),
    "regimens": list(common.REGIMENS),
    "reference": common.REFERENCE,
    "simultaneous_intervals": False,
    "g_bounds": list(common.G_BOUNDS),
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_for(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return common.fit(frame, cross_fit=True, configuration="primary")


def cleverly_rows(
    frame: pd.DataFrame, truth: dict[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    return common.result_rows(STUDY, fit_cleverly(frame), truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return common.draw_and_fit(STUDY, cross_fit=True, replicates=replicates, n=n, n_jobs=n_jobs)
