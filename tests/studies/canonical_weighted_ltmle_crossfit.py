"""Cross-fitted weighted end-of-study longitudinal TMLE evidence study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import weighted_longitudinal_common as common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.weighted_longitudinal_properties_common import property_cells

LMTP_VERSION = "1.5.4"
SEED = 20262000
PRIMARY_REPLICATES = common.PRIMARY_REPLICATES
PRIMARY_N = common.PRIMARY_N

STUDY = StudyRecord(
    name="cross-fitted weighted end-of-study longitudinal TMLE",
    slug="weighted-ltmle-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "weighted_lmtp_ltmle",
    document=(
        "docs/technical-reference/method-evidence/"
        "cross-fitted-weighted-end-of-study-longitudinal-tmle.md"
    ),
    anchor="cross-fitted-weighted-end-of-study-longitudinal-tmle",
    scenarios={common.SCENARIO: common.ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-weighted-ltmle",
    reference="lmtp-weighted",
    publication_policy="reporting",
    modules=(
        "tests/studies/canonical_weighted_ltmle_crossfit.py",
        "tests/studies/weighted_longitudinal_common.py",
        "tests/studies/weighted_ltmle_crossfit_properties.py",
        "tests/studies/weighted_longitudinal_properties_common.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit/audit.py",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_weighted_glm_adapter.R",
        "tests/canonical/ltmle_regimen_adapter.R",
        "tests/canonical/weighted_lmtp_ltmle/run_study.R",
    ),
    runner_module="tests.studies.canonical_weighted_ltmle_crossfit",
    properties_module="tests.studies.weighted_ltmle_crossfit_properties",
    property_cells=property_cells(),
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": "f04a2b47f46debc515ce4ae778e05ebfde922c44",
    "lmtp_tarball_sha256": "fd49d9f291d4ddabb78c36d152b25aaa234a7204b645b9921f998c152e3d2ba5",
    "r_base_image": (
        "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
    ),
}

CONFIGURATION = {
    "construction": "fold_specific_cross_fit_weighted",
    "selection_probability": {"W1_positive": 0.3, "otherwise": 0.9},
    "selected_n": common.PRIMARY_N,
    "cross_fit": True,
    "outer_folds": 5,
    "learner_folds": 2,
    "weights": "fixed_inverse_selection_probability",
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
    "g_bounds": list(common.G_BOUNDS),
    "regimens": list(common.REGIMENS),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_scenario(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return common.fit_cleverly(frame, cross_fit=True)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return common.rows_from_result(STUDY, fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    return common.replicate(STUDY, payload, cross_fit=True)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return common.draw_and_fit(STUDY, replicates=replicates, n=n, cross_fit=True, n_jobs=n_jobs)
