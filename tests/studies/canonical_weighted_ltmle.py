"""Ordinary weighted end-of-study longitudinal TMLE evidence study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import weighted_longitudinal_common as common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.weighted_longitudinal_properties_common import property_cells

LTMLE_VERSION = "1.3-0"
SEED = 20261101
PRIMARY_REPLICATES = common.ORDINARY_PRIMARY_REPLICATES
PRIMARY_N = common.PRIMARY_N

STUDY = StudyRecord(
    name="ordinary weighted end-of-study longitudinal TMLE",
    slug="weighted-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "weighted_ltmle",
    document=(
        "docs/technical-reference/method-evidence/"
        "ordinary-weighted-end-of-study-longitudinal-tmle.md"
    ),
    anchor="ordinary-weighted-end-of-study-longitudinal-tmle",
    scenarios={common.SCENARIO: common.ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-weighted-ltmle",
    reference="ltmle-weighted",
    publication_policy="reporting",
    modules=(
        "tests/studies/canonical_weighted_ltmle.py",
        "tests/studies/weighted_longitudinal_common.py",
        "tests/studies/weighted_ltmle_properties.py",
        "tests/studies/weighted_longitudinal_properties_common.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/ltmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/ltmle/Dockerfile",
        "tests/canonical/study_harness.R",
        "tests/canonical/ltmle_regimen_adapter.R",
        "tests/canonical/weighted_ltmle/run_study.R",
    ),
    runner_module="tests.studies.canonical_weighted_ltmle",
    properties_module="tests.studies.weighted_ltmle_properties",
    property_cells=property_cells(),
)

REFERENCE_METADATA = {
    "ltmle_version": LTMLE_VERSION,
    "ltmle_source_commit": "338c029dae9692ef20714125773da7037688993b",
    "ltmle_tarball_sha256": "fb31d0dd6ab81687b81f3279b414c21e91c655e10aac12f73fc6723efd848aad",
    "r_base_image": (
        "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
    ),
}

CONFIGURATION = {
    "construction": "ordinary_weighted",
    "selection_probability": {
        "W1_positive": common.SELECTION_LOW,
        "otherwise": common.SELECTION_HIGH,
    },
    "selected_n": common.PRIMARY_N,
    "cross_fit": False,
    "outer_folds": 1,
    "weights": "fixed_inverse_selection_probability",
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    "mechanism": "supplied_from_the_law_to_both",
    "g_bounds": list(common.G_BOUNDS),
    "regimens": list(common.REGIMENS),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_scenario(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return common.fit_cleverly(frame, cross_fit=False)


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
    return common.replicate(STUDY, payload, cross_fit=False)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return common.draw_and_fit(STUDY, replicates=replicates, n=n, cross_fit=False, n_jobs=n_jobs)
