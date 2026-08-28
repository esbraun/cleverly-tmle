"""Independent evidence for the original fold-evaluated CV-TMLE report."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import G_BOUNDS, N_FOLDS, cv_fit, rows_from_result
from tests.studies.canonical_tmle import draw_from_seed as canonical_tmle_draw_from_seed
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1000
SEED = 20240821
SUPPORTED = ("ey1", "ey0", "ate", "att", "atc")
SCENARIO_ESTIMANDS = {"binary": SUPPORTED, "continuous": SUPPORTED}

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
    "crossfit_overfitting": ("fold_evaluated_cvtmle", "in_sample_control"),
}

STUDY = StudyRecord(
    name="fold-evaluated point-treatment CV-TMLE",
    slug="fold-evaluated-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "cvtmle_fold",
    document="docs/technical-reference/method-evidence/fold-evaluated-point-treatment-cv-tmle.md",
    anchor="fold-evaluated-point-treatment-cv-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-fold-evaluated-cvtmle",
    reference=None,
    modules=(
        "tests/studies/fold_evaluated_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        # ``G_BOUNDS`` and ``N_FOLDS`` come from there, and both are result determining.
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/fold_cvtmle_properties.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.fold_evaluated_cvtmle",
    properties_module="tests.studies.fold_cvtmle_properties",
    property_cells=PROPERTY_CELLS,
)

CONFIGURATION = {
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "targeting_scheme": "pooled",
    "cv_evaluation": True,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "q_bounds": "sample outcome range",
    "fold_aggregation": "equal 1/V plug-in average with cross-validated variance",
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from *this* study's declared seed.

    The laws come from the ordinary-TMLE study; the samples do not.  This row is separate
    evidence, and it would not be if it re-used another study's draws.
    """
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample from an explicit seed, for the published-seed audit.

    The laws are the ordinary study's, so the draw is too.  What belongs to this study is
    the seed that reaches here, which :func:`draw_scenario` supplies from its own record.
    """
    return canonical_tmle_draw_from_seed(scenario, n, seed)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """The fold-evaluated construction: the shared build, with ``cv_evaluation=True``."""
    return cv_fit(
        frame,
        binary=scenario == "binary",
        estimands=SUPPORTED,
        n_folds=N_FOLDS,
        cv_evaluation=True,
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return rows_from_result(STUDY, fit_cleverly(frame, scenario), truth, scenario, replicate)


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for records in outcomes for row in records])
    return rows.loc[:, list(REPLICATE_COLUMNS)]
