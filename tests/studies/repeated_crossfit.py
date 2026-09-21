"""Registered evidence for repeated point-treatment cross-fitted TMLE.

The construction comes from :mod:`tests.studies.canonical_cvtmle`, so this row inherits that
module's two declarations as well: each draw's outer split reads neither the treatment nor
the outcome, and the continuous law's outcome scale is declared rather than derived.
:func:`~tests.studies.canonical_cvtmle.cv_fit` passes both and reads the realized plan back
off every fit.  What is this study's own is the repetition: three complete fold draws per
sample, a median point, and the median within-draw variance plus squared split displacement.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.canonical_cvtmle import (
    G_BOUNDS,
    Q_BOUNDS,
    STRATIFY_FOLDS,
    cv_fit,
    fitted_rows,
    rows_from_result,
)
from tests.studies.canonical_tmle import SCENARIO_ESTIMANDS
from tests.studies.canonical_tmle import draw_from_seed as canonical_tmle_draw_from_seed
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 800
PRIMARY_N = 1000
SEED = 20260924
N_FOLDS = 5
REPEATS = 3
FOLD_SEED_TRIALS = 400
REPEAT_STABILITY_N = 1_000
MAX_REPEAT_SPREAD_RATIO = 1.0

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
    "repeat_stability": ("three_repeats", "one_repeat_control"),
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
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/bounded_cv_laws.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
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
    "stratify_folds": STRATIFY_FOLDS,
    "q_bounds": (
        f"declared {list(Q_BOUNDS)} on the continuous law, whose outcome is a proportion; "
        "none on the binary law, whose scaler is already the identity"
    ),
    "folds": "unstratified five-fold assignments drawn from the estimator's own seed",
    "repeat_aggregation": (
        "median point; median of within-draw variance plus squared split displacement; "
        "ratios use the log scale"
    ),
    "repeat_stability": {
        "n": REPEAT_STABILITY_N,
        "paired_fold_seed_trials": FOLD_SEED_TRIALS,
        "spread_ratio_upper_boundary": MAX_REPEAT_SPREAD_RATIO,
        "seed_streams": ["sample", "fold_seed", "bootstrap"],
        "control": "first fold seed derived by each three-repeat plan",
    },
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    return canonical_tmle_draw_from_seed(scenario, n, seed)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """Fit the declared five-fold, three-draw stacked CV-TMLE.

    ``N_FOLDS`` here is this module's own 5, not the 10 that ``canonical_cvtmle`` declares
    under the same name.
    """
    return cv_fit(
        frame,
        binary=scenario == "binary",
        estimands=SCENARIO_ESTIMANDS[scenario],
        n_folds=N_FOLDS,
        repeats=REPEATS,
        cv_evaluation=False,
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema."""
    return rows_from_result(STUDY, fit_cleverly(frame, scenario), truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )
