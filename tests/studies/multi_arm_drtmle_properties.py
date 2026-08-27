"""Repeated-sampling properties for multi-arm DR-TMLE."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common, multi_arm_properties
from tests.studies.canonical_multi_arm_drtmle import STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish


def cells() -> tuple[PropertyCell, ...]:
    return (
        *multi_arm_properties.robustness_cells(seed=22_100),
        *multi_arm_properties.asymptotic_cells(seed=22_100, include_null_power=False),
    )


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    return lambda: DRTMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2000),
        cross_fit=True,
        n_folds=5,
        estimands="ate",
        reference=multi_arm_common.REFERENCE,
        simultaneous=False,
        g_bounds=multi_arm_common.G_BOUNDS,
        max_outer=100,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        guard=("Q", "g"),
        reduction="univariate",
        reduced_crossfit="pooled",
        update_order="drtmle",
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return finish(*apply_shared_verdicts(rows, STUDY))
