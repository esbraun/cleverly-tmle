"""Repeated-sampling properties for ordinary multi-arm TMLE."""

from __future__ import annotations

import pandas as pd

from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common, multi_arm_properties
from tests.studies.canonical_multi_arm_tmle import STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish


def cells() -> tuple[PropertyCell, ...]:
    return (
        *multi_arm_properties.robustness_cells(seed=21_100),
        *multi_arm_properties.asymptotic_cells(seed=21_100),
    )


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    return lambda: TMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=False,
        estimands="ate",
        reference=multi_arm_common.REFERENCE,
        simultaneous=False,
        g_bounds=multi_arm_common.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return finish(*apply_shared_verdicts(rows, STUDY))
