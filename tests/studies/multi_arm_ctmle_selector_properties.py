"""Repeated-sampling properties for selector-based multi-arm C-TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common, multi_arm_properties
from tests.studies.canonical_multi_arm_ctmle_selector import STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish

SELECTOR_RMSE_RATIO = 0.80


def cells() -> tuple[PropertyCell, ...]:
    return (
        *multi_arm_properties.selector_cells(seed=24_500),
        *multi_arm_properties.asymptotic_cells(seed=24_100),
    )


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    options: dict[str, Any] = {
        "strategy": "discrete",
        "candidates": ((), ("W1",), ("W1", "W2"), ("W1", "W2", "W3")),
    }
    if cell.property == "selector_necessity" and cell.cell == "empty_control":
        options["candidates"] = ((),)
    return lambda: CTMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=True,
        n_folds=5,
        selection_folds=3,
        selection_inner_folds=2,
        penalty=False,
        estimands="ate",
        ctmle_estimand="ate",
        reference=multi_arm_common.REFERENCE,
        simultaneous=False,
        g_bounds=multi_arm_common.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        **options,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=("rmse_ratio",))
    necessity = rows.loc[rows["property"] == "selector_necessity"]
    collaborative = necessity.loc[necessity["cell"] == "collaborative"].sort_values("replicate")
    control = necessity.loc[necessity["cell"] == "empty_control"].sort_values("replicate")
    if not np.array_equal(collaborative["replicate"], control["replicate"]):
        raise ValueError("selector-necessity cells are not paired on replication")
    collaborative_error = collaborative["estimate"].to_numpy() - collaborative["truth"].to_numpy()
    control_error = control["estimate"].to_numpy() - control["truth"].to_numpy()
    ratio = float(np.sqrt(np.mean(collaborative_error**2) / np.mean(control_error**2)))
    mask = summary["property"] == "selector_necessity"
    positive = mask & (summary["role"] == "positive")
    negative = mask & (summary["role"] == "control")
    summary.loc[mask, "rmse_ratio"] = ratio
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    summary.loc[mask, "property_passed"] = bool(
        summary.loc[mask, "passed"].all() and ratio <= SELECTOR_RMSE_RATIO
    )
    return finish(summary, rates)
