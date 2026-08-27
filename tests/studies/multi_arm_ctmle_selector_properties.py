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

#: The estimator arguments each selector-necessity cell adds to the shared configuration.
#: The control keeps the empty candidate and nothing else, so it stops before adjusting for
#: any multi-arm confounder; each positive cell may reach the same three covariates by the
#: path its own strategy defines.  Written here rather than inside the estimator factory so
#: the forced path and the free paths are read side by side.
SELECTOR_OPTIONS: dict[str, dict[str, Any]] = {
    "greedy": {"strategy": "greedy"},
    "ordered": {"strategy": "ordered", "ordering": ("W1", "W2", "W3")},
    "discrete": {
        "strategy": "discrete",
        "candidates": ((), ("W1",), ("W1", "W2"), ("W1", "W2", "W3")),
    },
    multi_arm_properties.SELECTOR_CONTROL: {"strategy": "discrete", "candidates": ((),)},
}


def cells() -> tuple[PropertyCell, ...]:
    return (
        *multi_arm_properties.selector_cells(seed=24_500),
        *multi_arm_properties.asymptotic_cells(seed=24_100),
    )


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    # Every other family keeps the discrete ladder the row's asymptotic cells were declared
    # with, so this factory varies the selector path only where the path is the subject.
    options = (
        SELECTOR_OPTIONS[cell.cell]
        if cell.property == "selector_necessity"
        else SELECTOR_OPTIONS["discrete"]
    )
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


def _root_mean_square_error(cell: pd.DataFrame) -> float:
    error = cell["estimate"].to_numpy() - cell["truth"].to_numpy()
    return float(np.sqrt(np.mean(error**2)))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=("rmse_ratio",))
    necessity = rows.loc[rows["property"] == "selector_necessity"]
    control = necessity.loc[necessity["cell"] == multi_arm_properties.SELECTOR_CONTROL].sort_values(
        "replicate"
    )
    control_error = _root_mean_square_error(control)
    ratios: dict[str, float] = {}
    for path in multi_arm_properties.SELECTOR_PATHS:
        chosen = necessity.loc[necessity["cell"] == path].sort_values("replicate")
        # Paired on the replication index, because the cells are run on the same draws
        # precisely so the ratio below is not two independent error curves divided.  A cell
        # that lost a replication would otherwise compare different samples and read as a
        # selector effect.
        if not np.array_equal(chosen["replicate"], control["replicate"]):
            raise ValueError(f"the {path!r} selector cell is not paired with its control")
        ratios[path] = _root_mean_square_error(chosen) / control_error

    mask = summary["property"] == "selector_necessity"
    for path, ratio in ratios.items():
        summary.loc[mask & (summary["cell"] == path), "rmse_ratio"] = ratio
    summary.loc[mask & (summary["role"] == "control"), "rmse_ratio"] = 1.0
    positive = mask & (summary["role"] == "positive")
    negative = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    # The family claim is joint over every declared path: selection has to be what produced
    # the result on each of them.  A path whose ratio reaches one chose the control's own
    # mechanism path, so the pair it forms is not a control at all, and reporting that as a
    # per-cell verdict alone would leave the family green on the strength of the others.
    summary.loc[mask, "property_passed"] = bool(
        summary.loc[mask, "passed"].all()
        and all(ratio <= SELECTOR_RMSE_RATIO for ratio in ratios.values())
    )
    return finish(summary, rates)
