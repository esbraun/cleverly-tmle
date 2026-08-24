"""Repeated-sampling properties for selector-based C-TMLE."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import instrument_dgp, linear_dgp
from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_properties
from tests.studies.canonical_ctmle_selector import G_BOUNDS, STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells

SELECTOR_RMSE_RATIO = 0.50


def cells() -> tuple[PropertyCell, ...]:
    linear = linear_dgp()
    robustness = (
        PropertyCell(
            "double_robustness",
            "both_correct",
            linear,
            LinearRegression,
            lambda: LogisticRegression(max_iter=1000),
            700,
            canonical_properties.DOUBLE_ROBUST_REPLICATES,
            12_100,
        ),
        PropertyCell(
            "double_robustness",
            "outcome_correct",
            linear,
            LinearRegression,
            lambda: DummyClassifier(strategy="prior"),
            700,
            canonical_properties.DOUBLE_ROBUST_REPLICATES,
            12_101,
        ),
        PropertyCell(
            "double_robustness",
            "treatment_correct",
            linear,
            DummyRegressor,
            lambda: LogisticRegression(max_iter=1000),
            # This is the slower leg of double robustness: at n=700 its O(n^-1)
            # remainder is still about 0.28 empirical SD.  At n=2,000 the same
            # fixed 0.25-SD equivalence margin can distinguish that remainder from
            # first-order bias without weakening the claim after observing it.
            2000,
            canonical_properties.DOUBLE_ROBUST_REPLICATES,
            12_102,
        ),
        PropertyCell(
            "double_robustness",
            "both_wrong",
            linear,
            DummyRegressor,
            lambda: DummyClassifier(strategy="prior"),
            700,
            canonical_properties.DOUBLE_ROBUST_REPLICATES,
            12_103,
            role="control",
        ),
    )
    forced = instrument_dgp()
    selector_necessity = (
        PropertyCell(
            "selector_necessity",
            "collaborative",
            forced,
            DummyRegressor,
            lambda: LogisticRegression(max_iter=1000),
            1500,
            canonical_properties.RATE_REPLICATES,
            12_200,
        ),
        PropertyCell(
            "selector_necessity",
            "empty_control",
            forced,
            DummyRegressor,
            lambda: LogisticRegression(max_iter=1000),
            1500,
            canonical_properties.RATE_REPLICATES,
            12_200,
            role="control",
        ),
    )
    inherited = tuple(
        replace(cell, seed=cell.seed + 4_000)
        for cell in canonical_properties.cells()
        if cell.property != "double_robustness"
    )
    return (*robustness, *selector_necessity, *inherited)


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    options: dict[str, Any] = {}
    if cell.property == "selector_necessity" and cell.cell == "empty_control":
        options = {"strategy": "discrete", "candidates": ((),)}
    return lambda: CTMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=True,
        n_folds=5,
        selection_folds=3,
        selection_inner_folds=2,
        estimands=("ate",),
        ctmle_estimand="ate",
        simultaneous=False,
        g_bounds=G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        **options,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    # Through ``extra_columns`` rather than assigned afterwards.  The rate rows are built from
    # ``summary.columns`` inside ``apply_shared_verdicts``, so a column added after the call is
    # absent when they are built and arrives on them as NaN by ``concat`` instead of through the
    # declared mechanism.  Same published value either way today; one of them is a coincidence.
    summary, rates = canonical_properties.apply_shared_verdicts(
        rows, STUDY, extra_columns=("rmse_ratio",)
    )
    necessity = rows.loc[rows["property"] == "selector_necessity"]
    collaborative = necessity.loc[necessity["cell"] == "collaborative"].sort_values("replicate")
    control = necessity.loc[necessity["cell"] == "empty_control"].sort_values("replicate")
    if not np.array_equal(collaborative["replicate"], control["replicate"]):
        raise ValueError("selector-necessity cells are not paired on replication")
    collaborative_error = collaborative["estimate"].to_numpy() - collaborative["truth"].to_numpy()
    control_error = control["estimate"].to_numpy() - control["truth"].to_numpy()
    ratio = float(np.sqrt(np.mean(collaborative_error**2) / np.mean(control_error**2)))
    mask = summary["property"] == "selector_necessity"
    summary.loc[mask, "rmse_ratio"] = ratio
    positive = mask & (summary["role"] == "positive")
    negative = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    joint = bool(summary.loc[mask, "passed"].all() and ratio <= SELECTOR_RMSE_RATIO)
    summary.loc[mask, "property_passed"] = joint
    return canonical_properties.finish(summary, rates)
