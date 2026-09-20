"""Repeated-sampling properties for selector-based C-TMLE.

Every cell here is cross-fitted, so every cell samples from a **bounded** law and declares
``q_bounds=(0, 1)``.  :mod:`tests.studies.bounded_cv_laws` says why at length: without a
declared outcome support the estimator reads the scale off every observed outcome, and each
fold's training predictions then depend on the rows it is predicting.  The split is declared
too, through ``stratify_folds="none"``, and it reaches the outer folds, the selection folds
and the nested folds of each candidate alike.

The cells are stated on their Gaussian originals and transformed rather than retyped.  The
inherited canonical block comes through
:func:`~tests.studies.bounded_cv_laws.bounded_cells` and the two families this study
declares itself through :func:`~tests.studies.bounded_cv_laws.bounded_twin`, so each one
keeps the budget, seed, role and size it always declared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import instrument_dgp, linear_dgp
from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import bounded_cv_laws, canonical_properties
from tests.studies.canonical_ctmle_selector import G_BOUNDS, STRATIFY_FOLDS, STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish

SELECTOR_RMSE_RATIO = 0.50


def cells() -> tuple[PropertyCell, ...]:
    linear = linear_dgp()
    gaussian_robustness = (
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
            # This is the slower leg of double robustness: it leans on inverse
            # weighting, so its O(n^-1) remainder is the largest of the four arms.
            # The size was declared on the Gaussian linear law, where 700 left that
            # remainder too close to the fixed 0.25-SD equivalence margin to be told
            # apart from first-order bias, and 2,000 separated the two.  It carries
            # over to the bounded twin with the budget and the seed through
            # ``bounded_twin``, unremeasured: this is a positive cell, and remeasuring
            # its own verdict statistic to choose its size is what the margin exists
            # to prevent.
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
    gaussian_necessity = (
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
    robustness = tuple(bounded_cv_laws.bounded_twin(cell) for cell in gaussian_robustness)
    selector_necessity = tuple(bounded_cv_laws.bounded_twin(cell) for cell in gaussian_necessity)
    inherited = bounded_cv_laws.bounded_cells(
        "ctmle_selector_properties", exclude=("double_robustness",)
    )
    return (*robustness, *selector_necessity, *inherited)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, so each committed truth is read back against its law.

    Returns
    -------
    tuple of PropertyCell
        The bounded double-robustness arms, the selector-necessity pair, and the inherited
        canonical block, in the order they run.
    """
    return cells()


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
        # Every cell here samples from a beta law, so every cell declares the support, and
        # every split -- outer, selection and nested -- is drawn without reading a label.
        q_bounds=bounded_cv_laws.Q_BOUNDS,
        stratify_folds=STRATIFY_FOLDS,
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
    summary.loc[mask, "rmse_ratio"] = ratio
    positive = mask & (summary["role"] == "positive")
    negative = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    joint = bool(summary.loc[mask, "passed"].all() and ratio <= SELECTOR_RMSE_RATIO)
    summary.loc[mask, "property_passed"] = joint
    return finish(summary, rates)
