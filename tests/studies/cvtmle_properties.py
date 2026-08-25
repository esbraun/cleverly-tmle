"""Shared repeated-sampling claims for the two literature-backed CV-TMLE reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_properties
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    crossfit_overfitting_verdicts,
    finish,
)
from tests.studies.evidence.registry import StudyRecord

OVERFIT_REPLICATES = 400
OVERFIT_N = 500


def cells(variant: str) -> tuple[PropertyCell, ...]:
    """The ordinary TMLE claims plus the overfitting experiment CV-TMLE exists for."""
    inherited = tuple(replace(cell) for cell in canonical_properties.cells())
    dgp = nonlinear_dgp()
    overfit = (
        PropertyCell(
            property="crossfit_overfitting",
            cell=f"{variant}_cvtmle",
            dgp=dgp,
            outcome_learner=lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            treatment_learner=canonical_properties.LogisticRegression,
            n=OVERFIT_N,
            replicates=OVERFIT_REPLICATES,
            seed=10_100,
        ),
        PropertyCell(
            property="crossfit_overfitting",
            cell="in_sample_control",
            dgp=dgp,
            outcome_learner=lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            treatment_learner=canonical_properties.LogisticRegression,
            n=OVERFIT_N,
            replicates=OVERFIT_REPLICATES,
            # The same seed as the cross-fit arm on purpose: the coverage-gain statement is
            # paired on ``replicate``, so the two arms must see the same samples and differ
            # only in whether the tree saw the rows it predicts.
            seed=10_100,
            role="control",
        ),
    )
    return (*inherited, *overfit)


def estimator(record: StudyRecord, variant: str) -> Callable[[PropertyCell], Callable[[], Any]]:
    def factory(cell: PropertyCell) -> Callable[[], Any]:
        control = cell.property == "crossfit_overfitting" and cell.cell == "in_sample_control"
        return lambda: TMLE(
            outcome_learner=cell.outcome_learner(),
            treatment_learner=cell.treatment_learner(),
            cross_fit=not control,
            n_folds=10,
            targeting_scheme="pooled",
            cv_evaluation=variant == "fold_evaluated" and not control,
            estimands=cell.estimand,
            simultaneous=False,
            g_bounds=(0.025, 0.975),
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )

    return factory


def generate(record: StudyRecord, variant: str, *, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(variant), estimator(record, variant), n_jobs=n_jobs)


def summarize(rows: pd.DataFrame, record: StudyRecord, variant: str) -> pd.DataFrame:
    """Summarize the shared cells and make the overfitting control load-bearing."""
    summary, rates = apply_shared_verdicts(
        rows,
        record,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
    )

    crossfit_overfitting_verdicts(summary, rows, record, positive_cell=f"{variant}_cvtmle")

    return finish(summary, rates)
