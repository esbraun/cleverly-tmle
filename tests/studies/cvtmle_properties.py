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
from tests.studies.evidence.properties import (
    PropertyCell,
    coverage_gain_interval,
    run_cells,
    se_ratio_interval,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed

OVERFIT_REPLICATES = 400
OVERFIT_N = 500
OVERFIT_SE_FLOOR = 0.85
OVERFIT_SE_CONTROL_CEILING = 0.75
OVERFIT_COVERAGE_GAIN = 0.15


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
    margins = record.margins
    summary, rates = canonical_properties.apply_shared_verdicts(
        rows,
        record,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
    )

    overfit_rows = rows.loc[rows["property"] == "crossfit_overfitting"]
    positive_name = f"{variant}_cvtmle"
    positive_rows = overfit_rows.loc[overfit_rows["cell"] == positive_name]
    control_rows = overfit_rows.loc[overfit_rows["cell"] == "in_sample_control"]
    positive_se = se_ratio_interval(
        positive_rows,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, "crossfit_overfitting", positive_name),
    )
    control_se = se_ratio_interval(
        control_rows,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, "crossfit_overfitting", "in_sample_control"),
    )
    gain = coverage_gain_interval(
        positive_rows,
        control_rows,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, "crossfit_overfitting", "coverage_gain"),
    )
    # Each row's own verdict from its own statement.  The cross-fit arm claims a calibrated
    # standard error; the control claims the opposite, that fitting in-sample understates it
    # by a wide margin.  One scalar broadcast across the property published the *positive*
    # arm's rule beside a control whose SE ratio was 0.58 and whose coverage was 0.65, so a
    # reader could not tell which statement the "Pass" belonged to.
    verdicts = {
        positive_name: bool(
            positive_se.low >= OVERFIT_SE_FLOOR and positive_se.high <= margins.se_ratio_sanity[1]
        ),
        "in_sample_control": bool(control_se.high <= OVERFIT_SE_CONTROL_CEILING),
    }
    # The third statement is about the pair and belongs to neither row alone: cross-fitting
    # has to *buy* coverage the in-sample fit does not have, on the same samples.
    joint = bool(all(verdicts.values()) and gain[0] >= OVERFIT_COVERAGE_GAIN)
    for cell, interval in ((positive_name, positive_se), ("in_sample_control", control_se)):
        mask = (summary["property"] == "crossfit_overfitting") & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "coverage_gain_ci_lower"] = gain[0]
        summary.loc[mask, "coverage_gain_ci_upper"] = gain[1]
        summary.loc[mask, "passed"] = verdicts[cell]
        summary.loc[mask, "property_passed"] = joint

    return canonical_properties.finish(summary, rates)
