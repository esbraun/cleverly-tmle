"""Shared repeated-sampling claims for point-treatment CV-TMLE reports."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.inference import cross_validated_variance
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


def cells(variant: str, *, include_overfitting: bool = True) -> tuple[PropertyCell, ...]:
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
    return (*inherited, *overfit) if include_overfitting else inherited


def estimator(
    record: StudyRecord,
    variant: str,
    *,
    repeats: int = 1,
    n_folds: int = 10,
    targeting_scheme: str = "pooled",
    cv_evaluation: bool | None = None,
) -> Callable[[PropertyCell], Callable[[], Any]]:
    evaluate_by_fold = variant == "fold_evaluated" if cv_evaluation is None else cv_evaluation

    def factory(cell: PropertyCell) -> Callable[[], Any]:
        control = cell.property == "crossfit_overfitting" and cell.cell == "in_sample_control"
        return lambda: TMLE(
            outcome_learner=cell.outcome_learner(),
            treatment_learner=cell.treatment_learner(),
            cross_fit=not control,
            n_folds=n_folds,
            repeats=repeats if not control else 1,
            targeting_scheme=targeting_scheme if not control else "pooled",
            cv_evaluation=evaluate_by_fold and not control,
            estimands=cell.estimand,
            simultaneous=False,
            g_bounds=(0.025, 0.975),
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )

    return factory


def assert_double_robustness_preflight(
    record: StudyRecord,
    variant: str,
    declared: tuple[PropertyCell, ...],
    *,
    repeats: int,
    n_folds: int,
    targeting_scheme: str,
    cv_evaluation: bool | None,
) -> None:
    """Run deterministic controls before a CV property driver spends its full budget."""
    cell = next(
        cell
        for cell in declared
        if cell.property == "double_robustness" and cell.cell == "treatment_correct"
    )
    frame, _ = cell.dgp.sample(400, seed=cell.seed)
    build = estimator(
        record,
        variant,
        repeats=repeats,
        n_folds=n_folds,
        targeting_scheme=targeting_scheme,
        cv_evaluation=cv_evaluation,
    )
    result = build(cell)().fit(frame, **cell.fit_kwargs).single()
    canonical_properties.assert_double_robustness_fit(
        cell,
        result,
        g_bounds=(0.025, 0.975),
    )

    fluctuation = result.fluctuations["mean"]
    if targeting_scheme == "fold":
        if len(fluctuation.folds) != n_folds:
            raise RuntimeError("fold targeting did not retain one fluctuation per fold")
        if any(np.max(np.abs(fold.epsilon)) <= 1e-8 for fold in fluctuation.folds):
            raise RuntimeError("a fold-targeted preflight epsilon is zero")
        if any(
            not fold.converged or np.max(np.abs(fold.score)) > 1e-9 for fold in fluctuation.folds
        ):
            raise RuntimeError("a fold-targeted preflight score is unsolved")

    evaluate_by_fold = variant == "fold_evaluated" if cv_evaluation is None else cv_evaluation
    if evaluate_by_fold and repeats == 1:
        detail = result.cv_targeting
        if detail is None:
            raise RuntimeError("fold evaluation produced no cross-validated targeting detail")
        expected_point = float(np.mean(detail.fold_estimates["ate"]))
        if result.psi("ate") != expected_point:
            raise RuntimeError("the fold-evaluated ATE is not the equal 1/V fold average")
        indices = (
            [fold.index for fold in fluctuation.folds]
            if fluctuation.folds
            else [test for _, test in result.nuisance.folds]
        )
        expected_variance = cross_validated_variance(
            result["ate"].influence_curve,
            indices,
        )
        if not np.isclose(result["ate"].variance, expected_variance, rtol=1e-12):
            raise RuntimeError("the fold-evaluated ATE did not retain the CV variance")


def generate(
    record: StudyRecord,
    variant: str,
    *,
    repeats: int = 1,
    n_folds: int = 10,
    targeting_scheme: str = "pooled",
    cv_evaluation: bool | None = None,
    include_overfitting: bool = True,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    declared = cells(variant, include_overfitting=include_overfitting)
    assert_double_robustness_preflight(
        record,
        variant,
        declared,
        repeats=repeats,
        n_folds=n_folds,
        targeting_scheme=targeting_scheme,
        cv_evaluation=cv_evaluation,
    )
    return run_cells(
        declared,
        estimator(
            record,
            variant,
            repeats=repeats,
            n_folds=n_folds,
            targeting_scheme=targeting_scheme,
            cv_evaluation=cv_evaluation,
        ),
        n_jobs=n_jobs,
    )


@overload
def summarize(
    rows: pd.DataFrame,
    record: StudyRecord,
    variant: str,
    *,
    extra_columns: Sequence[str] = (),
    include_overfitting: bool = True,
    return_parts: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def summarize(
    rows: pd.DataFrame,
    record: StudyRecord,
    variant: str,
    *,
    extra_columns: Sequence[str] = (),
    include_overfitting: bool = True,
    return_parts: Literal[True],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]: ...


def summarize(
    rows: pd.DataFrame,
    record: StudyRecord,
    variant: str,
    *,
    extra_columns: Sequence[str] = (),
    include_overfitting: bool = True,
    return_parts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Summarize the shared cells, optionally making the overfitting control load-bearing."""
    summary, rates = apply_shared_verdicts(
        rows,
        record,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper", *extra_columns),
    )

    if include_overfitting:
        crossfit_overfitting_verdicts(summary, rows, record, positive_cell=f"{variant}_cvtmle")

    return (summary, rates) if return_parts else finish(summary, rates)
