"""Repeated-sampling properties for outcome-adaptive C-TMLE."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import CTMLE
from tests.conftest import OracleOutcomeContinuous
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_properties, cvtmle_properties
from tests.studies.canonical_ctmle_oat import G_BOUNDS, STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells, se_ratio_interval
from tests.studies.evidence.seeds import stream_seed

OAT_NULL_REPLICATES = 800


def cells() -> tuple[PropertyCell, ...]:
    nonlinear = nonlinear_dgp()
    robustness = (
        PropertyCell(
            "robustness_contract",
            "outcome_correct",
            nonlinear,
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: LogisticRegression(max_iter=1000),
            700,
            canonical_properties.DOUBLE_ROBUST_REPLICATES,
            13_100,
        ),
        PropertyCell(
            "robustness_contract",
            "outcome_wrong",
            nonlinear,
            LinearRegression,
            lambda: LogisticRegression(max_iter=1000),
            700,
            canonical_properties.DOUBLE_ROBUST_REPLICATES,
            13_101,
            role="control",
        ),
    )
    inherited = tuple(
        replace(
            cell,
            seed=cell.seed + 5_000,
            # At 400 replications a valid 93.25% observed coverage has a 99%
            # Clopper-Pearson lower endpoint below 0.90.  Doubling only this cell
            # resolves the declared floor; it does not alter the floor or size margin.
            replicates=(
                OAT_NULL_REPLICATES if cell.property == "type_i_error" else cell.replicates
            ),
        )
        for cell in canonical_properties.cells()
        if cell.property != "double_robustness"
    )
    overfit = (
        PropertyCell(
            "crossfit_overfitting",
            "cross_fitted_oat",
            nonlinear,
            lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            lambda: LogisticRegression(max_iter=1000),
            cvtmle_properties.OVERFIT_N,
            cvtmle_properties.OVERFIT_REPLICATES,
            13_200,
        ),
        PropertyCell(
            "crossfit_overfitting",
            "in_sample_control",
            nonlinear,
            lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            lambda: LogisticRegression(max_iter=1000),
            cvtmle_properties.OVERFIT_N,
            cvtmle_properties.OVERFIT_REPLICATES,
            13_200,
            role="control",
        ),
    )
    return (*robustness, *inherited, *overfit)


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    in_sample = cell.property == "crossfit_overfitting" and cell.cell == "in_sample_control"
    return lambda: CTMLE(
        strategy="oat",
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=not in_sample,
        n_folds=5,
        estimands=("ate",),
        simultaneous=False,
        g_bounds=G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = canonical_properties.apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
    )

    robustness = summary["property"] == "robustness_contract"
    positive = robustness & (summary["role"] == "positive")
    control = robustness & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    overfit_rows = rows.loc[rows["property"] == "crossfit_overfitting"]
    positive_rows = overfit_rows.loc[overfit_rows["cell"] == "cross_fitted_oat"]
    control_rows = overfit_rows.loc[overfit_rows["cell"] == "in_sample_control"]
    positive_se = se_ratio_interval(
        positive_rows,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "crossfit_overfitting", "cross_fitted_oat"),
    )
    control_se = se_ratio_interval(
        control_rows,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "crossfit_overfitting", "in_sample_control"),
    )
    gain = cvtmle_properties._coverage_gain_interval(
        positive_rows,
        control_rows,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "crossfit_overfitting", "coverage_gain"),
    )
    verdicts = {
        "cross_fitted_oat": bool(
            positive_se.low >= cvtmle_properties.OVERFIT_SE_FLOOR
            and positive_se.high <= STUDY.margins.se_ratio_sanity[1]
        ),
        "in_sample_control": bool(control_se.high <= cvtmle_properties.OVERFIT_SE_CONTROL_CEILING),
    }
    joint = bool(all(verdicts.values()) and gain[0] >= cvtmle_properties.OVERFIT_COVERAGE_GAIN)
    for cell, interval in (
        ("cross_fitted_oat", positive_se),
        ("in_sample_control", control_se),
    ):
        mask = (summary["property"] == "crossfit_overfitting") & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "coverage_gain_ci_lower"] = gain[0]
        summary.loc[mask, "coverage_gain_ci_upper"] = gain[1]
        summary.loc[mask, "passed"] = verdicts[cell]
        summary.loc[mask, "property_passed"] = joint
    return canonical_properties.finish(summary, rates)
