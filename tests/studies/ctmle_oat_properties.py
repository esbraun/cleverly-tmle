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
from tests.studies.evidence.properties import (
    PropertyCell,
    coverage_gain_interval,
    run_cells,
    se_ratio_deficit_interval,
    se_ratio_interval,
)
from tests.studies.evidence.seeds import stream_seed

OAT_NULL_REPLICATES = 800

#: The two cells that measure what OAT's reported interval leaves out.
#:
#: OAT fits ``g`` on ``Qbar`` itself, so when ``Qbar`` is estimated the *model class*
#: ``g`` is chosen from is random too, and the influence curve does not see that.  These
#: cells are one law and one set of draws with a single difference: whether ``Qbar`` moves.
#: ``oracle_design`` pins it, and must be calibrated -- that is the positive claim, and it
#: is what says the machinery is right when the design is fixed.  ``estimated`` fits it, and
#: is a *control*: it must report a materially smaller standard error relative to its own
#: spread, because a study that could not detect the omission would report the same
#: "calibrated" verdict for both and establish nothing.
#:
#: The margin is a floor on a defect rather than a ceiling on one.  It is deliberately not
#: a tolerance that blesses the current behaviour: if the reported interval is ever made to
#: carry this term, this control stops being discriminated and the study goes red, which is
#: the correct signal that the documented limitation has become stale.
#: The confounded law with a real effect, at the size where the deficit was measured
#: before this cell was written: 0.981 against 0.017 over 2,000 draws, or 0.036 of a
#: reported standard error, and unchanged at ``n = 4,000``.  A cell whose law was chosen
#: without measuring it first could have landed where OAT's design happens to span the
#: true mechanism, where there is no omission to detect and the control fails for a reason
#: that is not the estimator's.
GENERATED_DESIGN_EFFECT = 0.3
GENERATED_DESIGN_N = 1000
GENERATED_DESIGN_REPLICATES = 1_200
GENERATED_DESIGN_DEFICIT = 0.01


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
            # This cell answers to two gates, and at 400 replications it cleared
            # neither with room: the exact coverage lower endpoint fell below the 0.90
            # floor, and the one-sided rejection bound sat above the 0.10 ceiling.
            # Doubling contracts both intervals around the rates actually observed
            # without moving either margin.  It does not make the cell comfortable --
            # see the OAT section on why this law's size sits above nominal at all --
            # and the published endpoints, not the point estimates, are what show that.
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
    generated_law = canonical_properties.null_dgp(GENERATED_DESIGN_EFFECT)
    generated = (
        PropertyCell(
            "generated_design",
            "oracle_design",
            generated_law,
            lambda: OracleOutcomeContinuous(generated_law),
            lambda: LogisticRegression(max_iter=1000),
            GENERATED_DESIGN_N,
            GENERATED_DESIGN_REPLICATES,
            13_300,
        ),
        PropertyCell(
            "generated_design",
            "estimated",
            generated_law,
            LinearRegression,
            lambda: LogisticRegression(max_iter=1000),
            GENERATED_DESIGN_N,
            GENERATED_DESIGN_REPLICATES,
            # The same seed as its pair on purpose: the deficit below is a paired
            # difference, and two cells drawn apart could not supply one.
            13_300,
            role="control",
        ),
    )
    return (*robustness, *inherited, *overfit, *generated)


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
        extra_columns=(
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            "se_ratio_deficit_lower",
            "se_ratio_deficit_upper",
        ),
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
    gain = coverage_gain_interval(
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

    generated = rows.loc[rows["property"] == "generated_design"]
    estimated_rows = generated.loc[generated["cell"] == "estimated"].sort_values("replicate")
    oracle_rows = generated.loc[generated["cell"] == "oracle_design"].sort_values("replicate")
    oracle_se = se_ratio_interval(
        oracle_rows,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", "oracle_design"),
    )
    deficit = se_ratio_deficit_interval(
        estimated_rows,
        oracle_rows,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", "deficit"),
    )
    estimated_se = se_ratio_interval(
        estimated_rows,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, "generated_design", "estimated"),
    )
    design_verdicts = {
        # Fixed design: the ordinary calibration claim, on the ordinary band.
        "oracle_design": bool(oracle_se.within(*STUDY.margins.calibration_se_ratio)),
        # Estimated design: the omission must be *visible*, not merely suspected.
        "estimated": bool(deficit.high <= -GENERATED_DESIGN_DEFICIT),
    }
    design_joint = bool(all(design_verdicts.values()))
    for cell, interval in (("oracle_design", oracle_se), ("estimated", estimated_se)):
        mask = (summary["property"] == "generated_design") & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "se_ratio_deficit_lower"] = deficit.low
        summary.loc[mask, "se_ratio_deficit_upper"] = deficit.high
        summary.loc[mask, "passed"] = design_verdicts[cell]
        summary.loc[mask, "property_passed"] = design_joint

    return canonical_properties.finish(summary, rates)
