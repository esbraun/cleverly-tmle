"""Repeated-sampling properties for outcome-adaptive C-TMLE.

Eleven of the twelve cells here are cross-fitted, so every cell samples from a **bounded**
law and declares ``q_bounds=(0, 1)``.  :mod:`tests.studies.bounded_cv_laws` says why at
length: without a declared outcome support the estimator reads the scale off every observed
outcome, and each fold's training predictions then depend on the rows it is predicting.
The twelfth is ``crossfit_overfitting/in_sample_control``, and it moves onto the bounded law
and declares the same support.  It is paired with the cross-fitted arm on one seed, so the
two must differ in whether the tree saw the rows it predicts and in nothing else.

The split is declared too, through ``stratify_folds="none"``.  ``strategy="oat"`` draws no
selection folds, so the declaration reaches the outer folds and the Super Learner's inner
folds alone.

The cells are stated on their Gaussian originals and transformed rather than retyped.  The
inherited canonical block comes through
:func:`~tests.studies.bounded_cv_laws.bounded_cells`, and the three families this study
declares itself through :func:`~tests.studies.bounded_cv_laws.bounded_twin`, so each one
keeps the budget, seed, role and size it always declared.  The one budget that does move is
the generated-design pair's, and :data:`~tests.studies.bounded_cv_laws.GENERATED_DESIGN_REPLICATES`
records why.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import CTMLE
from tests.conftest import OracleOutcomeContinuous
from tests.parallel import STUDY_JOBS
from tests.studies import bounded_cv_laws, canonical_properties, cvtmle_properties
from tests.studies.canonical_ctmle_oat import G_BOUNDS, STRATIFY_FOLDS, STUDY
from tests.studies.evidence.properties import (
    PropertyCell,
    run_cells,
    se_ratio_deficit_interval,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    robustness_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

OAT_NULL_REPLICATES = 800

#: The two cells that measure how OAT's reported interval behaves when its design moves.
#:
#: OAT fits ``g`` on ``Qbar`` itself, so when ``Qbar`` is estimated the design ``g`` is
#: fitted on is random too, and the influence curve is evaluated after that fit.  These
#: cells are one law and one set of draws with a single difference: whether ``Qbar`` moves.
#: ``oracle_design`` pins it, and must be calibrated -- that is the positive claim, and it
#: is what says the machinery is right when the design is fixed.  ``estimated`` fits it, and
#: is a *control*: it must report a materially smaller standard error relative to its own
#: spread, because a study that could not separate the two designs would report the same
#: "calibrated" verdict for both and establish nothing.
#:
#: The margin is a detection threshold rather than a tolerance.  It registers a paired
#: finite-sample difference between the two designs.  It does not decide whether an
#: asymptotic correction exists, and the deficit is not by itself proof of an omitted
#: first-order term.  ``docs/roadmap.md`` F19 carries that question, and a source-backed
#: answer there must say whether and how this cell changes.
#: A cell whose law was chosen without measuring it first could have landed where OAT's
#: design happens to span the true mechanism, where there is no difference to detect and
#: the control fails for a reason that is not the estimator's.
GENERATED_DESIGN_EFFECT = 0.3
GENERATED_DESIGN_N = 1000
GENERATED_DESIGN_DEFICIT = 0.01

#: How many observations the preflight fit below draws.  Small: it reads a fitted scaler,
#: which is a property of the declaration rather than of the sample, and it is paid once
#: before every run of this study.
PREFLIGHT_N = 400


def cells() -> tuple[PropertyCell, ...]:
    nonlinear = nonlinear_dgp()
    gaussian_robustness = (
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
    # This study's type-I cell answers to two gates, and at the canonical 400 replications
    # it cleared neither with room: the exact coverage lower endpoint fell below the 0.90
    # floor, and the one-sided rejection bound sat above the 0.10 ceiling.  Doubling
    # contracts both intervals around the rates actually observed without moving either
    # margin.  It does not make the cell comfortable -- see the OAT section on why this
    # law's size sits above nominal at all -- and the published endpoints, not the point
    # estimates, are what show that.  The budget carries over to the bounded twin
    # unchanged; the law under it is what moved.
    inherited = bounded_cv_laws.bounded_cells(
        "ctmle_oat_properties",
        exclude=("double_robustness",),
        replicates={"type_i_error": OAT_NULL_REPLICATES},
    )
    gaussian_overfit = (
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
    gaussian_generated = (
        PropertyCell(
            "generated_design",
            "oracle_design",
            generated_law,
            lambda: OracleOutcomeContinuous(generated_law),
            lambda: LogisticRegression(max_iter=1000),
            GENERATED_DESIGN_N,
            bounded_cv_laws.GENERATED_DESIGN_REPLICATES,
            13_300,
        ),
        PropertyCell(
            "generated_design",
            "estimated",
            generated_law,
            LinearRegression,
            lambda: LogisticRegression(max_iter=1000),
            GENERATED_DESIGN_N,
            bounded_cv_laws.GENERATED_DESIGN_REPLICATES,
            # The same seed as its pair on purpose: the deficit below is a paired
            # difference, and two cells drawn apart could not supply one.
            13_300,
            role="control",
        ),
    )
    robustness = tuple(bounded_cv_laws.bounded_twin(cell) for cell in gaussian_robustness)
    overfit = tuple(bounded_cv_laws.bounded_twin(cell) for cell in gaussian_overfit)
    generated = tuple(bounded_cv_laws.bounded_twin(cell) for cell in gaussian_generated)
    return (*robustness, *inherited, *overfit, *generated)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, so each committed truth is read back against its law.

    Returns
    -------
    tuple of PropertyCell
        The bounded robustness pair, the inherited canonical block, the overfitting pair
        and the generated-design pair, in the order they run.
    """
    return cells()


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
        # Every cell here samples from a beta law, so every cell declares the support.
        # The in-sample control declares it too: the two arms of the overfitting claim
        # differ in whether the tree saw the rows it predicts, and nothing else, so a
        # control on a derived scale would confound the comparison with a second change.
        q_bounds=bounded_cv_laws.Q_BOUNDS,
        stratify_folds=STRATIFY_FOLDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def assert_unit_outcome_scale(declared: tuple[PropertyCell, ...]) -> None:
    """Fit the oracle-outcome arm once, and refuse a fit that derived its outcome scale.

    :class:`~tests.conftest.OracleOutcomeUnit` returns the law's conditional mean on the
    unit scale, and it cannot check the condition that makes it exact.  A learner is handed
    the already-scaled outcome, and a scaler derived from the observed range maps it into a
    subset of what the identity produces, so no input distinguishes the two.  Here the
    scaler is on the result, and the check is exact.

    Two cells fit that oracle, ``robustness_contract/outcome_correct`` and
    ``generated_design/oracle_design``, and both reach it through :func:`_estimator`.  One
    fit therefore witnesses the declaration both of them rest on.

    Parameters
    ----------
    declared : tuple of PropertyCell
        The cells this run is about to spend its budget on.
    """
    cell = next(
        cell
        for cell in declared
        if cell.property == "robustness_contract" and cell.cell == "outcome_correct"
    )
    frame, _ = cell.dgp.sample(PREFLIGHT_N, seed=cell.seed)
    result = _estimator(cell)().fit(frame, **cell.fit_kwargs).single()
    bounded_cv_laws.assert_unit_outcome_scaler(result)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    declared = cells()
    assert_unit_outcome_scale(declared)
    return run_cells(declared, _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            "se_ratio_deficit_lower",
            "se_ratio_deficit_upper",
        ),
    )

    robustness_verdicts(summary, family="robustness_contract")

    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_oat")

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

    return finish(summary, rates)
