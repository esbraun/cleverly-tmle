"""Repeated-sampling properties for multi-arm DR-TMLE.

Every cell draws its own split from the estimator, so the cells declare
``stratify_folds="none"`` rather than supplying a vector.  The law is binary, so no cell
declares ``q_bounds``: the outcome scaler is already the identity.

The two ``fold_policy`` arms are the one exception to what that declaration buys.  They
are constructed with ``stratify_folds="none"`` like every other cell, so nothing is
refused, and the ``treatment_stratified`` arm then reaches a balanced split by replacing
:meth:`~cleverly.estimators.TMLE._fold_strata` through
:class:`~tests.studies.multi_arm_properties.FoldPolicyMixin`.  That family reports its
coverage and states no verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common, multi_arm_properties
from tests.studies.canonical_multi_arm_drtmle import STRATIFY_FOLDS, STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import (
    CONTRACTION_SCENARIOS,
    FOLD_POLICY_FAMILY,
    control_role,
    finish,
    fold_policy_diagnostics,
    summarize_contraction_properties,
)

#: Which design columns each misspecified nuisance keeps.
#:
#: Both drop ``W1`` and nothing else.  ``W1`` is the covariate the shared law puts in the arm
#: logits *and* in the outcome mean, so dropping it is a real confounding failure rather than
#: a dropped predictor a plug-in never needed.  ``W3`` enters the outcome only, and a model
#: that omits it stays consistent for this contrast, which is why the ``both_wrong`` control
#: has to reach past it.
#:
#: The outcome design is ``[arm indicator, arm indicator, W1, W2, W3]`` and the treatment
#: design is ``[W1, W2, W3]``.  Both misspecifications stay covariate-dependent on purpose.
#: A constant nuisance would hand this method's univariate guard regressions a constant single
#: regressor, and the reported standard error then inflates by orders of magnitude while the
#: cell still reads as a union-model measurement.
MISSPECIFIED_OUTCOME_COLUMNS = (0, 1, 3, 4)
MISSPECIFIED_TREATMENT_COLUMNS = (1, 2)

#: The sizes the contraction ladder is fitted over, and how many replications each rung gets.
#:
#: **Why this family exists.**  ``double_robustness`` judges the bias at one size against an
#: equivalence margin of a quarter of an empirical standard deviation, and on this law the
#: ``treatment_correct`` cell exceeds it at ``n = 2,000``.  A single red cell cannot say which
#: of two very different things happened: a second-order remainder that has not yet decayed,
#: which is what the binary theorem predicts and which leaves the interval eventually valid,
#: or an armwise extension that is not consistent at all.  Those look alike at one size and
#: mean opposite things, and this row exists to measure the armwise extension rather than to
#: inherit the binary result.
#:
#: Fitting log |bias| on log ``n`` separates them.  The ``both_wrong`` arm rides along as the
#: control that must fail to contract.  The ladder starts at the size the level cell is judged
#: at and doubles twice, so its first rung reproduces that regime rather than a milder one.
CONTRACTION_SIZES = (2000, 4000, 8000)
CONTRACTION_REPLICATES = 600


class ColumnLogistic(BaseEstimator, ClassifierMixin):
    """Unpenalized multinomial logistic regression on a fixed subset of design columns."""

    def __init__(self, columns: Sequence[int] | None = None) -> None:
        self.columns = columns

    def _select(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if self.columns is None:
            return values
        return values[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ColumnLogistic:
        self.model_ = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs").fit(
            self._select(design), target, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._select(design)), dtype=float)


def _misspecified_outcome() -> ColumnLogistic:
    return ColumnLogistic(MISSPECIFIED_OUTCOME_COLUMNS)


def _misspecified_treatment() -> ColumnLogistic:
    return ColumnLogistic(MISSPECIFIED_TREATMENT_COLUMNS)


def _nuisances(scenario: str) -> tuple[Any, Any]:
    """The learner pair one nuisance regime names, in the order a cell declares them."""
    outcome = (
        multi_arm_properties.correct_outcome()
        if scenario == "outcome_correct"
        else _misspecified_outcome
    )
    treatment = (
        multi_arm_properties.correct_treatment()
        if scenario == "treatment_correct"
        else _misspecified_treatment
    )
    return outcome, treatment


def _contraction_cells() -> tuple[PropertyCell, ...]:
    return tuple(
        PropertyCell(
            "double_robust_contraction",
            f"{scenario}_n{size}",
            multi_arm_properties.Sampler(),
            *_nuisances(scenario),
            size,
            CONTRACTION_REPLICATES,
            # One offset per rung, so no two rungs share a replication stream.  The ladder is
            # fitted across sizes, and a shared stream would correlate the rungs and narrow
            # the slope interval for a reason that has nothing to do with the estimator.
            24_000 + scenario_index * 300 + size_index * 100,
            role=control_role(scenario),
            estimand=multi_arm_properties.ESTIMAND,
        )
        for scenario_index, scenario in enumerate(CONTRACTION_SCENARIOS)
        for size_index, size in enumerate(CONTRACTION_SIZES)
    )


def cells() -> tuple[PropertyCell, ...]:
    return (
        *multi_arm_properties.robustness_cells(
            seed=22_100,
            misspecified_outcome=_misspecified_outcome,
            misspecified_treatment=_misspecified_treatment,
        ),
        *multi_arm_properties.asymptotic_cells(seed=22_100, include_null_power=False),
        *_contraction_cells(),
        *multi_arm_properties.fold_policy_cells(STUDY),
    )


def declared_cells() -> tuple[PropertyCell, ...]:
    """The cells :func:`generate_property_rows` runs, named for the truth-binding check.

    The same tuple :func:`cells` returns, exposed under the name
    ``tests/unit/test_method_evidence.py`` reads so that every committed truth is checked
    against the law the cell that produced it names.  The ``root_n_rate`` rows the document
    publishes are not here: :func:`summarize_properties` derives them from the size cells and
    they reach no replication file.

    Returns
    -------
    tuple of PropertyCell
        One entry per published ``(property, cell)`` pair in the replication rows.
    """
    return cells()


def _settings(cell: PropertyCell) -> dict[str, Any]:
    """The estimator arguments every cell of this study is fitted under.

    One dictionary rather than two estimator factories, because the fold-policy arms have
    to differ from the gated cells in the split policy and in nothing else.  A second
    literal copy of these arguments could drift from this one, and the drift would read as
    a fold-policy effect.

    Parameters
    ----------
    cell : PropertyCell
        The cell being fitted, which supplies the two nuisance learners.

    Returns
    -------
    dict
        The keyword arguments for :class:`~cleverly.estimators.DRTMLE`.
    """
    return {
        "outcome_learner": cell.outcome_learner(),
        "treatment_learner": cell.treatment_learner(),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(C=1e6, max_iter=2000),
        "cross_fit": True,
        "n_folds": 5,
        "stratify_folds": STRATIFY_FOLDS,
        "estimands": "ate",
        "reference": multi_arm_common.REFERENCE,
        "simultaneous": False,
        "g_bounds": multi_arm_common.G_BOUNDS,
        "max_outer": 100,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
        "guard": ("Q", "g"),
        "reduction": "univariate",
        "reduced_crossfit": "pooled",
        "update_order": "drtmle",
    }


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    # The fold-policy arms keep ``stratify_folds="none"`` like every other cell, so nothing
    # is refused at construction; the policy reaches this method's one split through
    # ``_fold_strata``.
    if cell.property == FOLD_POLICY_FAMILY:
        return lambda: multi_arm_properties.FoldPolicyDRTMLE(cell.cell, **_settings(cell))
    return lambda: DRTMLE(**_settings(cell))


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The contraction ladder's published summary, then the fold-policy numbers.

    The shared call is asked for its unfinished parts, because
    :func:`~tests.studies.evidence.property_verdicts.finish` is terminal and the
    fold-policy arms have their paired difference still to write.

    Parameters
    ----------
    rows : pandas.DataFrame
        Every replication row this study emitted.

    Returns
    -------
    pandas.DataFrame
        The published summary, in its published order.
    """
    summary, rates = summarize_contraction_properties(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        return_parts=True,
    )
    fold_policy_diagnostics(summary, rows, STUDY)
    return finish(summary, rates)
