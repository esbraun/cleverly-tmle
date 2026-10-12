"""Repeated-sampling properties for the lasso and learners C-TMLE sequences.

Every cell is cross-fitted: five outer folds, three selection folds and two inner folds, with
the split drawn unstratified.  Every two-arm cell samples a **bounded** law and declares
``q_bounds=(0, 1)``, for the reason :mod:`tests.studies.bounded_cv_laws` gives.  The two
``multi_arm`` cells sample the binary three-arm law of :mod:`tests.studies.multi_arm_common`,
whose outcome scaler is already the identity, so they declare no ``q_bounds``.

Every lasso cell replaces the default 101-penalty grid with a declared user grid of
:data:`PROPERTY_GRID_SIZE` penalties, the last of them 0, to bound the cost.  The positive
penalties follow the primary grid's rule (:func:`~tests.studies.canonical_ctmle_sequences.penalty_design`)
on the cell's own law and size, so each candidate is a distinct mechanism there too.

The declared candidates of every learners cell are :data:`LEARNER_COLUMNS`: a main-terms
logistic regression on ``W1``, then on ``W1`` and ``W2``.  ``treatment_learner`` is the
terminal: the main-terms logistic regression on every covariate.

**The outcome-correct law** (:func:`outcome_correct_dgp`) is the bounded linear law of
:func:`~tests.studies.bounded_cv_laws.linear_dgp` with the propensity of
:func:`~cleverly.datasets.nonlinear_dgp`.  The outcome mean is unchanged, so
:class:`~tests.studies.fractional_glm.QuasiBinomialGLM` on ``A`` and ``W1`` to ``W4`` still
specifies it correctly, and the ATE is the bounded linear law's.  The propensity carries
``W2^2``, ``W2 W3`` and ``1(W4 > 0)`` terms, so no candidate on either path is correct for it.
``tests/unit/test_ctmle_sequences_design.py`` checks both claims: the outcome learner's limit at
``n = 100,000``, and the main-terms logistic projection's KL gap above
:data:`MINIMUM_PROJECTION_KL`.  The law is built here rather than through
:func:`~tests.studies.bounded_cv_laws.bounded_twin`, because the twin table has no Gaussian
law with this propensity.

**Verdicts.**  ``double_robustness`` uses the shared union-model rule.  ``selector_necessity``
pairs each collaborative cell with the empty-path control on one seed, and the family claim
needs every positive cell's bias inside the margin, the control's outside it, and each
collaborative RMSE at most :data:`SELECTOR_RMSE_RATIO` of the control's.  ``multi_arm`` has no
RMSE clause: the positive cell's bias is inside the margin and the control's is outside it.

**Seeds** are the block ``16_100`` to ``16_400``, unused by every other property module.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cleverly.datasets import DGP, instrument_dgp, linear_dgp, nonlinear_dgp
from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import bounded_cv_laws, multi_arm_common
from tests.studies.canonical_ctmle_sequences import G_BOUNDS, STRATIFY_FOLDS, STUDY, penalty_design
from tests.studies.evidence.properties import PropertyBatch, PropertyCell
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.multi_arm_properties import ESTIMAND as MULTI_ARM_ESTIMAND
from tests.studies.multi_arm_properties import Sampler

#: The binary selector study's constant: each collaborative RMSE over the control's.
SELECTOR_RMSE_RATIO = 0.50

#: The declared lasso grid of every property cell: this many penalties, the last one 0.
PROPERTY_GRID_SIZE = 12

#: The smallest main-terms logistic KL gap the outcome-correct law's propensity may have.
MINIMUM_PROJECTION_KL = 0.01

#: The covariate columns, by position, of each declared learners candidate, least adaptive
#: first.  Positions rather than names, because a candidate learner reads the covariate matrix.
LEARNER_COLUMNS = ((0,), (0, 1))

#: Replication budgets.  1,200 for double robustness and the multi-arm pair, 800 for the
#: selector-necessity triple, as declared before the run.
ROBUST_REPLICATES = 1_200
NECESSITY_REPLICATES = 800

DOUBLE_ROBUST_SEED = 16_100
NECESSITY_SEED = 16_200
MULTI_ARM_SEED = 16_300


# --------------------------------------------------------------------------- laws


@functools.cache
def outcome_correct_dgp() -> DGP:
    """The bounded linear law with :func:`~cleverly.datasets.nonlinear_dgp`'s propensity.

    Returns
    -------
    DGP
        The law of the three outcome-correct and both-wrong cells.
    """
    return replace(
        bounded_cv_laws.linear_dgp(),
        name="bounded_linear_nonlinear_propensity",
        propensity=nonlinear_dgp().propensity,
    )


def _main_terms(columns: tuple[int, ...]) -> Pipeline:
    """A main-terms logistic regression on the covariate columns at ``columns``."""
    return Pipeline(
        [
            ("columns", ColumnTransformer([("keep", "passthrough", list(columns))])),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )


def declared_learners() -> list[Pipeline]:
    """The ``treatment_candidates=`` of every learners cell, least adaptive first.

    Returns
    -------
    list of Pipeline
        One unfitted candidate per entry of :data:`LEARNER_COLUMNS`.
    """
    return [_main_terms(columns) for columns in LEARNER_COLUMNS]


def _strategy(cell: str) -> str:
    if cell.startswith("lasso_"):
        return "lasso"
    if cell.startswith("learners_"):
        return "learners"
    return "discrete"


# --------------------------------------------------------------------------- cells


def cells() -> tuple[PropertyCell, ...]:
    gaussian_linear = linear_dgp()
    treatment_correct = tuple(
        bounded_cv_laws.bounded_twin(
            PropertyCell(
                "double_robustness",
                name,
                gaussian_linear,
                DummyRegressor,
                lambda: LogisticRegression(max_iter=1000),
                # The slower leg of double robustness, sized as the selector study sized it.
                2000,
                ROBUST_REPLICATES,
                seed,
            )
        )
        for name, seed in (
            ("lasso_treatment_correct", DOUBLE_ROBUST_SEED + 2),
            ("learners_treatment_correct", DOUBLE_ROBUST_SEED + 4),
        )
    )
    outcome_law = outcome_correct_dgp()
    outcome_correct = tuple(
        PropertyCell(
            "double_robustness",
            name,
            outcome_law,
            QuasiBinomialGLM,
            lambda: LogisticRegression(max_iter=1000),
            700,
            ROBUST_REPLICATES,
            seed,
        )
        for name, seed in (
            ("lasso_outcome_correct", DOUBLE_ROBUST_SEED + 1),
            ("learners_outcome_correct", DOUBLE_ROBUST_SEED + 3),
        )
    )
    both_wrong = PropertyCell(
        "double_robustness",
        "both_wrong",
        outcome_law,
        DummyRegressor,
        lambda: LogisticRegression(max_iter=1000),
        700,
        ROBUST_REPLICATES,
        DOUBLE_ROBUST_SEED + 5,
        role="control",
    )
    forced = instrument_dgp()
    necessity = tuple(
        bounded_cv_laws.bounded_twin(
            PropertyCell(
                "selector_necessity",
                name,
                forced,
                DummyRegressor,
                lambda: LogisticRegression(max_iter=1000),
                1500,
                NECESSITY_REPLICATES,
                NECESSITY_SEED,
                role=role,
            )
        )
        for name, role in (
            ("lasso_collaborative", "positive"),
            ("learners_collaborative", "positive"),
            ("empty_control", "control"),
        )
    )
    multi_arm = tuple(
        PropertyCell(
            "multi_arm",
            name,
            Sampler(),
            DummyRegressor,
            lambda: LogisticRegression(max_iter=1000),
            2000,
            ROBUST_REPLICATES,
            MULTI_ARM_SEED,
            role=role,
            estimand=MULTI_ARM_ESTIMAND,
        )
        for name, role in (("lasso_treatment_correct", "positive"), ("empty_control", "control"))
    )
    return (
        outcome_correct[0],
        treatment_correct[0],
        outcome_correct[1],
        treatment_correct[1],
        both_wrong,
        *necessity,
        *multi_arm,
    )


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, so each committed truth is read back against its law.

    Returns
    -------
    tuple of PropertyCell
        The five double-robustness cells, the selector-necessity triple and the multi-arm
        pair, in the order they run.
    """
    return cells()


# --------------------------------------------------------------------------- grids


def _covariates(cell: PropertyCell) -> tuple[str, ...]:
    if cell.property == "multi_arm":
        return ("W1", "W2", "W3")
    return tuple(cell.dgp.covariate_names)


def _sampler(cell: PropertyCell) -> Any:
    if cell.property == "multi_arm":

        def multi_arm(n: int, seed: int) -> pd.DataFrame:
            return cell.dgp(n, seed)[0]

        return multi_arm

    def two_arm(n: int, seed: int) -> pd.DataFrame:
        return cell.dgp.sample(n, seed=seed, backend="pandas")[0]

    return two_arm


@functools.cache
def _grid(law: str, n: int) -> tuple[float, ...]:
    matches = [
        cell
        for cell in cells()
        if _strategy(cell.cell) == "lasso" and cell.dgp.name == law and cell.n == n
    ]
    if not matches:
        raise KeyError(f"no lasso cell samples {law!r} at n = {n}")
    cell = matches[0]
    positive = penalty_design(
        _sampler(cell),
        n,
        size=PROPERTY_GRID_SIZE - 1,
        covariates=_covariates(cell),
        label=law,
    )
    return (*positive, 0.0)


def property_grid(cell: PropertyCell) -> tuple[float, ...]:
    """The declared lasso grid of one cell: the design rule on its law, then 0.

    Parameters
    ----------
    cell : PropertyCell
        A lasso cell.

    Returns
    -------
    tuple of float
        :data:`PROPERTY_GRID_SIZE` penalties, strictly decreasing, the last one 0.
    """
    return _grid(cell.dgp.name, cell.n)


# --------------------------------------------------------------------------- estimator


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    strategy = _strategy(cell.cell)
    options: dict[str, Any]
    if strategy == "lasso":
        options = {"strategy": "lasso", "lambdas": property_grid(cell)}
    elif strategy == "learners":
        options = {"strategy": "learners", "treatment_candidates": declared_learners()}
    else:
        options = {"strategy": "discrete", "candidates": ((),)}
    if cell.property == "multi_arm":
        # A binary outcome: the scaler is already the identity, so no ``q_bounds``.
        bounds: dict[str, Any] = {
            "reference": multi_arm_common.REFERENCE,
            "g_bounds": multi_arm_common.G_BOUNDS,
        }
    else:
        bounds = {"g_bounds": G_BOUNDS, "q_bounds": bounded_cv_laws.Q_BOUNDS}
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
        stratify_folds=STRATIFY_FOLDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        **bounds,
        **options,
    )


def sampling_batches() -> tuple[PropertyBatch, ...]:
    """The actual sampling calls, shared by complete and targeted regeneration."""
    return (PropertyBatch("properties", cells(), _estimator),)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return sampling_batches()[0].run(n_jobs=n_jobs)


# --------------------------------------------------------------------------- verdicts


def _rmse(cell: pd.DataFrame) -> float:
    error = cell["estimate"].to_numpy() - cell["truth"].to_numpy()
    return float(np.sqrt(np.mean(error**2)))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    # ``rmse_ratio`` goes through ``extra_columns``, so the rate rows are built with it.
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=("rmse_ratio",))

    necessity = rows.loc[rows["property"] == "selector_necessity"]
    control = necessity.loc[necessity["cell"] == "empty_control"].sort_values("replicate")
    ratios: dict[str, float] = {}
    for name in ("lasso_collaborative", "learners_collaborative"):
        chosen = necessity.loc[necessity["cell"] == name].sort_values("replicate")
        # Paired on the replication index: the cells share one seed, so the ratio divides
        # two error curves of the same draws.
        if not np.array_equal(chosen["replicate"], control["replicate"]):
            raise ValueError(f"the {name!r} selector cell is not paired with its control")
        ratios[name] = _rmse(chosen) / _rmse(control)
    mask = summary["property"] == "selector_necessity"
    for name, ratio in ratios.items():
        summary.loc[mask & (summary["cell"] == name), "rmse_ratio"] = ratio
    summary.loc[mask & (summary["role"] == "control"), "rmse_ratio"] = 1.0
    positive = mask & (summary["role"] == "positive")
    negative = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    summary.loc[mask, "property_passed"] = bool(
        summary.loc[mask, "passed"].all()
        and all(ratio <= SELECTOR_RMSE_RATIO for ratio in ratios.values())
    )

    multi = summary["property"] == "multi_arm"
    positive = multi & (summary["role"] == "positive")
    negative = multi & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    summary.loc[multi, "property_passed"] = bool(summary.loc[multi, "passed"].all())
    return finish(summary, rates)
