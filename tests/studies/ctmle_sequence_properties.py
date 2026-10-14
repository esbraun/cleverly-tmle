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
logistic regression on ``W2``, then on ``W1``.  ``treatment_learner`` is the terminal: the
main-terms logistic regression on every covariate.  On the instrument law of the
selector-necessity cells, ``W2`` is the instrument and ``W1`` the confounder, so index 0
leaves the confounding in place.  ``learners_collaborative`` passes only when the selector
steps past index 0.  ``tests/unit/test_ctmle_sequences_design.py`` checks that index 0 is
biased on that law and that index 1 is not.

**The outcome-correct law** (:func:`outcome_correct_dgp`, plan 3.4.3, B2) keeps the bounded
linear law's outcome terms and the linear law's propensity coefficients, and adds one bounded
interaction, ``f = tanh(W1) tanh(W2)``, to both mechanisms:
``g(W) = 0.05 + 0.90 expit(0.3 W1 - 0.2 W2 + 0.1 W3 + 2.0 f)`` and
``Qbar(a, W) = expit(-0.15 + 0.45 a + 0.30 (W1 + 0.5 W2 - 0.8 W3 + 0.4 W4) + 0.8 f)``, with
``Y ~ Beta(20 m, 20 (1 - m))``.  The correct outcome learner (:func:`correct_outcome_learner`)
appends ``f`` to the design and fits
:class:`~tests.studies.fractional_glm.QuasiBinomialGLM`, which specifies ``Qbar`` exactly.  No
candidate on either path reads ``f``, so every candidate mechanism is misspecified.
``tests/unit/test_ctmle_sequences_design.py`` checks both claims: the outcome learner's limit,
and that the main-terms projection leaves a TMLE with a constant ``Q`` biased by at least
1.0 SD at ``n = 700``, with a control law that drops ``f`` from ``g``.  The both-wrong controls
fit each strategy under test with a constant outcome on the same law.

**Verdicts.**  ``double_robustness`` uses the shared union-model rule.  ``selector_necessity``
pairs each collaborative cell with the empty-path control on one seed, and the family claim
needs every positive cell's bias inside the margin, the control's outside it, and the paired
bootstrap upper bound (:data:`RMSE_BOUND_LEVEL`) of each collaborative RMSE over the control's
at most :data:`SELECTOR_RMSE_RATIO`.  ``multi_arm`` has no RMSE clause: the positive cell's bias
is inside the margin and the control's is outside it.

**Selection.**  ``property-selection.csv`` records the selected and final index of every
lasso and learners fit (:func:`generate_property_artifacts`), so the page can report where
each path ended.

**Seeds** are the block ``16_100`` to ``16_400``, unused by every other property module.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from cleverly.datasets import DGP, instrument_dgp, linear_dgp
from cleverly.estimators import CTMLE
from cleverly.utils.bounds import expit
from cleverly.utils.parallel import map_parallel
from cleverly.validation import CoverageStudy
from cleverly.validation._seeds import sample_seed_streams
from tests.parallel import STUDY_JOBS
from tests.studies import bounded_cv_laws, multi_arm_common
from tests.studies.canonical_ctmle_sequences import (
    G_BOUNDS,
    PROPERTY_SELECTION_FILE,
    STRATIFY_FOLDS,
    STUDY,
    penalty_design,
)
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyBatch,
    PropertyCell,
    bootstrap_draw_blocks,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.seeds import stream_seed
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.multi_arm_properties import ESTIMAND as MULTI_ARM_ESTIMAND
from tests.studies.multi_arm_properties import Sampler

#: The binary selector study's constant: each collaborative RMSE over the control's.
SELECTOR_RMSE_RATIO = 0.50
#: The one-sided level of the paired bootstrap bound on that ratio (plan 3.4.3, S3, N3).
RMSE_BOUND_LEVEL = 0.99

#: The declared lasso grid of every property cell: this many penalties, the last one 0.
PROPERTY_GRID_SIZE = 12

#: The outcome-correct law (B2): the interaction's coefficient in the propensity logit and
#: in the outcome logit, and the beta precision.
OUTCOME_CORRECT_G_INTERACTION = 2.0
OUTCOME_CORRECT_Q_INTERACTION = 0.8
OUTCOME_CORRECT_CONCENTRATION = 20.0

#: The covariate columns, by position, of each declared learners candidate, in the declared
#: order.  Positions rather than names, because a candidate learner reads the covariate matrix.
#: Index 0 reads the instrument ``W2`` alone, so on the instrument law it omits the confounder
#: ``W1`` and is biased: a selector that never leaves index 0 fails the necessity cell.
LEARNER_COLUMNS = ((1,), (0,))

#: Replication budgets.  1,200 for double robustness and the multi-arm pair, 800 for the
#: selector-necessity triple, as declared before the run.
ROBUST_REPLICATES = 1_200
NECESSITY_REPLICATES = 800

DOUBLE_ROBUST_SEED = 16_100
NECESSITY_SEED = 16_200
MULTI_ARM_SEED = 16_300


# --------------------------------------------------------------------------- laws


def interaction(w: np.ndarray) -> np.ndarray:
    """The bounded interaction ``f = tanh(W1) tanh(W2)`` of the outcome-correct law.

    Parameters
    ----------
    w : ndarray of float
        The covariates ``W1`` to ``W4``, one row per unit.

    Returns
    -------
    ndarray of float
        One value per row, in ``(-1, 1)``.
    """
    return np.tanh(w[:, 0]) * np.tanh(w[:, 1])


@functools.cache
def outcome_correct_dgp(g_interaction: float = OUTCOME_CORRECT_G_INTERACTION) -> DGP:
    """The outcome-correct law of plan 3.4.3 (B2).

    Parameters
    ----------
    g_interaction : float
        The coefficient of ``f`` in the propensity logit.  The study uses
        :data:`OUTCOME_CORRECT_G_INTERACTION`.  The design test's control law sets it to 0.

    Returns
    -------
    DGP
        The law of the outcome-correct and both-wrong cells.
    """
    weights = np.asarray(bounded_cv_laws.LINEAR_WEIGHTS, dtype=float)

    def propensity(w: np.ndarray) -> np.ndarray:
        u = 0.3 * w[:, 0] - 0.2 * w[:, 1] + 0.1 * w[:, 2] + g_interaction * interaction(w)
        return np.asarray(0.05 + 0.90 * expit(u), dtype=float)

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        linear = (
            bounded_cv_laws.LINEAR_INTERCEPT
            + bounded_cv_laws.LINEAR_EFFECT * a
            + bounded_cv_laws.LINEAR_SCALE * (w[:, :4] @ weights)
            + OUTCOME_CORRECT_Q_INTERACTION * interaction(w)
        )
        return np.asarray(expit(linear), dtype=float)

    name = "x16_outcome_correct"
    if g_interaction != OUTCOME_CORRECT_G_INTERACTION:
        name = f"{name}(g_interaction={g_interaction:g})"
    return replace(
        bounded_cv_laws.linear_dgp(),
        name=name,
        propensity=propensity,
        outcome_mean=outcome_mean,
        concentration=OUTCOME_CORRECT_CONCENTRATION,
    )


def with_interaction(x: np.ndarray) -> np.ndarray:
    """Append ``f`` to an outcome design ``[A, W1, ..., W4]``.

    Parameters
    ----------
    x : array-like
        The design the package passes an outcome learner: ``A``, then the covariates.

    Returns
    -------
    ndarray of float
        The design with ``tanh(W1) tanh(W2)`` as its last column.
    """
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, interaction(x[:, 1:])])


def correct_outcome_learner() -> Pipeline:
    """The outcome learner that specifies the outcome-correct law exactly.

    Returns
    -------
    Pipeline
        ``with_interaction``, then :class:`~tests.studies.fractional_glm.QuasiBinomialGLM`.
    """
    return Pipeline(
        [("interaction", FunctionTransformer(with_interaction)), ("glm", QuasiBinomialGLM())]
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
    if cell.startswith("lasso_") or cell.endswith("_lasso"):
        return "lasso"
    if cell.startswith("learners_") or cell.endswith("_learners"):
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
            correct_outcome_learner,
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
    # Each control fits the strategy under test, with a constant outcome (plan 3.4.3).
    both_wrong = tuple(
        PropertyCell(
            "double_robustness",
            name,
            outcome_law,
            DummyRegressor,
            lambda: LogisticRegression(max_iter=1000),
            700,
            ROBUST_REPLICATES,
            seed,
            role="control",
        )
        for name, seed in (
            ("both_wrong_lasso", DOUBLE_ROBUST_SEED + 5),
            ("both_wrong_learners", DOUBLE_ROBUST_SEED + 6),
        )
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
        *both_wrong,
        *necessity,
        *multi_arm,
    )


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, so each committed truth is read back against its law.

    Returns
    -------
    tuple of PropertyCell
        The six double-robustness cells, the selector-necessity triple and the multi-arm
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


#: The columns of ``property-selection.csv``.  ``selected`` and ``final`` are 1-based, as
#: in ``r-selection.csv``, and ``final`` is missing where no final step ran.
SELECTION_COLUMNS = ("property", "cell", "replicate", "selected", "final", "n_candidates")

#: The selection of the last fit in this process, which :class:`_Recording` writes and
#: :func:`_replication` reads.  Each replication runs in one process, one fit at a time.
_LAST_SELECTION: dict[str, Any] = {}


@dataclass(frozen=True)
class _Recording:
    """An estimator that fits ``inner`` and records the selection of its result.

    Parameters
    ----------
    inner : Any
        The estimator.
    """

    inner: Any

    def fit(self, *arguments: Any, **keywords: Any) -> Any:
        _LAST_SELECTION.clear()
        result = self.inner.fit(*arguments, **keywords)
        selection = result.single().extra["ctmle"]
        _LAST_SELECTION.update(
            selected=int(selection.selected) + 1,
            final=math.nan if selection.final is None else int(selection.final) + 1,
            n_candidates=len(selection.path),
        )
        return result


def _recorded(factory: Any) -> Any:
    return lambda: _Recording(factory())


def _replication(study: CoverageStudy, index: int, seed: int) -> tuple[Any, dict[str, Any]]:
    outcome = study.run_replication(index, seed)
    return outcome, dict(_LAST_SELECTION)


@dataclass(frozen=True)
class SelectionBatch(PropertyBatch):
    """The property batch, with the selection of every lasso and learners fit.

    :meth:`run_with_selection` fits each replication once.  Its rows are those
    :func:`~tests.studies.evidence.properties.run_cells` returns for the same cells, and it
    also returns ``property-selection.csv``.
    """

    def run_with_selection(self, *, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the batch, and record each lasso and learners fit's selection.

        Parameters
        ----------
        n_jobs : int
            Worker processes.

        Returns
        -------
        tuple of DataFrame
            The property rows, and :data:`SELECTION_COLUMNS` rows.
        """
        budgets: dict[int, int] = {}
        for cell in self.cells:
            budgets[cell.seed] = max(budgets.get(cell.seed, 0), cell.replicates)
        sample_seeds = sample_seed_streams(budgets)
        frames: list[pd.DataFrame] = []
        selections: list[dict[str, Any]] = []
        for cell in self.cells:
            records_selection = _strategy(cell.cell) != "discrete"
            factory = self.estimator(cell)
            study = CoverageStudy(
                dgp=cell.dgp,
                estimator=_recorded(factory) if records_selection else factory,
                n=cell.n,
                n_replicates=cell.replicates,
                estimands=(cell.estimand,),
                fit_kwargs=dict(cell.fit_kwargs),
                seed=cell.seed,
                sample_seeds=sample_seeds[cell.seed][: cell.replicates],
                n_jobs=n_jobs,
            )
            outcomes = map_parallel(
                _replication,
                [
                    (study, index, int(seed))
                    for index, seed in enumerate(sample_seeds[cell.seed][: cell.replicates])
                ],
                n_jobs=n_jobs,
            )
            records = []
            failed = 0
            for outcome, selection in outcomes:
                if not isinstance(outcome, tuple):
                    failed += 1
                    continue
                kept = [record for record in outcome if record.estimand == cell.estimand]
                records.extend(kept)
                if records_selection and kept:
                    selections.append(
                        {
                            "property": cell.property,
                            "cell": cell.cell,
                            "replicate": kept[0].replicate,
                            **selection,
                        }
                    )
            if not records:
                raise RuntimeError(f"every replication of {cell.property}/{cell.cell} failed")
            frames.append(
                pd.DataFrame(
                    {
                        "property": cell.property,
                        "cell": cell.cell,
                        "role": cell.role,
                        "replicate": [record.replicate for record in records],
                        "n": cell.n,
                        "requested_replicates": cell.replicates,
                        "failed_replicates": failed,
                        "truth": [record.truth for record in records],
                        "estimate": [record.estimate for record in records],
                        "std_error": [record.std_error for record in records],
                        "covered": [int(record.covered) for record in records],
                        "rejected": [int(record.rejected) for record in records],
                    }
                )
            )
        rows = pd.concat(frames, ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]
        return rows, pd.DataFrame(selections, columns=list(SELECTION_COLUMNS))


def sampling_batches() -> tuple[PropertyBatch, ...]:
    """The actual sampling calls, shared by complete and targeted regeneration."""
    return (SelectionBatch("properties", cells(), _estimator),)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return sampling_batches()[0].run(n_jobs=n_jobs)


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The property rows and ``property-selection.csv``, from one set of fits.

    Parameters
    ----------
    n_jobs : int
        Worker processes.

    Returns
    -------
    tuple
        The property rows, and the side artefact by file name.
    """
    batch = sampling_batches()[0]
    assert isinstance(batch, SelectionBatch)
    rows, selection = batch.run_with_selection(n_jobs=n_jobs)
    return rows, {PROPERTY_SELECTION_FILE: selection}


# --------------------------------------------------------------------------- verdicts


def _rmse(cell: pd.DataFrame) -> float:
    error = cell["estimate"].to_numpy() - cell["truth"].to_numpy()
    return float(np.sqrt(np.mean(error**2)))


def rmse_ratio_bound(chosen: pd.DataFrame, control: pd.DataFrame, *, seed: int) -> float:
    """The paired bootstrap upper bound on one RMSE ratio, at :data:`RMSE_BOUND_LEVEL`.

    The resampling unit is the replication index, so each draw keeps a collaborative error
    beside the control's error on the same sample (plan 3.4.3, S3).

    Parameters
    ----------
    chosen : DataFrame
        The collaborative cell's rows, sorted by replicate.
    control : DataFrame
        The control's rows on the same replicates, sorted by replicate.
    seed : int
        The record's resampling stream for this cell.

    Returns
    -------
    float
        The :data:`RMSE_BOUND_LEVEL` quantile of the bootstrap ratios.
    """
    squared = np.column_stack(
        [
            (chosen["estimate"].to_numpy() - chosen["truth"].to_numpy()) ** 2,
            (control["estimate"].to_numpy() - control["truth"].to_numpy()) ** 2,
        ]
    )
    rng = np.random.default_rng(seed)
    ratios = []
    for block in bootstrap_draw_blocks(
        squared, replicates=STUDY.margins.bootstrap_replicates, rng=rng
    ):
        means = block.mean(axis=1)
        ratios.append(np.sqrt(means[:, 0] / means[:, 1]))
    return float(np.quantile(np.concatenate(ratios), RMSE_BOUND_LEVEL))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    # ``rmse_ratio`` and its bound go through ``extra_columns``, so the rate rows carry them.
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("rmse_ratio", "rmse_ratio_upper")
    )

    necessity = rows.loc[rows["property"] == "selector_necessity"]
    control = necessity.loc[necessity["cell"] == "empty_control"].sort_values("replicate")
    ratios: dict[str, tuple[float, float]] = {}
    for name in ("lasso_collaborative", "learners_collaborative"):
        chosen = necessity.loc[necessity["cell"] == name].sort_values("replicate")
        # Paired on the replication index: the cells share one seed, so the ratio divides
        # two error curves of the same draws.
        if not np.array_equal(chosen["replicate"], control["replicate"]):
            raise ValueError(f"the {name!r} selector cell is not paired with its control")
        bound = rmse_ratio_bound(
            chosen, control, seed=stream_seed(STUDY, "selector_necessity", name, "rmse_ratio")
        )
        ratios[name] = (_rmse(chosen) / _rmse(control), bound)
    mask = summary["property"] == "selector_necessity"
    for name, (ratio, bound) in ratios.items():
        summary.loc[mask & (summary["cell"] == name), "rmse_ratio"] = ratio
        summary.loc[mask & (summary["cell"] == name), "rmse_ratio_upper"] = bound
    # The control is the denominator, so it has no ratio of its own.
    summary.loc[mask & (summary["role"] == "control"), ["rmse_ratio", "rmse_ratio_upper"]] = (
        math.nan
    )
    positive = mask & (summary["role"] == "positive")
    negative = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    summary.loc[mask, "property_passed"] = bool(
        summary.loc[mask, "passed"].all()
        and all(bound <= SELECTOR_RMSE_RATIO for _, bound in ratios.values())
    )

    multi = summary["property"] == "multi_arm"
    positive = multi & (summary["role"] == "positive")
    negative = multi & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[negative, "passed"] = summary.loc[negative, "bias_discriminated"]
    summary.loc[multi, "property_passed"] = bool(summary.loc[multi, "passed"].all())
    return finish(summary, rates)
