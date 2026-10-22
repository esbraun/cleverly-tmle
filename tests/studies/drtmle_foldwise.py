r"""Registered evidence for the fold-wise reduced regressions of cross-fitted DR-TMLE (X83).

The subject is :class:`~cleverly.estimators.DRTMLE` with ``cross_fit=True`` and
``reduced_crossfit="foldwise"``, the shipped default.  Fold ``k``'s reduced regressions train
on a slab of leave-two-out designs, tilted along the slab's own covariates at the pooled
fluctuation.  The registered DR-TMLE studies (``canonical-drtmle``,
``canonical-multi-arm-drtmle``, ``canonical-stratified-drtmle``) pin ``"pooled"``, which is the
construction R ``drtmle`` runs with ``cvFolds``.  In it, fold ``k``'s data reach fold ``k``'s
reduced regressions through the other folds' designs.  This study carries the default for the
rows whose registered evidence is pooled R parity (plan ``X83-plan.md`` section 7.3).

**No comparator.**  R ``drtmle`` has no fold-wise construction, so the study has
``reference=None`` and runs no R.

**The primary phase** is the ``canonical-drtmle`` design with one setting changed: the three
nuisance regimes of the Benkeser et al. (2017) Section 5.1 law at n = 3,000, the
canonical-drtmle learners, ten folds, the unstratified fold vector in the sample, and
``reduced_crossfit="foldwise"``.  :data:`SEED` **is** ``canonical-drtmle``'s seed, and the
scenarios are listed in its order, so replication ``r < 800`` of a scenario is that study's
registered replication ``r``: the same rows and the same fold vector.  A reader pairs the two
row files on ``(scenario, replicate)`` to read the pooled-minus-foldwise difference on the
registered draws (plan section 7.4, the RM18 re-read).  Replications 800 and above continue the
same seed stream.  :data:`RESAMPLING_SEED` is fresh, so every bootstrap and every property
sample of this study is drawn from a stream no other study uses.

**The property cells** are :mod:`tests.studies.drtmle_foldwise_properties`.  Their reading
and their budgets are declared there.

**Failures.**  The record declares ``failure_bound={IMPLEMENTATION: 0.01}`` and the property
module declares ``FAILURE_BOUND = 0.01``, in the form ``docs/development/method-benchmarking.md``
("Property fitting failures", "The bound") defines.  A fit that raises, a non-finite score
audit and an active propensity bound each count as a failed replicate.

**Cost.**  The sizing pilots' seconds per fit, on four busy cores: primary 3.8 (``tc3000``,
taken for ``both_correct`` and the both-wrong control too) and 3.3 (``oc3000``), leak pair 5.3
for both fits at n = 1,000, ``benkeser_order`` 4.2, ``bivariate`` 3.3, ``bivariate_oc`` 3.0,
``k5`` 1.8, ``spline_reduced`` 17.9, multi-arm 1.6, stratified 11.8.  At the declared budgets
that is about 59 CPU-hours: 18.3 primary, 6.5 leak, 14.0 spline, 10.5 stratified, and 10 for
the other cells.

**Declared before any run:** the budgets and seeds of this module and of the property module,
``publication_policy="gated"``, and the reading the property module states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly.learners.crossfit import random_partition
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_drtmle as canonical
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import initial_estimates, primary_rows

IMPLEMENTATION = "cleverly-drtmle-foldwise"
#: ``canonical-drtmle``'s seed, on purpose: with the scenarios in that study's order it makes
#: replication ``r`` of each scenario that study's registered replication ``r``.
#: ``tests/unit/test_drtmle_foldwise_study_design.py`` pins the rows equal.
SEED = canonical.SEED
#: Fresh entropy for every bootstrap and property stream: no registered study uses it.
RESAMPLING_SEED = 2026083183
PRIMARY_N = canonical.PRIMARY_N
SCENARIOS = canonical.SCENARIOS
ESTIMANDS = canonical.ESTIMANDS
#: The replications ``canonical-drtmle`` registered per scenario, which this study pairs with.
REGISTERED_REPLICATES = canonical.PRIMARY_REPLICATES

#: The primary budget per scenario, the output of the arm-pooled joint predictive rule
#: (:mod:`tests.studies.drtmle_foldwise_properties`, "Sizes").  The rows ``r < 800`` pair with
#: ``canonical-drtmle``.
PRIMARY_REPLICATES = 6_000

#: The share of a cell's replicates that a run may lose, declared before the first run.
FAILURE_BOUND = 0.01

#: The fold-wise construction this study is about, and the R parity construction its leak
#: control fits.
FOLDWISE = "foldwise"
POOLED = "pooled"

#: The flexible reduced learners of the ``spline_reduced`` cell (review F10): a cubic
#: B-spline basis of the scalar reduced design with :data:`SPLINE_KNOTS` knots, under least
#: squares for the reduced outcome regression and a near-unpenalized logistic fit for the
#: reduced mechanism.  The fit predicts in sample at the slab's training rows, which is the
#: second-order gap F10 names, and the basis has more freedom than the canonical GLM.
#:
#: **Why not capped boosting**, the plan's example.  On two pilot draws at n = 3,000,
#: ``HistGradientBoosting`` with 10 iterations of depth 2 reached the 100-round alternation
#: cap under both ``"foldwise"`` and ``"pooled"``, at about 415 seconds a fit, and a
#: 100-neighbour k-NN did the same at about 225 seconds.  A piecewise-constant reduced learner
#: leaves equation (10)'s solve near singular in both constructions, so the cell would measure
#: the cap and not the fold-wise slab.  The spline basis is smooth and converged on the pilot.
#: :data:`REDUCED_LEARNER_LIMIT` states the finding as a limit of the evidence.
SPLINE_KNOTS = 5
SPLINE_LOGISTIC_C = 1e4
#: The limit the piecewise-constant pilot fits set, published in :data:`CONFIGURATION`.
REDUCED_LEARNER_LIMIT = (
    "Piecewise-constant reduced learners (HistGradientBoosting with 10 iterations of depth 2, "
    "100-neighbour k-NN) reached max_outer = 100 under both foldwise and pooled reduced "
    "regressions on two pilot draws of the paper law at n = 3,000. The study covers smooth "
    "reduced learners only: the canonical GLMs and the spline basis."
)


@dataclass(frozen=True)
class Fit:
    """One DR-TMLE configuration on the paper law.

    Every field but the one a cell varies keeps the ``canonical-drtmle`` setting.

    Parameters
    ----------
    reduced_crossfit : str
        ``"foldwise"`` (the subject) or ``"pooled"`` (the leak control).
    update_order : str
        ``"drtmle"`` or ``"benkeser"``.
    reduction : str
        ``"univariate"`` or ``"bivariate"``.
    n_folds : int
        The outer folds.  The sample's fold vector is drawn at this count.
    reduced : str
        ``"glm"`` (the canonical Gaussian and binomial GLMs) or ``"spline"`` (the
        :data:`SPLINE_KNOTS` basis).
    """

    reduced_crossfit: str = FOLDWISE
    update_order: str = "drtmle"
    reduction: str = "univariate"
    n_folds: int = canonical.N_FOLDS
    reduced: str = "glm"


#: The primary configuration.
PRIMARY_FIT = Fit()


def _reduced_learners(fit: Fit) -> tuple[Any, Any]:
    if fit.reduced == "glm":
        return LinearRegression(), canonical.ColumnLogistic()
    if fit.reduced == "spline":
        return (
            make_pipeline(SplineTransformer(n_knots=SPLINE_KNOTS, degree=3), LinearRegression()),
            make_pipeline(
                SplineTransformer(n_knots=SPLINE_KNOTS, degree=3),
                LogisticRegression(C=SPLINE_LOGISTIC_C, max_iter=2000),
            ),
        )
    raise ValueError(f"unknown reduced learner {fit.reduced!r}")


def estimator(assignment: np.ndarray, scenario: str, fit: Fit = PRIMARY_FIT) -> Any:
    """The fixed-fold DR-TMLE of one nuisance regime under one configuration.

    Parameters
    ----------
    assignment : numpy.ndarray
        The sample's fold vector.
    scenario : str
        The nuisance regime, one of :data:`SCENARIOS` or ``"both_wrong"``.
    fit : Fit
        The configuration.

    Returns
    -------
    FixedFoldDRTMLE
        The unfitted estimator.
    """
    outcome, treatment = canonical._learners(scenario)
    reduced_outcome, reduced_treatment = _reduced_learners(fit)
    return canonical.FixedFoldDRTMLE(
        assignment,
        outcome_learner=outcome,
        treatment_learner=treatment,
        reduced_outcome_learner=reduced_outcome,
        reduced_treatment_learner=reduced_treatment,
        cross_fit=True,
        n_folds=fit.n_folds,
        stratify_folds=canonical.STRATIFY_FOLDS,
        estimands=ESTIMANDS,
        simultaneous=False,
        g_bounds=canonical.G_BOUNDS,
        max_outer=canonical.MAX_OUTER,
        max_iter=canonical.CONFIGURATION["max_iter"],
        tol=1e-10,
        random_state=0,
        guard=tuple(canonical.CONFIGURATION["guard"]),
        reduction=fit.reduction,
        reduced_crossfit=fit.reduced_crossfit,
        update_order=fit.update_order,
    )


def fit_cleverly(frame: pd.DataFrame, scenario: str, fit: Fit = PRIMARY_FIT) -> Any:
    """Fit one sample of the paper law and audit it as ``canonical-drtmle`` does.

    Parameters
    ----------
    frame : pandas.DataFrame
        One sample with its ``fold`` column.
    scenario : str
        The nuisance regime.
    fit : Fit
        The configuration.

    Returns
    -------
    Any
        The single fitted result.

    Raises
    ------
    RuntimeError
        When the score audit is non-finite or the propensity bound activated.
    """
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    if int(assignment.max()) + 1 != fit.n_folds:
        raise ValueError(
            f"the sample carries {int(assignment.max()) + 1} folds and the fit declares "
            f"{fit.n_folds}"
        )
    result = (
        estimator(assignment, scenario, fit)
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W12"])
        .single()
    )
    canonical.assert_unstratified_scheme(result)
    score = result.diagnostics.score_equations()
    if not np.isfinite([float(row.score) for row in score.rows]).all():
        raise RuntimeError("DR-TMLE empirical score audit is non-finite")
    raw = np.asarray(result.repeats[0].nuisance.propensity.values, dtype=float)
    bounded = result.repeats[0].nuisance.bounded_propensity(canonical.G_BOUNDS)
    if not np.array_equal(raw, bounded):
        raise RuntimeError("the propensity bound activated")
    return result


def draw_paper(scenario: str, n: int, seed: int, n_folds: int = canonical.N_FOLDS) -> Any:
    """One sample of the paper law with an unstratified fold vector of ``n_folds`` folds.

    At ten folds this is ``canonical-drtmle``'s own draw.  At another count the fold vector is
    redrawn from the same derived seed, so the rows stay the same.

    Parameters
    ----------
    scenario : str
        The nuisance regime; the law does not depend on it.
    n : int
        Rows.
    seed : int
        The sample seed.
    n_folds : int
        The fold count of the vector.

    Returns
    -------
    tuple of pandas.DataFrame and dict
        The sample and the exact truths.
    """
    frame, truths = canonical.draw_from_seed(scenario, n, seed)
    if n_folds != canonical.N_FOLDS:
        frame["fold"] = random_partition(n, n_folds, seed=seed + 1).assignment
    return frame, truths


# ---------------------------------------------------------------------- the registration
#: Filled by :mod:`tests.studies.drtmle_foldwise_properties`, which this module cannot import;
#: ``tests/unit/test_drtmle_foldwise_study_design.py`` pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "fold_locality": ("ate__foldwise", "ate__pooled"),
    "double_robustness": (
        "ate__benkeser_order",
        "ate__bivariate",
        "ate__bivariate_oc",
        "ate__k5",
        "ate__spline_reduced",
        "ate__both_wrong",
        "marginal_ate__stratified",
        "v0_ate__stratified",
        "v1_ate__stratified",
        "v2_ate__stratified",
        "multi_arm_drtmle__treatment_correct",
    ),
}

STUDY = StudyRecord(
    name="DR-TMLE with fold-wise reduced regressions",
    slug="drtmle-foldwise",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_foldwise",
    document="docs/technical-reference/method-evidence/dr-tmle-foldwise.md",
    anchor="dr-tmle-with-fold-wise-reduced-regressions",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/discrete_law.py",
        "tests/studies/drtmle_foldwise.py",
        "tests/studies/drtmle_foldwise_properties.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/multi_arm_drtmle_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.drtmle_foldwise",
    properties_module="tests.studies.drtmle_foldwise_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION: dict[str, Any] = {
    "source_law": "Benkeser et al. (2017), Section 5.1",
    "cross_fit": True,
    "n_folds": canonical.N_FOLDS,
    "reduced_crossfit": FOLDWISE,
    "reduction": "univariate",
    "guard": list(canonical.CONFIGURATION["guard"]),
    "update_order": "drtmle",
    "max_outer": canonical.MAX_OUTER,
    "max_iter": canonical.CONFIGURATION["max_iter"],
    "g_bounds": list(canonical.G_BOUNDS),
    "stratify_folds": canonical.STRATIFY_FOLDS,
    "folds": (
        "canonical-drtmle's unstratified ten-fold vector, drawn by cleverly.random_partition "
        "from the sample's seed alone"
    ),
    "draws": (
        "canonical-drtmle's seed and scenario order, so replication r < 800 of each scenario "
        "is that study's registered replication r"
    ),
    "nuisance_models": dict(canonical.CONFIGURATION["nuisance_models"]),
    "comparator": "none: Python property evidence, no comparator",
    "comparator_reason": "R drtmle has no fold-wise reduced-regression construction",
    "spline_reduced_learners": (
        f"cubic B-spline basis with {SPLINE_KNOTS} knots; least squares for the reduced "
        f"outcome regression, logistic with C = {SPLINE_LOGISTIC_C:g} for the reduced mechanism"
    ),
    "reduced_learner_limit": REDUCED_LEARNER_LIMIT,
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed: ``canonical-drtmle``'s own draw."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return canonical.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema."""
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(fit_cleverly(frame, scenario), truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication.

    A run at the declared budget is refused while a unit of the sizing rule waits on its
    fallback (:func:`tests.studies.drtmle_foldwise_properties.assert_sized`).
    """
    if replicates >= PRIMARY_REPLICATES:
        from tests.studies.drtmle_foldwise_properties import assert_sized

        assert_sized()
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for scenario in SCENARIOS for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])
