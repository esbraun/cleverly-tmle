r"""Registered evidence for the centred fold cluster variance at few clusters per fold (X54 (e)).

A fold-evaluated ``TMLE`` (``cv_evaluation=True``) with ``id=`` reports the variance

.. math::

    \hat\sigma^2 = V^{-2}\sum_v J_v\,s_v^2/n_v^2,

with :math:`s_v^2` the divisor-:math:`(J_v-1)` variance of fold :math:`v`'s cluster totals of
the fold-centred curve (:func:`cleverly.inference.cluster.cross_validated_variance`), and its
interval on :math:`t_{J-V}`. Under i.i.d. cluster totals and equal folds,
:math:`E[s_v^2]` equals the totals' variance whatever their law, so the variance ratio
:math:`E[\hat\sigma^2]/\mathrm{Var}(\hat\psi)` has target 1 at every number of clusters per
fold, while the SE ratio has target :math:`c_k = E\sqrt{\chi^2_k/k} < 1` with :math:`k = J - V`
even under normality. The study therefore gates the variance ratio's lower end, against the
square of the calibration band's lower end, with the two-sided coverage of the :math:`t_{J-V}`
interval over the shown fits and the all-draw coverage floor G1 (plan X54 rev 3.2, section 2.4).
The variance ratio's upper side is reported: a fold whose nuisances are fitted on few clusters
occasionally reports a very large SE, which raises the mean of :math:`\hat\sigma^2` without
moving coverage.

**Property ``fold_variance_calibration``.** Twelve cells (:data:`CELLS`): two laws by
:math:`(V, J) \in \{(2,4), (2,8), (2,12), (5,10), (5,20), (5,30)\}`, which is 2, 4 and 6
clusters per validation fold at each :math:`V`. Cells are read by the clusters of each training
complement, :math:`J(V-1)/V` = 2, 4, 6, 8, 16 and 24 (:attr:`Cell.training`), because the pilot
showed the limit is the clusters the nuisances are fitted on, not the validation fold. Every cluster has the same size, so
``cv_evaluation=True`` runs (Fix-4 refuses unequal sizes):

* ``equal10``: 10 rows per cluster from the law of ``clustered_dgp(10, "binomial")``, which
  :func:`tests.studies.clustered_unequal_laws.draw_informative` reproduces with the arm-by-latent
  coefficient 8, no size terms and fixed sizes (the ``equal10`` law of
  ``clustered-few-cluster-tmle``);
* ``cluster_covariate``: 30 rows per cluster with ``W1`` shared within a cluster
  (:func:`tests.studies.clustered_unequal_laws.draw_cluster_covariate`), the X24 probe law,
  where the plug-in part of the curve carries the cluster effect.

The ``equal10`` cells measure the ATE and the ``cluster_covariate`` cells ``ey1``
(:data:`TARGETS`): the arm mean carries the plug-in part, and with it the cluster effect, which
the ATE nearly cancels on that law. Each draw is fitted with ``TMLE(cv_evaluation=True)`` and
``id=``, ``V`` grouped folds (``random_partition`` of the cluster labels), an unpenalized
logistic outcome regression and an unpenalized logistic propensity, both in ``W1`` and ``W2``.
Eight arms are published (:data:`ARMS`); the first six share that fit's point estimate:

==========================  ========================================================================
arm                         variance and reference
==========================  ========================================================================
``t_reference``             the shipped fold variance, :math:`t_{J-V}` (positive)
``iid_t_control``           the same construction with rows as units (the uncentred fold second
                            moment), :math:`t_{J-V}` (control: its median paired SE ratio to the
                            positive arm must lie below 0.95)
``deflated_control``        the shipped variance times :data:`DEFLATION`, :math:`0.80^2`
                            (control: it must fail the positive rule)
``pooled_centered``         the stacked cluster variance of the same fit, centred at the pooled
                            point, :math:`t_{J-V}` (diagnostic; :data:`POOLED_CENTERED_GATED`
                            states why no cell gates it)
``t_j_minus_2_reference``   the shipped variance on :math:`t_{J-2}` (diagnostic)
``normal_reference``        the shipped variance on the normal quantile (diagnostic)
``known_propensity``        a second fit of the same draw with the law's true propensity declared
                            known, its fold variance on :math:`t_{J-V}` (positive: the second
                            gated configuration, with its own training floor)
``known_deflated_control``  the second fit's variance times :data:`DEFLATION` (control: it must
                            fail the positive rule)
==========================  ========================================================================

The fit reports ``"few_cluster_plugin"`` below 10 clusters, which shows a labelled interval;
the study measures that interval. A fit that withholds its interval (a constant binary outcome
in an arm, or an estimate outside its bounds) is a miss in its arms and is left out of their
shown-fit statistics, and its share is published. A training complement with one outcome value
takes the constant-target rule; one without an arm is refused, and the draw is lost and counts
as a miss in G1. The rules are those of
:mod:`tests.studies.few_cluster_fold_variance_properties`.

**Primary.** The same fit on ``equal10`` at :math:`V = 5`, :math:`J = 20` (200 rows), 1,000
replications, reporting ``ey0``, ``ey1`` and ``ate``. No comparator implements a
fold-evaluated cluster variance, so the record declares no reference and publishes an empty
equivalence artifact. The policy is ``reporting``, declared before any run.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, t
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.inference.cluster import cross_validated_variance
from cleverly.learners.crossfit import check_integrity
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

PRIMARY_REPLICATES: Final[int] = 1_000
SEED: Final[int] = 20265451
RESAMPLING_SEED: Final[int] = 20265452
FAMILY: Final[str] = "fold_variance_calibration"
ESTIMANDS: Final[tuple[str, ...]] = ("ey0", "ey1", "ate")
#: The propensity bounds of a fitted treatment regression, as the canonical DR-TMLE study uses.
#: Both laws' true propensities reach 0.01 only beyond 6 SD of their logits.
G_BOUNDS: Final[tuple[float, float]] = (0.01, 0.99)
#: The multiplier of each deflated control's variance: :math:`0.80^2`, an SE 20% too small,
#: outside the SE-ratio band. Sized from the committed pilot columns before the run (plan rev
#: 3.4): at :math:`0.85^2` the estimated ``equal10__v2_j12`` control failed the rule with
#: probability 0.55, because the positive there is conservative (variance ratio 1.31, coverage
#: 0.955); at :math:`0.80^2` every control of both configurations fails it with plug-in
#: probability 1.000.
DEFLATION: Final[float] = 0.80**2

LAWS: Final[tuple[str, ...]] = ("equal10", "cluster_covariate")
#: The estimand each law's cells measure. On ``cluster_covariate`` the cluster effect sits in the
#: plug-in part of the curve, which the arm mean carries (as in the covariate pair of
#: ``clustered-unequal-cvtmle``) and the ATE nearly cancels: the pilot's paired IID/cluster SE
#: ratio of the ATE read 0.97 to 1.06 there, so no IID control could fail on it.
TARGETS: Final[dict[str, str]] = {"equal10": "ate", "cluster_covariate": "ey1"}
ROWS_PER_CLUSTER: Final[dict[str, int]] = {
    "equal10": 10,
    "cluster_covariate": laws.COVARIATE_LAW_ROWS,
}
#: ``(V, J)``: 2, 4 and 6 clusters per fold at V = 2 and at V = 5.
DESIGNS: Final[tuple[tuple[int, int], ...]] = ((2, 4), (2, 8), (2, 12), (5, 10), (5, 20), (5, 30))

#: The published arms. :func:`arm_role` gives each one's role.
ARMS: Final[tuple[str, ...]] = (
    "t_reference",
    "iid_t_control",
    "deflated_control",
    "pooled_centered",
    "t_j_minus_2_reference",
    "normal_reference",
    "known_propensity",
    "known_deflated_control",
)


@dataclasses.dataclass(frozen=True)
class Cell:
    """One property cell: a law, a fold count and a cluster count.

    Parameters
    ----------
    law : str
        One of :data:`LAWS`.
    folds : int
        The validation folds ``V``.
    clusters : int
        The clusters per draw ``J``, a multiple of ``V``.
    """

    law: str
    folds: int
    clusters: int

    @property
    def name(self) -> str:
        """The cell's published prefix."""
        return f"{self.law}__v{self.folds}_j{self.clusters}"

    @property
    def per_fold(self) -> int:
        """The clusters of each validation fold, ``J / V``."""
        return self.clusters // self.folds

    @property
    def training(self) -> int:
        """The clusters of each training complement, ``J (V - 1) / V``: the axis cells are read on."""
        return self.clusters - self.per_fold

    @property
    def df(self) -> int:
        """The reference degrees of freedom, ``J - V``."""
        return self.clusters - self.folds

    @property
    def n(self) -> int:
        """The rows of one draw."""
        return self.clusters * ROWS_PER_CLUSTER[self.law]


CELLS: Final[tuple[Cell, ...]] = tuple(
    Cell(law, folds, clusters) for law in LAWS for folds, clusters in DESIGNS
)
CELL_BY_NAME: Final[dict[str, Cell]] = {cell.name: cell for cell in CELLS}
#: The training-cluster counts of the design, the ladder each configuration's floor is read on.
TRAINING_LADDER: Final[tuple[int, ...]] = tuple(sorted({cell.training for cell in CELLS}))
#: The two gated configurations (plan 5.3, rev 3.2): the propensity estimated, and the same
#: draw refitted with the law's propensity declared known (``treatment_probabilities=``). Each
#: takes its own training floor.
POSITIVE_ARMS: Final[tuple[str, ...]] = ("t_reference", "known_propensity")
#: The mechanism each positive arm stands for, the key of a training floor.
MECHANISMS: Final[dict[str, str]] = {"t_reference": "estimated", "known_propensity": "known"}
#: Each positive arm's must-fail control: its variance times :data:`DEFLATION`.
DEFLATED_ARMS: Final[dict[str, str]] = {
    "t_reference": "deflated_control",
    "known_propensity": "known_deflated_control",
}
PRIMARY_CELL: Final[Cell] = CELL_BY_NAME["equal10__v5_j20"]
PRIMARY_N: Final[int] = PRIMARY_CELL.n
SCENARIO: Final[str] = PRIMARY_CELL.name

#: The cells whose ``pooled_centered`` arm is a gated must-fail control: none. The plan gates it
#: in the cells with 2 clusters per fold where the pilot reads its variance ratio above
#: :data:`POOLED_CENTERED_PILOT_TRIGGER`, expecting re-centring at the pooled point to add the
#: between-fold spread, a relative excess of about :math:`(1-1/V)/(J_v-1)`. It adds none in
#: expectation: each fold curve is centred at its own fold point, so the shift to the pooled
#: point adds the between-fold spread, :math:`(V-1)\tau^2` in sum, which replaces the
#: :math:`V-1` degrees of freedom the fold centring removed, and the pooled divisor
#: :math:`J-1` returns :math:`\tau^2`. The pilot's pooled-centred variance ratio over the fold
#: one reads 0.71 to 1.01 in every cell. The two cells with 2 training clusters read pooled ratios
#: of 6.1 and 12.4 from the same tail that puts the fold ratio at 7.6 and 17.4, so a gate there
#: could not see a mis-centring. The decision was recorded from the pilot before the declaration.
POOLED_CENTERED_PILOT_TRIGGER: Final[float] = 1.25
POOLED_CENTERED_GATED: Final[frozenset[str]] = frozenset()

#: The cells whose pilot was extended from 400 to 1,600 draws: the plug-in pass probability at
#: the 400-draw readings was below 0.90 (the calibration rule of the plans guide).
PILOT_EXTENDED: Final[frozenset[str]] = frozenset(
    {
        "equal10__v2_j4",
        "equal10__v2_j8",
        "equal10__v2_j12",
        "equal10__v5_j10",
        "equal10__v5_j20",
        "cluster_covariate__v2_j4",
        "cluster_covariate__v2_j8",
        "cluster_covariate__v5_j20",
    }
)
#: The positive cells of each configuration whose plug-in pass probability at 8,000 draws is
#: below 0.90 at the pilot's own readings (rev 3.2's three gates, with the pilot's draws a refusal
#: lost refitted under the constant-target rule): predicted reds, recorded before the
#: declaration. ``regenerate.py`` gives each one's reading and route.
PREDICTED_RED: Final[dict[str, frozenset[str]]] = {
    "t_reference": frozenset({"equal10__v2_j4", "cluster_covariate__v2_j4"}),
    "known_propensity": frozenset({"equal10__v2_j4", "equal10__v2_j8", "cluster_covariate__v2_j4"}),
}
#: Each configuration's training floor predicted from :data:`PREDICTED_RED`: the smallest count
#: of :data:`TRAINING_LADDER` with no predicted red at or above it (plan 5.3 predicts 4 and 6).
PREDICTED_FLOOR: Final[dict[str, int]] = {"t_reference": 4, "known_propensity": 6}
#: The deflated controls whose power to fail the positive rule is below 0.90 at the pilot's
#: readings: the estimated propensity's outlier SEs on 6 training clusters leave ``equal10`` a
#: variance ratio of 1.31, so the deflated variance still reads 0.94 and its coverage 0.927.
#: The red-cell owner of a positive cell below its configuration's training floor (the roadmap's
#: Red-cell owners table): it holds that cell's coverage-band and G1 legs under one of
#: :data:`TRAINING_FLOOR_READING` and :data:`FINITE_SAMPLE_READING`. A red variance-ratio lower
#: end, a red at 24 without a recorded diagnosis, or a control that does not do its job is a
#: defect suspect or an unlicensed configuration (:func:`~tests.studies.few_cluster_fold_variance_properties.declared_route`).
RED_OWNER: Final[str] = "X54-training-floor"
#: The owner's two readings (plan 5.8, rev 3.5). A red below the configuration's
#: :data:`PREDICTED_FLOOR`, with a passing variance-ratio lower end, reads as the training-floor
#: limit the pilot showed. A red at or above it, or a red count above a passing count of the
#: same configuration (the non-monotone signature: the training-complement mechanism is monotone
#: in the count), reads as a finite-sample miss: the plug-in figures size the run, and the
#: predictive figures of :data:`PREDICTIVE_FAMILY` (0.718 and 0.646) give such a red that chance.
TRAINING_FLOOR_READING: Final[str] = "training-floor limit"
FINITE_SAMPLE_READING: Final[str] = "finite-sample (plug-in sizing)"
#: The recorded diagnosis flag, per configuration, that a red at 24 training clusters (so no
#: training floor) is a finite-sample miss and no defect: the record of the diagnosis that found
#: it. Empty before the run. Only an entry here turns that configuration's ``defect suspect`` into
#: ``no floor (finite-sample)``
#: (:func:`~tests.studies.few_cluster_fold_variance_properties.declared_route`).
NO_DEFECT_DIAGNOSES: Final[dict[str, str]] = {}
#: The owner of each predicted red, declared before the run.
PREDICTED_RED_OWNERS: Final[dict[str, str]] = {
    cell_name: RED_OWNER
    for arm, cells in PREDICTED_RED.items()
    for cell_name in (f"{cell}__{arm}" for cell in sorted(cells))
}
#: The predictive family figures (plan section 5): the shown coverage of each cell integrated
#: over its Jeffreys posterior from the pilot, the variance ratio at its plug-in value, over the
#: cells at or above each predicted floor. The rule gates the plug-in figure; a chance red at 16 or
#: 24 training clusters is read against these.
PREDICTIVE_FAMILY: Final[dict[str, float]] = {"t_reference": 0.718, "known_propensity": 0.646}

#: Replications of every cell (plan 5.3, rev 3.4): 8,000, the plan's ceiling, where the plug-in
#: family product of the cells above each configuration's predicted floor, controls included,
#: reads 1.000 (estimated) and 0.997 (known), against 0.970 and 0.847 at 4,000.
PROPERTY_REPLICATES: Final[dict[str, int]] = {cell.name: 8_000 for cell in CELLS}


def cell_name(cell: str, arm: str) -> str:
    """The published name of one property cell."""
    return f"{cell}__{arm}"


def arm_role(cell: str, arm: str) -> str:
    """The role of one arm of one cell."""
    if arm in POSITIVE_ARMS:
        return "positive"
    if arm == "iid_t_control" or arm in DEFLATED_ARMS.values():
        return "control"
    if arm == "pooled_centered" and cell in POOLED_CENTERED_GATED:
        return "control"
    return "diagnostic"


PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(cell.name, arm) for cell in CELLS for arm in ARMS
)

STUDY = StudyRecord(
    name="the centred fold cluster variance at few clusters per fold",
    slug="few-cluster-fold-variance",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_fold_variance",
    document="docs/technical-reference/method-evidence/few-cluster-fold-variance.md",
    anchor="the-centred-fold-cluster-variance-at-few-clusters-per-fold",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": 0.05},
    modules=(
        "tests/studies/few_cluster_fold_variance.py",
        "tests/studies/few_cluster_fold_variance_properties.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.few_cluster_fold_variance",
    properties_module="tests.studies.few_cluster_fold_variance_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "TMLE(cv_evaluation=True) with id=: the centred fold cluster variance on "
    "t with J - V degrees of freedom",
    "laws": {
        "equal10": "clustered_dgp(10, 'binomial'): 10 rows per cluster, a cluster latent with "
        "arm-by-latent coefficient 8",
        "cluster_covariate": f"{laws.COVARIATE_LAW_ROWS} rows per cluster, W1 shared within a "
        "cluster (the X24 probe law)",
    },
    "designs": [list(design) for design in DESIGNS],
    "folds": "grouped random_partition folds from the cluster labels and random_state=0; "
    "J / V clusters in every fold",
    "targeting_scheme": "pooled",
    "outcome_learner": "unpenalized logistic regression (C=1e6) in W1 and W2",
    "treatment_learner": "unpenalized logistic regression (C=1e6) in W1 and W2",
    "g_bounds": list(G_BOUNDS),
    "simultaneous_intervals": False,
    "deflation": DEFLATION,
    "red_owner": RED_OWNER,
    "targets": dict(TARGETS),
    "known_propensity": "a second fit per draw with the law's true propensity declared known "
    "through treatment_probabilities= "
    "(the second gated configuration)",
    "training_ladder": list(TRAINING_LADDER),
    "primary": f"{SCENARIO}: ey0, ey1 and ate",
}


# ------------------------------------------------------------------------------------ laws


def law_truth(law: str) -> dict[str, float]:
    """The exact ``ey0``, ``ey1`` and ``ate`` of ``law``."""
    if law == "equal10":
        truth = laws.informative_truth(delta=0.0, gamma=0.0, effect_modifier=8.0)
        return {name: float(truth[name]) for name in ESTIMANDS}
    if law == "cluster_covariate":
        return {name: float(value) for name, value in laws.covariate_truth().items()}
    raise KeyError(law)


def draw(law: str, clusters: int, rng: np.random.Generator) -> pd.DataFrame:
    """One sample of ``clusters`` equal clusters of ``law``."""
    if law == "equal10":
        return laws.draw_informative(
            clusters,
            rng,
            sizes=np.full(clusters, ROWS_PER_CLUSTER[law]),
            delta=0.0,
            gamma=0.0,
            effect_modifier=8.0,
        )
    if law == "cluster_covariate":
        return laws.draw_cluster_covariate(clusters, rng)
    raise KeyError(law)


# ------------------------------------------------------------------------------------ fits


def learner() -> LogisticRegression:
    """The unpenalized main-terms logistic regression of both nuisances."""
    return LogisticRegression(C=1e6, max_iter=1000)


class ExactPropensity:
    """A law's true propensity in ``W1`` and ``W2``.

    ``treatment_probabilities=`` reads it as two frame columns; the pilot read it through
    ``OracleTreatment``, which the design test shows gives the same fit.

    Parameters
    ----------
    law : str
        One of :data:`LAWS`.
    """

    def __init__(self, law: str) -> None:
        self.law = law

    def propensity(self, covariates: Any) -> np.ndarray:
        """The true propensity of each row."""
        x = np.asarray(covariates, dtype=float)
        if self.law == "equal10":
            return laws.informative_propensity(x[:, 0], x[:, 1])
        return laws.covariate_propensity(x[:, 0], x[:, 1])


def fit(
    frame: pd.DataFrame,
    folds: int,
    estimands: Sequence[str],
    *,
    known: str | None = None,
    oracle: bool = False,
) -> Any:
    """The fold-evaluated ``TMLE`` of one draw, with ``id=`` and ``folds`` equal folds.

    ``known`` names the law whose true propensity is declared through
    ``treatment_probabilities=`` (the ``known_propensity`` configuration), so no treatment
    learner runs; ``None`` fits the propensity. ``oracle=True`` supplies the same propensity as an
    ``OracleTreatment`` learner instead, the form the pilot ran; the design test shows the two
    agree.
    """
    columns: dict[str, Any] = {}
    treatment: Any = learner()
    if known is not None:
        probability = ExactPropensity(known).propensity(frame[["W1", "W2"]].to_numpy())
        if oracle:
            treatment = OracleTreatment(ExactPropensity(known))
        else:
            frame = frame.assign(_p1=probability, _p0=1.0 - probability)
            columns = {"treatment_probabilities": {1: "_p1", 0: "_p0"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                outcome_learner=learner(),
                treatment_learner=treatment,
                cross_fit=True,
                n_folds=folds,
                cv_evaluation=True,
                targeting_scheme="pooled",
                stratify_folds="none",
                estimands=tuple(estimands),
                simultaneous=False,
                g_bounds=G_BOUNDS,
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                id="cluster",
                **columns,
            )
            .single()
        )
    cluster = np.asarray(frame["cluster"])
    check_integrity(result.nuisance.folds, cluster=cluster)
    per_fold = {int(np.unique(cluster[test]).size) for _, test in result.nuisance.folds}
    if len(per_fold) != 1:  # pragma: no cover - a design contract: equal folds
        raise AssertionError(f"unequal validation folds: {sorted(per_fold)} clusters")
    return result


def t_critical(df: int, alpha: float = STUDY.margins.alpha) -> float:
    """The two-sided t quantile with ``df`` degrees of freedom."""
    return float(t.ppf(1.0 - alpha / 2.0, df))


NORMAL_CRITICAL: Final[float] = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def jensen_factor(k: int) -> float:
    r""":math:`c_k = E\sqrt{\chi^2_k/k}`, the SE ratio of an exactly calibrated variance on
    :math:`k` degrees of freedom under normality."""
    return float(np.sqrt(2.0 / k) * np.exp(gammaln((k + 1) / 2) - gammaln(k / 2)))


def _checked(result: Any, cell: Cell) -> Any:
    estimate = result[TARGETS[cell.law]]
    if estimate.reference_df != cell.df:
        raise AssertionError(
            f"{cell.name}: the fit reports reference_df {estimate.reference_df}, not J - V"
        )
    return estimate


def draw_arms(frame: pd.DataFrame, cell: Cell, truth: float) -> dict[str, dict[str, Any]]:
    """The arms of :data:`ARMS` of one draw, off two fits.

    The first six read the study's fit; ``known_propensity`` and ``known_deflated_control`` read
    a second fit of the same draw with the law's true propensity declared known. Each arm holds its point, standard error, interval,
    ``withheld`` (the status of an estimate that withholds its interval, else ``""``),
    ``reference_df``, ``covered`` (0 when withheld) and ``rejected``.
    """
    target = TARGETS[cell.law]
    result = fit(frame, cell.folds, (target,))
    estimate = _checked(result, cell)
    known = _checked(fit(frame, cell.folds, (target,), known=cell.law), cell)
    fold_se = float(estimate.plugin_std_error)
    folds = [test for _, test in result.nuisance.folds]
    iid_se = float(np.sqrt(cross_validated_variance(estimate.influence_curve, folds, None)))
    pooled_se = float(np.sqrt(result.cv_targeting.pooled[target].variance))
    known_se = float(known.plugin_std_error)
    t_df = t_critical(cell.df)
    versions = {
        "t_reference": (estimate, fold_se, t_df),
        "iid_t_control": (estimate, iid_se, t_df),
        "deflated_control": (estimate, fold_se * np.sqrt(DEFLATION), t_df),
        "pooled_centered": (estimate, pooled_se, t_df),
        "t_j_minus_2_reference": (estimate, fold_se, t_critical(cell.clusters - 2)),
        "normal_reference": (estimate, fold_se, NORMAL_CRITICAL),
        "known_propensity": (known, known_se, t_df),
        "known_deflated_control": (known, known_se * np.sqrt(DEFLATION), t_df),
    }
    out: dict[str, dict[str, Any]] = {}
    for arm, (version, se, critical) in versions.items():
        shown = bool(version.supplies_inference)
        psi = float(version.psi)
        low, high = psi - critical * se, psi + critical * se
        out[arm] = {
            "psi": psi,
            "std_error": float(se),
            "low": float(low),
            "high": float(high),
            "withheld": "" if shown else str(version.inference),
            "reference_df": int(version.reference_df),
            "covered": int(shown and low <= truth <= high),
            "rejected": int(shown and not low <= 0.0 <= high),
        }
    return out


# --------------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw: ``n / 10`` clusters of ``equal10``."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    clusters = n // ROWS_PER_CLUSTER[PRIMARY_CELL.law]
    return draw(PRIMARY_CELL.law, clusters, np.random.default_rng(seed)), law_truth(
        PRIMARY_CELL.law
    )


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the shipped estimate and interval of each estimand."""
    result = fit(frame, PRIMARY_CELL.folds, ESTIMANDS)
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
        n=PRIMARY_N,
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )


__all__: Sequence[str] = (
    "CELLS",
    "STUDY",
    "draw",
    "draw_and_fit",
    "draw_arms",
    "fit",
)
