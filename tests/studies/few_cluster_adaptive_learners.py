r"""Registered evidence for few-cluster intervals with the default learner library (X54 part (h)).

Below 40 clusters a clustered estimate reports its interval on a Student t reference with
``J - 2`` degrees of freedom. The registered few-cluster studies measure that rule with
parametric nuisance models only (``clustered-few-cluster-tmle``,
``few-cluster-cross-fitted-ltmle``), and Wang, Park, Small and Li (2024), Remark 3, caution
against complex nuisance learners at about 20 clusters. This study measures the default learner
library (:func:`cleverly.learners.default_library`: histogram gradient boosting, a random forest
and an L1 logistic or lasso regression, stacked by a Super Learner with grouped inner folds) for
every nuisance, at 10, 20 and 40 clusters. A configuration that failed the declared caveat rule
would have taken an ``adaptive_nuisance`` shortfall of its support row. Every gated row passes in
the registered run, so none does, and the study is evidence of the rows its gated fits reach
(:data:`cleverly._support_rows.ADAPTIVE_LEARNER_CELLS`).

**Laws.** L1 ``equal10``: the equal-size law of ``clustered-few-cluster-tmle``, clusters of
10 rows, the arm-by-latent coefficient 8. L2 ``nonlinear10``: :mod:`tests.studies.clustered_nonlinear_law`,
the same layout with both nuisances nonlinear in the row covariates. L3 ``equal40``: the
end-of-study law of ``few-cluster-cross-fitted-ltmle`` at 20 clusters of 40 rows.

**Fits.** On L1 and L2 at 10, 20 and 40 clusters, every draw is fitted six times, each fit its
own key (:data:`POINT_FITS`):

=========================  ===================================================================
fit                        construction
=========================  ===================================================================
``tmle_in_sample``         in-sample ``TMLE``, the default library for both nuisances
``tmle_stacked``           stacked ``TMLE``, 5 grouped folds, the default library
``glm_in_sample``          in-sample ``TMLE``, main-terms logistic outcome and propensity
``glm_stacked``            stacked ``TMLE`` with those GLMs
``interpolating_control``  in-sample ``TMLE``, an unpruned tree outcome, logistic propensity
``stacked_control``        stacked ``TMLE`` whose out-of-fold outcome predictions are an
                           unpruned random forest fitted on the full sample
                           (:class:`FullSampleForest`)
=========================  ===================================================================

On L3 at 20 clusters: ``ltmle_crossfit`` (cross-fitted ``LTMLE``, 5 folds, the default library
for every outcome node, the known mechanisms) and ``ltmle_glm`` (the committed subject, the
quasi-binomial GLM).

**Arms.** A default-library fit publishes ``t_reference`` (the reported interval),
``iid_t_control`` (the same point and t quantile with the IID row SE, the two-clause rule of
the registered few-cluster studies) and ``wang_inflated`` (reported: candidate (B) of part (c),
the variance times :math:`J/(J-p)` on :math:`t_{J-p}` with :math:`p = \max(2, d) = 2`). On the
gated fits (:data:`POSITIVE_FITS`: stacked ``TMLE`` and cross-fitted ``LTMLE``) the first is
positive and the second a control. The in-sample default-library fit is reported only: it takes
the ``adaptive_nuisance`` shortfall by theory, and its rows publish the
same readings, among them the paired coverage difference to the GLM arm on L1 and L2. A GLM fit
publishes ``glm_reference`` (reported; the paired non-inferiority clause of the stacked L1
positives reads it). ``stacked_control`` is a must-fail control; ``interpolating_control`` is
reported. The rules are those of :mod:`tests.studies.few_cluster_adaptive_properties`.

**Primary.** L1 at 20 clusters, stacked default-library ``TMLE``, the ATE, 1,200 replications:
the smallest multiple of 400 of at least 1,000 whose predictive family-wise pass probability
reaches 0.95 on the pilot (0.956; 0.933 at 1,000,
:func:`tests.studies.few_cluster_adaptive_pilot.primary_replicates`). No comparator fits a
few-cluster TMLE with a Super Learner and a t reference, so the record declares no reference and
publishes an empty equivalence artifact. The policy is ``reporting``, declared before any run,
because the pilot predicts three positive rows red (:data:`PREDICTED_REDS`). The failure bound
is 0.01 per side.

**Pilot.** 400 draws of every fit of every cell on the stream of
:mod:`tests.studies.few_cluster_adaptive_pilot` (``pilot.csv.gz``; no fit raised). Every stacked
default-library cell reads a variance ratio whose 99% interval contains 1 (0.93 to 1.13), and
its paired all-draw coverage difference to the stacked GLM arm on the same draws lies in
:math:`[-0.008, 0.013]`, so the pilot confirms no departure of the stacked library from the
parametric baseline. At 10 clusters both laws read coverage 0.92 for the default library and
0.92 to 0.93 for the stacked GLM arm on the same draws: a few-cluster deficit the parametric
baseline shares (the committed exact-propensity GLM reads 0.935 there at 4,000 draws), which the
shown-coverage clause's 0.92 floor sees. The in-sample fits read variance ratios of 0.68 to 0.88
and coverage 1.5 to 3.3 points under their GLM arm at every count, the overfitting the theory
predicts. The stacked control's power to fail is at least 0.989 at every count.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from cleverly.estimators import TMLE
from tests.studies import clustered_nonlinear_law as nonlinear
from tests.studies import clustered_unequal_laws as informative
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.clustered_crossfit_ltmle import ID as LTMLE_ID
from tests.studies.clustered_crossfit_ltmle import NODES as LTMLE_NODES
from tests.studies.clustered_crossfit_ltmle import subject as ltmle_subject
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.few_cluster_crossfit_ltmle import TARGET as LTMLE_TARGET
from tests.studies.few_cluster_crossfit_ltmle import draw as draw_end_of_study
from tests.studies.point_study_helpers import initial_estimates, primary_rows

PRIMARY_REPLICATES: Final[int] = 1_200
PRIMARY_CLUSTERS: Final[int] = 20
PRIMARY_N: Final[int] = 10 * PRIMARY_CLUSTERS
#: Draws of every property cell, the reported in-sample cells included. At target
#: (variance ratio 1, nominal coverage, a paired difference of 0) every clause of every positive
#: row passes with probability at least 0.9993 on the pilot's spreads, so no cell is raised to
#: 4,800 (:func:`tests.studies.few_cluster_adaptive_pilot.at_target_pass`). The in-sample cells
#: keep 4,000 draws, paired with the stacked and GLM fits of the same draws, so their published
#: bias over SD and paired GLM difference compare with the stacked rows' at equal precision
#: (Remark 3's in-sample against cross-fitted reading).
PROPERTY_REPLICATES: Final[int] = 4_000
SEED: Final[int] = 20264581
RESAMPLING_SEED: Final[int] = 20264582
SCENARIO: Final[str] = "equal10_j20_stacked"
FAMILY: Final[str] = "adaptive_learners"
ESTIMANDS: Final[tuple[str, ...]] = ("ate",)
#: Grouped outer folds of every stacked or cross-fitted fit.
N_FOLDS: Final[int] = 5
#: The seed of every fit's folds and learners, as the registered few-cluster studies fix it.
RANDOM_STATE: Final[int] = 0

#: The point laws and the LTMLE law, with the cluster counts each runs at.
POINT_LAWS: Final[tuple[str, ...]] = ("equal10", "nonlinear10")
LTMLE_LAW: Final[str] = "equal40"
COUNTS: Final[dict[str, tuple[int, ...]]] = {
    "equal10": (10, 20, 40),
    "nonlinear10": (10, 20, 40),
    "equal40": (20,),
}
#: Rows per cluster of each law.
CLUSTER_SIZE: Final[dict[str, int]] = {"equal10": 10, "nonlinear10": 10, "equal40": 40}

POINT_FITS: Final[tuple[str, ...]] = (
    "tmle_in_sample",
    "tmle_stacked",
    "glm_in_sample",
    "glm_stacked",
    "interpolating_control",
    "stacked_control",
)
LTMLE_FITS: Final[tuple[str, ...]] = ("ltmle_crossfit", "ltmle_glm")
#: The default-library fits. Each publishes its interval, its IID control and candidate (B).
ADAPTIVE_FITS: Final[tuple[str, ...]] = ("tmle_in_sample", "tmle_stacked", "ltmle_crossfit")
#: The default-library fits whose ``t_reference`` arm is gated (positive). The in-sample fit is
#: reported only: it takes the ``adaptive_nuisance`` shortfall by theory
#: (:data:`cleverly._support.ADAPTIVE_IN_SAMPLE_REASON`), because
#: each learner predicts from its full-sample fit and no Donsker argument covers it.
POSITIVE_FITS: Final[tuple[str, ...]] = ("tmle_stacked", "ltmle_crossfit")
#: The GLM fit paired with each default-library fit (the non-inferiority clause on L1).
GLM_OF: Final[dict[str, str]] = {
    "tmle_in_sample": "glm_in_sample",
    "tmle_stacked": "glm_stacked",
    "ltmle_crossfit": "ltmle_glm",
}
#: The must-fail control whose failure the caveat rule needs for each gated fit.
#: The cross-fitted LTMLE cell reads the stacked control of L1 at 20 clusters: no control
#: plants an overfit LTMLE outcome, and the stacked control shows that the out-of-fold reading
#: can see an overfit outcome fit at that count (declared before any run).
MUST_FAIL_OF: Final[dict[str, str]] = {
    "tmle_stacked": "stacked_control",
    "ltmle_crossfit": "stacked_control",
}

#: Each fit's arms and their roles.
ARMS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    **dict.fromkeys(
        POSITIVE_FITS,
        (
            ("t_reference", "positive"),
            ("iid_t_control", "control"),
            ("wang_inflated", "diagnostic"),
        ),
    ),
    "tmle_in_sample": (
        ("t_reference", "diagnostic"),
        ("iid_t_control", "diagnostic"),
        ("wang_inflated", "diagnostic"),
    ),
    **dict.fromkeys(
        ("glm_in_sample", "glm_stacked", "ltmle_glm"), (("glm_reference", "diagnostic"),)
    ),
    # Reported: it reads what an interpolating in-sample outcome fit does to the interval.
    "interpolating_control": (("interpolating_control", "diagnostic"),),
    "stacked_control": (("stacked_control", "control"),),
}


def fits_of(law: str) -> tuple[str, ...]:
    """The fits a draw of ``law`` takes."""
    return LTMLE_FITS if law == LTMLE_LAW else POINT_FITS


def cell_name(fit: str, law: str, clusters: int, arm: str) -> str:
    """The published name of one property cell."""
    return f"{fit}__{law}__j{clusters}__{arm}"


#: Every (law, count) the study draws, in declaration order.
DRAW_CELLS: Final[tuple[tuple[str, int], ...]] = tuple(
    (law, clusters) for law in (*POINT_LAWS, LTMLE_LAW) for clusters in COUNTS[law]
)

PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(fit, law, clusters, arm)
    for law, clusters in DRAW_CELLS
    for fit in fits_of(law)
    for arm, _ in ARMS[fit]
)

STUDY = StudyRecord(
    name="few-cluster intervals with the default learner library",
    slug="few-cluster-adaptive-learners",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_adaptive_learners",
    document="docs/technical-reference/method-evidence/few-cluster-adaptive-learners.md",
    anchor="few-cluster-intervals-with-the-default-learner-library",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/few_cluster_adaptive_learners.py",
        "tests/studies/few_cluster_adaptive_properties.py",
        "tests/studies/clustered_nonlinear_law.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/clustered_crossfit_ltmle.py",
        "tests/studies/clustered_longitudinal_laws.py",
        "tests/studies/few_cluster_crossfit_ltmle.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_clustered_tmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/conftest.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.few_cluster_adaptive_learners",
    properties_module="tests.studies.few_cluster_adaptive_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "the shipped influence-curve interval on t with J - 2 degrees of freedom "
    "below 40 clusters, with the default learner library for every nuisance",
    "laws": {
        "equal10": "clustered-few-cluster-tmle's equal10 law, arm-by-latent coefficient 8",
        "nonlinear10": "tests/studies/clustered_nonlinear_law.py",
        "equal40": "few-cluster-cross-fitted-ltmle's end-of-study law, scaled latent",
    },
    "counts": {law: list(counts) for law, counts in COUNTS.items()},
    "default_library": "cleverly.learners.default_library: histogram gradient boosting, random "
    "forest, L1 logistic or lasso; Super Learner with grouped inner folds on the cluster id",
    "glm_reference": "main-terms unpenalized logistic outcome and propensity (point); the "
    "committed quasi-binomial GLM subject (LTMLE)",
    "controls": "an unpruned tree outcome in sample (reported); an unpruned random forest "
    "fitted on the full sample standing in for every out-of-fold prediction of a stacked fit "
    "(must fail)",
    "gated_fits": list(POSITIVE_FITS),
    "outer_folds": N_FOLDS,
    "random_state": RANDOM_STATE,
    "simultaneous_intervals": False,
}


# ------------------------------------------------------------------------------------ laws


def law_truth(law: str) -> float:
    """The exact ATE of ``law``."""
    if law == "equal10":
        return float(
            informative.informative_truth(delta=0.0, gamma=0.0, effect_modifier=8.0)["ate"]
        )
    if law == "nonlinear10":
        return float(nonlinear.truth()["ate"])
    if law == LTMLE_LAW:
        _, truth = draw_end_of_study("equal1", 2, 0)
        return float(truth[LTMLE_TARGET])
    raise KeyError(law)


def draw(law: str, clusters: int, seed: int) -> pd.DataFrame:
    """One draw of ``clusters`` clusters of ``law``."""
    rng = np.random.default_rng(seed)
    if law == "equal10":
        return informative.draw_informative(
            clusters,
            rng,
            delta=0.0,
            gamma=0.0,
            effect_modifier=8.0,
            sizes=np.full(clusters, CLUSTER_SIZE[law]),
        )
    if law == "nonlinear10":
        return nonlinear.draw(clusters, rng)
    if law == LTMLE_LAW:
        frame, _ = draw_end_of_study(law, clusters, seed)
        return frame
    raise KeyError(law)


# ------------------------------------------------------------------------------------ fits


def _logistic() -> LogisticRegression:
    """The unpenalized main-terms logistic regression of the GLM arms."""
    return LogisticRegression(C=1e6, max_iter=1000)


def _tree() -> DecisionTreeClassifier:
    """The unpruned tree of both controls: one leaf per distinct design row."""
    return DecisionTreeClassifier(min_samples_leaf=1, random_state=RANDOM_STATE)


#: Trees of the stacked control's forest.
CONTROL_TREES: Final[int] = 100


def _forest() -> RandomForestClassifier:
    """The unpruned forest of the stacked control: each tree grown to single-row leaves.

    Each tree fits its bootstrap sample to purity, so a row's prediction averages the trees
    that held it, which reproduce its outcome, with the about 37% that did not. The fitted
    values are pulled strongly toward the outcomes without equalling them, so the residuals
    shrink but do not vanish, and the fluctuation step and the propensity still act.
    """
    return RandomForestClassifier(
        n_estimators=CONTROL_TREES, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1
    )


class FullSampleForest(BaseEstimator, ClassifierMixin):
    """An unpruned random forest fitted on the full sample, whatever rows a fold passes.

    The stacked control feeds the stacked
    targeting and variance code out-of-fold predictions that are in-sample ones: each fold's
    "training" fit is the forest on every row, so the validation rows were seen. The outcome
    design of :class:`~cleverly.TMLE` is ``[A, W1, W2]``; :meth:`fit` refuses a training design
    whose rows are not rows of ``design``, so a changed column order stops the control rather
    than fitting a different design.

    Parameters
    ----------
    design : ndarray
        The full outcome design, ``[A, W1, W2]`` on every row.
    target : ndarray
        The full outcome.
    """

    def __init__(self, design: Any = None, target: Any = None) -> None:
        self.design = design
        self.target = target

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FullSampleForest:
        del y, sample_weight
        full = np.round(np.asarray(self.design, dtype=float), 12)
        rows = np.round(np.asarray(X, dtype=float), 12)
        known = {tuple(row) for row in full}
        if rows.shape[1] != full.shape[1] or any(tuple(row) not in known for row in rows):
            raise ValueError("FullSampleForest: a training row is not a row of the full design")
        self.model_ = _forest().fit(full, np.asarray(self.target, dtype=float))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(np.asarray(X, dtype=float)))


ROLES: Final[dict[str, Any]] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W1", "W2"],
    "id": "cluster",
}


def point_estimator(fit: str, frame: pd.DataFrame) -> TMLE:
    """The ``TMLE`` of one point fit; the default library wherever no learner is named."""
    settings: dict[str, Any] = {
        "estimands": ESTIMANDS,
        "simultaneous": False,
        "random_state": RANDOM_STATE,
        "n_folds": N_FOLDS,
    }
    stacked = fit in {"tmle_stacked", "glm_stacked", "stacked_control"}
    settings["cross_fit"] = stacked
    if fit in {"glm_in_sample", "glm_stacked"}:
        settings |= {"outcome_learner": _logistic(), "treatment_learner": _logistic()}
    elif fit == "interpolating_control":
        settings |= {"outcome_learner": _tree(), "treatment_learner": _logistic()}
    elif fit == "stacked_control":
        design = frame[["A", "W1", "W2"]].to_numpy(dtype=float)
        settings |= {
            "outcome_learner": FullSampleForest(design, frame["Y"].to_numpy(dtype=float)),
            "treatment_learner": _logistic(),
        }
    elif fit not in {"tmle_in_sample", "tmle_stacked"}:
        raise KeyError(fit)
    return TMLE(**settings)


def ltmle_estimator(fit: str) -> Any:
    """The ``LTMLE`` of one L3 fit: the committed subject, or it with the default library."""
    estimator = ltmle_subject()
    if fit == "ltmle_crossfit":
        estimator.outcome_learner = None
        estimator.pseudo_learner = None
    elif fit != "ltmle_glm":
        raise KeyError(fit)
    return estimator


@dataclasses.dataclass(frozen=True)
class Fitted:
    """One fit's ATE estimate and what the rows record beside it.

    Parameters
    ----------
    estimate : ParameterEstimate
        The reported estimate.
    sl_single : float
        1 when every Super Learner of the outcome put at least 0.999 of its weight on one
        candidate, 0 when one did not, ``nan`` for a fit with no outcome Super Learner.
    """

    estimate: Any
    sl_single: float


def _sl_single(result: Any) -> float:
    diagnostics = getattr(getattr(result, "nuisance", None), "diagnostics", None) or {}
    fits = diagnostics.get("outcome", ())
    weights = [np.asarray(getattr(d, "weights", ()), dtype=float) for d in fits]
    weights = [w for w in weights if w.size]
    if not weights:
        return float("nan")
    return float(all(w.max() >= 0.999 for w in weights))


def fit_one(fit: str, frame: pd.DataFrame) -> Fitted:
    """Fit one key: the estimate of the ATE and the Super Learner reading."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit in LTMLE_FITS:
            result = ltmle_estimator(fit).fit(frame, outcome="Y", id=LTMLE_ID, **LTMLE_NODES)
            return Fitted(result[LTMLE_TARGET], float("nan"))
        result = point_estimator(fit, frame).fit(frame, **ROLES).single()
        return Fitted(result["ate"], _sl_single(result))


# --------------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw: L1 at ``n // 10`` clusters."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return draw("equal10", n // CLUSTER_SIZE["equal10"], seed), {"ate": law_truth("equal10")}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the stacked default-library fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = point_estimator("tmle_stacked", frame).fit(frame, **ROLES).single()
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


def draw_and_fit(*, replicates: int, n: int, n_jobs: int | None = None) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    from tests.parallel import STUDY_JOBS

    return fitted_rows(
        STUDY,
        draw_scenario,
        cleverly_rows,
        replicates=replicates,
        n=n,
        n_jobs=STUDY_JOBS if n_jobs is None else n_jobs,
    )


# ------------------------------------------------------------------- the pilot's predictions

#: The predictive probability that each gated row passes at :data:`PROPERTY_REPLICATES` draws,
#: read from the committed 400-draw pilot
#: (:func:`tests.studies.few_cluster_adaptive_pilot.pilot_readings`) before the declaration: a
#: positive row's clauses as a product, an IID control's two clauses, and a must-fail control's
#: power to fail. G1 and the shown-coverage clause enter as their joint probability on one
#: predictive hit law, since the pilot withheld no interval. Each figure carries the pilot's own
#: error (Jeffreys beta-binomial for a rate, normal for the log variance ratio, the bias and the
#: paired difference).
PREDICTED_PASS: Final[dict[str, float]] = {
    "ltmle_crossfit__equal40__j20__iid_t_control": 1.000,
    "ltmle_crossfit__equal40__j20__t_reference": 0.927,
    "stacked_control__equal10__j10__stacked_control": 0.999,
    "stacked_control__equal10__j20__stacked_control": 1.000,
    "stacked_control__equal10__j40__stacked_control": 1.000,
    "stacked_control__nonlinear10__j10__stacked_control": 0.999,
    "stacked_control__nonlinear10__j20__stacked_control": 0.989,
    "stacked_control__nonlinear10__j40__stacked_control": 1.000,
    "tmle_stacked__equal10__j10__iid_t_control": 1.000,
    "tmle_stacked__equal10__j10__t_reference": 0.226,
    "tmle_stacked__equal10__j20__iid_t_control": 1.000,
    "tmle_stacked__equal10__j20__t_reference": 0.760,
    "tmle_stacked__equal10__j40__iid_t_control": 1.000,
    "tmle_stacked__equal10__j40__t_reference": 0.875,
    "tmle_stacked__nonlinear10__j10__iid_t_control": 1.000,
    "tmle_stacked__nonlinear10__j10__t_reference": 0.177,
    "tmle_stacked__nonlinear10__j20__iid_t_control": 1.000,
    "tmle_stacked__nonlinear10__j20__t_reference": 0.975,
    "tmle_stacked__nonlinear10__j40__iid_t_control": 1.000,
    "tmle_stacked__nonlinear10__j40__t_reference": 0.269,
}
#: A gated row is a predicted red when its predictive pass probability is below this: red is
#: more likely than not. The plug-in rule (pilot readings taken as true, figure below 0.90)
#: names the same three rows (0.022, 0.005 and 0.313).
PREDICTED_RED_BELOW: Final[float] = 0.50
#: The gated rows the pilot predicted red, declared before the run with their readings
#: (``tests/canonical/few_cluster_adaptive_learners/regenerate.py``). None went red in the
#: registered run: they cover 0.9425, 0.9437 and 0.9430 at 4,000 draws. At 10 clusters the
#: shown-coverage clause binds (predictive 0.26 and 0.21; coverage 0.92 on both laws, shared
#: with the paired GLM arm); on L2 at 40 the shown-coverage and variance-ratio clauses (0.46
#: and 0.58; coverage 0.930, variance ratio 0.93, the paired GLM arm 0.918 and 0.90).
PREDICTED_REDS: Final[tuple[str, ...]] = (
    "tmle_stacked__equal10__j10__t_reference",
    "tmle_stacked__nonlinear10__j10__t_reference",
    "tmle_stacked__nonlinear10__j40__t_reference",
)
#: No interim stop: every cell runs its declared draws.
INTERIM_DRAWS: Final[int | None] = None


__all__: Sequence[str] = (
    "PREDICTED_PASS",
    "PREDICTED_REDS",
    "STUDY",
    "draw",
    "draw_and_fit",
    "fit_one",
    "law_truth",
)
