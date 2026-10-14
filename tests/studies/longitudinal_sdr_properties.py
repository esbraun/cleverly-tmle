r"""Property families of ``longitudinal-sdr``.

Every fit is :class:`~cleverly.longitudinal.SDR` with five folds, the law's own mechanism and,
on a discrete law, saturated cell means, unless the table names another setting.

==================================  =======================================================
family                              cells, replications and size
==================================  =======================================================
``sequential_double_robustness``    ``ate_regimen[always vs never]`` of
                                    :mod:`tests.discrete_law_longitudinal_sdr` in the five
                                    patterns of ``X4-plan.md`` 6.8, at n = 2,000 and 8,000;
                                    1,000 each.  Two controls at n = 2,000: SDR with both
                                    nuisances wrong at node 2, and LTMLE in ``P_D``
``root_n_and_efficiency``           ``static`` at n = 1,000, 2,000 and 8,000; 700 each
``root_n_rate``                     the two rate rows of ``static``, from the ladder
``interval_calibration``            eight labels (below); 4,000 at n = 2,000 each, with the
                                    two derived controls of each
``weighted_calibration``            ``selection_weighted``: the selected law of
                                    ``weighted_longitudinal_properties_common`` with its
                                    weights, ``always vs never``; 4,000 at n = 2,000, with
                                    two derived controls
``clustered_calibration``           ``clustered_longitudinal_laws`` with grouped folds and
                                    GLM outcome regressions: ``clusters_200`` (200 clusters
                                    of 10) and ``clusters_20`` (20 clusters of 100,
                                    ``X4-plan.md`` 6.9); 4,000 each, with two derived
                                    controls of each
``type_i_error``                    ``static`` under the sharp null of
                                    ``ltmle_crossfit_properties``; 800 at n = 4,000
``power``                           ``static`` under the law; 800 at n = 4,000
``crossfit_overfitting``            fully grown trees on ``make_longitudinal``: five folds
                                    (positive) and one fold (control), paired draws; 8,000
                                    at n = 1,000
``simultaneous_coverage``           the band over the six regimen means of
                                    :mod:`tests.discrete_law_longitudinal` and its
                                    pointwise control
``construction_diagnostic``         no verdict.  On the overfitting draws: the stitched Step 2
                                    of Section 5.3 (``SDR_STITCHED_CROSS_FIT``), and the
                                    shipped construction with an estimated tree mechanism
                                    under the default bounds (0.01, 1)
``sdr_tmle_agreement``              no verdict.  :math:`|\hat\psi_{SDR} - \hat\psi_{LTMLE}|
                                    / \widehat{SE}_{SDR}` on the ``static`` calibration draws
==================================  =======================================================

The calibration labels:

======================  ==============================================================  ==========
label                   law and estimand                                                bound
======================  ==============================================================  ==========
``static``              :mod:`tests.discrete_law_longitudinal`, ``always vs never``     exact EIF
``dynamic``             the same fits, ``ey_regimen[treat_if_l2]``                      exact EIF
``survival_h2``         :mod:`tests.discrete_law_survival`, ``always vs never @ t=2``   exact EIF
``competing_h2``        :mod:`tests.discrete_law_competing`, death, ``always vs never``  exact EIF
``policy_mix``          the three-level law of ``stochastic-categorical-ltmle``,        exact EIF
                        ``mix vs low``
``mtp_up``              the continuous-dose law of ``longitudinal-mtp``, ``up vs        quadrature
                        natural``, with its oracle bins
``msm_identity``        the ``duration`` coefficient of the identity working model of   exact EIF
                        ``cross-fitted-longitudinal-msm``
``msm_logit``           the ``duration`` coefficient of its logit working model         exact EIF
======================  ==============================================================  ==========

A ``clustered_calibration`` label has no efficiency claim, because its GLM outcome
regressions are misspecified and the cluster latent moves the bound.  ``weighted_calibration``
has none either, for the reason :data:`NOISE_SD` gives.  The positive cell of each reads the
SE-ratio and coverage bands only.  Its noise control adds :data:`NOISE_SD` and reads the
SE-ratio rule.

**The ``sequential_double_robustness`` rule** (``X4-plan.md`` 6.8).  ``both_correct`` and
``P_C`` have exact ratios, so they gate the bias, the reported-error screen and the coverage
floor at both sizes.  ``P_A``, ``P_B`` and ``P_D`` gate the absolute bias at two sizes: the
pair passes when :math:`|\bar b| + 2\,\mathrm{MCSE}` at n = 8,000 is below
:data:`SDR_TOLERANCE` and below the same bound at n = 2,000, and the SE ratio at n = 8,000 lies
inside ``UNION_MODEL_SE_BAND``.  The MCSE halves between the sizes at a fixed replication
count, so the second clause also holds for a constant bias: the tolerance is the clause that
discriminates.  Their coverage is reported only, because Theorem 4 needs the rate product.
Each control passes when its whole bias interval lies at least :data:`CONTROL_FLOOR`
empirical standard deviations from zero, and its SE ratio lies inside the same band.

A replication that raises is not redrawn.  :func:`failure_probe` counts failed fits per fit
set and records no estimate.  A fit set with any failure is dropped before the declared run,
with its gap stated as a page limit.  A raise in the declared run stops it.  The failing cell
then drops to its red-cell owner with its failure count published, and the run repeats
without it with no other change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE, SDR
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_competing as competing_law
from tests import discrete_law_longitudinal as law
from tests import discrete_law_longitudinal_multivalue as multivalue
from tests import discrete_law_longitudinal_sdr as sdr_law
from tests import discrete_law_survival as survival_law
from tests.longitudinal_sdr import (
    CENSORING_WIDTH,
    COLUMNS,
    OUTCOME_WIDTH,
    TREATMENT_WIDTH,
    NodeLearner,
)
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_longitudinal_mtp as mtp_study
from tests.studies import canonical_stochastic_categorical_ltmle as policy_study
from tests.studies import categorical_longitudinal_common as categorical
from tests.studies import clustered_longitudinal_laws as clustered_law
from tests.studies import crossfit_longitudinal_msm_properties as msm_properties
from tests.studies import longitudinal_mtp_common as mtp
from tests.studies import longitudinal_mtp_properties as mtp_properties
from tests.studies import ltmle_competing_properties as competing_properties
from tests.studies import ltmle_crossfit_properties as end_properties
from tests.studies import ltmle_survival_crossfit_properties as survival_properties
from tests.studies import stochastic_categorical_ltmle_properties as policy_properties
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_longitudinal_sdr import (
    BAND_LABEL,
    CALIBRATION_LABELS,
    CLUSTERED_POSITIVE,
    G_BOUNDS,
    LEARNER_FOLDS,
    N_FOLDS,
    PRIMARY_N,
    SDR_PATTERNS,
    SDR_SIZES,
    SEPARATE_CALIBRATION,
    STUDY,
    plans_of,
    supplied_ratios,
)
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism, declared_regimens
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    absolute_bias_bound,
    control_row,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    UNION_MODEL_SE_BAND,
    _recompute_interval_columns,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    indexed_noise,
    se_ratio_leaves_band,
    simultaneous_coverage_verdicts,
    summary_interval,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import (
    joint_cells,
    joint_coverage_rows,
    joint_property_cells,
)
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.oracle_density_bins import fit_bytes, oracle_bins

SDR_REPLICATES = 1_000
CONTROL_N = 2_000
RATE_REPLICATES = 700
RATE_SIZES = (1_000, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
OVERFIT_REPLICATES = 8_000
OVERFIT_N = 1_000
#: The band cell's replications; ``tests/unit/test_longitudinal_sdr_design.py`` pins that they
#: meet the RM36 control-power rule at the measured ``p0``.
BAND_REPLICATES = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
#: The absolute bias bound of the ``P_A``, ``P_B`` and ``P_D`` cells at n = 8,000.
SDR_TOLERANCE = 0.01
#: The standardized bias each robustness control must establish (``X4-plan.md`` 6.8).
CONTROL_FLOOR = 3.0
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

STATIC = "ate_regimen[always vs never]"
DYNAMIC_MEAN = "ey_regimen[treat_if_l2]"
SURVIVAL_NAME = survival_properties.CONTRASTS["static_t2"]
COMPETING_NAME = competing_properties.CONTRASTS["death_static_t2"]
MIX = policy_properties.MIX
UP = mtp_properties.UP
MSM_NAME = msm_properties.NAMES["duration"]
#: The estimand of every overfitting and construction cell.
OVERFIT_NAME = STATIC
#: The shipped default bounds, for the tree-mechanism diagnostic only.  A leaf of 50 rows can hold
#: no row at a plan's arm, so its estimated probability is 0 and :data:`G_BOUNDS` would weight a
#: held-out row by 1e8: the failure probe read estimates near 1e5 in 6 of 2,000 draws.  No true
#: partial path probability of ``make_longitudinal`` reaches 0.02 (50,000 rows), so these bounds
#: move only estimated zeros.
TREE_G_BOUNDS = (0.01, 1.0)
#: The estimand of an ``sdr_tmle_agreement`` row.
AGREEMENT = "abs_difference_over_se"
#: The summary column that publishes the mean of :data:`AGREEMENT`.
AGREEMENT_COLUMN = "mean_abs_difference_over_se"
#: The column that publishes :math:`|\bar b| + 2\,\mathrm{MCSE}` of a robustness cell.
BIAS_BOUND_COLUMN = "absolute_bias_bound"

END_REGIMENS = declared_regimens(
    {label: law.REGIMEN_SPEC[label] for label in ("never", "always", "treat_if_l2")}
)
BAND_REGIMENS = declared_regimens(law.REGIMEN_SPEC)
#: Every parameter the band fit reports: the six regimen means and five contrasts.
BAND_NAMES = law.NAMES

#: The estimand of each calibration label.
CALIBRATION_ESTIMANDS: dict[str, str] = {
    "static": STATIC,
    "dynamic": DYNAMIC_MEAN,
    "survival_h2": SURVIVAL_NAME,
    "competing_h2": COMPETING_NAME,
    "policy_mix": MIX,
    "mtp_up": UP,
    "msm_identity": MSM_NAME,
    "msm_logit": MSM_NAME,
}
#: The law each calibration label is drawn from.  ``static`` and ``dynamic`` share their fits.
CALIBRATION_LAWS: dict[str, str] = {
    "static": "discrete",
    "dynamic": "discrete",
    "survival_h2": "survival",
    "competing_h2": "competing",
    "policy_mix": "policy",
    "mtp_up": "mtp",
    "msm_identity": "msm",
    "msm_logit": "msm_logit",
}
#: ``(clusters, sizes)`` of each ``clustered_calibration`` label, which is also its law's name.
CLUSTER_LAYOUT = {"clusters_200": (200, "equal10"), "clusters_20": (20, "equal100")}


def _sd(probs: np.ndarray, curve: np.ndarray) -> float:
    return float(np.sqrt(np.sum(probs * curve**2)))


#: The efficiency bound of each consistent calibration label, as a standard deviation.
EFFICIENCY_SD: dict[str, float] = {
    "static": end_properties.EFFICIENCY_SD["static"],
    "dynamic": _sd(law.PROBS, law.eif(DYNAMIC_MEAN)),
    "survival_h2": survival_properties.EFFICIENCY_SD["static_t2"],
    "competing_h2": competing_properties.EFFICIENCY_SD["death_static_t2"],
    "policy_mix": policy_properties.EFFICIENCY_SD["mix"],
    "mtp_up": mtp_properties.EFFICIENCY_SD["up"],
    "msm_identity": msm_properties.EFFICIENCY_SD["duration"],
    "msm_logit": msm_properties.EFFICIENCY_SD[msm_properties.LOGIT],
}
#: The root-n noise scale of each label of :data:`SEPARATE_CALIBRATION`.  A clustered label has
#: no bound: the scale is declared from a design probe of 400 draws on streams 100,000 and
#: above, which read an empirical root-n spread of 1.727 and 3.590.  ``selection_weighted``
#: uses its exact bound, but it claims no efficiency ratio.  The fixed-weight LTMLE cell of
#: ``weighted-ltmle-crossfit`` reads ``finite-sample, contracting`` against the same bound
#: (owner ``RM18-fixed-weights``), and a design probe of 1,500 draws read an empirical ratio
#: of 1.080 for SDR against the ceiling of 1.10.
NOISE_SD: dict[str, float] = {
    "clusters_200": 1.75,
    "clusters_20": 3.6,
    "selection_weighted": weighted.EFFICIENCY_SD["static"],
}

#: The truth of each calibration label that does not vary by draw.
TRUTH: dict[str, float] = {
    "static": float(law.TRUTH[STATIC]),
    "dynamic": float(law.TRUTH[DYNAMIC_MEAN]),
    "survival_h2": float(survival_law.TRUTH[SURVIVAL_NAME]),
    "competing_h2": float(competing_law.TRUTH[COMPETING_NAME]),
    "policy_mix": float(policy_study.TRUTH[MIX]),
    "mtp_up": float(mtp_study.TRUTH[mtp_study.CONTINUOUS][UP]),
    "msm_identity": float(msm_properties.TRUTH[MSM_NAME]),
    "msm_logit": float(msm_properties.LOGIT_TRUTH),
    "selection_weighted": float(law.TRUTH[STATIC]),
}


# ---------------------------------------------------------------------------- fits


def _sdr(regimens: Any, **settings: Any) -> SDR:
    defaults: dict[str, Any] = {
        "n_folds": N_FOLDS,
        "learner_folds": LEARNER_FOLDS,
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "random_state": 0,
    }
    defaults.update(settings)
    return SDR(regimens, **defaults)


def discrete_fit(
    frame: pd.DataFrame,
    *,
    regimens: Any = END_REGIMENS,
    simultaneous: bool = False,
    engine: type = SDR,
) -> Any:
    """The fit of every family on :mod:`tests.discrete_law_longitudinal`."""
    outcome, pseudo, treatment, censoring = end_properties._learners("both_correct")
    settings: dict[str, Any] = {
        "reference": "never",
        "outcome_learner": outcome,
        "pseudo_learner": pseudo,
        "treatment_learner": treatment,
        "censoring_learner": censoring,
        "simultaneous": simultaneous,
    }
    if engine is LTMLE:
        return end_properties.fit(frame)
    return _sdr(regimens, **settings).fit(frame, **COLUMNS)


def sdr_learners(first: str, second: str) -> dict[str, Any]:
    """The four learner slots of a ``sequential_double_robustness`` pattern.

    ``first`` and ``second`` name the wrong nuisances at nodes 1 and 2: ``m`` the outcome
    regression, ``r`` the treatment and censoring mechanism.  A right regression is the
    saturated cell mean and a right mechanism is the law's own.  A wrong regression at node 2
    is least squares on ``L2`` alone, at node 1 the sample mean; a wrong mechanism is the
    marginal class share.  These are the learners of ``tests/unit/test_sdr_double_robustness.py``
    with the law's mechanism in place of a cell mean, which an empty training cell could
    otherwise move.
    """

    def mechanism(kind: str, wrong: bool) -> Any:
        return DummyClassifier(strategy="prior") if wrong else sdr_law.KnownMechanism(kind)

    horizon = OUTCOME_WIDTH[2]
    return {
        "outcome_learner": NodeLearner(
            {horizon: LinearRegression() if "m" in second else law.CellMeans()},
            columns={horizon: (1,)} if "m" in second else None,
        ),
        "pseudo_learner": NodeLearner(
            {OUTCOME_WIDTH[1]: DummyRegressor() if "m" in first else law.CellMeans()}
        ),
        "treatment_learner": NodeLearner(
            {
                TREATMENT_WIDTH[1]: mechanism("treatment", "r" in first),
                TREATMENT_WIDTH[2]: mechanism("treatment", "r" in second),
            }
        ),
        "censoring_learner": NodeLearner(
            {
                CENSORING_WIDTH[1]: mechanism("censoring", "r" in first),
                CENSORING_WIDTH[2]: mechanism("censoring", "r" in second),
            }
        ),
    }


#: Wrong nuisances per node of each pattern, as ``(node 1, node 2)``.
PATTERN_NUISANCES: dict[str, tuple[str, str]] = {
    "both_correct": ("", ""),
    "P_A": ("r", "r"),
    "P_B": ("m", "r"),
    "P_C": ("m", "m"),
    "P_D": ("r", "m"),
}
#: ``(engine, wrong nuisances)`` of each robustness control.
CONTROLS: dict[str, tuple[type, tuple[str, str]]] = {
    "node2_both_wrong_control": (SDR, ("", "mr")),
    "ltmle_P_D_control": (LTMLE, PATTERN_NUISANCES["P_D"]),
}


def sdr_family_fit(frame: pd.DataFrame, nuisances: tuple[str, str], engine: type = SDR) -> Any:
    """One fit of the ``sequential_double_robustness`` law."""
    settings: dict[str, Any] = {
        "reference": sdr_law.REFERENCE,
        "n_folds": N_FOLDS,
        "learner_folds": LEARNER_FOLDS,
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "random_state": 0,
        **sdr_learners(*nuisances),
    }
    if engine is LTMLE:
        settings.update(max_iter=100, tol=1e-10)
    return engine(declared_regimens(sdr_law.REGIMEN_SPEC), **settings).fit(frame, **COLUMNS)


def survival_fit(frame: pd.DataFrame) -> Any:
    outcome, pseudo, treatment, censoring = survival_properties._learners("both_correct")
    return _sdr(
        survival_properties.REGIMENS,
        reference=survival_properties.REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
    ).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def competing_fit(frame: pd.DataFrame) -> Any:
    outcome, pseudo, treatment, censoring = competing_properties._learners("both_correct")
    return _sdr(
        competing_properties.REGIMENS,
        reference=competing_properties.REFERENCE,
        horizons=(2,),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
    ).fit(
        frame,
        outcome=competing_law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def policy_fit(frame: pd.DataFrame) -> Any:
    outcome, pseudo, treatment = categorical._learners("both_correct")
    return _sdr(
        policy_study.regimens(),
        reference=policy_study.REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
    ).fit(frame, outcome="Y", treatment=["A1", "A2"], baseline=["W"], time_varying=[[], ["L2"]])


def mtp_fit(frame: pd.DataFrame) -> Any:
    outcome, pseudo, hazard = mtp.continuous_learners("both_correct", mtp_study.edges_of(frame))
    return _sdr(
        mtp_properties._up_plans(),
        reference="natural",
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=hazard,
        density_bins=oracle_bins(len(frame)),
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        continuous_treatment=["A1", "A2"],
    )


def msm_fit(frame: pd.DataFrame, link: str = "identity") -> Any:
    outcome, pseudo, treatment, censoring = msm_properties._learners("both_correct")
    return _sdr(
        msm_properties.REGIMENS,
        msm=msm_properties.working_model(link=link),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
    ).fit(frame, **msm_properties.COLUMNS)


def weighted_fit(frame: pd.DataFrame) -> Any:
    outcome, pseudo, treatment, censoring = weighted._learners("both_correct", cross_fit=True)
    return _sdr(
        weighted.REGIMENS,
        reference=weighted.REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
    ).fit(frame, weights="obs_weight", **COLUMNS)


def clustered_draw(label: str, seed: int) -> tuple[pd.DataFrame, float]:
    clusters, sizes = CLUSTER_LAYOUT[label]
    frame, truth = clustered_law.draw_end_of_study(clusters, sizes, seed)
    return frame, float(truth[STATIC])


def clustered_fit(frame: pd.DataFrame) -> Any:
    """The primary subject of ``sdr_end_of_study`` with whole-cluster folds."""
    return _sdr(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=LinearRegression(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
    ).fit(
        frame,
        outcome="Y",
        id="id",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


# ------------------------------------------------------------- overfitting and construction


OVERFIT_PLANS = {"never": 0, "always": 1}
OVERFIT_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _grown() -> tuple[Any, Any]:
    return (
        DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
        DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
    )


def overfit_fit(frame: pd.DataFrame, *, n_folds: int, mechanism: str = "known") -> Any:
    """Fully grown outcome trees on ``make_longitudinal``.

    ``mechanism="known"`` is the law's own mechanism.  ``"tree"`` estimates it with trees of at
    least 50 rows a leaf, the diagnostic arm of ``X4-plan.md`` 6.11, under
    :data:`TREE_G_BOUNDS`.
    """
    outcome, pseudo = _grown()
    if mechanism == "known":
        treatment: Any = KnownLongitudinalMechanism("treatment")
        censoring: Any = KnownLongitudinalMechanism("censoring")
    else:
        treatment = DecisionTreeClassifier(min_samples_leaf=50, random_state=0)
        censoring = DecisionTreeClassifier(min_samples_leaf=50, random_state=0)
    bounds = G_BOUNDS if mechanism == "known" else TREE_G_BOUNDS
    return _sdr(
        OVERFIT_PLANS,
        reference="never",
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=n_folds,
        g_bounds=bounds,
    ).fit(frame, **OVERFIT_COLUMNS)


def longhand_transforms(
    frame: pd.DataFrame, result: Any, *, construction: str
) -> dict[str, np.ndarray]:
    r"""Each plan's first-node transform :math:`\varphi_1` of an overfitting fit, by hand.

    ``result`` is a known-mechanism :func:`overfit_fit`; its per-node ratios are fold-free, so
    the two constructions differ only in the outcome fits.  ``"training"`` is the shipped
    training-fold recursion: fold :math:`k` fits both regressions on its training rows and
    composes every transform from them.  ``"stitched"`` is Step 2 of Section 5.3 as printed:
    each row's node-2 transform reads the fold that holds it out, and node 1 regresses those
    stitched transforms.  ``tests/unit/test_longitudinal_sdr_design.py`` checks that
    ``"training"`` reproduces the fit.
    """
    assignment = result.folds.assignment
    folds = np.unique(assignment)
    baseline = frame[["W1", "W2"]].to_numpy(dtype=float)
    history = np.column_stack([baseline, np.nan_to_num(frame["L2"].to_numpy(dtype=float))])
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    outcome, pseudo = _grown()

    def horizon(train: np.ndarray, follow: np.ndarray) -> np.ndarray:
        model = clone(outcome).fit(history[train & follow], y[train & follow])
        return np.asarray(model.predict_proba(history)[:, -1], dtype=float)

    def earlier(train: np.ndarray, follow: np.ndarray, target: np.ndarray) -> np.ndarray:
        model = clone(pseudo).fit(baseline[train & follow], target[train & follow])
        return np.asarray(model.predict(baseline), dtype=float)

    out: dict[str, np.ndarray] = {}
    for plan in plans_of(result, OVERFIT_PLANS):
        ratios = supplied_ratios(result, plan)
        first = ratios[:, 0] > 0.0
        second = first & (ratios[:, 1] > 0.0)
        phi1 = np.zeros(len(frame))
        if construction == "training":
            for fold in folds:
                held = assignment == fold
                train = ~held if folds.size > 1 else np.ones(len(frame), dtype=bool)
                m2 = horizon(train, second)
                phi2 = np.where(first, m2 + ratios[:, 1] * (y - m2), 0.0)
                m1 = earlier(train, first, phi2)
                value = m1 + ratios[:, 0] * (m2 - m1) + ratios[:, 0] * ratios[:, 1] * (y - m2)
                phi1[held] = value[held]
        elif construction == "stitched":
            m2 = np.zeros(len(frame))
            for fold in folds:
                held = assignment == fold
                m2[held] = horizon(~held, second)[held]
            phi2 = np.where(first, m2 + ratios[:, 1] * (y - m2), 0.0)
            m1 = np.zeros(len(frame))
            for fold in folds:
                held = assignment == fold
                m1[held] = earlier(~held, first, phi2)[held]
            phi1 = m1 + ratios[:, 0] * (m2 - m1) + ratios[:, 0] * ratios[:, 1] * (y - m2)
        else:
            raise ValueError(f"unknown construction {construction!r}")
        out[plan.label] = phi1
    return out


def stitched_estimate(frame: pd.DataFrame, result: Any) -> tuple[float, float]:
    """The stitched Step 2 contrast ``always`` minus ``never`` and its standard error."""
    transforms = longhand_transforms(frame, result, construction="stitched")
    contrast = transforms["always"] - transforms["never"]
    return float(np.mean(contrast)), float(np.std(contrast, ddof=1) / np.sqrt(len(contrast)))


def overfit_draw(seed: int, n: int = OVERFIT_N) -> tuple[pd.DataFrame, float]:
    frame, truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, float(truth[OVERFIT_NAME])


# ---------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"sdr_family"``, ``"discrete"``, ``"discrete_null"``, ``"discrete_band"``,
        ``"overfit"``, ``"overfit_construction"``, ``"agreement"`` or a calibration law of
        :data:`CALIBRATION_LAWS`.  A diagnostic family reads its positive family's draws
        under a name of its own, so the collision gate can key a stream by law and seed.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "sdr_family":
            return dict(sdr_law.TRUTH)
        if self.name == "discrete_null":
            return {STATIC: float(end_properties.NULL_TRUTH)}
        if self.name == "discrete_band":
            return {name: float(law.TRUTH[name]) for name in BAND_NAMES}
        if self.name in {"overfit", "overfit_construction"}:
            return {OVERFIT_NAME: overfit_draw(0, n=10)[1]}
        if self.name == "agreement":
            return {AGREEMENT: 0.0}
        if self.name == "weighted":
            return {STATIC: TRUTH["selection_weighted"]}
        if self.name in CLUSTER_LAYOUT:
            return {STATIC: clustered_draw(self.name, 0)[1]}
        if self.name == "discrete":
            return {STATIC: TRUTH["static"], DYNAMIC_MEAN: TRUTH["dynamic"]}
        label = next(key for key, value in CALIBRATION_LAWS.items() if value == self.name)
        return {CALIBRATION_ESTIMANDS[label]: TRUTH[label]}


#: Every fit set: ``(family, stream label, configuration, n, replicates, law)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "sequential_double_robustness",
            f"{pattern}__n_{size}",
            pattern,
            size,
            SDR_REPLICATES,
            "sdr_family",
        )
        for pattern in SDR_PATTERNS
        for size in SDR_SIZES
    ),
    (
        "sequential_double_robustness",
        "controls",
        "controls",
        CONTROL_N,
        SDR_REPLICATES,
        "sdr_family",
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "discrete")
        for size in RATE_SIZES
    ),
    *(
        (
            "interval_calibration",
            stream,
            "both_correct",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            stream,
        )
        for stream in dict.fromkeys(CALIBRATION_LAWS.values())
    ),
    *(
        (
            "clustered_calibration",
            label,
            CLUSTERED_POSITIVE,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            label,
        )
        for label in CLUSTER_LAYOUT
    ),
    (
        "weighted_calibration",
        "selection_weighted",
        "correctly_specified",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "weighted",
    ),
    ("type_i_error", "sharp_null", "both_correct", NULL_N, NULL_REPLICATES, "discrete_null"),
    ("power", "alternative", "both_correct", NULL_N, NULL_REPLICATES, "discrete"),
    (
        "crossfit_overfitting",
        "paired",
        "overfit",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "overfit",
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The ``root_n_rate`` rows are fitted from the ladder's rows rather than sampled, so they
    are published in the summary and declared in ``PROPERTY_CELLS`` but not here.
    """
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        law_name: str,
        n: int,
        replicates: int,
        stream: str,
        estimand: str,
        *,
        stream_family: str | None = None,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(law_name),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=replicates,
                seed=_seed(stream_family or family, stream, 0),
                role=role,
                estimand=estimand,
            )
        )

    family = "sequential_double_robustness"
    for pattern in SDR_PATTERNS:
        for size in SDR_SIZES:
            stream = f"{pattern}__n_{size}"
            add(family, stream, "positive", "sdr_family", size, SDR_REPLICATES, stream, STATIC)
    for control in CONTROLS:
        add(family, control, "control", "sdr_family", CONTROL_N, SDR_REPLICATES, "controls", STATIC)
    for size in RATE_SIZES:
        add(
            "root_n_and_efficiency",
            f"static__n_{size}",
            "positive",
            "discrete",
            size,
            RATE_REPLICATES,
            f"n_{size}",
            STATIC,
        )
    for label in CALIBRATION_LABELS:
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                CALIBRATION_LAWS[label],
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                CALIBRATION_LAWS[label],
                CALIBRATION_ESTIMANDS[label],
            )
    for family, (labels, positive) in SEPARATE_CALIBRATION.items():
        for label in labels:
            law_name = "weighted" if label == "selection_weighted" else label
            for cell, role in (
                (positive, "positive"),
                ("shrunken_se_control", "control"),
                ("noise_control", "control"),
            ):
                add(
                    family,
                    f"{label}__{cell}",
                    role,
                    law_name,
                    CALIBRATION_N,
                    CALIBRATION_REPLICATES,
                    label,
                    STATIC,
                )
    add(
        "type_i_error",
        "static__sharp_null",
        "positive",
        "discrete_null",
        NULL_N,
        NULL_REPLICATES,
        "sharp_null",
        STATIC,
    )
    add(
        "power",
        "static__alternative",
        "positive",
        "discrete",
        NULL_N,
        NULL_REPLICATES,
        "alternative",
        STATIC,
    )
    for cell, role in (("cross_fitted_sdr", "positive"), ("in_sample_control", "control")):
        add(
            "crossfit_overfitting",
            cell,
            role,
            "overfit",
            OVERFIT_N,
            OVERFIT_REPLICATES,
            "paired",
            OVERFIT_NAME,
        )
    for cell in ("stitched_cross_fit", "estimated_flexible_mechanism"):
        add(
            "construction_diagnostic",
            cell,
            DIAGNOSTIC_ROLE,
            "overfit_construction",
            OVERFIT_N,
            OVERFIT_REPLICATES,
            "paired",
            OVERFIT_NAME,
            stream_family="crossfit_overfitting",
        )
    add(
        "sdr_tmle_agreement",
        "static",
        DIAGNOSTIC_ROLE,
        "agreement",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "discrete",
        AGREEMENT,
        stream_family="interval_calibration",
    )
    cells.extend(
        joint_property_cells(
            BAND_LABEL, n=PRIMARY_N, replicates=BAND_REPLICATES, seed=_seed(JOINT, BAND_LABEL, 0)
        )
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _calibration_frame(law_name: str, n: int, seed: int) -> tuple[pd.DataFrame, float | None]:
    """One calibration draw, with its truth where the truth varies by draw."""
    if law_name == "discrete":
        return end_properties.sample(law.PROBS, n, seed), None
    if law_name == "survival":
        return survival_properties.sample(survival_law.PROBS, n, seed), None
    if law_name == "competing":
        return competing_properties.sample(competing_law.PROBS, n, seed), None
    if law_name == "policy":
        return multivalue.sample(multivalue.PROBS, n, seed), None
    if law_name == "mtp":
        return mtp.sample_continuous(n, seed), None
    if law_name in {"msm", "msm_logit"}:
        return end_properties.sample(law.PROBS, n, seed), None
    if law_name == "weighted":
        return weighted.sample(law.PROBS, n, seed), None
    if law_name in CLUSTER_LAYOUT:
        frame, truth = clustered_draw(law_name, seed)
        if len(frame) != n:
            raise AssertionError(f"{law_name} drew {len(frame)} rows, not {n}")
        return frame, truth
    raise KeyError(law_name)


CALIBRATION_FITS = {
    "survival": survival_fit,
    "competing": competing_fit,
    "policy": policy_fit,
    "mtp": mtp_fit,
    "msm": msm_fit,
    "msm_logit": lambda frame: msm_fit(frame, link="logit"),
}


_Payload = tuple[str, str, str, int, int, int, str]
_Drawn = tuple[pd.DataFrame, float | None]


def _draw_fit_set(payload: _Payload) -> _Drawn:
    """The sample one draw of one fit set fits, with a truth that varies by draw."""
    family, label, _, replicate, n, _, law_name = payload
    seed = _seed(family, label, replicate)
    if family == "sequential_double_robustness":
        return sdr_law.sample(n, seed), None
    if family == "crossfit_overfitting":
        return overfit_draw(seed, n)
    if family in {"clustered_calibration", "weighted_calibration", "interval_calibration"}:
        return _calibration_frame(law_name, n, seed)
    probs = end_properties.NULL_PROBS if law_name == "discrete_null" else law.PROBS
    return end_properties.sample(probs, n, seed), None


def _fit_set_rows(payload: _Payload) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes, drawn and fitted in one call."""
    return _fit_set_drawn(payload, _draw_fit_set(payload))


def _fit_set_drawn(payload: _Payload, drawn: _Drawn) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes."""
    family, label, configuration, replicate, n, requested, law_name = payload
    alpha = STUDY.margins.alpha
    common = {"replicate": replicate, "n": n, "requested": requested}

    def row(
        cell: str, role: str, truth: float, estimate: Any, property_name: str = family
    ) -> dict[str, Any]:
        return replicate_row(
            property_name=property_name,
            cell=cell,
            role=role,
            truth=truth,
            estimate=estimate,
            alpha=alpha,
            **common,
        )

    frame, varying = drawn
    if family == "sequential_double_robustness":
        truth = float(sdr_law.TRUTH[STATIC])
        if configuration == "controls":
            return [
                row(cell, "control", truth, sdr_family_fit(frame, nuisances, engine)[STATIC])
                for cell, (engine, nuisances) in CONTROLS.items()
            ]
        result = sdr_family_fit(frame, PATTERN_NUISANCES[configuration])
        return [row(label, "positive", truth, result[STATIC])]
    if family == "crossfit_overfitting":
        assert varying is not None
        truth = varying
        cross_fitted = overfit_fit(frame, n_folds=N_FOLDS)
        flexible = overfit_fit(frame, n_folds=N_FOLDS, mechanism="tree")
        diagnostic = "construction_diagnostic"
        stitched = stitched_estimate(frame, cross_fitted)
        return [
            row("cross_fitted_sdr", "positive", truth, cross_fitted[OVERFIT_NAME]),
            row("in_sample_control", "control", truth, overfit_fit(frame, n_folds=1)[OVERFIT_NAME]),
            control_row(
                property_name=diagnostic,
                cell="stitched_cross_fit",
                truth=truth,
                estimate=stitched[0],
                standard_error=stitched[1],
                critical=CRITICAL,
                role=DIAGNOSTIC_ROLE,
                **common,
            ),
            row(
                "estimated_flexible_mechanism",
                DIAGNOSTIC_ROLE,
                truth,
                flexible[OVERFIT_NAME],
                diagnostic,
            ),
        ]
    if family == "clustered_calibration":
        estimate = clustered_fit(frame)[STATIC]
        assert varying is not None
        return [row(f"{label}__{CLUSTERED_POSITIVE}", "positive", float(varying), estimate)]
    if family == "weighted_calibration":
        estimate = weighted_fit(frame)[STATIC]
        return [row(f"{label}__{configuration}", "positive", TRUTH[label], estimate)]
    if family == "interval_calibration":
        if law_name == "discrete":
            result = discrete_fit(frame)
            ltmle = discrete_fit(frame, engine=LTMLE)
            estimate = result[STATIC]
            return [
                row("static__correctly_specified", "positive", TRUTH["static"], estimate),
                row(
                    "dynamic__correctly_specified",
                    "positive",
                    TRUTH["dynamic"],
                    result[DYNAMIC_MEAN],
                ),
                control_row(
                    property_name="sdr_tmle_agreement",
                    cell="static",
                    truth=0.0,
                    estimate=abs(float(estimate.psi) - float(ltmle[STATIC].psi))
                    / float(estimate.std_error),
                    standard_error=1.0,
                    critical=CRITICAL,
                    role=DIAGNOSTIC_ROLE,
                    **common,
                ),
            ]
        labels = [key for key, value in CALIBRATION_LAWS.items() if value == law_name]
        result = CALIBRATION_FITS[law_name](frame)
        return [
            row(
                f"{key}__correctly_specified",
                "positive",
                TRUTH[key] if varying is None else varying,
                result[CALIBRATION_ESTIMANDS[key]],
            )
            for key in labels
        ]
    truth = float(end_properties.NULL_TRUTH) if law_name == "discrete_null" else TRUTH["static"]
    result = discrete_fit(frame)
    return [row(f"static__{label}", "positive", truth, result[STATIC])]


def _draw_band(payload: tuple[int, int]) -> pd.DataFrame:
    return end_properties.sample(law.PROBS, PRIMARY_N, _seed(JOINT, BAND_LABEL, payload[0]))


def _band_rows(payload: tuple[int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    replicate, requested = payload
    result = discrete_fit(frame, regimens=BAND_REGIMENS, simultaneous=True)
    return joint_coverage_rows(
        result,
        {name: float(law.TRUTH[name]) for name in BAND_NAMES},
        BAND_NAMES,
        label=BAND_LABEL,
        replicate=replicate,
        n=PRIMARY_N,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


#: Each calibration positive arm to the controls derived from it.  ``interval_calibration``
#: arms come from :func:`calibration_derived`; the separate families derive the same two.
DERIVED_CELLS: dict[str, tuple[str, ...]] = {
    **calibration_derived(CALIBRATION_LABELS),
    **{
        f"{family}/{label}__{positive}": (
            f"{family}/{label}__shrunken_se_control",
            f"{family}/{label}__noise_control",
        )
        for family, (labels, positive) in SEPARATE_CALIBRATION.items()
        for label in labels
    },
}

#: A fit-set key is ``("fit_set", payload)`` and a band key is ``("band", (replicate, requested))``.
_Key = tuple[str, tuple[Any, ...]]


def _payloads(budget: int | None) -> list[_Key]:
    out: list[_Key] = []
    for family, label, configuration, n, replicates, law_name in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ("fit_set", (family, label, configuration, r, n, requested, law_name))
            for r in range(requested)
        )
    band = BAND_REPLICATES if budget is None else budget
    out.extend(("band", (r, band)) for r in range(band))
    return out


def _draw(key: _Key) -> Any:
    """The sample a key fits, a function of the key alone."""
    kind, payload = key
    return _draw_band(payload) if kind == "band" else _draw_fit_set(payload)


def _run(key: _Key, drawn: Any) -> list[dict[str, Any]]:
    kind, payload = key
    return list(_band_rows(payload, drawn) if kind == "band" else _fit_set_drawn(payload, drawn))


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    kind, payload = key
    if kind == "band":
        return [f"{JOINT}/{name}" for name in joint_cells(BAND_LABEL)]
    family, label, configuration, _, _, _, law_name = payload
    if family == "sequential_double_robustness":
        if configuration == "controls":
            return [f"{family}/{cell}" for cell in CONTROLS]
        return [f"{family}/{label}"]
    if family == "crossfit_overfitting":
        return [
            f"{family}/cross_fitted_sdr",
            f"{family}/in_sample_control",
            "construction_diagnostic/stitched_cross_fit",
            "construction_diagnostic/estimated_flexible_mechanism",
        ]
    if family in {"clustered_calibration", "weighted_calibration"}:
        positive = CLUSTERED_POSITIVE if family == "clustered_calibration" else configuration
        name = f"{family}/{label}__{positive}"
        return [name, *DERIVED_CELLS[name]]
    if family == "interval_calibration":
        if law_name == "discrete":
            names = [
                f"{family}/static__correctly_specified",
                f"{family}/dynamic__correctly_specified",
            ]
            derived = [control for name in names for control in DERIVED_CELLS[name]]
            return [*names, "sdr_tmle_agreement/static", *derived]
        names = [
            f"{family}/{key}__correctly_specified"
            for key, value in CALIBRATION_LAWS.items()
            if value == law_name
        ]
        return [*names, *(control for name in names for control in DERIVED_CELLS[name])]
    return [f"{family}/static__{label}"]


def _cell_of(key: _Key) -> str:
    kind, payload = key
    return f"{JOINT}/{BAND_LABEL}" if kind == "band" else f"{payload[0]}/{payload[1]}"


def _replicate_of(key: _Key) -> int:
    kind, payload = key
    return payload[0] if kind == "band" else payload[3]


def _size(key: _Key) -> int:
    """The rows of a payload's fit: its declared ``n``, or the primary size for a band row."""
    kind, payload = key
    return int(payload[4]) if kind == "fit_set" else PRIMARY_N


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check.

    The payloads run in one pool per size, smallest first.  Each pool's workers are capped by
    the free memory read just before it starts, at :func:`fit_bytes` per worker.  The worker
    count changes no row.
    """
    jobs = _payloads(budget)
    outcomes: list[Any] = []
    for size in sorted({_size(job) for job in jobs}):
        group = [job for job in jobs if _size(job) == size]
        workers = memory_capped_workers(n_jobs, fit_bytes(size))
        outcomes += fit_properties(
            group,
            _run,
            cells_of=_cells_of,
            cell_of=_cell_of,
            replicate_of=_replicate_of,
            n_jobs=workers,
            label=f"properties/n{size}",
            draw=_draw,
            derived=DERIVED_CELLS,
        )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
            separate_controls(rows),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def separate_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The two derived arms of each :data:`SEPARATE_CALIBRATION` label, as in ``interval_calibration``.

    The shrunken arm multiplies the reported standard error by :data:`SHRUNKEN_SE_FACTOR`.  The
    noise arm adds normal noise of scale :data:`NOISE_SD` over root n to each estimate.
    """
    controls: list[pd.DataFrame] = []
    for family, (labels, positive) in SEPARATE_CALIBRATION.items():
        for label in labels:
            controls.extend(_derived_arms(rows, family, label, positive))
    return pd.concat(controls, ignore_index=True)


def _derived_arms(rows: pd.DataFrame, family: str, label: str, positive: str) -> list[pd.DataFrame]:
    base = rows.loc[(rows["property"] == family) & (rows["cell"] == f"{label}__{positive}")]
    if base.empty:
        raise ValueError(f"{label}__{positive} has no rows to derive controls from")
    shrunken = base.copy()
    shrunken["cell"] = f"{label}__shrunken_se_control"
    shrunken["role"] = "control"
    shrunken["std_error"] *= SHRUNKEN_SE_FACTOR
    noisy = base.copy()
    rng = np.random.default_rng(stream_seed(STUDY, "efficiency_noise", label))
    noisy["cell"] = f"{label}__noise_control"
    noisy["role"] = "control"
    noisy["estimate"] += indexed_noise(
        rng, base, label, scale=NOISE_SD[label] / np.sqrt(CALIBRATION_N)
    )
    return [
        _recompute_interval_columns(shrunken, CRITICAL),
        _recompute_interval_columns(noisy, CRITICAL),
    ]


def failure_probe(draws: int, *, n_jobs: int = 1, start: int = 0) -> dict[str, int]:
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``.

    It records no estimate.  A fit set with any failure is dropped before the declared run.
    """
    payloads = [
        ((family, label, configuration, r, n, draws, law_name),)
        for family, label, configuration, n, _, law_name in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{label}": 0 for family, label, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


# ---------------------------------------------------------------------------- verdicts


def _bias_bound(summary: pd.DataFrame, index: Any) -> float:
    return absolute_bias_bound(summary.loc[index, "bias"], summary.loc[index, "bias_se"])


def sequential_double_robustness_verdicts(summary: pd.DataFrame) -> None:
    """The family's three rules (``X4-plan.md`` 6.8), stated in the module docstring."""
    margins = STUDY.margins
    family = summary["property"] == "sequential_double_robustness"
    for index in summary.index[family]:
        summary.loc[index, BIAS_BOUND_COLUMN] = _bias_bound(summary, index)
    scaled = summary["se_ratio"].between(*UNION_MODEL_SE_BAND)
    for pattern in SDR_PATTERNS:
        cells = {
            size: summary.index[family & (summary["cell"] == f"{pattern}__n_{size}")][0]
            for size in SDR_SIZES
        }
        if pattern in {"both_correct", "P_C"}:
            for index in cells.values():
                summary.loc[index, "passed"] = bool(
                    summary.loc[index, "bias_equivalent"]
                    and scaled[index]
                    and summary.loc[index, "coverage_ci_lower"] >= margins.coverage_floor
                )
            continue
        small, large = (summary.loc[cells[size], BIAS_BOUND_COLUMN] for size in SDR_SIZES)
        verdict = bool(large < SDR_TOLERANCE and large < small and scaled[cells[SDR_SIZES[1]]])
        for index in cells.values():
            summary.loc[index, "passed"] = verdict
    for control in CONTROLS:
        index = summary.index[family & (summary["cell"] == control)][0]
        low = float(summary.loc[index, "bias_ci_lower"])
        high = float(summary.loc[index, "bias_ci_upper"])
        scale = float(summary.loc[index, "empirical_se"])
        clear = np.sign(low) == np.sign(high) and min(abs(low), abs(high)) >= CONTROL_FLOOR * scale
        summary.loc[index, "passed"] = bool(clear and scaled[index])


def separate_calibration_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The rule of each :data:`SEPARATE_CALIBRATION` family: SE ratio and coverage only.

    The positive arm passes inside both calibration bands.  Both controls must leave the
    SE-ratio band below: the shrunken arm by its factor, the noise arm because added spread
    lowers the ratio.
    """
    margins = STUDY.margins
    for family, (_, positive) in SEPARATE_CALIBRATION.items():
        for index in summary.index[summary["property"] == family]:
            cell = str(summary.loc[index, "cell"])
            ratio = se_ratio_interval(
                rows.loc[(rows["property"] == family) & (rows["cell"] == cell)],
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, family, cell),
            )
            summary.loc[index, "se_ratio_ci_lower"] = ratio.low
            summary.loc[index, "se_ratio_ci_upper"] = ratio.high
            if cell.endswith(f"__{positive}"):
                coverage = summary_interval(summary, index, "coverage")
                passed = ratio.within(*margins.calibration_se_ratio) and coverage.within(
                    *margins.calibration_coverage
                )
            else:
                passed = se_ratio_leaves_band(ratio, margins.calibration_se_ratio, "below")
            summary.loc[index, "passed"] = bool(passed)


def diagnostic_rows(summary: pd.DataFrame) -> None:
    """Publish the two reported families with no verdict."""
    construction = summary["property"] == "construction_diagnostic"
    summary.loc[construction, "passed"] = True
    summary.loc[construction, "property_passed"] = True
    agreement = summary["property"] == "sdr_tmle_agreement"
    summary.loc[agreement, AGREEMENT_COLUMN] = summary.loc[agreement, "mean_estimate"]
    for column in (
        "bias_margin",
        "standardized_bias",
        "root_n_bias",
        "se_ratio",
        "rejection_rate",
        "rejection_ci_lower",
        "rejection_ci_upper",
    ):
        if column in summary:
            summary.loc[agreement, column] = np.nan
    for column in ("bias_equivalent", "bias_discriminated"):
        if column in summary:
            summary[column] = summary[column].astype(object)
            summary.loc[agreement, column] = np.nan
    summary.loc[agreement, "passed"] = True
    summary.loc[agreement, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            BIAS_BOUND_COLUMN,
            AGREEMENT_COLUMN,
        ),
        rate_labels=("static",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    separate_calibration_verdicts(summary, rows)
    sequential_double_robustness_verdicts(summary)
    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_sdr")
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    diagnostic_rows(summary)
    return finish(summary, rates)


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "failure_probe",
    "generate_property_rows",
    "summarize_properties",
]


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)
