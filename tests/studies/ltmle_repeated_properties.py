"""Independent repeated-sampling properties of repeated-outcome LTMLE (X32a).

Every binary family samples :func:`cleverly.datasets.make_longitudinal_repeated` on its own
stream ``stream_seed(STUDY, "property_sample", family, configuration, replicate)``.  Every
truth is enumerated.  The binary truths come from :func:`cleverly.datasets.repeated_truth`.
:mod:`tests.studies.ltmle_repeated_law` states the control limits and the continuous law.
The labels name the parameters: ``never_t3`` is ``ey_regimen[never @ t=3]`` and
``rule_vs_never_t3`` is the rule's contrast at ``t = 3``.  The five ``*_t3`` labels are the
gated labels.  The ``interval_calibration`` arms draw ``n = 8,000``
(:data:`CALIBRATION_N`).  So do ``repeated_outcome_necessity`` (:data:`ABSORBING_N`) and
``full_history_necessity`` (:data:`FULL_HISTORY_N`).  Every other family draws
``n = 2,000`` (:data:`N`).  Budgets and the revision notes say why.

Declared before any run
-----------------------

==============================  =========================================================
family and cells                what is fitted, replications and size
==============================  =========================================================
C1 ``interval_calibration``     ``*_t3__estimated_mechanisms``: in sample, the mechanisms
                                estimated by a main-terms logistic GLM on the declared
                                design (correctly specified), quasibinomial GLM outcome
                                regressions.  ``simultaneous=True``.
                                :data:`CALIBRATION_REPLICATES` replications.  The derived
                                ``shrunken_se_control`` and ``noise_control`` arms read
                                these rows.  The same fits read at ``t = 1`` and ``t = 2``
                                are the C1 cells of the eight earlier-horizon labels,
                                gated by the same rule
``small_sample_calibration``    ``*_t3__{estimated_mechanisms,continuous_gaussian}``: C1
                                and C6 refitted at ``n = 2,000``.  Role ``diagnostic``:
                                reported, never gated.  2,000 replications
C2 ``double_robustness``        Three arms on the five gated labels, ``n = 2,000``, 1,400
                                replications.  ``mechanism_correct``: the C1 mechanisms with
                                an outcome regression on ``W`` alone.  ``outcome_correct``:
                                saturated full-history outcome regressions
                                (:class:`CellMeansBackoff`) with mechanisms on ``W`` alone
                                (:class:`BaselineOnly`), which are wrong and still vary, so
                                the fluctuation moves.  ``both_wrong``: the ``W``
                                regression with intercept-only mechanisms.  The two
                                positive arms must be unbiased.  ``both_wrong`` is a
                                control that must miss the truth and reach
                                :data:`BOTH_WRONG_LIMIT`
C3 ``interval_calibration``     ``*_t3__crossfit_glm``: C1 at five folds
C4 ``simultaneous_coverage``    ``all_reported``: the default band of the C1 fits over all
                                fifteen reported parameters, and its pointwise control
C5 ``interval_calibration``     ``*_t3__crossfit_boosted``: five folds with a shallow
                                regularized booster (depth 2, 50 rounds, rate 0.1, L2
                                penalty 1, at least 20 rows a leaf, no early stopping) for
                                both outcome learners
C6 ``interval_calibration``     ``*_t3__continuous_gaussian``: the three-value law
                                :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`,
                                least-squares outcome regressions, ``q_bounds = (0, 1)``,
                                C1 mechanisms
C7 ``rule_necessity``           ``rule_t{2,3}__declared`` against ``__reversed_rule``, the
                                rule that treats after a zero outcome.  The positive arm
                                reads the declared rule's truth.  The control must miss it
                                and reach :data:`REVERSED_TRUTH`.  2,000 replications
N1 ``repeated_outcome_neces     ``{plan}_t{2,3}__repeated`` against ``__absorbing_reading``,
sity``                          an end-of-study fit of :math:`\\max_{s \\le h} Y_s` with the
                                earlier outcomes in the history.  The control must miss the
                                truth and reach :data:`ABSORBING_LIMIT`.  Its outcome
                                learner is :class:`CellMeansBackoff`.  ``n = 8,000``.
                                2,800 replications
N2 ``outcome_history_necessi    ``{never,always,always_vs_never}_t3__repeated`` against
ty``                            ``__history_free``, an end-of-study fit with every ``Y_s``
                                left out of the history and saturated cell-mean outcome
                                regressions.  The earlier outcomes are confounders here.
                                The control must miss the truth and reach
                                :data:`HISTORY_FREE_LIMIT`.  1,200 replications
N3 ``full_history_necessity``   ``{label}__full_history`` against ``__markov_history``,
                                five labels, ``n = 8,000``, 1,200 replications.  Both arms
                                use intercept-only mechanisms, so the outcome regression
                                alone carries the consistency of the positive arm.  The
                                positive arm regresses on the full history, and the control
                                on ``[W, Y_{t-1}, L_t]``.  The control must miss the truth
                                and reach :data:`MARKOV_LIMIT`, the exact limit of the
                                single-fold fit (``tmle_limit`` in the law module)
T1 ``targeting_necessity``      ``{label}__targeted`` against ``__untargeted``, five
                                labels, ``n = 2,000``, 1,600 replications.  The targeted
                                arm is the ``mechanism_correct`` fit.  The control is
                                :func:`untargeted`, the same sequential regression with no
                                fluctuation, written without cleverly.  It must move the
                                estimate by :data:`TARGETING_DISPLACEMENT` spreads of the
                                targeted arm and reach :data:`PLUG_IN_LIMIT`, which is
                                exact because no fluctuation exists
==============================  =========================================================

Gating, declared before any run:

* the SE-ratio band gates every C1 cell, at every horizon.  The other calibration arms
  gate the ``t = 3`` labels;
* every other positive cell answers to its family's shared rule, and each control to the
  opposite direction of the same instrument.  A control with an alternative target also
  answers to its own limit: its bias around the limit must lie inside the margin;
* a red row is diagnosed.  A defect is fixed, and a finite-sample shortfall gets the
  ``n`` that a pilot shows removes it.  Either change is re-declared and re-run on fresh
  seeds.  No red row is deferred to a later roadmap item.

Budgets.  Both budgets come from declared rules (:func:`primary_replicates` and
:func:`calibration_replicates`) applied to pilots on the stream
``stream_seed(STUDY, "pilot", ...)``.  No published cell reads that stream.  The
calibration rule is the smallest multiple of 400 whose predictive family-wise pass
probability (:func:`calibration_family_pass_probability`) reaches 0.95.  The pooled reading
below defines that probability.  The pilot numbers are in the section "Fourth declaration
pilots".

Pooled reading.  The first version of the calibration rule put each of the 28 cells at the
far end of its own 80% pilot interval at once, and multiplied the per-cell pass
probabilities.  That rule priced the pilot noise of a ratio once for each cell, and the
product priced it again.  It fitted no model of how the cells relate, and it set the budget
at 25,600 replications.  The pooled rule models the relation.

* Each calibration arm is a group.  The true SE ratio of a cell is the arm's true mean plus a
  deviation of standard deviation :math:`\\tau`.
* The pilot mean of an arm has the noise variance :math:`s^2 / k_{\\mathrm{eff}}`, where
  :math:`s^2 = 1 / (2 (R_0 - 1))` is the noise variance of one ratio at :math:`R_0 = 2{,}000`
  pilot draws.  Two cells of an arm have the pilot-noise correlation :math:`\\rho_{ij}^2`,
  the square of the correlation of their efficient influence curves under the law
  (:func:`label_correlations`).  So :math:`k_{\\mathrm{eff}} = k^2 / \\sum_{ij} \\rho_{ij}^2`
  (:func:`effective_cells`).  It is 6.69 for the 13 in-sample cells and 2.58 for each
  5-cell arm.  A test derives both from the law.
* :func:`heterogeneity_bound` reads :math:`\\tau` as the 80% upper confidence bound on the
  spread between cells, given the spread that the pilot noise alone explains.  The pilot
  spread is smaller than that noise, so the point estimate is zero and the bound is
  0.0059.
* :func:`pooled_pass_probability` integrates the cell pass probability
  (:func:`calibration_pass_probability`) over the arm mean and over :math:`\\tau`, and
  multiplies the arms.

The probability is 0.9523 at 8,000 replications and 0.9437 at 7,600, so
``CALIBRATION_REPLICATES = 8,000``.  At :math:`\\tau = 0` it is 0.9785.  The property
families read the same pooled reading for their bias checks.  Each family is a group, and
:func:`property_bias_replicates` sizes them.  See P4 below.

Revision after the first declared run (5728c3ce).  Every positive cell passed.  Two N1
controls failed their second condition: ``always_t3`` and ``rule_t3`` missed the truth as
declared, but did not reach :data:`ABSORBING_LIMIT`.  On one draw at ``n = 50,000`` the
control reaches the limit within 0.001.  The cause is the control's learner.  An unseen
cell fell back to the training mean, the fallback that :class:`CellMeansBackoff` replaces.
Over 150 draws at ``n = 2,000`` the backoff leaves a mean bias of 0.0096 for the rule, near
its margin.  Over 120 draws at ``n = 8,000`` every absorbing control is within 0.001 of its
limit.  N1 therefore draws ``n = 8,000`` with the backoff learner.  The first run also
read the earlier horizons as a reported diagnostic.  Their intervals ship, so C1 now gates
them.  The first run's rows informed both changes, so the second declaration drew fresh
seeds for every family.

Third run (52cb8596).  The third run changed the R reference initial estimate only.  The
property rows are the second declaration's rows.

Fourth declaration (``SEED = 20263204``).  A design review of the third run found defects
that this module's families address, before any new draw:

* No family showed that targeting matters.  ``targeting_necessity`` (T1) now compares the
  targeted fit with a longhand plug-in on the same draw.
* The law was first-order Markov in ``Y`` and ``L``, so no check read the history of the
  outcome regression beyond one node.  The law now has depth two, and N3 separates a
  full-history regression from a Markov one.  N2 measures the earlier outcomes as
  confounders of the later nodes.  N3 measures the depth of the outcome regression.
* The ``both_wrong`` and reversed-rule controls read a declared limit that no verdict
  used.  ``both_wrong`` now reads :data:`BOTH_WRONG_LIMIT` through
  ``alternative_target_verdicts``.  The old limit covered static plans only.  The new
  limit also covers the rule, whose clever covariate reads ``Y_{t-1}``, so the intercept
  fluctuation moves.  C7 reads :data:`REVERSED_TRUTH`.
* ``double_robustness`` read one positive arm.  ``outcome_correct`` now shows consistency
  with the outcome regression correct and the mechanisms wrong.
* The calibration budget came from a per-cell rule. It is now the pooled predictive
  family-wise probability over 28 cells, and every arm is piloted at its declared ``n``.
* ``W`` takes three values and the first outcome regression is no longer saturated.  The
  first-node fluctuation therefore moves.

Fourth declaration pilots.  Every pilot draws from ``stream_seed(STUDY, "pilot", ...)``.
The hooks are :func:`pilot_se_ratios` (P2, P3), :func:`pilot_property_rows` (P4) and
:func:`pilot_primary` (P1).

* P2, calibration, ``n = 8,000``, 2,000 draws per arm.  The SE ratios of the 28 gated cells
  read 0.9795 to 1.0278 for the 13 in-sample cells, 0.974 to 1.0136 for ``crossfit_glm``,
  1.0073 to 1.0279 for ``crossfit_boosted`` and 0.992 to 1.0055 for ``continuous_gaussian``.
  They are recorded in :data:`CALIBRATION_PILOT_RATIOS`.  Their mean is 1.006 and their
  spread across cells is 0.014, which is the sampling error of one ratio at 2,000 draws.
  The arm means are 1.0088 (13 in-sample cells), 0.9969 (``crossfit_glm``), 1.0162
  (``crossfit_boosted``) and 0.9966 (``continuous_gaussian``).  :func:`calibration_replicates`
  applies the pooled reading to them.  The family-wise pass probability first reaches 0.95
  at 8,000 replications (0.9523, against 0.9437 at 7,600), so
  ``CALIBRATION_REPLICATES = 8,000``.  The rule is applied as declared and is not capped.
* P3, ``n = 2,000``, 2,000 draws.  The in-sample ratios of the five gated labels read 0.946
  (``rule_t3``) to 1.003 (``never_t3``), and the continuous arm reads 0.970 to 0.995.  The
  shortfall is the ``1/n`` in-sample deficit, and the gated arms draw ``n = 8,000``.
* P4, 300 draws per family at its own size.  ``outcome_correct`` is unbiased: the largest
  bias is 0.060 empirical spreads (``rule_t3``), under the 0.10 that would move the family to
  ``n = 4,000``.  The largest bias of any positive arm is 0.137 spreads at 300 draws, whose
  own sampling error is 0.058.  Each control sits within 0.108 spreads of its limit.  The
  smallest control gap is 0.59 spreads (``always_t3`` of N2), then 0.80 (``rule_t3`` of
  ``both_wrong``).  The full-history outcome regression meets an unseen cell for 0.9% of the
  rule's rows at the third node and 0.4% to 0.2% of the static rows, so the backoff rarely
  acts.  The largest share in one draw is 3.3%.
* P4 biases.  The 57 standardized biases of the positive cells and of the controls around
  their limits are in :data:`PROPERTY_PILOT_BIASES`.  The declared reading gives each family
  its own mean (:func:`property_bias_groups`), because a finite-sample bias that a family
  shares is plausible.  The pilot supports it for N1, whose 12 cells average 0.058 against a
  noise of 0.024 in that mean.  The pilot spread of cells within families is smaller than the pilot
  noise, so the point estimate of the spread is zero, and the 80% upper bound on the spread between cells is 0.0325.  The probability that every
  bias check passes at 1,000 replications for each family is then 0.75, and N1 binds
  at 0.86.  :func:`property_bias_replicates` raises each family to the smallest multiple of 200
  at which its own probability reaches 0.95 to the power 1/6.  The sizes are 1,400
  (``double_robustness``), 1,600 (``targeting_necessity``), 2,000 (``rule_necessity``), 2,800
  (``repeated_outcome_necessity``) and 1,200 (``outcome_history_necessity`` and
  ``full_history_necessity``).  The probability that every check passes is 0.961.  At a spread
  of zero it is 0.999.  The cross-check puts all cells in one group with one mean, which
  the pilot does not argue for.  It gives a bound of 0.0329 and a probability of 0.994 at these
  sizes, and 0.962 at 1,000 replications for each family.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.metaestimators import available_if

from cleverly.datasets import make_longitudinal_repeated
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.utils.parallel import map_parallel
from tests.discrete_law_longitudinal import CellMeans
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_repeated_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.canonical_ltmle_repeated import (
    COLUMNS,
    EARLIER_LABELS,
    ESTIMANDS,
    G_BOUNDS,
    GATED_LABELS,
    PRIMARY_N,
    RULE_LABEL,
    STUDY,
    fit_cleverly,
    label_estimand,
    treat_after,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    Necessity,
    alternative_target_necessity_verdicts,
    alternative_target_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT_FAMILY
from tests.studies.evidence.simultaneous import joint_cells, joint_coverage_rows
from tests.studies.fractional_glm import QuasiBinomialGLM, QuasiBinomialSeparationWarning
from tests.studies.repeated_sizing import BUDGET_STEP as BUDGET_STEP
from tests.studies.repeated_sizing import FAMILY_WISE_PASS as FAMILY_WISE_PASS
from tests.studies.repeated_sizing import PILOT_LEVEL as PILOT_LEVEL
from tests.studies.repeated_sizing import PRIMARY_CELL_PASS as PRIMARY_CELL_PASS
from tests.studies.repeated_sizing import PRIMARY_DESIGN_BIAS as PRIMARY_DESIGN_BIAS
from tests.studies.repeated_sizing import PRIMARY_DESIGN_COVERAGE as PRIMARY_DESIGN_COVERAGE
from tests.studies.repeated_sizing import PROPERTY_STEP as PROPERTY_STEP
from tests.studies.repeated_sizing import PilotGroup as PilotGroup
from tests.studies.repeated_sizing import heterogeneity_bound as heterogeneity_bound
from tests.studies.repeated_sizing import pooled_pass_probability as pooled_pass_probability

N = 2_000
#: The sample size of every ``interval_calibration`` arm.  See the module docstring, Budgets.
CALIBRATION_N = 8_000
CALIBRATION_REPLICATES = 8_000
#: The sample size of the N1 family, ``repeated_outcome_necessity``.  See Budgets.
ABSORBING_N = 8_000
#: The sample size of the N3 family, ``full_history_necessity``.  See Budgets.
FULL_HISTORY_N = 8_000
#: The in-sample arms refitted at :data:`N` and reported under ``small_sample_calibration``.
SMALL_SAMPLE_ARMS = ("estimated_mechanisms", "continuous_gaussian")
SMALL_SAMPLE_REPLICATES = 2_000
#: The replications of each property family, from :func:`property_bias_replicates`.  A design
#: test re-derives them from the pilot biases.  See the module docstring, Budgets.
DOUBLE_ROBUST_REPLICATES = 1_400
TARGETING_REPLICATES = 1_600
RULE_REPLICATES = 2_000
ABSORBING_REPLICATES = 2_800
HISTORY_FREE_REPLICATES = 1_200
FULL_HISTORY_REPLICATES = 1_200
FOLDS = 5
SHRUNKEN_SE_FACTOR = 0.70
#: The least paired displacement, in the positive arm's empirical standard deviations, of
#: every necessity family here: the standardized-bias margin.
NECESSITY_DISPLACEMENT = 0.25
#: How far removing the fluctuation must move the estimate, in the targeted arm's empirical
#: standard deviations, before ``targeting_necessity`` calls the targeting step load bearing.
#: The standardized-bias margin again: the control must move the estimate by as much as the
#: study is prepared to overlook.
TARGETING_DISPLACEMENT = 0.25
#: The red-cell owner of a row the finite-sample rule admits.
FINITE_SAMPLE_OWNER = "X32-finite-sample"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The positive calibration arms, in the order the table above declares them.
CALIBRATION_ARMS = (
    "estimated_mechanisms",
    "crossfit_glm",
    "crossfit_boosted",
    "continuous_gaussian",
)
STATIC_GATED = ("never_t3", "always_t3", "always_vs_never_t3")
RULE_LABELS = ("rule_t2", "rule_t3")
ABSORBING_LABELS = ("never_t2", "never_t3", "always_t2", "always_t3", "rule_t2", "rule_t3")
HISTORY_LABELS = STATIC_GATED
JOINT_LABEL = "all_reported"

_PLAN = {"never": "never", "always": "always", RULE_LABEL: law.RULE, "reversed": "reversed"}
REVERSED = "reversed"

# ---------------------------------------------------------------------------- truths

_, TRUTH = make_longitudinal_repeated(n=1, seed=0, backend="pandas")
TRUTH = {name: float(value) for name, value in TRUTH.items()}


def _law_value(
    model: law.RepeatedOutcomeLaw, estimand: str, method: str = "g_formula", **options: Any
) -> float:
    """``estimand`` on ``model`` through one of its plan functionals."""
    if estimand.startswith("ate_regimen["):
        comparison, horizon = estimand[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        return _law_value(
            model, f"ey_regimen[{left} @ t={horizon}]", method, **options
        ) - _law_value(model, f"ey_regimen[{right} @ t={horizon}]", method, **options)
    plan, horizon = estimand[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
    return float(getattr(model, method)(_PLAN[plan], int(horizon), **options))


CONTINUOUS_TRUTH = {
    label: _law_value(law.CONTINUOUS, label_estimand(label)) for label in GATED_LABELS
}
ABSORBING_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), absorbing=True)
    for label in ABSORBING_LABELS
}
HISTORY_FREE_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), history="free") for label in HISTORY_LABELS
}
#: The limit of the plug-in whose outcome regression is the cell mean given ``W``: the
#: ``targeting_necessity`` control, exactly, because it has no fluctuation.
PLUG_IN_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), history="baseline")
    for label in GATED_LABELS
}
#: The limit of the single-fold fit with a ``W`` cell-mean outcome regression and intercept-only
#: mechanisms: the ``both_wrong`` control.  A static plan's fluctuation is inert there, so
#: the value equals :data:`PLUG_IN_LIMIT`.  The rule's does not, because its clever covariate
#: reads ``Y_{t-1}``.  The two limits differ by 0.046 empirical SDs for ``rule_t3`` at
#: ``n = 2,000``, which 1,400 replications cannot separate.  The claim that the rule's
#: fluctuation moves the fit rests on the exact sample test of the limit functional in
#: ``tests/unit/test_ltmle_repeated_method_study.py``, and not on this control.
BOTH_WRONG_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), "tmle_limit", q_history="baseline")
    for label in GATED_LABELS
}
#: The limit of the single-fold fit with an outcome regression on ``[W, Y_{t-1}, L_t]`` and
#: intercept-only mechanisms: the N3 control.
MARKOV_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), "tmle_limit", q_history="markov")
    for label in GATED_LABELS
}
#: The truth of the reversed rule, the alternative target of the C7 control.
REVERSED_TRUTH = {
    label: law.BINARY.counterfactual(REVERSED, int(label.rsplit("_t", 1)[1]))
    for label in RULE_LABELS
}


def _eif_sd(model: law.RepeatedOutcomeLaw, label: str) -> float:
    estimand = label_estimand(label)
    if estimand.startswith("ate_regimen["):
        comparison, horizon = estimand[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        return model.eif_sd(_PLAN[left], int(horizon), reference=_PLAN[right])
    plan, horizon = estimand[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
    return model.eif_sd(_PLAN[plan], int(horizon))


#: The efficient influence curve's standard deviation of each gated label.  It scales the
#: noise control.  The study claims no efficiency, so no efficiency band reads it.
EFFICIENCY_SD = {label: _eif_sd(law.BINARY, label) for label in GATED_LABELS}

# ---------------------------------------------------------------------------- learners


class BaselineOnly(BaseEstimator):
    """``learner`` fitted on the first design column alone, which is ``W``.

    The misspecified, non-constant mechanism of the ``outcome_correct`` arm: it drops every
    ``Y_{t-1}`` and ``L_t`` term, and still varies with ``W``, so the fluctuation it feeds
    moves.
    """

    def __init__(self, learner: Any) -> None:
        self.learner = learner

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> BaselineOnly:
        self.fitted_ = clone(self.learner).fit(
            np.asarray(X, dtype=float)[:, :1], y, sample_weight=sample_weight
        )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict(np.asarray(X, dtype=float)[:, :1]))

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict_proba(np.asarray(X, dtype=float)[:, :1]))


class CellMeansBackoff(BaseEstimator):
    """Cell means on a seen design row, and a main-terms quasibinomial GLM on an unseen one.

    The outcome learner of every control that reads cells of a coarse or full history.
    :class:`CellMeans` falls back to the training mean on a design row the fit never saw.
    The declared run at 5728c3ce read that fallback at the rule's third node, where some
    histories of the followers through node 2 have no follower through node 3.  The control
    then missed its own limit: ``rule_t3__absorbing_reading`` reached 0.569 against
    :data:`ABSORBING_LIMIT` 0.615, and ``always_t3`` 0.728 against 0.740.  A GLM on the same
    rows predicts an unseen cell from its neighbours, so the fit converges to the same limit
    with less finite-sample bias.

    The GLM separates by construction in the N1 control.  Once an earlier outcome is 1, the
    running maximum is 1, so that history column predicts the response exactly.  The last
    iterate then predicts close to 1 on such a row, which is the limit.  The fit therefore
    keeps it and drops :class:`QuasiBinomialSeparationWarning`.  A fit whose deviance does
    not settle still raises.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeansBackoff:
        matrix = np.asarray(X, dtype=float)
        self.cells_ = CellMeans().fit(matrix, y, sample_weight=sample_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QuasiBinomialSeparationWarning)
            self.backoff_ = QuasiBinomialGLM().fit(matrix, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        keys = self.cells_.keys_
        rounded = np.round(matrix, 9)
        seen = (rounded[:, None, :] == keys[None, :, :]).all(axis=2).any(axis=1)
        return np.where(
            seen, self.cells_.predict(matrix), np.asarray(self.backoff_.predict(matrix))
        )


class CellMeansOn(BaseEstimator):
    """``learner`` fitted on the columns of the outcome design that a history kind names.

    A repeated-outcome fit hands the outcome regression at node ``t`` the design
    ``[W, Y1, L2, ..., Y_{t-1}, L_t]``, so its width is 1, 3 or 5.  ``baseline`` keeps ``W``
    alone.  ``markov`` keeps ``[W, Y_{t-1}, L_t]``, the last two columns and ``W``, which is
    the whole design at widths 1 and 3.  ``full`` keeps every column.

    Parameters
    ----------
    kind : str
        ``"baseline"``, ``"markov"`` or ``"full"``.
    learner : estimator
        The learner fitted on the selected columns.  ``predict_proba`` is offered only when
        the learner has it, because a classifier-versus-regressor choice reads the attribute.
    """

    def __init__(self, kind: str, learner: Any) -> None:
        self.kind = kind
        self.learner = learner

    @staticmethod
    def columns(kind: str, width: int) -> list[int]:
        """The design columns ``kind`` keeps at a design of ``width`` columns."""
        if kind == "full":
            return list(range(width))
        if kind == "baseline" or width == 1:
            return [0]
        if kind == "markov":
            return [0, width - 2, width - 1]
        raise ValueError(kind)

    def _select(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return matrix[:, self.columns(self.kind, matrix.shape[1])]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeansOn:
        self.fitted_ = clone(self.learner).fit(self._select(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict(self._select(X)))

    @available_if(lambda self: hasattr(self.learner, "predict_proba"))
    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict_proba(self._select(X)))


def booster(kind: str) -> Any:
    """The shallow regularized booster of C5, never the library default."""
    options = {
        "max_depth": 2,
        "max_iter": 50,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": 0,
    }
    if kind == "classifier":
        return HistGradientBoostingClassifier(**options)
    return HistGradientBoostingRegressor(**options)


#: The configurations whose outcome regression is the cell mean given ``W`` alone.
BASELINE_Q = frozenset({"mechanism_correct", "both_wrong", "targeted"})
#: The configurations whose mechanisms are intercept-only.
INTERCEPT_G = frozenset({"both_wrong", "full_history", "markov_history"})


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    """``(outcome, pseudo, treatment, censoring)`` learners of a configuration."""
    if configuration == "crossfit_boosted":
        q, pseudo = booster("classifier"), booster("regressor")
    elif configuration == "continuous_gaussian":
        q, pseudo = LinearRegression(), LinearRegression()
    elif configuration in BASELINE_Q:
        q, pseudo = CellMeansOn("baseline", CellMeans()), CellMeansOn("baseline", CellMeans())
    elif configuration in {"outcome_correct", "full_history"}:
        q, pseudo = CellMeansBackoff(), CellMeansBackoff()
    elif configuration == "markov_history":
        q = CellMeansOn("markov", CellMeansBackoff())
        pseudo = CellMeansOn("markov", CellMeansBackoff())
    else:
        q, pseudo = QuasiBinomialGLM(), QuasiBinomialGLM()
    if configuration in INTERCEPT_G:
        g, c = DummyClassifier(strategy="prior"), DummyClassifier(strategy="prior")
    elif configuration == "outcome_correct":
        g, c = BaselineOnly(QuasiBinomialGLM()), BaselineOnly(QuasiBinomialGLM())
    else:
        g, c = QuasiBinomialGLM(), QuasiBinomialGLM()
    return q, pseudo, g, c


def regimens(*, reversed_rule: bool = False) -> dict[str, Any]:
    """The three plans, and the reversed rule for the C7 control."""
    plans: dict[str, Any] = {
        "never": 0,
        "always": 1,
        RULE_LABEL: DynamicRegimen(
            RULE_LABEL, (1, treat_after("Y1"), treat_after("Y2")), rule_kind="known"
        ),
    }
    if reversed_rule:
        plans[REVERSED] = DynamicRegimen(
            REVERSED, (1, treat_zero("Y1"), treat_zero("Y2")), rule_kind="known"
        )
    return plans


def treat_zero(previous: str) -> Callable[[Any], np.ndarray]:
    """The reversed rule's node: treat exactly when the outcome ``previous`` was zero."""

    def node(history: Any) -> np.ndarray:
        return (np.asarray(history[previous], dtype=float) <= 0.0).astype(int)

    return node


def fit(
    frame: pd.DataFrame,
    configuration: str = "estimated_mechanisms",
    *,
    simultaneous: bool = False,
    reversed_rule: bool = False,
) -> Any:
    """The repeated-outcome fit of one configuration."""
    q, pseudo, g, c = _learners(configuration)
    folds = FOLDS if configuration in {"crossfit_glm", "crossfit_boosted"} else 1
    return LTMLE(
        regimens(reversed_rule=reversed_rule),
        reference="never",
        outcome_learner=q,
        pseudo_learner=pseudo,
        treatment_learner=g,
        censoring_learner=c,
        n_folds=folds,
        g_bounds=G_BOUNDS,
        q_bounds=(0.0, 1.0) if configuration == "continuous_gaussian" else None,
        simultaneous=simultaneous,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **COLUMNS)


def absorbing_fit(frame: pd.DataFrame, horizon: int) -> Any:
    r"""The N1 control: an end-of-study fit of :math:`\max_{s \le h} Y_s` at ``horizon``.

    The earlier outcomes stay in the history, so the mechanisms and the rule read what the
    repeated fit reads.  Only the outcome is the absorbing reading.

    The outcome regressions are saturated cell means rather than the positive arm's GLM.
    ``Y_s = 1`` implies ``M = 1`` for every later regression, so a logistic fit separates
    on every draw by construction.  Every node is binary, so cell means are the exact
    regressions of the absorbing reading, and the fit converges to
    :data:`ABSORBING_LIMIT` through the outcome model as well as through the mechanisms.
    """
    panel = frame.copy()
    outcomes = panel[[f"Y{s}" for s in range(1, horizon + 1)]]
    panel["M"] = outcomes.max(axis=1, skipna=False)
    _, _, g, c = _learners("estimated_mechanisms")
    plans = {
        "never": 0,
        "always": 1,
        RULE_LABEL: DynamicRegimen(
            RULE_LABEL,
            (1, *(treat_after(f"Y{s}") for s in range(1, horizon))),
            rule_kind="known",
        ),
    }
    blocks: list[list[str]] = [[], *([f"Y{s - 1}", f"L{s}"] for s in range(2, horizon + 1))]
    return LTMLE(
        plans,
        reference="never",
        outcome_learner=CellMeansBackoff(),
        pseudo_learner=CellMeansBackoff(),
        treatment_learner=g,
        censoring_learner=c,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        panel,
        outcome="M",
        treatment=[f"A{s}" for s in range(1, horizon + 1)],
        baseline=["W"],
        time_varying=blocks,
        censoring=[f"C{s}" for s in range(1, horizon + 1)],
    )


def history_free_fit(frame: pd.DataFrame, horizon: int = 3) -> Any:
    """The N2 control: an end-of-study fit of ``Y_h`` with no ``Y_s`` in any history.

    The earlier outcomes are confounders of the later nodes, and this fit leaves them out of
    the outcome regression and both mechanisms.  Saturated cell-mean outcome regressions on
    ``[W, L_2, ..., L_t]`` are correct for that coarser history, so the fit converges to its
    coarse g-formula whatever the mechanisms.
    """
    _, _, g, c = _learners("estimated_mechanisms")
    return LTMLE(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=g,
        censoring_learner=c,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=f"Y{horizon}",
        treatment=[f"A{s}" for s in range(1, horizon + 1)],
        baseline=["W"],
        time_varying=[[], *([f"L{s}"] for s in range(2, horizon + 1))],
        censoring=[f"C{s}" for s in range(1, horizon + 1)],
    )


def _end_of_study_name(label: str) -> str:
    """The name an end-of-study fit reports for a label: the horizon leaves the bracket."""
    return label_estimand(label).rsplit(" @ t=", 1)[0] + "]"


# ---------------------------------------------------------------------------- plug-ins


def label_parts(label: str) -> tuple[str, str | None, int]:
    """``(plan, reference, horizon)`` of a property label, in the law module's plan names."""
    stem, horizon = label.rsplit("_t", 1)
    if "_vs_" in stem:
        left, right = stem.split("_vs_")
        return left, right, int(horizon)
    return stem, None, int(horizon)


def plug_in(
    frame: pd.DataFrame,
    plan: str,
    horizon: int,
    *,
    history: str = "baseline",
    learner: Any = None,
) -> float:
    r"""The sequential-regression plug-in of one plan, with **no** fluctuation.

    Backward from ``horizon``: regress the carried outcome among the units that followed
    the plan and stayed under observation through ``t``, on the history ``history`` names at
    ``t``, predict for every row, and carry the prediction back.  The estimate averages the
    first prediction.  With the default ``history="baseline"`` and no learner each regression
    is the cell mean given ``W``, written here with ``numpy`` and ``pandas`` alone.  Given a
    ``learner``, that learner is fitted on the history's columns.

    Written longhand rather than by calling the estimator with the fluctuation disabled, for
    the reason ``tests.studies.ltmle_properties.untargeted`` gives: a flag on the estimator
    would make the control a statement about a branch of the code it audits.
    """
    filled = frame.fillna(0.0)
    baseline = filled["W"].to_numpy()
    carried = frame[f"Y{horizon}"].to_numpy(dtype=float)
    for time_point in range(horizon, 0, -1):
        followers = np.ones(len(frame), dtype=bool)
        for node in range(1, time_point + 1):
            earlier = filled[f"Y{node - 1}"].to_numpy() if node > 1 else np.zeros(len(frame))
            followers &= (filled[f"A{node}"].to_numpy() == law.arm(plan, node, earlier)) & (
                filled[f"C{node}"].to_numpy() == 1.0
            )
        if learner is None:
            means = pd.Series(carried[followers]).groupby(baseline[followers]).mean()
            if not set(baseline) <= set(means.index):
                raise ValueError(
                    f"a baseline cell has no follower of {plan!r} at node {time_point}"
                )
            carried = pd.Series(baseline).map(means).to_numpy()
        else:
            design = filled[law.history_columns(time_point, history)].to_numpy()
            model = clone(learner).fit(design[followers], carried[followers])
            carried = np.asarray(model.predict(design), dtype=float)
    return float(np.mean(carried))


def untargeted(frame: pd.DataFrame, label: str) -> float:
    """The ``targeting_necessity`` control: the plug-in of a label, or of its contrast."""
    plan, reference, horizon = label_parts(label)
    value = plug_in(frame, plan, horizon)
    if reference is not None:
        value -= plug_in(frame, reference, horizon)
    return value


# ---------------------------------------------------------------------------- rows


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    requested: int,
    truth: float,
    estimate: Any,
    *,
    n: int | None = None,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=family_n(family) if n is None else n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def sample(configuration: str, seed: int, n: int = N) -> pd.DataFrame:
    """One draw of size ``n`` of the law a configuration reads."""
    if configuration == "continuous_gaussian":
        return law.CONTINUOUS.sample(n, seed)
    frame, _ = make_longitudinal_repeated(n=n, seed=seed, censoring=True, backend="pandas")
    return frame


def family_n(family: str) -> int:
    """The sample size a property family draws."""
    if family == "interval_calibration":
        return CALIBRATION_N
    if family == "repeated_outcome_necessity":
        return ABSORBING_N
    if family == "full_history_necessity":
        return FULL_HISTORY_N
    return N


def _paired_rows(
    family: str,
    labels: Sequence[str],
    arms: tuple[str, str],
    replicate: int,
    requested: int,
    positive: Any,
    control: Any,
    *,
    positive_truth: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """The positive and control rows of a paired family, both read at the true value."""
    rows: list[dict[str, Any]] = []
    for label in labels:
        name = label_estimand(label)
        truth = TRUTH[name] if positive_truth is None else positive_truth[label]
        rows.append(
            _row(
                family,
                f"{label}__{arms[0]}",
                "positive",
                replicate,
                requested,
                truth,
                positive[name],
            )
        )
        rows.append(
            _row(
                family, f"{label}__{arms[1]}", "control", replicate, requested, truth, control[name]
            )
        )
    return rows


_Payload = tuple[str, str, int, int, int]


def _draw(payload: _Payload) -> pd.DataFrame:
    """One replication's sample, drawn outside the fit."""
    family, configuration, _, _, seed = payload
    return sample(configuration, seed, family_n(family))


def _replication(payload: _Payload) -> list[dict[str, Any]]:
    """One replication's rows, drawn and fitted in one call: the pilots' and the timer's form."""
    return _fit_drawn(payload, _draw(payload))


def _fit_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One replication's rows from its drawn sample."""
    family, configuration, replicate, requested, _ = payload
    n = family_n(family)
    rows: list[dict[str, Any]] = []
    if family in {"interval_calibration", "small_sample_calibration"}:
        calibration = family == "interval_calibration"
        joint = calibration and configuration == "estimated_mechanisms"
        result = fit(frame, configuration, simultaneous=joint)
        truths = CONTINUOUS_TRUTH if configuration == "continuous_gaussian" else None
        for label in GATED_LABELS:
            name = label_estimand(label)
            truth = TRUTH[name] if truths is None else truths[label]
            rows.append(
                _row(
                    family,
                    f"{label}__{configuration}",
                    "positive" if calibration else DIAGNOSTIC_ROLE,
                    replicate,
                    requested,
                    truth,
                    result[name],
                    n=n,
                )
            )
        if joint:
            for label in EARLIER_LABELS:
                name = label_estimand(label)
                rows.append(
                    _row(
                        family,
                        f"{label}__{configuration}",
                        "positive",
                        replicate,
                        requested,
                        TRUTH[name],
                        result[name],
                        n=n,
                    )
                )
            rows.extend(
                joint_coverage_rows(
                    result,
                    {name: TRUTH[name] for name in result.estimates},
                    tuple(result.estimates),
                    label=JOINT_LABEL,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    if family == "double_robustness":
        result = fit(frame, configuration)
        role = "control" if configuration == "both_wrong" else "positive"
        for label in GATED_LABELS:
            name = label_estimand(label)
            rows.append(
                _row(
                    family,
                    f"{label}__{configuration}",
                    role,
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
        return rows
    if family == "targeting_necessity":
        result = fit(frame, configuration)
        for label in GATED_LABELS:
            name = label_estimand(label)
            rows.append(
                _row(
                    family,
                    f"{label}__targeted",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            # The plug-in has no influence curve of its own, so the row carries the targeted
            # fit's standard error and the family gates on bias alone.
            rows.append(
                control_row(
                    property_name=family,
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=TRUTH[name],
                    estimate=untargeted(frame, label),
                    standard_error=float(result[name].std_error),
                    critical=CRITICAL,
                )
            )
        return rows
    if family == "rule_necessity":
        result = fit(frame, reversed_rule=True)
        for label in RULE_LABELS:
            name = label_estimand(label)
            reversed_name = name.replace(RULE_LABEL, REVERSED)
            rows.append(
                _row(
                    family,
                    f"{label}__declared",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            rows.append(
                _row(
                    family,
                    f"{label}__reversed_rule",
                    "control",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[reversed_name],
                )
            )
        return rows
    if family == "repeated_outcome_necessity":
        result = fit(frame)
        controls = {horizon: absorbing_fit(frame, horizon) for horizon in (2, 3)}
        for label in ABSORBING_LABELS:
            name = label_estimand(label)
            horizon = int(label.rsplit("_t", 1)[1])
            rows.append(
                _row(
                    family,
                    f"{label}__repeated",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            rows.append(
                _row(
                    family,
                    f"{label}__absorbing_reading",
                    "control",
                    replicate,
                    requested,
                    TRUTH[name],
                    controls[horizon][_end_of_study_name(label)],
                )
            )
        return rows
    if family == "outcome_history_necessity":
        result = fit(frame)
        control = history_free_fit(frame)
        for label in HISTORY_LABELS:
            name = label_estimand(label)
            rows.append(
                _row(
                    family,
                    f"{label}__repeated",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            rows.append(
                _row(
                    family,
                    f"{label}__history_free",
                    "control",
                    replicate,
                    requested,
                    TRUTH[name],
                    control[_end_of_study_name(label)],
                )
            )
        return rows
    if family == "full_history_necessity":
        return _paired_rows(
            family,
            GATED_LABELS,
            ("full_history", "markov_history"),
            replicate,
            requested,
            fit(frame, "full_history"),
            fit(frame, "markov_history"),
        )
    raise KeyError(family)


#: Every fit set: ``(family, configuration, replications)``.
FIT_SETS: tuple[tuple[str, str, int], ...] = (
    *(("interval_calibration", arm, CALIBRATION_REPLICATES) for arm in CALIBRATION_ARMS),
    *(("small_sample_calibration", arm, SMALL_SAMPLE_REPLICATES) for arm in SMALL_SAMPLE_ARMS),
    ("double_robustness", "mechanism_correct", DOUBLE_ROBUST_REPLICATES),
    ("double_robustness", "outcome_correct", DOUBLE_ROBUST_REPLICATES),
    ("double_robustness", "both_wrong", DOUBLE_ROBUST_REPLICATES),
    ("targeting_necessity", "targeted", TARGETING_REPLICATES),
    ("rule_necessity", "paired", RULE_REPLICATES),
    ("repeated_outcome_necessity", "paired", ABSORBING_REPLICATES),
    ("outcome_history_necessity", "paired", HISTORY_FREE_REPLICATES),
    ("full_history_necessity", "paired", FULL_HISTORY_REPLICATES),
)


def _payloads(
    fit_sets: tuple[tuple[str, str, int], ...] = FIT_SETS,
) -> list[tuple[tuple[str, str, int, int, int]]]:
    payloads: list[tuple[tuple[str, str, int, int, int]]] = []
    for family, configuration, replicates in fit_sets:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", family, configuration, replicate)
            payloads.append(((family, configuration, replicate, replicates, seed),))
    return payloads


#: The calibration positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived(GATED_LABELS, positive_suffix="estimated_mechanisms")

#: Each family's cell suffixes and labels for one fit, beside the calibration families, whose
#: suffix is the configuration.
_PAIRED_CELLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "targeting_necessity": (GATED_LABELS, ("targeted", "untargeted")),
    "rule_necessity": (RULE_LABELS, ("declared", "reversed_rule")),
    "repeated_outcome_necessity": (ABSORBING_LABELS, ("repeated", "absorbing_reading")),
    "outcome_history_necessity": (HISTORY_LABELS, ("repeated", "history_free")),
    "full_history_necessity": (GATED_LABELS, ("full_history", "markov_history")),
}


def _cells_of(key: tuple[_Payload]) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    family, configuration = key[0][:2]
    if family in _PAIRED_CELLS:
        labels, arms = _PAIRED_CELLS[family]
        return [f"{family}/{label}__{arm}" for label in labels for arm in arms]
    names = [f"{family}/{label}__{configuration}" for label in GATED_LABELS]
    if family == "interval_calibration" and configuration == "estimated_mechanisms":
        names += [f"{family}/{label}__{configuration}" for label in EARLIER_LABELS]
        names += [f"{JOINT_FAMILY}/{name}" for name in joint_cells(JOINT_LABEL)]
    for name in list(names):
        names += DERIVED_CELLS.get(name, ())
    return names


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6).  Up to ``floor(0.01 R)`` lost
#: replicates are judged by the worst case of each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit every fit set and derive the calibration controls from the same draws."""
    outcomes = fit_properties(
        _payloads(),
        splat_drawn(_fit_drawn),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=GATED_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
                positive_suffix="estimated_mechanisms",
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts


#: The calibration families reported beside their SE-ratio interval and gated by nothing.
REPORTED_CALIBRATION = ("small_sample_calibration",)


def _calibration_report(summary: pd.DataFrame, rows: pd.DataFrame, name: str) -> None:
    """Publish a reported family's SE-ratio interval beside its coverage, gated by nothing."""
    margins = STUDY.margins
    family = summary["property"] == name
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == name) & (rows["cell"] == cell)]
        interval = se_ratio_interval(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, name, cell),
        )
        summary.loc[index, "se_ratio_ci_lower"] = interval.low
        summary.loc[index, "se_ratio_ci_upper"] = interval.high
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.  The
#: plug-in control reads its exact limit, :data:`PLUG_IN_LIMIT`.
TARGETING_NECESSITY = Necessity(
    family="targeting_necessity",
    labels=GATED_LABELS,
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    alternative_truths=PLUG_IN_LIMIT,
)


def _necessity(
    family: str, labels: Sequence[str], arms: tuple[str, str], limits: Mapping[str, float]
) -> Necessity:
    return Necessity(
        family=family,
        labels=tuple(labels),
        arms=arms,
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
        alternative_truths=limits,
    )


#: Every necessity family's declaration, read by its verdict and its rule.  Each control reads
#: its exact limit.
NECESSITIES = (
    TARGETING_NECESSITY,
    _necessity("rule_necessity", RULE_LABELS, ("declared", "reversed_rule"), REVERSED_TRUTH),
    _necessity(
        "repeated_outcome_necessity",
        ABSORBING_LABELS,
        ("repeated", "absorbing_reading"),
        ABSORBING_LIMIT,
    ),
    _necessity(
        "outcome_history_necessity",
        HISTORY_LABELS,
        ("repeated", "history_free"),
        HISTORY_FREE_LIMIT,
    ),
    _necessity(
        "full_history_necessity", GATED_LABELS, ("full_history", "markov_history"), MARKOV_LIMIT
    ),
)

#: The ``both_wrong`` limit of each gated label, by control cell, which
#: ``alternative_target_verdicts`` reads.
BOTH_WRONG_TARGETS = {f"{label}__both_wrong": BOTH_WRONG_LIMIT[label] for label in GATED_LABELS}


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("necessity_displacement", "targeting_displacement"),
        rate_labels=(),
    )
    margins = STUDY.margins

    # The C1 arm and its two derived controls read the instrument's rule for each kind.  The
    # other positive arms keep the two-sided rule ``apply_shared_verdicts`` wrote, which is
    # the same rule.
    primary = (summary["property"] == "interval_calibration") & ~summary["cell"].str.endswith(
        tuple(f"__{arm}" for arm in CALIBRATION_ARMS[1:])
    )
    subset = summary.loc[primary].copy()
    calibration_verdicts(subset, margins=margins, positive_suffix="estimated_mechanisms")
    summary.loc[subset.index, "passed"] = subset["passed"]

    for name in REPORTED_CALIBRATION:
        _calibration_report(summary, rows, name)
    simultaneous_coverage_verdicts(summary, margins=margins)

    # ``both_wrong`` reads its exact limit: the fit differs from the truth, and the limit says
    # by how much.  ``apply_shared_verdicts`` wrote the family's own verdicts.
    alternative_target_verdicts(
        summary,
        rows,
        STUDY,
        family="double_robustness",
        alternative_truths=BOTH_WRONG_TARGETS,
    )
    for necessity in NECESSITIES:
        alternative_target_necessity_verdicts(summary, rows, STUDY, necessity)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  ``double_robustness`` adds
#: the ``both_wrong`` limit to the union-model rule, and each necessity family reads its
#: declaration.  The calibration, simultaneous-coverage and union-model clauses are the shared
#: rules'.
PROPERTY_RULES = (
    property_verdicts.robustness_rule(alternative_truths=BOTH_WRONG_TARGETS),
    *property_verdicts.necessity_rules(*NECESSITIES),
)


# ---------------------------------------------------------------------------- design


# The sizing rules live in :mod:`tests.studies.repeated_sizing`.  They take the study's margins
# and its label-correlation function.  The functions below bind them to X32a, with its declared
# 0.95 target for each sized instrument.


def bias_equivalent_probability(bias: float, replicates: int) -> float:
    """P(an arm's 99% bias interval lies inside the margin), at standardized ``bias``.

    The margin is :attr:`~tests.studies.evidence.registry.Margins.standardized_bias` spreads of
    the arm, and the interval is the Student interval of the mean error, which is normal at
    these sizes.
    """
    return float(sizing.bias_equivalent_probability(STUDY.margins, bias, replicates))


def bias_discriminated_probability(gap: float, replicates: int) -> float:
    """P(a control's 99% bias interval clears the margin), at standardized ``gap``."""
    return sizing.bias_discriminated_probability(STUDY.margins, gap, replicates)


def coverage_floor_pass_probability(replicates: int, coverage: float) -> float:
    """The probability that the exact coverage interval's lower bound clears the floor.

    The gate is :attr:`~tests.studies.evidence.registry.Margins.coverage_floor` on the lower
    end of the two-sided Clopper-Pearson interval at the study's confidence level.
    """
    return sizing.coverage_floor_pass_probability(STUDY.margins, replicates, coverage)


def _accepted_counts(replicates: int) -> tuple[int, int] | None:
    """The covered counts whose exact interval lies inside the calibration coverage band."""
    return sizing.accepted_counts(STUDY.margins, replicates)


def calibration_pass_probability(
    replicates: int, *, ratio: float = 1.0, coverage: float | None = None
) -> float:
    """The probability that a calibration cell passes the two-sided rule.

    :func:`~tests.studies.repeated_sizing.calibration_pass` defines the two halves.
    ``coverage`` defaults to the coverage a Wald interval with this SE ratio attains,
    :math:`2\\Phi(1.96\\,r) - 1`, because a cell with a ratio off one is off in both halves.
    """
    return float(_calibration_pass(replicates, np.asarray(ratio, dtype=float), coverage))


def _calibration_pass(
    replicates: int, ratio: np.ndarray, coverage: float | None = None
) -> np.ndarray:
    """:func:`calibration_pass_probability` over an array of SE ratios."""
    return sizing.calibration_pass(STUDY.margins, replicates, ratio, coverage)


def primary_replicates(
    covered: Mapping[str, int], pilot_replicates: int, *, implementations: int = 2
) -> int:
    """The declared rule for ``PRIMARY_REPLICATES``.

    The smallest multiple of :data:`BUDGET_STEP` such that

    * a cell whose true standardized bias is the paired-difference margin
      (:data:`PRIMARY_DESIGN_BIAS`) passes the bias gate with probability at least 0.90;
    * the coverage-floor gate passes with probability at least :data:`PRIMARY_CELL_PASS` per
      cell at coverage :data:`PRIMARY_DESIGN_COVERAGE`; and
    * the product of the coverage-floor pass probabilities over every cell of every
      implementation is at least :data:`FAMILY_WISE_PASS`, with each cell at the one-sided
      80% lower Clopper-Pearson bound of its pilot coverage.  The last clause prices the
      pilot's own sampling error, so a pilot that happens to read high cannot under-size the
      run.  The paired cells are not binding, because the two implementations agree.

    Parameters
    ----------
    covered : Mapping[str, int]
        The covered count of each primary cell in the pilot.
    pilot_replicates : int
        The number of pilot replications.
    implementations : int
        How many implementations each cell is tested for (cleverly and R).
    """
    return sizing.primary_replicates(
        STUDY.margins, covered, pilot_replicates, implementations=implementations
    )


#: Pilot P1 (see ``tests.studies.canonical_ltmle_repeated``): the covered count of each
#: primary cell over :data:`PRIMARY_PILOT_REPLICATES` draws at ``n = 3,000`` from the pilot
#: stream.  :func:`primary_replicates` reads it, and a design test re-derives the budget.
PRIMARY_PILOT_REPLICATES = 2_000
PRIMARY_PILOT_COVERED = {
    "ey_regimen[never @ t=1]": 1914,
    "ey_regimen[never @ t=2]": 1900,
    "ey_regimen[never @ t=3]": 1893,
    "ey_regimen[always @ t=1]": 1881,
    "ey_regimen[always @ t=2]": 1888,
    "ey_regimen[always @ t=3]": 1879,
    "ey_regimen[treat then treat after an outcome @ t=2]": 1901,
    "ey_regimen[treat then treat after an outcome @ t=3]": 1870,
    "ate_regimen[always vs never @ t=1]": 1895,
    "ate_regimen[always vs never @ t=2]": 1904,
    "ate_regimen[always vs never @ t=3]": 1892,
    "ate_regimen[treat then treat after an outcome vs never @ t=2]": 1901,
    "ate_regimen[treat then treat after an outcome vs never @ t=3]": 1901,
}


# ---------------------------------------------------------------------------- pooled reading
#
# Both sizing rules below read a pilot as a set of noisy cells, not as one independent
# measurement per cell.  Cells that share a fit share noise, and cells that share a law differ
# by less than their own pilot noise.  A rule that moves every cell to the far end of its own
# interval prices the pilot noise once for every cell, and a family-wise product then prices it
# again.  The pooled reading prices the noise of a group once, and the spread between its cells
# separately.


def label_correlations(labels: Sequence[str]) -> np.ndarray:
    """The correlation, under the binary law, of the efficient influence curves of labels.

    Two estimates of the labels, from one sample, have this correlation to first order.  A
    label repeated in ``labels`` has correlation one with itself.
    """
    curves = []
    for label in labels:
        plan, reference, horizon = label_parts(label)
        curves.append(law.BINARY.eif(plan, horizon, reference=reference))
    stacked = np.array(curves)
    covariance = (stacked * law.BINARY.paths["p"].to_numpy()) @ stacked.T
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def effective_cells(labels: Sequence[str]) -> float:
    r"""How many independent cells ``labels`` are worth when their estimates are averaged.

    :math:`k_{\mathrm{eff}} = k^2 / \sum_{ij} \rho_{ij}^2` for the correlations
    :func:`label_correlations` returns.  The sampling noise of an SE ratio, the ratio of the
    mean reported SE to the empirical SD, has correlation :math:`\rho_{ij}^2` between two
    cells, so the mean of ``k`` ratios has the noise variance of one ratio divided by this
    number.
    """
    return sizing.effective_cells(labels, label_correlations)


#: Pilot P2 (see the module docstring): the SE ratio of each of the 28 gated calibration
#: cells over :data:`CALIBRATION_PILOT_REPLICATES` draws at :data:`CALIBRATION_N`, from the
#: pilot stream.  :func:`calibration_replicates` reads it, and a design test re-derives the
#: budget.  Four decimals are recorded, and the rule returns the same budget from them.
CALIBRATION_PILOT_REPLICATES = 2_000
CALIBRATION_PILOT_RATIOS = {
    "never_t3__estimated_mechanisms": 1.0003,
    "always_t3__estimated_mechanisms": 0.9967,
    "rule_t3__estimated_mechanisms": 0.9795,
    "always_vs_never_t3__estimated_mechanisms": 1.0128,
    "rule_vs_never_t3__estimated_mechanisms": 0.9883,
    "never_t1__estimated_mechanisms": 1.0183,
    "never_t2__estimated_mechanisms": 1.0246,
    "always_t1__estimated_mechanisms": 1.0225,
    "always_t2__estimated_mechanisms": 1.0061,
    "rule_t2__estimated_mechanisms": 1.0278,
    "always_vs_never_t1__estimated_mechanisms": 1.0159,
    "always_vs_never_t2__estimated_mechanisms": 1.0033,
    "rule_vs_never_t2__estimated_mechanisms": 1.0182,
    "never_t3__crossfit_glm": 1.0039,
    "always_t3__crossfit_glm": 1.0136,
    "rule_t3__crossfit_glm": 0.9837,
    "always_vs_never_t3__crossfit_glm": 1.0093,
    "rule_vs_never_t3__crossfit_glm": 0.9740,
    "never_t3__crossfit_boosted": 1.0073,
    "always_t3__crossfit_boosted": 1.0103,
    "rule_t3__crossfit_boosted": 1.0279,
    "always_vs_never_t3__crossfit_boosted": 1.0108,
    "rule_vs_never_t3__crossfit_boosted": 1.0245,
    "never_t3__continuous_gaussian": 0.9959,
    "always_t3__continuous_gaussian": 0.9935,
    "rule_t3__continuous_gaussian": 1.0055,
    "always_vs_never_t3__continuous_gaussian": 0.9920,
    "rule_vs_never_t3__continuous_gaussian": 0.9963,
}


def calibration_groups(ratios: Mapping[str, float], pilot_replicates: int) -> list[PilotGroup]:
    """One pilot group for each calibration arm.

    The pilot noise of a ratio has relative standard deviation ``1 / sqrt(2 (R - 1))``, and
    the noise of two cells of one arm has the correlation :func:`label_correlations` squares.

    Parameters
    ----------
    ratios : Mapping[str, float]
        The pilot SE ratio of each gated cell, by ``label__arm`` name.
    pilot_replicates : int
        The pilot replications behind each ratio.
    """
    return sizing.calibration_groups(ratios, pilot_replicates, label_correlations)


def calibration_family_pass_probability(
    replicates: int, groups: Sequence[PilotGroup], tau: float
) -> float:
    """The probability that every gated calibration cell passes at ``replicates``.

    Each arm reads :func:`pooled_pass_probability` with :func:`calibration_pass_probability`
    as the cell probability.
    """
    return sizing.calibration_family_pass_probability(STUDY.margins, replicates, groups, tau)


def calibration_replicates(ratios: Mapping[str, float], pilot_replicates: int) -> int:
    """The declared rule for ``CALIBRATION_REPLICATES``.

    The smallest multiple of :data:`BUDGET_STEP` at which
    :func:`calibration_family_pass_probability` is at least :data:`FAMILY_WISE_PASS`.  The
    true SE ratio of each arm is the arm's pilot mean, with the noise variance of that mean,
    and each cell lies :func:`heterogeneity_bound` away from it.  The earlier rule moved every
    cell to the far end of its own 80% interval at once.  It priced the noise of the pilot
    once for each of 28 cells, and again in the product over cells, and it fitted no model of
    how cells relate.

    Parameters
    ----------
    ratios : Mapping[str, float]
        The pilot SE ratio of each gated cell, by ``label__arm`` name.
    pilot_replicates : int
        The pilot replications behind each ratio.
    """
    return sizing.calibration_replicates(
        STUDY.margins, ratios, pilot_replicates, label_correlations
    )


#: Pilot P4 (see the module docstring): the standardized bias of each cell that a bias check
#: reads, over :data:`PROPERTY_PILOT_REPLICATES` draws at the cell's own size, from the pilot
#: stream.  The key is ``family/cell``.  A positive cell reads its bias around the truth, and a
#: control cell reads its bias around its own limit.  The unit is the empirical SD of the
#: cell.  :func:`property_bias_pass_probability` reads it, and a design test re-derives the
#: pass probability.
PROPERTY_PILOT_REPLICATES = 300
PROPERTY_PILOT_BIASES = {
    "double_robustness/always_t3__both_wrong": -0.079,
    "double_robustness/always_t3__mechanism_correct": 0.030,
    "double_robustness/always_t3__outcome_correct": -0.045,
    "double_robustness/always_vs_never_t3__both_wrong": -0.043,
    "double_robustness/always_vs_never_t3__mechanism_correct": 0.012,
    "double_robustness/always_vs_never_t3__outcome_correct": -0.023,
    "double_robustness/never_t3__both_wrong": -0.016,
    "double_robustness/never_t3__mechanism_correct": 0.008,
    "double_robustness/never_t3__outcome_correct": -0.006,
    "double_robustness/rule_t3__both_wrong": 0.021,
    "double_robustness/rule_t3__mechanism_correct": 0.072,
    "double_robustness/rule_t3__outcome_correct": 0.060,
    "double_robustness/rule_vs_never_t3__both_wrong": 0.025,
    "double_robustness/rule_vs_never_t3__mechanism_correct": 0.045,
    "double_robustness/rule_vs_never_t3__outcome_correct": 0.048,
    "full_history_necessity/always_t3__full_history": 0.021,
    "full_history_necessity/always_t3__markov_history": -0.011,
    "full_history_necessity/always_vs_never_t3__full_history": 0.014,
    "full_history_necessity/always_vs_never_t3__markov_history": -0.057,
    "full_history_necessity/never_t3__full_history": -0.001,
    "full_history_necessity/never_t3__markov_history": 0.065,
    "full_history_necessity/rule_t3__full_history": 0.025,
    "full_history_necessity/rule_t3__markov_history": 0.017,
    "full_history_necessity/rule_vs_never_t3__full_history": 0.019,
    "full_history_necessity/rule_vs_never_t3__markov_history": -0.042,
    "outcome_history_necessity/always_t3__history_free": -0.024,
    "outcome_history_necessity/always_t3__repeated": -0.003,
    "outcome_history_necessity/always_vs_never_t3__history_free": -0.035,
    "outcome_history_necessity/always_vs_never_t3__repeated": -0.041,
    "outcome_history_necessity/never_t3__history_free": 0.024,
    "outcome_history_necessity/never_t3__repeated": 0.050,
    "repeated_outcome_necessity/always_t2__absorbing_reading": 0.079,
    "repeated_outcome_necessity/always_t2__repeated": 0.064,
    "repeated_outcome_necessity/always_t3__absorbing_reading": 0.020,
    "repeated_outcome_necessity/always_t3__repeated": -0.029,
    "repeated_outcome_necessity/never_t2__absorbing_reading": 0.012,
    "repeated_outcome_necessity/never_t2__repeated": 0.080,
    "repeated_outcome_necessity/never_t3__absorbing_reading": 0.062,
    "repeated_outcome_necessity/never_t3__repeated": 0.134,
    "repeated_outcome_necessity/rule_t2__absorbing_reading": 0.108,
    "repeated_outcome_necessity/rule_t2__repeated": 0.093,
    "repeated_outcome_necessity/rule_t3__absorbing_reading": 0.092,
    "repeated_outcome_necessity/rule_t3__repeated": -0.023,
    "rule_necessity/rule_t2__declared": -0.105,
    "rule_necessity/rule_t2__reversed_rule": -0.021,
    "rule_necessity/rule_t3__declared": -0.040,
    "rule_necessity/rule_t3__reversed_rule": 0.032,
    "targeting_necessity/always_t3__targeted": -0.016,
    "targeting_necessity/always_t3__untargeted": -0.022,
    "targeting_necessity/always_vs_never_t3__targeted": 0.084,
    "targeting_necessity/always_vs_never_t3__untargeted": 0.046,
    "targeting_necessity/never_t3__targeted": -0.115,
    "targeting_necessity/never_t3__untargeted": -0.084,
    "targeting_necessity/rule_t3__targeted": 0.080,
    "targeting_necessity/rule_t3__untargeted": 0.051,
    "targeting_necessity/rule_vs_never_t3__targeted": 0.137,
    "targeting_necessity/rule_vs_never_t3__untargeted": 0.097,
}


def property_bias_groups(
    biases: Mapping[str, float], pilot_replicates: int
) -> dict[str, PilotGroup]:
    """One pilot group for each family: the standardized biases of its cells.

    The declared reading.  A family shares a finite-sample bias, so its cells share a mean that
    differs from zero and from the means of other families.  The pilot noise of a standardized
    bias has variance ``1 / R``.  Cells of one family share each draw, so their noise has the
    correlation of their labels' efficient influence curves.  A control cell and a positive
    cell of one label count as the same label, which is the most cautious reading.

    Parameters
    ----------
    biases : Mapping[str, float]
        The pilot bias of each cell, by ``family/cell`` name.
    pilot_replicates : int
        The pilot replications behind each bias.
    """
    return sizing.property_bias_groups(biases, pilot_replicates, label_correlations)


def property_bias_pool(biases: Mapping[str, float], pilot_replicates: int) -> PilotGroup:
    """The cross-check reading: every cell of every family in one group.

    A bias of a consistent estimator is zero up to a finite-sample term, so this reading gives
    all families one mean.  The spread between cells then includes the spread between
    families.  Cells of two families share no draw, so the noise is block diagonal.
    """
    return sizing.property_bias_pool(biases, pilot_replicates, label_correlations)


def property_bias_pass_probability(
    biases: Mapping[str, float],
    pilot_replicates: int,
    replicates: Mapping[str, int],
    tau: float | None = None,
    *,
    pooled: bool = False,
) -> float:
    """The probability that every bias check passes.

    The cell probability is :func:`bias_equivalent_probability` at the replications of the
    cell's family.  :func:`pooled_pass_probability` carries each group's pilot mean, the noise
    of that mean, and the spread ``tau`` between cells.  The groups are the families
    (:func:`property_bias_groups`).  With ``pooled=True`` the group is every cell
    (:func:`property_bias_pool`), the reported cross-check.

    Parameters
    ----------
    biases : Mapping[str, float]
        The pilot bias of each cell, by ``family/cell`` name.
    pilot_replicates : int
        The pilot replications behind each bias.
    replicates : Mapping[str, int]
        The replications of each family.
    tau : float, optional
        The spread between the true biases of the cells.  The default is the
        :func:`heterogeneity_bound` of the pilot under the same grouping.
    pooled : bool
        Read one group for all cells.
    """
    return sizing.property_bias_pass_probability(
        STUDY.margins,
        biases,
        pilot_replicates,
        replicates,
        label_correlations,
        tau,
        pooled=pooled,
    )


def property_bias_replicates(biases: Mapping[str, float], pilot_replicates: int) -> dict[str, int]:
    """The declared rule for the replications of each property family.

    The families draw independent streams, so the probability that every bias check passes is
    the product over families.  Each family takes the smallest multiple of
    :data:`PROPERTY_STEP` at which its own pass probability, with :math:`\\tau` at its 80%
    bound, reaches :data:`FAMILY_WISE_PASS` to the power ``1 / F`` for ``F`` families.  The
    product is then at least :data:`FAMILY_WISE_PASS`.

    Parameters
    ----------
    biases : Mapping[str, float]
        The pilot bias of each cell, by ``family/cell`` name.
    pilot_replicates : int
        The pilot replications behind each bias.
    """
    return sizing.property_bias_replicates(
        STUDY.margins, biases, pilot_replicates, label_correlations
    )


# ---------------------------------------------------------------------------- pilots
#
# Every pilot draws from ``stream_seed(STUDY, "pilot", ...)``, which no published cell reads.


def _pilot_replication(payload: tuple[str, int, int]) -> dict[str, tuple[float, float]]:
    configuration, replicate, n = payload
    frame = sample(configuration, stream_seed(STUDY, "pilot", configuration, replicate), n)
    result = fit(frame, configuration)
    return {
        label: (
            float(result[label_estimand(label)].psi),
            float(result[label_estimand(label)].std_error),
        )
        for label in (*GATED_LABELS, *EARLIER_LABELS)
    }


def pilot_se_ratios(
    replicates: int = 200,
    *,
    configuration: str = "estimated_mechanisms",
    n: int = CALIBRATION_N,
    n_jobs: int = STUDY_JOBS,
    offset: int = 0,
) -> dict[str, float]:
    """PILOT HOOK (P2, P3): the SE ratio of each reported label on pilot draws.

    Not run by the fast tier.  The stream is ``stream_seed(STUDY, "pilot", configuration,
    replicate)`` for ``replicate`` from ``offset``.  Run one call per calibration arm at
    :data:`CALIBRATION_N` for P2, and the in-sample arm at :data:`N` for P3, and size
    :data:`CALIBRATION_REPLICATES` with :func:`calibration_replicates`.  Record the pilot and the
    new budget in the declaration commit.
    """
    payloads = [
        ((configuration, replicate, n),) for replicate in range(offset, offset + replicates)
    ]
    draws = map_parallel(_pilot_replication, payloads, n_jobs=n_jobs)
    ratios: dict[str, float] = {}
    for label in (*GATED_LABELS, *EARLIER_LABELS):
        estimates = np.array([draw[label][0] for draw in draws])
        errors = np.array([draw[label][1] for draw in draws])
        ratios[label] = float(np.mean(errors) / np.std(estimates, ddof=1))
    return ratios


def _primary_pilot_replication(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, n = payload
    frame, _ = make_longitudinal_repeated(
        n=n, seed=stream_seed(STUDY, "pilot", "primary", replicate), backend="pandas"
    )
    result = fit_cleverly(frame)
    means: dict[tuple[str, int], float] = {}
    for plan in ("never", "always", law.RULE):
        for horizon in (1, 2, 3):
            if plan == law.RULE and horizon == 1:
                continue
            means[plan, horizon] = plug_in(
                frame, plan, horizon, history="full", learner=QuasiBinomialGLM()
            )
    rows = []
    for name in ESTIMANDS:
        if name.startswith("ate_regimen["):
            comparison, horizon_text = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
            left, right = comparison.split(" vs ")
            horizon = int(horizon_text)
            initial = means[_short(left), horizon] - means[_short(right), horizon]
        else:
            plan_name, horizon_text = name[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
            initial = means[_short(plan_name), int(horizon_text)]
        estimate = result[name]
        low, high = estimate.ci
        rows.append(
            {
                "replicate": replicate,
                "estimand": name,
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": int(low <= TRUTH[name] <= high),
                "plug_in": float(initial),
            }
        )
    return rows


def _short(plan: str) -> str:
    return law.RULE if plan == RULE_LABEL else plan


def pilot_primary(
    replicates: int = 200, *, n: int = PRIMARY_N, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    """PILOT HOOK (P1): cleverly's primary fits on the ``("pilot", "primary")`` stream.

    One row per replication and primary cell, with the estimate, its standard error, whether
    its interval covers the truth, and the plug-in of a quasibinomial GLM chain on the full
    history.  That chain stands in for the ``gcomp`` call of R, which cannot run here.
    """
    payloads = [((replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_primary_pilot_replication, payloads, n_jobs=n_jobs)
    return pd.DataFrame([row for rows in outcomes for row in rows])


def pilot_cell_summary(
    pilot: pd.DataFrame, targets: Mapping[str, float], *, cell: str = "estimand"
) -> pd.DataFrame:
    """Per cell: bias in empirical spreads, SE ratio, coverage and the covered count.

    Parameters
    ----------
    pilot : pandas.DataFrame
        Rows with ``estimate``, ``std_error`` and ``covered`` columns, grouped by ``cell``.
    targets : Mapping[str, float]
        The value each cell's bias is read around: the truth, or a control's limit.
    cell : str
        The grouping column.
    """
    records = []
    for name, group in pilot.groupby(cell, sort=False):
        spread = float(group["estimate"].std(ddof=1))
        records.append(
            {
                cell: name,
                "replicates": len(group),
                "bias_sd": float((group["estimate"].mean() - targets[str(name)]) / spread),
                "sd": spread,
                "se_ratio": float(group["std_error"].mean() / spread),
                "coverage": float(group["covered"].mean()),
                "covered": int(group["covered"].sum()),
            }
        )
    return pd.DataFrame(records)


def pilot_property_rows(
    family: str,
    configuration: str,
    replicates: int,
    *,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """PILOT HOOK (P4): a property family's replication rows on the pilot stream.

    The same fit and rows as the declared run, from
    ``stream_seed(STUDY, "pilot", family, configuration, replicate)``.
    """
    payloads = [
        (
            (
                family,
                configuration,
                replicate,
                replicates,
                stream_seed(STUDY, "pilot", family, configuration, replicate),
            ),
        )
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replication, payloads, n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def time_fit_sets(seed: int = 0) -> dict[tuple[str, str], float]:
    """Seconds for one replication of each fit set, single process, for the budget table."""
    seconds: dict[tuple[str, str], float] = {}
    for family, configuration, _ in FIT_SETS:
        start = time.perf_counter()
        _replication((family, configuration, 0, 1, seed))
        seconds[family, configuration] = time.perf_counter() - start
    return seconds


def declared_truths() -> Mapping[str, Mapping[str, float]]:
    """Every truth and control limit a cell reads, by kind."""
    return {
        "binary": TRUTH,
        "continuous": CONTINUOUS_TRUTH,
        "absorbing": ABSORBING_LIMIT,
        "history_free": HISTORY_FREE_LIMIT,
        "both_wrong": BOTH_WRONG_LIMIT,
        "plug_in": PLUG_IN_LIMIT,
        "markov": MARKOV_LIMIT,
        "reversed": REVERSED_TRUTH,
    }
