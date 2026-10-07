r"""Sequential regression: the backward recursion a longitudinal fit is built on.

The longitudinal g-formula is an iterated conditional expectation.  With
:math:`H_t = (W, L_1, A_1, C_1, \ldots, L_t)` the history just before the treatment
decision at time :math:`t`, and :math:`\bar a` a regimen,

.. math::

    \bar Q_{T+1} &= Y \\
    \bar Q_t(H_t) &= E\bigl[\bar Q_{t+1} \bigm| H_t,\, A_t = a_t,\, C_t = 1\bigr],
        \qquad t = T, \ldots, 1 \\
    \Psi(P) &= E\bigl[\bar Q_1(H_1)\bigr]

so the whole parameter is :math:`T` ordinary regressions run backwards, each one's
prediction the next one's outcome (Bang & Robins 2005).  Each regression is fitted on the
units that followed :math:`\bar a` and stayed under observation through :math:`t`, and
predicts for the units that did so through :math:`t - 1` -- which are exactly the units
the *previous* step is fitted on, and is what makes the recursion close.

That is the untargeted substitution estimator.  It does not generally solve the efficient
influence-curve equation.  Targeting solves that equation.  At each step the initial regression
is fluctuated along

.. math::

    \operatorname{logit} \bar Q^*_t = \operatorname{logit} \bar Q_t + \epsilon_t,

by logistic loss weighted with

.. math::

    h_t = \frac{\mathbb 1\{\bar A_t = \bar a_t,\, \bar C_t = 1\}}
               {\prod_{s \le t} g_s(a_s \mid H_s)\, c_s(H_s, a_s)}

whose score is the :math:`t`-th term of the efficient influence function

.. math::

    D^*(O) = \sum_{t=1}^{T} h_t \bigl(\bar Q^*_{t+1} - \bar Q^*_t\bigr)
             + \bar Q^*_1(H_1) - \Psi.

Solving all :math:`T` of them makes the estimator solve :math:`P_n D^* = 0`, which is
what buys the asymptotic linearity the reported variance assumes.  This placement of
:math:`h_t` in the loss follows the canonical ``ltmle::UpdateQ`` algorithm.  Putting it
in the submodel gives the same score at zero but a different finite-sample substitution
path.  Note the recursion
carries the *targeted* prediction forward, not the initial one: the outcome of step
:math:`t` is :math:`\bar Q^*_{t+1}`, so a residual left by one step is regressed away by
the next rather than accumulating.

With outer cross-fitting the recursion splits in two, following the cross-fitted
construction of Díaz, Williams, Hoffman and Schenck (2023, *JASA* 118(542), Section 5.2,
Steps 1-4).  First,
fold :math:`k` runs an *untargeted* backward recursion on its training complement: each
node regresses fold :math:`k`'s own untargeted prediction from the node after it, so no
held-out row enters any regression that predicts it.  The held-out predictions of the
``K`` folds are stitched into one out-of-fold initial estimate per node.  Second, one
pooled fluctuation per node, from :math:`t = T` down to :math:`1`, targets those stitched
predictions over *every* follower.  Its offset is the stitched initial prediction, its
outcome is the pooled targeted prediction from :math:`t + 1`, and its loss weight is the
out-of-fold cumulative mechanism.  Each fold's initial nuisance fit is therefore fixed given its
training rows, which is the conditional-independence step in the proof of the paper's
Theorem 3.  The pooled coefficient still depends on the full sample, as the theorem permits,
and its fluctuation solves :math:`P_n D^* = 0` exactly as the single-fold fit does.

The clever covariate is the reciprocal of a **cumulative** product, and that is the whole
positivity story of a longitudinal fit: :math:`T` probabilities multiply, so a mechanism
that looks harmless node by node can leave a handful of units carrying most of the
weight.  :attr:`RegimenFit.max_weight` and the effective sample size it implies are
reported for that reason rather than as decoration.

**A survival outcome** puts an absorbing :math:`Y_t` at every node, and the parameter
becomes the cumulative risk at a horizon :math:`k`,
:math:`\Psi_k(P) = P(Y_k^{\bar a} = 1)`.  The recursion above generalises by seeding
:math:`\bar Q_{k+1} = 0` and composing the event indicator into the pseudo-outcome:

.. math::

    Z_t &= Y_t + (1 - Y_t)\, \bar Q^*_{t+1} \\
    \bar Q_t(H_t) &= E\bigl[Z_t \bigm| H_t,\, A_t = a_t,\, C_t = 1\bigr]

fitted on the units at risk entering :math:`t` -- event-free through :math:`t - 1`, which
is one node *earlier* than the censoring factor, because a unit that has the event at
:math:`t` is exactly the observation that it happened and belongs in that regression.  So
one backward pass answers one horizon, and a curve is :math:`k` of them; the mechanism is
fitted once and shared across all of them, which is where the cost would otherwise be.

Note what does **not** change.  ``1{event-free through t-1}`` is a function of
:math:`H_t` -- it is part of the history, not an intervened node -- so it enters the
*indicator* of :math:`h_t` and never its denominator.  The cumulative product is still
over the :math:`2T` treatment and censoring factors, and the positivity assumption a
survival fit makes is the one an end-of-study fit makes.

**A known policy** at a node replaces the assigned arm by a draw from a density
:math:`q_t(\cdot \mid H_t)` that is fixed before the fit (Díaz, Williams, Hoffman and
Schenck 2023, Section 2 and Theorem 3, with the randomizer integrated out).  Write
:math:`\pi_t(a \mid H_t)` for the node's intervention density: the indicator of the assigned
arm at a label or rule node, and :math:`q_t` at a policy node.  Three things change.  The
regression at a policy node is fitted once with the current arm as a column, and the node
carries the policy-weighted mean of its per-arm predictions,
:math:`\bar Q_t(H_t) = \sum_a q_t(a \mid H_t)\, Q_t(a, H_t)`.  The clever covariate becomes
the ratio :math:`\prod_{s \le t} \pi_s(A_s \mid H_s) / \prod_{s \le t} g_s c_s` at the
observed arms, whose denominator alone is bounded.  And a row stays in the plan while its
observed arm has positive intervention density, which generalises "followed".  The
fluctuation keeps its intercept, offset and loss-weight placement, and moves every arm's
prediction by the same coefficient, as ``lmtp``'s update does.  A plan without a policy node
runs none of this, and its arrays are the ones it had before policies existed.

**Observation weights** are a tilt of the population and not a further node.  Every
regression here -- each mechanism factor, the outcome, every pseudo-outcome -- is fitted by
weighted loss, each node's fluctuation solves the weighted score
:math:`\sum_i w_i h_t(i) (Z_t(i) - \bar Q^*_t(i)) = 0`, the plug-in is a weighted average,
and the reported curve is :math:`w_i D^*(O_i)` with :math:`w` normalised to mean one.  So
the parameter is the one :mod:`cleverly.data.weighting` states, evaluated on the tilted law
at every node at once.  What a weight is emphatically **not** is a factor in
:math:`h_t`: the clever covariate's denominator is the :math:`2T` mechanism factors and
nothing else, and putting :math:`w` there would divide the estimating equation by the tilt
it is supposed to apply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.validate import arm_indicators
from ..data.weighting import effective_sample_size
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import LongitudinalError
from ..fluctuation.iterative import (
    Fluctuation,
    InitialFit,
    solve_fluctuation,
)
from ..fluctuation.submodel import Submodel
from ..interventions.policy import (
    induced_probabilities,
    lazy_frame,
    policy_branches,
)
from ..learners.crossfit import Folds
from ..learners.density import ConditionalDensity, fit_conditional_density
from ..learners.density_ratio import classifier_ratio
from ..learners.super_learner import SuperLearnerDiagnostics
from ..utils.bounds import OutcomeScaler, bound
from ..utils.parallel import map_parallel
from ..utils.phases import (
    PhaseProfile,
    collect_phases,
    merge_worker_phases,
    phase,
    profiling,
)
from .data import LongitudinalData, RegimenMasks
from .regimen import Plan, Regimen, RegimenSpec

__all__ = [
    "FoldRecursionCell",
    "Mechanism",
    "NodeInputs",
    "RegimenFit",
    "SequentialStep",
    "StitchedInitial",
    "fit_mechanism",
    "fit_regimen",
    "outcome_design",
    "preflight_mechanism_support",
    "preflight_terminal_outcomes",
    "prepare_node",
    "seed_carried",
    "untargeted_fold_recursions",
]


#: Filler for a prediction at a row the estimator never reads -- a unit censored before
#: the node in question.  Any finite number in ``(0, 1)`` would do; a half keeps ``logit``
#: at zero, so a filled row cannot make a Newton step look large.
_FILLER = 0.5

#: What a refusal about one outer training fold may offer.  The split is drawn from the
#: seed and, with ``id=``, the cluster labels, and reads no treatment, outcome or
#: covariate, so a fold count or a seed found by trying them until one fits would choose
#: the partition by the values it must not read.  ``n_folds=1`` is the in-sample fit,
#: which draws no split at all.
_CROSS_FIT_NODE_REMEDY = (
    "The split reads no treatment, outcome or covariate, so trying fold counts or seeds "
    "until one fits would choose the partition by the values it must not read. Fit in sample "
    "(CrossFitting(enabled=False), or n_folds=1 on the engine), or {alternative}."
)

#: The key under which a node's single counterfactual prediction is filed on its
#: :class:`~cleverly.fluctuation.iterative.InitialFit` and
#: :class:`~cleverly.fluctuation.submodel.Submodel`.
#:
#: An index, **not a treatment level**, and that is the point: under a dynamic rule
#: different units are assigned different arms at the same node, so there is no level to
#: key by.  ``mtp_submodel`` keys by shift index for the same reason.  Nothing observable
#: depends on the value -- ``check_arms`` requires only that it be a float, and
#: ``check_matching_arms`` only that the fit and the submodel agree on it.
_REGIMEN_ARM = 0.0


def _policy_arm(code: int) -> float:
    """The fluctuation key of a policy node's per-arm prediction at level ``code``.

    Offset by one from :data:`_REGIMEN_ARM`, which keeps the observed-arm prediction, so the
    two kinds of key cannot collide.  Like that key it is an index and not a treatment
    level.  :func:`~cleverly.fluctuation.iterative.apply_logistic` moves every key by the
    same step along the same covariate, and the score reads the observed key alone, so a
    per-arm key changes no coefficient.
    """
    return 1.0 + float(code)


def _internal_prediction_key(labels: Sequence[str], role: str) -> str:
    """Return a private prediction key that cannot collide with a regimen label."""
    occupied = set(labels)
    key = f"__cleverly_{role}__"
    while key in occupied:
        key += "_"
    return key


@dataclass(frozen=True)
class Mechanism:
    """Out-of-fold treatment and censoring probabilities, evaluated at each regimen.

    ``treatment[t][label]`` is
    :math:`P(A_t = d_t(H_t) \\mid H_t, \\bar A_{t-1} = \\bar d_{t-1})`
    with the earlier treatments set to the plan's ``values``, and
    ``censoring[t][label]`` is :math:`P(C_t = 1 \\mid H_t, \\bar A_t = \\bar a_t)`.  At a
    label or rule node ``values`` is what the regimen would have assigned.  At a policy node
    it is the observed arm, so both are evaluated at the observed history there.  One
    model per node serves every regimen: the *fit* is shared, and only where it is
    evaluated differs.

    Both are indexed from zero by node, so ``treatment[0]`` is the mechanism at
    :math:`t = 1`.

    Parameters
    ----------
    treatment : tuple of dict of str to FloatArray
        Regimen treatment probabilities at each node.
    censoring : tuple of dict of str to FloatArray
        Regimen retention probabilities at each node.
    treatment_observed : tuple of FloatArray
        Out-of-fold treatment probability matrices at the observed histories.
    censoring_observed : tuple of FloatArray
        Out-of-fold retention probabilities at the observed treatment histories.
    treatment_diagnostics : tuple of tuple of SuperLearnerDiagnostics
        Learner diagnostics from each treatment node and fitted fold.
    censoring_diagnostics : tuple of tuple of SuperLearnerDiagnostics
        Learner diagnostics from each censoring node and fitted fold.
    policy_numerators : dict of str to dict of int to FloatArray
        Ratio numerators of the modified treatment policy nodes, by plan and node.
    densities : dict of int to ConditionalDensity
        Out-of-fold conditional densities of the continuous nodes.
    no_censoring_nodes : tuple of int
        Censoring nodes with no censored unit in the eligible sample.
    no_censoring_folds : dict of int to tuple of int
        By censoring node, the training folds with no censored unit.
    """

    treatment: tuple[dict[str, FloatArray], ...]
    censoring: tuple[dict[str, FloatArray], ...]
    #: Out-of-fold probability matrix at the observed history and treatment, one per
    #: node. Empty only on a hand-built mechanism.
    treatment_observed: tuple[FloatArray, ...] = ()
    #: Out-of-fold retention probability at the observed treatment history, one per
    #: censoring node. Empty for complete data and on a hand-built mechanism.
    censoring_observed: tuple[FloatArray, ...] = ()
    #: Super Learner diagnostics from the same treatment fits that produced
    #: ``treatment_observed``. The inner tuple contains one record per fitted fold.
    treatment_diagnostics: tuple[tuple[SuperLearnerDiagnostics, ...], ...] = ()
    #: Super Learner diagnostics from the same censoring fits that produced
    #: ``censoring_observed``. Empty for complete data and for a non-Super-Learner fit.
    censoring_diagnostics: tuple[tuple[SuperLearnerDiagnostics, ...], ...] = ()
    #: By plan label, by node counted from one, the ratio numerator of a modified treatment
    #: policy node: :math:`g^d_t(A_t \mid H_t)` read from the out-of-fold multinomial at a
    #: categorical node, and :math:`r_t = g^d_t / g_t` from the out-of-fold density or
    #: classifier at a continuous one.  Empty without a policy node.
    policy_numerators: dict[str, dict[int, FloatArray]] = field(default_factory=dict)
    #: The out-of-fold conditional density of each continuous node, by node counted from
    #: one, on the density route.  Empty without a continuous node.
    densities: dict[int, ConditionalDensity] = field(default_factory=dict)
    #: The censoring nodes, counted from one, at which no eligible unit was censored.  The
    #: eligible sample is the rows at risk before the node, with positive weight.  Such a
    #: node's factor is exactly one at every plan, and no learner is fitted there
    #: (``survtmle`` sets ``G_dC = 1`` at ``t = 1`` and has a ``noCens`` branch for the same
    #: case).  The nuisance report shows an omission in place of a model row.
    no_censoring_nodes: tuple[int, ...] = ()
    #: By censoring node, counted from one, the one-based training folds of a cross-fitted
    #: fit that held no censored unit while the eligible sample did.  Each such fold predicts
    #: retention exactly one, which is its empirical rate, and fits no learner.
    no_censoring_folds: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def cumulative(
        self, data: LongitudinalData, plan: Plan, bounds: tuple[float, float]
    ) -> FloatArray:
        r"""``(n, T)`` cumulative product :math:`\prod_{s \le t} g_s c_s`, bounded.

        The raw factors are multiplied first and each cumulative prefix is then truncated
        into ``bounds``.  This is the ``CalcCumG`` convention of R's canonical ``ltmle``
        implementation and the meaning of that package's ``gbounds`` argument: bounds on
        estimated *cumulative* probabilities.  Bounding every factor first is a different
        regularisation whose discrepancy is invisible at one time point and grows with
        the number of treatment and censoring nodes.

        The arm is read per *unit*, since a dynamic rule assigns different units
        different arms at the same node.  That selection has already happened by the time
        this runs: :func:`fit_mechanism` picks each row's assigned column out of the
        node's multinomial, so what is multiplied here is
        :math:`g_t(d_t(H_t) \mid H_t)` itself.  There is no branch on the arm left to
        take, which is what makes the categorical case the same expression as the binary
        one rather than a generalisation of it.
        """
        return self.cumulative_with_unbounded(data, plan, bounds)[1]

    def cumulative_with_unbounded(
        self,
        data: LongitudinalData,
        plan: Plan,
        bounds: tuple[float, float],
    ) -> tuple[FloatArray, FloatArray]:
        """Raw and bounded cumulative mechanism probabilities for one regimen.

        Keeping the raw prefix is not optional diagnostic bookkeeping: without it a
        result cannot distinguish a naturally small path probability from one replaced
        by the configured floor.  :meth:`cumulative` retains its historical bounded-only
        interface and delegates here.
        """
        lower, upper = bounds
        running = np.ones(data.n)
        raw_columns = []
        bounded_columns = []
        for time in range(1, data.n_times + 1):
            treatment = self.treatment[time - 1][plan.label]
            censoring = self.censoring[time - 1][plan.label]
            running = running * treatment
            if data.censoring_names:
                running = running * censoring
            raw_columns.append(running)
            bounded_columns.append(bound(running, lower, upper))
        return np.column_stack(raw_columns), np.column_stack(bounded_columns)


@dataclass(frozen=True)
class _NodeRegression:
    """One node's fitted regression, before any targeting quantities are built."""

    time: int
    at_risk: BoolArray
    trained_on: BoolArray
    fitted_on: BoolArray
    pseudo_outcome: FloatArray
    initial: FloatArray
    learner_diagnostics: tuple[SuperLearnerDiagnostics, ...] = ()
    #: ``(n, K_t)`` per-arm predictions at a policy node, ``None`` elsewhere.
    initial_by_arm: FloatArray | None = None
    #: The policy-weighted mean of ``initial_by_arm``, ``None`` off a policy node.
    initial_marginal: FloatArray | None = None


@dataclass(frozen=True)
class NodeInputs:
    """Everything one node of the backward recursion needs before it is fluctuated.

    The split exists because *what* is fluctuated at a node is not always one regimen.
    A plain fit solves one score equation per node per regimen and can do the regression
    and the fluctuation in one breath; a working model over regimens
    (:mod:`cleverly.longitudinal.msm`) solves ``p`` equations per node **pooled across
    the declared plans**, so it needs every plan's regression at a node in hand before
    any of them is updated.  Both read the same regressions, from here.

    ``counterfactual`` is :math:`1/\\prod g` on the at-risk set and zero elsewhere -- the
    inverse-probability loss weight the *update* is fitted with.  ``clever`` is that
    masked down to the units that actually followed, which is the multiplier in the EIF
    and score.  On a plan with a policy node both carry the ratio numerator
    :math:`\\prod \\pi`, and "followed" means that the observed arm has positive
    intervention density at every node so far.  The logistic submodel itself is an
    intercept shift; putting ``clever`` in that submodel instead would solve the same
    score along a different path and cease to match the loss-weighted update in canonical
    R ``ltmle``.

    Parameters
    ----------
    time : int
        One-based node index.
    at_risk : BoolArray
        Rows whose histories remain observed and regimen-consistent at this node.
    trained_on : BoolArray
        Rows that followed the regimen through this node.
    fitted_on : BoolArray
        Rows this node's regression was fitted on, and the rows its fluctuation solves
        over. A node built here is a single-fold node, so these are the followers.
    pseudo_outcome : FloatArray
        Target supplied to the node regression.
    initial : FloatArray
        Initial node prediction before targeting.
    counterfactual : FloatArray
        Cumulative inverse-probability loss weight on the at-risk rows.
    clever : FloatArray
        Counterfactual weight restricted to rows that followed the regimen.
    learner_diagnostics : tuple of SuperLearnerDiagnostics
        Learner diagnostics from each fitted fold of the node regression.
    initial_by_arm : FloatArray or None
        ``(n, K_t)`` initial predictions at each level of a policy node, ``None`` elsewhere.
    initial_marginal : FloatArray or None
        The policy-weighted mean of ``initial_by_arm``, ``None`` off a policy node.
    policy : FloatArray or None
        The ``(n, K_t)`` policy density at a policy node, ``None`` elsewhere.
    """

    time: int
    at_risk: BoolArray
    #: Every row that followed the regimen through this node.  ``clever`` is nonzero
    #: exactly here, whichever rows the regression was fitted on.
    trained_on: BoolArray
    #: The rows the regression was fitted on, which equal ``trained_on`` here.  A
    #: cross-fitted fit builds no node inputs inside a fold: its one pooled fluctuation per
    #: node solves over every follower, against the stitched out-of-fold predictions.
    fitted_on: BoolArray
    pseudo_outcome: FloatArray
    initial: FloatArray
    counterfactual: FloatArray
    clever: FloatArray
    #: Super Learner diagnostics from this node's regression, one per fitted fold.
    learner_diagnostics: tuple[SuperLearnerDiagnostics, ...] = ()
    initial_by_arm: FloatArray | None = None
    initial_marginal: FloatArray | None = None
    policy: FloatArray | None = None


@dataclass(frozen=True)
class SequentialStep:
    """One retained node of the backward recursion.

    Parameters
    ----------
    time : int
        One-based node index.
    trained_on : BoolArray
        Rows eligible for the node regression.
    at_risk : BoolArray
        Rows whose history is observed and consistent with the regimen.
    pseudo_outcome : FloatArray
        Outcome of the node fluctuation and of the influence curve's node term.
    initial : FloatArray
        Initial node prediction before targeting.
    targeted : FloatArray
        Node prediction after targeting, at the arm the row took under the plan.  At a
        policy node that is the observed arm, and the influence curve's residual reads it.
    clever : FloatArray
        Cumulative inverse-probability multiplier on regimen followers.  On a plan with a
        policy node it is the cumulative ratio, the policy numerator over the bounded
        mechanism.
    fluctuation : Fluctuation
        Targeting solve retained for this node.
    learner_diagnostics : tuple of SuperLearnerDiagnostics
        Learner diagnostics from each fitted fold of the node regression.
    regression_target : FloatArray or None
        Target the node regression was fitted to, when it differs from ``pseudo_outcome``.
    marginal : FloatArray or None
        At a policy node, the policy-weighted mean of the targeted per-arm predictions,
        which the node carries to the earlier node and which the plug-in reads at the first
        node.  ``None`` elsewhere.
    targeted_by_arm : FloatArray or None
        ``(n, K_t)`` targeted predictions at each level of a policy node, ``None``
        elsewhere.
    initial_by_arm : FloatArray or None
        ``(n, K_t)`` initial predictions at each level of a policy node, the arrays the
        fluctuation moved into ``targeted_by_arm``.  ``None`` elsewhere.

    Attributes
    ----------
    n_trained : int
    value : FloatArray
    """

    time: int
    #: Rows eligible for the regression: followed the regimen and stayed under
    #: observation through this node.  With cross-fitting, each outer model uses the
    #: eligible rows in its training complement, and the one pooled fluctuation of the
    #: node solves its score over all of them.
    trained_on: BoolArray
    #: Rows whose history at this node is observed and regimen-consistent -- the set the
    #: regression *predicts* for, and the population the assigned arm is a statement
    #: about.  Equal to the previous node's ``trained_on`` on an end-of-study fit, which
    #: is what closes the recursion; on a survival fit it is that set less the units that
    #: had the event there, which closes it just as well and is the general statement.
    at_risk: BoolArray
    #: What this node's fluctuation was fitted *to*: the later node's targeted prediction
    #: on an end-of-study fit, and on a survival one the composition
    #: :math:`Z_t = Y_t + (1 - Y_t)\\,\\bar{Q}^*_{t+1}`.  Stored rather than recomputed
    #: because the influence curve's ``t``-th term needs the same quantity, and the one
    #: place this recursion could silently disagree with itself is by composing the
    #: pseudo-outcome twice and composing it differently.  On a single-fold fit the node
    #: regression was fitted to this array too.
    pseudo_outcome: FloatArray
    #: The stitched out-of-fold regression on a cross-fitted fit, and the in-sample one at
    #: a single fold.  Filled with ``0.5`` off ``at_risk``.
    initial: FloatArray
    targeted: FloatArray
    clever: FloatArray
    fluctuation: Fluctuation
    #: Super Learner diagnostics retained from the regression that produced ``initial``.
    learner_diagnostics: tuple[SuperLearnerDiagnostics, ...] = ()
    #: What the node regression was fitted to, where that is not ``pseudo_outcome``.  On a
    #: cross-fitted fit each fold regresses its own *untargeted* recursion, so row ``i``
    #: holds the target that row ``i``'s held-out fold composed from that fold's untargeted
    #: prediction at the later node.  A nuisance-loss diagnostic compares ``initial``
    #: against this array.  ``None`` on a single-fold fit, whose regression target is
    #: ``pseudo_outcome``.
    regression_target: FloatArray | None = None
    marginal: FloatArray | None = None
    targeted_by_arm: FloatArray | None = None
    initial_by_arm: FloatArray | None = None

    @property
    def n_trained(self) -> int:
        return int(self.trained_on.sum())

    @property
    def value(self) -> FloatArray:
        """What the node carries: ``marginal`` at a policy node, else ``targeted``."""
        return self.targeted if self.marginal is None else self.marginal


@dataclass(frozen=True)
class RegimenFit:
    """The estimate under one regimen, with the pieces that produced it."""

    regimen: RegimenSpec
    #: On the ``[0, 1]`` outcome scale, as everything inside the recursion is.
    psi_scaled: float
    influence_curve_scaled: FloatArray
    #: The node this fit's parameter is indexed by: ``T`` for an end-of-study outcome,
    #: and the horizon of the cumulative risk for a survival one.  Carried as a field
    #: rather than parsed back out of a report name, so ``diagnostics()`` and
    #: ``summary()`` read the regimen and the horizon rather than reconstructing them.
    horizon: int
    #: Which absorbing cause this fit's parameter is the incidence of, or ``None`` on a
    #: fit with a single event or an end-of-study outcome.  Carried as a field for the
    #: reason :attr:`horizon` is: ``diagnostics()`` and ``summary()`` read it rather than
    #: parsing it back out of a report name, so the two cannot drift.
    cause: str | None
    steps: tuple[SequentialStep, ...]
    #: Raw cumulative treatment-and-censoring probability before ``g_bounds``.
    cumulative_unbounded: FloatArray
    #: The same prefixes after applying ``g_bounds``.
    #:
    #: On a cross-fitted fit these are the out-of-fold prefixes, and they are the only
    #: mechanism the fit divides by: the pooled fluctuation at each node reads
    #: ``1 / cumulative[:, t - 1]`` on the followers and nothing else.  So
    #: ``step.clever`` is exactly that reciprocal on ``step.trained_on`` and zero elsewhere,
    #: on a cross-fitted fit and a single-fold one alike.  On a plan with a policy node it
    #: is ``cumulative_numerator[:, t - 1] / cumulative[:, t - 1]`` there: the bound applies
    #: to the denominator only.  A diagnostic that wants how much
    #: of the mechanism the bounds moved reads this against :attr:`cumulative_unbounded`.
    cumulative: FloatArray
    #: The ``(n, T)`` arms this regimen assigned *this* sample.  Constant down each
    #: column for a static plan; for a rule it is the thing ``diagnostics()`` reports,
    #: since what share of the at-risk units a rule would treat is a property of the
    #: data rather than of the declaration.
    assignment: FloatArray
    #: The observation weights the fit ran under, normalised to mean one and all-ones on
    #: an unweighted fit.  Held so the leverage below can be reported at
    #: :math:`w_i / \\prod g`: the weighting's cost and the clever covariate's *multiply*,
    #: and a diagnostic showing only one of them reads as comfortable on a fit that is thin
    #: on both -- the reasoning :mod:`cleverly.sensitivity.positivity` already applies at
    #: one time point.
    obs_weights: FloatArray
    #: The plan's evaluated policy densities, one entry per node and ``None`` off a policy
    #: node.  Empty on a plan without one.  At a policy node :attr:`assignment` holds the
    #: observed arm, and this holds what the policy would draw.
    policy: tuple[FloatArray | None, ...] = ()
    #: ``(n, T)`` running product of the intervention density at the observed arms, the
    #: numerator of the cumulative ratio.  ``None`` on a plan without a policy node.
    cumulative_numerator: FloatArray | None = None
    #: ``(n, T)`` raw per-node ratio of a plan with a modified treatment policy: the node's
    #: numerator over its unbounded treatment probability at the observed arm, which at a
    #: continuous node is the density ratio :math:`r_t` itself.  Not bounded by
    #: ``g_bounds``.  ``None`` on a plan without such a node.
    node_ratio: FloatArray | None = None

    @property
    def leverage(self) -> FloatArray:
        """Final node's clever covariate, weighted: :math:`w_i / \\prod_{s} g_s c_s`.

        What one unit can contribute to the estimating equation, which is the product of
        the two reweightings a fit applies and not either alone.
        """
        return np.asarray(self.obs_weights * self.steps[-1].clever, dtype=float)

    @property
    def max_weight(self) -> float:
        """Largest weighted clever-covariate value at the final node.

        The reciprocal of the smallest cumulative probability of following the regimen,
        times the unit's observation weight, so it is the leverage a single unit can have
        on the estimate.
        """
        weights = self.leverage
        return float(np.max(weights)) if weights.size else float("nan")

    @property
    def effective_n(self) -> float:
        """Kish effective sample size of the final node's weighted clever covariate.

        How many units the estimate is really averaging over once the weighting by
        :math:`w / \\prod g` is taken into account.  A number far below ``n`` says the
        regimen is supported by few units, whatever the reported standard error.
        """
        return effective_sample_size(self.leverage, on_degenerate=0.0)

    @property
    def converged(self) -> bool:
        return all(step.fluctuation.converged for step in self.steps)


def _check_categorical_fold_support(
    target: FloatArray,
    fit_mask: BoolArray,
    folds: Folds,
    classes: Sequence[float],
    levels: Sequence[object],
    node_name: str,
    *,
    every_level: bool = False,
) -> None:
    """Refuse a mechanism fit whose training law omits an observed treatment level.

    A missing class cannot be repaired by aligning a learner's probability columns: the
    requested assigned-arm probability is unidentified in that training fold.  Checking
    here also gives the analyst the original label, rather than a downstream matrix-shape
    error involving its internal dense code.

    **A two-level node is left alone by default**, and that is a compatibility decision
    rather than an oversight.  At two classes
    :func:`~cleverly.learners._fitting.predict_probabilities` delegates to ``predict_mean``
    and a degenerate training fold yields that fold's constant, which is the behaviour
    every binary fit has had; the diagnosis a binary panel reaches instead is
    :func:`prepare_node`'s "no unit followed regimen" refusal, which names the regimen
    rather than the fold.  Refusing here at ``K = 2`` as well would change an error a
    binary fit already had, for a case the existing path already reports.

    At three or more levels the fallback is not benign: the missing arm's column comes
    back zero, so its clever covariate is a division by zero and the cumulative product's
    reciprocal is infinite.  That is why the check exists at all, and why it starts here.

    ``every_level=True`` drops the two-level exemption, and the **first** node is where it
    is used.  Every unit is at risk there, so the first node's arms are the one stratum an
    unstratified split can be held to at all, and a split that loses an arm at the first
    node has lost it for every regimen the fit reports.  The later binary nodes keep the
    documented fallback, because their at-risk sets are regimen-dependent and the refusal
    a reader needs there names the regimen.

    Parameters
    ----------
    target : ndarray
        The node's arm code for every row.
    fit_mask : ndarray
        Which rows the node's mechanism is fitted on.
    folds : Folds
        The realized outer split.
    classes : sequence of float
        Every arm code the node declares.
    levels : sequence of object
        The caller-facing label of each code, in the same order.
    node_name : str
        The node's name, for the refusal.
    every_level : bool, default=False
        Whether to check a two-level node as well.
    """
    if len(classes) < 3 and not every_level:
        return
    eligible = np.asarray(fit_mask, dtype=bool)
    training_sets: list[tuple[int, IntArray]]
    if folds.is_single:
        training_sets = [(0, np.flatnonzero(eligible))]
    else:
        training_sets = [
            (fold, train[eligible[train]]) for fold, (train, _) in enumerate(folds, start=1)
        ]
    for fold, training in training_sets:
        present = set(np.asarray(target[training], dtype=float).tolist())
        missing = [levels[index] for index, code in enumerate(classes) if code not in present]
        if missing:
            where = "the eligible sample" if folds.is_single else f"training fold {fold}"
            remedy = (
                "Collect more observations at the rare level"
                if folds.is_single
                else _CROSS_FIT_NODE_REMEDY.format(
                    alternative="collect more observations at the rare level"
                )
            )
            raise LongitudinalError(
                f"treatment node {node_name!r} is missing level(s) {missing!r} in {where}; "
                "every treatment-mechanism training set must contain every observed level. "
                + remedy
            )


def preflight_mechanism_support(
    data: LongitudinalData,
    fit_masks: RegimenMasks,
    folds: Folds,
) -> None:
    """Check every node's mechanism training support before the first learner is fitted.

    The per-node check used to run inside the node loop, so node 3's missing arm was
    reported after two nodes had been fitted and a Super Learner library had run twice.
    Nothing it reads depends on a fit, so every node is asked here instead and a fit that
    cannot finish spends no learner time finding that out.

    The first node is checked at any level count, the later ones only from three levels
    up.  :func:`_check_categorical_fold_support` says why the two differ.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    fit_masks : RegimenMasks
        The masks :func:`fit_mechanism` builds for the observed treatment.
    folds : Folds
        The realized outer split.
    """
    for time in range(1, data.n_times + 1):
        if data.is_continuous_node(time) or data.is_held_node(time):
            continue
        at_risk = fit_masks.uncensored[:, time - 1] & fit_masks.event_free[:, time - 1]
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        classes = tuple(float(code) for code in range(len(data.treatment_levels[time - 1])))
        _check_categorical_fold_support(
            arm,
            at_risk,
            folds,
            classes,
            data.treatment_levels[time - 1],
            data.decision_name(time),
            every_level=time == 1 and not folds.is_single,
        )


def fit_mechanism(
    data: LongitudinalData,
    plans: Sequence[Plan],
    *,
    treatment_learner: Learner,
    censoring_learner: Learner,
    folds: Folds,
    n_jobs: int = 1,
    density_bins: int = 20,
    ratio: str = "density",
) -> Mechanism:
    """Fit the treatment and censoring mechanisms at every node, out of fold.

    Each node's model is fitted on the units still in the study *before* that node's
    decision -- not on the regimen's followers.  The conditioning set carries the earlier
    treatments as columns, so one model answers for every regimen and is simply evaluated
    at each one's arms.  Under a dynamic rule those arms differ by row, which changes
    where the model is *evaluated* and nothing about how it is fitted.

    On a survival fit "still in the study" excludes the units that have already had the
    event, and that exclusion is not cosmetic: such a unit has no treatment at this node,
    so its ``A_t`` is missing, and the design fills a missing arm with zero.  Left in the
    fit mask it would be trained on as an untreated observation and bias ``g_t`` -- and
    with it every clever covariate downstream of it.

    Both factors are fitted by weighted loss when the data carries observation weights, so
    what they estimate is the *tilted* law's mechanism -- which is what that law's
    influence function is built from, and what a weighted learner converges to.
    """
    treatment: list[dict[str, FloatArray]] = []
    censoring: list[dict[str, FloatArray]] = []
    treatment_observed: list[FloatArray] = []
    censoring_observed: list[FloatArray] = []
    treatment_diagnostics: list[tuple[SuperLearnerDiagnostics, ...]] = []
    censoring_diagnostics: list[tuple[SuperLearnerDiagnostics, ...]] = []
    numerators: dict[str, dict[int, FloatArray]] = {
        plan.label: {} for plan in plans if plan.has_mtp
    }
    densities: dict[int, ConditionalDensity] = {}
    no_censoring_nodes: list[int] = []
    no_censoring_folds: dict[int, tuple[int, ...]] = {}
    # Neither factor depends on a regimen, so one scan serves every node.  `followed` is
    # unused here and the all-true assignment makes that explicit rather than implicit.
    with phase("mask_construction"):
        fit_masks = data.regimen_masks(data.treatment)
    preflight_mechanism_support(data, fit_masks, folds)
    for time in range(1, data.n_times + 1):
        at_risk = fit_masks.uncensored[:, time - 1] & fit_masks.event_free[:, time - 1]
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        if data.is_held_node(time):
            # An identity node of a held design: its factor is exactly one in the
            # numerator and the denominator, and it fits no model.
            treatment.append({plan.label: np.ones(data.n) for plan in plans})
            treatment_observed.append(np.zeros((data.n, 0)))
            treatment_diagnostics.append(())
        elif data.is_continuous_node(time):
            with phase("mechanism_fit"):
                density = _continuous_node_ratios(
                    data,
                    plans,
                    time,
                    at_risk,
                    numerators,
                    treatment_learner=treatment_learner,
                    folds=folds,
                    density_bins=density_bins,
                    ratio=ratio,
                    n_jobs=n_jobs,
                )
            if density is not None:
                densities[time] = density
            # The density ratio is the whole treatment factor at a continuous node, carried
            # in the numerator; the denominator's treatment factor is one there, so the bound
            # applies to the censoring product and the categorical factors only.
            treatment.append({plan.label: np.ones(data.n) for plan in plans})
            treatment_observed.append(np.zeros((data.n, 0)))
            treatment_diagnostics.append(())
        else:
            _categorical_node(
                data,
                plans,
                time,
                at_risk,
                arm,
                treatment,
                treatment_observed,
                treatment_diagnostics,
                numerators,
                treatment_learner=treatment_learner,
                folds=folds,
                n_jobs=n_jobs,
            )
        if not data.censoring_names:
            censoring.append({plan.label: np.ones(data.n) for plan in plans})
            continue
        stayed = np.where(at_risk, data.uncensored[:, time - 1].astype(float), 0.0)
        # No eligible unit was censored at this node, so the retention factor is exactly
        # one and the learner has a constant target, which a standard classifier refuses.
        # This is the first node of every default integer grid.
        if _retains_every_eligible_unit(data, at_risk, time):
            censoring.append({plan.label: np.ones(data.n) for plan in plans})
            censoring_observed.append(np.ones(data.n))
            censoring_diagnostics.append(())
            no_censoring_nodes.append(time)
            continue
        censor_designs = {
            plan.label: data.history_design(time, treatment=plan.values, include_current=True)
            for plan in plans
        }
        censor_observed_key = _internal_prediction_key(tuple(censor_designs), "observed_censoring")
        censor_prediction_designs = {
            **censor_designs,
            censor_observed_key: data.history_design(time, include_current=True),
        }
        constant_folds: list[int] = []
        with phase("mechanism_fit"):
            predictions, diagnostics = cross_fit_predictions(
                censoring_learner,
                data.history_design(time, include_current=True),
                stayed,
                data.weights,
                folds,
                task="classification",
                predict_designs=censor_prediction_designs,
                fit_mask=at_risk,
                groups=data.cluster,
                clip=(0.0, 1.0),
                n_jobs=n_jobs,
                constant_folds=constant_folds,
            )
        if constant_folds:
            no_censoring_folds[time] = tuple(sorted(constant_folds))
        censoring_observed.append(np.asarray(predictions.pop(censor_observed_key), dtype=float))
        censoring_diagnostics.append(tuple(diagnostics))
        censoring.append(predictions)
    return Mechanism(
        tuple(treatment),
        tuple(censoring),
        tuple(treatment_observed),
        tuple(censoring_observed),
        tuple(treatment_diagnostics),
        tuple(censoring_diagnostics),
        numerators,
        densities,
        tuple(no_censoring_nodes),
        no_censoring_folds,
    )


def _retains_every_eligible_unit(data: LongitudinalData, at_risk: BoolArray, time: int) -> bool:
    """Whether no unit at risk before censoring node ``time``, with positive weight, left.

    A zero-weight row is in no weighted fit, so it cannot make the node's retention rate
    differ from one.
    """
    eligible = at_risk & (data.weights > 0.0)
    return not bool(np.any(eligible & ~data.uncensored[:, time - 1]))


def _categorical_node(
    data: LongitudinalData,
    plans: Sequence[Plan],
    time: int,
    at_risk: BoolArray,
    arm: FloatArray,
    treatment: list[dict[str, FloatArray]],
    treatment_observed: list[FloatArray],
    treatment_diagnostics: list[tuple[SuperLearnerDiagnostics, ...]],
    numerators: dict[str, dict[int, FloatArray]],
    *,
    treatment_learner: Learner,
    folds: Folds,
    n_jobs: int,
) -> None:
    r"""One categorical node's multinomial mechanism, and its policy nodes' numerators.

    At a modified treatment policy node the numerator is the discrete formula of Díaz,
    Williams, Hoffman and Schenck (2023, Section 4), :math:`g^d(A_t \mid H_t) =
    \sum_e p(e) \sum_s 1\{d_e(s, H_t) = A_t\}\, g(s \mid H_t)`, read from the same
    out-of-fold multinomial the denominator selects from.  It is not bounded; the bound
    applies to the running denominator, as for a known policy's :math:`q`.
    """
    designs = {plan.label: data.history_design(time, treatment=plan.values) for plan in plans}
    observed_key = _internal_prediction_key(tuple(designs), "observed_treatment")
    prediction_designs = {**designs, observed_key: data.history_design(time)}
    with phase("mechanism_fit"):
        classes = tuple(float(code) for code in range(len(data.treatment_levels[time - 1])))
        probabilities, diagnostics = cross_fit_predictions(
            treatment_learner,
            data.history_design(time),
            arm,
            data.weights,
            folds,
            task="classification",
            predict_designs=prediction_designs,
            fit_mask=at_risk,
            groups=data.cluster,
            clip=(0.0, 1.0),
            classes=classes,
            n_jobs=n_jobs,
        )
    treatment_observed.append(np.asarray(probabilities.pop(observed_key), dtype=float))
    treatment_diagnostics.append(tuple(diagnostics))
    rows = np.arange(data.n)
    treatment.append(
        {
            plan.label: probabilities[plan.label][rows, plan.arm(time).astype(np.int64)]
            for plan in plans
        }
    )
    observed = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
    for plan in plans:
        if not plan.is_mtp_node(time):
            continue
        assigned = plan.mtp_assignments[time - 1]
        assert assigned is not None
        g = np.asarray(probabilities[plan.label], dtype=float)
        codes = observed.astype(np.int64)
        numerator = np.zeros(data.n)
        for probability, level_map in assigned:
            numerator = numerator + probability * induced_probabilities(level_map, g)[rows, codes]
        numerators[plan.label][time] = np.where(at_risk, numerator, 0.0)


def _continuous_node_ratios(
    data: LongitudinalData,
    plans: Sequence[Plan],
    time: int,
    at_risk: BoolArray,
    numerators: dict[str, dict[int, FloatArray]],
    *,
    treatment_learner: Learner,
    folds: Folds,
    density_bins: int,
    ratio: str,
    n_jobs: int,
) -> ConditionalDensity | None:
    """A continuous node's policy ratios, by the binned density or by classification.

    The density route fits one pooled-hazard density of the dose given the observed
    history on the rows at risk, with bin edges from those rows of the whole sample, and
    evaluates each plan's policy by Equation (3) of Díaz, Williams, Hoffman and Schenck
    (2023).  The classifier route fits one stacked classifier per plan (their Section 5.4).
    Both are cross-fitted on ``folds``, so every row's ratio comes from a model that did not
    train on it.  Every plan holds a modified treatment policy at a continuous node, since
    :func:`~cleverly.longitudinal.regimen.refuse_continuous_assignments` refuses the rest.
    """
    dose = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
    history = data.history_design(time)
    frame = lazy_frame(lambda: data.policy_frame(time))
    density: ConditionalDensity | None = None
    if ratio == "density":
        density, _ = fit_conditional_density(
            treatment_learner,
            history,
            dose,
            data.weights,
            folds,
            n_bins=density_bins,
            groups=data.cluster,
            n_jobs=n_jobs,
            fit_mask=at_risk,
        )
    for plan in plans:
        targets = plan.carry_targets(time)
        assert targets is not None
        branches = policy_branches(plan.regimen.plan[time - 1])  # type: ignore[union-attr]
        if density is not None:
            numerator = np.zeros(data.n)
            for probability, branch in branches:
                numerator = numerator + probability * branch.ratio(dose, frame, density.density_at)
        else:
            (numerator,) = classifier_ratio(
                treatment_learner,
                history,
                dose,
                tuple(
                    (probability, target)
                    for (probability, _), target in zip(branches, targets, strict=True)
                ),
                data.weights,
                folds,
                evaluate_at=[dose],
                fit_mask=at_risk,
                groups=data.cluster,
                n_jobs=n_jobs,
            )
        numerators[plan.label][time] = np.where(at_risk, numerator, 0.0)
    return density


def seed_carried(data: LongitudinalData, scaler: OutcomeScaler) -> FloatArray:
    r""":math:`\bar Q_{k+1}`, what the backward recursion starts from.

    On a survival fit it is zero, so that the composition at the first node visited
    returns :math:`Y_k` exactly and the two statements of the recursion are one.  On an
    end-of-study fit it is the scaled outcome, filled at the rows no node reads.
    """
    if data.is_survival:
        return np.zeros(data.n)
    observed_outcome = data.uncensored_through(data.n_times)
    scaled = np.where(observed_outcome, scaler.scale(np.nan_to_num(data.outcome, nan=0.0)), _FILLER)
    return np.clip(scaled, 0.0, 1.0)


def preflight_terminal_outcomes(
    data: LongitudinalData,
    plans: Sequence[Plan],
    horizons: Sequence[int],
    folds: Folds,
) -> None:
    """Check that every reported horizon has followers, before any mechanism learner fits.

    Who follows a regimen through a horizon does not depend on a fitted nuisance, so a
    regimen, horizon or outer training complement with no follower is refused now.  A
    horizon whose followers all hold one event value is not refused: its regression is that
    value (:func:`constant_target`).

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        Resolved regimen assignments.
    horizons : sequence of int
        Reported outcome or event times.
    folds : Folds
        The realized outer split.
    """
    preflight_policy_support(data, plans, horizons, folds)
    for plan in plans:
        masks = plan.masks(data)
        for horizon in horizons:
            at_risk = masks.at_risk(horizon)
            followers = masks.following(horizon)
            for outer_fold, train_rows in _fit_rows(data, folds):
                _require_regimen_followers(
                    data, plan, horizon, at_risk, followers & train_rows, outer_fold=outer_fold
                )


def _fit_rows(data: LongitudinalData, folds: Folds) -> list[tuple[int | None, BoolArray]]:
    """Each outer training set as a row mask, with its fold index, or ``None`` in sample."""
    if folds.is_single:
        return [(None, np.ones(data.n, dtype=bool))]
    rows: list[tuple[int | None, BoolArray]] = []
    for fold, (train, _) in enumerate(folds):
        train_rows = np.zeros(data.n, dtype=bool)
        train_rows[train] = True
        rows.append((fold, train_rows))
    return rows


def preflight_policy_support(
    data: LongitudinalData,
    plans: Sequence[Plan],
    horizons: Sequence[int],
    folds: Folds,
) -> None:
    """Check every policy node's regression support before any learner is fitted.

    A policy node's regression predicts every level the policy can draw.  A level with
    positive policy probability on an at-risk row and no fitted row at that level has an
    all-zero design column, so the regression would extrapolate to it.  Nothing here reads a
    fit, so every policy node up to the last horizon, and every outer training set, is
    checked now.  The checks are the ones :func:`_fit_node_regression` repeats.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        Resolved regimen assignments.
    horizons : sequence of int
        Reported outcome or event times.
    folds : Folds
        The realized outer split.
    """
    last = max(horizons)
    for plan in plans:
        if not plan.has_policy:
            continue
        masks = plan.masks(data)
        for time in range(last, 0, -1):
            if not plan.is_policy_node(time):
                continue
            at_risk = masks.at_risk(time)
            followers = masks.following(time)
            for outer_fold, train_rows in _fit_rows(data, folds):
                fitted_on = followers & train_rows
                _require_regimen_followers(
                    data, plan, time, at_risk, fitted_on, outer_fold=outer_fold
                )
                _check_policy_arm_support(
                    data, plan, time, at_risk, fitted_on, outer_fold=outer_fold
                )


def _check_policy_arm_support(
    data: LongitudinalData,
    plan: Plan,
    time: int,
    at_risk: BoolArray,
    fitted_on: BoolArray,
    *,
    outer_fold: int | None,
) -> None:
    """Refuse a policy level that has policy mass at risk and no row to fit it on.

    The residuals of the influence curve read the observed arms only, so they cannot see
    the bias of a prediction at a level no fitted row received.  This check is the only
    guard against it.
    """
    if not plan.is_policy_node(time) or data.is_continuous_node(time):
        return
    density = plan.policy_at(time)
    observed = plan.arm(time)
    for code, level in enumerate(data.treatment_levels[time - 1]):
        if not np.any(density[at_risk, code] > 0.0):
            continue
        if np.any(fitted_on & (observed == float(code))):
            continue
        where = "" if outer_fold is None else f" in outer training fold {outer_fold + 1}"
        raise LongitudinalError(
            f"regimen {plan.label!r} puts probability on {level!r} at time {time}, but no "
            f"unit{where} that remains on the policy through time {time} received "
            f"{level!r} there. The regression at that node would predict {level!r} with no "
            "row to fit it on, so the policy is not supported by this sample."
            + (
                ""
                if outer_fold is None
                else " "
                + _CROSS_FIT_NODE_REMEDY.format(alternative="choose a policy this sample supports")
            )
        )


def _require_regimen_followers(
    data: LongitudinalData,
    plan: Plan,
    time: int,
    at_risk: BoolArray,
    fitted_on: BoolArray,
    *,
    outer_fold: int | None,
) -> None:
    """Refuse an empty regimen training set with the recursion's diagnostic."""
    if fitted_on.any():
        return
    where = (
        "while remaining in the study"
        if outer_fold is None
        else f"in outer training fold {outer_fold + 1}"
    )
    opening = (
        f"no unit's observed treatment history through time {time} has positive "
        f"probability under regimen {plan.label!r} {where}, so the sequential regression "
        "there has nothing to fit. The policy is not supported by this sample."
        if plan.has_policy
        else f"no unit followed regimen {plan.label!r} through time {time} {where}, so "
        "the sequential regression there has nothing to fit. The regimen is not "
        "supported by this sample."
    )
    raise LongitudinalError(
        opening
        + (
            ""
            if outer_fold is None
            else " " + _CROSS_FIT_NODE_REMEDY.format(alternative="choose a supported regimen")
        )
        + _risk_set_hint(data, plan, time)
        + _rule_hint(plan, data, at_risk, time)
    )


def outcome_design(
    data: LongitudinalData, plan: Plan, time: int, arm: int | None = None
) -> FloatArray:
    """The outcome regression's design at node ``time`` of ``plan``.

    Without a policy node it is :meth:`~cleverly.longitudinal.LongitudinalData.covariate_history`
    bit for bit, which ``tests/unit/test_sequential_design.py`` pins.  With one it adds a
    drop-first indicator block for the observed arm of every earlier **policy** node and,
    at a policy node, a block for the current arm.  A label or rule node adds no column: on
    the rows that remain on the plan its arm is a function of columns this design already
    holds, the argument of :meth:`~cleverly.longitudinal.LongitudinalData.history_design`.
    A policy node's arm is not.  It is drawn, so the regression must see it, and the node
    predicts each level by setting the current block.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan.
    time : int
        The node, counted from one.
    arm : int or None
        The level code to set the current policy node's block to, for a prediction, or at a
        continuous policy node the branch whose assigned dose to set.  ``None`` uses the
        observed arm.

    Returns
    -------
    FloatArray
        The design matrix, one row per unit.
    """
    history = data.covariate_history(time)
    if not plan.has_policy:
        return history
    blocks = [history]
    for node in range(1, time + 1):
        if not plan.is_policy_node(node):
            continue
        current = node == time and arm is not None
        if data.is_continuous_node(node):
            # A dose enters as itself.  At the current node a prediction sets it to the dose
            # one branch of the policy assigns, d_e(A_t, H_t), per row.
            targets = plan.carry_targets(node)
            dose = (
                targets[int(arm)]  # type: ignore[arg-type]
                if current and targets is not None
                else np.nan_to_num(plan.values[:, node - 1], nan=0.0)
            )
            blocks.append(np.asarray(dose, dtype=float).reshape(-1, 1))
            continue
        codes = np.full(data.n, float(arm)) if current else plan.values[:, node - 1]  # type: ignore[arg-type]
        blocks.append(arm_indicators(codes, len(data.treatment_levels[node - 1])))
    return np.hstack(blocks)


def _carried(policy: FloatArray, by_arm: FloatArray) -> FloatArray:
    r""":math:`\sum_j q_t(j \mid H_t)\, Q_t(j, H_t)`, the policy-weighted mean of a node.

    The one place a policy node's carried value is formed, from the initial per-arm
    predictions in an untargeted recursion and from the targeted ones after a fluctuation.
    """
    return np.asarray(np.sum(policy * by_arm, axis=1), dtype=float)


def _fit_node_regression(
    data: LongitudinalData,
    plan: Plan,
    carried: FloatArray,
    time: int,
    horizon: int,
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    cause: str | None = None,
    masks: RegimenMasks | None = None,
    fit_rows: BoolArray | None = None,
    outer_fold: int | None = None,
    n_jobs: int = 1,
) -> _NodeRegression:
    """Fit one node's regression without constructing targeting-only arrays.

    The outer cross-fitted recursion needs only this result: it stitches the held-out
    predictions and builds the clever covariate once in the parent during pooled targeting.
    Keeping that path here prevents each worker from constructing and profiling arrays that it
    immediately discards.
    """
    if masks is None:
        with phase("mask_construction"):
            masks = plan.masks(data)
    at_risk = masks.at_risk(time)
    trained_on = masks.following(time)
    fitted_on = trained_on if fit_rows is None else trained_on & fit_rows
    _require_regimen_followers(data, plan, time, at_risk, fitted_on, outer_fold=outer_fold)
    _check_policy_arm_support(data, plan, time, at_risk, fitted_on, outer_fold=outer_fold)
    with phase("pseudo_outcome"):
        next_outcome = _pseudo_outcome(data, carried, time, cause)
    design = outcome_design(data, plan, time)
    policy_node = plan.is_policy_node(time)
    levels = range(plan.policy_at(time).shape[1] if policy_node else 0)
    continuous_node = policy_node and data.is_continuous_node(time)
    predict_designs = (
        {f"arm_{code}": outcome_design(data, plan, time, arm=code) for code in levels}
        if policy_node
        else {"history": design}
    )
    if continuous_node:
        # A dose has no level to select the observed-arm prediction from, so it is
        # predicted at the observed dose; the branches are predicted at their own doses.
        predict_designs["history"] = design
    learner = outcome_learner if time == horizon else pseudo_learner
    task = "classification" if time == horizon and data.family == "binomial" else "regression"
    constant = constant_target(next_outcome, fitted_on)
    if constant is not None:
        # Every row the regression is fitted on has one target value, such as a grid node at
        # which no follower had the event.  The maximum-likelihood regression is that value,
        # so the node fits no learner, and the node's score is zero at that prediction.
        predictions = {name: np.full(data.n, constant) for name in predict_designs}
        diagnostics: list[SuperLearnerDiagnostics] = []
    else:
        with phase("outcome_learner_fit"):
            predictions, diagnostics = cross_fit_predictions(
                learner,
                design,
                next_outcome,
                data.weights,
                folds,
                task=task,  # type: ignore[arg-type]
                predict_designs=predict_designs,
                fit_mask=fitted_on,
                groups=data.cluster,
                clip=(0.0, 1.0),
                n_jobs=n_jobs,
                # A training fold whose rows hold one class predicts that class.
                constant_folds=[] if task == "classification" else None,
            )
    if not policy_node:
        return _NodeRegression(
            time=time,
            at_risk=at_risk,
            trained_on=trained_on,
            fitted_on=fitted_on,
            pseudo_outcome=next_outcome,
            initial=np.where(at_risk, predictions["history"], _FILLER),
            learner_diagnostics=tuple(diagnostics),
        )
    # One pooled fit, predicted at every level.  The observed-arm prediction is *selected*
    # from the per-arm predictions rather than predicted again, so the offset of the
    # fluctuation and the arm it moves are the same array bit for bit.
    by_arm = np.column_stack(
        [np.where(at_risk, predictions[f"arm_{code}"], _FILLER) for code in levels]
    )
    observed = (
        np.where(at_risk, predictions["history"], _FILLER)
        if continuous_node
        else by_arm[np.arange(data.n), plan.arm(time).astype(np.int64)]
    )
    return _NodeRegression(
        time=time,
        at_risk=at_risk,
        trained_on=trained_on,
        fitted_on=fitted_on,
        pseudo_outcome=next_outcome,
        initial=observed,
        learner_diagnostics=tuple(diagnostics),
        initial_by_arm=by_arm,
        initial_marginal=np.where(at_risk, _carried(plan.policy_at(time), by_arm), _FILLER),
    )


def prepare_node(
    data: LongitudinalData,
    plan: Plan,
    cumulative: FloatArray,
    carried: FloatArray,
    time: int,
    horizon: int,
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    cause: str | None = None,
    masks: RegimenMasks | None = None,
    numerator: FloatArray | None = None,
    n_jobs: int = 1,
) -> NodeInputs:
    """One node's masks, pseudo-outcome, regression and clever covariate.

    Everything the recursion does at a node *except* the fluctuation, which is split out
    because a working model over regimens pools that step across the declared plans and
    so has to hold every plan's regression at a node before any of them is updated.
    :func:`fit_regimen` calls this and fluctuates immediately, which is the recursion it
    always was.

    ``masks`` is this plan's prefix scans, built **once per regimen** by the caller.
    Rebuilding them here would be :math:`O(T^2 n)` over the pass, and on a survival fit
    that again per horizon; they are the same arrays either way, which is what
    ``tests/unit/test_longitudinal_masks.py`` checks and what makes the default -- build
    them for this one node -- a convenience rather than a second code path.

    This is the single-fold node.  A cross-fitted fit runs
    :func:`untargeted_fold_recursions` instead, whose fold regressions narrow the rows
    each regression is fitted on and build no clever covariate.

    ``numerator`` is the plan's :meth:`~cleverly.longitudinal.regimen.Plan.cumulative_numerator`,
    built once per regimen by the caller, as ``masks`` is.  On a plan with a policy node it
    enters both the loss weight and the curve multiplier.  ``None`` builds it here.
    """
    if numerator is None:
        numerator = plan.cumulative_numerator(data)
    regression = _fit_node_regression(
        data,
        plan,
        carried,
        time,
        horizon,
        outcome_learner=outcome_learner,
        pseudo_learner=pseudo_learner,
        folds=folds,
        cause=cause,
        masks=masks,
        n_jobs=n_jobs,
    )
    with phase("clever_covariate"):
        counterfactual, clever = _clever_covariate(
            regression.at_risk,
            regression.trained_on,
            cumulative,
            time,
            numerator=None if numerator is None else numerator[:, time - 1],
        )
    return NodeInputs(
        time=regression.time,
        at_risk=regression.at_risk,
        trained_on=regression.trained_on,
        fitted_on=regression.fitted_on,
        pseudo_outcome=regression.pseudo_outcome,
        initial=regression.initial,
        counterfactual=counterfactual,
        clever=clever,
        learner_diagnostics=regression.learner_diagnostics,
        initial_by_arm=regression.initial_by_arm,
        initial_marginal=regression.initial_marginal,
        policy=plan.policy_at(time) if plan.is_policy_node(time) else None,
    )


def _clever_covariate(
    at_risk: BoolArray,
    trained_on: BoolArray,
    cumulative: FloatArray,
    time: int,
    numerator: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Node ``time``'s inverse-probability loss weight and its follower-masked copy.

    One expression for the single-fold node and the pooled cross-fitted one, so the two
    cannot divide by different things.  Only a cross-fitted fit's *caller* differs: it
    hands in the out-of-fold cumulative prefixes.

    Parameters
    ----------
    at_risk : BoolArray
        Rows whose history at the node is observed and regimen-consistent.
    trained_on : BoolArray
        Rows that followed the regimen through the node.
    cumulative : FloatArray
        ``(n, T)`` bounded cumulative mechanism probabilities.
    time : int
        One-based node index.
    numerator : FloatArray or None
        The plan's cumulative intervention density at the observed arms through ``time``,
        on a plan with a policy node.  ``None`` keeps the numerator one, which is the
        expression a deterministic plan has always had.

    Returns
    -------
    tuple of FloatArray
        ``counterfactual``, which is ``numerator / cumulative[:, time - 1]`` on ``at_risk``
        and zero elsewhere, and ``clever``, which is that weight on ``trained_on`` and zero
        elsewhere.  Only the denominator is bounded.
    """
    denominator = np.where(at_risk, cumulative[:, time - 1], 1.0)
    if numerator is None:
        counterfactual = np.where(at_risk, 1.0 / denominator, 0.0)
    else:
        counterfactual = np.where(at_risk, numerator / denominator, 0.0)
    clever = np.where(trained_on, counterfactual, 0.0)
    return counterfactual, clever


def _fluctuate_node(
    pseudo_outcome: FloatArray,
    initial: FloatArray,
    loss_weights: FloatArray,
    fitted_on: BoolArray,
    *,
    label: str,
    time: int,
    alpha: float,
    max_iter: int,
    tol: float,
    arms: FloatArray | None = None,
) -> Fluctuation:
    """Solve one node's intercept fluctuation with the clever covariate in the loss weight.

    Canonical longitudinal TMLE uses an intercept fluctuation with the cumulative inverse
    probability in the *loss weight* (``ltmle::UpdateQ``), not as the logistic submodel's
    covariate.  Both choices solve :math:`\\sum H (Y - Q^*) = 0`.  They do not produce the
    same finite-sample substitution estimator when ``epsilon`` is nonzero, which is why the
    distinction is explicit here.

    Parameters
    ----------
    pseudo_outcome : FloatArray
        The node's outcome on the ``[0, 1]`` scale.
    initial : FloatArray
        The node's initial predictions, used as the offset.
    loss_weights : FloatArray
        Observation weight times the node's inverse-probability weight.
    fitted_on : BoolArray
        Rows the score is summed over.
    label : str
        Regimen label, used to name the coefficient.
    time : int
        One-based node index, used to name the coefficient.
    alpha : float
        Probability bound of the logistic submodel.
    max_iter : int
        Largest number of Newton iterations.
    tol : float
        Relative-score convergence tolerance.
    arms : FloatArray or None
        ``(n, K_t)`` per-arm initial predictions at a policy node.  Each column is moved by
        the same coefficient as ``initial`` and is filed under :func:`_policy_arm`.  The
        score reads ``initial`` alone, so the columns change no coefficient.

    Returns
    -------
    Fluctuation
        The solved fluctuation.  Its targeted predictions are filed under the regimen key.
    """
    intercept = np.ones((len(pseudo_outcome), 1))
    initial_arms = {_REGIMEN_ARM: initial}
    submodel_arms = {_REGIMEN_ARM: intercept}
    if arms is not None:
        for code in range(arms.shape[1]):
            initial_arms[_policy_arm(code)] = arms[:, code]
            submodel_arms[_policy_arm(code)] = intercept
    constant = constant_target(pseudo_outcome, fitted_on)
    if constant is not None and np.all(np.asarray(initial)[fitted_on] == constant):
        # The regression already equals its one target value on every row the score reads,
        # so the score is zero at epsilon = 0.  The logistic solver would first shrink a
        # prediction of 0 or 1 into the bounds and then chase it, so it is not called.
        names = (f"epsilon[{label}, t={time}]",)
        zero = np.zeros(1)
        return Fluctuation(
            epsilon=zero,
            targeted=InitialFit(initial, initial_arms),
            score=zero,
            converged=True,
            n_iter=0,
            trace=(0.0,),
            method="iterative",
            names=names,
            score_initial=zero,
        )
    return solve_fluctuation(
        pseudo_outcome,
        InitialFit(initial, initial_arms),
        Submodel(
            intercept,
            submodel_arms,
            (f"epsilon[{label}, t={time}]",),
            "sequential",
        ),
        loss_weights,
        fitted_on,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )


def _targeted_policy(
    fluctuation: Fluctuation, policy: FloatArray, at_risk: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """The targeted per-arm predictions of a policy node and their policy-weighted mean.

    Parameters
    ----------
    fluctuation : Fluctuation
        The node's solve, with one :func:`_policy_arm` key per level.
    policy : FloatArray
        ``(n, K_t)`` policy density at the node.
    at_risk : BoolArray
        The rows the node predicts for.

    Returns
    -------
    tuple of FloatArray
        ``(n, K_t)`` targeted per-arm predictions, and the marginal, filled with ``0.5``
        off ``at_risk``.
    """
    by_arm = np.column_stack(
        [fluctuation.targeted.arms[_policy_arm(code)] for code in range(policy.shape[1])]
    )
    return by_arm, np.where(at_risk, _carried(policy, by_arm), _FILLER)


def _pseudo_outcome(
    data: LongitudinalData, carried: FloatArray, time: int, cause: str | None
) -> FloatArray:
    """What node ``time`` regresses: the carried prediction, composed with the event."""
    if data.is_survival:
        # The numerator is *this* cause's event and the survival factor is
        # **all-cause**: a unit that left through a competing cause contributes a zero
        # here and carries nothing forward, because it is not going to have this
        # cause's event either.  Writing ``1 - event_by(time, cause)`` instead -- the
        # cause's own survival -- is the mistake competing risks invite, and it is
        # wrong by exactly the mass that left through the other causes.  With one
        # cause the two calls return the same array and this is the line it was.
        failed = data.event_by(time, cause)
        return np.asarray(failed + (1.0 - data.event_by(time)) * carried)
    return carried


def constant_target(target: FloatArray, fitted_on: BoolArray) -> float | None:
    """The one value ``target`` takes on ``fitted_on``, or ``None`` when it varies.

    A node whose regression rows hold one value, such as a grid node at which no follower
    had the event, has that value as its maximum-likelihood regression and fits no learner.
    The nuisance report shows ``LONGITUDINAL_CONSTANT_TARGET`` in place of its row.
    """
    values = np.unique(np.asarray(target, dtype=float)[np.asarray(fitted_on, dtype=bool)])
    return float(values[0]) if values.size == 1 else None


def _finish_regimen_fit(
    data: LongitudinalData,
    plan: Plan,
    steps: Sequence[SequentialStep],
    cumulative_unbounded: FloatArray,
    cumulative: FloatArray,
    *,
    horizon: int,
    cause: str | None,
    numerator: FloatArray | None = None,
) -> RegimenFit:
    """Assemble one regimen estimate and influence curve from its targeted steps."""
    retained = tuple(steps)
    # Every unit is at risk at the first node, so this averages predictions rather than
    # fillers. ``np.average`` against mean-one weights is bit-for-bit the old unweighted
    # mean when the weights are constant.
    # ``value`` is the carried prediction: the targeted one, or at a policy node the
    # policy-weighted mean of the targeted per-arm predictions.  The residual below reads
    # ``targeted``, the prediction at the arm the row took.
    psi = float(np.average(retained[0].value, weights=data.weights))
    influence = retained[0].value - psi
    for step in retained:
        # The t-th term reads the target the t-th regression was fitted to: the later
        # targeted prediction, or that prediction composed with this node's event.
        influence = influence + step.clever * (step.pseudo_outcome - step.targeted)
    # The weighted EIF is (w / E[w]) D*(P_w). The weight multiplies the whole centred
    # curve, not only its residual terms.
    influence = data.weights * influence
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=influence,
        horizon=horizon,
        cause=cause,
        steps=retained,
        cumulative_unbounded=cumulative_unbounded,
        cumulative=cumulative,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
        policy=plan.policy,
        cumulative_numerator=numerator,
    )


def node_ratios(data: LongitudinalData, plan: Plan, mechanism: Mechanism) -> FloatArray | None:
    """``(n, T)`` raw per-node ratio of a plan with a modified treatment policy, else ``None``.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan, with its policy numerators attached.
    mechanism : Mechanism
        The fitted mechanism.

    Returns
    -------
    FloatArray or None
        The node's intervention density over its unbounded treatment probability at the
        observed arm, zero where that probability is zero.
    """
    if not plan.has_mtp:
        return None
    density = plan.intervention_density(data)
    columns = []
    for time in range(1, data.n_times + 1):
        denominator = np.asarray(mechanism.treatment[time - 1][plan.label], dtype=float)
        safe = np.where(denominator > 0.0, denominator, 1.0)
        columns.append(np.where(denominator > 0.0, density[:, time - 1] / safe, 0.0))
    return np.column_stack(columns)


def fit_regimen(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    horizon: int | None = None,
    cause: str | None = None,
    alpha: float = 0.9995,
    max_iter: int = 20,
    tol: float = 1e-10,
    n_jobs: int = 1,
) -> RegimenFit:
    """Run the backward recursion for one regimen, targeting at every node.

    ``outcome_learner`` regresses the outcome itself at the last node and
    ``pseudo_learner`` the ``[0, 1]``-valued predictions at every earlier one.  They are
    separate arguments because the two regressions have different *types*: a binary
    outcome is a classification problem, and the pseudo-outcome that replaces it one node
    earlier never is, whatever the outcome's family.

    ``horizon`` says which node the parameter is indexed by.  On an end-of-study fit it
    is ``T`` and the recursion is the one Bang & Robins wrote down.  On a survival fit it
    is the horizon of a cumulative risk, the recursion starts there rather than at ``T``,
    and the pseudo-outcome carried back is composed with that node's event indicator:
    a unit that had the event contributes a one and a unit that did not contributes the
    later node's targeted prediction.  Seeding :math:`\\bar{Q}_{k+1} = 0` makes the two
    statements one, since at ``k`` the composition is exactly :math:`Y_k`.

    ``cause`` names which absorbing state the parameter is the incidence *of*, on a fit
    that declared competing risks, and is ``None`` when there is one.  It changes the
    pseudo-outcome and nothing else: the masks, the mechanism and the clever covariate are
    all-cause, because a competing event is part of the history rather than a node anyone
    intervenes on.  So the causes share every nuisance fit and differ only in what is
    regressed, which is also why a curve per cause costs ``J`` backward passes and one
    mechanism rather than ``J`` of each.

    Observation weights reach every regression, the fluctuation's score and the plug-in,
    and multiply the returned curve row-wise -- the module docstring says what that is and
    what it is not.
    """
    horizon = data.n_times if horizon is None else horizon
    if not 1 <= horizon <= data.n_times:
        raise LongitudinalError(f"horizon {horizon} is outside 1..{data.n_times}")
    if not folds.is_single:
        return _fit_regimen_crossfit(
            data,
            plan,
            mechanism,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=folds,
            scaler=scaler,
            g_bounds=g_bounds,
            horizon=horizon,
            cause=cause,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
        )
    cumulative_unbounded, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    carried = seed_carried(data, scaler)
    # Once per regimen, not once per node: the masks are prefix scans of one conjunction,
    # and rebuilding them at every node is what made this pass quadratic in T.
    masks = plan.masks(data)
    numerator = plan.cumulative_numerator(data)

    steps: list[SequentialStep] = []
    for time in range(horizon, 0, -1):
        node = prepare_node(
            data,
            plan,
            cumulative,
            carried,
            time,
            horizon,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=folds,
            cause=cause,
            masks=masks,
            numerator=numerator,
            n_jobs=n_jobs,
        )
        with phase("fluctuation"):
            fluctuation = _fluctuate_node(
                node.pseudo_outcome,
                node.initial,
                data.weights * node.counterfactual,
                node.fitted_on,
                label=plan.label,
                time=time,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                arms=node.initial_by_arm,
            )
        step = _targeted_step(node, fluctuation, regression_target=None)
        steps.append(step)
        carried = np.where(node.at_risk, step.value, _FILLER)

    steps.reverse()
    return _finish_regimen_fit(
        data,
        plan,
        steps,
        cumulative_unbounded,
        cumulative,
        horizon=horizon,
        cause=cause,
        numerator=numerator,
    )


def _targeted_step(
    node: NodeInputs, fluctuation: Fluctuation, *, regression_target: FloatArray | None
) -> SequentialStep:
    """One retained step from a node's inputs and its solved fluctuation.

    The one place a step is assembled for the per-regimen recursion, single-fold and
    pooled alike.  At a policy node it also holds the targeted per-arm predictions and their
    policy-weighted mean, which :attr:`SequentialStep.value` then returns.
    """
    targeted = fluctuation.targeted.arms[_REGIMEN_ARM]
    by_arm: FloatArray | None = None
    marginal: FloatArray | None = None
    if node.policy is not None:
        by_arm, marginal = _targeted_policy(fluctuation, node.policy, node.at_risk)
    return SequentialStep(
        time=node.time,
        trained_on=node.trained_on,
        at_risk=node.at_risk,
        pseudo_outcome=node.pseudo_outcome,
        initial=node.initial,
        targeted=targeted,
        clever=node.clever,
        fluctuation=fluctuation,
        learner_diagnostics=node.learner_diagnostics,
        regression_target=regression_target,
        marginal=marginal,
        targeted_by_arm=by_arm,
        initial_by_arm=node.initial_by_arm,
    )


@dataclass(frozen=True)
class FoldRecursionCell:
    """One backward recursion that every outer fold runs: a regimen, to one horizon.

    Parameters
    ----------
    plan : Plan
        The resolved regimen.
    masks : RegimenMasks
        The regimen's prefix scans, built once by the caller.
    horizon : int
        The node the recursion starts from.
    """

    plan: Plan
    masks: RegimenMasks
    horizon: int


@dataclass(frozen=True)
class StitchedInitial:
    """The untargeted out-of-fold recursion of one cell, stitched from the held-out rows.

    Parameters
    ----------
    initial : dict of int to FloatArray
        By node, row ``i``'s prediction from the fold that held row ``i`` out.  Filled with
        ``0.5`` off the node's at-risk rows.
    regression_target : dict of int to FloatArray
        By node, the target that row ``i``'s held-out fold composed from its own untargeted
        prediction at the later node.
    diagnostics : dict of int to tuple of SuperLearnerDiagnostics
        By node, the learner diagnostics of every fold's regression, in fold order.
    initial_by_arm : dict of int to FloatArray
        At each policy node, row ``i``'s ``(K_t,)`` per-arm predictions from the fold that
        held row ``i`` out.  ``initial`` at that node is the observed-arm column of it.
        Empty on a plan without a policy node.
    """

    initial: dict[int, FloatArray]
    regression_target: dict[int, FloatArray]
    diagnostics: dict[int, tuple[SuperLearnerDiagnostics, ...]]
    initial_by_arm: dict[int, FloatArray] = field(default_factory=dict)


_FoldOutputs = dict[
    int,
    tuple[FloatArray, FloatArray, tuple[SuperLearnerDiagnostics, ...], FloatArray | None],
]


def _untargeted_recursion_in_fold(
    data: LongitudinalData,
    cell: FoldRecursionCell,
    *,
    scaler: OutcomeScaler,
    cause: str | None,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    outer_train: BoolArray,
    fold: int,
    test: IntArray,
) -> _FoldOutputs:
    """Run one cell's untargeted recursion on one fold's training rows.

    The node regression is :func:`_fit_node_regression` with the fold's training complement
    as ``fit_rows`` and a one-fold split.  That is a fit on the named rows and a prediction
    everywhere, which is what an outer fold's model is.  So both refusal hints, and every
    mask, are the ones the single-fold pass uses.  Only the held-out slices are returned.
    """
    inner = Folds.single(data.n)
    outputs: _FoldOutputs = {}
    carried = seed_carried(data, scaler)
    for time in range(cell.horizon, 0, -1):
        node = _fit_node_regression(
            data,
            cell.plan,
            carried,
            time,
            cell.horizon,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=inner,
            cause=cause,
            masks=cell.masks,
            fit_rows=outer_train,
            outer_fold=fold,
        )
        outputs[time] = (
            node.initial[test],
            node.pseudo_outcome[test],
            node.learner_diagnostics,
            None if node.initial_by_arm is None else node.initial_by_arm[test],
        )
        # The fold's *untargeted* prediction is what the earlier node regresses.  Steps 1-4
        # of the Section 5.2 construction carry the untargeted prediction and target only in
        # the pooled pass.  Targeting here would make the fold's earlier regressions read a
        # fluctuation fitted on the fold's own training rows.  At a policy node the untargeted
        # prediction carried is the fold's policy-weighted mean of its per-arm predictions.
        untargeted = node.initial if node.initial_marginal is None else node.initial_marginal
        carried = np.where(node.at_risk, untargeted, _FILLER)
    return outputs


def untargeted_fold_recursions(
    data: LongitudinalData,
    cells: Sequence[FoldRecursionCell],
    *,
    folds: Folds,
    scaler: OutcomeScaler,
    cause: str | None,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    n_jobs: int = 1,
) -> list[StitchedInitial]:
    """Run every cell's untargeted recursion in every outer fold, and stitch the held-out rows.

    Step 1 of the cross-fitted construction of Díaz, Williams, Hoffman and Schenck (2023,
    *JASA* 118(542), Section 5.2).  There is one job per outer fold, and each job runs the
    recursion of **every** cell on that fold's training rows.  So a fit over ``C`` cells
    still fans out ``K`` jobs and not ``K * C``.  The cells do not interact inside a fold:
    no fold solves a fluctuation, and no fold reads another cell's predictions.  So cell
    ``c``'s stitched arrays are the ones a call with ``c`` alone returns.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    cells : sequence of FoldRecursionCell
        The recursions to run, in the order the result lists them.
    folds : Folds
        The realized outer split, with at least two folds.
    scaler : OutcomeScaler
        The outcome transformation.
    cause : str or None
        The absorbing cause, on a competing-risk fit.
    outcome_learner : Learner
        The regression at each cell's horizon.
    pseudo_learner : Learner
        The regression at every earlier node.
    n_jobs : int, default=1
        Parallel workers across the outer folds.

    Returns
    -------
    list of StitchedInitial
        One per cell, in the order of ``cells``.
    """
    # Read once, in the parent: a worker process has no collector of its own and so
    # cannot tell whether anybody asked for a profile.
    wanted = profiling()

    def run_fold(
        fold: int, train: IntArray, test: IntArray
    ) -> tuple[IntArray, list[_FoldOutputs], PhaseProfile | None]:
        outer_train = np.zeros(data.n, dtype=bool)
        outer_train[train] = True
        with collect_phases(wanted) as profile:
            outputs = [
                _untargeted_recursion_in_fold(
                    data,
                    cell,
                    scaler=scaler,
                    cause=cause,
                    outcome_learner=outcome_learner,
                    pseudo_learner=pseudo_learner,
                    outer_train=outer_train,
                    fold=fold,
                    test=test,
                )
                for cell in cells
            ]
        return test, outputs, profile

    jobs = [(fold, train, test) for fold, (train, test) in enumerate(folds)]
    # One parent phase over the whole fan-out, with the workers' phases merged underneath
    # it rather than into it.  Adding worker time to the parent's own totals would break
    # `sum(exclusive) <= total_seconds`: K folds running at once accumulate more processor
    # time than the parent spent waiting for them.
    with phase("outer_fold_recursion"):
        outcomes = map_parallel(run_fold, jobs, n_jobs=n_jobs)
    for _, _, profile in outcomes:
        merge_worker_phases(profile)
    stitched: list[StitchedInitial] = []
    for position, cell in enumerate(cells):
        nodes = range(1, cell.horizon + 1)
        initial = {time: np.full(data.n, _FILLER, dtype=float) for time in nodes}
        regression_target = {time: np.full(data.n, _FILLER, dtype=float) for time in nodes}
        diagnostics: dict[int, list[SuperLearnerDiagnostics]] = {time: [] for time in nodes}
        by_arm = {
            time: np.full((data.n, cell.plan.policy_at(time).shape[1]), _FILLER, dtype=float)
            for time in nodes
            if cell.plan.is_policy_node(time)
        }
        for test, outputs, _ in outcomes:
            for time, (held_out, target, fold_diagnostics, arms) in outputs[position].items():
                initial[time][test] = held_out
                regression_target[time][test] = target
                diagnostics[time].extend(fold_diagnostics)
                if arms is not None:
                    by_arm[time][test] = arms
        stitched.append(
            StitchedInitial(
                initial=initial,
                regression_target=regression_target,
                diagnostics={time: tuple(values) for time, values in diagnostics.items()},
                initial_by_arm=by_arm,
            )
        )
    return stitched


def _fit_regimen_crossfit(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    horizon: int,
    cause: str | None,
    alpha: float,
    max_iter: int,
    tol: float,
    n_jobs: int,
) -> RegimenFit:
    """Run untargeted fold recursions, stitch them, and target once per node over all rows.

    This is the cross-fitted construction of Díaz, Williams, Hoffman and Schenck (2023,
    *JASA* 118(542), Section 5.2, Steps 1-4).  Each outer fold runs a complete *untargeted*
    backward recursion on its training complement: node ``t - 1`` of fold ``k`` regresses
    fold ``k``'s own untargeted prediction at node ``t``.  So fold ``k``'s initial nuisance
    fits are fixed given its training rows, which is the conditional-independence step the
    proof of the paper's Theorem 3 uses.  No fold solves a fluctuation.  The parent stitches
    each fold's held-out predictions into one out-of-fold initial estimate per node, and
    :func:`_pooled_targeting` then solves one fluctuation per node over every follower.

    The fold recursions are :func:`untargeted_fold_recursions`, which the cross-fitted
    working model in :mod:`cleverly.longitudinal.msm` runs over every cell at once.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved regimen.
    mechanism : Mechanism
        The out-of-fold mechanism fit.
    outcome_learner : Learner
        The regression at the horizon.
    pseudo_learner : Learner
        The regression at every earlier node.
    folds : Folds
        The realized outer split, with at least two folds.
    scaler : OutcomeScaler
        The outcome transformation.
    g_bounds : tuple of float
        The cumulative mechanism bounds.
    horizon : int
        The node the parameter is indexed by.
    cause : str or None
        The absorbing cause, on a competing-risk fit.
    alpha : float
        Probability bound of the logistic submodel.
    max_iter : int
        Largest number of Newton iterations per node.
    tol : float
        Relative-score convergence tolerance.
    n_jobs : int
        Parallel workers across the outer folds.

    Returns
    -------
    RegimenFit
        The pooled-targeted fit.
    """
    # The out-of-fold pair is the only mechanism this fit divides by.
    cumulative_unbounded, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    with phase("mask_construction"):
        masks = plan.masks(data)
    (stitched,) = untargeted_fold_recursions(
        data,
        (FoldRecursionCell(plan, masks, horizon),),
        folds=folds,
        scaler=scaler,
        cause=cause,
        outcome_learner=outcome_learner,
        pseudo_learner=pseudo_learner,
        n_jobs=n_jobs,
    )
    numerator = plan.cumulative_numerator(data)
    steps = _pooled_targeting(
        data,
        plan,
        masks,
        cumulative,
        stitched,
        numerator=numerator,
        scaler=scaler,
        horizon=horizon,
        cause=cause,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )
    return _finish_regimen_fit(
        data,
        plan,
        steps,
        cumulative_unbounded,
        cumulative,
        horizon=horizon,
        cause=cause,
        numerator=numerator,
    )


def _pooled_targeting(
    data: LongitudinalData,
    plan: Plan,
    masks: RegimenMasks,
    cumulative: FloatArray,
    stitched: StitchedInitial,
    *,
    numerator: FloatArray | None,
    scaler: OutcomeScaler,
    horizon: int,
    cause: str | None,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[SequentialStep, ...]:
    r"""Solve one fluctuation per node over every follower, from the horizon backwards.

    Steps 2-4 of the Section 5.2 construction that :func:`_fit_regimen_crossfit` follows.
    Node ``t``'s fluctuation takes the stitched out-of-fold prediction as its offset.  Its
    outcome is :func:`_pseudo_outcome` applied to the pooled targeted prediction of node
    ``t + 1``, so a residual left at one node is regressed away at the next, as on a
    single-fold fit.  Its loss weight is the
    observation weight times the out-of-fold inverse cumulative mechanism.  The score is
    summed over every follower, so each node solves
    :math:`\sum_i w_i h_t(i) (Z_t(i) - \bar Q^*_t(i)) = 0` exactly, and the fit solves
    :math:`P_n D^* = 0` as a single-fold fit does.

    At a policy node the fluctuation also moves the stitched per-arm predictions by the same
    coefficient, and the node carries their policy-weighted mean.  The loss weight then
    carries the policy numerator.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved regimen.
    masks : RegimenMasks
        The regimen's prefix scans.
    cumulative : FloatArray
        ``(n, T)`` bounded out-of-fold cumulative mechanism probabilities.
    stitched : StitchedInitial
        The stitched out-of-fold initial predictions, regression targets and diagnostics.
    numerator : FloatArray or None
        The plan's cumulative intervention density, ``None`` without a policy node.
    scaler : OutcomeScaler
        The outcome transformation.
    horizon : int
        The node the parameter is indexed by.
    cause : str or None
        The absorbing cause, on a competing-risk fit.
    alpha : float
        Probability bound of the logistic submodel.
    max_iter : int
        Largest number of Newton iterations per node.
    tol : float
        Relative-score convergence tolerance.

    Returns
    -------
    tuple of SequentialStep
        One step per node, in time order.
    """
    steps: list[SequentialStep] = []
    carried = seed_carried(data, scaler)
    for time in range(horizon, 0, -1):
        node = pooled_node_inputs(
            data, plan, stitched, masks, cumulative, numerator, carried, time, cause
        )
        # One phase entry per node, as on a single-fold fit, and always in the parent.
        with phase("fluctuation"):
            fluctuation = _fluctuate_node(
                node.pseudo_outcome,
                node.initial,
                data.weights * node.counterfactual,
                node.fitted_on,
                label=plan.label,
                time=time,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                arms=node.initial_by_arm,
            )
        step = _targeted_step(node, fluctuation, regression_target=stitched.regression_target[time])
        steps.append(step)
        carried = np.where(node.at_risk, step.value, _FILLER)
    steps.reverse()
    return tuple(steps)


def pooled_node_inputs(
    data: LongitudinalData,
    plan: Plan,
    stitched: StitchedInitial,
    masks: RegimenMasks,
    cumulative: FloatArray,
    numerator: FloatArray | None,
    carried: FloatArray,
    time: int,
    cause: str | None,
) -> NodeInputs:
    """One node's inputs for the pooled update, over the stitched initial estimate.

    The pseudo-outcome composes the pooled targeted prediction of the later node, and the
    clever covariate divides by the out-of-fold cumulative mechanism.  The per-regimen
    pooled update and the cross-fitted working model both read this, so the two paths
    cannot divide by different things.  The score set is every follower.  At a policy node
    the inputs carry the stitched per-arm predictions and the policy.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan.
    stitched : StitchedInitial
        The plan's stitched out-of-fold recursion.
    masks : RegimenMasks
        The plan's prefix scans.
    cumulative : FloatArray
        ``(n, T)`` bounded out-of-fold cumulative mechanism probabilities.
    numerator : FloatArray or None
        The plan's cumulative intervention density, ``None`` without a policy node.
    carried : FloatArray
        The pooled targeted value of the later node.
    time : int
        The node, counted from one.
    cause : str or None
        The absorbing cause, on a competing-risk fit.

    Returns
    -------
    NodeInputs
        The node's inputs, with ``fitted_on`` equal to ``trained_on``.
    """
    at_risk = masks.at_risk(time)
    following = masks.following(time)
    with phase("pseudo_outcome"):
        pseudo_outcome = _pseudo_outcome(data, carried, time, cause)
    with phase("clever_covariate"):
        counterfactual, clever = _clever_covariate(
            at_risk,
            following,
            cumulative,
            time,
            numerator=None if numerator is None else numerator[:, time - 1],
        )
    policy_node = plan.is_policy_node(time)
    return NodeInputs(
        time=time,
        at_risk=at_risk,
        trained_on=following,
        fitted_on=following,
        pseudo_outcome=pseudo_outcome,
        initial=stitched.initial[time],
        counterfactual=counterfactual,
        clever=clever,
        learner_diagnostics=stitched.diagnostics[time],
        initial_by_arm=stitched.initial_by_arm[time] if policy_node else None,
        policy=plan.policy_at(time) if policy_node else None,
    )


def _risk_set_hint(data: LongitudinalData, plan: Plan, time: int) -> str:
    """On a survival fit, whether the risk set emptied because everybody had the event.

    "Nobody followed the regimen this far" and "everybody who did had already had the
    event" are different diagnoses -- the first is a positivity failure and the second is
    the study running out of people to observe -- and a single message covering both
    would send a reader to the wrong place.
    """
    if not data.is_survival:
        return ""
    masks = plan.masks(data)
    reached = masks.uncensored[:, time - 1] & masks.followed[:, time - 1]
    if not reached.any():
        return ""
    failed = int(np.sum(reached & ~data.event_free_through(time - 1)))
    if failed < int(reached.sum()):
        return ""
    return (
        f" All {failed} unit(s) that reached time {time} on this regimen had already had "
        "the event, so the risk set is empty rather than unsupported: the curve is "
        f"estimable only up to a horizon before {time}."
    )


def _rule_hint(plan: Plan, data: LongitudinalData, at_risk: BoolArray, time: int) -> str:
    """For an unsupported regimen, what the rule asked for where nobody was left.

    A static plan is unsupported because the sample happens not to contain the sequence.
    A rule can be unsupported because it asks for an arm nobody at risk received, and
    that is a different diagnosis -- so say which arms it wanted and how many units were
    there to give them to.

    Written in the node's *labels* rather than its dense codes.  Counting the rows whose
    code is ``1`` and calling them "arm 1" answers about whichever label sorts second,
    which on a three-armed node is a different arm from the one the reader is asking
    about and on a two-armed node reads as though only one arm existed.
    """
    if isinstance(plan.regimen, Regimen):
        return ""
    if not plan.regimen.is_rule(time):
        return ""
    assigned = plan.arm(time)[at_risk]
    levels = data.treatment_levels[time - 1]
    counts = ", ".join(
        f"{level!r} to {int(np.sum(assigned == float(code)))}" for code, level in enumerate(levels)
    )
    return (
        f" The rule at time {time} assigned {counts} of the "
        f"{int(at_risk.sum())} unit(s) at risk there."
    )
