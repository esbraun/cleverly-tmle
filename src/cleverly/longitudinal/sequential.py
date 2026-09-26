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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.weighting import effective_sample_size
from ..estimators._nuisance import cross_fit_companion, cross_fit_predictions
from ..exceptions import ConvergenceWarning, LongitudinalError
from ..fluctuation._score import score_columns, score_scale
from ..fluctuation.iterative import (
    Fluctuation,
    FoldFluctuation,
    InitialFit,
    TargetingFailure,
    TargetingLabel,
    dominant_failure,
    solve_fluctuation,
)
from ..fluctuation.submodel import Submodel
from ..learners.crossfit import Folds
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
    "Mechanism",
    "NodeInputs",
    "RegimenFit",
    "SequentialStep",
    "fit_mechanism",
    "fit_regimen",
    "preflight_mechanism_support",
    "preflight_terminal_outcomes",
    "prepare_node",
    "seed_carried",
]


#: Filler for a prediction at a row the estimator never reads -- a unit censored before
#: the node in question.  Any finite number in ``(0, 1)`` would do; a half keeps ``logit``
#: at zero, so a filled row cannot make a Newton step look large.
_FILLER = 0.5

#: What a refusal about one outer training fold may offer.  The split is drawn from the
#: seed alone and reads no treatment, outcome or covariate, so a fold count or a seed
#: found by trying them until one fits would choose the partition by the values it must
#: not read.  ``n_folds=1`` is the in-sample fit, which draws no split at all.
_CROSS_FIT_NODE_REMEDY = (
    "The split reads none of the data, so trying fold counts or seeds until one fits "
    "would choose the partition by the values it must not read. Fit in sample "
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
    with the earlier treatments set to what the regimen would have assigned, and
    ``censoring[t][label]`` is :math:`P(C_t = 1 \\mid H_t, \\bar A_t = \\bar a_t)`.  One
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
    treatment_by_fold : tuple of dict of str to FloatArray
        Treatment predictions from each outer-fold model on all rows.
    censoring_by_fold : tuple of dict of str to FloatArray
        Retention predictions from each outer-fold model on all rows.
    treatment_observed : tuple of FloatArray
        Out-of-fold treatment probability matrices at the observed histories.
    censoring_observed : tuple of FloatArray
        Out-of-fold retention probabilities at the observed treatment histories.
    treatment_diagnostics : tuple of tuple of SuperLearnerDiagnostics
        Learner diagnostics from each treatment node and fitted fold.
    censoring_diagnostics : tuple of tuple of SuperLearnerDiagnostics
        Learner diagnostics from each censoring node and fitted fold.
    """

    treatment: tuple[dict[str, FloatArray], ...]
    censoring: tuple[dict[str, FloatArray], ...]
    #: Predictions from each outer-fold model on every row.  Entry ``[t][label]`` has
    #: shape ``(K, n)``.  Only the engine-level cross-fitted working-model path
    #: (:mod:`cleverly.longitudinal.msm`) reads slab ``k``, for both the training and
    #: held-out rows of outer fold ``k``.  A per-regimen fit reads the out-of-fold
    #: probabilities above.  Empty only on a hand-built mechanism.
    treatment_by_fold: tuple[dict[str, FloatArray], ...] = ()
    censoring_by_fold: tuple[dict[str, FloatArray], ...] = ()
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
        *,
        fold: int | None = None,
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
            if fold is not None:
                # Both factors, not just the one this law happens to use: a mechanism
                # missing only its censoring slabs would sail through a treatment-only
                # check and then raise `IndexError` from inside the product below, which
                # names neither the cause nor the repair.
                if not self.treatment_by_fold or not self.censoring_by_fold:
                    raise LongitudinalError(
                        "this mechanism has no outer-fold prediction slabs; refit it before "
                        "using fold-specific longitudinal targeting"
                    )
                treatment = self.treatment_by_fold[time - 1][plan.label][fold]
                censoring = self.censoring_by_fold[time - 1][plan.label][fold]
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
    and score.  The logistic submodel itself is an intercept shift; putting ``clever`` in
    that submodel instead would solve the same score along a different path and cease to
    match the loss-weighted update in canonical R ``ltmle``.

    Parameters
    ----------
    time : int
        One-based node index.
    at_risk : BoolArray
        Rows whose histories remain observed and regimen-consistent at this node.
    trained_on : BoolArray
        Rows that followed the regimen through this node.
    fitted_on : BoolArray
        Rows this node's regression was fitted on. With one fold, the fluctuation
        also solves over these rows. With two or more folds, they are the followers
        in one fold's training complement and fit that fold's regression only. A
        per-regimen fit runs no fluctuation on them. The working-model path still
        solves its fold fluctuation over them.
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
    """

    time: int
    at_risk: BoolArray
    #: Every row that followed the regimen through this node.  ``clever`` is nonzero
    #: exactly here, whichever rows the regression was fitted on.
    trained_on: BoolArray
    #: The rows the regression was actually fitted on: ``trained_on``, and under outer
    #: cross-fitting that intersected with the fold's training complement.  A single-fold
    #: fit also solves its fluctuation over this mask.  A cross-fitted per-regimen fit
    #: does not fluctuate inside a fold: its one pooled fluctuation per node solves over
    #: every follower, against the stitched out-of-fold predictions.  The engine-level
    #: cross-fitted working-model path still solves each fold's fluctuation over this mask.
    fitted_on: BoolArray
    pseudo_outcome: FloatArray
    initial: FloatArray
    counterfactual: FloatArray
    clever: FloatArray
    #: Super Learner diagnostics from this node's regression, one per fitted fold.
    learner_diagnostics: tuple[SuperLearnerDiagnostics, ...] = ()


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
        Node prediction after targeting.
    clever : FloatArray
        Cumulative inverse-probability multiplier on regimen followers.
    fluctuation : Fluctuation
        Targeting solve retained for this node.
    learner_diagnostics : tuple of SuperLearnerDiagnostics
        Learner diagnostics from each fitted fold of the node regression.
    regression_target : FloatArray or None
        Target the node regression was fitted to, when it differs from ``pseudo_outcome``.
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

    @property
    def n_trained(self) -> int:
        return int(self.trained_on.sum())


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
    #: on a cross-fitted fit and a single-fold one alike.  A diagnostic that wants how much
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
        at_risk = fit_masks.uncensored[:, time - 1] & fit_masks.event_free[:, time - 1]
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        classes = tuple(float(code) for code in range(len(data.treatment_levels[time - 1])))
        _check_categorical_fold_support(
            arm,
            at_risk,
            folds,
            classes,
            data.treatment_levels[time - 1],
            data.treatment_names[time - 1],
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
    treatment_by_fold: list[dict[str, FloatArray]] = []
    censoring_by_fold: list[dict[str, FloatArray]] = []
    treatment_observed: list[FloatArray] = []
    censoring_observed: list[FloatArray] = []
    treatment_diagnostics: list[tuple[SuperLearnerDiagnostics, ...]] = []
    censoring_diagnostics: list[tuple[SuperLearnerDiagnostics, ...]] = []
    # Neither factor depends on a regimen, so one scan serves every node.  `followed` is
    # unused here and the all-true assignment makes that explicit rather than implicit.
    with phase("mask_construction"):
        fit_masks = data.regimen_masks(data.treatment)
    preflight_mechanism_support(data, fit_masks, folds)
    for time in range(1, data.n_times + 1):
        at_risk = fit_masks.uncensored[:, time - 1] & fit_masks.event_free[:, time - 1]
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        designs = {plan.label: data.history_design(time, treatment=plan.values) for plan in plans}
        observed_key = _internal_prediction_key(tuple(designs), "observed_treatment")
        prediction_designs = {**designs, observed_key: data.history_design(time)}
        with phase("mechanism_fit"):
            classes = tuple(float(code) for code in range(len(data.treatment_levels[time - 1])))
            probabilities, companion, diagnostics = cross_fit_companion(
                treatment_learner,
                data.history_design(time),
                arm,
                data.weights,
                folds,
                task="classification",
                predict_designs=prediction_designs,
                companion_designs=designs,
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
        fold_probabilities = {
            plan.label: companion[plan.label][:, rows, plan.arm(time).astype(np.int64)]
            for plan in plans
        }
        # Predicting one validation slice and predicting all rows can differ in the last
        # bit for a BLAS-backed learner.  The production OOF value is authoritative on the
        # held-out rows; the companion slab exists for the training complement around it.
        for fold, (_, test) in enumerate(folds):
            for plan in plans:
                fold_probabilities[plan.label][fold, test] = treatment[-1][plan.label][test]
        treatment_by_fold.append(fold_probabilities)

        if not data.censoring_names:
            censoring.append({plan.label: np.ones(data.n) for plan in plans})
            censoring_by_fold.append(
                {plan.label: np.ones((folds.n_folds, data.n), dtype=float) for plan in plans}
            )
            continue
        stayed = np.where(at_risk, data.uncensored[:, time - 1].astype(float), 0.0)
        censor_designs = {
            plan.label: data.history_design(time, treatment=plan.values, include_current=True)
            for plan in plans
        }
        censor_observed_key = _internal_prediction_key(tuple(censor_designs), "observed_censoring")
        censor_prediction_designs = {
            **censor_designs,
            censor_observed_key: data.history_design(time, include_current=True),
        }
        with phase("mechanism_fit"):
            predictions, censor_companion, diagnostics = cross_fit_companion(
                censoring_learner,
                data.history_design(time, include_current=True),
                stayed,
                data.weights,
                folds,
                task="classification",
                predict_designs=censor_prediction_designs,
                companion_designs=censor_designs,
                fit_mask=at_risk,
                groups=data.cluster,
                clip=(0.0, 1.0),
                n_jobs=n_jobs,
            )
        censoring_observed.append(np.asarray(predictions.pop(censor_observed_key), dtype=float))
        censoring_diagnostics.append(tuple(diagnostics))
        censoring.append(predictions)
        for fold, (_, test) in enumerate(folds):
            for plan in plans:
                censor_companion[plan.label][fold, test] = predictions[plan.label][test]
        censoring_by_fold.append(censor_companion)
    return Mechanism(
        tuple(treatment),
        tuple(censoring),
        tuple(treatment_by_fold),
        tuple(censoring_by_fold),
        tuple(treatment_observed),
        tuple(censoring_observed),
        tuple(treatment_diagnostics),
        tuple(censoring_diagnostics),
    )


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
    scaler: OutcomeScaler,
) -> None:
    """Check terminal classification regressions before any mechanism learner fits.

    The target at a reported horizon is the observed outcome or event indicator. It does
    not depend on a fitted nuisance or an earlier targeting step, so its support can be
    checked for every regimen, cause, horizon, and outer training complement now. The
    masks and target are the same ones :func:`prepare_node` uses at that horizon.

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
    scaler : OutcomeScaler
        The outcome transformation used by the recursion.
    """
    if data.family != "binomial":
        return
    carried = seed_carried(data, scaler)
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    for plan in plans:
        masks = data.regimen_masks(plan.values)
        for horizon in horizons:
            at_risk = masks.at_risk(horizon)
            followers = masks.following(horizon)
            for cause in causes:
                target = _pseudo_outcome(data, carried, horizon, cause)
                for fold, (train, _) in enumerate(folds):
                    outer_fold = None if folds.is_single else fold
                    fitted_on = followers.copy()
                    if not folds.is_single:
                        train_rows = np.zeros(data.n, dtype=bool)
                        train_rows[train] = True
                        fitted_on &= train_rows
                    _require_regimen_followers(
                        data, plan, horizon, at_risk, fitted_on, outer_fold=outer_fold
                    )
                    _check_outcome_varies(
                        data,
                        target,
                        fitted_on,
                        plan,
                        horizon,
                        horizon,
                        cause,
                        outer_fold=outer_fold,
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
    raise LongitudinalError(
        f"no unit followed regimen {plan.label!r} through time {time} {where}, so "
        "the sequential regression there has nothing to fit. The regimen is not "
        "supported by this sample."
        + (
            ""
            if outer_fold is None
            else " " + _CROSS_FIT_NODE_REMEDY.format(alternative="choose a supported regimen")
        )
        + _risk_set_hint(data, plan, time)
        + _rule_hint(plan, data, at_risk, time)
    )


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
            masks = data.regimen_masks(plan.values)
    at_risk = masks.at_risk(time)
    trained_on = masks.following(time)
    fitted_on = trained_on if fit_rows is None else trained_on & fit_rows
    _require_regimen_followers(data, plan, time, at_risk, fitted_on, outer_fold=outer_fold)
    with phase("pseudo_outcome"):
        next_outcome = _pseudo_outcome(data, carried, time, cause)
    design = data.covariate_history(time)
    learner = outcome_learner if time == horizon else pseudo_learner
    task = "classification" if time == horizon and data.family == "binomial" else "regression"
    if task == "classification":
        _check_outcome_varies(
            data, next_outcome, fitted_on, plan, time, horizon, cause, outer_fold=outer_fold
        )
    with phase("outcome_learner_fit"):
        predictions, diagnostics = cross_fit_predictions(
            learner,
            design,
            next_outcome,
            data.weights,
            folds,
            task=task,  # type: ignore[arg-type]
            predict_designs={"history": design},
            fit_mask=fitted_on,
            groups=data.cluster,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
    return _NodeRegression(
        time=time,
        at_risk=at_risk,
        trained_on=trained_on,
        fitted_on=fitted_on,
        pseudo_outcome=next_outcome,
        initial=np.where(at_risk, predictions["history"], _FILLER),
        learner_diagnostics=tuple(diagnostics),
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
    fit_rows: BoolArray | None = None,
    outer_fold: int | None = None,
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

    ``fit_rows`` narrows the rows the regression is **fitted** on without narrowing the
    rows ``clever`` is nonzero at.  The two are the same set on an ordinary pass and come
    apart under outer cross-fitting, where fold ``k``'s regression trains on the followers
    in its training complement while the clever covariate stays a statement about every
    follower.  Passing them as one mask -- which ``trained_on`` was until the outer
    recursion needed both -- silently zeroes the held-out rows' covariate and drops them
    from the score.  ``outer_fold`` names the fold in both refusals this node can raise --
    the empty risk set below and :func:`_check_outcome_varies` -- and reports the one-based
    number a reader can find in ``result.folds``.  Both refusals are statements about
    ``fitted_on``, so under cross-fitting both are statements about one fold's training
    rows, and neither may call that set the sample.
    """
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
        fit_rows=fit_rows,
        outer_fold=outer_fold,
        n_jobs=n_jobs,
    )
    with phase("clever_covariate"):
        counterfactual, clever = _clever_covariate(
            regression.at_risk, regression.trained_on, cumulative, time
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
    )


def _clever_covariate(
    at_risk: BoolArray, trained_on: BoolArray, cumulative: FloatArray, time: int
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

    Returns
    -------
    tuple of FloatArray
        ``counterfactual``, which is ``1 / cumulative[:, time - 1]`` on ``at_risk`` and
        zero elsewhere, and ``clever``, which is that weight on ``trained_on`` and zero
        elsewhere.
    """
    denominator = np.where(at_risk, cumulative[:, time - 1], 1.0)
    counterfactual = np.where(at_risk, 1.0 / denominator, 0.0)
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

    Returns
    -------
    Fluctuation
        The solved fluctuation.  Its targeted predictions are filed under the regimen key.
    """
    intercept = np.ones((len(pseudo_outcome), 1))
    return solve_fluctuation(
        pseudo_outcome,
        InitialFit(initial, {_REGIMEN_ARM: initial}),
        Submodel(
            intercept,
            {_REGIMEN_ARM: intercept},
            (f"epsilon[{label}, t={time}]",),
            "sequential",
        ),
        loss_weights,
        fitted_on,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )


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


def _check_outcome_varies(
    data: LongitudinalData,
    next_outcome: FloatArray,
    fitted_on: BoolArray,
    plan: Plan,
    time: int,
    horizon: int,
    cause: str | None,
    *,
    outer_fold: int | None = None,
) -> None:
    """Refuse a classification with nothing to separate, saying which case it is.

    The set the check reads is ``fitted_on``, the rows the regression is *fitted* on, and
    that set is what the refusal has to name.  On a single-fold pass it is every follower
    in the sample.  Under outer cross-fitting it is the followers in one fold's training
    complement, a strict subset, so the same frame can be estimable at ``n_folds=1`` and
    refused at ``n_folds=2``.  Calling the binding set "this sample" in both cases reports
    the second as a sample-size problem, which sends the reader to collect more data when
    the fold count is what moved.
    """
    seen = np.unique(next_outcome[fitted_on])
    if seen.size >= 2:
        return
    where = "" if outer_fold is None else f" in outer training fold {outer_fold + 1}"
    scope = "this sample" if outer_fold is None else f"outer training fold {outer_fold + 1}"
    crossfit_note = (
        ""
        if outer_fold is None
        else (
            " The check applies to each outer fold's training rows, not to the sample as a "
            "whole, so a cross-fitted fit needs the outcome to vary in every fold's "
            "training complement. That is stricter than a single-fold fit, which fits on "
            "every row, and the same frame can be estimable at n_folds=1. "
            + _CROSS_FIT_NODE_REMEDY.format(
                alternative="choose an estimand this fold count supports"
            )
        )
    )
    raise LongitudinalError(
        f"every unit following regimen {plan.label!r} through time {time}{where} has "
        f"the same outcome ({seen.tolist()}), so the regression there has "
        "nothing to separate. "
        + (
            (
                f"The incidence of {cause!r} at horizon {horizon} is not "
                f"estimable from {scope}: no unit following the regimen was "
                f"observed to leave through {cause!r}. A rare cause reaches "
                "this well before a common one does, so it is refused per "
                "cause rather than for the fit as a whole."
            )
            if cause is not None
            else (
                f"The risk at horizon {horizon} is not estimable from {scope}: "
                "no event was observed among the regimen's followers."
            )
            if data.is_survival
            else "The outcome does not vary among the regimen's followers."
        )
        + crossfit_note
    )


def _finish_regimen_fit(
    data: LongitudinalData,
    plan: Plan,
    steps: Sequence[SequentialStep],
    cumulative_unbounded: FloatArray,
    cumulative: FloatArray,
    *,
    horizon: int,
    cause: str | None,
) -> RegimenFit:
    """Assemble one regimen estimate and influence curve from its targeted steps."""
    retained = tuple(steps)
    # Every unit is at risk at the first node, so this averages predictions rather than
    # fillers. ``np.average`` against mean-one weights is bit-for-bit the old unweighted
    # mean when the weights are constant.
    psi = float(np.average(retained[0].targeted, weights=data.weights))
    influence = retained[0].targeted - psi
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
    )


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
    masks = data.regimen_masks(plan.values)

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
            )
        targeted = fluctuation.targeted.arms[_REGIMEN_ARM]
        steps.append(
            SequentialStep(
                time=time,
                trained_on=node.trained_on,
                at_risk=node.at_risk,
                pseudo_outcome=node.pseudo_outcome,
                initial=node.initial,
                targeted=targeted,
                clever=node.clever,
                fluctuation=fluctuation,
                learner_diagnostics=node.learner_diagnostics,
            )
        )
        carried = np.where(node.at_risk, targeted, _FILLER)

    steps.reverse()
    return _finish_regimen_fit(
        data,
        plan,
        steps,
        cumulative_unbounded,
        cumulative,
        horizon=horizon,
        cause=cause,
    )


@dataclass(frozen=True)
class _FoldSolve:
    """What the parent needs from one outer fold's node solve, without its arrays.

    Only the engine-level cross-fitted working-model path in
    :mod:`cleverly.longitudinal.msm` solves a fluctuation inside each outer fold, and so
    only it builds these.  The public estimator refuses ``msm=`` above one fold.  A
    per-regimen cross-fitted fit solves one pooled fluctuation per node instead, in
    :func:`_pooled_targeting`.

    A whole :class:`~cleverly.fluctuation.Fluctuation` carries a full-length
    :class:`~cleverly.fluctuation.InitialFit`, so returning ``K`` of them per node from
    ``K`` worker processes pickles ``K * T * 2n`` floats for the sake of a handful of
    scalars.  The stitched arrays the parent reports are assembled from each fold's
    held-out slice instead, which is ``n / K`` rows rather than ``n``.
    """

    record: FoldFluctuation
    #: Observation-weight mass of the fold's held-out rows.  The reported ``epsilon`` is
    #: the average across folds weighted by this, so a weighted fit averages by the
    #: weights it was fitted with rather than by row counts.
    mass: float
    failure: TargetingFailure | None
    hessian_condition: float
    loglik: float
    method: TargetingLabel
    names: tuple[str, ...]


def aggregate_fold_fluctuations(
    solves: Sequence[_FoldSolve],
    *,
    outcome: FloatArray,
    initial: FloatArray,
    targeted: FloatArray,
    covariate: FloatArray,
    loss_weights: FloatArray,
    mask: BoolArray,
) -> Fluctuation:
    r"""Combine outer-training solves without reporting a score that no array has.

    The one caller is the engine-level cross-fitted working-model path in
    :mod:`cleverly.longitudinal.msm`, which the public estimator refuses above one fold.
    A per-regimen cross-fitted fit no longer fluctuates inside a fold, so it has no fold
    solves to aggregate.

    The five arrays are the stitched fit's own, assembled by the caller.  A working-model
    node fluctuates by :math:`(dm/d\eta)\varphi` over the ``C`` live cells stacked, which is
    ``C * n`` rows and ``p`` columns.  Taking the arrays rather than rebuilding them here
    keeps the aggregation free of any one submodel's shape.

    **The score here is the score of the stitched fit**, computed from the arrays this
    object is returned beside, and not the average of the ``K`` per-fold scores.  Each of
    those is at solver tolerance by construction, so averaging them reports
    :math:`10^{-14}` for a fit whose pooled relative score is :math:`10^{-2}`, and
    :meth:`~cleverly.assessment.DiagnosticsFacade.score_equations` then signs off on a fit
    that nothing checked.  :meth:`cleverly.TMLE._solve_by_fold` recomputes the pooled score
    for the same reason, through the same helpers in :mod:`cleverly.fluctuation._score`.

    **The pooled score is not zero, and is not meant to be.**  Each fold fits its
    ``epsilon`` on its *training* complement, so the equation a fold solved is not the one
    its held-out rows pose.  What the pooled residual has to be is sampling noise about
    zero, which is a different claim and needs a different instrument:
    :attr:`~cleverly.fluctuation.Fluctuation.folds` carries the per-fold solves that did
    reach their roots, and :func:`~cleverly.assessment.score_equations` reports the two
    verdicts as separate rows.

    ``converged`` is therefore ``all`` of the fold solves rather than a relative-score test
    on the aggregate.  A solver that reached its root in every fold converged; that the
    pooled residual is nonzero is a property of the construction and not a failure.
    """
    masses = np.asarray([solve.mass for solve in solves], dtype=float)
    reasons = [solve.failure or "unknown" for solve in solves]
    failed = [index for index, solve in enumerate(solves) if not solve.record.converged]
    conditions = [solve.hessian_condition for solve in solves]
    finite = [value for value in conditions if np.isfinite(value)]
    return Fluctuation(
        epsilon=np.asarray(
            np.average(
                np.vstack([np.asarray(solve.record.epsilon, dtype=float) for solve in solves]),
                axis=0,
                weights=masses,
            ),
            dtype=float,
        ),
        targeted=InitialFit(targeted, {_REGIMEN_ARM: targeted}),
        score=score_columns(outcome, targeted, covariate, loss_weights, mask),
        converged=all(solve.record.converged for solve in solves),
        n_iter=sum(solve.record.n_iter for solve in solves),
        # Several fold solves have no single iteration trajectory. Their complete traces
        # live on the fold records instead of masquerading as one aggregate trace.
        trace=(),
        method=solves[0].method,
        names=solves[0].names,
        score_scale=score_scale(covariate, loss_weights, mask),
        folds=tuple(solve.record for solve in solves),
        score_initial=score_columns(outcome, initial, covariate, loss_weights, mask),
        n_solver_calls=len(solves),
        failure=dominant_failure(reasons, failed),
        # Not an average: a condition number says how badly identified the worst solve's
        # epsilon was, and averaging that with well-conditioned folds hides the one fold
        # the reader needs.  `nan` only when no fold reported one at all.
        hessian_condition=max(finite) if finite else float("nan"),
        loglik=float(np.average([solve.loglik for solve in solves], weights=masses)),
    )


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

    The node arithmetic inside a fold is :func:`prepare_node`'s, called with the fold's
    training complement as ``fit_rows`` and a one-fold split -- which is a fit on the named
    rows and a prediction everywhere, and is what an outer fold's model is.  So both
    refusal hints, and every mask, are the ones the single-fold pass uses.

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
    # The out-of-fold pair is the only mechanism this fit divides by.  The fold slabs the
    # mechanism also carries serve the engine-level cross-fitted working model alone.
    cumulative_unbounded, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    with phase("mask_construction"):
        masks = data.regimen_masks(plan.values)
    inner = Folds.single(data.n)
    # Read once, in the parent: a worker process has no collector of its own and so
    # cannot tell whether anybody asked for a profile.
    wanted = profiling()
    initial: dict[int, FloatArray] = {
        time: np.full(data.n, _FILLER, dtype=float) for time in range(1, horizon + 1)
    }
    regression_target: dict[int, FloatArray] = {
        time: np.full(data.n, _FILLER, dtype=float) for time in range(1, horizon + 1)
    }
    diagnostics: dict[int, list[SuperLearnerDiagnostics]] = {
        time: [] for time in range(1, horizon + 1)
    }

    def run_fold(
        fold: int, train: IntArray, test: IntArray
    ) -> tuple[
        IntArray,
        dict[int, tuple[FloatArray, FloatArray, tuple[SuperLearnerDiagnostics, ...]]],
        PhaseProfile | None,
    ]:
        outer_train = np.zeros(data.n, dtype=bool)
        outer_train[train] = True
        outputs: dict[int, tuple[FloatArray, FloatArray, tuple[SuperLearnerDiagnostics, ...]]] = {}
        with collect_phases(wanted) as profile:
            carried = seed_carried(data, scaler)
            for time in range(horizon, 0, -1):
                node = _fit_node_regression(
                    data,
                    plan,
                    carried,
                    time,
                    horizon,
                    outcome_learner=outcome_learner,
                    pseudo_learner=pseudo_learner,
                    folds=inner,
                    cause=cause,
                    masks=masks,
                    fit_rows=outer_train,
                    outer_fold=fold,
                )
                outputs[time] = (
                    node.initial[test],
                    node.pseudo_outcome[test],
                    node.learner_diagnostics,
                )
                # The fold's *untargeted* prediction is what the earlier node regresses.
                # The previously shipped construction instead fluctuated on the fold's
                # training rows and carried that targeted prediction into the fold's next
                # regression.  Steps 1-4 of the Section 5.2 construction carry the
                # untargeted prediction, and target only in the pooled pass.
                carried = np.where(node.at_risk, node.initial, _FILLER)
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
    for test, outputs, _ in outcomes:
        for time, (held_out, target, fold_diagnostics) in outputs.items():
            initial[time][test] = held_out
            regression_target[time][test] = target
            diagnostics[time].extend(fold_diagnostics)

    steps = _pooled_targeting(
        data,
        plan,
        masks,
        cumulative,
        initial,
        regression_target,
        {time: tuple(values) for time, values in diagnostics.items()},
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
    )


def _pooled_targeting(
    data: LongitudinalData,
    plan: Plan,
    masks: RegimenMasks,
    cumulative: FloatArray,
    initial: dict[int, FloatArray],
    regression_target: dict[int, FloatArray],
    diagnostics: dict[int, tuple[SuperLearnerDiagnostics, ...]],
    *,
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
    initial : dict of int to FloatArray
        Stitched out-of-fold initial predictions, by node.
    regression_target : dict of int to FloatArray
        Stitched targets the fold regressions were fitted to, by node.
    diagnostics : dict of int to tuple of SuperLearnerDiagnostics
        Learner diagnostics of every fold's regression, by node.
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
        at_risk = masks.at_risk(time)
        following = masks.following(time)
        with phase("pseudo_outcome"):
            pseudo_outcome = _pseudo_outcome(data, carried, time, cause)
        with phase("clever_covariate"):
            counterfactual, clever = _clever_covariate(at_risk, following, cumulative, time)
        # One phase entry per node, as on a single-fold fit, and always in the parent.
        with phase("fluctuation"):
            fluctuation = _fluctuate_node(
                pseudo_outcome,
                initial[time],
                data.weights * counterfactual,
                following,
                label=plan.label,
                time=time,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
            )
        targeted = fluctuation.targeted.arms[_REGIMEN_ARM]
        steps.append(
            SequentialStep(
                time=time,
                trained_on=following,
                at_risk=at_risk,
                pseudo_outcome=pseudo_outcome,
                initial=initial[time],
                targeted=targeted,
                clever=clever,
                fluctuation=fluctuation,
                learner_diagnostics=diagnostics[time],
                regression_target=regression_target[time],
            )
        )
        carried = np.where(at_risk, targeted, _FILLER)
    steps.reverse()
    return tuple(steps)


def warn_on_fold_convergence(nodes: Sequence[tuple[int, Fluctuation]], label: str) -> None:
    """Report the outer folds that did not converge, once, naming the modes.

    The engine-level cross-fitted working-model path in :mod:`cleverly.longitudinal.msm`
    is the one caller.  A per-regimen cross-fitted fit solves one pooled fluctuation per
    node, which warns for itself as a single-fold solve does.

    The per-fold solves run with ``warn=False`` so that ``K`` folds at ``T`` nodes cannot
    emit ``K * T`` warnings for one problem.  That alone would leave a fit able to fail in
    three folds of ten and say nothing at all, because the aggregate ``converged`` is then
    the only place it shows and the pooled score is nonzero on a healthy fit anyway.  So it
    is said here, once per working model, which is what
    :meth:`cleverly.TMLE._solve_by_fold` does for the point-treatment fold-targeting path.

    ``nodes`` is ``(time, aggregated fluctuation)`` rather than the steps it came from, so
    that the working-model path -- whose one fluctuation per node is shared by every live
    cell -- says it once for the model rather than once per cell saying the same thing.

    Parameters
    ----------
    nodes : sequence of tuple of int and Fluctuation
        Each node's time and its aggregated fluctuation, whose ``folds`` hold the solves.
    label : str
        What the warning names: a regimen label, or ``"msm"``.
    """
    failures = [
        (time, record)
        for time, fluctuation in nodes
        for record in fluctuation.folds
        if not record.converged
    ]
    if not failures:
        return
    modes = sorted(
        {fluctuation.failure or "unknown" for _, fluctuation in nodes if not fluctuation.converged}
    )
    times = sorted({time for time, _ in failures})
    warnings.warn(
        f"{len(failures)} outer-fold targeting solve(s) did not converge for {label!r}, "
        f"at node(s) {times} ({', '.join(modes)}). The stitched score cannot show this, "
        "because it is not the equation those solves posed; inspect "
        "step.fluctuation.folds for the per-fold detail.",
        ConvergenceWarning,
        stacklevel=3,
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
    reached = data.uncensored_through(time - 1) & data.followed_through(plan.values, time - 1)
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
