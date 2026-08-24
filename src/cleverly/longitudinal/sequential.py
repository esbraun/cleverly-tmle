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

With outer cross-fitting, that whole recursion is the unit of splitting.  Fold :math:`k`
fits every mechanism, regression and fluctuation on its training complement, carries those
fold-specific targeted predictions backwards, and evaluates the completed recursion only on
fold :math:`k`'s held-out rows.  The estimator stitches those held-out rows after all folds
finish.  A held-out outcome therefore cannot enter an earlier pseudo-outcome or fluctuation
that is used to predict its own row.

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
    "prepare_node",
    "seed_carried",
]


#: Filler for a prediction at a row the estimator never reads -- a unit censored before
#: the node in question.  Any finite number in ``(0, 1)`` would do; a half keeps ``logit``
#: at zero, so a filled row cannot make a Newton step look large.
_FILLER = 0.5

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
    """

    treatment: tuple[dict[str, FloatArray], ...]
    censoring: tuple[dict[str, FloatArray], ...]
    #: Predictions from each outer-fold model on every row.  Entry ``[t][label]`` has
    #: shape ``(K, n)``.  The recursion uses slab ``k`` for both the training and held-out
    #: rows of outer fold ``k``.  Empty only on mechanisms made by older persisted fits.
    treatment_by_fold: tuple[dict[str, FloatArray], ...] = ()
    censoring_by_fold: tuple[dict[str, FloatArray], ...] = ()

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
    """

    time: int
    at_risk: BoolArray
    #: Every row that followed the regimen through this node.  ``clever`` is nonzero
    #: exactly here, whichever rows the regression was fitted on.
    trained_on: BoolArray
    #: The rows the regression was actually fitted on: ``trained_on``, and under outer
    #: cross-fitting that intersected with the fold's training complement.  This is the
    #: mask the fluctuation solves its score over, so that a held-out row contributes to
    #: no coefficient that fluctuates it.
    fitted_on: BoolArray
    pseudo_outcome: FloatArray
    initial: FloatArray
    counterfactual: FloatArray
    clever: FloatArray


@dataclass(frozen=True)
class SequentialStep:
    """One node of the backward recursion, kept so a fit can be inspected node by node."""

    time: int
    #: Rows eligible for the regression: followed the regimen and stayed under
    #: observation through this node.  With cross-fitting, each outer model uses the
    #: eligible rows in its training complement.  ``fluctuation.folds`` identifies those
    #: complements through their held-out indices.
    trained_on: BoolArray
    #: Rows whose history at this node is observed and regimen-consistent -- the set the
    #: regression *predicts* for, and the population the assigned arm is a statement
    #: about.  Equal to the previous node's ``trained_on`` on an end-of-study fit, which
    #: is what closes the recursion; on a survival fit it is that set less the units that
    #: had the event there, which closes it just as well and is the general statement.
    at_risk: BoolArray
    #: What this node's regression was fitted *to*: the later node's targeted prediction
    #: on an end-of-study fit, and on a survival one the composition
    #: :math:`Z_t = Y_t + (1 - Y_t)\\,\\bar{Q}^*_{t+1}`.  Stored rather than recomputed
    #: because the influence curve's ``t``-th term needs the same quantity, and the one
    #: place this recursion could silently disagree with itself is by composing the
    #: pseudo-outcome twice and composing it differently.
    pseudo_outcome: FloatArray
    initial: FloatArray
    targeted: FloatArray
    clever: FloatArray
    fluctuation: Fluctuation

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
    #: **On a cross-fitted fit these are the out-of-fold prefixes and are not what the
    #: clever covariate was built from.**  Each outer fold's recursion reads its own
    #: mechanism slab, so ``step.clever`` is stitched from ``K`` fold-specific covariates
    #: and ``1 / cumulative`` does not reproduce it.  Both are honest reports of different
    #: things: this is the mechanism the fit estimated, and ``step.clever`` is the weight
    #: each row was actually targeted and scored with.  A diagnostic that wants the weight
    #: must read ``step.clever``; one that wants the mechanism, or how much of it the
    #: bounds moved, reads these.
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
) -> None:
    """Refuse a **categorical** mechanism fit whose training law omits an observed level.

    A missing class cannot be repaired by aligning a learner's probability columns: the
    requested assigned-arm probability is unidentified in that training fold.  Checking
    here also gives the analyst the original label, rather than a downstream matrix-shape
    error involving its internal dense code.

    **A two-level node is left alone**, and that is a compatibility decision rather than
    an oversight.  At two classes
    :func:`~cleverly.learners._fitting.predict_probabilities` delegates to ``predict_mean``
    and a degenerate training fold yields that fold's constant, which is the behaviour
    every binary fit has had; the diagnosis a binary panel reaches instead is
    :func:`prepare_node`'s "no unit followed regimen" refusal, which names the regimen
    rather than the fold.  Refusing here at ``K = 2`` as well would change an error a
    binary fit already had, for a case the existing path already reports.

    At three or more levels the fallback is not benign: the missing arm's column comes
    back zero, so its clever covariate is a division by zero and the cumulative product's
    reciprocal is infinite.  That is why the check exists at all, and why it starts here.
    """
    if len(classes) < 3:
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
            raise LongitudinalError(
                f"treatment node {node_name!r} is missing level(s) {missing!r} in {where}; "
                "every treatment-mechanism training set must contain every observed level. "
                "Use fewer folds, supply folds that preserve treatment support, or collect "
                "more observations at the rare level"
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
    # Neither factor depends on a regimen, so one scan serves every node.  `followed` is
    # unused here and the all-true assignment makes that explicit rather than implicit.
    with phase("mask_construction"):
        fit_masks = data.regimen_masks(data.treatment)
    for time in range(1, data.n_times + 1):
        at_risk = fit_masks.uncensored[:, time - 1] & fit_masks.event_free[:, time - 1]
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        designs = {plan.label: data.history_design(time, treatment=plan.values) for plan in plans}
        with phase("mechanism_fit"):
            classes = tuple(float(code) for code in range(len(data.treatment_levels[time - 1])))
            _check_categorical_fold_support(
                arm,
                at_risk,
                folds,
                classes,
                data.treatment_levels[time - 1],
                data.treatment_names[time - 1],
            )
            probabilities, companion, _ = cross_fit_companion(
                treatment_learner,
                data.history_design(time),
                arm,
                data.weights,
                folds,
                task="classification",
                predict_designs=designs,
                companion_designs=designs,
                fit_mask=at_risk,
                groups=data.cluster,
                clip=(0.0, 1.0),
                classes=classes,
                n_jobs=n_jobs,
            )
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
        with phase("mechanism_fit"):
            predictions, censor_companion, _ = cross_fit_companion(
                censoring_learner,
                data.history_design(time, include_current=True),
                stayed,
                data.weights,
                folds,
                task="classification",
                predict_designs=censor_designs,
                companion_designs=censor_designs,
                fit_mask=at_risk,
                groups=data.cluster,
                clip=(0.0, 1.0),
                n_jobs=n_jobs,
            )
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
    from the score.  ``outer_fold`` names the fold in the refusal below, and reports the
    one-based number a reader can find in ``result.folds``.
    """
    if masks is None:
        # Only around the scan, and only when there is one: reading two prefix slices off
        # masks the caller already built is not mask construction, and counting it as such
        # reports one entry per node for the work the once-per-regimen scan exists to
        # avoid -- which is the opposite of what the phase was added to show.
        with phase("mask_construction"):
            masks = data.regimen_masks(plan.values)
    at_risk = masks.at_risk(time)
    trained_on = masks.following(time)
    fitted_on = trained_on if fit_rows is None else trained_on & fit_rows
    if not fitted_on.any():
        where = (
            "while remaining in the study"
            if outer_fold is None
            else f"in outer training fold {outer_fold + 1}"
        )
        raise LongitudinalError(
            f"no unit followed regimen {plan.label!r} through time {time} {where}, so "
            "the sequential regression there has nothing to fit. The regimen is not "
            "supported by this sample."
            + ("" if outer_fold is None else " Use fewer folds, or a supported regimen.")
            + _risk_set_hint(data, plan, time)
            + _rule_hint(plan, data, at_risk, time)
        )
    with phase("pseudo_outcome"):
        next_outcome = _pseudo_outcome(data, carried, time, cause)
    design = data.covariate_history(time)
    learner = outcome_learner if time == horizon else pseudo_learner
    task = "classification" if time == horizon and data.family == "binomial" else "regression"
    if task == "classification":
        _check_outcome_varies(data, next_outcome, fitted_on, plan, time, horizon, cause)
    with phase("outcome_learner_fit"):
        predictions, _ = cross_fit_predictions(
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
    with phase("clever_covariate"):
        initial = np.where(at_risk, predictions["history"], _FILLER)
        denominator = np.where(at_risk, cumulative[:, time - 1], 1.0)
        counterfactual = np.where(at_risk, 1.0 / denominator, 0.0)
        clever = np.where(trained_on, counterfactual, 0.0)
    return NodeInputs(
        time=time,
        at_risk=at_risk,
        trained_on=trained_on,
        fitted_on=fitted_on,
        pseudo_outcome=next_outcome,
        initial=initial,
        counterfactual=counterfactual,
        clever=clever,
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
    trained_on: BoolArray,
    plan: Plan,
    time: int,
    horizon: int,
    cause: str | None,
) -> None:
    """Refuse a classification with nothing to separate, saying which case it is."""
    seen = np.unique(next_outcome[trained_on])
    if seen.size >= 2:
        return
    raise LongitudinalError(
        f"every unit following regimen {plan.label!r} through time {time} has "
        f"the same outcome ({seen.tolist()}), so the regression there has "
        "nothing to separate. "
        + (
            (
                f"The incidence of {cause!r} at horizon {horizon} is not "
                "estimable from this sample: no unit following the regimen was "
                f"observed to leave through {cause!r}. A rare cause reaches "
                "this well before a common one does, so it is refused per "
                "cause rather than for the fit as a whole."
            )
            if cause is not None
            else (
                f"The risk at horizon {horizon} is not estimable from this "
                "sample: no event was observed among the regimen's followers."
            )
            if data.is_survival
            else "The outcome does not vary among the regimen's followers."
        )
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
            # Canonical longitudinal TMLE uses an intercept fluctuation with the
            # cumulative inverse probability in the *loss weight* (ltmle::UpdateQ), not
            # as the logistic submodel's covariate.  Both choices solve sum H(Y-Q*)=0;
            # they do not produce the same finite-sample substitution estimator when
            # epsilon is nonzero, which is why the distinction is explicit here.
            intercept = np.ones((data.n, 1))
            fluctuation = solve_fluctuation(
                node.pseudo_outcome,
                InitialFit(node.initial, {_REGIMEN_ARM: node.initial}),
                Submodel(
                    intercept,
                    {_REGIMEN_ARM: intercept},
                    (f"epsilon[{plan.label}, t={time}]",),
                    "sequential",
                ),
                data.weights * node.counterfactual,
                node.fitted_on,
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
            )
        )
        carried = np.where(node.at_risk, targeted, _FILLER)

    steps.reverse()
    # Every unit is at risk at the first node, so this averages predictions rather than
    # fillers.  ``np.average`` against a mean-one weight vector is the ``np.mean`` it
    # replaced entry for entry when the weights are constant, which is what keeps an
    # unweighted fit bit-for-bit what it was.
    psi = float(np.average(steps[0].targeted, weights=data.weights))
    influence = steps[0].targeted - psi
    for step in steps:
        # The ``t``-th term of the efficient influence function reads the very array the
        # ``t``-th regression was fitted to -- the later node's targeted prediction, or,
        # on a survival fit, that prediction composed with this node's event indicator.
        influence = influence + step.clever * (step.pseudo_outcome - step.targeted)
    # Row-wise at the end rather than term by term: the weighted EIF is
    # ``(w / E[w]) D*(P_w)``, one factor multiplying the whole curve, and the *centring*
    # inside the bracket is what linearises the Hajek ratio above.  Multiplying only the
    # residual terms, or subtracting ``psi`` outside the weight, is the classic error and
    # would leave a curve whose mean is not zero.
    influence = data.weights * influence
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=influence,
        horizon=horizon,
        cause=cause,
        steps=tuple(steps),
        cumulative_unbounded=cumulative_unbounded,
        cumulative=cumulative,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
    )


@dataclass(frozen=True)
class _FoldSolve:
    """What the parent needs from one outer fold's node solve, without its arrays.

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

    The five arrays are the stitched fit's own, assembled by the caller, because the two
    callers stack them differently.  A per-regimen node fluctuates by an intercept over
    ``n`` rows; a working-model node fluctuates by :math:`(dm/d\eta)\varphi` over the ``C``
    live cells stacked, which is ``C * n`` rows and ``p`` columns.  Taking the arrays rather
    than rebuilding them here is what lets one aggregation serve both -- the first version
    hardcoded the intercept, and a second copy is how the two would drift apart.

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
    """Run one complete backward recursion per outer fold and stitch held-out rows.

    The node arithmetic is :func:`prepare_node`'s, called once per fold with that fold's
    mechanism slab, its training complement as ``fit_rows``, and a one-fold split -- which
    is a fit on the named rows and a prediction everywhere, and is what an outer fold's
    model is.  Writing the node out a second time here is what let the first version of
    this function drift: it dropped both refusal hints, and every mask was a second chance
    to disagree with the pass it is supposed to be.
    """
    cumulative_unbounded, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    with phase("mask_construction"):
        masks = data.regimen_masks(plan.values)
    inner = Folds.single(data.n)
    # Read once, in the parent: a worker process has no collector of its own and so
    # cannot tell whether anybody asked for a profile.
    wanted = profiling()
    intercept = np.ones((data.n, 1))
    weights = np.asarray(data.weights, dtype=float)
    stitched: dict[int, dict[str, FloatArray]] = {
        time: {
            name: np.full(data.n, _FILLER, dtype=float)
            for name in ("pseudo", "initial", "targeted", "clever")
        }
        for time in range(1, horizon + 1)
    }
    fold_solves: dict[int, list[_FoldSolve]] = {time: [] for time in range(1, horizon + 1)}

    def run_fold(
        fold: int, train: IntArray, test: IntArray
    ) -> tuple[
        IntArray,
        dict[int, tuple[FloatArray, FloatArray, FloatArray, FloatArray, _FoldSolve]],
        PhaseProfile | None,
    ]:
        _, fold_cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds, fold=fold)
        outer_train = np.zeros(data.n, dtype=bool)
        outer_train[train] = True
        outputs: dict[int, tuple[FloatArray, FloatArray, FloatArray, FloatArray, _FoldSolve]] = {}
        with collect_phases(wanted) as profile:
            carried = seed_carried(data, scaler)
            for time in range(horizon, 0, -1):
                node = prepare_node(
                    data,
                    plan,
                    fold_cumulative,
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
                with phase("fluctuation"):
                    # `warn=False`, and the count reported once by
                    # `warn_on_fold_convergence`: K folds at T nodes would otherwise emit
                    # K * T warnings for one problem.
                    fluctuation = solve_fluctuation(
                        node.pseudo_outcome,
                        InitialFit(node.initial, {_REGIMEN_ARM: node.initial}),
                        Submodel(
                            intercept,
                            {_REGIMEN_ARM: intercept},
                            (f"epsilon[{plan.label}, t={time}]",),
                            "sequential",
                        ),
                        weights * node.counterfactual,
                        node.fitted_on,
                        alpha=alpha,
                        max_iter=max_iter,
                        tol=tol,
                        warn=False,
                    )
                targeted = fluctuation.targeted.arms[_REGIMEN_ARM]
                outputs[time] = (
                    node.pseudo_outcome[test],
                    node.initial[test],
                    targeted[test],
                    node.clever[test],
                    _FoldSolve(
                        record=FoldFluctuation(
                            index=test,
                            epsilon=fluctuation.epsilon,
                            score=fluctuation.score,
                            converged=fluctuation.converged,
                            n_iter=fluctuation.n_iter,
                            trace=fluctuation.trace,
                            score_scale=fluctuation.score_scale,
                        ),
                        mass=float(weights[test].sum()),
                        failure=fluctuation.failure,
                        hessian_condition=fluctuation.hessian_condition,
                        loglik=fluctuation.loglik,
                        method=fluctuation.method,
                        names=fluctuation.names,
                    ),
                )
                carried = np.where(node.at_risk, targeted, _FILLER)
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
        for time, (pseudo, initial, targeted, clever, solve) in outputs.items():
            stitched[time]["pseudo"][test] = pseudo
            stitched[time]["initial"][test] = initial
            stitched[time]["targeted"][test] = targeted
            stitched[time]["clever"][test] = clever
            fold_solves[time].append(solve)

    steps = tuple(
        SequentialStep(
            time=time,
            trained_on=masks.following(time),
            at_risk=masks.at_risk(time),
            pseudo_outcome=stitched[time]["pseudo"],
            initial=stitched[time]["initial"],
            targeted=stitched[time]["targeted"],
            clever=stitched[time]["clever"],
            fluctuation=aggregate_fold_fluctuations(
                fold_solves[time],
                outcome=stitched[time]["pseudo"],
                initial=stitched[time]["initial"],
                targeted=stitched[time]["targeted"],
                # The sequential submodel is an intercept and the cumulative inverse
                # probability rides in the loss weight, so the score's `h` is that
                # intercept and its weights are `w * clever`.  `clever` is already zero off
                # the followers, which is why the mask can be the population rather than a
                # second place the same fact is written down.
                covariate=intercept,
                loss_weights=weights * stitched[time]["clever"],
                mask=masks.following(time),
            ),
        )
        for time in range(1, horizon + 1)
    )
    warn_on_fold_convergence([(step.time, step.fluctuation) for step in steps], plan.label)
    psi = float(np.average(steps[0].targeted, weights=data.weights))
    influence = steps[0].targeted - psi
    for step in steps:
        influence = influence + step.clever * (step.pseudo_outcome - step.targeted)
    influence = data.weights * influence
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=influence,
        horizon=horizon,
        cause=cause,
        steps=steps,
        cumulative_unbounded=cumulative_unbounded,
        cumulative=cumulative,
        assignment=np.asarray(plan.values),
        obs_weights=weights,
    )


def warn_on_fold_convergence(nodes: Sequence[tuple[int, Fluctuation]], label: str) -> None:
    """Report the outer folds that did not converge, once, naming the modes.

    The per-fold solves run with ``warn=False`` so that ``K`` folds at ``T`` nodes cannot
    emit ``K * T`` warnings for one problem.  That alone would leave a fit able to fail in
    three folds of ten and say nothing at all, because the aggregate ``converged`` is then
    the only place it shows and the pooled score is nonzero on a healthy fit anyway.  So it
    is said here, once per regimen or working model, which is what
    :meth:`cleverly.TMLE._solve_by_fold` does for the point-treatment fold-targeting path.

    ``nodes`` is ``(time, aggregated fluctuation)`` rather than the steps it came from, so
    that the working-model path -- whose one fluctuation per node is shared by every live
    cell -- says it once for the model rather than once per cell saying the same thing.
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
