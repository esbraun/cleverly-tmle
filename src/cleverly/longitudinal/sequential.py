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

That is the substitution estimator; it is not efficient and has no influence curve of its
own.  Targeting supplies both.  At each step the initial regression is fluctuated along

.. math::

    \operatorname{logit} \bar Q^*_t = \operatorname{logit} \bar Q_t + \epsilon_t\, h_t,
    \qquad
    h_t = \frac{\mathbb 1\{\bar A_t = \bar a_t,\, \bar C_t = 1\}}
               {\prod_{s \le t} g_s(a_s \mid H_s)\, c_s(H_s, a_s)}

whose score is the :math:`t`-th term of the efficient influence function

.. math::

    D^*(O) = \sum_{t=1}^{T} h_t \bigl(\bar Q^*_{t+1} - \bar Q^*_t\bigr)
             + \bar Q^*_1(H_1) - \Psi.

Solving all :math:`T` of them makes the estimator solve :math:`P_n D^* = 0`, which is
what buys the asymptotic linearity the reported variance assumes.  Note the recursion
carries the *targeted* prediction forward, not the initial one: the outcome of step
:math:`t` is :math:`\bar Q^*_{t+1}`, so a residual left by one step is regressed away by
the next rather than accumulating.

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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, Learner
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import LongitudinalError
from ..fluctuation.iterative import Fluctuation, InitialFit, solve_fluctuation
from ..fluctuation.submodel import Submodel
from ..learners.crossfit import Folds
from ..utils.bounds import OutcomeScaler, bound
from .data import LongitudinalData
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

    ``treatment[t][label]`` is :math:`P(A_t = 1 \\mid H_t, \\bar A_{t-1} = \\bar a_{t-1})`
    with the earlier treatments set to what the regimen would have assigned, and
    ``censoring[t][label]`` is :math:`P(C_t = 1 \\mid H_t, \\bar A_t = \\bar a_t)`.  One
    model per node serves every regimen: the *fit* is shared, and only where it is
    evaluated differs.

    Both are indexed from zero by node, so ``treatment[0]`` is the mechanism at
    :math:`t = 1`.
    """

    treatment: tuple[dict[str, FloatArray], ...]
    censoring: tuple[dict[str, FloatArray], ...]

    def cumulative(
        self, data: LongitudinalData, plan: Plan, bounds: tuple[float, float]
    ) -> FloatArray:
        r"""``(n, T)`` cumulative product :math:`\prod_{s \le t} g_s c_s`, bounded.

        Each factor is truncated into ``bounds`` *before* multiplying rather than the
        product afterwards, so a single near-deterministic node cannot be rescued by
        the others -- a distinction that does not arise at one time point, where the
        two are the same operation.

        The arm is read per *unit*, since a dynamic rule assigns different units
        different arms at the same node.  Under a static plan the column is constant and
        the ``where`` picks the same branch for every row, which is why this is the old
        expression rather than a generalisation of it.
        """
        lower, upper = bounds
        running = np.ones(data.n)
        columns = []
        for time in range(1, data.n_times + 1):
            arm = plan.arm(time)
            # ``g1`` is clipped and the control arm is its complement, so the two sum to
            # one -- the same convention ``Propensity.bounded`` keeps at one time point.
            g1 = bound(self.treatment[time - 1][plan.label], lower, upper)
            running = running * np.where(arm == 1.0, g1, 1.0 - g1)
            if data.censoring_names:
                running = running * bound(self.censoring[time - 1][plan.label], lower, upper)
            columns.append(running.copy())
        return np.column_stack(columns)


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
    covariate the *update* is applied at.  ``clever`` is that masked down to the units
    that actually followed, which is the covariate the *score* is taken against.  The
    two differ exactly as ``submodel.arms[a]`` differs from ``submodel.observed`` at one
    time point, and reading the wrong one is the mistake that stops every node after the
    first from being updated at all.
    """

    time: int
    at_risk: BoolArray
    trained_on: BoolArray
    pseudo_outcome: FloatArray
    initial: FloatArray
    counterfactual: FloatArray
    clever: FloatArray


@dataclass(frozen=True)
class SequentialStep:
    """One node of the backward recursion, kept so a fit can be inspected node by node."""

    time: int
    #: Rows the regression was fitted on: followed the regimen and stayed under
    #: observation through this node.
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
        weights = self.leverage
        total = float(np.sum(weights))
        if total <= 0:
            return 0.0
        return float(total**2 / np.sum(weights**2))

    @property
    def converged(self) -> bool:
        return all(step.fluctuation.converged for step in self.steps)


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
    for time in range(1, data.n_times + 1):
        at_risk = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        designs = {plan.label: data.history_design(time, treatment=plan.values) for plan in plans}
        predictions, _ = cross_fit_predictions(
            treatment_learner,
            data.history_design(time),
            arm,
            data.weights,
            folds,
            task="classification",
            predict_designs=designs,
            fit_mask=at_risk,
            groups=data.cluster,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        treatment.append(predictions)

        if not data.censoring_names:
            censoring.append({plan.label: np.ones(data.n) for plan in plans})
            continue
        stayed = np.where(at_risk, data.uncensored[:, time - 1].astype(float), 0.0)
        censor_designs = {
            plan.label: data.history_design(time, treatment=plan.values, include_current=True)
            for plan in plans
        }
        predictions, _ = cross_fit_predictions(
            censoring_learner,
            data.history_design(time, include_current=True),
            stayed,
            data.weights,
            folds,
            task="classification",
            predict_designs=censor_designs,
            fit_mask=at_risk,
            groups=data.cluster,
            clip=(0.0, 1.0),
            n_jobs=n_jobs,
        )
        censoring.append(predictions)
    return Mechanism(tuple(treatment), tuple(censoring))


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
    n_jobs: int = 1,
) -> NodeInputs:
    """One node's masks, pseudo-outcome, regression and clever covariate.

    Everything the recursion does at a node *except* the fluctuation, which is split out
    because a working model over regimens pools that step across the declared plans and
    so has to hold every plan's regression at a node before any of them is updated.
    :func:`fit_regimen` calls this and fluctuates immediately, which is the recursion it
    always was.
    """
    at_risk = data.at_risk(plan.values, time)
    trained_on = data.following(plan.values, time)
    if not trained_on.any():
        raise LongitudinalError(
            f"no unit followed regimen {plan.label!r} through time {time} while "
            "remaining in the study, so the sequential regression there has nothing "
            "to fit. The regimen is not supported by this sample."
            + _risk_set_hint(data, plan, time)
            + _rule_hint(plan, at_risk, time)
        )
    if data.is_survival:
        # The numerator is *this* cause's event and the survival factor is
        # **all-cause**: a unit that left through a competing cause contributes a zero
        # here and carries nothing forward, because it is not going to have this
        # cause's event either.  Writing ``1 - event_by(time, cause)`` instead -- the
        # cause's own survival -- is the mistake competing risks invite, and it is
        # wrong by exactly the mass that left through the other causes.  With one
        # cause the two calls return the same array and this is the line it was.
        failed = data.event_by(time, cause)
        next_outcome = failed + (1.0 - data.event_by(time)) * carried
    else:
        next_outcome = carried
    design = data.covariate_history(time)
    learner = outcome_learner if time == horizon else pseudo_learner
    task = "classification" if time == horizon and data.family == "binomial" else "regression"
    if task == "classification":
        seen = np.unique(next_outcome[trained_on])
        if seen.size < 2:
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
    predictions, _ = cross_fit_predictions(
        learner,
        design,
        next_outcome,
        data.weights,
        folds,
        task=task,  # type: ignore[arg-type]
        predict_designs={"history": design},
        fit_mask=trained_on,
        groups=data.cluster,
        clip=(0.0, 1.0),
        n_jobs=n_jobs,
    )
    initial = np.where(at_risk, predictions["history"], _FILLER)

    denominator = np.where(at_risk, cumulative[:, time - 1], 1.0)
    counterfactual = np.where(at_risk, 1.0 / denominator, 0.0)
    clever = np.where(trained_on, counterfactual, 0.0)
    return NodeInputs(
        time=time,
        at_risk=at_risk,
        trained_on=trained_on,
        pseudo_outcome=next_outcome,
        initial=initial,
        counterfactual=counterfactual,
        clever=clever,
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
    cumulative = mechanism.cumulative(data, plan, g_bounds)
    carried = seed_carried(data, scaler)

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
            n_jobs=n_jobs,
        )
        fluctuation = solve_fluctuation(
            node.pseudo_outcome,
            InitialFit(node.initial, {_REGIMEN_ARM: node.initial}),
            Submodel(
                node.clever.reshape(-1, 1),
                {_REGIMEN_ARM: node.counterfactual.reshape(-1, 1)},
                (f"h[{plan.label}, t={time}]",),
                "sequential",
            ),
            data.weights,
            node.trained_on,
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
        cumulative=cumulative,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
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


def _rule_hint(plan: Plan, at_risk: BoolArray, time: int) -> str:
    """For an unsupported regimen, what the rule asked for where nobody was left.

    A static plan is unsupported because the sample happens not to contain the sequence.
    A rule can be unsupported because it asks for an arm nobody at risk received, and
    that is a different diagnosis -- so say which arm it wanted and how many units were
    there to give it to.
    """
    if isinstance(plan.regimen, Regimen):
        return ""
    if not plan.regimen.is_rule(time):
        return ""
    assigned = plan.arm(time)[at_risk]
    treated = int(np.sum(assigned == 1.0))
    return (
        f" The rule at time {time} assigned arm 1 to {treated} of the "
        f"{int(at_risk.sum())} unit(s) at risk there."
    )
