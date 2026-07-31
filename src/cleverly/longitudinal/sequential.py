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
    "RegimenFit",
    "SequentialStep",
    "fit_mechanism",
    "fit_regimen",
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
class SequentialStep:
    """One node of the backward recursion, kept so a fit can be inspected node by node."""

    time: int
    #: Rows the regression was fitted on: followed the regimen and stayed under
    #: observation through this node.
    trained_on: BoolArray
    #: Rows whose history at this node is observed and regimen-consistent -- the set the
    #: regression *predicts* for, and the population the assigned arm is a statement
    #: about.  Equal to the previous node's ``trained_on``, which is what closes the
    #: recursion.
    at_risk: BoolArray
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
    steps: tuple[SequentialStep, ...]
    cumulative: FloatArray
    #: The ``(n, T)`` arms this regimen assigned *this* sample.  Constant down each
    #: column for a static plan; for a rule it is the thing ``diagnostics()`` reports,
    #: since what share of the at-risk units a rule would treat is a property of the
    #: data rather than of the declaration.
    assignment: FloatArray

    @property
    def max_weight(self) -> float:
        """Largest clever-covariate value at the final node.

        The reciprocal of the smallest cumulative probability of following the regimen,
        so it is the leverage a single unit can have on the estimate.
        """
        weights = self.steps[-1].clever
        return float(np.max(weights)) if weights.size else float("nan")

    @property
    def effective_n(self) -> float:
        """Kish effective sample size of the final node's clever covariate.

        How many units the estimate is really averaging over once the weighting by
        :math:`1 / \\prod g` is taken into account.  A number far below ``n`` says the
        regimen is supported by few units, whatever the reported standard error.
        """
        weights = self.steps[-1].clever
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

    Each node's model is fitted on the units still under observation *before* that
    node's decision -- not on the regimen's followers.  The conditioning set carries the
    earlier treatments as columns, so one model answers for every regimen and is simply
    evaluated at each one's arms.  Under a dynamic rule those arms differ by row, which
    changes where the model is *evaluated* and nothing about how it is fitted.
    """
    treatment: list[dict[str, FloatArray]] = []
    censoring: list[dict[str, FloatArray]] = []
    for time in range(1, data.n_times + 1):
        at_risk = data.uncensored_through(time - 1)
        arm = np.nan_to_num(data.treatment[:, time - 1], nan=0.0)
        designs = {plan.label: data.history_design(time, treatment=plan.values) for plan in plans}
        predictions, _ = cross_fit_predictions(
            treatment_learner,
            data.history_design(time),
            arm,
            np.ones(data.n),
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
            np.ones(data.n),
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
    """
    cumulative = mechanism.cumulative(data, plan, g_bounds)
    observed_outcome = data.uncensored_through(data.n_times)
    scaled = np.where(observed_outcome, scaler.scale(np.nan_to_num(data.outcome, nan=0.0)), _FILLER)
    next_outcome = np.clip(scaled, 0.0, 1.0)

    steps: list[SequentialStep] = []
    for time in range(data.n_times, 0, -1):
        at_risk = data.at_risk(plan.values, time)
        trained_on = data.following(plan.values, time)
        if not trained_on.any():
            raise LongitudinalError(
                f"no unit followed regimen {plan.label!r} through time {time} while "
                "remaining under observation, so the sequential regression there has "
                "nothing to fit. The regimen is not supported by this sample."
                + _rule_hint(plan, at_risk, time)
            )
        design = data.covariate_history(time)
        learner = outcome_learner if time == data.n_times else pseudo_learner
        task = (
            "classification" if time == data.n_times and data.family == "binomial" else "regression"
        )
        predictions, _ = cross_fit_predictions(
            learner,
            design,
            next_outcome,
            np.ones(data.n),
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

        fluctuation = solve_fluctuation(
            next_outcome,
            InitialFit(initial, {_REGIMEN_ARM: initial}),
            Submodel(
                clever.reshape(-1, 1),
                {_REGIMEN_ARM: counterfactual.reshape(-1, 1)},
                (f"h[{plan.label}, t={time}]",),
                "sequential",
            ),
            np.ones(data.n),
            trained_on,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        targeted = fluctuation.targeted.arms[_REGIMEN_ARM]
        steps.append(
            SequentialStep(
                time=time,
                trained_on=trained_on,
                at_risk=at_risk,
                initial=initial,
                targeted=targeted,
                clever=clever,
                fluctuation=fluctuation,
            )
        )
        next_outcome = np.where(at_risk, targeted, _FILLER)

    steps.reverse()
    psi = float(np.mean(steps[0].targeted))
    influence = steps[0].targeted - psi
    final = np.clip(scaled, 0.0, 1.0)
    for index, step in enumerate(steps):
        later = steps[index + 1].targeted if index + 1 < len(steps) else final
        influence = influence + step.clever * (later - step.targeted)
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=influence,
        steps=tuple(steps),
        cumulative=cumulative,
        assignment=np.asarray(plan.values),
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
