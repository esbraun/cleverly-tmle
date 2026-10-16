r"""Sequential g-computation and inverse probability weighting: the longitudinal baselines.

Both estimators target the parameters of :class:`~cleverly.longitudinal.LTMLE` under the same
identifying assumptions.  Neither solves the efficient influence-curve equation, so neither is
asymptotically linear with data-adaptive nuisances.  With every regression the estimator reads
a :class:`~cleverly.learners.GLM`, each is a regular parametric M-estimator, and its interval
comes from the stacked estimating equations of the GLM fits and the estimate.  That interval
covers the target only when every GLM it reads is correctly specified.

**Sequential g-computation** is the iterated conditional expectation form of the g-formula
(Bang and Robins 2005, Section 4): the untargeted backward recursion of
:func:`~cleverly.longitudinal.sequential.untargeted_recursion` on every row, and
:math:`\psi = P_n w \bar Q_1`.  Node :math:`t` solves

.. math::

    U_t(\beta_t, \beta_{t+1}) = P_n\bigl[w R_t \phi_t \{Z_t(\beta_{t+1})
        - m_t(\phi_t^\top \beta_t)\}\bigr] = 0,

where :math:`\phi_t` is the GLM's active design, :math:`m_t` its inverse canonical link,
:math:`R_t` the indicator of the node's fit rows, and :math:`Z_t` the pseudo-outcome built from
the later node.  Stacking :math:`(\beta_1, \ldots, \beta_T, \psi)` gives a block-triangular
Jacobian and the curve

.. math::

    \mathrm{IF} = w (\bar Q_1 - \psi)
        + \sum_t a_t^\top M_t^{-1} w R_t \phi_t (Z_t - m_t),

with :math:`M_t = P_n[w R_t m_t' \phi_t \phi_t^\top]`,
:math:`N_t = P_n[w R_t \phi_t (\partial Z_t / \partial \beta_{t+1})^\top]`,
:math:`a_1 = P_n[w\, \partial \bar Q_1 / \partial \beta_1]` and
:math:`a_{t+1}^\top = a_t^\top M_t^{-1} N_t`.

**Inverse probability weighting** reweights the followers by the cumulative ratio
:math:`\lambda_t = \prod_{s \le t} \pi_s / \prod_{s \le t} g_s c_s` that a targeted fit's
clever covariate holds.  The Horvitz-Thompson form is :math:`P_n[w \lambda_T R_T Y]`, and the
Hajek form divides by :math:`P_n[w \lambda_T R_T]`.  A survival risk is the sum of the
end-of-study identity over the nodes, and its Hajek form is the product of the IPW discrete
hazards (the Aalen-Johansen sum with competing causes).  Each estimated mechanism factor adds
the projection term :math:`b^\top I^{-1} S` of its GLM score :math:`S`, with
:math:`b = P_n[\partial (w \lambda R Y) / \partial \gamma]` (Lunceford and Davidian 2004,
Section 3.1).  A cumulative prefix that ``g_bounds`` moved has derivative zero, which is the
estimator's derivative as computed.

With another learner in a regression the estimator reads, the estimate stands and the curve is
a diagnostic: the efficient influence curve at the untargeted fits for g-computation, and the
fixed-mechanism curve for IPW.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray
from ..estimators._nuisance import ConstantFit
from ..exceptions import DataError
from ..learners._fitting import Task, as_target
from ..learners.glm import GLM
from ..utils.bounds import OutcomeScaler
from .data import LongitudinalData
from .regimen import Plan
from .sequential import (
    _FILLER,
    Mechanism,
    RegimenFit,
    SequentialStep,
    _clever_covariate,
    _finish_regimen_fit,
    _NodeRegression,
    constant_target,
    outcome_design,
    seed_carried,
    untargeted_recursion,
)

__all__ = [
    "IPW_NORMALIZATIONS",
    "fit_gcomp_cell",
    "fit_ipw_cell",
    "mechanism_reads_parametric",
    "reads_parametric",
]

#: The two IPW normalizations: the Hajek ratio, which is R ``ltmle``'s ``CalcIPTW``, and the
#: Horvitz-Thompson mean.
IPW_NORMALIZATIONS: tuple[str, ...] = ("hajek", "horvitz_thompson")
#: How far a mechanism score sum may sit from zero, relative to its absolute mass.
_SCORE_TOLERANCE = 1e-6


def reads_parametric(model: Any) -> bool:
    """Whether a fitted node model is one the stacked equations can read.

    Parameters
    ----------
    model : object
        A fitted model, a constant fit, or ``None`` at a node that fits none.

    Returns
    -------
    bool
        ``True`` for a :class:`~cleverly.learners.GLM`, a constant fit and ``None``.
    """
    return model is None or isinstance(model, (GLM, ConstantFit))


def mechanism_reads_parametric(mechanism: Mechanism) -> bool:
    """Whether every fitted factor of an IPW denominator is a GLM.

    Parameters
    ----------
    mechanism : Mechanism
        A mechanism fitted with ``keep_models=True``.

    Returns
    -------
    bool
        ``True`` when every kept treatment and censoring model is one the stacked equations
        read.
    """
    return all(reads_parametric(model) for model in mechanism.treatment_models) and all(
        reads_parametric(model) for model in mechanism.censoring_models
    )


# --------------------------------------------------------------------- g-computation


def fit_gcomp_cell(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    *,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    horizon: int,
    cause: str | None,
    outcome_learner: Any,
    pseudo_learner: Any,
    stacked: bool,
) -> RegimenFit:
    """One regimen's sequential g-computation estimate at one cause and horizon.

    The estimate is the first-node mean of the untargeted recursion on every row.  The
    mechanism enters only the clever covariate the diagnostics and the diagnostic curve read.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan.
    mechanism : Mechanism
        The diagnostic mechanism.
    scaler : OutcomeScaler
        The outcome transformation.
    g_bounds : tuple of float
        The cumulative bounds of the diagnostic clever covariate.
    horizon : int
        The node the parameter is indexed by.
    cause : str or None
        The absorbing cause, on a competing-risk fit.
    outcome_learner : object
        The regression at the horizon.
    pseudo_learner : object
        The regression at every earlier node.
    stacked : bool
        Whether the fit's status reads the stacked curve.  ``False`` gives the efficient
        influence curve at the untargeted fits, the diagnostic curve.

    Returns
    -------
    RegimenFit
        The estimate, its curve, and one step per node.

    Raises
    ------
    RuntimeError
        When ``stacked`` is ``True`` and a fitted node model is not one the stacked equations
        read.  The status and the curve would then disagree.
    """
    masks = plan.masks(data)
    nodes = untargeted_recursion(
        data,
        plan,
        masks=masks,
        horizon=horizon,
        scaler=scaler,
        cause=cause,
        outcome_learner=outcome_learner,
        pseudo_learner=pseudo_learner,
        keep_models=True,
    )
    nodes.reverse()
    for node in nodes:
        _refuse_aliased_extrapolation(data, plan, node)
    cumulative_unbounded, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    numerator = plan.cumulative_numerator(data)
    steps = []
    for node in nodes:
        _, clever = _clever_covariate(
            node.at_risk,
            node.trained_on,
            cumulative,
            node.time,
            numerator=None if numerator is None else numerator[:, node.time - 1],
        )
        steps.append(
            SequentialStep(
                time=node.time,
                trained_on=node.trained_on,
                at_risk=node.at_risk,
                pseudo_outcome=node.pseudo_outcome,
                initial=node.initial,
                targeted=node.initial,
                clever=clever,
                fluctuation=None,
                learner_diagnostics=node.learner_diagnostics,
                marginal=node.initial_marginal,
                targeted_by_arm=node.initial_by_arm,
                initial_by_arm=node.initial_by_arm,
            )
        )
    # The efficient influence curve at the untargeted fits: the diagnostic curve.
    fit = _finish_regimen_fit(
        data,
        plan,
        steps,
        cumulative_unbounded,
        cumulative,
        horizon=horizon,
        cause=cause,
        numerator=numerator,
    )
    if not stacked:
        return fit
    if not all(reads_parametric(node.model) for node in nodes):
        raise RuntimeError(
            "sequential g-computation: the status reads the stacked curve, but a fitted node "
            "model is not a GLM"
        )
    curve = stacked_gcomp_curve(data, plan, nodes, fit.psi_scaled, cause=cause)
    return _replace_curve(fit, curve)


def stacked_gcomp_curve(
    data: LongitudinalData,
    plan: Plan,
    nodes: Sequence[_NodeRegression],
    psi: float,
    *,
    cause: str | None,
) -> FloatArray:
    """The stacked estimating-equation curve of sequential g-computation.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan.
    nodes : sequence of node regressions
        The recursion's nodes in time order, each with its kept model.
    psi : float
        The estimate on the ``[0, 1]`` scale.
    cause : str or None
        The absorbing cause, on a competing-risk fit.

    Returns
    -------
    FloatArray
        The weighted curve on the ``[0, 1]`` scale, one value per row.
    """
    n = data.n
    weights = np.asarray(data.weights, dtype=float)
    first = nodes[0]
    carried = first.initial if first.initial_marginal is None else first.initial_marginal
    curve = weights * (carried - psi)
    derivative = _carried_derivative(data, plan, first)
    if derivative is None:
        return np.asarray(curve, dtype=float)
    chain = (weights[:, None] * derivative).sum(axis=0) / n
    for position, node in enumerate(nodes):
        model = node.model
        assert isinstance(model, GLM)
        design = outcome_design(data, plan, node.time)
        fit_weights = np.where(node.fitted_on, weights, 0.0)
        target = as_target(node.pseudo_outcome, _node_task(data, node.time, nodes[-1].time))
        information = model.information(design, fit_weights) / n
        projection = _solve(information, chain)
        curve = curve + model.score_contributions(design, target, fit_weights) @ projection
        if position + 1 == len(nodes):
            break
        later = _carried_derivative(data, plan, nodes[position + 1])
        if later is None:
            # The later node read a constant target, so this node's pseudo-outcome does not
            # move with any coefficient, and the chain ends here.
            break
        # The factor (1 - Y_t) is redundant: ``later`` is zero off the rows at risk at
        # t + 1, which excludes every row with an event at t.  It states the survival
        # pseudo-outcome Z_t = Y_t + (1 - Y_t) Q_{t+1} rather than relying on that zero.
        factor = 1.0 - data.event_by(node.time) if data.is_survival else np.ones(n)
        link = model.design(design).T @ ((fit_weights * factor)[:, None] * later) / n
        chain = link.T @ projection
    return np.asarray(curve, dtype=float)


def _node_task(data: LongitudinalData, time: int, horizon: int) -> Task:
    """The task the node's regression was fitted under, as ``_fit_node_regression`` sets it."""
    return "classification" if time == horizon and data.family == "binomial" else "regression"


def _carried_derivative(
    data: LongitudinalData, plan: Plan, node: _NodeRegression
) -> FloatArray | None:
    """``(n, p)`` derivative of what node ``t`` carries back, in its coefficients.

    At a label or rule node it is the mean Jacobian at the observed history.  At a policy node
    it is the policy-weighted sum of the Jacobians at each level, or at each branch dose of a
    continuous node.  A prediction the ``(0, 1)`` clip moved has derivative zero.  ``None`` at
    a node that read a constant target and fitted no model.
    """
    model = node.model
    if model is None or isinstance(model, ConstantFit):
        return None
    assert isinstance(model, GLM)
    if not plan.is_policy_node(node.time):
        return _clipped_jacobian(model, outcome_design(data, plan, node.time), node.at_risk)
    policy = plan.policy_at(node.time)
    total: FloatArray | None = None
    for code in range(policy.shape[1]):
        design = outcome_design(data, plan, node.time, arm=code)
        term = policy[:, code][:, None] * _clipped_jacobian(model, design, node.at_risk)
        total = term if total is None else total + term
    assert total is not None
    return total


def _prediction_designs(
    data: LongitudinalData, plan: Plan, node: _NodeRegression
) -> list[tuple[FloatArray, BoolArray]]:
    """Each design a node predicts at, with the rows that read it.

    A label or rule node predicts at its design on the rows at risk.  A policy node predicts at
    each level, or each branch dose of a continuous node, on the rows that the policy sends
    there with positive probability.
    """
    at_risk = np.asarray(node.at_risk, dtype=bool)
    if not plan.is_policy_node(node.time):
        return [(outcome_design(data, plan, node.time), at_risk)]
    policy = plan.policy_at(node.time)
    return [
        (outcome_design(data, plan, node.time, arm=code), at_risk & (policy[:, code] > 0.0))
        for code in range(policy.shape[1])
    ]


def _refuse_aliased_extrapolation(
    data: LongitudinalData, plan: Plan, node: _NodeRegression
) -> None:
    """Refuse a node whose ``GLM`` predicts through a column its fit rows aliased.

    A covariate that is constant, or a combination of other columns, on the followers has no
    coefficient.  Where a row at risk departs from that relation, the prediction treats the
    column as having no effect.  No unit with that history follows the plan, so the regimen
    is not identified under the model.
    """
    model = node.model
    if not isinstance(model, GLM):
        return
    leaving = np.zeros(data.n, dtype=bool)
    for design, rows in _prediction_designs(data, plan, node):
        leaving |= rows & model.outside_span(design)
    if np.any(leaving):
        raise DataError(
            f"regimen {plan.label!r} is not identified by sequential g-computation at node "
            f"{data.decision_name(node.time)!r}: the GLM's fit rows hold a covariate column "
            "that is constant or a combination of other columns, so it has no coefficient, "
            f"and {int(np.count_nonzero(leaving))} row(s) at risk depart from that relation. "
            "No unit with those histories follows the regimen, so its mean would assume the "
            "column has no effect. Remove the regimen, or remove the column from the design"
        )


def _clipped_jacobian(model: GLM, design: FloatArray, at_risk: BoolArray) -> FloatArray:
    """The mean Jacobian at ``design``, zero where the clip moved the prediction or off risk."""
    jacobian = np.asarray(model.mean_jacobian(design), dtype=float)
    keep = np.asarray(at_risk, dtype=bool)
    if model.family_ == "gaussian":
        mean = model.predict(design)
        keep = keep & (mean >= 0.0) & (mean <= 1.0)
    return np.where(keep[:, None], jacobian, 0.0)


def _replace_curve(fit: RegimenFit, curve: FloatArray) -> RegimenFit:
    return replace(fit, influence_curve_scaled=curve)


def _solve(information: FloatArray, vector: FloatArray) -> FloatArray:
    """``information^{-1} vector``, refusing a singular information matrix."""
    try:
        return np.asarray(np.linalg.solve(information, vector), dtype=float)
    except np.linalg.LinAlgError as error:
        raise DataError(
            "a GLM's information matrix is singular, so its stacked estimating equation has "
            "no solution. Remove the collinear design columns"
        ) from error


# ------------------------------------------------------------------------------- IPW


@dataclass(frozen=True)
class _Factor:
    """One estimated mechanism factor: its score rows and its log-derivative at the follower.

    Parameters
    ----------
    node : int
        The node, counted from one.
    projected : FloatArray
        ``(n, p)`` score rows, already multiplied by the inverse information:
        :math:`S I^{-1}`, with :math:`I` over ``n``.
    log_derivative : FloatArray
        ``(n, p)`` derivative of the factor's log at each row's observed value.
    """

    node: int
    projected: FloatArray
    log_derivative: FloatArray


def _mechanism_factors(data: LongitudinalData, mechanism: Mechanism) -> list[_Factor]:
    """Every estimated GLM factor of the IPW denominator, with its projected score rows."""
    n = data.n
    weights = np.asarray(data.weights, dtype=float)
    masks = data.regimen_masks(data.treatment)
    factors: list[_Factor] = []
    ones = np.ones(n)
    for time in range(1, data.n_times + 1):
        at_risk = masks.uncensored[:, time - 1] & masks.event_free[:, time - 1]
        fit_weights = np.where(at_risk, weights, 0.0)
        models = mechanism.treatment_models
        model = models[time - 1] if models else None
        if isinstance(model, GLM):
            design = data.history_design(time)
            # A GLM treatment model is fitted at a categorical node only, and ``node_codes``
            # refuses a mixed node rather than read its ``0.0`` slot.
            arm = as_target(np.nan_to_num(data.node_codes(time), nan=0.0), "classification")
            factors.append(
                _factor(time, model, design, arm, fit_weights, ones, n),
            )
        censoring = mechanism.censoring_models
        model = censoring[time - 1] if censoring else None
        if isinstance(model, GLM):
            design = data.history_design(time, include_current=True)
            stayed = as_target(
                np.where(at_risk, data.uncensored[:, time - 1].astype(float), 0.0),
                "classification",
            )
            factors.append(_factor(time, model, design, stayed, fit_weights, ones, n))
    return factors


def _factor(
    time: int,
    model: GLM,
    design: FloatArray,
    label: FloatArray,
    fit_weights: FloatArray,
    ones: FloatArray,
    n: int,
) -> _Factor:
    information = model.information(design, fit_weights) / n
    score = model.score_contributions(design, label, fit_weights)
    # The rows, design and label are rebuilt here, not read from the fit.  At the fitted
    # coefficients the score sums to zero on the fit's own rows only, so a drift between the
    # two constructions shows here rather than as a wrong curve.
    total = np.abs(score.sum(axis=0))
    if np.any(total > _SCORE_TOLERANCE * np.maximum(1.0, np.abs(score).sum(axis=0))):
        raise RuntimeError(
            f"the GLM mechanism at node {time} does not solve its score equation on the rows "
            "the stacked curve reads"
        )
    projected = _solve(information, score.T).T
    return _Factor(
        node=time,
        projected=np.asarray(projected, dtype=float),
        log_derivative=np.asarray(model.score_contributions(design, label, ones), dtype=float),
    )


class _Means:
    """Sample means of weighted IPW summands, each with its stacked (or fixed-mechanism) curve.

    A summand :math:`x = w u` at node ``t`` is a function of the cumulative ratio through
    ``t``.  The observation weights have mean one, so :math:`P_n x` is the mean of ``u`` in the
    weighted law, and its curve is :math:`w (u - P_n x)`, as every weighted curve here is.
    The stacked curve adds :math:`b^\\top I^{-1} S` over every factor at a node up to ``t``,
    with :math:`b = -P_n[x\\, \\partial \\log(g_s c_s)]` on the rows whose prefix through
    ``t`` the bounds left unmoved.
    """

    def __init__(
        self,
        data: LongitudinalData,
        factors: Sequence[_Factor] | None,
        unbounded: FloatArray,
        bounded: FloatArray,
    ) -> None:
        self.n = data.n
        self.weights = np.asarray(data.weights, dtype=float)
        self.factors = factors
        self.free = unbounded == bounded

    def __call__(self, summand: FloatArray, time: int) -> tuple[float, FloatArray]:
        mean = float(np.mean(summand))
        curve = summand - self.weights * mean
        if self.factors is None:
            return mean, np.asarray(curve, dtype=float)
        moving = np.where(self.free[:, time - 1], summand, 0.0)
        for factor in self.factors:
            if factor.node > time:
                continue
            slope = -(moving[:, None] * factor.log_derivative).sum(axis=0) / self.n
            curve = curve + factor.projected @ slope
        return mean, np.asarray(curve, dtype=float)


def fit_ipw_cell(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    *,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    horizon: int,
    cause: str | None,
    normalization: str,
    stacked: bool,
) -> RegimenFit:
    """One regimen's IPW estimate at one cause and horizon.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan.
    mechanism : Mechanism
        The fitted mechanism, with its models kept.
    scaler : OutcomeScaler
        The outcome transformation.
    g_bounds : tuple of float
        The bounds applied to each cumulative mechanism probability.
    horizon : int
        The node the parameter is indexed by.
    cause : str or None
        The absorbing cause, on a competing-risk fit.
    normalization : {"hajek", "horvitz_thompson"}
        How the weighted sum is normalized.
    stacked : bool
        Whether the curve carries the mechanism terms.  ``False`` gives the fixed-mechanism
        diagnostic curve.

    Returns
    -------
    RegimenFit
        The estimate, its curve, and one step per node with the node's clever covariate.
    """
    weights = np.asarray(data.weights, dtype=float)
    masks = plan.masks(data)
    numerator = plan.cumulative_numerator(data)
    unbounded, bounded = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    factors = _mechanism_factors(data, mechanism) if stacked else None
    means = _Means(data, factors, unbounded, bounded)
    survival = data.is_survival
    clever: dict[int, FloatArray] = {}
    terms: dict[int, FloatArray] = {}
    constant: list[int] = []
    for time in range(1, horizon + 1):
        _, clever[time] = _clever_covariate(
            masks.at_risk(time),
            masks.following(time),
            bounded,
            time,
            numerator=None if numerator is None else numerator[:, time - 1],
        )
        if survival:
            terms[time] = data.event_by(time, cause)
            followers = masks.following(time)
            # The all-cause check is conservative for Horvitz-Thompson, which reads the
            # cause term only.  The Hajek hazard product reads both.
            if constant_target(terms[time], followers) is not None or (
                cause is not None and constant_target(data.event_by(time), followers) is not None
            ):
                constant.append(time)
    if not survival:
        terms[horizon] = seed_carried(data, scaler, horizon=horizon)
        if constant_target(terms[horizon], masks.following(horizon)) is not None:
            constant.append(horizon)
        psi, curve = _end_of_study(
            weights, clever[horizon], terms[horizon], horizon, means, normalization, scaler
        )
    elif normalization == "horvitz_thompson":
        # A sum of end-of-study identities, one per node, each with its own prefix.
        psi, curve = 0.0, np.zeros(data.n)
        for time in range(1, horizon + 1):
            term, term_curve = means(weights * clever[time] * terms[time], time)
            psi, curve = psi + term, curve + term_curve
    else:
        psi, curve = _hazard_product(data, weights, clever, terms, horizon, cause, means)
    steps = tuple(
        SequentialStep(
            time=time,
            trained_on=masks.following(time),
            at_risk=masks.at_risk(time),
            pseudo_outcome=terms.get(time, np.zeros(data.n)),
            initial=np.full(data.n, _FILLER),
            targeted=np.full(data.n, _FILLER),
            clever=clever[time],
            fluctuation=None,
        )
        for time in range(1, horizon + 1)
    )
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=curve,
        horizon=horizon,
        cause=cause,
        steps=steps,
        cumulative_unbounded=unbounded,
        cumulative=bounded,
        assignment=np.asarray(plan.values),
        obs_weights=weights,
        policy=plan.policy,
        cumulative_numerator=numerator,
        constant_terms=tuple(constant),
    )


def _end_of_study(
    weights: FloatArray,
    clever: FloatArray,
    outcome: FloatArray,
    horizon: int,
    means: _Means,
    normalization: str,
    scaler: OutcomeScaler,
) -> tuple[float, FloatArray]:
    """The end-of-study IPW estimate and its curve, on the scaled outcome.

    The scaled Horvitz-Thompson sum :math:`P_n[w \\lambda R (Y - a)] / (b - a)` maps back to
    :math:`P_n[w \\lambda R Y] - a (P_n[w \\lambda R] - 1)`, which moves with :math:`a`.  The
    offset :math:`a (P_n[w \\lambda R] - 1) / (b - a)` restores the outcome-scale sum, and its
    curve is the mass curve times :math:`a / (b - a)`.  Hajek is invariant to the offset.
    """
    weighted = weights * clever
    total, total_curve = means(weighted * np.where(clever > 0.0, outcome, 0.0), horizon)
    mass, mass_curve = means(weighted, horizon)
    if normalization == "horvitz_thompson":
        offset = scaler.lower / scaler.range
        return total + offset * (mass - 1.0), np.asarray(
            total_curve + offset * mass_curve, dtype=float
        )
    psi = total / mass
    return psi, np.asarray((total_curve - psi * mass_curve) / mass, dtype=float)


def _hazard_product(
    data: LongitudinalData,
    weights: FloatArray,
    clever: dict[int, FloatArray],
    terms: dict[int, FloatArray],
    horizon: int,
    cause: str | None,
    means: _Means,
) -> tuple[float, FloatArray]:
    """The Hajek survival risk or cause incidence: the IPW hazards in the Aalen-Johansen sum.

    The cause hazard :math:`h_t` and the all-cause hazard :math:`H_t` share the denominator
    :math:`P_n[w \\lambda_t R_t]`.  The incidence is
    :math:`F(k) = \\sum_{t \\le k} S(t - 1) h_t` with
    :math:`S(t) = \\prod_{s \\le t} (1 - H_s)`.  With one cause :math:`h = H` and
    :math:`F(k) = 1 - S(k)`.
    """
    cause_hazard: list[tuple[float, FloatArray]] = []
    all_hazard: list[tuple[float, FloatArray]] = []
    for time in range(1, horizon + 1):
        weighted = weights * clever[time]
        mass, mass_curve = means(weighted, time)
        events, events_curve = means(weighted * terms[time], time)
        hazard = events / mass
        cause_hazard.append((hazard, (events_curve - hazard * mass_curve) / mass))
        if cause is None:
            all_hazard.append(cause_hazard[-1])
            continue
        every, every_curve = means(weighted * data.event_by(time), time)
        total = every / mass
        all_hazard.append((total, (every_curve - total * mass_curve) / mass))
    psi = 0.0
    curve = np.zeros(data.n)
    survival = 1.0
    survival_curve = np.zeros(data.n)
    for (hazard, hazard_curve), (total, total_curve) in zip(cause_hazard, all_hazard, strict=True):
        psi += survival * hazard
        curve = curve + survival * hazard_curve + hazard * survival_curve
        survival_curve = (1.0 - total) * survival_curve - survival * total_curve
        survival *= 1.0 - total
    return float(psi), np.asarray(curve, dtype=float)
