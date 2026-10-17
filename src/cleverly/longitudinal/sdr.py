r"""Sequentially doubly robust estimation of a longitudinal regimen.

The estimator of Díaz, Williams, Hoffman and Schenck (2023, *JASA* 118(542), Section 5.3).
It shares every nuisance with :mod:`cleverly.longitudinal.sequential` and fits no
fluctuation.  Write :math:`\pi_t` for the intervention density at the observed arm (the
indicator of the assigned arm at a label or rule node, :math:`q_t` at a policy node, and the
policy numerator at a modified treatment policy node), :math:`p_t = g_t c_t` for the node's
mechanism at the observed arm on the plan, and :math:`\check m_t` for the node regression.
The transformed outcome of Equation (4) is

.. math::

    \varphi_t = v_t + \sum_{s=t}^{h} W_{t,s} \bigl(Z_s - \check m_s(A_s, H_s)\bigr),
    \qquad
    W_{t,s} = \frac{\prod_{k=t}^{s} \pi_k}{\operatorname{bound}\bigl(\prod_{k=t}^{s} p_k\bigr)}

on the rows that follow the plan through :math:`s`, and zero elsewhere.  Here :math:`v_t` is
the value the node carries (the prediction at the plan's arm, or the policy-weighted mean of
the per-arm predictions at a policy node), and :math:`Z_s` is :math:`v_{s+1}` composed with
the node's event, with :math:`v_{h+1}` the seed of
:func:`~cleverly.longitudinal.sequential.seed_carried`.  The regression at node
:math:`t < h` is fitted to :math:`\varphi_{t+1}` composed with the node's event, and the
regression at the horizon to the outcome itself.  The estimate is the weighted mean of
:math:`\varphi_1`, and the curve is :math:`w(\varphi_1 - \hat\psi)`.

``g_bounds`` bounds each partial path probability :math:`\prod_{k=t}^{s} p_k`.  At
:math:`t = 1` that is the bounded cumulative product a targeted fit divides by, equal to
rounding (the factors multiply in another order).
Where no bound binds, :math:`W_{t,s}` is a product of per-node ratios and Equation (4),
Lemmas 3 and 4 and Theorem 4 of the paper apply.  Where one binds, the bounded weight acts as
a misspecified ratio on that path.  The share of such paths is recorded per node, over the
paths the estimate reads: a cross-fitted fit counts each fold's held-out rows.

``ratio_trim=q`` trims each per-node ratio :math:`\pi_k / p_k` at :math:`Q_q` before the
partial products (:func:`trimmed_factors`), so the unbounded weight is the ``lmtp`` weight
:math:`\prod_{k=t}^{s} \min(r_k, Q_q)`.  The threshold is one per plan, the quantile of the
pooled out-of-fold ratios (:mod:`cleverly.longitudinal.ratio_trim`), and every fold trims
its own factors at it.  So a fold's transform reads the held-out rows through that one number,
as a continuous node's bin edges do.

**Cross-fitting is the training-fold recursion.**  Fold :math:`k` fits every nuisance on its
training rows: the mechanism (``fit_mechanism(fit_rows=...)``), and the backward recursion
with its own outcome fits and its own mechanism in every training row's transform.  Its
held-out rows take :math:`\varphi_1` from fold :math:`k`'s models.  So each fold's transform
is fixed given its training rows, which is the conditional-independence step of the proof of
the paper's Theorem 3 (supplement Section 6).  One exception: a continuous node's fold density
bins on the out-of-fold fit's edges, which are quantiles of the dose over every row at risk.
Those quantiles read the held-out doses, so the fold's transform depends on them through the
edges alone.  The printed Step 2 of Section 5.3 composes a
training row's transform from the folds that hold it out, and ``lmtp`` reads a training row's
ratio from the out-of-fold mechanism.  Neither is fixed given the training rows.  With
fold-free ratios (known, or declared), the training-fold recursion equals ``lmtp::cf_sdr``.
The estimate pools every row: :math:`\hat\psi = n^{-1}\sum_i w_i \varphi_1(Z_i;
\hat\eta_{j(i)})`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
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
from .regimen import Plan, attach_policy_numerators
from .sequential import (
    _FILLER,
    Mechanism,
    RegimenFit,
    SequentialStep,
    _clever_covariate,
    _fit_node_regression,
    _pseudo_outcome,
    constant_target,
    fit_mechanism,
    seed_carried,
    tuning_term,
)

__all__ = [
    "FoldMechanism",
    "SDRCell",
    "fit_cells_sdr",
    "fit_fold_mechanisms",
    "node_factors",
    "partial_weights",
]


@dataclass(frozen=True)
class SDRCell:
    """One sequentially doubly robust recursion: a regimen, a cause and a horizon.

    Parameters
    ----------
    label : str
        The regimen label.
    cause : str or None
        The absorbing cause, on a competing-risk fit.
    horizon : int
        The node the recursion starts from.
    """

    label: str
    cause: str | None
    horizon: int


@dataclass(frozen=True)
class _Node:
    """One node of one fold's recursion, on every row."""

    time: int
    at_risk: BoolArray
    trained_on: BoolArray
    regression_target: FloatArray
    residual_target: FloatArray
    observed: FloatArray
    value: FloatArray
    by_arm: FloatArray | None
    transform: FloatArray
    clever: FloatArray
    diagnostics: tuple[SuperLearnerDiagnostics, ...]
    constant: bool
    #: Per row, the followed ``(row, s)`` paths from this node and those the bounds moved.
    binding: tuple[IntArray, IntArray]
    #: ``(n, 2)`` predictions at a population-tuned node's boundary and last moved dose.
    probes: FloatArray | None = None


def node_factors(data: LongitudinalData, plan: Plan, mechanism: Mechanism) -> FloatArray:
    r"""``(n, T)`` per-node mechanism :math:`p_t = g_t c_t` at the observed arm on the plan.

    The factors :meth:`~cleverly.longitudinal.sequential.Mechanism.cumulative_with_unbounded`
    multiplies, unbounded.  At a continuous node the treatment factor is one, because the
    density ratio is the whole treatment factor and sits in the numerator.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan.
    mechanism : Mechanism
        The fitted mechanism.

    Returns
    -------
    FloatArray
        One column per node.
    """
    columns = []
    for time in range(1, data.n_times + 1):
        factor = np.asarray(mechanism.treatment[time - 1][plan.label], dtype=float)
        if data.censoring_names:
            factor = factor * np.asarray(mechanism.censoring[time - 1][plan.label], dtype=float)
        columns.append(factor)
    return np.column_stack(columns)


def trimmed_factors(numerator: FloatArray, factors: FloatArray, threshold: float) -> FloatArray:
    r"""``(n, T)`` per-node mechanism that a ratio trim at ``threshold`` implies.

    Where the per-node ratio :math:`r_t = \pi_t / p_t` lies above :math:`Q_q`, the factor
    becomes :math:`\pi_t / Q_q`, so that :math:`\pi_t / \tilde p_t = \min(r_t, Q_q)`.  A
    partial product of the trimmed ratios is then the ``lmtp`` weight
    :math:`\prod_{k=t}^{s} \min(r_k, Q_q)` (``eif.R``, ``compute_weights``) wherever the
    bounds do not bind.  An entry with zero density or a zero factor keeps its factor.

    Parameters
    ----------
    numerator : FloatArray
        ``(n, T)`` intervention density :math:`\pi_t` at the observed arm.
    factors : FloatArray
        ``(n, T)`` unbounded per-node mechanism :math:`p_t`.
    threshold : float
        The trim threshold :math:`Q_q`, above zero.

    Returns
    -------
    FloatArray
        The factors, raised to :math:`\pi_t / Q_q` where the ratio lies above the threshold.
    """
    positive = (numerator > 0.0) & (factors > 0.0)
    ratio = np.where(positive, numerator / np.where(positive, factors, 1.0), 0.0)
    return np.where(positive & (ratio > threshold), numerator / threshold, factors)


def partial_weights(
    numerator: FloatArray,
    factors: FloatArray,
    masks: RegimenMasks,
    start: int,
    stop: int,
    bounds: tuple[float, float],
) -> tuple[list[FloatArray], IntArray, IntArray]:
    r"""The weights :math:`W_{t,s}` for ``s = start .. stop``, and the binding counts per row.

    Parameters
    ----------
    numerator : FloatArray
        ``(n, T)`` intervention density :math:`\pi_s` at the observed arm.
    factors : FloatArray
        ``(n, T)`` unbounded per-node mechanism :math:`p_s`.
    masks : RegimenMasks
        The plan's prefix scans.
    start : int
        The node :math:`t` the paths start at, counted from one.
    stop : int
        The last node :math:`s`, the horizon.
    bounds : tuple of float
        The bound pair applied to each partial product.

    Returns
    -------
    tuple
        One ``(n,)`` weight per ``s`` in order; per row, the number of followed
        ``(row, s)`` paths, and the number of those whose partial product the bounds moved.
    """
    lower, upper = bounds
    running_numerator = np.ones(numerator.shape[0])
    running_factor = np.ones(numerator.shape[0])
    weights = []
    paths = np.zeros(numerator.shape[0], dtype=np.intp)
    binding = np.zeros(numerator.shape[0], dtype=np.intp)
    for node in range(start, stop + 1):
        running_numerator = running_numerator * numerator[:, node - 1]
        running_factor = running_factor * factors[:, node - 1]
        bounded = bound(running_factor, lower, upper)
        following = masks.following(node)
        weights.append(np.where(following, running_numerator / bounded, 0.0))
        paths += following
        binding += following & (bounded != running_factor)
    return weights, paths, binding


def _recursion(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    masks: RegimenMasks,
    cell: SDRCell,
    *,
    scaler: OutcomeScaler,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    g_bounds: tuple[float, float],
    fit_rows: BoolArray | None,
    outer_fold: int | None,
    trim_threshold: float | None = None,
    carry_plan: Plan | None = None,
) -> list[_Node]:
    """One cell's recursion with one fold's nuisances, on every row, in time order.

    ``carry_plan`` is a cross-fitted population-target plan's map at the fold's own
    training tuner :math:`\\theta^{(-k)}`, with the fold mechanism's numerators.  The
    regressions are then fitted to the transform under ``carry_plan``, so they are fixed
    given the fold's training rows, and the returned arrays, which the held-out rows read,
    are those under ``plan``, the map at :math:`\\theta_n`.
    """
    single = Folds.single(data.n)
    numerator = plan.intervention_density(data)
    raw_factors = node_factors(data, plan, mechanism)
    factors = raw_factors
    if trim_threshold is not None:
        factors = trimmed_factors(numerator, raw_factors, trim_threshold)
    _, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    cumulative_numerator = plan.cumulative_numerator(data)
    carry_numerator = None if carry_plan is None else carry_plan.intervention_density(data)
    # The training transform's ratios are the fold map's own, trimmed at the same threshold.
    carry_factors = raw_factors
    if trim_threshold is not None and carry_numerator is not None:
        carry_factors = trimmed_factors(carry_numerator, raw_factors, trim_threshold)
    horizon = cell.horizon
    seed = seed_carried(data, scaler, horizon=horizon)
    values: dict[int, FloatArray] = {horizon + 1: seed}
    carry_values: dict[int, FloatArray] = {horizon + 1: seed}
    observed: dict[int, FloatArray] = {}
    residual: dict[int, FloatArray] = {}
    carry_residual: dict[int, FloatArray] = {}
    carried = seed
    nodes: list[_Node] = []
    for time in range(horizon, 0, -1):
        regression = _fit_node_regression(
            data,
            plan,
            carried,
            time,
            horizon,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            folds=single,
            cause=cell.cause,
            masks=masks,
            fit_rows=fit_rows,
            outer_fold=outer_fold,
            clip=None,
            carry_plan=carry_plan,
        )
        at_risk = regression.at_risk
        value = (
            regression.initial
            if regression.initial_marginal is None
            else regression.initial_marginal
        )
        values[time] = np.where(at_risk, value, _FILLER)
        observed[time] = regression.initial
        with phase("pseudo_outcome"):
            residual[time] = _pseudo_outcome(data, values[time + 1], time, cell.cause)
        with phase("clever_covariate"):
            weights, paths, binding = partial_weights(
                numerator, factors, masks, time, horizon, g_bounds
            )
            transform = np.array(values[time], dtype=float, copy=True)
            for node, weight in zip(range(time, horizon + 1), weights, strict=True):
                transform = transform + weight * (residual[node] - observed[node])
            transform = np.where(at_risk, transform, _FILLER)
            training = transform
            if carry_numerator is not None:
                # The fold's own map: what its earlier regression is fitted to.
                carry_value = (
                    value if regression.carry_marginal is None else regression.carry_marginal
                )
                carry_values[time] = np.where(at_risk, carry_value, _FILLER)
                carry_residual[time] = _pseudo_outcome(
                    data, carry_values[time + 1], time, cell.cause
                )
                carry_weights, _, _ = partial_weights(
                    carry_numerator, carry_factors, masks, time, horizon, g_bounds
                )
                training = np.array(carry_values[time], dtype=float, copy=True)
                for node, weight in zip(range(time, horizon + 1), carry_weights, strict=True):
                    training = training + weight * (carry_residual[node] - observed[node])
                training = np.where(at_risk, training, _FILLER)
            _, clever = _clever_covariate(
                at_risk,
                regression.trained_on,
                cumulative,
                time,
                numerator=None
                if cumulative_numerator is None
                else cumulative_numerator[:, time - 1],
            )
        nodes.append(
            _Node(
                time=time,
                at_risk=at_risk,
                trained_on=regression.trained_on,
                regression_target=regression.pseudo_outcome,
                residual_target=residual[time],
                observed=regression.initial,
                value=values[time],
                by_arm=regression.initial_by_arm,
                transform=transform,
                clever=clever,
                diagnostics=regression.learner_diagnostics,
                constant=constant_target(regression.pseudo_outcome, regression.fitted_on)
                is not None,
                binding=(paths, binding),
                probes=regression.probes,
            )
        )
        carried = training
    nodes.reverse()
    return nodes


#: Per cell, per node in time order: the held-out slice of every array the result keeps.
_FoldCell = list[tuple[_Node, IntArray]]


def _slice(node: _Node, rows: IntArray) -> _Node:
    """The node's arrays at ``rows``; the masks stay whole."""
    return _Node(
        time=node.time,
        at_risk=node.at_risk,
        trained_on=node.trained_on,
        regression_target=node.regression_target[rows],
        residual_target=node.residual_target[rows],
        observed=node.observed[rows],
        value=node.value[rows],
        by_arm=None if node.by_arm is None else node.by_arm[rows],
        transform=node.transform[rows],
        clever=node.clever[rows],
        diagnostics=node.diagnostics,
        constant=node.constant,
        binding=(node.binding[0][rows], node.binding[1][rows]),
        probes=None if node.probes is None else node.probes[rows],
    )


@dataclass(frozen=True)
class FoldMechanism:
    """One outer training fold's own mechanism, fitted on its training rows alone.

    Parameters
    ----------
    mechanism : Mechanism
        The fold's treatment and censoring fits, evaluated at every row.
    plans : tuple of Plan
        The plans with the fold's own policy numerators attached: the plans a held-out
        row's transform reads.
    carry_plans : tuple of Plan
        Per plan, the plan the fold's training transform reads, or ``None`` where it is the
        plan itself.  On a cross-fitted population-target tuned plan it is the map at the
        fold's own training tuner, with the fold's numerators.
    """

    mechanism: Mechanism
    plans: tuple[Plan, ...]
    carry_plans: tuple[Plan | None, ...] = ()


def fit_fold_mechanisms(
    data: LongitudinalData,
    plans: Sequence[Plan],
    *,
    treatment_learner: Learner,
    censoring_learner: Learner,
    folds: Folds,
    density_bins: int | None = None,
    ratio: str = "density",
    hazard_pooling: str = "node",
    n_jobs: int = 1,
) -> tuple[FoldMechanism, ...]:
    """Fit each outer training fold's own mechanism, on its training rows alone.

    Fold ``k``'s model is the one the out-of-fold mechanism uses at fold ``k``'s held-out
    rows, refitted here so that it also answers at fold ``k``'s training rows.  A
    sequentially doubly robust transform of a training row reads it there.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans.
    treatment_learner : Learner
        The treatment mechanism learner.
    censoring_learner : Learner
        The censoring mechanism learner.
    folds : Folds
        The realized outer split, with at least two folds.
    density_bins : int or None
        Bins of a continuous node's density.
    ratio : {"density", "classifier"}
        How a continuous node's policy ratio is estimated.
    hazard_pooling : {"node", "pooled"}
        Whether each fold fits one censoring model per node or one pooled over the nodes.
    n_jobs : int
        Parallel workers across the outer folds.

    Returns
    -------
    tuple of FoldMechanism
        One per outer fold, in fold order.
    """
    wanted = profiling()

    def run_fold(fold: int, train: IntArray) -> tuple[FoldMechanism, PhaseProfile | None]:
        outer_train = np.zeros(data.n, dtype=bool)
        outer_train[train] = True
        # A tuned plan fits the fold's mechanism under the fold's own map, at its training
        # tuner, so the classifier route stacks no held-out row's map.
        own = [plan.fold_plans[fold] if plan.fold_plans else plan for plan in plans]
        with collect_phases(wanted) as profile, phase("mechanism_fit"):
            mechanism = fit_mechanism(
                data,
                own,
                treatment_learner=treatment_learner,
                censoring_learner=censoring_learner,
                folds=Folds.single(data.n),
                density_bins=density_bins,
                ratio=ratio,
                hazard_pooling=hazard_pooling,
                fit_rows=outer_train,
            )
        attached = attach_policy_numerators(own, mechanism.policy_numerators)
        if not any(plan.fold_plans for plan in plans):
            return FoldMechanism(mechanism, tuple(attached)), profile
        evaluated: list[Plan] = []
        carry: list[Plan | None] = []
        for plan, fold_plan in zip(plans, attached, strict=True):
            if not plan.fold_plans or plan.fold_average:
                # A fold-average plan reads its fold's map on both kinds of row.
                evaluated.append(fold_plan)
                carry.append(None)
                continue
            evaluated.append(
                _at_full_sample_tuner(
                    data,
                    plan,
                    fold_plan,
                    mechanism,
                    ratio,
                    refit=partial(
                        _mixed_numerator,
                        data,
                        fit_rows=outer_train,
                        treatment_learner=treatment_learner,
                        density_bins=density_bins,
                    ),
                )
            )
            carry.append(fold_plan)
        return FoldMechanism(mechanism, tuple(evaluated), tuple(carry)), profile

    with phase("fold_mechanisms"):
        outcomes = map_parallel(
            run_fold, [(fold, train) for fold, (train, _) in enumerate(folds)], n_jobs=n_jobs
        )
    for _, profile in outcomes:
        merge_worker_phases(profile)
    return tuple(fold for fold, _ in outcomes)


def _mixed_numerator(
    data: LongitudinalData,
    plan: Plan,
    time: int,
    *,
    fit_rows: BoolArray,
    treatment_learner: Learner,
    density_bins: int | None,
) -> FloatArray:
    """One fold's joint density at a mixed node, evaluated under ``plan``'s map."""
    from .sequential import _mixed_node_density

    at_risk = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
    numerators: dict[str, dict[int, FloatArray]] = {plan.label: {}}
    _mixed_node_density(
        data,
        [plan],
        time,
        at_risk,
        numerators,
        fit_at=at_risk & fit_rows,
        treatment_learner=treatment_learner,
        folds=Folds.single(data.n),
        density_bins=density_bins,
        n_jobs=1,
    )
    return numerators[plan.label][time]


def _at_full_sample_tuner(
    data: LongitudinalData,
    plan: Plan,
    fold_plan: Plan,
    mechanism: Mechanism,
    ratio: str,
    *,
    refit: Any = None,
) -> Plan:
    """A population-target plan at :math:`\\theta_n`, with one fold's ratio numerators.

    On the density route each tuned node's ratio is the fold's density evaluated under the
    map at :math:`\\theta_n`; on the classifier route it is the fold's classifier, fitted
    at the fold's own map, at the observed dose.  Every other node keeps the fold plan's
    numerator, since its map does not depend on a tuner.  At a mixed node the induced
    joint density is not a function of the stored component densities alone, so
    ``refit(plan, time)`` fits the fold's joint density again, on the same rows, and
    evaluates it under the map at :math:`\\theta_n`.
    """
    from dataclasses import replace

    from ..interventions.policy import lazy_frame
    from .sequential import _density_numerator

    numerators = list(fold_plan.mtp_numerators)
    if ratio == "density":
        # A node with several tuned components is refitted once, at all of their maps.
        for time in sorted({tuning.time for tuning in plan.tuning}):
            if any(t.component is not None for t in plan.tuning if t.time == time):
                numerators[time - 1] = refit(plan, time)
                continue
            density = mechanism.densities[(time, data.treatment_names[time - 1])]
            dose = np.nan_to_num(data.node_codes(time), nan=0.0)
            frame = lazy_frame(partial(data.policy_frame, time))
            at_risk = data.uncensored_through(time - 1) & data.event_free_through(time - 1)
            numerators[time - 1] = np.where(
                at_risk, _density_numerator(plan, time, dose, frame, density), 0.0
            )
    return replace(plan, mtp_numerators=tuple(numerators))


def fit_cells_sdr(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    cells: Sequence[SDRCell],
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    fold_mechanisms: Sequence[FoldMechanism] = (),
    n_jobs: int = 1,
) -> list[RegimenFit]:
    """Run every cell's sequentially doubly robust recursion, in sample or cross-fitted.

    In sample, every cell reads ``mechanism`` and fits on every follower.  Above one fold,
    there is one job per outer fold.  Each job runs every cell on that fold's training rows
    with that fold's own mechanism from ``fold_mechanisms``.  The held-out rows of each fold
    are stitched, and each cell's estimate is the weighted mean of its stitched
    :math:`\\varphi_1`.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans, with the policy numerators of ``mechanism`` attached.
    mechanism : Mechanism
        The in-sample mechanism, or the out-of-fold one above one fold.  The result reports
        its cumulative products.  Above one fold no transform reads its factors.  Under
        ``ratio_trim=``, every fold's transform trims at its ``Mechanism.ratio_trims``
        thresholds.
    cells : sequence of SDRCell
        The recursions to run, in the order the result lists them.
    outcome_learner : Learner
        The regression at each cell's horizon.
    pseudo_learner : Learner
        The regression at every earlier node.
    folds : Folds
        The realized outer split.
    scaler : OutcomeScaler
        The outcome transformation.
    g_bounds : tuple of float
        The bound pair applied to each partial path probability.
    fold_mechanisms : sequence of FoldMechanism
        Above one fold, each fold's own mechanism, from :func:`fit_fold_mechanisms`.  Empty
        in sample.
    n_jobs : int
        Parallel workers across the outer folds.

    Returns
    -------
    list of RegimenFit
        One per cell, in the order of ``cells``.
    """
    plan_of = {plan.label: plan for plan in plans}
    masks = {plan.label: plan.masks(data) for plan in plans}
    # One threshold per plan, from the pooled out-of-fold ratios, as ``lmtp`` trims them.
    thresholds = {label: trim.threshold for label, trim in mechanism.ratio_trims.items()}
    wanted = profiling()

    def run(
        fold_mechanism: Mechanism,
        fold_plans: dict[str, Plan],
        fit_rows: BoolArray | None,
        outer_fold: int | None,
        carry_plans: dict[str, Plan | None] | None = None,
    ) -> list[list[_Node]]:
        return [
            _recursion(
                data,
                fold_plans[cell.label],
                fold_mechanism,
                masks[cell.label],
                cell,
                scaler=scaler,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                g_bounds=g_bounds,
                fit_rows=fit_rows,
                outer_fold=outer_fold,
                trim_threshold=thresholds.get(cell.label),
                carry_plan=None if carry_plans is None else carry_plans.get(cell.label),
            )
            for cell in cells
        ]

    if folds.is_single:
        every = np.arange(data.n)
        parts = [
            [[(node, every) for node in nodes]] for nodes in run(mechanism, plan_of, None, None)
        ]
    else:
        if len(fold_mechanisms) != folds.n_folds:
            raise ValueError("a cross-fitted SDR fit needs one fold mechanism per outer fold")

        def run_fold(
            fold: int, train: IntArray, test: IntArray
        ) -> tuple[IntArray, list[list[_Node]], PhaseProfile | None]:
            outer_train = np.zeros(data.n, dtype=bool)
            outer_train[train] = True
            own = fold_mechanisms[fold]
            carry = (
                {plan.label: c for plan, c in zip(own.plans, own.carry_plans, strict=True)}
                if own.carry_plans
                else None
            )
            with collect_phases(wanted) as profile:
                recursions = run(
                    own.mechanism,
                    {plan.label: plan for plan in own.plans},
                    outer_train,
                    fold,
                    carry,
                )
            sliced = [[_slice(node, test) for node in nodes] for nodes in recursions]
            return test, sliced, profile

        jobs = [(fold, train, test) for fold, (train, test) in enumerate(folds)]
        with phase("outer_fold_recursion"):
            outcomes = map_parallel(run_fold, jobs, n_jobs=n_jobs)
        for _, _, profile in outcomes:
            merge_worker_phases(profile)
        parts = [
            [[(node, test) for node in recursions[position]] for test, recursions, _ in outcomes]
            for position in range(len(cells))
        ]
    return [
        _finish(
            data,
            plan_of[cell.label],
            cell,
            _merge(data, cell_parts),
            cumulative=_bounded(data, plan_of[cell.label], mechanism, g_bounds),
        )
        for cell, cell_parts in zip(cells, parts, strict=True)
    ]


def _bounded(
    data: LongitudinalData, plan: Plan, mechanism: Mechanism, g_bounds: tuple[float, float]
) -> tuple[FloatArray, FloatArray]:
    """The raw and bounded cumulative mechanism the result reports."""
    return mechanism.cumulative_with_unbounded(data, plan, g_bounds)


@dataclass(frozen=True)
class _Stitched:
    """One node's arrays stitched over the folds, on every row."""

    node: _Node
    constant_folds: tuple[int, ...]
    diagnostics: tuple[SuperLearnerDiagnostics, ...]
    binding: tuple[int, int]


def _merge(
    data: LongitudinalData, parts: Sequence[Sequence[tuple[_Node, IntArray]]]
) -> list[_Stitched]:
    """Stitch each fold's held-out node arrays into one array per node, in time order.

    ``parts`` holds one sequence per fold, of ``(node, rows)`` pairs in time order, where
    ``node``'s arrays are already the held-out slice at ``rows``.  A single fold passes every
    row.
    """
    first = parts[0]
    stitched = []
    for index, (template, _) in enumerate(first):

        def blank(width: int | None = None) -> FloatArray:
            shape: tuple[int, ...] = (data.n,) if width is None else (data.n, width)
            return np.full(shape, _FILLER, dtype=float)

        arrays = {
            name: blank()
            for name in (
                "regression_target",
                "residual_target",
                "observed",
                "value",
                "transform",
            )
        }
        clever = np.zeros(data.n)
        by_arm = None if template.by_arm is None else blank(template.by_arm.shape[1])
        probes = None if template.probes is None else blank(template.probes.shape[1])
        diagnostics: list[SuperLearnerDiagnostics] = []
        constant_folds: list[int] = []
        paths = 0
        binding = 0
        for fold, fold_nodes in enumerate(parts):
            node, rows = fold_nodes[index]
            for name in arrays:
                arrays[name][rows] = getattr(node, name)
            clever[rows] = node.clever
            if by_arm is not None:
                assert node.by_arm is not None
                by_arm[rows] = node.by_arm
            if probes is not None:
                assert node.probes is not None
                probes[rows] = node.probes
            diagnostics.extend(node.diagnostics)
            if node.constant:
                constant_folds.append(fold + 1)
            # The held-out rows only: those are the paths the estimate reads.
            paths += int(node.binding[0].sum())
            binding += int(node.binding[1].sum())
        stitched.append(
            _Stitched(
                node=_Node(
                    time=template.time,
                    at_risk=template.at_risk,
                    trained_on=template.trained_on,
                    by_arm=by_arm,
                    clever=clever,
                    diagnostics=tuple(diagnostics),
                    constant=bool(constant_folds),
                    binding=(
                        np.array([paths], dtype=np.intp),
                        np.array([binding], dtype=np.intp),
                    ),
                    probes=probes,
                    **arrays,
                ),
                constant_folds=tuple(constant_folds) if len(parts) > 1 else (),
                diagnostics=tuple(diagnostics),
                binding=(paths, binding),
            )
        )
    return stitched


def _finish(
    data: LongitudinalData,
    plan: Plan,
    cell: SDRCell,
    nodes: Sequence[_Stitched],
    *,
    cumulative: tuple[FloatArray, FloatArray],
) -> RegimenFit:
    """One cell's estimate, the mean of :math:`\\varphi_1`, and its curve."""
    steps = tuple(
        SequentialStep(
            time=item.node.time,
            trained_on=item.node.trained_on,
            at_risk=item.node.at_risk,
            pseudo_outcome=item.node.residual_target,
            initial=item.node.observed,
            targeted=item.node.observed,
            clever=item.node.clever,
            fluctuation=None,
            learner_diagnostics=item.diagnostics,
            constant_folds=item.constant_folds,
            regression_target=item.node.regression_target,
            marginal=item.node.value if item.node.by_arm is not None else None,
            targeted_by_arm=item.node.by_arm,
            initial_by_arm=item.node.by_arm,
            transform=item.node.transform,
            bound_binding=(item.binding[1] / item.binding[0]) if item.binding[0] else 0.0,
            probes=item.node.probes,
        )
        for item in nodes
    )
    first = steps[0].transform
    assert first is not None
    unbounded, bounded = cumulative
    if plan.fold_average:
        return _fold_average_sdr(data, plan, cell, steps, first, unbounded, bounded)
    psi = float(np.average(first, weights=data.weights))
    influence = data.weights * (first - psi)
    term, derivatives = tuning_term(data, plan, steps)
    if term is not None:
        influence = influence + term
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=influence,
        horizon=cell.horizon,
        cause=cell.cause,
        steps=steps,
        cumulative_unbounded=unbounded,
        cumulative=bounded,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
        policy=plan.policy,
        cumulative_numerator=plan.cumulative_numerator(data),
        tuning_term=term,
        tuning_derivatives=derivatives,
    )


def _fold_average_sdr(
    data: LongitudinalData,
    plan: Plan,
    cell: SDRCell,
    steps: tuple[SequentialStep, ...],
    first: FloatArray,
    unbounded: FloatArray,
    bounded: FloatArray,
) -> RegimenFit:
    """The fold-average estimate and curve of a sequentially doubly robust fit.

    Fold ``k``'s estimate is the weighted mean of :math:`\\varphi_1` over its held-out rows,
    built from fold ``k``'s nuisances under fold ``k``'s map.  The estimate and the curve
    are assembled as in :func:`~cleverly.longitudinal.sequential._finish_fold_average`.
    """
    assert plan.folds is not None
    owner = np.asarray(plan.folds.assignment)
    n_folds = plan.folds.n_folds
    estimates = []
    influence = np.zeros(data.n)
    for k in range(n_folds):
        rows = owner == k
        mass = float(np.sum(data.weights[rows]))
        psi_k = float(np.average(first[rows], weights=data.weights[rows]))
        estimates.append(psi_k)
        influence = np.where(
            rows, data.n * data.weights * (first - psi_k) / (n_folds * mass), influence
        )
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=float(np.mean(estimates)),
        influence_curve_scaled=influence,
        horizon=cell.horizon,
        cause=cell.cause,
        steps=steps,
        cumulative_unbounded=unbounded,
        cumulative=bounded,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
        policy=plan.policy,
        cumulative_numerator=plan.cumulative_numerator(data),
        fold_estimates=tuple(estimates),
    )
