r"""Incremental propensity-score interventions over time: Kennedy's estimator.

The estimator of Kennedy (2019, *JASA* 114(526), Section 4.2, Algorithm 1), with the dropout
weights of Kim, Kennedy and Naimi (2021, *J Causal Inference* 9(1), Appendix A, Algorithm 1).
At a tilt node :math:`t` the plan draws the arm from

.. math::

    q_t(1 \mid H_t) = \frac{\delta_t \pi_t(H_t)}{\delta_t \pi_t(H_t) + 1 - \pi_t(H_t)},

Equation (1) of Kennedy (2019), with :math:`\pi_t` the treatment mechanism.  A plan can
hold label, rule and known-policy nodes beside its tilt nodes.

**The recursion.**  From the horizon down to the first node, the regression
:math:`m_t(H_t, A_t)` of :math:`Z_{t+1}` is fitted on the rows that follow the plan through
:math:`t`, and the node carries :math:`\bar m_t = \sum_a q_t(a \mid H_t) m_t(H_t, a)`
(Kennedy 2019, Remark 2).  :math:`Z_{h+1}` is the outcome, and
:math:`Z_{t+1} = Y_t + (1 - Y_t)\bar m_{t+1}` on a survival fit, with the all-cause
survival factor on a competing-risk fit.

**The influence function.**  The uncentered efficient influence function at the fitted
nuisances is

.. math::

    \varphi = \bar m_1(H_1) + \sum_{t=1}^{h} W_t \{Z_{t+1} - m_t(H_t, A_t)\}
        + \sum_{t \in \text{tilt}} W_{t-1}
          \frac{\delta_t \{m_t(H_t, 1) - m_t(H_t, 0)\}(A_t - \pi_t)}
               {(\delta_t \pi_t + 1 - \pi_t)^2},

Kennedy (2019, Theorem 2 and Appendix Lemmas 2 to 4) and Kim et al. (2021, Theorem 4.1).
The weight :math:`W_t` multiplies one factor per node up to :math:`t`: the cancelled tilt
factor :math:`(\delta_s A_s + 1 - A_s)/(\delta_s \pi_s + 1 - \pi_s)` at a tilt node, the
plan's intervention density over the mechanism at any other node, and the censoring
factor :math:`1\{C_s = 1\}/\omega_s` at every node.  ``g_bounds`` bounds the cumulative
product of the non-tilt mechanism factors alone, and the tilt product multiplies in after
the bound.  The tilt factor is never computed as :math:`q_t / \pi_t`.  The last sum is the
*mechanism term*.  It is absent at a node whose treatment mechanism is declared known.

**The estimate** is :math:`\hat\psi = \sum_i w_i \varphi_i / \sum_i w_i`, and the curve is
:math:`w_i (\varphi_i - \hat\psi)`.  Nothing is fluctuated.

**Cross-fitting is the training-fold recursion.**  Fold :math:`k` fits the mechanism and
every regression on its training rows, and builds each training row's target from its own
predictions (Kennedy 2019, Algorithm 1: "construct ... using only :math:`D_0`").  Its
held-out rows take :math:`\varphi` from fold :math:`k`'s models.  The estimate pools every
row, :math:`P_n \varphi(Z; \hat\eta_{-S})`, the form of Theorem 3.  Algorithm 1 averages the
fold means instead.  The two agree at equal fold sizes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..learners.crossfit import Folds
from ..learners.super_learner import SuperLearnerDiagnostics
from ..utils.bounds import OutcomeScaler
from ..utils.parallel import map_parallel
from ..utils.phases import PhaseProfile, collect_phases, merge_worker_phases, phase, profiling
from .data import LongitudinalData, RegimenMasks
from .regimen import Plan, _node_numerator
from .sdr import FoldMechanism
from .sequential import (
    _FILLER,
    Mechanism,
    RegimenFit,
    SequentialStep,
    _fit_node_regression,
    constant_target,
    seed_carried,
)

__all__ = [
    "IPSICell",
    "fit_cells_ipsi",
    "tilt_propensities",
    "tilted_plans",
]


@dataclass(frozen=True)
class IPSICell:
    """One recursion of Kennedy's estimator: a regimen, a cause and a horizon.

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


def tilt_propensities(
    data: LongitudinalData, plan: Plan, mechanism: Mechanism
) -> dict[int, FloatArray]:
    r"""By tilt node, the mechanism :math:`\pi_t(H_t)` of the plan's second level.

    It is read at the observed history, which on the rows at risk under the plan is the
    plan's history: a tilt node keeps every row, and a label or rule node keeps the rows
    whose observed arm is the assigned one.

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
    dict of int to FloatArray
        One ``(n,)`` probability per tilt node, counted from one.
    """
    out: dict[int, FloatArray] = {}
    for time in range(1, data.n_times + 1):
        if not plan.is_tilt_node(time):
            continue
        matrix = np.asarray(mechanism.treatment_observed[time - 1], dtype=float)
        out[time] = np.nan_to_num(matrix[:, 1], nan=0.0)
    return out


def tilted_plans(
    data: LongitudinalData, plans: Sequence[Plan], mechanism: Mechanism
) -> tuple[Plan, ...]:
    """Each plan with its tilt densities read from ``mechanism``.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans.
    mechanism : Mechanism
        The fitted mechanism.

    Returns
    -------
    tuple of Plan
        The plans, unchanged where they hold no tilt node.
    """
    return tuple(
        plan.with_tilts(tilt_propensities(data, plan, mechanism)) if plan.has_tilt else plan
        for plan in plans
    )


@dataclass(frozen=True)
class _Weights:
    """The weights :math:`W_0, \\ldots, W_h` of one plan, and the bound diagnostics."""

    weights: tuple[FloatArray, ...]
    raw: FloatArray
    bounded: FloatArray
    paths: tuple[int, ...]
    binding: tuple[int, ...]


def _weights(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    masks: RegimenMasks,
    propensity: Mapping[int, FloatArray],
    horizon: int,
    g_bounds: tuple[float, float],
) -> _Weights:
    """:math:`W_t` for ``t = 0 .. horizon``, zero off the rows that follow through ``t``."""
    raw, bounded = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    numerator = np.ones(data.n)
    tilt = np.ones(data.n)
    weights = [np.ones(data.n)]
    paths = []
    binding = []
    for time in range(1, horizon + 1):
        if plan.is_tilt_node(time):
            delta = plan.tilt_delta(time)
            pi = np.clip(np.asarray(propensity[time], dtype=float), 0.0, 1.0)
            arm = np.nan_to_num(data.node_codes(time), nan=0.0)
            tilt = tilt * (delta * arm + 1.0 - arm) / (delta * pi + 1.0 - pi)
        else:
            numerator = numerator * _node_numerator(plan, data, time)
        following = masks.following(time)
        limited = bounded[:, time - 1]
        weights.append(np.where(following, numerator * tilt / limited, 0.0))
        paths.append(int(np.count_nonzero(following)))
        binding.append(int(np.count_nonzero(following & (limited != raw[:, time - 1]))))
    return _Weights(
        weights=tuple(weights),
        raw=raw,
        bounded=bounded,
        paths=tuple(paths),
        binding=tuple(binding),
    )


@dataclass(frozen=True)
class _Node:
    """One node of one fold's recursion, on every row."""

    time: int
    at_risk: BoolArray
    trained_on: BoolArray
    target: FloatArray
    observed: FloatArray
    value: FloatArray
    by_arm: FloatArray | None
    weight: FloatArray
    mechanism_term: FloatArray | None
    diagnostics: tuple[SuperLearnerDiagnostics, ...]
    constant: bool
    binding: tuple[int, int]


@dataclass(frozen=True)
class _Run:
    """One cell's recursion with one fold's nuisances: its nodes and :math:`\\varphi`.

    ``mechanism_term`` is ``(n, T)``: column ``t - 1`` holds node ``t``'s mechanism term.
    """

    nodes: tuple[_Node, ...]
    phi: FloatArray
    mechanism_term: FloatArray


def _recursion(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    masks: RegimenMasks,
    cell: IPSICell,
    *,
    scaler: OutcomeScaler,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    g_bounds: tuple[float, float],
    fit_rows: BoolArray | None,
    outer_fold: int | None,
) -> _Run:
    """One cell's recursion and influence function, with one fold's nuisances, on every row."""
    single = Folds.single(data.n)
    propensity = tilt_propensities(data, plan, mechanism)
    tilted = plan.with_tilts(propensity) if plan.has_tilt else plan
    horizon = cell.horizon
    known = set(mechanism.known_treatment_nodes)
    with phase("clever_covariate"):
        weights = _weights(data, tilted, mechanism, masks, propensity, horizon, g_bounds)
    carried = seed_carried(data, scaler, horizon=horizon)
    nodes: list[_Node] = []
    for time in range(horizon, 0, -1):
        regression = _fit_node_regression(
            data,
            tilted,
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
        )
        at_risk = regression.at_risk
        value = (
            regression.initial
            if regression.initial_marginal is None
            else regression.initial_marginal
        )
        value = np.where(at_risk, value, _FILLER)
        term = None
        if plan.is_tilt_node(time) and time not in known:
            assert regression.initial_by_arm is not None
            with phase("influence_curve"):
                delta = plan.tilt_delta(time)
                pi = np.clip(np.asarray(propensity[time], dtype=float), 0.0, 1.0)
                arm = np.nan_to_num(data.node_codes(time), nan=0.0)
                blip = regression.initial_by_arm[:, 1] - regression.initial_by_arm[:, 0]
                derivative = delta * blip * (arm - pi) / (delta * pi + 1.0 - pi) ** 2
                term = np.where(at_risk, weights.weights[time - 1] * derivative, 0.0)
        nodes.append(
            _Node(
                time=time,
                at_risk=at_risk,
                trained_on=regression.trained_on,
                target=regression.pseudo_outcome,
                observed=regression.initial,
                value=value,
                by_arm=regression.initial_by_arm,
                weight=weights.weights[time],
                mechanism_term=term,
                diagnostics=regression.learner_diagnostics,
                constant=constant_target(regression.pseudo_outcome, regression.fitted_on)
                is not None,
                binding=(weights.paths[time - 1], weights.binding[time - 1]),
            )
        )
        carried = value
    nodes.reverse()
    with phase("influence_curve"):
        phi = np.array(nodes[0].value, dtype=float, copy=True)
        mechanism_term = np.zeros((data.n, data.n_times))
        for node in nodes:
            residual = np.where(node.weight != 0.0, node.target - node.observed, 0.0)
            phi = phi + node.weight * residual
            if node.mechanism_term is not None:
                mechanism_term[:, node.time - 1] = node.mechanism_term
        phi = phi + mechanism_term.sum(axis=1)
    return _Run(nodes=tuple(nodes), phi=phi, mechanism_term=mechanism_term)


def _slice(run: _Run, rows: IntArray) -> _Run:
    """The run's arrays at ``rows``; the masks stay whole."""
    return _Run(
        nodes=tuple(
            _Node(
                time=node.time,
                at_risk=node.at_risk,
                trained_on=node.trained_on,
                target=node.target[rows],
                observed=node.observed[rows],
                value=node.value[rows],
                by_arm=None if node.by_arm is None else node.by_arm[rows],
                weight=node.weight[rows],
                mechanism_term=None if node.mechanism_term is None else node.mechanism_term[rows],
                diagnostics=node.diagnostics,
                constant=node.constant,
                binding=node.binding,
            )
            for node in run.nodes
        ),
        phi=run.phi[rows],
        mechanism_term=run.mechanism_term[rows],
    )


def fit_cells_ipsi(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    cells: Sequence[IPSICell],
    *,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    fold_mechanisms: Sequence[FoldMechanism] = (),
    n_jobs: int = 1,
) -> list[RegimenFit]:
    """Run every cell's recursion, in sample or by the training-fold recursion.

    In sample, every cell reads ``mechanism`` and fits on every follower.  Above one fold,
    there is one job per outer fold.  Each job runs every cell on that fold's training rows
    with that fold's own mechanism from ``fold_mechanisms``.  The held-out rows of each fold
    are stitched, and each cell's estimate is the weighted mean of its stitched
    :math:`\\varphi`.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans, without tilt densities.
    mechanism : Mechanism
        The in-sample mechanism, or the out-of-fold one above one fold.  The result reports
        its cumulative products.  Above one fold no influence function reads it.
    cells : sequence of IPSICell
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
        The bound pair of the cumulative non-tilt mechanism product.
    fold_mechanisms : sequence of FoldMechanism
        Above one fold, each fold's own mechanism.  Empty in sample.
    n_jobs : int
        Parallel workers across the outer folds.

    Returns
    -------
    list of RegimenFit
        One per cell, in the order of ``cells``.
    """
    plan_of = {plan.label: plan for plan in plans}
    masks = {plan.label: plan.masks(data) for plan in plans}
    wanted = profiling()

    def run(
        fold_mechanism: Mechanism, fit_rows: BoolArray | None, outer_fold: int | None
    ) -> list[_Run]:
        return [
            _recursion(
                data,
                plan_of[cell.label],
                fold_mechanism,
                masks[cell.label],
                cell,
                scaler=scaler,
                outcome_learner=outcome_learner,
                pseudo_learner=pseudo_learner,
                g_bounds=g_bounds,
                fit_rows=fit_rows,
                outer_fold=outer_fold,
            )
            for cell in cells
        ]

    if folds.is_single:
        every = np.arange(data.n)
        parts = [[(cell_run, every)] for cell_run in run(mechanism, None, None)]
    else:
        if len(fold_mechanisms) != folds.n_folds:
            raise ValueError("a cross-fitted IPSI fit needs one fold mechanism per outer fold")

        def run_fold(
            fold: int, train: IntArray, test: IntArray
        ) -> tuple[IntArray, list[_Run], PhaseProfile | None]:
            outer_train = np.zeros(data.n, dtype=bool)
            outer_train[train] = True
            with collect_phases(wanted) as profile:
                runs = run(fold_mechanisms[fold].mechanism, outer_train, fold)
            return test, [_slice(cell_run, test) for cell_run in runs], profile

        jobs = [(fold, train, test) for fold, (train, test) in enumerate(folds)]
        with phase("outer_fold_recursion"):
            outcomes = map_parallel(run_fold, jobs, n_jobs=n_jobs)
        for _, _, profile in outcomes:
            merge_worker_phases(profile)
        parts = [
            [(runs[position], test) for test, runs, _ in outcomes] for position in range(len(cells))
        ]
    reported = {
        plan.label: plan.with_tilts(tilt_propensities(data, plan, mechanism))
        if plan.has_tilt
        else plan
        for plan in plans
    }
    fits = []
    for cell, cell_parts in zip(cells, parts, strict=True):
        plan = reported[cell.label]
        cumulative = _weights(
            data,
            plan,
            mechanism,
            masks[cell.label],
            tilt_propensities(data, plan, mechanism),
            cell.horizon,
            g_bounds,
        )
        fits.append(_finish(data, plan, cell, cell_parts, cumulative))
    return fits


def _finish(
    data: LongitudinalData,
    plan: Plan,
    cell: IPSICell,
    parts: Sequence[tuple[_Run, IntArray]],
    cumulative: _Weights,
) -> RegimenFit:
    """Stitch one cell's folds; its estimate, the mean of :math:`\\varphi`, and its curve."""
    phi = np.full(data.n, np.nan)
    mechanism_term = np.zeros((data.n, data.n_times))
    for cell_run, rows in parts:
        phi[rows] = cell_run.phi
        mechanism_term[rows] = cell_run.mechanism_term
    template = parts[0][0].nodes
    steps = []
    for index, node in enumerate(template):

        def blank(width: int | None = None) -> FloatArray:
            shape: tuple[int, ...] = (data.n,) if width is None else (data.n, width)
            return np.full(shape, _FILLER, dtype=float)

        target = blank()
        observed = blank()
        value = blank()
        weight = np.zeros(data.n)
        by_arm = None if node.by_arm is None else blank(node.by_arm.shape[1])
        diagnostics: list[SuperLearnerDiagnostics] = []
        constant_folds: list[int] = []
        paths = 0
        binding = 0
        for fold, (cell_run, rows) in enumerate(parts):
            part = cell_run.nodes[index]
            target[rows] = part.target
            observed[rows] = part.observed
            value[rows] = part.value
            weight[rows] = part.weight
            if by_arm is not None:
                assert part.by_arm is not None
                by_arm[rows] = part.by_arm
            diagnostics.extend(part.diagnostics)
            if part.constant:
                constant_folds.append(fold + 1)
            paths += part.binding[0]
            binding += part.binding[1]
        steps.append(
            SequentialStep(
                time=node.time,
                trained_on=node.trained_on,
                at_risk=node.at_risk,
                pseudo_outcome=target,
                initial=observed,
                targeted=observed,
                clever=weight,
                fluctuation=None,
                learner_diagnostics=tuple(diagnostics),
                constant_folds=tuple(constant_folds) if len(parts) > 1 else (),
                regression_target=target,
                marginal=value if by_arm is not None else None,
                targeted_by_arm=by_arm,
                initial_by_arm=by_arm,
                transform=phi if index == 0 else None,
                bound_binding=(binding / paths) if paths else 0.0,
            )
        )
    psi = float(np.average(phi, weights=data.weights))
    influence = data.weights * (phi - psi)
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=influence,
        horizon=cell.horizon,
        cause=cell.cause,
        steps=tuple(steps),
        cumulative_unbounded=cumulative.raw,
        cumulative=cumulative.bounded,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
        policy=plan.policy,
        cumulative_numerator=None,
        mechanism_term=mechanism_term,
    )
