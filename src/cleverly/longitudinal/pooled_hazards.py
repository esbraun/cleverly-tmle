"""One censoring regression pooled over every node, as ``survtmle`` fits it.

``hazard_pooling="pooled"`` replaces the censoring model of each node with one regression
over every at-risk ``(unit, node)`` row.  The pooled design is the node's design plus one
numeric column that holds the node's grid time, as ``survtmle``'s ``glm.ctime`` reads ``t``.
A linear time column is invariant to a constant offset, because the intercept absorbs it.

Every node's design must have the same columns, so the option needs one baseline treatment
and no time-varying covariate.  A unit keeps its fold at every node, so its stacked rows are
never split between a training fold and its held-out fold.  A node whose retention the design
declares known keeps the declared value, and its rows stay out of the pooled fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from .._typing import BoolArray, FloatArray, Learner
from ..estimators._nuisance import cross_fit_predictions
from ..exceptions import CapabilityError
from ..learners.crossfit import Folds
from ..learners.super_learner import SuperLearnerDiagnostics
from .data import LongitudinalData
from .regimen import Plan

__all__ = [
    "HAZARD_POOLINGS",
    "HazardPooling",
    "PooledCensoring",
    "check_hazard_pooling",
    "fit_pooled_censoring",
    "node_time",
    "refuse_pooled_designs",
    "stacked_design",
]

HazardPooling = Literal["node", "pooled"]

#: The accepted values of ``hazard_pooling=``.
HAZARD_POOLINGS: Final[tuple[str, ...]] = ("node", "pooled")


def check_hazard_pooling(hazard_pooling: str) -> None:
    """Refuse a ``hazard_pooling`` that is not ``"node"`` or ``"pooled"``.

    Parameters
    ----------
    hazard_pooling : str
        The declared pooling.

    Raises
    ------
    ValueError
        When the value is not one of :data:`HAZARD_POOLINGS`.
    """
    if hazard_pooling not in HAZARD_POOLINGS:
        raise ValueError(f"hazard_pooling must be 'node' or 'pooled'; got {hazard_pooling!r}")


def refuse_pooled_designs(data: LongitudinalData) -> None:
    """Refuse a pooled fit on a design whose node designs differ.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.

    Raises
    ------
    CapabilityError
        When the design has a treatment decision at more than one node, or time-varying
        covariates, so its node designs have different columns.
    """
    reasons = []
    if not data.treatment_held and data.n_times > 1:
        reasons.append(f"a treatment decision at each of its {data.n_times} nodes")
    if any(names for names in data.time_varying_names):
        reasons.append("time-varying covariates")
    if reasons:
        raise CapabilityError(
            "hazard_pooling='pooled' fits one regression over every node, so every node needs "
            f"the same design. This design has {' and '.join(reasons)}, so its node designs "
            "differ. Use hazard_pooling='node'"
        )


def node_time(data: LongitudinalData, time: int) -> float:
    """The grid time of node ``time``: the grid value, or the node index without a grid.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    time : int
        The node, counted from one.

    Returns
    -------
    float
        ``time_grid[time - 1]`` on a container with a grid, else ``time``.
    """
    if data.time_grid is not None:
        return float(data.time_grid[time - 1])
    return float(time)


@dataclass(frozen=True)
class PooledCensoring:
    """The pooled censoring model's out-of-fold retention at each pooled node.

    Parameters
    ----------
    nodes : tuple of int
        The pooled nodes, counted from one.
    by_plan : dict of int to dict of str to FloatArray
        Per node, the ``(n,)`` retention at each plan's arms.
    observed : dict of int to FloatArray
        Per node, the ``(n,)`` retention at the observed arms.
    diagnostics : tuple of SuperLearnerDiagnostics
        The pooled fit's diagnostics, one per fold.
    constant_folds : tuple of int
        The folds whose training rows hold one retention value.
    """

    nodes: tuple[int, ...]
    by_plan: dict[int, dict[str, FloatArray]]
    observed: dict[int, FloatArray]
    diagnostics: tuple[SuperLearnerDiagnostics, ...]
    constant_folds: tuple[int, ...]


def stacked_design(
    data: LongitudinalData, nodes: Sequence[int], plan: Plan | None = None
) -> FloatArray:
    """``(n * len(nodes), p + 1)`` node designs stacked by node, with the grid time last.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    nodes : sequence of int
        The pooled nodes, counted from one.
    plan : Plan or None
        The plan whose arms set the treatment blocks, or ``None`` for the observed arms.

    Returns
    -------
    FloatArray
        Node ``nodes[j]``'s rows occupy rows ``j * n`` to ``(j + 1) * n``.
    """
    blocks = []
    for time in nodes:
        design = data.history_design(
            time, treatment=None if plan is None else plan.values, include_current=True
        )
        blocks.append(np.column_stack([design, np.full(data.n, node_time(data, time))]))
    return np.vstack(blocks)


def fit_pooled_censoring(
    data: LongitudinalData,
    plans: Sequence[Plan],
    nodes: Sequence[int],
    at_risk: dict[int, BoolArray],
    *,
    learner: Learner,
    folds: Folds,
    n_jobs: int = 1,
) -> PooledCensoring:
    """Fit one retention model over the at-risk rows of ``nodes``, out of fold.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans.
    nodes : sequence of int
        The nodes whose retention the fit estimates, counted from one.
    at_risk : dict of int to BoolArray
        Per node, the rows at risk of censoring there.
    learner : Learner
        The censoring learner.
    folds : Folds
        The unit-level split.  Each unit keeps its fold at every node.
    n_jobs : int
        Parallel jobs.

    Returns
    -------
    PooledCensoring
        The out-of-fold retention at each node.
    """
    nodes = tuple(nodes)
    count = len(nodes)
    stayed = np.concatenate(
        [np.where(at_risk[t], data.uncensored[:, t - 1].astype(float), 0.0) for t in nodes]
    )
    fit_mask = np.concatenate([at_risk[t] for t in nodes])
    stacked_folds = Folds(np.tile(folds.assignment, count), folds.n_folds)
    groups = None if data.cluster is None else np.tile(data.cluster, count)
    observed_key = "\0observed_censoring"
    designs = {plan.label: stacked_design(data, nodes, plan) for plan in plans}
    designs[observed_key] = stacked_design(data, nodes)
    constant: list[int] = []
    predictions, diagnostics = cross_fit_predictions(
        learner,
        designs[observed_key],
        stayed,
        np.tile(data.weights, count),
        stacked_folds,
        task="classification",
        predict_designs=designs,
        fit_mask=fit_mask,
        groups=groups,
        clip=(0.0, 1.0),
        n_jobs=n_jobs,
        constant_folds=constant,
    )
    n = data.n
    by_plan: dict[int, dict[str, FloatArray]] = {}
    observed: dict[int, FloatArray] = {}
    for position, time in enumerate(nodes):
        rows = slice(position * n, (position + 1) * n)
        by_plan[time] = {
            plan.label: np.asarray(predictions[plan.label][rows], dtype=float) for plan in plans
        }
        observed[time] = np.asarray(predictions[observed_key][rows], dtype=float)
    return PooledCensoring(nodes, by_plan, observed, tuple(diagnostics), tuple(sorted(constant)))
