r"""The frequency-weighted bootstrap replicate of an in-sample sequential fit.

A nonparametric bootstrap resample of :math:`n` rows holds row :math:`i` of the sample
:math:`c_i` times, with :math:`\sum_i c_i = n`.  A sequential fit reads each row through sums
over rows, so the resample's fit is a fit on the **distinct** rows of the sample, each with
the frequency :math:`c_i`.  This module computes that fit directly: no resample is
materialized, and every quantity that does not depend on the counts is computed once.

The replicate equals a materialized refit (``run_bootstrap`` on
:meth:`~cleverly.longitudinal.LongitudinalData.subset`) to rounding.  The six cases where a
frequency-weighted fit could differ from a materialized one are handled as follows.

=====================================  ==========================================================
case                                   handling here
=====================================  ==========================================================
row presence                           every presence test reads the rows with :math:`c_i > 0`
                                       only: the follower, constant-node and policy-arm checks
                                       of each node regression, and the fluctuation's rows.  A
                                       row of count zero is not in the replicate at all
solver path and separation             each learner starts from its own initial value, as a
                                       materialized refit does.  The weighted sums equal the
                                       replicated sums, so the iterates are the same in exact
                                       arithmetic.  Nothing is warm-started.  At separation no
                                       finite maximum exists: :class:`~cleverly.learners.GLM`
                                       raises on both paths, so both drop the replicate.  A
                                       learner that keeps a diverging last iterate amplifies
                                       the rounding difference of the two sums, so the two
                                       paths then agree only to that learner's tolerance
frequency, not sampling, weights       the counts reach the learners and the fluctuation as
                                       weights.  The standard error is the frequency-weighted
                                       variance :math:`\sum_i c_i (D_i - \bar D)^2 / (n - 1)`
                                       over :math:`n`, which is what
                                       :func:`~cleverly.inference.influence_variance` gives
                                       on the materialized curve.  The observation-weight
                                       channel of the data, which reads weights as sampling
                                       weights, is not used
learner eligibility                    only a learner declared exact under frequency weights
                                       (:func:`frequency_exact`) runs here.  Every other fit
                                       takes the materialized refit
data-dependent transforms              the outcome scaler is recomputed from the rows present
                                       in the replicate.  A continuous or mixed treatment node,
                                       whose density bins at quantiles of the sample, takes
                                       the materialized refit
resample draw                          the replicate reads the row index that
                                       :func:`~cleverly.inference.bootstrap.run_bootstrap`
                                       draws for the same seed, and counts it through the
                                       distinct-row map.  Both paths see the same replicate
=====================================  ==========================================================

The checks a fit runs before any learner split by kind.  A check that every row must pass
(a declared probability that the data contradicts, a cumulative product outside the bounds)
passes on every replicate when it passes on the sample, because a replicate holds a subset of
the sample's rows.  A check that some row must exist (a follower at a node, an observed level
at a policy node) is repeated per node by the recursion, with the presence mask.  So the fast
path fails on exactly the replicates a materialized refit fails on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..estimators._nuisance import cross_fit_predictions
from ..learners.crossfit import Folds
from ..utils.bounds import OutcomeScaler
from .data import LongitudinalData
from .regimen import Plan
from .sdr import node_factors, partial_weights
from .sequential import (
    _FILLER,
    _REGIMEN_ARM,
    Mechanism,
    _carried,
    _check_policy_arm_support,
    _clever_covariate,
    _fluctuate_node,
    _policy_arm,
    _pseudo_outcome,
    _require_regimen_followers,
    constant_target,
    outcome_design,
    seed_carried,
)

__all__ = [
    "FREQUENCY_EXACT_ATTRIBUTE",
    "FrequencyReplicate",
    "ReportedName",
    "distinct_rows",
    "frequency_exact",
]

#: The class attribute a learner sets to ``True`` to declare that a fit with nonnegative
#: integer ``sample_weight`` equals the fit on the rows repeated that many times, from the same
#: initial value.  An unpenalized maximum-likelihood or least-squares fit has that property.
#: A penalized fit does not (the penalty does not scale with the weights), and neither does a
#: learner that cross-validates internally (a repeated row can enter a training and a
#: validation fold at once).
FREQUENCY_EXACT_ATTRIBUTE = "frequency_exact"


def frequency_exact(learner: Learner) -> bool:
    """Whether ``learner`` is declared exact under frequency weights.

    The declaration is the class attribute :data:`FREQUENCY_EXACT_ATTRIBUTE`, read from the
    learner's own class and not from its bases.  A subclass that adds a penalty would
    inherit a base declaration that no longer holds, so every class opts in by setting the
    attribute itself.  Scikit-learn's
    unpenalized :class:`~sklearn.linear_model.LinearRegression` is the one class outside the
    package that is accepted without it: its weighted least squares solve the replicated
    normal equations.

    Parameters
    ----------
    learner : Learner
        A resolved, unfitted learner.

    Returns
    -------
    bool
        ``True`` when the frequency-weighted fit equals the replicated fit.
    """
    if vars(type(learner)).get(FREQUENCY_EXACT_ATTRIBUTE, False) is True:
        return True
    from sklearn.linear_model import LinearRegression

    return type(learner) is LinearRegression


def distinct_rows(data: LongitudinalData) -> tuple[LongitudinalData, IntArray]:
    """The distinct rows of ``data``, and the distinct row each original row equals.

    Two rows are equal when every array a fit reads holds the same values at both: the
    outcome, the baseline, the treatment, the retention, every time-varying block, the event
    matrices, the observation weight and the declared mechanism.  Missing values compare by
    position, so a ``nan`` equals a ``nan`` and nothing else.  Unit identifiers are not read
    by a fit, so they do not separate rows.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.

    Returns
    -------
    tuple
        The distinct rows, in the order of their first occurrence, and the ``(n,)`` map from
        each row of ``data`` to its distinct row.
    """
    columns: list[FloatArray] = [
        np.asarray(data.outcome, dtype=float).reshape(data.n, -1),
        np.asarray(data.baseline, dtype=float).reshape(data.n, -1),
        np.asarray(data.treatment, dtype=float).reshape(data.n, -1),
        np.asarray(data.uncensored, dtype=float).reshape(data.n, -1),
        np.asarray(data.weights, dtype=float).reshape(data.n, -1),
        *(np.asarray(block, dtype=float).reshape(data.n, -1) for block in data.time_varying),
    ]
    for optional in (data.event, data.repeated):
        if optional is not None:
            columns.append(np.asarray(optional, dtype=float).reshape(data.n, -1))
    if data.cause_event is not None:
        columns.append(np.asarray(data.cause_event, dtype=float).reshape(data.n, -1))
    for values in data.component_values:
        if values is not None:
            columns.append(np.asarray(values, dtype=float).reshape(data.n, -1))
    known = data.known_mechanisms
    if known is not None:
        for time in range(1, data.n_times + 1):
            for declared in (known.treatment_at(time), known.censoring_at(time)):
                if declared is not None:
                    columns.append(np.asarray(declared, dtype=float).reshape(data.n, -1))
    if data.cluster is not None:
        columns.append(np.asarray(data.cluster, dtype=float).reshape(data.n, -1))
    matrix = np.column_stack(columns)
    missing = np.isnan(matrix)
    key = np.ascontiguousarray(np.column_stack([np.where(missing, 0.0, matrix), missing]))
    # ``-0.0`` and ``0.0`` are one value to every computation, so they are one key.
    key = key + 0.0
    view = key.view(np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).reshape(-1)
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return data.subset(first[order]), np.asarray(rank[inverse.reshape(-1)], dtype=np.int64)


def merge_equal_rows(
    design: FloatArray, target: FloatArray, weight: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Merge rows whose design and target are equal, summing their weights.

    For a learner exact under frequency weights, a row of weight :math:`a + b` fits as two
    equal rows of weights :math:`a` and :math:`b`, so the merge changes no fitted value.  Two
    rows merge only when every entry is equal.  The rows are ordered by a random projection
    first, and only neighbours in that order are compared, so equal rows that the order
    separates stay apart, which is also exact.

    Parameters
    ----------
    design : ndarray
        ``(r, p)`` design of the rows a regression is fitted on.
    target : ndarray
        ``(r,)`` regression target.
    weight : ndarray
        ``(r,)`` frequencies.

    Returns
    -------
    tuple of ndarray
        The merged design, target and summed frequencies.
    """
    matrix = np.column_stack([design, target])
    if matrix.shape[0] < 2:
        return design, target, weight
    projection = np.random.default_rng(0).random(matrix.shape[1])
    order = np.argsort(matrix @ projection, kind="stable")
    ordered = matrix[order]
    start = np.ones(ordered.shape[0], dtype=bool)
    start[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    group = np.cumsum(start) - 1
    merged = ordered[start]
    summed = np.bincount(group, weights=weight[order])
    return merged[:, :-1], merged[:, -1], summed


@dataclass(frozen=True)
class ReportedName:
    """One reported parameter of the fit, as the replicate composes it.

    Parameters
    ----------
    name : str
        The reported name.
    label : str
        The regimen whose mean the parameter reads.
    reference : str or None
        The reference regimen of a contrast, ``None`` for a mean.
    """

    name: str
    label: str
    reference: str | None


@dataclass(frozen=True)
class _Node:
    """The count-independent arrays of one node of one plan, on the distinct rows."""

    time: int
    at_risk: BoolArray
    trained_on: BoolArray
    design: FloatArray
    predict_designs: Mapping[str, FloatArray]
    #: The observed arm's code at a categorical policy node, ``None`` elsewhere.
    arm: IntArray | None
    policy: FloatArray | None
    counterfactual: FloatArray
    clever: FloatArray
    #: On an SDR fit, the weights :math:`W_{t,s}` for ``s = t .. h``.
    partial: tuple[FloatArray, ...] = ()


@dataclass(frozen=True)
class _PlanInputs:
    plan: Plan
    nodes: Mapping[int, _Node]


@dataclass(frozen=True)
class FrequencyReplicate:
    """Replicate an in-sample ``LTMLE`` or ``SDR`` fit from bootstrap counts.

    Build one with :meth:`build`.  Calling the object with a row index of the original data,
    as :func:`~cleverly.inference.bootstrap.run_bootstrap` draws it, returns each reported
    parameter's estimate and influence-curve standard error on the resample.

    Parameters
    ----------
    data : LongitudinalData
        The distinct rows of the prepared panel.
    inverse : ndarray
        ``(n,)`` map from each original row to its distinct row.
    plans : tuple of _PlanInputs
        Each plan, with its count-independent node arrays.
    names : tuple of ReportedName
        The reported parameters, in report order.
    estimator : {"tmle", "sdr"}
        Which recursion runs.
    horizon : int
        The node every recursion starts from.
    outcome_learner : Learner
        The regression at the horizon.
    pseudo_learner : Learner
        The regression at every earlier node.
    q_bounds : tuple of float or None
        The declared outcome bounds of a continuous outcome.
    targeting : dict
        The fluctuation settings ``alpha``, ``max_iter`` and ``tol`` of a targeted fit.
    """

    data: LongitudinalData
    inverse: IntArray
    plans: tuple[_PlanInputs, ...]
    names: tuple[ReportedName, ...]
    estimator: str
    horizon: int
    outcome_learner: Learner
    pseudo_learner: Learner
    q_bounds: tuple[float, float] | None
    targeting: Mapping[str, Any]

    @classmethod
    def build(
        cls,
        prepared: LongitudinalData,
        *,
        resolve: Callable[[LongitudinalData], tuple[tuple[Plan, ...], Mechanism]],
        names: Sequence[ReportedName],
        estimator: str,
        horizon: int,
        outcome_learner: Learner,
        pseudo_learner: Learner,
        g_bounds: tuple[float, float],
        q_bounds: tuple[float, float] | None,
        targeting: Mapping[str, Any],
    ) -> FrequencyReplicate:
        """Collapse the sample to its distinct rows and compute every fixed array once.

        Parameters
        ----------
        prepared : LongitudinalData
            The prepared panel of the fit.
        resolve : callable
            Maps a panel to its resolved plans and its mechanism, as the fit resolves them.
            The mechanism must be the declared one, which reads each row alone.
        names : sequence of ReportedName
            The reported parameters, in report order.
        estimator : {"tmle", "sdr"}
            Which recursion runs.
        horizon : int
            The node every recursion starts from.
        outcome_learner : Learner
            The regression at the horizon.
        pseudo_learner : Learner
            The regression at every earlier node.
        g_bounds : tuple of float
            The cumulative bound pair.
        q_bounds : tuple of float or None
            The declared outcome bounds of a continuous outcome.
        targeting : mapping
            The fluctuation settings of a targeted fit.

        Returns
        -------
        FrequencyReplicate
            The replicate, ready to be called with row indices.
        """
        data, inverse = distinct_rows(prepared)
        plans, mechanism = resolve(data)
        inputs = []
        for plan in plans:
            masks = plan.masks(data)
            _, cumulative = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
            numerator = plan.cumulative_numerator(data)
            density = plan.intervention_density(data) if estimator == "sdr" else None
            factors = node_factors(data, plan, mechanism) if estimator == "sdr" else None
            nodes = {}
            for time in range(horizon, 0, -1):
                at_risk = masks.at_risk(time)
                trained_on = masks.following(time)
                counterfactual, clever = _clever_covariate(
                    at_risk,
                    trained_on,
                    cumulative,
                    time,
                    numerator=None if numerator is None else numerator[:, time - 1],
                )
                policy_node = plan.is_policy_node(time)
                levels = range(plan.policy_at(time).shape[1] if policy_node else 0)
                design = outcome_design(data, plan, time)
                predict_designs = (
                    {f"arm_{code}": outcome_design(data, plan, time, arm=code) for code in levels}
                    if policy_node
                    else {"history": design}
                )
                partial: tuple[FloatArray, ...] = ()
                if density is not None and factors is not None:
                    weights, _, _ = partial_weights(
                        density, factors, masks, time, horizon, g_bounds
                    )
                    partial = tuple(weights)
                nodes[time] = _Node(
                    time=time,
                    at_risk=at_risk,
                    trained_on=trained_on,
                    design=design,
                    predict_designs=predict_designs,
                    arm=plan.arm(time).astype(np.int64) if policy_node else None,
                    policy=plan.policy_at(time) if policy_node else None,
                    counterfactual=counterfactual,
                    clever=clever,
                    partial=partial,
                )
            inputs.append(_PlanInputs(plan=plan, nodes=nodes))
        return cls(
            data=data,
            inverse=inverse,
            plans=tuple(inputs),
            names=tuple(names),
            estimator=estimator,
            horizon=horizon,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            q_bounds=q_bounds,
            targeting=dict(targeting),
        )

    def counts(self, index: IntArray) -> FloatArray:
        """The frequency of every distinct row in the resample ``index``.

        Parameters
        ----------
        index : ndarray
            Row indices of the original data, repeats kept.

        Returns
        -------
        ndarray
            ``(m,)`` counts over the distinct rows, as floats.
        """
        mapped = self.inverse[np.asarray(index, dtype=np.int64)]
        return np.bincount(mapped, minlength=self.data.n).astype(float)

    def __call__(self, index: IntArray) -> dict[str, tuple[float, float]]:
        """One replicate: every reported estimate and its standard error on the resample.

        Parameters
        ----------
        index : ndarray
            Row indices of the original data, repeats kept.

        Returns
        -------
        dict of str to tuple of float
            ``(estimate, standard error)`` per reported name.
        """
        return self.from_counts(self.counts(index))

    def from_counts(self, counts: FloatArray) -> dict[str, tuple[float, float]]:
        """The replicate of the resample whose distinct-row frequencies are ``counts``.

        Parameters
        ----------
        counts : ndarray
            ``(m,)`` nonnegative integer frequencies over the distinct rows.

        Returns
        -------
        dict of str to tuple of float
            ``(estimate, standard error)`` per reported name.
        """
        counts = np.asarray(counts, dtype=float)
        present = counts > 0.0
        scaler = self._scaler(present)
        seed = seed_carried(self.data, scaler, horizon=self.horizon)
        level: dict[str, tuple[float, FloatArray]] = {}
        for inputs in self.plans:
            run = self._targeted if self.estimator == "tmle" else self._doubly_robust
            level[inputs.plan.label] = run(inputs, counts, present, seed)
        rows = np.flatnonzero(present)
        weight = counts[rows]
        total = float(weight.sum())
        out: dict[str, tuple[float, float]] = {}
        for item in self.names:
            psi, curve = level[item.label]
            if item.reference is None:
                value = scaler.unscale_level(psi)
            else:
                base, base_curve = level[item.reference]
                value = scaler.unscale_difference(psi - base)
                curve = curve - base_curve
            influence = scaler.unscale_influence(curve[rows])
            mean = float(np.sum(weight * influence) / total)
            variance = float(np.sum(weight * (influence - mean) ** 2) / (total - 1.0))
            out[item.name] = (float(value), float(np.sqrt(variance / total)))
        return out

    @staticmethod
    def _fit_rows(node: _Node, present: BoolArray) -> BoolArray:
        """The node's followers that the replicate holds: the rows its regression reads."""
        return node.trained_on & present

    def _scaler(self, present: BoolArray) -> OutcomeScaler:
        """The outcome scaler of the rows present in the replicate."""
        if self.data.family == "binomial":
            return OutcomeScaler.identity()
        rows = self.data.subset(np.flatnonzero(present))
        return OutcomeScaler.from_outcome(rows.observed_outcomes(), self.q_bounds)

    def _regress(
        self,
        node: _Node,
        target: FloatArray,
        counts: FloatArray,
        present: BoolArray,
        rows: IntArray,
        plan: Plan,
        *,
        clip: tuple[float, float] | None,
    ) -> tuple[FloatArray, FloatArray | None]:
        """One node regression on the present rows at risk.

        Returns the prediction at the observed arm and, at a policy node, the ``(r, K_t)``
        predictions at every level, both on ``rows``.
        """
        fitted = self._fit_rows(node, present)
        _require_regimen_followers(
            self.data, plan, node.time, node.at_risk & present, fitted, outer_fold=None
        )
        _check_policy_arm_support(
            self.data, plan, node.time, node.at_risk & present, fitted, outer_fold=None
        )
        last = node.time == self.horizon
        learner = self.outcome_learner if last else self.pseudo_learner
        task = "classification" if last and self.data.family == "binomial" else "regression"
        constant = constant_target(target, fitted)
        if constant is not None:
            predictions = {name: np.full(rows.size, constant) for name in node.predict_designs}
        else:
            fit_rows = np.flatnonzero(fitted)
            design, response, weight = merge_equal_rows(
                node.design[fit_rows], target[fit_rows], counts[fit_rows]
            )
            predictions, _ = cross_fit_predictions(
                learner,
                design,
                response,
                weight,
                Folds.single(response.size),
                task=task,  # type: ignore[arg-type]
                predict_designs={
                    name: matrix[rows] for name, matrix in node.predict_designs.items()
                },
                clip=clip,
                constant_folds=[] if task == "classification" else None,
            )
        if node.policy is None:
            return predictions["history"], None
        assert node.arm is not None
        by_arm = np.column_stack(
            [predictions[f"arm_{code}"] for code in range(node.policy.shape[1])]
        )
        return by_arm[np.arange(rows.size), node.arm[rows]], by_arm

    def _targeted(
        self,
        inputs: _PlanInputs,
        counts: FloatArray,
        present: BoolArray,
        seed: FloatArray,
    ) -> tuple[float, FloatArray]:
        """The targeted recursion of one plan: its scaled estimate and curve per row."""
        plan = inputs.plan
        carried = seed
        curve = np.zeros(self.data.n)
        value = seed
        for time in range(self.horizon, 0, -1):
            node = inputs.nodes[time]
            rows = np.flatnonzero(node.at_risk & present)
            target = _pseudo_outcome(self.data, carried, time, None)
            initial, by_arm = self._regress(
                node, target, counts, present, rows, plan, clip=(0.0, 1.0)
            )
            fitted = self._fit_rows(node, present)[rows]
            fluctuation = _fluctuate_node(
                target[rows],
                initial,
                counts[rows] * node.counterfactual[rows],
                fitted,
                label=plan.label,
                time=time,
                alpha=self.targeting["alpha"],
                max_iter=self.targeting["max_iter"],
                tol=self.targeting["tol"],
                arms=by_arm,
            )
            targeted = fluctuation.targeted.arms[_REGIMEN_ARM]
            if node.policy is None:
                carried_rows = targeted
            else:
                moved = np.column_stack(
                    [
                        fluctuation.targeted.arms[_policy_arm(code)]
                        for code in range(node.policy.shape[1])
                    ]
                )
                carried_rows = _carried(node.policy[rows], moved)
            curve[rows] += node.clever[rows] * (target[rows] - targeted)
            value = np.full(self.data.n, _FILLER)
            value[rows] = carried_rows
            carried = value
        rows = np.flatnonzero(present)
        psi = float(np.average(value[rows], weights=counts[rows]))
        curve[rows] += value[rows] - psi
        return psi, curve

    def _doubly_robust(
        self,
        inputs: _PlanInputs,
        counts: FloatArray,
        present: BoolArray,
        seed: FloatArray,
    ) -> tuple[float, FloatArray]:
        """The sequentially doubly robust recursion of one plan: estimate and curve."""
        plan = inputs.plan
        values: dict[int, FloatArray] = {self.horizon + 1: seed}
        observed: dict[int, FloatArray] = {}
        residual: dict[int, FloatArray] = {}
        carried = seed
        transform = seed
        for time in range(self.horizon, 0, -1):
            node = inputs.nodes[time]
            rows = np.flatnonzero(node.at_risk & present)
            target = _pseudo_outcome(self.data, carried, time, None)
            initial, by_arm = self._regress(node, target, counts, present, rows, plan, clip=None)
            value_rows = initial if by_arm is None else _carried(node.policy[rows], by_arm)  # type: ignore[index]
            values[time] = np.full(self.data.n, _FILLER)
            values[time][rows] = value_rows
            observed[time] = np.full(self.data.n, _FILLER)
            observed[time][rows] = initial
            residual[time] = _pseudo_outcome(self.data, values[time + 1], time, None)
            transform = np.array(values[time], dtype=float, copy=True)
            for later, weight in zip(range(time, self.horizon + 1), node.partial, strict=True):
                transform = transform + weight * (residual[later] - observed[later])
            transform = np.where(node.at_risk, transform, _FILLER)
            carried = transform
        rows = np.flatnonzero(present)
        psi = float(np.average(transform[rows], weights=counts[rows]))
        curve = np.zeros(self.data.n)
        curve[rows] = transform[rows] - psi
        return psi, curve
