r"""The change-of-measure sequential plug-in of the influence-curve variance.

``LTMLE(variance="policy_weighted")`` and ``SDR(variance="policy_weighted")`` report, for
every level and contrast, the larger of the influence-curve variance and the variance
computed here (:mod:`cleverly.inference.policy`).  This module computes the second one.

**The representation.**  A fit's influence curve is
:math:`D = b + \sum_t \omega_t e_t`, with :math:`b = Q^d_1 - \psi`, the node residual
:math:`e_t = Q^d_{t+1} - Q_t(A_t, H_t)` (``pseudo_outcome - targeted`` of each
:class:`~cleverly.longitudinal.sequential.SequentialStep`) and :math:`\omega_t` the cumulative
ratio (``clever``).  The intervened law :math:`g^d_t` is the push-forward of
:math:`g_t` by the policy, so for :math:`r_t = g^d_t / g_t` and any integrable :math:`f`,

.. math::

    E[r_t(A_t, H_t)^k f(A_t, H_t) \mid H_t] = E[r_t(A^d_t, H_t)^{k-1} f(A^d_t, H_t) \mid H_t].

Iterated expectation then writes :math:`\sigma^2 = \operatorname{Var} D` as a backward
recursion.  Where every residual has conditional mean zero given :math:`(A_t, H_t)`, as at
the truth, its cross terms vanish and

.. math::

    \sigma^2 = E[b^2] + E[r_1(A^d_1) S_1(A^d_1, H_1)], \quad
    S_t = E[e_t^2 + r_{t+1}(A^d_{t+1}) S_{t+1}(A^d_{t+1}, H_{t+1}) \mid A_t, H_t],

with :math:`S_{\tau+1} = 0`.  This *diagonal form* is what ships.  Each :math:`S_t` is a
regression fitted at the observed :math:`(A_t, H_t)` and evaluated at the policy dose
:math:`A^d_t`, so a node's term reads the ratio at every row's policy dose rather than the
few rows whose observed ratio is large.  The residual itself is always read at the observed
dose.  Under a misspecified outcome model the dropped cross terms need not vanish, and the
diagonal form may converge to a number other than :math:`\sigma^2`; the reported maximum
with the influence-curve variance keeps the interval at least as wide as the default.

**Covariances.**  For two plans :math:`d` and :math:`d'` the same recursion runs on
:math:`e^d_t e^{d'}_t` with the factor
:math:`E[r^d r^{d'} f \mid H] = E[r^{d'}(A^d) f(A^d) \mid H]`.  At a categorical node the
change of measure is the exact sum :math:`\sum_a g^d(a) g^{d'}(a) / g(a)\, f(a)` over the
levels, which covers a static, dynamic or stochastic plan as well as a modified treatment
policy.  At a continuous node it is :math:`\sum_\beta p_\beta\, r^{d'}(d_\beta(A))
f(d_\beta(A))` over the policy's branches.  A node whose policy moves no dose has ratio one
exactly.  A plan that moves no dose at any node is the natural course: its change of measure
is the identity, so its variance is the influence-curve variance itself.

The point-treatment closed form of Tran, Petersen, Schwab and van der Laan (2023, J. Causal
Inference 11: 20210067, Section 3.1.2) covers static and dynamic regimes; the density-ratio
form above follows from the lemma, which needs no invertibility of the policy map.

**The variance regressions** sit behind :func:`_fit_variance_regression`: a Poisson-loss
gradient-boosted regression for the non-negative variance targets and a squared-loss one for
the covariance targets, with the settings of :data:`VARIANCE_LEARNER`.  The interval is valid
under any regression because of the maximum; the settings fix which variance was measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .._typing import BoolArray, FloatArray
from ..data.validate import arm_indicators
from ..interventions.policy import induced_probabilities, policy_branches, policy_moves_dose
from ..learners._threads import thread_limit
from .data import LongitudinalData
from .regimen import Plan
from .sequential import Mechanism, RegimenFit

__all__ = ["VARIANCE_LEARNER", "natural_levels", "policy_weighted_variance"]

#: The settings of every variance regression: a gradient-boosted tree regression with learning
#: rate 0.1, an L2 penalty of 1, no early stopping, depth 5, 200 iterations and leaves of at
#: least 20 rows.  No setting is exposed: the maximum makes any regression valid, and only
#: these settings are measured (``mtp-policy-variance``).
VARIANCE_LEARNER: Final[Mapping[str, Any]] = {
    "learning_rate": 0.1,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "max_depth": 5,
    "max_iter": 200,
    "min_samples_leaf": 20,
}

Loss = Literal["poisson", "squared_error"]


def policy_weighted_variance(
    data: LongitudinalData,
    fits: Sequence[RegimenFit],
    plans: Mapping[str, Plan],
    mechanism: Mechanism,
    *,
    seed: int | None = None,
) -> FloatArray:
    r"""The policy-weighted covariance of a fit's level estimates, on the targeting scale.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel: end of study, unclustered, unweighted and uncensored.
    fits : sequence of RegimenFit
        One fit per plan, in the order of the matrix.
    plans : mapping of str to Plan
        Each fit's plan by label, with its policy numerators attached.
    mechanism : Mechanism
        The fitted mechanism, with the ratios at the policy doses
        (``fit_mechanism(..., shifted_ratios=True)``).
    seed : int or None, default=None
        The ``random_state`` of each variance regression.

    Returns
    -------
    ndarray
        ``(k, k)`` second moments divided by ``n``: the diagonal form of each plan's variance
        and of each pair's covariance, on the ``[0, 1]`` outcome scale the recursion runs
        on.  A plan that moves no dose takes its influence-curve variance on the diagonal.
        An entry whose regressions could not be fitted is ``nan``.
    """
    n = data.n
    parts = [_Parts.of(fit, plans[fit.regimen.label], data) for fit in fits]
    k = len(parts)
    matrix = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i, k):
            left, right = parts[i], parts[j]
            if i == j and left.natural:
                value = float(np.var(left.curve, ddof=1))
            else:
                carried = _recursion(
                    data, mechanism, left, right, power=2 if i == j else 1, seed=seed
                )
                value = float(np.mean(left.base * right.base) + np.mean(carried))
            matrix[i, j] = matrix[j, i] = value / n
    return matrix


class _Parts:
    """What the recursion reads from one plan's fit."""

    def __init__(
        self,
        plan: Plan,
        curve: FloatArray,
        base: FloatArray,
        residuals: tuple[FloatArray, ...],
        trained_on: tuple[BoolArray, ...],
        at_risk: tuple[BoolArray, ...],
        natural: bool,
    ) -> None:
        self.plan = plan
        self.curve = curve
        self.base = base
        self.residuals = residuals
        self.trained_on = trained_on
        self.at_risk = at_risk
        self.natural = natural

    @classmethod
    def of(cls, fit: RegimenFit, plan: Plan, data: LongitudinalData) -> _Parts:
        steps = sorted(fit.steps, key=lambda step: step.time)
        residuals = _residuals(fit)
        curve = np.asarray(fit.influence_curve_scaled, dtype=float)
        base = curve - sum(
            (
                np.asarray(step.clever, dtype=float) * e
                for step, e in zip(steps, residuals, strict=True)
            ),
            start=np.zeros(data.n),
        )
        natural = _natural(plan, data)
        return cls(
            plan,
            curve,
            base,
            residuals,
            tuple(np.asarray(step.trained_on, dtype=bool) for step in steps),
            tuple(np.asarray(step.at_risk, dtype=bool) for step in steps),
            natural,
        )


def natural_levels(
    data: LongitudinalData, fits: Sequence[RegimenFit], plans: Mapping[str, Plan]
) -> tuple[bool, ...]:
    """By fit, whether its plan moves no dose at any node: the natural course.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    fits : sequence of RegimenFit
        The fits, in the order of the matrix of :func:`policy_weighted_variance`.
    plans : mapping of str to Plan
        Each fit's plan by label.

    Returns
    -------
    tuple of bool
        ``True`` where :func:`policy_weighted_variance` puts the influence-curve variance on
        the diagonal.
    """
    return tuple(_natural(plans[fit.regimen.label], data) for fit in fits)


def _natural(plan: Plan, data: LongitudinalData) -> bool:
    """Whether ``plan`` is a policy at every node that moves no dose."""
    return all(
        plan.is_mtp_node(time) and not _moves(plan, time) for time in range(1, data.n_times + 1)
    )


def _residuals(fit: RegimenFit) -> tuple[FloatArray, ...]:
    """Each node's residual :math:`e_t`, read at the observed dose, and zero off its rows."""
    return tuple(
        np.where(step.trained_on, np.asarray(step.pseudo_outcome) - np.asarray(step.targeted), 0.0)
        for step in sorted(fit.steps, key=lambda step: step.time)
    )


def _moves(plan: Plan, time: int) -> bool:
    """Whether a policy node of ``plan`` moves some dose at ``time`` by its declaration."""
    return bool(policy_moves_dose(plan.regimen.plan[time - 1]))  # type: ignore[union-attr]


def _drawn(plan: Plan, time: int) -> bool:
    """Whether ``plan`` draws its arm at ``time`` rather than assigning it by a rule."""
    return plan.is_mtp_node(time) or plan.is_policy_node(time)


def _recursion(
    data: LongitudinalData,
    mechanism: Mechanism,
    left: _Parts,
    right: _Parts,
    *,
    power: int,
    seed: int | None,
) -> FloatArray:
    """The first node's carried term of the diagonal recursion for one plan or one pair."""
    carry = np.zeros(data.n)
    loss: Loss = "poisson" if power == 2 else "squared_error"
    for time in range(data.n_times, 0, -1):
        index = time - 1
        fit_rows = left.trained_on[index] & right.trained_on[index]
        target = left.residuals[index] * right.residuals[index] + carry
        if loss == "poisson":
            target = np.maximum(target, 0.0)
        observed, points = _measure(data, mechanism, left.plan, right.plan, time)
        if all(not np.any(weight) for _, weight in points):
            # No history reaches a dose both plans draw (two static plans that differ here),
            # so the carried term is zero and no regression is needed.
            carry = np.zeros(data.n)
            continue
        if not fit_rows.any():
            return np.full(data.n, np.nan)
        predictions = _fit_variance_regression(
            observed[fit_rows],
            target[fit_rows],
            [design for design, _ in points],
            loss,
            seed=seed,
        )
        moved = sum(
            (
                weight * prediction
                for (_, weight), prediction in zip(points, predictions, strict=True)
            ),
            start=np.zeros(data.n),
        )
        carry = np.where(left.at_risk[index] & right.at_risk[index], moved, 0.0)
    return carry


def _measure(
    data: LongitudinalData,
    mechanism: Mechanism,
    left: Plan,
    right: Plan,
    time: int,
) -> tuple[FloatArray, list[tuple[FloatArray, FloatArray]]]:
    r"""The observed design of node ``time``, and the change of measure as weighted designs.

    Returns the design the regression is fitted on and the pairs
    ``(design, weight)`` such that
    :math:`E[r^{left} r^{right} f \mid H] = \sum_j w_j f(\text{design}_j)` row by row.
    """
    history = data.history_design(time)
    if data.node_kind(time) == "continuous":
        dose = np.nan_to_num(np.asarray(data.node_codes(time), dtype=float), nan=0.0)
        observed = np.column_stack([dose, history])
        moves_left, moves_right = _moves(left, time), _moves(right, time)
        if not (moves_left or moves_right):
            return observed, [(observed, np.ones(data.n))]
        # Push forward through a plan that moves the dose; the other plan's ratio is read at
        # its policy doses, and is one where that plan moves nothing.
        source, other, other_moves = (
            (left, right, moves_right) if moves_left else (right, left, moves_left)
        )
        branches = policy_branches(source.regimen.plan[time - 1])  # type: ignore[union-attr]
        targets = source.carry_targets(time)
        assert targets is not None
        points = []
        for branch, ((probability, _), target) in enumerate(zip(branches, targets, strict=True)):
            ratio = (
                mechanism.shifted_ratios[other.label][time][(source.label, branch)]
                if other_moves
                else np.ones(data.n)
            )
            design = np.column_stack([np.asarray(target, dtype=float), history])
            points.append((design, probability * np.asarray(ratio, dtype=float)))
        return observed, points
    # A categorical node: the exact sum over the levels.
    g = np.asarray(mechanism.treatment_observed[time - 1], dtype=float)
    weight = _intervened(left, time, g) * _intervened(right, time, g)
    safe = np.where(g > 0.0, g, 1.0)
    weight = np.where(g > 0.0, weight / safe, 0.0)
    levels = g.shape[1]
    if not (_drawn(left, time) or _drawn(right, time)):
        # Both plans assign the arm by a rule: the regression is fitted on the rows that
        # took it, and holds no treatment column, as the outcome regression does.
        return history, [(history, weight.sum(axis=1))]
    codes = np.nan_to_num(np.asarray(data.node_codes(time), dtype=float), nan=0.0)
    observed = np.column_stack([arm_indicators(codes.astype(np.int64), levels), history])
    points = [
        (
            np.column_stack(
                [arm_indicators(np.full(data.n, level, dtype=np.int64), levels), history]
            ),
            weight[:, level],
        )
        for level in range(levels)
    ]
    return observed, points


def _intervened(plan: Plan, time: int, g: FloatArray) -> FloatArray:
    r"""``(n, K)`` :math:`g^d_t(\cdot \mid H_t)`, the law the plan draws a categorical arm from."""
    n, levels = g.shape
    rows = np.arange(n)
    if plan.is_mtp_node(time):
        assigned = plan.mtp_assignments[time - 1]
        assert assigned is not None
        out = np.zeros((n, levels))
        for probability, level_map in assigned:
            out = out + probability * induced_probabilities(level_map, g)
        return out
    if plan.is_policy_node(time):
        return np.asarray(plan.policy_at(time), dtype=float)
    arm = np.nan_to_num(np.asarray(plan.arm(time), dtype=float), nan=0.0).astype(np.int64)
    out = np.zeros((n, levels))
    out[rows, arm] = 1.0
    return out


def _fit_variance_regression(
    design: FloatArray,
    target: FloatArray,
    evaluate_at: Sequence[FloatArray],
    loss: Loss,
    *,
    seed: int | None = None,
) -> list[FloatArray]:
    """Fit one variance regression and predict it at each design of ``evaluate_at``.

    The one place the variance regressions are fitted, so a targeted variant can replace it.

    Parameters
    ----------
    design : ndarray
        ``(m, p)`` the observed design of the rows the regression is fitted on.
    target : ndarray
        ``(m,)`` the target; non-negative under ``loss="poisson"``.
    evaluate_at : sequence of ndarray
        ``(n, p)`` designs to predict at.
    loss : {"poisson", "squared_error"}
        The loss: Poisson for a variance target, squared error for a covariance target.
    seed : int or None, default=None
        The regression's ``random_state``.

    Returns
    -------
    list of ndarray
        One ``(n,)`` prediction per design.  A constant target predicts the constant.
    """
    values = np.asarray(target, dtype=float)
    if values.size < 2 or np.ptp(values) == 0.0:
        constant = float(values.mean()) if values.size else float("nan")
        return [np.full(np.asarray(points).shape[0], constant) for points in evaluate_at]
    model = HistGradientBoostingRegressor(loss=loss, random_state=seed, **VARIANCE_LEARNER)
    # The nuisance fits' thread limit (cleverly.learners.set_thread_limit): an unlimited
    # OpenMP pool beside other busy processes slowed one fit by two orders of magnitude.
    with thread_limit():
        model.fit(np.asarray(design, dtype=float), values)
        return [
            np.asarray(model.predict(np.asarray(points, dtype=float)), dtype=float)
            for points in evaluate_at
        ]
