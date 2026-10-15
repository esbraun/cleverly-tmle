"""Internal mutations and exact readings for the X39 long-horizon witnesses.

Each mutation is a context manager that patches one module-global name of
:mod:`cleverly.longitudinal.sequential` or :class:`cleverly.longitudinal.data.LongitudinalData`
for the duration of a fit, so a unit test and the design driver run the same mutation.  The
input-level mutations (a retention column set to one, a lagged rule, a policy without its
earlier-arm input, a two-arm complement mechanism) need no patch: they are other plans, other
frames or other learners, and live in :mod:`tests.studies.long_horizon_common`.

The readings take a fit on the weighted exact frame of :mod:`tests.discrete_law_longitudinal_horizon`
and return the quantity a witness asserts on: the estimate, a node's fluctuation, a node's score,
and the departure of the fit's curve from the oracle curve at the fit's own regressions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest import mock

import numpy as np

from cleverly.fluctuation.iterative import Fluctuation, InitialFit
from cleverly.longitudinal import sequential
from cleverly.longitudinal.data import (
    LongitudinalData,
    RegimenMasks,
    _prefix_all,
    assignment_matrix,
)
from cleverly.longitudinal.regimen import Plan
from tests import discrete_law_longitudinal_horizon as law_
from tests.discrete_law_longitudinal_horizon import Law


# ----------------------------------------------------------------------------- mutations
@contextmanager
def skip_update(nodes: Collection[int]) -> Iterator[None]:
    """Node ``t`` in ``nodes`` keeps its initial regression: ``epsilon_t = 0``, no solve."""
    real = sequential._fluctuate_node

    def patched(
        pseudo_outcome: Any,
        initial: Any,
        loss_weights: Any,
        fitted_on: Any,
        *,
        label: str,
        time: int,
        arms: Any = None,
        **settings: Any,
    ) -> Fluctuation:
        if time not in nodes:
            return real(
                pseudo_outcome,
                initial,
                loss_weights,
                fitted_on,
                label=label,
                time=time,
                arms=arms,
                **settings,
            )
        initial_arms = {sequential._REGIMEN_ARM: initial}
        if arms is not None:
            for code in range(arms.shape[1]):
                initial_arms[sequential._policy_arm(code)] = arms[:, code]
        zero = np.zeros(1)
        return Fluctuation(
            epsilon=zero,
            targeted=InitialFit(initial, initial_arms),
            score=zero,
            converged=True,
            n_iter=0,
            trace=(0.0,),
            method="iterative",
            names=(f"epsilon[{label}, t={time}]",),
            score_initial=zero,
        )

    with mock.patch.object(sequential, "_fluctuate_node", patched):
        yield


@contextmanager
def untargeted_pseudo_outcome(nodes: Collection[int]) -> Iterator[None]:
    """Node ``t - 1`` regresses node ``t``'s untargeted value, for each ``t`` in ``nodes``.

    The step at node ``t`` keeps its solved fluctuation and reports it, but the value it carries
    to node ``t - 1`` is the initial regression.  ``nodes`` holds the later node of each pair, so
    the every-node mutation is ``range(2, K + 1)``.
    """
    real = sequential._targeted_step

    def patched(node: Any, fluctuation: Any, **settings: Any) -> Any:
        step = real(node, fluctuation, **settings)
        if node.time not in nodes:
            return step
        return dataclasses.replace(
            step,
            targeted=node.initial,
            marginal=node.initial_marginal,
            targeted_by_arm=node.initial_by_arm,
        )

    with mock.patch.object(sequential, "_targeted_step", patched):
        yield


@contextmanager
def censored_units_follow() -> Iterator[None]:
    """The plan's censoring mask is dropped: a unit censored at node ``t`` still follows it.

    The patch is on :meth:`Plan.masks`, which the recursion reads.  The mechanism fit and the
    declared-mechanism preflight read the data's own masks and keep the censoring.
    """
    real = Plan.masks

    def patched(self: Plan, data: LongitudinalData) -> RegimenMasks:
        masks = real(self, data)
        return RegimenMasks(
            uncensored=np.ones_like(masks.uncensored),
            followed=masks.followed,
            event_free=masks.event_free,
        )

    with mock.patch.object(Plan, "masks", patched):
        yield


@contextmanager
def rule_check_dropped(node: int) -> Iterator[None]:
    """The plan's follower mask at ``node`` keeps every treated row, whatever the plan assigned."""
    real = Plan.masks

    def patched(self: Plan, data: LongitudinalData) -> RegimenMasks:
        masks = real(self, data)
        matches = data.treatment == assignment_matrix(self.values, data.n, data.n_times)
        if self.has_policy:
            matches = matches & np.asarray(self.support(data), dtype=bool)
        matches[:, node - 1] = ~np.isnan(data.treatment[:, node - 1])
        return RegimenMasks(
            uncensored=masks.uncensored,
            followed=_prefix_all(matches),
            event_free=masks.event_free,
        )

    with mock.patch.object(Plan, "masks", patched):
        yield


class _AtRiskFollowing:
    """A plan's masks whose ``following(node)`` reads ``at_risk(node)``."""

    def __init__(self, masks: RegimenMasks, node: int) -> None:
        self._masks = masks
        self._node = node

    def at_risk(self, time: int) -> Any:
        return self._masks.at_risk(time)

    def following(self, time: int) -> Any:
        if time == self._node:
            return self._masks.at_risk(time)
        return self._masks.following(time)


@contextmanager
def fit_on_at_risk(node: int) -> Iterator[None]:
    """Node ``node``'s learner is fitted on the rows at risk instead of the followers.

    Only the learner's rows change: the clever covariate and the fluctuation still read the
    followers.
    """
    real = sequential._fit_node_regression

    def patched(
        data: Any, plan: Any, carried: Any, time: int, horizon: int, **settings: Any
    ) -> Any:
        if time != node:
            return real(data, plan, carried, time, horizon, **settings)
        masks = settings.pop("masks", None) or plan.masks(data)
        regression = real(
            data,
            plan,
            carried,
            time,
            horizon,
            masks=cast(Any, _AtRiskFollowing(masks, node)),
            **settings,
        )
        following = masks.following(time)
        return dataclasses.replace(regression, trained_on=following, fitted_on=following)

    with mock.patch.object(sequential, "_fit_node_regression", patched):
        yield


# ----------------------------------------------------------------------------- readings
def estimate(result: Any, plan: str) -> float:
    """The reported estimate of ``plan``."""
    return float(result[f"ey_regimen[{plan}]"].psi)


def epsilon(result: Any, plan: str, t: int) -> float:
    """Node ``t``'s fluctuation coefficient."""
    return float(result.fits[plan].steps[t - 1].fluctuation.epsilon[0])


def node_score(result: Any, plan: str, t: int) -> float:
    """Node ``t``'s weighted score ``mean(obs_weight * clever * (pseudo - targeted))``."""
    fit = result.fits[plan]
    step = fit.steps[t - 1]
    weight = np.asarray(fit.obs_weights, dtype=float)
    residual = np.where(step.trained_on, step.pseudo_outcome - step.targeted, 0.0)
    return float(np.mean(weight * step.clever * residual))


def regressions(result: Any, plan: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The fit's targeted node regressions in the oracle's ``curve()`` form."""
    return {
        step.time: (np.asarray(step.targeted, float), np.asarray(step.value, float))
        for step in result.fits[plan].steps
    }


def curve_departure(
    result: Any, plan: str, K: int, m: int, law: Law, weight: np.ndarray, oracle_plan: str = ""
) -> float:
    """Sup norm of the fit's per-unit curve minus the oracle curve at the fit's regressions.

    The oracle curve uses the law's true ratios of ``oracle_plan`` (``plan`` when empty).
    """
    name = f"ey_regimen[{plan}]"
    curve = np.asarray(result[name].influence_curve, dtype=float)
    per_unit = curve / (len(weight) * weight / np.sum(weight))
    oracle = law_.curve(K, m, oracle_plan or plan, law, regressions(result, plan))
    return float(np.max(np.abs(per_unit - oracle)))


def eif_departure(result: Any, plan: str, K: int, m: int, law: Law, weight: np.ndarray) -> float:
    """Sup norm of the fit's per-unit curve minus the efficient influence function."""
    name = f"ey_regimen[{plan}]"
    curve = np.asarray(result[name].influence_curve, dtype=float)
    per_unit = curve / (len(weight) * weight / np.sum(weight))
    return float(np.max(np.abs(per_unit - law_.eif(K, m, plan, law))))
