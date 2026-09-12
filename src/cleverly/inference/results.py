"""Shared scalar-result composition.

Point-treatment and longitudinal estimators produce different scientific artifacts, but
their mapping, covariance, and delta-method operations have exactly the same algebra.
Keeping that algebra here prevents result classes from drifting while leaving their
method-specific reporting and diagnostics separate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .._typing import FloatArray
from .cluster import (
    influence_covariance,
    stacked_second_moment_covariance,
    stacked_second_moment_variance,
)
from .delta import delta_method
from .influence import CovarianceRule, ParameterEstimate, Scale, make_estimate

__all__ = [
    "covariance_rule",
    "estimate_covariance",
    "estimate_curves",
    "select_estimates",
    "smooth_contrast",
    "sole_estimate",
]


def sole_estimate(estimates: Mapping[str, ParameterEstimate]) -> ParameterEstimate:
    """Return the sole estimate, refusing to guess on a multi-parameter result."""
    if len(estimates) != 1:
        raise ValueError(
            "this result contains multiple parameters; index the one you want from "
            f"{list(estimates)}"
        )
    return next(iter(estimates.values()))


def select_estimates(
    estimates: Mapping[str, ParameterEstimate], names: Sequence[str] | None
) -> tuple[str, ...]:
    """Normalize and validate a requested ordered subset of estimates."""
    chosen = tuple(estimates) if names is None else tuple(names)
    if not chosen:
        raise ValueError(f"no parameters selected; this result reports {list(estimates)}")
    missing = [name for name in chosen if name not in estimates]
    if missing:
        raise KeyError(f"unknown parameter(s) {missing}; this result reports {list(estimates)}")
    return chosen


def estimate_curves(estimates: Mapping[str, ParameterEstimate]) -> dict[str, FloatArray]:
    """Influence curves in the result's stable report order."""
    return {name: estimate.influence_curve for name, estimate in estimates.items()}


def covariance_rule(
    estimates: Mapping[str, ParameterEstimate],
    names: Sequence[str],
    *,
    cluster: Any = None,
) -> CovarianceRule:
    """The one covariance rule every selected estimate declares.

    A selection that mixes rules is refused. No derivation here supplies the
    cross-covariance between a centred curve and a raw second moment.
    """
    rules = {estimates[name].covariance_rule for name in names}
    if len(rules) != 1:
        raise ValueError(
            f"the selected estimates {list(names)} declare different covariance rules "
            f"{sorted(rules)}; select estimates that share one rule"
        )
    rule = rules.pop()
    if rule == "second_moment" and cluster is not None:
        raise ValueError(
            "the raw second-moment covariance rule is defined for independent rows only, "
            "and this result declares clusters"
        )
    return rule


def estimate_covariance(
    estimates: Mapping[str, ParameterEstimate],
    names: Sequence[str] | None,
    *,
    cluster: Any = None,
) -> FloatArray:
    """Joint covariance under the rule the selected estimates declare.

    The ``"centered"`` rule is the sample covariance at the observation or declared cluster
    unit. The ``"second_moment"`` rule is the raw second moment. Both rules give the same
    diagonal for any selection size.
    """
    chosen = select_estimates(estimates, names)
    rule = covariance_rule(estimates, chosen, cluster=cluster)
    curves = np.column_stack([estimates[name].influence_curve for name in chosen])
    if rule == "second_moment":
        return stacked_second_moment_covariance(curves)
    return influence_covariance(curves, cluster=cluster)


def smooth_contrast(
    estimates: Mapping[str, ParameterEstimate],
    function: Callable[[FloatArray], float],
    names: Sequence[str],
    *,
    n: int,
    cluster: Any = None,
    alpha: float = 0.05,
    name: str | None = None,
    scale: Scale = "difference",
    gradient: Callable[[FloatArray], FloatArray] | None = None,
) -> ParameterEstimate:
    """Apply the delta method to a smooth function of jointly estimated parameters.

    The derived estimate inherits the covariance rule of its inputs, and its variance
    applies that rule to the derived influence curve. Under either rule, that variance
    equals the quadratic form of the gradient with :func:`estimate_covariance`.
    """
    chosen = select_estimates(estimates, names)
    rule = covariance_rule(estimates, chosen, cluster=cluster)
    value, curve = delta_method(
        function,
        [estimates[key].psi for key in chosen],
        [estimates[key].influence_curve for key in chosen],
        gradient=gradient,
    )
    estimate = make_estimate(
        name or f"contrast({', '.join(chosen)})",
        value,
        curve,
        n=n,
        cluster=cluster,
        scale=scale,
        alpha=alpha,
    )
    if rule == "second_moment":
        estimate = replace(
            estimate,
            variance=stacked_second_moment_variance(curve),
            covariance_rule=rule,
        )
    return estimate
