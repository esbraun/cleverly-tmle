"""Shared scalar-result composition.

Point-treatment and longitudinal estimators produce different scientific artifacts, but
their mapping, covariance, and delta-method operations have exactly the same algebra.
Keeping that algebra here prevents result classes from drifting while leaving their
method-specific reporting and diagnostics separate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .._typing import FloatArray
from .cluster import influence_covariance
from .delta import delta_method
from .influence import ParameterEstimate, Scale, make_estimate

__all__ = [
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


def estimate_covariance(
    estimates: Mapping[str, ParameterEstimate],
    names: Sequence[str] | None,
    *,
    cluster: Any = None,
) -> FloatArray:
    """Joint covariance at the observation or declared cluster unit."""
    chosen = select_estimates(estimates, names)
    curves = np.column_stack([estimates[name].influence_curve for name in chosen])
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
    """Apply the delta method to a smooth function of jointly estimated parameters."""
    chosen = select_estimates(estimates, names)
    value, curve = delta_method(
        function,
        [estimates[key].psi for key in chosen],
        [estimates[key].influence_curve for key in chosen],
        gradient=gradient,
    )
    return make_estimate(
        name or f"contrast({', '.join(chosen)})",
        value,
        curve,
        n=n,
        cluster=cluster,
        scale=scale,
        alpha=alpha,
    )
