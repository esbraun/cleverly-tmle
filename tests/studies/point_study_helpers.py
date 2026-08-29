"""Shared result transcription for binary point-treatment evidence studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def initial_estimates(result: Any) -> dict[str, float]:
    """Return untargeted binary-arm parameters on their native reporting scales."""
    weights = np.asarray(result.data.weights, dtype=float)
    arms = {
        arm: float(
            np.average(
                result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[arm]),
                weights=weights,
            )
        )
        for arm in (0.0, 1.0)
    }
    ey0, ey1 = arms[0.0], arms[1.0]
    return {
        "ey0": ey0,
        "ey1": ey1,
        "ate": ey1 - ey0,
        "rr": ey1 / ey0,
        "or": (ey1 / (1.0 - ey1)) / (ey0 / (1.0 - ey0)),
    }


def primary_rows(
    *,
    result: Any,
    truth: Mapping[str, float],
    implementation: str,
    scenario: str,
    replicate: int,
    estimands: Sequence[str],
) -> list[dict[str, Any]]:
    """Convert a binary point fit to the evidence schema with native ratio inference."""
    initials = initial_estimates(result)
    rows: list[dict[str, Any]] = []
    for name in estimands:
        estimate = result[name]
        reference = float(truth[name])
        low, high = estimate.ci
        ratio = estimate.scale == "ratio"
        if ratio and estimate.log_psi is None:  # pragma: no cover - estimator contract guard
            raise AssertionError(f"ratio estimand {name!r} has no log-scale estimate")
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.data.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": (float(estimate.log_psi) if ratio else float(estimate.psi)),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if ratio else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": initials[name],
            }
        )
    return rows
