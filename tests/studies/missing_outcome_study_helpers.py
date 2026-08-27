"""Shared finite-law helpers for missing-outcome evidence studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tests import discrete_law_mar as mar


def probabilities(
    q: np.ndarray = mar.Q,
    *,
    g: np.ndarray = mar.G,
    pi: np.ndarray = mar.PI,
    p_w: np.ndarray = mar.P_W,
) -> np.ndarray:
    """Build an observed-data MAR law from its three nuisance functions."""
    out = np.empty((len(p_w), 2, 3), dtype=float)
    for w, a, kind in mar.SUPPORT:
        arm = g[w] if a == 1 else 1.0 - g[w]
        observed = pi[w, a]
        if kind == mar.UNOBSERVED:
            cell = 1.0 - observed
        else:
            outcome = q[w, a] if kind == mar.OBSERVED_ONE else 1.0 - q[w, a]
            cell = observed * outcome
        out[w, a, kind] = p_w[w] * arm * cell
    if not np.isclose(out.sum(), 1.0):  # pragma: no cover - declaration guard
        raise AssertionError("the declared MAR cell probabilities do not sum to one")
    return out


def sample_discrete(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` observed-data rows from a declared finite MAR law."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(mar.SUPPORT), size=n, p=np.asarray(probs).reshape(-1))
    values = np.asarray(mar.SUPPORT, dtype=float)[cells]
    kind = values[:, 2]
    return pd.DataFrame(
        {
            "W": values[:, 0],
            "A": values[:, 1],
            "Y": np.where(kind == mar.UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == mar.UNOBSERVED, 0.0, 1.0),
        }
    )


def truths(probs: np.ndarray, estimands: Sequence[str]) -> dict[str, float]:
    """Evaluate the independent observed-data oracle for each estimand."""
    return {name: float(mar.functional(probs, name)) for name in estimands}


def initial_arm_estimates(result: Any) -> dict[str, float]:
    """Return the untargeted treatment-specific means and their difference."""
    weights = np.asarray(result.data.weights, dtype=float)
    means = {
        arm: float(
            np.average(
                result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[arm]),
                weights=weights,
            )
        )
        for arm in result.data.arm_codes
    }
    return {"ey0": means[0.0], "ey1": means[1.0], "ate": means[1.0] - means[0.0]}


def primary_rows(
    *,
    result: Any,
    reference: Mapping[str, float],
    implementation: str,
    scenario: str,
    replicate: int,
    estimands: Sequence[str],
) -> list[dict[str, Any]]:
    """Convert one missing-outcome fit to the registered primary schema."""
    initials = initial_arm_estimates(result)
    rows: list[dict[str, Any]] = []
    for name in estimands:
        estimate = result[name]
        low, high = estimate.ci
        truth = float(reference[name])
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.data.n,
                "estimand": name,
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": initials[name],
            }
        )
    return rows


def efficiency_sd(probs: np.ndarray, estimand: str) -> float:
    """Standard deviation of the oracle observed-data influence curve."""
    base = np.asarray(probs, dtype=complex)
    step = 1e-30
    curve = np.empty(len(mar.SUPPORT))
    for point, support in enumerate(mar.SUPPORT):
        mass = np.zeros_like(base)
        mass[support] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        curve[point] = np.imag(mar.functional(perturbed, estimand)) / step
    return float(np.sqrt(np.sum(np.asarray(probs).reshape(-1) * curve**2)))
