"""Exact continuous law for learned weighted point-treatment studies."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SELECTION_SLOPE = 0.75
EFFECT_MODIFICATION = 2.0
TARGET_ATE = 1.0
SELECTED_W1_MEAN = SELECTION_SLOPE / 3.0
SELECTED_ATE = TARGET_ATE + EFFECT_MODIFICATION * SELECTED_W1_MEAN


def weighted_ate_efficiency_sd() -> float:
    """Exact selected-law SD of the target-population ATE influence curve."""
    slope = SELECTION_SLOPE
    integral = (
        -2.0 / slope**2 + (1.0 + 1.0 / slope**2) * math.log((1.0 + slope) / (1.0 - slope)) / slope
    )
    return math.sqrt(2.0 * integral)


def selected_density(w1: np.ndarray) -> np.ndarray:
    """Density of the selected ``W1`` law on ``[-1, 1]``."""
    values = np.asarray(w1, dtype=float)
    return (1.0 + SELECTION_SLOPE * values) / 2.0


def inverse_selection_weight(w1: np.ndarray) -> np.ndarray:
    """Radon-Nikodym derivative from the selected law to the uniform target law."""
    values = np.asarray(w1, dtype=float)
    return 1.0 / (1.0 + SELECTION_SLOPE * values)


def outcome_mean(
    w1: np.ndarray, w2: np.ndarray, treatment: np.ndarray, *, effect: float
) -> np.ndarray:
    """Conditional outcome mean under the declared effect and effect modification."""
    return (
        0.5
        + 0.5 * np.asarray(w1, dtype=float)
        + 0.25 * np.asarray(w2, dtype=float)
        + np.asarray(treatment, dtype=float)
        * (effect + EFFECT_MODIFICATION * np.asarray(w1, dtype=float))
    )


def truths(*, effect: float = TARGET_ATE) -> dict[str, float]:
    """Exact target-law arm means and ATE."""
    return {"ey0": 0.5, "ey1": 0.5 + effect, "ate": effect}


def sample_selected(n: int, seed: int, *, effect: float = TARGET_ATE) -> pd.DataFrame:
    """Draw exactly ``n`` rows from the selected continuous law."""
    rng = np.random.default_rng(seed)
    uniform = rng.uniform(size=n)
    slope = SELECTION_SLOPE
    # The other quadratic root is outside [-1, 1], so this branch is exact.
    w1 = (-1.0 + np.sqrt((1.0 - slope) ** 2 + 4.0 * slope * uniform)) / slope
    w2 = rng.uniform(-1.0, 1.0, size=n)
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    outcome = outcome_mean(w1, w2, treatment, effect=effect) + rng.normal(size=n)
    return pd.DataFrame(
        {
            "Y": outcome,
            "A": treatment,
            "W1": w1,
            "W2": w2,
            "obs_weight": inverse_selection_weight(w1),
        }
    )
