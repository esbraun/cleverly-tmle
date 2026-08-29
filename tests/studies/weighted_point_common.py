"""Finite-law helpers for the weighted point-treatment evidence study."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

#: Population law. Arrays indexed by the three levels of ``W`` and, for ``Q``, by arm.
P_W = np.array([0.50, 0.30, 0.20])
G = np.array([0.30, 0.60, 0.40])
Q = np.array([[0.20, 0.30], [0.30, 0.70], [0.10, 0.90]])

#: Inclusion probabilities create the biased sampling law. The supplied probability weights
#: invert this selection exactly; their overall scale is immaterial to the Hajek estimator.
SELECTION = np.array([0.15, 0.50, 0.90])
SELECTION_RATE = float(P_W @ SELECTION)
SELECTED_P_W = P_W * SELECTION / SELECTION_RATE
OBSERVATION_WEIGHTS = 1.0 / SELECTION

SUPPORT: tuple[tuple[int, int, int], ...] = tuple(
    itertools.product(range(len(P_W)), range(2), range(2))
)


class FinitePointLaw:
    """Exact binary nuisance functions for the declared finite law."""

    def __init__(self, *, q: np.ndarray = Q, g: np.ndarray = G) -> None:
        self.q = np.asarray(q, dtype=float)
        self.g = np.asarray(g, dtype=float)

    @staticmethod
    def _levels(covariates: Any) -> np.ndarray:
        design = np.asarray(covariates, dtype=float)
        return np.rint(design.reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        """Return ``P(A=1 | W)`` for each supplied row."""
        return self.g[self._levels(covariates)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: Any) -> np.ndarray:
        """Return ``P(Y=1 | A=arm, W)`` for each supplied row."""
        del intermediate
        return self.q[self._levels(covariates), int(arm)]


def truth_for(q: np.ndarray = Q, *, p_w: np.ndarray = P_W) -> dict[str, float]:
    """Evaluate all registered parameters directly under a supplied marginal law."""
    means = np.asarray(p_w, dtype=float) @ np.asarray(q, dtype=float)
    ey0, ey1 = (float(means[0]), float(means[1]))
    return {
        "ey0": ey0,
        "ey1": ey1,
        "ate": ey1 - ey0,
        "rr": ey1 / ey0,
        "or": (ey1 / (1.0 - ey1)) / (ey0 / (1.0 - ey0)),
    }


def population_truth(q: np.ndarray = Q) -> dict[str, float]:
    """Return the target recovered after inverse-selection weighting."""
    return truth_for(q, p_w=P_W)


def selected_truth(q: np.ndarray = Q) -> dict[str, float]:
    """Return the target recovered when the inverse-selection weights are omitted."""
    return truth_for(q, p_w=SELECTED_P_W)


def sample_selected(q: np.ndarray, n: int, seed: int, *, g: np.ndarray = G) -> pd.DataFrame:
    """Draw exactly ``n`` rows directly from the biased selected law."""
    rng = np.random.default_rng(seed)
    w = rng.choice(len(P_W), size=n, p=SELECTED_P_W)
    a = rng.binomial(1, np.asarray(g, dtype=float)[w])
    y = rng.binomial(1, np.asarray(q, dtype=float)[w, a])
    return pd.DataFrame(
        {
            "Y": y.astype(float),
            "A": a.astype(float),
            "W": w.astype(float),
            "obs_weight": OBSERVATION_WEIGHTS[w],
        }
    )


def selected_probabilities(q: np.ndarray = Q, *, g: np.ndarray = G) -> np.ndarray:
    """Return ``P_selected(W, A, Y)`` on :data:`SUPPORT`."""
    probabilities = np.empty((len(P_W), 2, 2), dtype=float)
    for w, a, y in SUPPORT:
        treatment = g[w] if a else 1.0 - g[w]
        outcome = q[w, a] if y else 1.0 - q[w, a]
        probabilities[w, a, y] = SELECTED_P_W[w] * treatment * outcome
    if not np.isclose(probabilities.sum(), 1.0):  # pragma: no cover - declaration guard
        raise AssertionError("the selected finite-law probabilities do not sum to one")
    return probabilities


def weighted_ate_eif(q: np.ndarray = Q, *, g: np.ndarray = G) -> np.ndarray:
    """Return the weighted ATE influence curve at each selected-law support point.

    The rows follow the selected law. Tilting them by ``1 / selection(W)`` recovers the
    population law, so the outer density ratio is ``selection_rate / selection(W)``.
    """
    truth = population_truth(q)["ate"]
    curve = np.empty(len(SUPPORT), dtype=float)
    for index, (w, a, y) in enumerate(SUPPORT):
        residual = y - q[w, a]
        clever = a / g[w] - (1.0 - a) / (1.0 - g[w])
        population_eif = clever * residual + q[w, 1] - q[w, 0] - truth
        curve[index] = SELECTION_RATE / SELECTION[w] * population_eif
    return curve


def weighted_ate_efficiency_sd(q: np.ndarray = Q, *, g: np.ndarray = G) -> float:
    """Return ``sqrt(E_selected[D*(O)^2])`` from the exact finite law."""
    probabilities = selected_probabilities(q, g=g).reshape(-1)
    curve = weighted_ate_eif(q, g=g)
    return float(np.sqrt(np.sum(probabilities * np.square(curve))))
