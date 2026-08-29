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


def truth_for(q: np.ndarray, p_w: np.ndarray) -> dict[str, float]:
    """Evaluate all registered parameters directly under a supplied marginal law.

    The study distinguishes two targets on the same outcome regression, so the marginal
    law is an argument rather than a default.  Pass :data:`P_W` for the population target
    the weights recover, and :data:`SELECTED_P_W` for the target the omitted-weight
    control recovers instead.

    Parameters
    ----------
    q : numpy.ndarray
        ``P(Y = 1 | A = a, W = w)``, indexed by level and then arm.
    p_w : numpy.ndarray
        The marginal law of ``W`` to average the arms over.

    Returns
    -------
    dict of str to float
        Each registered parameter under that marginal law.
    """
    means = np.asarray(p_w, dtype=float) @ np.asarray(q, dtype=float)
    ey0, ey1 = (float(means[0]), float(means[1]))
    return {
        "ey0": ey0,
        "ey1": ey1,
        "ate": ey1 - ey0,
        "rr": ey1 / ey0,
        "or": (ey1 / (1.0 - ey1)) / (ey0 / (1.0 - ey0)),
    }


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


def weighted_arm_eifs(q: np.ndarray = Q, *, g: np.ndarray = G) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted ``ey0`` and ``ey1`` influence curves on :data:`SUPPORT`.

    The rows follow the selected law. Tilting them by ``1 / selection(W)`` recovers the
    population law, so the outer density ratio is ``selection_rate / selection(W)``.

    Every contrast this study reports is a smooth function of the two arm means, so each
    contrast's curve is a fixed linear combination of these two.  Writing the arms once is
    what lets the log-risk-ratio and log-odds-ratio bounds below share one derivation with
    the ATE bound rather than restate it twice.

    Parameters
    ----------
    q : numpy.ndarray, optional
        ``P(Y = 1 | A = a, W = w)``, indexed by level and then arm.
    g : numpy.ndarray, optional
        ``P(A = 1 | W = w)``, indexed by level.

    Returns
    -------
    tuple of numpy.ndarray
        The ``ey0`` curve and the ``ey1`` curve, in that order.
    """
    truth = truth_for(q, P_W)
    zero = np.empty(len(SUPPORT), dtype=float)
    one = np.empty(len(SUPPORT), dtype=float)
    for index, (w, a, y) in enumerate(SUPPORT):
        residual = y - q[w, a]
        ratio = SELECTION_RATE / SELECTION[w]
        zero[index] = ratio * ((1.0 - a) / (1.0 - g[w]) * residual + q[w, 0] - truth["ey0"])
        one[index] = ratio * (a / g[w] * residual + q[w, 1] - truth["ey1"])
    return zero, one


def weighted_ate_eif(q: np.ndarray = Q, *, g: np.ndarray = G) -> np.ndarray:
    """Return the weighted ATE influence curve at each selected-law support point."""
    zero, one = weighted_arm_eifs(q, g=g)
    return one - zero


def weighted_logrr_eif(q: np.ndarray = Q, *, g: np.ndarray = G) -> np.ndarray:
    """Return the weighted log-risk-ratio influence curve, ``D1 / mu1 - D0 / mu0``."""
    truth = truth_for(q, P_W)
    zero, one = weighted_arm_eifs(q, g=g)
    return one / truth["ey1"] - zero / truth["ey0"]


def weighted_logor_eif(q: np.ndarray = Q, *, g: np.ndarray = G) -> np.ndarray:
    """Return the weighted log-odds-ratio influence curve on :data:`SUPPORT`.

    The delta-method factor of ``log(mu / (1 - mu))`` is ``1 / (mu (1 - mu))``, so the
    curve is ``D1 / (mu1 (1 - mu1)) - D0 / (mu0 (1 - mu0))``.

    Parameters
    ----------
    q : numpy.ndarray, optional
        ``P(Y = 1 | A = a, W = w)``, indexed by level and then arm.
    g : numpy.ndarray, optional
        ``P(A = 1 | W = w)``, indexed by level.

    Returns
    -------
    numpy.ndarray
        The curve at each support point, in support order.
    """
    truth = truth_for(q, P_W)
    zero, one = weighted_arm_eifs(q, g=g)
    ey0, ey1 = truth["ey0"], truth["ey1"]
    return one / (ey1 * (1.0 - ey1)) - zero / (ey0 * (1.0 - ey0))


def _efficiency_sd(curve: np.ndarray, q: np.ndarray, g: np.ndarray) -> float:
    """Return ``sqrt(E_selected[D*(O)^2])`` for one curve under the exact finite law."""
    probabilities = selected_probabilities(q, g=g).reshape(-1)
    return float(np.sqrt(np.sum(probabilities * np.square(curve))))


def weighted_ate_efficiency_sd(q: np.ndarray = Q, *, g: np.ndarray = G) -> float:
    """Return the exact ATE efficiency bound, scaled by the square root of the size."""
    return _efficiency_sd(weighted_ate_eif(q, g=g), q, g)


def weighted_logrr_efficiency_sd(q: np.ndarray = Q, *, g: np.ndarray = G) -> float:
    """Return the exact log-risk-ratio efficiency bound, on the reported inference scale."""
    return _efficiency_sd(weighted_logrr_eif(q, g=g), q, g)


def weighted_logor_efficiency_sd(q: np.ndarray = Q, *, g: np.ndarray = G) -> float:
    """Return the exact log-odds-ratio efficiency bound, on the reported inference scale."""
    return _efficiency_sd(weighted_logor_eif(q, g=g), q, g)
