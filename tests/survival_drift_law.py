r"""Finite laws for the drift-corrected survival TMLE, in the source's indexing.

The implementation never enters this module.  A law is a finite support of baseline points
:math:`W` with probabilities, a binary treatment and :math:`K` time points.  Each row of the
expanded law is one atom :math:`(W, A, \text{path})`, with the path one of: censored at
:math:`m = 0, \ldots, K - 1` (after the event check at :math:`m`), an event at
:math:`t = 1, \ldots, K`, or event-free and uncensored through :math:`K`.  The order is that of
Díaz (2019): :math:`R_0, L_1, R_1, L_2, \ldots, R_{K-1}, L_K`.  Each row carries its exact
probability, so every population mean is a weighted sum.

:func:`population_regression` is the exact regression on a scalar index: the weighted mean of
the outcome over the subpopulation rows sharing the index value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["FiniteLaw", "expand", "population_regression"]

Nuisance = Callable[[int, np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class FiniteLaw:
    """The expanded rows of a finite law.

    Parameters
    ----------
    weights : numpy.ndarray
        ``(N,)`` the probability of each atom.
    point : numpy.ndarray
        ``(N,)`` the index of the baseline point.
    treatment : numpy.ndarray
        ``(N,)`` the treatment.
    at_risk : numpy.ndarray
        ``(N, K)`` :math:`I_t` at column ``t - 1``.
    events : numpy.ndarray
        ``(N, K)`` :math:`L_t` at column ``t - 1``.
    eligible : numpy.ndarray
        ``(N, K)`` :math:`J_m` at column ``m``.
    censored : numpy.ndarray
        ``(N, K)`` :math:`R_m` at column ``m``.
    """

    weights: np.ndarray
    point: np.ndarray
    treatment: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    eligible: np.ndarray
    censored: np.ndarray


def expand(
    w_prob: np.ndarray,
    g_a: Callable[[np.ndarray], np.ndarray],
    g_r: Nuisance,
    h: Nuisance,
    n_times: int,
) -> FiniteLaw:
    """Expand a law into its atoms.

    Parameters
    ----------
    w_prob : numpy.ndarray
        ``(P,)`` the probability of each baseline point.
    g_a : callable
        ``g_a(points)``: :math:`P(A = 1 \\mid W)` at the point indices.
    g_r : callable
        ``g_r(m, points, a)``: :math:`P(R_m = 1 \\mid J_m = 1, A = a, W)`.
    h : callable
        ``h(t, points, a)``: :math:`P(L_t = 1 \\mid I_t = 1, A = a, W)`.
    n_times : int
        :math:`K`.

    Returns
    -------
    FiniteLaw
        The atoms.
    """
    points = np.arange(w_prob.size)
    rows: list[tuple[np.ndarray, np.ndarray, int, str, int]] = []
    weights, point, treatment, kind, time = [], [], [], [], []
    for a in (0, 1):
        pa = g_a(points) if a == 1 else 1.0 - g_a(points)
        alive = w_prob * pa
        for m in range(n_times):
            # R_m, then L_{m+1}.
            censor = g_r(m, points, a)
            weights.append(alive * censor)
            point.append(points)
            treatment.append(np.full(points.size, a))
            kind.append(np.full(points.size, 0))
            time.append(np.full(points.size, m))
            alive = alive * (1.0 - censor)
            event = h(m + 1, points, a)
            weights.append(alive * event)
            point.append(points)
            treatment.append(np.full(points.size, a))
            kind.append(np.full(points.size, 1))
            time.append(np.full(points.size, m + 1))
            alive = alive * (1.0 - event)
        weights.append(alive)
        point.append(points)
        treatment.append(np.full(points.size, a))
        kind.append(np.full(points.size, 2))
        time.append(np.full(points.size, n_times))
    del rows
    w = np.concatenate(weights)
    p = np.concatenate(point)
    a_ = np.concatenate(treatment)
    k_ = np.concatenate(kind)
    t_ = np.concatenate(time)
    n = w.size
    at_risk = np.zeros((n, n_times), dtype=bool)
    events = np.zeros((n, n_times))
    eligible = np.zeros((n, n_times), dtype=bool)
    censored = np.zeros((n, n_times))
    for i in range(n):
        if k_[i] == 0:  # censored at m = t_
            m = t_[i]
            at_risk[i, :m] = True  # I_t for t <= m
            eligible[i, : m + 1] = True  # J for 0..m
            censored[i, m] = 1.0
        elif k_[i] == 1:  # event at t = t_
            t = t_[i]
            at_risk[i, :t] = True
            events[i, t - 1] = 1.0
            eligible[i, : t - 1 + 1] = True  # J_0..J_{t-1}
        else:
            at_risk[i, :] = True
            eligible[i, :] = True
    return FiniteLaw(w, p, a_, at_risk, events, eligible, censored)


def population_regression(
    weights: np.ndarray, digits: int = 11
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """The exact regression on a scalar index under the law's weights.

    Parameters
    ----------
    weights : numpy.ndarray
        ``(N,)`` the atom probabilities.
    digits : int
        The rounding that identifies equal index values.

    Returns
    -------
    callable
        ``regress(x, y, rows)``: at every row, the weighted mean of ``y`` over the ``rows``
        sharing that row's index value, zero where no such row has mass.
    """

    def regress(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
        keys = np.round(np.asarray(x, dtype=float), digits)
        out = np.zeros(keys.size)
        mass = np.where(rows, weights, 0.0)
        for value in np.unique(keys):
            at = keys == value
            total = mass[at].sum()
            if total > 0.0:
                out[at] = (mass[at] * y[at]).sum() / total
        return out

    return regress
