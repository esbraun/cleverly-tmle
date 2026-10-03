r"""A four-node survival law with an exact truth for RMST and RMTL.

Every node is binary and absorbing structure is kept:

.. code-block:: text

    W -> L1 -> A1 -> C1 -> Y1 -> L2 -> A2 -> C2 -> Y2 -> ... -> Y4

``C_t = 1`` means the unit is still under observation after node ``t``, and ``Y_t = 1``
is the event at node ``t``.  A censored or failed unit has no later node.  Every treatment
probability lies in ``[0.15, 0.95]``, and treatment persists: a unit treated at one node is
treated at the next with probability of at least 0.75.  Censoring is light and the hazards lie in
``[0.04, 0.26]``.  So at ``n = 1000`` the ``always`` and ``never`` regimens keep about a hundred
followers at node 4, with events, and positivity holds without truncation.

The law serves three uses.

1. :func:`rmst_truth` and :func:`rmtl_truth` state the truth by direct enumeration over
   the counterfactual event times, from the structural conditionals alone.  They sum no
   risk curve, so a fit that sums its reported risks over the wrong range disagrees with
   them.
2. :func:`weighted_frame` lays out one row per observable history with its probability
   as an observation weight.  The weighted empirical law then *is* this law, so a
   saturated fit is exact.  A count layout would need about ``2 * 4**16`` rows.
3. :func:`sample` draws an observed sample for the registered study
   ``full-refit-bootstrap-and-derived-contrasts``.

:func:`functional` is the g-formula of the observed law over :data:`SUPPORT`, and
:func:`eif` its Gateaux derivative by complex step.  Both are written from the observed
probabilities and read no structural parameter.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PROBS",
    "REGIMENS",
    "SUPPORT",
    "K",
    "columns",
    "eif",
    "event_time_pmf",
    "fit_columns",
    "functional",
    "rmst_truth",
    "rmtl_truth",
    "sample",
    "simulate_event_times",
    "weighted_frame",
]

#: Number of treatment nodes.
K = 4

#: ``P(W = 1)``.
P_W1 = 0.4

#: Static regimens: one arm per node.  The first is the reference.
REGIMENS: dict[str, tuple[int, ...]] = {
    "never": (0, 0, 0, 0),
    "always": (1, 1, 1, 1),
    "early": (1, 1, 0, 0),
}


def p_l(w: int, a_prev: int) -> float:
    """``P(L_t = 1 | W, A_{t-1})``, with ``A_0 = 0``."""
    return 0.2 + 0.3 * w + 0.3 * a_prev


def p_a(w: int, l_t: int, a_prev: int) -> float:
    """``P(A_t = 1 | W, L_t, A_{t-1})``."""
    return 0.15 + 0.1 * w + 0.1 * l_t + 0.6 * a_prev


def p_c(w: int, l_t: int, a_t: int) -> float:
    """``P(C_t = 1 | W, L_t, A_t)``, the probability of staying under observation."""
    return 0.95 - 0.03 * l_t - 0.03 * a_t - 0.02 * w


def hazard(w: int, l_t: int, a_t: int, t: int) -> float:
    """``P(Y_t = 1 | W, L_t, A_t, C_t = 1, Y_{t-1} = 0)``."""
    return 0.08 + 0.05 * w + 0.1 * l_t - 0.04 * a_t + 0.01 * (t - 1)


def columns() -> tuple[str, ...]:
    """The wide columns in time order."""
    out = ["W"]
    for t in range(1, K + 1):
        out.extend([f"L{t}", f"A{t}", f"C{t}", f"Y{t}"])
    return tuple(out)


def fit_columns() -> dict[str, Any]:
    """The column keywords of :meth:`cleverly.longitudinal.LTMLE.fit` for this law."""
    return {
        "outcome": [f"Y{t}" for t in range(1, K + 1)],
        "treatment": [f"A{t}" for t in range(1, K + 1)],
        "baseline": ["W"],
        "time_varying": [[f"L{t}"] for t in range(1, K + 1)],
        "censoring": [f"C{t}" for t in range(1, K + 1)],
    }


# ------------------------------------------------------------------ exact truth


def _counterfactual_paths(plan: tuple[int, ...]) -> list[tuple[float, int]]:
    """``(probability, event time)`` of every counterfactual path under ``plan``.

    The event time is ``K + 1`` for a unit event-free through node ``K``.  Censoring is
    intervened away, so it never appears.
    """
    out: list[tuple[float, int]] = []
    for w in (0, 1):
        mass_w = P_W1 if w else 1.0 - P_W1

        def walk(t: int, mass: float, a_prev: int, w: int = w) -> None:
            if t > K:
                out.append((mass, K + 1))
                return
            a_t = plan[t - 1]
            for l_t in (0, 1):
                p = p_l(w, a_prev)
                mass_l = mass * (p if l_t else 1.0 - p)
                h = hazard(w, l_t, a_t, t)
                out.append((mass_l * h, t))
                walk(t + 1, mass_l * (1.0 - h), a_t)

        walk(1, mass_w, 0)
    return out


def rmst_truth(regimen: str, horizon: int) -> float:
    r"""``E[min(T^d, \tau)]`` by enumeration over the counterfactual event times."""
    return float(
        sum(mass * min(time, horizon) for mass, time in _counterfactual_paths(REGIMENS[regimen]))
    )


def rmtl_truth(regimen: str, horizon: int) -> float:
    r"""``E[\tau - min(T^d, \tau)]`` by enumeration.  One cause, so ``1{J = j}`` is one."""
    return float(
        sum(
            mass * (horizon - min(time, horizon))
            for mass, time in _counterfactual_paths(REGIMENS[regimen])
        )
    )


def risk_truth(regimen: str, horizon: int) -> float:
    """``P(T^d <= horizon)`` by enumeration."""
    return float(
        sum(mass for mass, time in _counterfactual_paths(REGIMENS[regimen]) if time <= horizon)
    )


def event_time_pmf(risks: Mapping[int, float]) -> dict[int, float]:
    """The event-time distribution from a risk curve ``{t: F(t)}`` on ``1..K``.

    ``P(T = t) = F(t) - F(t - 1)`` and ``P(T = K + 1) = 1 - F(K)``.  The two-node laws use it
    to state an enumeration truth from their own longhand risks.
    """
    times = sorted(risks)
    out: dict[int, float] = {}
    previous = 0.0
    for t in times:
        out[t] = risks[t] - previous
        previous = risks[t]
    out[times[-1] + 1] = 1.0 - previous
    return out


def enumerate_rmst(pmf: Mapping[int, float], horizon: int) -> float:
    """``sum_t min(t, horizon) P(T = t)``."""
    return float(sum(mass * min(time, horizon) for time, mass in pmf.items()))


# ------------------------------------------------------------- observed support


def _observed_paths() -> list[tuple[tuple[Any, ...], float]]:
    """Every observable history and its probability under the law."""
    out: list[tuple[tuple[Any, ...], float]] = []
    for w in (0, 1):
        mass_w = P_W1 if w else 1.0 - P_W1

        def walk(t: int, prefix: tuple[Any, ...], mass: float, a_prev: int, w: int = w) -> None:
            for l_t, a_t in itertools.product((0, 1), (0, 1)):
                pl = p_l(w, a_prev)
                pa = p_a(w, l_t, a_prev)
                pc = p_c(w, l_t, a_t)
                m = mass * (pl if l_t else 1.0 - pl) * (pa if a_t else 1.0 - pa)
                out.append(((*prefix, l_t, a_t, 0), m * (1.0 - pc)))
                h = hazard(w, l_t, a_t, t)
                out.append(((*prefix, l_t, a_t, 1, 1), m * pc * h))
                stay = (*prefix, l_t, a_t, 1, 0)
                if t == K:
                    out.append((stay, m * pc * (1.0 - h)))
                else:
                    walk(t + 1, stay, m * pc * (1.0 - h), a_t)

        walk(1, (w,), mass_w, 0)
    return out


_PATHS = _observed_paths()

#: One tuple of node values per observable history, in :func:`columns` order and
#: truncated where the unit was censored or had the event.
SUPPORT: tuple[tuple[Any, ...], ...] = tuple(path for path, _ in _PATHS)

#: The probability of each history in :data:`SUPPORT`.
PROBS = np.array([mass for _, mass in _PATHS])


def weighted_frame() -> pd.DataFrame:
    """One row per support point, with its probability in the column ``w``."""
    names = columns()
    rows = {
        name: [float(path[j]) if j < len(path) else np.nan for path in SUPPORT]
        for j, name in enumerate(names)
    }
    frame = pd.DataFrame(rows)
    frame["w"] = PROBS
    return frame


@cache
def _prefix_index() -> dict[tuple[Any, ...], np.ndarray]:
    """Support positions that share each prefix."""
    out: dict[tuple[Any, ...], list[int]] = {}
    for position, path in enumerate(SUPPORT):
        for length in range(1, len(path) + 1):
            out.setdefault(path[:length], []).append(position)
    return {key: np.asarray(value) for key, value in out.items()}


def _risk(mass: Any, regimen: str, horizon: int) -> Any:
    """The g-formula risk, given the mass of each history prefix.

    ``mass(())`` is the total mass.  Every conditional is a ratio of two prefix masses.
    """
    plan = REGIMENS[regimen]

    def risk(prefix: tuple[Any, ...], t: int) -> Any:
        if t > horizon:
            return 0.0
        total = mass(prefix)
        out: Any = 0.0
        a_t = plan[t - 1]
        for l_t in (0, 1):
            with_l = (*prefix, l_t)
            p_l_t = mass(with_l) / total
            uncensored = (*with_l, a_t, 1)
            h = mass((*uncensored, 1)) / mass(uncensored)
            later = risk((*uncensored, 0), t + 1) if t < K else 0.0
            out = out + p_l_t * (h + (1.0 - h) * later)
        return out

    return sum(mass((w,)) / mass(()) * risk((w,), 1) for w in (0, 1))


def functional(probs: Any, regimen: str, horizon: int) -> Any:
    """The risk ``P(T^d <= horizon)`` by the g-formula over the observed law ``probs``.

    Reads only ``probs`` and the support, never a structural parameter.
    """
    index = _prefix_index()

    def mass(prefix: tuple[Any, ...]) -> Any:
        if not prefix:
            return probs.sum()
        positions = index.get(prefix)
        return 0.0 if positions is None else probs[positions].sum()

    return _risk(mass, regimen, horizon)


def eif(regimen: str, horizon: int, *, step: float = 1e-30) -> np.ndarray:
    r"""The Gateaux derivative of the risk at every support point.

    :math:`D(o) = \sum_i \partial\Psi/\partial P_i \, (1\{i = o\} - P_i)`.  The
    functional reads the law only through prefix masses, so each partial derivative is the
    sum over the prefixes of support point ``i`` of the complex-step derivative in that
    prefix's mass.  That is exact to double precision and needs one evaluation per prefix
    the functional reads, rather than one per support point.
    """
    index = _prefix_index()
    real = {prefix: float(PROBS[positions].sum()) for prefix, positions in index.items()}
    real[()] = float(PROBS.sum())
    used: list[tuple[Any, ...]] = []

    def record(prefix: tuple[Any, ...]) -> float:
        used.append(prefix)
        return real[prefix]

    _risk(record, regimen, horizon)
    partial: dict[tuple[Any, ...], float] = {}
    for target in dict.fromkeys(used):

        def bumped(prefix: tuple[Any, ...], target: tuple[Any, ...] = target) -> complex:
            return real[prefix] + (1j * step if prefix == target else 0.0)

        partial[target] = float(np.imag(_risk(bumped, regimen, horizon)) / step)
    gradient = np.array(
        [
            sum(partial.get(path[:length], 0.0) for length in range(len(path) + 1))
            for path in SUPPORT
        ]
    )
    return gradient - float(PROBS @ gradient)


# --------------------------------------------------------------------- sampling


def sample(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` units from the observed law, wide, with ``nan`` after leaving."""
    names = columns()
    out = {name: np.full(n, np.nan) for name in names}
    w = (rng.random(n) < P_W1).astype(int)
    out["W"] = w.astype(float)
    alive = np.ones(n, dtype=bool)
    a_prev = np.zeros(n, dtype=int)
    for t in range(1, K + 1):
        idx = np.flatnonzero(alive)
        wl = w[idx]
        l_t = (rng.random(idx.size) < p_l(wl, a_prev[idx])).astype(int)
        a_t = (rng.random(idx.size) < p_a(wl, l_t, a_prev[idx])).astype(int)
        c_t = (rng.random(idx.size) < p_c(wl, l_t, a_t)).astype(int)
        h = hazard(wl, l_t, a_t, t)
        y_t = np.where(c_t == 1, (rng.random(idx.size) < h).astype(float), np.nan)
        out[f"L{t}"][idx] = l_t
        out[f"A{t}"][idx] = a_t
        out[f"C{t}"][idx] = c_t
        out[f"Y{t}"][idx] = y_t
        a_prev[idx] = a_t
        alive[idx] = (c_t == 1) & (y_t == 0)
    return pd.DataFrame(out)


def simulate_event_times(regimen: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw counterfactual event times under ``regimen``, ``K + 1`` for no event.

    A Monte Carlo statement of the same structural law, which a test compares with the
    enumeration in :func:`rmst_truth`.
    """
    plan = REGIMENS[regimen]
    w = (rng.random(n) < P_W1).astype(int)
    times = np.full(n, K + 1)
    alive = np.ones(n, dtype=bool)
    a_prev = 0
    for t in range(1, K + 1):
        a_t = plan[t - 1]
        l_t = (rng.random(n) < p_l(w, a_prev)).astype(int)
        h = hazard(w, l_t, a_t, t)
        event = alive & (rng.random(n) < h)
        times[event] = t
        alive &= ~event
        a_prev = a_t
    return times
