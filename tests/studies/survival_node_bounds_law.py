r"""The law, truths, valid bounds and efficiency bounds of ``survival-node-bounds`` (X33).

The law is the declaration of ``X22-plan.md`` II-3.4: a point treatment, binary ``A``, binary
``W1`` and ``W2``, and ``K = 5`` integer visits.

==============================  =============================================================
factor                          law
==============================  =============================================================
``W1``, ``W2``                  independent Bernoulli(:data:`P_W1`), Bernoulli(:data:`P_W2`)
treatment                       :math:`g(1 \mid W) = \operatorname{expit}(0.3 + 0.5 W_1 - 0.4 W_2)`
event hazard at visit ``t``     :math:`h_t = \operatorname{expit}(-4.5 + 0.5 A + 0.6 W_1 - 0.4 W_2 + 0.1 t)`
censoring hazard at ``t``       :math:`c_t = \operatorname{expit}(-4.0 + 0.2 A + 0.3 W_1 + 0.15 t)`
==============================  =============================================================

The declaration names no marginal law for ``W``.  This module fixes both at one half
(:data:`P_W1`, :data:`P_W2`); the design note records the choice.

At visit ``t`` an event is checked first, and then a unit without an event is censored with
probability :math:`c_t`.  A unit censored at visit ``t`` has ``time = t`` and ``event = 0``, so
:meth:`~cleverly.longitudinal.LongitudinalData.from_time_to_event` observes it through node
``t`` and censors it before node ``t + 1``.  That is ``survtmle``'s order (an event before a
censoring at one time).  Censoring at visit 5 ends nothing, because the study ends there.  So
node ``t`` retains a unit at risk with probability :math:`1 - c_{t-1}`, and node 1 retains
every unit.

Every truth, iterated mean, bound and efficiency bound below is a finite sum over the four
covariate cells, the two arms and the ten observed histories.  None of it calls the estimator.
:func:`exact_frame` is the same law written as 80 weighted records, so a fit on it with
observation weights fits the law itself (``P_n = P_0``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from cleverly.longitudinal import LongitudinalData

#: The node count and the grid of the time-to-event layout.
K = 5
GRID = tuple(float(t) for t in range(1, K + 1))
#: The marginal laws of the two baseline covariates (not named by the declaration).
P_W1 = 0.5
P_W2 = 0.5
#: The horizons the declared bounds read, and the horizons every fit reports.
BOUND_HORIZONS = (3, 5)
HORIZONS = (3, 5)
ARMS = (0, 1)
REGIMENS = {"arm1": 1, "arm0": 0}
REFERENCE = "arm0"
#: The declared factor of the valid bounds, and the factor the design test may drop it to.
DECLARED_FACTOR = 1.5
REDUCED_FACTOR = 1.1

CELLS: tuple[tuple[int, int], ...] = tuple((w1, w2) for w1 in (0, 1) for w2 in (0, 1))


def p_w(w1: int, w2: int) -> float:
    """The probability of one covariate cell."""
    return (P_W1 if w1 else 1.0 - P_W1) * (P_W2 if w2 else 1.0 - P_W2)


def g1(w1: Any, w2: Any) -> Any:
    """:math:`P(A = 1 \\mid W)`."""
    return expit(0.3 + 0.5 * np.asarray(w1, dtype=float) - 0.4 * np.asarray(w2, dtype=float))


def g(a: int, w1: int, w2: int) -> float:
    """:math:`P(A = a \\mid W)`."""
    p = float(g1(w1, w2))
    return p if a == 1 else 1.0 - p


def hazard(t: Any, a: Any, w1: Any, w2: Any) -> Any:
    """The event hazard at visit ``t`` among the units at risk there."""
    return expit(
        -4.5
        + 0.5 * np.asarray(a, dtype=float)
        + 0.6 * np.asarray(w1, dtype=float)
        - 0.4 * np.asarray(w2, dtype=float)
        + 0.1 * np.asarray(t, dtype=float)
    )


def censoring(t: Any, a: Any, w1: Any, w2: Any) -> Any:
    """The censoring hazard at visit ``t`` among the units with no event through ``t``."""
    del w2
    return expit(
        -4.0
        + 0.2 * np.asarray(a, dtype=float)
        + 0.3 * np.asarray(w1, dtype=float)
        + 0.15 * np.asarray(t, dtype=float)
    )


def retention(t: int, a: int, w1: int, w2: int) -> float:
    """:math:`P(C_t = 1 \\mid` at risk at node ``t``, uncensored before it): ``1 - c_{t-1}``."""
    return 1.0 if t == 1 else 1.0 - float(censoring(t - 1, a, w1, w2))


def iterated_mean(t: int, horizon: int, a: int, w1: int, w2: int) -> float:
    """:math:`Q_t = P(T \\le \\text{horizon} \\mid T \\ge t, A = a, W)` at a node ``t <= horizon``."""
    if not 1 <= t <= horizon <= K:
        raise ValueError(f"need 1 <= t <= horizon <= {K}; got t={t}, horizon={horizon}")
    survive = 1.0
    for s in range(t, horizon + 1):
        survive *= 1.0 - float(hazard(s, a, w1, w2))
    return 1.0 - survive


def risk(horizon: int, a: int) -> float:
    """:math:`F(\\text{horizon} \\mid a) = E_W[P(T \\le \\text{horizon} \\mid A = a, W)]`."""
    return sum(p_w(*cell) * iterated_mean(1, horizon, a, *cell) for cell in CELLS)


def risk_name(label: str, horizon: int) -> str:
    """The name a fit reports the risk of ``label`` at ``horizon`` by."""
    return f"risk_regimen[{label} @ t={horizon}]"


def contrast_name(horizon: int) -> str:
    """The name of the ``arm1 vs arm0`` risk difference at ``horizon``."""
    return f"ate_regimen[arm1 vs {REFERENCE} @ t={horizon}]"


def truths() -> dict[str, float]:
    """Every reported level and contrast: ``F(t | a)`` and ``RD(t)`` at the reported horizons."""
    out: dict[str, float] = {}
    for horizon in HORIZONS:
        for label, a in REGIMENS.items():
            out[risk_name(label, horizon)] = risk(horizon, a)
        out[contrast_name(horizon)] = risk(horizon, 1) - risk(horizon, 0)
    return out


TRUTH: dict[str, float] = truths()


# ------------------------------------------------------------------------------ bounds


def at_risk_maximum(t: int) -> float:
    """The largest at-risk iterated mean at node ``t`` over ``a``, ``W`` and the bound horizons.

    A horizon below ``t`` has no node ``t``, so it does not enter.
    """
    return max(
        iterated_mean(t, horizon, a, *cell)
        for horizon in BOUND_HORIZONS
        if horizon >= t
        for a in ARMS
        for cell in CELLS
    )


def hazard_maximum(t: int) -> float:
    """The largest true event hazard at node ``t`` over ``a`` and ``W``."""
    return max(float(hazard(t, a, *cell)) for a in ARMS for cell in CELLS)


def mean_bounds(factor: float) -> dict[int, tuple[float, float]]:
    """The valid bounds of the iterated-mean fit: ``(0, factor * at-risk maximum)`` per node.

    Every node of a horizon-5 fit is named.  A horizon-3 fit reads nodes 1 to 3 of the same
    mapping.
    """
    return {t: (0.0, factor * at_risk_maximum(t)) for t in range(1, K + 1)}


def hazard_bounds(factor: float) -> dict[int, tuple[float, float]]:
    """The valid bounds of the hazard fit: ``(0, factor * hazard maximum)`` per node."""
    return {t: (0.0, factor * hazard_maximum(t)) for t in range(1, K + 1)}


# ------------------------------------------------------------------------------ histories


@dataclass(frozen=True)
class History:
    """One observed history after ``(W, A)``.

    Parameters
    ----------
    time : int
        The recorded follow-up time.
    event : int
        ``1`` for an event at ``time``, ``0`` for a censoring at ``time`` or the study's end.
    mass : float
        Its probability given ``(W, A)``.
    """

    time: int
    event: int
    mass: float


def histories(a: int, w1: int, w2: int) -> Iterator[History]:
    """The ten observed histories given ``(W, A)``: an event at 1..5, a censoring at 1..4, or
    event-free through 5."""
    alive = 1.0
    for t in range(1, K + 1):
        h = float(hazard(t, a, w1, w2))
        yield History(t, 1, alive * h)
        alive *= 1.0 - h
        if t < K:
            c = float(censoring(t, a, w1, w2))
            yield History(t, 0, alive * c)
            alive *= 1.0 - c
    yield History(K, 0, alive)


def exact_frame() -> pd.DataFrame:
    """The law as weighted records: one row per ``(W, A, history)``, ``weight`` its probability."""
    rows = []
    for w1, w2 in CELLS:
        for a in ARMS:
            for history in histories(a, w1, w2):
                weight = p_w(w1, w2) * g(a, w1, w2) * history.mass
                rows.append(
                    {
                        "W1": float(w1),
                        "W2": float(w2),
                        "A": float(a),
                        "time": float(history.time),
                        "event": int(history.event),
                        "weight": weight,
                    }
                )
    frame = pd.DataFrame(rows)
    total = float(frame["weight"].sum())
    if abs(total - 1.0) > 1e-12:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the exact frame has mass {total}")
    return frame


# ------------------------------------------------------------------------------ the curve


def marginal_g(a: int) -> float:
    """:math:`P(A = a)`: the limit of an intercept-only treatment mechanism."""
    return sum(p_w(*cell) * g(a, *cell) for cell in CELLS)


def influence(
    horizon: int,
    arm: int,
    a: int,
    w1: int,
    w2: int,
    history: History,
    *,
    intercept_only: bool = False,
) -> float:
    r"""The efficient influence function of :math:`F(\text{horizon} \mid \text{arm})` at a record.

    With ``intercept_only`` the treatment probability is :func:`marginal_g`.  That is the
    influence function of a fit with a correct outcome regression and an intercept-only
    treatment mechanism, the positive arm of N.

    .. math::

        D = \frac{1\{A = \text{arm}\}}{g(\text{arm} \mid W)} \sum_{t \le k}
            \frac{1\{\text{at risk and observed at } t\}}{\prod_{s \le t} r_s}
            (Z_t - Q_t) + Q_1(\text{arm}, W) - \psi,

    with :math:`Z_t = 1` after an event at ``t``, :math:`Z_t = Q_{t+1}` after none, and
    :math:`Z_k` the event indicator at ``k`` (Benkeser, Carone and Gilbert 2018, Section 2.3).
    """
    psi = risk(horizon, arm)
    total = iterated_mean(1, horizon, arm, w1, w2) - psi
    if a != arm:
        return total
    ratio = 1.0 / (marginal_g(arm) if intercept_only else g(arm, w1, w2))
    retained = 1.0
    for t in range(1, horizon + 1):
        retained *= retention(t, a, w1, w2)
        # Observed at node t: no event before t, and not censored before node t.
        if history.time < t:
            break
        if history.time == t - 1 and history.event == 0:  # pragma: no cover - covered by break
            break
        q = iterated_mean(t, horizon, a, w1, w2)
        if history.time == t and history.event == 1:
            z = 1.0
        elif t == horizon:
            z = 0.0
        else:
            z = iterated_mean(t + 1, horizon, a, w1, w2)
        total += ratio / retained * (z - q)
        if history.time == t:
            # An event ends the history; a censoring at t ends observation after node t.
            break
    return total


def records() -> Iterator[tuple[float, int, int, int, History]]:
    """Every ``(probability, a, w1, w2, history)`` of the law."""
    for w1, w2 in CELLS:
        for a in ARMS:
            for history in histories(a, w1, w2):
                yield p_w(w1, w2) * g(a, w1, w2) * history.mass, a, w1, w2, history


def efficiency_sd(name: str, *, intercept_only: bool = False) -> float:
    """The standard deviation of the efficient influence function of a reported parameter.

    ``intercept_only`` reads :func:`influence` with :func:`marginal_g`.
    """
    horizon, terms = _terms(name)
    first = 0.0
    second = 0.0
    for probability, a, w1, w2, history in records():
        value = sum(
            coefficient * influence(horizon, arm, a, w1, w2, history, intercept_only=intercept_only)
            for arm, coefficient in terms
        )
        first += probability * value
        second += probability * value**2
    if abs(first) > 1e-12:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the influence function of {name} has mean {first}")
    return float(np.sqrt(second))


def _terms(name: str) -> tuple[int, tuple[tuple[int, float], ...]]:
    for horizon in HORIZONS:
        if name == contrast_name(horizon):
            return horizon, ((1, 1.0), (0, -1.0))
        for label, arm in REGIMENS.items():
            if name == risk_name(label, horizon):
                return horizon, ((arm, 1.0),)
    raise KeyError(f"{name} is not a reported parameter")


EFFICIENCY_SD: dict[str, float] = {name: efficiency_sd(name) for name in TRUTH}


# ------------------------------------------------------------------------------ sampling


def expected_events(n: int, horizon: int = K) -> dict[int, float]:
    """The expected number of observed events by ``horizon`` in each arm of a sample of ``n``."""
    out = {}
    for arm in ARMS:
        total = 0.0
        for probability, a, _, _, history in records():
            if a == arm and history.event == 1 and history.time <= horizon:
                total += probability
        out[arm] = n * total
    return out


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` units: ``W1``, ``W2``, ``A``, ``time`` and ``event``."""
    rng = np.random.default_rng(seed)
    w1 = (rng.random(n) < P_W1).astype(float)
    w2 = (rng.random(n) < P_W2).astype(float)
    a = (rng.random(n) < g1(w1, w2)).astype(float)
    time = np.full(n, float(K))
    event = np.zeros(n, dtype=int)
    open_ = np.ones(n, dtype=bool)
    for t in range(1, K + 1):
        fired = open_ & (rng.random(n) < hazard(t, a, w1, w2))
        time[fired] = float(t)
        event[fired] = 1
        open_ &= ~fired
        censored = rng.random(n) < censoring(t, a, w1, w2)
        if t < K:
            dropped = open_ & censored
            time[dropped] = float(t)
            open_ &= ~dropped
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "time": time, "event": event})


def container(frame: pd.DataFrame, *, weights: str | None = None) -> LongitudinalData:
    """The fit's container: the time-to-event converter on the integer grid."""
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=GRID,
        weights=weights,
    )


def as_mapping(bounds: Mapping[int, tuple[float, float]]) -> dict[float, tuple[float, float]]:
    """The bounds keyed by grid time, as ``node_bounds=`` takes them."""
    return {GRID[t - 1]: (float(lo), float(hi)) for t, (lo, hi) in bounds.items()}


__all__ = [
    "EFFICIENCY_SD",
    "GRID",
    "HORIZONS",
    "REFERENCE",
    "REGIMENS",
    "TRUTH",
    "K",
    "container",
    "draw",
    "exact_frame",
    "hazard_bounds",
    "mean_bounds",
]
