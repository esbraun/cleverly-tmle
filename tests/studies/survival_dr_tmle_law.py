r"""Law L-S of the ``survival-dr-tmle`` study, with its truths and efficient spreads.

:math:`W_1, W_2` are independent standard normals.  Five integer times; at each time
:math:`t = 1, \ldots, 5` the event fires with hazard

.. math::

    h_0(t, a, w) = \operatorname{expit}(-2 + 0.5 a - w_1 + 2 w_2 + 0.1 t),

and an event-free unit is then censored with hazard
:math:`c_0(w) = \operatorname{expit}(-2.5 + 0.4 w_1 + 0.6 w_2)` for :math:`t < 5`, so the
source's censoring hazard is :math:`g_{R,0}(0) = 0` and :math:`g_{R,0}(m) = c_0` for
:math:`m = 1, \ldots, 4`.  The treatment is :math:`A \sim \operatorname{Bern}(0.5)` on the trial
law and :math:`\operatorname{Bern}\{\operatorname{expit}(0.6 w_1 + 0.3 w_2)\}` on the observational
law.  The audit (``X91-F-audit.md``) chose the law so that neither nuisance's error is a
function of the other's index.

**Truth.** :math:`F_a(5) = 1 - E_W \prod_{t \le 5}\{1 - h_0(t, a, W)\}` depends on :math:`W`
through :math:`s = -W_1 + 2 W_2 \sim N(0, 5)` only, so one Gauss-Hermite rule of 200 nodes gives
it to rounding (:func:`truth`).

**Efficient spread.** With :math:`\pi_a(t, w) = g_a(w) G_0(t, w)`, :math:`S_a` the survival and
:math:`D_a` the risk's efficient influence function,

.. math::

    E\{D_a^2\} = \operatorname{Var}\{S_a(5, W)\} + E_W \sum_{t \le 5}
    \frac{S_a(t - 1)\{S_a(5)/S_a(t)\}^2 h_0(t)\{1 - h_0(t)\}}{\pi_a(t)},

and the risk difference adds the two martingale sums to the variance of
:math:`S_1(5) - S_0(5)`, since the arms' martingale terms are orthogonal to each other and to
every function of :math:`W`.  A product Gauss-Hermite rule of 80 nodes per axis evaluates it
(:func:`efficiency_sd`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

#: The number of times, the horizon and the grid.
K = 5
HORIZON = 5
GRID = tuple(float(t) for t in range(1, K + 1))
ARMS = {"a1": 1, "a0": 0}
REFERENCE = "a0"
ESTIMANDS = (
    f"risk_regimen[a1 @ t={HORIZON}]",
    f"risk_regimen[a0 @ t={HORIZON}]",
    f"ate_regimen[a1 vs a0 @ t={HORIZON}]",
)


def hazard(t: int, a: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """The event hazard :math:`h_0(t, a, w)`."""
    return np.asarray(expit(-2.0 + 0.5 * a - w1 + 2.0 * w2 + 0.1 * t), dtype=float)


def censoring(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """The censoring hazard :math:`c_0(w)` after the event check at ``t < 5``."""
    return np.asarray(expit(-2.5 + 0.4 * w1 + 0.6 * w2), dtype=float)


def treatment(w1: np.ndarray, w2: np.ndarray, *, trial: bool) -> np.ndarray:
    """:math:`P(A = 1 \\mid W)` on the trial or observational law."""
    if trial:
        return np.full(np.shape(w1), 0.5)
    return np.asarray(expit(0.6 * w1 + 0.3 * w2), dtype=float)


def sample(n: int, seed: int, *, trial: bool) -> pd.DataFrame:
    """One draw as a time-to-event frame.

    The trial frame also carries the declared probabilities: ``p1``, ``p0`` and the retention
    ``r1`` to ``r4`` after the event check at grid times 1 to 4.
    """
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, treatment(w1, w2, trial=trial)).astype(float)
    leave = censoring(w1, w2)
    time = np.full(n, float(K))
    event = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for t in range(1, K + 1):
        hit = alive & (rng.uniform(size=n) < hazard(t, a, w1, w2))
        time[hit], event[hit] = float(t), 1
        alive &= ~hit
        if t < K:
            gone = alive & (rng.uniform(size=n) < leave)
            time[gone] = float(t)
            alive &= ~gone
    frame = pd.DataFrame({"W1": w1, "W2": w2, "A": a, "time": time, "event": event})
    if trial:
        frame["p1"], frame["p0"] = 0.5, 0.5
        for g in range(1, K):
            frame[f"r{g}"] = 1.0 - leave
    return frame


def _survival(a: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """``(..., K + 1)`` :math:`S_a(t, w)` for ``t = 0, ..., K``."""
    out = [np.ones(np.broadcast_shapes(np.shape(w1), np.shape(w2)))]
    for t in range(1, K + 1):
        out.append(out[-1] * (1.0 - hazard(t, a, w1, w2)))
    return np.stack(out, axis=-1)


def truth(nodes: int = 200) -> dict[str, float]:
    """:math:`F_1(5)`, :math:`F_0(5)` and their difference, through :math:`s = -W_1 + 2 W_2`."""
    x, weight = hermegauss(nodes)
    weight = weight / weight.sum()
    s = np.sqrt(5.0) * x
    # h depends on w through -w1 + 2 w2 = s, so (w1, w2) = (-s, 0) carries it.
    risk = {a: 1.0 - float(weight @ _survival(a, -s, np.zeros_like(s))[:, HORIZON]) for a in (0, 1)}
    return {ESTIMANDS[0]: risk[1], ESTIMANDS[1]: risk[0], ESTIMANDS[2]: risk[1] - risk[0]}


TRUTH = truth()


def efficiency_sd(*, trial: bool, nodes: int = 80) -> dict[str, float]:
    """:math:`\\sqrt{E D^2}` of each estimand on the trial or observational law."""
    x, weight = hermegauss(nodes)
    weight = weight / weight.sum()
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    mass = np.outer(weight, weight)
    g1 = treatment(w1, w2, trial=trial)
    leave = censoring(w1, w2)
    martingale, level = {}, {}
    for a in (0, 1):
        survival = _survival(a, w1, w2)
        g = g1 if a == 1 else 1.0 - g1
        total = np.zeros_like(w1)
        for t in range(1, HORIZON + 1):
            h = hazard(t, a, w1, w2)
            retained = (1.0 - leave) ** (t - 1)
            # S(5)/S(t) as the product of the later factors, which never divides by zero.
            ratio = np.prod([1.0 - hazard(m, a, w1, w2) for m in range(t + 1, HORIZON + 1)], axis=0)
            total += survival[..., t - 1] * ratio**2 * h * (1.0 - h) / (g * retained)
        martingale[a] = float(np.sum(mass * total))
        level[a] = survival[..., HORIZON]

    def variance(values: np.ndarray) -> float:
        mean = float(np.sum(mass * values))
        return float(np.sum(mass * (values - mean) ** 2))

    return {
        ESTIMANDS[0]: float(np.sqrt(variance(level[1]) + martingale[1])),
        ESTIMANDS[1]: float(np.sqrt(variance(level[0]) + martingale[0])),
        ESTIMANDS[2]: float(np.sqrt(variance(level[1] - level[0]) + martingale[1] + martingale[0])),
    }
