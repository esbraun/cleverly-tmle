r"""The laws of the ``continuous-survival-variance`` study (X6 part (a)), with their truths.

Each law is the registered study's base law (:mod:`tests.studies.continuous_survival_law`) with
one declared change (X6 plan rev 5, section 5.1):

* ``rare_cause2`` (cell V1): the cause-2 log hazard's intercept is :math:`-2.6` in place of
  :math:`-2`, a rarer cause, with target times re-read as this law's event-time quartiles;
* ``heavy_weights`` (V2): the treatment model's coefficients are :math:`\omega = (2, -1, 1.5)`,
  which leaves the risks unchanged and makes the control arm's weights heavier;
* ``base`` (V3 at :math:`n = 500` and V4 in sample);
* ``quadratic`` (V7): the cause-1 log hazard gains :math:`0.3\,(X_3^2 - 1)`, which no default Cox
  member holds (each is linear in :math:`X_3`), with the propensity and the censoring unchanged.

Every hazard stays constant given :math:`(Z, X)`, so :math:`F_j(t \mid z, x) = h_j / h
\cdot (1 - e^{-h t})` and the truth is the base law's quadrature over :math:`X_3` with the changed
rates.  The efficient standard deviation (the oracle standard error times :math:`\sqrt n`) is the
base law's closed form with the changed rates and propensity
(:func:`efficiency_sd`, :func:`tests.studies.continuous_survival_law.efficiency_sd`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.special import expit

from tests.studies import continuous_survival_law as base

#: The V7 coefficient of :math:`X_3^2 - 1` in the cause-1 log hazard, fixed before the pilot.
#: The plan's pilot rule halves it once, to 0.15, if the influence-curve interval's coverage of
#: a cause-1 target at the median or upper quartile is predicted below 0.93 (5.1).
QUADRATIC = 0.3
#: The V1 cause-2 intercept.
RARE_INTERCEPT = -2.6
#: The V2 treatment-model coefficients on :math:`(X_1, X_2, X_3)`.
HEAVY_OMEGA = (2.0, -1.0, 1.5)


@dataclass(frozen=True)
class Law:
    """One of the study's laws.

    Parameters
    ----------
    name : str
        The law's name.
    cause2_intercept : float
        The intercept of the cause-2 log hazard (the base law's is -2).
    omega : tuple of float
        The treatment model's coefficients on :math:`(X_1, X_2, X_3)`.
    quadratic : float
        The coefficient of :math:`X_3^2 - 1` in the cause-1 log hazard.
    times : tuple of float
        The target times: the law's event-time quartiles (the base law's for the laws that
        leave the event times' scale alone).
    """

    name: str
    cause2_intercept: float = -2.0
    omega: tuple[float, float, float] = base.OMEGA
    quadratic: float = 0.0
    times: tuple[float, float, float] = base.TIMES

    def rates(
        self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, z: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The cause hazards :math:`(h_1, h_2)` given the arm and the covariates."""
        h1, h2 = base.rates(x1, x2, x3, z)
        x3 = np.asarray(x3, dtype=float)
        h1 = h1 * np.exp(self.quadratic * (x3**2 - 1.0))
        h2 = h2 * np.exp(self.cause2_intercept + 2.0)
        return np.asarray(h1, dtype=float), np.asarray(h2, dtype=float)

    def sample(self, n: int, seed: int | np.random.Generator) -> pd.DataFrame:
        """One draw as a frame with ``time``, ``event``, ``Z`` and ``X1`` to ``X4``."""
        rng = np.random.default_rng(seed)
        factor = rng.integers(0, 3, n)
        x1, x2 = (factor == 1).astype(float), (factor == 2).astype(float)
        x3 = rng.normal(size=n)
        w = self.omega
        z = rng.binomial(1, expit(w[0] * x1 + w[1] * x2 + w[2] * x3)).astype(float)
        h1, h2 = self.rates(x1, x2, x3, z)
        event_time = rng.exponential(1.0 / (h1 + h2))
        cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
        censor_time = rng.exponential(1.0 / base.CENSORING_RATE, size=n)
        return pd.DataFrame(
            {
                "time": np.minimum(event_time, censor_time),
                "event": np.where(event_time <= censor_time, cause, 0),
                "Z": z,
                "X1": x1,
                "X2": x2,
                "X3": x3,
                "X4": (x3 < 1.0).astype(float),
            }
        )

    def risk(self, z: int, cause: int, t: float) -> float:
        """:math:`E\\,F_j(t \\mid z, X)`, by quadrature over :math:`X_3` split at 1."""
        total = 0.0
        for x1, x2 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):

            def integrand(x3: float, x1: float = x1, x2: float = x2) -> float:
                h1, h2 = self.rates(np.array(x1), np.array(x2), np.array(x3), z)
                h = h1 + h2
                hj = h1 if cause == 1 else h2
                return float(hj / h * (1.0 - np.exp(-h * t)) * stats.norm.pdf(x3))

            for low, high in ((-12.0, 1.0), (1.0, 12.0)):
                value, _ = integrate.quad(integrand, low, high, epsabs=1e-13, epsrel=1e-12)
                total += value / 3.0
        return total

    def truth(self) -> dict[str, float]:
        """Every arm's risk of each cause at each time, and the risk differences."""
        out: dict[str, float] = {}
        for arm, z in base.ARMS.items():
            for code, cause in base.CAUSES.items():
                for t in self.times:
                    out[base.level_name(arm, cause, t)] = self.risk(z, code, t)
        for cause in base.CAUSES.values():
            for t in self.times:
                out[base.difference_name(cause, t)] = (
                    out[base.level_name("treated", cause, t)]
                    - out[base.level_name("control", cause, t)]
                )
        return out

    def efficiency_sd(self, cause: int, t: float, nodes: int = 200) -> dict[str, float]:
        r""":math:`\sqrt{E D^2}` of each arm's risk of ``cause`` at ``t`` and of its difference.

        The closed form of :func:`tests.studies.continuous_survival_law.efficiency_sd` with this
        law's rates and propensity: censoring is the base law's constant rate.
        """
        x3, w3 = base._x3_rule(nodes)
        u, wu = np.polynomial.legendre.leggauss(nodes)
        u, wu = t / 2.0 * (u + 1.0), t / 2.0 * wu
        weights = np.concatenate([w3 / 3.0] * 3)
        martingale: dict[int, float] = {}
        level: dict[int, np.ndarray] = {}
        for z in (0, 1):
            totals, levels = [], []
            for x1, x2 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
                h1, h2 = self.rates(np.full_like(x3, x1), np.full_like(x3, x2), x3, z)
                h = h1 + h2
                hj = h1 if cause == 1 else h2
                p1 = expit(self.omega[0] * x1 + self.omega[1] * x2 + self.omega[2] * x3)
                pi = p1 if z == 1 else 1.0 - p1
                g = (hj / h)[:, None] * (1.0 - np.exp(-h[:, None] * (t - u[None, :])))
                terms = hj[:, None] * (1.0 - g) ** 2 + (h - hj)[:, None] * g**2
                ratio = np.exp(-(h[:, None] - base.CENSORING_RATE) * u[None, :])
                totals.append((terms * ratio) @ wu / pi)
                levels.append(hj / h * (1.0 - np.exp(-h * t)))
            martingale[z] = float(weights @ np.concatenate(totals))
            level[z] = np.concatenate(levels)

        def variance(values: np.ndarray) -> float:
            mean = float(weights @ values)
            return float(weights @ (values - mean) ** 2)

        return {
            "treated": float(np.sqrt(variance(level[1]) + martingale[1])),
            "control": float(np.sqrt(variance(level[0]) + martingale[0])),
            "difference": float(
                np.sqrt(variance(level[1] - level[0]) + martingale[1] + martingale[0])
            ),
        }

    def event_time_quartiles(self, n: int = 2_000_000, seed: int = 2026911951) -> tuple[float, ...]:
        """The quartiles of the observed event times, from one large draw, to two decimals."""
        frame = self.sample(n, seed)
        observed = frame.loc[frame["event"] > 0, "time"].to_numpy()
        return tuple(float(v) for v in np.round(np.quantile(observed, [0.25, 0.5, 0.75]), 2))


#: V1's target times: :meth:`Law.event_time_quartiles` of the rare-cause law (the plan's 1.30 for
#: the median read 1.29 on the fixed-seed draw of 2,000,000), checked by the
#: design test.
RARE_TIMES = (0.52, 1.29, 2.69)

LAWS: Mapping[str, Law] = {
    "base": Law("base"),
    "rare_cause2": Law("rare_cause2", cause2_intercept=RARE_INTERCEPT, times=RARE_TIMES),
    "heavy_weights": Law("heavy_weights", omega=HEAVY_OMEGA),
    "quadratic": Law("quadratic", quadratic=QUADRATIC),
}


@cache
def truth(name: str) -> dict[str, float]:
    """The truths of law ``name``, keyed by the engine's parameter names."""
    return LAWS[name].truth()


@cache
def efficiency(name: str) -> dict[str, float]:
    """The efficient standard deviation of every target of law ``name``, keyed by name."""
    law = LAWS[name]
    out: dict[str, float] = {}
    for code, cause in base.CAUSES.items():
        for t in law.times:
            sd = law.efficiency_sd(code, t)
            out[base.level_name("treated", cause, t)] = sd["treated"]
            out[base.level_name("control", cause, t)] = sd["control"]
            out[base.difference_name(cause, t)] = sd["difference"]
    return out


def first_quartile(name: str) -> tuple[str, ...]:
    """The six targets at law ``name``'s first target time, read but not gated."""
    early = LAWS[name].times[0]
    return tuple(key for key in truth(name) if f"@ t={early:g}]" in key)


def gated(name: str) -> tuple[str, ...]:
    """The twelve targets at the median and upper quartile, gated by the primary rule."""
    early = set(first_quartile(name))
    return tuple(key for key in truth(name) if key not in early)
