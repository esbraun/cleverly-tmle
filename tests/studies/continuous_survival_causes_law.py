r"""The laws of the ``continuous-survival-causes`` study (X6 part (b)), with their truths.

Each law is the registered study's base law (:mod:`tests.studies.continuous_survival_law`) with
the number of causes changed (X6 plan rev 5, section 5.3):

* ``one_cause`` (L1): the base law's cause-1 hazard alone, :math:`h_2 \equiv 0`, so every event is
  of one type;
* ``three_causes`` (L3): the base law's :math:`h_1, h_2` and a third cause with
  :math:`h_3 = \exp(-2.5 + 0.5 X_1 + 0.5 X_3 + 0.5 Z)`.

The covariates, the propensity (:math:`\omega = (1, -1, 1)`) and the censoring (exponential at
rate 0.15, independent of :math:`(Z, X)`) are the base law's.  Every hazard is constant given
:math:`(Z, X)`, so with :math:`h = \sum_l h_l`

.. math::

    F_j(t \mid z, x) = \frac{h_j}{h}\, (1 - e^{-h t}),

and :math:`E\,F_j(t \mid z, X)` is the base law's quadrature over :math:`X_3`, split at
:math:`X_3 = 1` where :math:`X_4` jumps (:meth:`Law.risk`).  The efficient standard deviation is
the base law's closed form with :math:`h` the sum over this law's causes
(:meth:`Law.efficiency_sd`); it scales the calibration cell's noise control only.  The target
times are each law's event-time quartiles, read once from a draw of 2,000,000 with the base law's
fixed seed (:meth:`Law.event_time_quartiles`) and rounded to two decimals.
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

#: The third cause's log-hazard intercept and its coefficients on :math:`(X_1, X_3, Z)`.
CAUSE3 = (-2.5, 0.5, 0.5, 0.5)
ARMS = base.ARMS
REFERENCE = base.REFERENCE
BASELINE = base.BASELINE


@dataclass(frozen=True)
class Law:
    """One of the study's laws.

    Parameters
    ----------
    name : str
        The law's name.
    causes : int
        The number of causes, 1 or 3.
    times : tuple of float
        The target times: the law's event-time quartiles.
    """

    name: str
    causes: int
    times: tuple[float, float, float]

    @property
    def labels(self) -> dict[int, str]:
        """The engine's ``causes=``: code to label."""
        return {code: f"cause{code}" for code in range(1, self.causes + 1)}

    def rates(
        self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, z: np.ndarray | float
    ) -> tuple[np.ndarray, ...]:
        """The cause hazards :math:`(h_1, \\dots, h_J)` given the arm and the covariates."""
        h1, h2 = base.rates(x1, x2, x3, z)
        if self.causes == 1:
            return (h1,)
        intercept, b1, b3, bz = CAUSE3
        h3 = np.exp(intercept + b1 * np.asarray(x1) + b3 * np.asarray(x3) + bz * np.asarray(z))
        return h1, h2, np.asarray(h3, dtype=float)

    def sample(self, n: int, seed: int | np.random.Generator) -> pd.DataFrame:
        """One draw as a frame with ``time``, ``event``, ``Z`` and ``X1`` to ``X4``."""
        rng = np.random.default_rng(seed)
        factor = rng.integers(0, 3, n)
        x1, x2 = (factor == 1).astype(float), (factor == 2).astype(float)
        x3 = rng.normal(size=n)
        w = base.OMEGA
        z = rng.binomial(1, expit(w[0] * x1 + w[1] * x2 + w[2] * x3)).astype(float)
        hazards = np.vstack(self.rates(x1, x2, x3, z))
        total = hazards.sum(axis=0)
        event_time = rng.exponential(1.0 / total)
        # The cause by inversion of its conditional probabilities h_l / h.
        cumulative = np.cumsum(hazards / total, axis=0)
        cause = 1 + (rng.uniform(size=n)[None, :] > cumulative[:-1]).sum(axis=0)
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
                hazards = self.rates(np.array(x1), np.array(x2), np.array(x3), z)
                h = sum(hazards)
                return float(hazards[cause - 1] / h * (1.0 - np.exp(-h * t)) * stats.norm.pdf(x3))

            for low, high in ((-12.0, 1.0), (1.0, 12.0)):
                value, _ = integrate.quad(integrand, low, high, epsabs=1e-13, epsrel=1e-12)
                total += value / 3.0
        return total

    def level_name(self, arm: str, code: int, t: float) -> str:
        """The engine's name of an arm's risk of one cause at one time."""
        if self.causes == 1:
            return f"risk_regimen[{arm} @ t={t:g}]"
        return base.level_name(arm, self.labels[code], t)

    def difference_name(self, code: int, t: float) -> str:
        """The engine's name of one cause's risk difference at one time."""
        if self.causes == 1:
            return f"ate_regimen[treated vs control @ t={t:g}]"
        return base.difference_name(self.labels[code], t)

    def truth(self) -> dict[str, float]:
        """Every arm's risk of each cause at each time, and the risk differences."""
        out: dict[str, float] = {}
        for arm, z in ARMS.items():
            for code in self.labels:
                for t in self.times:
                    out[self.level_name(arm, code, t)] = self.risk(z, code, t)
        for code in self.labels:
            for t in self.times:
                out[self.difference_name(code, t)] = (
                    out[self.level_name("treated", code, t)]
                    - out[self.level_name("control", code, t)]
                )
        return out

    def efficiency_sd(self, cause: int, t: float, nodes: int = 200) -> dict[str, float]:
        r""":math:`\sqrt{E D^2}` of each arm's risk of ``cause`` at ``t`` and of its difference.

        The closed form of :func:`tests.studies.continuous_survival_law.efficiency_sd` with
        :math:`h` the sum over this law's causes: the martingale term is
        :math:`\int \{h_j (1 - g)^2 + (h - h_j) g^2\} S / S_c\, du / \pi`, which holds for any
        number of causes, since every cause other than :math:`j` carries the same clever
        covariate :math:`-g`.  With one cause :math:`h - h_j = 0`.
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
                hazards = self.rates(np.full_like(x3, x1), np.full_like(x3, x2), x3, z)
                h = sum(hazards)
                hj = hazards[cause - 1]
                p1 = expit(base.OMEGA[0] * x1 + base.OMEGA[1] * x2 + base.OMEGA[2] * x3)
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


#: L1's and L3's target times: :meth:`Law.event_time_quartiles` of each law on the fixed-seed draw
#: of 2,000,000 (checked by the design test on a smaller draw).
ONE_CAUSE_TIMES = (0.58, 1.48, 3.24)
THREE_CAUSE_TIMES = (0.37, 0.9, 1.86)

LAWS: Mapping[str, Law] = {
    "one_cause": Law("one_cause", 1, ONE_CAUSE_TIMES),
    "three_causes": Law("three_causes", 3, THREE_CAUSE_TIMES),
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
    for code in law.labels:
        for t in law.times:
            sd = law.efficiency_sd(code, t)
            out[law.level_name("treated", code, t)] = sd["treated"]
            out[law.level_name("control", code, t)] = sd["control"]
            out[law.difference_name(code, t)] = sd["difference"]
    return out


def first_quartile(name: str) -> tuple[str, ...]:
    """The targets at law ``name``'s first target time, read but not gated."""
    early = LAWS[name].times[0]
    return tuple(key for key in truth(name) if f"@ t={early:g}]" in key)


def gated(name: str) -> tuple[str, ...]:
    """The targets at the median and upper quartile, gated by the primary rule."""
    early = set(first_quartile(name))
    return tuple(key for key in truth(name) if key not in early)
