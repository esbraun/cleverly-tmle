r"""The laws, truths, efficiency bounds and learners of ``hazard-mtp``.

**The laws.**  :math:`W_1 \in \{0, 1\}` and :math:`W_2 \in \{0, \dots, 3\}` are independent and
uniform.  Each law has one cause, and at visit :math:`k` the event occurs among the units at risk
with probability

.. math::

    \lambda(k \mid f, a, W) = \operatorname{expit}\bigl(\alpha + \tau (k - 1) + \beta a
        + \gamma_1 W_1 + \gamma_2 W_2 + \phi f\bigr),

so a logistic regression on ``(W1, W2, A)`` at each visit (and ``F`` on the vector law) is
correctly specified and not saturated.  A unit entering visit :math:`k \ge 2` drops out before
the visit's event with probability
:math:`\operatorname{expit}(d_0 + d_a a + d_1 W_1 + d_2 W_2 + d_f f)`; on the one-visit vector
law the drop-out acts at visit 1.  Every mechanism is logistic (or, for the dose, a Gaussian
location model), so a parametric learner is correct without being saturated.

=================  ==========================================================================
law                treatment
=================  ==========================================================================
``levels``         a dose on :math:`\{0, \dots, 5\}` with
                   :math:`P(A = a \mid W) \propto \exp\{-(a - c(W))^2 / (2 s^2)\}`, a
                   multinomial logit linear in ``W``; five visits
``gaussian``       :math:`A \mid W \sim N(c(W), s^2)`; five visits
``vector``         a binary arm :math:`F` with :math:`P(F = 1 \mid W)` logistic and
                   :math:`A \mid F, W \sim N(c(W) + c_F F, s^2)`; one visit, because a vector
                   held over several visits cannot be declared in the data container
=================  ==========================================================================

with :math:`c(W) = c_0 + c_1 W_1 + c_2 W_2`.

**The policies.**  On ``levels`` the policy is "one level up, the top stays",
:math:`d(a) = \min(a + 1, 5)`.  On ``gaussian`` it is the additive shift :math:`d(a) = a + \delta`,
and on ``vector`` the joint policy that shifts the dose by :math:`\delta` and keeps the arm.
Each law's reference is the natural course, the identity policy.

**Truths.**  :math:`\psi_d(t) = E\,F(t \mid d(F, A), W)`, an exact sum over the dose levels or a
Gauss-Hermite quadrature of :data:`QUADRATURE_NODES` nodes over the Gaussian dose, which reaches
``1e-10`` because the integrand is analytic in the dose (the design test checks it against an
adaptive quadrature of its own recursion).

**Efficiency bounds.**  The efficient influence function of :math:`\psi_d(t)` at one decision is
(Díaz, Williams, Hoffman and Schenck 2023, Theorem 1 at one decision)

.. math::

    D = \sum_{k \le t} \frac{r(F, A, W)}{G_k(F, A, W)}\, R_k\, (Z_k - Q_k(F, A, W))
        + Q_1(d(F, A), W) - \psi,

with :math:`r = g^d / g`, :math:`G_k` the retention through visit :math:`k`, and :math:`Q_k` the
law's own backward regression of the event by :math:`t`.  :func:`efficiency_sd` sums its square
over the support, with the same quadrature over a Gaussian dose.  Nothing here calls the
estimator.

**The wrong hazard learner.**  :class:`NoWLogistic` regresses each visit's event on the dose
alone (and the arm on ``vector``): no ``W``.  ``W`` confounds the dose, so its dose slope is
biased.  :func:`wrong_hazard_limit` computes its limit by Newton's method on the law laid out as
weighted data and returns the plug-in of each target.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.longitudinal import LongitudinalData

#: The dose levels of the ``levels`` law.
LEVELS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
#: The baseline strata and their mass.
STRATA = tuple((w1, w2) for w1 in (0, 1) for w2 in (0, 1, 2, 3))
MASS = 0.125
#: The Gauss-Hermite nodes over a Gaussian dose.
QUADRATURE_NODES = 120
_NODES, _WEIGHTS = hermegauss(QUADRATURE_NODES)
_WEIGHTS = _WEIGHTS / math.sqrt(2.0 * math.pi)
#: The Gauss-Legendre nodes on each smooth piece of a truncated dose.
PIECE_NODES = 60
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = leggauss(PIECE_NODES)


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class Hazard:
    """The one-cause hazard on the logit scale.

    Parameters
    ----------
    intercept : float
        :math:`\\alpha`, at visit 1.
    trend : float
        :math:`\\tau`, per visit.
    dose : float
        :math:`\\beta`, per unit of dose.
    w1 : float
        :math:`\\gamma_1`.
    w2 : float
        :math:`\\gamma_2`.
    flag : float
        :math:`\\phi`, the arm's coefficient on the vector law.
    """

    intercept: float
    trend: float
    dose: float
    w1: float
    w2: float
    flag: float = 0.0

    def __call__(self, node: int, f: Any, a: Any, w1: Any, w2: Any) -> Any:
        """:math:`\\lambda(k \\mid f, a, W)`."""
        return _expit(
            self.intercept
            + self.trend * (node - 1)
            + self.dose * np.asarray(a, dtype=float)
            + self.w1 * np.asarray(w1, dtype=float)
            + self.w2 * np.asarray(w2, dtype=float)
            + self.flag * np.asarray(f, dtype=float)
        )


@dataclass(frozen=True)
class Point:
    """One treatment value given ``W``, with its conditional mass.

    Parameters
    ----------
    f : int
        The arm (zero on a law with no arm).
    a : float
        The dose.
    mass : float
        :math:`P(F = f, A = a \\mid W)`, or the quadrature weight times the arm's probability.
    """

    f: int
    a: float
    mass: float


@dataclass(frozen=True)
class Law:
    """A held baseline treatment, a one-cause survival outcome and a policy.

    Parameters
    ----------
    name : str
        The law's label.
    kind : {"levels", "gaussian", "vector"}
        How the treatment is drawn.
    n_times : int
        The visit count.
    hazard : Hazard
        The event hazard.
    centre : tuple of float
        :math:`(c_0, c_1, c_2)` of the dose's location on ``(1, W1, W2)``.
    scale : float
        :math:`s`, the dose's scale.
    dropout : tuple of float
        :math:`(d_0, d_a, d_1, d_2, d_f)` of the drop-out probability on the logit scale.
    shift : float
        :math:`\\delta`, the additive shift of a Gaussian dose.
    flag_shift : float
        :math:`c_F`, the arm's shift of the dose location on ``vector``.
    arm : tuple of float
        :math:`(a_0, a_1, a_2)` of :math:`P(F = 1 \\mid W)` on ``(1, W1, W2)``, logit scale.
    bounds : tuple of float or None
        :math:`(l, u)`, the fixed interval a Gaussian dose is truncated to, or ``None``.
    cap : float or None
        The shift's cap: a unit whose shifted dose would pass it keeps its own dose.
    """

    name: str
    kind: str
    n_times: int
    hazard: Hazard
    centre: tuple[float, float, float]
    scale: float
    dropout: tuple[float, float, float, float, float]
    shift: float = 0.0
    flag_shift: float = 0.0
    arm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds: tuple[float, float] | None = None
    cap: float | None = None
    _cache: dict[Any, Any] = field(default_factory=dict, compare=False, repr=False, hash=False)

    @property
    def grid(self) -> tuple[float, ...]:
        return tuple(float(k) for k in range(1, self.n_times + 1))

    @property
    def continuous(self) -> bool:
        """Whether the dose is continuous."""
        return self.kind != "levels"

    # ------------------------------------------------------------------ factors

    def location(self, f: Any, w1: Any, w2: Any) -> Any:
        """The dose's location :math:`c(W) + c_F f`."""
        c0, c1, c2 = self.centre
        return (
            c0
            + c1 * np.asarray(w1, dtype=float)
            + c2 * np.asarray(w2, dtype=float)
            + self.flag_shift * np.asarray(f, dtype=float)
        )

    def level_probabilities(self, w1: Any, w2: Any) -> np.ndarray:
        """``(n, 6)``: :math:`P(A = a \\mid W)` of the ``levels`` law."""
        centre = np.atleast_1d(np.asarray(self.location(0.0, w1, w2), dtype=float))
        levels = np.asarray(LEVELS, dtype=float)
        weight = np.exp(-((levels[None, :] - centre[:, None]) ** 2) / (2.0 * self.scale**2))
        return np.asarray(weight / weight.sum(axis=1, keepdims=True))

    def flagged(self, w1: Any, w2: Any) -> Any:
        """:math:`P(F = 1 \\mid W)` on ``vector``; zero elsewhere."""
        if self.kind != "vector":
            return np.zeros_like(np.asarray(w1, dtype=float))
        a0, a1, a2 = self.arm
        return _expit(a0 + a1 * np.asarray(w1, dtype=float) + a2 * np.asarray(w2, dtype=float))

    def density(self, a: Any, f: Any, w1: Any, w2: Any) -> Any:
        """The Gaussian dose density :math:`g(a \\mid f, W)`, truncated to :attr:`bounds`."""
        a = np.asarray(a, dtype=float)
        location = self.location(f, w1, w2)
        z = (a - location) / self.scale
        if self.bounds is None:
            return norm.pdf(z) / self.scale
        low, high = self.bounds
        mass = norm.cdf((high - location) / self.scale) - norm.cdf((low - location) / self.scale)
        inside = (a >= low) & (a <= high)
        return np.where(inside, norm.pdf(z) / (self.scale * mass), 0.0)

    def drop(self, f: Any, a: Any, w1: Any, w2: Any) -> Any:
        """The drop-out probability at a visit where drop-out acts."""
        d0, da, d1, d2, df = self.dropout
        return _expit(
            d0
            + da * np.asarray(a, dtype=float)
            + d1 * np.asarray(w1, dtype=float)
            + d2 * np.asarray(w2, dtype=float)
            + df * np.asarray(f, dtype=float)
        )

    def drops_at(self, node: int) -> bool:
        """Whether drop-out acts at ``node``: from visit 2, or at visit 1 on ``vector``."""
        return node >= 2 or self.kind == "vector"

    def retained(self, node: int, f: Any, a: Any, w1: Any, w2: Any) -> Any:
        """The probability of staying in follow-up at ``node``."""
        if not self.drops_at(node):
            return np.ones_like(np.asarray(a, dtype=float)) * 1.0
        return 1.0 - self.drop(f, a, w1, w2)

    # ------------------------------------------------------------------ the policy

    def policy(self, f: Any, a: Any) -> tuple[Any, Any]:
        """The policy's treatment :math:`d(f, a)`."""
        if self.kind == "levels":
            return f, np.minimum(np.asarray(a, dtype=float) + 1.0, float(LEVELS[-1]))
        moved = np.asarray(a, dtype=float) + self.shift
        if self.cap is None:
            return f, moved
        return f, np.where(moved > self.cap, np.asarray(a, dtype=float), moved)

    def ratio(self, f: int, a: float, w1: int, w2: int) -> float:
        """:math:`r = g^d(f, a \\mid W) / g(f, a \\mid W)` of the policy."""
        if self.kind == "levels":
            g = self.level_probabilities(w1, w2)[0]
            index = round(a)
            if index == 0:
                return 0.0
            mapped = g[index - 1] + (g[index] if index == LEVELS[-1] else 0.0)
            return float(mapped / g[index])
        moved = (
            0.0
            if self.cap is not None and a > self.cap
            else self.density(a - self.shift, f, w1, w2)
        )
        held = 1.0 if self.cap is not None and a > self.cap - self.shift else 0.0
        return float(moved / self.density(a, f, w1, w2)) + held

    def pieces(self, f: int, w1: int, w2: int) -> tuple[tuple[float, float], ...]:
        """The smooth pieces of a truncated dose: the ratio and the policy are smooth inside."""
        assert self.bounds is not None
        low, high = self.bounds
        cuts = {low, high, low + self.shift}
        if self.cap is not None:
            cuts |= {self.cap, self.cap - self.shift}
        edges = sorted(c for c in cuts if low <= c <= high)
        del f, w1, w2
        return tuple((a, b) for a, b in itertools.pairwise(edges) if b > a)

    def points(self, w1: int, w2: int) -> tuple[Point, ...]:
        """Every treatment value given ``W``, an exact list or a quadrature over the dose."""
        key = ("points", w1, w2)
        if key in self._cache:
            return self._cache[key]
        out: list[Point] = []
        if self.kind == "levels":
            g = self.level_probabilities(w1, w2)[0]
            out = [Point(0, float(a), float(g[i])) for i, a in enumerate(LEVELS)]
        else:
            arms: tuple[tuple[int, float], ...] = ((0, 1.0),)
            if self.kind == "vector":
                p = float(self.flagged(w1, w2))
                arms = ((0, 1.0 - p), (1, p))
            for f, share in arms:
                if self.bounds is not None:
                    for low, high in self.pieces(f, w1, w2):
                        half = (high - low) / 2.0
                        doses = low + half * (_LEGENDRE_NODES + 1.0)
                        mass = self.density(doses, f, w1, w2) * half * _LEGENDRE_WEIGHTS
                        out += [
                            Point(f, float(a), share * float(m))
                            for a, m in zip(doses, mass, strict=True)
                        ]
                    continue
                centre = float(self.location(f, w1, w2))
                out += [
                    Point(f, centre + self.scale * float(z), share * float(weight))
                    for z, weight in zip(_NODES, _WEIGHTS, strict=True)
                ]
        self._cache[key] = tuple(out)
        return tuple(out)

    # ------------------------------------------------------------------ the curve

    def curve(self, f: Any, a: Any, w1: Any, w2: Any) -> np.ndarray:
        """``(T, ...)``: :math:`F(t \\mid f, a, W)` at each visit."""
        survival = np.ones_like(np.asarray(a, dtype=float))
        running = np.zeros_like(survival)
        out = []
        for node in range(1, self.n_times + 1):
            lam = self.hazard(node, f, a, w1, w2)
            running = running + survival * lam
            survival = survival * (1.0 - lam)
            out.append(running)
        return np.array(out)

    def regression(self, f: Any, a: Any, w1: Any, w2: Any, horizon: int) -> np.ndarray:
        """``Q_k``, ``k = 1..horizon + 1``: P(event by ``horizon`` | at risk at ``k``, f, a, W)."""
        out = np.zeros(horizon + 2)
        for node in range(horizon, 0, -1):
            lam = float(self.hazard(node, f, a, w1, w2))
            out[node] = lam + (1.0 - lam) * out[node + 1]
        return out

    def truth(self, horizon: int, *, natural: bool = False) -> float:
        """:math:`\\psi(t)` of the policy, or of the natural course."""
        total = 0.0
        for w1, w2 in STRATA:
            for point in self.points(w1, w2):
                f, a = (point.f, point.a) if natural else self.policy(point.f, point.a)
                total += MASS * point.mass * float(self.curve(f, a, w1, w2)[horizon - 1])
        return total

    # ------------------------------------------------------------------ sampling

    def draw_treatment(self, rng: np.random.Generator, w1: np.ndarray, w2: np.ndarray) -> Any:
        """``(F, A)`` of ``n`` units."""
        n = w1.size
        if self.kind == "levels":
            g = self.level_probabilities(w1, w2)
            cumulative = np.cumsum(g, axis=1)
            u = rng.random(n)[:, None]
            a = np.minimum((u > cumulative).sum(axis=1), len(LEVELS) - 1).astype(float)
            return np.zeros(n), a
        f = (rng.random(n) < self.flagged(w1, w2)).astype(float)
        if self.bounds is not None:
            location = self.location(f, w1, w2)
            low, high = (norm.cdf((b - location) / self.scale) for b in self.bounds)
            u = low + (high - low) * rng.random(n)
            return f, np.asarray(location + self.scale * norm.ppf(u), dtype=float)
        a = self.location(f, w1, w2) + self.scale * rng.standard_normal(n)
        return f, np.asarray(a, dtype=float)

    def draw(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` units.

        On the five-visit laws the frame is ``W1, W2, A, time, event``: an event at visit ``k``
        is ``time = k`` with ``event = 1``, a drop-out at visit ``k >= 2`` is ``time = k - 1``
        with ``event = 0``, and a unit followed through the last visit is ``time = T`` with
        ``event = 0``.  On ``vector`` it is ``W1, W2, F, A, C1, Y1`` (``Y1`` missing when
        ``C1 = 0``).
        """
        rng = np.random.default_rng(seed)
        w1 = rng.integers(0, 2, size=n).astype(float)
        w2 = rng.integers(0, 4, size=n).astype(float)
        f, a = self.draw_treatment(rng, w1, w2)
        if self.kind == "vector":
            kept = (rng.random(n) >= self.drop(f, a, w1, w2)).astype(float)
            event = (rng.random(n) < self.hazard(1, f, a, w1, w2)).astype(float)
            return pd.DataFrame(
                {
                    "W1": w1,
                    "W2": w2,
                    "F": f,
                    "A": a,
                    "C1": kept,
                    "Y1": np.where(kept == 1.0, event, np.nan),
                }
            )
        time = np.full(n, float(self.n_times))
        event = np.zeros(n, dtype=int)
        open_ = np.ones(n, dtype=bool)
        for node in range(1, self.n_times + 1):
            if self.drops_at(node):
                lost = open_ & (rng.random(n) < self.drop(f, a, w1, w2))
                time[lost] = float(node - 1)
                open_ &= ~lost
            fired = open_ & (rng.random(n) < self.hazard(node, f, a, w1, w2))
            time[fired] = float(node)
            event[fired] = 1
            open_ &= ~fired
        return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "time": time, "event": event})

    def container(self, frame: pd.DataFrame, *, weights: str | None = None) -> LongitudinalData:
        """The fit's container: the time-to-event converter, or the wide vector layout."""
        if self.kind == "vector":
            return LongitudinalData.from_frame(
                frame,
                outcome=["Y1"],
                treatment=[["F", "A"]],
                continuous_treatment=["A"],
                baseline=["W1", "W2"],
                censoring=["C1"],
                weights=weights,
            )
        return LongitudinalData.from_time_to_event(
            frame,
            time="time",
            event="event",
            treatment="A",
            baseline=["W1", "W2"],
            grid=self.grid,
            continuous_treatment=self.continuous,
            weights=weights,
        )

    # ------------------------------------------------------------------ outcomes

    def outcomes(self, f: int, a: float, w1: int, w2: int) -> list[tuple[float, float, int]]:
        """Every outcome after ``(W, F, A)``: ``(probability, time, event)``, as :meth:`draw` codes it.

        On ``vector`` a drop-out is ``time = 0``.
        """
        out: list[tuple[float, float, int]] = []
        alive = 1.0
        for node in range(1, self.n_times + 1):
            stay = float(self.retained(node, f, a, w1, w2))
            if stay < 1.0:
                out.append((alive * (1.0 - stay), float(node - 1), 0))
            alive *= stay
            lam = float(self.hazard(node, f, a, w1, w2))
            out.append((alive * lam, float(node), 1))
            alive *= 1.0 - lam
        out.append((alive, float(self.n_times), 0))
        return out

    def exact_frame(self) -> pd.DataFrame:
        """The ``levels`` law laid out as weighted time-to-event rows; ``wt`` is each mass."""
        if self.kind != "levels":
            raise ValueError("only the levels law has a finite support")
        rows: list[dict[str, float]] = []
        for w1, w2 in STRATA:
            for point in self.points(w1, w2):
                for mass, time, event in self.outcomes(0, point.a, w1, w2):
                    rows.append(
                        {
                            "W1": float(w1),
                            "W2": float(w2),
                            "A": point.a,
                            "time": time,
                            "event": event,
                            "wt": MASS * point.mass * mass,
                        }
                    )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ the bound

    def unit_curve(
        self,
        horizon: int,
        natural: bool,
        f: int,
        a: float,
        w1: int,
        w2: int,
        time: float,
        event: int,
    ) -> float:
        """The uncentred curve :math:`D + \\psi` of one target at one outcome.

        The iterated-mean form: :math:`\\sum_{k \\le t} \\frac{r}{G_k} R_k (Z_k - Q_k) +
        Q_1(d(f, a), W)`, with :math:`Z_k` the event indicator at an event and
        :math:`Q_{k + 1}` otherwise.
        """
        mapped_f, mapped_a = (f, a) if natural else self.policy(f, a)
        total = float(self.regression(mapped_f, mapped_a, w1, w2, horizon)[1])
        ratio = 1.0 if natural else self.ratio(f, a, w1, w2)
        if ratio == 0.0:
            return total
        q = self.regression(f, a, w1, w2, horizon)
        last = int(time) if event else int(time) + 1
        cumulative = 1.0
        for node in range(1, horizon + 1):
            if node > last:
                break
            cumulative *= float(self.retained(node, f, a, w1, w2))
            if not event and node == last:
                break  # dropped out at this visit
            z = (1.0 if event else 0.0) if (event and node == last) else float(q[node + 1])
            total += ratio / cumulative * (z - float(q[node]))
        return total

    def ratios(self, f: Any, a: Any, w1: Any, w2: Any) -> np.ndarray:
        """:meth:`ratio` at one treatment per row."""
        f, a = np.asarray(f, dtype=float), np.asarray(a, dtype=float)
        if self.kind == "levels":
            g = self.level_probabilities(w1, w2)
            index = np.rint(a).astype(int)
            rows = np.arange(index.size)
            below = g[rows, np.maximum(index - 1, 0)]
            top = np.where(index == LEVELS[-1], g[rows, index], 0.0)
            return np.asarray(np.where(index == 0, 0.0, (below + top) / g[rows, index]))
        cap = np.inf if self.cap is None else self.cap
        moved = np.where(a > cap, 0.0, self.density(a - self.shift, f, w1, w2))
        return np.asarray(moved / self.density(a, f, w1, w2) + (a > cap - self.shift))

    def unit_curves(
        self, horizon: int, natural: bool, f: Any, a: Any, w1: Any, w2: Any, time: Any, event: Any
    ) -> np.ndarray:
        """:meth:`unit_curve` at one outcome per row, vectorized over the rows."""
        f, a = np.asarray(f, dtype=float), np.asarray(a, dtype=float)
        w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
        time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=int)

        def regression(dose: np.ndarray) -> np.ndarray:
            out = np.zeros((horizon + 2, dose.size))
            for node in range(horizon, 0, -1):
                lam = self.hazard(node, f, dose, w1, w2)
                out[node] = lam + (1.0 - lam) * out[node + 1]
            return out

        mapped = a if natural else self.policy(f, a)[1]
        total = regression(np.asarray(mapped, dtype=float))[1]
        ratio = np.ones_like(a) if natural else self.ratios(f, a, w1, w2)
        q = regression(a)
        last = np.where(event == 1, time, time + 1.0).astype(int)
        cumulative = np.ones_like(a)
        for node in range(1, horizon + 1):
            active = node <= last
            cumulative = np.where(
                active, cumulative * self.retained(node, f, a, w1, w2), cumulative
            )
            dropped = (event == 0) & (node == last)
            z = np.where((event == 1) & (node == last), 1.0, q[node + 1])
            step = np.divide(ratio, cumulative) * (z - q[node])
            total = total + np.where(active & ~dropped & (ratio != 0.0), step, 0.0)
        return np.asarray(total)

    def frame_curves(self, target: Target, frame: pd.DataFrame) -> np.ndarray:
        """The centred efficient influence function of ``target`` at each row of a drawn frame."""
        if self.kind == "vector":
            kept = frame["C1"].to_numpy(dtype=float) == 1.0
            fired = kept & (frame["Y1"].fillna(0.0).to_numpy(dtype=float) == 1.0)
            f = frame["F"].to_numpy(dtype=float)
            time, event = np.where(kept, 1.0, 0.0), fired.astype(int)
        else:
            f = np.zeros(len(frame))
            time, event = frame["time"].to_numpy(dtype=float), frame["event"].to_numpy(dtype=int)
        columns = (frame["A"].to_numpy(dtype=float), frame["W1"], frame["W2"])
        curve = sum(
            c * self.unit_curves(t, natural, f, *columns, time, event)
            for c, t, natural in target.terms
        )
        return np.asarray(curve - target.value(self))


# ---------------------------------------------------------------------------- targets


@dataclass(frozen=True)
class Target:
    """A linear combination of risks: ``coefficient * psi(horizon)`` per term.

    Parameters
    ----------
    terms : tuple of tuple
        ``(coefficient, horizon, natural)`` triples.
    """

    terms: tuple[tuple[float, int, bool], ...]

    def value(self, law: Law) -> float:
        return sum(c * law.truth(t, natural=natural) for c, t, natural in self.terms)


def efficiency_sd(law: Law, target: Target) -> float:
    """The exact efficiency bound of ``target`` as a standard deviation.

    The function raises when the enumerated curve does not have mean zero.
    """
    psi = target.value(law)
    first = second = 0.0
    for w1, w2 in STRATA:
        for point in law.points(w1, w2):
            for mass, time, event in law.outcomes(point.f, point.a, w1, w2):
                curve = (
                    sum(
                        c * law.unit_curve(t, natural, point.f, point.a, w1, w2, time, event)
                        for c, t, natural in target.terms
                    )
                    - psi
                )
                probability = MASS * point.mass * mass
                first += probability * curve
                second += probability * curve**2
    if abs(first) > 1e-9:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the influence function has mean {first}, not zero")
    return math.sqrt(second)


# ---------------------------------------------------------------------------- learners


class _Logistic(BaseEstimator, ClassifierMixin):
    """An unpenalised logistic regression on the columns :meth:`_features` picks."""

    def _features(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Logistic:
        self.model_ = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
        self.model_.fit(self._features(X), np.asarray(y, dtype=float), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._features(X)))


def dose_column(X: Any, kind: str) -> np.ndarray:
    """The dose of a hazard design ``[W1, W2, (F,) dose block]``.

    The ``levels`` block is the five drop-first indicators of the six levels.
    """
    matrix = np.asarray(X, dtype=float)
    if kind == "levels":
        block = matrix[:, 2:]
        if block.shape[1] != len(LEVELS) - 1:
            raise ValueError(
                f"a levels design has {2 + len(LEVELS) - 1} columns, not {matrix.shape[1]}"
            )
        return np.where(block.any(axis=1), np.argmax(block, axis=1) + 1.0, 0.0)
    return matrix[:, -1]


class HazardLogistic(_Logistic):
    """The correct hazard learner: unpenalised logistic on ``(W1, W2, (F,) A)``.

    The dose enters linearly, read from the levels' indicators on ``levels``.

    Parameters
    ----------
    kind : str
        The law's kind.
    """

    def __init__(self, kind: str = "levels") -> None:
        self.kind = kind

    def _features(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        dose = dose_column(matrix, self.kind)
        columns = [matrix[:, 0], matrix[:, 1]]
        if self.kind == "vector":
            columns.append(matrix[:, 2])
        return np.column_stack([*columns, dose])


class NoWLogistic(HazardLogistic):
    """The wrong hazard learner: unpenalised logistic on the dose (and the arm), no ``W``."""

    def _features(self, X: Any) -> np.ndarray:
        return super()._features(X)[:, 2:]


class LevelsTreatment(_Logistic):
    """The correct ``levels`` treatment learner: multinomial logistic on ``(W1, W2)``.

    The law's log odds of level ``a`` against level ``0`` are linear in ``W``, so the model is
    correct, with fifteen coefficients over forty free cell probabilities.
    """

    def _features(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != 2:
            raise ValueError(f"a treatment design has 2 columns [W1, W2], not {matrix.shape[1]}")
        return matrix


class CensoringLogistic(_Logistic):
    """The correct censoring learner: unpenalised logistic on ``(W1, W2, (F,) A)``.

    Parameters
    ----------
    kind : str
        The law's kind.
    """

    def __init__(self, kind: str = "levels") -> None:
        self.kind = kind

    def _features(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        dose = dose_column(matrix, self.kind)
        columns = [matrix[:, 0], matrix[:, 1]]
        if self.kind == "vector":
            columns.append(matrix[:, 2])
        return np.column_stack([*columns, dose])


class RatioLogistic(_Logistic):
    """The correct classifier of the ratio route: logistic on ``(W1, W2, (F,) A)``.

    The log odds of a shifted copy against an observed one are
    :math:`\\log g(a - \\delta \\mid f, W) - \\log g(a \\mid f, W)`, linear in ``a`` and ``W``
    under a Gaussian location model with a constant scale.
    """


class DoseOnlyRatio(_Logistic):
    """The wrong classifier of the ratio route: logistic on the dose alone, no ``W``."""

    def _features(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, -1:]


class GaussianDoseHazard(BaseEstimator):
    """A Gaussian location density of the dose, fitted, read through the binned hazards.

    The density layer hands this the history design followed by the bin index
    (``bin_design = "index"``).  ``fit`` reads the units' doses and histories from the
    constructor's ``frame`` (the sample being fitted, so the fit uses the data and nothing
    else), regresses the dose on ``(W1, W2, (F))`` by least squares, and takes the residual
    scale; ``predict_proba`` returns the probability that the dose stops in bin ``b`` given it
    reached ``b`` under that normal law.  ``marginal=True`` is the wrong mechanism: the same
    family with no covariates.

    On ``vector`` the same learner is the mechanism of the arm, whose design is ``[W1, W2]``
    with no bin index: it is then the logistic regression on ``(W1, W2)`` (the prior share when
    ``marginal``).  The fit reads the whole ``frame``, so it serves only fits with one fold.

    Parameters
    ----------
    edges : tuple of float
        The density's bin edges.
    covariates : tuple of str
        The history columns, in design order.
    frame : DataFrame or None
        The sample's history and dose.
    marginal : bool
        Whether to ignore the history.
    bounds : tuple of float or None
        The interval the dose is truncated to.  The location model is then the truncated
        normal regression, fitted by maximum likelihood with the interval known.
    """

    bin_design = "index"

    def __init__(
        self,
        edges: tuple[float, ...] = (),
        covariates: tuple[str, ...] = ("W1", "W2"),
        frame: pd.DataFrame | None = None,
        marginal: bool = False,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        self.edges = edges
        self.covariates = covariates
        self.frame = frame
        self.marginal = marginal
        self.bounds = bounds

    def _is_arm(self, X: Any) -> bool:
        return "F" in self.covariates and np.asarray(X).shape[1] == len(self.covariates) - 1

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> GaussianDoseHazard:
        if self._is_arm(X):
            matrix = np.asarray(X, dtype=float)
            self.arm_ = (
                None
                if self.marginal
                else LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(
                    matrix, np.asarray(y, dtype=float), sample_weight=sample_weight
                )
            )
            share = np.average(np.asarray(y, dtype=float), weights=sample_weight)
            self.arm_share_ = float(share)
            self.classes_ = np.array([0.0, 1.0])
            return self
        del X, y, sample_weight
        if self.frame is None:
            raise ValueError("the dose hazard needs the sample's frame")
        history = self.frame.loc[:, list(self.covariates)].to_numpy(dtype=float)
        dose = self.frame["A"].to_numpy(dtype=float)
        if self.marginal:
            history = np.zeros((dose.size, 0))
        model = LinearRegression().fit(history, dose) if history.shape[1] else None
        fitted = model.predict(history) if model is not None else np.full(dose.size, dose.mean())
        self.coef_ = np.asarray(model.coef_ if model is not None else [], dtype=float)
        self.intercept_ = float(model.intercept_) if model is not None else float(dose.mean())
        self.scale_ = float(np.sqrt(np.mean((dose - fitted) ** 2)))
        if self.bounds is not None:
            self.intercept_, self.coef_, self.scale_ = _truncated_fit(
                history, dose, self.bounds, self.intercept_, self.coef_, self.scale_
            )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _cdf(self, x: np.ndarray, mean: Any) -> np.ndarray:
        if self.bounds is None:
            return np.asarray(norm.cdf((x - mean) / self.scale_))
        low, high = (norm.cdf((b - mean) / self.scale_) for b in self.bounds)
        value = (norm.cdf((x - mean) / self.scale_) - low) / (high - low)
        return np.asarray(np.clip(value, 0.0, 1.0))

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        if self._is_arm(design):
            if self.arm_ is None:
                p = np.full(design.shape[0], self.arm_share_)
                return np.column_stack([1.0 - p, p])
            return np.asarray(self.arm_.predict_proba(design))
        width = design.shape[1] - 1
        if width != len(self.covariates):
            raise ValueError(
                f"a dose design has {len(self.covariates) + 1} columns, not {width + 1}"
            )
        index = np.rint(design[:, width]).astype(int)
        mean = self.intercept_ + (design[:, :width] @ self.coef_ if self.coef_.size else 0.0)
        edges = np.asarray(self.edges, dtype=float)
        last = len(edges) - 2
        lower = np.where(index == 0, -np.inf, edges[np.clip(index, 0, last + 1)])
        upper = np.where(index >= last, np.inf, edges[np.clip(index + 1, 0, last + 1)])
        cdf_lower = self._cdf(lower, mean)
        stopped = self._cdf(upper, mean) - cdf_lower
        survived = 1.0 - cdf_lower
        hazard = np.divide(stopped, survived, out=np.ones_like(stopped), where=survived > 0)
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


def _truncated_fit(
    history: np.ndarray,
    dose: np.ndarray,
    bounds: tuple[float, float],
    intercept: float,
    coef: np.ndarray,
    scale: float,
) -> tuple[float, np.ndarray, float]:
    """The maximum-likelihood truncated normal regression of ``dose`` on ``history``."""
    from scipy.optimize import minimize

    design = np.column_stack([np.ones(dose.size), history])
    low, high = bounds

    def loss(theta: np.ndarray) -> float:
        mean, sd = design @ theta[:-1], math.exp(theta[-1])
        mass = norm.cdf((high - mean) / sd) - norm.cdf((low - mean) / sd)
        z = (dose - mean) / sd
        return float(np.mean(theta[-1] + 0.5 * z**2 + np.log(np.maximum(mass, 1e-300))))

    start = np.concatenate([[intercept], coef, [math.log(scale)]])
    theta = minimize(loss, start, method="BFGS", options={"gtol": 1e-10, "maxiter": 1_000}).x
    return float(theta[0]), np.asarray(theta[1:-1], dtype=float), float(math.exp(theta[-1]))


# ---------------------------------------------------------------------------- the wrong limit


def _logistic_limit(rows: np.ndarray, events: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """The weighted logistic MLE of ``events / totals`` on ``rows``, by Newton's method."""
    coefficients = np.zeros(rows.shape[1])
    for _ in range(200):
        p = _expit(rows @ coefficients)
        score = rows.T @ (events - totals * p)
        hessian = (rows.T * (totals * p * (1.0 - p))) @ rows
        step = np.linalg.solve(hessian, score)
        coefficients = coefficients + step
        if np.max(np.abs(step)) < 1e-14:
            break
    return coefficients


def wrong_hazard_coefficients(law: Law) -> list[np.ndarray]:
    """The limit of :class:`NoWLogistic` at each visit, on ``(1, (F,) A)``.

    The units at risk at visit ``k`` are those entering event-free and retained, so their mass
    is :math:`P(W) g(f, a \\mid W) S(k - 1) G_k`.
    """
    keys = [(w1, w2, point) for w1, w2 in STRATA for point in law.points(w1, w2)]
    entering = np.array([MASS * point.mass for _, _, point in keys])

    def row(f: float, a: float) -> list[float]:
        return [1.0, f, a] if law.kind == "vector" else [1.0, a]

    rows = np.array([row(point.f, point.a) for _, _, point in keys])
    out = []
    for node in range(1, law.n_times + 1):
        at_risk = entering * np.array(
            [float(law.retained(node, p.f, p.a, w1, w2)) for w1, w2, p in keys]
        )
        lam = np.array([float(law.hazard(node, p.f, p.a, w1, w2)) for w1, w2, p in keys])
        out.append(_logistic_limit(rows, at_risk * lam, at_risk))
        entering = at_risk * (1.0 - lam)
    return out


def wrong_hazard_limit(law: Law, target: Target) -> float:
    """The plug-in of ``target`` at the limit of :class:`NoWLogistic`."""
    coefficients = wrong_hazard_coefficients(law)

    def curve(f: float, a: float) -> np.ndarray:
        survival, running, out = 1.0, 0.0, []
        for node in range(law.n_times):
            row = np.array([1.0, f, a] if law.kind == "vector" else [1.0, a])
            lam = float(_expit(row @ coefficients[node]))
            running += survival * lam
            survival *= 1.0 - lam
            out.append(running)
        return np.array(out)

    total = 0.0
    for coefficient, horizon, natural in target.terms:
        for w1, w2 in STRATA:
            for point in law.points(w1, w2):
                f, a = (point.f, point.a) if natural else law.policy(point.f, point.a)
                total += (
                    coefficient * MASS * point.mass * float(curve(float(f), float(a))[horizon - 1])
                )
    return total


def positivity(law: Law) -> dict[str, float]:
    """The overlap of a law over every covariate value and treatment point.

    ``ratio_max`` is the largest policy ratio over the support (over the central 99.9% of an
    untruncated Gaussian dose, over the whole interval of a truncated one), ``g_min`` the smallest level probability on ``levels``, ``retained_min``
    the smallest retention through the last visit, and ``arm_min`` the smallest arm
    probability on ``vector``.
    """
    out: dict[str, float] = {"ratio_max": 0.0, "retained_min": 1.0}
    if law.kind == "levels":
        out["g_min"] = float(min(law.level_probabilities(w1, w2).min() for w1, w2 in STRATA))
    if law.kind == "vector":
        p = [float(law.flagged(w1, w2)) for w1, w2 in STRATA]
        out["arm_min"] = min(min(p), 1.0 - max(p))
    edge = float(norm.ppf(0.9995))
    for w1, w2 in STRATA:
        if law.kind == "levels":
            values: Sequence[tuple[int, float]] = [(0, float(a)) for a in LEVELS]
        elif law.bounds is not None:
            values = [(0, float(a)) for a in np.linspace(*law.bounds, 401)]
        else:
            fs = (0, 1) if law.kind == "vector" else (0,)
            values = [
                (f, float(law.location(f, w1, w2)) + law.scale * z)
                for f in fs
                for z in np.linspace(-edge, edge, 41)
            ]
        for f, a in values:
            out["ratio_max"] = max(out["ratio_max"], law.ratio(f, a, w1, w2))
            kept = float(
                np.prod([law.retained(k, f, a, w1, w2) for k in range(1, law.n_times + 1)])
            )
            out["retained_min"] = min(out["retained_min"], kept)
    return out


#: The hazard of the five-visit laws.
HAZARD = Hazard(-2.4, 0.08, -0.25, 0.6, 0.45)
#: The drop-out of every law: logistic in ``(A, W1, W2)``, 0.03 to 0.10 a visit.
DROPOUT = (-3.5, 0.08, 0.3, 0.12, 0.0)
#: The dose location of every law.
CENTRE = (1.5, 1.0, 0.5)
SCALE = 1.5
#: The scale of the ``levels`` dose.  At 1.5 the smallest product of a level probability and the
#: retention through the last visit is 0.0072, under the default cumulative bound of 0.01, so
#: the bound would truncate the natural course's weight (an exact-sample fit with the wrong
#: hazard learner then misses the natural risk by 7e-5).  At 2.0 the product is 0.028.
LEVELS_SCALE = 2.0

LEVELS_LAW = Law("levels", "levels", 5, HAZARD, CENTRE, LEVELS_SCALE, DROPOUT)
GAUSSIAN_LAW = Law("gaussian", "gaussian", 5, HAZARD, CENTRE, SCALE, DROPOUT, shift=1.0)
VECTOR_LAW = Law(
    "vector",
    "vector",
    1,
    Hazard(-1.2, 0.0, -0.25, 0.6, 0.45, flag=-0.5),
    CENTRE,
    SCALE,
    (-2.5, 0.08, 0.3, 0.12, 0.2),
    shift=1.0,
    flag_shift=0.5,
    arm=(-0.6, 0.8, 0.2),
)
#: The gated law of the density route: the Gaussian dose truncated to :data:`TRUNCATION` and the
#: shift capped at its top, so every dose the ratio is read at lies where the density's
#: quantile bins are narrower than the shift.  On the untruncated law the top bins are wider
#: than the shift and the binned ratio misses there (``test_hazard_mtp_study_design.py``,
#: ``TestTheInfluenceCurveGap``).
TRUNCATION = (0.0, 5.5)
TRUNCATED_LAW = Law(
    "truncated",
    "gaussian",
    5,
    HAZARD,
    CENTRE,
    SCALE,
    DROPOUT,
    shift=1.0,
    bounds=TRUNCATION,
    cap=TRUNCATION[1],
)
LAWS = {
    "levels": LEVELS_LAW,
    "gaussian": GAUSSIAN_LAW,
    "truncated": TRUNCATED_LAW,
    "vector": VECTOR_LAW,
}
