r"""The law, truths, efficiency bounds and learners of ``hazard-time-varying-multinode``.

**Law (M).**  :math:`W_1 \in \{0, 1\}` and :math:`W_2 \in \{0, \dots, 3\}` are independent and
uniform, on the strata of :data:`tests.studies.hazard_time_varying_law.STRATA`.  At each of
:math:`T = 3` nodes a binary covariate :math:`L_k`, coded :math:`-1` and :math:`+1` in the frame,
is measured first (not at node 1), then the binary treatment :math:`A_k`, then dropout, then the
event:

.. math::

    P(L_k = +1 \mid W, A_{k-1}, L_{k-1}) &= \operatorname{expit}(l_0 + l_1 W_1 + l_A A_{k-1}
        + l_L [L_{k-1} = +1]), \\
    g_1(1 \mid W) &= \operatorname{expit}(a_0 + a_1 W_1 + a_2 W_2), \\
    g_k(1 \mid W, A_{k-1}, L_k) &= \operatorname{expit}(b_0 + b_1 W_1 + b_A A_{k-1}
        + b_L [L_k = +1]), \quad k \ge 2, \\
    c_k(W, A_k, L_k) &= \operatorname{expit}(c_0 + c_1 W_1 + c_2 W_2 + c_A A_k
        + c_L [L_k = +1]), \\
    \lambda(k \mid W, A_{k-1}, A_k, L_k) &= \operatorname{expit}(h_0 + h_t (k - 1) + h_1 W_1
        + h_2 W_2 + h_P A_{k-1} + h_A A_k + h_L [L_k = +1]),

with :math:`A_0 = 0` and :math:`L_1` absent.  One cause.  Every factor at node :math:`k` reads
:math:`(W, A_{k-1}, L_k)` and :math:`A_k`, so the law is Markov: a logistic regression on
:math:`(W_1, W_2, L_k, A_{k-1}, A_k)` is a correctly specified hazard (and on
:math:`(W_1, W_2, L_k, A_k)` a correct retention model) without being saturated.  The hazard
reads :math:`A_{k-1}` (plan X70 rev 3.6, plan review 4 finding 3), so a slot that sets the
earlier treatment to the slot's level, or reads the held arm, misses.

**Regimens.**  Always, never, the rule (treat when :math:`W_1 = 1` at node 1 and when
:math:`L_k = +1` after) and the known policy :math:`q_1(1 \mid W_1) = 0.3 + 0.4 [W_1 = 1]` and
:math:`q_k(1 \mid L_k, A_{k-1}) = 0.05 + 0.3 [L_k = +1] + 0.6 A_{k-1}` after (plan X70
rev 3.7: the large :math:`A_{k-1}` term lets a defect of the policy's prefix or slot show).

**Truths.**  :math:`\psi_r(t) = E\,Q^r_1(H_1)` with
:math:`Q^r_k(H_k) = \sum_a q_{r,k}(a \mid H_k)[\lambda(k \mid H_k, a) + (1 - \lambda(k \mid H_k,
a)) E\{Q^r_{k+1}(H_{k+1}) \mid H_k, a\}]` for :math:`k < t`, the bracket's second term dropped
at :math:`k = t`.  On this Markov law :math:`Q^r_k` reads :math:`(W, A_{k-1}, L_k)` only, so the
recursion is an exact finite sum (:meth:`Law.q_bar`).

**Efficiency bounds.**  The efficient influence function is the iterated-mean one (Díaz,
Williams, Hoffman and Schenck 2023, Theorem 1),

.. math::

    D = Q^r_1(H_1) - \psi + \sum_{k \le t} h^r_k R_k \bigl(Z_{k+1} - \bar Q^r_k(H_k, A_k)\bigr),
    \qquad h^r_k = \prod_{s \le k} \frac{q_{r,s}(A_s \mid H_s)}{g_s(A_s \mid H_s)\, c_s},

with :math:`\bar Q^r_k(H_k, a)` the bracket above and :math:`Z_{k+1}` the event indicator at
:math:`k`, or :math:`Q^r_{k+1}` at the observed :math:`H_{k+1}` after a node without an event
(zero at :math:`k = t`).  :func:`efficiency_sd` sums its square over the support.  Nothing here
calls the estimator.

**The learners.**  ``L`` columns are the only columns of a design with a negative entry, and a
node design is :math:`[W_1, W_2, L_2, \dots, L_k, A_1, \dots, A_k]`.  :class:`Window` keeps
:math:`W`, the last ``L`` column and the last ``arms`` treatment columns; with ``drop_l`` it keeps
no ``L`` column.  The continuation design (``outcome_design``) holds :math:`W`, the covariates,
the arm blocks of the policy nodes and one indicator per later node, and
:class:`~tests.studies.hazard_time_varying_law.LastL` keeps all but the earlier ``L`` columns.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from tests.studies.hazard_time_varying_law import (
    MASS,
    STRATA,
    CellMeans,
    LastL,
    Logistic,
    NoL,
)

#: A regimen's density at node ``k``: ``(node, W1, A_{k-1}, L_k) -> P(A_k = 1)``.
Density = Callable[[int, int, int, int | None], float]


def _flag(l: int | None) -> float:
    return 1.0 if l == 1 else 0.0


@dataclass(frozen=True)
class Coefficients:
    """The coefficients of law (M), named as in the module docstring.

    Parameters
    ----------
    covariate : tuple of float
        ``(l_0, l_1, l_A, l_L)``.
    first : tuple of float
        ``(a_0, a_1, a_2)``, the node-1 treatment.
    later : tuple of float
        ``(b_0, b_1, b_A, b_L)``, the treatment at ``k >= 2``.
    retention : tuple of float
        ``(c_0, c_1, c_2, c_A, c_L)``.
    hazard : tuple of float
        ``(h_0, h_t, h_1, h_2, h_P, h_A, h_L)``.
    """

    covariate: tuple[float, float, float, float]
    first: tuple[float, float, float]
    later: tuple[float, float, float, float]
    retention: tuple[float, float, float, float, float]
    hazard: tuple[float, float, float, float, float, float, float]


#: The declared coefficients.  The design test checks every positivity and witness condition
#: of plan X70 section 5 on them before any pilot.  ``h_P`` and ``h_A`` are large so a slot
#: defect moves the policy's limit; ``h_0`` keeps every follower's event cell dense enough that
#: a held cell stays rare at the ladder's n = 1,000, and every hazard inside the fit's clip.
#: ``b_L`` and ``c_L`` size the wrong mechanism, the factor every ``outcome_correct`` witness
#: multiplies, as far as a cumulative mechanism of at least ``g_bounds``' 0.01 on every
#: treatment path allows.
COEFFICIENTS = Coefficients(
    covariate=(-0.6, 0.8, -0.8, 3.5),
    first=(-0.4, 0.6, 0.15),
    later=(-1.4, 0.3, 0.4, 2.0),
    retention=(4.4, 0.2, -0.15, 0.3, -3.0),
    hazard=(-1.6, 0.2, 0.3, 0.15, -2.3, -2.0, 3.2),
)


def always(node: int, w1: int, previous: int, l: int | None) -> float:
    """Always treat."""
    del node, w1, previous, l
    return 1.0


def never(node: int, w1: int, previous: int, l: int | None) -> float:
    """Never treat."""
    del node, w1, previous, l
    return 0.0


def rule(node: int, w1: int, previous: int, l: int | None) -> float:
    """Treat when ``W1 = 1`` at node 1 and when ``L_k = +1`` after."""
    del previous
    return float(w1 == 1) if node == 1 else _flag(l)


#: The known policy at ``k >= 2``: ``(base, L_k = +1 term, A_{k-1} term)``.  It keeps
#: ``q_k`` in ``[0.05, 0.95]`` on every path.
POLICY_LATER = (0.05, 0.3, 0.6)


def policy(node: int, w1: int, previous: int, l: int | None) -> float:
    """The known policy: ``0.3 + 0.4 [W1 = 1]``, then ``0.05 + 0.3 [L_k = +1] + 0.6 A_{k-1}``."""
    if node == 1:
        return 0.3 + 0.4 * w1
    base, covariate, prefix = POLICY_LATER
    return base + covariate * _flag(l) + prefix * previous


#: The regimens by label, in the order a fit reports them.
REGIMENS: dict[str, Density] = {
    "always": always,
    "never": never,
    "rule": rule,
    "policy": policy,
}
REFERENCE = "never"


@dataclass(frozen=True)
class Law:
    """Law (M), with its support enumerated.

    A support point is ``(W1, W2)`` followed by ``(L_k, A_k, C_k, Y_k)`` at each node, ``None``
    where not reached; ``L_1`` is ``None``.

    Parameters
    ----------
    name : str
        The law's name.
    coefficients : Coefficients
        Its coefficients.
    n_times : int
        The nodes ``T``.
    """

    name: str = "multinode"
    coefficients: Coefficients = COEFFICIENTS
    n_times: int = 3
    support: tuple[tuple[Any, ...], ...] = field(init=False, repr=False, compare=False)
    probs: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w1, w2 in STRATA:
            self._node((w1, w2), MASS, 1, 0, None, points)
        object.__setattr__(self, "support", tuple(point for point, _ in points))
        object.__setattr__(self, "probs", np.array([mass for _, mass in points]))

    # ---------------------------------------------------------------- the tables

    @property
    def width(self) -> int:
        return 2 + 4 * self.n_times

    def p_l(self, w1: int, previous_arm: int, previous: int | None) -> float:
        """``P(L_k = +1 | W1, A_{k-1}, L_{k-1})``."""
        l0, l1, la, ll = self.coefficients.covariate
        return float(expit(l0 + l1 * w1 + la * previous_arm + ll * _flag(previous)))

    def g(self, node: int, w1: int, w2: int, previous_arm: int, l: int | None) -> float:
        """``P(A_k = 1 | H_k)``."""
        if node == 1:
            a0, a1, a2 = self.coefficients.first
            return float(expit(a0 + a1 * w1 + a2 * w2))
        b0, b1, ba, bl = self.coefficients.later
        return float(expit(b0 + b1 * w1 + ba * previous_arm + bl * _flag(l)))

    def retention(self, w1: int, w2: int, a: int, l: int | None) -> float:
        """``P(C_k = 1 | W, A_k, L_k)``."""
        c0, c1, c2, ca, cl = self.coefficients.retention
        return float(expit(c0 + c1 * w1 + c2 * w2 + ca * a + cl * _flag(l)))

    def hazard(
        self, node: int, w1: int, w2: int, previous_arm: int, a: int, l: int | None
    ) -> float:
        """``lambda(k | W, A_{k-1}, A_k, L_k)``, the event hazard given followed."""
        h0, ht, h1, h2, hp, ha, hl = self.coefficients.hazard
        eta = h0 + ht * (node - 1) + h1 * w1 + h2 * w2 + hp * previous_arm + ha * a + hl * _flag(l)
        return float(expit(eta))

    # ---------------------------------------------------------------- the support

    def _node(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        previous_arm: int,
        previous: int | None,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w1, w2 = prefix[:2]
        if node == 1:
            branches: list[tuple[int | None, float]] = [(None, 1.0)]
        else:
            p = self.p_l(w1, previous_arm, previous)
            branches = [(1, p), (-1, 1.0 - p)]
        for l, share in branches:
            g = self.g(node, w1, w2, previous_arm, l)
            for a, arm_share in ((1, g), (0, 1.0 - g)):
                self._enter((*prefix, l, a), mass * share * arm_share, node, previous_arm, l, out)

    def _enter(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        previous_arm: int,
        l: int | None,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        w1, w2 = prefix[:2]
        a = prefix[-1]
        kept = self.retention(w1, w2, a, l)
        point = (*prefix, 0)
        out.append((point + (None,) * (self.width - len(point)), mass * (1.0 - kept)))
        lam = self.hazard(node, w1, w2, previous_arm, a, l)
        point = (*prefix, 1, 1)
        out.append((point + (None,) * (self.width - len(point)), mass * kept * lam))
        self._node((*prefix, 1, 0), mass * kept * (1.0 - lam), node + 1, a, l, out)

    @staticmethod
    def value(point: tuple[Any, ...], name: str, node: int) -> Any:
        """The value of ``L``, ``A``, ``C`` or ``Y`` at ``node`` in a support point."""
        return point[2 + 4 * (node - 1) + {"L": 0, "A": 1, "C": 2, "Y": 3}[name]]

    # ---------------------------------------------------------------- frames

    def columns(self, *, weights: str | None = None) -> dict[str, Any]:
        """The keywords of ``LTMLE.fit`` for a frame of this law."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": [f"A{k}" for k in nodes],
            "baseline": ["W1", "W2"],
            "censoring": [f"C{k}" for k in nodes],
            "outcome": [f"Y{k}" for k in nodes],
            "time_varying": [[] if k == 1 else [f"L{k}"] for k in nodes],
        }
        if weights is not None:
            out["weights"] = weights
        return out

    def _frame_of(self, rows: Sequence[tuple[Any, ...]]) -> pd.DataFrame:
        def column(getter: Callable[[tuple[Any, ...]], Any]) -> np.ndarray:
            return np.array([np.nan if getter(p) is None else float(getter(p)) for p in rows])

        out: dict[str, np.ndarray] = {
            "W1": column(lambda p: p[0]),
            "W2": column(lambda p: p[1]),
        }
        for node in range(1, self.n_times + 1):
            for name in ("L", "A", "C", "Y"):
                if name == "L" and node == 1:
                    continue
                out[f"{name}{node}"] = column(lambda p, n=name, k=node: self.value(p, n, k))
        return pd.DataFrame(out)

    def exact_frame(self) -> pd.DataFrame:
        """The law as data: one row per support point, its probability in ``wt``."""
        frame = self._frame_of(self.support)
        frame["wt"] = self.probs
        return frame

    def draw(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` independent units, in the layout of :meth:`exact_frame` without ``wt``."""
        rng = np.random.default_rng(seed)
        w1 = rng.integers(0, 2, size=n)
        w2 = rng.integers(0, 4, size=n)
        out: dict[str, np.ndarray] = {"W1": w1.astype(float), "W2": w2.astype(float)}
        open_ = np.ones(n, dtype=bool)
        previous_arm = np.zeros(n, dtype=int)
        previous = np.zeros(n, dtype=int)
        c = self.coefficients
        for node in range(1, self.n_times + 1):
            if node > 1:
                l0, l1, la, ll = c.covariate
                p = expit(l0 + l1 * w1 + la * previous_arm + ll * (previous == 1))
                drawn = np.where(rng.random(n) < p, 1, -1)
                out[f"L{node}"] = np.where(open_, drawn, np.nan)
                current = np.where(open_, drawn, 0)
                b0, b1, ba, bl = c.later
                g = expit(b0 + b1 * w1 + ba * previous_arm + bl * (current == 1))
            else:
                current = np.zeros(n, dtype=int)
                a0, a1, a2 = c.first
                g = expit(a0 + a1 * w1 + a2 * w2)
            a = (rng.random(n) < g).astype(int)
            out[f"A{node}"] = np.where(open_, a, np.nan)
            c0, c1, c2, ca, cl = c.retention
            kept = rng.random(n) < expit(c0 + c1 * w1 + c2 * w2 + ca * a + cl * (current == 1))
            out[f"C{node}"] = np.where(open_, kept.astype(float), np.nan)
            followed = open_ & kept
            h0, ht, h1, h2, hp, ha, hl = c.hazard
            eta = (
                h0
                + ht * (node - 1)
                + h1 * w1
                + h2 * w2
                + hp * previous_arm
                + ha * a
                + hl * (current == 1)
            )
            event = rng.random(n) < expit(eta)
            out[f"Y{node}"] = np.where(followed, event.astype(float), np.nan)
            open_ = followed & ~event
            previous_arm = np.where(open_, a, 0)
            previous = np.where(open_, current, 0)
        ordered = ["W1", "W2"]
        for node in range(1, self.n_times + 1):
            ordered += [f"L{node}"] if node > 1 else []
            ordered += [f"A{node}", f"C{node}", f"Y{node}"]
        return pd.DataFrame(out).loc[:, ordered]

    # ---------------------------------------------------------------- the truth

    def q_bar(
        self,
        density: Density,
        node: int,
        w1: int,
        w2: int,
        previous_arm: int,
        l: int | None,
        t: int,
    ) -> float:
        r""":math:`Q^r_k(H_k)`: the probability of an event by node ``t`` under the regimen."""
        return _q_bar(self, density, node, w1, w2, previous_arm, l, t)

    def q_arm(
        self,
        density: Density,
        node: int,
        w1: int,
        w2: int,
        previous_arm: int,
        l: int | None,
        a: int,
        t: int,
    ) -> float:
        r""":math:`\bar Q^r_k(H_k, a)`: the bracket of the recursion at treatment ``a``."""
        lam = self.hazard(node, w1, w2, previous_arm, a, l)
        if node >= t:
            return lam
        p = self.p_l(w1, a, l)
        later = p * _q_bar(self, density, node + 1, w1, w2, a, 1, t) + (1.0 - p) * _q_bar(
            self, density, node + 1, w1, w2, a, -1, t
        )
        return lam + (1.0 - lam) * later

    def truth(self, density: Density, t: int) -> float:
        """``E Q^r_1(H_1)``."""
        return float(sum(MASS * self.q_bar(density, 1, w1, w2, 0, None, t) for w1, w2 in STRATA))

    def eif(self, density: Density, t: int) -> np.ndarray:
        """The efficient influence function of ``psi_r(t)`` at every support point."""
        psi = self.truth(density, t)
        return np.array([self._eif_at(point, density, t) for point in self.support]) - psi

    def _eif_at(self, point: tuple[Any, ...], density: Density, t: int) -> float:
        w1, w2 = point[:2]
        value = self.q_bar(density, 1, w1, w2, 0, None, t)
        weight = 1.0
        previous_arm = 0
        for node in range(1, t + 1):
            l = self.value(point, "L", node)
            a = self.value(point, "A", node)
            if a is None:
                break
            q = density(node, w1, previous_arm, l)
            g = self.g(node, w1, w2, previous_arm, l)
            weight *= (q if a == 1 else 1.0 - q) / (g if a == 1 else 1.0 - g)
            weight /= self.retention(w1, w2, a, l)
            if self.value(point, "C", node) == 0 or weight == 0.0:
                break
            y = self.value(point, "Y", node)
            if y == 0 and node < t:
                z = self.q_bar(density, node + 1, w1, w2, a, self.value(point, "L", node + 1), t)
            else:
                z = float(y == 1)
            value += weight * (z - self.q_arm(density, node, w1, w2, previous_arm, l, a, t))
            if y == 1:
                break
            previous_arm = a
        return value

    def frame_curves(self, target: Target, frame: pd.DataFrame) -> np.ndarray:
        """The centred efficient influence function of ``target`` at each unit of ``frame``."""
        table = dict(zip(self.support, target.curve(self), strict=True))
        return np.array([table[key] for key in _keys(self, frame)])


@cache
def _q_bar(
    law: Law,
    density: Density,
    node: int,
    w1: int,
    w2: int,
    previous_arm: int,
    l: int | None,
    t: int,
) -> float:
    q = density(node, w1, previous_arm, l)
    total = 0.0
    for a, share in ((1, q), (0, 1.0 - q)):
        if share:
            total += share * law.q_arm(density, node, w1, w2, previous_arm, l, a, t)
    return total


def _keys(law: Law, frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Each unit of ``frame`` as the support point it realises."""

    def cell(record: dict[str, Any], name: str) -> Any:
        if name not in record or pd.isna(record[name]):
            return None
        return int(record[name])

    out = []
    for record in frame.to_dict("records"):
        point: list[Any] = [int(record["W1"]), int(record["W2"])]
        for node in range(1, law.n_times + 1):
            point += [cell(record, f"{name}{node}") for name in ("L", "A", "C", "Y")]
        out.append(tuple(point))
    return out


# -------------------------------------------------------------------- targets


@dataclass(frozen=True)
class Target:
    """A linear combination of risks: ``(coefficient, regimen label, horizon)`` terms.

    Parameters
    ----------
    terms : tuple of tuple
        Each ``(coefficient, regimen label, horizon)``.
    """

    terms: tuple[tuple[float, str, int], ...]

    def value(self, law: Law) -> float:
        return float(sum(c * law.truth(REGIMENS[r], t) for c, r, t in self.terms))

    def curve(self, law: Law) -> np.ndarray:
        """The centred efficient influence function at every support point."""
        return np.asarray(sum(c * law.eif(REGIMENS[r], t) for c, r, t in self.terms))


def efficiency_sd(law: Law, target: Target) -> float:
    """The standard deviation of the efficient influence function of ``target``."""
    curve = target.curve(law)
    return float(math.sqrt(float(np.sum(law.probs * curve**2))))


def positivity(law: Law) -> dict[str, float]:
    r"""The range of every treatment probability and hazard, the smallest retention, and the
    smallest cumulative mechanism.

    ``g_min`` and ``g_max`` run over every node, stratum, earlier treatment and covariate, and
    ``hazard_min`` and ``hazard_max`` over every history a node can hold.
    ``cumulative_min`` is the smallest :math:`\prod_{s \le k} g_s(A_s \mid H_s)` and
    :math:`\prod_{s \le k} g_s(A_s \mid H_s)\, c_s` over every node and every path of
    treatments and covariates: the known policy puts mass on every path, so each is the
    cumulative weight of a follower.
    """
    values = []
    hazards = []
    for w1, w2 in STRATA:
        values.append(law.g(1, w1, w2, 0, None))
        hazards += [law.hazard(1, w1, w2, 0, a, None) for a in (0, 1)]
        for node in range(2, law.n_times + 1):
            for previous_arm in (0, 1):
                for l in (1, -1):
                    values.append(law.g(node, w1, w2, previous_arm, l))
                    hazards += [law.hazard(node, w1, w2, previous_arm, a, l) for a in (0, 1)]
    retained = min(
        law.retention(w1, w2, a, l) for w1, w2 in STRATA for a in (0, 1) for l in (None, 1, -1)
    )
    cumulative = 1.0
    for w1, w2 in STRATA:
        for arms in itertools.product((0, 1), repeat=law.n_times):
            for path in itertools.product((1, -1), repeat=law.n_times - 1):
                weight, previous_arm = 1.0, 0
                for node, a in enumerate(arms, start=1):
                    l = None if node == 1 else path[node - 2]
                    g = law.g(node, w1, w2, previous_arm, l)
                    weight *= g if a == 1 else 1.0 - g
                    cumulative = min(cumulative, weight)
                    weight *= law.retention(w1, w2, a, l)
                    cumulative = min(cumulative, weight)
                    previous_arm = a
    return {
        "g_min": float(min(values)),
        "g_max": float(max(values)),
        "hazard_min": float(min(hazards)),
        "hazard_max": float(max(hazards)),
        "retention_min": retained,
        "cumulative_min": float(cumulative),
    }


def follower_mass(law: Law, density: Density, node: int) -> float:
    """The mass at risk at ``node`` whose treatments through ``node`` the regimen puts mass on.

    A follower of a known policy counts with its weight one; the mass is that of the units
    at risk at ``node`` whose every observed treatment has positive regimen probability.
    """
    total = 0.0
    for point, mass in zip(law.support, law.probs, strict=True):
        w1 = point[0]
        previous_arm = 0
        follows = True
        for k in range(1, node + 1):
            a = law.value(point, "A", k)
            if a is None or (k < node and law.value(point, "Y", k) != 0):
                follows = False
                break
            q = density(k, w1, previous_arm, law.value(point, "L", k))
            if (q if a == 1 else 1.0 - q) == 0.0:
                follows = False
                break
            if k < node and law.value(point, "C", k) != 1:
                follows = False
                break
            previous_arm = a
        if follows:
            total += mass
    return total


# -------------------------------------------------------------------- learners


def _l_columns(matrix: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.any(matrix < 0.0, axis=0))


class Window(BaseEstimator):
    """``base`` on ``W``, the last ``L`` column and the last ``arms`` treatment columns.

    A node design is ``[W1, W2, L_2, ..., L_k, A_1, ...]``, so the treatment columns are those
    after the baseline and the ``L`` columns.  ``drop_l`` keeps no ``L`` column.

    Parameters
    ----------
    base : estimator
        The learner fitted on the kept columns.
    arms : int
        The trailing treatment columns kept.
    drop_l : bool
        Whether to drop every ``L`` column.
    baseline : int
        The leading baseline columns.
    """

    def __init__(self, base: Any, arms: int = 2, drop_l: bool = False, baseline: int = 2) -> None:
        self.base = base
        self.arms = arms
        self.drop_l = drop_l
        self.baseline = baseline

    def _kept(self, matrix: np.ndarray) -> np.ndarray:
        covariates = _l_columns(matrix)
        first_arm = self.baseline + covariates.size
        treatments = np.arange(first_arm, matrix.shape[1])
        kept = [*range(self.baseline)]
        if covariates.size and not self.drop_l:
            kept.append(int(covariates[-1]))
        kept += [int(column) for column in treatments[-self.arms :]] if self.arms else []
        return np.array(sorted(kept), dtype=int)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Window:
        matrix = np.asarray(X, dtype=float)
        self.keep_ = self._kept(matrix)
        self.model_ = clone(self.base)
        self.model_.fit(matrix[:, self.keep_], y, sample_weight=sample_weight)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        return self

    def _reduce(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self.keep_]

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._reduce(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._reduce(X)))


class NewtonLogistic(Logistic):
    """An unpenalised logistic regression solved by Newton steps to ``tol=1e-12``.

    The quasi-Newton solve of :class:`~tests.studies.hazard_time_varying_law.Logistic` stops at
    its gradient tolerance, which on this law's exact sample leaves the cells whose hazard is
    near 1e-3 a few 1e-9 from their maximum-likelihood fit: the exact limits then miss by more
    than ``EXACT``.  The model is the same.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NewtonLogistic:
        self.model_ = LogisticRegression(
            C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=1_000
        )
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self


def hazard_learner(correct: bool) -> Window:
    """Logistic on ``(W, L_k, A_{k-1}, A_k)`` when correct; without ``L_k`` when not."""
    return Window(NewtonLogistic(), arms=2, drop_l=not correct)


def continuation_learner(correct: bool) -> BaseEstimator:
    """The cell mean over every continuation column but the earlier ``L``; without ``L``."""
    return LastL(CellMeans()) if correct else NoL(CellMeans())


def treatment_learner(correct: bool) -> Window:
    """Logistic on ``(W, L_k, A_{k-1})`` when correct; without ``L_k`` when not."""
    return Window(NewtonLogistic(), arms=1, drop_l=not correct)


def censoring_learner(correct: bool) -> Window:
    """Logistic on ``(W, L_k, A_k)`` when correct; without ``L_k`` when not."""
    return Window(NewtonLogistic(), arms=1, drop_l=not correct)


LAW = Law()

__all__ = [
    "COEFFICIENTS",
    "LAW",
    "POLICY_LATER",
    "REFERENCE",
    "REGIMENS",
    "Coefficients",
    "Law",
    "NewtonLogistic",
    "Target",
    "Window",
    "censoring_learner",
    "continuation_learner",
    "efficiency_sd",
    "follower_mass",
    "hazard_learner",
    "positivity",
    "treatment_learner",
]
