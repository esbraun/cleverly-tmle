r"""The laws, truths, efficiency bounds and learners of ``hazard-time-varying``.

**The laws.**  :math:`W_1 \in \{0, 1\}` and :math:`W_2 \in \{0, \dots, 3\}` are independent and
uniform.  A baseline treatment :math:`A` is drawn from :math:`g(a \mid W)` and held over the
nodes.  At node :math:`k \ge 2` a binary covariate :math:`L_k` is measured first, coded
:math:`-1` and :math:`+1` in the frame, from

.. math::

    P(L_k = +1 \mid W, A, L_{k-1}) = \operatorname{expit}(l_0 + l_1 W_1 + l_A(A)
        + l_L [L_{k-1} = +1]),

then the unit drops out with probability
:math:`\operatorname{expit}(d_0 + d_t (k - 1) + d_A(A) + d_1 W_1 + d_2 W_2 + d_L [L_k = +1])`,
and a unit still followed has the event of cause :math:`j` with the sequential pseudo-hazard

.. math::

    \rho_j(k \mid W, A, L_k) = \operatorname{expit}(h_{j0} + h_{jt} (k - 1) + h_{jA}(A)
        + h_{j1} W_1 + h_{j2} W_2 + h_{jL} [L_k = +1]),

so the cause-specific hazards are :math:`\lambda_1 = \rho_1` and
:math:`\lambda_2 = \rho_2 (1 - \rho_1)`, the order in which
:func:`cleverly.longitudinal.hazard.fit_hazards` fits them.  :math:`L_1` is not measured, so at
node 1 the terms in :math:`L` vanish.  Each law is Markov: :math:`L_k` reads :math:`L_{k-1}`
only and every factor at node :math:`k` reads :math:`L_k` only.  So a logistic regression on
:math:`(W_1, W_2, A, L_k)` at each node is a correctly specified hazard and censoring model,
not a saturated one, and the cell mean over :math:`(W, A, L_k, s)` is the true regression of the
future: the conditional distribution of the later events given the node's history.

=================  ==========================================================================
law                design
=================  ==========================================================================
``one_cause``      binary :math:`A`, one cause, five nodes
``competing``      binary :math:`A`, two causes, four nodes
``three_arm``      :math:`A \in \{0, 1, 2\}` with a multinomial logit :math:`g`, one cause,
                   five nodes; the regimen "one level up, the top stays" is a modified
                   treatment policy at node 1
=================  ==========================================================================

**Truths.**  :math:`\psi_d(t, j) = E\,Q_1(W, d(A); t, j)` with the backward recursion
:math:`Q_k(W, a, L_k) = \lambda_j(k) + (1 - \textstyle\sum_i \lambda_i(k))\,
E[Q_{k+1}(W, a, L_{k+1}) \mid W, a, L_k]` over the binary covariate paths: an exact finite sum.
A regimen is a map :math:`d` of the natural treatment: a constant for a static regimen, the
identity for the natural course and :math:`\min(a + 1, 2)` for "one level up".

**Efficiency bounds.**  The efficient influence function of :math:`\psi_d(t, j)` is the
iterated-mean form (van der Laan and Gruber 2012; Díaz, Williams, Hoffman and Schenck 2023,
Theorem 1, for the modified treatment policy at one decision)

.. math::

    D = Q_1(W, d(A)) - \psi + \sum_{k \le t} \frac{r_d(A, W)}{\prod_{i \le k} c_i}
        R_k \bigl(Z_{k+1} - Q_k(W, A, L_k)\bigr),

with :math:`r_d = g^d / g`, :math:`g^d` the law of :math:`d(A)` given :math:`W`, :math:`R_k`
the indicator of being followed and event-free at node :math:`k`, and :math:`Z_{k+1}` the
indicator of a cause-:math:`j` event at :math:`k`, or :math:`Q_{k+1}` at the observed
:math:`L_{k+1}` after a node without an event.  :func:`efficiency_sd` sums its square over the
law's support.  Nothing here calls the estimator.

**The learners.**  ``L`` columns are the only columns of a design with a negative entry.
:class:`LastL` keeps the other columns and the last ``L`` column (:math:`L_k` on a node design);
:class:`NoL` drops every ``L`` column.  The wrong hazard and continuation learners are
:class:`NoL` fits, whose plug-in treats the drop-out as ignorable given :math:`(W, A)`; on these
laws the drop-out reads :math:`L_k`, so that plug-in is biased.  :func:`wrong_plug_in_limit`
computes its limit from the law alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression

#: The baseline strata and their mass.
STRATA = tuple((w1, w2) for w1 in (0, 1) for w2 in (0, 1, 2, 3))
MASS = 0.125
#: The cause labels of the two-cause law, by event code.
CAUSES = {1: "relapse", 2: "death"}

Regimen = Callable[[int], int]


@dataclass(frozen=True)
class Logit:
    """One logistic factor: ``intercept + trend (k - 1) + arm[a] + w1 W1 + w2 W2 + l [L = +1]``.

    Parameters
    ----------
    intercept : float
        The log odds at node 1, arm 0, ``W = 0`` and ``L`` absent.
    trend : float
        Per node after the first.
    arm : tuple of float
        The effect of each treatment level; level 0 reads ``arm[0]``.
    w1 : float
        The coefficient of ``W1``.
    w2 : float
        The coefficient of ``W2``.
    l : float
        The coefficient of ``[L_k = +1]``.
    """

    intercept: float
    trend: float
    arm: tuple[float, ...]
    w1: float
    w2: float
    l: float

    def __call__(self, node: int, a: int, w1: int, w2: int, l: int | None) -> float:
        flag = 1.0 if l == 1 else 0.0
        eta = (
            self.intercept
            + self.trend * (node - 1)
            + self.arm[a]
            + self.w1 * w1
            + self.w2 * w2
            + self.l * flag
        )
        return float(expit(eta))


@dataclass(frozen=True)
class Covariate:
    """``P(L_k = +1 | W1, A, L_{k-1}) = expit(intercept + w1 W1 + arm[a] + previous [L_{k-1} = +1])``.

    Parameters
    ----------
    intercept : float
        The log odds at ``W1 = 0``, arm 0 and ``L_{k-1}`` absent or ``-1``.
    w1 : float
        The coefficient of ``W1``.
    arm : tuple of float
        The effect of each treatment level.
    previous : float
        The coefficient of ``[L_{k-1} = +1]``.
    """

    intercept: float
    w1: float
    arm: tuple[float, ...]
    previous: float

    def __call__(self, a: int, w1: int, previous: int | None) -> float:
        flag = 1.0 if previous == 1 else 0.0
        return float(expit(self.intercept + self.w1 * w1 + self.arm[a] + self.previous * flag))


@dataclass(frozen=True)
class Law:
    """One law of the study, with its support enumerated.

    Parameters
    ----------
    name : str
        The law's key in :data:`LAWS`.
    n_times : int
        The nodes ``T``.
    treatment : tuple of tuple of float
        The multinomial logit of ``A`` against level 0: one ``(intercept, w1, w2)`` per level
        after the first.
    covariate : Covariate
        The law of ``L_k``.
    dropout : Logit
        The probability of dropping out at a node, given the node's ``L_k``.
    hazards : tuple of Logit
        The sequential pseudo-hazard of each cause.
    """

    name: str
    n_times: int
    treatment: tuple[tuple[float, float, float], ...]
    covariate: Covariate
    dropout: Logit
    hazards: tuple[Logit, ...]
    support: tuple[tuple[Any, ...], ...] = field(init=False, repr=False, compare=False)
    probs: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w1, w2 in STRATA:
            g = self.g(w1, w2)
            for a in range(self.arms):
                self._walk((w1, w2, a), MASS * g[a], 1, None, points)
        object.__setattr__(self, "support", tuple(point for point, _ in points))
        object.__setattr__(self, "probs", np.array([mass for _, mass in points]))

    # ---------------------------------------------------------------- the tables

    @property
    def arms(self) -> int:
        return len(self.treatment) + 1

    @property
    def causes(self) -> int:
        return len(self.hazards)

    def g(self, w1: int, w2: int) -> np.ndarray:
        """``P(A = a | W)`` for every level."""
        logits = np.array([0.0, *(b0 + b1 * w1 + b2 * w2 for b0, b1, b2 in self.treatment)])
        weights = np.exp(logits - logits.max())
        return np.asarray(weights / weights.sum())

    def retention(self, node: int, a: int, w1: int, w2: int, l: int | None) -> float:
        """``P(C_k = 1 | W, A, L_k)``."""
        return 1.0 - self.dropout(node, a, w1, w2, l)

    def cause_hazards(self, node: int, a: int, w1: int, w2: int, l: int | None) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given followed and event-free."""
        out: list[float] = []
        taken = 0.0
        for factor in self.hazards:
            value = factor(node, a, w1, w2, l) * (1.0 - taken)
            out.append(value)
            taken += value
        return tuple(out)

    # ---------------------------------------------------------------- the support

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        previous: int | None,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w1, _, a = prefix[:3]
        if node == 1:
            self._enter((*prefix, None), mass, node, None, out)
            return
        p = self.covariate(a, w1, previous)
        for value, share in ((1, p), (-1, 1.0 - p)):
            self._enter((*prefix, value), mass * share, node, value, out)

    def _enter(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        l: int | None,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        width = 3 + 3 * self.n_times
        w1, w2, a = prefix[:3]
        kept = self.retention(node, a, w1, w2, l)
        point = (*prefix, 0)
        out.append((point + (None,) * (width - len(point)), mass * (1.0 - kept)))
        lam = self.cause_hazards(node, a, w1, w2, l)
        for code, share in enumerate(lam, start=1):
            point = (*prefix, 1, code)
            out.append((point + (None,) * (width - len(point)), mass * kept * share))
        self._walk((*prefix, 1, 0), mass * kept * (1.0 - sum(lam)), node + 1, l, out)

    @staticmethod
    def value(point: tuple[Any, ...], name: str, node: int) -> Any:
        """The value of ``L``, ``C`` or ``Y`` at ``node`` in a support point."""
        return point[3 + 3 * (node - 1) + {"L": 0, "C": 1, "Y": 2}[name]]

    # ---------------------------------------------------------------- frames

    def outcome_columns(self) -> list[str]:
        if self.causes == 1:
            return [f"Y{k}" for k in range(1, self.n_times + 1)]
        return [f"{prefix}{k}" for prefix in ("R", "D") for k in range(1, self.n_times + 1)]

    def columns(self, *, weights: str | None = None) -> dict[str, Any]:
        """The keywords of ``LTMLE.fit`` for a frame of this law."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": "A",
            "baseline": ["W1", "W2"],
            "censoring": [f"C{k}" for k in nodes],
            "time_varying": [[] if k == 1 else [f"L{k}"] for k in nodes],
        }
        if self.causes == 2:
            out["outcome"] = {
                CAUSES[1]: [f"R{k}" for k in nodes],
                CAUSES[2]: [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        if weights is not None:
            out["weights"] = weights
        return out

    def _frame_of(self, rows: Sequence[tuple[Any, ...]]) -> pd.DataFrame:
        def column(getter: Callable[[tuple[Any, ...]], Any]) -> np.ndarray:
            return np.array([np.nan if getter(p) is None else float(getter(p)) for p in rows])

        out: dict[str, np.ndarray] = {
            "W1": column(lambda p: p[0]),
            "W2": column(lambda p: p[1]),
            "A": column(lambda p: p[2]),
        }
        for node in range(2, self.n_times + 1):
            out[f"L{node}"] = column(lambda p, k=node: self.value(p, "L", k))
        for node in range(1, self.n_times + 1):
            out[f"C{node}"] = column(lambda p, k=node: self.value(p, "C", k))
        codes = {
            node: column(lambda p, k=node: self.value(p, "Y", k))
            for node in range(1, self.n_times + 1)
        }
        _outcomes(out, codes, self.causes)
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
        g = np.array([self.g(int(x1), int(x2)) for x1, x2 in STRATA])
        stratum = w1 * 4 + w2
        cumulative = np.cumsum(g[stratum], axis=1)
        a = (rng.random(n)[:, None] > cumulative[:, :-1]).sum(axis=1)
        out: dict[str, np.ndarray] = {
            "W1": w1.astype(float),
            "W2": w2.astype(float),
            "A": a.astype(float),
        }
        open_ = np.ones(n, dtype=bool)
        previous = np.zeros(n, dtype=int)
        codes: dict[int, np.ndarray] = {}
        for node in range(1, self.n_times + 1):
            if node > 1:
                p = _vector(self.covariate, a, w1, previous)
                draw_l = np.where(rng.random(n) < p, 1, -1)
                out[f"L{node}"] = np.where(open_, draw_l, np.nan)
                current = np.where(open_, draw_l, 0)
            else:
                current = np.zeros(n, dtype=int)
            lost = rng.random(n) < _vector_logit(self.dropout, node, a, w1, w2, current)
            out[f"C{node}"] = np.where(open_, (~lost).astype(float), np.nan)
            followed = open_ & ~lost
            u = rng.random(n)
            code = np.zeros(n)
            taken = np.zeros(n)
            for index, factor in enumerate(self.hazards, start=1):
                share = _vector_logit(factor, node, a, w1, w2, current) * (1.0 - taken)
                code = np.where((code == 0) & (u >= taken) & (u < taken + share), index, code)
                taken = taken + share
            codes[node] = np.where(followed, code, np.nan)
            open_ = followed & (code == 0)
            previous = current
        columns = {f"L{k}": out.pop(f"L{k}") for k in range(2, self.n_times + 1)}
        censoring = {f"C{k}": out.pop(f"C{k}") for k in range(1, self.n_times + 1)}
        out.update(columns)
        out.update(censoring)
        _outcomes(out, codes, self.causes)
        return pd.DataFrame(out)

    # ---------------------------------------------------------------- the truth

    def q_bar(self, node: int, a: int, w1: int, w2: int, l: int | None, t: int, j: int) -> float:
        r""":math:`Q_k(W, a, L_k)`: the probability of a cause-``j`` event by node ``t``."""
        lam = self.cause_hazards(node, a, w1, w2, l)
        value = lam[j - 1]
        if node < t:
            p = self.covariate(a, w1, l)
            later = p * self.q_bar(node + 1, a, w1, w2, 1, t, j) + (1.0 - p) * self.q_bar(
                node + 1, a, w1, w2, -1, t, j
            )
            value += (1.0 - sum(lam)) * later
        return value

    def pushforward(self, regimen: Regimen, w1: int, w2: int) -> np.ndarray:
        """``g^d(a | W)``: the law of ``d(A)`` given ``W``."""
        g = self.g(w1, w2)
        out = np.zeros(self.arms)
        for a in range(self.arms):
            out[regimen(a)] += g[a]
        return out

    def truth(self, regimen: Regimen, t: int, j: int = 1) -> float:
        """``E Q_1(W, d(A); t, j)``."""
        total = 0.0
        for w1, w2 in STRATA:
            g = self.g(w1, w2)
            total += MASS * sum(
                g[a] * self.q_bar(1, regimen(a), w1, w2, None, t, j) for a in range(self.arms)
            )
        return total

    def eif(self, regimen: Regimen, t: int, j: int = 1) -> np.ndarray:
        """The efficient influence function of ``psi_d(t, j)`` at every support point."""
        psi = self.truth(regimen, t, j)
        return np.array([self._eif_at(point, regimen, t, j) for point in self.support]) - psi

    def _eif_at(self, point: tuple[Any, ...], regimen: Regimen, t: int, j: int) -> float:
        """The uncentred efficient influence function at one support point."""
        w1, w2, a = point[:3]
        value = self.q_bar(1, regimen(a), w1, w2, None, t, j)
        q = self.pushforward(regimen, w1, w2)
        weight = q[a] / self.g(w1, w2)[a]
        if weight == 0.0:
            return value
        l: int | None = None
        for node in range(1, t + 1):
            if node > 1:
                l = self.value(point, "L", node)
            c = self.value(point, "C", node)
            weight /= self.retention(node, a, w1, w2, l)
            if c == 0:
                break
            y = self.value(point, "Y", node)
            if y == 0 and node < t:
                z = self.q_bar(node + 1, a, w1, w2, self.value(point, "L", node + 1), t, j)
            else:
                z = 1.0 if y == j else 0.0
            value += weight * (z - self.q_bar(node, a, w1, w2, l, t, j))
            if y != 0:
                break
        return value

    def frame_curves(self, target: Target, frame: pd.DataFrame) -> np.ndarray:
        """The centred efficient influence function of ``target`` at each unit of ``frame``."""
        table = dict(zip(self.support, target.curve(self), strict=True))
        keys = _keys(self, frame)
        return np.array([table[key] for key in keys])


def _outcomes(out: dict[str, np.ndarray], codes: dict[int, np.ndarray], causes: int) -> None:
    for node, code in codes.items():
        if causes == 1:
            out[f"Y{node}"] = code
    if causes == 2:
        for node, code in codes.items():
            out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
        for node, code in codes.items():
            out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))


def _vector_logit(
    factor: Logit, node: int, a: np.ndarray, w1: np.ndarray, w2: np.ndarray, l: np.ndarray
) -> np.ndarray:
    eta = (
        factor.intercept
        + factor.trend * (node - 1)
        + np.asarray(factor.arm)[a]
        + factor.w1 * w1
        + factor.w2 * w2
        + factor.l * (l == 1)
    )
    return np.asarray(expit(eta))


def _vector(factor: Covariate, a: np.ndarray, w1: np.ndarray, previous: np.ndarray) -> np.ndarray:
    eta = factor.intercept + factor.w1 * w1 + np.asarray(factor.arm)[a] + factor.previous * (
        previous == 1
    )
    return np.asarray(expit(eta))


def _keys(law: Law, frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Each unit of ``frame`` as the support point it realises."""
    width = 3 + 3 * law.n_times

    def cell(value: Any) -> Any:
        return None if pd.isna(value) else int(value)

    out = []
    names = [*("W1", "W2", "A")]
    for node in range(1, law.n_times + 1):
        names += [f"L{node}" if node > 1 else "", f"C{node}"]
        names += ["" if law.causes == 2 else f"Y{node}"]
    records = frame.to_dict("records")
    for record in records:
        point: list[Any] = []
        for position, name in enumerate(names):
            node = (position - 3) // 3 + 1
            if position >= 3 and (position - 3) % 3 == 2 and law.causes == 2:
                relapse, death = record[f"R{node}"], record[f"D{node}"]
                point.append(
                    None if pd.isna(relapse) else 1 if relapse == 1 else 2 if death == 1 else 0
                )
            elif name == "":
                point.append(None)
            else:
                point.append(cell(record[name]))
        out.append(tuple(point) + (None,) * (width - len(point)))
    return out


# -------------------------------------------------------------------- regimens and targets


def static(level: int) -> Regimen:
    """The static regimen that sets every unit's treatment to ``level``."""
    return lambda a: level


def natural(a: int) -> int:
    """The natural course: the identity map."""
    return a


def up(a: int) -> int:
    """One level up, the top stays."""
    return min(a + 1, 2)


@dataclass(frozen=True)
class Target:
    """A linear combination of risks: ``(coefficient, regimen, horizon, cause)`` terms.

    Parameters
    ----------
    terms : tuple of tuple
        Each ``(coefficient, regimen, horizon, cause)``.
    """

    terms: tuple[tuple[float, Regimen, int, int], ...]

    def value(self, law: Law) -> float:
        return float(sum(c * law.truth(d, t, j) for c, d, t, j in self.terms))

    def curve(self, law: Law) -> np.ndarray:
        """The centred efficient influence function at every support point."""
        return np.asarray(sum(c * law.eif(d, t, j) for c, d, t, j in self.terms))


def efficiency_sd(law: Law, target: Target) -> float:
    """The standard deviation of the efficient influence function of ``target``."""
    curve = target.curve(law)
    return float(math.sqrt(float(np.sum(law.probs * curve**2))))


# -------------------------------------------------------------------- the wrong plug-in


def _at_risk_cells(law: Law, node: int) -> dict[tuple[int, int, int], tuple[float, float]]:
    """``(W1, W2, A) -> (mass followed at node k, mass of a cause-1 event there)``.

    The mass is that of the units followed and event-free at node ``k`` (uncensored through
    ``C_k``), with ``L_k`` summed out: the rows a node fit without ``L_k`` trains on.
    """
    out: dict[tuple[int, int, int], list[float]] = {}
    for point, mass in zip(law.support, law.probs, strict=True):
        if law.value(point, "C", node) != 1:
            continue
        key = (point[0], point[1], point[2])
        entry = out.setdefault(key, [0.0, 0.0])
        entry[0] += mass
        entry[1] += mass * float(law.value(point, "Y", node) == 1)
    return {key: (total, events) for key, (total, events) in out.items()}


def _logistic(rows: np.ndarray, events: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """The weighted logistic regression of ``events / totals`` on ``rows``, by Newton's method."""
    beta = np.zeros(rows.shape[1])
    for _ in range(100):
        p = expit(rows @ beta)
        score = rows.T @ (events - totals * p)
        hessian = (rows * (totals * p * (1.0 - p))[:, None]).T @ rows
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return beta


def wrong_hazards(law: Law) -> list[dict[tuple[int, int, int], float]]:
    """The limit of the node hazard fitted without ``L_k`` (one cause), by node and stratum.

    The model is the study's wrong learner: logistic regression on ``(W1, W2, A)``, with the
    levels of a three-level treatment as indicators.
    """
    if law.causes != 1:
        raise ValueError("the wrong plug-in limit is declared for one cause")
    fitted = []
    for node in range(1, law.n_times + 1):
        cells = _at_risk_cells(law, node)
        keys = sorted(cells)

        def row(key: tuple[int, int, int]) -> list[float]:
            w1, w2, a = key
            arms = [1.0 if a == level else 0.0 for level in range(1, law.arms)]
            return [1.0, float(w1), float(w2), *arms]

        rows = np.array([row(key) for key in keys])
        totals = np.array([cells[key][0] for key in keys])
        events = np.array([cells[key][1] for key in keys])
        beta = _logistic(rows, events, totals)
        fitted.append({key: float(expit(np.array(row(key)) @ beta)) for key in keys})
    return fitted


def wrong_plug_in_limit(law: Law, target: Target) -> float:
    """The limit of the plug-in whose hazard and continuation both omit ``L``.

    Both fits omit ``L``, so every node's fitted curve is a function of ``(W, A)``.  The
    continuation without ``L`` then reproduces the next node's curve exactly, and the node-1
    curve is the product of the node hazards fitted without ``L_k``:
    :math:`1 - \\prod_{k \\le t} (1 - \\tilde\\lambda_k(W, a))`, averaged over
    :math:`W` and over :math:`g^d`.
    """
    fitted = wrong_hazards(law)
    total = 0.0
    for coefficient, regimen, horizon, _ in target.terms:
        for w1, w2 in STRATA:
            q = law.pushforward(regimen, w1, w2)
            for a in range(law.arms):
                if q[a] == 0.0:
                    continue
                survival = 1.0
                for node in range(horizon):
                    survival *= 1.0 - fitted[node][(w1, w2, a)]
                total += coefficient * MASS * q[a] * (1.0 - survival)
    return total


def positivity(law: Law) -> dict[str, float]:
    """The smallest treatment probability and retention, and their smallest product.

    ``cumulative_min`` is the smallest :math:`g(a \\mid W) \\prod_{k \\le T} c_k` over every
    stratum, level and covariate path.
    """
    g_min = min(float(np.min(law.g(w1, w2))) for w1, w2 in STRATA)
    retained = 1.0
    cumulative = 1.0
    for w1, w2 in STRATA:
        g = law.g(w1, w2)
        for a in range(law.arms):
            for path in _paths(law.n_times):
                kept = 1.0
                for node, l in enumerate(path, start=1):
                    kept *= law.retention(node, a, w1, w2, l)
                retained = min(retained, kept)
                cumulative = min(cumulative, g[a] * kept)
    return {"g_min": g_min, "retained_min": retained, "cumulative_min": cumulative}


def _paths(n_times: int) -> list[tuple[int | None, ...]]:
    paths: list[tuple[int | None, ...]] = [(None,)]
    for _ in range(2, n_times + 1):
        paths = [(*path, value) for path in paths for value in (1, -1)]
    return paths


# -------------------------------------------------------------------- learners


def _l_columns(matrix: np.ndarray) -> np.ndarray:
    """The columns holding a negative entry: the ``L`` columns, coded ``-1`` and ``+1``."""
    return np.flatnonzero(np.any(matrix < 0.0, axis=0))


class _Columns(BaseEstimator):
    """A learner fitted on the columns :meth:`_kept` picks at fit time."""

    def __init__(self, base: Any) -> None:
        self.base = base

    def _kept(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Columns:
        matrix = np.asarray(X, dtype=float)
        self.keep_ = self._kept(matrix)
        self.model_ = clone(self.base)
        self.model_.fit(self._reduce(matrix), y, sample_weight=sample_weight)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        return self

    def _reduce(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        reduced = matrix[:, self.keep_]
        return reduced if reduced.shape[1] else np.ones((matrix.shape[0], 1))

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._reduce(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._reduce(X)))


class LastL(_Columns):
    """``base`` on every column but the earlier ``L`` columns: ``L_k`` is the last one kept.

    Parameters
    ----------
    base : estimator
        The learner fitted on the kept columns.
    """

    def _kept(self, matrix: np.ndarray) -> np.ndarray:
        dropped = _l_columns(matrix)[:-1]
        return np.setdiff1d(np.arange(matrix.shape[1]), dropped)


class NoL(_Columns):
    """``base`` on every column but the ``L`` columns.

    Parameters
    ----------
    base : estimator
        The learner fitted on the kept columns.
    """

    def _kept(self, matrix: np.ndarray) -> np.ndarray:
        return np.setdiff1d(np.arange(matrix.shape[1]), _l_columns(matrix))


class Logistic(BaseEstimator, ClassifierMixin):
    """An unpenalised logistic regression, additive in its design (multinomial on 3 classes)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Logistic:
        self.model_ = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(np.asarray(X, dtype=float)))


class CellMeans(BaseEstimator, RegressorMixin):
    """The weighted mean of ``y`` within each distinct design row; an unseen row reads the mean.

    Fitted on the columns a :class:`LastL` keeps, it is the cell mean over
    :math:`(W, A, L_k, s)`, the true regression of the future on these Markov laws.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeans:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        keys, inverse = np.unique(matrix, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        totals = np.bincount(inverse, weights=weights * target, minlength=keys.shape[0])
        sizes = np.bincount(inverse, weights=weights, minlength=keys.shape[0])
        self.keys_ = keys
        self.means_ = np.where(sizes > 0, totals / np.where(sizes > 0, sizes, 1.0), 0.0)
        total = float(np.sum(weights))
        self.default_ = float(np.sum(weights * target) / total) if total > 0 else 0.0
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        lookup = {tuple(key): mean for key, mean in zip(self.keys_, self.means_, strict=True)}
        return np.array([lookup.get(tuple(row), self.default_) for row in matrix])

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


def hazard_learner(correct: bool) -> _Columns:
    """Logistic on ``(W, A, L_k)`` when correct; on ``(W, A)`` when not."""
    return LastL(Logistic()) if correct else NoL(Logistic())


def continuation_learner(correct: bool) -> _Columns:
    """The cell mean over ``(W, A, L_k, s)`` when correct; over ``(W, A, s)`` when not."""
    return LastL(CellMeans()) if correct else NoL(CellMeans())


def censoring_learner(correct: bool) -> _Columns:
    """Logistic on ``(W, A, L_k)`` when correct; on ``(W, A)`` when not."""
    return LastL(Logistic()) if correct else NoL(Logistic())


# -------------------------------------------------------------------- the laws

#: The treatment of the binary laws: ``P(A = 1 | W) = expit(-0.4 + 0.7 W1 + 0.2 W2)``.
BINARY_TREATMENT = ((-0.4, 0.7, 0.2),)
#: The three-level treatment: levels 1 and 2 against 0.
THREE_LEVEL_TREATMENT = ((0.1, 0.4, -0.1), (-0.5, 0.8, 0.15))
COVARIATE = Covariate(-1.2, 0.6, (0.0, -2.5, -4.0), 3.5)
DROPOUT = Logit(-4.3, 0.0, (0.0, -0.6, -1.0), 0.15, 0.1, 3.4)
HAZARD = Logit(-4.0, 0.1, (0.0, -0.25, -0.5), 0.3, 0.2, 3.3)
SECOND_CAUSE = Logit(-2.8, 0.05, (0.0, 0.3, 0.5), -0.2, 0.15, 0.8)

ONE_CAUSE = Law("one_cause", 5, BINARY_TREATMENT, COVARIATE, DROPOUT, (HAZARD,))
COMPETING = Law("competing", 4, BINARY_TREATMENT, COVARIATE, DROPOUT, (HAZARD, SECOND_CAUSE))
THREE_ARM = Law("three_arm", 5, THREE_LEVEL_TREATMENT, COVARIATE, DROPOUT, (HAZARD,))

LAWS = {law.name: law for law in (ONE_CAUSE, COMPETING, THREE_ARM)}

__all__ = [
    "CAUSES",
    "LAWS",
    "STRATA",
    "CellMeans",
    "LastL",
    "Law",
    "Logistic",
    "NoL",
    "Target",
    "censoring_learner",
    "continuation_learner",
    "efficiency_sd",
    "hazard_learner",
    "natural",
    "positivity",
    "static",
    "up",
    "wrong_plug_in_limit",
]
