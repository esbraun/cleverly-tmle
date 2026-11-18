r"""A held baseline treatment with a binary covariate after baseline, enumerated exactly (X70 (b1)).

The implementation never enters this module.  Law (H) of the X70 plan, section 4.4:

``W1, W2, A, C1, Y1, L2, C2, Y2, L3, C3, Y3, L4, C4, Y4``

with ``W = (W1, W2)`` binary, ``A`` held over the nodes (two levels, or three in the
three-arm variant), ``L_k`` binary from ``(W1, A, L_{k-1})``, dropout from ``(A, W, L_k)`` and
the cause-specific hazards from ``(k, A, W, L_k)``.  ``L_k`` is coded ``-1`` and ``+1`` so a
test learner can tell its column apart from the 0/1 columns of a design (:class:`NoL`).

The law is passed as data: one row per support point, weighted by its probability, so an
estimate whose nuisances are right, or whose mechanism is right with a targeted plug-in,
equals the truth to floating tolerance.  The truth and the efficient influence function are
sums over the support computed from the tables here, with the iterated expectations
:math:`Q_k(W, a, L_k)` evaluated by recursion over the covariate paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, clone

from .discrete_law_longitudinal import CellMeans
from .discrete_law_longitudinal_multivalue import CellProbabilities

__all__ = ["CellMeans", "CellProbabilities", "HeldLaw", "MultiNodeLaw", "NoL"]

#: The cause labels of the two-cause variant, by event code.
CAUSES = {1: "relapse", 2: "death"}


class NoL(BaseEstimator):
    """A learner that drops every column holding a negative entry, which is each ``L`` column.

    ``NoL(CellMeans())`` is a cell mean over the design without ``L_k``, and ``NoL(Logistic())``
    a logistic regression without it: both are wrong on a law whose hazard reads ``L_k``.

    Parameters
    ----------
    base : estimator
        The learner fitted on the kept columns.
    """

    def __init__(self, base: Any) -> None:
        self.base = base

    @staticmethod
    def _kept(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(~np.any(matrix < 0.0, axis=0))

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NoL:
        matrix = np.asarray(X, dtype=float)
        self.keep_ = self._kept(matrix)
        self.model_ = clone(self.base)
        reduced = matrix[:, self.keep_]
        if reduced.shape[1] == 0:
            reduced = np.ones((matrix.shape[0], 1))
        self.model_.fit(reduced, y, sample_weight=sample_weight)
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


def _flag(value: int | None) -> float:
    """``1`` where ``L = +1``, else ``0`` (``L_1`` is not measured)."""
    return 1.0 if value == 1 else 0.0


@dataclass(frozen=True)
class HeldLaw:
    """Law (H): a held treatment, a binary covariate after baseline, dropout and an event.

    Parameters
    ----------
    n_times : int
        The nodes ``T``.
    causes : int
        One cause, or two coded ``R`` and ``D`` in the frame.
    arms : int
        The treatment levels, two or three.
    censored : bool
        Whether units drop out.  Without dropout every retention is one.
    l_moves_hazard : bool
        Whether ``L_k`` enters the hazard.  ``False`` gives the control in which no later
        hazard reads a covariate after baseline (dropout still does).
    last_only : bool
        Whether events happen at the last node only.
    no_late_second : bool
        Whether cause 2 cannot occur after node 2 (two causes only).
    flat : int or None
        A node whose hazards are scaled near zero, for the coherence witness.
    follow : Callable or None
        ``(w1, w2) -> level``: a deterministic treatment assignment, for the observed-slot
        control in which every unit follows the one-hot policy.
    """

    n_times: int = 4
    causes: int = 1
    arms: int = 2
    censored: bool = True
    l_moves_hazard: bool = True
    last_only: bool = False
    no_late_second: bool = False
    flat: int | None = None
    follow: Callable[[int, int], int] | None = None
    support: tuple[tuple[Any, ...], ...] = field(init=False, repr=False)
    probs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w1 in (0, 1):
            for w2 in (0, 1):
                pw = self.p_w(w1, w2)
                for a in range(self.arms):
                    mass = pw * self.g(a, w1, w2)
                    if mass > 0.0:
                        self._walk((w1, w2, a), mass, 1, None, points)
        object.__setattr__(self, "support", tuple(point for point, _ in points))
        object.__setattr__(self, "probs", np.array([mass for _, mass in points]))

    # --------------------------------------------------------------- the tables

    @staticmethod
    def p_w(w1: int, w2: int) -> float:
        """``P(W1 = w1, W2 = w2)``."""
        p1 = 0.4 if w1 else 0.6
        p2 = (0.55 if w2 else 0.45) if w1 else (0.3 if w2 else 0.7)
        return p1 * p2

    def g(self, a: int, w1: int, w2: int) -> float:
        """``P(A = a | W)``."""
        if self.follow is not None:
            return 1.0 if a == self.follow(w1, w2) else 0.0
        if self.arms == 2:
            p = 0.3 + 0.2 * w1 + 0.25 * w2
            return p if a == 1 else 1.0 - p
        table = (0.5 - 0.2 * w1, 0.3, 0.2 + 0.2 * w1)
        return table[a]

    @staticmethod
    def p_l(w1: int, a: int, previous: int | None) -> float:
        """``P(L_k = +1 | W1, A, L_{k-1})``."""
        return float(expit(-0.3 + 0.8 * w1 - 0.7 * a + 1.0 * _flag(previous)))

    def retention(self, node: int, w1: int, w2: int, a: int, l: int | None) -> float:
        """``P(C_k = 1 | W, A, L_k)``."""
        if not self.censored:
            return 1.0
        return float(expit(1.8 - 0.4 * w2 + 0.3 * a - 1.2 * _flag(l) + 0.1 * node))

    def hazards(self, node: int, w1: int, w2: int, a: int, l: int | None) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given at risk."""
        if self.last_only and node < self.n_times:
            return (0.0,) * self.causes
        reads = _flag(l) if self.l_moves_hazard else 0.0
        total = float(expit(-1.6 + 0.2 * node - 0.6 * a + 0.5 * w1 + 1.1 * reads + 0.4 * a * reads))
        if self.flat is not None and node == self.flat:
            total *= 1e-3
        if self.causes == 1:
            return (total,)
        share = float(expit(0.4 - 0.8 * a + 0.6 * w2 - 0.9 * reads))
        if self.no_late_second and node > 2:
            share = 1.0
        return (total * share, total * (1.0 - share))

    # ---------------------------------------------------------------- the support

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        l: int | None,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        width = 3 + 3 * self.n_times
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w1, _, a = prefix[:3]
        if node > 1:
            p = self.p_l(w1, a, l)
            for value, share in ((1, p), (-1, 1.0 - p)):
                if share > 0.0:
                    self._enter((*prefix, value), mass * share, node, value, out, width)
            return
        self._enter((*prefix, None), mass, node, None, out, width)

    def _enter(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        l: int | None,
        out: list[tuple[tuple[Any, ...], float]],
        width: int,
    ) -> None:
        w1, w2, a = prefix[:3]
        c = self.retention(node, w1, w2, a, l)
        if c < 1.0:
            point = (*prefix, 0)
            out.append((point + (None,) * (width - len(point)), mass * (1.0 - c)))
        lam = self.hazards(node, w1, w2, a, l)
        for code, share in enumerate(lam, start=1):
            if share > 0.0:
                point = (*prefix, 1, code)
                out.append((point + (None,) * (width - len(point)), mass * c * share))
        self._walk((*prefix, 1, 0), mass * c * (1.0 - sum(lam)), node + 1, l, out)

    # ----------------------------------------------------------------- the frame

    def value(self, point: tuple[Any, ...], name: str, node: int) -> Any:
        """The value of ``L``, ``C`` or ``Y`` at ``node`` in a support point."""
        offset = {"L": 0, "C": 1, "Y": 2}[name]
        return point[3 + 3 * (node - 1) + offset]

    def frame(self) -> pd.DataFrame:
        """The law as data: one row per support point, its probability in ``wt``."""

        def column(getter: Callable[[tuple[Any, ...]], Any]) -> np.ndarray:
            return np.array(
                [np.nan if getter(p) is None else float(getter(p)) for p in self.support]
            )

        out: dict[str, np.ndarray] = {
            "W1": column(lambda p: p[0]),
            "W2": column(lambda p: p[1]),
            "A": column(lambda p: p[2]),
        }
        for node in range(1, self.n_times + 1):
            if node > 1:
                out[f"L{node}"] = column(lambda p, k=node: self.value(p, "L", k))
            out[f"C{node}"] = column(lambda p, k=node: self.value(p, "C", k))
            code = column(lambda p, k=node: self.value(p, "Y", k))
            if self.causes == 2:
                out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
            else:
                out[f"Y{node}"] = code
        out["wt"] = self.probs * 1000.0
        return pd.DataFrame(out)

    def columns(self, *, l1: bool = False) -> dict[str, Any]:
        """The ``fit`` keywords; ``l1`` declares ``W2`` as a node-1 block instead of baseline."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": "A",
            "baseline": ["W1"] if l1 else ["W1", "W2"],
            "censoring": [f"C{k}" for k in nodes],
            "time_varying": [(["W2"] if l1 else []) if k == 1 else [f"L{k}"] for k in nodes],
            "weights": "wt",
        }
        if self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    # ----------------------------------------------------------------- the truth

    def q_bar(self, node: int, w1: int, w2: int, a: int, l: int | None, t: int, j: int) -> float:
        r""":math:`\bar Q_k(W, a, L_k)` for cause ``j`` (one-based) at horizon ``t``."""
        lam = self.hazards(node, w1, w2, a, l)
        value = lam[j - 1]
        if node < t:
            p = self.p_l(w1, a, l)
            later = p * self.q_bar(node + 1, w1, w2, a, 1, t, j) + (1.0 - p) * self.q_bar(
                node + 1, w1, w2, a, -1, t, j
            )
            value += (1.0 - sum(lam)) * later
        return value

    def truth(self, density: Callable[[int, int], np.ndarray], t: int, j: int = 1) -> float:
        """``E sum_a q(a | W) Q_1(W, a)`` for the node-1 density ``density(w1, w2)``."""
        psi = 0.0
        for w1 in (0, 1):
            for w2 in (0, 1):
                q = density(w1, w2)
                psi += self.p_w(w1, w2) * sum(
                    q[a] * self.q_bar(1, w1, w2, a, None, t, j) for a in range(self.arms)
                )
        return psi

    def eif(self, density: Callable[[int, int], np.ndarray], t: int, j: int = 1) -> np.ndarray:
        """The efficient influence function at every support point, for a node-1 density."""
        psi = self.truth(density, t, j)
        out = np.empty(len(self.support))
        for row, point in enumerate(self.support):
            w1, w2, a = point[:3]
            q = density(w1, w2)
            value = sum(q[b] * self.q_bar(1, w1, w2, b, None, t, j) for b in range(self.arms)) - psi
            weight = q[a] / self.g(a, w1, w2) if self.g(a, w1, w2) > 0 else 0.0
            l: int | None = None
            for node in range(1, t + 1):
                if node > 1:
                    l = self.value(point, "L", node)
                    if l is None:
                        break
                c = self.value(point, "C", node)
                if c is None:
                    break
                weight /= self.retention(node, w1, w2, a, l)
                if c == 0:
                    break
                y = self.value(point, "Y", node)
                if y == 0 and node < t:
                    nxt = self.value(point, "L", node + 1)
                    z = self.q_bar(node + 1, w1, w2, a, nxt, t, j)
                else:
                    z = 1.0 if y == j else 0.0
                value += weight * (z - self.q_bar(node, w1, w2, a, l, t, j))
                if y != 0:
                    break
            out[row] = value
        return out

    def true_retention(self, frame: pd.DataFrame, node: int) -> np.ndarray:
        """``(n,)`` ``P(C_k = 1 | W, A, L_k)`` at each row's observed history, one off it."""
        w1 = frame["W1"].to_numpy(dtype=int)
        w2 = frame["W2"].to_numpy(dtype=int)
        a = frame["A"].to_numpy(dtype=int)
        l = (
            np.nan_to_num(frame[f"L{node}"].to_numpy(dtype=float), nan=0.0)
            if node > 1
            else np.zeros(len(frame))
        )
        return np.array(
            [
                self.retention(node, w1[i], w2[i], a[i], int(l[i]) if l[i] != 0 else None)
                for i in range(len(frame))
            ]
        )


# ------------------------------------------------------------------------ law (M)


def _rule_level(node: int, w1: int, l: int | None) -> int:
    """The rule "treat when ``L_k = +1``" (node 1: when ``W1 = 1``)."""
    return w1 if node == 1 else int(l == 1)


@dataclass(frozen=True)
class MultiNodeLaw:
    r"""Law (M): a treatment at every node, a binary covariate after baseline, dropout and an event.

    ``W1, W2, A1, C1, Y1, L2, A2, C2, Y2, L3, A3, C3, Y3`` with ``W = (W1, W2)`` binary,
    ``A_k`` from ``(W, A_{k-1}, L_k)`` (node 1 from ``W``), ``L_k \in \{-1, +1\}`` at ``k \ge 2``
    from ``(W1, A_{k-1}, L_{k-1})``, dropout from ``(W, A_k, L_k)`` and the hazards from
    ``(k, W, A_{k-1}, A_k, L_k)``.  Each factor at node ``k`` reads ``L_k`` and ``A_{k-1}``
    only, so the law is order-0 Markov in the sense of ``markov_order=0``.

    Parameters
    ----------
    n_times : int
        The nodes ``T``.
    causes : int
        One cause, or two coded ``R`` and ``D`` in the frame.
    levels : tuple of int
        The level count of each node's treatment.
    censored : bool
        Whether units drop out.
    l_moves : bool
        Whether ``L_k`` enters the hazards and the treatment.  ``False`` leaves it in the
        dropout and its own law only: the control in which the covariate term vanishes.
    prefix_hazard : bool
        Whether the hazard reads ``A_{k-1}``.
    policy_prefix : bool
        Whether the known policy reads ``A_{k-1}``.
    copies : frozenset of int
        The nodes whose treatment copies ``A_1``.
    follow : bool
        Whether every unit follows the rule at every node.
    last_only : bool
        Whether events happen at the last node only.
    no_event : frozenset of tuple
        ``(node, level)`` pairs under which no event occurs at that node.
    """

    n_times: int = 3
    causes: int = 1
    levels: tuple[int, ...] = (2, 2, 2)
    censored: bool = True
    l_moves: bool = True
    prefix_hazard: bool = True
    policy_prefix: bool = True
    copies: frozenset[int] = frozenset()
    follow: bool = False
    last_only: bool = False
    no_event: frozenset[tuple[int, int]] = frozenset()
    support: tuple[tuple[Any, ...], ...] = field(init=False, repr=False)
    probs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w1 in (0, 1):
            for w2 in (0, 1):
                self._node((w1, w2), self.p_w(w1, w2), 1, points)
        object.__setattr__(self, "support", tuple(point for point, _ in points))
        object.__setattr__(self, "probs", np.array([mass for _, mass in points]))

    # --------------------------------------------------------------- the tables

    p_w = staticmethod(HeldLaw.p_w)

    def g(self, node: int, history: tuple[Any, ...]) -> np.ndarray:
        """``P(A_k = . | H_k)`` over node ``k``'s levels; ``history`` ends with ``L_k``."""
        w1, w2 = history[:2]
        arms, covariates = self.arms(history), self.covariates(history)
        l = covariates[-1]
        width = self.levels[node - 1]
        if self.follow:
            return np.eye(width)[_rule_level(node, w1, l)]
        if node in self.copies:
            return np.eye(width)[arms[0]]
        reads = _flag(l) if self.l_moves else 0.0
        if node == 1:
            lin = -0.2 + 0.6 * w1 - 0.4 * w2
        else:
            lin = -0.3 + 0.5 * w1 + 0.8 * arms[-1] + 0.7 * reads - 0.3
        if width == 2:
            p = float(expit(lin))
            return np.array([1.0 - p, p])
        raw = np.exp(np.array([0.0, 0.3 + 0.5 * reads, -0.2 + 0.3 * w1 + 0.4 * reads]))
        return raw / raw.sum()

    @staticmethod
    def p_l(w1: int, previous_arm: int, previous: int | None) -> float:
        """``P(L_k = +1 | W1, A_{k-1}, L_{k-1})``."""
        return float(expit(-0.3 + 0.8 * w1 - 0.7 * previous_arm + 1.0 * _flag(previous)))

    def retention(self, node: int, w1: int, w2: int, a: int, l: int | None) -> float:
        """``P(C_k = 1 | W, A_k, L_k)``."""
        if not self.censored:
            return 1.0
        return float(expit(1.8 - 0.4 * w2 + 0.3 * a - 1.2 * _flag(l) + 0.1 * node))

    def hazards(
        self, node: int, w1: int, w2: int, previous_arm: int, a: int, l: int | None
    ) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given at risk."""
        if (self.last_only and node < self.n_times) or (node, a) in self.no_event:
            return (0.0,) * self.causes
        reads = _flag(l) if self.l_moves else 0.0
        # 1.2 sizes the policy-prefix witness: at 0.8 the prefix term and the -0.6 A_k term
        # nearly cancel downstream, and the mutation misses by 9.9e-4, under the 1e-3 floor.
        prefix = 1.2 * previous_arm if (self.prefix_hazard and node > 1) else 0.0
        total = float(
            expit(-1.6 + 0.2 * node - 0.6 * a + 0.5 * w1 + 1.1 * reads + 0.4 * a * reads + prefix)
        )
        if self.causes == 1:
            return (total,)
        share = float(expit(0.4 - 0.8 * a + 0.6 * w2 - 0.9 * reads))
        return (total * share, total * (1.0 - share))

    # ------------------------------------------------------------ the regimens

    def policy(self, node: int, history: tuple[Any, ...]) -> np.ndarray:
        r"""The known policy: :math:`0.3 + 0.4 [W_1 = 1]` at node 1, then
        :math:`0.2 + 0.4 [L_k = +1] + 0.2 A_{k-1}` (``policy_prefix``) or
        :math:`0.3 + 0.4 [L_k = +1]`; over three levels, a fixed table moved by ``L_k``."""
        w1 = history[0]
        arms, l = self.arms(history), self.covariates(history)[-1]
        width = self.levels[node - 1]
        if node == 1:
            p = 0.3 + 0.4 * w1
        elif self.policy_prefix:
            p = 0.2 + 0.4 * _flag(l) + 0.2 * min(arms[-1], 1)
        else:
            p = 0.3 + 0.4 * _flag(l)
        if width == 2:
            return np.array([1.0 - p, p])
        return np.array([0.2 + 0.2 * _flag(l), 0.3, 0.5 - 0.2 * _flag(l)])

    def rule(self, node: int, history: tuple[Any, ...]) -> np.ndarray:
        """The rule as a one-hot density."""
        level = _rule_level(node, history[0], self.covariates(history)[-1])
        return np.eye(self.levels[node - 1])[level]

    def static(self, level: int) -> Callable[[int, tuple[Any, ...]], np.ndarray]:
        """The static arm ``level`` at every node."""
        return lambda node, history: np.eye(self.levels[node - 1])[level]

    # ---------------------------------------------------------------- histories

    @staticmethod
    def arms(history: tuple[Any, ...]) -> tuple[int, ...]:
        """The treatments in a history ``(W1, W2, L1, A1, L2, A2, ...)``."""
        return tuple(history[3::2])

    @staticmethod
    def covariates(history: tuple[Any, ...]) -> tuple[int | None, ...]:
        """The covariates in a history, ``L_1`` (``None``) first."""
        return tuple(history[2::2])

    # ---------------------------------------------------------------- the support

    def _node(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        """Enumerate from node ``node`` on; ``prefix`` is the path through node ``k - 1``.

        A support point is ``(W1, W2)`` followed, per node, by ``(L_k, A_k, C_k, Y_k)``.
        """
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w1, w2 = prefix[:2]
        nodes = [prefix[2 + 4 * s : 6 + 4 * s] for s in range(node - 1)]
        arms = [entry[1] for entry in nodes]
        previous_l = nodes[-1][0] if nodes else None
        if node == 1:
            branches: list[tuple[int | None, float]] = [(None, 1.0)]
        else:
            p = self.p_l(w1, arms[-1], previous_l)
            branches = [(1, p), (-1, 1.0 - p)]
        width = 2 + 4 * self.n_times
        for l, share in branches:
            if share <= 0.0:
                continue
            history = self._history(prefix, node, l)
            for a, ga in enumerate(self.g(node, history)):
                if ga <= 0.0:
                    continue
                base = mass * share * ga
                c = self.retention(node, w1, w2, a, l)
                if c < 1.0:
                    point = (*prefix, l, a, 0)
                    out.append((point + (None,) * (width - len(point)), base * (1.0 - c)))
                lam = self.hazards(node, w1, w2, arms[-1] if arms else 0, a, l)
                for code, hazard in enumerate(lam, start=1):
                    if hazard > 0.0:
                        point = (*prefix, l, a, 1, code)
                        out.append((point + (None,) * (width - len(point)), base * c * hazard))
                self._node((*prefix, l, a, 1, 0), base * c * (1.0 - sum(lam)), node + 1, out)

    def _history(self, prefix: tuple[Any, ...], node: int, l: int | None) -> tuple[Any, ...]:
        """``H_k = (W1, W2, L_1, A_1, ..., A_{k-1}, L_k)`` from a support prefix."""
        out: list[Any] = list(prefix[:2])
        for s in range(node - 1):
            entry = prefix[2 + 4 * s : 6 + 4 * s]
            out.extend([entry[0], entry[1]])
        out.append(l)
        return tuple(out)

    def value(self, point: tuple[Any, ...], name: str, node: int) -> Any:
        """The value of ``L``, ``A``, ``C`` or ``Y`` at ``node`` in a support point."""
        return point[2 + 4 * (node - 1) + {"L": 0, "A": 1, "C": 2, "Y": 3}[name]]

    # ----------------------------------------------------------------- the frame

    def frame(self) -> pd.DataFrame:
        """The law as data: one row per support point, its probability in ``wt``."""

        def column(getter: Callable[[tuple[Any, ...]], Any]) -> np.ndarray:
            return np.array(
                [np.nan if getter(p) is None else float(getter(p)) for p in self.support]
            )

        out: dict[str, np.ndarray] = {
            "W1": column(lambda p: p[0]),
            "W2": column(lambda p: p[1]),
        }
        for node in range(1, self.n_times + 1):
            if node > 1:
                out[f"L{node}"] = column(lambda p, k=node: self.value(p, "L", k))
            out[f"A{node}"] = column(lambda p, k=node: self.value(p, "A", k))
            out[f"C{node}"] = column(lambda p, k=node: self.value(p, "C", k))
            code = column(lambda p, k=node: self.value(p, "Y", k))
            if self.causes == 2:
                out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
            else:
                out[f"Y{node}"] = code
        out["wt"] = self.probs * 1000.0
        return pd.DataFrame(out)

    def columns(self, *, l1: bool = False) -> dict[str, Any]:
        """The ``fit`` keywords; ``l1`` declares ``W2`` as a node-1 block and drops ``L_k``."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": [f"A{k}" for k in nodes],
            "baseline": ["W1"] if l1 else ["W1", "W2"],
            "censoring": [f"C{k}" for k in nodes],
            "time_varying": [
                (["W2"] if l1 else []) if k == 1 else ([] if l1 else [f"L{k}"]) for k in nodes
            ],
            "weights": "wt",
        }
        if self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    # ----------------------------------------------------------------- the truth

    def q(
        self,
        density: Callable[[int, tuple[Any, ...]], np.ndarray],
        node: int,
        history: tuple[Any, ...],
        t: int,
        j: int,
    ) -> float:
        """:math:`Q^r_k(H_k)` for cause ``j`` (one-based) at horizon ``t``."""
        q = density(node, history)
        return float(
            sum(
                q[a] * self.q_bar(density, node, history, a, t, j)
                for a in range(len(q))
                if q[a] > 0.0
            )
        )

    def q_bar(
        self,
        density: Callable[[int, tuple[Any, ...]], np.ndarray],
        node: int,
        history: tuple[Any, ...],
        a: int,
        t: int,
        j: int,
    ) -> float:
        r""":math:`\bar Q^r_k(H_k, a)`, the regression of the node's iterated outcome."""
        w1, w2 = history[:2]
        arms, l = self.arms(history), self.covariates(history)[-1]
        lam = self.hazards(node, w1, w2, arms[-1] if arms else 0, a, l)
        value = lam[j - 1]
        if node < t:
            p = self.p_l(w1, a, l)
            later = p * self.q(density, node + 1, (*history, a, 1), t, j) + (1.0 - p) * self.q(
                density, node + 1, (*history, a, -1), t, j
            )
            value += (1.0 - sum(lam)) * later
        return float(value)

    def truth(
        self, density: Callable[[int, tuple[Any, ...]], np.ndarray], t: int, j: int = 1
    ) -> float:
        """:math:`\\psi_{r,j}(t) = E\\,Q^r_1(H_1)` by the backward recursion."""
        return float(
            sum(
                self.p_w(w1, w2) * self.q(density, 1, (w1, w2, None), t, j)
                for w1 in (0, 1)
                for w2 in (0, 1)
            )
        )

    def eif(
        self, density: Callable[[int, tuple[Any, ...]], np.ndarray], t: int, j: int = 1
    ) -> np.ndarray:
        """The efficient influence function at every support point."""
        psi = self.truth(density, t, j)
        out = np.empty(len(self.support))
        for row, point in enumerate(self.support):
            w1, w2 = point[:2]
            history: tuple[Any, ...] = (w1, w2, None)
            value = self.q(density, 1, history, t, j) - psi
            weight = 1.0
            for node in range(1, t + 1):
                if node > 1:
                    l = self.value(point, "L", node)
                    if l is None:
                        break
                    history = (*history, l)
                a = self.value(point, "A", node)
                c = self.value(point, "C", node)
                if a is None or c is None:
                    break
                ga = self.g(node, history)[a]
                weight *= density(node, history)[a] / ga
                weight /= self.retention(node, w1, w2, a, self.covariates(history)[-1])
                if c == 0 or weight == 0.0:
                    break
                y = self.value(point, "Y", node)
                if y == 0 and node < t:
                    z = self.q(
                        density, node + 1, (*history, a, self.value(point, "L", node + 1)), t, j
                    )
                else:
                    z = 1.0 if y == j else 0.0
                value += weight * (z - self.q_bar(density, node, history, a, t, j))
                if y != 0:
                    break
                history = (*history, a)
            out[row] = value
        return out

    def true_g(self, frame: pd.DataFrame, node: int) -> np.ndarray:
        """``(n, K_k)`` the treatment probabilities at each row's observed history."""
        return np.array(
            [self.g(node, self._row_history(frame, i, node)) for i in range(len(frame))]
        )

    def true_retention(self, frame: pd.DataFrame, node: int) -> np.ndarray:
        """``(n,)`` ``P(C_k = 1 | W, A_k, L_k)`` at each row's observed history, one off it."""
        out = np.ones(len(frame))
        for i in range(len(frame)):
            a = frame[f"A{node}"].iloc[i]
            if np.isnan(a):
                continue
            history = self._row_history(frame, i, node)
            out[i] = self.retention(
                node, history[0], history[1], int(a), self.covariates(history)[-1]
            )
        return out

    def _row_history(self, frame: pd.DataFrame, i: int, node: int) -> tuple[Any, ...]:
        """``H_k`` of row ``i``, with a missing entry filled by 0 (a row no fit reads)."""
        row = frame.iloc[i]
        out: list[Any] = [int(row["W1"]), int(row["W2"]), None]
        for s in range(1, node):
            out.append(int(np.nan_to_num(row[f"A{s}"])))
            value = row[f"L{s + 1}"]
            out.append(None if np.isnan(value) else int(value))
        return tuple(out)
