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

__all__ = ["CellMeans", "CellProbabilities", "HeldLaw", "NoL"]

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
