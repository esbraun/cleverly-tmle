r"""Point-treatment laws with a held baseline treatment that a finite sample realises exactly.

The implementation never enters this module.  Each law is a finite support with every
conditional probability a multiple of one quarter, so ``N`` rows laid out in the cell
proportions make the empirical law *equal* to the law.  The parameter is the g-formula
written as quotients of cell masses, and complex-step derivatives of it give the efficient
influence function, with no clever covariate and no mechanism product.

Two layouts share the machinery.

==================  =======================================================================
law                 nodes
==================  =======================================================================
survival            ``W, A, C1, Y1, [L2], C2, Y2, C3, Y3``; one cause, or two causes coded
                    ``1`` (relapse) and ``2`` (death) in the event node
end of study        ``W, A, C1, [L2], C2, Y``
==================  =======================================================================

``A`` takes three levels.  It is decided once, at baseline, and held: a fit declares it as
``treatment="A"``.  The wide frame also carries ``A1, A2, A3`` (``A1, A2`` at the end of
study), copies of ``A`` on the rows still in the study, so that the same law can be fitted
the shipped way, with one treatment column per node.

Every censoring and event probability depends on ``(W, A)`` and on ``L2`` where present,
so a fit that dropped a censoring factor or mixed up two arms misses the truth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .discrete_law_longitudinal import CellMeans
from .discrete_law_longitudinal_multivalue import CellProbabilities

__all__ = ["CellMeans", "CellProbabilities", "PointLaw", "end_of_study_law", "survival_law"]

_QUARTERS = (0.25, 0.5, 0.75)

#: ``P(A = a | W = w)``, indexed ``[w, a]``.
G = np.array([[0.25, 0.25, 0.50], [0.50, 0.25, 0.25]])

#: The labels of the two causes, by event code.
CAUSE_LABELS = {1: "relapse", 2: "death"}


def _retained(node: int, w: int, a: int, l: int) -> float:
    """``P(C_k = 1 | W, A, L)``."""
    return 0.75 - 0.25 * ((w + a + node + l) % 2)


#: The one-cause hazard by node, indexed ``[w][a]``.  A table and not a cyclic formula: a
#: hazard that runs through the same three values at every arm, in another order, gives
#: every arm the same risk at the last node, a parameter no bug can move.
HAZARD = (
    ((0.25, 0.50, 0.50), (0.25, 0.25, 0.50), (0.50, 0.25, 0.25)),
    ((0.50, 0.50, 0.75), (0.25, 0.50, 0.25), (0.50, 0.75, 0.50)),
)

#: The two-cause split by node, ``(P(cause 1), P(cause 2))``, indexed ``[w][a]``.
_PAIRS = ((0.25, 0.25), (0.50, 0.25), (0.25, 0.50))
SPLIT = (
    ((0, 1, 1), (2, 0, 1), (1, 2, 2)),
    ((1, 0, 2), (0, 0, 1), (2, 1, 0)),
)


def _hazard(node: int, w: int, a: int, l: int) -> float:
    """``P(Y_k = 1 | at risk, W, A, L)`` with one cause.  ``L2 = 1`` adds a quarter."""
    return min(HAZARD[w][a][node - 1] + 0.25 * l, 0.75)


def _split(node: int, w: int, a: int, l: int) -> tuple[float, float]:
    """``(P(cause 1), P(cause 2))`` at node ``k`` given at risk, with two causes."""
    return _PAIRS[(SPLIT[w][a][node - 1] + l) % 3]


def _covariate(w: int, a: int) -> float:
    """``P(L2 = 1 | W, A, still in the study)``."""
    return _QUARTERS[(w + a) % 3]


def _outcome(w: int, a: int, l: int) -> float:
    """``E[Y | W, A, L2, uncensored]`` at the end of the study."""
    return _QUARTERS[(w + 2 * a + l) % 3]


@dataclass(frozen=True)
class PointLaw:
    """One finite law, its support, its exact sample and its g-formula.

    Parameters
    ----------
    kind : {"survival", "end_of_study"}
        The outcome layout.
    n_times : int
        The number of nodes.
    causes : int
        One or two causes, on a survival law.
    with_l2 : bool
        Whether ``L2`` sits before node 2.
    censor_first : bool
        Whether units can be censored at node 1.  A time-to-event input cannot hold such a
        unit, since its censoring time would be ``g_0 = 0``.
    """

    kind: str
    n_times: int = 3
    causes: int = 1
    with_l2: bool = False
    censor_first: bool = True
    support: tuple[tuple[Any, ...], ...] = field(init=False)
    masses: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    n: int = field(init=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w in (0, 1):
            for a in range(3):
                self._walk((w, a), 0.5 * G[w, a], 1, 0, points)
        support = tuple(point for point, _ in points)
        masses = np.array([mass for _, mass in points])
        depth = max(len([v for v in point if v is not None]) for point in support)
        n = 2 * 4 ** (depth - 1)
        values = masses * n
        counts = np.rint(values)
        if np.max(np.abs(values - counts)) > 1e-9:  # pragma: no cover - guards the tables
            raise AssertionError("cell probabilities are not multiples of 1/N")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "counts", counts.astype(int))
        object.__setattr__(self, "n", int(n))

    # ------------------------------------------------------------------ support

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        l: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        """Every history after ``prefix``, with its probability, ``None`` past the exit."""
        if node > self.n_times:
            if self.kind == "end_of_study":
                w, a = prefix[0], prefix[1]
                y = _outcome(w, a, l)
                out.append(((*prefix, 1), mass * y))
                out.append(((*prefix, 0), mass * (1.0 - y)))
            else:
                out.append((prefix, mass))
            return
        w, a = prefix[0], prefix[1]
        if self.with_l2 and node == 2:
            p = _covariate(w, a)
            for value, share in ((1, p), (0, 1.0 - p)):
                self._enter((*prefix, value), mass * share, node, value, out)
            return
        self._enter(prefix, mass, node, l, out)

    def _enter(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        l: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        w, a = prefix[0], prefix[1]
        width = len(self.columns)
        c = 1.0 if node == 1 and not self.censor_first else _retained(node, w, a, l)
        if c < 1.0:
            out.append(((*prefix, 0) + (None,) * (width - len(prefix) - 1), mass * (1.0 - c)))
        stayed = (*prefix, 1)
        if self.kind == "end_of_study":
            self._walk(stayed, mass * c, node + 1, l, out)
            return
        if self.causes == 2:
            first, second = _split(node, w, a, l)
            branches = ((1, first), (2, second), (0, 1.0 - first - second))
        else:
            h = _hazard(node, w, a, l)
            branches = ((1, h), (0, 1.0 - h))
        for code, share in branches:
            point = (*stayed, code)
            if code:
                out.append((point + (None,) * (width - len(point)), mass * c * share))
            else:
                self._walk(point, mass * c * share, node + 1, l, out)

    @property
    def columns(self) -> tuple[str, ...]:
        """The support's node names, in time order."""
        out = ["W", "A"]
        for node in range(1, self.n_times + 1):
            if self.with_l2 and node == 2:
                out.append("L2")
            out.append(f"C{node}")
            if self.kind == "survival":
                out.append(f"E{node}")
        if self.kind == "end_of_study":
            out.append("Y")
        return tuple(out)

    @property
    def probs(self) -> np.ndarray:
        """``P`` over the support, from the counts, so it is the sample's empirical law."""
        return self.counts / self.n

    # ------------------------------------------------------------------- sample

    def frame(self) -> pd.DataFrame:
        """The ``N``-row sample, with the event node split into fit columns."""
        cells = np.repeat(np.arange(len(self.support)), self.counts)
        raw = {
            name: np.array(
                [np.nan if point[i] is None else float(point[i]) for point in self.support]
            )[cells]
            for i, name in enumerate(self.columns)
        }
        frame: dict[str, np.ndarray] = {"W": raw["W"], "A": raw["A"]}
        alive = np.ones(cells.size, dtype=bool)
        for node in range(1, self.n_times + 1):
            frame[f"A{node}"] = np.where(alive, raw["A"], np.nan)
            if self.with_l2 and node == 2:
                frame["L2"] = raw["L2"]
            frame[f"C{node}"] = raw[f"C{node}"]
            if self.kind == "survival":
                code = raw[f"E{node}"]
                if self.causes == 2:
                    frame[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                    frame[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
                else:
                    frame[f"Y{node}"] = code
                alive = alive & (raw[f"C{node}"] == 1) & (code == 0)
            else:
                alive = alive & (raw[f"C{node}"] == 1)
        if self.kind == "end_of_study":
            frame["Y"] = raw["Y"]
        return pd.DataFrame(frame)

    def long_frame(self, grid: Sequence[float], *, offset: float = 0.0) -> pd.DataFrame:
        """The same sample as one row per unit: ``W``, ``A``, ``time`` and ``event``.

        An event at node ``k`` is at ``g_k - offset``, inside the interval
        ``(g_{k-1}, g_k]`` for an offset below the spacing.  A unit censored at node
        ``k`` was last seen at ``g_{k-1}``, and a unit event-free at the end is censored at
        ``g_K``.  Needs ``censor_first=False`` and no ``L2``.
        """
        if self.censor_first or self.with_l2 or self.kind != "survival":
            raise ValueError("a long layout needs a survival law with censor_first=False, no L2")
        g = np.asarray(grid, dtype=float)
        index = {name: i for i, name in enumerate(self.columns)}
        time = np.empty(len(self.support))
        code = np.zeros(len(self.support), dtype=np.int64)
        for row, point in enumerate(self.support):
            time[row] = g[-1]
            for node in range(1, self.n_times + 1):
                if point[index[f"C{node}"]] == 0:
                    time[row] = g[node - 2]
                    break
                event = point[index[f"E{node}"]]
                if event:
                    time[row] = g[node - 1] - offset
                    code[row] = event
                    break
        cells = np.repeat(np.arange(len(self.support)), self.counts)
        w = np.array([point[0] for point in self.support], dtype=float)
        a = np.array([point[1] for point in self.support], dtype=float)
        return pd.DataFrame(
            {"W": w[cells], "A": a[cells], "time": time[cells], "event": code[cells]}
        )

    def fit_columns(self, *, held: bool = True) -> dict[str, Any]:
        """The column keywords of ``LTMLE.fit`` for this law."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": "A" if held else [f"A{k}" for k in nodes],
            "baseline": ["W"],
            "censoring": [f"C{k}" for k in nodes],
            "time_varying": [["L2"] if self.with_l2 and k == 2 else [] for k in nodes],
        }
        if self.kind == "end_of_study":
            out["outcome"] = "Y"
        elif self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    def first_row_of(self) -> np.ndarray:
        """Index of the first sample row of each support point."""
        return np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    # --------------------------------------------------------------- g-formula

    def _mass(self, probs: Any, pattern: dict[str, int]) -> Any:
        positions = self._positions(tuple(sorted(pattern.items())))
        return sum(probs[i] for i in positions)

    def _positions(self, items: tuple[tuple[str, int], ...]) -> tuple[int, ...]:
        cache = self.__dict__.setdefault("_cache", {})
        if items not in cache:
            index = {name: i for i, name in enumerate(self.columns)}
            cache[items] = tuple(
                i
                for i, point in enumerate(self.support)
                if all(point[index[name]] == value for name, value in items)
            )
        return cache[items]

    def functional(
        self,
        probs: Any,
        assign: Callable[[Any, int], Sequence[Any]],
        horizon: int | None = None,
        cause: int = 1,
    ) -> Any:
        r"""The risk (or cause incidence, or end-of-study mean) under a node-1 intervention.

        ``assign(probs, w)`` returns the intervention's probability of each arm given
        ``W = w``.  It may read ``probs``, which is how a modified treatment policy states
        the law of the shifted treatment.  Every operation is a sum, a product or a
        quotient, so the value is analytic in ``probs``.
        """
        total = self._mass(probs, {})
        psi: Any = 0.0
        for w in (0, 1):
            share = self._mass(probs, {"W": w}) / total
            weights = assign(probs, w)
            for a in range(3):
                psi = psi + share * weights[a] * self._regression(
                    probs, {"W": w, "A": a}, 1, horizon, cause
                )
        return psi

    def _regression(
        self, probs: Any, history: dict[str, int], node: int, horizon: int | None, cause: int
    ) -> Any:
        """``Q_t(history)``: the iterated conditional expectation from node ``t``."""
        if self.with_l2 and node == 2:
            base = self._mass(probs, history)
            value: Any = 0.0
            for l in (0, 1):
                with_l = {**history, "L2": l}
                value = value + self._mass(probs, with_l) / base * self._after_l(
                    probs, with_l, node, horizon, cause
                )
            return value
        return self._after_l(probs, history, node, horizon, cause)

    def _after_l(
        self, probs: Any, history: dict[str, int], node: int, horizon: int | None, cause: int
    ) -> Any:
        kept = {**history, f"C{node}": 1}
        if self.kind == "end_of_study":
            if node == self.n_times:
                return self._mass(probs, {**kept, "Y": 1}) / self._mass(probs, kept)
            return self._regression(probs, kept, node + 1, horizon, cause)
        assert horizon is not None
        reached = self._mass(probs, kept)
        fired = self._mass(probs, {**kept, f"E{node}": cause}) / reached
        if node == horizon:
            return fired
        survived = {**kept, f"E{node}": 0}
        free = self._mass(probs, survived) / reached
        return fired + free * self._regression(probs, survived, node + 1, horizon, cause)

    def gateaux(
        self,
        assign: Callable[[Any, int], Sequence[Any]],
        point: int,
        horizon: int | None = None,
        cause: int = 1,
        *,
        step: float = 1e-30,
    ) -> float:
        """The Gateaux derivative of the functional at support point ``point``."""
        base = self.probs.astype(complex)
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        return float(np.imag(self.functional(perturbed, assign, horizon, cause)) / step)

    def eif(
        self,
        assign: Callable[[Any, int], Sequence[Any]],
        horizon: int | None = None,
        cause: int = 1,
    ) -> np.ndarray:
        """The EIF at every support point, in support order."""
        return np.array([self.gateaux(assign, i, horizon, cause) for i in range(len(self.support))])


def static(arm: int) -> Callable[[Any, int], Sequence[float]]:
    """The intervention that sets ``A = arm``."""
    return lambda probs, w: tuple(1.0 if a == arm else 0.0 for a in range(3))


def known(policy: np.ndarray) -> Callable[[Any, int], Sequence[float]]:
    """A known stochastic policy, ``policy[w, a]``."""
    return lambda probs, w: tuple(float(policy[w, a]) for a in range(3))


def shifted(law: PointLaw, mapping: Sequence[int]) -> Callable[[Any, int], Sequence[Any]]:
    """The law of ``mapping[A]`` given ``W``: a modified treatment policy on the levels."""

    def assign(probs: Any, w: int) -> Sequence[Any]:
        base = law._mass(probs, {"W": w})
        out: list[Any] = [0.0, 0.0, 0.0]
        for a in range(3):
            out[mapping[a]] = out[mapping[a]] + law._mass(probs, {"W": w, "A": a}) / base
        return tuple(out)

    return assign


def survival_law(*, causes: int = 1, with_l2: bool = False, censor_first: bool = True) -> PointLaw:
    """The three-node survival law."""
    return PointLaw("survival", 3, causes, with_l2, censor_first)


def end_of_study_law(*, with_l2: bool = False) -> PointLaw:
    """The two-censoring-node end-of-study law."""
    return PointLaw("end_of_study", 2, 1, with_l2)
