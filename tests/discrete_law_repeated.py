r"""Two-time-point *repeated-outcome* laws that a finite sample realises exactly.

The counterpart of :mod:`tests.discrete_law_survival` for an outcome that is measured at
every node and is **not** absorbing.  The node order is

.. code-block:: text

    W  ->  A1  ->  C1  ->  Y1  ->  L2  ->  A2  ->  C2  ->  Y2

and every conditional probability is a multiple of ``1/4``, so every cell probability is a
multiple of ``1 / N`` with ``N = 2 * 4**7``.  Laying ``N`` rows out in the cell proportions
makes the empirical distribution equal to the data-generating one.

Three structural facts are what these laws are for.

**The outcome returns.**  A unit with ``Y1 = 1`` has ``Y2 = 0`` with probability at least
``0.3``, so the absorbing reading :math:`E[\max_{s \le h} Y_s^d]` misses the truth by a
declared gap (:data:`ABSORBING_GAP`).

**The outcome confounds the next node.**  ``A2`` and ``C2`` depend on ``Y1``, and ``Y2``
depends on ``Y1``.  A fit whose history omits ``Y1`` converges to a different g-formula,
whose exact value :func:`history_free_value` states; it misses the truth by
:data:`HISTORY_GAP`.

**Censoring is missing at random given the history.**  ``C_t`` depends on
``(W, A_1, Y_{t-1}, L_t, A_t)`` and never on ``Y_t``.  A design test checks the tables.

Two laws share this structure.  :data:`BINARY` has binary ``Y_t``.  :data:`CONTINUOUS`
has three values per node, with ``Y1`` on ``{-0.5, 0.5, 2.0}`` and ``Y2`` on
``{0.0, 0.5, 1.0}``, so ``Y1``'s range exceeds ``Y2``'s.  A fit reads it as ``gaussian``.

Every truth is enumerated twice.  :func:`RepeatedLaw.functional` writes the g-formula on
the cell probabilities, which :func:`RepeatedLaw.gateaux` differentiates by a complex
step.  :func:`RepeatedLaw.counterfactual_value` composes the conditionals directly under
the intervention.  A test asserts that the two agree.  Nothing the library supplies enters
either.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from .discrete_law_longitudinal import CellMeans

__all__ = [
    "ABSORBING_GAP",
    "BINARY",
    "CONTINUOUS",
    "HISTORY_GAP",
    "HORIZONS",
    "REGIMEN_REFERENCE",
    "CellMeans",
    "RepeatedLaw",
]

#: Rows in the realised sample: one factor of two for ``P(W)`` and seven of four.
N = 2 * 4**7

#: The horizons a fit on these laws reports.
HORIZONS: tuple[int, ...] = (1, 2)

#: The declared least gap between the truth and the absorbing reading at ``h = 2``.
ABSORBING_GAP = 0.05

#: The declared least gap between the truth and the history-free g-formula at ``h = 2``.
HISTORY_GAP = 0.02

REGIMEN_REFERENCE = "never"

_NODES = ("w", "a1", "c1", "y1", "l2", "a2", "c2", "y2")
_COLUMNS = ("W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")

P_W = np.array([0.5, 0.5])

#: ``P(A1 = 1 | W)``.
G1 = np.array([0.50, 0.25])

#: ``P(C1 = 1 | W, A1)``.
C1 = np.array([[0.75, 0.50], [0.75, 0.75]])

#: ``P(L2 = 1 | W, A1, Y1 high)``, indexed ``[w, a1, high]``.  ``L2`` reads ``Y1``.
P_L2 = np.array([[[0.25, 0.75], [0.50, 0.75]], [[0.25, 0.50], [0.50, 0.75]]])

#: ``P(A2 = 1 | W, A1, Y1 high, L2)``, indexed ``[w, a1, high, l2]``.  ``A2`` reads ``Y1``.
G2 = np.array(
    [
        [[[0.25, 0.50], [0.75, 0.75]], [[0.25, 0.25], [0.75, 0.50]]],
        [[[0.25, 0.25], [0.50, 0.75]], [[0.50, 0.25], [0.75, 0.75]]],
    ]
)

#: ``P(C2 = 1 | W, A1, Y1 high, L2, A2)``, indexed ``[w, a1, high, l2, a2]``.  ``C2`` reads
#: ``Y1`` and not ``Y2``: censoring is missing at random given the history.
C2 = np.array(
    [
        [
            [[[0.75, 0.25], [0.75, 0.50]], [[0.25, 0.75], [0.50, 0.75]]],
            [[[0.75, 0.25], [0.50, 0.25]], [[0.25, 0.75], [0.25, 0.50]]],
        ],
        [
            [[[0.50, 0.25], [0.75, 0.25]], [[0.25, 0.75], [0.25, 0.75]]],
            [[[0.75, 0.50], [0.75, 0.25]], [[0.25, 0.50], [0.25, 0.75]]],
        ],
    ]
)


@dataclass(frozen=True)
class RepeatedLaw:
    """One repeated-outcome law with ``T = 2`` and its exact sample.

    Parameters
    ----------
    name : str
        A label for test identifiers.
    y1_values : tuple of float
        The support of ``Y1``, in level order.
    y2_values : tuple of float
        The support of ``Y2``, in level order.
    y1_table : callable
        ``(w, a1) -> probabilities`` over the levels of ``Y1``.
    y2_table : callable
        ``(w, a1, y1_level, l2, a2) -> probabilities`` over the levels of ``Y2``.
    family : str
        The family a fit resolves for this law.
    """

    name: str
    y1_values: tuple[float, ...]
    y2_values: tuple[float, ...]
    y1_table: Callable[[int, int], tuple[float, ...]]
    y2_table: Callable[[int, int, int, int, int], tuple[float, ...]]
    family: str
    _index_cache: dict[tuple[tuple[str, int], ...], tuple[int, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ------------------------------------------------------------ the law

    def high(self, y1_level: int) -> int:
        """Whether ``Y1`` is at its top level, which every later node reads."""
        return int(y1_level == len(self.y1_values) - 1)

    @cached_property
    def support(self) -> tuple[tuple[Any, ...], ...]:
        """One point per observable history, as level indices; ``None`` is not reached."""
        k1, k2 = len(self.y1_values), len(self.y2_values)
        points: list[tuple[Any, ...]] = []
        for w, a1 in itertools.product(range(2), range(2)):
            points.append((w, a1, 0, None, None, None, None, None))
        for w, a1, y1, l2, a2 in itertools.product(
            range(2), range(2), range(k1), range(2), range(2)
        ):
            points.append((w, a1, 1, y1, l2, a2, 0, None))
        for w, a1, y1, l2, a2, y2 in itertools.product(
            range(2), range(2), range(k1), range(2), range(2), range(k2)
        ):
            points.append((w, a1, 1, y1, l2, a2, 1, y2))
        return tuple(points)

    def _mass_of(self, point: tuple[Any, ...]) -> float:
        w, a1, c1, y1, l2, a2, c2, y2 = point
        mass = P_W[w] * (G1[w] if a1 == 1 else 1.0 - G1[w])
        if c1 == 0:
            return float(mass * (1.0 - C1[w, a1]))
        mass *= C1[w, a1] * self.y1_table(w, a1)[y1]
        high = self.high(y1)
        mass *= P_L2[w, a1, high] if l2 == 1 else 1.0 - P_L2[w, a1, high]
        mass *= G2[w, a1, high, l2] if a2 == 1 else 1.0 - G2[w, a1, high, l2]
        stay = C2[w, a1, high, l2, a2]
        if c2 == 0:
            return float(mass * (1.0 - stay))
        return float(mass * stay * self.y2_table(w, a1, y1, l2, a2)[y2])

    @cached_property
    def counts(self) -> np.ndarray:
        """Rows per support point in an ``N``-row sample that realises the law exactly."""
        values = np.array([self._mass_of(point) * N for point in self.support])
        rounded = np.rint(values)
        if np.max(np.abs(values - rounded)) > 1e-6:  # pragma: no cover - guards the constants
            raise AssertionError("the cell probabilities are not multiples of 1/N")
        if abs(rounded.sum() - N) > 0.5:  # pragma: no cover - guards the constants
            raise AssertionError("the cell probabilities do not sum to one")
        return rounded.astype(int)

    @cached_property
    def probs(self) -> np.ndarray:
        """``P`` over the support, bit for bit the empirical law of :meth:`frame`."""
        return self.counts / N

    # ------------------------------------------------------------ regimens

    #: The oracle side: the arm at node 2 as a function of ``(w, y1_level, l2)``.
    def arm2(self, regimen: str, w: int, y1: int, l2: int) -> int:
        """The arm a regimen assigns at node 2."""
        if regimen == "never":
            return 0
        if regimen == "always":
            return 1
        if regimen == "treat_if_high":
            return self.high(y1)
        raise ValueError(regimen)

    @staticmethod
    def arm1(regimen: str) -> int:
        """The arm a regimen assigns at node 1."""
        return 0 if regimen == "never" else 1

    def regimen_spec(self) -> dict[str, Any]:
        """The estimator side: what ``LTMLE`` is handed as ``regimens=``."""
        cut = self.y1_values[-1]
        return {
            "never": 0,
            "always": 1,
            # Treat at node 1, then treat at node 2 only when the first outcome is high.
            "treat_if_high": (1, lambda h: (h["Y1"] >= cut).astype(int)),
        }

    @property
    def regimens(self) -> tuple[str, ...]:
        return ("never", "always", "treat_if_high")

    @property
    def names(self) -> tuple[str, ...]:
        """The parameter names a repeated fit reports on this law, in report order."""
        means = tuple(f"ey_regimen[{d} @ t={h}]" for d in self.regimens for h in HORIZONS)
        contrasts = tuple(
            f"ate_regimen[{d} vs {REGIMEN_REFERENCE} @ t={h}]"
            for d in self.regimens
            if d != REGIMEN_REFERENCE
            for h in HORIZONS
        )
        return means + contrasts

    # ------------------------------------------------------------ the functional

    def _index(self, **pattern: int) -> tuple[int, ...]:
        key = tuple(sorted(pattern.items()))
        if key not in self._index_cache:
            self._index_cache[key] = tuple(
                position
                for position, point in enumerate(self.support)
                if all(point[_NODES.index(node)] == value for node, value in pattern.items())
            )
        return self._index_cache[key]

    def _mass(self, probs: Any, **pattern: int) -> Any:
        return sum(probs[position] for position in self._index(**pattern))

    def functional(self, probs: Any, estimand: str, *, absorbing: bool = False) -> Any:
        r"""The g-formula :math:`E[Y_h^d]` as a function of the cell probabilities.

        .. math::

            \Psi_1 = \sum_w P(w)\, E[Y_1 \mid w, a_1, C_1 = 1], \qquad
            \Psi_2 = \sum_w P(w) \sum_{y_1, l_2} P(y_1, l_2 \mid w, a_1, C_1 = 1)\,
                     E[Y_2 \mid w, a_1, y_1, l_2, a_2, C_2 = 1]

        with ``a_2 = d_2(w, y_1, l_2)``.  ``absorbing=True`` gives the reading
        :math:`E[\max(Y_1, Y_2)^d]` of a binary law at ``h = 2``, the M1 limit.  Every
        operation is a sum or a quotient of sums, so a complex step differentiates it.
        """
        p = probs
        if estimand.startswith("ate_regimen["):
            index = estimand[len("ate_regimen[") : -1]
            contrast, _, horizon = index.rpartition(" @ t=")
            left, right = contrast.split(" vs ")
            return self.functional(
                p, f"ey_regimen[{left} @ t={horizon}]", absorbing=absorbing
            ) - self.functional(p, f"ey_regimen[{right} @ t={horizon}]", absorbing=absorbing)
        label, _, text = estimand[len("ey_regimen[") : -1].rpartition(" @ t=")
        horizon = int(text)
        a1 = self.arm1(label)
        total = self._mass(p)
        psi: Any = 0.0
        for w in (0, 1):
            share = self._mass(p, w=w) / total
            reached = self._mass(p, w=w, a1=a1, c1=1)
            if horizon == 1:
                inner: Any = 0.0
                for y1, value in enumerate(self.y1_values):
                    inner = inner + value * self._mass(p, w=w, a1=a1, c1=1, y1=y1) / reached
                psi = psi + share * inner
                continue
            inner = 0.0
            for y1, l2 in itertools.product(range(len(self.y1_values)), range(2)):
                density = self._mass(p, w=w, a1=a1, c1=1, y1=y1, l2=l2) / reached
                a2 = self.arm2(label, w, y1, l2)
                kept = self._mass(p, w=w, a1=a1, c1=1, y1=y1, l2=l2, a2=a2, c2=1)
                mean: Any = 0.0
                for y2, value in enumerate(self.y2_values):
                    cell = self._mass(p, w=w, a1=a1, c1=1, y1=y1, l2=l2, a2=a2, c2=1, y2=y2)
                    mean = mean + value * cell / kept
                if absorbing:
                    mean = self.y1_values[y1] + (1.0 - self.y1_values[y1]) * mean
                inner = inner + density * mean
            psi = psi + share * inner
        return psi

    def history_free_value(self, estimand: str) -> float:
        """The g-formula with ``Y1`` dropped from every later history: the M3 limit.

        A fit whose node-2 history omits ``Y1`` regresses ``Y2`` on ``(W, A1, L2, A2)``
        among the uncensored, and its mechanisms omit ``Y1`` too.  Both nuisances are then
        correct for the coarser history, so the estimator converges to the g-formula on
        that history.  ``C2`` reads ``Y1``, so this differs from the truth.
        """
        p = self.probs
        label, _, text = estimand[len("ey_regimen[") : -1].rpartition(" @ t=")
        if int(text) != 2 or label == "treat_if_high":
            raise ValueError("the history-free limit is stated for static plans at h = 2")
        a1 = self.arm1(label)
        a2 = self.arm2(label, 0, 0, 0)
        total = self._mass(p)
        psi = 0.0
        for w in (0, 1):
            share = self._mass(p, w=w) / total
            reached = self._mass(p, w=w, a1=a1, c1=1)
            inner = 0.0
            for l2 in (0, 1):
                density = self._mass(p, w=w, a1=a1, c1=1, l2=l2) / reached
                kept = self._mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1)
                mean = sum(
                    value * self._mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, y2=y2) / kept
                    for y2, value in enumerate(self.y2_values)
                )
                inner += density * mean
            psi += share * inner
        return float(psi)

    def counterfactual_value(self, estimand: str) -> float:
        """:math:`E[Y_h^d]` composed from the conditionals under the intervention.

        Independent of :meth:`functional`: it reads the tables, not the cells, and sets
        every censoring node to one.
        """
        if estimand.startswith("ate_regimen["):
            index = estimand[len("ate_regimen[") : -1]
            contrast, _, horizon = index.rpartition(" @ t=")
            left, right = contrast.split(" vs ")
            return self.counterfactual_value(
                f"ey_regimen[{left} @ t={horizon}]"
            ) - self.counterfactual_value(f"ey_regimen[{right} @ t={horizon}]")
        label, _, text = estimand[len("ey_regimen[") : -1].rpartition(" @ t=")
        horizon = int(text)
        a1 = self.arm1(label)
        value = 0.0
        for w in (0, 1):
            first = self.y1_table(w, a1)
            for y1, p1 in enumerate(first):
                if horizon == 1:
                    value += P_W[w] * p1 * self.y1_values[y1]
                    continue
                high = self.high(y1)
                for l2 in (0, 1):
                    pl = P_L2[w, a1, high] if l2 == 1 else 1.0 - P_L2[w, a1, high]
                    a2 = self.arm2(label, w, y1, l2)
                    second = self.y2_table(w, a1, y1, l2, a2)
                    mean = sum(q * v for q, v in zip(second, self.y2_values, strict=True))
                    value += P_W[w] * p1 * pl * mean
        return float(value)

    @cached_property
    def truth(self) -> dict[str, float]:
        """Population value of every reported parameter."""
        return {name: float(self.functional(self.probs, name)) for name in self.names}

    def gateaux(self, estimand: str, point: int, *, step: float = 1e-30) -> float:
        """The Gateaux derivative of ``estimand`` at support point ``point``."""
        base = self.probs.astype(complex)
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        return float(np.imag(self.functional(perturbed, estimand)) / step)

    def eif(self, estimand: str) -> np.ndarray:
        """The EIF of ``estimand`` at every support point, in support order."""
        return np.array([self.gateaux(estimand, position) for position in range(len(self.support))])

    # ------------------------------------------------------------ the sample

    def frame(self) -> pd.DataFrame:
        """The ``N``-row sample whose empirical distribution is exactly this law."""
        cells = np.repeat(np.arange(len(self.support)), self.counts)
        columns: dict[str, np.ndarray] = {}
        for position, name in enumerate(_COLUMNS):
            values = []
            for point in self.support:
                level = point[position]
                if level is None:
                    values.append(np.nan)
                elif name == "Y1":
                    values.append(self.y1_values[level])
                elif name == "Y2":
                    values.append(self.y2_values[level])
                else:
                    values.append(float(level))
            columns[name] = np.array(values)[cells]
        return pd.DataFrame(columns)

    def first_row_of(self) -> np.ndarray:
        """Index of the first sample row of each support point, in support order."""
        return np.concatenate([[0], np.cumsum(self.counts)[:-1]])


# ---------------------------------------------------------------- the binary law

#: ``P(Y1 = 1 | W, A1)``, indexed ``[w, a1]``.
_H1 = np.array([[0.50, 0.25], [0.75, 0.50]])

#: ``P(Y2 = 1 | W, A1, Y1, L2, A2)``, indexed ``[w, a1, y1, l2, a2]``.  At ``Y1 = 1`` every
#: entry is at most ``0.7``, so the outcome returns to zero with probability at least 0.3.
_H2 = np.array(
    [
        [
            [[[0.75, 0.50], [0.75, 0.75]], [[0.25, 0.25], [0.50, 0.25]]],
            [[[0.50, 0.75], [0.75, 0.50]], [[0.25, 0.50], [0.25, 0.25]]],
        ],
        [
            [[[0.50, 0.75], [0.75, 0.75]], [[0.25, 0.50], [0.25, 0.25]]],
            [[[0.75, 0.50], [0.75, 0.75]], [[0.50, 0.25], [0.25, 0.25]]],
        ],
    ]
)

BINARY = RepeatedLaw(
    name="binary",
    y1_values=(0.0, 1.0),
    y2_values=(0.0, 1.0),
    y1_table=lambda w, a1: (1.0 - _H1[w, a1], _H1[w, a1]),
    y2_table=lambda w, a1, y1, l2, a2: (1.0 - _H2[w, a1, y1, l2, a2], _H2[w, a1, y1, l2, a2]),
    family="binomial",
)

# ---------------------------------------------------------------- the continuous law

_LOW, _MID, _TOP = (0.50, 0.25, 0.25), (0.25, 0.50, 0.25), (0.25, 0.25, 0.50)

#: ``P(Y1 = level | W, A1)``, indexed ``[w][a1]``.
_Y1_TABLE = ((_LOW, _MID), (_MID, _TOP))


def _y2_continuous(w: int, a1: int, y1: int, l2: int, a2: int) -> tuple[float, ...]:
    """Three-level ``Y2``: higher after a high ``Y1`` and after treatment, with exceptions."""
    score = (y1 >= 2) + a2 + (w == 1 and l2 == 1) - (a1 == 1 and y1 == 0)
    return (_LOW, _MID, _TOP, _MID)[int(np.clip(score, 0, 3))]


CONTINUOUS = RepeatedLaw(
    name="continuous",
    y1_values=(-0.5, 0.5, 2.0),
    y2_values=(0.0, 0.5, 1.0),
    y1_table=lambda w, a1: _Y1_TABLE[w][a1],
    y2_table=_y2_continuous,
    family="gaussian",
)
