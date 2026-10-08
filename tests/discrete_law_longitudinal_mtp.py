r"""A two-node law for longitudinal modified treatment policies, written longhand.

The implementation never enters this module (``tests/unit/test_oracle_independence.py``).
The parameter under a plan of modified treatment policies is the g-formula of Díaz,
Williams, Hoffman and Schenck (2023), Theorem 1, with each node's policy reading the unit's
own treatment and the *intervened* history (their Definition 1):

.. math::

    \Psi = \sum_w P(w) \sum_{a_1} g_1(a_1 \mid w) \sum_e p_1(e)
           \sum_{l} P(l \mid w, a_1^d) \sum_{a_2} g_2(a_2 \mid w, a_1^d, l)
           \sum_f p_2(f)\, \bar Q\bigl(w, a_1^d, l, d_{2,f}(a_2, w, a_1^d, l)\bigr),

with :math:`a_1^d = d_{1,e}(a_1, w)`.  Every conditional is a quotient of cell masses and
every policy is a fixed map of support indices, so the functional is analytic in the cell
probabilities and a complex step differentiates it to full precision.

Two laws share the cell masses.  :data:`UNIT` puts both nodes on the doses ``{0, 1, 2, 3}``
with unit-width bins.  :data:`SCALED` relabels node 2 as ``{1, 2, 4, 8}`` with bins that
double in width, so a scale by two there meets the exactness condition of
``tests/discrete_law_policy_point.py``: :math:`w(b(a)) = |b'(a)|\, w(a)`.  The same cells
also realise a categorical reading of node 2 and a vector reading, in which level ``k`` is
the pair ``(k // 2, k % 2)`` of two binary components.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

#: ``P(W = w)``.
P_W = np.array([0.5, 0.5])
#: ``g_1(a_1 | w)`` in eighths, indexed ``[w, a1]``.
G1 = np.array([[1.0, 2.0, 2.0, 3.0], [3.0, 2.0, 2.0, 1.0]]) / 8.0
#: ``P(L_2 = 1 | w, a1)`` in quarters.
P_L2 = np.array([[0.25, 0.50, 0.75, 0.50], [0.75, 0.50, 0.25, 0.50]])
_ROWS = np.array(
    [[1.0, 2.0, 2.0, 3.0], [3.0, 2.0, 2.0, 1.0], [2.0, 3.0, 1.0, 2.0], [2.0, 1.0, 3.0, 2.0]]
)
#: ``g_2(a_2 | w, a1, l2)`` in eighths, indexed ``[w, a1, l2, a2]``.
G2 = np.array(
    [[[_ROWS[(w + a1 + 2 * l2) % 4] / 8.0 for l2 in range(2)] for a1 in range(4)] for w in range(2)]
)
_Q_VALUES = (0.25, 0.50, 0.75)
#: ``E[Y | w, a1, l2, a2]`` in quarters, non-monotone in both doses.  Chosen by a search over
#: modular patterns for the law whose plan means lie farthest apart: from the natural course
#: by 0.011, from one another by 0.0025, from every static mean by 0.0053, and the history
#: reading of node 2 moves its plan by 0.0137.
Q = np.array(
    [
        [
            [[_Q_VALUES[(w + a1 + l2 + 2 * a2 * a2) % 3] for a2 in range(4)] for l2 in range(2)]
            for a1 in range(4)
        ]
        for w in range(2)
    ]
)

#: ``(w, a1, l2, a2, y)`` support points, in the order :func:`frame` emits rows.
SUPPORT = tuple(itertools.product(range(2), range(4), range(2), range(4), range(2)))
#: Rows: every cell probability is a multiple of ``1 / N``.
N = 2 * 8 * 4 * 8 * 4


def _cell(point: tuple[int, int, int, int, int]) -> float:
    w, a1, l2, a2, y = point
    p_l2 = P_L2[w, a1] if l2 else 1.0 - P_L2[w, a1]
    p_y = Q[w, a1, l2, a2] if y else 1.0 - Q[w, a1, l2, a2]
    return float(P_W[w] * G1[w, a1] * p_l2 * G2[w, a1, l2, a2] * p_y)


def _counts() -> np.ndarray:
    counts = np.array([_cell(point) * N for point in SUPPORT])
    assert np.allclose(counts, np.round(counts), atol=1e-9), "the law is not realised by N rows"
    assert int(np.round(counts).sum()) == N
    return np.round(counts).astype(int)


COUNTS = _counts()
PROBS = COUNTS / N


@dataclass(frozen=True)
class Grid:
    """The dose values of one node and the bin edges that isolate them.

    Parameters
    ----------
    doses : tuple of float
        The dose of each support index.
    edges : tuple of float
        Bin edges, one support point per bin.
    """

    doses: tuple[float, ...]
    edges: tuple[float, ...]


UNIT_GRID = Grid((0.0, 1.0, 2.0, 3.0), (-0.5, 0.5, 1.5, 2.5, 3.5))
SCALED_GRID = Grid((1.0, 2.0, 4.0, 8.0), (0.75, 1.5, 3.0, 6.0, 12.0))


def frame(
    node1: Grid = UNIT_GRID, node2: Grid = UNIT_GRID, *, vector: bool = False
) -> pd.DataFrame:
    """``N`` rows realising the law, in :data:`SUPPORT` order.

    ``vector=True`` writes each node as two binary columns, ``A1a, A1b`` and ``A2a, A2b``,
    with level ``k`` the pair ``(k // 2, k % 2)``.
    """
    rows = []
    for point, count in zip(SUPPORT, COUNTS, strict=True):
        w, a1, l2, a2, y = point
        rows.extend([(float(w), a1, float(l2), a2, float(y))] * int(count))
    raw = np.array(rows, dtype=float)
    a1 = raw[:, 1].astype(int)
    a2 = raw[:, 3].astype(int)
    columns: dict[str, Any] = {"W": raw[:, 0], "L2": raw[:, 2], "Y": raw[:, 4]}
    if vector:
        columns.update({"A1a": a1 // 2, "A1b": a1 % 2, "A2a": a2 // 2, "A2b": a2 % 2})
    else:
        columns["A1"] = np.asarray(node1.doses)[a1]
        columns["A2"] = np.asarray(node2.doses)[a2]
    return pd.DataFrame(columns)


def sample(n: int, seed: int, node1: Grid = UNIT_GRID, node2: Grid = UNIT_GRID) -> pd.DataFrame:
    """Draw ``n`` rows from the law, for fits whose nuisances are not exact."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(SUPPORT), size=n, p=np.asarray(PROBS, dtype=float))
    points = np.asarray(SUPPORT, dtype=int)[cells]
    return pd.DataFrame(
        {
            "W": points[:, 0].astype(float),
            "A1": np.asarray(node1.doses)[points[:, 1]],
            "L2": points[:, 2].astype(float),
            "A2": np.asarray(node2.doses)[points[:, 3]],
            "Y": points[:, 4].astype(float),
        }
    )


def first_row_of() -> np.ndarray:
    """Index of the first row of each support point."""
    return np.concatenate([[0], np.cumsum(COUNTS)[:-1]]).astype(int)


# ------------------------------------------------------------------ policies

#: A node policy: ``(probability, map)`` branches.  Node 1 maps are ``[w][a1]``, node 2 maps
#: ``[w][a1][l2][a2]``, each entry a support index.
Branches = tuple[tuple[float, Any], ...]


def _node1(assign: Any) -> Any:
    return tuple(tuple(int(assign(w, a)) for a in range(4)) for w in range(2))


def _node2(assign: Any) -> Any:
    return tuple(
        tuple(
            tuple(tuple(int(assign(w, a1, l2, a)) for a in range(4)) for l2 in range(2))
            for a1 in range(4)
        )
        for w in range(2)
    )


IDENTITY1: Branches = ((1.0, _node1(lambda w, a: a)),)
IDENTITY2: Branches = ((1.0, _node2(lambda w, a1, l2, a: a)),)
#: ``a + 1`` held at the top index: ``Shift(1, cap=3)`` on the unit grid, ``Scale(2, cap=8)``
#: on the scaled one.
UP1: Branches = ((1.0, _node1(lambda w, a: min(a + 1, 3))),)
UP2: Branches = ((1.0, _node2(lambda w, a1, l2, a: min(a + 1, 3))),)
#: Node 2 moves up one only when the (intervened) earlier dose is at least 2.
HISTORY2: Branches = ((1.0, _node2(lambda w, a1, l2, a: a + 1 if a1 >= 2 and a < 3 else a)),)
#: Node 2 reverses the dose, ``3 - a``.
DECREASING2: Branches = ((1.0, _node2(lambda w, a1, l2, a: 3 - a)),)
#: With probability one quarter move up one (held at the top), else stay.
RANDOM1: Branches = ((0.25, UP1[0][1]), (0.75, IDENTITY1[0][1]))
RANDOM2: Branches = ((0.25, UP2[0][1]), (0.75, IDENTITY2[0][1]))
#: The vector policy: a unit with both components one has its second set to zero.
VECTOR2: Branches = ((1.0, _node2(lambda w, a1, l2, a: 2 if a == 3 else a)),)
VECTOR1: Branches = ((1.0, _node1(lambda w, a: 2 if a == 3 else a)),)

#: Every plan, as its two node policies.
PLANS: dict[str, tuple[Branches, Branches]] = {
    "natural": (IDENTITY1, IDENTITY2),
    "up": (UP1, UP2),
    "history": (IDENTITY1, HISTORY2),
    "decreasing": (UP1, DECREASING2),
    "random": (RANDOM1, RANDOM2),
    "up at 2": (IDENTITY1, UP2),
    "up then history": (UP1, HISTORY2),
    "vector": (VECTOR1, VECTOR2),
}


def _mass(probs: Any, **pattern: int) -> Any:
    return sum(probs[index] for index in _index(**pattern))


@cache
def _index(**pattern: int) -> tuple[int, ...]:
    names = ("w", "a1", "l2", "a2", "y")
    return tuple(
        index
        for index, point in enumerate(SUPPORT)
        if all(point[names.index(name)] == value for name, value in pattern.items())
    )


def functional_mtp(
    probs: Any, plan: tuple[Branches, Branches], *, intervened_history: bool = True
) -> Any:
    """The g-formula under a plan of modified treatment policies, from cell masses.

    ``intervened_history=False`` is the mutation of plan section 6.12: node 2's policy reads
    the *natural* earlier dose rather than the intervened one.  The tests assert that this
    gives a different number, so the history semantics are witnessed.
    """
    total = _mass(probs)
    psi: Any = 0.0
    for w in range(2):
        p_w = _mass(probs, w=w) / total
        for a1 in range(4):
            g1 = _mass(probs, w=w, a1=a1) / _mass(probs, w=w)
            for p_e, map1 in plan[0]:
                a1d = map1[w][a1]
                reached = _mass(probs, w=w, a1=a1d)
                read = a1d if intervened_history else a1
                for l2 in range(2):
                    p_l2 = _mass(probs, w=w, a1=a1d, l2=l2) / reached
                    stratum = _mass(probs, w=w, a1=a1d, l2=l2)
                    for a2 in range(4):
                        g2 = _mass(probs, w=w, a1=a1d, l2=l2, a2=a2) / stratum
                        for p_f, map2 in plan[1]:
                            a2d = map2[w][read][l2][a2]
                            cell = _mass(probs, w=w, a1=a1d, l2=l2, a2=a2d)
                            events = _mass(probs, w=w, a1=a1d, l2=l2, a2=a2d, y=1)
                            psi = psi + p_w * g1 * p_e * p_l2 * g2 * p_f * events / cell
    return psi


def functional(probs: Any, estimand: str) -> Any:
    """An ``ey_regimen`` or ``ate_regimen`` name over :data:`PLANS`."""
    if estimand.startswith("ate_regimen["):
        left, right = estimand[len("ate_regimen[") : -1].split(" vs ")
        return functional(probs, f"ey_regimen[{left}]") - functional(probs, f"ey_regimen[{right}]")
    return functional_mtp(probs, PLANS[estimand[len("ey_regimen[") : -1]])


def truth(estimand: str) -> float:
    """The parameter at the law."""
    return float(np.real(functional(PROBS, estimand)))


def gateaux_at(estimand: str, point: int, *, step: float = 1e-30) -> float:
    """The complex-step Gateaux derivative of ``estimand`` at one support point."""
    base = np.asarray(PROBS, dtype=complex)
    mass = np.zeros_like(base)
    mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


@cache
def eif(estimand: str) -> np.ndarray:
    """The efficient influence function at every support point."""
    return np.array([gateaux_at(estimand, point) for point in range(len(SUPPORT))])


def static_means() -> dict[str, float]:
    """``E[Y^{a1, a2}]`` for every static plan, the foil of the nonzero witnesses."""
    out = {}
    for a1, a2 in itertools.product(range(4), range(4)):
        plan = (
            ((1.0, _node1(lambda w, a, a1=a1: a1)),),
            ((1.0, _node2(lambda w, b1, l2, a, a2=a2: a2)),),
        )
        out[f"{a1}/{a2}"] = float(np.real(functional_mtp(PROBS, plan)))
    return out


# ------------------------------------------------------------------ survival and competing risks

#: Binary-node policies on :mod:`tests.discrete_law_survival` and
#: :mod:`tests.discrete_law_competing`: node 1 keeps the treatment with probability one
#: quarter and is otherwise untreated (a risk-ratio tilt of 0.25, asymmetric so that a swap of
#: the two branch weights moves the answer), and node 2 treats every unit
#: with ``L2 = 1`` and leaves the rest.  Node maps are ``[w][a1]`` and ``[w][a1][l2][a2]``.
SURVIVAL_NODE1: Branches = (
    (0.25, tuple(tuple(a for a in range(2)) for _ in range(2))),
    (0.75, tuple(tuple(0 for _ in range(2)) for _ in range(2))),
)
SURVIVAL_NODE2: Branches = (
    (
        1.0,
        tuple(
            tuple(
                tuple(tuple(1 if l2 == 1 else a for a in range(2)) for l2 in range(2))
                for _ in range(2)
            )
            for _ in range(2)
        ),
    ),
)


def functional_mtp_survival(probs: Any, horizon: int, *, cause: str | None = None) -> Any:
    """The cumulative risk, or a cause's incidence, at ``horizon`` under the binary MTP plan.

    The policy reads the unit's own treatment at each node and the intervened earlier one,
    as :func:`functional_mtp` does; the event composes as in
    ``tests.discrete_law_longitudinal_policy.functional_policy_survival``.
    """
    from . import discrete_law_competing as competing
    from . import discrete_law_survival as survival

    if cause is None:
        mass = survival._mass
        event1, event2, alive1 = {"y1": 1}, {"y2": 1}, {"y1": 0}
    else:
        j = competing.CAUSES.index(cause) + 1
        mass = competing._mass
        event1, event2, alive1 = {"j1": j}, {"j2": j}, {"j1": 0}
    total = mass(probs)
    psi: Any = 0.0
    for w in (0, 1):
        share = mass(probs, w=w) / total
        for a1 in (0, 1):
            g1 = mass(probs, w=w, a1=a1) / mass(probs, w=w)
            for p_e, map1 in SURVIVAL_NODE1:
                a1d = map1[w][a1]
                reached = mass(probs, w=w, a1=a1d, c1=1)
                hazard1 = mass(probs, w=w, a1=a1d, c1=1, **event1) / reached
                if horizon == 1:
                    psi = psi + share * g1 * p_e * hazard1
                    continue
                survived = mass(probs, w=w, a1=a1d, c1=1, **alive1)
                remaining: Any = 0.0
                for l2 in (0, 1):
                    stratum = mass(probs, w=w, a1=a1d, c1=1, **alive1, l2=l2)
                    density = stratum / survived
                    for a2 in (0, 1):
                        g2 = mass(probs, w=w, a1=a1d, c1=1, **alive1, l2=l2, a2=a2) / stratum
                        for p_f, map2 in SURVIVAL_NODE2:
                            a2d = map2[w][a1d][l2][a2]
                            uncensored = mass(
                                probs, w=w, a1=a1d, c1=1, **alive1, l2=l2, a2=a2d, c2=1
                            )
                            events = mass(
                                probs, w=w, a1=a1d, c1=1, **alive1, l2=l2, a2=a2d, c2=1, **event2
                            )
                            remaining = remaining + density * g2 * p_f * events / uncensored
                psi = psi + share * g1 * p_e * (hazard1 + (survived / reached) * remaining)
    return psi
