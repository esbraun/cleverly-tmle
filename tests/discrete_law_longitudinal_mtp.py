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

#: The atom plans of X40 part (b), on the unit grid with the dose 3 at node 2 declared an
#: atom.  "to atom" moves 0 and 1 up one and sends 2 to the atom; the atom stays.  On the
#: support it is :data:`UP2`, so the discrete formula and the mixed ratio must agree.  "hold
#: atom" moves 0 and 1 up one and leaves 2 and the atom in place.  "atom random" takes
#: "to atom" with probability 0.4 and "hold atom" otherwise.
TO_ATOM2: Branches = ((1.0, _node2(lambda w, a1, l2, a: min(a + 1, 3))),)
HOLD_ATOM2: Branches = ((1.0, _node2(lambda w, a1, l2, a: a + 1 if a < 2 else a)),)
ATOM_RANDOM2: Branches = ((0.4, TO_ATOM2[0][1]), (0.6, HOLD_ATOM2[0][1]))
#: The declared atom of node 2 in the atom plans, as a support index and a dose.
ATOM_INDEX = 3
ATOM_DOSE = 3.0

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
    "to atom": (IDENTITY1, TO_ATOM2),
    "hold atom": (IDENTITY1, HOLD_ATOM2),
    "atom random": (IDENTITY1, ATOM_RANDOM2),
}


def atom_ratio_cells() -> np.ndarray:
    """The mixed ratio of "to atom" at the atom, in every node-2 stratum ``(w, a1, l2)``.

    The atom's preimage is the interval ``[1.5, 2.5)``, which holds the dose 2, and the atom
    itself, so the ratio is ``(g_2(2) + g_2(3)) / g_2(3)`` (``X40-plan.md`` section 2.3).
    """
    return np.array(
        [
            (G2[w, a1, l2, 2] + G2[w, a1, l2, ATOM_INDEX]) / G2[w, a1, l2, ATOM_INDEX]
            for w in range(2)
            for a1 in range(4)
            for l2 in range(2)
        ]
    )


def preimage_mass() -> float:
    """``P(A_2 = 2)``: the mass of the interval the "to atom" plan sends to the atom."""
    return float(_mass(PROBS, a2=2))


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


# ------------------------------------------------------------------ mixed vector nodes
#
# A coarse law whose two treatment nodes are each a vector ``A_t = (F_t, D_t)``: a binary
# flag and a dose on the grid ``{0, 1, 2}``.  Support index ``k = 3 f + d``.  The fit reads
# ``D_t`` as a continuous component of a mixed node, so this is the oracle of the classifier
# route at a vector node with a continuous component (Díaz, Williams, Hoffman and Schenck
# 2023, Section 2, Theorem 1 and Section 5.4).
#
# The saturated classifier returns the exact ratio under three conditions, which every fit
# of this law meets:
#
# 1. ``Folds.single``, unit weights, and a frame that realises the law by integer counts
#    (:data:`MIXED_N` rows, or :data:`MIXED_CENSORED_N` with censoring).
# 2. Every policy stays on the grid, by a cap at the grid maximum, so every stacked cell
#    with label one also holds label-zero mass.  Otherwise the predicted probability is
#    one and the odds guard of ``cleverly.learners.density_ratio`` replaces the ratio.
# 3. The learners partition by distinct design row, so the encoding of the flag, an
#    indicator or a code, cannot change a prediction.

#: The flag and the dose of each support index ``k = 3 f + d``.
MIXED_CELLS = tuple((f, d) for f in range(2) for d in range(3))
#: The dose values of the grid.
MIXED_DOSES = (0.0, 1.0, 2.0)
#: ``g_1(k | w)`` in eighths, indexed ``[w, k]``: both components depend on ``W``.
MIXED_G1 = np.array([[1.0, 2.0, 1.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 1.0, 2.0, 1.0]]) / 8.0
_MIXED_ROWS = np.array(
    [
        [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
        [2.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 2.0, 2.0, 1.0],
        [2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
    ]
)


def _mixed_p_l2(w: int, k1: int) -> float:
    """``P(L_2 = 1 | w, k1)`` in quarters, reading both components of node 1."""
    f, d = MIXED_CELLS[k1]
    return (0.25, 0.50, 0.75, 0.50)[(w + f + 2 * d) % 4]


def _mixed_g2(w: int, k1: int, l2: int, k2: int) -> float:
    """``g_2(k2 | w, k1, l2)`` in eighths, which reads both components of node 1."""
    return float(_MIXED_ROWS[(w + k1 + 2 * l2 + 5) % 6][k2] / 8.0)


def _mixed_q(w: int, k1: int, l2: int, k2: int) -> float:
    """``E[Y | w, k1, l2, k2]`` in quarters; node 1 enters directly, through both components.

    Chosen by a search over modular patterns for the law whose plan means lie farthest apart:
    the natural course, "dose up", "flag gated" and its shifted-flag reading differ pairwise
    by at least 0.02.
    """
    f1, d1 = MIXED_CELLS[k1]
    f2, d2 = MIXED_CELLS[k2]
    return _Q_VALUES[(w + 2 * f1 + d1 + l2 + 2 * f2 + 2 * d2 * d2 + f2 * d2) % 3]


def _mixed_retained(w: int, k1: int) -> float:
    """``P(C_1 = 1 | w, k1)``: retention after node 1 reads both of its components."""
    f, d = MIXED_CELLS[k1]
    return (0.75, 0.50)[(w + f + d) % 2]


def _mixed_hazard(w: int, k1: int) -> float:
    """``P(Y_1 = 1 | w, k1)`` of the survival reading, which reads both components of node 1."""
    f, d = MIXED_CELLS[k1]
    return (0.25, 0.50)[(w + f + 2 * d) % 2] if d != 1 else 0.25 * (1 + f)


#: The three readings of the mixed law.  ``"end"`` has one outcome at the end.
#: ``"censored"`` adds ``C_1`` after node 1, and ``"survival"`` adds the event ``Y_1`` after
#: node 1, which is absorbing.  The middle field of a point is ``c1`` or ``y1``; a point that
#: left after node 1 holds ``-1`` for every later field.
MIXED_DESIGNS = ("end", "censored", "survival")


def _mixed_support(design: str) -> tuple[tuple[int, ...], ...]:
    later = tuple(itertools.product(range(2), range(6), range(2), range(6), range(2)))
    if design == "end":
        return later
    left = 0 if design == "censored" else 1
    return tuple(
        [(w, k1, left, -1, -1, -1) for w in range(2) for k1 in range(6)]
        + [(w, k1, 1 - left, l2, k2, y) for w, k1, l2, k2, y in later]
    )


def _mixed_names(design: str) -> tuple[str, ...]:
    middle = {"end": (), "censored": ("c1",), "survival": ("y1",)}[design]
    return ("w", "k1", *middle, "l2", "k2", "y")


def _mixed_cell(point: tuple[int, ...], design: str) -> float:
    names = _mixed_names(design)
    values = dict(zip(names, point, strict=True))
    w, k1 = values["w"], values["k1"]
    base = P_W[w] * MIXED_G1[w, k1]
    if design == "censored":
        stay = _mixed_retained(w, k1)
        if values["c1"] == 0:
            return float(base * (1.0 - stay))
        base = base * stay
    if design == "survival":
        hazard = _mixed_hazard(w, k1)
        if values["y1"] == 1:
            return float(base * hazard)
        base = base * (1.0 - hazard)
    l2, k2, y = values["l2"], values["k2"], values["y"]
    p_l2 = _mixed_p_l2(w, k1) if l2 else 1.0 - _mixed_p_l2(w, k1)
    q = _mixed_q(w, k1, l2, k2)
    return float(base * p_l2 * _mixed_g2(w, k1, l2, k2) * (q if y else 1.0 - q))


#: Rows of each reading: every cell probability is a multiple of ``1 / n``.
MIXED_N = {
    "end": 2 * 8 * 4 * 8 * 4,
    "censored": 2 * 8 * 4 * 4 * 8 * 4,
    "survival": 2 * 8 * 4 * 4 * 8 * 4,
}
MIXED_SUPPORT = {design: _mixed_support(design) for design in MIXED_DESIGNS}


def _mixed_counts(design: str) -> np.ndarray:
    n = MIXED_N[design]
    counts = np.array([_mixed_cell(point, design) * n for point in MIXED_SUPPORT[design]])
    assert np.allclose(counts, np.round(counts), atol=1e-9), "the law is not realised by n rows"
    assert int(np.round(counts).sum()) == n
    assert np.all(np.round(counts) > 0)
    return np.round(counts).astype(int)


MIXED_COUNTS = {design: _mixed_counts(design) for design in MIXED_DESIGNS}
MIXED_PROBS = {design: MIXED_COUNTS[design] / MIXED_N[design] for design in MIXED_DESIGNS}


def mixed_frame(design: str = "end") -> pd.DataFrame:
    """The rows realising one reading of the mixed law, in support order.

    The columns are ``W``, ``F1``, ``D1``, ``L2``, ``F2``, ``D2`` and ``Y``.  The censored
    reading adds ``C1`` and ``C2``, and nobody is censored at ``C2``.  The survival reading
    adds ``Y1`` and names the end outcome ``Y2``; a unit with ``Y1 = 1`` carries ``Y2 = 1``.
    A unit that left after node 1 holds ``nan`` at ``L2``, ``F2`` and ``D2``.
    """
    names = _mixed_names(design)
    rows = []
    for point, count in zip(MIXED_SUPPORT[design], MIXED_COUNTS[design], strict=True):
        values = dict(zip(names, point, strict=True))
        f1, d1 = MIXED_CELLS[values["k1"]]
        middle = float(values.get("c1", values.get("y1", 1)))
        if values["k2"] < 0:
            later = [np.nan, np.nan, np.nan, 1.0 if design == "survival" else np.nan]
        else:
            f2, d2 = MIXED_CELLS[values["k2"]]
            later = [float(values["l2"]), float(f2), MIXED_DOSES[d2], float(values["y"])]
        rows.extend([[float(values["w"]), float(f1), MIXED_DOSES[d1], middle, *later]] * int(count))
    raw = np.array(rows, dtype=float)
    columns: dict[str, Any] = {
        "W": raw[:, 0],
        "F1": raw[:, 1],
        "D1": raw[:, 2],
        "L2": raw[:, 4],
        "F2": raw[:, 5],
        "D2": raw[:, 6],
    }
    if design == "censored":
        columns["C1"] = raw[:, 3]
        columns["C2"] = np.where(raw[:, 3] == 1.0, 1.0, np.nan)
    if design == "survival":
        columns["Y1"] = raw[:, 3]
        columns["Y2"] = raw[:, 7]
    else:
        columns["Y"] = raw[:, 7]
    return pd.DataFrame(columns)


def mixed_sample(n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows from the end-of-study mixed law, for fits whose nuisances are not exact."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(MIXED_PROBS["end"], dtype=float)
    cells = rng.choice(len(MIXED_SUPPORT["end"]), size=n, p=probs)
    points = np.asarray(MIXED_SUPPORT["end"], dtype=int)[cells]
    cells1 = np.asarray(MIXED_CELLS)[points[:, 1]]
    cells2 = np.asarray(MIXED_CELLS)[points[:, 3]]
    return pd.DataFrame(
        {
            "W": points[:, 0].astype(float),
            "F1": cells1[:, 0].astype(float),
            "D1": np.asarray(MIXED_DOSES)[cells1[:, 1]],
            "L2": points[:, 2].astype(float),
            "F2": cells2[:, 0].astype(float),
            "D2": np.asarray(MIXED_DOSES)[cells2[:, 1]],
            "Y": points[:, 4].astype(float),
        }
    )


def _wide_q(d1: Any, d2: Any) -> Any:
    """``E[Y | ...]`` of :func:`mixed_wide_sample`: ``0.05 + 0.9 (d_1 d_2 / 4)^2``, from 0.05 to 0.95."""
    return 0.05 + 0.9 * (np.asarray(d1) * np.asarray(d2) / 4.0) ** 2


def mixed_wide_sample(n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows of the mixed law with an outcome that a dose policy moves widely.

    The covariates and both treatment nodes are :func:`mixed_sample`'s.  The outcome is
    redrawn from :func:`_wide_q`, which reads only the doses and is convex in their product.
    "dose up" moves the mean of a row at doses ``(1, 1)`` from 0.11 to 0.95.  A main-terms
    outcome regression misses the product, so each W8 fluctuation stays away from zero.
    """
    frame = mixed_sample(n, seed)
    rng = np.random.default_rng((seed, 8))
    mean = _wide_q(frame["D1"].to_numpy(), frame["D2"].to_numpy())
    return frame.assign(Y=(rng.random(n) < mean).astype(float))


def mixed_first_row_of(design: str = "end") -> np.ndarray:
    """Index of the first row of each support point of one reading of the mixed law."""
    return np.concatenate([[0], np.cumsum(MIXED_COUNTS[design])[:-1]]).astype(int)


def _mixed1(assign: Any) -> Any:
    return tuple(tuple(int(assign(w, k)) for k in range(6)) for w in range(2))


def _mixed2(assign: Any) -> Any:
    return tuple(
        tuple(
            tuple(tuple(int(assign(w, k1, l2, k)) for k in range(6)) for l2 in range(2))
            for k1 in range(6)
        )
        for w in range(2)
    )


def _index_of(f: int, d: int) -> int:
    return 3 * f + d


def _up(k: int) -> int:
    f, d = MIXED_CELLS[k]
    return _index_of(f, min(d + 1, 2))


def _flag_gated(k: int) -> int:
    """W4: set the flag to one, and raise the dose by one where the **natural** flag is one."""
    f, d = MIXED_CELLS[k]
    return _index_of(1, min(d + 1, 2) if f == 1 else d)


def _flag_gated_shifted(k: int) -> int:
    """The M3 reading of W4: the dose map reads the flag after the policy set it to one."""
    _, d = MIXED_CELLS[k]
    return _index_of(1, min(d + 1, 2))


def _dose_gated(k: int) -> int:
    """W10: keep the flag, and raise the dose by one where the flag is one."""
    f, d = MIXED_CELLS[k]
    return _index_of(f, min(d + 1, 2) if f == 1 else d)


def _flag_reads_dose(k: int) -> int:
    """W18: the flag goes to one where the **natural** dose is at least one; the dose rises."""
    f, d = MIXED_CELLS[k]
    return _index_of(1 if d >= 1 else f, min(d + 1, 2))


def _node_pair(assign: Any) -> tuple[Branches, Branches]:
    return (
        ((1.0, _mixed1(lambda w, k: assign(k))),),
        ((1.0, _mixed2(lambda w, k1, l2, k: assign(k))),),
    )


#: Every mixed plan, as its two node policies.
MIXED_PLANS: dict[str, tuple[Branches, Branches]] = {
    "natural": _node_pair(lambda k: k),
    "dose up": _node_pair(_up),
    "flag gated": _node_pair(_flag_gated),
    "flag gated (shifted flag)": _node_pair(_flag_gated_shifted),
    "dose gated": _node_pair(_dose_gated),
    "flag reads dose": _node_pair(_flag_reads_dose),
    "random": (
        ((0.25, _mixed1(lambda w, k: _up(k))), (0.75, _mixed1(lambda w, k: k))),
        ((0.25, _mixed2(lambda w, k1, l2, k: _up(k))), (0.75, _mixed2(lambda w, k1, l2, k: k))),
    ),
    # H5: a risk-ratio tilt of the flag at each node, delta 4 (keep with probability 1/4,
    # else set to one), crossed with the dose randomizer of "random".  The draws of the two
    # nodes are independent.
    "flag tilt up, random dose": tuple(
        tuple(
            (p, nest(assign))
            for p, assign in (
                (0.25 * 0.25, _up),
                (0.25 * 0.75, lambda k: k),
                (0.75 * 0.25, lambda k: _index_of(1, MIXED_CELLS[_up(k)][1])),
                (0.75 * 0.75, lambda k: _index_of(1, MIXED_CELLS[k][1])),
            )
        )
        for nest in (
            lambda assign: _mixed1(lambda w, k: assign(k)),
            lambda assign: _mixed2(lambda w, k1, l2, k: assign(k)),
        )
    ),
}


def _mixed_mass(probs: Any, design: str, **pattern: int) -> Any:
    return sum(probs[index] for index in _mixed_index(design, **pattern))


@cache
def _mixed_index(design: str, **pattern: int) -> tuple[int, ...]:
    names = _mixed_names(design)
    return tuple(
        index
        for index, point in enumerate(MIXED_SUPPORT[design])
        if all(point[names.index(name)] == value for name, value in pattern.items())
    )


def functional_mixed(
    probs: Any, plan: tuple[Branches, Branches], design: str = "end", horizon: int = 2
) -> Any:
    r"""The g-formula under a plan of joint policies on the mixed law, from cell masses.

    Each node's map reads the unit's natural index at that node, so a dose map reads the
    natural flag.  Node 2's map may read the intervened node-1 index.  On the censored
    reading every conditional after ``C_1`` is among the units still observed.  On the
    survival reading the parameter is the cumulative risk at ``horizon``:
    :math:`h_1 + (1 - h_1)\,E[Y_2 \mid Y_1 = 0, \ldots]` under the plan.
    """
    kept: dict[str, int] = {"end": {}, "censored": {"c1": 1}, "survival": {"y1": 0}}[design]
    total = _mixed_mass(probs, design)
    psi: Any = 0.0
    for w in range(2):
        p_w = _mixed_mass(probs, design, w=w) / total
        for k1 in range(6):
            g1 = _mixed_mass(probs, design, w=w, k1=k1) / _mixed_mass(probs, design, w=w)
            for p_e, map1 in plan[0]:
                k1d = map1[w][k1]
                at_node = _mixed_mass(probs, design, w=w, k1=k1d)
                reached = _mixed_mass(probs, design, w=w, k1=k1d, **kept)
                first: Any = 0.0
                if design == "survival":
                    first = _mixed_mass(probs, design, w=w, k1=k1d, y1=1) / at_node
                    if horizon == 1:
                        psi = psi + p_w * g1 * p_e * first
                        continue
                later: Any = 0.0
                for l2 in range(2):
                    stratum = _mixed_mass(probs, design, w=w, k1=k1d, **kept, l2=l2)
                    for k2 in range(6):
                        g2 = _mixed_mass(probs, design, w=w, k1=k1d, **kept, l2=l2, k2=k2) / stratum
                        for p_f, map2 in plan[1]:
                            k2d = map2[w][k1d][l2][k2]
                            cell = _mixed_mass(probs, design, w=w, k1=k1d, **kept, l2=l2, k2=k2d)
                            events = _mixed_mass(
                                probs, design, w=w, k1=k1d, **kept, l2=l2, k2=k2d, y=1
                            )
                            later = later + (stratum / reached) * g2 * p_f * events / cell
                survived = reached / at_node if design == "survival" else 1.0
                psi = psi + p_w * g1 * p_e * (first + survived * later)
    return psi


def mixed_truth(label: str, design: str = "end", horizon: int = 2) -> float:
    """The parameter of ``label`` at one reading of the mixed law."""
    probs = MIXED_PROBS[design]
    return float(np.real(functional_mixed(probs, MIXED_PLANS[label], design, horizon)))


@cache
def mixed_eif(label: str, design: str = "end", horizon: int = 2) -> np.ndarray:
    """The complex-step efficient influence function of ``label`` at every support point."""
    base = np.asarray(MIXED_PROBS[design], dtype=complex)
    step = 1e-30
    out = []
    for point in range(base.size):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        value = functional_mixed(perturbed, MIXED_PLANS[label], design, horizon)
        out.append(float(np.imag(value) / step))
    return np.array(out)
