r"""Categorical incremental interventions over time, on exactly realised three-level laws.

The implementation never enters this module.  Each parameter below is the g-formula under a
categorical odds tilt (Kennedy 2019, Theorem 1, with the finite partition of the arm; Kim,
Kennedy and Naimi 2021, Theorem 3.1 for dropout), written directly as ratios of
finite-support masses.  At a tilt node :math:`t` the plan draws the arm from

.. math::

    q_t(a \mid H_t) = \frac{\delta_{t,a}\, \pi_t(a \mid H_t)}{\sum_b \delta_{t,b}\, \pi_t(b \mid H_t)},

where :math:`\pi_t` is the treatment mechanism *of the law being evaluated*.  The functional
reads :math:`\pi_t` off the perturbed cell masses, so a complex-step Gateaux derivative
contains the mechanism term at every tilt node without being told about it.  A node named in
``known`` reads :math:`\pi_t` off the unperturbed law instead.  That is the model with a
known treatment mechanism at that node, whose influence function has no mechanism term there.

The laws share one node layout and differ in what follows the second treatment:

* :data:`LAW`, the **end-of-study law**:
  :math:`W \to A_1 \to C_1 \to L_2 \to A_2 \to C_2 \to Y`, with three treatment levels at
  each node, history-dependent dropout after each node, and every conditional on the quarter
  grid.  :data:`UNCENSORED` is the same law with no dropout.
* :data:`SURVIVAL`, with an absorbing event :math:`Y_1`, :math:`Y_2` at each node.
* :data:`COMPETING`, with two absorbing causes :math:`J_1`, :math:`J_2`.

Tilts are keyed by the analyst's labels.  The raw arm order of every law is
:data:`ARM_LABELS` = ``("standard", "high", "low")``, and ``CausalData`` sorts the labels to
``("high", "low", "standard")``.  A tilt resolved by position instead of by label lands on the
wrong level.

The module asserts at import that no term of the efficient influence function can vanish
unnoticed (:func:`_check_nonzero_witnesses`):

* each level's mechanism score coefficient, which is the mechanism term at that level of the
  node, is at least ``0.02`` in absolute value at some history with mass, at every node and
  for every tilt plan;
* each node's mechanism term holds at least ``0.05`` of the influence function's variance on the
  end-of-study laws and at least ``0.02`` on the survival and competing-risk laws;
* the tilt moves the mean by more than ``0.01`` from the natural course, and ignoring the
  factor of any one named level moves it by more than ``0.01`` (on the competing-risk law, the
  cumulative incidence of some cause);
* the complete-case functional of the dropout law differs from the parameter by more than
  ``0.01``, and the influence function of the complete-case functional differs from the
  parameter's by more than ``0.01`` somewhere, so a fit that drops the censoring factor
  fails.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

#: The raw arm order of every law.  The data layer sorts these labels.
ARM_LABELS: tuple[str, ...] = ("standard", "high", "low")
K = len(ARM_LABELS)

#: A node of a plan: ``("tilt", {label: factor})`` or a static arm ``("arm", raw index)``.
Node = tuple[str, Any]


def tilt(factors: dict[str, float]) -> Node:
    """An odds-tilt node.  A level the mapping does not name keeps the factor ``1``."""
    return ("tilt", dict(factors))


def arm(label: str) -> Node:
    """A static node, by label."""
    return ("arm", ARM_LABELS.index(label))


def factor_vector(factors: dict[str, float]) -> np.ndarray:
    """The factor of every level, on the raw arm axis (``ARM_LABELS`` order)."""
    return np.array([factors.get(label, 1.0) for label in ARM_LABELS], dtype=float)


def _q(node: Node, pi: list[Any]) -> list[Any]:
    """The plan's probability of every arm at a node with mechanism ``pi`` (one entry per arm)."""
    kind, value = node
    if kind == "arm":
        return [1.0 if a == int(value) else 0.0 for a in range(K)]
    f = factor_vector(value)
    normaliser = sum(f[a] * pi[a] for a in range(K))
    return [f[a] * pi[a] / normaliser for a in range(K)]


@cache
def _index(
    support: tuple[tuple[Any, ...], ...], names: tuple[str, ...], pattern: Any
) -> tuple[int, ...]:
    return tuple(
        position
        for position, point in enumerate(support)
        if all(point[names.index(node)] == value for node, value in pattern)
    )


def _masser(support: tuple[tuple[Any, ...], ...], names: tuple[str, ...]) -> Callable[..., Any]:
    def mass(probs: Any, **pattern: int) -> Any:
        return sum(probs[i] for i in _index(support, names, tuple(sorted(pattern.items()))))

    return mass


STEP = 1e-30


def eif(function: Callable[[Any], Any], probs: Any) -> np.ndarray:
    """The influence function of ``function`` at every support point, by complex step.

    Column ``j`` of the batch is the law moved toward point ``j``, and the functional
    broadcasts over the columns.  A point with no mass reports ``nan``: no row sits there.
    """
    probs = np.asarray(probs, dtype=float)
    size = len(probs)
    batch = (1.0 - 1j * STEP) * probs[:, None] + 1j * STEP * np.eye(size)
    values = np.imag(np.asarray(function(batch))) / STEP
    return np.where(probs > 0, values, np.nan)


def _exact_counts(masses: np.ndarray, n: int) -> np.ndarray:
    values = masses * n
    rounded = np.rint(values)
    if np.max(np.abs(values - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError("the cell probabilities are not multiples of 1/N")
    return rounded.astype(int)


def _bern(p: float, x: int) -> float:
    return p if x == 1 else 1.0 - p


# ------------------------------------------------------------------ the end-of-study law
#
#     W -> A1 -> C1 -> L2 -> A2 -> C2 -> Y        (A has three levels, the rest are binary)
#
# Every conditional is a multiple of 1/4, so 2 * 4**6 rows realise the law exactly.

N = 2 * 4**6
NAMES = ("w", "a1", "c1", "l2", "a2", "c2", "y")
COLUMNS = ("W", "A1", "C1", "L2", "A2", "C2", "Y")

SUPPORT: tuple[tuple[Any, ...], ...] = tuple(
    [(w, a1, 0, None, None, None, None) for w, a1 in itertools.product(range(2), range(K))]
    + [
        (w, a1, 1, l2, a2, 0, None)
        for w, a1, l2, a2 in itertools.product(range(2), range(K), range(2), range(K))
    ]
    + [
        (w, a1, 1, l2, a2, 1, y)
        for w, a1, l2, a2, y in itertools.product(range(2), range(K), range(2), range(K), range(2))
    ]
)
mass = _masser(SUPPORT, NAMES)

TABLES: dict[str, np.ndarray] = {}  # filled below


def _mass_of(point: tuple[Any, ...], t: dict[str, np.ndarray]) -> float:
    w, a1, c1, l2, a2, c2, y = point
    m = t["P_W"][w] * t["G1"][w, a1]
    if c1 == 0:
        return float(m * (1.0 - t["C1"][w, a1]))
    m *= t["C1"][w, a1]
    m *= _bern(t["P_L2"][w, a1], l2)
    m *= t["G2"][w, a1, l2, a2]
    if c2 == 0:
        return float(m * (1.0 - t["C2"][w, a1, l2, a2]))
    m *= t["C2"][w, a1, l2, a2]
    return float(m * _bern(t["Q"][w, a1, l2, a2], y))


def counts_of(t: dict[str, np.ndarray]) -> np.ndarray:
    """Rows per support point in an ``N``-row sample that realises the tables exactly."""
    return _exact_counts(np.array([_mass_of(point, t) for point in SUPPORT]), N)


def functional(
    probs: Any, plan: tuple[Node, Node], *, base: Any = None, known: tuple[int, ...] = ()
) -> Any:
    r""":math:`E[Y^{Q}]` on the end-of-study law, as a function of the masses.

    .. math::

        \sum_w P(w) \sum_{a_1} q_1(a_1 \mid w) \sum_l P(l \mid w, a_1, C_1 = 1)
        \sum_{a_2} q_2(a_2 \mid w, a_1, l)\, E[Y \mid w, a_1, l, a_2, C_2 = 1]

    ``base`` is the unperturbed law and ``known`` the nodes whose mechanism is read from it.
    """
    p = probs
    base = PROBS if base is None else base
    total = mass(p)
    psi = 0.0
    for w in range(2):
        source = base if 1 in known else p
        pi1 = [mass(source, w=w, a1=a) / mass(source, w=w) for a in range(K)]
        q1 = _q(plan[0], pi1)
        share = mass(p, w=w) / total
        for a1 in range(K):
            reached = mass(p, w=w, a1=a1, c1=1)
            for l2 in range(2):
                density = mass(p, w=w, a1=a1, c1=1, l2=l2) / reached
                source = base if 2 in known else p
                denominator = mass(source, w=w, a1=a1, c1=1, l2=l2)
                pi2 = [mass(source, w=w, a1=a1, c1=1, l2=l2, a2=a) / denominator for a in range(K)]
                q2 = _q(plan[1], pi2)
                for a2 in range(K):
                    uncensored = mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1)
                    events = mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, y=1)
                    psi = psi + share * q1[a1] * density * q2[a2] * (events / uncensored)
    return psi


def complete_case(probs: Any, plan: tuple[Node, Node], *, known: tuple[int, ...] = ()) -> Any:
    r"""The same functional on the units that never dropped out, ignoring the censoring.

    It is the parameter of a fit that restricts to the rows with :math:`C_1 = C_2 = 1` and
    models no censoring: the law of those rows, with the mechanisms read from it.
    """
    p = probs
    kept = mass(p, c1=1, c2=1)
    total = kept
    psi = 0.0
    base = PROBS
    for w in range(2):
        source = base if 1 in known else p
        pi1 = [
            mass(source, w=w, a1=a, c1=1, c2=1) / mass(source, w=w, c1=1, c2=1) for a in range(K)
        ]
        q1 = _q(plan[0], pi1)
        share = mass(p, w=w, c1=1, c2=1) / total
        for a1 in range(K):
            reached = mass(p, w=w, a1=a1, c1=1, c2=1)
            for l2 in range(2):
                density = mass(p, w=w, a1=a1, c1=1, c2=1, l2=l2) / reached
                source = base if 2 in known else p
                denominator = mass(source, w=w, a1=a1, c1=1, c2=1, l2=l2)
                pi2 = [
                    mass(source, w=w, a1=a1, c1=1, c2=1, l2=l2, a2=a) / denominator
                    for a in range(K)
                ]
                q2 = _q(plan[1], pi2)
                for a2 in range(K):
                    cell = mass(p, w=w, a1=a1, c1=1, c2=1, l2=l2, a2=a2)
                    events = mass(p, w=w, a1=a1, c1=1, c2=1, l2=l2, a2=a2, y=1)
                    psi = psi + share * q1[a1] * density * q2[a2] * (events / cell)
    return psi


# ------------------------------------------------------------------ survival and competing
#
#     W -> A1 -> C1 -> Y1 -> L2 -> A2 -> C2 -> Y2        (survival; Y is absorbing)
#     W -> A1 -> C1 -> J1 -> L2 -> A2 -> C2 -> J2        (competing; J in {0, relapse, death})
#
# A unit with an event at the first node has no later node.

SURVIVAL_N = 2 * 4**7
SURVIVAL_NAMES = ("w", "a1", "c1", "y1", "l2", "a2", "c2", "y2")
SURVIVAL_COLUMNS = ("W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")
SURVIVAL_SUPPORT: tuple[tuple[Any, ...], ...] = tuple(
    [(w, a1, 0, None, None, None, None, None) for w, a1 in itertools.product(range(2), range(K))]
    + [(w, a1, 1, 1, None, None, None, None) for w, a1 in itertools.product(range(2), range(K))]
    + [
        (w, a1, 1, 0, l2, a2, 0, None)
        for w, a1, l2, a2 in itertools.product(range(2), range(K), range(2), range(K))
    ]
    + [
        (w, a1, 1, 0, l2, a2, 1, y2)
        for w, a1, l2, a2, y2 in itertools.product(range(2), range(K), range(2), range(K), range(2))
    ]
)
survival_mass = _masser(SURVIVAL_SUPPORT, SURVIVAL_NAMES)

#: The two absorbing causes, in report order.  A cause's code is its position plus one.
CAUSES: tuple[str, ...] = ("relapse", "death")
COMPETING_N = 4**8
COMPETING_NAMES = ("w", "a1", "c1", "j1", "l2", "a2", "c2", "j2")
COMPETING_COLUMNS = ("W", "A1", "C1", "R1", "D1", "L2", "A2", "C2", "R2", "D2")
COMPETING_SUPPORT: tuple[tuple[Any, ...], ...] = tuple(
    [(w, a1, 0, None, None, None, None, None) for w, a1 in itertools.product(range(2), range(K))]
    + [
        (w, a1, 1, j1, None, None, None, None)
        for w, a1, j1 in itertools.product(range(2), range(K), range(1, len(CAUSES) + 1))
    ]
    + [
        (w, a1, 1, 0, l2, a2, 0, None)
        for w, a1, l2, a2 in itertools.product(range(2), range(K), range(2), range(K))
    ]
    + [
        (w, a1, 1, 0, l2, a2, 1, j2)
        for w, a1, l2, a2, j2 in itertools.product(
            range(2), range(K), range(2), range(K), range(len(CAUSES) + 1)
        )
    ]
)
competing_mass = _masser(COMPETING_SUPPORT, COMPETING_NAMES)


def _event_mass_of(point: tuple[Any, ...], t: dict[str, np.ndarray], cause_count: int) -> float:
    """``P`` of one support point of the survival (one cause) or competing (two) law."""
    w, a1, c1, e1, l2, a2, c2, e2 = point
    m = t["P_W"][w] * t["G1"][w, a1]
    if c1 == 0:
        return float(m * (1.0 - t["C1"][w, a1]))
    m *= t["C1"][w, a1]
    h1 = t["H1"].reshape(cause_count, 2, K)[:, w, a1]
    if e1 != 0:
        return float(m * h1[(e1 if cause_count > 1 else 1) - 1])
    m *= 1.0 - h1.sum()
    m *= _bern(t["P_L2"][w, a1], l2)
    m *= t["G2"][w, a1, l2, a2]
    if c2 == 0:
        return float(m * (1.0 - t["C2"][w, a1, l2, a2]))
    m *= t["C2"][w, a1, l2, a2]
    h2 = t["H2"].reshape(cause_count, 2, K, 2, K)[:, w, a1, l2, a2]
    if e2 != 0:
        return float(m * h2[(e2 if cause_count > 1 else 1) - 1])
    return float(m * (1.0 - h2.sum()))


def survival_counts_of(t: dict[str, np.ndarray]) -> np.ndarray:
    """Rows per support point of the survival law."""
    return _exact_counts(np.array([_event_mass_of(p, t, 1) for p in SURVIVAL_SUPPORT]), SURVIVAL_N)


def competing_counts_of(t: dict[str, np.ndarray]) -> np.ndarray:
    """Rows per support point of the competing law."""
    return _exact_counts(
        np.array([_event_mass_of(p, t, len(CAUSES)) for p in COMPETING_SUPPORT]), COMPETING_N
    )


def functional_event(
    probs: Any,
    plan: tuple[Node, Node],
    horizon: int,
    *,
    cause: int | None = None,
    base: Any = None,
    known: tuple[int, ...] = (),
) -> Any:
    r"""The cumulative incidence of one cause by ``horizon`` under a tilt plan.

    ``cause=None`` reads the survival law, whose event nodes take the values 0 and 1.  A
    cause code reads the competing law.  The second node's population is the units free of
    *every* cause after the first node, and the cause-specific events are the numerator:

    .. math::

        F(2) = \sum_w P(w) \sum_{a_1} q_1 \Bigl[h_1 + S_1 \sum_l P(l \mid \cdot)
               \sum_{a_2} q_2\, h_2\Bigr].
    """
    survival_law = cause is None
    event_mass = survival_mass if survival_law else competing_mass
    event = "y" if survival_law else "j"
    code = 1 if survival_law else cause
    base = (SURVIVAL_PROBS if survival_law else COMPETING_PROBS) if base is None else base
    p = probs
    total = event_mass(p)
    psi = 0.0
    for w in range(2):
        source = base if 1 in known else p
        pi1 = [event_mass(source, w=w, a1=a) / event_mass(source, w=w) for a in range(K)]
        q1 = _q(plan[0], pi1)
        share = event_mass(p, w=w) / total
        for a1 in range(K):
            reached = event_mass(p, w=w, a1=a1, c1=1)
            hazard1 = event_mass(p, w=w, a1=a1, c1=1, **{f"{event}1": code}) / reached
            if horizon == 1:
                psi = psi + share * q1[a1] * hazard1
                continue
            free = {f"{event}1": 0}
            survived = event_mass(p, w=w, a1=a1, c1=1, **free)
            later = 0.0
            for l2 in range(2):
                density = event_mass(p, w=w, a1=a1, c1=1, l2=l2, **free) / survived
                source = base if 2 in known else p
                denominator = event_mass(source, w=w, a1=a1, c1=1, l2=l2, **free)
                pi2 = [
                    event_mass(source, w=w, a1=a1, c1=1, l2=l2, a2=a, **free) / denominator
                    for a in range(K)
                ]
                q2 = _q(plan[1], pi2)
                for a2 in range(K):
                    uncensored = event_mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, **free)
                    events = event_mass(
                        p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, **free, **{f"{event}2": code}
                    )
                    later = later + density * q2[a2] * (events / uncensored)
            psi = psi + share * q1[a1] * (hazard1 + (survived / reached) * later)
    return psi


# ------------------------------------------------------------------ the tables
#
# The tables came from a hill climb over the quarter grid (``tests/unit/test_categorical_ipsi_laws.py``
# asserts the result).  It maximised the smallest margin of the rules in the module docstring.

TABLES = {
    "P_W": np.array([0.5, 0.5]),
    "G1": np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]),
    "C1": np.array([[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]]),
    "P_L2": np.array([[0.75, 0.25, 0.25], [0.5, 0.25, 0.75]]),
    "G2": np.array(
        [
            [
                [[0.25, 0.5, 0.25], [0.25, 0.25, 0.5]],
                [[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]],
                [[0.25, 0.5, 0.25], [0.25, 0.25, 0.5]],
            ],
            [
                [[0.25, 0.25, 0.5], [0.25, 0.5, 0.25]],
                [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]],
                [[0.25, 0.25, 0.5], [0.25, 0.5, 0.25]],
            ],
        ]
    ),
    "C2": np.array(
        [
            [
                [[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]],
                [[0.5, 0.5, 0.75], [0.75, 0.75, 0.75]],
                [[0.75, 0.75, 0.5], [0.75, 0.75, 0.75]],
            ],
            [
                [[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]],
                [[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]],
                [[0.75, 0.75, 0.75], [0.75, 0.75, 0.75]],
            ],
        ]
    ),
    "Q": np.array(
        [
            [
                [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
                [[0.75, 0.75, 0.75], [0.75, 0.75, 0.25]],
                [[0.25, 0.25, 0.75], [0.25, 0.25, 0.25]],
            ],
            [
                [[0.25, 0.25, 0.75], [0.25, 0.5, 0.75]],
                [[0.25, 0.75, 0.75], [0.25, 0.75, 0.25]],
                [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
            ],
        ]
    ),
}

COUNTS = counts_of(TABLES)
PROBS = COUNTS / N

#: The same law with no dropout: ``C1 = C2 = 1`` with probability one.
UNCENSORED_TABLES = {
    **TABLES,
    "C1": np.ones_like(TABLES["C1"]),
    "C2": np.ones_like(TABLES["C2"]),
}
UNCENSORED_COUNTS = counts_of(UNCENSORED_TABLES)
UNCENSORED_PROBS = UNCENSORED_COUNTS / N

#: The survival and competing laws share the treatment, covariate and dropout tables with
#: the end-of-study law and add the hazards.
SURVIVAL_TABLES = {
    **TABLES,
    **{
        "H1": np.array([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]),
        "H2": np.array(
            [
                [
                    [[0.75, 0.25, 0.75], [0.5, 0.25, 0.75]],
                    [[0.75, 0.75, 0.5], [0.25, 0.75, 0.25]],
                    [[0.25, 0.25, 0.75], [0.75, 0.25, 0.25]],
                ],
                [
                    [[0.75, 0.25, 0.25], [0.75, 0.25, 0.75]],
                    [[0.75, 0.5, 0.75], [0.25, 0.75, 0.75]],
                    [[0.25, 0.25, 0.5], [0.25, 0.25, 0.75]],
                ],
            ]
        ),
    },
}
SURVIVAL_COUNTS = survival_counts_of(SURVIVAL_TABLES)
SURVIVAL_PROBS = SURVIVAL_COUNTS / SURVIVAL_N
HORIZONS: tuple[int, ...] = (1, 2)

COMPETING_TABLES = {
    **TABLES,
    "P_W": np.array([0.25, 0.75]),
    **{
        "H1": np.array(
            [[[0.5, 0.0, 0.0], [0.25, 0.0, 0.5]], [[0.25, 0.5, 0.75], [0.0, 0.25, 0.0]]]
        ),
        "H2": np.array(
            [
                [
                    [
                        [[0.5, 0.0, 0.25], [0.0, 0.25, 0.75]],
                        [[0.0, 0.0, 0.25], [0.25, 0.5, 0.5]],
                        [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25]],
                    ],
                    [
                        [[0.25, 0.75, 0.75], [0.25, 0.5, 0.0]],
                        [[0.25, 0.5, 0.0], [0.5, 0.25, 0.25]],
                        [[0.75, 0.0, 0.0], [0.25, 0.75, 0.75]],
                    ],
                ],
                [
                    [
                        [[0.0, 0.25, 0.25], [0.5, 0.0, 0.0]],
                        [[0.5, 0.25, 0.25], [0.0, 0.0, 0.25]],
                        [[0.25, 0.25, 0.75], [0.0, 0.0, 0.25]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.5]],
                        [[0.0, 0.25, 0.75], [0.0, 0.25, 0.25]],
                        [[0.0, 0.25, 0.25], [0.25, 0.0, 0.0]],
                    ],
                ],
            ]
        ),
    },
}
COMPETING_COUNTS = competing_counts_of(COMPETING_TABLES)
COMPETING_PROBS = COMPETING_COUNTS / COMPETING_N


# ------------------------------------------------------------------ plans and names

#: The tilts, by label.  A level a tilt does not name keeps the factor ``1``.
T_A: dict[str, float] = {"high": 2.0, "low": 0.5}
T_B: dict[str, float] = {"standard": 0.25}
NATURAL: dict[str, float] = {"high": 1.0}

#: The plans of the end-of-study law, by the label the estimator side gives them.  They cover
#: a constant tilt, a per-node pair, a tilt then a static node, and a static node then a tilt
#: (the label factor enters :math:`W_{t-1}` at a tilt node).
PLANS: dict[str, tuple[Node, Node]] = {
    "natural": (tilt(NATURAL), tilt(NATURAL)),
    "T_a": (tilt(T_A), tilt(T_A)),
    "T_b": (tilt(T_B), tilt(T_B)),
    "per_node": (tilt(T_A), tilt(T_B)),
    "tilt_then_static": (tilt(T_A), arm("low")),
    "static_then_tilt": (arm("high"), tilt(T_B)),
}
REFERENCE = "natural"

#: The plans of the survival and competing laws.
EVENT_PLANS: dict[str, tuple[Node, Node]] = {
    label: PLANS[label] for label in ("natural", "T_a", "per_node", "static_then_tilt")
}


def names(plans: dict[str, Any], reference: str, horizons: tuple[int, ...] = ()) -> tuple[str, ...]:
    """The names a fit of ``plans`` reports, in report order.

    With ``horizons`` the names are survival risks, regimen outer and horizon inner.
    """
    if not horizons:
        return tuple(f"ey_regimen[{label}]" for label in plans) + tuple(
            f"ate_regimen[{label} vs {reference}]" for label in plans if label != reference
        )
    return tuple(
        f"risk_regimen[{label} @ t={horizon}]" for label in plans for horizon in horizons
    ) + tuple(
        f"ate_regimen[{label} vs {reference} @ t={horizon}]"
        for label in plans
        if label != reference
        for horizon in horizons
    )


def competing_names(reference: str = REFERENCE) -> tuple[str, ...]:
    """The names a competing-risk fit of :data:`EVENT_PLANS` reports, in report order."""
    return tuple(
        f"cif_regimen[{label}, {cause} @ t={horizon}]"
        for label in EVENT_PLANS
        for cause in CAUSES
        for horizon in HORIZONS
    ) + tuple(
        f"ate_regimen[{label} vs {reference}, {cause} @ t={horizon}]"
        for label in EVENT_PLANS
        if label != reference
        for cause in CAUSES
        for horizon in HORIZONS
    )


def _contrast(name: str, mean: Callable[[str, str], Any]) -> Any:
    """``mean(label, suffix)`` for a mean, and the difference of two for a contrast."""
    head, _, index = name.partition("[")
    index = index[:-1]
    label, at, horizon = index.partition(" @ ")
    suffix = at + horizon
    if head == "ate_regimen":
        left, right = label.split(" vs ")
        return mean(left, suffix) - mean(right, suffix)
    return mean(label, suffix)


def end_function(
    name: str,
    *,
    plans: dict[str, tuple[Node, Node]] = PLANS,
    base: Any = None,
    known: tuple[int, ...] = (),
) -> Callable[[Any], Any]:
    """The functional behind ``name`` on the end-of-study law, as a function of the masses."""

    def at(probs: Any) -> Any:
        return _contrast(
            name, lambda label, _: functional(probs, plans[label], base=base, known=known)
        )

    return at


def survival_function(name: str, *, known: tuple[int, ...] = ()) -> Callable[[Any], Any]:
    """The functional behind a survival ``name``, as a function of the masses."""

    def at(probs: Any) -> Any:
        return _contrast(
            name,
            lambda label, suffix: functional_event(
                probs, EVENT_PLANS[label], int(suffix.rpartition("=")[2]), known=known
            ),
        )

    return at


def competing_function(name: str, *, known: tuple[int, ...] = ()) -> Callable[[Any], Any]:
    """The functional behind a competing-risk ``name``, as a function of the masses."""

    def at(probs: Any) -> Any:
        head, _, index = name.partition("[")
        index = index[:-1]
        rest, _, horizon = index.rpartition(" @ t=")
        label, _, cause = rest.rpartition(", ")
        code = CAUSES.index(cause) + 1

        def mean(plan_label: str) -> Any:
            return functional_event(
                probs, EVENT_PLANS[plan_label], int(horizon), cause=code, known=known
            )

        if head == "ate_regimen":
            left, right = label.split(" vs ")
            return mean(left) - mean(right)
        return mean(label)

    return at


# ------------------------------------------------------------------ frames


def first_rows(counts: np.ndarray) -> np.ndarray:
    """The first row of each support point.  A point with no rows reports ``-1``."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.where(counts > 0, starts, -1)


def _frame(
    support: tuple[tuple[Any, ...], ...],
    counts: np.ndarray,
    node_names: tuple[str, ...],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Rows laid out in support order; an arm column holds the label, a missing node ``nan``."""
    cells = np.repeat(np.arange(len(support)), counts)
    out: dict[str, np.ndarray] = {}
    for name in columns:
        node = name.lower()
        if node in ("r1", "d1", "r2", "d2"):
            position, code = node_names.index("j" + node[1]), 1 if node[0] == "r" else 2
        else:
            position, code = node_names.index(node), None
        values = []
        for point in support:
            raw = point[position]
            if raw is None:
                values.append(np.nan)
            elif node in ("a1", "a2"):
                values.append(ARM_LABELS[raw])
            elif code is not None:
                values.append(float(raw == code))
            else:
                values.append(float(raw))
        out[name] = np.array(values, dtype=object if node in ("a1", "a2") else float)[cells]
    return pd.DataFrame(out)


def end_frame() -> pd.DataFrame:
    """The sample that realises the end-of-study law exactly, dropout included."""
    return _frame(SUPPORT, COUNTS, NAMES, COLUMNS)


def uncensored_frame() -> pd.DataFrame:
    """The sample that realises the law with no dropout; the censoring columns are dropped."""
    return _frame(SUPPORT, UNCENSORED_COUNTS, NAMES, COLUMNS).drop(columns=["C1", "C2"])


def survival_frame() -> pd.DataFrame:
    """The sample that realises the survival law exactly."""
    return _frame(SURVIVAL_SUPPORT, SURVIVAL_COUNTS, SURVIVAL_NAMES, SURVIVAL_COLUMNS)


def competing_frame() -> pd.DataFrame:
    """The sample that realises the competing law: one indicator column per cause per node."""
    return _frame(COMPETING_SUPPORT, COMPETING_COUNTS, COMPETING_NAMES, COMPETING_COLUMNS)


def competing_outcome_columns() -> dict[str, list[str]]:
    """``outcome=`` for a fit on the competing law: a cause to its indicator per node."""
    return {"relapse": ["R1", "R2"], "death": ["D1", "D2"]}


# ------------------------------------------------------------------ the nonzero witnesses

#: The smallest values that keep a term of the influence function from vanishing unnoticed.
MIN_COEFFICIENT = 0.02
MIN_SHARE = 0.05
MIN_EVENT_SHARE = 0.02
MIN_MOVE = 0.01


def _arm_at(support: tuple[tuple[Any, ...], ...], names: tuple[str, ...], node: int) -> np.ndarray:
    """The raw arm index of treatment node ``node`` at each support point, ``-1`` if unreached."""
    position = names.index(f"a{node}")
    return np.array([-1 if point[position] is None else point[position] for point in support])


def _family_margins(
    label: str,
    make: Callable[..., Callable[[Any], Any]],
    probs: np.ndarray,
    support: tuple[tuple[Any, ...], ...],
    names: tuple[str, ...],
    entries: tuple[tuple[str, tuple[Node, Node], bool], ...],
    share: float,
    nodes: tuple[int, ...] = (1, 2),
) -> dict[str, float]:
    """Each rule of the module docstring as a margin: the value over its threshold.

    ``make(plan, known)`` is the functional of a plan with the mechanisms of the nodes in
    ``known`` read off the unperturbed law.  A margin of at least ``1`` passes.
    """
    natural = (tilt(NATURAL), tilt(NATURAL))
    margins: dict[str, float] = {}
    keep = probs > 0
    for plan_label, plan, ignore in entries:
        tag = f"{label}/{plan_label}"
        psi = float(make(plan, ())(probs))
        margins[f"{tag}: moves the mean"] = abs(psi - float(make(natural, ())(probs))) / MIN_MOVE
        full = eif(make(plan, ()), probs)
        mean = np.nansum(probs * full)
        variance = np.nansum(probs * (full - mean) ** 2)
        for node in nodes:
            if plan[node - 1][0] != "tilt":
                continue
            term = full - eif(make(plan, (node,)), probs)
            arms = _arm_at(support, names, node)
            for level in range(K):
                chosen = (arms == level) & keep
                if chosen.any():
                    margins[f"{tag}: node {node} coefficient of level {level}"] = (
                        float(np.nanmax(np.abs(term[chosen]))) / MIN_COEFFICIENT
                    )
            term_mean = np.nansum(probs * term)
            term_variance = np.nansum(probs * (term - term_mean) ** 2)
            margins[f"{tag}: node {node} share of the variance"] = (
                float(term_variance / variance) / share
            )
            if ignore:
                for named in plan[node - 1][1]:
                    reduced = {k: v for k, v in plan[node - 1][1].items() if k != named}
                    altered = list(plan)
                    altered[node - 1] = tilt(reduced or NATURAL)
                    other = float(make((altered[0], altered[1]), ())(probs))
                    margins[f"{tag}: node {node} factor of {named!r} moves the mean"] = (
                        abs(other - psi) / MIN_MOVE
                    )
    return margins


def nonzero_margins() -> dict[str, float]:
    """Every nonzero rule of every law as a margin.  A margin of at least ``1`` passes."""
    tilted = (
        ("T_a", PLANS["T_a"], True),
        ("T_b", PLANS["T_b"], True),
        ("per_node", PLANS["per_node"], True),
    )
    margins: dict[str, float] = {}
    for label, probs in (("end", PROBS), ("uncensored", UNCENSORED_PROBS)):
        margins.update(
            _family_margins(
                label,
                lambda plan, known, base=probs: (
                    lambda p: functional(p, plan, base=base, known=known)
                ),
                probs,
                SUPPORT,
                NAMES,
                tilted,
                MIN_SHARE,
            )
        )
    # The dropout rules: a fit that ignores the censoring is a different parameter, and a curve
    # that drops the censoring factor is a different curve.
    for plan_label, plan, _ in tilted:
        psi = float(functional(PROBS, plan))
        complete = float(complete_case(PROBS, plan))
        margins[f"end/{plan_label}: complete case differs"] = abs(complete - psi) / MIN_MOVE
        curve = eif(lambda p, plan=plan: functional(p, plan, base=PROBS), PROBS)
        complete_curve = eif(lambda p, plan=plan: complete_case(p, plan), PROBS)
        gap = np.nanmax(np.abs(curve - complete_curve))
        margins[f"end/{plan_label}: complete-case curve differs"] = float(gap) / MIN_MOVE
    event_entries = (
        ("T_a", PLANS["T_a"], True),
        ("per_node", PLANS["per_node"], True),
        ("static_then_tilt", PLANS["static_then_tilt"], False),
    )
    for horizon in HORIZONS:
        nodes = (1,) if horizon == 1 else (1, 2)
        entries = (
            tuple((label, plan, False) for label, plan, _ in event_entries[:1])
            if horizon == 1
            else event_entries
        )
        margins.update(
            _family_margins(
                f"survival h{horizon}",
                lambda plan, known, h=horizon: (
                    lambda p: functional_event(p, plan, h, base=SURVIVAL_PROBS, known=known)
                ),
                SURVIVAL_PROBS,
                SURVIVAL_SUPPORT,
                SURVIVAL_NAMES,
                entries,
                MIN_EVENT_SHARE,
                nodes,
            )
        )
        competing_entries = tuple((label, plan, False) for label, plan, _ in entries)
        for code, cause in enumerate(CAUSES, start=1):
            margins.update(
                _family_margins(
                    f"competing {cause} h{horizon}",
                    lambda plan, known, h=horizon, c=code: (
                        lambda p: functional_event(
                            p, plan, h, cause=c, base=COMPETING_PROBS, known=known
                        )
                    ),
                    COMPETING_PROBS,
                    COMPETING_SUPPORT,
                    COMPETING_NAMES,
                    competing_entries,
                    MIN_EVENT_SHARE,
                    nodes,
                )
            )
        # A level's factor must move the cumulative incidence of some cause.
        for plan_label, plan, ignore in entries:
            if not ignore:
                continue
            for node in (1, 2):
                for named in plan[node - 1][1]:
                    reduced = {k: v for k, v in plan[node - 1][1].items() if k != named}
                    altered = list(plan)
                    altered[node - 1] = tilt(reduced or NATURAL)
                    moves = [
                        abs(
                            float(
                                functional_event(
                                    COMPETING_PROBS,
                                    (altered[0], altered[1]),
                                    horizon,
                                    cause=code,
                                    base=COMPETING_PROBS,
                                )
                            )
                            - float(
                                functional_event(
                                    COMPETING_PROBS, plan, horizon, cause=code, base=COMPETING_PROBS
                                )
                            )
                        )
                        for code in range(1, len(CAUSES) + 1)
                    ]
                    margins[
                        f"competing h{horizon}/{plan_label}: node {node} factor of {named!r} "
                        "moves the mean of some cause"
                    ] = max(moves) / MIN_MOVE
    return margins


def _check_nonzero_witnesses() -> None:
    """Refuse an import whose tables let a term of the influence function vanish."""
    failing = {name: value for name, value in nonzero_margins().items() if value < 1.0}
    if failing:  # pragma: no cover - guards the constants
        raise AssertionError(f"the tables violate the nonzero rules: {failing}")


_check_nonzero_witnesses()
