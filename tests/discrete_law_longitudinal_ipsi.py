r"""Incremental propensity-score interventions on exactly realised longitudinal laws.

The implementation never enters this module.  Each parameter below is the g-formula under
an odds tilt (Kennedy 2019, Theorem 1; Kim, Kennedy and Naimi 2021, Theorem 3.1), written
directly as ratios of finite-support masses.  Complex-step derivatives of it supply the
efficient influence function independently of the fitted weights and recursion.

A tilt node multiplies the odds of treatment by :math:`\delta_t`:

.. math::

    q_t(1 \mid H_t) = \frac{\delta_t \pi_t(H_t)}{\delta_t \pi_t(H_t) + 1 - \pi_t(H_t)},

where :math:`\pi_t` is the treatment mechanism *of the law being evaluated*.  The functional
reads :math:`\pi_t` off the perturbed cell masses, so the Gateaux derivative contains the
mechanism term at every tilt node without being told about it.  With ``known=True`` the
functional reads :math:`\pi_t` off the unperturbed law instead.  That is the model with a
known treatment mechanism, whose influence function has no mechanism term.

Dropout is prevented in every plan: the functional conditions on ``C_t = 1`` at each node.

The laws:

* :mod:`tests.discrete_law_longitudinal`, the two-node censored end-of-study law.  Its plans
  are :data:`PLANS`, and :func:`functional_binary` is the oracle.
* :mod:`tests.discrete_law_survival` and :mod:`tests.discrete_law_competing`, by
  :func:`functional_survival` and :func:`functional_competing`.
* The three-node law of :func:`three_node_frame`, with a nonzero tilt at every node.
* The no-positivity law of :func:`positivity_frame`, with one cell at :math:`\pi = 1/32`.

A plan is a tuple with one entry per node.  A float entry is an odds tilt with that
:math:`\delta`.  An ``int`` entry is a static arm.  Every plan here is a constant of the
module, so no comparison ever touches the perturbed masses.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from . import discrete_law_competing as competing
from . import discrete_law_longitudinal as binary
from . import discrete_law_survival as survival

#: A node of a plan: an odds tilt ``("tilt", delta)`` or a static arm ``("arm", a)``.
Node = tuple[str, float]


def tilt(delta: float) -> Node:
    """An odds-tilt node."""
    return ("tilt", float(delta))


def arm(a: int) -> Node:
    """A static node."""
    return ("arm", float(a))


#: The plans of the two-node law, by the label the estimator side gives them.  They cover
#: constant tilts, a per-node tilt, a tilt then a static node, a static node then a tilt
#: (the M16 witness: the label factor enters :math:`W_{t-1}` at a tilt node), and a static
#: plan beside them.
PLANS: dict[str, tuple[Node, Node]] = {
    "natural": (tilt(1.0), tilt(1.0)),
    "half": (tilt(0.5), tilt(0.5)),
    "double": (tilt(2.0), tilt(2.0)),
    "triple": (tilt(3.0), tilt(3.0)),
    "per_node": (tilt(2.0), tilt(0.5)),
    "tilt_then_never": (tilt(2.0), arm(0)),
    "always_then_tilt": (arm(1), tilt(3.0)),
    "never": (arm(0), arm(0)),
}
REFERENCE = "natural"


def _q(node: Node, pi: Any, a: int) -> Any:
    """The plan's probability of arm ``a`` at a node with mechanism ``pi``."""
    kind, value = node
    if kind == "arm":
        return 1.0 if a == int(value) else 0.0
    treated = value * pi / (value * pi + 1.0 - pi)
    return treated if a == 1 else 1.0 - treated


@cache
def _index(support: tuple[tuple[Any, ...], ...], names: tuple[str, ...], pattern: Any) -> Any:
    return tuple(
        position
        for position, point in enumerate(support)
        if all(point[names.index(node)] == value for node, value in pattern)
    )


def _masser(support: tuple[tuple[Any, ...], ...], names: tuple[str, ...]) -> Callable[..., Any]:
    def mass(probs: Any, **pattern: int) -> Any:
        return sum(probs[i] for i in _index(support, names, tuple(sorted(pattern.items()))))

    return mass


_BINARY_NAMES = ("w", "a1", "c1", "l2", "a2", "c2", "y")
_binary_mass = _masser(binary.SUPPORT, _BINARY_NAMES)


def functional_binary(
    probs: Any,
    plan: tuple[Node, Node],
    *,
    base: Any = None,
    known: tuple[int, ...] = (),
) -> Any:
    r""":math:`E[Y^{Q}]` on a two-node law over :data:`tests.discrete_law_longitudinal.SUPPORT`.

    .. math::

        \sum_w P(w) \sum_{a_1} q_1(a_1 \mid w) \sum_{l} P(l \mid w, a_1, C_1 = 1)
        \sum_{a_2} q_2(a_2 \mid w, a_1, l)\, E[Y \mid w, a_1, l, a_2, C_2 = 1]

    ``base`` is the unperturbed law, :data:`tests.discrete_law_longitudinal.PROBS` unless a
    caller names another.  A node in ``known`` reads :math:`\pi_t` off ``base``.  A history
    with no mass in ``base`` is skipped.  The plan gives it no weight there, so the skip is a
    constant of the law and not a comparison on the perturbed masses.
    """
    p = probs
    base = binary.PROBS if base is None else base
    total = _binary_mass(p)
    psi = 0.0
    for w in (0, 1):
        source = base if 1 in known else p
        pi1 = _binary_mass(source, w=w, a1=1) / _binary_mass(source, w=w)
        for a1 in (0, 1):
            q1 = _q(plan[0], pi1, a1)
            if _binary_mass(base, w=w, a1=a1, c1=1) == 0:
                continue
            reached = _binary_mass(p, w=w, a1=a1, c1=1)
            for l2 in (0, 1):
                density = _binary_mass(p, w=w, a1=a1, c1=1, l2=l2) / reached
                source = base if 2 in known else p
                pi2 = _binary_mass(source, w=w, a1=a1, c1=1, l2=l2, a2=1) / _binary_mass(
                    source, w=w, a1=a1, c1=1, l2=l2
                )
                for a2 in (0, 1):
                    if _binary_mass(base, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1) == 0:
                        continue
                    q2 = _q(plan[1], pi2, a2)
                    uncensored = _binary_mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1)
                    events = _binary_mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, y=1)
                    share = _binary_mass(p, w=w) / total
                    psi = psi + share * q1 * density * q2 * (events / uncensored)
    return psi


STEP = 1e-30


def eif(function: Callable[[Any], Any], probs: Any) -> np.ndarray:
    """The influence function of ``function`` at every support point, by complex step.

    Every support point is perturbed at once: column ``j`` of the batch is the law moved
    toward point ``j``, and the functional broadcasts over the columns.  A point with no
    mass reports ``nan``: no row sits there.
    """
    probs = np.asarray(probs, dtype=float)
    size = len(probs)
    batch = (1.0 - 1j * STEP) * probs[:, None] + 1j * STEP * np.eye(size)
    values = np.imag(np.asarray(function(batch))) / STEP
    return np.where(probs > 0, values, np.nan)


def weighted(probs: Any, weights: Any) -> Any:
    r"""The weighted law :math:`dP_w = w\,dP / E_P[w]`, for one law or a batch of columns."""
    cell = np.asarray(weights, dtype=float).reshape((-1,) + (1,) * (np.ndim(probs) - 1))
    tilted = cell * probs
    return tilted / tilted.sum(axis=0)


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


def binary_function(
    name: str,
    *,
    plans: dict[str, tuple[Node, Node]] = PLANS,
    base: Any = None,
    known: tuple[int, ...] = (),
    weights: Any = None,
) -> Callable[[Any], Any]:
    """The functional behind ``name`` on a two-node law, as a function of the masses.

    ``weights`` is a weight per support point; the functional is then evaluated at the
    weighted law :math:`P_w`, as :func:`weighted` builds it.
    """

    def at(probs: Any) -> Any:
        law = probs if weights is None else weighted(probs, weights)
        anchor = base
        if weights is not None:
            anchor = weighted(binary.PROBS if base is None else base, weights)
        return _contrast(
            name, lambda label, _: functional_binary(law, plans[label], base=anchor, known=known)
        )

    return at


NAMES = names(PLANS, REFERENCE)

#: Population values of every reported parameter on the two-node law.
TRUTH = {name: float(binary_function(name)(binary.PROBS)) for name in NAMES}


# ------------------------------------------------------------------ two-node laws
#
# Every two-node law here shares the support and node layout of
# :mod:`tests.discrete_law_longitudinal` and differs only in its tables.


def _two_node_mass(point: tuple[Any, ...], tables: dict[str, np.ndarray]) -> float:
    w, a1, c1, l2, a2, c2, y = point
    g1, c1_, l2_, g2, c2_, q = (tables[key] for key in ("G1", "C1", "P_L2", "G2", "C2", "Q"))
    mass = tables["P_W"][w] * (g1[w] if a1 == 1 else 1.0 - g1[w])
    if c1 == 0:
        return float(mass * (1.0 - c1_[w, a1]))
    mass *= c1_[w, a1]
    mass *= l2_[w, a1] if l2 == 1 else 1.0 - l2_[w, a1]
    mass *= g2[w, a1, l2] if a2 == 1 else 1.0 - g2[w, a1, l2]
    if c2 == 0:
        return float(mass * (1.0 - c2_[w, a1, l2, a2]))
    mass *= c2_[w, a1, l2, a2]
    outcome = q[w, a1, l2, a2]
    return float(mass * (outcome if y == 1 else 1.0 - outcome))


def _exact_counts(masses: np.ndarray, n: int) -> np.ndarray:
    values = masses * n
    rounded = np.rint(values)
    if np.max(np.abs(values - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError("the cell probabilities are not multiples of 1/N")
    return rounded.astype(int)


def _two_node_counts(tables: dict[str, np.ndarray], n: int) -> np.ndarray:
    return _exact_counts(np.array([_two_node_mass(point, tables) for point in binary.SUPPORT]), n)


def _rows(support: Any, counts: np.ndarray, columns: tuple[str, ...]) -> pd.DataFrame:
    cells = np.repeat(np.arange(len(support)), counts)
    return pd.DataFrame(
        {
            name: np.array(
                [np.nan if point[position] is None else float(point[position]) for point in support]
            )[cells]
            for position, name in enumerate(columns)
        }
    )


_TWO_NODE_COLUMNS = ("W", "A1", "C1", "L2", "A2", "C2", "Y")


def first_rows(counts: np.ndarray) -> np.ndarray:
    """The first row of each support point.  A point with no rows reports ``-1``."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.where(counts > 0, starts, -1)


_BASE_TABLES = {
    "P_W": binary.P_W,
    "G1": binary.G1,
    "C1": binary.C1,
    "P_L2": binary.P_L2,
    "G2": binary.G2,
    "C2": binary.C2,
    "Q": binary.Q,
}

#: The tilt law: the censored two-node law whose mechanism terms are large.
#:
#: On :mod:`tests.discrete_law_longitudinal` each node's mechanism term holds less than 4 %
#: of the influence function's variance, so a test that dropped it could pass by a small
#: margin.  These tables came from a hill climb over the quarter grid (outcome entries may
#: be 0 or 1).  It maximised the smaller of each node's mechanism share at
#: :math:`\delta \in \{0.5, 2\}` and ten times the separation of the constant and per-node
#: plans.  :func:`tilt_witnesses` states the result.
TILT_TABLES: dict[str, np.ndarray] = {
    "P_W": np.array([0.5, 0.5]),
    "G1": np.array([0.5, 0.25]),
    "C1": np.array([[0.75, 0.75], [0.5, 0.75]]),
    "P_L2": np.array([[0.75, 0.25], [0.75, 0.25]]),
    "G2": np.array([[[0.5, 0.75], [0.25, 0.5]], [[0.25, 0.25], [0.75, 0.5]]]),
    "C2": np.array(
        [
            [[[0.75, 0.75], [0.25, 0.5]], [[0.5, 0.25], [0.75, 0.75]]],
            [[[0.5, 0.5], [0.5, 0.25]], [[0.25, 0.5], [0.5, 0.5]]],
        ]
    ),
    "Q": np.array(
        [
            [[[0.0, 0.75], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.75]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    ),
}
TILT_N = binary.N
TILT_COUNTS = _two_node_counts(TILT_TABLES, TILT_N)
TILT_PROBS = TILT_COUNTS / TILT_N


def tilt_frame() -> pd.DataFrame:
    """The sample that realises the tilt law exactly."""
    return _rows(binary.SUPPORT, TILT_COUNTS, _TWO_NODE_COLUMNS)


#: Population values of every reported parameter on the tilt law.
TILT_TRUTH = {name: float(binary_function(name, base=TILT_PROBS)(TILT_PROBS)) for name in NAMES}

#: The no-positivity law: the law of :mod:`tests.discrete_law_longitudinal` with three
#: treatment cells changed at node 2.  ``N`` grows by a factor of eight, so ``1/32`` is a
#: whole number of rows.
#:
#: ``G2[0, 0, 0]`` is ``1/32``, below the ``0.05`` bound the T18 fit declares, and the
#: observed arm there is the rare one.  ``G2[1, 1, 1]`` is 0 and ``G2[1, 0, 1]`` is 1, so no
#: unit in those cells receives the other arm.
POSITIVITY_TABLES = {**_BASE_TABLES, "G2": binary.G2.copy()}
POSITIVITY_TABLES["G2"][0, 0, 0] = 1.0 / 32.0
POSITIVITY_TABLES["G2"][1, 1, 1] = 0.0
POSITIVITY_TABLES["G2"][1, 0, 1] = 1.0
POSITIVITY_N = 2 * 4**5 * 32
POSITIVITY_COUNTS = _two_node_counts(POSITIVITY_TABLES, POSITIVITY_N)
POSITIVITY_PROBS = POSITIVITY_COUNTS / POSITIVITY_N


def positivity_frame() -> pd.DataFrame:
    """The sample that realises the no-positivity law exactly."""
    return _rows(binary.SUPPORT, POSITIVITY_COUNTS, _TWO_NODE_COLUMNS)


# ------------------------------------------------------------------ survival and competing

_SURVIVAL_NAMES = ("w", "a1", "c1", "y1", "l2", "a2", "c2", "y2")
_survival_mass = _masser(survival.SUPPORT, _SURVIVAL_NAMES)
_COMPETING_NAMES = ("w", "a1", "c1", "j1", "l2", "a2", "c2", "j2")
_competing_mass = _masser(competing.SUPPORT, _COMPETING_NAMES)

#: The plans of the survival and competing laws.
EVENT_PLANS: dict[str, tuple[Node, Node]] = {
    "natural": (tilt(1.0), tilt(1.0)),
    "double": (tilt(2.0), tilt(2.0)),
    "per_node": (tilt(2.0), tilt(0.5)),
    "always": (arm(1), arm(1)),
}


def functional_event(
    probs: Any,
    plan: tuple[Node, Node],
    horizon: int,
    *,
    cause: int | None = None,
    known: tuple[int, ...] = (),
) -> Any:
    r"""The cumulative incidence of one cause by ``horizon`` under a tilt plan.

    ``cause=None`` reads the survival law, whose event node takes the values 0 and 1.  A
    cause code reads the competing law.  The second node's population is the units free
    of *every* cause after the first node, and the cause-specific events are the
    numerator:

    .. math::

        F(2) = \sum_w P(w) \sum_{a_1} q_1 \Bigl[h_1 + S_1 \sum_l P(l \mid \cdot)
               \sum_{a_2} q_2\, h_2\Bigr].
    """
    survival_law = cause is None
    mass = _survival_mass if survival_law else _competing_mass
    event = "y" if survival_law else "j"
    code = 1 if survival_law else cause
    base = survival.PROBS if survival_law else competing.PROBS
    p = probs
    total = mass(p)
    psi = 0.0
    for w in (0, 1):
        source = base if 1 in known else p
        pi1 = mass(source, w=w, a1=1) / mass(source, w=w)
        share = mass(p, w=w) / total
        for a1 in (0, 1):
            q1 = _q(plan[0], pi1, a1)
            reached = mass(p, w=w, a1=a1, c1=1)
            hazard1 = mass(p, w=w, a1=a1, c1=1, **{f"{event}1": code}) / reached
            if horizon == 1:
                psi = psi + share * q1 * hazard1
                continue
            free = {f"{event}1": 0}
            survived = mass(p, w=w, a1=a1, c1=1, **free)
            later = 0.0
            for l2 in (0, 1):
                density = mass(p, w=w, a1=a1, c1=1, l2=l2, **free) / survived
                source = base if 2 in known else p
                pi2 = mass(source, w=w, a1=a1, c1=1, l2=l2, a2=1, **free) / mass(
                    source, w=w, a1=a1, c1=1, l2=l2, **free
                )
                for a2 in (0, 1):
                    q2 = _q(plan[1], pi2, a2)
                    uncensored = mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, **free)
                    events = mass(
                        p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, **free, **{f"{event}2": code}
                    )
                    later = later + density * q2 * (events / uncensored)
            psi = psi + share * q1 * (hazard1 + (survived / reached) * later)
    return psi


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


def competing_function(name: str) -> Callable[[Any], Any]:
    """The functional behind a competing-risk ``name``, as a function of the masses."""

    def at(probs: Any) -> Any:
        head, _, index = name.partition("[")
        index = index[:-1]
        rest, _, horizon = index.rpartition(" @ t=")
        label, _, cause = rest.rpartition(", ")
        code = competing.CAUSES.index(cause) + 1

        def mean(plan_label: str) -> Any:
            return functional_event(probs, EVENT_PLANS[plan_label], int(horizon), cause=code)

        if head == "ate_regimen":
            left, right = label.split(" vs ")
            return mean(left) - mean(right)
        return mean(label)

    return at


def competing_names(reference: str = REFERENCE) -> tuple[str, ...]:
    """The names a competing-risk fit of :data:`EVENT_PLANS` reports, in report order."""
    return tuple(
        f"cif_regimen[{label}, {cause} @ t={horizon}]"
        for label in EVENT_PLANS
        for cause in competing.CAUSES
        for horizon in competing.HORIZONS
    ) + tuple(
        f"ate_regimen[{label} vs {reference}, {cause} @ t={horizon}]"
        for label in EVENT_PLANS
        if label != reference
        for cause in competing.CAUSES
        for horizon in competing.HORIZONS
    )


# ------------------------------------------------------------------ the three-node law
#
#     W  ->  A1  ->  L2  ->  A2  ->  L3  ->  A3  ->  Y
#
# No censoring.  Every treatment and covariate conditional is a multiple of 1/4 in
# [0.25, 0.75], and the outcome conditional is a multiple of 1/4 in [0, 1], so ``2 * 4**6``
# rows realise the law exactly.  The tables came from a hill climb over the quarter grid.
# It maximised the smallest of each node's mechanism share at delta 0.5 and 2, and ten times
# the separation of the constant plans.  :func:`three_witnesses` states the result.

THREE_N = 2 * 4**6
#: ``P(A1 = 1 | W)``.
THREE_G1 = np.array([0.75, 0.25])
#: ``P(L2 = 1 | W, A1)``.
THREE_L2 = np.array([[0.75, 0.75], [0.75, 0.25]])
#: ``P(A2 = 1 | W, A1, L2)``.
THREE_G2 = np.array([[[0.25, 0.75], [0.5, 0.75]], [[0.75, 0.75], [0.25, 0.25]]])
#: ``P(L3 = 1 | W, L2, A2)``.
THREE_L3 = np.array([[[0.75, 0.75], [0.25, 0.5]], [[0.75, 0.5], [0.75, 0.75]]])
#: ``P(A3 = 1 | W, A2, L3)``.
THREE_G3 = np.array([[[0.75, 0.75], [0.75, 0.75]], [[0.75, 0.25], [0.5, 0.75]]])
#: ``P(Y = 1 | W, A1, A2, L3, A3)``.
THREE_Q = np.array(
    [
        [
            [[[0.5, 0.0], [0.0, 0.25]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.5], [0.5, 0.25]], [[0.75, 1.0], [0.75, 1.0]]],
        ],
        [
            [[[0.25, 0.75], [0.25, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, 0.0], [0.25, 1.0]], [[0.75, 0.25], [0.0, 0.0]]],
        ],
    ]
)

THREE_SUPPORT: tuple[tuple[int, ...], ...] = tuple(itertools.product(range(2), repeat=7))
_THREE_NAMES = ("w", "a1", "l2", "a2", "l3", "a3", "y")
_three_mass = _masser(THREE_SUPPORT, _THREE_NAMES)


def _bern(p: float, x: int) -> float:
    return p if x == 1 else 1.0 - p


THREE_COUNTS = _exact_counts(
    np.array(
        [
            0.5
            * _bern(THREE_G1[w], a1)
            * _bern(THREE_L2[w, a1], l2)
            * _bern(THREE_G2[w, a1, l2], a2)
            * _bern(THREE_L3[w, l2, a2], l3)
            * _bern(THREE_G3[w, a2, l3], a3)
            * _bern(THREE_Q[w, a1, a2, l3, a3], y)
            for w, a1, l2, a2, l3, a3, y in THREE_SUPPORT
        ]
    ),
    THREE_N,
)
THREE_PROBS = THREE_COUNTS / THREE_N

#: The plans of the three-node law.
THREE_PLANS: dict[str, tuple[Node, Node, Node]] = {
    "natural": (tilt(1.0), tilt(1.0), tilt(1.0)),
    "half": (tilt(0.5), tilt(0.5), tilt(0.5)),
    "double": (tilt(2.0), tilt(2.0), tilt(2.0)),
    "mixed": (tilt(2.0), arm(1), tilt(0.5)),
}


def functional_three(
    probs: Any,
    plan: tuple[Node, Node, Node],
    *,
    base: Any = None,
    known: tuple[int, ...] = (),
) -> Any:
    """:math:`E[Y^{Q}]` on the three-node law, by finite sums.

    ``base`` and ``known`` read as in :func:`functional_binary`.
    """
    p = probs
    base = THREE_PROBS if base is None else base
    mass = _three_mass
    total = mass(p)

    def pi(node: int, **history: int) -> Any:
        source = base if node in known else p
        return mass(source, **history, **{f"a{node}": 1}) / mass(source, **history)

    psi = 0.0
    for w in (0, 1):
        share = mass(p, w=w) / total
        pi1 = pi(1, w=w)
        for a1 in (0, 1):
            q1 = _q(plan[0], pi1, a1)
            for l2 in (0, 1):
                d2 = mass(p, w=w, a1=a1, l2=l2) / mass(p, w=w, a1=a1)
                pi2 = pi(2, w=w, a1=a1, l2=l2)
                for a2 in (0, 1):
                    q2 = _q(plan[1], pi2, a2)
                    for l3 in (0, 1):
                        h3 = {"w": w, "a1": a1, "l2": l2, "a2": a2}
                        d3 = mass(p, **h3, l3=l3) / mass(p, **h3)
                        pi3 = pi(3, **h3, l3=l3)
                        for a3 in (0, 1):
                            q3 = _q(plan[2], pi3, a3)
                            cell = {**h3, "l3": l3, "a3": a3}
                            mean = mass(p, **cell, y=1) / mass(p, **cell)
                            psi = psi + share * q1 * d2 * q2 * d3 * q3 * mean
    return psi


def three_function(name: str, *, known: tuple[int, ...] = ()) -> Callable[[Any], Any]:
    """The functional behind ``name`` on the three-node law."""

    def at(probs: Any) -> Any:
        return _contrast(
            name, lambda label, _: functional_three(probs, THREE_PLANS[label], known=known)
        )

    return at


def three_frame() -> pd.DataFrame:
    """The sample that realises the three-node law exactly."""
    return _rows(THREE_SUPPORT, THREE_COUNTS, ("W", "A1", "L2", "A2", "L3", "A3", "Y"))


def mechanism_share(
    function: Callable[..., Callable[[Any], Any]], name: str, probs: Any, node: int
) -> float:
    """The share of the influence function's variance that node ``node``'s mechanism term holds.

    The mechanism term at a node is the difference between the influence function and the
    influence function with that node's :math:`\\pi_t` held at the law.  Both come from the
    complex step alone.
    """
    full = eif(function(name), probs)
    held = eif(function(name, known=(node,)), probs)
    keep = probs > 0
    term = full[keep] - held[keep]
    weights = probs[keep]
    centred = full[keep] - np.average(full[keep], weights=weights)
    return float(np.average(term**2, weights=weights) / np.average(centred**2, weights=weights))


# ------------------------------------------------------------------ nonzero witnesses


def arm_gaps_binary(probs: np.ndarray, plan: tuple[Node, Node]) -> dict[int, np.ndarray]:
    r""":math:`|m_t(H_t, 1) - m_t(H_t, 0)|` at every history with mass, by node.

    A mechanism term is :math:`W_{t-1}\delta(m_t(H_t,1)-m_t(H_t,0))(A_t-\pi_t)/D_t^2`, so it
    is zero wherever this gap is.  A test that drops the term passes there.
    """

    def outcome(w: int, a1: int, l2: int, a2: int) -> float:
        cell = {"w": w, "a1": a1, "c1": 1, "l2": l2, "a2": a2, "c2": 1}
        return float(_binary_mass(probs, **cell, y=1) / _binary_mass(probs, **cell))

    def carried(w: int, a1: int, l2: int) -> float:
        history = {"w": w, "a1": a1, "c1": 1, "l2": l2}
        pi = float(_binary_mass(probs, **history, a2=1) / _binary_mass(probs, **history))
        return sum(_q(plan[1], pi, a2) * outcome(w, a1, l2, a2) for a2 in (0, 1))

    def first(w: int, a1: int) -> float:
        reached = _binary_mass(probs, w=w, a1=a1, c1=1)
        return sum(
            float(_binary_mass(probs, w=w, a1=a1, c1=1, l2=l2) / reached) * carried(w, a1, l2)
            for l2 in (0, 1)
        )

    second = [
        abs(outcome(w, a1, l2, 1) - outcome(w, a1, l2, 0))
        for w, a1, l2 in itertools.product(range(2), repeat=3)
    ]
    return {1: np.array([abs(first(w, 1) - first(w, 0)) for w in (0, 1)]), 2: np.array(second)}


def arm_gaps_three(probs: np.ndarray, plan: tuple[Node, Node, Node]) -> dict[int, np.ndarray]:
    """:func:`arm_gaps_binary` on the three-node law."""
    mass = _three_mass

    def m3(h: dict[str, int], a3: int) -> float:
        return float(mass(probs, **h, a3=a3, y=1) / mass(probs, **h, a3=a3))

    def carried3(h: dict[str, int]) -> float:
        pi = float(mass(probs, **h, a3=1) / mass(probs, **h))
        return sum(_q(plan[2], pi, a3) * m3(h, a3) for a3 in (0, 1))

    def m2(h: dict[str, int], a2: int) -> float:
        reached = mass(probs, **h, a2=a2)
        return sum(
            float(mass(probs, **h, a2=a2, l3=l3) / reached) * carried3({**h, "a2": a2, "l3": l3})
            for l3 in (0, 1)
        )

    def carried2(h: dict[str, int]) -> float:
        pi = float(mass(probs, **h, a2=1) / mass(probs, **h))
        return sum(_q(plan[1], pi, a2) * m2(h, a2) for a2 in (0, 1))

    def m1(w: int, a1: int) -> float:
        reached = mass(probs, w=w, a1=a1)
        return sum(
            float(mass(probs, w=w, a1=a1, l2=l2) / reached) * carried2({"w": w, "a1": a1, "l2": l2})
            for l2 in (0, 1)
        )

    histories3 = [
        dict(zip(("w", "a1", "l2", "a2", "l3"), cell, strict=True))
        for cell in itertools.product(range(2), repeat=5)
    ]
    histories2 = [
        dict(zip(("w", "a1", "l2"), cell, strict=True))
        for cell in itertools.product(range(2), repeat=3)
    ]
    return {
        1: np.array([abs(m1(w, 1) - m1(w, 0)) for w in (0, 1)]),
        2: np.array([abs(m2(h, 1) - m2(h, 0)) for h in histories2]),
        3: np.array([abs(m3(h, 1) - m3(h, 0)) for h in histories3]),
    }
