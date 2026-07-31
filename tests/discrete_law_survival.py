r"""A two-time-point *survival* law that a finite sample realises exactly.

The counterpart of :mod:`tests.discrete_law_longitudinal` for an outcome that is a node
at every time point rather than one at the end.  It exists separately rather than as a
wider version of that law for two reasons: that one has to go on proving the end-of-study
derivation unchanged, and a law answering two derivations at once cannot be read as
evidence for either.  Every node is binary --

.. code-block:: text

    W  ->  A1  ->  C1  ->  Y1  ->  L2  ->  A2  ->  C2  ->  Y2

-- and every conditional probability is a multiple of ``1/4``, so every cell probability
is a multiple of ``1 / N`` with ``N = 2 * 4**7``.  Laying ``N`` rows out in the cell
proportions makes the empirical distribution *equal* to the data-generating one, which is
what makes the assertions here exact rather than statistical.

Two structural facts are what this law is for, and neither can be checked on a law whose
outcome sits at the end.

**The event is absorbing, so it moves the risk set.**  A unit with ``Y1 = 1`` has no
``L2``, no ``A2``, no ``C2`` and no ``Y2``: the support below carries a fourth
missingness pattern for it, beside the two censoring patterns and the complete one.  So a
fit that kept the failures in the second node's regression, or dropped them from the
first node's, answers for a different population and misses the truth here.

**The parameter is a curve.**  The cumulative risk at ``t = 1`` and at ``t = 2`` are two
parameters of one distribution with almost the same influence curve, and
:func:`functional` states both longhand.

Positivity holds comfortably -- every conditional lies in ``[0.25, 0.75]`` -- so no
truncation is active and the estimator runs on the unmodified mechanism.

Three regimens rather than the six of the longitudinal law: the dynamic machinery is
already proved there, and here every regimen is two parameters, so a fourth would buy
coverage of an axis that is not this law's business.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from .discrete_law_longitudinal import CellMeans

__all__ = [
    "NAMES",
    "PROBS",
    "REGIMEN_ARMS",
    "REGIMEN_REFERENCE",
    "REGIMEN_SPEC",
    "SUPPORT",
    "TRUTH",
    "CellMeans",
    "N",
    "eif",
    "first_row_of",
    "frame",
    "functional",
    "gateaux",
]

#: Rows in the realised sample.  ``2 * 4**7``: one factor of two for ``P(W)`` and seven
#: of four for the conditionals that follow it -- one more node than the end-of-study law
#: has, because the hazard at the first time point is a node in its own right.
N = 2 * 4**7

P_W = np.array([0.5, 0.5])

#: ``P(A1 = 1 | W = w)``.
G1 = np.array([0.50, 0.25])

#: ``P(C1 = 1 | W, A1)``.  Depends on the history, and has to: with a censoring
#: probability that did not, censoring would be MCAR and a fit that dropped the censoring
#: factors would still find the truth -- which would leave the negative control in
#: ``test_influence_gateaux_survival`` unable to fail.
C1 = np.array([[0.75, 0.50], [0.75, 0.75]])

#: ``P(Y1 = 1 | W, A1, C1 = 1)`` -- the hazard at the first time point, indexed
#: ``[w, a1]``.  Treatment lowers it, so the two regimens' curves separate at ``t = 1``
#: rather than only at the end.
H1 = np.array([[0.50, 0.25], [0.75, 0.50]])

#: ``P(L2 = 1 | W, A1, C1 = 1, Y1 = 0)``.  Conditioned on surviving the first node, since
#: that is the only population that has an ``L2``.  Depends on ``A1``: this is the node
#: that makes the problem longitudinal rather than two cross-sections.
P_L2 = np.array([[0.25, 0.75], [0.50, 0.75]])

#: ``P(A2 = 1 | W, A1, Y1 = 0, L2)``, indexed ``[w, a1, l2]``.
G2 = np.array([[[0.50, 0.75], [0.25, 0.50]], [[0.75, 0.50], [0.50, 0.25]]])

#: ``P(C2 = 1 | W, A1, L2, A2)``, indexed ``[w, a1, l2, a2]``.
C2 = np.array(
    [
        [[[0.75, 0.50], [0.75, 0.75]], [[0.50, 0.75], [0.75, 0.50]]],
        [[[0.75, 0.75], [0.50, 0.75]], [[0.75, 0.50], [0.75, 0.75]]],
    ]
)

#: ``P(Y2 = 1 | W, A1, L2, A2, C2 = 1)``, indexed ``[w, a1, l2, a2]`` -- the hazard at the
#: second time point among the units still at risk there.
#:
#:
#: The ``[1, 0, 1, 1]`` entry is ``0.25`` rather than the ``0.50`` its neighbours would
#: suggest, and deliberately: with ``0.50`` the rule's curve came out *exactly* equal to
#: ``never``'s at both horizons -- the two cells they differ in cancelling to the last bit
#: -- which would have left a parameter no bug could move.
#: :func:`tests.unit.test_influence_gateaux_survival.test_the_rule_is_a_parameter_no_static
#: _plan_reaches` is the assertion that found that and the one that keeps it found.
H2 = np.array(
    [
        [[[0.25, 0.50], [0.50, 0.75]], [[0.50, 0.75], [0.75, 0.50]]],
        [[[0.50, 0.25], [0.75, 0.25]], [[0.25, 0.50], [0.50, 0.75]]],
    ]
)

#: The regimens the estimands are checked against, in report order; the first is the
#: reference.  **The oracle side**, as in the end-of-study law: arms as a lookup over cell
#: *indices*, a scalar being a constant node and an array at the second node indexed
#: ``[w, l2]``.  :data:`REGIMEN_SPEC` states the same three plans as callables, so a slip
#: in one is a wrong number rather than a mistake that cancels on both sides.
REGIMEN_ARMS: dict[str, tuple[Any, Any]] = {
    "never": (0, 0),
    "always": (1, 1),
    # Treat everyone, then continue only for the responders: d_2 = 1{L2 = 1}.  Under a
    # survival outcome its followers are a set that moves twice over -- with the
    # covariate, as at one node, and with who is still alive to have a covariate at all.
    # It treats at the first node so that its curve separates from ``never`` at *both*
    # horizons; a rule that idles at the first node is exactly ``never`` at ``t = 1``,
    # which is a true statement about the law and a parameter no bug can move.
    "continue_if_l2": (1, np.array([[0, 1], [0, 1]])),
}

#: The estimator side of :data:`REGIMEN_ARMS`: what ``LTMLE`` is handed as ``regimens=``.
REGIMEN_SPEC: dict[str, Any] = {
    "never": 0,
    "always": 1,
    "continue_if_l2": (1, lambda h: h["L2"]),
}

REGIMEN_REFERENCE = "never"

if set(REGIMEN_ARMS) != set(REGIMEN_SPEC):  # pragma: no cover - guards the two tables
    raise AssertionError(
        "REGIMEN_ARMS and REGIMEN_SPEC must state the same regimens; a plan the oracle "
        "knows and the estimator is never handed proves nothing, and the reverse is a "
        "parameter with no truth to check against"
    )

#: The horizons a fit on this law reports, in report order.
HORIZONS: tuple[int, ...] = (1, 2)

#: One support point per observable history.  ``None`` marks a node the unit never
#: reached, and there are now two ways to fail to reach one: a unit censored at the first
#: node has no ``Y1`` onwards, and a unit that **had the event** at the first node has no
#: ``L2`` onwards.  The second block is the one an end-of-study law has no analogue for.
SUPPORT: tuple[tuple[Any, ...], ...] = tuple(
    [(w, a1, 0, None, None, None, None, None) for w, a1 in itertools.product(range(2), range(2))]
    + [(w, a1, 1, 1, None, None, None, None) for w, a1 in itertools.product(range(2), range(2))]
    + [
        (w, a1, 1, 0, l2, a2, 0, None)
        for w, a1, l2, a2 in itertools.product(range(2), range(2), range(2), range(2))
    ]
    + [
        (w, a1, 1, 0, l2, a2, 1, y2)
        for w, a1, l2, a2, y2 in itertools.product(range(2), range(2), range(2), range(2), range(2))
    ]
)

_NODES = ("w", "a1", "c1", "y1", "l2", "a2", "c2", "y2")

_COLUMNS = ("W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")


def _mass_of(point: tuple[Any, ...]) -> float:
    """``P`` of one support point, as a product of the conditionals above."""
    w, a1, c1, y1, l2, a2, c2, y2 = point
    mass = P_W[w] * (G1[w] if a1 == 1 else 1.0 - G1[w])
    if c1 == 0:
        return float(mass * (1.0 - C1[w, a1]))
    mass *= C1[w, a1]
    mass *= H1[w, a1] if y1 == 1 else 1.0 - H1[w, a1]
    if y1 == 1:
        return float(mass)
    mass *= P_L2[w, a1] if l2 == 1 else 1.0 - P_L2[w, a1]
    mass *= G2[w, a1, l2] if a2 == 1 else 1.0 - G2[w, a1, l2]
    if c2 == 0:
        return float(mass * (1.0 - C2[w, a1, l2, a2]))
    mass *= C2[w, a1, l2, a2]
    return float(mass * (H2[w, a1, l2, a2] if y2 == 1 else 1.0 - H2[w, a1, l2, a2]))


def _counts() -> np.ndarray:
    counts = np.array([_mass_of(point) * N for point in SUPPORT])
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- keep every conditional a multiple of 1/4"
        )
    return rounded.astype(int)


COUNTS = _counts()

#: ``P`` over the support, taken from the counts so it is bit-for-bit the empirical law
#: of :func:`frame`.
PROBS = COUNTS / N


@cache
def _index(**pattern: int) -> tuple[int, ...]:
    """Support points matching every named node, memoised as in the end-of-study law."""
    return tuple(
        position
        for position, point in enumerate(SUPPORT)
        if all(point[_NODES.index(node)] == value for node, value in pattern.items())
    )


def _arm(node: Any, *index: int) -> int:
    """The arm a plan's node assigns in the cell named by ``index``.

    Selecting by cell *index* is what keeps :func:`functional` analytic: the arm is fixed
    before the complex perturbation is applied, so no comparison ever touches ``probs``.
    """
    return int(node) if np.ndim(node) == 0 else int(node[index])


def _mass(probs: Any, **pattern: int) -> Any:
    """Total probability of the matching support points -- a linear form in ``probs``."""
    return sum(probs[position] for position in _index(**pattern))


def _split_horizon(index: str) -> tuple[str, int]:
    """``"always @ t=2"`` -> ``("always", 2)``.

    Split off the horizon *before* anything splits on ``" vs "``: a contrast's index is
    ``"always vs never @ t=2"``, and splitting the other way round leaves the horizon
    attached to the reference regimen's label.
    """
    label, _, horizon = index.rpartition(" @ t=")
    return label, int(horizon)


def functional(probs: Any, estimand: str) -> Any:
    r"""The cumulative risk under a regimen, as a closed-form function of the cells.

    Written straight off the iterated conditional expectation for an absorbing event, and
    sharing no code with the library:

    .. math::

        \Psi_1 &= \sum_w P(W = w)\, h_1(w, a_1) \\
        \Psi_2 &= \sum_w P(W = w)\Bigl[h_1 + (1 - h_1)
                  \sum_{l} P(L_2 = l \mid W, A_1 = a_1, C_1 = 1, Y_1 = 0)\,
                  h_2(w, a_1, l, a_2)\Bigr]

    where every conditional is a quotient of sums of cell probabilities.  Note where the
    second sum conditions: on ``Y1 = 0`` as well as ``C1 = 1``, because the units that had
    the event have no ``L2`` and are not the population the second hazard is about.  That
    conditioning is the whole of what a survival outcome adds, and getting it wrong is
    what the negative controls in the test module are written to catch.

    Every operation is a sum, a difference or a quotient of sums, so this stays analytic
    in the cell probabilities -- which is what lets :func:`gateaux` differentiate it by a
    complex step.  Do not introduce ``clip``, ``abs`` or a comparison here.
    """
    p = probs
    if estimand.startswith("ate_regimen["):
        index = estimand[len("ate_regimen[") : -1]
        contrast, horizon = _split_horizon(index)
        left, right = contrast.split(" vs ")
        return functional(p, f"risk_regimen[{left} @ t={horizon}]") - functional(
            p, f"risk_regimen[{right} @ t={horizon}]"
        )
    if not estimand.startswith("risk_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")

    label, horizon = _split_horizon(estimand[len("risk_regimen[") : -1])
    node1, node2 = REGIMEN_ARMS[label]
    total = _mass(p)
    psi = 0.0
    for w in (0, 1):
        a1 = _arm(node1, w)
        share = _mass(p, w=w) / total
        reached = _mass(p, w=w, a1=a1, c1=1)
        hazard1 = _mass(p, w=w, a1=a1, c1=1, y1=1) / reached
        if horizon == 1:
            psi = psi + share * hazard1
            continue
        survived = _mass(p, w=w, a1=a1, c1=1, y1=0)
        later = 0.0
        for l2 in (0, 1):
            a2 = _arm(node2, w, l2)
            density = _mass(p, w=w, a1=a1, c1=1, y1=0, l2=l2) / survived
            uncensored = _mass(p, w=w, a1=a1, c1=1, y1=0, l2=l2, a2=a2, c2=1)
            events = _mass(p, w=w, a1=a1, c1=1, y1=0, l2=l2, a2=a2, c2=1, y2=1)
            later = later + density * (events / uncensored)
        psi = psi + share * (hazard1 + (1.0 - hazard1) * later)
    return psi


#: The parameter names a survival fit reports on this law, in report order: regimen
#: outer, horizon inner, means before contrasts -- which is the order ``_estimates``
#: builds them in, and the gate in the test module compares the two sets both ways.
NAMES: tuple[str, ...] = tuple(
    f"risk_regimen[{label} @ t={horizon}]" for label in REGIMEN_ARMS for horizon in HORIZONS
) + tuple(
    f"ate_regimen[{label} vs {REGIMEN_REFERENCE} @ t={horizon}]"
    for label in REGIMEN_ARMS
    if label != REGIMEN_REFERENCE
    for horizon in HORIZONS
)

#: Population values of every reported parameter.
TRUTH = {name: float(functional(PROBS, name)) for name in NAMES}


def gateaux(estimand: str, point: int, *, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\,
                 \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)\right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function -- derived from :func:`functional` alone, with no clever
    covariate, no cumulative product and nothing else the library supplies.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif(estimand: str) -> np.ndarray:
    """The EIF of ``estimand`` at every support point, in support order."""
    return np.array([gateaux(estimand, position) for position in range(len(SUPPORT))])


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    Rows are laid out in :data:`SUPPORT` order, one contiguous block per support point.
    A node the unit never reached is ``nan`` -- including every node after the one where
    it had the event, which is the missingness pattern
    :class:`~cleverly.longitudinal.LongitudinalData` requires of an absorbing outcome.
    """
    cells = np.repeat(np.arange(len(SUPPORT)), COUNTS)
    columns = {
        name: np.array(
            [np.nan if point[position] is None else float(point[position]) for point in SUPPORT]
        )[cells]
        for position, name in enumerate(_COLUMNS)
    }
    return pd.DataFrame(columns)


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    return np.concatenate([[0], np.cumsum(COUNTS)[:-1]])
