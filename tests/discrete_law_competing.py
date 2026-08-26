r"""A two-time-point *competing-risks* law that a finite sample realises exactly.

The third of the longitudinal laws, beside :mod:`tests.discrete_law_longitudinal` for an
outcome at the end of the study and :mod:`tests.discrete_law_survival` for a single
absorbing event at every node.  It is a separate module rather than a wider version of
either for the reason the survival law gives for itself: those two have to go on proving
their own derivations unchanged, and a law answering three derivations at once cannot be
read as evidence for any of them.

Every node is binary except the outcome, which takes **three** values --

.. code-block:: text

    W  ->  A1  ->  C1  ->  J1  ->  L2  ->  A2  ->  C2  ->  J2

with ``J_t in {0, 1, 2}``: no event, *relapse*, or *death*.  Both are absorbing and they
are mutually exclusive, so a unit with ``J_1 != 0`` has no ``L2``, no ``A2``, no ``C2``
and no ``J_2``.  Every conditional probability is a multiple of ``1/4``, so every cell
probability is a multiple of ``1 / N`` with ``N = 4**8``.  Laying ``N`` rows out in the
cell proportions makes the empirical distribution *equal* to the data-generating one,
which is what makes the assertions here exact rather than statistical.

**What this law is for, and what neither sibling can check.**  The cumulative incidence of
one cause is not the risk of that cause treated as if the other did not exist.  Its
recursion carries a *cause-specific* numerator against an **all-cause** survival factor:

.. math::

    F_j(2) = \sum_w P(W = w)\Bigl[h_{1j}
             + \bigl(1 - \textstyle\sum_{j'} h_{1j'}\bigr)
               \sum_l P(L_2 = l \mid \cdot)\, h_{2j}\Bigr]

Writing :math:`(1 - h_{1j})` there -- the cause's *own* survival -- is the mistake this
module exists to catch.  It reads like the obvious generalisation of the single-event
recursion, it is wrong by exactly the mass that left through the other cause, and it
leaves every score at machine zero and every convergence flag green.  :func:`functional`
therefore writes the survival factor as ``survived / reached``, a quotient of sums over
cells, and never as one minus a hazard.

A competing event is part of the *history*, not an intervened node.  It enters the
indicator of the clever covariate and never its denominator, so the positivity assumption
here is the one an end-of-study fit makes -- over the ``2T`` treatment and censoring
factors and nothing else.  A fit that put the competing cause in the denominator would be
answering the *other* competing-risks question, the one that eliminates the competing
event, and that is a different estimand rather than a tuning choice.

Positivity holds comfortably: every treatment and censoring conditional lies in
``[0.25, 0.75]``, so no truncation is active and the estimator runs on the unmodified
mechanism.

Three constraints on the hazards, each of which the constants below satisfy by
construction rather than by luck:

* every cause-specific hazard is a multiple of ``1/4`` and **strictly positive**.  A zero
  would empty a support block, and an empty block would silently misalign
  :func:`first_row_of` against :func:`eif` -- the test would then compare a curve at the
  wrong row and could still pass;
* the all-cause hazard leaves at least ``0.25`` surviving at each node, so the second node
  has a risk set;
* ``P(W)`` is ``(0.25, 0.75)`` rather than the siblings' ``(0.5, 0.5)``.  With equal
  weights and quarter-valued hazards the marginal incidences land on such coarse fractions
  that several sit exactly on ``0.5``, which is
  :data:`~cleverly.longitudinal.sequential._FILLER` -- the value a prediction takes at a
  row nothing reads, and so the one value a truth must not equal if the test is to notice
  a filled row leaking into the estimate.

The constants were chosen by search under those conditions plus the non-degeneracy ones
the test module relies on: no contrast within ``0.02`` of zero, the three regimens
separated at each ``(cause, horizon)``, the two causes separated at each
``(regimen, horizon)``, the two causes' contrasts not mirror images of one another -- a
bug that swapped the causes would otherwise cancel rather than show -- and the rule
separated from ``always`` at ``t = 2``.  The winner leaves a worst-case separation of
``0.027``.  Treatment *raises* the incidence of relapse and *lowers* that of death, which
is the substantive shape a competing-risks parameter is for.

Three regimens, as in the survival law and for the same reason: the dynamic machinery is
proved in the end-of-study law, and here every regimen is already four parameters.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from .discrete_law_longitudinal import CellMeans

__all__ = [
    "CAUSES",
    "HORIZONS",
    "NAMES",
    "PROBS",
    "REGIMEN_ARMS",
    "REGIMEN_REFERENCE",
    "REGIMEN_SPEC",
    "SUPPORT",
    "TRUTH",
    "CellMeans",
    "N",
    "counts",
    "eif",
    "first_row_of",
    "frame",
    "functional",
    "gateaux",
    "outcome_columns",
    "probabilities",
]

#: Rows in the realised sample.  ``4**8``: one factor of four for ``P(W)`` -- see the
#: module docstring on why it is not a half -- and seven more for the conditionals that
#: follow it.  A three-valued node whose distribution is over quarters still contributes
#: a single factor of four, so the event nodes cost no more than the binary ones.
N = 4**8

#: ``P(W = w)``.  Deliberately not ``(0.5, 0.5)``; see the module docstring.
P_W = np.array([0.25, 0.75])

#: ``P(A1 = 1 | W = w)``.
G1 = np.array([0.50, 0.25])

#: ``P(C1 = 1 | W, A1)``.  Depends on the history, and has to: with a censoring
#: probability that did not, censoring would be MCAR and a fit that dropped the censoring
#: factors would still find the truth -- which would leave the negative control in
#: ``test_influence_gateaux_competing`` unable to fail.
C1 = np.array([[0.75, 0.50], [0.75, 0.75]])

#: The names of the two absorbing causes, in report order.  A cause's *code* in the
#: support tuples is its position here plus one, ``0`` being "no event".
CAUSES: tuple[str, ...] = ("relapse", "death")

#: ``P(J1 = j | W, A1, C1 = 1)`` for each cause, indexed ``[cause - 1, w, a1]``.  The
#: all-cause hazard is the sum over the first axis, and ``1`` minus it is what survives to
#: the second node -- the quantity :func:`functional` must multiply by, and the one a
#: cause-specific reading gets wrong.
#:
#: Treatment raises the relapse hazard at ``w = 0`` and lowers the death hazard at
#: ``w = 1``, so the two causes' contrasts differ in sign as well as size.
H1 = np.array(
    [
        [[0.25, 0.50], [0.25, 0.25]],  # relapse
        [[0.25, 0.25], [0.50, 0.25]],  # death
    ]
)

#: ``P(L2 = 1 | W, A1, C1 = 1, J1 = 0)``.  Conditioned on leaving the first node
#: event-free by *either* cause, since that is the only population that has an ``L2``.
#: Depends on ``A1``: this is the node that makes the problem longitudinal rather than two
#: cross-sections.
P_L2 = np.array([[0.25, 0.75], [0.50, 0.75]])

#: ``P(A2 = 1 | W, A1, J1 = 0, L2)``, indexed ``[w, a1, l2]``.
G2 = np.array([[[0.50, 0.75], [0.25, 0.50]], [[0.75, 0.50], [0.50, 0.25]]])

#: ``P(C2 = 1 | W, A1, L2, A2)``, indexed ``[w, a1, l2, a2]``.
C2 = np.array(
    [
        [[[0.75, 0.50], [0.75, 0.75]], [[0.50, 0.75], [0.75, 0.50]]],
        [[[0.75, 0.75], [0.50, 0.75]], [[0.75, 0.50], [0.75, 0.75]]],
    ]
)

#: ``P(J2 = j | W, A1, L2, A2, C2 = 1)`` for each cause, indexed
#: ``[cause - 1, w, a1, l2, a2]`` -- the hazards among the units still at risk at the
#: second node.
H2 = np.array(
    [
        [  # relapse
            [[[0.25, 0.50], [0.25, 0.25]], [[0.50, 0.25], [0.25, 0.25]]],
            [[[0.25, 0.25], [0.25, 0.25]], [[0.50, 0.25], [0.25, 0.25]]],
        ],
        [  # death
            [[[0.25, 0.25], [0.25, 0.50]], [[0.25, 0.50], [0.50, 0.25]]],
            [[[0.25, 0.50], [0.50, 0.25]], [[0.25, 0.50], [0.25, 0.25]]],
        ],
    ]
)

#: The regimens the estimands are checked against, in report order; the first is the
#: reference.  **The oracle side**, as in both sibling laws: arms as a lookup over cell
#: *indices*, a scalar being a constant node and an array at the second node indexed
#: ``[w, l2]``.  :data:`REGIMEN_SPEC` states the same three plans as callables, so a slip
#: in one is a wrong number rather than a mistake that cancels on both sides.
REGIMEN_ARMS: dict[str, tuple[Any, Any]] = {
    "never": (0, 0),
    "always": (1, 1),
    # Treat everyone, then continue only for the responders: d_2 = 1{L2 = 1}.  Under
    # competing risks its followers are a set that moves twice over -- with the covariate,
    # as at one node, and with who is still event-free *by either cause* to have a
    # covariate at all.  It treats at the first node, so at ``t = 1`` it **is** ``always``
    # -- a true statement about the law, and the one the bit-for-bit rule test asserts --
    # while at ``t = 2`` the constants above keep it a parameter no static plan reaches.
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
#: reached, and there are now three ways to fail to reach one: censoring at either node,
#: and -- the block neither sibling law has -- an event of *either* cause at the first
#: node, which ends the unit's follow-up as surely as censoring does but keeps it in that
#: node's regression rather than dropping it.
SUPPORT: tuple[tuple[Any, ...], ...] = tuple(
    [(w, a1, 0, None, None, None, None, None) for w, a1 in itertools.product(range(2), range(2))]
    + [
        (w, a1, 1, j1, None, None, None, None)
        for w, a1, j1 in itertools.product(range(2), range(2), range(1, len(CAUSES) + 1))
    ]
    + [
        (w, a1, 1, 0, l2, a2, 0, None)
        for w, a1, l2, a2 in itertools.product(range(2), range(2), range(2), range(2))
    ]
    + [
        (w, a1, 1, 0, l2, a2, 1, j2)
        for w, a1, l2, a2, j2 in itertools.product(
            range(2), range(2), range(2), range(2), range(len(CAUSES) + 1)
        )
    ]
)

_NODES = ("w", "a1", "c1", "j1", "l2", "a2", "c2", "j2")

#: How the three-valued event nodes are written out as the columns a fit is handed: one
#: **indicator per cause per node**, which is the declaration
#: :class:`~cleverly.longitudinal.LongitudinalData` takes.  The support tuple stays
#: three-valued because that is what makes exclusivity structural here and keeps
#: :func:`_mass_of` a plain chain rule; the expansion happens once, in :func:`frame`.
#: Each entry is ``(column, node, cause code or None)``.
_COLUMNS: tuple[tuple[str, str, int | None], ...] = (
    ("W", "w", None),
    ("A1", "a1", None),
    ("C1", "c1", None),
    ("R1", "j1", 1),
    ("D1", "j1", 2),
    ("L2", "l2", None),
    ("A2", "a2", None),
    ("C2", "c2", None),
    ("R2", "j2", 1),
    ("D2", "j2", 2),
)


def outcome_columns() -> dict[str, list[str]]:
    """``outcome=`` for a fit on this law: a cause to its indicator column per node.

    Built from :data:`_COLUMNS` rather than written out again, so the frame and the
    declaration cannot drift apart.
    """
    return {
        cause: [name for name, _, code in _COLUMNS if code == position + 1]
        for position, cause in enumerate(CAUSES)
    }


def _mass_of(point: tuple[Any, ...], *, h1: np.ndarray = H1, h2: np.ndarray = H2) -> float:
    """``P`` of one support point, as a product of the conditionals above."""
    w, a1, c1, j1, l2, a2, c2, j2 = point
    mass = P_W[w] * (G1[w] if a1 == 1 else 1.0 - G1[w])
    if c1 == 0:
        return float(mass * (1.0 - C1[w, a1]))
    mass *= C1[w, a1]
    # An event of either cause ends the unit's history here, exactly as censoring does --
    # but it contributes its *own* hazard, and what survives to the next node is one minus
    # the sum over causes rather than one minus this cause's.
    if j1 != 0:
        return float(mass * h1[j1 - 1, w, a1])
    mass *= 1.0 - h1[:, w, a1].sum()
    mass *= P_L2[w, a1] if l2 == 1 else 1.0 - P_L2[w, a1]
    mass *= G2[w, a1, l2] if a2 == 1 else 1.0 - G2[w, a1, l2]
    if c2 == 0:
        return float(mass * (1.0 - C2[w, a1, l2, a2]))
    mass *= C2[w, a1, l2, a2]
    if j2 != 0:
        return float(mass * h2[j2 - 1, w, a1, l2, a2])
    return float(mass * (1.0 - h2[:, w, a1, l2, a2].sum()))


def counts(h1: Any = None, h2: Any = None) -> np.ndarray:
    """Rows per support point in an ``N``-row sample that realises the law exactly.

    ``None`` means this module's own hazards; an array replaces that node's cause-specific
    hazards and leaves the treatment, censoring and covariate laws alone.  Both guards below
    apply to a derived law as much as to this one, which is the reason
    :func:`probabilities` routes through here rather than returning the raw products.
    """
    h1 = H1 if h1 is None else h1
    h2 = H2 if h2 is None else h2
    values = np.array([_mass_of(point, h1=h1, h2=h2) * N for point in SUPPORT])
    rounded = np.rint(values)
    if np.max(np.abs(values - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- keep every conditional a multiple of 1/4"
        )
    if np.min(rounded) < 1:  # pragma: no cover - guards the constants
        raise AssertionError(
            "a support point has no rows, which would leave first_row_of() pointing at "
            "the next block and silently compare the influence curve at the wrong row -- "
            "keep every cause-specific hazard strictly positive"
        )
    return rounded.astype(int)


def probabilities(h1: Any = None, h2: Any = None) -> np.ndarray:
    """Cell probabilities, optionally with the cause-specific hazards replaced.

    Taken from the counts rather than from the products, as both sibling laws do, so a law
    derived here is bit-for-bit the empirical law of the sample :func:`frame` would lay out
    for it.  Every exactness argument in this module rests on that property, and a derived
    law has to *inherit* it rather than approximately share it -- which is why this route
    rounds to whole rows and refuses a hazard off the quarter grid instead of normalising
    whatever it was handed.

    Property studies build their sharp-null laws through here, so the guard is not
    hypothetical: a null off the grid samples perfectly well and silently stops agreeing
    with any exact-law control built on :func:`frame`.
    """
    return counts(h1, h2) / N


COUNTS = counts()

#: ``P`` over the support, taken from the counts so it is bit-for-bit the empirical law
#: of :func:`frame`.
PROBS = COUNTS / N


@cache
def _index(**pattern: int) -> tuple[int, ...]:
    """Support points matching every named node, memoised as in the sibling laws.

    A ``None`` entry never equals an integer, so a point whose follow-up ended early is
    excluded from any pattern naming a later node without that having to be said.
    """
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
    """``"always, death @ t=2"`` -> ``("always, death", 2)``.

    Split the horizon off the end *first*, before the cause and before ``" vs "``: a
    contrast's index is ``"always vs never, death @ t=2"``, and taking the pieces in any
    other order leaves the horizon or the cause attached to a regimen label.
    """
    label, _, horizon = index.rpartition(" @ t=")
    return label, int(horizon)


def _split_cause(index: str) -> tuple[str, str]:
    """``"always vs never, death"`` -> ``("always vs never", "death")``.

    Run after :func:`_split_horizon` and before any split on ``" vs "``.
    """
    label, _, cause = index.rpartition(", ")
    return label, cause


def functional(probs: Any, estimand: str) -> Any:
    r"""The cause-specific cumulative incidence, as a closed-form function of the cells.

    Written straight off the iterated conditional expectation for competing absorbing
    events, and sharing no code with the library:

    .. math::

        F_j(1) &= \sum_w P(W = w)\, h_{1j}(w, a_1) \\
        F_j(2) &= \sum_w P(W = w)\Bigl[h_{1j} + S_1(w, a_1)
                  \sum_{l} P(L_2 = l \mid W, A_1 = a_1, C_1 = 1, J_1 = 0)\,
                  h_{2j}(w, a_1, l, a_2)\Bigr]

    with :math:`S_1 = P(J_1 = 0 \mid W, A_1 = a_1, C_1 = 1)`, the probability of leaving
    the first node event-free **by either cause**.  That factor is the whole of what
    competing risks add, and it is why ``survived`` below is a mass over the cells with
    ``j1 = 0`` rather than ``1 - hazard1``: the two agree only when there is one cause,
    and they differ by exactly the mass that left through the other one.  The numerator is
    cause-specific and the survival factor is all-cause; swapping either reading is the
    mistake the negative controls in the test module are written to catch.

    Note also where the second sum conditions -- on ``j1 = 0`` as well as ``c1 = 1``,
    because a unit that had *either* event has no ``L2`` and is not the population the
    second hazard is about.

    Every operation is a sum, a difference or a quotient of sums, so this stays analytic
    in the cell probabilities -- which is what lets :func:`gateaux` differentiate it by a
    complex step.  Do not introduce ``clip``, ``abs`` or a comparison here.
    """
    p = probs
    if estimand.startswith("ate_regimen["):
        rest, horizon = _split_horizon(estimand[len("ate_regimen[") : -1])
        contrast, cause = _split_cause(rest)
        left, right = contrast.split(" vs ")
        return functional(p, f"cif_regimen[{left}, {cause} @ t={horizon}]") - functional(
            p, f"cif_regimen[{right}, {cause} @ t={horizon}]"
        )
    if not estimand.startswith("cif_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")

    rest, horizon = _split_horizon(estimand[len("cif_regimen[") : -1])
    label, cause = _split_cause(rest)
    j = CAUSES.index(cause) + 1
    node1, node2 = REGIMEN_ARMS[label]
    total = _mass(p)
    psi = 0.0
    for w in (0, 1):
        a1 = _arm(node1, w)
        share = _mass(p, w=w) / total
        reached = _mass(p, w=w, a1=a1, c1=1)
        hazard1 = _mass(p, w=w, a1=a1, c1=1, j1=j) / reached
        if horizon == 1:
            psi = psi + share * hazard1
            continue
        # All-cause, and deliberately not ``1 - hazard1``.
        survived = _mass(p, w=w, a1=a1, c1=1, j1=0)
        later = 0.0
        for l2 in (0, 1):
            a2 = _arm(node2, w, l2)
            density = _mass(p, w=w, a1=a1, c1=1, j1=0, l2=l2) / survived
            uncensored = _mass(p, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1)
            events = _mass(p, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1, j2=j)
            later = later + density * (events / uncensored)
        psi = psi + share * (hazard1 + (survived / reached) * later)
    return psi


def survival_functional(probs: Any, label: str, horizon: int) -> Any:
    """``P(event-free by ``horizon`` under ``label``)`` -- all causes, longhand.

    Not a parameter any fit reports.  It exists so the test module can state the identity
    that ties the causes together, :math:`\\sum_j F_j(k) + S(k) = 1`, against something
    other than one minus the sum of the very numbers being checked.
    """
    node1, node2 = REGIMEN_ARMS[label]
    total = _mass(probs)
    psi = 0.0
    for w in (0, 1):
        a1 = _arm(node1, w)
        share = _mass(probs, w=w) / total
        reached = _mass(probs, w=w, a1=a1, c1=1)
        survived = _mass(probs, w=w, a1=a1, c1=1, j1=0)
        if horizon == 1:
            psi = psi + share * (survived / reached)
            continue
        later = 0.0
        for l2 in (0, 1):
            a2 = _arm(node2, w, l2)
            density = _mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2) / survived
            uncensored = _mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1)
            free = _mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1, j2=0)
            later = later + density * (free / uncensored)
        psi = psi + share * (survived / reached) * later
    return psi


#: The parameter names a competing-risks fit reports on this law, in report order:
#: regimen outer, then cause, then horizon, means before contrasts -- which is the order
#: ``_estimates`` builds them in, and the gate in the test module compares the two sets
#: both ways.
NAMES: tuple[str, ...] = tuple(
    f"cif_regimen[{label}, {cause} @ t={horizon}]"
    for label in REGIMEN_ARMS
    for cause in CAUSES
    for horizon in HORIZONS
) + tuple(
    f"ate_regimen[{label} vs {REGIMEN_REFERENCE}, {cause} @ t={horizon}]"
    for label in REGIMEN_ARMS
    if label != REGIMEN_REFERENCE
    for cause in CAUSES
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
    The three-valued event nodes are expanded here into one indicator column per cause,
    which is the form a fit is handed; a node the unit never reached is ``nan`` in *every*
    cause's column, including every node after the one where it had an event, which is the
    missingness pattern :class:`~cleverly.longitudinal.LongitudinalData` requires of an
    absorbing outcome.
    """
    cells = np.repeat(np.arange(len(SUPPORT)), COUNTS)
    columns = {}
    for name, node, code in _COLUMNS:
        position = _NODES.index(node)
        values = np.array(
            [
                np.nan
                if point[position] is None
                else float(point[position] if code is None else point[position] == code)
                for point in SUPPORT
            ]
        )
        columns[name] = values[cells]
    return pd.DataFrame(columns)


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    return np.concatenate([[0], np.cumsum(COUNTS)[:-1]])
