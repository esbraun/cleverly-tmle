r"""A two-time-point law that a finite sample realises *exactly*.

The longitudinal counterpart of :mod:`tests.discrete_law`, and it exists for the same
reason: the efficient influence function of the sequential g-formula is a property of a
distribution, so checking that the library computes the right one needs a distribution
the test can hold in its hand.  Every node is binary --

.. code-block:: text

    W  ->  A1  ->  C1  ->  L2  ->  A2  ->  C2  ->  Y

-- and every conditional probability is a multiple of ``1/4``, so every cell probability
is a multiple of ``1 / N`` with ``N = 2 * 4**6``.  Laying ``N`` rows out in the cell
proportions makes the empirical distribution *equal* to the data-generating one, which
buys the three things the point-treatment law buys: closed-form population quantities, an
initial fit that is exact in the sample (so every targeting step's score is zero at
:math:`\epsilon = 0` and the reported influence curve is the EIF at :math:`P_0`), and
assertions that can be exact rather than statistical.

The law is deliberately *not* a sequence of two point-treatment problems.  ``L2`` depends
on ``A1`` and both ``A2`` and ``Y`` depend on ``L2``, so it is a confounder of the second
decision and a consequence of the first -- the structure that makes the sequential
regression necessary.  Censoring depends on the history at both nodes, so a fit that
dropped the censoring factors from the cumulative product would miss the truth here
rather than merely lose efficiency.

Positivity holds comfortably: every conditional probability lies in ``[0.25, 0.75]``, so
no truncation is active and the estimator runs on the unmodified mechanism.

Both kinds of regimen answer to the same law.  Because ``W`` and ``L2`` are binary, a
dynamic rule :math:`d_t(H_t)` on this law is a lookup over four cells, which is what lets
the oracle state one longhand and the estimator be handed the same plan as a callable --
see :data:`REGIMEN_ARMS` and :data:`REGIMEN_SPEC`.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

#: Rows in the realised sample.  ``2 * 4**6``: one factor of two for ``P(W)`` and six of
#: four for the conditional probabilities that follow it.
N = 2 * 4**6

P_W = np.array([0.5, 0.5])

#: ``P(A1 = 1 | W = w)``.
G1 = np.array([0.50, 0.25])

#: ``P(C1 = 1 | W, A1)`` -- still under observation after the first node.
C1 = np.array([[0.75, 0.50], [0.75, 0.75]])

#: ``P(L2 = 1 | W, A1, C1 = 1)``.  Depends on ``A1``: this is the node that makes the
#: problem longitudinal rather than two cross-sections.
P_L2 = np.array([[0.25, 0.75], [0.50, 0.75]])

#: ``P(A2 = 1 | W, A1, L2, C1 = 1)``, indexed ``[w, a1, l2]``.
G2 = np.array([[[0.50, 0.75], [0.25, 0.50]], [[0.75, 0.50], [0.50, 0.25]]])

#: ``P(C2 = 1 | W, A1, L2, A2)``, indexed ``[w, a1, l2, a2]``.
C2 = np.array(
    [
        [[[0.75, 0.50], [0.75, 0.75]], [[0.50, 0.75], [0.75, 0.50]]],
        [[[0.75, 0.75], [0.50, 0.75]], [[0.75, 0.50], [0.75, 0.75]]],
    ]
)

#: ``P(Y = 1 | W, A1, L2, A2, C2 = 1)``, indexed ``[w, a1, l2, a2]``.
Q = np.array(
    [
        [[[0.25, 0.50], [0.50, 0.75]], [[0.50, 0.75], [0.75, 0.50]]],
        [[[0.50, 0.25], [0.75, 0.50]], [[0.25, 0.50], [0.50, 0.75]]],
    ]
)

#: The regimens the estimands are checked against, in report order.  The first is the
#: reference every contrast is taken against, as the estimator defaults.
#:
#: **The oracle side.**  Arms as a lookup over *cell indices*: a scalar is a constant
#: node, an array at the first node is indexed ``[w]`` and at the second ``[w, l2]``.
#: The estimator is handed :data:`REGIMEN_SPEC` instead, which states the same six plans
#: as callables over a dataframe.  Writing them twice, in two representations, is
#: deliberate: a slip in one is a wrong number rather than a mistake that cancels against
#: itself on both sides of the comparison.
REGIMEN_ARMS: dict[str, tuple[Any, Any]] = {
    "never": (0, 0),
    "always": (1, 1),
    "early": (1, 0),
    # A rule that ignores the history.  It must reproduce ``always`` exactly, which is
    # the strongest single assertion here: it fails if the follower masks, the mechanism's
    # arm selection, the censoring model's current-arm column, the outcome design or the
    # submodel's key treats a rule differently from a constant.
    "always_rule": (np.array([1, 1]), np.array([[1, 1], [1, 1]])),
    # d_2 = 1{L2 = 1}.  ``L2`` is caused by ``A1`` and causes ``Y``, so no static plan
    # reaches this parameter -- which is the case the module exists for.
    "treat_if_l2": (0, np.array([[0, 1], [0, 1]])),
    # Dynamic at the *first* node, which ``treat_if_l2`` never exercises: the only case
    # where the follower mask at ``t = 1`` compares against a per-row value.
    "respond": (np.array([0, 1]), np.array([[1, 0], [1, 0]])),
}

#: The estimator side of :data:`REGIMEN_ARMS`: what ``LTMLE`` is handed as ``regimens=``.
#: A rule is called with ``[W]`` at the first node and ``[W, L2]`` at the second, so
#: ``treat_if_l2`` cannot be written as one callable for both nodes.
REGIMEN_SPEC: dict[str, Any] = {
    "never": 0,
    "always": 1,
    "early": (1, 0),
    "always_rule": (lambda h: np.ones(len(h)), lambda h: np.ones(len(h))),
    "treat_if_l2": (0, lambda h: h["L2"]),
    "respond": (lambda h: h["W"], lambda h: 1.0 - h["L2"]),
}

REGIMEN_REFERENCE = "never"

if set(REGIMEN_ARMS) != set(REGIMEN_SPEC):  # pragma: no cover - guards the two tables
    raise AssertionError(
        "REGIMEN_ARMS and REGIMEN_SPEC must state the same regimens; a plan the oracle "
        "knows and the estimator is never handed proves nothing, and the reverse is a "
        "parameter with no truth to check against"
    )

#: One support point per observable history.  ``None`` marks a node the unit never
#: reached: a unit censored at the first time point has no ``L2``, no ``A2`` and no ``Y``,
#: which is exactly the missingness pattern the container requires.
SUPPORT: tuple[tuple[Any, ...], ...] = tuple(
    [(w, a1, 0, None, None, None, None) for w, a1 in itertools.product(range(2), range(2))]
    + [
        (w, a1, 1, l2, a2, 0, None)
        for w, a1, l2, a2 in itertools.product(range(2), range(2), range(2), range(2))
    ]
    + [
        (w, a1, 1, l2, a2, 1, y)
        for w, a1, l2, a2, y in itertools.product(range(2), range(2), range(2), range(2), range(2))
    ]
)

_NODES = ("w", "a1", "c1", "l2", "a2", "c2", "y")


def _mass_of(point: tuple[Any, ...], outcome: Any = None) -> float:
    """``P`` of one support point, as a product of the conditionals above.

    ``outcome`` replaces :data:`Q` and nothing else, which is what lets a caller state a law
    that shares this one's treatment, censoring and confounder mechanisms while answering
    differently at the final node -- a sharp null, for instance.  Everything upstream of ``Y``
    is deliberately not a parameter: a null that also moved the mechanism would be a different
    experiment rather than the same one under a different truth.
    """
    w, a1, c1, l2, a2, c2, y = point
    q = Q if outcome is None else outcome
    mass = P_W[w] * (G1[w] if a1 == 1 else 1.0 - G1[w])
    if c1 == 0:
        return float(mass * (1.0 - C1[w, a1]))
    mass *= C1[w, a1]
    mass *= P_L2[w, a1] if l2 == 1 else 1.0 - P_L2[w, a1]
    mass *= G2[w, a1, l2] if a2 == 1 else 1.0 - G2[w, a1, l2]
    if c2 == 0:
        return float(mass * (1.0 - C2[w, a1, l2, a2]))
    mass *= C2[w, a1, l2, a2]
    return float(mass * (q[w, a1, l2, a2] if y == 1 else 1.0 - q[w, a1, l2, a2]))


def counts(outcome: Any = None) -> np.ndarray:
    """Rows per support point in an ``N``-row sample that realises the law exactly."""
    values = np.array([_mass_of(point, outcome) * N for point in SUPPORT])
    rounded = np.rint(values)
    if np.max(np.abs(values - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- keep every conditional a multiple of 1/4"
        )
    return rounded.astype(int)


def probabilities(outcome: Any = None) -> np.ndarray:
    """Cell probabilities, optionally with :data:`Q` replaced by ``outcome``.

    Taken from the counts rather than from the products, so a law built here is bit-for-bit
    the empirical law of the sample :func:`frame` would lay out for it -- the property every
    exactness argument in this module rests on, and one a derived law has to inherit rather
    than approximately share.
    """
    return counts(outcome) / N


COUNTS = counts()

#: ``P`` over the support, taken from the counts so it is bit-for-bit the empirical law
#: of :func:`frame`.
PROBS = COUNTS / N


@cache
def _index(**pattern: int) -> tuple[int, ...]:
    """Support points matching every named node.

    Memoised because :func:`functional` calls it about thirty times per evaluation and
    :func:`eif` evaluates the functional once per support point per parameter -- and the
    number of parameters went from five to eleven when dynamic rules were added.  The
    result is an immutable tuple over a fixed ``SUPPORT``, so caching it cannot go stale.
    """
    return tuple(
        position
        for position, point in enumerate(SUPPORT)
        if all(point[_NODES.index(node)] == value for node, value in pattern.items())
    )


def _arm(node: Any, *index: int) -> int:
    """The arm a plan's node assigns in the cell named by ``index``.

    A scalar node is a constant; an array node is a rule, read as a lookup over the
    cells.  Selecting by cell *index* is what keeps :func:`functional` analytic in the
    cell probabilities: the arm is fixed before the complex perturbation is applied, so
    no comparison ever touches ``probs``.
    """
    return int(node) if np.ndim(node) == 0 else int(node[index])


def _mass(probs: Any, **pattern: int) -> Any:
    """Total probability of the matching support points -- a linear form in ``probs``."""
    return sum(probs[position] for position in _index(**pattern))


#: How long each regimen treats for, as a constant of the *plan*.
#:
#: Declared here rather than computed from anything, for the reason :func:`_arm` selects
#: an arm by cell index: the design has to be fixed before the complex perturbation
#: reaches ``probs``, or :func:`functional` stops being analytic.  Two of the six share a
#: value, which is deliberate -- a design that separated every regimen would be saturated
#: however few its terms.
REGIMEN_INTENSITY: dict[str, float] = {
    "never": 0.0,
    "always": 2.0,
    "early": 1.0,
    "always_rule": 2.0,
    "treat_if_l2": 0.5,
    "respond": 1.0,
}

#: Terms of the working model the coefficients below belong to.
MSM_REGIMEN_TERMS = ("(intercept)", "intensity", "W")

#: :math:`\varphi(c, W = w)` as ``(2, R, 3)`` and :math:`h(c, w)` as ``(2, R)``, built
#: outside :func:`functional` so that no integer index or comparison sits inside a
#: function that must stay analytic.
#:
#: **Deliberately not saturated**: three coefficients against twelve ``(w, regimen)``
#: cells, so :math:`\beta` really is a projection and code that dropped :math:`M^{-1}`,
#: or that reported some regimen's mean under a coefficient's name, cannot pass.  A
#: saturated design agrees with the means whatever the projection code does.
#:
#: **Deliberately non-uniform** ``h``: with :math:`h \equiv 1` the design can go
#: orthogonal across the cells and a coefficient collapse into something the per-regimen
#: report already gives.  ``test_influence_gateaux_longitudinal_msm`` asserts on this
#: law that the two weightings give different answers, so the choice is shown to be
#: load-bearing rather than asserted to be.
MSM_REGIMEN_DESIGN = np.array(
    [[[1.0, REGIMEN_INTENSITY[label], float(w)] for label in REGIMEN_ARMS] for w in range(2)]
)
MSM_REGIMEN_WEIGHTS = np.array(
    [[1.0 + 0.5 * REGIMEN_INTENSITY[label] + 0.25 * w for label in REGIMEN_ARMS] for w in range(2)]
)

#: A second, **saturated** design: one indicator per regimen, uniform weights.  Its
#: coefficients must be the six ``ey_regimen`` truths exactly, which is what says a
#: working model summarises the regimens rather than replacing them.
MSM_SATURATED_TERMS = tuple(REGIMEN_ARMS)
MSM_SATURATED_DESIGN = np.array(
    [[list(row) for row in np.eye(len(REGIMEN_ARMS))] for _ in range(2)]
)
MSM_SATURATED_WEIGHTS = np.ones((2, len(REGIMEN_ARMS)))

#: ``(inverse, dm/deta, d2m/deta2)`` per link, as functions of the *mean*.  Written
#: longhand rather than imported, as ``tests/discrete_law.py`` writes them: ``expit`` is
#: spelled out because the argument here is complex and ``scipy.special.expit`` is not.
MSM_LINKS: dict[str, tuple[Any, Any, Any]] = {
    "identity": (lambda eta: eta, lambda m: np.ones_like(m), lambda m: np.zeros_like(m)),
    "log": (np.exp, lambda m: m, lambda m: m),
    "logit": (
        lambda eta: 1.0 / (1.0 + np.exp(-eta)),
        lambda m: m * (1.0 - m),
        lambda m: m * (1.0 - m) * (1.0 - 2.0 * m),
    ),
}

#: Newton steps taken to solve a linked working model's normal equations, run **without a
#: convergence test** -- a comparison against a tolerance is not analytic, and a functional
#: that branched on one could not be differentiated by a complex step at all.  Newton
#: converges quadratically in the value and in the derivative alike, so past the point
#: where the real part stops moving the imaginary part is exact too; a test doubles this
#: and checks that neither moves.  The same constant, and the same reasoning, as
#: ``tests/discrete_law.MSM_NEWTON_STEPS``.
MSM_NEWTON_STEPS = 40


def _conditional_mean(probs: Any, label: str, w: int) -> Any:
    r""":math:`E[Y^{\bar a} \mid W = w]` for one regimen.

    Written out separately rather than factored out of :func:`functional`'s
    ``ey_regimen`` branch, and deliberately: two representations of one formula is the
    house style here -- :data:`REGIMEN_ARMS` against :data:`REGIMEN_SPEC` -- because a
    shared inner loop lets a slip cancel against itself on both sides of a comparison.
    """
    node1, node2 = REGIMEN_ARMS[label]
    a1 = _arm(node1, w)
    reached = _mass(probs, w=w, a1=a1, c1=1)
    mean = 0.0
    for l2 in (0, 1):
        a2 = _arm(node2, w, l2)
        density = _mass(probs, w=w, a1=a1, c1=1, l2=l2) / reached
        uncensored = _mass(probs, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1)
        events = _mass(probs, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, y=1)
        mean = mean + density * (events / uncensored)
    return mean


def msm_coefficients(
    probs: Any,
    *,
    design: Any = MSM_REGIMEN_DESIGN,
    weights: Any = MSM_REGIMEN_WEIGHTS,
    link: str = "identity",
    steps: int | None = None,
) -> Any:
    r"""The working model's coefficient vector: the :math:`h`-weighted projection.

    :math:`\beta` minimises
    :math:`\sum_w P(w) \sum_c h(c, w)\,(E[Y^c \mid w] - m(c, w; \beta))^2`, which under
    the identity link is two einsums and a solve and under a link is Newton's method on
    the same normal equations.

    ``np.linalg.solve`` on complex input is what lets the complex step differentiate
    *through* the solve -- including through ``p_w``, which is where :math:`M`'s own
    contribution to the influence curve comes from.  Code treating :math:`M` as a
    constant fails on exactly that term.
    """
    p = probs
    total = _mass(p)
    p_w = np.array([_mass(p, w=0) / total, _mass(p, w=1) / total])
    q = np.array(
        [[_conditional_mean(p, label, w) for label in REGIMEN_ARMS] for w in range(2)],
        dtype=p_w.dtype,
    )
    inverse, slope, curvature = MSM_LINKS[link]
    if link == "identity":
        gram = np.einsum("wcp,wcq,wc,w->pq", design, design, weights, p_w)
        moment = np.einsum("wcp,wc,wc,w->p", design, weights, q, p_w)
        return np.linalg.solve(gram, moment)
    beta = np.zeros(design.shape[2], dtype=p_w.dtype)
    for _ in range(MSM_NEWTON_STEPS if steps is None else steps):
        m = inverse(np.einsum("wcp,p->wc", design, beta))
        residual = q - m
        first, second = slope(m), curvature(m)
        score = np.einsum("wcp,wc,w->p", design, weights * first * residual, p_w)
        jacobian = np.einsum(
            "wcp,wcq,wc,w->pq", design, design, weights * (first**2 - residual * second), p_w
        )
        beta = beta + np.linalg.solve(jacobian, score)
    return beta


#: ``family -> (design, weights, terms)``.  The families are named apart in the *oracle*
#: only: a fit reports every one of them as ``msm_regimen[<term>]``, since which design
#: and which link it declared is a statement it made rather than part of the name.  The
#: estimator side of each is built in the test module, from these very arrays.
_MSM_FAMILIES: dict[str, tuple[Any, Any, tuple[str, ...]]] = {
    "identity": (MSM_REGIMEN_DESIGN, MSM_REGIMEN_WEIGHTS, MSM_REGIMEN_TERMS),
    "log": (MSM_REGIMEN_DESIGN, MSM_REGIMEN_WEIGHTS, MSM_REGIMEN_TERMS),
    "logit": (MSM_REGIMEN_DESIGN, MSM_REGIMEN_WEIGHTS, MSM_REGIMEN_TERMS),
    "saturated": (MSM_SATURATED_DESIGN, MSM_SATURATED_WEIGHTS, MSM_SATURATED_TERMS),
}


def msm_names(family: str) -> tuple[str, ...]:
    """The oracle's names for one working-model family, in report order."""
    terms = _MSM_FAMILIES[family][2]
    head = "msm_regimen[" if family == "identity" else f"msm_regimen_{family}["
    return tuple(f"{head}{term}]" for term in terms)


def functional(probs: Any, estimand: str) -> Any:
    r"""The target parameter as a closed-form function of the cell probabilities.

    Written straight off the longitudinal g-formula,

    .. math::

        \Psi = \sum_w P(W = w) \sum_{l} P(L_2 = l \mid W, A_1 = a_1, C_1 = 1)\,
               E[Y \mid W, A_1 = a_1, L_2 = l, A_2 = a_2, C_2 = 1]

    and sharing no code with the library, so comparing against it is a check rather than
    a restatement.  Every operation is a sum or a quotient of sums, so this stays
    analytic in the cell probabilities -- which is what lets :func:`gateaux` differentiate
    it by a complex step.  Do not introduce ``clip``, ``abs`` or a comparison here.

    A **dynamic** regimen changes only where the arms come from: :func:`_arm` reads them
    off the cell being summed over, so :math:`a_1` depends on ``w`` and :math:`a_2` on
    ``(w, l2)``.  That keeps the analyticity above intact, because the arms are integers
    fixed by the index before any perturbation reaches ``probs``.  A rule expressed as an
    indicator *of the probabilities* would break it, and would break it silently -- the
    complex step would come back real.
    """
    p = probs
    if estimand.startswith("ate_regimen["):
        left, right = estimand[len("ate_regimen[") : -1].split(" vs ")
        return functional(p, f"ey_regimen[{left}]") - functional(p, f"ey_regimen[{right}]")
    for name, (design, weights, terms) in _MSM_FAMILIES.items():
        head = "msm_regimen[" if name == "identity" else f"msm_regimen_{name}["
        if estimand.startswith(head):
            column = terms.index(estimand[len(head) : -1])
            link = "identity" if name in ("identity", "saturated") else name
            return msm_coefficients(p, design=design, weights=weights, link=link)[column]
    if not estimand.startswith("ey_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")

    node1, node2 = REGIMEN_ARMS[estimand[len("ey_regimen[") : -1]]
    total = _mass(p)
    psi = 0.0
    for w in (0, 1):
        a1 = _arm(node1, w)
        share = _mass(p, w=w) / total
        reached = _mass(p, w=w, a1=a1, c1=1)
        for l2 in (0, 1):
            a2 = _arm(node2, w, l2)
            density = _mass(p, w=w, a1=a1, c1=1, l2=l2) / reached
            uncensored = _mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1)
            events = _mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, y=1)
            psi = psi + share * density * (events / uncensored)
    return psi


#: The parameter names a longitudinal fit reports on this law, in report order.
NAMES: tuple[str, ...] = tuple(f"ey_regimen[{label}]" for label in REGIMEN_ARMS) + tuple(
    f"ate_regimen[{label} vs {REGIMEN_REFERENCE}]"
    for label in REGIMEN_ARMS
    if label != REGIMEN_REFERENCE
)

#: Population values of every reported parameter.
TRUTH = {name: float(functional(PROBS, name)) for name in NAMES}

#: The working-model parameter names, per family.  Kept **apart** from :data:`NAMES`:
#: the gate in ``test_influence_gateaux_longitudinal`` asserts that a fit with no ``msm=``
#: reports exactly ``NAMES``, and a working model replaces that report rather than adding
#: to it.
MSM_NAMES: dict[str, tuple[str, ...]] = {family: msm_names(family) for family in _MSM_FAMILIES}

#: Population values of every working-model coefficient, per family.
MSM_TRUTH: dict[str, float] = {
    name: float(functional(PROBS, name)) for names in MSM_NAMES.values() for name in names
}


def gateaux(estimand: str, point: int, *, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\,
                 \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)\right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function -- derived here from :func:`functional` alone, with no
    clever covariate, no cumulative product and nothing else the library supplies.

    Differentiation is by complex step, for the reason ``tests/discrete_law.py`` gives:
    the imaginary part is carried separately, so there is no subtractive cancellation and
    the derivative comes back to full double precision.
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

    Rows are laid out in :data:`SUPPORT` order, one contiguous block per support point,
    so :func:`first_row_of` locates a representative row for each.  A node the unit never
    reached is ``nan``, which is what :class:`~cleverly.longitudinal.LongitudinalData`
    requires of the columns after a censoring time.
    """
    cells = np.repeat(np.arange(len(SUPPORT)), COUNTS)
    columns = {
        name: np.array(
            [np.nan if point[position] is None else float(point[position]) for point in SUPPORT]
        )[cells]
        for position, name in enumerate(("W", "A1", "C1", "L2", "A2", "C2", "Y"))
    }
    return pd.DataFrame(columns)


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    return np.concatenate([[0], np.cumsum(COUNTS)[:-1]])


# ------------------------------------------------------------------ observation weights
#
# The same construction as ``tests/discrete_law.py``, and it exists here for the same
# reason: ``cleverly.data.weighting`` claims a weighted fit estimates ``Psi(P_w)`` on the
# tilted law, and that claim is checked by writing ``Psi(P_w)`` out longhand rather than by
# agreeing with the library's own arithmetic.  A weight is a function of the observed row,
# so on a law with finite support it is one number per support point.


def cell_weights(weight_of: Any) -> np.ndarray:
    """A weight per support point, from a function of the whole observed history.

    ``weight_of`` is called with ``(w, a1, c1, l2, a2, c2, y)``, the support point itself,
    so a weight may read any node -- including one a censored unit never reached, which
    arrives as ``None`` exactly as it does in :data:`SUPPORT`.
    """
    return np.array([float(weight_of(*point)) for point in SUPPORT], dtype=float)


def row_weights(weights: np.ndarray) -> np.ndarray:
    """Cell weights expanded to one value per row of :func:`frame`."""
    return np.repeat(np.asarray(weights, dtype=float), COUNTS)


def tilt(probs: Any, weights: Any) -> Any:
    r"""The weighted law :math:`dP_w = w\,dP / E_P[w]`, as cell probabilities.

    Kept analytic in ``probs`` -- a ratio of linear functions -- so
    :func:`weighted_gateaux` can differentiate through it by a complex step.
    """
    tilted = np.asarray(weights, dtype=float) * np.asarray(probs)
    return tilted / tilted.sum()


def weighted_functional(probs: Any, estimand: str, weights: Any) -> Any:
    """``Psi(P_w)`` -- the estimand of the tilted law, longhand.

    Tilt the law, then apply the same sequential g-formula :func:`functional` already
    spells out.  Nothing here touches a clever covariate, a cumulative product or a
    weighted score equation, so comparing a weighted fit against it is a check of the
    claim rather than a restatement of the implementation.
    """
    return functional(tilt(probs, weights), estimand)


def weighted_gateaux(
    estimand: str,
    point: int,
    weights: Any,
    *,
    base: Any | None = None,
    step: float = 1e-30,
) -> float:
    r"""Gateaux derivative of :math:`P \mapsto \Psi(P_w)` at support point ``point``.

    The contamination is of :math:`P`, the law the *rows are drawn from* -- not of
    :math:`P_w`.  That is the whole content of the check: the weights are part of the
    data-generating experiment, so the influence function has to be taken with respect to
    the law that generates them.
    """
    base_array = np.asarray(PROBS if base is None else base, dtype=complex)
    mass = np.zeros_like(base_array)
    mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base_array + 1j * step * mass
    return float(np.imag(weighted_functional(perturbed, estimand, weights)) / step)


def weighted_eif(estimand: str, weights: Any, *, base: Any | None = None) -> np.ndarray:
    """The EIF of ``Psi(P_w)`` at every support point, in support order."""
    return np.array(
        [weighted_gateaux(estimand, point, weights, base=base) for point in range(len(SUPPORT))]
    )


class CellMeans(BaseEstimator):
    """The saturated learner: the sample mean of ``y`` within each distinct design row.

    On a law this sample realises exactly, an unpenalised saturated fit *is* the oracle,
    which is what lets the test hand the estimator exact nuisances without writing each
    one out by hand.  That matters most for the sequential regression, whose oracle at the
    earlier node is an expectation of the *later* node's regression -- a quantity nobody
    would want to transcribe.

    A design row the fit never saw falls back to the training mean.  Only rows the
    estimator masks away can hit that path: a unit censored before the node in question
    has its history filled with zeros, and nothing reads its prediction.

    ``sample_weight`` is **honoured, not accepted and dropped**, and that is what makes the
    saturated fit the oracle of the *tilted* law as well as of ``P_0``.  A weighted cell
    mean is ``sum_cell w y / sum_cell w``, which on a sample that realises the law exactly
    is ``E_{P_w}[y | cell]`` exactly.  Discarding the weights would leave the estimator
    holding ``P_0``'s conditionals while its estimand is at ``P_w``, so ``epsilon`` would
    not come back zero and every exactness argument in the weighted test module would
    quietly become an approximation.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeans:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        keys, inverse = np.unique(np.round(matrix, 9), axis=0, return_inverse=True)
        totals = np.bincount(inverse, weights=weights * target, minlength=keys.shape[0])
        sizes = np.bincount(inverse, weights=weights, minlength=keys.shape[0])
        self.keys_ = keys
        # ``np.maximum(sizes, 0)`` would not do: a cell can carry zero total weight when
        # every row in it was zero-weighted, and the fallback there is the same "nothing
        # reads this prediction" case as an unseen design row.
        self.means_ = np.where(sizes > 0, totals / np.where(sizes > 0, sizes, 1.0), 0.0)
        self.default_ = float(np.average(target, weights=weights))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        out = np.full(matrix.shape[0], self.default_)
        for position, key in enumerate(self.keys_):
            out[np.all(matrix == key, axis=1)] = self.means_[position]
        return out

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])
