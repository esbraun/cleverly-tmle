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
"""

from __future__ import annotations

import itertools
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
REGIMENS: dict[str, tuple[float, float]] = {
    "never": (0.0, 0.0),
    "always": (1.0, 1.0),
    "early": (1.0, 0.0),
}

REGIMEN_REFERENCE = "never"

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


def _mass_of(point: tuple[Any, ...]) -> float:
    """``P`` of one support point, as a product of the conditionals above."""
    w, a1, c1, l2, a2, c2, y = point
    mass = P_W[w] * (G1[w] if a1 == 1 else 1.0 - G1[w])
    if c1 == 0:
        return float(mass * (1.0 - C1[w, a1]))
    mass *= C1[w, a1]
    mass *= P_L2[w, a1] if l2 == 1 else 1.0 - P_L2[w, a1]
    mass *= G2[w, a1, l2] if a2 == 1 else 1.0 - G2[w, a1, l2]
    if c2 == 0:
        return float(mass * (1.0 - C2[w, a1, l2, a2]))
    mass *= C2[w, a1, l2, a2]
    return float(mass * (Q[w, a1, l2, a2] if y == 1 else 1.0 - Q[w, a1, l2, a2]))


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


def _index(**pattern: int) -> tuple[int, ...]:
    """Support points matching every named node."""
    return tuple(
        position
        for position, point in enumerate(SUPPORT)
        if all(point[_NODES.index(node)] == value for node, value in pattern.items())
    )


def _mass(probs: Any, **pattern: int) -> Any:
    """Total probability of the matching support points -- a linear form in ``probs``."""
    return sum(probs[position] for position in _index(**pattern))


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
    """
    p = probs
    if estimand.startswith("ate_regimen["):
        left, right = estimand[len("ate_regimen[") : -1].split(" vs ")
        return functional(p, f"ey_regimen[{left}]") - functional(p, f"ey_regimen[{right}]")
    if not estimand.startswith("ey_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")

    a1, a2 = (int(arm) for arm in REGIMENS[estimand[len("ey_regimen[") : -1]])
    total = _mass(p)
    psi = 0.0
    for w in (0, 1):
        share = _mass(p, w=w) / total
        reached = _mass(p, w=w, a1=a1, c1=1)
        for l2 in (0, 1):
            density = _mass(p, w=w, a1=a1, c1=1, l2=l2) / reached
            uncensored = _mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1)
            events = _mass(p, w=w, a1=a1, c1=1, l2=l2, a2=a2, c2=1, y=1)
            psi = psi + share * density * (events / uncensored)
    return psi


#: The parameter names a longitudinal fit reports on this law, in report order.
NAMES: tuple[str, ...] = tuple(f"ey_regimen[{label}]" for label in REGIMENS) + tuple(
    f"ate_regimen[{label} vs {REGIMEN_REFERENCE}]"
    for label in REGIMENS
    if label != REGIMEN_REFERENCE
)

#: Population values of every reported parameter.
TRUTH = {name: float(functional(PROBS, name)) for name in NAMES}


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
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeans:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        keys, inverse = np.unique(np.round(matrix, 9), axis=0, return_inverse=True)
        totals = np.bincount(inverse, weights=target, minlength=keys.shape[0])
        sizes = np.bincount(inverse, minlength=keys.shape[0])
        self.keys_ = keys
        self.means_ = totals / np.maximum(sizes, 1)
        self.default_ = float(target.mean())
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
