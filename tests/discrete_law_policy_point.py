"""Finite laws for checking point-treatment modified treatment policies exactly.

The companion of ``tests/discrete_law_shift.py`` for every policy class beyond the additive
shift: a multiplicative scale, a piecewise policy whose pieces have different slopes, a
declared decreasing piece, a piece boundary that depends on the covariate, a randomized
policy, and policies on a binary and a three-level treatment.  The functional is written
longhand and shares no code with ``src/``; each policy is a fixed map of support indices
per covariate value, computed once outside the functional, so the functional stays
analytic in the cell probabilities and a complex step differentiates it to full precision.

**The exactness condition for a non-unit slope.**  A binned density is
:math:`\\hat g(a) = p(b(a)) / w(b(a))`, with :math:`w` the width of the bin holding
:math:`a`.  Equation (3) of Díaz, Williams, Hoffman and Schenck (2023) multiplies
:math:`\\hat g(b_j(a))` by :math:`|b_j'(a)|`, so it equals the discrete formula
:math:`g^d(a) = \\sum_s 1\\{d(s) = a\\}\\, p(s)` exactly when
:math:`w(b_j(a)) = |b_j'(a)|\\, w(a)` at every moved support point.  The multiplicative grid
doubles its bin width with the dose, and the piecewise grid gives the slope-two piece a
source bin half as wide as its target bin.  :func:`check_exactness` asserts the condition
for every grid and policy, so a law edited into violating it fails before any test reads it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#: Rows.  Every cell probability is a multiple of ``1 / N``: ``P(W)`` in tenths, ``g`` in
#: twentieths and ``Qbar`` in tenths.
N = 2000

#: ``P(W = w)``, in tenths.
P_W = np.array([0.50, 0.30, 0.20])

#: A policy as one or more ``(probability, map)`` branches.  ``map[w][k]`` is the support
#: index the branch assigns a unit with covariate ``w`` and observed dose index ``k``.
Branches = tuple[tuple[float, tuple[tuple[int, ...], ...]], ...]


@dataclass(frozen=True)
class Grid:
    """One finite law: a dose support, its bin edges, ``g(a | w)`` and ``Qbar(a, w)``.

    Parameters
    ----------
    name : str
        The grid's name.
    doses : tuple of float
        The support of the treatment.
    edges : ndarray
        Bin edges of the density, each support point in its own bin; empty bins hold no
        mass.
    g : ndarray
        ``(3, K)`` treatment probabilities, rows summing to one.
    q : ndarray
        ``(3, K)`` outcome probabilities.
    """

    name: str
    doses: tuple[float, ...]
    edges: np.ndarray
    g: np.ndarray
    q: np.ndarray

    @property
    def support(self) -> tuple[tuple[int, int, int], ...]:
        """``(w, k, y)`` in the order :meth:`frame` emits rows."""
        return tuple(itertools.product(range(3), range(len(self.doses)), range(2)))

    @property
    def counts(self) -> np.ndarray:
        """``N * P(w, a, y)`` as integers, asserting that the constants are exact."""
        out = np.zeros((3, len(self.doses), 2))
        for w, k, y in self.support:
            probability = P_W[w] * self.g[w, k] * (self.q[w, k] if y else 1.0 - self.q[w, k])
            exact = probability * N
            assert abs(exact - round(exact)) < 1e-9, (self.name, w, k, y, probability)
            out[w, k, y] = round(exact)
        assert out.sum() == N
        return out.astype(int)

    @property
    def probs(self) -> np.ndarray:
        """``(3, K, 2)`` cell probabilities."""
        return self.counts / N

    def frame(self, treatment: Callable[[int], object] | None = None) -> pd.DataFrame:
        """``N`` rows realising the law exactly, blocks in :attr:`support` order."""
        counts = self.counts
        label = (lambda k: float(self.doses[k])) if treatment is None else treatment
        rows = [
            (float(w), label(k), float(y))
            for (w, k, y) in self.support
            for _ in range(int(counts[w, k, y]))
        ]
        return pd.DataFrame(rows, columns=["W", "A", "Y"])

    def first_row_of(self) -> np.ndarray:
        """Index of the first row of each support point, in :attr:`support` order."""
        counts = self.counts
        starts = np.cumsum([0] + [int(counts[w, k, y]) for (w, k, y) in self.support])[:-1]
        return np.asarray(starts, dtype=int)

    def bins(self, rows: np.ndarray) -> np.ndarray:
        """``(n, B)`` bin probabilities from ``(n, K)`` support probabilities.

        An empty bin, which holds no support point, has probability zero.
        """
        rows = np.asarray(rows, dtype=float)
        out = np.zeros((rows.shape[0], self.edges.size - 1))
        for k, dose in enumerate(self.doses):
            out[:, int(np.digitize(dose, self.edges) - 1)] = rows[:, k]
        return out

    def bin_width(self, k: int) -> float:
        """The width of the bin holding support point ``k``."""
        index = int(np.digitize(self.doses[k], self.edges) - 1)
        return float(self.edges[index + 1] - self.edges[index])

    def index_of(self, value: float) -> int:
        """The support index of a dose, refusing a dose off the support."""
        matches = [k for k, dose in enumerate(self.doses) if dose == value]
        assert len(matches) == 1, (self.name, value)
        return matches[0]


#: Unit-width bins on ``{0, 1, 2, 3}``, the shift law's support.
UNIT = Grid(
    "unit",
    (0.0, 1.0, 2.0, 3.0),
    np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
    np.array([[8.0, 6.0, 4.0, 2.0], [2.0, 4.0, 6.0, 8.0], [5.0, 5.0, 5.0, 5.0]]) / 20.0,
    np.array([[0.2, 0.4, 0.6, 0.8], [0.3, 0.3, 0.5, 0.7], [0.1, 0.5, 0.5, 0.9]]),
)

#: A multiplicative support ``{1, 2, 4, 8}`` whose bins double in width with the dose, so a
#: scale by two satisfies the exactness condition of the module docstring.
MULTIPLICATIVE = Grid(
    "multiplicative",
    (1.0, 2.0, 4.0, 8.0),
    np.array([0.75, 1.5, 3.0, 6.0, 12.0]),
    np.array([[6.0, 6.0, 4.0, 4.0], [2.0, 4.0, 6.0, 8.0], [5.0, 3.0, 7.0, 5.0]]) / 20.0,
    np.array([[0.2, 0.5, 0.4, 0.8], [0.3, 0.4, 0.6, 0.7], [0.1, 0.6, 0.5, 0.9]]),
)

#: A support for a piecewise policy with slopes two and one: the slope-two piece maps 1 to
#: 2, whose bin is twice as wide as 1's, and the slope-one piece maps 2, 4 and 6 two up, onto
#: bins of equal width.  The empty bins between the support points hold no mass.
PIECEWISE = Grid(
    "piecewise",
    (1.0, 2.0, 4.0, 6.0, 8.0),
    np.array([0.75, 1.25, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
    np.array([[5.0, 4.0, 4.0, 4.0, 3.0], [2.0, 3.0, 5.0, 5.0, 5.0], [4.0, 4.0, 4.0, 4.0, 4.0]])
    / 20.0,
    np.array([[0.2, 0.4, 0.5, 0.7, 0.8], [0.3, 0.3, 0.6, 0.5, 0.9], [0.1, 0.5, 0.4, 0.8, 0.6]]),
)

#: A binary treatment, coded 0 and 1.
BINARY = Grid(
    "binary",
    (0.0, 1.0),
    np.array([-0.5, 0.5, 1.5]),
    np.array([[14.0, 6.0], [8.0, 12.0], [10.0, 10.0]]) / 20.0,
    np.array([[0.2, 0.6], [0.3, 0.5], [0.4, 0.9]]),
)

#: A three-level treatment with labels 1, 2 and 3.
THREE_LEVEL = Grid(
    "three level",
    (1.0, 2.0, 3.0),
    np.array([0.5, 1.5, 2.5, 3.5]),
    np.array([[8.0, 6.0, 6.0], [4.0, 6.0, 10.0], [6.0, 8.0, 6.0]]) / 20.0,
    np.array([[0.2, 0.5, 0.8], [0.4, 0.3, 0.7], [0.1, 0.6, 0.5]]),
)

GRIDS = {grid.name: grid for grid in (UNIT, MULTIPLICATIVE, PIECEWISE, BINARY, THREE_LEVEL)}


def _deterministic(grid: Grid, assign: Callable[[int, float], float]) -> Branches:
    """One branch from ``assign(w, dose)``, as support indices per covariate value."""
    return (
        (
            1.0,
            tuple(tuple(grid.index_of(assign(w, dose)) for dose in grid.doses) for w in range(3)),
        ),
    )


def _identity(grid: Grid) -> Branches:
    return _deterministic(grid, lambda w, a: a)


#: The boundary of the covariate-dependent piece: doses below ``1.5 + 0.5 w`` move up one.
def boundary(w: float) -> float:
    """``u(w)``, the upper end of the moving piece of the ``"boundary"`` policy."""
    return 1.5 + 0.5 * w


#: Every policy the tests fit, by grid and report label.  Each is the policy written as
#: maps of support indices; the estimator side declares the same policy as an object.
POLICIES: dict[str, dict[str, Branches]] = {
    "unit": {
        "natural course": _identity(UNIT),
        "decreasing": _deterministic(UNIT, lambda w, a: 3.0 - a),
        "boundary": _deterministic(UNIT, lambda w, a: a + 1.0 if a < boundary(w) else a),
        "randomized": (
            (0.25, _deterministic(UNIT, lambda w, a: a - 1.0 if a >= 2.5 else a)[0][1]),
            (0.75, _identity(UNIT)[0][1]),
        ),
    },
    "multiplicative": {
        "natural course": _identity(MULTIPLICATIVE),
        "x2": _deterministic(MULTIPLICATIVE, lambda w, a: 2.0 * a if 2.0 * a <= 8.0 else a),
    },
    "piecewise": {
        "natural course": _identity(PIECEWISE),
        "piecewise": _deterministic(
            PIECEWISE, lambda w, a: 2.0 * a if a < 1.5 else a + 2.0 if a < 7.0 else a
        ),
    },
    "binary": {
        "natural course": _identity(BINARY),
        "rr 0.5": (
            (0.5, _identity(BINARY)[0][1]),
            (0.5, _deterministic(BINARY, lambda w, a: 0.0)[0][1]),
        ),
        "rr 4": (
            (0.25, _identity(BINARY)[0][1]),
            (0.75, _deterministic(BINARY, lambda w, a: 1.0)[0][1]),
        ),
        "treat W=1": _deterministic(BINARY, lambda w, a: 1.0 if w == 1 else a),
    },
    "three level": {
        "natural course": _identity(THREE_LEVEL),
        "drop above 2": _deterministic(THREE_LEVEL, lambda w, a: a - 1.0 if a > 2.0 else a),
    },
}

#: The Jacobian ``|b'|`` of each moved support point, by grid and policy, for the
#: exactness condition.  A unit-slope policy is absent: its condition is equal widths.
_SLOPES: dict[tuple[str, str], Callable[[float], float]] = {
    ("multiplicative", "x2"): lambda a: 0.5,
    ("piecewise", "piecewise"): lambda a: 0.5 if a < 1.5 else 1.0,
}


def check_exactness() -> None:
    """Assert ``w(source) = |b'| w(target)`` at every moved point of every continuous law.

    Raises
    ------
    AssertionError
        If a grid's bins do not make Equation (3) equal the discrete formula.
    """
    for grid_name in ("unit", "multiplicative", "piecewise"):
        grid = GRIDS[grid_name]
        for label, branches in POLICIES[grid_name].items():
            slope = _SLOPES.get((grid_name, label), lambda a: 1.0)
            for _, mapping in branches:
                for w in range(3):
                    for k, target in enumerate(mapping[w]):
                        if target == k:
                            continue
                        lhs = grid.bin_width(k)
                        rhs = slope(grid.doses[k]) * grid.bin_width(target)
                        assert abs(lhs - rhs) < 1e-12, (grid_name, label, w, k, lhs, rhs)


check_exactness()


def functional(probs: Any, grid: str, estimand: str) -> Any:
    """The estimand, longhand, sharing no code with ``src/``.

    ``probs`` is the ``(3, K, 2)`` array of cell probabilities, and may be complex.
    Arithmetic only: every comparison lives in :data:`POLICIES`.

    Parameters
    ----------
    probs : ndarray
        Cell probabilities.
    grid : str
        The law's grid.
    estimand : str
        ``"<name>[label]"`` or ``"<name>[label vs reference]"`` for a level or a contrast,
        where ``<name>`` is ``ey_policy``, ``ate_policy``, ``ey_rr_tilt`` or
        ``ate_rr_tilt``.

    Returns
    -------
    complex or float
        The parameter at ``probs``.
    """
    joint = probs[:, :, 0] + probs[:, :, 1]
    qbar = probs[:, :, 1] / joint
    name, _, rest = estimand.partition("[")
    inner = rest[:-1]
    if name in ("ey_policy", "ey_rr_tilt"):
        return _mean_under(joint, qbar, POLICIES[grid][inner])
    if name in ("ate_policy", "ate_rr_tilt"):
        left, right = inner.split(" vs ")
        return _mean_under(joint, qbar, POLICIES[grid][left]) - _mean_under(
            joint, qbar, POLICIES[grid][right]
        )
    raise ValueError(f"no oracle branch for {estimand!r}")


def _mean_under(joint: Any, qbar: Any, branches: Branches) -> Any:
    """``sum_e p(e) sum_{w, a} P(w, a) Qbar(d_e(a, w), w)``."""
    total: Any = 0.0
    for probability, mapping in branches:
        for w in range(joint.shape[0]):
            for k, target in enumerate(mapping[w]):
                total = total + probability * joint[w, k] * qbar[w, target]
    return total


def truth(grid: str, estimand: str) -> float:
    """The parameter at the law."""
    return float(np.real(functional(GRIDS[grid].probs, grid, estimand)))


def gateaux(grid: str, estimand: str, point: int, *, step: float = 1e-30) -> float:
    """The Gateaux derivative at one support point, by complex step."""
    law = GRIDS[grid]
    base = law.probs.astype(complex)
    mass = np.zeros_like(base)
    mass[law.support[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, grid, estimand)) / step)


def eif(grid: str, estimand: str) -> np.ndarray:
    """The influence curve at every support point, in :attr:`Grid.support` order."""
    return np.array([gateaux(grid, estimand, point) for point in range(len(GRIDS[grid].support))])
