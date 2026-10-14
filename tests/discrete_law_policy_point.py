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
        # A piece closed on the right at the support point 1: the doses 0 and 1 move up one.
        # Closed on the left, the dose 1 would stay, so the closed end decides the answer.
        "right closed": _deterministic(UNIT, lambda w, a: a + 1.0 if a <= 1.0 else a),
        "right closed piece": _deterministic(UNIT, lambda w, a: a + 1.0 if a <= 1.0 else a),
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
        # The same policy with no piece above 7: an uncovered dose keeps its value.
        "piecewise gap": _deterministic(
            PIECEWISE, lambda w, a: 2.0 * a if a < 1.5 else a + 2.0 if a < 7.0 else a
        ),
    },
    "binary": {
        "natural course": _identity(BINARY),
        # Asymmetric on purpose: at delta = 0.5 the keep and drop branches have equal weight,
        # so a swap of the two probabilities would not move the mean.
        "rr 0.25": (
            (0.25, _identity(BINARY)[0][1]),
            (0.75, _deterministic(BINARY, lambda w, a: 0.0)[0][1]),
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
        # Closed on the right at the level 2, which is a support point.
        "right closed": _deterministic(THREE_LEVEL, lambda w, a: a + 1.0 if a <= 2.0 else a),
    },
}

#: The Jacobian ``|b'|`` of each moved support point, by grid and policy, for the
#: exactness condition.  A unit-slope policy is absent: its condition is equal widths.
_SLOPES: dict[tuple[str, str], Callable[[float], float]] = {
    ("multiplicative", "x2"): lambda a: 0.5,
    ("piecewise", "piecewise"): lambda a: 0.5 if a < 1.5 else 1.0,
    ("piecewise", "piecewise gap"): lambda a: 0.5 if a < 1.5 else 1.0,
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


# ------------------------------------------------------------------ declared atoms (X40)
#
# The unit grid with the dose 3 declared an atom.  The policies mirror the longitudinal atom
# plans of ``tests/discrete_law_longitudinal_mtp.py``: "to atom" moves 0 and 1 up one and
# sends 2 to the atom, "hold atom" leaves 2 and the atom in place, and "atom random" takes
# the first with probability 0.4.  The mixed ratio with respect to Lebesgue plus counting
# measure on the atom equals the discrete formula on this grid, because every moved
# continuous dose lands in a bin of the same width.

#: The declared atom, as a support index of :data:`UNIT` and as a dose.
ATOM_INDEX = 3
ATOM_DOSE = 3.0

ATOM_POLICIES: dict[str, Branches] = {
    "natural course": _identity(UNIT),
    "to atom": _deterministic(UNIT, lambda w, a: min(a + 1.0, 3.0)),
    "hold atom": _deterministic(UNIT, lambda w, a: a + 1.0 if a < 2.0 else a),
    "atom random": (
        (0.4, _deterministic(UNIT, lambda w, a: min(a + 1.0, 3.0))[0][1]),
        (0.6, _deterministic(UNIT, lambda w, a: a + 1.0 if a < 2.0 else a)[0][1]),
    ),
}


def atom_functional(probs: Any, label: str) -> Any:
    """``E[Y^d]`` under one of :data:`ATOM_POLICIES`, longhand."""
    joint = probs[:, :, 0] + probs[:, :, 1]
    return _mean_under(joint, probs[:, :, 1] / joint, ATOM_POLICIES[label])


def atom_truth(label: str) -> float:
    """The parameter at the law."""
    return float(np.real(atom_functional(UNIT.probs, label)))


def atom_eif(label: str, *, step: float = 1e-30) -> np.ndarray:
    """The influence curve at every support point of :data:`UNIT`, by complex step."""
    out = []
    base = UNIT.probs.astype(complex)
    for point in UNIT.support:
        mass = np.zeros_like(base)
        mass[point] = 1.0
        value = atom_functional((1.0 - 1j * step) * base + 1j * step * mass, label)
        out.append(float(np.imag(value) / step))
    return np.array(out)


# ------------------------------------------------------------------ mixed vector treatment
#
# A coarse law whose treatment is a vector ``A = (F, D)``: a binary flag and a dose on the
# grid ``{0, 1, 2}``.  Support index ``k = 3 f + d``, covariate ``W`` in ``{0, 1}``.  The fit
# reads ``D`` as a continuous component, so this is the oracle of the classifier route at a
# point vector treatment with a continuous component (Díaz, Williams, Hoffman and Schenck
# 2023, Section 2, Theorem 1 and Section 5.4).
#
# The saturated classifier returns the exact ratio under three conditions, which every exact
# fit of this law meets:
#
# 1. ``Folds.single``, unit weights, and a frame of :data:`MIXED_N` rows that realises the
#    law by integer counts.
# 2. Every policy stays on the grid, by a cap at the grid maximum, so every stacked cell with
#    label one also holds label-zero mass.  Otherwise the predicted probability is one and the
#    odds guard of ``cleverly.learners.density_ratio`` replaces the ratio.
# 3. The learners partition by distinct design row, so the encoding of the flag, an indicator
#    or a code, cannot change a prediction.
#
# The outcome learner is saturated too, so the targeting step moves nothing; the targeting
# witness on a sample with a misspecified outcome regression is W8.

#: The flag and the dose of each support index ``k = 3 f + d``.
MIXED_CELLS = tuple((f, d) for f in range(2) for d in range(3))
#: The dose values of the grid.
MIXED_DOSES = (0.0, 1.0, 2.0)
#: ``P(W = w)``.
MIXED_P_W = np.array([0.5, 0.5])
#: ``g(k | w)`` in eighths, indexed ``[w, k]``: both components depend on ``W``.
MIXED_G = np.array([[1.0, 2.0, 1.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 1.0, 2.0, 1.0]]) / 8.0
#: The outcome probabilities of ``Qbar``, in quarters.
_MIXED_Q_VALUES = (0.25, 0.5, 0.75)


def mixed_q(w: int, k: int) -> float:
    """``Qbar(k, w)``, which reads both components: the flag enters through ``2 f d``.

    Chosen by a search over modular patterns for the law whose policy means lie farthest
    apart: the natural course, "dose up", "flag gated", its shifted-flag reading and "dose
    gated" differ pairwise by at least 0.015, and "flag gated" and its shifted-flag reading
    by 0.047.
    """
    f, d = MIXED_CELLS[k]
    return _MIXED_Q_VALUES[(w + d + 2 * f * d + d * d) % 3]


#: Rows: every cell probability is a multiple of ``1 / 64``.
MIXED_N = 4 * 64
#: ``(w, k, y)`` in the order :func:`mixed_frame` emits rows.
MIXED_SUPPORT = tuple(itertools.product(range(2), range(6), range(2)))


def _mixed_counts() -> np.ndarray:
    counts = np.array(
        [
            MIXED_P_W[w] * MIXED_G[w, k] * (mixed_q(w, k) if y else 1.0 - mixed_q(w, k)) * MIXED_N
            for w, k, y in MIXED_SUPPORT
        ]
    )
    assert np.allclose(counts, np.round(counts), atol=1e-9), "the law is not realised by n rows"
    assert int(np.round(counts).sum()) == MIXED_N
    assert np.all(np.round(counts) > 0)
    return np.round(counts).astype(int)


MIXED_COUNTS = _mixed_counts()
MIXED_PROBS = MIXED_COUNTS / MIXED_N


def mixed_frame() -> pd.DataFrame:
    """``MIXED_N`` rows realising the mixed law, in :data:`MIXED_SUPPORT` order.

    The columns are ``W``, ``F``, ``D`` and ``Y``.
    """
    rows = []
    for (w, k, y), count in zip(MIXED_SUPPORT, MIXED_COUNTS, strict=True):
        f, d = MIXED_CELLS[k]
        rows.extend([(float(w), float(f), MIXED_DOSES[d], float(y))] * int(count))
    return pd.DataFrame(rows, columns=["W", "F", "D", "Y"])


def mixed_sample(n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows from the mixed law, for fits whose nuisances are not exact."""
    rng = np.random.default_rng(seed)
    points = np.asarray(MIXED_SUPPORT)[rng.choice(len(MIXED_SUPPORT), size=n, p=MIXED_PROBS)]
    cells = np.asarray(MIXED_CELLS)[points[:, 1]]
    return pd.DataFrame(
        {
            "W": points[:, 0].astype(float),
            "F": cells[:, 0].astype(float),
            "D": np.asarray(MIXED_DOSES)[cells[:, 1]],
            "Y": points[:, 2].astype(float),
        }
    )


def mixed_first_row_of() -> np.ndarray:
    """Index of the first row of each support point, in :data:`MIXED_SUPPORT` order."""
    return np.concatenate([[0], np.cumsum(MIXED_COUNTS)[:-1]]).astype(int)


def _index(f: int, d: int) -> int:
    return 3 * f + d


def _up(k: int) -> int:
    f, d = MIXED_CELLS[k]
    return _index(f, min(d + 1, 2))


def _flag_gated(k: int) -> int:
    """W4: set the flag to one, and raise the dose by one where the **natural** flag is one."""
    f, d = MIXED_CELLS[k]
    return _index(1, min(d + 1, 2) if f == 1 else d)


def _flag_gated_shifted(k: int) -> int:
    """The M3 reading of W4: the dose map reads the flag after the policy set it to one."""
    _, d = MIXED_CELLS[k]
    return _index(1, min(d + 1, 2))


def _dose_gated(k: int) -> int:
    """Keep the flag, and raise the dose by one where the flag is one."""
    f, d = MIXED_CELLS[k]
    return _index(f, min(d + 1, 2) if f == 1 else d)


def _flag_reads_dose(k: int) -> int:
    """W18: the flag goes to one where the **natural** dose is at least one; the dose rises.

    The categorical map reads a dose that the policy moves, so its preimage indicator is
    evaluated at the preimage of the dose.
    """
    f, d = MIXED_CELLS[k]
    return _index(1 if d >= 1 else f, min(d + 1, 2))


def _mixed_map(assign: Callable[[int], int]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(assign(k) for k in range(6)) for _ in range(2))


#: Every mixed policy, as ``(probability, map)`` branches with ``map[w][k]`` the target index.
MIXED_POLICIES: dict[str, Branches] = {
    "natural": ((1.0, _mixed_map(lambda k: k)),),
    "dose up": ((1.0, _mixed_map(_up)),),
    "flag gated": ((1.0, _mixed_map(_flag_gated)),),
    "flag gated (shifted flag)": ((1.0, _mixed_map(_flag_gated_shifted)),),
    "dose gated": ((1.0, _mixed_map(_dose_gated)),),
    "flag reads dose": ((1.0, _mixed_map(_flag_reads_dose)),),
    "random": ((0.25, _mixed_map(_up)), (0.75, _mixed_map(lambda k: k))),
    # The static intervention that sets every unit to ``(f, d) = (1, 2)``: its mean is
    # ``E[Y(1, 2)]``, the ``ey`` of that tuple level on an all-categorical reading.
    "static (1, 2)": ((1.0, _mixed_map(lambda k: _index(1, 2))),),
    # H5: a risk-ratio tilt of the flag is a two-point randomizer.  ``delta = 1/4`` keeps the
    # flag with probability 1/4 and sets it to zero otherwise; ``delta = 4`` keeps it with
    # probability 1/4 and sets it to one otherwise.  The last plan is the product of the
    # tilt up and the dose randomizer of "random", with product probabilities.
    "flag tilt down": (
        (0.25, _mixed_map(lambda k: k)),
        (0.75, _mixed_map(lambda k: _index(0, MIXED_CELLS[k][1]))),
    ),
    "flag tilt up": (
        (0.25, _mixed_map(lambda k: k)),
        (0.75, _mixed_map(lambda k: _index(1, MIXED_CELLS[k][1]))),
    ),
    "flag tilt up, random dose": (
        (0.25 * 0.25, _mixed_map(_up)),
        (0.25 * 0.75, _mixed_map(lambda k: k)),
        (0.75 * 0.25, _mixed_map(lambda k: _index(1, MIXED_CELLS[_up(k)][1]))),
        (0.75 * 0.75, _mixed_map(lambda k: _index(1, MIXED_CELLS[k][1]))),
    ),
}


def functional_mixed(probs: Any, label: str) -> Any:
    """The policy mean of ``label`` from the ``(2 * 6 * 2,)`` cell probabilities, longhand.

    ``probs`` may be complex.  Arithmetic only: every comparison lives in
    :data:`MIXED_POLICIES`.
    """
    cells = np.asarray(probs).reshape(2, 6, 2)
    joint = cells[:, :, 0] + cells[:, :, 1]
    qbar = cells[:, :, 1] / joint
    return _mean_under(joint, qbar, MIXED_POLICIES[label])


def mixed_truth(label: str) -> float:
    """The parameter of ``label`` at the mixed law."""
    return float(np.real(functional_mixed(MIXED_PROBS, label)))


def mixed_eif(label: str) -> np.ndarray:
    """The complex-step efficient influence function at every support point."""
    base = np.asarray(MIXED_PROBS, dtype=complex)
    step = 1e-30
    out = []
    for point in range(base.size):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        out.append(float(np.imag(functional_mixed(perturbed, label)) / step))
    return np.array(out)
