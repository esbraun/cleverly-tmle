"""A coarse law of a vector treatment with two doses, for the density route's exact witnesses.

The treatment is ``A = (F, D1, D2)``: a binary flag and two doses on the geometric grid
``{1, 2, 4, 8}``.  Support index ``k = 16 f + 4 i1 + i2``, with ``i1`` and ``i2`` the grid
positions of the doses, and covariate ``W`` in ``{0, 1}``.  The law factors as

.. math::

    P(w)\\, g_F(f \\mid w)\\, g_1(d_1 \\mid f, w)\\, g_2(d_2 \\mid d_1, f, w)\\,
    \\bar Q(f, d_1, d_2, w),

and ``g_2`` reads ``d_1``, so a policy that moves dose 1 changes the conditioning of dose 2.
The same frame is read twice: as a point treatment by ``TMLE`` and as one mixed node by
``LTMLE``.  Both readings estimate the same policy mean, so they share this oracle.

**The exactness conditions** (plan review rev 3, H3).  The binned density of each dose reads
the law's grid, whose bin ``[0.75 x, 1.5 x)`` around a point ``x`` has width ``0.75 x``.  A
``Scale(2)`` map has inverse derivative ``1 / 2``, and the source bin is half as wide as its
target bin, so Equation (3) of Díaz, Williams, Hoffman and Schenck (2023) equals the discrete
formula at every moved point (the condition of ``tests/discrete_law_policy_point.py``).  The
hazard learner is saturated over its design rows (``CellProbabilities``).  The frame realises
the law by integer counts.  Every cell ``(c, d_1)`` has positive probability, so every
conditioning design of dose 2 at a positive earlier factor is in training.  A preimage off the
grid, such as ``1 / 2``, has an estimated density of zero, and the product of the factors is
then zero whatever dose 2's model predicts there.

Each policy is a fixed map of support indices per covariate value, and the functional is
written longhand from the cell probabilities, so a complex step differentiates it to full
precision.  The parameters were chosen by a search over small rational tables for the law
whose policy means lie farthest apart: "ordered" and its shifted-dose-1 reading differ by
``0.039``, and "full" and its shifted-flag reading by ``0.032``.  Every pair of
deterministic policies differs by at least ``0.0014``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .discrete_law_policy_point import Branches, _mean_under

#: The dose grid of both doses, and its bin edges, ``0.75 x`` below each point.
DOSES = (1.0, 2.0, 4.0, 8.0)
EDGES = (0.75, 1.5, 3.0, 6.0, 12.0)
#: ``(f, i1, i2)`` of each support index.
CELLS = tuple(itertools.product(range(2), range(4), range(4)))
#: ``P(W = w)``.
P_W = np.array([0.5, 0.5])
#: ``P(F = 1 | W = w)``, in quarters.
P_F = np.array([0.25, 0.75])
#: Dose distributions in eighths.  Every entry is positive.
_POOL = (
    (1, 2, 3, 2),
    (2, 3, 2, 1),
    (3, 2, 2, 1),
    (1, 1, 3, 3),
    (2, 2, 2, 2),
    (1, 3, 1, 3),
    (4, 2, 1, 1),
    (1, 1, 2, 4),
)
#: ``g_1(. | f, w)`` by ``(w, f)``, as an index into the pool.
_G1 = {(0, 0): 1, (0, 1): 3, (1, 0): 2, (1, 1): 6}
#: The outcome probabilities of ``Qbar``, in eighths.
_Q_VALUES = (0.125, 0.375, 0.625, 0.875)


def g1(w: int, f: int, i1: int) -> float:
    """``g_1(d_1 | f, w)``."""
    return _POOL[_G1[(w, f)]][i1] / 8.0


def g2(w: int, f: int, i1: int, i2: int) -> float:
    """``g_2(d_2 | d_1, f, w)``, which reads the first dose."""
    return _POOL[(3 * w + 7 * f + 3 * i1) % 8][i2] / 8.0


def qbar(w: int, k: int) -> float:
    """``Qbar(f, d_1, d_2, w)``."""
    f, i1, i2 = CELLS[k]
    return _Q_VALUES[(2 * w + 2 * f + 3 * i1 + i2 + 3 * i1 * i2) % 4]


def g(w: int, k: int) -> float:
    """``g(f, d_1, d_2 | w)``, by the chain rule."""
    f, i1, i2 = CELLS[k]
    flag = P_F[w] if f == 1 else 1.0 - P_F[w]
    return float(flag * g1(w, f, i1) * g2(w, f, i1, i2))


#: Rows: every cell probability is a multiple of ``1 / N``.
N = 4096
#: ``(w, k, y)`` in the order :func:`frame` emits rows.
SUPPORT = tuple(itertools.product(range(2), range(len(CELLS)), range(2)))


def _counts() -> np.ndarray:
    counts = np.array(
        [P_W[w] * g(w, k) * (qbar(w, k) if y else 1.0 - qbar(w, k)) * N for w, k, y in SUPPORT]
    )
    assert np.allclose(counts, np.round(counts), atol=1e-9), "the law is not realised by N rows"
    assert int(np.round(counts).sum()) == N
    assert np.all(np.round(counts) > 0)
    return np.round(counts).astype(int)


COUNTS = _counts()
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """``N`` rows realising the law, in :data:`SUPPORT` order: ``W, F, D1, D2, Y``."""
    rows = []
    for (w, k, y), count in zip(SUPPORT, COUNTS, strict=True):
        f, i1, i2 = CELLS[k]
        rows.extend([(float(w), float(f), DOSES[i1], DOSES[i2], float(y))] * int(count))
    return pd.DataFrame(rows, columns=["W", "F", "D1", "D2", "Y"])


def sample(n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows from the law, for fits whose nuisances are not exact."""
    rng = np.random.default_rng(seed)
    points = np.asarray(SUPPORT)[rng.choice(len(SUPPORT), size=n, p=PROBS)]
    cells = np.asarray(CELLS)[points[:, 1]]
    return pd.DataFrame(
        {
            "W": points[:, 0].astype(float),
            "F": cells[:, 0].astype(float),
            "D1": np.asarray(DOSES)[cells[:, 1]],
            "D2": np.asarray(DOSES)[cells[:, 2]],
            "Y": points[:, 2].astype(float),
        }
    )


def first_row_of() -> np.ndarray:
    """Index of the first row of each support point, in :data:`SUPPORT` order."""
    return np.concatenate([[0], np.cumsum(COUNTS)[:-1]]).astype(int)


# ------------------------------------------------------------------ policies


def _index(f: int, i1: int, i2: int) -> int:
    return 16 * f + 4 * i1 + i2


def _up(i: int) -> int:
    """``Scale(2, cap=8)`` on the grid: double, and hold the top."""
    return min(i + 1, 3)


def _ordered(f: int, i1: int, i2: int) -> int:
    """W15: double dose 1, and double dose 2 where the **natural** dose 1 is at most 2."""
    return _index(f, _up(i1), _up(i2) if i1 <= 1 else i2)


def _ordered_shifted(f: int, i1: int, i2: int) -> int:
    """The M13 reading of W15: dose 2's map reads dose 1 after the policy doubled it."""
    return _index(f, _up(i1), _up(i2) if _up(i1) <= 1 else i2)


def _reverse_ordered(f: int, i1: int, i2: int) -> int:
    """Double the doses 1 and 2 where the **natural** dose 2 is at most 2.

    Dose 1's map reads dose 2, so the policy declares ``order=("D2", "D1")``.  The law is
    still factored with dose 1 first: the order governs reads, not the factorization.
    """
    small = i2 <= 1
    return _index(f, _up(i1) if small else i1, _up(i2) if small else i2)


def _full(f: int, i1: int, i2: int) -> int:
    """Every term of the formula at once.

    The flag goes to one where the natural dose 1 is at least 4: a categorical map that
    reads a dose, and many-to-one.  Dose 1 doubles where the natural flag is one: a dose map
    that reads the candidate cell, with a non-unit Jacobian.  Dose 2 doubles where the
    natural dose 1 is at most 2: an ordered read.
    """
    flag = 1 if i1 >= 2 else f
    return _index(flag, _up(i1) if f == 1 else i1, _up(i2) if i1 <= 1 else i2)


def _full_shifted(f: int, i1: int, i2: int) -> int:
    """The M16 reading of "full": dose 1's map reads the flag after the policy set it."""
    flag = 1 if i1 >= 2 else f
    return _index(flag, _up(i1) if flag == 1 else i1, _up(i2) if i1 <= 1 else i2)


def _scale_both(f: int, i1: int, i2: int) -> int:
    return _index(f, _up(i1), _up(i2))


def _map(assign: Callable[[int, int, int], int]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(assign(*CELLS[k]) for k in range(len(CELLS))) for _ in range(2))


#: Every policy, as ``(probability, map)`` branches with ``map[w][k]`` the target index.
POLICIES: dict[str, Branches] = {
    "natural": ((1.0, _map(lambda f, i1, i2: _index(f, i1, i2))),),
    "ordered": ((1.0, _map(_ordered)),),
    "ordered (shifted dose 1)": ((1.0, _map(_ordered_shifted)),),
    "reverse ordered": ((1.0, _map(_reverse_ordered)),),
    "full": ((1.0, _map(_full)),),
    "full (shifted flag)": ((1.0, _map(_full_shifted)),),
    "scale both": ((1.0, _map(_scale_both)),),
    "random": ((0.5, _map(_full)), (0.5, _map(_scale_both))),
}


def functional(probs: Any, label: str) -> Any:
    """The policy mean of ``label`` from the ``(2 * 32 * 2,)`` cell probabilities, longhand."""
    cells = np.asarray(probs).reshape(2, len(CELLS), 2)
    joint = cells[:, :, 0] + cells[:, :, 1]
    q = cells[:, :, 1] / joint
    return _mean_under(joint, q, POLICIES[label])


def truth(label: str) -> float:
    """The parameter of ``label`` at the law."""
    return float(np.real(functional(PROBS, label)))


def eif(label: str) -> np.ndarray:
    """The complex-step efficient influence function at every support point."""
    base = np.asarray(PROBS, dtype=complex)
    step = 1e-30
    out = []
    for point in range(base.size):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        out.append(float(np.imag(functional(perturbed, label)) / step))
    return np.array(out)
