r"""A three-arm finite-support law with outcomes missing at random, realised exactly.

The three-arm sibling of :mod:`tests.discrete_law_mar`. It carries the exact-law checks of
the multi-arm missing-outcome DR-TMLE in ``tests/unit/test_drtmle_missing_multi_arm.py``.
A row reveals ``(W, A, Delta)`` always and ``Y`` only when ``Delta = 1``, so the support is

.. math::

    (w, a, k), \qquad k \in \{\,\Delta = 1, Y = 0;\ \ \Delta = 1, Y = 1;\ \ \Delta = 0\,\},

27 cells in a ``(3, 3, 3)`` array. Each cell probability is a multiple of ``1 / N``, so
:func:`frame` *is* the law and not a sample from it. The module asserts that at import.

**Why ``W`` has three levels.** Each arm's outcome tilt has two parameters: one on
``1{A=a} Delta / (g_a pi_a)`` and one on the outcome-drift covariate. With two ``W`` levels
per arm the tilt saturates: its maximum-likelihood solution is the cell mean of ``Y`` in each
``(a, w)`` cell. On an exact frame that mean is the true ``Q``, so every reduction vanishes
and a both-wrong control cannot fail. :mod:`tests.discrete_law_mar` uses three levels for the
same reason.

The tables are quarters and eighths. In every row, any two arm columns differ, and every arm
takes three distinct ``Q`` and ``PI`` values across ``W``, so a column swap is visible. The
smallest cell is ``(1/4)(1/8)(1/4)(1/4) = 1/512``, so ``N = 1024`` gives every cell at least
two rows. ``min g * pi = 1/32``, so no default bound binds.

The law is also exposed as a :class:`tests.studies.mar_arm_indexed_laws.Law`, so the oracle
learners ``LawOutcome``, ``LawTreatment`` and ``LawResponse`` serve it. Those learners read
only the ``Law`` tables.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.mar_arm_indexed_laws import Law

#: Rows in the realised sample.
N = 1024

#: The treatment labels, in the column order of every table. ``CausalData`` sorts them, so
#: arm code 0 is ``"high"``, and ``"high"`` is the reference arm of every contrast.
LABELS: tuple[str, ...] = ("low", "mid", "high")
CODES: tuple[str, ...] = tuple(sorted(LABELS))
REFERENCE = "high"
K = len(LABELS)

#: ``P(W = w)``.
P_W = np.array([1 / 2, 1 / 4, 1 / 4])

#: ``P(A = a | W = w)``, indexed ``[w, a]``. A known W-stratified randomization.
G = np.array([[4 / 8, 3 / 8, 1 / 8], [2 / 8, 1 / 8, 5 / 8], [1 / 8, 4 / 8, 3 / 8]])

#: ``P(Delta = 1 | A = a, W = w)``, indexed ``[w, a]``.
PI = np.array([[2 / 4, 3 / 4, 1 / 4], [1 / 4, 2 / 4, 3 / 4], [3 / 4, 1 / 4, 2 / 4]])

#: ``P(Y = 1 | A = a, Delta = 1, W = w)``, indexed ``[w, a]``.
Q = np.array([[1 / 4, 2 / 4, 3 / 4], [3 / 4, 1 / 4, 2 / 4], [2 / 4, 3 / 4, 1 / 4]])

#: The observation drift: a wrong ``P(Delta = 1 | A, W)``.
WRONG_PI = np.array([[3 / 4, 1 / 4, 2 / 4], [2 / 4, 3 / 4, 1 / 4], [1 / 4, 2 / 4, 3 / 4]])

#: The outcome drift: a wrong ``Q``.
WRONG_Q = 1.0 - Q

OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = 0, 1, 2

#: The support, ordered ``(w, a, k)`` with ``a`` the table column. Row blocks follow it.
SUPPORT: tuple[tuple[int, int, int], ...] = tuple(itertools.product(range(3), range(K), range(3)))


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = a, K = k)`` as a ``(3, K, 3)`` integer array."""
    counts = np.empty((3, K, 3))
    for w, a, k in SUPPORT:
        base = P_W[w] * G[w, a]
        if k == UNOBSERVED:
            counts[w, a, k] = base * (1.0 - PI[w, a]) * N
        else:
            outcome = Q[w, a] if k == OBSERVED_ONE else 1.0 - Q[w, a]
            counts[w, a, k] = base * PI[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-9:  # pragma: no cover - guards the constants
        raise AssertionError("a cell probability is not a multiple of 1/N")
    if rounded.min() < 2:  # pragma: no cover - guards the constants
        raise AssertionError("every cell needs at least two rows")
    return rounded.astype(int)


#: Cell counts in the realised sample. Integral by construction, checked above.
COUNTS = _cell_counts()

#: ``P(W, A, K)``, bit for bit the empirical law of :func:`frame`.
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    ``A`` carries the labels. ``Y`` is ``NaN`` wherever ``Delta`` is zero.
    """
    counts = [COUNTS[w, a, k] for w, a, k in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    kind = columns[:, 2]
    return pd.DataFrame(
        {
            "W": columns[:, 0],
            "A": np.asarray(LABELS, dtype=object)[columns[:, 1].astype(int)],
            "Y": np.where(kind == UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
        }
    )


def cell_of_row() -> np.ndarray:
    """The support index of each row of :func:`frame`."""
    counts = [COUNTS[w, a, k] for w, a, k in SUPPORT]
    return np.repeat(np.arange(len(SUPPORT)), counts)


def _others() -> list[str]:
    return [label for label in CODES if label != REFERENCE]


#: Every reported name on a binary outcome, in the engine's report order by stem.
NAMES: tuple[str, ...] = (
    *(f"ey[{label}]" for label in CODES),
    *(f"ate[{label} vs {REFERENCE}]" for label in _others()),
    *(f"rr[{label} vs {REFERENCE}]" for label in _others()),
    *(f"or[{label} vs {REFERENCE}]" for label in _others()),
)


def functional(probs: Any, name: str) -> Any:
    r"""The target as a closed-form function of the cell probabilities.

    ``E(Y_a) = sum_w P(W = w) P(Y = 1 | A = a, Delta = 1, W = w)``. ``rr`` and ``or`` are on
    the log scale, which is the scale of their reported influence curves. Every operation is
    arithmetic or :func:`numpy.log`, so the complex step in :func:`gateaux` stays exact.
    """
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))
    observed = p[:, :, OBSERVED_ZERO] + p[:, :, OBSERVED_ONE]
    q = p[:, :, OBSERVED_ONE] / observed

    def mean(label: str) -> Any:
        return (p_w * q[:, LABELS.index(label)]).sum()

    head, _, rest = name.partition("[")
    rest = rest.rstrip("]")
    if head == "ey":
        return mean(rest)
    arm, reference = rest.split(" vs ")
    one, zero = mean(arm), mean(reference)
    if head == "ate":
        return one - zero
    if head == "rr":
        return np.log(one) - np.log(zero)
    if head == "or":
        return np.log(one / (1.0 - one)) - np.log(zero / (1.0 - zero))
    raise ValueError(f"unknown estimand {name!r}")


def natural(name: str) -> float:
    """The truth on the reported scale: ``rr`` and ``or`` as ratios, not logs."""
    value = float(functional(PROBS, name))
    return float(np.exp(value)) if name.startswith(("rr[", "or[")) else value


def gateaux(name: str, point: int, *, step: float = 1e-30) -> float:
    """The Gateaux derivative of ``name`` at support point ``point``, by complex step."""
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, name)) / step)


def eif(name: str) -> np.ndarray:
    """The EIF of ``name`` at every support point, in support order."""
    return np.array([gateaux(name, point) for point in range(len(SUPPORT))])


#: The law as the arm-indexed oracle learners read it. ``key`` only names it.
LAW = Law(
    key="x3",
    scenario="three_arm_mar_exact",
    p_w=P_W,
    g=G,
    pi=PI,
    mu=Q,
    labels=LABELS,
    continuous=False,
)


def probabilities_by_label(frame_: pd.DataFrame, table: np.ndarray = G) -> dict[str, np.ndarray]:
    """Known treatment probabilities as a mapping keyed by level, one column per arm."""
    w = np.rint(frame_["W"].to_numpy(dtype=float)).astype(int)
    return {label: table[w, LABELS.index(label)] for label in LABELS}
