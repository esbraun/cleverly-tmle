r"""The point mixed law of :mod:`tests.discrete_law_policy_point`, with ``Delta`` and ``Z``.

The treatment is the vector ``A = (F, D)``: a binary flag and a dose on ``{0, 1, 2}``,
support index ``k = 3 f + d``, covariate ``W`` in ``{0, 1}``.  This module adds a recorded
outcome indicator ``Delta`` and a binary intermediate ``Z``, so a vector policy fit declared
with ``delta=``, ``intermediate=`` or ``strata=`` has an exact oracle.  It is the vector
analogue of :mod:`tests.discrete_law_shift_cde`, and it keeps that module's time order:

.. math::

    p(w, k, z, j) = p_W(w)\, g(k \mid w)\, q_z(k, w)\, \pi(k, w)\,
                    p_Y(\,\cdot \mid k, z, w, \Delta = 1),

with ``j`` in ``{Delta = 1, Y = 0; Delta = 1, Y = 1; Delta = 0}``, 72 cells in a
``(2, 6, 2, 3)`` array.  :math:`\pi` conditions on ``(A, W)`` and not on ``Z``.

**What makes the law see the vector.**  :data:`PI` and :data:`QZ` read both components:
each moves with the flag at a fixed dose and with the dose at a fixed flag.  A missingness
or intermediate design that drops a component therefore fits a different mechanism, and a
mechanism read at the observed value instead of the policy's value differs on every moved
cell.  :func:`check_nondegenerate` asserts each claim with a margin.

Every factor has a power-of-two denominator: ``P_W`` in halves, ``G`` in eighths, and
``QZ``, ``PI`` and ``QBAR`` in quarters, so each cell probability is a multiple of
``1 / N`` with ``N = 1024``.  :func:`frame` is the law, not a sample from it.  The
functional is arithmetic only, so a complex step gives its Gateaux derivative, which is the
efficient influence function of the observed-data model.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from tests.discrete_law_policy_point import (
    MIXED_CELLS,
    MIXED_DOSES,
    MIXED_G,
    MIXED_POLICIES,
    Branches,
)

#: Rows.  Every cell probability is a multiple of ``1 / N``.
N = 1024

#: ``P(W = w)``.
P_W = np.array([0.5, 0.5])

#: ``g(k | w)``, the point mixed law's, in eighths.
G = MIXED_G

#: ``pi(k, w) = P(Delta = 1 | A = cell k, W = w)``, indexed ``[w, k]``; ``k = 3 f + d``.
#: ``PI``, ``QZ`` and ``QBAR`` are the output of a search over quarters for the law whose
#: smallest margin in :func:`check_nondegenerate` is largest (0.038).
PI = np.array(
    [
        [0.50, 0.25, 0.50, 0.25, 0.50, 0.75],
        [0.50, 0.50, 0.25, 0.25, 0.25, 0.75],
    ]
)

#: ``q_z(k, w) = P(Z = 1 | A = cell k, W = w)``, indexed ``[w, k]``.
QZ = np.array(
    [
        [0.25, 0.25, 0.50, 0.50, 0.50, 0.50],
        [0.50, 0.25, 0.75, 0.25, 0.75, 0.75],
    ]
)

#: ``Qbar(k, z, w) = P(Y = 1 | A = cell k, Z = z, Delta = 1, W = w)``, indexed ``[w, k, z]``.
QBAR = np.array(
    [
        [[0.25, 0.25], [0.50, 0.75], [0.75, 0.75], [0.50, 0.25], [0.75, 0.75], [0.25, 0.25]],
        [[0.75, 0.25], [0.75, 0.25], [0.75, 0.75], [0.50, 0.25], [0.75, 0.50], [0.50, 0.25]],
    ]
)

#: The levels a controlled direct effect can be taken at.
LEVELS = (0, 1)

#: The last axis: outcome recorded and zero, recorded and one, or not recorded.
OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = 0, 1, 2

#: ``(w, k, z, j)`` in the order :func:`frame` emits rows.
SUPPORT: tuple[tuple[int, int, int, int], ...] = tuple(
    itertools.product(range(2), range(6), range(2), range(3))
)

#: The policies, as ``(probability, map)`` branches from the point mixed law.
POLICIES: dict[str, Branches] = MIXED_POLICIES


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = k, Z = z, J = j)`` as a ``(2, 6, 2, 3)`` integer array."""
    counts = np.empty((2, 6, 2, 3))
    for w, k, z, j in SUPPORT:
        base = P_W[w] * G[w, k] * (QZ[w, k] if z == 1 else 1.0 - QZ[w, k])
        if j == UNOBSERVED:
            counts[w, k, z, j] = base * (1.0 - PI[w, k]) * N
        else:
            outcome = QBAR[w, k, z] if j == OBSERVED_ONE else 1.0 - QBAR[w, k, z]
            counts[w, k, z, j] = base * PI[w, k] * outcome * N
    rounded = np.rint(counts)
    assert np.max(np.abs(counts - rounded)) < 1e-9, "the law is not realised by N rows"
    assert np.all(rounded > 0), "every support point holds a row"
    assert int(rounded.sum()) == N
    return rounded.astype(int)


#: Cell counts of the realised sample.
COUNTS = _cell_counts()
#: ``P(W, A, Z, J)``, from the counts, so it is bit for bit the empirical law of the frame.
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """The ``N`` rows whose empirical law is exactly this law, in :data:`SUPPORT` order.

    The columns are ``W``, ``F``, ``D``, ``Z``, ``Y`` and ``Delta``.  ``Y`` is ``NaN``
    where ``Delta`` is zero; ``Z`` is recorded on every row.
    """
    counts = [COUNTS[point] for point in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=int)[cells]
    flag_dose = np.asarray(MIXED_CELLS)[columns[:, 1]]
    kind = columns[:, 3]
    return pd.DataFrame(
        {
            "W": columns[:, 0].astype(float),
            "F": flag_dose[:, 0].astype(float),
            "D": np.asarray(MIXED_DOSES)[flag_dose[:, 1]],
            "Z": columns[:, 2].astype(float),
            "Y": np.where(kind == UNOBSERVED, np.nan, kind.astype(float)),
            "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
        }
    )


def first_row_of() -> np.ndarray:
    """Index of the first row of each support point, in :data:`SUPPORT` order."""
    counts = np.array([COUNTS[point] for point in SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)


def _parts(probs: Any, level: int | None) -> tuple[Any, Any]:
    """``P(w, k)`` over everyone and ``Qbar`` among the recorded rows, at ``level``."""
    joint = probs.sum(axis=(2, 3))
    if level is None:
        recorded = (probs[:, :, :, OBSERVED_ZERO] + probs[:, :, :, OBSERVED_ONE]).sum(axis=2)
        qbar = probs[:, :, :, OBSERVED_ONE].sum(axis=2) / recorded
    else:
        recorded = probs[:, :, level, OBSERVED_ZERO] + probs[:, :, level, OBSERVED_ONE]
        qbar = probs[:, :, level, OBSERVED_ONE] / recorded
    return joint, qbar


def _mean_under(joint: Any, qbar: Any, label: str, stratum: int | None) -> Any:
    """``sum_e p(e) sum_{w, k} P(w, k) Qbar(d_e(k, w), w)``, within ``W = stratum``."""
    total: Any = 0.0
    rows = range(joint.shape[0]) if stratum is None else (stratum,)
    for probability, mapping in POLICIES[label]:
        for w in rows:
            for k, target in enumerate(mapping[w]):
                total = total + probability * joint[w, k] * qbar[w, target]
    if stratum is None:
        return total
    return total / joint[stratum].sum()


def functional(
    probs: Any, estimand: str, level: int | None = None, stratum: int | None = None
) -> Any:
    """The parameter, longhand, sharing no code with ``src/``.

    Parameters
    ----------
    probs : ndarray
        The ``(2, 6, 2, 3)`` cell probabilities; complex for :func:`gateaux`.
    estimand : str
        ``"ey_policy[label]"`` or ``"ate_policy[label vs reference]"``.
    level : int or None
        ``None`` leaves ``Z`` alone (a ``delta=``-only fit); ``0`` or ``1`` is the
        controlled direct effect under the policy.
    stratum : int or None
        ``None`` is the marginal parameter; ``w`` is the parameter within ``W = w``.

    Returns
    -------
    complex or float
        The parameter at ``probs``.
    """
    joint, qbar = _parts(probs, level)
    name, _, rest = estimand.partition("[")
    inner = rest[:-1]
    if name == "ey_policy":
        return _mean_under(joint, qbar, inner, stratum)
    if name == "ate_policy":
        left, right = inner.split(" vs ")
        return _mean_under(joint, qbar, left, stratum) - _mean_under(joint, qbar, right, stratum)
    raise ValueError(f"no oracle branch for {estimand!r}")


def truth(estimand: str, level: int | None = None, stratum: int | None = None) -> float:
    """The parameter at the law."""
    return float(np.real(functional(PROBS, estimand, level, stratum)))


def eif(estimand: str, level: int | None = None, stratum: int | None = None) -> np.ndarray:
    """The complex-step efficient influence function at every support point."""
    base = PROBS.astype(complex)
    step = 1e-30
    out = []
    for point in SUPPORT:
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        out.append(float(np.imag(functional(perturbed, estimand, level, stratum)) / step))
    return np.array(out)


def complete_case_truth(estimand: str, level: int | None = None) -> float:
    """**Negative control.**  The parameter on the rows with a recorded outcome only."""
    complete = PROBS.copy()
    complete[:, :, :, UNOBSERVED] = 0.0
    return float(np.real(functional(complete / complete.sum(), estimand, level)))


def check_nondegenerate() -> dict[str, float]:
    """The margins by which the law sees each component, asserted above ``1e-2``.

    Returns
    -------
    dict
        ``pi`` and ``qz``: the smallest change of the mechanism with the flag at a fixed
        dose, and with the dose at a fixed flag, over the cells a policy moves.
        ``levels``: the smallest gap between the policy means at ``Z = 0``, ``Z = 1`` and
        with ``Z`` left alone.  ``complete case``: the smallest gap between the policy mean
        and its complete-case reading.
    """
    margins: dict[str, float] = {}
    for name, mechanism in (("pi", PI), ("qz", QZ)):
        by_flag = np.abs(mechanism[:, 3:] - mechanism[:, :3]).max(axis=1)
        by_dose = np.abs(np.diff(mechanism.reshape(2, 2, 3), axis=2)).max(axis=(1, 2))
        margins[name] = float(min(by_flag.min(), by_dose.min()))
    labels = ("dose up", "flag gated", "dose gated", "random")
    gaps = []
    for label in labels:
        values = [truth(f"ey_policy[{label}]", level) for level in (None, 0, 1)]
        gaps.extend(abs(a - b) for a, b in itertools.combinations(values, 2))
    margins["levels"] = float(min(gaps))
    margins["complete case"] = float(
        min(
            abs(truth(f"ey_policy[{label}]") - complete_case_truth(f"ey_policy[{label}]"))
            for label in labels
        )
    )
    for name, value in margins.items():
        assert value > 1e-2, (name, value)
    return margins
