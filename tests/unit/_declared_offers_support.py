"""A finite two-node law with three offers per node, a declared mechanism and censoring.

The law has a binary baseline ``L0``, a three-level offer ``A1``, a binary ``L1`` that the first
offer moves, a three-level offer ``A2`` that ``L1`` drives, and a binary outcome.  Units leave
after node one with a retention that depends on ``L0``.  Every conditional probability is a
whole number of quarters or twentieths, so each cell holds a whole number of rows and the
empirical distribution *is* the law, with no weights and no sampling error.

At that law a baseline that divides by the true offer mechanism and by a correctly specified
censoring model reproduces the g-formula, so an estimate equals the truth written out here from
the tables.  The offer labels sort as ``discount, none, shipping``, which is not the order of the
mechanism columns, so a declaration read by position instead of by label fails.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

#: The offer labels in the order of the declared probability columns.
LEVELS = ("none", "discount", "shipping")

#: ``P(L0 = 1)``.
P_L0 = 0.5
#: ``P(A1 = level | L0)``, in quarters, keyed by ``L0``, columns in :data:`LEVELS` order.
G1 = {0: (0.50, 0.25, 0.25), 1: (0.25, 0.25, 0.50)}
#: ``P(L1 = 1 | L0, A1)``, in quarters, keyed ``(l0, level)``.
P_L1 = {
    (0, "none"): 0.25,
    (0, "discount"): 0.75,
    (0, "shipping"): 0.50,
    (1, "none"): 0.50,
    (1, "discount"): 0.75,
    (1, "shipping"): 0.25,
}
#: ``P(A2 = level | L1)``, in quarters, columns in :data:`LEVELS` order.
G2 = {0: (0.25, 0.50, 0.25), 1: (0.50, 0.25, 0.25)}
#: ``P(C1 = 1 | L0)``, the retention after node one, in quarters.
RETENTION = {0: 0.75, 1: 0.50}
#: Rows per baseline cell, which makes every split whole.
ROWS_PER_CELL = 4**4 * 20


def outcome_probability(l0: int, a1: str, l1: int, a2: str) -> float:
    """``P(Y = 1 | L0, A1, L1, A2)`` in twentieths, from 0.10 to 0.90.

    The table has no additive structure on the logit scale, so a main-terms outcome model is
    wrong and only a saturated one reproduces it.
    """
    code = 7 * l0 + 3 * LEVELS.index(a1) + 5 * l1 + 11 * LEVELS.index(a2)
    code += 2 * LEVELS.index(a1) * l1
    return float((code % 17 + 2) / 20.0)


def _whole(value: float) -> int:
    count = round(value)
    assert abs(value - count) < 1e-9, value
    return int(count)


def frame() -> pd.DataFrame:
    """The law as a wide frame whose empirical distribution is the law.

    Columns ``L0``, ``A1``, ``C1``, ``L1``, ``A2``, ``C2``, ``Y``, then the declared offer
    probabilities ``p1_<level>`` and ``p2_<level>``.  A unit that leaves after node one has
    missing later nodes, and ``C2`` is one for every unit still observed.
    """
    rows: list[dict[str, Any]] = []
    for l0 in (0, 1):
        for a1_index, a1 in enumerate(LEVELS):
            n_a1 = _whole(ROWS_PER_CELL * G1[l0][a1_index])
            kept = _whole(n_a1 * RETENTION[l0])
            rows += [_row(l0, a1, None, None, None)] * (n_a1 - kept)
            for l1 in (0, 1):
                n_l1 = _whole(kept * (P_L1[(l0, a1)] if l1 else 1 - P_L1[(l0, a1)]))
                for a2_index, a2 in enumerate(LEVELS):
                    n_a2 = _whole(n_l1 * G2[l1][a2_index])
                    events = _whole(n_a2 * outcome_probability(l0, a1, l1, a2))
                    rows += [_row(l0, a1, l1, a2, float(i < events)) for i in range(n_a2)]
    return pd.DataFrame(rows)


def _row(l0: int, a1: str, l1: int | None, a2: str | None, y: float | None) -> dict[str, Any]:
    observed = l1 is not None
    row: dict[str, Any] = {
        "L0": float(l0),
        "A1": a1,
        "C1": 1.0 if observed else 0.0,
        "L1": float(l1) if observed else np.nan,
        "A2": a2 if observed else None,
        "C2": 1.0 if observed else np.nan,
        "Y": y if y is not None else np.nan,
    }
    for index, level in enumerate(LEVELS):
        row[f"p1_{level}"] = G1[l0][index]
        row[f"p2_{level}"] = G2[l1][index] if observed else np.nan
    return row


#: The column roles of the frame.
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["L0"],
    "time_varying": [[], ["L1"]],
    "censoring": ["C1", "C2"],
}


def declared(*, swap: bool = False) -> dict[str, dict[str, dict[str, str]]]:
    """The column form of the declared offer mechanism.

    ``swap=True`` exchanges the columns of ``discount`` and ``shipping`` at both nodes: the
    declaration of the same shape that is wrong.
    """
    exchange = {"discount": "shipping", "shipping": "discount"} if swap else {}
    return {
        f"A{node}": {level: f"p{node}_{exchange.get(level, level)}" for level in LEVELS}
        for node in (1, 2)
    }


def truth(first: str, second: Any) -> float:
    """``E[Y^d]`` for the offer ``first`` and the rule ``second`` of ``L1``, by enumeration."""
    total = 0.0
    for l0 in (0, 1):
        for l1 in (0, 1):
            p_l1 = P_L1[(l0, first)] if l1 else 1 - P_L1[(l0, first)]
            arm = second(l1) if callable(second) else second
            total += 0.5 * p_l1 * outcome_probability(l0, first, l1, arm)
    return total


class DeclaredOffers(BaseEstimator, ClassifierMixin):
    """A treatment "learner" that returns the law's offer mechanism from the history design.

    The node-one design is ``[L0]`` and the node-two design is ``[L0, L1, A1 ...]``.  A fit with
    this learner divides by the values a declaration holds.  The data layer codes the labels in
    sorted order, ``discount, none, shipping``, so the columns follow that order.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DeclaredOffers:
        self.classes_ = np.arange(len(LEVELS), dtype=float)
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if values.shape[1] == 1:
            by_level = np.array([G1[int(v)] for v in values[:, 0]], dtype=float)
        else:
            by_level = np.array([G2[int(v)] for v in values[:, 1]], dtype=float)
        return by_level[:, [LEVELS.index(level) for level in sorted(LEVELS)]]
