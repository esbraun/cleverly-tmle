"""Finite laws whose empirical distribution equals the law, for the known-mechanism tests.

Each law has a few covariate cells, a rational treatment mechanism and a rational outcome
regression.  Every cell holds a whole number of rows in the exact law proportions: the arms
split each cell by the mechanism, and the outcomes split each cell-arm by the outcome
regression.  So the empirical distribution ``P_n`` *is* ``P_0``, with no weights and no
sampling error.

At ``P_n = P_0`` a TMLE that divides by the true mechanism solves ``P_n D*(Qbar*, g0) = 0``,
and its remainder ``P_0[(g - g0) / g (Qbar - Qbar0)]`` is zero, so its estimate equals the
truth whatever the outcome regression, up to the tolerance of the fluctuation solver
(Moore and van der Laan 2009, Section 2).  The truth is computed here from the tables by
exact enumeration.

The point-treatment law has ``W1`` on three points and ``W2`` in ``{0, 1}``.  The mechanism
is in quarters and the outcome regression in twentieths, so 800 rows per cell make every
split whole: ``n = 4800``.  The longitudinal law is a two-node SMART with a binary baseline
``L0``, a binary ``L1`` and optional censoring after node one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

#: Rows per covariate cell of the point-treatment law.
ROWS_PER_CELL = 800

#: The covariate cells ``(W1, W2)``, each with probability one sixth.
CELLS: tuple[tuple[float, float], ...] = tuple(
    (w1, w2) for w2 in (0.0, 1.0) for w1 in (-1.0, 0.0, 1.0)
)

#: ``P(A = 1 | cell)`` of the two-arm law, in quarters.
BINARY_MECHANISM: tuple[float, ...] = (0.25, 0.5, 0.75, 0.75, 0.25, 0.5)

#: ``P(A = a | cell)`` of the three-arm law, in quarters, arms in code order.
THREE_ARM_MECHANISM: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.25, 0.5),
    (0.5, 0.25, 0.25),
    (0.25, 0.5, 0.25),
    (0.5, 0.25, 0.25),
    (0.25, 0.25, 0.5),
    (0.25, 0.5, 0.25),
)

#: ``Qbar0(a, cell)`` in twentieths, one row per cell and one column per arm.  The
#: columns are not a logistic function of main terms, so a main-terms outcome learner
#: is wrong here.
OUTCOME: tuple[tuple[float, float, float], ...] = (
    (0.15, 0.45, 0.30),
    (0.30, 0.40, 0.65),
    (0.55, 0.60, 0.20),
    (0.20, 0.85, 0.50),
    (0.65, 0.70, 0.35),
    (0.40, 0.90, 0.75),
)

#: The population weight ``w(W)`` of the weighted law, one per cell.
CELL_WEIGHTS: tuple[float, ...] = (1.0, 2.0, 1.0, 3.0, 1.0, 2.0)


def point_frame(arms: int = 2) -> pd.DataFrame:
    """The point-treatment law as a frame whose empirical law is the law.

    Columns: ``W1``, ``W2``, ``A``, ``Y``, ``wt`` (the cell weight) and one probability
    column per arm, ``p0``, ``p1`` (and ``p2``).

    Parameters
    ----------
    arms : int
        2 or 3.

    Returns
    -------
    pandas.DataFrame
        ``6 * ROWS_PER_CELL`` rows in cell, arm and outcome order.
    """
    rows: list[dict[str, float]] = []
    for index, (w1, w2) in enumerate(CELLS):
        probabilities = (
            (1.0 - BINARY_MECHANISM[index], BINARY_MECHANISM[index])
            if arms == 2
            else THREE_ARM_MECHANISM[index]
        )
        for arm, share in enumerate(probabilities):
            count = _whole(ROWS_PER_CELL * share)
            events = _whole(count * OUTCOME[index][arm])
            for row in range(count):
                record = {
                    "W1": w1,
                    "W2": w2,
                    "A": float(arm),
                    "Y": 1.0 if row < events else 0.0,
                    "wt": CELL_WEIGHTS[index],
                }
                for code, value in enumerate(probabilities):
                    record[f"p{code}"] = value
                rows.append(record)
    return pd.DataFrame(rows)


def _whole(value: float) -> int:
    """``value`` as a whole number of rows, which the law's denominators guarantee."""
    count = round(value)
    assert abs(value - count) < 1e-9, value
    return int(count)


def mechanism(frame: pd.DataFrame, arms: int = 2) -> np.ndarray:
    """The ``(n, K)`` true mechanism of ``frame``'s rows, in arm-code order."""
    return frame[[f"p{code}" for code in range(arms)]].to_numpy(dtype=float)


def point_truth(arms: int = 2, *, weighted: bool = False) -> dict[str, float]:
    """The arm means and contrasts of the point-treatment law, by exact enumeration.

    ``weighted=True`` gives the parameters of the law tilted by :data:`CELL_WEIGHTS`, which
    depend on ``W`` only, so the tilted law's mechanism is the design mechanism.

    Parameters
    ----------
    arms : int
        2 or 3.
    weighted : bool
        Whether to tilt the cell probabilities by the cell weights.

    Returns
    -------
    dict of str to float
        ``ey[a]`` for each arm, and at two arms ``ate``, ``rr``, ``or``, ``att`` and ``atc``.
    """
    mass = np.array(CELL_WEIGHTS if weighted else (1.0,) * len(CELLS), dtype=float)
    mass = mass / mass.sum()
    outcome = np.array(OUTCOME, dtype=float)[:, :arms]
    means = mass @ outcome
    truth = {f"ey[{float(a)}]": float(means[a]) for a in range(arms)}
    if arms == 2:
        g1 = np.array(BINARY_MECHANISM, dtype=float)
        blip = outcome[:, 1] - outcome[:, 0]
        truth["ey1"] = float(means[1])
        truth["ey0"] = float(means[0])
        truth["ate"] = float(means[1] - means[0])
        truth["rr"] = float(means[1] / means[0])
        truth["or"] = float((means[1] / (1 - means[1])) / (means[0] / (1 - means[0])))
        truth["att"] = float(np.sum(mass * g1 * blip) / np.sum(mass * g1))
        truth["atc"] = float(np.sum(mass * (1 - g1) * blip) / np.sum(mass * (1 - g1)))
    return truth


class FixedMechanism(BaseEstimator, ClassifierMixin):
    """A treatment "learner" that returns the law's mechanism, read from ``(W1, W2)``.

    The known-equals-fitted control: a fit with this learner divides by the same values as a
    fit on data that declares them, so the two must agree to the last few bits.
    """

    def __init__(self, arms: int = 2) -> None:
        self.arms = arms

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> FixedMechanism:
        self.classes_ = np.arange(self.arms, dtype=float)
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        cells = [CELLS.index((float(w1), float(w2))) for w1, w2 in values[:, -2:]]
        if self.arms == 2:
            one = np.array([BINARY_MECHANISM[cell] for cell in cells])
            return np.column_stack([1.0 - one, one])
        return np.array([THREE_ARM_MECHANISM[cell] for cell in cells], dtype=float)


# --------------------------------------------------------------------- longitudinal

#: ``P(A1 = 1)``, the first-stage randomization.
SMART_A1 = 0.5
#: ``P(L1 = 1 | L0, A1)``, in quarters, keyed ``(l0, a1)``.
SMART_L1 = {(0, 0): 0.25, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.75}
#: ``P(A2 = 1 | L1)``, the second-stage randomization, in quarters.
SMART_A2 = {0: 0.25, 1: 0.75}
#: ``P(C1 = 1 | L0)``, the retention after node one, in quarters.
SMART_C1 = {0: 0.75, 1: 0.5}
#: ``P(Y = 1 | L0, A1, L1, A2)`` in twentieths.
SMART_Y = {
    (0, 0, 0, 0): 0.15,
    (0, 0, 0, 1): 0.35,
    (0, 0, 1, 0): 0.40,
    (0, 0, 1, 1): 0.80,
    (0, 1, 0, 0): 0.25,
    (0, 1, 0, 1): 0.60,
    (0, 1, 1, 0): 0.55,
    (0, 1, 1, 1): 0.30,
    (1, 0, 0, 0): 0.45,
    (1, 0, 0, 1): 0.20,
    (1, 0, 1, 0): 0.65,
    (1, 0, 1, 1): 0.90,
    (1, 1, 0, 0): 0.35,
    (1, 1, 0, 1): 0.75,
    (1, 1, 1, 0): 0.50,
    (1, 1, 1, 1): 0.85,
}
#: Units per baseline cell, which makes every split of the SMART law whole.
SMART_ROWS_PER_CELL = 2 * 4 * 4 * 4 * 20


def smart_frame(*, censoring: bool) -> pd.DataFrame:
    """The two-node SMART law as a wide frame whose empirical law is the law.

    Columns ``L0``, ``A1``, ``C1``, ``L1``, ``A2``, ``C2``, ``Y`` and the declared
    probabilities ``g1_0``, ``g1_1``, ``g2_0``, ``g2_1``, ``r1``, ``r2``.  Without
    ``censoring`` the ``C`` columns are dropped and every unit is observed.  With it, a
    unit censored after node one has missing later nodes, and ``C2`` is one for every
    unit still observed.

    Parameters
    ----------
    censoring : bool
        Whether units leave after node one.

    Returns
    -------
    pandas.DataFrame
        ``2 * SMART_ROWS_PER_CELL`` units.
    """
    rows: list[dict[str, float]] = []
    for l0 in (0, 1):
        total = SMART_ROWS_PER_CELL
        for a1 in (0, 1):
            n_a1 = _whole(total * (SMART_A1 if a1 else 1 - SMART_A1))
            kept = _whole(n_a1 * SMART_C1[l0]) if censoring else n_a1
            for _ in range(n_a1 - kept):
                rows.append(_smart_row(l0, a1, None, None, None, censoring))
            for l1 in (0, 1):
                n_l1 = _whole(kept * (SMART_L1[(l0, a1)] if l1 else 1 - SMART_L1[(l0, a1)]))
                for a2 in (0, 1):
                    n_a2 = _whole(n_l1 * (SMART_A2[l1] if a2 else 1 - SMART_A2[l1]))
                    events = _whole(n_a2 * SMART_Y[(l0, a1, l1, a2)])
                    for row in range(n_a2):
                        rows.append(_smart_row(l0, a1, l1, a2, float(row < events), censoring))
    frame = pd.DataFrame(rows)
    if not censoring:
        frame = frame.drop(columns=["C1", "C2", "r1", "r2"])
    return frame


def _smart_row(
    l0: int, a1: int, l1: int | None, a2: int | None, y: float | None, censoring: bool
) -> dict[str, float]:
    observed = l1 is not None
    g2 = np.nan if l1 is None else SMART_A2[l1]
    return {
        "L0": float(l0),
        "A1": float(a1),
        "C1": 1.0 if observed else 0.0,
        "L1": float(l1) if observed else np.nan,
        "A2": float(a2) if a2 is not None else np.nan,
        "C2": 1.0 if observed else np.nan,
        "Y": y if y is not None else np.nan,
        "g1_0": 1 - SMART_A1,
        "g1_1": SMART_A1,
        "g2_0": 1 - g2,
        "g2_1": g2,
        "r1": SMART_C1[l0] if censoring else 1.0,
        "r2": 1.0 if observed else np.nan,
    }


def smart_truth(a1: int, a2: int) -> float:
    """``E[Y^{a1, a2}]`` of the SMART law, by exact enumeration."""
    total = 0.0
    for l0 in (0, 1):
        for l1 in (0, 1):
            p_l1 = SMART_L1[(l0, a1)] if l1 else 1 - SMART_L1[(l0, a1)]
            total += 0.5 * p_l1 * SMART_Y[(l0, a1, l1, a2)]
    return total


#: The column roles of the SMART frame.
SMART_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["L0"],
    "time_varying": [[], ["L1"]],
}

#: The column form of the SMART law's declared treatment mechanism.
SMART_TREATMENT = {"A1": {0.0: "g1_0", 1.0: "g1_1"}, "A2": {0.0: "g2_0", 1.0: "g2_1"}}
#: The column form of the SMART law's declared retention.
SMART_CENSORING = {"C1": "r1", "C2": "r2"}
