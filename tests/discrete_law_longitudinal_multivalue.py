r"""An exactly realised two-time-point law with three treatment levels per node.

The implementation never enters this module.  The parameter below is the longitudinal
g-formula written directly as ratios of finite-support masses; complex-step derivatives
of that functional supply the efficient influence function independently of the fitted
clever covariate.  Every conditional probability is a multiple of one quarter, so 512
rows realise the law exactly.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

N = 2 * 4**4
ARM_LABELS = ("low", "standard", "high")
P_W = np.array([0.5, 0.5])
G1 = np.array([[0.25, 0.25, 0.50], [0.50, 0.25, 0.25]])
P_L2 = np.array([[0.25, 0.50, 0.75], [0.75, 0.25, 0.50]])
G2 = np.array(
    [
        [
            [[0.25, 0.25, 0.50], [0.50, 0.25, 0.25]],
            [[0.25, 0.50, 0.25], [0.25, 0.25, 0.50]],
            [[0.50, 0.25, 0.25], [0.25, 0.50, 0.25]],
        ],
        [
            [[0.25, 0.50, 0.25], [0.50, 0.25, 0.25]],
            [[0.50, 0.25, 0.25], [0.25, 0.25, 0.50]],
            [[0.25, 0.25, 0.50], [0.25, 0.50, 0.25]],
        ],
    ]
)
Q = np.array(
    [
        [
            [[0.25, 0.50, 0.75], [0.50, 0.75, 0.25]],
            [[0.50, 0.25, 0.75], [0.75, 0.50, 0.25]],
            [[0.75, 0.50, 0.25], [0.25, 0.75, 0.50]],
        ],
        [
            [[0.50, 0.25, 0.75], [0.75, 0.50, 0.25]],
            [[0.25, 0.75, 0.50], [0.50, 0.25, 0.75]],
            [[0.75, 0.25, 0.50], [0.25, 0.50, 0.75]],
        ],
    ]
)

REGIMEN_ARMS: dict[str, tuple[Any, Any]] = {
    "low": (0, 0),
    "high": (2, 2),
    "step_down": (2, 1),
    "respond": (1, np.array([0, 2])),
}
REGIMEN_SPEC: dict[str, Any] = {
    "low": "low",
    "high": "high",
    "step_down": ("high", "standard"),
    "respond": ("standard", lambda h: np.where(h["L2"] == 1, "high", "low")),
}
REFERENCE = "low"

SUPPORT = tuple(itertools.product(range(2), range(3), range(2), range(3), range(2)))


def _arm(node: Any, l2: int | None = None) -> int:
    return int(node) if np.ndim(node) == 0 else int(node[l2])


def _mass_of(point: tuple[int, ...]) -> float:
    w, a1, l2, a2, y = point
    mass = P_W[w] * G1[w, a1]
    mass *= P_L2[w, a1] if l2 else 1.0 - P_L2[w, a1]
    mass *= G2[w, a1, l2, a2]
    return float(mass * (Q[w, a1, l2, a2] if y else 1.0 - Q[w, a1, l2, a2]))


COUNTS = np.rint(np.array([_mass_of(point) * N for point in SUPPORT])).astype(int)
if (
    COUNTS.sum() != N
    or np.max(np.abs(COUNTS - np.array([_mass_of(point) * N for point in SUPPORT]))) > 1e-10
):  # pragma: no cover - guards the exact-law constants
    raise AssertionError("the declared categorical law is not exactly realisable at N=512")
PROBS = COUNTS / N


def _indices(**pattern: int) -> np.ndarray:
    names = ("w", "a1", "l2", "a2", "y")
    return np.array(
        [
            index
            for index, point in enumerate(SUPPORT)
            if all(point[names.index(name)] == value for name, value in pattern.items())
        ],
        dtype=int,
    )


def _mass(probs: Any, **pattern: int) -> Any:
    return np.sum(np.asarray(probs)[_indices(**pattern)])


def functional(probs: Any, estimand: str) -> Any:
    """The intervention-specific mean, directly from the finite-support g-formula."""
    if estimand.startswith("ate_regimen["):
        left, right = estimand[len("ate_regimen[") : -1].split(" vs ")
        return functional(probs, f"ey_regimen[{left}]") - functional(probs, f"ey_regimen[{right}]")
    if not estimand.startswith("ey_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")
    label = estimand[len("ey_regimen[") : -1]
    node1, node2 = REGIMEN_ARMS[label]
    total = _mass(probs)
    psi = 0.0
    for w in range(2):
        a1 = _arm(node1)
        reached = _mass(probs, w=w, a1=a1)
        for l2 in range(2):
            a2 = _arm(node2, l2)
            p_l2 = _mass(probs, w=w, a1=a1, l2=l2) / reached
            treated = _mass(probs, w=w, a1=a1, l2=l2, a2=a2)
            events = _mass(probs, w=w, a1=a1, l2=l2, a2=a2, y=1)
            psi += (_mass(probs, w=w) / total) * p_l2 * events / treated
    return psi


NAMES = tuple(f"ey_regimen[{label}]" for label in REGIMEN_ARMS) + tuple(
    f"ate_regimen[{label} vs {REFERENCE}]" for label in REGIMEN_ARMS if label != REFERENCE
)
TRUTH = {name: float(functional(PROBS, name)) for name in NAMES}


def gateaux_at(probs: np.ndarray, estimand: str, point: int, *, step: float = 1e-30) -> float:
    base = np.asarray(probs, dtype=complex)
    point_mass = np.zeros_like(base)
    point_mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * point_mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif_at(probs: np.ndarray, estimand: str) -> np.ndarray:
    return np.array([gateaux_at(probs, estimand, point) for point in range(len(SUPPORT))])


def frame() -> pd.DataFrame:
    cells = np.repeat(np.arange(len(SUPPORT)), COUNTS)
    points = np.asarray(SUPPORT, dtype=int)[cells]
    return pd.DataFrame(
        {
            "W": points[:, 0].astype(float),
            "A1": np.asarray(ARM_LABELS, dtype=object)[points[:, 1]],
            "L2": points[:, 2].astype(float),
            "A2": np.asarray(ARM_LABELS, dtype=object)[points[:, 3]],
            "Y": points[:, 4].astype(float),
        }
    )


def first_row_of() -> np.ndarray:
    return np.concatenate([[0], np.cumsum(COUNTS)[:-1]])


class CellProbabilities(BaseEstimator):
    """Weighted categorical probabilities within each distinct design row."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellProbabilities:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        target = np.asarray(y, dtype=float)
        weights = np.ones(target.size) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(target)
        self.keys_, inverse = np.unique(matrix, axis=0, return_inverse=True)
        self.probabilities_ = np.zeros((self.keys_.shape[0], self.classes_.size))
        for cell in range(self.keys_.shape[0]):
            rows = inverse == cell
            denominator = np.sum(weights[rows])
            for column, arm in enumerate(self.classes_):
                self.probabilities_[cell, column] = (
                    np.sum(weights[rows & (target == arm)]) / denominator
                )
        totals = np.array([np.sum(weights[target == arm]) for arm in self.classes_])
        self.default_ = totals / totals.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        out = np.broadcast_to(self.default_, (matrix.shape[0], self.classes_.size)).copy()
        for cell, key in enumerate(self.keys_):
            out[np.all(matrix == key, axis=1)] = self.probabilities_[cell]
        return out
