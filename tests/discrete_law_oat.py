r"""A two-armed finite-support law that separates the per-arm and the shared OAT designs.

The outcome-adaptive C-TMLE of :class:`~cleverly.estimators.CTMLE` fits its treatment
mechanism on outcome predictions. Under ``oat_design="per_arm"`` arm ``a`` reads only
``Qbar(a, W)``. Under ``oat_design="shared"`` one mechanism reads every arm's prediction.
The two designs coincide whenever every ``Qbar(a, .)`` is a bijection of ``W``. This law
breaks that for arm 0 alone:

* ``W`` takes four values with equal mass;
* ``Qbar(1, .)`` is injective, so the shared design ``[Qbar(0, W), Qbar(1, W)]`` separates
  every value of ``W``;
* ``Qbar(0, 0) = Qbar(0, 1)``, so the per-arm design of arm 0 ties ``W = 0`` and ``W = 1``;
* ``g_0(0) != g_0(1)``, so the tie changes the arm-0 mechanism.

A saturated mechanism on the per-arm design of arm 0 therefore returns the pooled
``P(A = 0 | W in {0, 1})`` on those rows, and the shared design returns the separated values.
Every cell probability is a multiple of ``1 / N``, so :func:`frame` realises the law
exactly, as :mod:`tests.discrete_law_multi` does for three arms.

:class:`SaturatedCategorical` is the oracle learner of both designs here, and the oracle
of the shared design in :mod:`tests.unit.test_multi_arm_collaborative`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

#: Rows in the realised sample.
N = 1600

#: ``P(W = w)`` for ``w in {0, 1, 2, 3}``.
P_W = np.array([0.25, 0.25, 0.25, 0.25])

#: ``g_0(w) = P(A = 1 | W = w)``.  The first two differ, which is what the tie acts on.
G1 = np.array([0.30, 0.60, 0.50, 0.70])

#: ``Qbar(a, w) = P(Y = 1 | A = a, W = w)``, indexed ``[w, a]``.  Column 1 is injective.
#: Column 0 ties ``w = 0`` and ``w = 1``.
Q = np.array(
    [
        [0.30, 0.20],
        [0.30, 0.40],
        [0.50, 0.60],
        [0.70, 0.80],
    ]
)


def _counts() -> np.ndarray:
    """``N * P(W = w, A = a, Y = y)`` as a ``(4, 2, 2)`` integer array."""
    counts = np.empty((4, 2, 2))
    g = np.column_stack([1.0 - G1, G1])
    for w in range(4):
        for a in range(2):
            for y in range(2):
                outcome = Q[w, a] if y == 1 else 1.0 - Q[w, a]
                counts[w, a, y] = P_W[w] * g[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError("the cell probabilities are not multiples of 1/N")
    return rounded.astype(int)


#: Cell counts of the realised sample.
COUNTS = _counts()

#: ``E[Y(a)]`` of the realised law.
TRUTH = {a: float(np.sum(P_W * Q[:, a])) for a in range(2)}


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law."""
    rows = [
        (w, a, y)
        for w in range(4)
        for a in range(2)
        for y in range(2)
        for _ in range(COUNTS[w, a, y])
    ]
    values = np.array(rows, dtype=float)
    return pd.DataFrame({"W": values[:, 0], "A": values[:, 1], "Y": values[:, 2]})


def projection(arm: int) -> np.ndarray:
    """``P(A = arm | Qbar(arm, W) = Qbar(arm, w))`` for each ``w``: the per-arm limit.

    Pools the values of ``w`` that share ``Qbar(arm, w)``, weighted by ``P(W = w)``.
    """
    g = np.column_stack([1.0 - G1, G1])[:, arm]
    keys = Q[:, arm]
    out = np.empty(4)
    for w in range(4):
        tied = np.isclose(keys, keys[w], rtol=0.0, atol=1e-12)
        out[w] = float(np.sum(P_W[tied] * g[tied]) / np.sum(P_W[tied]))
    return out


class OracleOutcome(BaseEstimator):
    """The law's own ``E[Y | A, W]``, plus a declared constant ``shift`` on the unit scale.

    The design is ``[A, W]``, the binary-treatment layout of
    :meth:`~cleverly.data.CausalData.treatment_design`.
    """

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOutcome:
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        arm = np.rint(design[:, 0]).astype(int)
        w = np.rint(design[:, 1]).astype(int)
        return np.clip(Q[w, arm] + self.shift, 0.01, 0.99)


class SaturatedCategorical(BaseEstimator):
    """``P(A = a | design)`` as the weighted class frequencies within each design row.

    A ``K``-class sibling of :class:`tests.discrete_law_longitudinal.CellMeans`, and here
    for the same reason: on a sample that realises its law exactly, an unpenalised
    saturated fit *is* the oracle, so "did the mechanism come out right" becomes an exact
    assertion rather than a tolerance.  A binary target is the two-class case, which is
    what the per-arm design fits.

    :attr:`designs` collects what every ``fit`` was handed, so one test can check what the
    treatment model was *given* and another what it learned from it.  It is a **class**
    attribute rather than a constructor argument because ``sklearn.clone`` rebuilds an
    estimator from ``get_params`` and deep-copies anything that is not itself an
    estimator: a list passed in would arrive at the fitted clone as a copy, and the caller
    would watch an empty one.
    """

    designs: ClassVar[list[np.ndarray]] = []

    def __init__(self, record: bool = False) -> None:
        self.record = record

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SaturatedCategorical:
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if self.record:
            type(self).designs.append(design.copy())
        self.classes_ = np.unique(target)
        keys, inverse = np.unique(np.round(design, 9), axis=0, return_inverse=True)
        totals = np.zeros((keys.shape[0], self.classes_.size))
        for column, level in enumerate(self.classes_):
            totals[:, column] = np.bincount(
                inverse, weights=weights * (target == level), minlength=keys.shape[0]
            )
        sizes = totals.sum(axis=1, keepdims=True)
        self.keys_ = keys
        self.frequencies_ = np.where(sizes > 0, totals / np.where(sizes > 0, sizes, 1.0), 0.0)
        self.default_ = totals.sum(axis=0) / totals.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.round(np.asarray(X, dtype=float), 9)
        out = np.tile(self.default_, (design.shape[0], 1))
        for position, key in enumerate(self.keys_):
            out[np.all(design == key, axis=1)] = self.frequencies_[position]
        return out
