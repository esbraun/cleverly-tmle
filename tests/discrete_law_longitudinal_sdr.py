r"""The two-node law of the ``sequential_double_robustness`` family of ``longitudinal-sdr``.

The node layout, the support, the regimens and the closed-form functional are those of
:mod:`tests.discrete_law_longitudinal`:

.. code-block:: text

    W  ->  A1  ->  C1  ->  L2  ->  A2  ->  C2  ->  Y

Only the conditional tables differ.  Every entry is a multiple of ``1/4`` in ``[0.25, 0.75]``,
so :data:`N` rows realise the law exactly and no bound binds.

The tables come from a search over the quarter grid (``X4-plan.md`` 6.8).  The search kept the
law that maximised the smaller of two exact-law limits of the standardized bias of
``ate_regimen[always vs never]`` at ``n = 2,000``:

=================================  ==================================================  =======
control                            nuisances                                           limit
=================================  ==================================================  =======
SDR, both wrong at node 2          node 2: least squares on ``L2`` and a prior share   4.535
LTMLE in pattern ``P_D``           node 1: a prior share; node 2: least squares on      4.578
                                   ``L2``
=================================  ==================================================  =======

A limit is the exact-law error times :math:`\sqrt{2000}`, divided by the standard deviation of
the misspecified fit's own influence curve.  That is the denominator the harness reads.  The
floor is 3, and ``tests/unit/test_longitudinal_sdr_design.py`` recomputes both limits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tests import discrete_law_longitudinal as base

N = base.N
SUPPORT = base.SUPPORT
P_W = np.array([0.5, 0.5])

#: ``P(A1 = 1 | W = w)``.
G1 = np.array([0.75, 0.25])
#: ``P(C1 = 1 | W, A1)``.
C1 = np.array([[0.75, 0.75], [0.75, 0.75]])
#: ``P(L2 = 1 | W, A1, C1 = 1)``.
P_L2 = np.array([[0.25, 0.25], [0.25, 0.25]])
#: ``P(A2 = 1 | W, A1, L2, C1 = 1)``, indexed ``[w, a1, l2]``.
G2 = np.array([[[0.25, 0.75], [0.75, 0.75]], [[0.25, 0.50], [0.75, 0.25]]])
#: ``P(C2 = 1 | W, A1, L2, A2)``, indexed ``[w, a1, l2, a2]``.
C2 = np.array(
    [
        [[[0.25, 0.75], [0.75, 0.75]], [[0.75, 0.75], [0.25, 0.75]]],
        [[[0.75, 0.75], [0.50, 0.75]], [[0.50, 0.25], [0.75, 0.25]]],
    ]
)
#: ``P(Y = 1 | W, A1, L2, A2, C2 = 1)``, indexed ``[w, a1, l2, a2]``.
Q = np.array(
    [
        [[[0.75, 0.75], [0.75, 0.25]], [[0.25, 0.75], [0.75, 0.75]]],
        [[[0.25, 0.50], [0.75, 0.75]], [[0.25, 0.25], [0.25, 0.25]]],
    ]
)

#: The plans the family fits, as the estimator is handed them.
REGIMEN_SPEC: dict[str, Any] = {"never": 0, "always": 1}
REFERENCE = "never"
#: The estimand of every cell of the family.
ESTIMAND = "ate_regimen[always vs never]"
NAMES = ("ey_regimen[never]", "ey_regimen[always]", ESTIMAND)


def _mass_of(point: tuple[Any, ...]) -> float:
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


def counts() -> np.ndarray:
    """Rows per support point in an ``N``-row sample that realises the law exactly."""
    values = np.array([_mass_of(point) * N for point in SUPPORT])
    rounded = np.rint(values)
    if np.max(np.abs(values - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError("every conditional probability must be a multiple of 1/4")
    return rounded.astype(int)


COUNTS = counts()
PROBS = COUNTS / N

#: The g-formula value of each name, through the base law's closed-form functional.
TRUTH = {name: float(base.functional(PROBS, name)) for name in NAMES}


def eif(name: str, *, step: float = 1e-30) -> np.ndarray:
    """The efficient influence function of ``name`` at every support point, by complex step."""
    points = np.eye(len(SUPPORT))
    probs = PROBS.astype(complex)
    return np.array(
        [
            float(np.imag(base.functional((1.0 - 1j * step) * probs + 1j * step * mass, name)))
            / step
            for mass in points
        ]
    )


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical law is exactly this law."""
    cells = np.repeat(np.arange(len(SUPPORT)), COUNTS)
    return _rows(cells)


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` independent rows of the law."""
    cells = np.random.default_rng(seed).choice(len(SUPPORT), size=n, p=PROBS)
    return _rows(cells)


def _rows(cells: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: np.array(
                [np.nan if point[position] is None else float(point[position]) for point in SUPPORT]
            )[cells]
            for position, name in enumerate(("W", "A1", "C1", "L2", "A2", "C2", "Y"))
        }
    )


class KnownMechanism(BaseEstimator):
    """The law's own treatment or censoring probability, read off the design width.

    The engine's designs are ``[W]`` and ``[W, L2, A1]`` for treatment, and ``[W, A1]`` and
    ``[W, L2, A1, A2]`` for censoring.

    Parameters
    ----------
    kind : str
        ``"treatment"`` or ``"censoring"``.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w = matrix[:, 0].astype(int)
        if self.kind == "treatment" and matrix.shape[1] == 1:
            probability = G1[w]
        elif self.kind == "treatment" and matrix.shape[1] == 3:
            probability = G2[w, matrix[:, 2].astype(int), matrix[:, 1].astype(int)]
        elif self.kind == "censoring" and matrix.shape[1] == 2:
            probability = C1[w, matrix[:, 1].astype(int)]
        elif self.kind == "censoring" and matrix.shape[1] == 4:
            l2, a1, a2 = (matrix[:, column].astype(int) for column in (1, 2, 3))
            probability = C2[w, a1, l2, a2]
        else:
            raise ValueError(f"unexpected {self.kind} mechanism design {matrix.shape}")
        return np.column_stack([1.0 - probability, probability])
