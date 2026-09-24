r"""The exact tilt law of the regime-density witness, shared by the declaration tests.

``tests/unit/test_stochastic_regime_densities.py`` (RM25) measures on this law what a
regime curve omits when the odds tilt :math:`g^\star` of the sample mechanism is declared
known.  The law, the two tilts, and the closed-form curves live here, so a later test of a
declared function can reuse the same law and its measured values.  The parts that do not
depend on the law are in ``tests/unit/_declaration_support.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests import discrete_law as law
from tests.unit._declaration_support import cell_p

#: A variant law on which the omitted term is material and asymmetric in delta.  ``g``
#: differs across ``W``, and ``Qbar(1, w) - Qbar(0, w)`` is 0.8 or 0.9 at every ``w``.
#: Every cell is a multiple of ``1 / N``, so the 1000 rows realise the law exactly.
TILT_COUNTS = law.cell_counts(
    p_w=[0.4, 0.4, 0.2], g=[0.25, 0.5, 0.75], q=[[0.05, 0.95], [0.1, 0.9], [0.1, 0.9]]
)
TILT_PROBS = TILT_COUNTS / law.N
CELL_P = cell_p(TILT_PROBS)
W_OF = np.array([w for w, _, _ in law.SUPPORT])
A_OF = np.array([a for _, a, _ in law.SUPPORT], dtype=float)
DGP = law.DiscreteLaw(TILT_PROBS)

#: The two tilts, by the labels the oracle's ``ey_ipsi`` keys on.
DELTAS = {label: law.IPSI_DELTAS[label] for label in ("odds x2", "odds x0.5")}

#: The reported SE over the exact estimated-density SE at delta = 2 must stay below this.
#: The probe measured 0.6226 before this bound was chosen.  A curve that carried the
#: Kennedy term would read 1.  The bound claims an understatement of more than 30 percent and
#: pins no digit; the approx line records the measured value, as RM13 records 0.7417.
UNDERSTATEMENT_BOUND = 0.7


def levels(frame: Any) -> np.ndarray:
    return np.rint(np.asarray(frame["W"], dtype=float)).astype(int)


def odds_tilt(g: np.ndarray, delta: float) -> np.ndarray:
    """Kennedy's tilt, ``q(1 | w) = delta g / (delta g + 1 - g)``, as ``(3, 2)`` columns."""
    one = delta * g / (delta * g + 1.0 - g)
    return np.column_stack([1.0 - one, one])


class KnownTilt:
    """The tilt of the law's true mechanism, a fixed function of ``W``."""

    def __init__(self, delta: float) -> None:
        self.star = odds_tilt(DGP.g, delta)

    def __call__(self, frame: Any) -> np.ndarray:
        return self.star[levels(frame)]


class SampleTilt:
    """The user lie: the tilt of the sample's treated share in each stratum."""

    def __init__(self, sample: Any, delta: float) -> None:
        w, a = levels(sample), np.asarray(sample["A"], dtype=float)
        self.star = odds_tilt(np.array([a[w == k].mean() for k in range(3)]), delta)

    def __call__(self, frame: Any) -> np.ndarray:
        return self.star[levels(frame)]


def fixed_eif(star: np.ndarray, *, step: float = 1e-30) -> np.ndarray:
    """The Gateaux derivative of ``sum_w P(w) sum_a q*(a | w) Qbar(a, w)``, ``q*`` frozen.

    It is :func:`law.gateaux_eif` of that functional.  Only ``P(W)`` and ``Qbar`` move with
    the law; the density does not.
    """

    def psi(p: Any) -> Any:
        q = p[:, :, 1] / p.sum(axis=2)
        return (p.sum(axis=(1, 2)) * (star * q).sum(axis=1)).sum()

    return law.gateaux_eif(psi, TILT_PROBS, step=step)


def kennedy_term(delta: float) -> np.ndarray:
    """``T = delta (Qbar(1, W) - Qbar(0, W)) / D^2 (A - g(W))``, at every support point."""
    g, q = DGP.g, DGP.q
    slope = delta * (q[:, 1] - q[:, 0]) / (delta * g + 1.0 - g) ** 2
    return slope[W_OF] * (A_OF - g[W_OF])


def tilt_curve(result: Any) -> np.ndarray:
    return np.asarray(result.estimates["ey_regime[tilt]"].influence_curve)
