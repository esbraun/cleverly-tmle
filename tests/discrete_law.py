r"""A finite-support law that a finite sample realises *exactly*.

The efficient influence function is a property of a distribution, not of a sample, so
checking that the library computes the right one needs a distribution the test can hold
in its hand.  This module supplies one: ``W`` takes three values, ``A`` and ``Y`` are
binary, and every cell probability is an exact multiple of ``1 / N``.  Laying out ``N``
rows in the cell proportions therefore makes the empirical distribution *equal* to the
data-generating one, which buys three things:

* every population quantity -- the truth, the propensity, the conditional means -- is a
  closed-form function of the twelve cell probabilities, computable without touching any
  library code;
* handed the oracle nuisances, the initial fit is exactly correct *in the sample*, so the
  targeting step's score is zero at :math:`\epsilon = 0`, ``epsilon_hat`` is zero, and the
  influence curve the estimator reports is the EIF at :math:`P_0` rather than an estimate
  of it;
* there is no sampling error anywhere, so the assertions can be exact rather than
  statistical.

Positivity holds comfortably by construction (propensities in ``[0.25, 0.6]``, conditional
means in ``[0.2, 0.8]``), so no truncation or shrinkage is active and the estimator runs on
the unmodified nuisances.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

#: Rows in the realised sample.  Every cell probability below is a multiple of ``1 / N``.
N = 1000

#: ``P(W = w)`` for ``w in {0, 1, 2}``.
P_W = np.array([0.50, 0.30, 0.20])

#: ``g(w) = P(A = 1 | W = w)``.  Well inside ``(0, 1)``, so ``g_bounds`` never binds.
G = np.array([0.40, 0.60, 0.25])

#: ``Qbar(a, w) = P(Y = 1 | A = a, W = w)``, indexed ``[w, a]``.
Q = np.array([[0.40, 0.70], [0.20, 0.50], [0.60, 0.80]])

#: The support, ordered ``(w, a, y)``.  Row blocks of the sample follow this order.
SUPPORT: tuple[tuple[int, int, int], ...] = tuple(itertools.product(range(3), range(2), range(2)))


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = a, Y = y)`` as a ``(3, 2, 2)`` integer array."""
    counts = np.empty((3, 2, 2))
    for w, a, y in SUPPORT:
        arm = G[w] if a == 1 else 1.0 - G[w]
        outcome = Q[w, a] if y == 1 else 1.0 - Q[w, a]
        counts[w, a, y] = P_W[w] * arm * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- adjust P_W, G or Q"
        )
    return rounded.astype(int)


#: Cell counts in the realised sample.  Integral by construction -- checked above.
COUNTS = _cell_counts()

#: ``P(W, A, Y)``, shape ``(3, 2, 2)``.  Taken from the counts rather than from the
#: constants above, so it is bit-for-bit the empirical law of :func:`frame`.
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    Rows are laid out in :data:`SUPPORT` order, one contiguous block per support point,
    so :func:`first_row_of` locates a representative row for each.
    """
    counts = [COUNTS[w, a, y] for w, a, y in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    return pd.DataFrame({"W": columns[:, 0], "A": columns[:, 1], "Y": columns[:, 2]})


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    counts = np.array([COUNTS[w, a, y] for w, a, y in SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


def functional(probs: Any, estimand: str) -> Any:
    r"""The target parameter as a closed-form function of the cell probabilities.

    Written out longhand from the identification formula and sharing no code with the
    library, so comparing against it is a genuine check rather than a restatement.
    Ratios are returned on the *log* scale, which is the scale their influence curve and
    confidence interval live on.

    Every operation is arithmetic (or :func:`numpy.log`), so this stays analytic in the
    cell probabilities -- which is what lets :func:`gateaux` differentiate it by a complex
    step.  Do not introduce ``clip``, ``abs`` or a comparison here.
    """
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))  # P(W = w)
    p_wa = p.sum(axis=2)  # P(W = w, A = a)
    q = p[:, :, 1] / p_wa  # E[Y | A = a, W = w]

    psi_one = (p_w * q[:, 1]).sum()
    psi_zero = (p_w * q[:, 0]).sum()

    if estimand == "ey1":
        return psi_one
    if estimand == "ey0":
        return psi_zero
    if estimand == "ate":
        return psi_one - psi_zero
    if estimand == "rr":
        return np.log(psi_one) - np.log(psi_zero)
    if estimand == "or":
        return np.log(psi_one / (1.0 - psi_one)) - np.log(psi_zero / (1.0 - psi_zero))
    if estimand in ("att", "atc"):
        arm = 1 if estimand == "att" else 0
        share = p_wa[:, arm] / p_wa[:, arm].sum()  # P(W = w | A = arm)
        return (share * (q[:, 1] - q[:, 0])).sum()
    raise ValueError(f"unknown estimand {estimand!r}")


#: Population values of every estimand, on the scale :func:`functional` returns.
TRUTH = {
    name: float(functional(PROBS, name)) for name in ("ey1", "ey0", "ate", "rr", "or", "att", "atc")
}


def gateaux(estimand: str, point: int, *, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)
                 \right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function, and here is derived from :func:`functional` alone --
    no clever covariate, no submodel, nothing the library supplies.

    Differentiation is by complex step rather than finite difference.  ``Psi`` along the
    contamination path is a rational function of ``t``, hence analytic, so
    :math:`\operatorname{Im}\Psi(ih)/h = \Psi'(0) + O(h^2)`; because the imaginary part is
    carried separately there is no subtractive cancellation, and ``h`` can be taken small
    enough that the truncation term is far below machine precision.  The result is the
    derivative to full double precision, which is what makes the comparison exact instead
    of merely close.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif(estimand: str) -> np.ndarray:
    """The EIF of ``estimand`` evaluated at every support point, in support order."""
    return np.array([gateaux(estimand, point) for point in range(len(SUPPORT))])


#: ``P(A = 1 | W = w)`` and ``E[Y | A = a, W = w]`` as the *realised sample* has them.
#: Equal to :data:`G` and :data:`Q` mathematically; derived from the counts so that the
#: oracle nuisances are exact in the sample down to the last bit.
G_EXACT = PROBS.sum(axis=2)[:, 1] / PROBS.sum(axis=(1, 2))
Q_EXACT = PROBS[:, :, 1] / PROBS.sum(axis=2)


class DiscreteLaw:
    """The law, duck-typed as a ``DGP`` for the oracle learners in :mod:`tests.conftest`.

    Only ``propensity`` and ``outcome_mean`` are needed: those are the two methods
    :class:`~tests.conftest.OracleTreatment` and :class:`~tests.conftest.OracleOutcome`
    call.  Both read the cell counts, so the oracle and the sample cannot drift apart.
    """

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        return np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        return G_EXACT[self._index(covariates)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        return Q_EXACT[self._index(covariates), int(arm)]
