r"""The finite-support law of :mod:`tests.discrete_law`, with an intermediate variable.

:mod:`tests.discrete_law` and :mod:`tests.discrete_law_mar` carry the library's only
non-circular proofs that the reported influence curve *is* the efficient influence
function.  Neither reaches the ``intermediate=`` path, and that path is the one that most
needs reaching: :mod:`cleverly.estimators.direct_effect` derives its influence function by
argument -- a sequential-regression term that vanishes because nothing is measured between
:math:`A` and :math:`Z` -- and an argument is not a machine-checked fact.  This module
supplies the law that makes it one.

The observed-data support is ``(w, a, z, k)``.  ``Z`` is realised after ``A`` and before
``Y``, so it is recorded whether or not the outcome is:

.. math::

    (w, a, z, k), \qquad
    k \in \{\,\Delta = 1, Y = 0;\ \ \Delta = 1, Y = 1;\ \ \Delta = 0\,\},

thirty-six cells in a ``(3, 2, 2, 3)`` array.  Keeping :math:`\Delta` costs a dimension and
buys the thing this law exists to check: the controlled direct effect's clever covariate
divides by the **three-way** product :math:`g_a(W)\, q_z(a, W)\, \pi_a(W)`, and a law with
:math:`\pi \equiv 1` would never tell a two-way product from a three-way one.

The cell probabilities follow the time ordering the derivation assumes,

.. math::

    p(w, a, z, k) = p_W(w)\, g_a(w)\, q_z(a, w)\, \pi_a(w)\,
                    p_Y(\,\cdot \mid a, z, w, \Delta = 1),

with :math:`\pi` conditioning on ``(A, W)`` and *not* on ``Z`` -- which is the missingness
assumption :meth:`~cleverly.data.causal_data.CausalData.missingness_design` states, encoded
here rather than assumed.

Every property that makes the parent modules' assertions exact is preserved: each cell
probability is a multiple of ``1 / N``, so :func:`frame` *is* the law rather than a sample
from it; handed the oracle nuisances the initial fit is exactly right in the sample, so
``epsilon_hat`` is zero and the reported curve is the EIF at :math:`P_0` rather than an
estimate of it; and nothing is truncated -- ``g`` and ``q_z`` lie in ``[0.25, 0.75]``
against an ``auto`` bound near 0.023 and a ``nuisance_bound`` of 0.01, :math:`\pi` in
``[0.5, 0.75]``, and ``Qbar`` in ``[0.25, 0.75]`` against the ``alpha`` shrinkage window.

The constants are not chosen for symmetry.  They are the output of a search over laws with
denominators in ``{2, 4}``, scored by the *smallest* margin any of this module's claims
would have to clear, so that no negative control is marginal.  What that buys:

* the controlled direct effect **changes sign** between the two levels -- ``-0.1875`` at
  ``z = 0`` against ``+0.3125`` at ``z = 1``.  Confusing the levels does not perturb the
  answer, it inverts it.
* neither equals the total effect (``+0.09375``), which is what marginalising ``Z`` over
  its conditional law rather than intervening on it would give.
* ``q_z`` varies with both the arm and the covariate, so the two columns of the
  intermediate nuisance are not interchangeable and intervening on ``Z`` genuinely
  reweights the covariate distribution.
* :math:`\tau_z(w) = \bar Q(1, z, w) - \bar Q(0, z, w)` is correlated with ``g(w)`` at both
  levels, so ``att > ate > atc`` strictly -- see :func:`ordering`.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

#: Rows in the realised sample.  Every cell probability below is a multiple of ``1 / N``.
N = 1024

#: ``P(W = w)`` for ``w in {0, 1, 2}``.
P_W = np.array([0.50, 0.25, 0.25])

#: ``g(w) = P(A = 1 | W = w)``.  Decreasing in ``w``, well inside ``(0, 1)``.
G = np.array([0.75, 0.50, 0.25])

#: ``q_1(a, w) = P(Z = 1 | A = a, W = w)``, indexed ``[w, a]``.  Depends on both arguments:
#: without the arm dependence the two columns of the intermediate nuisance would be
#: interchangeable, and without the covariate dependence intervening on ``Z`` would not
#: reweight anything and the ``Z``-stratified control below would have nothing to detect.
QZ = np.array([[0.75, 0.75], [0.75, 0.25], [0.25, 0.25]])

#: ``pi(a, w) = P(Delta = 1 | A = a, W = w)``, indexed ``[w, a]``.  Note it does not
#: depend on ``z``: that is assumption 5 of the derivation, not an oversight.
PI = np.array([[0.50, 0.50], [0.75, 0.50], [0.75, 0.50]])

#: ``Qbar(a, z, w) = P(Y = 1 | A = a, Z = z, Delta = 1, W = w)``, indexed ``[w, a, z]``.
#: Carries a genuine ``A x Z`` interaction -- without one the controlled direct effect
#: would be the same parameter at both levels and the module could not tell them apart.
QBAR = np.array(
    [
        [[0.75, 0.25], [0.75, 0.75]],  # w = 0
        [[0.75, 0.50], [0.50, 0.75]],  # w = 1
        [[0.75, 0.25], [0.25, 0.25]],  # w = 2
    ]
)

#: Index of the fourth axis: the outcome was observed and zero, observed and one, or not
#: observed.  ``Y`` is undefined -- not zero -- in the last case.
OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = 0, 1, 2

#: The levels of ``Z`` a controlled direct effect can be targeted at.
LEVELS: tuple[int, int] = (0, 1)

#: The support, ordered ``(w, a, z, k)``.  Row blocks of the sample follow this order.
SUPPORT: tuple[tuple[int, int, int, int], ...] = tuple(
    itertools.product(range(3), range(2), range(2), range(3))
)


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = a, Z = z, K = k)`` as a ``(3, 2, 2, 3)`` integer array."""
    counts = np.empty((3, 2, 2, 3))
    for w, a, z, k in SUPPORT:
        arm = G[w] if a == 1 else 1.0 - G[w]
        level = QZ[w, a] if z == 1 else 1.0 - QZ[w, a]
        base = P_W[w] * arm * level
        if k == UNOBSERVED:
            counts[w, a, z, k] = base * (1.0 - PI[w, a]) * N
        else:
            outcome = QBAR[w, a, z] if k == OBSERVED_ONE else 1.0 - QBAR[w, a, z]
            counts[w, a, z, k] = base * PI[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- adjust P_W, G, QZ, PI or QBAR"
        )
    return rounded.astype(int)


#: Cell counts in the realised sample.  Integral by construction -- checked above.
COUNTS = _cell_counts()

#: ``P(W, A, Z, K)``, shape ``(3, 2, 2, 3)``.  Taken from the counts rather than from the
#: constants above, so it is bit-for-bit the empirical law of :func:`frame`.
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    ``Y`` is ``NaN`` wherever ``Delta`` is zero; ``Z`` is recorded on every row, including
    those, because it is realised before the outcome.  Rows are laid out in :data:`SUPPORT`
    order, one contiguous block per support point, so :func:`first_row_of` locates a
    representative row for each.
    """
    counts = [COUNTS[w, a, z, k] for w, a, z, k in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    kind = columns[:, 3]
    return pd.DataFrame(
        {
            "W": columns[:, 0],
            "A": columns[:, 1],
            "Z": columns[:, 2],
            "Y": np.where(kind == UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
        }
    )


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    counts = np.array([COUNTS[w, a, z, k] for w, a, z, k in SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


def functional(probs: Any, estimand: str, level: int) -> Any:
    r"""The target parameter at ``Z = level``, as a function of the cell probabilities.

    Written out longhand from the identification formula

    .. math::

        \Psi_{a,z}(P) = \sum_w P(W = w)\,
            P\bigl(Y = 1 \mid A = a,\, Z = z,\, \Delta = 1,\, W = w\bigr),

    sharing no code with the library, so comparing against it is a genuine check rather
    than a restatement.  Two things in that display are the whole content of the parameter
    and each has its own negative control below: the conditional mean is taken *within* the
    ``Z = z`` stratum, while the covariate distribution it is averaged against is the
    marginal over **everyone** -- every arm, every level, observed or not.  Averaging
    against the ``Z = z`` stratum's own covariate marginal instead is
    :func:`z_stratified_functional`; averaging the conditional mean over ``Z`` rather than
    fixing it is :func:`total_effect_functional`.

    Ratios are returned on the *log* scale, which is the scale their influence curve and
    confidence interval live on.

    Every operation is arithmetic (or :func:`numpy.log`), so this stays analytic in the
    cell probabilities -- which is what lets :func:`gateaux` differentiate it by a complex
    step.  Do not introduce ``clip``, ``abs`` or a comparison here.
    """
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2, 3))  # P(W = w) -- W is recorded for everyone
    p_wa = p.sum(axis=(2, 3))  # P(W = w, A = a) -- so is A
    observed = p[:, :, level, OBSERVED_ZERO] + p[:, :, level, OBSERVED_ONE]
    q = p[:, :, level, OBSERVED_ONE] / observed  # E[Y | A = a, Z = z, Delta = 1, W = w]

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


def total_effect_functional(probs: Any, estimand: str) -> Any:
    """:func:`functional` with ``Z`` *marginalised* rather than intervened on.

    Not the estimand -- the point of having it is that it is not.  This is the answer to
    the question "what is the effect of ``A``", which is a different question from "what is
    the effect of ``A`` with ``Z`` held at ``z``", and on this law it is ``+0.09375``
    against controlled direct effects of ``-0.1875`` and ``+0.3125``.  An implementation
    that built the clever covariate without the ``q_z`` factor, or that ignored the
    intermediate level when predicting the outcome, would be estimating this.

    Collapsing the ``Z`` axis of the cell probabilities is exactly the marginalisation:
    it replaces ``P(Y = 1 | A, Z = z, Delta = 1, W)`` with ``P(Y = 1 | A, Delta = 1, W)``.
    """
    p = np.asarray(probs)
    collapsed = p.sum(axis=2)
    p_w = collapsed.sum(axis=(1, 2))
    p_wa = collapsed.sum(axis=2)
    observed = collapsed[:, :, OBSERVED_ZERO] + collapsed[:, :, OBSERVED_ONE]
    q = collapsed[:, :, OBSERVED_ONE] / observed

    psi_one = (p_w * q[:, 1]).sum()
    psi_zero = (p_w * q[:, 0]).sum()

    if estimand == "ey1":
        return psi_one
    if estimand == "ey0":
        return psi_zero
    if estimand == "ate":
        return psi_one - psi_zero
    if estimand in ("att", "atc"):
        arm = 1 if estimand == "att" else 0
        share = p_wa[:, arm] / p_wa[:, arm].sum()
        return (share * (q[:, 1] - q[:, 0])).sum()
    raise ValueError(f"unknown estimand {estimand!r}")


def z_stratified_functional(probs: Any, estimand: str, level: int) -> Any:
    """:func:`functional` with the covariate marginal taken *within* the ``Z = z`` stratum.

    The classic wrong answer: conditioning on the subpopulation that happened to receive
    ``Z = z`` instead of intervening to set ``Z = z`` for everybody.  The conditional means
    are unchanged -- they were already taken within the stratum -- so the only thing that
    moves is the marginal of ``W`` they are averaged against, and it moves because ``q_z``
    depends on ``W``.
    """
    p = np.asarray(probs)
    stratum = p[:, :, level, :].sum(axis=(1, 2))  # P(W = w, Z = z)
    p_w = stratum / stratum.sum()  # P(W = w | Z = z)
    observed = p[:, :, level, OBSERVED_ZERO] + p[:, :, level, OBSERVED_ONE]
    q = p[:, :, level, OBSERVED_ONE] / observed

    psi_one = (p_w * q[:, 1]).sum()
    psi_zero = (p_w * q[:, 0]).sum()

    if estimand == "ey1":
        return psi_one
    if estimand == "ey0":
        return psi_zero
    if estimand == "ate":
        return psi_one - psi_zero
    raise ValueError(f"unknown estimand {estimand!r}")


#: Every estimand at every level, on the scale :func:`functional` returns.
#: Keyed ``TRUTH[level][name]``.
TRUTH: dict[int, dict[str, float]] = {
    level: {
        name: float(functional(PROBS, name, level))
        for name in ("ey1", "ey0", "ate", "rr", "or", "att", "atc")
    }
    for level in LEVELS
}

#: The total effect, for use as a negative control.
TOTAL_EFFECT: dict[str, float] = {
    name: float(total_effect_functional(PROBS, name))
    for name in ("ey1", "ey0", "ate", "att", "atc")
}


def ordering(level: int) -> tuple[float, float, float]:
    """``(att, ate, atc)`` at ``level``, which this law makes *strictly* decreasing.

    A constant-effect law cannot tell a correct ``att`` from a swapped conditioning
    population or an inverted propensity-odds factor: every contrast comes out the same
    number.  Here :math:`\\tau_z(w)` is correlated with ``g(w)``, so the treated are drawn
    from the covariate values with the larger effect and the three contrasts separate.
    """
    return tuple(TRUTH[level][name] for name in ("att", "ate", "atc"))  # type: ignore[return-value]


def gateaux(estimand: str, point: int, level: int, *, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at ``level``, at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)
                 \right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function -- here of the observed-data model, contaminated at an
    observed-data support point, and derived from :func:`functional` alone.  No clever
    covariate, no submodel, nothing the library supplies.

    Thirty of the thirty-six points contribute no residual term at all: the residual lives
    only where the row was in the targeted arm, took the targeted level, *and* had its
    outcome recorded.  Whether the derivative comes out as
    :math:`\bar Q(a, z, W) - \Psi` exactly at the other thirty is the sharpest single
    statement that all three indicators are in the right place.

    Differentiation is by complex step rather than finite difference, for the reasons given
    in :func:`tests.discrete_law.gateaux`: full double precision, hence an exact comparison
    rather than a close one.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand, level)) / step)


def eif(estimand: str, level: int) -> np.ndarray:
    """The EIF of ``estimand`` at ``level``, evaluated at every support point."""
    return np.array([gateaux(estimand, point, level) for point in range(len(SUPPORT))])


#: The nuisances as the *realised sample* has them.  Equal to :data:`G`, :data:`QZ`,
#: :data:`PI` and :data:`QBAR` mathematically; derived from the counts so that the oracle
#: nuisances are exact in the sample down to the last bit.
_OBSERVED = PROBS[:, :, :, OBSERVED_ZERO] + PROBS[:, :, :, OBSERVED_ONE]
G_EXACT = PROBS.sum(axis=(2, 3))[:, 1] / PROBS.sum(axis=(1, 2, 3))
QZ_EXACT = PROBS[:, :, 1, :].sum(axis=2) / PROBS.sum(axis=(2, 3))
PI_EXACT = _OBSERVED.sum(axis=2) / PROBS.sum(axis=3).sum(axis=2)
QBAR_EXACT = PROBS[:, :, :, OBSERVED_ONE] / _OBSERVED


class DiscreteLaw:
    """The law, duck-typed as a ``DGP`` for the oracle learners in :mod:`tests.conftest`.

    Supplies the four nuisances the controlled-direct-effect path needs: ``propensity``,
    ``intermediate_mean``, ``missingness`` and ``outcome_mean``.  All four read the cell
    counts, so the oracles and the sample cannot drift apart.

    Note ``outcome_mean`` takes the intermediate level and *uses* it -- the counterparts in
    :mod:`tests.discrete_law` and :mod:`tests.discrete_law_mar` accept the argument and
    ignore it, because on those laws there is nothing to condition on.  That is the whole
    difference between an ordinary counterfactual mean and a controlled direct effect, so
    the signature is where it has to show up.
    """

    def __init__(self, probs: Any = None) -> None:
        p = PROBS if probs is None else np.asarray(probs, dtype=float)
        observed = p[:, :, :, OBSERVED_ZERO] + p[:, :, :, OBSERVED_ONE]
        self.probs = p
        self.g = p.sum(axis=(2, 3))[:, 1] / p.sum(axis=(1, 2, 3))
        self.qz = p[:, :, 1, :].sum(axis=2) / p.sum(axis=(2, 3))
        self.pi = observed.sum(axis=2) / p.sum(axis=3).sum(axis=2)
        self.q = p[:, :, :, OBSERVED_ONE] / observed

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        return np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        return self.g[self._index(covariates)]

    def intermediate_mean(self, covariates: Any, arm: float) -> np.ndarray:
        """``P(Z = 1 | A = arm, W)``."""
        return self.qz[self._index(covariates), int(arm)]

    def missingness(self, covariates: Any, arm: float) -> np.ndarray:
        """``P(Delta = 1 | A = arm, W)`` -- deliberately not a function of ``Z``."""
        return self.pi[self._index(covariates), int(arm)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        """``E[Y | A = arm, Z = intermediate, Delta = 1, W]``.

        ``intermediate`` is required: without a level there is no controlled direct effect
        to speak of, and silently defaulting to one of them is the failure mode
        :func:`cleverly.estimators.direct_effect.check_level` exists to prevent.
        """
        if intermediate is None:
            raise ValueError(
                "outcome_mean needs an intermediate level on this law; a controlled direct "
                "effect is a different parameter at each level of Z"
            )
        return self.q[self._index(covariates), int(arm), int(intermediate)]
