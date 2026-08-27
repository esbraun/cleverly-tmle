r"""The finite-support law of :mod:`tests.discrete_law`, with outcomes missing at random.

:mod:`tests.discrete_law` carries the library's only non-circular proofs that the
influence curve is *the* efficient influence function and that its remainder carries both
nuisance errors.  Neither proof reaches the ``delta=`` path, because that law has no
:math:`\Delta` dimension: its support is ``(w, a, y)`` and its identification formula has
no missingness anywhere.  This module supplies the missing-at-random counterpart, so the
same two arguments can be made about the same estimator run with ``delta="Delta"``.

The observed-data law is what has to be held in the hand here, not the full-data one.  A
row reveals ``(W, A, Delta)`` always and ``Y`` only when ``Delta = 1``, so the support is

.. math::

    (w, a, k), \qquad k \in \{\,\Delta = 1, Y = 0;\ \ \Delta = 1, Y = 1;\ \ \Delta = 0\,\},

eighteen cells whose probabilities are the free parameters of the observed-data
distribution.  Keeping them in a rectangular ``(3, 2, 3)`` array is what lets the Gateaux
derivative be taken by the same complex step :mod:`tests.discrete_law` uses -- the
parameter stays an analytic function of the cell probabilities, contaminated cell by cell.

Every property that makes the parent module's assertions exact is preserved:

* each cell probability is a multiple of ``1 / N``, so :func:`frame` *is* the law rather
  than a sample from it;
* handed the oracle nuisances the initial fit is exactly right in the sample -- within a
  ``(w, a)`` cell the clever covariate is constant and the observed outcomes average to
  exactly ``Qbar(a, w)``, so the score is zero at :math:`\epsilon = 0` and
  ``epsilon_hat`` is zero;
* nothing is truncated: ``g`` lies in ``[0.25, 0.60]`` against an ``auto`` bound near
  0.023, :math:`\pi` in ``[0.25, 0.90]`` against ``nuisance_bound = 0.01``, and ``Qbar``
  in ``[0.2, 0.8]`` against the ``alpha`` shrinkage window.

Two constants are chosen to give the negative controls teeth rather than for symmetry
with the parent module.  :data:`PI` varies with **both** ``W`` and ``A`` -- a mechanism
depending only on ``A`` would leave the covariate marginal alone and hide an estimator
that conditions on being observed.  And ``Q[:, 1]`` decreases in ``w`` while missingness
pushes mass towards large ``w``, so the two do not cancel: ``P(W | Delta = 1)`` is
``[0.282, 0.352, 0.366]`` against ``P(W) = [0.50, 0.30, 0.20]``, and averaging the
targeted predictions over the observed rows instead of all of them gives 0.475 where the
truth is 0.590.  Roughly half the sample has no outcome at all.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

#: Rows in the realised sample.  Every cell probability below is a multiple of ``1 / N``.
N = 1000

#: ``P(W = w)`` for ``w in {0, 1, 2}``.  As in :mod:`tests.discrete_law`.
P_W = np.array([0.50, 0.30, 0.20])

#: ``g(w) = P(A = 1 | W = w)``.  As in :mod:`tests.discrete_law`.
G = np.array([0.40, 0.60, 0.25])

#: ``pi(a, w) = P(Delta = 1 | A = a, W = w)``, indexed ``[w, a]``.  Depends on both
#: arguments, which is what makes a complete-case analysis target a different covariate
#: distribution *and* what makes the two columns of the missingness nuisance differ.
PI = np.array([[0.25, 0.30], [0.50, 0.60], [0.90, 0.80]])

#: ``Qbar(a, w) = P(Y = 1 | A = a, Delta = 1, W = w)``, indexed ``[w, a]``.  Under MAR
#: this equals ``P(Y = 1 | A = a, W = w)``; the estimator only ever sees the former.
Q = np.array([[0.40, 0.80], [0.20, 0.50], [0.60, 0.20]])

#: Index of the third axis: the outcome was observed and zero, observed and one, or not
#: observed.  ``Y`` is undefined -- not zero -- in the last case.
OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = 0, 1, 2

#: The support, ordered ``(w, a, k)``.  Row blocks of the sample follow this order.
SUPPORT: tuple[tuple[int, int, int], ...] = tuple(itertools.product(range(3), range(2), range(3)))


def _cell_counts() -> np.ndarray:
    """``N * P(W = w, A = a, K = k)`` as a ``(3, 2, 3)`` integer array."""
    counts = np.empty((3, 2, 3))
    for w, a, k in SUPPORT:
        arm = G[w] if a == 1 else 1.0 - G[w]
        base = P_W[w] * arm
        if k == UNOBSERVED:
            counts[w, a, k] = base * (1.0 - PI[w, a]) * N
        else:
            outcome = Q[w, a] if k == OBSERVED_ONE else 1.0 - Q[w, a]
            counts[w, a, k] = base * PI[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError(
            "the cell probabilities are not multiples of 1/N, so no sample of N rows can "
            "realise the law exactly -- adjust P_W, G, PI or Q"
        )
    return rounded.astype(int)


#: Cell counts in the realised sample.  Integral by construction -- checked above.
COUNTS = _cell_counts()

#: ``P(W, A, K)``, shape ``(3, 2, 3)``.  Taken from the counts rather than from the
#: constants above, so it is bit-for-bit the empirical law of :func:`frame`.
PROBS = COUNTS / N


def frame() -> pd.DataFrame:
    """The ``N``-row sample whose empirical distribution is exactly this law.

    ``Y`` is ``NaN`` wherever ``Delta`` is zero, which is how a caller would really hand
    missing outcomes to :meth:`~cleverly.TMLE.fit`.  Rows are laid out in :data:`SUPPORT`
    order, one contiguous block per support point, so :func:`first_row_of` locates a
    representative row for each.
    """
    counts = [COUNTS[w, a, k] for w, a, k in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    kind = columns[:, 2]
    return pd.DataFrame(
        {
            "W": columns[:, 0],
            "A": columns[:, 1],
            "Y": np.where(kind == UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
        }
    )


def first_row_of() -> np.ndarray:
    """Index of the first sample row belonging to each support point, in support order."""
    counts = np.array([COUNTS[w, a, k] for w, a, k in SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


#: The incremental interventions the ``ipsi`` estimands are checked against, as odds
#: multipliers.  Restated rather than imported from :mod:`tests.discrete_law`, on the same
#: terms :data:`P_W` and :data:`G` are: an oracle that reached into another module for the
#: estimand it states would make the two laws agree by construction.  That the two lists
#: are in fact the same is asserted from the outside, in
#: ``tests/unit/test_influence_gateaux_ipsi_mar.py``.
#:
#: What the missingness changes is *not* the tilt.  ``q_delta = delta g / (delta g + 1 - g)``
#: is a functional of ``P(A | W)``, and ``A`` and ``W`` are recorded for every row, so the
#: intervention is defined on a fully observed sub-law however much of ``Y`` is missing.
#: Only :math:`\bar Q` is reached by :data:`PI`, which is why the influence curve picks up
#: :math:`\pi` in the outcome term and nowhere else -- the claim under test.
IPSI_DELTAS: dict[str, float] = {"natural course": 1.0, "odds x2": 2.0, "odds x0.5": 0.5}

#: The tilt contrasts are taken against; the first supplied, as the estimator defaults.
IPSI_REFERENCE = "natural course"


def functional(probs: Any, estimand: str) -> Any:
    r"""The target parameter as a closed-form function of the cell probabilities.

    Identical in structure to :func:`tests.discrete_law.functional` bar one line, and that
    line is the whole missing-data content: the conditional mean is taken among the rows
    whose outcome was *recorded*,

    .. math::

        \bar Q(a, w) = P(Y = 1 \mid A = a, \Delta = 1, W = w),

    while the covariate distribution it is averaged against is the marginal over
    *everyone*.  Under missingness at random those two are the identification formula;
    an estimator that took both from the observed rows would be estimating something
    else, and :func:`observed_only_functional` says what.

    Every operation is arithmetic (or :func:`numpy.log`), so this stays analytic in the
    cell probabilities -- which is what lets :func:`gateaux` differentiate it by a complex
    step.  Do not introduce ``clip``, ``abs`` or a comparison here.
    """
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))  # P(W = w) -- W is recorded for everyone
    p_wa = p.sum(axis=2)  # P(W = w, A = a) -- so is A
    observed = p[:, :, OBSERVED_ZERO] + p[:, :, OBSERVED_ONE]  # P(W = w, A = a, Delta = 1)
    q = p[:, :, OBSERVED_ONE] / observed  # E[Y | A = a, Delta = 1, W = w]

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

    # The incremental estimands, written exactly as tests/discrete_law.py writes them bar
    # the definition of `q` above.  That is the point: the missingness enters this law's
    # `Qbar` and nothing else, so if the estimator needs more than a `pi` in the outcome
    # half of the clever covariate, the derivative below will say so.
    #
    # `g` is taken from the *whole* sample, not from the complete cases -- `A` is recorded
    # for everyone, so `P(A = 1 | W)` is not a missing-data problem and estimating it off
    # the observed rows would be an error this branch must not share with the estimator.
    if estimand.startswith("ey_ipsi["):
        delta = IPSI_DELTAS[estimand[len("ey_ipsi[") : -1]]
        g = p_wa[:, 1] / p_w
        d = delta * g + (1.0 - g)
        return (p_w * (delta * g * q[:, 1] + (1.0 - g) * q[:, 0]) / d).sum()
    if estimand.startswith("ate_ipsi["):
        left, right = estimand[len("ate_ipsi[") : -1].split(" vs ")
        return functional(p, f"ey_ipsi[{left}]") - functional(p, f"ey_ipsi[{right}]")

    raise ValueError(f"unknown estimand {estimand!r}")


def observed_only_functional(probs: Any, estimand: str) -> Any:
    """:func:`functional` with the covariate marginal taken among observed rows.

    Not the estimand -- the point of having it is that it is *not*.  This is what a
    plug-in that averaged the targeted predictions over the complete cases rather than
    over the whole sample would converge to, and on this law it differs from
    :func:`functional` by about 0.115 for ``ey1``.  Used as a negative control, so that
    "the plug-in averages over all ``n`` rows" is asserted rather than assumed.
    """
    p = np.asarray(probs)
    # Re-normalise onto the complete cases, then apply the same formula.  The conditional
    # means are unchanged -- they were already taken among observed rows -- so the only
    # thing that moves is the marginal of ``W`` they are averaged against.  On an ``ipsi``
    # estimand ``g`` moves too, since it is a ratio of cell probabilities and those have
    # been re-normalised; that is still the right negative control, because a complete-case
    # analysis really would fit the mechanism on the complete cases as well.
    complete = np.zeros_like(p)
    complete[:, :, OBSERVED_ZERO] = p[:, :, OBSERVED_ZERO]
    complete[:, :, OBSERVED_ONE] = p[:, :, OBSERVED_ONE]
    return functional(complete / complete.sum(), estimand)


#: The parameter names the ``ipsi`` targets report on this law, in the order the estimator
#: reports them.  This law is deliberately *not* one of the two the registry coverage gate
#: in ``tests/unit/test_registry.py`` walks -- its estimand names are the parent law's, and
#: two laws claiming one name would make ``truth_for`` depend on law order -- so these are
#: for the modules that check the ``delta=`` path to read, not for that gate.
PER_ARM_NAMES: dict[str, tuple[str, ...]] = {
    "ey_ipsi": tuple(f"ey_ipsi[{label}]" for label in IPSI_DELTAS),
    "ate_ipsi": tuple(
        f"ate_ipsi[{label} vs {IPSI_REFERENCE}]" for label in IPSI_DELTAS if label != IPSI_REFERENCE
    ),
}

#: Population values of every estimand, on the scale :func:`functional` returns.
TRUTH = {
    name: float(functional(PROBS, name))
    for name in (
        "ey1",
        "ey0",
        "ate",
        "rr",
        "or",
        "att",
        "atc",
        *PER_ARM_NAMES["ey_ipsi"],
        *PER_ARM_NAMES["ate_ipsi"],
    )
}


def gateaux(estimand: str, point: int, *, probs: Any = None, step: float = 1e-30) -> float:
    r"""The Gateaux derivative of ``estimand`` at support point ``point``.

    .. math::

        D^*(o) = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)
                 \right|_{t = 0}

    which for a pathwise-differentiable parameter in a nonparametric model *is* the
    efficient influence function -- here of the *observed-data* model, contaminated at an
    observed-data support point.  Six of the eighteen points are ``Delta = 0``, where the
    residual term of the influence curve cannot contribute and the derivative must come
    out as :math:`\bar Q(a, W) - \psi` exactly; that is the part of the claim no test in
    the suite currently reaches.

    Differentiation is by complex step rather than finite difference, for the reasons
    given in :func:`tests.discrete_law.gateaux`: full double precision, hence an exact
    comparison rather than a close one.

    ``probs`` defaults to :data:`PROBS`, the law this module declares.  Another set of cell
    probabilities gets that law's influence curve instead, which is what a study built on a
    variant of this law needs -- the randomized missing-outcome study fixes ``g`` at one half
    and still wants an efficiency bound off the same derivative rather than off a second copy
    of it.
    """
    base = (PROBS if probs is None else np.asarray(probs, dtype=float)).astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif(estimand: str, *, probs: Any = None) -> np.ndarray:
    """The EIF of ``estimand`` evaluated at every support point, in support order."""
    return np.array([gateaux(estimand, point, probs=probs) for point in range(len(SUPPORT))])


#: The nuisances as the *realised sample* has them.  Equal to :data:`G`, :data:`PI` and
#: :data:`Q` mathematically; derived from the counts so that the oracle nuisances are
#: exact in the sample down to the last bit.
G_EXACT = PROBS.sum(axis=2)[:, 1] / PROBS.sum(axis=(1, 2))
PI_EXACT = (PROBS[:, :, OBSERVED_ZERO] + PROBS[:, :, OBSERVED_ONE]) / PROBS.sum(axis=2)
Q_EXACT = PROBS[:, :, OBSERVED_ONE] / (PROBS[:, :, OBSERVED_ZERO] + PROBS[:, :, OBSERVED_ONE])


class DiscreteLaw:
    """The law, duck-typed as a ``DGP`` for the oracle learners in :mod:`tests.conftest`.

    Supplies ``propensity``, ``outcome_mean`` and -- unlike its counterpart in
    :mod:`tests.discrete_law` -- ``missingness``, which
    :class:`~tests.conftest.OracleMissingness` calls.  All three read the cell counts, so
    the oracles and the sample cannot drift apart.

    Constructed on a different set of cell probabilities -- a *tilted* law, say -- it
    supplies the nuisances of that law instead, which is what an oracle for a weighted fit
    has to be.
    """

    def __init__(self, probs: Any = None) -> None:
        p = PROBS if probs is None else np.asarray(probs, dtype=float)
        observed = p[:, :, OBSERVED_ZERO] + p[:, :, OBSERVED_ONE]
        self.probs = p
        self.g = p.sum(axis=2)[:, 1] / p.sum(axis=(1, 2))
        self.pi = observed / p.sum(axis=2)
        self.q = p[:, :, OBSERVED_ONE] / observed

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        return np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        return self.g[self._index(covariates)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        return self.q[self._index(covariates), int(arm)]

    def missingness(self, covariates: Any, arm: float) -> np.ndarray:
        return self.pi[self._index(covariates), int(arm)]


# --------------------------------------------------------------------- weighting


def cell_weights(weight_of: Any) -> np.ndarray:
    """A weight per support point, from a function of ``(w, a, k)``.

    Observation weights are a function of the observed row, so on a law with finite
    support they are eighteen numbers.  A weight may depend on ``Y`` only through ``k``,
    which is as it should be: a row with no recorded outcome cannot be weighted by one.
    """
    return np.array([float(weight_of(w, a, k)) for w, a, k in SUPPORT], dtype=float)


def row_weights(weights: np.ndarray) -> np.ndarray:
    """Cell weights expanded to one value per row of :func:`frame`."""
    counts = [COUNTS[w, a, k] for w, a, k in SUPPORT]
    return np.repeat(np.asarray(weights, dtype=float), counts)


def tilt(probs: Any, weights: Any) -> Any:
    r"""The weighted law :math:`dP_w = w\,dP / E_P[w]`, as cell probabilities.

    Kept analytic in ``probs`` -- a ratio of linear functions -- so :func:`weighted_gateaux`
    can differentiate through it by a complex step.
    """
    p = np.asarray(probs)
    w = np.asarray(weights, dtype=float).reshape(len(SUPPORT))
    cells = np.zeros_like(p)
    for index, (a, b, c) in enumerate(SUPPORT):
        cells[a, b, c] = w[index]
    tilted = cells * p
    return tilted / tilted.sum()


def weighted_functional(probs: Any, estimand: str, weights: Any) -> Any:
    """``Psi(P_w)`` -- the estimand of the tilted law, longhand.

    The tilt and the missingness interact: the weighted fit's missingness model converges
    to :math:`P_w(\\Delta = 1 \\mid A, W)`, not to :math:`P_0`'s, and the covariate
    marginal the plug-in averages against is the tilted one.  Writing the parameter this
    way makes both statements testable rather than assumed.
    """
    return functional(tilt(probs, weights), estimand)


def weighted_gateaux(estimand: str, point: int, weights: Any, *, step: float = 1e-30) -> float:
    r"""Gateaux derivative of :math:`P \mapsto \Psi(P_w)` at support point ``point``.

    The contamination is of :math:`P`, the law the *rows are drawn from* -- not of
    :math:`P_w`.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(weighted_functional(perturbed, estimand, weights)) / step)


def weighted_eif(estimand: str, weights: Any) -> np.ndarray:
    """The EIF of ``Psi(P_w)`` at every support point, in support order."""
    return np.array([weighted_gateaux(estimand, point, weights) for point in range(len(SUPPORT))])
