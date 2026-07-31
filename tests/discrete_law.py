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


#: The regimes the regime-indexed estimands are checked against, ``g*(a | W = w)`` as a
#: ``(3, 2)`` array per label.  Declared here, in the oracle, rather than in the test that
#: fits them: the oracle keys on the *reported parameter name*, which carries the regime's
#: label, so the label is part of what is being checked.  ``tests/unit/test_regimes.py``
#: builds the matching :class:`~cleverly.interventions.Intervention` objects and asserts
#: their densities equal these, which is what ties the two sides together.
#:
#: One of each kind, deliberately: a static regime, a deterministic rule that actually
#: depends on ``W``, and a stochastic one that is degenerate nowhere.  Two static regimes
#: could not tell code that mixes over the arms from code that picks one column.
REGIMES: dict[str, np.ndarray] = {
    "never": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    "rule": np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    "tilt": np.array([[0.75, 0.25], [0.50, 0.50], [0.25, 0.75]]),
}

#: The regime contrasts are taken against; the first supplied, as the estimator defaults.
REGIME_REFERENCE = "never"


#: The incremental interventions the ``ipsi`` estimands are checked against, as odds
#: multipliers.  Unlike :data:`REGIMES` these are *not* densities: the density
#: ``q_delta = delta g / (delta g + 1 - g)`` is a function of the law, so it has to be
#: computed inside :func:`functional` from the cell probabilities.  That is exactly what
#: is under test -- the complex step then differentiates through ``g`` and produces the
#: extra influence-curve term for free, which no ``REGIMES`` entry can exercise.
#:
#: One above one, one below, and one *at* one.  A sign error in ``dm/dg`` survives only
#: one side of one, and ``delta = 1`` is the natural course, where ``q_1 = g`` exactly and
#: the influence curve collapses to ``Y - E[Y]`` row by row whatever the nuisances are.
IPSI_DELTAS: dict[str, float] = {"natural course": 1.0, "odds x2": 2.0, "odds x0.5": 0.5}

#: The tilt contrasts are taken against; the first supplied, as the estimator defaults.
IPSI_REFERENCE = "natural course"


#: The working model the ``msm`` estimand is checked against: ``m(a, W) = b0 + b1 a + b2 W``,
#: with ``W`` read as a number.  Its term names are part of what is checked, because they
#: are what the reported parameter names carry.
MSM_TERMS: tuple[str, ...] = ("(intercept)", "a", "W")

#: ``phi(a, W = w)`` as a ``(3, 2, 3)`` array indexed ``[w, a, term]``, and the weights
#: ``h(a, W = w)`` as ``(3, 2)``.  Built here, outside :func:`functional`, for the reason
#: ``tests/discrete_law_shift.py`` precomputes its shift map: both are constants of the
#: law, and constructing them inside would put a comparison or an integer index into a
#: function that has to stay analytic in the cell probabilities.
#:
#: Three coefficients against six ``(w, a)`` cells, so the model is **not** saturated and
#: ``beta`` really is a projection rather than a reparameterisation of the six conditional
#: means.  That is the point of choosing it: a saturated working model would agree with the
#: means whatever the projection code did, and could not tell a correct ``M^-1`` from a
#: missing one.
#:
#: The weights are **not** uniform, and that is load-bearing rather than decoration.  With
#: ``h = 1`` the ``a`` column is orthogonal to ``{1, W}`` across the cells -- each ``w``
#: carries both arms at the same mass -- so ``beta_a`` collapses to the marginal ATE
#: *identically*, as a functional and not just numerically.  Its influence curve would then
#: be the ATE's too, and code that reported the ATE under the name ``msm[a]`` would pass
#: every check here.  A weight that varies in both arguments breaks that orthogonality and
#: is the only thing that exercises ``h`` at all.
MSM_DESIGN = np.array([[[1.0, float(a), float(w)] for a in range(2)] for w in range(3)])
MSM_WEIGHTS = np.array([[1.0 + 0.5 * a + 0.25 * w for a in range(2)] for w in range(3)])

#: The links the ``msm`` estimand is checked under, and their mean functions written out
#: *here* rather than imported.  The whole point of an oracle is that it shares no code
#: with what it checks, and ``dm/deta`` is the factor a linked clever covariate carries --
#: so taking it from :mod:`cleverly.msm` would be checking that module against itself.
#:
#: ``expit`` is spelled ``1 / (1 + exp(-eta))`` because :func:`gateaux` evaluates this at a
#: *complex* argument, and ``scipy.special.expit`` is real-only.  Everything below is
#: arithmetic and :func:`numpy.exp`, which is what keeps the whole functional analytic.
MSM_LINKS: dict[str, Any] = {
    "identity": (lambda eta: eta, lambda m: np.ones_like(m), lambda m: np.zeros_like(m)),
    "log": (np.exp, lambda m: m, lambda m: m),
    "logit": (
        lambda eta: 1.0 / (1.0 + np.exp(-eta)),
        lambda m: m * (1.0 - m),
        lambda m: m * (1.0 - m) * (1.0 - 2.0 * m),
    ),
}

#: Newton steps taken to solve a linked working model's normal equations, run **without a
#: convergence test** -- a comparison against a tolerance is not analytic, and a functional
#: that branched on one could not be differentiated by a complex step.
#:
#: A fixed count is not a compromise here.  Newton converges quadratically in the value and
#: the derivative alike, so past the point where the real part stops moving the imaginary
#: part is exact too; ``TestTheNewtonSolveHasConverged`` doubles this and checks nothing
#: moves, which is the statement that matters rather than the number itself.
MSM_NEWTON_STEPS = 40


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

    if estimand in ("ey1", "ey[1]"):
        return psi_one
    if estimand in ("ey0", "ey[0]"):
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

    # Psi_r = sum_w P(W = w) sum_a g*_r(a | w) E[Y | A = a, W = w], written straight off
    # the g-formula for a regime. It reduces to psi_one above when g* puts all its mass
    # on arm 1, which is the sense in which a static intervention is a special case.
    if estimand.startswith("ey_regime["):
        star = REGIMES[estimand[len("ey_regime[") : -1]]
        return (p_w * (star * q).sum(axis=1)).sum()
    if estimand.startswith("ate_regime["):
        left, right = estimand[len("ate_regime[") : -1].split(" vs ")
        return functional(p, f"ey_regime[{left}]") - functional(p, f"ey_regime[{right}]")

    # beta = M^-1 b, the h-weighted least-squares projection of the counterfactual means
    # onto m(a, W; beta), written straight off the normal equations:
    #
    #   M = sum_w P(w) sum_a h(a, w) phi(a, w) phi(a, w)'
    #   b = sum_w P(w) sum_a h(a, w) phi(a, w) E[Y | A = a, W = w]
    #
    # M depends on the law only through P(W), which is where its own contribution to the
    # influence curve comes from -- an implementation that treated M as a constant would
    # pass every check that fixes P(W) and fail this one.
    if estimand.startswith("ey_ipsi["):
        # Psi(delta) = sum_w P(W = w) sum_a q_delta(a | w) E[Y | A = a, W = w], with
        # q_delta the odds of g multiplied by delta. `g` is a ratio of linear forms in the
        # cell probabilities, so this stays analytic and the complex step differentiates
        # through the mechanism as well as through Qbar -- which is the whole point.
        delta = IPSI_DELTAS[estimand[len("ey_ipsi[") : -1]]
        g = p_wa[:, 1] / p_w
        d = delta * g + (1.0 - g)
        return (p_w * (delta * g * q[:, 1] + (1.0 - g) * q[:, 0]) / d).sum()
    if estimand.startswith("ate_ipsi["):
        left, right = estimand[len("ate_ipsi[") : -1].split(" vs ")
        return functional(p, f"ey_ipsi[{left}]") - functional(p, f"ey_ipsi[{right}]")

    if estimand.startswith("msm["):
        gram = np.einsum("wap,waq,wa,w->pq", MSM_DESIGN, MSM_DESIGN, MSM_WEIGHTS, p_w)
        moment = np.einsum("wap,wa,wa,w->p", MSM_DESIGN, MSM_WEIGHTS, q, p_w)
        beta = np.linalg.solve(gram, moment)
        return beta[MSM_TERMS.index(estimand[len("msm[") : -1])]

    # The same projection through a link, where the normal equations are no longer linear
    # in beta:
    #
    #   U(beta) = sum_w P(w) sum_a h phi (dm/deta) (E[Y | a, w] - m(a, w; beta)) = 0
    #
    # solved by Newton with the exact Jacobian. The Jacobian carries the curvature term
    # -(Qbar - m) d2m/deta2, which vanishes only where the model fits -- and this one is
    # three coefficients against six cells, so it does not.
    for name, (inverse, slope, curvature) in MSM_LINKS.items():
        if name == "identity" or not estimand.startswith(f"msm_{name}["):
            continue
        beta = np.zeros(len(MSM_TERMS), dtype=p.dtype)
        for _ in range(MSM_NEWTON_STEPS):
            m = inverse(np.einsum("wap,p->wa", MSM_DESIGN, beta))
            residual = q - m
            first, second = slope(m), curvature(m)
            score = np.einsum("wap,wa,w->p", MSM_DESIGN, MSM_WEIGHTS * first * residual, p_w)
            jacobian = np.einsum(
                "wap,waq,wa,w->pq",
                MSM_DESIGN,
                MSM_DESIGN,
                MSM_WEIGHTS * (first**2 - residual * second),
                p_w,
            )
            beta = beta + np.linalg.solve(jacobian, score)
        return beta[MSM_TERMS.index(estimand[len(f"msm_{name}[") : -1])]

    raise ValueError(f"unknown estimand {estimand!r}")


#: The parameter names a target reports on this two-armed law, for targets that report
#: one parameter per arm rather than one under their own name.  ``ey`` is the arm-general
#: counterfactual mean: with two arms it reports ``ey[0]`` and ``ey[1]``, which are the
#: same two numbers ``ey0`` and ``ey1`` report and are checked against the same oracle.
PER_ARM_NAMES: dict[str, tuple[str, ...]] = {
    "ey": ("ey[0]", "ey[1]"),
    "ey_regime": tuple(f"ey_regime[{label}]" for label in REGIMES),
    "ey_ipsi": tuple(f"ey_ipsi[{label}]" for label in IPSI_DELTAS),
    "ate_ipsi": tuple(
        f"ate_ipsi[{label} vs {IPSI_REFERENCE}]" for label in IPSI_DELTAS if label != IPSI_REFERENCE
    ),
    "ate_regime": tuple(
        f"ate_regime[{label} vs {REGIME_REFERENCE}]"
        for label in REGIMES
        if label != REGIME_REFERENCE
    ),
    # One family per link.  The *estimator* reports every one of them as ``msm[term]`` --
    # a fit declares one link and its coefficients are its coefficients -- so the oracle
    # names them apart, since a law's truths are one dict and three families of three
    # would collide under one name. ``tests/unit/test_influence_gateaux_msm.py`` maps
    # between the two, and the coverage gate in ``test_registry.py`` walks these.
    "msm": tuple(
        f"msm[{term}]" if link == "identity" else f"msm_{link}[{term}]"
        for link in MSM_LINKS
        for term in MSM_TERMS
    ),
}


def msm_names(link: str) -> tuple[str, ...]:
    """This law's names for a working model's coefficients under ``link``."""
    stem = "msm" if link == "identity" else f"msm_{link}"
    return tuple(f"{stem}[{term}]" for term in MSM_TERMS)


def oracle_names(target: str) -> tuple[str, ...]:
    """The parameter name(s) ``target`` reports here, or none if this law does not own it.

    A target is one *functional*, and a functional can report more than one number --
    which is what made this indirection necessary once a treatment could have more than
    two arms.  The coverage gate in ``tests/unit/test_registry.py`` walks these rather
    than the bare target names, so a per-arm target still cannot ship without an oracle.

    Returning ``()`` for a target this law does not cover is what lets that gate walk
    *several* laws: a shift is checked against ``tests/discrete_law_shift.py``, whose
    treatment has four ordered doses rather than two arms, and neither law can express
    the other's estimands.  The bare-name fallback is guarded by :data:`TRUTH` so that
    "this law has no branch for it" and "this law owns it under its own name" stay
    distinguishable.
    """
    if target in PER_ARM_NAMES:
        return PER_ARM_NAMES[target]
    return (target,) if target in TRUTH else ()


#: Population values of every estimand, on the scale :func:`functional` returns.
TRUTH = {
    name: float(functional(PROBS, name))
    for name in (
        "ey1",
        "ey0",
        "ey[0]",
        "ey[1]",
        "ate",
        "rr",
        "or",
        "att",
        "atc",
        *PER_ARM_NAMES["ey_regime"],
        *PER_ARM_NAMES["ate_regime"],
        *PER_ARM_NAMES["ey_ipsi"],
        *PER_ARM_NAMES["ate_ipsi"],
        *PER_ARM_NAMES["msm"],
    )
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

    Constructed on a different set of cell probabilities -- a *tilted* law, say -- it
    supplies the nuisances of that law instead, which is what an oracle for a weighted
    fit has to be: weighted learners converge to the tilted conditionals, not to
    :math:`P_0`'s.
    """

    def __init__(self, probs: Any = None) -> None:
        p = PROBS if probs is None else np.asarray(probs, dtype=float)
        self.probs = p
        self.g = p.sum(axis=2)[:, 1] / p.sum(axis=(1, 2))
        self.q = p[:, :, 1] / p.sum(axis=2)

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        return np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        return self.g[self._index(covariates)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        return self.q[self._index(covariates), int(arm)]


# --------------------------------------------------------------------- weighting


def cell_weights(weight_of: Any) -> np.ndarray:
    """A weight per support point, from a function of ``(w, a, y)``.

    Observation weights are a function of the observed row, so on a law with finite
    support they are twelve numbers.
    """
    return np.array([float(weight_of(w, a, y)) for w, a, y in SUPPORT], dtype=float)


def row_weights(weights: np.ndarray) -> np.ndarray:
    """Cell weights expanded to one value per row of :func:`frame`."""
    counts = [COUNTS[w, a, y] for w, a, y in SUPPORT]
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

    This is the parameter :mod:`cleverly.data.weighting` says a weighted fit estimates,
    written here without reference to any library code: tilt the law, then apply the same
    identification formula :func:`functional` already spells out.
    """
    return functional(tilt(probs, weights), estimand)


def weighted_gateaux(estimand: str, point: int, weights: Any, *, step: float = 1e-30) -> float:
    r"""Gateaux derivative of :math:`P \mapsto \Psi(P_w)` at support point ``point``.

    The contamination is of :math:`P`, the law the *rows are drawn from* -- not of
    :math:`P_w`.  That is the whole content of the check: the weights are part of the
    data-generating experiment, so the influence function has to be taken with respect to
    the law that generates them.
    """
    base = PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(weighted_functional(perturbed, estimand, weights)) / step)


def weighted_eif(estimand: str, weights: Any) -> np.ndarray:
    """The EIF of ``Psi(P_w)`` at every support point, in support order."""
    return np.array([weighted_gateaux(estimand, point, weights) for point in range(len(SUPPORT))])
