r"""The laws of the ``small-sample-wald`` study: four diagnosis laws and three held-out laws.

**Diagnosis laws.**  The four laws of the Fix-3 planning simulations (plans
``rows/Fix-3/exploratory/sim_fresh.py``), moved here unchanged.  They chose the rule, so the
study labels their cells as diagnosis cells and no falsifier reads them.

==============  =========================================================================
law             definition, ``n`` rows
==============  =========================================================================
``lognormal``   ``V`` in {0, 1, 2} with ``P(V) = (0.6, 0.3, 0.1)``; ``W ~ N(0, 1)``;
                ``A ~ Bernoulli(expit(-0.2 + 0.6 W - 0.3 V))``;
                ``Y = 1 + A + 0.5 W + 0.3 V + exp(N(0, 1.2^2))`` (error skewness 11.2)
``lognormal06`` as ``lognormal`` with ``exp(N(0, 0.6^2))`` (error skewness 2.3)
``student``     as ``lognormal`` with a Student t error on 3 df, scale 1, no skewness
``rare``        ``V`` in {0, 1, 2} with ``P(V) = (0.55, 0.30, 0.15)``; ``W ~ N(0, 1)``;
                ``A ~ Bernoulli(expit(0.1 + 0.5 W - 0.2 V))``;
                ``Y ~ Bernoulli(expit(-2.2 - 0.8 A + 0.7 W - 0.4 V))``
==============  =========================================================================

The continuous truths are closed form (``E[W] = 0`` in every stratum); the ``rare`` truths are
Monte Carlo means over 4,000,000 normal draws from ``default_rng(1)``, as the planning
simulations used them.  :func:`rare_quadrature_truths` gives the same truths by 80-node
Gauss-Hermite quadrature, and the design test checks the two agree.  The fit of every
diagnosis law is :func:`diagnosis_fit`: stratified in-sample TMLE, a correct linear or
logistic outcome model, a correct logistic treatment model, ``g_bounds = (0.01, 0.99)``.

**Held-out laws** (Fix-3 plan rev 4, section 5.2.2), declared before any simulation on them.
No estimator has been fitted on them.  Their truths are exact: finite sums over the support,
or closed-form moments.

``H1`` (:func:`h1_sample`): a sparse treated arm by treatment rarity, no strata.
``W`` in {0, 1, 2, 3} with probabilities (0.4, 0.3, 0.2, 0.1);
``A | W ~ Bernoulli(expit(-3.4 + 0.25 W))``; ``Y | A, W ~ Bernoulli(expit(-0.6 + 2.4 A + 0.35 W))``.
Fit (:func:`h1_fit`): plain TMLE in sample, unpenalized logistic outcome model on ``(A, W)``
and treatment model on ``W`` (both correct), default ``g_bounds``.

``H2`` (:func:`h2_sample`): a continuous outcome with a point mass and gamma skew.  ``V`` in
{0, 1, 2} with probabilities (0.60, 0.34, 0.06); ``W | V`` with the rows of
:data:`tests.studies.stratified_law.P_W_GIVEN_V`; ``A ~ Bernoulli(expit(-0.2 + 0.5 W - 0.4 V))``;
``Y = B G`` with ``B ~ Bernoulli(expit(0.4 + 0.5 A - 0.3 W + 0.2 V))`` and
``G ~ Gamma(shape 1.5, scale exp(0.3 + 0.4 A + 0.25 W + 0.1 V))``, independent given
``(A, W, V)``, so ``E[Y | A, W, V] = 1.5 pi theta``.  Fit (:func:`h2_fit`): stratified TMLE in
sample, the saturated cell-mean least-squares outcome model :class:`CellMeans` on the 18
``(A, W, V)`` cells (correct), a logistic treatment model on ``(W, V)`` (correct).  The model
refuses a sample with an empty cell (:data:`EMPTY_DESIGN_CELL`), and the study records that
replicate as a failed fit (plan rev 4, B-1).

``H3`` (:func:`h3_sample`): L1's support (``V``, ``W`` in {0, 1, 2}, binary ``A`` and ``Y``) with
``P(V) = (0.45, 0.40, 0.15)``, ``P(W | V)`` as L1, ``g = expit(-1.2 + 0.6 W - 0.5 V)`` and
``Q = expit(-0.4 + 0.9 A - 0.4 W + 0.5 V)``.  Fit: the logit MSM ``(1, a, W)`` of
:func:`tests.studies.stratified_incremental_msm_properties.fit_msm`.  The working model
contains ``Q`` inside each stratum, so the truths are ``Q``'s coefficients.

The design quantities of every law (the population moment df ``nu_pop = 2 m / (kappa - 1)``,
the exact constant-cell probabilities, the distance to the nearer bound, the empty-cell
probability of H2) are computed here without sampling, by the functions of the planning
script ``heldout_design.py``.  ``tests/unit/test_small_sample_wald_design.py`` recomputes every
number that script printed.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from tests import discrete_law
from tests.studies import stratified_alternating_law as alternating
from tests.studies import stratified_law as l1

# ------------------------------------------------------------------------ diagnosis laws

#: The lognormal error's log-scale SD of ``lognormal``.
S_LOG = 1.2
DIAGNOSIS_LAWS = ("rare", "student", "lognormal06", "lognormal")
#: The outcome family each diagnosis law's outcome has.
DIAGNOSIS_FAMILY = {
    "rare": "binomial",
    "student": "gaussian",
    "lognormal06": "gaussian",
    "lognormal": "gaussian",
}
STRATUM_NAMES = ("ey[1][V=2]", "ey[0][V=2]", "ate[V=2]")
#: Normal draws behind the Monte Carlo truths of ``rare``.
RARE_TRUTH_DRAWS = 4_000_000


def diagnosis_sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows ``(V, W, A, Y)`` of a diagnosis law from ``seed``."""
    rng = np.random.default_rng(seed)
    if law in ("lognormal", "lognormal06", "student"):
        v = rng.choice(3, n, p=[0.6, 0.3, 0.1])
        w = rng.normal(size=n)
        a = rng.binomial(1, expit(-0.2 + 0.6 * w - 0.3 * v))
        if law == "student":
            error = rng.standard_t(3, n)
        else:
            error = np.exp(rng.normal(0, S_LOG if law == "lognormal" else 0.6, n))
        y: Any = 1.0 + 1.0 * a + 0.5 * w + 0.3 * v + error
    elif law == "rare":
        v = rng.choice(3, n, p=[0.55, 0.30, 0.15])
        w = rng.normal(size=n)
        a = rng.binomial(1, expit(0.1 + 0.5 * w - 0.2 * v))
        y = rng.binomial(1, expit(-2.2 - 0.8 * a + 0.7 * w - 0.4 * v))
    else:
        raise KeyError(law)
    return pd.DataFrame({"V": v, "W": w, "A": a, "Y": y})


@functools.cache
def _diagnosis_truths(law: str) -> tuple[tuple[str, float], ...]:
    if law in ("lognormal", "lognormal06", "student"):
        s = {"lognormal": S_LOG, "lognormal06": 0.6}.get(law)
        base = 1.0 + 0.3 * 2 + (float(np.exp(s**2 / 2)) if s else 0.0)
        values = {"ey[1][V=2]": base + 1.0, "ey[0][V=2]": base, "ate[V=2]": 1.0}
    elif law == "rare":
        w = np.random.default_rng(1).normal(size=RARE_TRUTH_DRAWS)
        e1 = float(expit(-2.2 - 0.8 + 0.7 * w - 0.8).mean())
        e0 = float(expit(-2.2 + 0.7 * w - 0.8).mean())
        values = {"ey[1][V=2]": e1, "ey[0][V=2]": e0, "ate[V=2]": e1 - e0}
    else:
        raise KeyError(law)
    return tuple(values.items())


def diagnosis_truths(law: str) -> dict[str, float]:
    """The three stratum-2 truths of a diagnosis law."""
    return dict(_diagnosis_truths(law))


def logistic() -> LogisticRegression:
    """The unpenalized logistic regression of every binary nuisance here."""
    return LogisticRegression(C=1e6, max_iter=2000)


def diagnosis_fit(frame: pd.DataFrame, law: str) -> Any:
    """The planning simulations' fit: stratified in-sample TMLE with correct models."""
    outcome = logistic() if law == "rare" else LinearRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=outcome,
                treatment_learner=logistic(),
                cross_fit=False,
                estimands=("ey", "ate"),
                simultaneous=False,
                g_bounds=(0.01, 0.99),
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
            .single()
        )


# ------------------------------------------------------------------ moments and the df


def binary_moments(q: Any) -> tuple[Any, Any, Any]:
    """Central moments 2 to 4 of ``Y - q`` for ``Y ~ Bernoulli(q)``."""
    return q * (1 - q), q * (1 - q) * (1 - 2 * q), q * (1 - q) * (1 - 3 * q + 3 * q * q)


def central_from_raw(r1: Any, r2: Any, r3: Any, r4: Any) -> tuple[Any, Any, Any]:
    """Central moments 2 to 4 from raw moments 1 to 4."""
    m2 = r2 - r1**2
    m3 = r3 - 3 * r1 * r2 + 2 * r1**3
    m4 = r4 - 4 * r1 * r3 + 6 * r1**2 * r2 - 3 * r1**4
    return m2, m3, m4


def lognormal_moments(s: float, shape: Any) -> tuple[tuple[Any, Any, Any], float]:
    """Central moments 2 to 4 of ``exp(N(0, s^2))``, broadcast to ``shape``, and its mean."""
    m1 = float(np.exp(s * s / 2))
    raw = [np.exp(j * j * s * s / 2) for j in (1, 2, 3, 4)]
    moments = central_from_raw(*raw)
    return tuple(np.full(shape, x) for x in moments), m1  # type: ignore[return-value]


def hurdle_gamma_raw(pi: Any, theta: Any, k: float) -> list[Any]:
    """Raw moments 1 to 4 of ``B G`` with ``B ~ Bernoulli(pi)``, ``G ~ Gamma(k, theta)``."""
    return [pi * theta**j * np.exp(gammaln(k + j) - gammaln(k)) for j in (1, 2, 3, 4)]


def kappa_pt(
    pw: Any, g: Any, q1: Any, q0: Any, mom1: Any, mom0: Any, estimand: str
) -> tuple[float, float, float]:
    """Truth, kurtosis and SD of a point-treatment curve over the rows that carry it.

    ``pw`` holds the covariate cell weights given the stratum, ``mom1`` and ``mom0`` the
    central moments 2 to 4 of ``Y - Q`` in each arm, and ``estimand`` is ``ey1``, ``ey0`` or
    ``ate``.  The curve is ``A / g (Y - Q1) + Q1 - psi`` and its mirror images.
    """
    pw = np.asarray(pw, dtype=float)
    pw = pw / pw.sum()
    zero = (0.0, 0.0, 0.0)
    if estimand == "ey1":
        psi = float(np.sum(pw * q1))
        c = q1 - psi
        terms = [(g, 1 / g, mom1), (1 - g, 0.0, zero)]
    elif estimand == "ey0":
        psi = float(np.sum(pw * q0))
        c = q0 - psi
        terms = [(1 - g, 1 / (1 - g), mom0), (g, 0.0, zero)]
    elif estimand == "ate":
        psi = float(np.sum(pw * (q1 - q0)))
        c = q1 - q0 - psi
        terms = [(g, 1 / g, mom1), (1 - g, -1 / (1 - g), mom0)]
    else:
        raise KeyError(estimand)
    e2 = np.sum(pw * c * c)
    e4 = np.sum(pw * c**4)
    for pa, b, (m2, m3, m4) in terms:
        e2 = e2 + np.sum(pw * pa * b * b * m2)
        e4 = e4 + np.sum(pw * pa * (6 * c * c * b * b * m2 + 4 * c * b**3 * m3 + b**4 * m4))
    return psi, float(e4 / e2**2), float(np.sqrt(e2))


def nu_pop(kappa: float, m: float) -> float:
    """The population moment df ``2 m / (kappa - 1)``; ``inf`` when ``kappa <= 1``."""
    return float("inf") if kappa <= 1 else float(2 * m / (kappa - 1))


def const_prob(n: int, q_a: float, q_a1: float) -> float:
    """P(an arm holds rows and all of them share one binary outcome value), iid rows.

    ``q_a = P(stratum, A = a)`` and ``q_a1 = P(stratum, A = a, Y = 1)``.
    """
    q_a0 = q_a - q_a1
    return float((1 - q_a0) ** n + (1 - q_a1) ** n - 2 * (1 - q_a) ** n)


def either(p: float, q: float) -> float:
    """The withheld share of an estimand that reads two arms, constant independently."""
    return p + q - p * q


GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = _GH_W / _GH_W.sum()


def l1_cells(v: int) -> tuple[Any, Any, Any, Any]:
    """L1's ``P(W | V = v)``, ``g``, ``Q(1, .)`` and ``Q(0, .)`` on ``W`` in {0, 1, 2}."""
    w = np.arange(3.0)
    return l1.P_W_GIVEN_V[v], l1.propensity(w, v), l1.outcome(1, w, v), l1.outcome(0, w, v)


def l1_pt(v: int, estimand: str) -> tuple[float, float, float]:
    """:func:`kappa_pt` of an L1 stratum parameter."""
    pw, g, q1, q0 = l1_cells(v)
    return kappa_pt(pw, g, q1, q0, binary_moments(q1), binary_moments(q0), estimand)


def l1_const(n: int, v: int, arm: int, probs: Any = None) -> float:
    """The exact constant-cell probability of an arm in stratum ``v`` on L1's support."""
    probs = l1.PROBS if probs is None else probs
    support = np.asarray(l1.SUPPORT)
    selected = (support[:, 0] == v) & (support[:, 2] == arm)
    return const_prob(
        n, float(probs[selected].sum()), float(probs[selected & (support[:, 3] == 1)].sum())
    )


def msm_kappa(
    probs: Any, v: int, term: str, link: str = "logit"
) -> tuple[float, float, float, float]:
    """Truth, kurtosis over the stratum, ``P(V = v)`` and efficiency SD of an MSM coefficient.

    The curve is the complex-step Gateaux derivative of
    :func:`tests.studies.stratified_alternating_law.msm_beta` along the contamination path.
    """
    index = alternating.MSM_TERMS.index(term)

    def functional(p: Any) -> Any:
        return alternating.msm_beta(p, link, v)[index]

    cells = [(i,) for i in range(len(alternating.SUPPORT))]
    curve = np.asarray(discrete_law.contamination_eif(functional, probs, cells), dtype=float)
    inside = np.array([point[0] for point in alternating.SUPPORT]) == v
    p = probs[inside] / probs[inside].sum()
    u = curve[inside]
    u = u - np.sum(p * u)
    kappa = float(np.sum(p * u**4) / np.sum(p * u * u) ** 2)
    sd = float(np.sqrt(np.sum(probs * curve * curve)))
    return float(np.real(functional(probs))), kappa, float(probs[inside].sum()), sd


def continuous_pt(law: str, estimand: str) -> tuple[float, float, float]:
    """:func:`kappa_pt` of a continuous diagnosis law in stratum 2, by quadrature over ``W``."""
    w, pw, v = GH_X, GH_W, 2
    g = expit(-0.2 + 0.6 * w - 0.3 * v)
    if law == "student":
        # The t on 3 df has no fourth moment: the kurtosis is infinite and nu_pop is 0.
        return float("nan"), float("inf"), float("nan")
    s = {"lognormal": S_LOG, "lognormal06": 0.6}[law]
    moments, m1 = lognormal_moments(s, w.shape)
    q0 = 1.0 + 0.5 * w + 0.3 * v + m1
    q1 = q0 + 1.0
    return kappa_pt(pw, g, q1, q0, moments, moments, estimand)


def rare_cells() -> tuple[Any, Any, Any, Any]:
    """The ``rare`` law in stratum 2 on the quadrature nodes of ``W``."""
    w, pw, v = GH_X, GH_W, 2
    g = expit(0.1 + 0.5 * w - 0.2 * v)
    q1 = expit(-2.2 - 0.8 + 0.7 * w - 0.4 * v)
    q0 = expit(-2.2 + 0.7 * w - 0.4 * v)
    return pw, g, q1, q0


def rare_pt(estimand: str) -> tuple[float, float, float]:
    """:func:`kappa_pt` of a ``rare`` stratum-2 parameter."""
    pw, g, q1, q0 = rare_cells()
    return kappa_pt(pw, g, q1, q0, binary_moments(q1), binary_moments(q0), estimand)


def rare_const(n: int, arm: int) -> float:
    """The exact constant-cell probability of an arm of ``rare`` in stratum 2."""
    pw, g, q1, q0 = rare_cells()
    share = 0.15
    if arm == 1:
        q_a, q_a1 = share * np.sum(pw * g), share * np.sum(pw * g * q1)
    else:
        q_a, q_a1 = share * np.sum(pw * (1 - g)), share * np.sum(pw * (1 - g) * q0)
    return const_prob(n, float(q_a), float(q_a1))


def rare_quadrature_truths() -> dict[str, float]:
    """The ``rare`` truths by quadrature, the check on the Monte Carlo truths."""
    one, zero = rare_pt("ey1")[0], rare_pt("ey0")[0]
    return {"ey[1][V=2]": one, "ey[0][V=2]": zero, "ate[V=2]": one - zero}


# ---------------------------------------------------------------------------- H1


H1_PW = np.array([0.4, 0.3, 0.2, 0.1])
H1_NAMES = ("ey[1]", "ey[0]", "ate")


def h1_cells() -> tuple[Any, Any, Any, Any]:
    """H1's ``P(W)``, ``g``, ``Q(1, .)`` and ``Q(0, .)`` on ``W`` in {0, 1, 2, 3}."""
    w = np.arange(4.0)
    g = expit(-3.4 + 0.25 * w)
    return H1_PW, g, expit(-0.6 + 2.4 + 0.35 * w), expit(-0.6 + 0.35 * w)


def h1_pt(estimand: str) -> tuple[float, float, float]:
    """:func:`kappa_pt` of an H1 parameter."""
    pw, g, q1, q0 = h1_cells()
    return kappa_pt(pw, g, q1, q0, binary_moments(q1), binary_moments(q0), estimand)


def h1_const(n: int) -> tuple[float, float]:
    """The exact constant-cell probabilities of H1's treated and untreated arms."""
    pw, g, q1, q0 = h1_cells()
    treated = const_prob(n, float(np.sum(pw * g)), float(np.sum(pw * g * q1)))
    untreated = const_prob(n, float(np.sum(pw * (1 - g))), float(np.sum(pw * (1 - g) * q0)))
    return treated, untreated


def h1_truths() -> dict[str, float]:
    """H1's three truths."""
    return {
        name: h1_pt(short)[0] for name, short in zip(H1_NAMES, ("ey1", "ey0", "ate"), strict=True)
    }


def h1_sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows ``(W, A, Y)`` of H1 from ``seed``."""
    rng = np.random.default_rng(seed)
    pw, g, _, _ = h1_cells()
    w = rng.choice(4, size=n, p=pw)
    a = rng.binomial(1, g[w])
    y = rng.binomial(1, expit(-0.6 + 2.4 * a + 0.35 * w))
    return pd.DataFrame({"W": w, "A": a, "Y": y})


def h1_fit(frame: pd.DataFrame) -> Any:
    """Plain in-sample TMLE with correct logistic models and the default ``g_bounds``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=logistic(),
                treatment_learner=logistic(),
                cross_fit=False,
                estimands=("ey", "ate"),
                simultaneous=False,
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"])
            .single()
        )


# ---------------------------------------------------------------------------- H2


H2_PV = np.array([0.60, 0.34, 0.06])
H2_SHAPE = 1.5
H2_NAMES = STRATUM_NAMES
#: The reason :class:`CellMeans` gives when it refuses a sample with an empty cell.
EMPTY_DESIGN_CELL = "empty_design_cell"


def h2_pi(a: Any, w: Any, v: Any) -> Any:
    """``P(B = 1 | A, W, V)``, the share of positive outcomes."""
    return expit(
        0.4 + 0.5 * np.asarray(a, float) - 0.3 * np.asarray(w, float) + 0.2 * np.asarray(v, float)
    )


def h2_theta(a: Any, w: Any, v: Any) -> Any:
    """The gamma scale given ``(A, W, V)``."""
    return np.exp(
        0.3 + 0.4 * np.asarray(a, float) + 0.25 * np.asarray(w, float) + 0.1 * np.asarray(v, float)
    )


def h2_propensity(w: Any, v: Any) -> Any:
    """``g(w, v) = P(A = 1 | W = w, V = v)``."""
    return expit(-0.2 + 0.5 * np.asarray(w, float) - 0.4 * np.asarray(v, float))


def h2_cells(v: int = 2) -> tuple[Any, Any, dict[int, tuple[Any, Any, Any]]]:
    """H2's ``P(W | V = v)``, ``g`` and per arm ``(Q, central moments of Y - Q, pi)``."""
    w = np.arange(3.0)
    out = {}
    for a in (0, 1):
        pi = h2_pi(a, w, v)
        raw = hurdle_gamma_raw(pi, h2_theta(a, w, v), H2_SHAPE)
        out[a] = (raw[0], central_from_raw(*raw), pi)
    return l1.P_W_GIVEN_V[v], h2_propensity(w, v), out


def h2_pt(estimand: str) -> tuple[float, float, float]:
    """:func:`kappa_pt` of an H2 stratum-2 parameter."""
    pw, g, arms = h2_cells()
    (q1, mom1, _), (q0, mom0, _) = arms[1], arms[0]
    return kappa_pt(pw, g, q1, q0, mom1, mom0, estimand)


def h2_truths() -> dict[str, float]:
    """H2's three stratum-2 truths."""
    return {
        name: h2_pt(short)[0] for name, short in zip(H2_NAMES, ("ey1", "ey0", "ate"), strict=True)
    }


def h2_cell_probabilities(v: int = 2) -> dict[tuple[int, int], float]:
    """``P(A = a, W = w, V = v)`` of each design cell of stratum ``v``."""
    out = {}
    for a in (0, 1):
        for w in range(3):
            g = float(h2_propensity(w, v))
            out[(a, w)] = float(H2_PV[v] * l1.P_W_GIVEN_V[v, w] * (g if a else 1.0 - g))
    return out


def h2_empty_cell(n: int) -> float:
    """P(at least one stratum-2 design cell is empty), cells independent (review 3, B-1)."""
    empty = np.array([(1.0 - p) ** n for p in h2_cell_probabilities().values()])
    return float(1.0 - np.prod(1.0 - empty))


def h2_const(n: int, arm: int) -> float:
    """P(arm ``arm`` of stratum 2 holds rows and every one is 0): its only constant value."""
    pw, g, arms = h2_cells()
    pi = arms[arm][2]
    share = g if arm == 1 else 1.0 - g
    q_a = float(H2_PV[2] * np.sum(pw * share))
    q_zero = float(H2_PV[2] * np.sum(pw * share * (1.0 - pi)))
    return float((1.0 - (q_a - q_zero)) ** n - (1.0 - q_a) ** n)


def h2_sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows ``(V, W, A, Y)`` of H2 from ``seed``."""
    rng = np.random.default_rng(seed)
    v = rng.choice(3, size=n, p=H2_PV)
    cumulative = np.cumsum(l1.P_W_GIVEN_V, axis=1)[v]
    w = (rng.uniform(size=n)[:, None] > cumulative).sum(axis=1)
    a = rng.binomial(1, h2_propensity(w, v))
    b = rng.binomial(1, h2_pi(a, w, v))
    g = rng.gamma(H2_SHAPE, h2_theta(a, w, v))
    return pd.DataFrame({"V": v, "W": w, "A": a, "Y": b * g})


class CellMeans(BaseEstimator, RegressorMixin):
    """The saturated least-squares regression: the mean of ``y`` in each declared cell.

    The design's columns are integer codes, and ``levels`` declares every value each column
    can take.  The fit refuses a sample in which a declared cell holds no row, instead of
    falling back to another value: the cell's prediction would then rest on nothing.

    Parameters
    ----------
    levels : sequence of sequence of int
        The values of each design column.  The cells are their product.
    """

    def __init__(self, levels: Sequence[Sequence[int]] = ((0, 1), (0, 1, 2), (0, 1, 2))) -> None:
        self.levels = levels

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> CellMeans:
        codes = np.rint(np.asarray(design, dtype=float)).astype(np.int64)
        y = np.asarray(target, dtype=float).reshape(-1)
        weight = np.ones(y.size) if sample_weight is None else np.asarray(sample_weight, float)
        self.means_: dict[tuple[int, ...], float] = {}
        for cell in _product(self.levels):
            rows = np.all(codes == np.asarray(cell), axis=1) & (weight > 0)
            if not rows.any():
                raise DataError(
                    f"{EMPTY_DESIGN_CELL}: the design cell {cell} holds no row, so the "
                    "saturated outcome regression has no value there"
                )
            self.means_[cell] = float(np.average(y[rows], weights=weight[rows]))
        return self

    def predict(self, design: Any) -> np.ndarray:
        codes = np.rint(np.asarray(design, dtype=float)).astype(np.int64)
        return np.array([self.means_[tuple(int(c) for c in row)] for row in codes])


def _product(levels: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    cells: list[tuple[int, ...]] = [()]
    for values in levels:
        cells = [(*cell, int(value)) for cell in cells for value in values]
    return cells


def h2_fit(frame: pd.DataFrame) -> Any:
    """Stratified in-sample TMLE with :class:`CellMeans` and a logistic treatment model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=CellMeans(),
                treatment_learner=logistic(),
                cross_fit=False,
                estimands=("ey", "ate"),
                simultaneous=False,
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
            .single()
        )


# ---------------------------------------------------------------------------- H3


H3_PV = np.array([0.45, 0.40, 0.15])
H3_NAMES = tuple(f"msm[{term}][V=2]" for term in alternating.MSM_TERMS)


def h3_g(w: Any, v: Any) -> Any:
    """H3's propensity."""
    return expit(-1.2 + 0.6 * np.asarray(w, float) - 0.5 * np.asarray(v, float))


def h3_q(a: Any, w: Any, v: Any) -> Any:
    """H3's outcome regression."""
    return expit(
        -0.4 + 0.9 * np.asarray(a, float) - 0.4 * np.asarray(w, float) + 0.5 * np.asarray(v, float)
    )


@functools.cache
def _h3_probs() -> tuple[float, ...]:
    support = np.asarray(alternating.SUPPORT, float)
    v, w, a, y = support.T
    p_vw = H3_PV[v.astype(int)] * l1.P_W_GIVEN_V[v.astype(int), w.astype(int)]
    g, q = h3_g(w, v), h3_q(a, w, v)
    return tuple(p_vw * np.where(a == 1, g, 1 - g) * np.where(y == 1, q, 1 - q))


def h3_probs() -> np.ndarray:
    """H3's 36 cell probabilities, ordered as L1's support."""
    return np.array(_h3_probs())


def h3_truths() -> dict[str, float]:
    """H3's three stratum-2 logit-MSM coefficients."""
    beta = alternating.msm_beta(h3_probs(), "logit", 2)
    return {name: float(np.real(value)) for name, value in zip(H3_NAMES, beta, strict=True)}


def h3_sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows ``(V, W, A, Y)`` of H3 from ``seed``."""
    rng = np.random.default_rng(seed)
    probs = h3_probs()
    cells = rng.choice(len(alternating.SUPPORT), size=n, p=probs / probs.sum())
    support = np.asarray(alternating.SUPPORT, dtype=np.int64)[cells]
    return pd.DataFrame(support, columns=["V", "W", "A", "Y"])


def h3_fit(frame: pd.DataFrame) -> Any:
    """The L1 logit-MSM fit of :func:`~tests.studies.stratified_incremental_msm_properties.fit_msm`."""
    from tests.studies.stratified_incremental_msm_properties import fit_msm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_msm(frame)


# ---------------------------------------------------------------- constant-cell shares


def constant_cell_share(law: str, n: int, estimand: str) -> float:
    """The exact probability that an arm the estimand reads is a constant cell.

    ``nan`` for a law with no closed form here.  An estimand that reads both arms takes
    :func:`either` of the two arms' probabilities.
    """
    arms = _arms(estimand)
    if law in ("boundary", "l1_logit"):
        values = {arm: l1_const(n, 2, arm) for arm in (0, 1)}
    elif law == "rare":
        values = {arm: rare_const(n, arm) for arm in (0, 1)}
    elif law == "h1":
        treated, untreated = h1_const(n)
        values = {1: treated, 0: untreated}
    elif law == "h2":
        values = {arm: h2_const(n, arm) for arm in (0, 1)}
    elif law == "h3":
        values = {arm: l1_const(n, 2, arm, h3_probs()) for arm in (0, 1)}
    else:
        return float("nan")
    if len(arms) == 1:
        return values[arms[0]]
    return either(values[0], values[1])


def _arms(estimand: str) -> tuple[int, ...]:
    if estimand.startswith("ey[1]"):
        return (1,)
    if estimand.startswith("ey[0]"):
        return (0,)
    return (0, 1)


def truths(law: str) -> Mapping[str, float]:
    """Every truth of a diagnosis or held-out law, by the package's estimand names."""
    if law in DIAGNOSIS_LAWS:
        return diagnosis_truths(law)
    return {"h1": h1_truths, "h2": h2_truths, "h3": h3_truths}[law]()
