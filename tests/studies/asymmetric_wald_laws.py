r"""The held-out laws of ``asymmetric-wald`` (Fix-3 PR 3, plan rev 6, section 9.5).

Five laws, declared before any simulation on them.  **No estimator has been fitted on any of
them** before the declared run: every number here is a finite sum over a discrete support or an
80-node Gauss-Legendre or Gauss-Hermite quadrature over a continuous covariate, with closed-form
conditional moments (the planning script ``rev5_design.py`` and, for the width predictor,
``width_predict.py``).  ``tests/unit/test_asymmetric_wald_design.py`` recomputes every number
those scripts printed from the functions here.

=====  ======================================================================================
law    definition, ``n`` rows, fit
=====  ======================================================================================
D1     ``V`` in {0, 1, 2}, ``P(V) = (0.50, 0.30, 0.20)``; ``W | V`` in {0, 1, 2} with the rows
       of :data:`tests.studies.stratified_law.P_W_GIVEN_V`;
       ``A ~ Bernoulli(expit(-1.4 + 0.3 W - 0.6 V))``;
       ``Y ~ Bernoulli(expit(-0.9 - 1.3 A + 0.35 W - 0.2 V))``; n = 3,000.  Stratified TMLE
       in sample, unpenalized logistic outcome model on ``(A, W, V)`` and treatment model on
       ``(W, V)``, both correct, ``g_bounds = (0.01, 0.99)``
D2     ``W ~ N(0, 1)``; ``A ~ Bernoulli(expit(2.9 + 0.35 W))``;
       ``Y ~ Bernoulli(expit(0.6 + 2.0 (1 - A) + 0.5 W))``; n = 2,000.  Plain TMLE in sample,
       logistic outcome model on ``(A, W)`` and treatment model on ``W``, both correct, the
       default ``g_bounds``
C1     ``V`` in {0, 1, 2}, ``P(V) = (0.55, 0.33, 0.12)``; ``W ~ Uniform(-1, 1)``;
       ``A ~ Bernoulli(expit(-0.3 + 0.8 W - 0.3 V))``;
       ``Y = 0.5 + 0.8 A + 0.6 W + 0.4 V + s_A (G - 2)``, ``G ~ Gamma(2, 1)``, ``s_1 = 1.6``,
       ``s_0 = 1.0``; n = 2,500.  Stratified TMLE in sample, linear outcome model on
       ``(A, W, V)``, logistic treatment model on ``(W, V)``, ``g_bounds = (0.01, 0.99)``
C2     ``V`` in {0, 1, 2}, ``P(V) = (0.50, 0.35, 0.15)``; ``W | V`` in {0, 1, 2, 3} with the
       rows of :data:`C2_PW`; ``A ~ Bernoulli(expit(-0.4 + 0.5 W - 0.6 V))``;
       ``Y ~ Beta(mu phi, (1 - mu) phi)``, ``mu = 0.66 + 0.07 A + 0.045 W + 0.03 V``,
       ``phi = 5``; n = 2,000.  As C1's fit, ``W`` numeric
C3     as C1 with ``Y = 1 + 0.7 A + 0.5 W + 0.3 V + e``, ``e ~ Laplace(0, b_A)``,
       ``b_1 = 1.2``, ``b_0 = 0.8``, ``P(V) = (0.60, 0.28, 0.12)``; n = 2,500
=====  ======================================================================================

The design quantities of a point-treatment curve over the rows that carry it (a stratum's rows,
or every row) are its raw moments ``E[u^k]``, ``k = 0..6`` (:func:`curve_moments`), from which
the population moment df ``nu_pop = 2 m / (kappa - 1)``, the skewness, the acceleration
``a_pop = gamma / (6 sqrt(m))`` and the width predictor of plan 9.6 (:func:`width_prediction`)
follow, and the share of the variance each arm's residual term carries
(:func:`variance_shares`).  The constant-cell probabilities are exact (:func:`const_prob`).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, gammaln
from sklearn.linear_model import LinearRegression

from cleverly.estimators import TMLE
from tests.studies import small_sample_laws as diagnosis
from tests.studies import stratified_law as l1
from tests.studies.small_sample_laws import const_prob, either, logistic

#: The held-out laws, in declaration order.
HELDOUT_LAWS = ("d1", "d2", "c1", "c2", "c3")

Z = float(stats.norm.ppf(0.975))
#: ``2 z^2 + 1`` at the 0.95 level.
K = 2.0 * Z * Z + 1.0

GL_X, _GL_W = np.polynomial.legendre.leggauss(80)
#: Gauss-Legendre weights of ``W ~ Uniform(-1, 1)``, normalised to 1.
GL_W = _GL_W / _GL_W.sum()
GH_X, GH_W = diagnosis.GH_X, diagnosis.GH_W


# ------------------------------------------------------------------------------ moments


def central_from_raw(raw: Sequence[Any]) -> list[Any]:
    """Central moments ``0..6`` from raw moments ``0..6`` (``raw[0] = 1``)."""
    mu = raw[1]
    return [sum(comb(k, j) * raw[j] * (-mu) ** (k - j) for j in range(k + 1)) for k in range(7)]


def binary_central(q: Any) -> list[Any]:
    """Central moments ``0..6`` of ``Y - q``, ``Y ~ Bernoulli(q)``."""
    q = np.asarray(q, dtype=float)
    return central_from_raw([np.ones_like(q)] + [q] * 6)


def gamma_central(shape: float, scale: float) -> list[float]:
    """Central moments ``0..6`` of ``scale * G``, ``G ~ Gamma(shape, 1)``."""
    raw = [float(np.exp(gammaln(shape + j) - gammaln(shape))) * scale**j for j in range(7)]
    return [float(x) for x in central_from_raw(raw)]


def laplace_central(b: float) -> list[float]:
    """Central moments ``0..6`` of ``Laplace(0, b)``."""
    return [1.0, 0.0, 2.0 * b**2, 0.0, 24.0 * b**4, 0.0, 720.0 * b**6]


def beta_central(mu: Any, phi: float) -> list[Any]:
    """Central moments ``0..6`` of ``Beta(mu phi, (1 - mu) phi)``."""
    a, b = np.asarray(mu, float) * phi, (1.0 - np.asarray(mu, float)) * phi
    raw = [
        np.exp(gammaln(a + j) - gammaln(a) + gammaln(a + b) - gammaln(a + b + j)) for j in range(7)
    ]
    return central_from_raw(raw)


def lognormal_central(s: float) -> list[float]:
    """Central moments ``0..6`` of ``exp(N(0, s^2))``."""
    return [float(x) for x in central_from_raw([np.exp(j * j * s * s / 2) for j in range(7)])]


def hurdle_central(pi: Any, theta: Any, shape: float = diagnosis.H2_SHAPE) -> list[Any]:
    """Central moments ``0..6`` of ``B G``, ``B ~ Bernoulli(pi)``, ``G ~ Gamma(shape, theta)``."""
    pi = np.asarray(pi, float)
    raw = [np.ones_like(pi)] + [
        pi * np.asarray(theta, float) ** j * np.exp(gammaln(shape + j) - gammaln(shape))
        for j in range(1, 7)
    ]
    return central_from_raw(raw)


def _broadcast(central: Sequence[Any], shape: tuple[int, ...]) -> list[np.ndarray]:
    return [np.broadcast_to(np.asarray(x, float), shape) for x in central]


def _terms(
    pw: Any, g: Any, q1: Any, q0: Any, estimand: str
) -> tuple[np.ndarray, float, np.ndarray, list[tuple[Any, Any, bool]]]:
    pw = np.asarray(pw, float)
    pw = pw / pw.sum()
    g, q1, q0 = (np.asarray(x, float) for x in (g, q1, q0))
    if estimand == "ey1":
        psi = float(np.sum(pw * q1))
        terms = [(g, 1.0 / g, True), (1.0 - g, 0.0 * g, False)]
        centre = q1 - psi
    elif estimand == "ey0":
        psi = float(np.sum(pw * q0))
        terms = [(g, 0.0 * g, False), (1.0 - g, 1.0 / (1.0 - g), True)]
        centre = q0 - psi
    elif estimand == "ate":
        psi = float(np.sum(pw * (q1 - q0)))
        terms = [(g, 1.0 / g, True), (1.0 - g, -1.0 / (1.0 - g), True)]
        centre = q1 - q0 - psi
    else:
        raise KeyError(estimand)
    return pw, psi, centre, terms


def curve_moments(
    pw: Any,
    g: Any,
    q1: Any,
    q0: Any,
    central1: Sequence[Any],
    central0: Sequence[Any],
    estimand: str,
) -> tuple[float, np.ndarray]:
    """The truth and ``E[u^k]``, ``k = 0..6``, of a point-treatment curve over its carrying rows.

    The curve of ``ey1`` is ``A / g (Y - Q1) + Q1 - psi``, of ``ey0`` its mirror image, and of
    ``ate`` their difference.  ``pw`` holds the covariate weights given the carrying rows,
    ``central1`` and ``central0`` the central moments ``0..6`` of ``Y - Q`` in each arm.
    """
    pw, psi, centre, terms = _terms(pw, g, q1, q0, estimand)
    moments = np.zeros(7)
    centrals = (_broadcast(central1, centre.shape), _broadcast(central0, centre.shape))
    for k in range(7):
        total = np.zeros_like(centre)
        for (share, slope, reads), central in zip(terms, centrals, strict=True):
            if not reads:
                total = total + share * centre**k
            else:
                total = total + share * sum(
                    comb(k, j) * centre ** (k - j) * slope**j * central[j] for j in range(k + 1)
                )
        moments[k] = float(np.sum(pw * total))
    return psi, moments


def variance_shares(
    pw: Any,
    g: Any,
    q1: Any,
    q0: Any,
    central1: Sequence[Any],
    central0: Sequence[Any],
    estimand: str,
) -> dict[str, float]:
    """The share of the curve's variance carried by ``Q`` (``c``) and each arm's residual term."""
    pw, _, centre, terms = _terms(pw, g, q1, q0, estimand)
    parts = {"c": float(np.sum(pw * centre**2))}
    for name, (share, slope, _), central in zip(
        ("1", "0"),
        terms,
        (_broadcast(central1, centre.shape), _broadcast(central0, centre.shape)),
        strict=True,
    ):
        parts[name] = float(np.sum(pw * share * slope**2 * central[2]))
    total = sum(parts.values())
    return {name: value / total for name, value in parts.items()}


@dataclass(frozen=True)
class Design:
    """The design quantities of one estimand of one law.

    Parameters
    ----------
    truth : float
        The parameter.
    moments : ndarray
        ``E[u^k]``, ``k = 0..6``, of the curve over its carrying rows.
    m : float
        The expected number of carrying rows.
    shares : dict
        :func:`variance_shares`.
    """

    truth: float
    moments: np.ndarray
    m: float
    shares: Mapping[str, float]

    @property
    def kappa(self) -> float:
        """The curve's kurtosis over its carrying rows."""
        return float(self.moments[4] / self.moments[2] ** 2)

    @property
    def skewness(self) -> float:
        """The curve's skewness over its carrying rows."""
        return float(self.moments[3] / self.moments[2] ** 1.5)

    @property
    def nu_pop(self) -> float:
        """``2 m / (kappa - 1)``."""
        return diagnosis.nu_pop(self.kappa, self.m)

    @property
    def a_pop(self) -> float:
        """``gamma / (6 sqrt(m))``, the acceleration of the population curve."""
        return self.skewness / (6.0 * float(np.sqrt(self.m)))

    @property
    def efficiency_se(self) -> float:
        """``sqrt(E[u^2] / m)``."""
        return float(np.sqrt(self.moments[2] / self.m))

    def distance(self, bounds: tuple[float, float]) -> float:
        """The distance to the nearer bound, in efficiency SEs."""
        return min(self.truth - bounds[0], bounds[1] - self.truth) / self.efficiency_se


def width_prediction(moments: np.ndarray, m: float) -> dict[str, float]:
    """The width predictor of plan 9.6 (``width_predict.py``), from ``E[u^k]`` over ``m`` rows.

    ``a_hat = g / (6 sqrt(m))`` with the delta-method variance of the sample skewness ``g`` of
    ``m`` terms; under a normal approximation to ``a_hat``, the skew-widened interval's mean
    width over the capped t's is ``1 + K E|a_hat| / (2 t)``, ``t = t_{0.975, max(nu_pop, 5)}``.
    A prediction, not a law: it errs high on a law with finite moments and fails on a law with a
    huge sixth moment (plan 9.6).
    """
    mu2, mu3, mu4 = moments[2], moments[3], moments[4]
    cov = np.array(
        [[moments[i + j] - moments[i] * moments[j] for j in (1, 2, 3)] for i in (1, 2, 3)]
    )
    grad = np.array([-3.0 * mu2 / mu2**1.5, -1.5 * mu3 / mu2**2.5, 1.0 / mu2**1.5])
    var_g = float(grad @ cov @ grad) / m
    a_pop = float(mu3 / mu2**1.5 / (6.0 * np.sqrt(m)))
    sd = float(np.sqrt(var_g) / (6.0 * np.sqrt(m)))
    e_abs = sd * float(np.sqrt(2.0 / np.pi)) * float(
        np.exp(-(a_pop**2) / (2.0 * sd**2))
    ) + a_pop * (1.0 - 2.0 * float(stats.norm.cdf(-a_pop / sd)))
    nu = diagnosis.nu_pop(float(mu4 / mu2**2), m)
    t = float(stats.t.ppf(0.975, max(nu, 5.0)))
    return {
        "a_pop": a_pop,
        "sd": sd,
        "e_abs": e_abs,
        "nu_pop": nu,
        "ratio": 1.0 + K * e_abs / (2.0 * t),
    }


# ------------------------------------------------------------------------------ the laws


@dataclass(frozen=True)
class Law:
    """One held-out law's constants.

    Parameters
    ----------
    name : str
        ``d1``, ``d2``, ``c1``, ``c2`` or ``c3``.
    n : int
        Rows per draw.
    p_v : tuple of float or None
        ``P(V)``, or ``None`` for an unstratified law.
    family : str
        The declared outcome family.
    """

    name: str
    n: int
    p_v: tuple[float, ...] | None
    family: str


LAWS: dict[str, Law] = {
    "d1": Law("d1", 3_000, (0.50, 0.30, 0.20), "binomial"),
    "d2": Law("d2", 2_000, None, "binomial"),
    "c1": Law("c1", 2_500, (0.55, 0.33, 0.12), "gaussian"),
    "c2": Law("c2", 2_000, (0.50, 0.35, 0.15), "gaussian"),
    "c3": Law("c3", 2_500, (0.60, 0.28, 0.12), "gaussian"),
}

#: ``P(W | V)`` of C2, ``W`` in {0, 1, 2, 3}.
C2_PW = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]])
C2_PHI = 5.0
C1_SCALE = {1: 1.6, 0: 1.0}
C1_SHAPE = 2.0
C3_SCALE = {1: 1.2, 0: 0.8}


def d1_g(w: Any, v: Any) -> Any:
    """D1's propensity."""
    return expit(-1.4 + 0.3 * np.asarray(w, float) - 0.6 * np.asarray(v, float))


def d1_q(a: Any, w: Any, v: Any) -> Any:
    """D1's outcome regression."""
    return expit(
        -0.9 - 1.3 * np.asarray(a, float) + 0.35 * np.asarray(w, float) - 0.2 * np.asarray(v, float)
    )


def d2_g(w: Any) -> Any:
    """D2's propensity."""
    return expit(2.9 + 0.35 * np.asarray(w, float))


def d2_q(a: Any, w: Any) -> Any:
    """D2's outcome regression."""
    return expit(0.6 + 2.0 * (1.0 - np.asarray(a, float)) + 0.5 * np.asarray(w, float))


def c_g(w: Any, v: Any) -> Any:
    """C1's and C3's propensity."""
    return expit(-0.3 + 0.8 * np.asarray(w, float) - 0.3 * np.asarray(v, float))


def c1_q(a: Any, w: Any, v: Any) -> Any:
    """C1's outcome regression."""
    return (
        0.5 + 0.8 * np.asarray(a, float) + 0.6 * np.asarray(w, float) + 0.4 * np.asarray(v, float)
    )


def c2_g(w: Any, v: Any) -> Any:
    """C2's propensity."""
    return expit(-0.4 + 0.5 * np.asarray(w, float) - 0.6 * np.asarray(v, float))


def c2_mu(a: Any, w: Any, v: Any) -> Any:
    """C2's outcome mean."""
    return (
        0.66
        + 0.07 * np.asarray(a, float)
        + 0.045 * np.asarray(w, float)
        + 0.03 * np.asarray(v, float)
    )


def c3_q(a: Any, w: Any, v: Any) -> Any:
    """C3's outcome regression."""
    return (
        1.0 + 0.7 * np.asarray(a, float) + 0.5 * np.asarray(w, float) + 0.3 * np.asarray(v, float)
    )


def _discrete_w(rng: np.random.Generator, rows: np.ndarray, v: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(rows, axis=1)[v]
    return np.asarray((rng.uniform(size=v.size)[:, None] > cumulative).sum(axis=1))


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of a held-out law from ``seed``."""
    rng = np.random.default_rng(seed)
    spec = LAWS[law]
    w: Any
    y: Any
    if law == "d2":
        w = rng.normal(size=n)
        a = rng.binomial(1, d2_g(w))
        y = rng.binomial(1, d2_q(a, w))
        return pd.DataFrame({"W": w, "A": a, "Y": y})
    assert spec.p_v is not None
    v = rng.choice(3, size=n, p=np.asarray(spec.p_v))
    if law == "d1":
        w = _discrete_w(rng, l1.P_W_GIVEN_V, v)
        a = rng.binomial(1, d1_g(w, v))
        y = rng.binomial(1, d1_q(a, w, v))
    elif law == "c1":
        w = rng.uniform(-1.0, 1.0, size=n)
        a = rng.binomial(1, c_g(w, v))
        scale = np.where(a == 1, C1_SCALE[1], C1_SCALE[0])
        y = c1_q(a, w, v) + scale * (rng.gamma(C1_SHAPE, 1.0, size=n) - C1_SHAPE)
    elif law == "c2":
        w = _discrete_w(rng, C2_PW, v)
        a = rng.binomial(1, c2_g(w, v))
        mu = c2_mu(a, w, v)
        y = rng.beta(mu * C2_PHI, (1.0 - mu) * C2_PHI)
    elif law == "c3":
        w = rng.uniform(-1.0, 1.0, size=n)
        a = rng.binomial(1, c_g(w, v))
        scale = np.where(a == 1, C3_SCALE[1], C3_SCALE[0])
        y = c3_q(a, w, v) + rng.laplace(0.0, scale, size=n)
    else:
        raise KeyError(law)
    return pd.DataFrame({"V": v, "W": w, "A": a, "Y": y})


# ------------------------------------------------------------------------- the design


def _cells(law: str, v: int) -> tuple[Any, Any, Any, Any, Sequence[Any], Sequence[Any]]:
    """``P(W | carrying rows)``, ``g``, ``Q1``, ``Q0`` and the residual central moments."""
    if law == "d1":
        w = np.arange(3.0)
        q1, q0 = d1_q(1, w, v), d1_q(0, w, v)
        return l1.P_W_GIVEN_V[v], d1_g(w, v), q1, q0, binary_central(q1), binary_central(q0)
    if law == "d2":
        q1, q0 = d2_q(1, GH_X), d2_q(0, GH_X)
        return GH_W, d2_g(GH_X), q1, q0, binary_central(q1), binary_central(q0)
    if law == "c1":
        return (
            GL_W, c_g(GL_X, v), c1_q(1, GL_X, v), c1_q(0, GL_X, v),
            gamma_central(C1_SHAPE, C1_SCALE[1]), gamma_central(C1_SHAPE, C1_SCALE[0]),
        )  # fmt: skip
    if law == "c2":
        w = np.arange(4.0)
        q1, q0 = c2_mu(1, w, v), c2_mu(0, w, v)
        return C2_PW[v], c2_g(w, v), q1, q0, beta_central(q1, C2_PHI), beta_central(q0, C2_PHI)
    if law == "c3":
        return (
            GL_W, c_g(GL_X, v), c3_q(1, GL_X, v), c3_q(0, GL_X, v),
            laplace_central(C3_SCALE[1]), laplace_central(C3_SCALE[0]),
        )  # fmt: skip
    raise KeyError(law)


def stratum_of(estimand: str) -> int | None:
    """The stratum an estimand reads, or ``None`` for every row."""
    if "[V=" not in estimand:
        return None
    return int(estimand.split("[V=", 1)[1].split("]", 1)[0])


def short_name(estimand: str) -> str:
    """``ey1``, ``ey0`` or ``ate`` of a package estimand name."""
    if estimand.startswith("ey[1]"):
        return "ey1"
    if estimand.startswith("ey[0]"):
        return "ey0"
    if estimand.startswith("ate"):
        return "ate"
    raise KeyError(estimand)


@functools.cache
def design(law: str, estimand: str) -> Design:
    """The design quantities of one estimand of a held-out law, by quadrature or finite sums."""
    spec = LAWS[law]
    v = stratum_of(estimand)
    stratum = 0 if v is None else v
    pw, g, q1, q0, central1, central0 = _cells(law, stratum)
    short = short_name(estimand)
    truth, moments = curve_moments(pw, g, q1, q0, central1, central0, short)
    shares = variance_shares(pw, g, q1, q0, central1, central0, short)
    m = float(spec.n if v is None else spec.n * (spec.p_v or (1.0,))[v])
    return Design(truth, moments, m, shares)


#: The estimands of each held-out law, by the package's names.
ESTIMANDS: dict[str, tuple[str, ...]] = {
    "d1": ("ey[1][V=2]", "ey[0][V=2]", "ate[V=2]", "ey[1][V=0]", "ey[0][V=0]", "ate[V=0]"),
    "d2": ("ey[1]", "ey[0]", "ate"),
    "c1": ("ey[1][V=2]", "ey[0][V=2]", "ate[V=2]"),
    "c2": ("ey[1][V=2]", "ey[0][V=2]", "ate[V=2]"),
    "c3": ("ey[1][V=2]", "ey[0][V=2]", "ate[V=2]"),
}


def truths(law: str) -> dict[str, float]:
    """Every truth of a held-out law, exact."""
    return {name: design(law, name).truth for name in ESTIMANDS[law]}


def arm_counts(law: str, v: int | None) -> dict[str, float]:
    """``P(carrying, A = a)`` and ``P(carrying, A = a, Y = 1)`` of a binary law, per arm."""
    spec = LAWS[law]
    pw, g, q1, q0, _, _ = _cells(law, 0 if v is None else v)
    pw = np.asarray(pw, float) / np.sum(pw)
    share = 1.0 if v is None else (spec.p_v or (1.0,))[v]
    return {
        "q_1": float(share * np.sum(pw * g)),
        "q_11": float(share * np.sum(pw * g * q1)),
        "q_0": float(share * np.sum(pw * (1 - g))),
        "q_01": float(share * np.sum(pw * (1 - g) * q0)),
    }


def constant_cell(law: str, estimand: str) -> float:
    """The exact probability that an arm the estimand reads is a constant cell (withheld)."""
    spec = LAWS[law]
    if spec.family != "binomial":
        return 0.0
    counts = arm_counts(law, stratum_of(estimand))
    one = const_prob(spec.n, counts["q_1"], counts["q_11"])
    zero = const_prob(spec.n, counts["q_0"], counts["q_01"])
    short = short_name(estimand)
    return {"ey1": one, "ey0": zero, "ate": either(one, zero)}[short]


def d2_default_bound_tail() -> float:
    """P(``1 - g`` falls below D2's default propensity bound ``5 / (sqrt(n) log n)``)."""
    n = LAWS["d2"].n
    bound = 5.0 / (np.sqrt(n) * np.log(n))
    return float(stats.norm.sf((np.log((1 - bound) / bound) - 2.9) / 0.35))


# ------------------------------------------------------------- diagnosis-law references


def lognormal06_design(estimand: str) -> Design:
    """``lognormal06``'s stratum-2 curve, for scale (a diagnosis law; plan 9.5)."""
    v, s = 2, 0.6
    g = expit(-0.2 + 0.6 * GH_X - 0.3 * v)
    q0 = 1.0 + 0.5 * GH_X + 0.3 * v + float(np.exp(s * s / 2))
    central = lognormal_central(s)
    short = short_name(estimand)
    truth, moments = curve_moments(GH_W, g, q0 + 1.0, q0, central, central, short)
    return Design(
        truth, moments, 200.0, variance_shares(GH_W, g, q0 + 1.0, q0, central, central, short)
    )


def h1_design(estimand: str) -> Design:
    """H1's curve (``ho_sparse_treat``, a diagnosis law since study 1's run)."""
    pw, g, q1, q0 = diagnosis.h1_cells()
    short = short_name(estimand)
    c1, c0 = binary_central(q1), binary_central(q0)
    truth, moments = curve_moments(pw, g, q1, q0, c1, c0, short)
    return Design(truth, moments, 2_000.0, variance_shares(pw, g, q1, q0, c1, c0, short))


def h2_design(estimand: str) -> Design:
    """H2's stratum-2 curve (``ho_hurdle``, a diagnosis law since study 1's run)."""
    w, v = np.arange(3.0), 2
    pw, g = l1.P_W_GIVEN_V[v], diagnosis.h2_propensity(w, v)
    pis = {a: diagnosis.h2_pi(a, w, v) for a in (0, 1)}
    thetas = {a: diagnosis.h2_theta(a, w, v) for a in (0, 1)}
    qs = {a: diagnosis.H2_SHAPE * pis[a] * thetas[a] for a in (0, 1)}
    c1, c0 = hurdle_central(pis[1], thetas[1]), hurdle_central(pis[0], thetas[0])
    short = short_name(estimand)
    truth, moments = curve_moments(pw, g, qs[1], qs[0], c1, c0, short)
    m = 2_000.0 * float(diagnosis.H2_PV[v])
    return Design(truth, moments, m, variance_shares(pw, g, qs[1], qs[0], c1, c0, short))


# ------------------------------------------------------------------------------ the fits


def _stratified(frame: pd.DataFrame, outcome_learner: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=outcome_learner,
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


def fit(law: str, frame: pd.DataFrame) -> Any:
    """The declared fit of a held-out law (plan 9.5): correct parametric nuisance models."""
    if law == "d1":
        return _stratified(frame, logistic())
    if law in ("c1", "c2", "c3"):
        return _stratified(frame, LinearRegression())
    if law == "d2":
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
    raise KeyError(law)
