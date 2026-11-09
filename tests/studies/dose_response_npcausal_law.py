r"""The law of the ``dose-response-npcausal`` comparator study (X93 S2), with exact truths.

Covariates :math:`W \sim N(0, I_3)`.  The dose is a normal location model, linear in
:math:`W` so that R ``SuperLearner``'s ``SL.glm.interaction`` holds its mean and its constant
variance exactly,

.. math::

    A = m(W) + \sigma\varepsilon, \qquad m(W) = 2.5 + 0.5 W_1 + 0.3 W_2 - 0.3 W_3,
    \qquad \sigma = 1.5,

with :math:`\varepsilon \sim N(0, 1)`.  The Gaussian outcome reads :math:`W` through the index
:math:`U = 0.7 W_1 - 0.5 W_3` of the S1 law, with a dose-varying slope:

.. math::

    Y = \beta_0 + \beta_1 A + c(A) U + \eta, \qquad c(a) = 0.8 + 0.1 a, \qquad
    \eta \sim N(0, 1).

:math:`W_1` confounds, since it moves both the dose and the outcome.  :math:`E U = 0`, so the
curve is the line :math:`\theta(a) = \beta_0 + \beta_1 a` and every local-linear smoother
reproduces it: the Lebesgue-measure equivalent kernel of ``DoseResponse`` and npcausal
``ctseff``'s kernel regression against the marginal dose density both return
:math:`\theta(a_0)` at every dose and bandwidth.  That is what makes the two targets one
target at this law (the plan's "comparison of target recovery, not of the smoothing
measure").  The outcome regression :math:`\bar Q_0(a, W) = \beta_0 + \beta_1 a + c(a)U` lies in
the span of the main terms and pairwise products of :math:`(a, W)`, the design both
implementations fit.

The dose mean :math:`m(W)` is normal with variance :math:`s^2 = 0.43`, so the inverse density
has the closed-form expectation of :func:`inverse_density_mean`, finite because
:math:`s^2 < \sigma^2`.  The implementation never enters this module.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from cleverly.dose_response import KernelName, equivalent_kernel

SIGMA = 1.5
SUPPORT = (0.0, 5.0)
DOSE_INTERCEPT = 2.5
DOSE_LOADINGS = (0.5, 0.3, -0.3)
#: :math:`s^2 = \operatorname{Var} m(W)`.
DOSE_MEAN_VARIANCE = float(sum(b * b for b in DOSE_LOADINGS))
U_LOADINGS = (0.7, 0.0, -0.5)
U_VARIANCE = float(sum(b * b for b in U_LOADINGS))
BETA = (0.2, 0.5)
SLOPE = (0.8, 0.1)
NOISE_SD = 1.0

#: Gauss-Legendre nodes on each smooth piece of a dose integral.
_LEGENDRE = np.polynomial.legendre.leggauss(64)


def dose_mean(w: np.ndarray) -> np.ndarray:
    """:math:`m(W)`, row by row."""
    return DOSE_INTERCEPT + w @ np.asarray(DOSE_LOADINGS)


def index(w: np.ndarray) -> np.ndarray:
    """:math:`U = 0.7 W_1 - 0.5 W_3`."""
    return w @ np.asarray(U_LOADINGS)


def c(a: np.ndarray | float) -> np.ndarray:
    return SLOPE[0] + SLOPE[1] * np.asarray(a, dtype=float)


def curve(a: np.ndarray | float) -> np.ndarray:
    """:math:`\\theta(a) = \\beta_0 + \\beta_1 a`, exact."""
    return BETA[0] + BETA[1] * np.asarray(a, dtype=float)


def q_bar(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """:math:`\\bar Q_0(a, W)`."""
    return curve(a) + c(a) * index(w)


def density(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """:math:`g_0(a \\mid W)`."""
    z = (np.asarray(a, dtype=float) - dose_mean(w)) / SIGMA
    return np.exp(-0.5 * z * z) / (SIGMA * np.sqrt(2.0 * np.pi))


def inverse_density_mean(a: np.ndarray | float) -> np.ndarray:
    r""":math:`E_W\{1 / g_0(a \mid W)\}`, in closed form.

    With :math:`a - m(W) \sim N(a - 2.5, s^2)` and :math:`t = 1/(2\sigma^2)`,
    :math:`E e^{t(a - m)^2} = (1 - 2ts^2)^{-1/2}\exp\{t(a - 2.5)^2/(1 - 2ts^2)\}`.
    """
    a = np.asarray(a, dtype=float)
    t = 1.0 / (2.0 * SIGMA * SIGMA)
    shrink = 1.0 - 2.0 * t * DOSE_MEAN_VARIANCE
    return np.asarray(
        SIGMA
        * np.sqrt(2.0 * np.pi)
        * np.exp(t * (a - DOSE_INTERCEPT) ** 2 / shrink)
        / np.sqrt(shrink)
    )


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows with columns ``W1, W2, W3, A, Y``."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    a = dose_mean(w) + SIGMA * rng.standard_normal(n)
    y = q_bar(a, w) + NOISE_SD * rng.standard_normal(n)
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2], "A": a, "Y": y})


def _pieces(*pairs: tuple[float, float]) -> np.ndarray:
    lower, upper = SUPPORT
    breaks = {lower, upper}
    for h, a0 in pairs:
        for edge in (a0 - h, a0 + h):
            if lower < edge < upper:
                breaks.add(edge)
    return np.array(sorted(breaks))


def dose_rule(*pairs: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre doses and weights on the pieces of :math:`D` that every pair's weight sees."""
    nodes, weights = _LEGENDRE
    points, masses = [], []
    for left, right in itertools.pairwise(_pieces(*pairs)):
        half = 0.5 * (right - left)
        points.append(left + half * (nodes + 1.0))
        masses.append(half * weights)
    return np.concatenate(points), np.concatenate(masses)


def smoothed(h: float, a0: float, kernel: KernelName) -> float:
    """:math:`\\theta_h(a_0) = \\int_D w_{h,a_0}\\theta`, by the composite rule (a check of linearity)."""
    a, mass = dose_rule((h, a0))
    return float(np.sum(mass * equivalent_kernel(a, a0, h, SUPPORT, kernel) * curve(a)))


def efficiency_covariance(pairs: tuple[tuple[float, float], ...], kernel: KernelName) -> np.ndarray:
    r"""The covariance of the efficient influence curves of the ``DoseResponse`` targets.

    .. math::

        \operatorname{Cov}(D_j, D_k) = \int_D w_j w_k\, E_W\{\sigma^2 / g_0(a \mid W)\}\,da
        + \operatorname{Cov}_W\Bigl(\int_D w_j \bar Q_0, \int_D w_k \bar Q_0\Bigr).

    The residual variance is :math:`\sigma^2 = 1`.  :math:`\int_D w_j \bar Q_0(\cdot, W) =
    \theta(a_j) + U\int_D w_j c`, so the second term is :math:`U`'s variance times the two
    integrals of :math:`c`.  Both are one-dimensional rules, exact to rounding.
    """
    a, mass = dose_rule(*pairs)
    weights = np.stack([equivalent_kernel(a, a0, h, SUPPORT, kernel) for h, a0 in pairs])
    residual = (weights * (NOISE_SD**2 * inverse_density_mean(a) * mass)[None, :]) @ weights.T
    slopes = weights @ (c(a) * mass)
    return np.asarray(residual + U_VARIANCE * np.outer(slopes, slopes), dtype=float)


def efficiency_sd(h: float, a0: float, kernel: KernelName) -> float:
    """:math:`\\operatorname{sd} D_{h,a_0}(P_0)(O)` (:func:`efficiency_covariance` of one pair)."""
    return float(np.sqrt(efficiency_covariance(((h, a0),), kernel)[0, 0]))
