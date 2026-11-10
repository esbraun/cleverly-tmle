r"""The law of the ``missingness-tilt-inference`` study, with its exact tilted truths.

Covariates :math:`W_1 \sim N(0, 1)` and :math:`W_2 \sim \mathrm{Bernoulli}(0.5)`, independent.
Every mechanism is a main-terms logistic model, and the outcome has one interaction:

.. math::

    \operatorname{logit} g_0(1 \mid W) &= 0.1 + 0.5 W_1 - 0.4 W_2, \\
    \operatorname{logit} \pi_0(a, W) &= 1.2 + 0.5 a + 0.25 W_1 - 0.4 W_2, \\
    \operatorname{logit} \bar Q_0(a, W) &= -1.6 + 0.8 a + 0.6 W_1 - 0.5 W_2 + 0.5 a W_1,

with :math:`\pi_0(a, W) = P(\Delta = 1 \mid A = a, W)` and :math:`\bar Q_0` the regression of
the binary :math:`Y` among the observed rows.  The outcome is missing at random under the
sampling law: the study's truths are not causal claims about an unobserved outcome but the
values of the declared tilted *functional* of the observed-data law, which is what the targeted
tilt estimates.  On the central 99.9% of :math:`W_1` the propensity lies in
:math:`[0.125, 0.851]` and the response probability in :math:`[0.49, 0.93]`
(``tests/unit/test_missingness_tilt_inference_design.py`` recomputes both).

The tilted functional of arm :math:`a` at :math:`\gamma` is

.. math::

    m_{a,\gamma}(W) = \pi_0(a, W)\bar Q_0(a, W) + \{1 - \pi_0(a, W)\}
        \operatorname{expit}\{\operatorname{logit}\bar Q_0(a, W) + \gamma\},
    \qquad \psi_\gamma(a) = E\, m_{a,\gamma}(W),

``ate`` is :math:`\psi_\gamma(1) - \psi_\gamma(0)`, ``rr`` is their ratio (read on the log
scale), ``att`` is :math:`E[g_0(1 \mid W)\{m_{1,\gamma} - m_{0,\gamma}\}] / E g_0(1 \mid W)`,
``ey_obs`` is :math:`E \sum_a g_0(a \mid W) m_{a,\gamma}(W)` and ``par`` is
``ey_obs - ey0``.  Every expectation is an 80-node probabilists' Gauss-Hermite rule in
:math:`W_1` and a two-point sum over :math:`W_2`, exact to rounding for these smooth
integrands.  The estimator never enters this module.

The weighted scenario's observation weight is :math:`\omega(W) = \exp(0.5 W_1)`, a function of
the covariate, so a weighted fit estimates the parameters of the weight-tilted covariate law:
every mean becomes :math:`E[\omega f] / E[\omega]` and ``att`` becomes
:math:`E[\omega g_0 (m_{1,\gamma} - m_{0,\gamma})] / E[\omega g_0]`.  The weighted truths sit
well away from the unweighted ones (``ey1`` at :math:`\gamma = 2` moves from 0.372 to 0.466), so a
fit that ignored its weights would be seen.

:func:`efficiency_sd` is the standard deviation of the efficient influence curve of the tilted
``ate``, from its closed form

.. math::

    D_\gamma = \sum_a \frac{(2a - 1) 1\{A = a\}}{g_0(a \mid W)}
        \Bigl[\frac{\Delta \kappa_a (Y - \bar Q_a)}{\pi_a} + (\bar Q_a - h_a)(\Delta - \pi_a)\Bigr]
        + m_{1,\gamma} - m_{0,\gamma} - \psi_\gamma,

with :math:`h_a = \operatorname{expit}(\operatorname{logit}\bar Q_a + \gamma)` and
:math:`\kappa_a = \pi_a + (1 - \pi_a) h_a(1 - h_a) / \{\bar Q_a(1 - \bar Q_a)\}`.  Its three
parts are orthogonal, so the variance is a quadrature of their conditional variances.

The outcome is rare (:math:`E\,\bar Q_0 \approx 0.23` among the observed) so that a positive
tilt moves the unobserved means toward one half, where :math:`h(1-h)` exceeds
:math:`\bar Q(1-\bar Q)`.  That makes the tilted efficient SD of ``ate`` at :math:`\gamma = 2`
(1.105) clearly larger than the MAR one (0.898), which the A1-era control needs: a MAR standard
error that understates the tilted spread.  With a common outcome the two nearly coincide (0.99
of each other at an intercept of -0.3), and the control would not separate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import expit, logit

#: Probabilists' Gauss-Hermite nodes for the expectation over :math:`W_1`.
_HERMITE = np.polynomial.hermite_e.hermegauss(80)

G_COEF = (0.1, 0.5, -0.4)
PI_COEF = (1.2, 0.5, 0.25, -0.4)
Q_COEF = (-1.6, 0.8, 0.6, -0.5, 0.5)
#: The observation weight of the weighted scenario, :math:`\omega(W) = \exp(0.5 W_1)`.
WEIGHT_SLOPE = 0.5


def weight(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """:math:`\\omega(W)`, the weighted scenario's observation weight."""
    del w2  # the weight reads W1 alone
    return np.asarray(np.exp(WEIGHT_SLOPE * np.asarray(w1, dtype=float)))


#: Every target the law evaluates.
TARGETS = ("ey1", "ey0", "ate", "att", "rr", "ey_obs", "par")


def propensity(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """:math:`g_0(1 \\mid W)`."""
    b0, b1, b2 = G_COEF
    return np.asarray(expit(b0 + b1 * w1 + b2 * w2))


def response(a: np.ndarray | float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """:math:`\\pi_0(a, W)`."""
    b0, ba, b1, b2 = PI_COEF
    return np.asarray(expit(b0 + ba * np.asarray(a, dtype=float) + b1 * w1 + b2 * w2))


def outcome(a: np.ndarray | float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """:math:`\\bar Q_0(a, W)`."""
    b0, ba, b1, b2, baw = Q_COEF
    arm = np.asarray(a, dtype=float)
    return np.asarray(expit(b0 + ba * arm + b1 * w1 + b2 * w2 + baw * arm * w1))


def sample(n: int, seed: int, *, weighted: bool = False) -> pd.DataFrame:
    """``n`` rows ``W1, W2, A, Delta, Y`` (and ``wt`` when weighted), ``Y`` NaN when unobserved.

    The weight ``wt`` is :func:`weight`, a function of ``W1``, so the weighted population tilts
    the covariate law and :func:`truths` with ``weighted=True`` gives its parameters.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, propensity(w1, w2)).astype(float)
    delta = rng.binomial(1, response(a, w1, w2))
    y = rng.binomial(1, outcome(a, w1, w2)).astype(float)
    frame = pd.DataFrame(
        {"W1": w1, "W2": w2, "A": a, "Delta": delta, "Y": np.where(delta == 1, y, np.nan)}
    )
    if weighted:
        frame["wt"] = weight(w1, w2)
    return frame


def _expect(integrand: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """:math:`E f(W_1, W_2)` by Gauss-Hermite in :math:`W_1` and a sum over :math:`W_2`."""
    nodes, weights = _HERMITE
    weights = weights / np.sqrt(2.0 * np.pi)
    return float(
        sum(0.5 * np.sum(weights * integrand(nodes, np.full_like(nodes, w2))) for w2 in (0.0, 1.0))
    )


def tilted_arm(a: int, gamma: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """:math:`m_{a,\\gamma}(W)`."""
    q = outcome(a, w1, w2)
    pi = response(a, w1, w2)
    return np.asarray(pi * q + (1.0 - pi) * expit(logit(q) + gamma))


def truths(gamma: float, *, weighted: bool = False) -> dict[str, float]:
    """Every target of :data:`TARGETS` at ``gamma``; ``rr`` is the ratio, not its log.

    With ``weighted=True`` every expectation is taken under the weight-tilted covariate law,
    :math:`E[\\omega f] / E[\\omega]`, and ``att`` under :math:`E[\\omega g_0 f] / E[\\omega g_0]`.
    """

    def mean(f: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
        if not weighted:
            return _expect(f)
        return _expect(lambda w1, w2: weight(w1, w2) * f(w1, w2)) / _expect(weight)

    def m1(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return tilted_arm(1, gamma, w1, w2)

    def m0(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        return tilted_arm(0, gamma, w1, w2)

    ey1 = mean(m1)
    ey0 = mean(m0)
    att = mean(lambda w1, w2: propensity(w1, w2) * (m1(w1, w2) - m0(w1, w2))) / mean(propensity)
    ey_obs = mean(
        lambda w1, w2: propensity(w1, w2) * m1(w1, w2) + (1.0 - propensity(w1, w2)) * m0(w1, w2)
    )
    return {
        "ey1": ey1,
        "ey0": ey0,
        "ate": ey1 - ey0,
        "att": att,
        "rr": ey1 / ey0,
        "ey_obs": ey_obs,
        "par": ey_obs - ey0,
    }


def _residual_variance(a: int, gamma: float, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """:math:`E[\\text{residual block}^2 \\mid A = a, W] / g_0(a \\mid W)`."""
    q = outcome(a, w1, w2)
    pi = response(a, w1, w2)
    h = expit(logit(q) + gamma)
    kappa = pi + (1.0 - pi) * h * (1.0 - h) / (q * (1.0 - q))
    g = propensity(w1, w2) if a == 1 else 1.0 - propensity(w1, w2)
    return np.asarray((kappa**2 * q * (1.0 - q) / pi + (q - h) ** 2 * pi * (1.0 - pi)) / g)


def efficiency_sd(gamma: float) -> float:
    """The SD of the efficient influence curve of the tilted ``ate`` at ``gamma``."""
    psi = truths(gamma)["ate"]
    residual = _expect(
        lambda w1, w2: _residual_variance(1, gamma, w1, w2) + _residual_variance(0, gamma, w1, w2)
    )
    spread = _expect(
        lambda w1, w2: (tilted_arm(1, gamma, w1, w2) - tilted_arm(0, gamma, w1, w2) - psi) ** 2
    )
    return float(np.sqrt(residual + spread))
