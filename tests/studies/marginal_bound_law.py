r"""The laws of the X82 SH cells, and the sharp marginal sensitivity model set of each.

Both cells draw ``W`` as 20 independent standard normal columns. ``W1`` to ``W4`` carry the
signal. ``W5`` to ``W20`` are declared weak confounders with coefficient ``0.05`` and
alternating sign in both indices (parent plan section 8.2 pads to 20 covariates).

====  ===================================================  ==================================
cell  outcome                                              fit
====  ===================================================  ==================================
SH1   ``Y = 0.5 + A + eta_Y(W) + eps``, ``eps ~ N(0, 1)``  linear outcome GLM
SH2   ``Y ~ Bernoulli(expit(-1.3 + 0.8 A + eta_Y(W)))``    logistic outcome GLM
====  ===================================================  ==================================

The propensity is ``expit(eta_A(W))`` for both cells. ``eta_A`` and ``eta_Y`` are linear in
``W``, so every fitted GLM is correctly specified. So are the conditional quantile and the tail
regression of the SH1 bound: both are ``eta_Y`` plus a constant.

The truth is the sharp set of the ATE at ``Gamma`` from the observed law. Each arm bound is
:math:`E[1\{A = a\} Y + 1\{A \ne a\} \rho(W)]`, with :math:`\rho` the sharp conditional bound:

* Gaussian: :math:`\rho^\pm = m \pm \sigma k`, with
  :math:`k = (1 - 1/\Gamma)\,\varphi(z_\tau) / (1 - \tau)` and :math:`\tau = \Gamma / (1 + \Gamma)`;
* binary: :math:`\rho^+ = \min(\Gamma p, 1 - (1 - p)/\Gamma)` and
  :math:`\rho^- = \max(1 - \Gamma (1 - p), p / \Gamma)`.

``(eta_A, eta_Y)`` is bivariate normal, so each expectation is a two-dimensional integral. A
tensor Gauss-Legendre rule on ``[-9, 9]`` in each standard normal axis evaluates it. The binary
bound has a kink, so the rule is not spectral there. :func:`sharp_set` takes the node count, and
the design test checks that doubling it moves no endpoint by more than ``1e-6``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

CELLS = ("SH1", "SH2")
COVARIATES = tuple(f"W{j}" for j in range(1, 21))

#: The weak-confounder coefficient, with alternating sign on ``W5`` to ``W20``.
WEAK = 0.05

_WEAK_SIGNS = np.array([1.0 if j % 2 else -1.0 for j in range(16)])

#: ``eta_A(W) = A_COEF . W``.
A_COEF = np.concatenate([[0.6, -0.4, 0.3, 0.0], WEAK * _WEAK_SIGNS])

#: ``eta_Y(W) = Y_COEF . W``.
Y_COEF = np.concatenate([[0.8, 0.5, 0.0, -0.4], WEAK * _WEAK_SIGNS])

#: SH1: intercept, treatment effect and noise SD.
GAUSSIAN_INTERCEPT = 0.5
GAUSSIAN_EFFECT = 1.0
GAUSSIAN_SD = 1.0

#: SH2: intercept and treatment log odds.
BINARY_INTERCEPT = -1.3
BINARY_EFFECT = 0.8

#: The tensor Gauss-Legendre node count of :func:`sharp_set`, on each standard normal axis.
NODES = 1_000

#: Each standard normal axis is integrated on ``[-HALF_RANGE, HALF_RANGE]``.
HALF_RANGE = 9.0


def sample(cell: str, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows of ``cell`` from ``seed``."""
    if cell not in CELLS:
        raise KeyError(f"unknown cell {cell!r}; choose from {CELLS}")
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, len(COVARIATES)))
    a = rng.binomial(1, expit(w @ A_COEF))
    eta = w @ Y_COEF
    if cell == "SH1":
        y = GAUSSIAN_INTERCEPT + GAUSSIAN_EFFECT * a + eta + GAUSSIAN_SD * rng.standard_normal(n)
    else:
        y = rng.binomial(1, expit(BINARY_INTERCEPT + BINARY_EFFECT * a + eta)).astype(float)
    frame = pd.DataFrame(w, columns=list(COVARIATES))
    frame["A"] = a
    frame["Y"] = y
    return frame


def gaussian_shift(gamma: float) -> float:
    """``k``: the sharp Gaussian bound is the mean plus ``sigma * k``."""
    tau = gamma / (1.0 + gamma)
    z = float(stats.norm.ppf(tau))
    return (1.0 - 1.0 / gamma) * float(stats.norm.pdf(z)) / (1.0 - tau)


def _nodes(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(eta_A, eta_Y, weight)`` on a tensor rule for the bivariate normal of the indices."""
    x, h = np.polynomial.legendre.leggauss(count)
    x = HALF_RANGE * x
    h = HALF_RANGE * h * stats.norm.pdf(x)
    covariance = np.array([[A_COEF @ A_COEF, A_COEF @ Y_COEF], [A_COEF @ Y_COEF, Y_COEF @ Y_COEF]])
    root = np.linalg.cholesky(covariance)
    u, v = np.meshgrid(x, x, indexing="ij")
    standard = np.stack([u.ravel(), v.ravel()])
    eta_a, eta_y = root @ standard
    return eta_a, eta_y, np.outer(h, h).ravel()


def _arm_means(cell: str, eta_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cell == "SH1":
        base = GAUSSIAN_INTERCEPT + eta_y
        return base, base + GAUSSIAN_EFFECT
    return expit(BINARY_INTERCEPT + eta_y), expit(BINARY_INTERCEPT + BINARY_EFFECT + eta_y)


def _rho(cell: str, mean: np.ndarray, gamma: float, upper: bool) -> np.ndarray:
    if cell == "SH1":
        shift = GAUSSIAN_SD * gaussian_shift(gamma)
        return mean + shift if upper else mean - shift
    if upper:
        return np.minimum(gamma * mean, 1.0 - (1.0 - mean) / gamma)
    return np.maximum(1.0 - gamma * (1.0 - mean), mean / gamma)


def arm_bound(cell: str, arm: int, gamma: float, upper: bool, nodes: int = NODES) -> float:
    """The sharp upper or lower bound of ``E[Y(arm)]`` at ``gamma``."""
    eta_a, eta_y, weight = _nodes(nodes)
    e1 = expit(eta_a)
    share = e1 if arm == 1 else 1.0 - e1
    mean = _arm_means(cell, eta_y)[arm]
    value = share * mean + (1.0 - share) * _rho(cell, mean, gamma, upper)
    return float(weight @ value)


def sharp_set(cell: str, gamma: float, nodes: int = NODES) -> tuple[float, float]:
    """The sharp set ``(lower, upper)`` of the ATE at ``gamma``."""
    lower = arm_bound(cell, 1, gamma, False, nodes) - arm_bound(cell, 0, gamma, True, nodes)
    upper = arm_bound(cell, 1, gamma, True, nodes) - arm_bound(cell, 0, gamma, False, nodes)
    return lower, upper


def ate(cell: str, nodes: int = NODES) -> float:
    """The ATE: the sharp set at ``Gamma = 1``."""
    lower, upper = sharp_set(cell, 1.0, nodes)
    return 0.5 * (lower + upper)
