r"""The law of the ``dose-response`` study (X93), with its exact smoothed curves.

Covariates :math:`W \sim N(0, I_3)`.  The dose is a location model with a mean that is
nonlinear and bounded in :math:`W`,

.. math::

    A = m(W) + \sigma \varepsilon, \qquad m(W) = 2.5 + 0.7 \tanh W_1 + 0.4 \tanh(W_2 - W_3),

with :math:`\varepsilon \sim N(0, 1)` and :math:`\sigma = 1.5`, so its support is the real line
and :math:`g_0(a \mid W) \ge \varphi(3.6 / \sigma) / \sigma > 0` on the declared interval
:math:`D = [0, 5]`.  The binary outcome reads :math:`W` through one linear index,

.. math::

    \bar Q_0(a, W) = \operatorname{expit}\{f(a) + c(a) U\}, \qquad U = 0.7 W_1 - 0.5 W_3,
    \qquad f(a) = -0.3 + 0.9 \sin\{1.1 (a - 1.5)\}, \qquad c(a) = 0.8 + 0.1 a,

so :math:`\theta(a) = E \operatorname{expit}\{f(a) + c(a) U\}` with :math:`U \sim N(0, 0.74)`
is a one-dimensional Gauss-Hermite integral, exact to rounding.  :math:`W_1` confounds, since
it moves both the dose and the outcome.  The Gaussian outcome of the unbounded scenario has the
same mean on the identity scale, :math:`f(a) + c(a) U`, with unit noise.

Every smoothed truth :math:`\theta_h(a_0) = \int_D w_{h,a_0}(a)\theta(a)\,da` is a composite
Gauss-Legendre rule on the pieces between :math:`D`'s endpoints and :math:`a_0 \pm h`, where the
weight is a polynomial and :math:`\theta` is smooth.  A stratified truth conditions on
:math:`V = 1\{W_1 > 0\}`, under which :math:`U` is a skew normal; it is a two-dimensional
Gauss-Hermite integral over :math:`(W_1, W_3)`.  The implementation never enters this module.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
import pandas as pd
from scipy.special import expit

from cleverly.dose_response import DoseResponse, equivalent_kernel

SIGMA = 1.5
SUPPORT = (0.0, 5.0)
U_LOADINGS = (0.7, 0.0, -0.5)
U_VARIANCE = float(sum(b * b for b in U_LOADINGS))

#: Gauss-Hermite (probabilists') nodes behind every expectation over a normal index.
_HERMITE = np.polynomial.hermite_e.hermegauss(80)
#: Gauss-Legendre nodes on each smooth piece of a smoothed truth.
_LEGENDRE = np.polynomial.legendre.leggauss(64)


def dose_mean(w: np.ndarray) -> np.ndarray:
    """:math:`m(W)`, row by row."""
    return 2.5 + 0.7 * np.tanh(w[:, 0]) + 0.4 * np.tanh(w[:, 1] - w[:, 2])


def f(a: np.ndarray | float) -> np.ndarray:
    return -0.3 + 0.9 * np.sin(1.1 * (np.asarray(a, dtype=float) - 1.5))


def c(a: np.ndarray | float) -> np.ndarray:
    return 0.8 + 0.1 * np.asarray(a, dtype=float)


def index(w: np.ndarray) -> np.ndarray:
    """:math:`U = 0.7 W_1 - 0.5 W_3`."""
    return w @ np.asarray(U_LOADINGS)


def q_bar(a: np.ndarray, w: np.ndarray, *, gaussian: bool = False) -> np.ndarray:
    """:math:`\\bar Q_0(a, W)` on the outcome's scale."""
    linear = f(a) + c(a) * index(w)
    return linear if gaussian else expit(linear)


def density(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """:math:`g_0(a \\mid W)`."""
    z = (np.asarray(a, dtype=float) - dose_mean(w)) / SIGMA
    return np.exp(-0.5 * z * z) / (SIGMA * np.sqrt(2.0 * np.pi))


def sample(n: int, seed: int, *, gaussian: bool = False) -> pd.DataFrame:
    """``n`` rows with columns ``W1, W2, W3, V, A, Y``.

    ``V = 1{W1 > 0}`` is an integer column, so a stratified fit labels its strata ``V=0`` and
    ``V=1``, the suffixes :func:`truths` writes.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    a = dose_mean(w) + SIGMA * rng.standard_normal(n)
    mean = q_bar(a, w, gaussian=gaussian)
    y = mean + rng.standard_normal(n) if gaussian else rng.binomial(1, mean).astype(float)
    return pd.DataFrame(
        {
            "W1": w[:, 0],
            "W2": w[:, 1],
            "W3": w[:, 2],
            "V": (w[:, 0] > 0).astype(int),
            "A": a,
            "Y": y,
        }
    )


def curve(
    a: np.ndarray | float, *, gaussian: bool = False, stratum: int | None = None
) -> np.ndarray:
    """:math:`\\theta(a)`, or :math:`E\\{\\bar Q_0(a, W) \\mid V = v\\}` for ``stratum=v``."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    nodes, weights = _HERMITE
    weights = weights / weights.sum()
    if stratum is None:
        u = np.sqrt(U_VARIANCE) * nodes
        linear = f(a)[:, None] + c(a)[:, None] * u[None, :]
        values = linear if gaussian else expit(linear)
        return np.asarray(values @ weights)
    # V = 1{W1 > 0}: W1 is a half normal on the stratum's side, W3 an independent normal.
    half, half_weights = _half_normal(stratum)
    u = U_LOADINGS[0] * half[:, None] + U_LOADINGS[2] * nodes[None, :]
    joint = half_weights[:, None] * weights[None, :]
    linear = f(a)[:, None, None] + c(a)[:, None, None] * u[None, :, :]
    values = linear if gaussian else expit(linear)
    return np.asarray(np.tensordot(values, joint, axes=([1, 2], [0, 1])))


@cache
def _half_normal(stratum: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of :math:`N(0, 1)` conditioned on the sign ``stratum`` gives."""
    nodes, weights = np.polynomial.legendre.leggauss(200)
    upper = 9.0
    x = 0.5 * upper * (nodes + 1.0)
    density_ = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    mass = 0.5 * upper * weights * density_ * 2.0
    sign = 1.0 if stratum == 1 else -1.0
    return sign * x, mass / mass.sum()


def _pieces(h: float, a0: float) -> np.ndarray:
    lower, upper = SUPPORT
    breaks = {lower, upper}
    for edge in (a0 - h, a0 + h):
        if lower < edge < upper:
            breaks.add(edge)
    return np.array(sorted(breaks))


def smoothed(
    h: float,
    a0: float,
    *,
    kernel: str = "epanechnikov",
    gaussian: bool = False,
    stratum: int | None = None,
) -> float:
    """:math:`\\theta_h(a_0)`, exact to rounding."""
    nodes, weights = _LEGENDRE
    total = 0.0
    breaks = _pieces(h, a0)
    for left, right in itertools.pairwise(breaks):
        half = 0.5 * (right - left)
        a = left + half * (nodes + 1.0)
        w = equivalent_kernel(a, a0, h, SUPPORT, kernel)  # type: ignore[arg-type]
        total += float(np.sum(half * weights * w * curve(a, gaussian=gaussian, stratum=stratum)))
    return total


def truths(
    curve_: DoseResponse, *, gaussian: bool = False, strata: bool = False
) -> dict[str, float]:
    """Every pair's truth under the parameter names the fit reports."""
    from cleverly.dose_response import pair_name

    out: dict[str, float] = {}
    for h, a0 in curve_.pairs:
        if strata:
            for v in (0, 1):
                name = f"{pair_name(h, a0)}[V={v}]"
                out[name] = smoothed(h, a0, kernel=curve_.kernel, gaussian=gaussian, stratum=v)
        else:
            out[pair_name(h, a0)] = smoothed(h, a0, kernel=curve_.kernel, gaussian=gaussian)
    return out


#: Gauss-Hermite nodes per axis of the three-dimensional rule over :math:`W`.
_W_NODES = 30


@cache
def _w_rule() -> tuple[np.ndarray, np.ndarray]:
    """A tensor Gauss-Hermite rule for :math:`W \\sim N(0, I_3)`: nodes ``(K, 3)``, weights ``(K,)``."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(_W_NODES)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    mass = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
    return grid, mass


def _dose_rule(*pairs: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre doses and weights on the pieces of :math:`D` that every pair's weight sees.

    The pieces break at :math:`D`'s endpoints and at each :math:`a_0 \\pm h` inside it, so each
    weight is a polynomial on every piece.
    """
    nodes, weights = _LEGENDRE
    breaks = sorted({float(edge) for h, a0 in pairs for edge in _pieces(h, a0)})
    points, masses = [], []
    for left, right in itertools.pairwise(breaks):
        half = 0.5 * (right - left)
        points.append(left + half * (nodes + 1.0))
        masses.append(half * weights)
    return np.concatenate(points), np.concatenate(masses)


def efficiency_covariance(
    pairs: tuple[tuple[float, float], ...],
    *,
    kernel: str = "epanechnikov",
    gaussian: bool = False,
) -> np.ndarray:
    r"""The covariance of the efficient influence curves of ``pairs``, ``(h, a0)`` each.

    .. math::

        \operatorname{Cov}(D_j, D_k) = E_W \int_D \frac{w_j(a) w_k(a) \sigma^2(a, W)}
        {g_0(a \mid W)}\,da + \operatorname{Cov}_W\Bigl(\int_D w_j \bar Q_0(\cdot, W),
        \int_D w_k \bar Q_0(\cdot, W)\Bigr),

    with :math:`\sigma^2 = \bar Q_0(1 - \bar Q_0)` for the binary outcome and 1 for the
    Gaussian one.  The first term is the residual part of :math:`D` (its conditional covariance
    given :math:`(A, W)`, integrated against :math:`g_0`), the second the plug-in part; the two
    parts are uncorrelated.  Both are tensor Gauss-Hermite rules over :math:`W` against one
    Gauss-Legendre rule over the pieces of :math:`D`, exact to rounding on this smooth law.
    """
    w_nodes, w_mass = _w_rule()
    a, a_mass = _dose_rule(*pairs)
    weights = np.stack(
        [equivalent_kernel(a, a0, h, SUPPORT, kernel) for h, a0 in pairs]  # type: ignore[arg-type]
    )
    linear = f(a)[None, :] + c(a)[None, :] * index(w_nodes)[:, None]
    mean = linear if gaussian else expit(linear)
    variance = np.ones_like(mean) if gaussian else mean * (1.0 - mean)
    z = (a[None, :] - dose_mean(w_nodes)[:, None]) / SIGMA
    g0 = np.exp(-0.5 * z * z) / (SIGMA * np.sqrt(2.0 * np.pi))
    residual_density = w_mass @ (variance / g0)
    residual = (weights * (residual_density * a_mass)[None, :]) @ weights.T
    integrals = mean @ (weights * a_mass[None, :]).T
    centred = integrals - w_mass @ integrals
    plug_in = centred.T @ (centred * w_mass[:, None])
    return np.asarray(residual + plug_in, dtype=float)


def efficiency_sd(
    h: float, a0: float, *, kernel: str = "epanechnikov", gaussian: bool = False
) -> float:
    """:math:`\\operatorname{sd} D_{h,a_0}(P_0)(O)` (:func:`efficiency_covariance` of one pair)."""
    return float(np.sqrt(efficiency_covariance(((h, a0),), kernel=kernel, gaussian=gaussian)[0, 0]))


def expected_kish(h: float, a0: float, n: int, *, kernel: str = "epanechnikov") -> float:
    r"""The Kish size of the clever covariate under the true density, at ``n`` rows.

    With :math:`H = w(A)/g_0(A \mid W)`, :math:`E|H| = \int_D |w|` and
    :math:`E H^2 = \int_D w(a)^2 E_W\{1/g_0(a \mid W)\}\,da`, so the Kish size
    :math:`(\sum_i |H_i|)^2 / \sum_i H_i^2` is :math:`n (E|H|)^2 / E H^2` to first order.  The
    fit reports the same quantity with its own binned density in place of :math:`g_0`.
    """
    w_nodes, w_mass = _w_rule()
    a, a_mass = _dose_rule((h, a0))
    weight = equivalent_kernel(a, a0, h, SUPPORT, kernel)  # type: ignore[arg-type]
    z = (a[None, :] - dose_mean(w_nodes)[:, None]) / SIGMA
    inverse = SIGMA * np.sqrt(2.0 * np.pi) * np.exp(0.5 * z * z)
    first = float(np.abs(weight) @ a_mass)
    second = float((weight**2 * (w_mass @ inverse)) @ a_mass)
    return float(n * first * first / second)
