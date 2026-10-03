"""Clustered laws at unequal cluster sizes, and their exact truths.

Two laws serve ``clustered-unequal-cvtmle``, ``clustered-few-cluster-tmle`` and the unit tests
of the cluster-ratio variance.

**The informative law.** It keeps the mean and the propensity of
``cleverly.datasets.clustered_dgp(family="binomial")`` and adds the cluster size. The size
:math:`N_j` is uniform on ``{2, ..., 18}``, with mean 10. Its finite support is Assumption 1(b)
of Wang, Park, Small and Li (2024). With :math:`s(N)=(N-10)/\\sqrt{24}`, the outcome logit adds
:math:`\\delta\\,s(N)+\\gamma\\,a\\,s(N)`. The size does not enter the propensity, so it does not
confound. ``W1`` and ``W2`` are row-level standard normals. The third latent is a cluster-level
standard normal, shared within the cluster and hidden from the fit, as in ``clustered_dgp``.

The row-weighted mean is :math:`\\mu_I(a)=E\\{N\\,m_a(N)\\}/E(N)`, and the cluster-average mean is
:math:`\\mu_C(a)=E\\{m_a(N)\\}`, where :math:`m_a(N)` is the mean outcome under arm ``a`` in a
cluster of size ``N``. The two coincide at :math:`\\delta=\\gamma=0`.

**The cluster-level covariate law.** ``W1`` is shared within a cluster, ``W2`` is row level,
and every cluster holds the same number of rows. The plug-in part of the curve then carries the
cluster effect. It is the law of the X24 review probe that found the fold-evaluated variance
defect.

Truths are exact sums over ``N`` and one-dimensional integrals. The latents are independent
standard normals that enter the logit linearly, so their sum given ``N`` and the arm is one
normal, and each mean is a logistic-normal integral. Gauss-Hermite tensor quadrature in the
three latents converges slowly here, because the arm-by-latent coefficient of 8 makes the
integrand steep. ``tests/unit/test_cluster_ratio_variance.py`` checks that two integration
tolerances agree, and that a tensor quadrature agrees to its own accuracy.
"""

from __future__ import annotations

import functools
from typing import Final

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.special import expit
from scipy.stats import norm

#: The support of the cluster size of the informative law.
SIZES: Final[np.ndarray] = np.arange(2, 19)
#: The mean and the variance of the uniform size on ``SIZES``.
SIZE_MEAN: Final[float] = 10.0
SIZE_VARIANCE: Final[float] = 24.0
#: The size coefficient of the outcome logit in both arms.
DELTA: Final[float] = 0.5
#: The arm-by-size coefficient. The pilot rule of ``clustered-unequal-cvtmle`` sets it; see
#: :data:`GAMMA_PILOT`.
GAMMA: Final[float] = 2.0
#: The record of the pilot that set :data:`GAMMA`, written before any registered run.
GAMMA_PILOT: Final[str] = "pending"
#: The arm-by-latent coefficient of ``clustered_dgp(family="binomial")``. The shared latent
#: modifies the effect, which correlates the curve within a cluster.
EFFECT_MODIFIER: Final[float] = 8.0
#: The absolute and relative tolerances of the one-dimensional truth integrals.
TRUTH_TOLERANCE: Final[float] = 1e-13
#: Rows per cluster of the cluster-level covariate law.
COVARIATE_LAW_ROWS: Final[int] = 30


def size_score(size: np.ndarray) -> np.ndarray:
    """The standardized cluster size :math:`(N - 10) / \\sqrt{24}`."""
    return (np.asarray(size, dtype=float) - SIZE_MEAN) / np.sqrt(SIZE_VARIANCE)


def informative_propensity(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """The propensity of ``clustered_dgp``, which the size does not enter."""
    return np.asarray(expit(0.3 * w1 + 0.6 * w2), dtype=float)


def informative_mean(
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    a: float | np.ndarray,
    size: np.ndarray,
    *,
    delta: float,
    gamma: float,
    effect_modifier: float = EFFECT_MODIFIER,
) -> np.ndarray:
    """The outcome mean of the informative law."""
    score = size_score(size)
    logit = (
        -0.4
        + 0.8 * a
        + 0.5 * w1
        + 0.3 * w2
        + 0.6 * w3
        + effect_modifier * a * w3
        + delta * score
        + gamma * a * score
    )
    return np.asarray(expit(logit), dtype=float)


def draw_informative(
    n_clusters: int,
    rng: np.random.Generator,
    *,
    delta: float = DELTA,
    gamma: float = GAMMA,
    effect_modifier: float = EFFECT_MODIFIER,
    sizes: np.ndarray | None = None,
) -> pd.DataFrame:
    """One sample of ``n_clusters`` independent clusters of the informative law.

    ``sizes`` fixes the cluster sizes, for an equal-size control. ``None`` draws them.
    The frame holds ``Y``, ``A``, ``W1``, ``W2``, ``cluster`` and ``size``, the row count
    of the row's cluster.
    """
    drawn = (
        rng.integers(int(SIZES.min()), int(SIZES.max()) + 1, size=n_clusters)
        if sizes is None
        else np.asarray(sizes, dtype=int)
    )
    cluster = np.repeat(np.arange(n_clusters), drawn)
    rows = cluster.size
    w1 = rng.normal(size=rows)
    w2 = rng.normal(size=rows)
    w3 = rng.normal(size=n_clusters)[cluster]
    size = drawn[cluster].astype(float)
    a = rng.binomial(1, informative_propensity(w1, w2)).astype(float)
    mean = informative_mean(
        w1, w2, w3, a, size, delta=delta, gamma=gamma, effect_modifier=effect_modifier
    )
    y = rng.binomial(1, mean).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "cluster": cluster, "size": size})


def logistic_normal_mean(center: float, scale: float, tolerance: float = TRUTH_TOLERANCE) -> float:
    r""":math:`E\{\operatorname{expit}(c + \sigma Z)\}` for a standard normal :math:`Z`.

    Every truth here reduces to this one-dimensional integral. Each latent enters the logit
    linearly and the latents are independent standard normals, so their sum is one normal.
    """

    def integrand(z: float) -> float:
        return float(expit(center + scale * z) * norm.pdf(z))

    value, _ = integrate.quad(
        integrand, -np.inf, np.inf, epsabs=tolerance, epsrel=tolerance, limit=400
    )
    return float(value)


@functools.cache
def size_means(
    delta: float,
    gamma: float,
    effect_modifier: float = EFFECT_MODIFIER,
    tolerance: float = TRUTH_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """``m_0(N)`` and ``m_1(N)`` on :data:`SIZES`.

    Given ``N`` and the arm, the logit is a constant plus
    :math:`0.5 W_1 + 0.3 W_2 + (0.6 + k a) W_3`, a normal with variance
    :math:`0.34 + (0.6 + k a)^2`, where :math:`k` is the effect modifier.
    """
    by_arm = []
    for arm in (0.0, 1.0):
        scale = float(np.sqrt(0.5**2 + 0.3**2 + (0.6 + effect_modifier * arm) ** 2))
        centers = -0.4 + 0.8 * arm + (delta + gamma * arm) * size_score(SIZES)
        by_arm.append(np.array([logistic_normal_mean(float(c), scale, tolerance) for c in centers]))
    return by_arm[0], by_arm[1]


def informative_truth(
    delta: float = DELTA,
    gamma: float = GAMMA,
    effect_modifier: float = EFFECT_MODIFIER,
    tolerance: float = TRUTH_TOLERANCE,
) -> dict[str, float]:
    r"""The row-weighted and the cluster-average truths of the informative law.

    Keys ``ey0``, ``ey1`` and ``ate`` are the row-weighted :math:`\mu_I`. Keys with the
    suffix ``_cluster`` are the cluster-average :math:`\mu_C`.
    """
    m0, m1 = size_means(delta, gamma, effect_modifier, tolerance)
    size = SIZES.astype(float)
    individual = {"ey0": float(size @ m0 / size.sum()), "ey1": float(size @ m1 / size.sum())}
    cluster = {"ey0": float(m0.mean()), "ey1": float(m1.mean())}
    return {
        "ey0": individual["ey0"],
        "ey1": individual["ey1"],
        "ate": individual["ey1"] - individual["ey0"],
        "ey0_cluster": cluster["ey0"],
        "ey1_cluster": cluster["ey1"],
        "ate_cluster": cluster["ey1"] - cluster["ey0"],
    }


def covariate_propensity(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """The propensity of the cluster-level covariate law."""
    return np.asarray(expit(0.3 * w1 + 0.3 * w2), dtype=float)


def covariate_mean(w1: np.ndarray, w2: np.ndarray, a: float | np.ndarray) -> np.ndarray:
    """The outcome mean of the cluster-level covariate law."""
    return np.asarray(expit(-0.2 + 0.5 * a + 2.0 * w1 + 0.3 * w2), dtype=float)


def draw_cluster_covariate(
    n_clusters: int, rng: np.random.Generator, *, rows: int = COVARIATE_LAW_ROWS
) -> pd.DataFrame:
    """One sample of the cluster-level covariate law: ``W1`` shared within a cluster."""
    cluster = np.repeat(np.arange(n_clusters), rows)
    w1 = rng.normal(size=n_clusters)[cluster]
    w2 = rng.normal(size=cluster.size)
    a = rng.binomial(1, covariate_propensity(w1, w2)).astype(float)
    y = rng.binomial(1, covariate_mean(w1, w2, a)).astype(float)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "cluster": cluster})


@functools.cache
def covariate_truth(tolerance: float = TRUTH_TOLERANCE) -> dict[str, float]:
    """``ey0``, ``ey1`` and ``ate`` of the cluster-level covariate law.

    The logit is a constant plus :math:`2 W_1 + 0.3 W_2`, a normal with variance 4.09.
    """
    scale = float(np.sqrt(2.0**2 + 0.3**2))
    ey0 = logistic_normal_mean(-0.2, scale, tolerance)
    ey1 = logistic_normal_mean(0.3, scale, tolerance)
    return {"ey0": ey0, "ey1": ey1, "ate": ey1 - ey0}
