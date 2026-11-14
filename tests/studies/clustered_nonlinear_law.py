r"""The nonlinear clustered law of ``few-cluster-adaptive-learners`` (L2), and its exact truth.

L2 is the ``equal10`` law of ``clustered-few-cluster-tmle`` (L1) with both nuisances made
nonlinear in the row covariates. Each cluster holds :data:`CLUSTER_SIZE` rows. ``W1`` and
``W2`` are row-level standard normals, and ``U`` is a cluster-level standard normal, shared
within the cluster and hidden from the fit, with the arm-by-latent coefficient 8 of L1:

.. math::

    \operatorname{logit} g(W) &= 0.3 + 0.3 W_1 + 0.6 W_2 - 0.25 W_1^2, \\
    \operatorname{logit} \bar Q(a, W, U) &= -0.4 + 0.8 a + \sin(2 W_1) + 0.6 W_1 W_2
        + 0.3 W_2 + (0.6 + 8 a) U.

A main-terms GLM in :math:`(A, W_1, W_2)` is misspecified for both nuisances, while the
default learner library can fit the curvature (plan X54 section 5.4). The propensity stays in
:math:`[0.017, 0.84]` for :math:`|W_1| \le 3` and :math:`|W_2| \le 2`.

**Truth.** Given the arm, the latent enters the logit linearly, so
:math:`E\{Y(a)\} = E_W\,\ell\{c_a(W), s_a\}` with
:math:`\ell(c, s) = E\{\operatorname{expit}(c + s Z)\}`, :math:`Z` standard normal,
:math:`c_a` the logit without the latent and :math:`s_a = 0.6 + 8a`. The outer expectation is
a tensor Gauss-Hermite rule in :math:`(W_1, W_2)`, whose integrand is entire. The inner one
is a trapezoid rule in :math:`Z` on :math:`[-12, 12]`, which converges geometrically for a
smooth integrand with Gaussian tails. A Gauss-Hermite rule in :math:`Z` converges slowly at
:math:`s_1 = 8.6`, where the integrand is a near step (the note of
:mod:`tests.studies.clustered_unequal_laws`). :func:`truth` takes both resolutions, and the
design test checks that the two declared resolutions agree to :data:`TRUTH_AGREEMENT` and that
a large draw agrees to its sampling error.
"""

from __future__ import annotations

import functools
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import expit

#: Rows per cluster.
CLUSTER_SIZE: Final[int] = 10
#: The arm-by-latent coefficient, as in L1 (``equal10``).
EFFECT_MODIFIER: Final[float] = 8.0
#: The declared truth resolution: Gauss-Hermite nodes per covariate, trapezoid points in the
#: latent. :data:`CHECK_RESOLUTION` doubles both.
RESOLUTION: Final[tuple[int, int]] = (100, 4_001)
CHECK_RESOLUTION: Final[tuple[int, int]] = (200, 8_001)
#: The largest difference the two resolutions may show in any truth.
TRUTH_AGREEMENT: Final[float] = 1e-8
#: The half-width of the latent's trapezoid range, in standard deviations.
LATENT_RANGE: Final[float] = 12.0


def propensity(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """The treatment probability of L2."""
    w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
    return np.asarray(expit(0.3 + 0.3 * w1 + 0.6 * w2 - 0.25 * w1**2), dtype=float)


def outcome_logit(w1: np.ndarray, w2: np.ndarray, a: float | np.ndarray) -> np.ndarray:
    """The outcome logit of L2 without the latent term."""
    w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
    return np.asarray(-0.4 + 0.8 * a + np.sin(2.0 * w1) + 0.6 * w1 * w2 + 0.3 * w2, dtype=float)


def latent_scale(a: float | np.ndarray) -> np.ndarray:
    """The coefficient of the cluster latent in arm ``a``."""
    return np.asarray(0.6 + EFFECT_MODIFIER * np.asarray(a, dtype=float), dtype=float)


def draw(n_clusters: int, rng: np.random.Generator) -> pd.DataFrame:
    """One sample of ``n_clusters`` clusters of :data:`CLUSTER_SIZE` rows.

    The frame holds ``Y``, ``A``, ``W1``, ``W2``, ``cluster`` and ``size``, the columns of
    :func:`tests.studies.clustered_unequal_laws.draw_informative`.
    """
    cluster = np.repeat(np.arange(n_clusters), CLUSTER_SIZE)
    rows = cluster.size
    w1 = rng.normal(size=rows)
    w2 = rng.normal(size=rows)
    u = rng.normal(size=n_clusters)[cluster]
    a = rng.binomial(1, propensity(w1, w2)).astype(float)
    mean = expit(outcome_logit(w1, w2, a) + latent_scale(a) * u)
    y = rng.binomial(1, mean).astype(float)
    return pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W1": w1,
            "W2": w2,
            "cluster": cluster,
            "size": np.full(rows, float(CLUSTER_SIZE)),
        }
    )


def logistic_normal(center: np.ndarray, scale: float, points: int) -> np.ndarray:
    r""":math:`E\{\operatorname{expit}(c + s Z)\}` for each ``c``, by the trapezoid rule in ``Z``."""
    z = np.linspace(-LATENT_RANGE, LATENT_RANGE, points)
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    step = z[1] - z[0]
    weights = np.full(points, step)
    weights[[0, -1]] *= 0.5
    values = expit(np.asarray(center, dtype=float)[:, None] + scale * z[None, :])
    return np.asarray(values @ (weights * density), dtype=float)


@functools.cache
def truth(resolution: tuple[int, int] = RESOLUTION) -> dict[str, float]:
    """``ey0``, ``ey1`` and ``ate`` of L2 at a quadrature resolution.

    Parameters
    ----------
    resolution : tuple of int
        Gauss-Hermite nodes per covariate, and trapezoid points in the latent.
    """
    nodes, points = resolution
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    w1, w2 = (grid.ravel() for grid in np.meshgrid(x, x, indexing="ij"))
    weight = np.outer(w, w).ravel()
    means = {}
    for arm in (0, 1):
        inner = logistic_normal(outcome_logit(w1, w2, arm), float(latent_scale(arm)), points)
        means[f"ey{arm}"] = float(weight @ inner)
    return {"ey0": means["ey0"], "ey1": means["ey1"], "ate": means["ey1"] - means["ey0"]}
