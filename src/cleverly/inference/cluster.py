r"""Influence-curve variance, with optional clustering.

For an asymptotically linear estimator, :math:`\hat\psi - \psi_0 \approx
\frac{1}{n}\sum_i \mathrm{IC}_i`, so the variance estimate is
:math:`\widehat{\mathrm{Var}}(\hat\psi) = \widehat{\mathrm{Var}}(\mathrm{IC}) / n`.

When observations are grouped -- repeated measures, households, clinics -- the
:math:`\mathrm{IC}_i` are not independent and that formula understates the
variance.  Summing the influence curve within each cluster restores
independence across the resulting :math:`n_c` terms:

.. math::

    \hat\psi - \psi_0 \approx \frac{1}{n}\sum_{c=1}^{n_c} S_c,
    \qquad S_c = \sum_{i \in c} \mathrm{IC}_i,
    \qquad
    \widehat{\mathrm{Var}}(\hat\psi) = \frac{n_c\,\widehat{\mathrm{Var}}(S_c)}{n^2}.

With singleton clusters :math:`S_c = \mathrm{IC}_c` and :math:`n_c = n`, so the
expression collapses to the independent case -- a property the tests assert
directly.
"""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, IntArray

__all__ = ["cluster_sums", "influence_variance"]


def cluster_sums(influence_curve: FloatArray, cluster: IntArray) -> FloatArray:
    """Sum the influence curve within each cluster.

    Works for a 1-d influence curve or an ``(n, m)`` matrix of several estimands'
    curves, in which case each column is summed independently.
    """
    ic = np.asarray(influence_curve, dtype=float)
    codes = np.asarray(cluster).reshape(-1)
    if ic.shape[0] != codes.shape[0]:
        raise ValueError(
            f"influence curve has {ic.shape[0]} rows but cluster has {codes.shape[0]} entries"
        )
    unique, inverse = np.unique(codes, return_inverse=True)
    # np.bincount rather than np.add.at: the latter is unbuffered and measures about
    # twice as slow here, and this runs on every estimate and every multiplier draw.
    inverse = inverse.reshape(-1)
    n_clusters = unique.size
    if ic.ndim == 1:
        return np.bincount(inverse, weights=ic, minlength=n_clusters).astype(float)
    return np.column_stack(
        [
            np.bincount(inverse, weights=ic[:, column], minlength=n_clusters)
            for column in range(ic.shape[1])
        ]
    ).astype(float)


def influence_variance(
    influence_curve: FloatArray,
    cluster: IntArray | None = None,
) -> float:
    """Variance of an estimator from its influence curve."""
    ic = np.asarray(influence_curve, dtype=float).reshape(-1)
    n = ic.shape[0]
    if n < 2:
        raise ValueError("need at least 2 observations to estimate a variance")

    if cluster is None:
        return float(np.var(ic, ddof=1) / n)

    sums = cluster_sums(ic, cluster)
    n_clusters = sums.shape[0]
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters to estimate a cluster-robust variance")
    return float(n_clusters * np.var(sums, ddof=1) / n**2)


def influence_covariance(
    influence_curves: FloatArray,
    cluster: IntArray | None = None,
) -> FloatArray:
    """Covariance matrix across several estimands' influence curves.

    Needed for the delta method on a function of estimands and for simultaneous
    confidence bands, both of which depend on how the estimands covary rather
    than only on their individual variances.
    """
    ic = np.asarray(influence_curves, dtype=float)
    if ic.ndim != 2:
        raise ValueError(f"expected an (n, m) matrix of influence curves; got shape {ic.shape}")
    n = ic.shape[0]
    if cluster is None:
        return np.asarray(np.cov(ic, rowvar=False, ddof=1) / n, dtype=float).reshape(
            ic.shape[1], ic.shape[1]
        )
    sums = cluster_sums(ic, cluster)
    n_clusters = sums.shape[0]
    covariance = np.cov(sums, rowvar=False, ddof=1).reshape(ic.shape[1], ic.shape[1])
    return np.asarray(n_clusters * covariance / n**2, dtype=float)
