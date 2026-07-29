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

:func:`cross_validated_variance` is the CV-TMLE counterpart: the same quantity
averaged over validation folds rather than pooled, which is the variance estimator
Zheng & van der Laan pair with the cross-validated targeting step.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .._typing import FloatArray, IntArray

__all__ = ["cluster_sums", "cross_validated_variance", "influence_variance"]


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


def cross_validated_variance(
    influence_curve: FloatArray,
    folds: Iterable[IntArray],
    cluster: IntArray | None = None,
) -> float:
    r"""Variance of a CV-TMLE estimator, averaged over validation folds.

    Zheng & van der Laan pair the cross-validated targeting step with a
    cross-validated variance: each fold contributes the second moment of *its own*
    influence curve, computed from nuisance fits that never saw those rows, and the
    folds are then averaged.

    .. math::

        \hat\sigma^2_{CV} = \frac{1}{V} \sum_{v=1}^{V}
                            \frac{1}{n_v} \sum_{i \in \mathcal V_v} \mathrm{IC}_i^2,
        \qquad
        \widehat{\mathrm{Var}}(\hat\psi) = \hat\sigma^2_{CV} / n.

    The second moment rather than the centred variance is deliberate, and what it costs
    depends on which influence curve you hand it.  Given the *fold-specific* curves that
    canonical CV-TMLE produces -- each centred at its own fold's estimate -- the mean
    within a fold is already exactly zero, so not centring again discards nothing and
    this is the estimator Zheng & van der Laan define.  Given a *pooled* curve, mean-zero
    only over the whole sample, centring within a fold would throw away a real
    contribution, so the uncentred form remains the right choice but the result is an
    approximation to the same quantity rather than that quantity.

    At equal fold sizes the fold weights cancel and this reduces exactly to
    ``mean(IC**2) / n`` -- which is why it agrees with :func:`influence_variance` on a
    well-solved fit, and why the two can be compared as a check rather than trusted
    separately.

    Folds are weighted equally, at ``1/V``.  The point estimate they go with is averaged
    the same way, so the two stay consistent without a weighting argument.

    Parameters
    ----------
    folds:
        Validation index arrays, one per fold -- ``[test for _, test in folds]`` for a
        :class:`~cleverly.learners.crossfit.Folds`.  They must partition the sample.
    """
    ic = np.asarray(influence_curve, dtype=float).reshape(-1)
    n = ic.shape[0]
    if n < 2:
        raise ValueError("need at least 2 observations to estimate a variance")

    partition: Sequence[IntArray] = [np.asarray(index).reshape(-1) for index in folds]
    if not partition:
        raise ValueError("need at least one validation fold")
    covered = np.concatenate(partition) if len(partition) > 1 else partition[0]
    if covered.shape[0] != n or np.unique(covered).shape[0] != n:
        raise ValueError(
            f"validation folds must partition the {n} observations; "
            f"got {covered.shape[0]} index/indices covering "
            f"{np.unique(covered).shape[0]} distinct rows"
        )

    if cluster is None:
        moments = [float(np.mean(ic[index] ** 2)) for index in partition if index.size]
        return float(np.mean(moments) / n)

    codes = np.asarray(cluster).reshape(-1)
    moments = []
    for index in partition:
        if index.size == 0:
            continue
        sums = cluster_sums(ic[index], codes[index])
        moments.append(float(np.mean(sums**2)))
    n_clusters = int(np.unique(codes).size)
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters to estimate a cluster-robust variance")
    return float(n_clusters * np.mean(moments) / n**2)


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
