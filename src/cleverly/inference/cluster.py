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


def _contiguous_codes(codes: IntArray) -> int | None:
    """``C`` if ``codes`` is already ``0..C-1`` with every code used, else ``None``.

    Worth checking because it usually is.  :func:`cleverly.data.validate.encode_clusters`
    densifies the identifiers **once**, when the container is built, and
    :meth:`CausalData.subset` re-derives them; so every call from inside the package hands
    this function contiguous codes, and the ``np.unique`` that used to run here was
    re-deriving an encoding it had already been given.  That sort is the majority of the
    cost at a small estimand count: at ``n = 1e6``, ``m = 5`` it is 44 ms of a 78 ms call.

    The check is three linear passes against that sort's ``O(n log n)`` -- measured at
    about a twentieth of what it saves -- and it is exact rather than optimistic. Codes
    that skip a value are *not* contiguous in the sense that matters: ``np.unique`` would
    return one row per *observed* label, where ``np.bincount`` returns one per slot, and
    the empty row would go on to change a variance.  Every caller therefore gets the same
    answer it did, including one passing arbitrary labels.
    """
    if not np.issubdtype(codes.dtype, np.integer) or codes.size == 0:
        return None
    high = int(codes.max())
    # A contiguous encoding has at most one code per row, so this also bounds the count
    # array below -- an ``id`` column of raw integers would otherwise allocate by its
    # largest value rather than by its cardinality.
    if high >= codes.size or int(codes.min()) != 0:
        return None
    counts = np.bincount(codes, minlength=high + 1)
    return high + 1 if bool(counts.all()) else None


def cluster_sums(influence_curve: FloatArray, cluster: IntArray) -> FloatArray:
    """Sum the influence curve within each cluster.

    Works for a 1-d influence curve or an ``(n, m)`` matrix of several estimands'
    curves, in which case each column is summed independently.

    Rows come back in sorted cluster-label order, which is what every caller assumes and
    what :func:`_contiguous_codes` exists to preserve while skipping the sort that used to
    establish it.
    """
    ic = np.asarray(influence_curve, dtype=float)
    codes = np.asarray(cluster).reshape(-1)
    if ic.shape[0] != codes.shape[0]:
        raise ValueError(
            f"influence curve has {ic.shape[0]} rows but cluster has {codes.shape[0]} entries"
        )
    n_clusters = _contiguous_codes(codes)
    if n_clusters is None:
        unique, inverse = np.unique(codes, return_inverse=True)
        codes = inverse.reshape(-1)
        n_clusters = int(unique.size)
    # np.bincount rather than np.add.at: the latter is unbuffered and measures about
    # twice as slow here, and this runs on every estimate and every multiplier draw.
    if ic.ndim == 1:
        return np.asarray(np.bincount(codes, weights=ic, minlength=n_clusters), dtype=float)
    # One pass per estimand over the same index vector.  A single ``bincount`` over a
    # flattened ``(row, column)`` index does fuse them, and was measured: 2.3x at
    # ``m = 20, n = 1e5`` and **0.5x** at ``n = 1e6``, where its ``8nm``-byte index array
    # stops fitting anywhere useful.  One code path, at the size that matters.
    return np.asarray(
        np.column_stack(
            [
                np.bincount(codes, weights=ic[:, column], minlength=n_clusters)
                for column in range(ic.shape[1])
            ]
        ),
        dtype=float,
    )


def influence_variance(
    influence_curve: FloatArray,
    cluster: IntArray | None = None,
) -> float:
    """Variance of an estimator from its influence curve.

    Parameters
    ----------
    influence_curve : ndarray
        ``(n,)`` influence curve.
    cluster : ndarray or None
        ``(n,)`` cluster codes. ``None`` treats the rows as independent.

    Returns
    -------
    float
        Variance of the estimator on the inference scale.
    """
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

    For the equally weighted fold estimator
    :math:`\hat\psi_{CV}=V^{-1}\sum_v\hat\psi_v`, the finite-sample scaling is

    .. math::

        \widehat{\mathrm{Var}}(\hat\psi_{CV}) =
        \frac{1}{V^2}\sum_{v=1}^{V}\frac{1}{n_v^2}
        \sum_{i \in \mathcal V_v} \mathrm{IC}_{v,i}^2.

    The second moment rather than a fold-centred variance is deliberate.  Canonical
    CV-TMLE uses one common fluctuation coefficient, so a validation fold's efficient
    influence curve need not have empirical mean zero by itself; the fold scores cancel
    only after the validation risks are aggregated.  Recentring within folds would erase
    that real component.

    At equal fold sizes this reduces exactly to ``mean(IC**2) / n``.  Keeping the
    :math:`n_v^{-2}` factors is essential when a partition is not exactly balanced:
    dividing a fold-averaged second moment by the total ``n`` instead estimates a
    different, row-weighted aggregation.

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

    raw_partition = [np.asarray(index).reshape(-1) for index in folds]
    if any(not np.issubdtype(index.dtype, np.integer) for index in raw_partition):
        raise ValueError("validation-fold indices must be integers")
    partition: Sequence[IntArray] = [index.astype(np.int64, copy=False) for index in raw_partition]
    if not partition:
        raise ValueError("need at least one validation fold")
    if any(index.size == 0 for index in partition):
        raise ValueError("validation folds must not be empty")
    covered = np.concatenate(partition) if len(partition) > 1 else partition[0]
    if np.any(covered < 0) or np.any(covered >= n):
        raise ValueError(f"validation-fold indices must lie in [0, {n}); got {covered.tolist()}")
    if covered.shape[0] != n or np.unique(covered).shape[0] != n:
        raise ValueError(
            f"validation folds must partition the {n} observations; "
            f"got {covered.shape[0]} index/indices covering "
            f"{np.unique(covered).shape[0]} distinct rows"
        )

    nonempty = list(partition)
    n_folds = len(nonempty)
    if cluster is None:
        return float(
            sum(float(np.sum(ic[index] ** 2)) / index.size**2 for index in nonempty) / n_folds**2
        )

    codes = np.asarray(cluster).reshape(-1)
    if codes.shape[0] != n:
        raise ValueError(f"cluster has {codes.shape[0]} entries but influence curve has {n} rows")
    cluster_folds: dict[object, int] = {}
    contributions = []
    for fold_number, index in enumerate(nonempty):
        for code in np.unique(codes[index]):
            key = code.item() if hasattr(code, "item") else code
            previous = cluster_folds.setdefault(key, fold_number)
            if previous != fold_number:
                raise ValueError(
                    f"cluster {key!r} appears in validation folds {previous} and "
                    f"{fold_number}; clusters must be assigned whole to one fold"
                )
        sums = cluster_sums(ic[index], codes[index])
        contributions.append(float(np.sum(sums**2)) / index.size**2)
    n_clusters = int(np.unique(codes).size)
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters to estimate a cluster-robust variance")
    return float(sum(contributions) / n_folds**2)


def influence_covariance(
    influence_curves: FloatArray,
    cluster: IntArray | None = None,
) -> FloatArray:
    """Covariance matrix across several estimands' influence curves.

    Needed for the delta method on a function of estimands and for simultaneous
    confidence bands, both of which depend on how the estimands covary rather
    than only on their individual variances.

    Parameters
    ----------
    influence_curves : ndarray
        ``(n, k)`` influence curves, one column per estimand.
    cluster : ndarray or None
        ``(n,)`` cluster codes. ``None`` treats the rows as independent.

    Returns
    -------
    ndarray
        ``(k, k)`` covariance matrix of the estimators.
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
