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
from typing import Final

import numpy as np

from .._inference_status import (
    FEW_CLUSTER_THRESHOLD,
    MINIMUM_INTERVAL_CLUSTERS,
    InferenceStatus,
)
from .._typing import BoolArray, FloatArray, IntArray

__all__ = [
    "WEIGHT_MASS_RTOL",
    "cluster_inference_status",
    "cluster_reference_df",
    "cluster_sizes",
    "cluster_sums",
    "cluster_weight_mass",
    "cross_validated_variance",
    "fewest_clusters",
    "influence_variance",
    "positive_mass_clause",
    "stacked_second_moment_covariance",
    "stacked_second_moment_variance",
    "unequal_cluster_sizes",
]


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

    With ``cluster``, each fold contributes the cluster-robust variance of its own rows,
    :math:`J_v\,\widehat{\mathrm{var}}(S_{vj};\,\mathrm{ddof}=1)/n_v^2`, where
    :math:`S_{vj}` sums the curve over the rows of cluster :math:`j` in fold :math:`v`
    and :math:`J_v` counts the clusters of the fold. That is
    :func:`influence_variance` applied to the fold rows. The fold curve is centred at the
    fold's own plug-in :math:`\hat\psi_v`, so its plug-in part sums to zero inside the
    fold. An uncentred second moment of the :math:`J_v` cluster sums therefore keeps only
    :math:`(J_v-1)/J_v` of the between-cluster variance of that part, and a fold of four
    clusters loses a quarter of it. The centred ``ddof=1`` variance restores that share.
    It also removes the fold mean of the residual part, whose expectation is zero under
    the pooled fluctuation, so it is unbiased for both parts. Rows as units keep the
    uncentred second moment above: there :math:`J_v=n_v`, the loss is :math:`1/n_v`, and
    the construction is Zheng and van der Laan's. Each validation fold must hold at least
    two clusters.

    Parameters
    ----------
    influence_curve : ndarray
        ``(n,)`` fold-specific influence curve. Entry ``i`` is the curve of the fold that
        holds row ``i`` out.
    folds : iterable of ndarray
        Validation index arrays, one per fold -- ``[test for _, test in folds]`` for a
        :class:`~cleverly.learners.crossfit.Folds`.  They must partition the sample.
    cluster : ndarray or None, default=None
        ``(n,)`` cluster codes. Each cluster must lie whole in one fold. ``None`` treats
        the rows as independent.

    Returns
    -------
    float
        Variance of the equally weighted fold estimator.
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
        fold_clusters = int(np.unique(codes[index]).size)
        if fold_clusters < 2:
            raise ValueError(
                f"validation fold {fold_number} holds {fold_clusters} cluster; the "
                "fold-evaluated variance compares cluster totals inside each fold, so every "
                "validation fold needs at least 2 clusters"
            )
        contributions.append(influence_variance(ic[index], codes[index]))
    return float(sum(contributions) / n_folds**2)


def stacked_second_moment_variance(influence_curve: FloatArray) -> float:
    r"""Variance of a stacked CV-TMLE estimate from the raw second moment of its curve.

    .. math::

        \widehat{\mathrm{Var}}(\hat\psi) = \frac{1}{n^2}\sum_{i=1}^{n} \mathrm{IC}_i^2.

    The stacked cross-fitted natural-course mean (its stacked contract) evaluates its point
    and curve on all held-out rows at once, with one pooled fluctuation.  That fluctuation
    solves the stacked score, and the point is the row mean of the targeted predictions,
    so the curve has empirical mean zero to targeting tolerance.  This rule therefore
    equals the centered-rule variance
    :math:`\{n(n-1)\}^{-1}\sum_i(\mathrm{IC}_i-\overline{\mathrm{IC}})^2` times
    ``(n - 1) / n``, and
    ``tests/unit/test_natural_course_crossfit.py`` checks that identity.  The rows are
    weighted equally at ``1/n`` because the point estimate is the row-weighted mean of the
    stacked predictions.  :func:`cross_validated_variance` is the equal-fold-weight
    counterpart, and differs from this whenever the folds are unequal.

    The arithmetic is ``mean(IC**2) / n`` in that association.  The committed stacked study
    artifacts were generated from it, and a different association moves their last bits.

    Parameters
    ----------
    influence_curve : ndarray
        ``(n,)`` stacked influence curve, one entry per held-out row.

    Returns
    -------
    float
        Variance of the estimator.
    """
    ic = np.asarray(influence_curve, dtype=float).reshape(-1)
    n = ic.shape[0]
    if n < 2:
        raise ValueError("need at least 2 observations to estimate a variance")
    return float(np.mean(np.square(ic)) / n)


def stacked_second_moment_covariance(influence_curves: FloatArray) -> FloatArray:
    r"""Joint raw second-moment covariance of several stacked CV-TMLE curves.

    Entry :math:`(j, k)` is :math:`n^{-2}\sum_i \mathrm{IC}_{ij}\mathrm{IC}_{ik}`.  The
    diagonal comes from :func:`stacked_second_moment_variance`, so a one-estimate
    selection returns exactly the stored variance.

    Parameters
    ----------
    influence_curves : ndarray
        ``(n, k)`` influence curves, one column per estimand.

    Returns
    -------
    ndarray
        ``(k, k)`` covariance matrix of the estimators.
    """
    ic = np.asarray(influence_curves, dtype=float)
    if ic.ndim != 2:
        raise ValueError(f"expected an (n, m) matrix of influence curves; got shape {ic.shape}")
    n, k = ic.shape
    covariance = np.empty((k, k), dtype=float)
    for row in range(k):
        covariance[row, row] = stacked_second_moment_variance(ic[:, row])
        for column in range(row):
            value = float(np.mean(ic[:, row] * ic[:, column]) / n)
            covariance[row, column] = value
            covariance[column, row] = value
    return covariance


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


def cluster_sizes(cluster: IntArray) -> IntArray:
    """The number of rows in each distinct cluster, in sorted label order.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster label of each row.

    Returns
    -------
    ndarray of int
        One row count per distinct label.
    """
    _, counts = np.unique(np.asarray(cluster).reshape(-1), return_counts=True)
    return np.asarray(counts, dtype=np.int64)


#: Two cluster weight masses count as equal when they differ by at most this share of the
#: largest one. Weights are normalised to mean one, so the same weights summed in another
#: order differ by a few units in the last place, far below this, and a design that
#: gives clusters different mass differs far above it.
WEIGHT_MASS_RTOL: Final[float] = 1e-9


def cluster_weight_mass(cluster: IntArray, weights: FloatArray) -> FloatArray:
    """The summed observation weight of each distinct cluster, in sorted label order.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster label of each row.
    weights : ndarray of float
        The observation weight of each row.

    Returns
    -------
    ndarray of float
        One weight sum per distinct label, in the order of :func:`cluster_sizes`.
    """
    _, inverse = np.unique(np.asarray(cluster).reshape(-1), return_inverse=True)
    return np.asarray(
        np.bincount(inverse.reshape(-1), weights=np.asarray(weights, dtype=float).reshape(-1)),
        dtype=float,
    )


def unequal_cluster_sizes(cluster: IntArray, weights: FloatArray | None = None) -> bool:
    """Whether the clusters differ in size: in row count, or in weight mass when weighted.

    Equal weight mass makes the weight-weighted and equal-cluster means coincide for
    every possible outcome. Equal row counts do not ensure equal mass, so a weighted
    fit reads both. Two masses count as equal within
    :data:`WEIGHT_MASS_RTOL` of the largest.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster label of each row.
    weights : ndarray of float or None, default None
        The observation weight of each row, or ``None`` for an unweighted fit.

    Returns
    -------
    bool
        ``True`` when two clusters hold different numbers of rows, or different weight
        mass beyond the tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.inference.cluster import unequal_cluster_sizes
    >>> cluster = np.repeat(np.arange(4), 3)
    >>> unequal_cluster_sizes(cluster)
    False
    >>> unequal_cluster_sizes(cluster, weights=np.where(cluster % 2 == 0, 0.5, 2.0))
    True
    """
    counts = cluster_sizes(cluster)
    if counts.size == 0:
        return False
    if int(counts.min()) != int(counts.max()):
        return True
    if weights is None:
        return False
    mass = cluster_weight_mass(cluster, weights)
    return bool(float(np.ptp(mass)) > WEIGHT_MASS_RTOL * float(np.max(np.abs(mass))))


def fewest_clusters(
    cluster: IntArray,
    strata: IntArray | None = None,
    weights: FloatArray | None = None,
) -> int:
    """The fewest clusters with positive weight mass that one estimate reads.

    A fit without strata reads every positive-mass cluster. A fit with baseline strata also reports
    one estimate per stratum. A cluster with zero weight mass contributes nothing to
    that estimate, so it does not count, including inside a stratum.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster label of each row.
    strata : ndarray of int or None, default None
        The baseline stratum code of each row, or ``None`` for a fit without strata.
    weights : ndarray of float or None, default None
        The observation weight of each row. ``None`` counts every cluster.

    Returns
    -------
    int
        The distinct positive-mass cluster count of the whole fit, or the smallest
        positive-mass count within one reported stratum when that is smaller.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.inference.cluster import fewest_clusters
    >>> cluster = np.repeat(np.arange(50), 2)
    >>> fewest_clusters(cluster)
    50
    >>> fewest_clusters(cluster, strata=(cluster < 6).astype(int))
    6
    """
    labels = np.asarray(cluster).reshape(-1)
    positive = (
        np.ones(labels.size, dtype=bool)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1) > 0.0
    )
    count = int(np.unique(labels[positive]).size)
    if strata is None:
        return count
    levels = np.asarray(strata).reshape(-1)
    return min(
        count,
        *(int(np.unique(labels[positive & (levels == level)]).size) for level in np.unique(levels)),
    )


def positive_mass_clause(cluster: IntArray, weights: FloatArray | None) -> str:
    """The summary clause that counts the clusters with positive weight mass.

    A cluster with zero weight mass contributes nothing to any estimate, so the few-cluster
    rule does not count it. A summary prints this clause after the cluster count when the
    two counts differ, so the reader sees the count that the status reads.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster label of each row.
    weights : ndarray of float or None
        The observation weight of each row, or ``None`` for an unweighted fit.

    Returns
    -------
    str
        ``", positive weight mass in N"`` when fewer than all clusters have positive
        weight mass, and the empty string otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.inference.cluster import positive_mass_clause
    >>> cluster = np.repeat(np.arange(4), 2)
    >>> positive_mass_clause(cluster, np.where(cluster == 0, 0.0, 1.0))
    ', positive weight mass in 3'
    >>> positive_mass_clause(cluster, None)
    ''
    """
    if weights is None:
        return ""
    active = fewest_clusters(cluster, weights=weights)
    if active < np.unique(np.asarray(cluster).reshape(-1)).size:
        return f", positive weight mass in {active}"
    return ""


def cluster_inference_status(
    cluster: IntArray | None,
    *,
    strata: IntArray | None = None,
    weights: FloatArray | None = None,
    minimum: int | None = None,
) -> InferenceStatus:
    r"""The inference status the fit's cluster labels, strata, and weights determine.

    ``"few_cluster_plugin"`` when one reported estimate reads fewer distinct clusters with
    positive weight mass than
    :data:`~cleverly._inference_status.MINIMUM_INTERVAL_CLUSTERS`. That is the whole fit,
    or, with baseline strata, any one stratum: :func:`fewest_clusters` gives the count.
    The fit takes one status, so a stratum with too few clusters withholds the interval of
    every estimate. Otherwise ``"influence_curve"``, in sample or cross-fitted, at equal or
    unequal cluster sizes. Below
    :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` clusters an estimate takes a
    Student t reference, which :func:`cluster_reference_df` gives.

    The cluster sizes do not enter. The variance sums the curve within each cluster, and
    the curve already holds the term :math:`-\hat\psi` on every row. The cluster total is
    then :math:`A_j - N_j\hat\psi`, the delta-method curve of the ratio of the mean
    cluster total to the mean cluster size, which is the row-weighted mean.

    Parameters
    ----------
    cluster : ndarray of int or None
        The cluster label of each row, or ``None`` for an unclustered fit.
    strata : ndarray of int or None, default None
        The baseline stratum code of each row, or ``None`` for a fit without strata.
    weights : ndarray of float or None, default None
        The observation weight of each row, or ``None`` for an unweighted fit.
    minimum : int or None, default None
        A floor above :data:`~cleverly._inference_status.MINIMUM_INTERVAL_CLUSTERS`, or
        ``None`` for that floor alone. The cross-fitted
        ``LTMLE`` passes
        :data:`~cleverly._inference_status.MINIMUM_CROSS_FITTED_LONGITUDINAL_CLUSTERS`,
        the smallest count its registered study measures.

    Returns
    -------
    str
        One of :data:`~cleverly.inference.influence.InferenceStatus`.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.inference.cluster import cluster_inference_status
    >>> cluster_inference_status(np.repeat(np.arange(10), 3))
    'influence_curve'
    >>> cluster_inference_status(np.repeat(np.arange(9), 3))
    'few_cluster_plugin'
    >>> equal = np.repeat(np.arange(40), 10)
    >>> cluster_inference_status(equal, strata=(equal < 3).astype(int))
    'few_cluster_plugin'
    """
    if cluster is None:
        return "influence_curve"
    floor = (
        MINIMUM_INTERVAL_CLUSTERS if minimum is None else max(minimum, MINIMUM_INTERVAL_CLUSTERS)
    )
    if fewest_clusters(cluster, strata, weights) < floor:
        return "few_cluster_plugin"
    return "influence_curve"


def cluster_reference_df(
    cluster: IntArray,
    weights: FloatArray | None = None,
    rows: BoolArray | None = None,
) -> int | None:
    """The degrees of freedom of the Student t reference of one clustered estimate.

    ``J - 2``, where ``J`` counts the clusters with positive weight mass among ``rows``,
    when ``J`` is below :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD`.
    ``None``, the normal reference, at that count and above. Nugent et al. (2024),
    Section 2.2, last paragraph, give the rule. Only an inferential fit reads it, and its
    status guarantees ``J`` of at least
    :data:`~cleverly._inference_status.MINIMUM_INTERVAL_CLUSTERS`.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster label of each row.
    weights : ndarray of float or None, default None
        The observation weight of each row. ``None`` counts every cluster.
    rows : ndarray of bool or None, default None
        The rows the estimate reads, such as one baseline stratum. ``None`` reads all.

    Returns
    -------
    int or None
        The degrees of freedom, or ``None`` for the normal reference.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.inference.cluster import cluster_reference_df
    >>> cluster = np.repeat(np.arange(12), 3)
    >>> cluster_reference_df(cluster)
    10
    >>> cluster_reference_df(cluster, rows=cluster < 10)
    8
    >>> cluster_reference_df(np.repeat(np.arange(40), 3)) is None
    True
    """
    labels = np.asarray(cluster).reshape(-1)
    keep = np.ones(labels.size, dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    if weights is not None:
        keep = keep & (np.asarray(weights, dtype=float).reshape(-1) > 0.0)
    count = int(np.unique(labels[keep]).size)
    if count >= FEW_CLUSTER_THRESHOLD:
        return None
    if count < MINIMUM_INTERVAL_CLUSTERS:
        raise ValueError(
            f"a Student t reference needs at least {MINIMUM_INTERVAL_CLUSTERS} clusters with "
            f"positive weight mass; these rows hold {count}. Only a fit whose status supplies "
            "inference reads the reference"
        )
    return count - 2
