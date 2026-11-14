r"""The delete-one-cluster jackknife variance of a clustered fit (X54 part (c)).

``TMLE(cluster_variance="jackknife")`` and ``LTMLE(cluster_variance="jackknife")`` refit the
whole estimator once per positive-mass cluster with that cluster's rows removed: every
nuisance is refitted and the targeting is redone, and a cross-fitted fit keeps the drawn split
less the removed cluster. With :math:`\hat\theta^{-g}` the vector of reported estimates of the
refit without cluster :math:`g`, each on its inference scale (the log of a ratio, the
transform's scale of a transformed estimate), and :math:`\bar\theta` their mean over the
:math:`J` refits, the covariance is

.. math::

    \hat\Sigma_J = \frac{J - 1}{J} \sum_{g=1}^{J}
        (\hat\theta^{-g} - \bar\theta)(\hat\theta^{-g} - \bar\theta)^\top

(Fang, Wang, Hu and Li, arXiv:2510.08438v1, Section 3.3). A reported estimate's variance is its
diagonal entry, and an estimate derived after the fit, a contrast, a ratio or a transform,
carries its gradient over the reported estimates through the delta method, so its variance is
:math:`\lambda^\top\hat\Sigma_J\lambda`. The interval keeps the fit's reference: a Student t
with :math:`J - 2` degrees of freedom below 40 clusters, the normal from 40.

**Why it is a natural extension, and where it stops.** With every nuisance a finite-dimensional
model fitted by smooth estimating equations (a GLM, a Super Learner of GLMs, or a declared
known mechanism), a TMLE with a fixed number of logistic fluctuation steps is an M-estimator
over independent clusters. A one-step expansion of its stacked estimating equations gives
:math:`\hat\theta^{-g} - \hat\theta = -\tilde\psi_g/(J - 1) + o_P(J^{-1})`, with
:math:`\tilde\psi_g` the cluster-level influence function of the stack, nuisance-estimation
terms included, so :math:`\hat\Sigma_J` equals the cluster variance of :math:`\tilde\psi` to
first order and the interval has the influence-curve interval's limit. With an estimated,
correctly specified parametric mechanism and a misspecified outcome regression the
influence-curve variance of the efficient curve is conservative, while the jackknife reads the
actual variance. A data-adaptive learner needs a stability bound on its leave-one-cluster-out
refits that no source read here gives, so the estimators refuse it before any learner. A
cross-fitted fit holds the split fixed, which needs every fold to keep a positive-mass cluster
after any removal, so a split with a fold of fewer than two is refused too.

A refit that raises, or whose estimate takes a parameter status such as a constant arm cell
(:func:`refit_withheld`), leaves the jackknife of that estimate undefined. A refit on
:math:`J - 1` clusters may fall under a cluster-count status of its own; that says nothing about
its point value and does not count. The estimate then takes ``"jackknife_refit_plugin"``
and withholds its interval; :attr:`JackknifeVariance.failed_refits` names the clusters and
:attr:`JackknifeVariance.failures` what happened.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import numpy as np

from .._inference_status import (
    JACKKNIFE_REFIT_STATUS,
    OUT_OF_RANGE_STATUS,
    PARAMETER_STATUSES,
    supplies_inference,
)
from .._refusal import Refusal, RefusalReason
from .._typing import FloatArray, IntArray
from ..learners.crossfit import Folds

__all__ = [
    "CLUSTER_VARIANCES",
    "JACKKNIFE_LEARNER_REFUSAL",
    "ClusterVariance",
    "JackknifeCovariance",
    "JackknifeVariance",
    "jackknife_covariance",
    "jackknife_estimates",
    "jackknife_fold_refusal",
    "jackknife_learner",
    "jackknife_learner_refusal",
    "jackknife_summary_line",
    "leave_one_cluster_out",
    "positive_mass_clusters",
    "refit_folds",
    "refit_withheld",
    "with_jackknife",
]

#: The variance of a clustered estimate: the influence curve's, or the cluster jackknife's.
ClusterVariance = Literal["influence_curve", "jackknife"]

#: The accepted values of ``cluster_variance=``.
CLUSTER_VARIANCES: Final[tuple[str, ...]] = ("influence_curve", "jackknife")

#: The R5 refusal of a nuisance the jackknife has no linearization for. ``{what}`` names it.
JACKKNIFE_LEARNER_REFUSAL: Final[str] = (
    "cluster_variance='jackknife' refits every nuisance with one cluster left out and redoes "
    "the targeting. Its variance agrees with the influence-curve variance to first order when "
    "each nuisance is a finite-dimensional model fitted by smooth estimating equations: a "
    "GLM, a Super Learner of GLMs, or a declared known mechanism. {what} is not, and no result "
    "read here shows that leave-one-cluster-out refits of {what} linearize. Fang et al. "
    "(arXiv:2510.08438, Section 3.3) use Cox models only and prove no result."
)

#: The alternative every R5 refusal names.
JACKKNIFE_ALTERNATIVE: Final[str] = (
    "Use the default cluster variance (cluster_variance='influence_curve'), or pass GLM learners."
)

#: The R5 refusal of a cross-fitted split with a fold of fewer than two positive-mass clusters.
JACKKNIFE_FOLD_REFUSAL: Final[str] = (
    "cluster_variance='jackknife' on a cross-fitted fit holds the drawn split fixed and removes "
    "one cluster from its fold. With {clusters} clusters in {folds} folds, fold {fold} holds "
    "{held} positive-mass cluster(s), so a refit would lose that fold and change the estimator "
    "it is meant to resample."
)


def jackknife_learner_refusal(what: str) -> Refusal:
    """The R5 refusal for a nuisance or a construction the jackknife does not cover.

    Parameters
    ----------
    what : str
        The first offending learner or construction, for example ``"the outcome learner
        HistGradientBoostingClassifier"``.

    Returns
    -------
    Refusal
        Code ``inference_gap``, reopened by F28.
    """
    return Refusal(
        RefusalReason.INFERENCE_GAP,
        JACKKNIFE_LEARNER_REFUSAL.format(what=what),
        alternative=JACKKNIFE_ALTERNATIVE,
        roadmap="F28",
    )


def jackknife_summary_line(estimates: Mapping[str, Any]) -> str | None:
    """The Method details line of a ``cluster_variance="jackknife"`` fit, or ``None``.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The fit's estimates.

    Returns
    -------
    str or None
        The line, naming the refit count and any cluster whose refit failed, or ``None`` when
        no estimate carries a jackknife record.
    """
    records = [item.jackknife for item in estimates.values() if item.jackknife is not None]
    if not records:
        return None
    covariance = records[0].covariance
    failed = sorted({code for record in records for code in record.failed_refits})
    line = (
        "Variance: the delete-one-cluster jackknife, from "
        f"{len(covariance.clusters)} refits that each leave out one cluster's rows and "
        "refit every model. Each standard error is its square root, read against the same "
        "reference as the influence-curve standard error."
    )
    if failed:
        line += (
            f" The refits without cluster(s) {', '.join(str(code) for code in failed)} "
            "failed, so the estimates that read them show no interval."
        )
    return line


def jackknife_learner(learner: Any) -> bool:
    """Whether the jackknife covers ``learner``: a parametric learner or a Super Learner of them.

    The parametric classes are those of :func:`cleverly._support.learner_family`: the exact
    classes ``LogisticRegression``, ``LinearRegression`` and :class:`~cleverly.learners.GLM`,
    and any class that declares itself parametric. The default library (``None``) is not.

    Parameters
    ----------
    learner : object or None
        The learner as passed to the estimator.

    Returns
    -------
    bool
        ``True`` when the R5 learner rule admits it.
    """
    from .._support import learner_family
    from ..learners.super_learner import SuperLearner

    if isinstance(learner, SuperLearner):
        library = learner.library
        candidates = library.values() if isinstance(library, Mapping) else library
        items = list(candidates or ())
        return bool(items) and all(learner_family(item) == "parametric" for item in items)
    return learner_family(learner) == "parametric"


def positive_mass_clusters(cluster: IntArray, weights: FloatArray | None) -> IntArray:
    """The cluster codes with positive total weight, sorted.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster code of each row.
    weights : ndarray of float or None
        The observation weights, or ``None`` for unweighted rows.

    Returns
    -------
    ndarray of int
        The codes the jackknife leaves out, one refit each.
    """
    codes = np.asarray(cluster)
    labels = np.unique(codes)
    if weights is None:
        return labels
    mass = np.bincount(np.searchsorted(labels, codes), weights=np.asarray(weights, dtype=float))
    return labels[mass > 0.0]


def jackknife_fold_refusal(
    cluster: IntArray, weights: FloatArray | None, assignment: IntArray, n_folds: int
) -> Refusal | None:
    """The R5 fold refusal for a drawn split, or ``None`` when every fold keeps two clusters.

    The trigger reads the split, not ``J < 2V``: a stratified split need not balance the
    cluster counts.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster code of each row.
    weights : ndarray of float or None
        The observation weights, or ``None``.
    assignment : ndarray of int
        The fold of each row.
    n_folds : int
        The fold count of the split.

    Returns
    -------
    Refusal or None
        The refusal naming the first fold that holds fewer than two positive-mass clusters.
    """
    if n_folds < 2:
        return None
    kept = set(positive_mass_clusters(cluster, weights).tolist())
    codes = np.asarray(cluster)
    folds = np.asarray(assignment)
    for fold in range(n_folds):
        held = len(kept & set(np.unique(codes[folds == fold]).tolist()))
        if held < 2:
            return Refusal(
                RefusalReason.INFERENCE_GAP,
                JACKKNIFE_FOLD_REFUSAL.format(
                    clusters=len(kept), folds=n_folds, fold=fold, held=held
                ),
                alternative=(
                    f"Use at least {2 * n_folds} clusters with unstratified folds, fewer folds "
                    f"(n_folds <= {len(kept) // 2}), or the default cluster variance."
                ),
                roadmap="F28",
            )
    return None


def leave_one_cluster_out(cluster: IntArray, code: int) -> IntArray:
    """The rows a refit keeps when cluster ``code`` is left out.

    Parameters
    ----------
    cluster : ndarray of int
        The cluster code of each row.
    code : int
        The cluster to remove.

    Returns
    -------
    ndarray of int
        The kept row indices, in order.
    """
    return np.flatnonzero(np.asarray(cluster) != code)


def refit_folds(folds: Folds, keep: IntArray) -> Folds:
    """The drawn split restricted to the kept rows, with no redraw.

    Parameters
    ----------
    folds : Folds
        The full fit's split.
    keep : ndarray of int
        The rows the refit keeps.

    Returns
    -------
    Folds
        The kept rows' labels, with the same fold count and no relabelling. A fold the
        removal empties raises :class:`~cleverly.exceptions.DataError` rather than shrinking
        the split, and a split :func:`jackknife_fold_refusal` admits never empties one.
    """
    if folds.is_single:
        return Folds.single(int(np.asarray(keep).size))
    labels = np.asarray(folds.assignment)[np.asarray(keep)]
    return Folds(labels.astype(np.int64), int(folds.n_folds))


def jackknife_covariance(leave_one_out: FloatArray) -> FloatArray:
    r"""The jackknife covariance of a ``(J, k)`` array of leave-one-out estimates.

    :math:`\frac{J-1}{J}\sum_g(\hat\theta^{-g}-\bar\theta)(\hat\theta^{-g}-\bar\theta)^\top`.

    Parameters
    ----------
    leave_one_out : ndarray
        ``(J, k)``: row ``g`` holds the ``k`` estimates of the refit without cluster ``g``.

    Returns
    -------
    ndarray
        ``(k, k)`` covariance.

    Examples
    --------
    For a mean of cluster totals the jackknife variance is the cluster variance over ``J``:

    >>> import numpy as np
    >>> from cleverly.inference.cluster_jackknife import jackknife_covariance
    >>> totals = np.array([1.0, 2.0, 4.0, 7.0])
    >>> loo = np.array([np.delete(totals, g).mean() for g in range(4)])[:, None]
    >>> bool(np.isclose(jackknife_covariance(loo)[0, 0], totals.var(ddof=1) / 4, rtol=1e-12))
    True
    """
    values = np.asarray(leave_one_out, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    clusters = values.shape[0]
    if clusters < 2:
        raise ValueError(f"the jackknife needs at least two leave-one-out refits; got {clusters}")
    centred = values - values.mean(axis=0)
    return np.asarray((clusters - 1) / clusters * (centred.T @ centred), dtype=float)


@dataclass(frozen=True, eq=False)
class JackknifeCovariance:
    r"""The jackknife covariance of one fit's reported estimates, shared by every estimate.

    Parameters
    ----------
    names : tuple of str
        The reported estimates whose refit values the covariance reads, in its row order.
    matrix : ndarray
        ``(k, k)`` :math:`\hat\Sigma_J` on each estimate's inference scale.
    leave_one_out : ndarray
        ``(J, k)`` the refit values the matrix is computed from.
    clusters : ndarray of int
        ``(J,)`` the cluster code each row of :attr:`leave_one_out` leaves out.
    failures : tuple of tuple
        ``(code, what)`` for each refit that raised, ``what`` naming the error.
    """

    names: tuple[str, ...]
    matrix: FloatArray
    leave_one_out: FloatArray
    clusters: IntArray
    failures: tuple[tuple[int, str], ...] = ()

    def quadratic(self, gradient: FloatArray) -> float:
        r""":math:`\lambda^\top\hat\Sigma_J\lambda`, ``nan`` when an entry it reads is not finite.

        Parameters
        ----------
        gradient : ndarray
            ``(k,)`` in the order of :attr:`names`.

        Returns
        -------
        float
            The jackknife variance of the functional.
        """
        weights = np.asarray(gradient, dtype=float).reshape(-1)
        used = weights != 0.0
        block = self.matrix[np.ix_(used, used)]
        if not np.all(np.isfinite(block)):
            return float("nan")
        return float(weights[used] @ block @ weights[used])

    def same_as(self, other: JackknifeCovariance) -> bool:
        """Whether ``other`` is this fit's covariance: the same object, or equal content.

        Parameters
        ----------
        other : JackknifeCovariance
            The covariance of another estimate.

        Returns
        -------
        bool
            ``True`` when the names and the matrix agree.
        """
        return other is self or (
            other.names == self.names
            and bool(np.array_equal(other.matrix, self.matrix, equal_nan=True))
        )


@dataclass(frozen=True)
class JackknifeVariance:
    r"""Both variances of one estimate under ``cluster_variance="jackknife"``.

    Parameters
    ----------
    covariance : JackknifeCovariance
        The fit's shared covariance of its reported estimates.
    gradient : ndarray
        ``(k,)`` :math:`\lambda`, the estimate's gradient on its inference scale with respect
        to the reported estimates of :attr:`covariance`: a unit vector for a reported
        estimate.
    influence_variance : float
        :math:`\hat\sigma^2_{IC}`, the variance of the estimate's influence curve, kept as a
        diagnostic.
    failed_refits : tuple of int
        The cluster codes whose refit raised or withheld one of the estimates the gradient
        reads. Non-empty means the jackknife of this estimate is undefined.

    Attributes
    ----------
    jackknife_variance : float
    failures : tuple of tuple
    """

    covariance: JackknifeCovariance
    gradient: FloatArray
    influence_variance: float
    failed_refits: tuple[int, ...] = field(default=())

    @property
    def jackknife_variance(self) -> float:
        r""":math:`\lambda^\top\hat\Sigma_J\lambda`, ``nan`` when a refit failed."""
        if self.failed_refits:
            return float("nan")
        return self.covariance.quadratic(self.gradient)

    @property
    def failures(self) -> tuple[tuple[int, str], ...]:
        """``(code, what)`` for each failed refit this estimate reads."""
        failed = set(self.failed_refits)
        recorded = dict(self.covariance.failures)
        return tuple((code, recorded.get(code, "withheld its interval")) for code in sorted(failed))


def refit_withheld(status: str) -> bool:
    """Whether a refit's estimate under ``status`` leaves the jackknife undefined.

    A parameter status (a constant arm cell, a constant node, a failed stack) says the point
    value of the refit is not the estimator's, so the jackknife of that estimate is undefined.
    A fit-level status, such as the cluster-count statuses a refit on ``J - 1`` clusters
    takes, says nothing about the point value, and an estimate on or outside its range keeps
    a finite value, so neither counts.

    Parameters
    ----------
    status : str
        The refit estimate's inference status.

    Returns
    -------
    bool
        ``True`` for a parameter status other than ``"out_of_range_plugin"``.
    """
    return status in PARAMETER_STATUSES and status != OUT_OF_RANGE_STATUS


def jackknife_estimates(
    estimates: Mapping[str, Any],
    refits: Sequence[tuple[int, Mapping[str, Any] | None, str]],
    *,
    support_of: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Each reported estimate with its jackknife variance, or withheld where a refit failed.

    Parameters
    ----------
    estimates : Mapping of str to ParameterEstimate
        The full-sample estimates, in report order.
    refits : sequence of tuple
        One ``(code, estimates, what)`` per left-out cluster: the refit's estimates, or
        ``None`` with ``what`` naming the error when the refit raised.
    support_of : callable or None
        The support record of the fit at a status, for an estimate that takes the withheld
        refit status. ``None`` keeps each estimate's record.

    Returns
    -------
    dict of str to ParameterEstimate
        The estimates. One that supplies inference and whose every refit gives a finite value
        (:func:`refit_withheld`) carries :class:`JackknifeVariance` and stores its jackknife
        variance. One whose refit failed
        takes ``"jackknife_refit_plugin"``. One that supplies no inference is unchanged.
    """
    names = tuple(name for name, estimate in estimates.items() if estimate.supplies_inference)
    codes = np.array([code for code, _, _ in refits], dtype=np.int64)
    values = np.full((len(refits), len(names)), np.nan)
    failed: dict[str, list[int]] = {name: [] for name in names}
    raised: list[tuple[int, str]] = []
    for row, (code, refit, what) in enumerate(refits):
        if refit is None:
            raised.append((int(code), what))
            for name in names:
                failed[name].append(int(code))
            continue
        for column, name in enumerate(names):
            estimate = refit.get(name)
            value = None if estimate is None else float(estimate.inference_value)
            if (
                estimate is None
                or refit_withheld(estimate.inference)
                or value is None
                or not np.isfinite(value)
            ):
                failed[name].append(int(code))
                continue
            values[row, column] = value
    matrix = np.full((len(names), len(names)), np.nan)
    complete = [column for column, name in enumerate(names) if not failed[name]]
    if complete and len(refits) >= 2:
        block = jackknife_covariance(values[:, complete])
        matrix[np.ix_(complete, complete)] = block
    shared = JackknifeCovariance(
        names=names,
        matrix=matrix,
        leave_one_out=values,
        clusters=codes,
        failures=tuple(raised),
    )
    out: dict[str, Any] = {}
    for name, estimate in estimates.items():
        if name not in names:
            out[name] = estimate
            continue
        gradient = np.zeros(len(names))
        gradient[names.index(name)] = 1.0
        record = JackknifeVariance(
            covariance=shared,
            gradient=gradient,
            influence_variance=float(
                estimate.variance if estimate.node is None else estimate.node.influence_variance
            ),
            failed_refits=tuple(failed[name]),
        )
        out[name] = with_jackknife(estimate, record, support_of=support_of)
    return out


def with_jackknife(
    estimate: Any, record: JackknifeVariance, *, support_of: Callable[[str], Any] | None = None
) -> Any:
    """``estimate`` carrying ``record``: its variance, or the withheld refit status.

    Parameters
    ----------
    estimate : ParameterEstimate
        An estimate that supplies inference.
    record : JackknifeVariance
        Its jackknife record.
    support_of : callable or None
        The support record of the fit at a status, read when the estimate takes the withheld
        refit status.

    Returns
    -------
    ParameterEstimate
        The estimate with :attr:`~ParameterEstimate.jackknife` set. Its variance is the
        jackknife variance, plus the boundary-node supplement when it carries one. When a
        refit failed it takes ``"jackknife_refit_plugin"``, keeps its curve variance as the
        plug-in diagnostic and drops its t reference.
    """
    if record.failed_refits:
        status = JACKKNIFE_REFIT_STATUS
        return replace(
            estimate,
            jackknife=record,
            inference=status,
            reference_df=estimate.reference_df if supplies_inference(status) else None,
            support=estimate.support if support_of is None else support_of(status),
        )
    variance = record.jackknife_variance
    if estimate.node is not None:
        variance += estimate.node.node_variance
    return replace(estimate, jackknife=record, variance=float(variance))
