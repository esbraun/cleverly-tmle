"""Full-refit bootstrap inference.

The influence-curve variance is the right default: it is cheap and, when the
nuisance estimators converge fast enough, correct.  It is also asymptotic, and it
can be optimistic in exactly the situations practitioners care about -- small
samples, weak overlap, a heavily targeted fit.  R's ``tmle`` therefore offers
``B > 1`` for a bootstrap, and so does this library.

The bootstrap here is a *full refit*: every replicate re-runs the whole procedure,
nuisance fits included.  Bootstrapping only the targeting step while holding the
nuisance fits fixed would understate the variance, because the nuisance
estimation is itself a source of uncertainty.  The name states that procedure: a
targeted bootstrap is a distinct construction. docs/roadmap.md X55 holds the HAL-TMLE
bootstrap, and F2 holds a targeted bootstrap for other learners.

With clusters, whole clusters are resampled. Resampling rows would destroy the
dependence structure the cluster variance exists to account for. Each sampled
occurrence receives a distinct cluster code, including repeated draws of one source
cluster.

A replicate that cannot run is dropped rather than redrawn, and :attr:`BootstrapResult.n_failed`
counts the drops.  A cross-fitted replicate draws its own split and puts it through the same
preflight an ordinary fit does, so a resample that leaves a treatment arm or an outcome class
out of some training complement is refused there, before any learner.  Redrawing until a
replicate succeeded would select the resamples by the values the split must not read, so the
reported interval is conditional on the replicates that ran and the count says how many did.
Under a rare arm that count is worth reading: the dropped replicates are the ones the arm is
rarest in, so the surviving spread is narrower than the unconditional one.

Observation weights are resampled with their rows and renormalised within each replicate,
which is what keeps every replicate aimed at the same tilted parameter (see
:mod:`cleverly.data.weighting`).  What the bootstrap does *not* do is re-derive the
weights: a replicate inherits the numbers it was given, so if the weights came out of a
fitted selection or calibration model, these intervals condition on that fit exactly as
the influence-curve ones do.  Re-deriving the weights inside each replicate would need
the model that produced them, which the package never sees.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeVar

import numpy as np

from .._support import UNCLASSIFIED, Support
from .._typing import FloatArray, IntArray
from .._warnings import suppressed as _suppressed
from .._warnings import warn as _warn
from ..utils.parallel import map_parallel
from .delta import normal_cdf, normal_quantile
from .influence import BootstrapInterval, BootstrapSummary

__all__ = [
    "BootstrapBand",
    "BootstrapInterval",
    "BootstrapResult",
    "bca_acceleration",
    "bca_interval",
    "bootstrap_band",
    "bootstrap_indices",
    "cluster_members",
    "run_bootstrap",
    "run_index_bootstrap",
    "studentized_critical_value",
]


class Resamplable(Protocol):
    """A data container the bootstrap can resample: a row count, clusters and a subset.

    :class:`~cleverly.data.CausalData` and
    :class:`~cleverly.longitudinal.LongitudinalData` both satisfy it.
    """

    @property
    def n(self) -> int: ...

    @property
    def cluster(self) -> IntArray | None: ...

    def subset(self, index: Any) -> Any: ...


_DataT = TypeVar("_DataT", bound=Resamplable)

#: One replicate's refit: each estimand's alias mapped to ``(estimate, standard error)``.
#: The standard error is the replicate fit's own influence-curve standard error on the
#: inference scale.  A studentized band divides by it, so a refit that returns a bare
#: estimate is refused with a :class:`TypeError`.
RefitFn = Callable[[Any], Mapping[str, tuple[float, float]]]
#: A refit that reads the replicate's row index of the original data instead of a
#: materialized resample.  It returns what :data:`RefitFn` returns.
IndexRefitFn = Callable[[IntArray], Mapping[str, tuple[float, float]]]
Resampling = Literal["auto", "iid", "cluster"]
#: The text a refit that breaks the :data:`RefitFn` contract is refused with.
REFIT_CONTRACT = (
    "a bootstrap refit must map each estimand to (estimate, standard error); the standard "
    "error is the replicate fit's own influence-curve standard error"
)


@dataclass(frozen=True)
class _BootstrapDraw:
    """One reproducible draw from a resolved bootstrap design."""

    replicate: int
    seed: int
    sequence: np.random.SeedSequence


@dataclass(frozen=True)
class _BootstrapDesign:
    """Resolved bootstrap samples shared by inference and refit assessments."""

    draws: tuple[_BootstrapDraw, ...]
    resampling: Literal["iid", "cluster"]
    cluster: IntArray | None
    members: tuple[IntArray, ...] | None

    def indices(self, data: _DataT, draw: _BootstrapDraw) -> tuple[IntArray, IntArray | None]:
        """Return the row positions of one draw, and the distinct cluster occurrence codes."""
        return _bootstrap_draw(
            data.n,
            self.cluster,
            np.random.default_rng(draw.sequence),
            None if self.members is None else list(self.members),
        )

    def sample(self, data: _DataT, draw: _BootstrapDraw) -> _DataT:
        """Materialize one sample with distinct codes for cluster occurrences."""
        index, occurrence_codes = self.indices(data, draw)
        sample = data.subset(index)
        if occurrence_codes is None:
            return sample
        return replace(sample, cluster=occurrence_codes)


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap draws for every estimand, plus how many replicates failed.

    Parameters
    ----------
    draws : dict of str to ndarray
        One array of replicate estimates per estimand alias.
    n_requested : int
        Replicates asked for.
    n_failed : int
        Replicates that raised and were dropped. Weak overlap can leave a resample
        with an empty treatment arm in some stratum, and a cross-fitted replicate is
        also refused by the fold preflight when its own draw leaves an arm or an
        outcome class out of a training complement. Either way the interval below is
        *conditional* on the replicates that ran, and this count is how far it is from
        the one that was asked for.
    resampling : str
        Whether rows or clusters were resampled.
    std_errors : dict of str to ndarray, default={}
        One array of replicate standard errors per estimand alias, aligned with
        ``draws``. Each entry is the replicate fit's own influence-curve standard error.
        Empty for draws derived after the fit, whose replicates carry no curve.
    """

    draws: dict[str, FloatArray]
    n_requested: int
    n_failed: int
    resampling: str
    std_errors: dict[str, FloatArray] = field(default_factory=dict)

    def summary(
        self,
        name: str,
        alpha: float = 0.05,
        *,
        interval: BootstrapInterval = "percentile",
        estimate: float | None = None,
        acceleration: float | None = None,
    ) -> BootstrapSummary:
        """The bootstrap interval and standard error for one estimand.

        The percentile interval takes ``np.quantile`` of the finite draws. It is the
        default: the point of running a bootstrap here is usually that the sampling
        distribution is skewed, and a symmetric interval would throw that information
        away. The BCa interval (:func:`bca_interval`) also corrects the median bias and
        the skew of the draws.

        Parameters
        ----------
        name : str
            Estimand alias to summarise.
        alpha : float
            Significance level of the interval.
        interval : {"percentile", "bca"}, default="percentile"
            Which interval to compute.
        estimate : float or None, default=None
            The full-sample estimate, which the BCa bias correction compares the draws
            with. Required with ``interval="bca"``.
        acceleration : float or None, default=None
            The BCa acceleration, from :func:`bca_acceleration`. Required with
            ``interval="bca"``.

        Returns
        -------
        BootstrapSummary
            The interval and the bootstrap standard error for that estimand.

        Raises
        ------
        ValueError
            When ``interval`` is not one of the two, or ``"bca"`` lacks ``estimate`` or
            ``acceleration``.
        """
        if interval not in ("percentile", "bca"):
            raise ValueError(f"interval must be 'percentile' or 'bca'; got {interval!r}")
        if interval == "bca" and (estimate is None or acceleration is None):
            raise ValueError("a BCa interval needs the full-sample estimate and acceleration")
        draws = self.draws[name]
        finite = draws[np.isfinite(draws)]
        nan = float("nan")
        if finite.size < 2:
            return BootstrapSummary(
                std_error=nan,
                ci=(nan, nan),
                ci_one_sided_lower=nan,
                ci_one_sided_upper=nan,
                n_replicates=int(finite.size),
                n_failed=self.n_failed,
                draws=finite,
                interval=interval,
            )
        spread = float(np.std(finite, ddof=1))
        if interval == "bca":
            assert estimate is not None and acceleration is not None
            low, high, z0 = bca_interval(finite, estimate, acceleration, alpha)
            one_lower, _, _ = bca_interval(finite, estimate, acceleration, 2.0 * alpha)
            _, one_upper, _ = bca_interval(finite, estimate, acceleration, 2.0 * alpha)
            return BootstrapSummary(
                std_error=spread,
                ci=(low, high),
                ci_one_sided_lower=one_lower,
                ci_one_sided_upper=one_upper,
                n_replicates=int(finite.size),
                n_failed=self.n_failed,
                draws=finite,
                interval="bca",
                z0=z0,
                acceleration=float(acceleration),
            )
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        return BootstrapSummary(
            std_error=spread,
            ci=(float(low), float(high)),
            ci_one_sided_lower=float(np.quantile(finite, alpha)),
            ci_one_sided_upper=float(np.quantile(finite, 1.0 - alpha)),
            n_replicates=int(finite.size),
            n_failed=self.n_failed,
            draws=finite,
        )


def cluster_sums(curve: FloatArray, cluster: IntArray | None) -> FloatArray:
    """The influence curve summed within each resampled unit.

    A cluster bootstrap draws whole clusters, so the unit of its acceleration and of its
    influence measures is the cluster sum of the row curve.

    Parameters
    ----------
    curve : ndarray
        ``(n,)`` row influence curve.
    cluster : ndarray or None
        ``(n,)`` cluster codes, or ``None`` for independent rows.

    Returns
    -------
    ndarray
        ``curve`` for ``None``, else one sum per cluster in sorted cluster order.
    """
    values = np.asarray(curve, dtype=float).reshape(-1)
    if cluster is None:
        return values
    _, codes = np.unique(np.asarray(cluster).reshape(-1), return_inverse=True)
    return np.asarray(np.bincount(codes, weights=values), dtype=np.float64)


def bca_acceleration(influence_curve: FloatArray, cluster: IntArray | None = None) -> float:
    r"""The BCa acceleration of an estimate, from its own influence curve.

    :math:`\hat a = \sum_i D_i^3 / (6 (\sum_i D_i^2)^{3/2})`, with :math:`D_i` the
    centred curve of the resampled unit. It equals the skewness of the curve over
    :math:`6\sqrt n`, which is the infinitesimal-jackknife acceleration of Efron (1987) that
    Coyle's thesis (pp. 11-13) applies to a targeted estimate. A positive multiple of the
    curve gives the same value, so a ratio's curve on the log scale and on the ratio scale
    agree. A cluster bootstrap resamples clusters, so with ``cluster`` the curve is first
    summed within each cluster.

    Parameters
    ----------
    influence_curve : ndarray
        ``(n,)`` influence curve of the estimate.
    cluster : ndarray or None, default=None
        ``(n,)`` cluster codes when the bootstrap resamples clusters.

    Returns
    -------
    float
        The acceleration, ``0.0`` for a curve with no spread.
    """
    curve = cluster_sums(influence_curve, cluster)
    centred = curve - curve.mean()
    second = float(np.sum(centred**2))
    if second <= 0.0:
        return 0.0
    return float(np.sum(centred**3) / (6.0 * second**1.5))


def bca_bias_correction(draws: FloatArray, estimate: float) -> float:
    r"""The BCa bias correction :math:`z_0 = \Phi^{-1}(\#\{\psi^*_b < \hat\psi\} / B)`.

    A draw equal to the estimate counts for half, the usual convention for tied draws.

    Parameters
    ----------
    draws : ndarray
        The finite replicate estimates.
    estimate : float
        The full-sample estimate.

    Returns
    -------
    float
        :math:`z_0`, infinite when every draw lies on one side of the estimate.
    """
    values = np.asarray(draws, dtype=float)
    below = np.count_nonzero(values < estimate) + 0.5 * np.count_nonzero(values == estimate)
    return normal_quantile(float(below) / values.size)


#: Rounding slack on ``level (B + 1)``. A level that is an integer rank in exact arithmetic,
#: such as ``Phi(Phi^-1(0.025)) * 200 = 5``, can land a few ulp below it after the round trip.
_RANK_SLACK = 1e-9


def _lower_rank(position: float) -> int:
    """``floor(position)``, treating a position within the slack of an integer as that integer."""
    return math.floor(position + _RANK_SLACK)


def _upper_rank(position: float) -> int:
    """``ceil(position)``, treating a position within the slack of an integer as that integer."""
    return math.ceil(position - _RANK_SLACK)


def _lower_order_statistic(ordered: FloatArray, level: float) -> float:
    """The ``floor(level (B + 1))``-th order statistic, clamped to ``1 .. B``."""
    rank = min(max(_lower_rank(level * (ordered.size + 1)), 1), ordered.size)
    return float(ordered[rank - 1])


def _upper_order_statistic(ordered: FloatArray, level: float) -> float:
    """The ``ceil(level (B + 1))``-th order statistic, clamped to ``1 .. B``."""
    rank = min(max(_upper_rank(level * (ordered.size + 1)), 1), ordered.size)
    return float(ordered[rank - 1])


def bca_interval(
    draws: FloatArray, estimate: float, acceleration: float, alpha: float
) -> tuple[float, float, float]:
    r"""The two-sided BCa interval of finite bootstrap draws.

    The bias correction is :math:`z_0 = \Phi^{-1}(\#\{\psi^*_b < \hat\psi\} / B)`. Each
    limit is read at the level
    :math:`\Phi\bigl(z_0 + (z_0 + z_q) / (1 - \hat a (z_0 + z_q))\bigr)` for
    :math:`q = \alpha/2` and :math:`1 - \alpha/2`. The lower limit is the
    :math:`\lfloor \text{level}\,(B + 1) \rfloor`-th order statistic and the upper limit the
    :math:`\lceil \text{level}\,(B + 1) \rceil`-th, so each limit is a replicate value and
    the interval does not narrow by interpolation.

    Parameters
    ----------
    draws : ndarray
        The finite replicate estimates.
    estimate : float
        The full-sample estimate.
    acceleration : float
        The acceleration :math:`\hat a` (:func:`bca_acceleration`).
    alpha : float
        Significance level of the two-sided interval.

    Returns
    -------
    tuple of float
        The lower limit, the upper limit and :math:`z_0`. The limits are ``nan`` when no
        draw lies on one side of the estimate (so :math:`z_0` is infinite) or when
        :math:`1 - \hat a (z_0 + z_q)` is not positive.
    """
    ordered = np.sort(np.asarray(draws, dtype=float))
    nan = float("nan")
    z0 = bca_bias_correction(ordered, estimate)
    if not math.isfinite(z0):
        return nan, nan, z0
    levels = []
    for quantile in (alpha / 2.0, 1.0 - alpha / 2.0):
        shifted = z0 + normal_quantile(quantile)
        denominator = 1.0 - acceleration * shifted
        if denominator <= 0.0:
            return nan, nan, z0
        levels.append(normal_cdf(z0 + shifted / denominator))
    return (
        _lower_order_statistic(ordered, levels[0]),
        _upper_order_statistic(ordered, levels[1]),
        z0,
    )


@dataclass(frozen=True)
class BootstrapBand:
    r"""A studentized bootstrap sup-t band over several estimands.

    Parameters
    ----------
    critical_value : float
        The :math:`\lceil (1 - \alpha)(B + 1) \rceil`-th order statistic of the replicate
        maxima :math:`\max_j |\psi^*_{bj} - \hat\psi_j| / SE^*_{bj}`.
    bands : dict of str to tuple of float
        One joint interval per estimand alias, :math:`\hat\psi_j \pm c^* \widehat{SE}_j`.
    alpha : float
        Joint significance level.
    n_replicates : int
        Replicates whose every estimate and standard error was finite and positive.
    support : Support, default=UNCLASSIFIED
        Whether a registered row measures the band for the fit. A row that lists the
        ``bootstrap_band`` kind, with every banded estimate supported on it, makes it
        ``supported``. Every other band is ``experimental``.
    """

    critical_value: float
    bands: dict[str, tuple[float, float]]
    alpha: float
    n_replicates: int
    support: Support = UNCLASSIFIED


def studentized_critical_value(
    estimates: FloatArray, draws: FloatArray, std_errors: FloatArray, alpha: float
) -> tuple[float, int]:
    r"""The studentized sup-t critical value of bootstrap draws.

    Each replicate gives :math:`T_b = \max_j |\psi^*_{bj} - \hat\psi_j| / SE^*_{bj}`, with
    :math:`SE^*_{bj}` the replicate fit's own standard error. The critical value is the
    :math:`\lceil (1 - \alpha)(B + 1) \rceil`-th order statistic of the :math:`T_b`.

    Parameters
    ----------
    estimates : ndarray
        ``(J,)`` full-sample estimates.
    draws : ndarray
        ``(B, J)`` replicate estimates.
    std_errors : ndarray
        ``(B, J)`` replicate standard errors.
    alpha : float
        Joint significance level.

    Returns
    -------
    tuple
        The critical value, and the number of replicates it read. A replicate with a
        missing estimate or a standard error that is not positive is not read. The value
        is ``nan`` when the order statistic lies beyond the replicates read.
    """
    psi = np.asarray(estimates, dtype=float).reshape(1, -1)
    values = np.asarray(draws, dtype=float)
    scale = np.asarray(std_errors, dtype=float)
    usable = np.all(np.isfinite(values) & np.isfinite(scale) & (scale > 0.0), axis=1)
    maxima = np.sort(np.max(np.abs(values[usable] - psi) / scale[usable], axis=1))
    rank = _upper_rank((1.0 - alpha) * (maxima.size + 1))
    if maxima.size == 0 or rank > maxima.size:
        return float("nan"), int(maxima.size)
    return float(maxima[rank - 1]), int(maxima.size)


def bootstrap_band(
    bootstrap: BootstrapResult,
    estimates: Mapping[str, float],
    std_errors: Mapping[str, float],
    *,
    alpha: float,
    support: Support = UNCLASSIFIED,
) -> BootstrapBand:
    """The studentized bootstrap sup-t band over ``estimates``.

    Parameters
    ----------
    bootstrap : BootstrapResult
        The replicate draws, with their standard errors.
    estimates : mapping of str to float
        The full-sample estimates to band, by alias. Every alias must have draws and
        replicate standard errors.
    std_errors : mapping of str to float
        The full-sample standard error of each alias.
    alpha : float
        Joint significance level.
    support : Support, default=UNCLASSIFIED
        The band's support, from :func:`~cleverly._support.band_support` with the
        ``bootstrap_band`` kind.

    Returns
    -------
    BootstrapBand
        One joint interval per alias.
    """
    names = list(estimates)
    draws = np.column_stack([bootstrap.draws[name] for name in names])
    scale = np.column_stack([bootstrap.std_errors[name] for name in names])
    centre = np.array([estimates[name] for name in names], dtype=float)
    critical, used = studentized_critical_value(centre, draws, scale, alpha)
    bands = {
        name: (
            float(estimates[name] - critical * std_errors[name]),
            float(estimates[name] + critical * std_errors[name]),
        )
        for name in names
    }
    return BootstrapBand(
        critical_value=critical,
        bands=bands,
        alpha=alpha,
        n_replicates=used,
        support=support,
    )


def cluster_members(cluster: IntArray) -> list[IntArray]:
    """Row indices belonging to each cluster, in sorted cluster order.

    One stable ``argsort`` and a split, rather than a ``codes == code`` scan per
    cluster: the latter is ``O(n_clusters * n)``, which dominates a cluster bootstrap
    once there are more than a handful of clusters.
    """
    codes = np.asarray(cluster).reshape(-1)
    order = np.argsort(codes, kind="stable")
    _, starts = np.unique(codes[order], return_index=True)
    return [group.astype(np.int64) for group in np.split(order, starts[1:])]


def bootstrap_indices(
    n: int,
    cluster: IntArray | None,
    rng: np.random.Generator,
    members: list[IntArray] | None = None,
) -> IntArray:
    """Row indices for one bootstrap replicate.

    Without clusters this is an ordinary ``n``-out-of-``n`` resample.  With clusters,
    ``n_c`` clusters are drawn with replacement and every row of each drawn cluster
    is included, so the replicate has the same dependence structure as the original
    sample (and, for unbalanced clusters, a slightly different row count).

    Parameters
    ----------
    members : list of ndarray or None
        Precomputed :func:`cluster_members` output.  Building it is ``O(n log n)`` and
        does not depend on the draw, so :func:`run_bootstrap` builds it once and passes
        it to every replicate rather than paying for it thousands of times.
    """
    index, _ = _bootstrap_draw(n, cluster, rng, members)
    return index


def _bootstrap_draw(
    n: int,
    cluster: IntArray | None,
    rng: np.random.Generator,
    members: list[IntArray] | None = None,
) -> tuple[IntArray, IntArray | None]:
    """Draw rows and, for cluster samples, one code per sampled occurrence."""
    if cluster is None:
        return rng.integers(0, n, size=n, dtype=np.int64), None

    groups = cluster_members(cluster) if members is None else members
    drawn = rng.integers(0, len(groups), size=len(groups))
    selected = [groups[int(k)] for k in drawn]
    index = np.concatenate(selected).astype(np.int64)
    occurrence_codes = np.repeat(
        np.arange(len(selected), dtype=np.int64),
        [group.size for group in selected],
    )
    return index, occurrence_codes


def _bootstrap_design(
    data: Resamplable,
    *,
    n_replicates: int,
    resampling: Resampling,
    random_state: int | None,
) -> _BootstrapDesign:
    """Resolve and draw one bootstrap design without changing inference draw order."""
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be positive; got {n_replicates}")
    if resampling not in ("auto", "iid", "cluster"):
        raise ValueError("resampling must be 'auto', 'iid', or 'cluster'")
    use_clusters = data.cluster is not None if resampling == "auto" else resampling == "cluster"
    if use_clusters and data.cluster is None:
        raise ValueError("resampling='cluster' requires the data to carry cluster ids")

    sequences = np.random.SeedSequence(random_state).spawn(n_replicates)
    codes = data.cluster if use_clusters else None
    members = None if codes is None else cluster_members(codes)
    draws = []
    for replicate, sequence in enumerate(sequences):
        draws.append(
            _BootstrapDraw(
                replicate=replicate,
                seed=int(sequence.generate_state(1)[0]),
                sequence=sequence,
            )
        )
    return _BootstrapDesign(
        draws=tuple(draws),
        resampling="cluster" if use_clusters else "iid",
        cluster=codes,
        members=None if members is None else tuple(members),
    )


def run_bootstrap(
    data: Resamplable,
    refit: RefitFn,
    *,
    n_replicates: int,
    resampling: Resampling = "auto",
    random_state: int | None = None,
    n_jobs: int = 1,
) -> BootstrapResult:
    """Run ``n_replicates`` full-refit bootstrap replicates.

    Parameters
    ----------
    data : CausalData or LongitudinalData
        Validated study data to resample.
    refit : callable
        Maps a resampled container of the same type to a mapping of estimand name to
        ``(estimate, standard error)`` (:data:`RefitFn`).  Replicates that raise are
        dropped and counted rather than aborting the run: with weak overlap a resample
        can easily end up with an empty treatment arm in some stratum.
    n_replicates : int
        How many replicates to draw.
    resampling : {"auto", "iid", "cluster"}
        ``"auto"`` resamples clusters when the data has them, and rows otherwise.
        ``"iid"`` always resamples rows. ``"cluster"`` requires cluster ids and is
        refused without them. Any other value is refused.
    random_state : int or None
        Seed for the replicate draws.
    n_jobs : int
        Number of joblib workers.

    Returns
    -------
    BootstrapResult
        The replicate estimates and standard errors, and how many replicates failed.

    Raises
    ------
    TypeError
        When a replicate's refit returns anything but ``(estimate, standard error)`` pairs.
    """
    if n_replicates < 2:
        raise ValueError(f"n_replicates must be at least 2; got {n_replicates}")
    design = _bootstrap_design(
        data,
        n_replicates=n_replicates,
        resampling=resampling,
        random_state=random_state,
    )
    return _run_draws(design, lambda draw: refit(design.sample(data, draw)), n_replicates, n_jobs)


def run_index_bootstrap(
    data: Resamplable,
    refit: IndexRefitFn,
    *,
    n_replicates: int,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> BootstrapResult:
    """Run row-resampling replicates whose refit reads the row index, not a resample.

    The draws are the ones :func:`run_bootstrap` takes for the same ``random_state`` with
    ``resampling="iid"``: replicate ``b`` reads the row index that :func:`run_bootstrap`
    would subset ``data`` with.  A refit that computes the resample's fit from the index,
    such as a frequency-weighted replicate on the distinct rows, therefore reports the
    replicates a materialized refit reports.

    Parameters
    ----------
    data : CausalData or LongitudinalData
        Validated study data. Its clusters, if any, are refused: this draw is by row.
    refit : callable
        Maps a ``(n,)`` row index, repeats kept, to a mapping of estimand name to
        ``(estimate, standard error)``. Replicates that raise are dropped and counted.
    n_replicates : int
        How many replicates to draw.
    random_state : int or None
        Seed for the replicate draws.
    n_jobs : int
        Number of joblib workers.

    Returns
    -------
    BootstrapResult
        The replicate estimates and standard errors, and how many replicates failed.

    Raises
    ------
    ValueError
        When ``data`` carries cluster ids.
    TypeError
        When a replicate's refit returns anything but ``(estimate, standard error)`` pairs.
    """
    if n_replicates < 2:
        raise ValueError(f"n_replicates must be at least 2; got {n_replicates}")
    if data.cluster is not None:
        raise ValueError("an index bootstrap resamples rows; clustered data resamples clusters")
    design = _bootstrap_design(
        data, n_replicates=n_replicates, resampling="iid", random_state=random_state
    )

    def call(draw: _BootstrapDraw) -> Mapping[str, tuple[float, float]]:
        index, _ = _bootstrap_draw(data.n, None, np.random.default_rng(draw.sequence))
        return refit(index)

    return _run_draws(design, call, n_replicates, n_jobs)


def _pair(value: Any) -> tuple[float, float]:
    """One refit entry as ``(estimate, standard error)``, or the contract's ``TypeError``."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{REFIT_CONTRACT}; got {value!r}")
    return float(value[0]), float(value[1])


def _run_draws(
    design: _BootstrapDesign,
    call: Callable[[_BootstrapDraw], Mapping[str, Any]],
    n_replicates: int,
    n_jobs: int,
) -> BootstrapResult:
    """Run one refit per draw, drop the replicates that raise, and collect the rest."""

    def replicate(draw: _BootstrapDraw) -> Mapping[str, Any] | None:
        try:
            return call(draw)
        except Exception:
            return None

    with _suppressed():
        # Replicates routinely trip positivity and convergence warnings; surfacing
        # them once per replicate would bury the real output, and recording them would
        # put a resample's warning on the main fit.
        outcomes = map_parallel(replicate, design.draws, n_jobs=n_jobs)

    # Outside the guard above: a refit that breaks the contract is a programming error, and
    # counting it as a failed replicate would report an interval from no replicates.
    successes = [
        {name: _pair(value) for name, value in row.items()} for row in outcomes if row is not None
    ]
    n_failed = len(outcomes) - len(successes)
    if not successes:
        raise RuntimeError(
            f"all {n_replicates} bootstrap replicates failed; the fit is too unstable to "
            "bootstrap. Check res.diagnostics.support() and consider tighter g_bounds."
        )
    if n_failed:
        _warn(
            f"{n_failed} of {n_replicates} bootstrap replicates failed and were dropped",
            UserWarning,
            stacklevel=3,
        )

    names = list(successes[0].keys())
    missing = (np.nan, np.nan)
    draws = {
        name: np.array([row.get(name, missing)[0] for row in successes], dtype=float)
        for name in names
    }
    std_errors = {
        name: np.array([row.get(name, missing)[1] for row in successes], dtype=float)
        for name in names
    }
    return BootstrapResult(
        draws=draws,
        n_requested=n_replicates,
        n_failed=n_failed,
        resampling=design.resampling,
        std_errors=std_errors,
    )
