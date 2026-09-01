"""Targeted bootstrap inference.

The influence-curve variance is the right default: it is cheap and, when the
nuisance estimators converge fast enough, correct.  It is also asymptotic, and it
can be optimistic in exactly the situations practitioners care about -- small
samples, weak overlap, a heavily targeted fit.  R's ``tmle`` therefore offers
``B > 1`` for a bootstrap, and so does this library.

The bootstrap here is *targeted*: every replicate re-runs the whole procedure,
nuisance fits included.  Bootstrapping only the targeting step while holding the
nuisance fits fixed would understate the variance, because the nuisance
estimation is itself a source of uncertainty.

With clusters, whole clusters are resampled -- resampling rows would destroy the
dependence structure the cluster variance exists to account for.

Observation weights are resampled with their rows and renormalised within each replicate,
which is what keeps every replicate aimed at the same tilted parameter (see
:mod:`cleverly.data.weighting`).  What the bootstrap does *not* do is re-derive the
weights: a replicate inherits the numbers it was given, so if the weights came out of a
fitted selection or calibration model, these intervals condition on that fit exactly as
the influence-curve ones do.  Re-deriving the weights inside each replicate would need
the model that produced them, which the package never sees.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from ..data.causal_data import CausalData
from ..utils.parallel import map_parallel
from .influence import BootstrapSummary

__all__ = ["BootstrapResult", "bootstrap_indices", "cluster_members", "run_bootstrap"]

RefitFn = Callable[[CausalData], Mapping[str, float]]
Resampling = Literal["auto", "iid", "cluster"]


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

    def sample(self, data: CausalData, draw: _BootstrapDraw) -> CausalData:
        """Materialize one sample without retaining every replicate index."""
        index = bootstrap_indices(
            data.n,
            self.cluster,
            np.random.default_rng(draw.sequence),
            None if self.members is None else list(self.members),
        )
        return data.subset(index)


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
        with an empty treatment arm in some stratum.
    resampling : str
        Whether rows or clusters were resampled.
    """

    draws: dict[str, FloatArray]
    n_requested: int
    n_failed: int
    resampling: str

    def summary(self, name: str, alpha: float = 0.05) -> BootstrapSummary:
        """Percentile intervals and bootstrap standard error for one estimand.

        Percentile rather than normal-approximation intervals: the point of running
        a bootstrap here is usually that the sampling distribution is skewed, and a
        symmetric interval would throw that information away.

        Parameters
        ----------
        name : str
            Estimand alias to summarise.
        alpha : float
            Significance level of the percentile interval.

        Returns
        -------
        BootstrapSummary
            Percentile interval and bootstrap standard error for that estimand.
        """
        draws = self.draws[name]
        finite = draws[np.isfinite(draws)]
        if finite.size < 2:
            return BootstrapSummary(
                std_error=float("nan"),
                ci=(float("nan"), float("nan")),
                ci_one_sided_lower=float("nan"),
                ci_one_sided_upper=float("nan"),
                n_replicates=int(finite.size),
                n_failed=self.n_failed,
                draws=finite,
            )
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        return BootstrapSummary(
            std_error=float(np.std(finite, ddof=1)),
            ci=(float(low), float(high)),
            ci_one_sided_lower=float(np.quantile(finite, alpha)),
            ci_one_sided_upper=float(np.quantile(finite, 1.0 - alpha)),
            n_replicates=int(finite.size),
            n_failed=self.n_failed,
            draws=finite,
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
    members:
        Precomputed :func:`cluster_members` output.  Building it is ``O(n log n)`` and
        does not depend on the draw, so :func:`run_bootstrap` builds it once and passes
        it to every replicate rather than paying for it thousands of times.
    """
    if cluster is None:
        return rng.integers(0, n, size=n, dtype=np.int64)

    groups = cluster_members(cluster) if members is None else members
    drawn = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(k)] for k in drawn]).astype(np.int64)


def _bootstrap_design(
    data: CausalData,
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
    if not hasattr(data, "subset"):
        raise TypeError(
            f"the bootstrap resamples rows and refits, which needs a subset() on the "
            f"data container; {type(data).__name__} has none. A longitudinal fit is not "
            "bootstrappable for that reason: subsetting has to carry every node and the "
            "whole backward recursion has to run again per replicate"
        )
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
    data: CausalData,
    refit: RefitFn,
    *,
    n_replicates: int,
    resampling: Resampling = "auto",
    random_state: int | None = None,
    n_jobs: int = 1,
) -> BootstrapResult:
    """Run ``n_replicates`` targeted bootstrap replicates.

    Parameters
    ----------
    data : CausalData
        Validated study data to resample.
    refit : callable
        Maps a resampled :class:`~cleverly.data.CausalData` to a mapping of estimand
        name to point estimate.  Replicates that raise are dropped and counted
        rather than aborting the run: with weak overlap a resample can easily end
        up with an empty treatment arm in some stratum.
    n_replicates : int
        How many replicates to draw.
    resampling : {"auto", "cluster", "row"}
        ``"auto"`` resamples clusters when the data has them, rows otherwise.

    random_state : int or None
        Seed for the replicate draws.
    n_jobs : int
        Number of joblib workers.

    Returns
    -------
    BootstrapResult
        The replicate estimates, and how many replicates failed.
    """
    if n_replicates < 2:
        raise ValueError(f"n_replicates must be at least 2; got {n_replicates}")
    design = _bootstrap_design(
        data,
        n_replicates=n_replicates,
        resampling=resampling,
        random_state=random_state,
    )

    def replicate(draw: _BootstrapDraw) -> Mapping[str, float] | None:
        try:
            return refit(design.sample(data, draw))
        except Exception:
            return None

    with warnings.catch_warnings():
        # Replicates routinely trip positivity and convergence warnings; surfacing
        # them once per replicate would bury the real output.
        warnings.simplefilter("ignore")
        outcomes = map_parallel(replicate, design.draws, n_jobs=n_jobs)

    successes = [row for row in outcomes if row is not None]
    n_failed = len(outcomes) - len(successes)
    if not successes:
        raise RuntimeError(
            f"all {n_replicates} bootstrap replicates failed; the fit is too unstable to "
            "bootstrap. Check res.diagnostics.support() and consider tighter g_bounds."
        )
    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_replicates} bootstrap replicates failed and were dropped",
            UserWarning,
            stacklevel=2,
        )

    names = list(successes[0].keys())
    draws = {
        name: np.array([float(row.get(name, np.nan)) for row in successes], dtype=float)
        for name in names
    }
    return BootstrapResult(
        draws=draws,
        n_requested=n_replicates,
        n_failed=n_failed,
        resampling=design.resampling,
    )
