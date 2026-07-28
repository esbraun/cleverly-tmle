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

__all__ = ["BootstrapResult", "bootstrap_indices", "run_bootstrap"]

RefitFn = Callable[[CausalData], Mapping[str, float]]
Resampling = Literal["auto", "iid", "cluster"]


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap draws for every estimand, plus how many replicates failed."""

    draws: dict[str, FloatArray]
    n_requested: int
    n_failed: int
    resampling: str

    def summary(self, name: str, alpha: float = 0.05) -> BootstrapSummary:
        """Percentile intervals and bootstrap standard error for one estimand.

        Percentile rather than normal-approximation intervals: the point of running
        a bootstrap here is usually that the sampling distribution is skewed, and a
        symmetric interval would throw that information away.
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


def bootstrap_indices(
    n: int,
    cluster: IntArray | None,
    rng: np.random.Generator,
) -> IntArray:
    """Row indices for one bootstrap replicate.

    Without clusters this is an ordinary ``n``-out-of-``n`` resample.  With clusters,
    ``n_c`` clusters are drawn with replacement and every row of each drawn cluster
    is included, so the replicate has the same dependence structure as the original
    sample (and, for unbalanced clusters, a slightly different row count).
    """
    if cluster is None:
        return rng.integers(0, n, size=n, dtype=np.int64)

    codes = np.asarray(cluster).reshape(-1)
    unique = np.unique(codes)
    members = [np.flatnonzero(codes == code) for code in unique]
    drawn = rng.integers(0, unique.size, size=unique.size)
    return np.concatenate([members[int(k)] for k in drawn]).astype(np.int64)


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
    refit:
        Maps a resampled :class:`~cleverly.data.CausalData` to a mapping of estimand
        name to point estimate.  Replicates that raise are dropped and counted
        rather than aborting the run: with weak overlap a resample can easily end
        up with an empty treatment arm in some stratum.
    resampling:
        ``"auto"`` resamples clusters when the data has them, rows otherwise.
    """
    if n_replicates < 2:
        raise ValueError(f"n_replicates must be at least 2; got {n_replicates}")
    use_clusters = data.cluster is not None if resampling == "auto" else resampling == "cluster"
    if use_clusters and data.cluster is None:
        raise ValueError("resampling='cluster' requires the data to carry cluster ids")

    seeds = np.random.SeedSequence(random_state).spawn(n_replicates)

    def replicate(seed: np.random.SeedSequence) -> Mapping[str, float] | None:
        rng = np.random.default_rng(seed)
        try:
            index = bootstrap_indices(data.n, data.cluster if use_clusters else None, rng)
            return refit(data.subset(index))
        except Exception:
            return None

    with warnings.catch_warnings():
        # Replicates routinely trip positivity and convergence warnings; surfacing
        # them once per replicate would bury the real output.
        warnings.simplefilter("ignore")
        outcomes = map_parallel(replicate, seeds, n_jobs=n_jobs)

    successes = [row for row in outcomes if row is not None]
    n_failed = len(outcomes) - len(successes)
    if not successes:
        raise RuntimeError(
            f"all {n_replicates} bootstrap replicates failed; the fit is too unstable to "
            "bootstrap. Check res.sensitivity.positivity() and consider tighter g_bounds."
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
        resampling="cluster" if use_clusters else "iid",
    )
