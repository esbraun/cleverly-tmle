"""Decision primitives shared by every method-evidence study.

Every function here answers a question of the form "is this quantity inside a margin I
declared before I looked", never "is this quantity exactly zero".  The distinction is the
governing rule of the evidence studies: a Monte Carlo study accumulates evidence by adding
replications, so an accept-decision has to become *easier* as replications grow.  A
significance test does the opposite -- it converges on rejecting any estimator whose
finite-sample remainder is not identically zero, which is every estimator -- and would turn
the study red for the one reason that is not a defect.

:func:`gate_is_margin_bounded` in the framework tests exercises that property directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm, t


@dataclass(frozen=True)
class Interval:
    """A confidence interval, with the containment questions asked of it."""

    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def within(self, low: float, high: float) -> bool:
        """The interval lies inside ``[low, high]`` -- the equivalence verdict."""
        return self.low >= low and self.high <= high

    def outside(self, low: float, high: float) -> bool:
        """The interval lies entirely outside ``[low, high]`` -- the discrimination verdict."""
        return self.low > high or self.high < low

    def resolution(self, reference: float) -> float:
        """How far from ``reference`` this interval still reaches.

        The smallest equivalence margin the data could have supported: quoting it is what
        keeps a wide margin from reading as a tight calibration proof.
        """
        return max(abs(self.low - reference), abs(self.high - reference))


def student_interval(values: np.ndarray, *, confidence_level: float) -> Interval:
    """Two-sided Student interval for the mean of ``values``."""
    count = len(values)
    if count < 2:
        raise ValueError(f"a Student interval needs at least two values; got {count}")
    centre = float(np.mean(values))
    half_width = float(
        t.ppf(0.5 + confidence_level / 2.0, count - 1) * np.std(values, ddof=1) / math.sqrt(count)
    )
    return Interval(centre - half_width, centre + half_width)


def clopper_pearson(successes: int, trials: int, *, confidence_level: float) -> Interval:
    """The exact two-sided binomial interval, used for coverage.

    Exact rather than Wald because coverage sits near one, where the normal approximation
    is worst and where the whole question is whether the shortfall is real.
    """
    tail = (1.0 - confidence_level) / 2.0
    low = float(beta.ppf(tail, successes, trials - successes + 1)) if successes else 0.0
    high = (
        float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
        if successes < trials
        else 1.0
    )
    return Interval(low, high)


def coverage_for_se_ratio(ratio: float, *, alpha: float) -> float:
    r"""Actual coverage of a Wald interval whose reported SE is ``ratio`` times the truth.

    :math:`2\Phi(z_{\alpha/2} r) - 1`.  This is what ties the SE-calibration band to the
    coverage floor: the two margins are one statement about interval validity written on two
    scales, and declaring them independently invites a pair that cannot both bind.
    """
    return float(2.0 * norm.cdf(norm.ppf(1.0 - alpha / 2.0) * ratio) - 1.0)


def se_ratio_for_coverage(coverage: float, *, alpha: float) -> float:
    """The inverse of :func:`coverage_for_se_ratio`."""
    return float(norm.ppf(0.5 + coverage / 2.0) / norm.ppf(1.0 - alpha / 2.0))


def bootstrap(
    arrays: Mapping[str, np.ndarray],
    statistics: Mapping[str, Callable[[dict[str, np.ndarray]], np.ndarray]],
    *,
    replicates: int,
    seed: int,
    batch_size: int = 1_000,
) -> dict[str, np.ndarray]:
    """Deterministic paired percentile bootstrap over jointly resampled columns.

    One index draw per batch feeds every array, so statistics computed here compare the same
    replications -- the pairing that makes a cross-implementation comparison a comparison
    rather than two independent studies.  Deterministic given ``seed`` because the published
    artefacts have to be recomputable from the committed rows.
    """
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"bootstrap columns have different lengths: {sorted(lengths)}")
    count = lengths.pop()
    rng = np.random.default_rng(seed)
    out = {name: np.empty(replicates, dtype=float) for name in statistics}
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, count, size=(stop - start, count))
        sampled = {name: array[indices] for name, array in arrays.items()}
        for name, statistic in statistics.items():
            out[name][start:stop] = statistic(sampled)
    return out


def percentile_interval(samples: np.ndarray, *, confidence_level: float) -> Interval:
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return Interval(float(low), float(high))


def upper_bound(samples: np.ndarray, *, confidence_level: float) -> float:
    """One-sided upper confidence bound -- the non-inferiority direction for a loss."""
    return float(np.quantile(samples, confidence_level))


def lower_bound(samples: np.ndarray, *, confidence_level: float) -> float:
    """One-sided lower confidence bound -- the non-inferiority direction for a benefit."""
    return float(np.quantile(samples, 1.0 - confidence_level))


@dataclass(frozen=True)
class BiasVerdict:
    """Standardized-bias equivalence, in the two-one-sided-tests sense."""

    bias: float
    interval: Interval
    margin: float
    scale: float

    @property
    def equivalent(self) -> bool:
        """The bias is negligible relative to the spread, within the declared margin."""
        return self.interval.within(-self.margin, self.margin)

    @property
    def discriminated(self) -> bool:
        """The bias is *established* to exceed the margin -- the negative-control verdict."""
        return self.interval.outside(-self.margin, self.margin)

    @property
    def standardized(self) -> float:
        return self.bias / self.scale if self.scale > 0.0 else math.nan


def standardized_bias_verdict(
    errors: np.ndarray,
    *,
    margin: float,
    confidence_level: float,
    scale: float | None = None,
) -> BiasVerdict:
    """Is the mean error inside ``margin`` standard deviations of zero?

    Standardizing by the sampling spread rather than by the Monte Carlo standard error is
    the whole point.  ``margin * empirical_sd`` is a property of the estimator at this
    sample size and does not move when replications are added, so more replications shrink
    the interval inside a fixed target.  ``|bias| <= k * bias_se`` shrinks the *target*
    instead, and fails once the Monte Carlo error drops below the estimator's real
    second-order remainder.
    """
    spread = float(np.std(errors, ddof=1)) if scale is None else scale
    return BiasVerdict(
        bias=float(np.mean(errors)),
        interval=student_interval(errors, confidence_level=confidence_level),
        margin=margin * spread,
        scale=spread,
    )
