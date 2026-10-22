r"""The statistics that the refutation tests read, without any verdict.

Each function takes arrays and returns numbers and the reference law they sit against. The
registry (:mod:`cleverly.validation._interpretation`) decides whether a reading may print a
verdict, so nothing here compares a statistic with a threshold except the equivalence bands of
:class:`EquivalenceBands`, whose references are exact.

The functions follow the X82 plan: the placebo ``t`` (5.1.3), the subset scatter ratio (5.3),
the paired shift (5.2) and the equivalence readings of the generated-outcome and plasmode
reports (4.2.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy import stats

__all__ = [
    "DEFAULT_BAND_REPLICATES",
    "BandReading",
    "EquivalenceBands",
    "PlaceboReading",
    "SubsetReading",
    "band_readings",
    "paired_shift",
    "placebo_reading",
    "subset_reading",
]

#: The smallest draw count at which a calibrated estimator passes all three equivalence
#: readings with probability at least 0.90 (X82 plan 4.2.1; the design test recomputes it).
DEFAULT_BAND_REPLICATES: Final[int] = 400

#: A spread below this multiple of the standard error is zero for the purpose of a
#: studentized statistic (AGENT-BRIEF "Near-zero comparisons").
_DEGENERATE_SPREAD: Final[float] = 1e-12


@dataclass(frozen=True)
class PlaceboReading:
    """The one-sample ``t`` of the placebo draws against the exact null.

    Parameters
    ----------
    t : float
        ``mean(d) / (sd(d) / sqrt(R))`` with ``d`` the draws minus the null, on the log
        scale for a ratio. ``nan`` for a constant zero ``d``, and infinite for a constant
        nonzero ``d``.
    pvalue : float
        Two-sided probability under ``t_{R-1}``. ``nan`` for a constant zero ``d``.
    mean : float
        Mean of ``d``.
    sd : float
        Sample standard deviation of ``d``.
    df : int
        ``R - 1``.
    kind : {"t", "constant_zero", "constant_nonzero"}
        ``"constant_zero"`` when the estimate does not move with the treatment, and
        ``"constant_nonzero"`` when every draw sits at one nonzero offset.
    scale_ratio : float or None
        Mean standard error over the standard deviation of the draws. ``None`` without
        standard errors.
    """

    t: float
    pvalue: float
    mean: float
    sd: float
    df: int
    kind: Literal["t", "constant_zero", "constant_nonzero"]
    scale_ratio: float | None = None

    @property
    def rejects(self) -> bool:
        """Whether the two-sided test rejects at 0.05, or the offset is constant nonzero."""
        if self.kind == "constant_zero":
            return False
        if self.kind == "constant_nonzero":
            return True
        return bool(self.pvalue < 0.05)


def placebo_reading(
    deviations: Sequence[float],
    *,
    std_error: float | None = None,
    plugin_std_errors: Sequence[float] = (),
    scale: float | None = None,
) -> PlaceboReading:
    """Compute the placebo ``t`` from the draws' deviations from the null.

    Parameters
    ----------
    deviations : sequence of float
        ``psi*_r - null`` (``log psi*_r`` for a ratio), one per draw.
    std_error : float or None
        The fit's standard error, which sets the degenerate floor.
    plugin_std_errors : sequence of float
        The refits' own standard errors, for the descriptive scale ratio.
    scale : float or None
        Magnitude of the estimate, for the floor when the fit has no standard error.

    Returns
    -------
    PlaceboReading
        The statistic, and its degenerate kind where the spread is below the floor.
    """
    d = np.asarray(deviations, dtype=float)
    r = int(d.size)
    if r < 2:
        raise ValueError("the placebo t needs at least two draws")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if std_error is not None and np.isfinite(std_error) and std_error > 0:
        floor = _DEGENERATE_SPREAD * float(std_error)
    else:
        floor = _DEGENERATE_SPREAD * max(1.0, abs(float(scale)) if scale is not None else 1.0)
    ratio = None
    if len(plugin_std_errors) == r and sd > floor:
        ratio = float(np.mean(plugin_std_errors)) / sd
    if sd <= floor:
        kind: Literal["t", "constant_zero", "constant_nonzero"] = (
            "constant_zero" if abs(mean) <= floor else "constant_nonzero"
        )
        t = float("nan") if kind == "constant_zero" else float(np.sign(mean) * np.inf)
        p = float("nan") if kind == "constant_zero" else 0.0
        return PlaceboReading(t, p, mean, sd, r - 1, kind, ratio)
    t = mean / (sd / np.sqrt(r))
    p = float(2.0 * stats.t.sf(abs(t), r - 1))
    return PlaceboReading(float(t), p, mean, sd, r - 1, "t", ratio)


@dataclass(frozen=True)
class SubsetReading:
    """The scatter of subset refits against the sampling reference.

    Parameters
    ----------
    rho : float
        ``sd_r(psi_r) / (SE * sqrt((n/m - 1) n/(n-1)))``.
    statistic : float
        ``(R - 1) rho^2``, approximately ``chi^2_{R-1}`` given the data.
    pvalue : float
        Two-sided probability under ``chi^2_{R-1}``. Both tails matter: a large ``rho``
        reads "the estimator varies more than its standard error says", and a small one
        reads "the standard error is too large".
    reference_sd : float
        The reference standard deviation ``SE * sqrt((n/m - 1) n/(n-1))``.
    spread : float
        The sample standard deviation of the refits.
    """

    rho: float
    statistic: float
    pvalue: float
    reference_sd: float
    spread: float


def subset_reference_sd(std_error: float, n: int, m: int) -> float:
    r"""Return the standard deviation of a subset refit around the full-sample estimate.

    For an asymptotically linear estimator with influence curve :math:`D`,
    :math:`\psi_S - \psi_n \approx m^{-1}\sum_{i \in S} D_i - n^{-1}\sum_i D_i`. Given the
    data, the variance over random subsets of size :math:`m` is
    :math:`\sigma_D^2 (1/m - 1/n) n/(n-1)`, so the reference is
    :math:`\mathrm{SE}\sqrt{(n/m - 1)\, n/(n-1)}`.

    Parameters
    ----------
    std_error : float
        The fit's standard error, :math:`\sigma_D/\sqrt{n}`.
    n : int
        Number of sampling units in the fit.
    m : int
        Number of units drawn.

    Returns
    -------
    float
        The reference standard deviation.
    """
    if not 0 < m <= n:
        raise ValueError(f"the drawn size m={m} must lie in (0, n={n}]")
    if n < 2:
        raise ValueError("the subset reference needs at least two units")
    return float(std_error) * float(np.sqrt((n / m - 1.0) * n / (n - 1.0)))


def subset_reading(values: Sequence[float], *, std_error: float, n: int, m: int) -> SubsetReading:
    """Compute the scatter ratio and its two-sided chi-squared probability.

    Parameters
    ----------
    values : sequence of float
        The refits on the verdict scale (log for a ratio).
    std_error : float
        The fit's standard error on that scale.
    n : int
        Number of sampling units (clusters on a clustered fit).
    m : int
        Number of units each subset drew.

    Returns
    -------
    SubsetReading
        The statistic and its reference. ``rho`` is ``nan`` for a subset equal to the full
        sample or a non-positive standard error.
    """
    v = np.asarray(values, dtype=float)
    r = int(v.size)
    if r < 2:
        raise ValueError("the subset reading needs at least two refits")
    spread = float(v.std(ddof=1))
    if m >= n or not np.isfinite(std_error) or std_error <= 0:
        nan = float("nan")
        return SubsetReading(nan, nan, nan, nan, spread)
    reference = subset_reference_sd(std_error, n, m)
    rho = spread / reference
    statistic = (r - 1) * rho**2
    low = stats.chi2.cdf(statistic, r - 1)
    pvalue = float(min(1.0, 2.0 * min(low, 1.0 - low)))
    return SubsetReading(float(rho), float(statistic), pvalue, reference, spread)


def paired_shift(
    noise: Sequence[float], base: Sequence[float], *, std_error: float
) -> tuple[float, float, float]:
    """Compute the paired shift of the random-common-cause refits.

    Each noise refit pairs with a refit that shares its seed, and so its folds, and omits the
    noise column. The shift is the mean paired difference in standard errors.

    Parameters
    ----------
    noise, base : sequence of float
        Refits with and without the noise column, on the verdict scale, paired by seed.
    std_error : float
        The fit's standard error on that scale.

    Returns
    -------
    tuple of float
        The mean paired shift and the sample standard deviation of the paired differences,
        both in standard errors, then the standard deviation of the baseline refits in
        standard errors (the seed-only spread). All three are ``nan`` without a positive
        standard error.
    """
    a = np.asarray(noise, dtype=float)
    b = np.asarray(base, dtype=float)
    if a.shape != b.shape or a.size < 1:
        raise ValueError("the paired refits must have one value per seed")
    if not np.isfinite(std_error) or std_error <= 0:
        nan = float("nan")
        return nan, nan, nan
    diff = a - b
    spread = float(diff.std(ddof=1)) / std_error if diff.size > 1 else float("nan")
    seed_only = float(b.std(ddof=1)) / std_error if b.size > 1 else float("nan")
    return float(diff.mean()) / std_error, spread, seed_only


@dataclass(frozen=True)
class EquivalenceBands:
    """The equivalence bands of the generated-outcome and plasmode readings.

    An exact reference fixes the sampling law of each reading, and it does not make an exact
    null true: no finite-sample estimator has bias exactly zero or a standard-error ratio
    exactly one. A reading therefore passes when its 95% interval lies inside its band.

    Parameters
    ----------
    bias : float
        Half-width of the bias band in units of the standard deviation of the draws.
    se_ratio : tuple of float
        Band for the mean reported standard error over the standard deviation of the draws.
    coverage : tuple of float
        Band for the coverage of the nominal 95% interval.
    minimum_draws : int
        The fewest draws at which the bands are calibrated. Below it a reading prints at
        ``L1``.
    """

    bias: float = 0.20
    se_ratio: tuple[float, float] = (0.85, 1.15)
    coverage: tuple[float, float] = (0.88, 0.99)
    minimum_draws: int = DEFAULT_BAND_REPLICATES

    def __post_init__(self) -> None:
        if not self.bias > 0.0:
            raise ValueError("the bias band must have a positive half-width")
        for name, (low, high) in (("se_ratio", self.se_ratio), ("coverage", self.coverage)):
            if not low < high:
                raise ValueError(f"the {name} band must have its lower limit below its upper")
        if self.minimum_draws < 2:
            raise ValueError("minimum_draws must be at least two")


@dataclass(frozen=True)
class BandReading:
    """One equivalence reading with its 95% interval and band.

    Parameters
    ----------
    name : {"bias", "se_ratio", "coverage"}
        The reading.
    estimate : float
        Point value. The bias is in units of the standard deviation of the draws.
    lower, upper : float
        The 95% interval.
    band : tuple of float
        The equivalence band.
    inside : bool
        Whether the interval lies inside the band.
    """

    name: Literal["bias", "se_ratio", "coverage"]
    estimate: float
    lower: float
    upper: float
    band: tuple[float, float]
    inside: bool


def band_readings(
    estimates: Sequence[float],
    std_errors: Sequence[float],
    truth: float,
    bands: EquivalenceBands,
    *,
    truth_variance: float = 0.0,
    critical: float = 1.959963984540054,
) -> tuple[BandReading, BandReading, BandReading]:
    r"""Compute the three equivalence readings of an exact-reference report.

    Parameters
    ----------
    estimates, std_errors : sequence of float
        One refit per draw: the point estimate and its reported standard error.
    truth : float
        The exact target of the generated law.
    bands : EquivalenceBands
        The bands to judge against.
    truth_variance : float
        The Monte Carlo variance of ``truth``. Zero for an exact truth.
    critical : float
        The normal critical value of the coverage reading and of the nominal intervals.

    Returns
    -------
    tuple of BandReading
        Bias, standard-error ratio and coverage, in that order.

        The bias interval is :math:`\bar\psi - T \pm 1.96\sqrt{s^2/R + V_T}` in units of the
        draws' standard deviation :math:`s`. The ratio interval is the delta method on
        :math:`\log \overline{se} - \tfrac12 \log \overline{(\psi_r - \bar\psi)^2}`,
        from the joint sample moments. Coverage is the two-sided Clopper-Pearson interval of
        the share of draws with :math:`|\psi_r - T| \le 1.96\, se_r`.
    """
    psi = np.asarray(estimates, dtype=float)
    se = np.asarray(std_errors, dtype=float)
    r = int(psi.size)
    if r < 3 or se.size != r:
        raise ValueError("the equivalence readings need at least three draws with a standard error")
    sd = float(psi.std(ddof=1))
    if sd <= 0:
        raise ValueError("the draws do not vary, so the equivalence readings are undefined")

    bias = float(psi.mean() - truth)
    half = critical * float(np.sqrt(sd**2 / r + truth_variance))
    bias_reading = BandReading(
        "bias",
        bias / sd,
        (bias - half) / sd,
        (bias + half) / sd,
        (-bands.bias, bands.bias),
        bool(abs(bias) + half <= bands.bias * sd),
    )

    m = float(se.mean())
    d2 = (psi - psi.mean()) ** 2
    v = float(d2.mean())
    ratio = m / sd
    influence = (se - m) / m - (d2 - v) / (2.0 * v)
    s_log = float(influence.std(ddof=1)) / np.sqrt(r)
    low, high = ratio * np.exp(-critical * s_log), ratio * np.exp(critical * s_log)
    ratio_reading = BandReading(
        "se_ratio",
        float(ratio),
        float(low),
        float(high),
        bands.se_ratio,
        bool(bands.se_ratio[0] <= low and high <= bands.se_ratio[1]),
    )

    hits = int(np.count_nonzero(np.abs(psi - truth) <= critical * se))
    cp_low = 0.0 if hits == 0 else float(stats.beta.ppf(0.025, hits, r - hits + 1))
    cp_high = 1.0 if hits == r else float(stats.beta.ppf(0.975, hits + 1, r - hits))
    coverage_reading = BandReading(
        "coverage",
        hits / r,
        cp_low,
        cp_high,
        bands.coverage,
        bool(bands.coverage[0] <= cp_low and cp_high <= bands.coverage[1]),
    )
    return bias_reading, ratio_reading, coverage_reading
