"""Sizing rules shared by the repeated-outcome studies (X32a, X32b).

A study declares how many replications each instrument draws as a rule applied to a pilot.
This module holds the rules.  Each rule takes the study's
:class:`~tests.studies.evidence.registry.Margins`, and the rules that read an influence curve
take the study's correlation function, so the incremental (``IPSI``) and sequential-regression
(``SDR``, ``LTMLE``) studies share one implementation.

The pooled reading
------------------

A rule reads a pilot as a set of noisy cells, not as one independent measurement per cell.
Cells that share a fit share noise, and cells that share a law differ by less than their own
pilot noise.  A rule that moves every cell to the far end of its own interval prices the pilot
noise once for every cell, and a family-wise product then prices it again.
:func:`pooled_pass_probability` prices the noise of a group once, and
:func:`heterogeneity_bound` prices the spread between its cells separately.

The family-wise allocation
--------------------------

One study passes when every gated verdict passes.  Some verdicts share rows, so the study
allocates failure probability by the union bound, which holds under any dependence.  The
constants below declare the shares before any pilot exists:

=============================  =============================================================
constant                       failure share
=============================  =============================================================
:data:`DISCRIMINATION_FAIL`    0.001 for all discrimination checks together.  Each of ``k``
                               checks fails with probability at most
                               ``1 - `` :func:`discrimination_floor`.
:data:`SIZED_FAIL`             :math:`d = 0.049 / 3` for each of the three sized instruments:
                               primary, calibration, property bias.
:data:`PRIMARY_CLAUSE_SHARE`   The coverage clause and the bias clause of the primary each
                               fail with probability at most :math:`d / 2`.
:func:`family_share`           A property family fails with probability at most
                               :math:`d / F` for ``F`` families.
:data:`C6_SHARE`               The C6 ``n`` rule fails with probability at most :math:`d / 4`.
=============================  =============================================================

The study fails with probability at most :math:`0.001 + 3d = 0.05`.  X32a declares
:data:`FAMILY_WISE_PASS` = 0.95 for each sized instrument, as a product.  Its sizes reproduce
exactly, because the X32a rules keep that target as their default.

The robust reading
------------------

:func:`robust_probability` is the pooled reading after a declared outlier test.  The pooled
reading assumes exchangeable cells, and one outlying cell moves the group mean and the spread
only a little.  The test finds the most extreme cell of a judged group and reads that cell
alone.  Clause (d) of :func:`primary_replicates` and every ``n`` rule use it.

The ``n`` rule
--------------

:func:`n_rule_fires` is the one predictive rule that decides whether a declared ``n`` doubles.
It reads the robust probability that every bias check of a group passes at
:data:`N_RULE_REFERENCE_REPLICATES`.  The reference is the pilot-free floor of the primary
rule.  A bias check passes more often as the replications grow when the true bias lies below
the margin, so a rule that holds at the reference holds at every larger size.
:func:`c6_rule_fires` is the rule of the continuous-law calibration arm.  That arm has no bias
gate, so the rule reads the coverage half of its calibration gate at the cap.

The calibration cap
-------------------

:func:`calibration_size` searches up to :data:`CALIBRATION_CAP`.  If no size meets the budget,
it returns the cap and marks the budget unmet.  The run then uses the cap and the gate gives
the verdict.  Reaching the cap is not a diagnosis.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq
from scipy.stats import beta, binom, chi2, norm

from tests.studies.evidence.registry import Margins

#: The one-sided 80% level of the pilot's own sampling error in the sizing rules.
PILOT_LEVEL = 0.80
#: The multiples of this many replications are the candidate budgets of the sizing rules.
BUDGET_STEP = 400
#: The multiples of this many replications are the candidate sizes of a property family.
PROPERTY_STEP = 200
#: No rule searches past this many replications.
MAXIMUM_REPLICATES = 50_000
#: The family-wise pass probability X32a requires of each sized instrument.
FAMILY_WISE_PASS = 0.95
#: The per-cell coverage-floor pass probability the primary rule requires at the stated coverage.
PRIMARY_CELL_PASS = 0.99
PRIMARY_DESIGN_COVERAGE = 0.93
PRIMARY_DESIGN_BIAS = 0.15

#: The failure probability a whole study may have.
STUDY_FAIL = 0.05
#: The failure share of every discrimination check together.
DISCRIMINATION_FAIL = 0.001
#: The failure share of each sized instrument: the primary, the calibration and the property bias
#: checks.
SIZED_FAIL = (STUDY_FAIL - DISCRIMINATION_FAIL) / 3.0
#: The share of one sized instrument.
SIZED_SHARE = 1.0 - SIZED_FAIL
#: The primary splits its failure share equally between the coverage clause and the bias clause.
PRIMARY_CLAUSE_SHARE = 1.0 - SIZED_FAIL / 2.0
#: The calibration terms of ``IPSI`` (C1, C3, C6, the C4 joint band) and of ``SDR`` (adding C5).
IPSI_CALIBRATION_TERMS = 4
SDR_CALIBRATION_TERMS = 5
#: The share of the C6 ``n`` rule: one of the ``IPSI`` calibration terms.
C6_SHARE = 1.0 - SIZED_FAIL / IPSI_CALIBRATION_TERMS
#: The calibration pilots (P2, P5) draw this many replications per arm.
CALIBRATION_PILOT_DRAWS = 8_000
#: Every other pilot draws this many replications per scenario or family.
PILOT_DRAWS = 2_000
#: The ``n`` rule reads the bias checks at this many replications.
N_RULE_REFERENCE_REPLICATES = 2_400
#: The calibration rule searches up to this many replications.  At the cap the gate decides.
CALIBRATION_CAP = MAXIMUM_REPLICATES
#: The two-sided level of the outlier test in each judged group, Bonferroni over its cells.
OUTLIER_LEVEL = 0.01


def discrimination_floor(checks: int) -> float:
    """The pass probability each of ``checks`` discrimination checks must hold.

    The study allocates :data:`DISCRIMINATION_FAIL` to all of them.  The union bound gives
    each check a failure share of ``DISCRIMINATION_FAIL / checks``, whatever the dependence.
    """
    return 1.0 - DISCRIMINATION_FAIL / checks


def family_share(families: int) -> float:
    """The share of one of ``families`` property families: ``1 - SIZED_FAIL / families``."""
    return 1.0 - SIZED_FAIL / families


# ---------------------------------------------------------------------------- cell probabilities


def _confidence_z(margins: Margins) -> float:
    return float(norm.ppf(0.5 + margins.confidence_level / 2.0))


def bias_equivalent_probability(
    margins: Margins, bias: float | np.ndarray, replicates: int
) -> float | np.ndarray:
    """P(an arm's 99% bias interval lies inside the margin), at standardized ``bias``.

    The margin is :attr:`~tests.studies.evidence.registry.Margins.standardized_bias` spreads of
    the arm, and the interval is the Student interval of the mean error, which is normal at
    these sizes.  An array of biases gives an array of probabilities.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    bias : float or numpy.ndarray
        The true standardized bias of the arm.
    replicates : int
        The replications of the arm.
    """
    z = _confidence_z(margins)
    root = np.sqrt(replicates)
    value = np.asarray(bias, dtype=float)
    probability = norm.cdf(root * (margins.standardized_bias - value) - z) - norm.cdf(
        -root * (margins.standardized_bias + value) + z
    )
    return float(probability) if probability.ndim == 0 else np.asarray(probability)


def bias_discriminated_probability(margins: Margins, gap: float, replicates: int) -> float:
    """P(a control's 99% bias interval clears the margin), at standardized ``gap``.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    gap : float
        The distance between the control's limit and the truth, in spreads of the arm.
    replicates : int
        The replications of the arm.
    """
    z = _confidence_z(margins)
    return float(norm.cdf(np.sqrt(replicates) * (abs(gap) - margins.standardized_bias) - z))


@functools.cache
def _first_passing_count(margins: Margins, replicates: int) -> int | None:
    """The smallest covered count whose exact interval's lower end clears the coverage floor."""
    tail = (1.0 - margins.confidence_level) / 2.0
    counts = np.arange(1, replicates + 1)
    lower = beta.ppf(tail, counts, replicates - counts + 1)
    passing = counts[lower >= margins.coverage_floor]
    return int(passing[0]) if passing.size else None


def coverage_floor_pass_probability(margins: Margins, replicates: int, coverage: float) -> float:
    """The probability that the exact coverage interval's lower bound clears the floor.

    The gate is :attr:`~tests.studies.evidence.registry.Margins.coverage_floor` on the lower
    end of the two-sided Clopper-Pearson interval at the study's confidence level.  The lower
    end rises with the covered count, so the gate passes from one count upward and the
    probability is a binomial tail.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of the arm.
    coverage : float
        The true coverage of the arm.
    """
    first = _first_passing_count(margins, replicates)
    if first is None:
        return 0.0
    return float(binom.sf(first - 1, replicates, coverage))


@functools.cache
def accepted_counts(margins: Margins, replicates: int) -> tuple[int, int] | None:
    """The covered counts whose exact interval lies inside the calibration coverage band.

    The Clopper-Pearson bounds both rise with the covered count, so the accepted counts form
    one run.  ``None`` means that no count is accepted at this many replications.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of the arm.
    """
    band_low, band_high = margins.calibration_coverage
    counts = np.arange(replicates + 1)
    tail = (1.0 - margins.confidence_level) / 2.0
    lower = np.where(
        counts > 0, beta.ppf(tail, counts, np.maximum(replicates - counts + 1, 1)), 0.0
    )
    upper = np.where(
        counts < replicates,
        beta.ppf(1.0 - tail, counts + 1, np.maximum(replicates - counts, 1)),
        1.0,
    )
    inside = counts[(lower >= band_low) & (upper <= band_high)]
    if inside.size == 0:
        return None
    return int(inside.min()), int(inside.max())


def wald_coverage(
    margins: Margins, ratio: np.ndarray, bias: float | np.ndarray = 0.0
) -> np.ndarray:
    r"""The coverage of a Wald interval with SE ratio ``ratio`` and standardized ``bias``.

    :math:`c(r, b) = \Phi(1.96 r - b) - \Phi(-1.96 r - b)`.  At zero bias it is
    :math:`2\Phi(1.96 r) - 1`, which is X32a's reading.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    ratio : numpy.ndarray
        The true ratio of the mean reported SE to the empirical SD.
    bias : float or numpy.ndarray
        The standardized bias of the estimate.
    """
    critical = float(norm.ppf(1.0 - margins.alpha / 2.0))
    if np.all(np.asarray(bias) == 0.0):
        # ``coverage_for_se_ratio`` over an array: a unit test pins the two together.
        return np.asarray(2.0 * norm.cdf(critical * ratio) - 1.0)
    return np.asarray(norm.cdf(critical * ratio - bias) - norm.cdf(-critical * ratio - bias))


def calibration_pass(
    margins: Margins,
    replicates: int,
    ratio: np.ndarray,
    coverage: float | None = None,
    *,
    bias: float = 0.0,
    one_sided: bool = False,
) -> np.ndarray:
    """The probability that a calibration cell passes, over an array of SE ratios.

    The SE-ratio half reads a normal approximation: the empirical SD of ``R`` normal
    estimates has relative standard deviation ``1 / sqrt(2 (R - 1))``, and the bootstrap 99%
    interval has half-width ``z_{0.995}`` of those.  The mean reported SE is treated as exact.
    The coverage half is exact: it sums the binomial law of the covered count over the counts
    whose Clopper-Pearson interval lies inside the band.  The two halves are independent under
    this approximation, so the probability is their product.

    ``coverage`` defaults to the coverage a Wald interval with this SE ratio and ``bias``
    attains (:func:`wald_coverage`), because a cell with a ratio off one is off in both halves.
    A one-sided cell reads only the lower end of the band on both scales (the ``SDR`` C5 arm).

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of the arm.
    ratio : numpy.ndarray
        The true SE ratios.
    coverage : float, optional
        The true coverage, in place of the Wald coverage.
    bias : float
        The standardized bias that the Wald coverage reads.
    one_sided : bool
        Read the lower end of the band only.
    """
    ratio = np.asarray(ratio, dtype=float)
    low, high = margins.calibration_se_ratio
    z = _confidence_z(margins)
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    half = z * spread
    if one_sided:
        se_pass = 1.0 - norm.cdf((low + half - ratio) / spread)
    else:
        se_pass = np.maximum(
            norm.cdf((high - half - ratio) / spread) - norm.cdf((low + half - ratio) / spread),
            0.0,
        )
    wald = (
        np.full_like(se_pass, coverage)
        if coverage is not None
        else wald_coverage(margins, ratio, bias)
    )
    if one_sided:
        tail = (1.0 - margins.confidence_level) / 2.0
        counts = np.arange(1, replicates + 1)
        lower = beta.ppf(tail, counts, replicates - counts + 1)
        accepted = counts[lower >= margins.calibration_coverage[0]]
        if accepted.size == 0:
            return np.zeros_like(se_pass)
        return np.asarray(se_pass * binom.sf(int(accepted[0]) - 1, replicates, wald))
    accepted_run = accepted_counts(margins, replicates)
    if accepted_run is None:
        return np.zeros_like(se_pass)
    coverage_pass = binom.cdf(accepted_run[1], replicates, wald) - binom.cdf(
        accepted_run[0] - 1, replicates, wald
    )
    return np.asarray(se_pass * coverage_pass)


def calibration_control_probability(margins: Margins, replicates: int, ratio: float) -> float:
    """P(a derived calibration control with SE ratio ``ratio`` lies below the band).

    The control passes when the 99% interval of its SE ratio lies below the band's lower end.
    The shrunken-SE control has the ratio 0.70 and the noise control has ``1 / sqrt(2)``.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of the arm.
    ratio : float
        The true SE ratio of the control.
    """
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    return float(
        norm.cdf(
            (margins.calibration_se_ratio[0] - _confidence_z(margins) * spread - ratio) / spread
        )
    )


def joint_band_pass_probability(margins: Margins, replicates: int, coverage: float) -> float:
    """The probability that the exact interval of the joint coverage lies inside the band (C4).

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of the arm.
    coverage : float
        The true joint coverage.
    """
    accepted = accepted_counts(margins, replicates)
    if accepted is None:
        return 0.0
    return float(
        binom.cdf(accepted[1], replicates, coverage)
        - binom.cdf(accepted[0] - 1, replicates, coverage)
    )


# ---------------------------------------------------------------------------- pooled reading


def influence_correlations(curves: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """The correlation, under the path weights, of influence curves.

    Two estimates from one sample have this correlation to first order.  A curve repeated in
    ``curves`` has correlation one with itself.

    Parameters
    ----------
    curves : Sequence[numpy.ndarray]
        The centered influence curve of each label on every path.
    weights : numpy.ndarray
        The probability of each path.
    """
    stacked = np.array(curves)
    covariance = (stacked * np.asarray(weights)) @ stacked.T
    scale = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(scale, scale))


def cell_labels(names: Sequence[str]) -> list[str]:
    """The label of each ``label__configuration`` cell name, ``label`` alone if it has no suffix."""
    return [name.split("__", 1)[0].rsplit("/", 1)[-1] for name in names]


Correlations = Callable[[Sequence[str]], np.ndarray]
#: One label-correlation function, or one for each calibration arm when the arms draw from
#: different laws (``IPSI`` C6 reads the continuous law).
ArmCorrelations = Correlations | Mapping[str, Correlations]


def effective_cells(labels: Sequence[str], correlations: Correlations) -> float:
    r"""How many independent cells ``labels`` are worth when their estimates are averaged.

    :math:`k_{\mathrm{eff}} = k^2 / \sum_{ij} \rho_{ij}^2` for the correlations that
    ``correlations`` returns.  The sampling noise of an SE ratio, the ratio of the mean
    reported SE to the empirical SD, has correlation :math:`\rho_{ij}^2` between two cells, so
    the mean of ``k`` ratios has the noise variance of one ratio divided by this number.

    Parameters
    ----------
    labels : Sequence[str]
        The labels of the cells.
    correlations : Callable
        The study's label-correlation function.
    """
    matrix = correlations(labels)
    return float(len(labels) ** 2 / np.sum(matrix**2))


@dataclass(frozen=True)
class PilotGroup:
    """Pilot values of cells that share a law, and the covariance of their pilot noise.

    Parameters
    ----------
    values : numpy.ndarray
        The pilot value of each cell.
    noise : numpy.ndarray
        The covariance of the pilot noise of the cells, one row and column for each cell.
    """

    values: np.ndarray
    noise: np.ndarray

    @property
    def mean_variance(self) -> float:
        """The noise variance of the group mean."""
        return float(np.sum(self.noise)) / len(self.values) ** 2


def heterogeneity_bound(groups: Sequence[PilotGroup], *, level: float = PILOT_LEVEL) -> float:
    r"""The one-sided upper confidence bound on the spread :math:`\tau` between cells.

    A cell of a group has a true value equal to the group's true mean plus a deviation of
    standard deviation :math:`\tau`.  The statistic is the sum over groups of the squared
    deviations of the pilot values from their group mean.  It is a quadratic form in normal
    noise, and a Satterthwaite chi-square matches its first two moments at each
    :math:`\tau`.  The bound is the :math:`\tau` at which the observed statistic sits at the
    ``1 - level`` quantile of that law, and zero if the observed statistic already lies below
    it at :math:`\tau = 0`.  The cells' noise covariance is part of the law, so a spread that
    the noise alone explains yields a small bound.

    Parameters
    ----------
    groups : Sequence[PilotGroup]
        The pilot groups.
    level : float
        The confidence level of the one-sided bound.
    """
    groups = [group for group in groups if len(group.values) > 1]
    if not groups:
        return 0.0
    observed = float(sum(np.sum((group.values - np.mean(group.values)) ** 2) for group in groups))

    def tail(tau: float) -> float:
        eigenvalues: list[float] = []
        for group in groups:
            size = len(group.values)
            centre = np.eye(size) - np.ones((size, size)) / size
            covariance = tau**2 * np.eye(size) + group.noise
            eigenvalues.extend(np.linalg.eigvalsh(centre @ covariance @ centre))
        spectrum = np.clip(np.asarray(eigenvalues), 0.0, None)
        mean = float(spectrum.sum())
        if mean <= 0.0:  # no noise and no spread: the statistic is exactly zero
            return 1.0
        freedom = 2.0 * mean**2 / (2.0 * float(np.sum(spectrum**2)))
        return float(chi2.cdf(observed * freedom / mean, freedom))

    target = 1.0 - level
    if tail(0.0) <= target:
        return 0.0
    upper = 0.01
    while tail(upper) > target:
        upper *= 2.0
    return float(brentq(lambda tau: tail(tau) - target, 0.0, upper, xtol=1e-9))


#: Gauss-Hermite nodes and weights (probabilists') of the pooled predictive integral.
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(31)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def pooled_pass_probability(
    groups: Sequence[PilotGroup],
    tau: float,
    cell_pass: Callable[[np.ndarray], np.ndarray] | Sequence[Callable[[np.ndarray], np.ndarray]],
) -> float:
    r"""The family-wise pass probability of the cells of ``groups`` under the pooled reading.

    The true value of a group's cells is :math:`\mu \sim N(\bar v, \sigma^2)`, where
    :math:`\bar v` is the pilot group mean and :math:`\sigma^2` is its noise variance.  Given
    :math:`\mu`, each cell has the true value :math:`\mu + \tau z` with its own independent
    :math:`z \sim N(0, 1)`.  The probability is the product over groups of
    :math:`E_\mu[\prod_c E_z\, p_c(\mu + \tau z)]` for the cell pass probabilities :math:`p_c`.
    Cells that share a fit are positively dependent, which makes the product a lower bound.

    Parameters
    ----------
    groups : Sequence[PilotGroup]
        The pilot groups.
    tau : float
        The spread between the true values of the cells of a group.
    cell_pass : Callable or Sequence[Callable]
        The probability that one cell passes, given its true value, over an array of values.
        One callable serves every cell.  A sequence gives each cell of the pooled groups its
        own, in the order of the groups' cells.
    """
    nodes = _NODES
    total = 1.0
    offset = 0
    for group in groups:
        size = len(group.values)
        cells = (
            [cell_pass] * size if callable(cell_pass) else list(cell_pass)[offset : offset + size]
        )
        offset += size
        centres = float(np.mean(group.values)) + float(np.sqrt(group.mean_variance)) * nodes
        inner: dict[int, np.ndarray] = {}
        product = np.ones(len(nodes))
        for cell in cells:
            if id(cell) not in inner:
                if tau > 0.0:
                    inner[id(cell)] = cell(centres[:, None] + tau * nodes[None, :]) @ _WEIGHTS
                else:
                    inner[id(cell)] = cell(centres)
            product = product * inner[id(cell)]
        total *= float(np.sum(_WEIGHTS * product))
    return total


def _bias_gate(margins: Margins, replicates: int) -> Callable[[np.ndarray], np.ndarray]:
    """The pass probability of a bias check at ``replicates``, over an array of biases."""

    def gate(bias: np.ndarray) -> np.ndarray:
        return np.asarray(bias_equivalent_probability(margins, bias, replicates))

    return gate


def _calibration_cell(
    margins: Margins, replicates: int, *, bias: float = 0.0, one_sided: bool = False
) -> Callable[[np.ndarray], np.ndarray]:
    """The pass probability of a calibration cell at ``replicates``, over an array of ratios."""

    def cell(ratio: np.ndarray) -> np.ndarray:
        return calibration_pass(margins, replicates, ratio, bias=bias, one_sided=one_sided)

    return cell


# ---------------------------------------------------------------------------- primary


def primary_replicates(
    margins: Margins,
    covered: Mapping[str, int],
    pilot_replicates: int,
    *,
    implementations: int = 2,
    coverage_pass: float = FAMILY_WISE_PASS,
    bias_groups: Sequence[PilotGroup] | None = None,
    bias_pass: float = FAMILY_WISE_PASS,
) -> int:
    """The declared rule for the replications of the primary cells.

    The smallest multiple of :data:`BUDGET_STEP` such that

    * (a) a cell whose true standardized bias is the paired-difference margin
      (:data:`PRIMARY_DESIGN_BIAS`) passes the bias gate with probability at least 0.90;
    * (b) the coverage-floor gate passes with probability at least :data:`PRIMARY_CELL_PASS`
      per cell at coverage :data:`PRIMARY_DESIGN_COVERAGE`;
    * (c) the product of the coverage-floor pass probabilities over every cell of every
      implementation is at least ``coverage_pass``, with each cell at the one-sided 80% lower
      Clopper-Pearson bound of its pilot coverage.  The clause prices the pilot's own sampling
      error, so a pilot that happens to read high cannot under-size the run.  The paired
      cells are not binding, because the two implementations agree; and
    * (d) when ``bias_groups`` is given, the robust probability
      (:func:`robust_probability`) that every bias gate passes is at least ``bias_pass``.  The
      groups carry the pilot biases of the cells, one group for each scenario.

    X32a passes neither ``bias_groups`` nor a share, so it keeps clauses (a) to (c) at 0.95.
    X32b passes :data:`PRIMARY_CLAUSE_SHARE` for both shares and the pilot biases.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    covered : Mapping[str, int]
        The covered count of each primary cell in the pilot.
    pilot_replicates : int
        The number of pilot replications.
    implementations : int
        How many implementations each cell is tested for (cleverly and R).
    coverage_pass : float
        The target of clause (c).
    bias_groups : Sequence[PilotGroup], optional
        The pilot biases of the cells, for clause (d).
    bias_pass : float
        The target of clause (d).
    """
    lower = {
        cell: float(beta.ppf(1.0 - PILOT_LEVEL, count, pilot_replicates - count + 1))
        if count
        else 0.0
        for cell, count in covered.items()
    }
    reading = robust_reading(bias_groups) if bias_groups is not None else None
    replicates = BUDGET_STEP
    while True:
        product = (
            np.prod(
                [
                    coverage_floor_pass_probability(margins, replicates, value)
                    for value in lower.values()
                ]
            )
            ** implementations
        )
        if (
            bias_equivalent_probability(margins, PRIMARY_DESIGN_BIAS, replicates) >= 0.90
            and coverage_floor_pass_probability(margins, replicates, PRIMARY_DESIGN_COVERAGE)
            >= PRIMARY_CELL_PASS
            and product >= coverage_pass
            and (
                reading is None or reading.probability(_bias_gate(margins, replicates)) >= bias_pass
            )
        ):
            return replicates
        replicates += BUDGET_STEP
        if replicates > MAXIMUM_REPLICATES:  # pragma: no cover - a pilot this poor needs a larger n
            raise ValueError("no budget up to 50,000 replications meets the primary rule")


# ---------------------------------------------------------------------------- calibration


def calibration_arms(ratios: Mapping[str, float]) -> dict[str, list[str]]:
    """The cell names of each calibration arm, in the order of ``ratios``."""
    arms: dict[str, list[str]] = {}
    for name in ratios:
        arms.setdefault(name.split("__", 1)[1], []).append(name)
    return arms


def calibration_groups(
    ratios: Mapping[str, float], pilot_replicates: int, correlations: ArmCorrelations
) -> list[PilotGroup]:
    """One pilot group for each calibration arm.

    The pilot noise of a ratio has relative standard deviation ``1 / sqrt(2 (R - 1))``, and
    the noise of two cells of one arm has the squared correlation of their labels.

    Parameters
    ----------
    ratios : Mapping[str, float]
        The pilot SE ratio of each gated cell, by ``label__arm`` name.
    pilot_replicates : int
        The pilot replications behind each ratio.
    correlations : Callable or Mapping[str, Callable]
        The study's label-correlation function, or one for each arm name.
    """
    sampling = 1.0 / (2.0 * (pilot_replicates - 1))
    return [
        PilotGroup(
            np.array([ratios[name] for name in names]),
            sampling
            * (correlations[arm] if isinstance(correlations, Mapping) else correlations)(
                cell_labels(names)
            )
            ** 2,
        )
        for arm, names in calibration_arms(ratios).items()
    ]


def calibration_family_pass_probability(
    margins: Margins, replicates: int, groups: Sequence[PilotGroup], tau: float
) -> float:
    """The probability that every gated calibration cell passes at ``replicates``.

    Each arm reads :func:`pooled_pass_probability` with :func:`calibration_pass` as the cell
    probability.  This is X32a's product reading.
    """
    return pooled_pass_probability(groups, tau, _calibration_cell(margins, replicates))


def calibration_replicates(
    margins: Margins,
    ratios: Mapping[str, float],
    pilot_replicates: int,
    correlations: ArmCorrelations,
    *,
    family_pass: float = FAMILY_WISE_PASS,
) -> int:
    """X32a's rule for the replications of the calibration instrument.

    The smallest multiple of :data:`BUDGET_STEP` at which
    :func:`calibration_family_pass_probability` is at least ``family_pass``.  The true SE
    ratio of each arm is the arm's pilot mean, with the noise variance of that mean, and each
    cell lies :func:`heterogeneity_bound` away from it.  X32b uses :func:`calibration_size`.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    ratios : Mapping[str, float]
        The pilot SE ratio of each gated cell, by ``label__arm`` name.
    pilot_replicates : int
        The pilot replications behind each ratio.
    correlations : Callable or Mapping[str, Callable]
        The study's label-correlation function, or one for each arm name.
    family_pass : float
        The family-wise pass probability the rule requires.
    """
    groups = calibration_groups(ratios, pilot_replicates, correlations)
    tau = heterogeneity_bound(groups)
    replicates = BUDGET_STEP
    while calibration_family_pass_probability(margins, replicates, groups, tau) < family_pass:
        replicates += BUDGET_STEP
        if replicates > MAXIMUM_REPLICATES:  # pragma: no cover - a pilot this far off needs a fix
            raise ValueError("no budget up to 50,000 replications meets the calibration rule")
    return replicates


def calibration_failure_bound(
    margins: Margins,
    replicates: int,
    groups: Sequence[PilotGroup],
    tau: float,
    names: Sequence[str],
    *,
    one_sided_arms: Sequence[str] = (),
    biases: Mapping[str, float] | None = None,
    joint_coverage: float | None = None,
) -> float:
    """The union bound of the predictive failure probabilities of the calibration terms.

    Each arm is a term, read with :func:`pooled_pass_probability` and the common ``tau``.
    The C4 joint band is one more term.  The union bound holds under any dependence between
    the terms.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of every arm.
    groups : Sequence[PilotGroup]
        The pilot groups, one for each arm (:func:`calibration_groups`).
    tau : float
        The spread between the true ratios of the cells of an arm.
    names : Sequence[str]
        The cell names, in the order of the groups' cells.
    one_sided_arms : Sequence[str]
        The arms whose cells read only the lower end of the band (``SDR`` C5).
    biases : Mapping[str, float], optional
        The pilot standardized bias of each cell.  The coverage half reads the Wald coverage at
        the cell's ratio and this bias.
    joint_coverage : float, optional
        The pilot joint coverage of the C4 band.
    """
    cells: list[Callable[[np.ndarray], np.ndarray]] = []
    shared = {
        one_sided: _calibration_cell(margins, replicates, one_sided=one_sided)
        for one_sided in (False, True)
    }
    for name in names:
        sided = name.split("__", 1)[1] in one_sided_arms
        if biases is None:
            cells.append(shared[sided])
        else:
            cells.append(_calibration_cell(margins, replicates, bias=biases[name], one_sided=sided))
    failure = 0.0
    offset = 0
    for group in groups:
        size = len(group.values)
        failure += 1.0 - pooled_pass_probability([group], tau, cells[offset : offset + size])
        offset += size
    if joint_coverage is not None:
        failure += 1.0 - joint_band_pass_probability(margins, replicates, joint_coverage)
    return failure


class CalibrationSize(NamedTuple):
    """The size :func:`calibration_size` chooses, and whether it meets the budget."""

    replicates: int
    #: ``False`` at the cap: no size up to the cap meets the budget, and the gate decides.
    budget_met: bool


def calibration_size(
    margins: Margins,
    ratios: Mapping[str, float],
    pilot_replicates: int,
    correlations: ArmCorrelations,
    *,
    budget: float = SIZED_FAIL,
    one_sided_arms: Sequence[str] = (),
    biases: Mapping[str, float] | None = None,
    joint_coverage: float | None = None,
    cap: int = CALIBRATION_CAP,
) -> CalibrationSize:
    """The declared rule for the replications of the calibration instrument (X32b).

    The smallest multiple of :data:`BUDGET_STEP` at which :func:`calibration_failure_bound`
    is at most ``budget``.  If no size up to ``cap`` meets it, the size is ``cap`` and
    ``budget_met`` is ``False``.  The study module then runs at the cap and reads the gate.
    The failure bound falls as the size grows, so the search brackets the size on a coarse
    grid of 4,000 and bisects.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    ratios : Mapping[str, float]
        The pilot SE ratio of each gated cell, by ``label__arm`` name.
    pilot_replicates : int
        The pilot replications behind each ratio.
    correlations : Callable or Mapping[str, Callable]
        The study's label-correlation function, or one for each arm name.
    budget : float
        The failure probability the calibration instrument may have.
    one_sided_arms : Sequence[str]
        The arms whose cells read only the lower end of the band.
    biases : Mapping[str, float], optional
        The pilot standardized bias of each cell.
    joint_coverage : float, optional
        The pilot joint coverage of the C4 band.
    cap : int
        The largest size the rule chooses.
    """
    groups = calibration_groups(ratios, pilot_replicates, correlations)
    tau = heterogeneity_bound(groups)
    names = [name for arm in calibration_arms(ratios).values() for name in arm]

    def fails(replicates: int) -> float:
        return calibration_failure_bound(
            margins,
            replicates,
            groups,
            tau,
            names,
            one_sided_arms=one_sided_arms,
            biases=biases,
            joint_coverage=joint_coverage,
        )

    previous = 0
    hit = None
    for replicates in (*range(4_000, cap + 1, 4_000), cap):
        if fails(replicates) <= budget:
            hit = replicates
            break
        previous = replicates
    if hit is None:
        return CalibrationSize(cap, False)
    low, high = previous // BUDGET_STEP, hit // BUDGET_STEP
    while high - low > 1:
        middle = (low + high) // 2
        if fails(middle * BUDGET_STEP) <= budget:
            high = middle
        else:
            low = middle
    return CalibrationSize(high * BUDGET_STEP, True)


# ---------------------------------------------------------------------------- property bias


def bias_families(biases: Mapping[str, float]) -> dict[str, list[str]]:
    """The cell names of each family, in the order of ``biases``."""
    families: dict[str, list[str]] = {}
    for name in biases:
        families.setdefault(name.split("/", 1)[0], []).append(name)
    return families


def property_bias_groups(
    biases: Mapping[str, float], pilot_replicates: int, correlations: Correlations
) -> dict[str, PilotGroup]:
    """One pilot group for each family: the standardized biases of its cells.

    A family shares a finite-sample bias, so its cells share a mean that differs from zero and
    from the means of other families.  The pilot noise of a standardized bias has variance
    ``1 / R``.  Cells of one family share each draw, so their noise has the correlation of
    their labels' efficient influence curves.  A control cell and a positive cell of one label
    count as the same label, which is the most cautious reading.

    Parameters
    ----------
    biases : Mapping[str, float]
        The pilot bias of each cell, by ``family/cell`` name.
    pilot_replicates : int
        The pilot replications behind each bias.
    correlations : Callable
        The study's label-correlation function.
    """
    return {
        family: PilotGroup(
            np.array([biases[name] for name in names]),
            correlations(cell_labels(names)) / pilot_replicates,
        )
        for family, names in bias_families(biases).items()
    }


def property_bias_pool(
    biases: Mapping[str, float], pilot_replicates: int, correlations: Correlations
) -> PilotGroup:
    """The cross-check reading: every cell of every family in one group.

    A bias of a consistent estimator is zero up to a finite-sample term, so this reading gives
    all families one mean.  The spread between cells then includes the spread between
    families.  Cells of two families share no draw, so the noise is block diagonal.
    """
    groups = property_bias_groups(biases, pilot_replicates, correlations)
    return PilotGroup(
        np.concatenate([group.values for group in groups.values()]),
        block_diag(*(group.noise for group in groups.values())),
    )


def _bias_cell_passes(
    margins: Margins, biases: Mapping[str, float], replicates: Mapping[str, int]
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """The pass probability of each cell, in the order of ``biases``, at its family's size."""
    by_replicates = {count: _bias_gate(margins, count) for count in set(replicates.values())}
    return [by_replicates[replicates[name.split("/", 1)[0]]] for name in biases]


def property_bias_pass_probability(
    margins: Margins,
    biases: Mapping[str, float],
    pilot_replicates: int,
    replicates: Mapping[str, int],
    correlations: Correlations,
    tau: float | None = None,
    *,
    pooled: bool = False,
) -> float:
    """The probability that every bias check passes.

    The cell probability is :func:`bias_equivalent_probability` at the replications of the
    cell's family.  :func:`pooled_pass_probability` carries each group's pilot mean, the noise
    of that mean, and the spread ``tau`` between cells.  The groups are the families
    (:func:`property_bias_groups`).  With ``pooled=True`` the group is every cell
    (:func:`property_bias_pool`), the reported cross-check.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    biases : Mapping[str, float]
        The pilot bias of each cell, by ``family/cell`` name.
    pilot_replicates : int
        The pilot replications behind each bias.
    replicates : Mapping[str, int]
        The replications of each family.
    correlations : Callable
        The study's label-correlation function.
    tau : float, optional
        The spread between the true biases of the cells.  The default is the
        :func:`heterogeneity_bound` of the pilot under the same grouping.
    pooled : bool
        Read one group for all cells.
    """
    groups = (
        [property_bias_pool(biases, pilot_replicates, correlations)]
        if pooled
        else list(property_bias_groups(biases, pilot_replicates, correlations).values())
    )
    if tau is None:
        tau = heterogeneity_bound(groups)
    return pooled_pass_probability(groups, tau, _bias_cell_passes(margins, biases, replicates))


def property_bias_replicates(
    margins: Margins,
    biases: Mapping[str, float],
    pilot_replicates: int,
    correlations: Correlations,
    *,
    family_pass: float = FAMILY_WISE_PASS,
    per_family: float | None = None,
) -> dict[str, int]:
    """The declared rule for the replications of each property family.

    The families draw independent streams, so the probability that every bias check passes is
    the product over families.  Each family takes the smallest multiple of
    :data:`PROPERTY_STEP` at which its own pass probability, with :math:`\\tau` at its 80%
    bound read from every family together, reaches ``family_pass`` to the power ``1 / F`` for
    ``F`` families.  The product is then at least ``family_pass``.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    biases : Mapping[str, float]
        The pilot bias of each cell, by ``family/cell`` name.
    pilot_replicates : int
        The pilot replications behind each bias.
    correlations : Callable
        The study's label-correlation function.
    family_pass : float
        The family-wise pass probability the rule requires.
    per_family : float, optional
        The pass probability each family must reach, in place of ``family_pass ** (1 / F)``.
        X32b passes :func:`family_share` (``1 - d / F``).
    """
    groups = property_bias_groups(biases, pilot_replicates, correlations)
    tau = heterogeneity_bound(list(groups.values()))
    target = family_pass ** (1.0 / len(groups)) if per_family is None else per_family
    sizes: dict[str, int] = {}
    for family, group in groups.items():
        count = PROPERTY_STEP
        while True:

            def cell(bias: np.ndarray, count: int = count) -> np.ndarray:
                return np.asarray(bias_equivalent_probability(margins, bias, count))

            if pooled_pass_probability([group], tau, cell) >= target:
                break
            count += PROPERTY_STEP
            if count > MAXIMUM_REPLICATES:  # pragma: no cover - a pilot this far off needs a fix
                raise ValueError(f"no size up to 50,000 replications meets the rule for {family}")
        sizes[family] = count
    return sizes


# ---------------------------------------------------------------------------- the robust reading


def _residual_variance(noise: np.ndarray, cell: int, tau: float) -> float:
    """The variance of ``b_i`` minus the mean of the other cells, under noise and spread ``tau``."""
    size = noise.shape[0]
    weights = np.full(size, -1.0 / (size - 1))
    weights[cell] = 1.0
    return float(weights @ noise @ weights) + tau**2 * (1.0 + 1.0 / (size - 1))


class RobustReading(NamedTuple):
    """The pilot groups after the outlier test: what :func:`robust_probability` evaluates."""

    #: The judged groups without their outliers.
    kept: list[PilotGroup]
    #: The groups that only feed the spread.
    extra: list[PilotGroup]
    #: The pooled spread, read from the kept and extra groups.
    tau: float
    #: The pilot bias and the pilot-noise variance of each outlier.
    outliers: list[tuple[float, float]]

    def probability(self, cell_pass: Callable[[np.ndarray], np.ndarray]) -> float:
        """The probability that every cell passes: the pooled failure and each outlier's."""
        failure = 1.0 - pooled_pass_probability(self.kept, self.tau, cell_pass)
        for bias, variance in self.outliers:
            failure += float(1.0 - cell_pass(bias + np.sqrt(variance) * _NODES) @ _WEIGHTS)
        return 1.0 - failure


def robust_reading(
    groups: Sequence[PilotGroup], *, extra: Sequence[PilotGroup] = ()
) -> RobustReading:
    r"""Apply the declared outlier test to the judged ``groups`` (review C2).

    ``groups`` are the judged groups: a primary's scenarios, or one property family.
    ``extra`` are groups that only feed :math:`\tau`: the study's other property families.
    In each judged group of ``m >= 3`` cells, the candidate is the cell with the largest
    :math:`|b_i - \bar b_{-i}|` in units of the pilot-noise standard deviation of that
    difference.  Its standardized residual uses the noise variance plus
    :math:`\tau_{-i}^2 (1 + 1 / (m - 1))`, with :math:`\tau_{-i}` re-read without the
    candidate.  The candidate is an outlier when the residual exceeds
    :math:`\Phi^{-1}(1 - 0.01 / (2m))`.  An outlier leaves the pooled reading and is read
    alone: its true value is :math:`N(b_i, \sigma_i^2)` with its own pilot-noise variance and
    no pooling.  The reading does not depend on the replications, so a sizing rule computes it
    once.

    Parameters
    ----------
    groups : Sequence[PilotGroup]
        The judged pilot groups.
    extra : Sequence[PilotGroup]
        Groups that feed the spread and are not judged.
    """
    judged = list(groups)
    extra = list(extra)
    kept: list[PilotGroup] = []
    outliers: list[tuple[float, float]] = []
    for position, group in enumerate(judged):
        values, noise = group.values, group.noise
        size = len(values)
        if size < 3:
            kept.append(group)
            continue
        noise_sd = np.array([np.sqrt(_residual_variance(noise, i, 0.0)) for i in range(size)])
        residual = np.array(
            [values[i] - (values.sum() - values[i]) / (size - 1) for i in range(size)]
        )
        candidate = int(np.argmax(np.abs(residual) / noise_sd))
        keep = np.arange(size) != candidate
        reduced = PilotGroup(values[keep], noise[np.ix_(keep, keep)])
        others = [g for j, g in enumerate(judged) if j != position] + [reduced] + extra
        tau_without = heterogeneity_bound(others)
        score = abs(residual[candidate]) / np.sqrt(
            _residual_variance(noise, candidate, tau_without)
        )
        if score > norm.ppf(1.0 - OUTLIER_LEVEL / (2.0 * size)):
            kept.append(reduced)
            outliers.append((float(values[candidate]), float(noise[candidate, candidate])))
        else:
            kept.append(group)
    return RobustReading(kept, extra, heterogeneity_bound(kept + extra), outliers)


def robust_probability(
    groups: Sequence[PilotGroup],
    cell_pass: Callable[[np.ndarray], np.ndarray],
    *,
    extra: Sequence[PilotGroup] = (),
) -> tuple[float, int]:
    r"""The pooled reading of ``groups`` after a declared outlier test (review C2).

    :func:`robust_reading` applies the test.  The probability is 1 minus the pooled failure
    probability of the remaining cells (:math:`\tau` re-read without the outliers) minus the
    failure probability of each outlier (union bound).  With no outlier it is
    :func:`pooled_pass_probability` at the 80% bound of :func:`heterogeneity_bound`.

    Parameters
    ----------
    groups : Sequence[PilotGroup]
        The judged pilot groups.
    cell_pass : Callable
        The probability that one cell passes, given its true value, over an array of values.
    extra : Sequence[PilotGroup]
        Groups that feed the spread and are not judged.

    Returns
    -------
    tuple of (float, int)
        The probability and the number of outliers.
    """
    reading = robust_reading(groups, extra=extra)
    return reading.probability(cell_pass), len(reading.outliers)


# ---------------------------------------------------------------------------- the n rule


def n_rule_statistic(
    margins: Margins,
    groups: Sequence[PilotGroup],
    *,
    extra: Sequence[PilotGroup] = (),
    replicates: int = N_RULE_REFERENCE_REPLICATES,
) -> float:
    """The robust probability that every bias check of ``groups`` passes at ``replicates``.

    The cell probability is :func:`bias_equivalent_probability`.  A primary judges its
    scenarios.  A property family judges its own group, and the other families of the study
    feed the spread as ``extra``.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    groups : Sequence[PilotGroup]
        The pilot biases of the checks: one group for each scenario or family judged.
    extra : Sequence[PilotGroup]
        Groups that only feed the spread.
    replicates : int
        The replications at which the checks are read.
    """
    return robust_probability(groups, _bias_gate(margins, replicates), extra=extra)[0]


def n_rule_fires(
    margins: Margins,
    groups: Sequence[PilotGroup],
    share: float,
    *,
    extra: Sequence[PilotGroup] = (),
    replicates: int = N_RULE_REFERENCE_REPLICATES,
) -> bool:
    """Whether the declared ``n`` doubles: :func:`n_rule_statistic` is below ``share``.

    The share is :data:`PRIMARY_CLAUSE_SHARE` for a primary and :func:`family_share` for a
    property family.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    groups : Sequence[PilotGroup]
        The pilot biases of the checks.
    share : float
        The group's share of the study's pass probability.
    extra : Sequence[PilotGroup]
        Groups that only feed the spread.
    replicates : int
        The replications at which the checks are read.
    """
    return n_rule_statistic(margins, groups, extra=extra, replicates=replicates) < share


def coverage_half_probability(
    margins: Margins, replicates: int, bias: float | np.ndarray
) -> np.ndarray:
    """The coverage half of the calibration gate at SE ratio 1, as a function of the bias.

    The Wald coverage at ratio 1 and standardized ``bias`` must put the exact interval of the
    covered count inside the calibration band.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    replicates : int
        The replications of the arm.
    bias : float or numpy.ndarray
        The true standardized bias.
    """
    accepted = accepted_counts(margins, replicates)
    value = np.asarray(bias, dtype=float)
    if accepted is None:
        return np.zeros_like(value)
    coverage = wald_coverage(margins, np.ones_like(value), value)
    return np.asarray(
        binom.cdf(accepted[1], replicates, coverage)
        - binom.cdf(accepted[0] - 1, replicates, coverage)
    )


def c6_rule_statistic(
    margins: Margins,
    c6: PilotGroup,
    *,
    extra: Sequence[PilotGroup] = (),
    replicates: int = CALIBRATION_CAP,
) -> float:
    """The robust probability that every C6 cell passes the coverage half at the cap.

    C6 has no bias gate.  Its gates are the SE-ratio band and the coverage band.  More
    replications cure every bias at which the Wald coverage lies inside the band, and the
    calibration rule prices it.  A larger ``n`` is the only cure for a bias that fails the
    band at the cap.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    c6 : PilotGroup
        The pilot biases of the C6 cells.
    extra : Sequence[PilotGroup]
        The pilot biases of the other calibration arms, which feed the spread.
    replicates : int
        The replications at which the coverage half is read.
    """

    def cell(bias: np.ndarray) -> np.ndarray:
        return coverage_half_probability(margins, replicates, bias)

    return robust_probability([c6], cell, extra=extra)[0]


def c6_rule_fires(
    margins: Margins,
    c6: PilotGroup,
    *,
    extra: Sequence[PilotGroup] = (),
    share: float = C6_SHARE,
    replicates: int = CALIBRATION_CAP,
) -> bool:
    """Whether the C6 ``n`` doubles: :func:`c6_rule_statistic` is below ``share``.

    Parameters
    ----------
    margins : Margins
        The study's margins.
    c6 : PilotGroup
        The pilot biases of the C6 cells.
    extra : Sequence[PilotGroup]
        The pilot biases of the other calibration arms.
    share : float
        The share of the rule.
    replicates : int
        The replications at which the coverage half is read.
    """
    return c6_rule_statistic(margins, c6, extra=extra, replicates=replicates) < share
