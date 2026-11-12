r"""The arm-pooled joint predictive sizing rule of the Fix-6 fold-wise DR-TMLE records.

Three gated records of Fix-6 are sized by this rule: ``drtmle-foldwise-n1500`` (its two paired
property cells), ``multi-arm-drtmle-foldwise`` and ``stratified-drtmle-foldwise`` (their primary
phases).  The rule is the one :mod:`tests.studies.drtmle_foldwise_properties` declared for
X83, written once here for any set of planning summaries, so that a sizing pilot only has to
supply numbers.

**The rows each budget is judged by.**  A primary row reads the shared ``Margins()`` of
``performance-tests.csv``: the 99% Student interval of the standardized bias inside 0.25
empirical SD, the 99% exact coverage interval above the 0.90 floor, and the 99% interval of the
SE ratio inside the 0.8 to 1.2 sanity band.  A ``double_robustness`` property cell reads the
bias clause alone, because its union-model SE band (0.1 to 10) never binds a fit whose SE ratio
lies near one.  A control row must put its bias interval wholly outside the margin.

**The predictive rule.**  Each :class:`Pilot` is the planning summary of the rows one fit
estimates on one draw: each row's standardized bias ``mean(error) / sd(error)`` with Monte Carlo
SE ``1 / sqrt(R_pilot)``, the rows' per-replication error correlation, and for a primary row its
SE ratio and coverage.  Separate pilots are independent.  Rows that the same mechanism drives
may be declared as the arms of a regime (:attr:`Rule.regimes`): their mean and between-arm SD
``tau`` are estimated together, ``tau`` at its one-sided 80% Q-profile upper bound, and each
arm's planning value is its normal-normal posterior; every other row of an arm's pilot moves
with the arm by its conditional regression.  :meth:`Rule.planning_draws` integrates over this
planning uncertainty with :attr:`Rule.draws` joint draws from :attr:`Rule.seed`.  At each draw
the study's own Monte Carlo error is drawn too, and a unit passes when every clause of every
row passes.  A unit's budget is the smallest multiple of :attr:`Rule.step` at or above its floor
whose predictive pass reaches the unit share ``target ** (1 / len(units))``.  A unit that no
budget up to :attr:`Rule.ceiling` sizes is reported unsized, and its record's guard refuses the
declared run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from scipy.stats import t as student

from tests.studies.evidence.registry import Margins
from tests.studies.evidence.schema import truth_on_inference_scale

#: The clauses the rule reads.  ``se_ratio`` and ``coverage`` bind only the rows whose pilot
#: gives them.
CLAUSES = ("bias", "se_ratio", "coverage")


@dataclass(frozen=True)
class Pilot:
    """The planning summary of the rows one fit estimates on one draw.

    Parameters
    ----------
    source : str
        Where the summary comes from: a committed artifact or a sizing pilot's CSV.
    replicates : int
        The replications the summary reads.
    rows : tuple of str
        The rows, each an estimand or a cell.
    bias : tuple of float
        Each row's standardized bias, ``mean(error) / sd(error)``.
    correlation : tuple of tuple of float
        The per-replication correlation of the rows' errors.
    controls : tuple of str
        The rows whose bias must fall outside the margin.
    se_ratio : tuple of float
        Each row's mean reported SE over its error SD; empty for a property cell.
    coverage : tuple of float
        Each row's 95% Wald coverage; empty for a property cell.
    """

    source: str
    replicates: int
    rows: tuple[str, ...]
    bias: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    controls: tuple[str, ...] = ()
    se_ratio: tuple[float, ...] = ()
    coverage: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        size = len(self.rows)
        if len(self.bias) != size or np.shape(self.correlation) != (size, size):
            raise ValueError(f"{self.source}: bias and correlation must match the {size} rows")
        if len(self.se_ratio) != len(self.coverage) or (
            self.se_ratio and len(self.se_ratio) != size
        ):
            raise ValueError(f"{self.source}: give an SE ratio and a coverage for every row")
        unknown = set(self.controls) - set(self.rows)
        if unknown:
            raise ValueError(f"{self.source}: controls name unknown rows {sorted(unknown)}")


def single(source: str, replicates: int, row: str, bias: float) -> Pilot:
    """The planning summary of one property cell, read on its bias alone.

    Parameters
    ----------
    source : str
        Where the planning value comes from.
    replicates : int
        The replications behind it, which set its planning uncertainty.
    row : str
        The cell.
    bias : float
        Its standardized bias.

    Returns
    -------
    Pilot
        A one-row summary with no SE-ratio or coverage clause.
    """
    return Pilot(source, replicates, (row,), (bias,), ((1.0,),))


def pilot_from_rows(
    rows: pd.DataFrame,
    *,
    source: str,
    estimands: Sequence[str] | None = None,
    calibration: bool = True,
) -> Pilot:
    """The planning summary of one fit group, from rows in the replication schema.

    The errors are read on each row's inference scale, as ``performance-tests.csv`` reads them:
    ``inference_estimate`` minus the truth on that scale, so a ratio contrast is planned on its
    log scale.  Rows without ``inference_estimate`` (property rows) read ``estimate`` on the
    identity scale.  Only replications that report every estimand enter the correlation.

    Parameters
    ----------
    rows : pandas.DataFrame
        One fit group's rows: ``replicate``, ``estimand``, ``truth``, ``std_error``,
        ``covered`` and ``estimate``, with ``inference_estimate`` and ``inference_scale`` when
        the group has them.
    source : str
        Where the rows come from.
    estimands : sequence of str or None
        The rows of the summary, in order; every estimand of ``rows`` when ``None``.
    calibration : bool
        Whether the summary carries each row's SE ratio and coverage (a primary row).

    Returns
    -------
    Pilot
        The summary.
    """
    names = tuple(sorted(rows["estimand"].unique())) if estimands is None else tuple(estimands)
    frame = rows.loc[rows["estimand"].isin(names)].copy()
    if "inference_estimate" in frame:
        frame["value"] = frame["inference_estimate"].astype(float)
        frame["target"] = [
            truth_on_inference_scale(str(name), float(truth), str(scale))
            for name, truth, scale in zip(
                frame["estimand"], frame["truth"], frame["inference_scale"], strict=True
            )
        ]
    else:
        frame["value"] = frame["estimate"].astype(float)
        frame["target"] = frame["truth"].astype(float)
    frame["error"] = frame["value"] - frame["target"]
    errors = frame.pivot(index="replicate", columns="estimand", values="error").loc[:, list(names)]
    errors = errors.dropna()
    if len(errors) < 3:
        raise ValueError(f"{source}: fewer than three complete replications")
    bias: list[float] = []
    se_ratio: list[float] = []
    coverage: list[float] = []
    for name in names:
        group = frame.loc[frame["estimand"] == name]
        error = group["error"].to_numpy(dtype=float)
        spread = float(np.std(group["value"].to_numpy(dtype=float), ddof=1))
        bias.append(float(np.mean(error) / spread))
        se_ratio.append(float(group["std_error"].astype(float).mean() / spread))
        coverage.append(float(group["covered"].astype(float).mean()))
    correlation = np.corrcoef(errors.to_numpy(dtype=float), rowvar=False).reshape(
        len(names), len(names)
    )
    return Pilot(
        source,
        len(errors),
        names,
        tuple(round(value, 4) for value in bias),
        tuple(tuple(round(float(value), 4) for value in row) for row in correlation),
        se_ratio=tuple(round(value, 4) for value in se_ratio) if calibration else (),
        coverage=tuple(round(value, 4) for value in coverage) if calibration else (),
    )


def _factor(correlation: Sequence[Sequence[float]]) -> np.ndarray:
    """A square root of a correlation matrix that may be singular (``ate = ey1 - ey0``)."""
    values, vectors = np.linalg.eigh(np.asarray(correlation, dtype=float))
    return np.asarray(vectors * np.sqrt(np.clip(values, 0.0, None)))


@dataclass(frozen=True)
class _Draws:
    bias: Mapping[str, np.ndarray]
    se_ratio: Mapping[str, np.ndarray]
    coverage: Mapping[str, np.ndarray]
    noise: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Rule:
    """The arm-pooled joint predictive rule over a set of planning summaries.

    Parameters
    ----------
    pilots : Mapping
        Name to :class:`Pilot`.
    units : Mapping
        Unit name to ``(pilot names, floor)``: the rows one budget serves and the smallest
        budget the unit may take.
    seed : int
        The seed of the planning draws.  Fixed, so every number the rule returns is exact.
    regimes : tuple
        Each regime is a tuple of arms ``(pilot name, row)`` that one mechanism drives.
    margins : Margins
        The margins every row is judged by.
    target : float
        The family-wise pass probability the budgets must reach when nothing is wrong.
    step : int
        The budget grid.
    ceiling : int
        The largest budget the rule tries.
    draws : int
        The joint planning draws.
    tau_level : float
        The one-sided level of the between-arm SD's upper bound.
    """

    pilots: Mapping[str, Pilot]
    units: Mapping[str, tuple[tuple[str, ...], int]]
    seed: int
    regimes: tuple[tuple[tuple[str, str], ...], ...] = ()
    margins: Margins = field(default_factory=Margins)
    target: float = 0.95
    step: int = 400
    ceiling: int = 8_000
    draws: int = 20_000
    tau_level: float = 0.80

    def __post_init__(self) -> None:
        for unit, (names, floor) in self.units.items():
            unknown = set(names) - set(self.pilots)
            if unknown:
                raise ValueError(f"unit {unit!r} names unknown pilots {sorted(unknown)}")
            if floor < 2:
                raise ValueError(f"unit {unit!r} needs a floor of at least two replications")
        for regime in self.regimes:
            if len(regime) < 2:
                raise ValueError("a regime pools at least two arms")
            for name, row in regime:
                if row not in self.pilots[name].rows:
                    raise ValueError(f"regime arm {(name, row)!r} names no pilot row")

    @property
    def unit_target(self) -> float:
        """The pass share each unit must reach, ``target ** (1 / len(units))``."""
        return float(self.target ** (1.0 / len(self.units)))

    @property
    def z(self) -> float:
        """The two-sided normal critical value of the coverage and SE-ratio intervals."""
        return float(norm.ppf(0.5 + self.margins.confidence_level / 2))

    def regime_tau(self, regime: Sequence[tuple[str, str]]) -> tuple[float, float, float]:
        """A regime's between-arm SD at its :attr:`tau_level` upper bound, and the pooled mean.

        The bound is the one-sided Q-profile bound: the ``tau`` at which the generalized ``Q``
        statistic equals the ``1 - tau_level`` quantile of chi-square on ``k - 1`` degrees of
        freedom, or zero when ``Q`` at zero already lies below it.

        Parameters
        ----------
        regime : sequence of tuple
            The arms ``(pilot name, row)``.

        Returns
        -------
        tuple of float
            ``(tau, pooled mean, SE of the pooled mean)``.
        """
        y = np.array(
            [self.pilots[name].bias[self.pilots[name].rows.index(row)] for name, row in regime]
        )
        v = np.array([1.0 / self.pilots[name].replicates for name, _ in regime])

        def q(t2: float) -> float:
            w = 1.0 / (v + t2)
            mu = float(np.sum(w * y) / np.sum(w))
            return float(np.sum(w * (y - mu) ** 2))

        critical = float(chi2.ppf(1.0 - self.tau_level, len(y) - 1))
        t2 = 0.0 if q(0.0) <= critical else float(brentq(lambda t: q(t) - critical, 0.0, 1.0))
        w = 1.0 / (v + t2)
        return math.sqrt(t2), float(np.sum(w * y) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))

    @cached_property
    def planning_draws(self) -> _Draws:
        """The joint planning draws and the study's own Monte Carlo noise, from one seed.

        Each pilot's rows are drawn jointly normal around their summary with covariance
        ``correlation / replicates``.  Each regime's arms are then replaced by their
        normal-normal posterior at the bounded ``tau``, with the pooled mean drawn once per
        planning draw, so the arms share its error.  The other rows of an arm's pilot follow
        the arms by the conditional regression of the pilot's correlation.  The SE ratio and
        coverage of a primary row are drawn around the pilot's value with SEs
        ``ratio / sqrt(2 R)`` and ``sqrt(c (1 - c) / R)``.

        Returns
        -------
        _Draws
            The planning values and standard normal noise for the study's realization.
        """
        rng = np.random.default_rng(self.seed)
        bias: dict[str, np.ndarray] = {}
        for name, pilot in self.pilots.items():
            shocks = rng.standard_normal((self.draws, len(pilot.rows)))
            bias[name] = np.asarray(pilot.bias) + shocks @ _factor(pilot.correlation).T / math.sqrt(
                pilot.replicates
            )
        for regime in self.regimes:
            tau, mean, spread = self.regime_tau(regime)
            mu = mean + spread * rng.standard_normal(self.draws)
            moved: dict[str, dict[int, np.ndarray]] = {}
            for name, row in regime:
                pilot = self.pilots[name]
                index = pilot.rows.index(row)
                v = 1.0 / pilot.replicates
                shrink = tau**2 / (v + tau**2)
                arm = (
                    mu
                    + shrink * (pilot.bias[index] - mu)
                    + math.sqrt(v * shrink) * rng.standard_normal(self.draws)
                )
                moved.setdefault(name, {})[index] = arm - bias[name][:, index]
            for name, shifts in moved.items():
                correlation = np.asarray(self.pilots[name].correlation, dtype=float)
                arms = sorted(shifts)
                others = [i for i in range(correlation.shape[0]) if i not in shifts]
                delta = np.column_stack([shifts[i] for i in arms])
                if others:
                    gain = correlation[np.ix_(others, arms)] @ np.linalg.pinv(
                        correlation[np.ix_(arms, arms)]
                    )
                    bias[name][:, others] += delta @ gain.T
                bias[name][:, arms] += delta
        se_ratio: dict[str, np.ndarray] = {}
        coverage: dict[str, np.ndarray] = {}
        for name, pilot in self.pilots.items():
            if not pilot.se_ratio:
                continue
            ratio = np.asarray(pilot.se_ratio)
            covered = np.asarray(pilot.coverage)
            shape = (self.draws, len(pilot.rows))
            se_ratio[name] = ratio + rng.standard_normal(shape) * ratio / math.sqrt(
                2 * pilot.replicates
            )
            coverage[name] = np.clip(
                covered
                + rng.standard_normal(shape) * np.sqrt(covered * (1 - covered) / pilot.replicates),
                0.0,
                1.0,
            )
        noise = {
            name: tuple(rng.standard_normal((self.draws, len(pilot.rows))) for _ in range(3))
            for name, pilot in self.pilots.items()
        }
        return _Draws(bias, se_ratio, coverage, noise)  # type: ignore[arg-type]

    def pilot_passes(
        self, name: str, replicates: int, clauses: tuple[str, ...] = CLAUSES
    ) -> np.ndarray:
        """Whether one pilot's rows all pass at ``replicates``, at each planning draw.

        Parameters
        ----------
        name : str
            A key of :attr:`pilots`.
        replicates : int
            The budget.
        clauses : tuple of str
            The clauses read, a subset of :data:`CLAUSES`; a test drops one to witness it.

        Returns
        -------
        numpy.ndarray
            One boolean per planning draw.
        """
        pilot = self.pilots[name]
        draws = self.planning_draws
        bias_noise, ratio_noise, coverage_noise = draws.noise[name]
        root = math.sqrt(replicates)
        level = self.margins.confidence_level
        half = float(student.ppf(0.5 + level / 2, replicates - 1)) / root
        estimate = draws.bias[name] + bias_noise @ _factor(pilot.correlation).T / root
        margin = self.margins.standardized_bias
        passed = np.ones(self.draws, dtype=bool)
        if "bias" in clauses:
            for index, row in enumerate(pilot.rows):
                size = np.abs(estimate[:, index])
                passed &= size - half > margin if row in pilot.controls else size + half <= margin
        if pilot.se_ratio and "se_ratio" in clauses:
            ratio = draws.se_ratio[name]
            read = ratio + ratio_noise * ratio / math.sqrt(2 * replicates)
            width = self.z * read / math.sqrt(2 * replicates)
            low, high = self.margins.se_ratio_sanity
            passed &= ((read - width >= low) & (read + width <= high)).all(axis=1)
        if pilot.coverage and "coverage" in clauses:
            covered = draws.coverage[name]
            read = np.clip(
                covered + coverage_noise * np.sqrt(covered * (1 - covered) / replicates), 0.0, 1.0
            )
            floor = read - self.z * np.sqrt(read * (1 - read) / replicates)
            passed &= (floor >= self.margins.coverage_floor).all(axis=1)
        return passed

    def unit_passes(
        self, unit: str, replicates: int, clauses: tuple[str, ...] = CLAUSES
    ) -> np.ndarray:
        """Whether every row of a unit passes at ``replicates``, at each planning draw.

        Parameters
        ----------
        unit : str
            A key of :attr:`units`.
        replicates : int
            The budget.
        clauses : tuple of str
            The clauses read.

        Returns
        -------
        numpy.ndarray
            One boolean per planning draw.
        """
        names, _ = self.units[unit]
        passed = np.ones(self.draws, dtype=bool)
        for name in names:
            passed &= self.pilot_passes(name, replicates, clauses)
        return passed

    def unit_pass(self, unit: str, replicates: int, clauses: tuple[str, ...] = CLAUSES) -> float:
        """A unit's predictive pass probability at ``replicates``.

        Parameters
        ----------
        unit : str
            A key of :attr:`units`.
        replicates : int
            The budget.
        clauses : tuple of str
            The clauses read.

        Returns
        -------
        float
            The share of planning draws at which every row passes.
        """
        return float(self.unit_passes(unit, replicates, clauses).mean())

    def grid(self, unit: str) -> range:
        """The budgets the rule tries for a unit: multiples of :attr:`step` from its floor.

        Parameters
        ----------
        unit : str
            A key of :attr:`units`.

        Returns
        -------
        range
            The candidate budgets, ascending, up to :attr:`ceiling`.
        """
        _, floor = self.units[unit]
        start = self.step * math.ceil(floor / self.step)
        return range(start, self.ceiling + 1, self.step)

    def budget(self, unit: str) -> tuple[int, float, bool]:
        """``(R, predictive pass, sized)``: the smallest grid budget reaching the unit share.

        Parameters
        ----------
        unit : str
            A key of :attr:`units`.

        Returns
        -------
        tuple
            The budget, its predictive pass, and whether it reaches :attr:`unit_target`.  A
            unit that no budget up to :attr:`ceiling` sizes returns the ceiling and ``False``.
        """
        for replicates in self.grid(unit):
            value = self.unit_pass(unit, replicates)
            if value >= self.unit_target:
                return replicates, value, True
        return self.ceiling, self.unit_pass(unit, self.ceiling), False

    def family_pass(self, budgets: Mapping[str, int]) -> float:
        """The family-wise predictive pass probability, read jointly over the planning draws.

        Parameters
        ----------
        budgets : Mapping
            Unit name to budget.

        Returns
        -------
        float
            The share of planning draws at which every unit passes at its budget.
        """
        passed = np.ones(self.draws, dtype=bool)
        for unit in self.units:
            passed &= self.unit_passes(unit, budgets[unit])
        return float(passed.mean())


# --------------------------------------------------------------- the contraction ladder
def slope_half_width(
    standardized: Sequence[float], replicates: Sequence[int], sizes: Sequence[int], level: float
) -> float:
    """The delta-method half-width of the log |bias| on log n slope interval.

    A rung's log absolute bias has variance about ``1 / (z^2 R)``, with ``z`` its standardized
    bias and ``R`` its replications, and the rungs are independent.  The slope is the least
    squares slope on ``log n``, so its variance is ``sum_k w_k^2 / (z_k^2 R_k)`` with
    ``w_k = (x_k - mean x) / sum (x - mean x)^2``.  On three log-equally-spaced rungs the middle
    rung's weight is zero.

    Parameters
    ----------
    standardized : sequence of float
        Each rung's planning standardized bias.
    replicates : sequence of int
        Each rung's replications.
    sizes : sequence of int
        Each rung's n.
    level : float
        The two-sided confidence level of the interval.

    Returns
    -------
    float
        The projected half-width.
    """
    x = np.log(np.asarray(sizes, dtype=float))
    centred = x - x.mean()
    weights = centred / np.sum(centred**2)
    z = np.asarray(standardized, dtype=float)
    r = np.asarray(replicates, dtype=float)
    variance = float(np.sum(weights**2 / (z**2 * r)))
    return float(norm.ppf(0.5 + level / 2)) * math.sqrt(variance)


def outer_rung_budget(
    standardized: Sequence[float],
    sizes: Sequence[int],
    middle: int,
    *,
    target: float,
    level: float,
    step: int = 100,
    ceiling: int = 20_000,
) -> int | None:
    """The smallest outer-rung budget whose projected slope half-width reaches ``target``.

    Both outer rungs take the same budget, and the middle rung keeps ``middle``.

    Parameters
    ----------
    standardized : sequence of float
        Each rung's planning standardized bias.
    sizes : sequence of int
        The three rungs' n.
    middle : int
        The middle rung's replications.
    target : float
        The declared resolution target, a half-width.
    level : float
        The two-sided confidence level of the interval.
    step : int
        The budget grid.
    ceiling : int
        The largest budget tried.

    Returns
    -------
    int or None
        The budget, or ``None`` when no budget up to ``ceiling`` reaches the target.
    """
    for budget in range(step, ceiling + 1, step):
        width = slope_half_width(standardized, (budget, middle, budget), sizes, level)
        if width <= target:
            return budget
    return None


def as_rows(pilot: Pilot) -> list[dict[str, Any]]:
    """A pilot's per-row planning values, as rows a CSV can carry.

    Parameters
    ----------
    pilot : Pilot
        The summary.

    Returns
    -------
    list of dict
        One row per estimand.
    """
    out: list[dict[str, Any]] = []
    for index, row in enumerate(pilot.rows):
        out.append(
            {
                "source": pilot.source,
                "replicates": pilot.replicates,
                "estimand": row,
                "standardized_bias": pilot.bias[index],
                "se_ratio": pilot.se_ratio[index] if pilot.se_ratio else math.nan,
                "coverage": pilot.coverage[index] if pilot.coverage else math.nan,
            }
        )
    return out
