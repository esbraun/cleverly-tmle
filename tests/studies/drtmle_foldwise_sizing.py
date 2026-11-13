r"""The arm-pooled joint predictive sizing rule of the Fix-6 fold-wise DR-TMLE records.

The three gated records of Fix-6 are sized here: ``drtmle-foldwise-n1500`` (its ``both_correct``
primary rows and its two paired property cells, jointly), ``multi-arm-drtmle-foldwise`` (its
primary rows and its ``outcome_correct`` cell, jointly) and ``stratified-drtmle-foldwise`` (its
primary rows less the declared finite-sample row ``ey[1][V=2]``, and its contraction ladder by
:func:`predictive_outer_budget` and :func:`control_budget`).  The rule extends the one
:mod:`tests.studies.drtmle_foldwise_properties` declared for X83 so that a sizing pilot only has
to supply rows.

**The rows each budget is judged by.**  A primary row reads the shared ``Margins()`` of
``performance-tests.csv``: the 99% Student interval of the standardized bias inside 0.25
empirical SD, the 99% exact coverage interval above the 0.90 floor, and the 99% interval of the
SE ratio inside the 0.8 to 1.2 sanity band.  A ``double_robustness`` property cell reads the
bias clause alone, because its union-model SE band (0.1 to 10) never binds a fit whose SE ratio
lies near one.  A control row must put its bias interval wholly outside the margin.

**One fit's rows are drawn jointly.**  Each :class:`Pilot` summarizes the rows one fit
estimates on one draw.  For each replication and row it forms the influence of that
replication on the row's standardized bias, its coverage and its SE ratio,

.. math::

    \phi_b = d - b\,(d^2 - 1)/2, \quad \phi_c = c - \bar c, \quad
    \phi_\rho = (s - \bar s)/\mathrm{sd} - \rho\,(d^2 - 1)/2,

with ``d`` the centred error in SD units, ``c`` the coverage indicator and ``s`` the reported
SE, and keeps their covariance :attr:`Pilot.influence` across every row and statistic.  An
estimate over ``R`` replications then has covariance ``influence / R``: the planning values are
drawn from it at the pilot's ``R``, and the study's own realization at the candidate budget, so
the bias, coverage and SE-ratio rows of every estimand of one fit move together as they do in a
run.

**Shared configurations and regimes.**  Two pilots of one configuration share their planning
draw (:attr:`Rule.shares`): the second reads the first's planning values row by row and draws
only its own study noise.  Rows that the same mechanism drives may be declared as the arms of a
regime (:attr:`Rule.regimes`): their mean and between-arm SD ``tau`` are estimated together,
``tau`` at its one-sided 80% Q-profile upper bound, and each arm's planning value is its
normal-normal posterior; every other statistic of an arm's pilot moves with the arm by its
conditional regression on the influence covariance.

**Budgets.**  A unit passes at a planning draw when every clause of every row passes.  A unit's
budget is the smallest multiple of :attr:`Rule.step` at or above its floor whose predictive pass
reaches the unit share ``target ** (1 / len(units))``.  A unit that no budget up to
:attr:`Rule.ceiling` sizes is reported unsized, and its record's guard refuses the declared run.

**Rows set aside.**  A coverage or SE-ratio row of a primary whose predictive pass at the
ceiling is below :data:`SET_ASIDE_THRESHOLD`, and which no acceptance text reads
(:attr:`Rule.protected`), may be set aside (:func:`set_aside_rows`): it is declared before the
run as an expected finite-sample row, read under the finite-sample rule, and the budget is sized
over the other rows.  A bias row is never set aside, and no margin moves.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from scipy.stats import t as student

from tests.studies.evidence.registry import ROOT, Margins
from tests.studies.evidence.schema import truth_on_inference_scale

#: The clauses the rule reads.  ``se_ratio`` and ``coverage`` bind only the rows whose pilot
#: gives them.
CLAUSES = ("bias", "se_ratio", "coverage")
#: The clauses a row may be set aside on; a bias row never is.
SET_ASIDE_CLAUSES = ("se_ratio", "coverage")
#: A calibration row is set aside only when its predictive pass at the ceiling is below this.
SET_ASIDE_THRESHOLD = 0.80


@dataclass(frozen=True)
class Pilot:
    """The planning summary of the rows one fit estimates on one draw.

    Parameters
    ----------
    source : str
        Where the summary comes from: a committed artifact or a sizing pilot's rows.
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
    influence : tuple of tuple of float
        The covariance of one replication's influence on the statistics, ordered as every
        row's bias, then (when the pilot has them) every row's coverage, then every row's SE
        ratio.  Empty for a summary without rows, which then falls back on ``correlation`` for
        the bias and on independent rows for the calibration.
    """

    source: str
    replicates: int
    rows: tuple[str, ...]
    bias: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    controls: tuple[str, ...] = ()
    se_ratio: tuple[float, ...] = ()
    coverage: tuple[float, ...] = ()
    influence: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        size = len(self.rows)
        if len(self.bias) != size or np.shape(self.correlation) != (size, size):
            raise ValueError(f"{self.source}: bias and correlation must match the {size} rows")
        if len(self.se_ratio) != len(self.coverage) or (
            self.se_ratio and len(self.se_ratio) != size
        ):
            raise ValueError(f"{self.source}: give an SE ratio and a coverage for every row")
        if self.influence and np.shape(self.influence) != (self.width, self.width):
            raise ValueError(f"{self.source}: the influence covariance must be {self.width} wide")
        unknown = set(self.controls) - set(self.rows)
        if unknown:
            raise ValueError(f"{self.source}: controls name unknown rows {sorted(unknown)}")

    @property
    def calibrated(self) -> bool:
        """Whether the pilot carries coverage and SE-ratio rows."""
        return bool(self.se_ratio)

    @property
    def width(self) -> int:
        """The length of the statistic vector: ``k`` rows, or ``3 k`` with calibration."""
        return len(self.rows) * (3 if self.calibrated else 1)

    def index(self, statistic: str, row: str) -> int:
        """The position of one row's statistic in the statistic vector.

        Parameters
        ----------
        statistic : str
            ``bias``, ``coverage`` or ``se_ratio``.
        row : str
            A row of the pilot.

        Returns
        -------
        int
            The position.
        """
        block = {"bias": 0, "coverage": 1, "se_ratio": 2}[statistic]
        if block and not self.calibrated:
            raise KeyError(f"{self.source} has no {statistic} rows")
        return block * len(self.rows) + self.rows.index(row)

    @property
    def point(self) -> np.ndarray:
        """The statistic vector at the pilot's values."""
        values = list(self.bias)
        if self.calibrated:
            values += list(self.coverage) + list(self.se_ratio)
        return np.asarray(values, dtype=float)

    @property
    def covariance(self) -> np.ndarray:
        """One replication's covariance of the statistics (``influence``, or its fallback)."""
        if self.influence:
            return np.asarray(self.influence, dtype=float)
        k = len(self.rows)
        out = np.zeros((self.width, self.width))
        out[:k, :k] = np.asarray(self.correlation, dtype=float)
        if self.calibrated:
            covered = np.asarray(self.coverage)
            ratio = np.asarray(self.se_ratio)
            out[k : 2 * k, k : 2 * k] = np.diag(covered * (1 - covered))
            out[2 * k :, 2 * k :] = np.diag(ratio**2 / 2)
        return out


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


def without(pilot: Pilot, rows: Sequence[str]) -> Pilot:
    """A pilot with ``rows`` dropped, its covariances restricted to the rows it keeps.

    A record sizes its budget over the rows the budget can make pass.  A row declared before
    the run as an expected finite-sample row is read under its own rule, so its record drops it
    from the sizing family with this.

    Parameters
    ----------
    pilot : Pilot
        The planning summary.
    rows : sequence of str
        The rows to drop; each must be a row of ``pilot``.

    Returns
    -------
    Pilot
        The summary of the remaining rows.
    """
    unknown = set(rows) - set(pilot.rows)
    if unknown:
        raise ValueError(f"{pilot.source}: no rows {sorted(unknown)} to drop")
    keep = [index for index, row in enumerate(pilot.rows) if row not in set(rows)]
    k = len(pilot.rows)
    blocks = 3 if pilot.calibrated else 1
    wide = [block * k + index for block in range(blocks) for index in keep]

    def pick(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(values[index] for index in keep) if values else ()

    return Pilot(
        pilot.source,
        pilot.replicates,
        tuple(pilot.rows[index] for index in keep),
        pick(pilot.bias),
        tuple(tuple(pilot.correlation[i][j] for j in keep) for i in keep),
        controls=tuple(row for row in pilot.controls if row not in set(rows)),
        se_ratio=pick(pilot.se_ratio),
        coverage=pick(pilot.coverage),
        influence=(
            tuple(tuple(pilot.influence[i][j] for j in wide) for i in wide)
            if pilot.influence
            else ()
        ),
    )


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
    identity scale.  Only replications that report every estimand enter.

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
        The summary, with the influence covariance of its statistics.
    """
    names = tuple(sorted(rows["estimand"].unique())) if estimands is None else tuple(estimands)
    frame = rows.loc[rows["estimand"].isin(names)].copy()
    if "inference_estimate" in frame and frame["inference_estimate"].notna().all():
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

    def wide(column: str) -> pd.DataFrame:
        return frame.pivot(index="replicate", columns="estimand", values=column).loc[:, list(names)]

    complete = wide("error").dropna().index
    if len(complete) < 3:
        raise ValueError(f"{source}: fewer than three complete replications")
    error = wide("error").loc[complete].to_numpy(dtype=float)
    value = wide("value").loc[complete].to_numpy(dtype=float)
    spread = value.std(axis=0, ddof=1)
    centred = (error - error.mean(axis=0)) / spread
    bias = error.mean(axis=0) / spread
    square = (centred**2 - 1) / 2
    blocks = [centred - bias * square]
    se_ratio: tuple[float, ...] = ()
    coverage: tuple[float, ...] = ()
    if calibration:
        covered = wide("covered").loc[complete].to_numpy(dtype=float)
        reported = wide("std_error").loc[complete].to_numpy(dtype=float)
        ratio = reported.mean(axis=0) / spread
        blocks.append(covered - covered.mean(axis=0))
        blocks.append((reported - reported.mean(axis=0)) / spread - ratio * square)
        se_ratio = tuple(round(float(v), 4) for v in ratio)
        coverage = tuple(round(float(v), 4) for v in covered.mean(axis=0))
    influence = np.cov(np.hstack(blocks), rowvar=False).reshape(
        len(names) * len(blocks), len(names) * len(blocks)
    )
    correlation = np.corrcoef(error, rowvar=False).reshape(len(names), len(names))
    return Pilot(
        source,
        len(complete),
        names,
        tuple(round(float(v), 4) for v in bias),
        tuple(tuple(round(float(v), 4) for v in row) for row in correlation),
        se_ratio=se_ratio,
        coverage=coverage,
        influence=tuple(tuple(round(float(v), 6) for v in row) for row in influence),
    )


def _factor(covariance: np.ndarray) -> np.ndarray:
    """A square root of a covariance matrix that may be singular (``ate = ey1 - ey0``)."""
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    return np.asarray(vectors * np.sqrt(np.clip(values, 0.0, None)))


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
    shares : Mapping
        Pilot name to the pilot of the same configuration whose planning draw it reads.
    protected : frozenset
        ``(pilot name, row)`` pairs an acceptance text reads; never set aside.
    set_aside : frozenset
        ``(pilot name, row, clause)`` triples declared as expected finite-sample rows, which
        the budget is not sized against (:func:`set_aside_rows`).
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
    shares: Mapping[str, str] = field(default_factory=dict)
    protected: frozenset[tuple[str, str]] = frozenset()
    set_aside: frozenset[tuple[str, str, str]] = frozenset()
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
        for name, source in self.shares.items():
            if name not in self.pilots or source not in self.pilots or source in self.shares:
                raise ValueError(f"share {name!r} -> {source!r} names no unshared pilot pair")
            if not set(self.pilots[name].rows) <= set(self.pilots[source].rows):
                raise ValueError(f"{name!r} has rows {source!r} does not plan")
        for name, row, clause in self.set_aside:
            if clause not in SET_ASIDE_CLAUSES:
                raise ValueError(f"a {clause} row is never set aside")
            if (name, row) in self.protected:
                raise ValueError(f"{(name, row)!r} is read by an acceptance text")
            if not self.pilots[name].calibrated or row not in self.pilots[name].rows:
                raise ValueError(f"{(name, row)!r} has no calibration row to set aside")

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
    def planning_draws(self) -> Mapping[str, np.ndarray]:
        """Each pilot's statistic vector at every planning draw, from one seed.

        Each unshared pilot's statistics are drawn jointly normal around its summary with
        covariance ``covariance / replicates``.  Each regime's arms are then replaced by their
        normal-normal posterior at the bounded ``tau``, with the pooled mean drawn once per
        planning draw, so the arms share its error; the other statistics of an arm's pilot
        follow by their conditional regression on the arms.  A shared pilot then reads its
        source's values row by row.

        Returns
        -------
        Mapping
            Pilot name to a ``(draws, width)`` array.
        """
        rng = np.random.default_rng(self.seed)
        values: dict[str, np.ndarray] = {}
        for name, pilot in self.pilots.items():
            shocks = rng.standard_normal((self.draws, pilot.width))
            values[name] = pilot.point + shocks @ _factor(pilot.covariance).T / math.sqrt(
                pilot.replicates
            )
        for regime in self.regimes:
            tau, mean, spread = self.regime_tau(regime)
            mu = mean + spread * rng.standard_normal(self.draws)
            moved: dict[str, dict[int, np.ndarray]] = {}
            for name, row in regime:
                pilot = self.pilots[name]
                index = pilot.index("bias", row)
                v = 1.0 / pilot.replicates
                shrink = tau**2 / (v + tau**2)
                arm = (
                    mu
                    + shrink * (pilot.bias[pilot.rows.index(row)] - mu)
                    + math.sqrt(v * shrink) * rng.standard_normal(self.draws)
                )
                moved.setdefault(name, {})[index] = arm - values[name][:, index]
            for name, shifts in moved.items():
                covariance = self.pilots[name].covariance
                arms = sorted(shifts)
                others = [i for i in range(covariance.shape[0]) if i not in shifts]
                delta = np.column_stack([shifts[i] for i in arms])
                if others:
                    gain = covariance[np.ix_(others, arms)] @ np.linalg.pinv(
                        covariance[np.ix_(arms, arms)]
                    )
                    values[name][:, others] += delta @ gain.T
                values[name][:, arms] += delta
        for name, source in self.shares.items():
            pilot, origin = self.pilots[name], self.pilots[source]
            for statistic in ("bias", "coverage", "se_ratio"):
                if statistic != "bias" and not (pilot.calibrated and origin.calibrated):
                    continue
                for row in pilot.rows:
                    values[name][:, pilot.index(statistic, row)] = values[source][
                        :, origin.index(statistic, row)
                    ]
        return values

    @cached_property
    def _noise(self) -> Mapping[str, np.ndarray]:
        """Standard normal draws of each pilot's study realization, one block per pilot."""
        rng = np.random.default_rng([self.seed, 1])
        return {
            name: rng.standard_normal((self.draws, pilot.width))
            for name, pilot in self.pilots.items()
        }

    def row_clauses(
        self, name: str, replicates: int, clauses: tuple[str, ...] = CLAUSES
    ) -> dict[tuple[str, str], np.ndarray]:
        """Whether each row's clause passes at ``replicates``, at each planning draw.

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
        dict
            ``(row, clause)`` to one boolean per planning draw.
        """
        pilot = self.pilots[name]
        covariance = pilot.covariance
        realized = self.planning_draws[name] + self._noise[name] @ _factor(
            covariance
        ).T / math.sqrt(replicates)
        level = self.margins.confidence_level
        half = float(student.ppf(0.5 + level / 2, replicates - 1)) / math.sqrt(replicates)
        margin = self.margins.standardized_bias
        out: dict[tuple[str, str], np.ndarray] = {}
        for row in pilot.rows:
            if "bias" in clauses:
                size = np.abs(realized[:, pilot.index("bias", row)])
                out[(row, "bias")] = (
                    size - half > margin if row in pilot.controls else size + half <= margin
                )
            if not pilot.calibrated:
                continue
            if "se_ratio" in clauses:
                index = pilot.index("se_ratio", row)
                read = realized[:, index]
                width = self.z * math.sqrt(max(covariance[index, index], 0.0) / replicates)
                low, high = self.margins.se_ratio_sanity
                out[(row, "se_ratio")] = (read - width >= low) & (read + width <= high)
            if "coverage" in clauses:
                read = np.clip(realized[:, pilot.index("coverage", row)], 0.0, 1.0)
                floor = read - self.z * np.sqrt(read * (1 - read) / replicates)
                out[(row, "coverage")] = floor >= self.margins.coverage_floor
        return out

    def pilot_passes(
        self, name: str, replicates: int, clauses: tuple[str, ...] = CLAUSES
    ) -> np.ndarray:
        """Whether one pilot's rows all pass at ``replicates``, at each planning draw.

        The clauses in :attr:`set_aside` are not read.

        Parameters
        ----------
        name : str
            A key of :attr:`pilots`.
        replicates : int
            The budget.
        clauses : tuple of str
            The clauses read.

        Returns
        -------
        numpy.ndarray
            One boolean per planning draw.
        """
        passed = np.ones(self.draws, dtype=bool)
        for (row, clause), value in self.row_clauses(name, replicates, clauses).items():
            if (name, row, clause) not in self.set_aside:
                passed &= value
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

    def ceiling_passes(self) -> list[tuple[str, str, str, float]]:
        """Every row clause's own predictive pass at the ceiling, lowest first.

        Returns
        -------
        list of tuple
            ``(pilot, row, clause, pass)``.
        """
        out = []
        for name in self.pilots:
            for (row, clause), value in self.row_clauses(name, self.ceiling).items():
                out.append((name, row, clause, float(value.mean())))
        return sorted(out, key=lambda item: item[3])


def set_aside_rows(
    rule: Rule, threshold: float = SET_ASIDE_THRESHOLD
) -> tuple[Rule, tuple[tuple[str, str, str, float], ...]]:
    """The rule with every eligible row below ``threshold`` at the ceiling set aside.

    A row is eligible when its clause is a coverage or SE-ratio clause and no acceptance text
    reads it (:attr:`Rule.protected`).  Each set-aside row is returned with its pass at the
    ceiling, which the declaration names.

    Parameters
    ----------
    rule : Rule
        The rule.
    threshold : float
        The predictive pass at the ceiling below which a row is set aside.

    Returns
    -------
    tuple
        The rule with :attr:`Rule.set_aside` filled, and the rows set aside.
    """
    chosen = tuple(
        (name, row, clause, value)
        for name, row, clause, value in rule.ceiling_passes()
        if clause in SET_ASIDE_CLAUSES and (name, row) not in rule.protected and value < threshold
    )
    aside = frozenset((name, row, clause) for name, row, clause, _ in chosen)
    return dataclasses.replace(rule, set_aside=rule.set_aside | aside), chosen


# --------------------------------------------------------------- the contraction ladder
def _slope_weights(sizes: Sequence[int]) -> np.ndarray:
    x = np.log(np.asarray(sizes, dtype=float))
    centred = x - x.mean()
    return np.asarray(centred / np.sum(centred**2))


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
        Each rung's standardized bias.
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
    weights = _slope_weights(sizes)
    z = np.asarray(standardized, dtype=float)
    r = np.asarray(replicates, dtype=float)
    variance = float(np.sum(weights**2 / (z**2 * r)))
    return float(norm.ppf(0.5 + level / 2)) * math.sqrt(variance)


@dataclass(frozen=True)
class LadderPlan:
    """The planning values of a ladder's read labels at one rung, with their uncertainty.

    Parameters
    ----------
    source : str
        Where the values come from.
    labels : tuple of str
        The read labels.
    size : int
        The rung the values were measured at.
    standardized : tuple of float
        Each label's standardized bias at that rung.
    correlation : tuple of tuple of float
        The labels' per-replication error correlation at that rung.
    replicates : int
        The replications behind the values.
    """

    source: str
    labels: tuple[str, ...]
    size: int
    standardized: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    replicates: int

    def __post_init__(self) -> None:
        k = len(self.labels)
        if len(self.standardized) != k or np.shape(self.correlation) != (k, k):
            raise ValueError(f"{self.source}: values and correlation must match the labels")


def ladder_plan_from_rows(
    rows: pd.DataFrame, *, source: str, labels: Mapping[str, str], size: int
) -> LadderPlan:
    """A ladder plan from one rung's per-replication rows.

    Parameters
    ----------
    rows : pandas.DataFrame
        The rung's rows: ``replicate``, ``estimand``, ``truth`` and ``estimate``.
    source : str
        Where the rows come from.
    labels : Mapping
        Read label to its estimand name in ``rows``.
    size : int
        The rung.

    Returns
    -------
    LadderPlan
        The plan.
    """
    pilot = pilot_from_rows(
        rows.drop(columns=["inference_estimate"], errors="ignore"),
        source=source,
        estimands=tuple(labels.values()),
        calibration=False,
    )
    return LadderPlan(source, tuple(labels), size, pilot.bias, pilot.correlation, pilot.replicates)


def second_order_pool(top: LadderPlan, low: LadderPlan) -> LadderPlan:
    """The top rung's planning values pooled with the lower rung's, carried up as second order.

    Under a second-order bias the standardized bias scales as ``sqrt(low.size / top.size)``
    between the rungs, so the lower rung's value times that factor is a second estimate of the
    top rung's, with variance ``(low.size / top.size) / low.replicates``.  The pool is their
    inverse-variance mean, with the top rung's correlation and the pooled precision as its
    replicates.  This is the hypothesis the ladder's acceptance tests, not a margin.

    Parameters
    ----------
    top : LadderPlan
        The values at the top rung.
    low : LadderPlan
        The values at a lower rung, for the same labels.

    Returns
    -------
    LadderPlan
        The pooled plan at the top rung.
    """
    if top.labels != low.labels:
        raise ValueError("the two plans must read the same labels")
    ratio = low.size / top.size
    carried = [value * math.sqrt(ratio) for value in low.standardized]
    weight_top, weight_low = top.replicates, low.replicates / ratio
    pooled = tuple(
        round((a * weight_top + b * weight_low) / (weight_top + weight_low), 4)
        for a, b in zip(top.standardized, carried, strict=True)
    )
    return LadderPlan(
        f"{top.source}, pooled with {low.source} carried up as second order",
        top.labels,
        top.size,
        pooled,
        top.correlation,
        round(weight_top + weight_low),
    )


def write_planning(
    path: Path, pilots: Mapping[str, Pilot | LadderPlan], seconds: Mapping[str, float]
) -> None:
    """Write planning summaries and seconds per fit as JSON, the form a record commits.

    Parameters
    ----------
    path : Path
        The file.
    pilots : Mapping
        Name to :class:`Pilot` or :class:`LadderPlan`.
    seconds : Mapping
        Group to its mean seconds per fit.
    """
    body = {
        "pilots": {
            name: {"kind": type(value).__name__, **dataclasses.asdict(value)}
            for name, value in pilots.items()
        },
        "seconds_per_fit": {name: round(float(value), 3) for name, value in seconds.items()},
    }
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


#: The committed planning summaries of the Fix-6 records: the sizing pilot's merged groups
#: (1,000 replications per primary and cell group, 600 per ladder rung), written by
#: ``python -m tests.diagnostics.fix6_foldwise_sizing.run --planning``.
PLANNING_PATH = ROOT / "tests" / "studies" / "drtmle_foldwise_pilot.json"


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def read_planning(path: Path) -> tuple[dict[str, Pilot | LadderPlan], dict[str, float]]:
    """Read what :func:`write_planning` wrote.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    tuple
        Name to :class:`Pilot` or :class:`LadderPlan`, and group to seconds per fit.
    """
    body = json.loads(path.read_text(encoding="utf-8"))
    kinds = {"Pilot": Pilot, "LadderPlan": LadderPlan}
    pilots: dict[str, Pilot | LadderPlan] = {}
    for name, fields in body["pilots"].items():
        kind = kinds[fields.pop("kind")]
        pilots[name] = kind(**{key: _tuples(value) for key, value in fields.items()})
    return pilots, {name: float(value) for name, value in body["seconds_per_fit"].items()}


@cache
def committed_planning() -> tuple[dict[str, Pilot | LadderPlan], dict[str, float]]:
    """The committed planning summaries (:data:`PLANNING_PATH`).

    Returns
    -------
    tuple
        Pilot group to its summary, and group to its mean seconds per fit.
    """
    return read_planning(PLANNING_PATH)


def planned(planning: Mapping[str, Pilot | LadderPlan] | None, group: str, kind: type[Any]) -> Any:
    """One pilot group's planning summary, checked to be of ``kind``.

    Parameters
    ----------
    planning : Mapping or None
        Pilot group to its summary; :func:`committed_planning` when ``None``.
    group : str
        The group.
    kind : type
        :class:`Pilot` or :class:`LadderPlan`.

    Returns
    -------
    Pilot or LadderPlan
        The summary.

    Raises
    ------
    TypeError
        When the group's summary is not of ``kind``.
    """
    plans = committed_planning()[0] if planning is None else planning
    value = plans[group]
    if not isinstance(value, kind):
        raise TypeError(
            f"planning group {group!r} is a {type(value).__name__}, not {kind.__name__}"
        )
    return value


def ladder_resolution_pass(
    plan: LadderPlan,
    sizes: Sequence[int],
    budgets: Sequence[int],
    *,
    target: float,
    level: float,
    draws: int = 20_000,
    seed: int = 0,
) -> float:
    """The predictive probability that every read label's slope interval reaches ``target``.

    The bias is second order: each label's standardized bias at rung ``n`` is its planning value
    at the plan's rung times ``sqrt(plan.size / n)``.  The planning values are drawn jointly
    with covariance ``correlation / replicates``; at each rung the study's standardized bias is
    the planned one plus its own error, jointly across labels with the same correlation and
    variance ``1 / R``; the half-width is the delta-method one at the realized values.

    Parameters
    ----------
    plan : LadderPlan
        The planning values.
    sizes : sequence of int
        The rungs.
    budgets : sequence of int
        Each rung's replications.
    target : float
        The declared half-width.
    level : float
        The interval's two-sided confidence level.
    draws : int
        The joint planning draws.
    seed : int
        Their seed.

    Returns
    -------
    float
        The share of draws at which every label's half-width is at most ``target``.
    """
    rng = np.random.default_rng(seed)
    factor = _factor(np.asarray(plan.correlation, dtype=float))
    k = len(plan.labels)
    planned = np.asarray(plan.standardized) + rng.standard_normal((draws, k)) @ factor.T / (
        math.sqrt(plan.replicates)
    )
    weights = _slope_weights(sizes)
    critical = float(norm.ppf(0.5 + level / 2))
    variance = np.zeros((draws, k))
    for size, budget, weight in zip(sizes, budgets, weights, strict=True):
        scale = math.sqrt(plan.size / size)
        realized = planned * scale + rng.standard_normal((draws, k)) @ factor.T / math.sqrt(budget)
        variance += weight**2 / (realized**2 * budget)
    return float((critical * np.sqrt(variance) <= target).all(axis=1).mean())


def predictive_outer_budget(
    plan: LadderPlan,
    sizes: Sequence[int],
    middle: int,
    *,
    target: float,
    level: float,
    probability: float = 0.95,
    floor: int = 800,
    step: int = 400,
    ceiling: int = 8_000,
    draws: int = 20_000,
    seed: int = 0,
) -> tuple[int, float, bool]:
    """The smallest outer-rung budget whose predictive resolution reaches ``probability``.

    Both outer rungs take the budget and the middle rung keeps ``middle``.

    Parameters
    ----------
    plan : LadderPlan
        The planning values.
    sizes : sequence of int
        The three rungs.
    middle : int
        The middle rung's replications.
    target : float
        The declared half-width.
    level : float
        The interval's two-sided confidence level.
    probability : float
        The predictive probability every read label must reach the target with.
    floor : int
        The smallest budget tried.
    step : int
        The budget grid.
    ceiling : int
        The largest budget tried.
    draws : int
        The joint planning draws.
    seed : int
        Their seed.

    Returns
    -------
    tuple
        ``(R, predictive pass, sized)``; the ceiling and ``False`` when no budget reaches it.
    """
    value = 0.0
    for budget in range(step * math.ceil(floor / step), ceiling + 1, step):
        value = ladder_resolution_pass(
            plan,
            sizes,
            (budget, middle, budget),
            target=target,
            level=level,
            draws=draws,
            seed=seed,
        )
        if value >= probability:
            return budget, value, True
    return ceiling, value, False


def control_fails_pass(
    plan: LadderPlan,
    sizes: Sequence[int],
    budget: int,
    *,
    first_order: float,
    level: float,
    draws: int = 20_000,
    seed: int = 0,
) -> float:
    """The predictive probability that every read label's control slope fails to contract.

    The control's bias does not decay: its standardized bias grows as ``sqrt(n / plan.size)``.
    A label fails to contract when its slope interval's upper end is at or above
    ``first_order``; the slope and its half-width are the delta-method ones at the realized
    values.

    Parameters
    ----------
    plan : LadderPlan
        The control's planning values.
    sizes : sequence of int
        The rungs.
    budget : int
        Every rung's replications.
    first_order : float
        The slope the upper end must reach, -1/2.
    level : float
        The interval's two-sided confidence level.
    draws : int
        The joint planning draws.
    seed : int
        Their seed.

    Returns
    -------
    float
        The share of draws at which every label fails to contract.
    """
    rng = np.random.default_rng(seed)
    factor = _factor(np.asarray(plan.correlation, dtype=float))
    k = len(plan.labels)
    planned = np.asarray(plan.standardized) + rng.standard_normal((draws, k)) @ factor.T / (
        math.sqrt(plan.replicates)
    )
    weights = _slope_weights(sizes)
    critical = float(norm.ppf(0.5 + level / 2))
    slope = np.zeros((draws, k))
    variance = np.zeros((draws, k))
    for size, weight in zip(sizes, weights, strict=True):
        truth = planned * math.sqrt(size / plan.size)
        realized = truth + rng.standard_normal((draws, k)) @ factor.T / math.sqrt(budget)
        # log |bias| in SD units at the rung; the SD's own sqrt(n) scaling enters the slope.
        slope += weight * (np.log(np.abs(realized)) - 0.5 * math.log(size))
        variance += weight**2 / (realized**2 * budget)
    upper = slope + critical * np.sqrt(variance)
    return float((upper >= first_order).all(axis=1).mean())


def control_budget(
    plan: LadderPlan,
    sizes: Sequence[int],
    *,
    first_order: float,
    level: float,
    probability: float = 0.99,
    floor: int = 400,
    step: int = 400,
    ceiling: int = 8_000,
    draws: int = 20_000,
    seed: int = 0,
) -> tuple[int, float, bool]:
    """The smallest per-rung control budget whose failure to contract reaches ``probability``.

    Parameters
    ----------
    plan : LadderPlan
        The control's planning values.
    sizes : sequence of int
        The rungs.
    first_order : float
        The slope the upper end must reach, -1/2.
    level : float
        The interval's two-sided confidence level.
    probability : float
        The predictive probability the control must fail to contract with.
    floor : int
        The smallest budget tried.
    step : int
        The budget grid.
    ceiling : int
        The largest budget tried.
    draws : int
        The joint planning draws.
    seed : int
        Their seed.

    Returns
    -------
    tuple
        ``(R, predictive pass, sized)``.
    """
    value = 0.0
    for budget in range(step * math.ceil(floor / step), ceiling + 1, step):
        value = control_fails_pass(
            plan, sizes, budget, first_order=first_order, level=level, draws=draws, seed=seed
        )
        if value >= probability:
            return budget, value, True
    return ceiling, value, False


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
