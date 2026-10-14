"""Recorded property failures: the records, the stamps, and the check that they agree.

RM39 part b2.  A property loop runs through
:func:`tests.studies.evidence.replication.fit_properties`, so a recordable error in one fit
costs the cells that fit writes one replicate, not the run.  The regeneration driver collects
the records (:func:`tests.studies.evidence.replication.recording`) and publishes them as
``property-failures.csv``, with the columns of the primary phase's ``failures.csv``.  A
record's ``scenarios`` holds the ``property/cell`` names its key writes, ``;``-joined.

The driver is the one writer of ``failed_replicates``: :func:`stamp` sets each cell's count
from the records, and :func:`reconcile` refuses rows whose gaps the records do not explain.
With no record, :func:`stamp` returns the rows unchanged, so no artifact moves.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies.evidence.failures import (
    BOUND_EXCEEDED,
    COMPLETE_CASE_FAILS,
    FAILURE_COLUMNS,
    NOT_ACCOUNTABLE,
    WITHIN_BOUND,
    WORST_CASE_FAILS,
    Leg,
    filled,
    vertex,
)
from tests.studies.evidence.inference import Interval, clopper_pearson, student_interval
from tests.studies.evidence.properties import (
    mean_difference_interval,
    paired_spread_ratio_interval,
    rate,
    ratio_intervals,
    se_ratio_deficit_interval,
    spread_values,
)
from tests.studies.evidence.registry import Margins, StudyRecord
from tests.studies.evidence.replication import Recording
from tests.studies.evidence.seeds import stream_seed

#: The published record file and the published accounting file of the property phase.  Each
#: exists only when a property fit failed, so a study without one publishes exactly the
#: artifacts it did before.
PROPERTY_FAILURES = "property-failures.csv"
PROPERTY_ACCOUNTING = "property-failure-accounting.csv"

#: The columns :func:`expand` returns: one row per cell replicate a record costs.
LOST_COLUMNS = ("property", "cell", "replicate", "group_id")


def frame(recorded: Recording) -> pd.DataFrame | None:
    """The run's property records in published order, or ``None`` when no fit failed.

    Parameters
    ----------
    recorded : Recording
        What the driver's collector gathered.

    Returns
    -------
    pandas.DataFrame or None
        The records with :data:`~tests.studies.evidence.failures.FAILURE_COLUMNS`, sorted by
        group id.
    """
    failures = recorded.failures
    if failures is None:
        return None
    return failures.loc[:, list(FAILURE_COLUMNS)].sort_values("group_id", ignore_index=True)


def read(path: Path) -> pd.DataFrame | None:
    """The records in ``path``, or ``None`` when the file does not exist.

    Parameters
    ----------
    path : pathlib.Path
        A ``property-failures.csv``.

    Returns
    -------
    pandas.DataFrame or None
        The records, with ``scenarios`` and ``replicates`` read as text.
    """
    if not path.exists():
        return None
    published = pd.read_csv(
        path, dtype={"scenarios": str, "replicates": str, "seed": str}, keep_default_na=False
    )
    return published.loc[:, list(FAILURE_COLUMNS)]


def expand(records: pd.DataFrame | None) -> pd.DataFrame:
    """Every ``(property, cell, replicate)`` the records cost, with the group that cost it.

    Parameters
    ----------
    records : pandas.DataFrame or None
        Property failure records.

    Returns
    -------
    pandas.DataFrame
        One row per cell replicate, with :data:`LOST_COLUMNS`.

    Raises
    ------
    ValueError
        When a record names a malformed cell, or two records cost one cell replicate.
    """
    if records is None or records.empty:
        return pd.DataFrame(columns=list(LOST_COLUMNS))
    out: list[tuple[str, str, int, str]] = []
    for row in records.itertuples(index=False):
        replicates = [int(part) for part in str(row.replicates).split(";") if part]
        for name in (part for part in str(row.scenarios).split(";") if part):
            if name.count("/") != 1:
                raise ValueError(f"failure {row.group_id} names {name!r}, not a property/cell")
            family, cell = name.split("/")
            out.extend((family, cell, replicate, str(row.group_id)) for replicate in replicates)
    lost = pd.DataFrame(out, columns=list(LOST_COLUMNS))
    duplicated = lost.duplicated(["property", "cell", "replicate"])
    if duplicated.any():
        raise ValueError(
            "two property failure records cost one cell replicate:\n"
            + lost.loc[duplicated].to_string(index=False)
        )
    return lost


def counts(records: pd.DataFrame | None) -> dict[tuple[str, str], int]:
    """How many replicates the records cost each ``(property, cell)``.

    Parameters
    ----------
    records : pandas.DataFrame or None
        Property failure records.

    Returns
    -------
    dict
        The count of each cell a record names.
    """
    lost = expand(records)
    return {
        (str(family), str(cell)): len(group)
        for (family, cell), group in lost.groupby(["property", "cell"], sort=True)
    }


def stamp(rows: pd.DataFrame, records: pd.DataFrame | None) -> pd.DataFrame:
    """``rows`` with each cell's ``failed_replicates`` set to the replicates its records cost.

    The one writer of the column.  With no record, the rows come back unchanged, so the bytes
    of a run without a failure do not move.

    Parameters
    ----------
    rows : pandas.DataFrame
        The property replication rows.
    records : pandas.DataFrame or None
        Property failure records.

    Returns
    -------
    pandas.DataFrame
        The stamped rows.
    """
    if records is None or records.empty:
        return rows
    lost = counts(records)
    out = rows.copy()
    keys = zip(out["property"].astype(str), out["cell"].astype(str), strict=True)
    out["failed_replicates"] = [lost.get(key, 0) for key in keys]
    return out


def reconcile(
    rows: pd.DataFrame,
    records: pd.DataFrame | None,
    *,
    derived: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Refuse property rows whose missing replicates are not exactly the recorded failures.

    Per cell, the surviving and the recorded replicate ids are disjoint and together cover
    ``0 .. requested_replicates - 1``, and the stamp equals the recorded count.  A record that
    names a cell with no surviving row is refused, and so is a short cell with no record.
    ``derived`` maps each base cell to its derived controls, by ``property/cell`` name.  A
    control derived from a base must have lost exactly the base's replicates.

    Parameters
    ----------
    rows : pandas.DataFrame
        The stamped property replication rows.
    records : pandas.DataFrame or None
        Property failure records.
    derived : Mapping or None
        A study's ``DERIVED_CELLS``.

    Raises
    ------
    ValueError
        When any cell breaks a rule above.
    """
    lost = expand(records)
    by_cell: dict[tuple[str, str], set[int]] = {
        (str(family), str(cell)): set(group["replicate"].astype(int))
        for (family, cell), group in lost.groupby(["property", "cell"], sort=True)
    }
    wrong: list[str] = []
    present = set()
    for (family, cell), group in rows.groupby(["property", "cell"], sort=True):
        key = (str(family), str(cell))
        present.add(key)
        name = f"{family}/{cell}"
        survived = [int(value) for value in group["replicate"]]
        recorded = by_cell.get(key, set())
        requested = group["requested_replicates"].unique()
        stamped = group["failed_replicates"].unique()
        if len(requested) != 1 or len(stamped) != 1:
            wrong.append(f"{name}: inconsistent requested or failed replication counts")
            continue
        if len(survived) != len(set(survived)):
            wrong.append(f"{name}: a replicate appears twice")
        elif set(survived) & recorded:
            wrong.append(f"{name}: replicates both survived and failed")
        elif set(survived) | recorded != set(range(int(requested[0]))):
            wrong.append(
                f"{name}: {len(survived)} surviving and {len(recorded)} failed replicates do "
                f"not cover the requested {int(requested[0])}"
            )
        if int(stamped[0]) != len(recorded):
            wrong.append(f"{name}: stamped {int(stamped[0])} failures, recorded {len(recorded)}")
    for key in sorted(set(by_cell) - present):
        wrong.append(f"{key[0]}/{key[1]}: a record names a cell with no surviving row")
    for base, controls in (derived or {}).items():
        family, cell = base.split("/")
        for control in controls:
            control_key = tuple(control.split("/"))
            if by_cell.get(control_key, set()) != by_cell.get((family, cell), set()):
                wrong.append(f"{control}: lost other replicates than its base {base}")
    if wrong:
        raise ValueError(
            "property cells whose replicates are not the recorded failures:\n" + "\n".join(wrong)
        )


# ---- the accounting (RM39-plan 4.5, RM39b2-plan section 7) ------------------------------------

#: The columns of ``property-failure-accounting.csv``: one row per cell and clause, after one
#: ``failure bound`` row.  ``complete_case`` and ``worst_case`` hold the clause's slack, the
#: distance by which it clears its threshold, so a negative value fails.
PROPERTY_ACCOUNTING_COLUMNS = (
    "property",
    "cell",
    "role",
    "requested",
    "failed",
    "rate",
    "bound",
    "allowed",
    "leg",
    "complete_case",
    "worst_case",
    "imputation",
    "leg_passed",
    "gate_passed",
    "reason",
)

#: The largest ``FAILURE_BOUND`` a property module may declare (RM39-plan a2 Q3).
MAXIMUM_BOUND = 0.05

#: The leg every accounted cell writes first, and the leg of a cell no rule accounts for.
BOUND_LEG = "failure bound"
NOT_ACCOUNTABLE_LEG = "failures not accountable"

#: Rounding slack on ``bound * requested``, as :data:`tests.studies.evidence.failures` keeps.
_FLOOR_SLACK = 1e-9

#: ``(study slug, family)`` pairs whose verdict a study computes itself, beyond a shared
#: function (D3).  A family listed here is not accountable in that study until the study
#: declares a non-zero ``FAILURE_BOUND`` and the rule that matches its own code.  The
#: reproduction and differential tests of ``tests/unit/test_method_evidence.py`` decide each
#: pair: every pair of a shared family that is not listed must reproduce the committed
#: verdicts exactly.
#:
#: | entries | why the shared rule does not hold |
#: | --- | --- |
#: | ``clustered_inference`` and ``crossfit_overfitting`` of the three multi-pair studies | each calls ``_paired_cell_verdicts`` once per pair of its own ``PAIRS`` list, so a joint verdict spans one pair, which the summary does not record |
#: | ``corrected_mar_inference`` of the three MAR DRTMLE studies | each adds a coverage floor and an SE-ratio sanity conjunct to the positive cells after ``robustness_verdicts`` |
STUDY_LOCAL: Mapping[str, frozenset[str]] = {
    "clustered-unequal-cvtmle": frozenset({"clustered_inference"}),
    "clustered-cross-fitted-ltmle": frozenset({"clustered_inference"}),
    "stacked-mar-arm-indexed-cvtmle": frozenset({"crossfit_overfitting"}),
    "mar-drtmle": frozenset({"corrected_mar_inference"}),
    "multi-arm-mar-drtmle": frozenset({"corrected_mar_inference"}),
    "composite-missing-drtmle": frozenset({"corrected_mar_inference"}),
}


def allowed(bound: float, requested: int) -> int:
    """How many replicates of a cell of ``requested`` may fail before the cell is red.

    Parameters
    ----------
    bound : float
        The study's ``FAILURE_BOUND``.
    requested : int
        The cell's requested replicates.

    Returns
    -------
    int
        ``floor(bound * requested)``.
    """
    return math.floor(bound * requested + _FLOOR_SLACK)


@dataclass(frozen=True)
class Judged:
    """One clause on one candidate: its slack and whether it holds.

    Parameters
    ----------
    slack : float
        The distance by which the clause clears its threshold; negative when it fails.
    passed : bool
        Whether the clause holds.  It is not always ``slack >= 0``: a strict inequality fails
        at zero slack.
    """

    slack: float
    passed: bool


@dataclass
class Context:
    """What a clause reads: the study, its surviving rows, the complete-case summary.

    Parameters
    ----------
    record : StudyRecord
        Supplies the margins, the seed streams and ``truth_varies_by_replicate``.
    rows : pandas.DataFrame
        The stamped surviving replication rows.
    summary : pandas.DataFrame
        The complete-case summary the study's own verdicts computed.
    lost : dict
        The replicates the records cost each ``(property, cell)``.
    parameters : dict
        Study declarations a rule reads, such as ``efficiency_band``.
    """

    record: StudyRecord
    rows: pd.DataFrame
    summary: pd.DataFrame
    lost: dict[tuple[str, str], int] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def margins(self) -> Margins:
        """The study's margins."""
        return self.record.margins

    def cell(self, family: str, cell: str) -> pd.DataFrame:
        """The surviving rows of one cell, in the order the rows hold them."""
        rows = self.rows
        return rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]

    def k(self, row: pd.Series) -> int:
        """The replicates the summary row's verdict lost."""
        value = float(row["failed_replicates"])
        if math.isfinite(value):
            return int(value)
        return sum(self.lost.get(source, 0) for source in sources(row, self.rows))


Judge = Callable[[Context, pd.Series, Interval], Judged]
Applies = Callable[[Context, pd.Series], bool]


@dataclass(frozen=True)
class Clause:
    """One clause of a family's verdict, and the candidate set its statistic type declares.

    Parameters
    ----------
    name : str
        The leg name the accounting file publishes.
    statistic : str
        The statistic type, ``S1`` to ``S10`` (RM39b2-plan section 7.2).
    applies : Callable
        Whether the clause is part of a summary row's verdict.
    read : Callable
        The complete-case value, read off the summary row.
    candidates : Callable
        ``(label, value)`` for each candidate imputation of the lost replicates.
    target : str
        ``passed`` when the clause feeds the row's own verdict, ``property_passed`` when it
        feeds the family's joint clause alone.
    """

    name: str
    statistic: str
    applies: Applies
    read: Callable[[Context, pd.Series], Judged]
    candidates: Callable[[Context, pd.Series], list[tuple[str, Judged]]]
    target: str = "passed"


@dataclass(frozen=True)
class FamilyRule:
    """The declared clause tree of a family whose whole verdict a shared function computes.

    Parameters
    ----------
    families : frozenset of str
        The property families the rule reads.
    shared : str
        The ``property_verdicts`` function that computes the verdict.
    clauses : tuple of Clause
        Every clause, each with the rows it applies to.  A row's verdict is their conjunction.
    joint : str
        ``row`` when a row's ``property_passed`` is its own verdict.  ``group`` when every row
        of the family shares one joint verdict: all rows pass and every ``property_passed``
        clause holds.
    accountable : Callable
        Whether a summary row of the family is accountable at all.  The contraction slope rows
        are not (F1).
    """

    families: frozenset[str]
    shared: str
    clauses: tuple[Clause, ...]
    joint: str = "row"
    accountable: Callable[[pd.Series], bool] = field(default=lambda row: True)

    def clause(self, name: str) -> Clause:
        """The clause named ``name``."""
        for clause in self.clauses:
            if clause.name == name:
                return clause
        raise KeyError(f"{sorted(self.families)} has no clause {name!r}")


# ---- judges ---------------------------------------------------------------------------------


def floor_judge(threshold: Callable[[Context], float]) -> Judge:
    """The lower endpoint at or above a threshold."""
    return lambda ctx, row, iv: Judged(iv.low - threshold(ctx), bool(iv.low >= threshold(ctx)))


def ceiling_judge(threshold: Callable[[Context], float]) -> Judge:
    """The upper endpoint at or below a threshold."""
    return lambda ctx, row, iv: Judged(threshold(ctx) - iv.high, bool(iv.high <= threshold(ctx)))


def below_judge(threshold: Callable[[Context], float]) -> Judge:
    """The upper endpoint strictly below a threshold."""
    return lambda ctx, row, iv: Judged(threshold(ctx) - iv.high, bool(iv.high < threshold(ctx)))


def above_judge(threshold: Callable[[Context], float]) -> Judge:
    """The lower endpoint strictly above a threshold."""
    return lambda ctx, row, iv: Judged(iv.low - threshold(ctx), bool(iv.low > threshold(ctx)))


def band_judge(band: Callable[[Context], tuple[float, float]]) -> Judge:
    """The whole interval inside a band."""

    def judge(ctx: Context, row: pd.Series, iv: Interval) -> Judged:
        low, high = band(ctx)
        return Judged(min(iv.low - low, high - iv.high), bool(iv.within(low, high)))

    return judge


def resolve_judge(ctx: Context, row: pd.Series, iv: Interval) -> Judged:
    """The root-n control's rule: clear of nominal below, or at the floor above."""
    margins = ctx.margins
    below = 1.0 - margins.alpha - iv.high
    above = iv.low - margins.coverage_floor
    return Judged(max(below, above), bool(iv.high < 1.0 - margins.alpha or above >= 0.0))


def roles(*names: str) -> Applies:
    """A clause that applies to rows of the named roles."""
    return lambda ctx, row: str(row["role"]) in names


# ---- candidate imputations ------------------------------------------------------------------


def _errors(ctx: Context, group: pd.DataFrame) -> np.ndarray:
    estimates = group["estimate"].to_numpy(dtype=float)
    if ctx.record.truth_varies_by_replicate:
        return estimates - group["truth"].to_numpy(dtype=float)
    return estimates - float(group["truth"].iloc[0])


def _spread_values(group: pd.DataFrame, truth_varies: bool) -> np.ndarray:
    return spread_values(group, truth_varies=truth_varies)


def _widest(values: np.ndarray, k: int) -> np.ndarray:
    """The vertex split of ``k`` fills with the largest standard deviation."""
    options = [vertex(values, k, j) for j in range(k + 1)]
    return max(options, key=lambda option: float(np.std(option, ddof=1)))


def _imputed(
    group: pd.DataFrame,
    values: np.ndarray,
    std_errors: np.ndarray,
    *,
    truth_varies: bool,
) -> pd.DataFrame:
    """A cell frame of the survivors and the filled rows.

    ``values`` and ``std_errors`` hold the survivors first and the fills after them.
    ``values`` are the spread values: estimates, or errors when the truth varies, whose
    filled rows then take the cell's mean truth.
    """
    k = len(values) - len(group)
    truth = group["truth"].to_numpy(dtype=float)
    estimates = group["estimate"].to_numpy(dtype=float)
    if truth_varies:
        fill_truth = np.full(k, float(np.mean(truth)))
        estimates = np.concatenate([estimates, fill_truth + values[len(group) :]])
        truth = np.concatenate([truth, fill_truth])
    else:
        estimates = np.concatenate([estimates, values[len(group) :]])
        truth = np.concatenate([truth, np.full(k, float(truth[0]))])
    return pd.DataFrame(
        {
            "property": group["property"].iloc[0],
            "cell": group["cell"].iloc[0],
            "n": int(group["n"].iloc[0]),
            "truth": truth,
            "estimate": estimates,
            "std_error": std_errors,
        }
    )


def se_ratio_sets(
    group: pd.DataFrame, k: int, *, truth_varies: bool
) -> list[tuple[str, pd.DataFrame]]:
    """The S4 candidate frames of a cell that lost ``k`` replicates (RM39a2's SE-ratio set).

    The SEs at the survivor maximum with the spread values at their mean, then the SEs at the
    survivor minimum with the spread values at each vertex split.

    Parameters
    ----------
    group : pandas.DataFrame
        The cell's surviving rows.
    k : int
        The lost replicates.
    truth_varies : bool
        Whether the spread reads each row's error.

    Returns
    -------
    list of (str, pandas.DataFrame)
        Each candidate's label and frame.
    """
    values = _spread_values(group, truth_varies)
    std_errors = group["std_error"].to_numpy(dtype=float)
    out = [
        (
            "SE at max, estimates at mean",
            _imputed(
                group,
                filled(values, k, float(np.mean(values))),
                filled(std_errors, k, float(np.max(std_errors))),
                truth_varies=truth_varies,
            ),
        )
    ]
    out += [
        (
            f"SE at min, estimates j={j} at min",
            _imputed(
                group,
                vertex(values, k, j),
                filled(std_errors, k, float(np.min(std_errors))),
                truth_varies=truth_varies,
            ),
        )
        for j in range(k + 1)
    ]
    return out


def _se_ratio(frame: pd.DataFrame, truth_varies: bool) -> float:
    spread = float(np.std(_spread_values(frame, truth_varies), ddof=1))
    return float(frame["std_error"].mean() / spread) if spread > 0 else math.nan


# ---- statistic types ------------------------------------------------------------------------


def bias_clause(name: str, *, kind: str, applies: Applies) -> Clause:
    """S1: the standardized-bias interval, at every vertex split with the scale held.

    ``kind`` is ``equivalent`` (inside the margin) or ``discriminated`` (wholly outside it,
    on the side of the complete-case bias, m14).

    Parameters
    ----------
    name : str
        The leg name.
    kind : str
        ``equivalent`` or ``discriminated``.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S1 clause.
    """

    def judge(row: pd.Series, iv: Interval) -> Judged:
        margin = float(row["bias_margin"])
        if kind == "equivalent":
            return Judged(margin - max(-iv.low, iv.high), bool(iv.within(-margin, margin)))
        if float(row["bias"]) >= 0.0:
            return Judged(iv.low - margin, bool(iv.low > margin))
        return Judged(-margin - iv.high, bool(iv.high < -margin))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(row, Interval(float(row["bias_ci_lower"]), float(row["bias_ci_upper"])))

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        errors = _errors(ctx, ctx.cell(row["property"], row["cell"]))
        k = ctx.k(row)
        level = ctx.margins.confidence_level
        return [
            (
                f"j={j} at min",
                judge(row, student_interval(vertex(errors, k, j), confidence_level=level)),
            )
            for j in range(k + 1)
        ]

    if kind not in {"equivalent", "discriminated"}:
        raise ValueError(f"unknown bias clause kind {kind!r}")
    return Clause(name, "S1", applies, read, candidates)


def proportion_clause(name: str, *, statistic: str, judge: Judge, applies: Applies) -> Clause:
    """S2 (coverage) or S3 (rejection): the exact interval at every split of the lost replicates.

    Parameters
    ----------
    name : str
        The leg name.
    statistic : str
        ``S2`` for coverage, ``S3`` for rejection.
    judge : Callable
        The clause's rule on the exact interval.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S2 or S3 clause.
    """
    prefix, rate_column = {"S2": ("coverage", "coverage"), "S3": ("rejection", "rejection_rate")}[
        statistic
    ]

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(float(row[f"{prefix}_ci_lower"]), float(row[f"{prefix}_ci_upper"]))
        return judge(ctx, row, iv)

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        survivors = int(row["replicates"])
        hits = round(float(row[rate_column]) * survivors)
        k = ctx.k(row)
        level = ctx.margins.confidence_level
        word = "covered" if statistic == "S2" else "rejected"
        return [
            (
                f"j={j} {word}",
                judge(ctx, row, clopper_pearson(hits + j, survivors + k, confidence_level=level)),
            )
            for j in range(k + 1)
        ]

    return Clause(name, statistic, applies, read, candidates)


def se_point_clause(
    name: str, *, band: Callable[[Context], tuple[float, float]], applies: Applies
) -> Clause:
    """S4: the point SE ratio inside a band, over the RM39a2 SE-ratio set.

    Parameters
    ----------
    name : str
        The leg name.
    band : Callable
        The band the ratio must sit inside.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S4 clause.
    """

    def judge(ctx: Context, ratio: float) -> Judged:
        low, high = band(ctx)
        return Judged(min(ratio - low, high - ratio), bool(low <= ratio <= high))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(ctx, float(row["se_ratio"]))

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        varies = ctx.record.truth_varies_by_replicate
        group = ctx.cell(row["property"], row["cell"])
        return [
            (label, judge(ctx, _se_ratio(frame, varies)))
            for label, frame in se_ratio_sets(group, ctx.k(row), truth_varies=varies)
        ]

    return Clause(name, "S4", applies, read, candidates)


def se_interval_clause(
    name: str,
    *,
    which: str,
    judge: Judge,
    seed: Callable[[Context, pd.Series], int],
    applies: Applies,
    truth_varies: Callable[[Context], bool] | None = None,
) -> Clause:
    """S5: a bootstrap SE-ratio or efficiency interval over the S4 set, with the verdict's seed.

    ``which`` is ``se_ratio``, ``efficiency_empirical`` or ``efficiency_reported``.  An
    efficiency interval reads the cell's efficiency bound back off its complete-case ratio,
    ``sqrt(n) * mean_std_error / efficiency_reported_ratio``.

    Parameters
    ----------
    name : str
        The leg name.
    which : str
        The interval the clause reads.
    judge : Callable
        The clause's rule on the interval.
    seed : Callable
        The verdict's own resampling seed for the row.
    applies : Callable
        The rows the clause is part of.
    truth_varies : Callable or None
        Whether the spread reads each row's error; the study's flag by default.

    Returns
    -------
    Clause
        The S5 clause.
    """

    def varies(ctx: Context) -> bool:
        return ctx.record.truth_varies_by_replicate if truth_varies is None else truth_varies(ctx)

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(float(row[f"{which}_ci_lower"]), float(row[f"{which}_ci_upper"]))
        return judge(ctx, row, iv)

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        group = ctx.cell(row["property"], row["cell"])
        bound = None
        if which != "se_ratio":
            bound = (
                math.sqrt(int(row["n"]))
                * float(row["mean_std_error"])
                / float(row["efficiency_reported_ratio"])
            )
        out = []
        for label, frame in se_ratio_sets(group, ctx.k(row), truth_varies=varies(ctx)):
            intervals = ratio_intervals(
                frame,
                replicates=ctx.margins.bootstrap_replicates,
                confidence_level=ctx.margins.confidence_level,
                seed=seed(ctx, row),
                bound=bound,
                truth_varies=varies(ctx),
            )
            out.append((label, judge(ctx, row, intervals[which])))
        return out

    return Clause(name, "S5", applies, read, candidates)


def rate_slope_clause(
    name: str,
    *,
    judge: Judge,
    ladder: Callable[[pd.Series], tuple[str, str, str, tuple[str, ...]]],
    applies: Applies,
) -> Clause:
    """S6: a fitted rate slope with each lost rung at its attainable extremes.

    ``ladder(row)`` gives the ladder's property, the cell prefix of its rungs, the statistic
    (``spread`` or ``reported``) and the seed labels.  Per lost rung the spread sits at its
    attainable maximum (the widest vertex split) or minimum (the survivor mean); for
    ``reported`` the SEs sit at the survivor maximum or minimum.  Each rung takes the side
    that pushes the slope up, then the side that pushes it down.

    Parameters
    ----------
    name : str
        The leg name.
    judge : Callable
        The rule on the slope interval.
    ladder : Callable
        The ladder a rate row is fitted over.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S6 clause.
    """

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(ctx, row, Interval(float(row["slope_ci_lower"]), float(row["slope_ci_upper"])))

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family, prefix, statistic, labels = ladder(row)
        varies = ctx.record.truth_varies_by_replicate
        selected = ctx.rows.loc[
            (ctx.rows["property"] == family) & ctx.rows["cell"].str.startswith(prefix)
        ]
        rungs = [
            (int(group["n"].iloc[0]), str(cell), group)
            for cell, group in selected.groupby("cell", sort=True)
        ]
        logs = np.log([size for size, _, _ in rungs])
        centred = logs - logs.mean()
        out = []
        for direction, sign in (("up", 1.0), ("down", -1.0)):
            frames = []
            for (_size, cell, group), weight in zip(rungs, centred, strict=True):
                k = ctx.lost.get((family, cell), 0)
                if not k:
                    frames.append(
                        group.loc[:, ["property", "cell", "n", "truth", "estimate", "std_error"]]
                    )
                    continue
                high = weight * sign >= 0.0
                values = _spread_values(group, varies and statistic != "reported")
                std_errors = group["std_error"].to_numpy(dtype=float)
                if statistic == "reported":
                    values = filled(values, k, float(np.mean(values)))
                    std_errors = filled(
                        std_errors, k, float(np.max(std_errors) if high else np.min(std_errors))
                    )
                else:
                    values = (
                        _widest(values, k) if high else filled(values, k, float(np.mean(values)))
                    )
                    std_errors = filled(std_errors, k, float(np.mean(std_errors)))
                frames.append(
                    _imputed(
                        group, values, std_errors, truth_varies=varies and statistic != "reported"
                    )
                )
            fitted = rate(
                pd.concat(frames, ignore_index=True),
                property_name=family,
                statistic=statistic,
                bootstrap_replicates=ctx.margins.bootstrap_replicates,
                confidence_level=ctx.margins.confidence_level,
                seed=stream_seed(ctx.record, *labels),
                truth_varies=varies,
            )
            out.append((f"lost rungs push the slope {direction}", judge(ctx, row, fitted.interval)))
        return out

    return Clause(name, "S6", applies, read, candidates)


def _arms(ctx: Context, row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The positive and control arms of a two-cell family, from the summary's roles."""
    family = ctx.summary.loc[ctx.summary["property"] == row["property"]]
    positive = family.loc[family["role"] == "positive", "cell"]
    control = family.loc[family["role"] == "control", "cell"]
    if len(positive) != 1 or len(control) != 1:
        raise ValueError(f"{row['property']} is not one positive and one control cell")
    return (
        ctx.cell(row["property"], positive.iloc[0]),
        ctx.cell(row["property"], control.iloc[0]),
    )


def coverage_gain_clause(
    name: str, *, floor: float, seed: Callable[[Context, pd.Series], int], applies: Applies
) -> Clause:
    """S7: the paired coverage gain with every lost paired difference at -1, then at +1.

    A replicate either arm lost is a lost difference.  Exact when both arms lost the same
    replicates; conservative across keys, where a fill can be infeasible for the arm that
    survived.  It feeds the family's ``property_passed`` alone.

    Parameters
    ----------
    name : str
        The leg name.
    floor : float
        What the gain's lower endpoint must reach.
    seed : Callable
        The verdict's own resampling seed.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S7 clause.
    """

    def judge(iv: Interval) -> Judged:
        return Judged(iv.low - floor, bool(iv.low >= floor))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(
            Interval(float(row["coverage_gain_ci_lower"]), float(row["coverage_gain_ci_upper"]))
        )

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        positive, control = _arms(ctx, row)
        merged = positive[["replicate", "covered"]].merge(
            control[["replicate", "covered"]], on="replicate", suffixes=("_positive", "_control")
        )
        differences = merged["covered_positive"].to_numpy(dtype=float) - merged[
            "covered_control"
        ].to_numpy(dtype=float)
        requested = int(positive["requested_replicates"].iloc[0])
        k = requested - len(merged)
        return [
            (
                f"lost differences at {value:+.0f}",
                judge(
                    mean_difference_interval(
                        filled(differences, k, value),
                        replicates=ctx.margins.bootstrap_replicates,
                        confidence_level=ctx.margins.confidence_level,
                        seed=seed(ctx, row),
                    )
                ),
            )
            for value in (-1.0, 1.0)
        ]

    return Clause(name, "S7", applies, read, candidates, target="property_passed")


def _aligned(
    group: pd.DataFrame, requested: int, values: np.ndarray, std_errors: np.ndarray
) -> pd.DataFrame:
    """Every requested replicate of one arm: its real row where it survived, a fill where not."""
    missing = sorted(set(range(requested)) - {int(value) for value in group["replicate"]})
    fills = pd.DataFrame(
        {
            "replicate": missing,
            "estimate": values[len(group) :],
            "std_error": std_errors[len(group) :],
        }
    )
    kept = group.loc[:, ["replicate", "estimate", "std_error"]]
    return pd.concat([kept, fills], ignore_index=True).sort_values("replicate", ignore_index=True)


def deficit_clause(
    name: str,
    *,
    cells: tuple[str, str],
    judge: Judge,
    seed: Callable[[Context, pd.Series], int],
    applies: Applies,
) -> Clause:
    """S8: one cell's SE ratio minus a paired cell's, over the product of both S4 sets.

    The arms are aligned by replicate (b1 M2): the real value where an arm survived, its fill
    where it lost.  The summary row carries ``se_ratio_deficit_ci_lower`` and ``_upper``.

    Parameters
    ----------
    name : str
        The leg name.
    cells : tuple of str
        The subject and reference cells of the row's family.
    judge : Callable
        The rule on the deficit interval.
    seed : Callable
        The verdict's own resampling seed.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S8 clause.
    """

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(
            float(row["se_ratio_deficit_ci_lower"]), float(row["se_ratio_deficit_ci_upper"])
        )
        return judge(ctx, row, iv)

    def sets(ctx: Context, group: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        requested = int(group["requested_replicates"].iloc[0])
        k = requested - len(group)
        return [
            (
                label,
                _aligned(
                    group,
                    requested,
                    frame["estimate"].to_numpy(dtype=float),
                    frame["std_error"].to_numpy(dtype=float),
                ),
            )
            for label, frame in se_ratio_sets(group, k, truth_varies=False)
        ]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        subject = ctx.cell(row["property"], cells[0])
        reference = ctx.cell(row["property"], cells[1])
        out = []
        for subject_label, subject_frame in sets(ctx, subject):
            for reference_label, reference_frame in sets(ctx, reference):
                iv = se_ratio_deficit_interval(
                    subject_frame,
                    reference_frame,
                    replicates=ctx.margins.bootstrap_replicates,
                    confidence_level=ctx.margins.confidence_level,
                    seed=seed(ctx, row),
                )
                out.append(
                    (f"subject {subject_label}; reference {reference_label}", judge(ctx, row, iv))
                )
        return out

    return Clause(name, "S8", applies, read, candidates)


def spread_ratio_clause(
    name: str,
    *,
    cells: tuple[str, str],
    side: str,
    judge: Judge,
    seed: Callable[[Context, pd.Series], int],
    applies: Applies,
) -> Clause:
    """S9: a paired spread ratio, pushed toward the clause's boundary.

    For an ``upper`` clause the numerator's lost values sit at each vertex split and the
    denominator's at the survivor mean; a ``lower`` clause reverses the roles.  The arms are
    aligned by replicate.  The summary row carries ``spread_ratio_ci_lower`` and ``_upper``.

    Parameters
    ----------
    name : str
        The leg name.
    cells : tuple of str
        The numerator and denominator cells of the row's family.
    side : str
        ``upper`` or ``lower``: which endpoint the clause bounds.
    judge : Callable
        The rule on the ratio interval.
    seed : Callable
        The verdict's own resampling seed.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S9 clause.
    """
    if side not in {"upper", "lower"}:
        raise ValueError(f"unknown spread ratio side {side!r}")

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(float(row["spread_ratio_ci_lower"]), float(row["spread_ratio_ci_upper"]))
        return judge(ctx, row, iv)

    def arm(group: pd.DataFrame, spread: bool) -> list[tuple[str, pd.DataFrame]]:
        requested = int(group["requested_replicates"].iloc[0])
        k = requested - len(group)
        values = group["estimate"].to_numpy(dtype=float)
        std_errors = group["std_error"].to_numpy(dtype=float)
        if spread:
            options = [(f"j={j} at min", vertex(values, k, j)) for j in range(k + 1)]
        else:
            options = [("at mean", filled(values, k, float(np.mean(values))))]
        return [
            (label, _aligned(group, requested, option, filled(std_errors, k, 1.0)))
            for label, option in options
        ]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        numerator = ctx.cell(row["property"], cells[0])
        denominator = ctx.cell(row["property"], cells[1])
        out = []
        for numerator_label, numerator_frame in arm(numerator, side == "upper"):
            for denominator_label, denominator_frame in arm(denominator, side == "lower"):
                ratio = paired_spread_ratio_interval(
                    numerator_frame,
                    denominator_frame,
                    replicates=ctx.margins.bootstrap_replicates,
                    confidence_level=ctx.margins.confidence_level,
                    seed=seed(ctx, row),
                )
                out.append(
                    (
                        f"numerator {numerator_label}; denominator {denominator_label}",
                        judge(ctx, row, ratio.interval),
                    )
                )
        return out

    return Clause(name, "S9", applies, read, candidates)


def extra_mean_clause(
    name: str,
    *,
    column: str,
    summary_column: str,
    side: str,
    limit: float,
    applies: Applies,
) -> Clause:
    """S10: the mean of an extra row column, with the lost rows at the survivor max or min.

    A ``cap`` clause needs the mean at or below ``limit`` and fills the lost rows at the
    survivor maximum.  A ``floor`` clause needs it at or above and fills at the minimum.

    Parameters
    ----------
    name : str
        The leg name.
    column : str
        The row column the mean reads.
    summary_column : str
        The summary column that publishes the complete-case mean.
    side : str
        ``cap`` or ``floor``.
    limit : float
        The declared cap or floor.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S10 clause.
    """
    if side not in {"cap", "floor"}:
        raise ValueError(f"unknown extra mean side {side!r}")

    def judge(mean: float) -> Judged:
        if side == "cap":
            return Judged(limit - mean, bool(mean <= limit))
        return Judged(mean - limit, bool(mean >= limit))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(float(row[summary_column]))

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        values = ctx.cell(row["property"], row["cell"])[column].to_numpy(dtype=float)
        fill = float(np.max(values) if side == "cap" else np.min(values))
        label = "lost rows at survivor max" if side == "cap" else "lost rows at survivor min"
        return [(label, judge(float(np.mean(filled(values, ctx.k(row), fill)))))]

    return Clause(name, "S10", applies, read, candidates)


# ---- the summary rows a verdict reads -------------------------------------------------------


def sources(row: pd.Series, rows: pd.DataFrame) -> list[tuple[str, str]]:
    """The replication cells a summary row's verdict reads.

    A cell row reads itself.  A ``root_n_rate`` row reads its ladder, a contraction slope row
    its scenario's rungs, and a ``one_wrong_inference`` growth row its smallest and largest
    rungs.  Any other row with no replication cell reads none.

    Parameters
    ----------
    row : pandas.Series
        One summary row.
    rows : pandas.DataFrame
        The replication rows.

    Returns
    -------
    list of tuple
        The ``(property, cell)`` pairs.
    """
    from tests.studies.evidence import property_verdicts

    family, cell = str(row["property"]), str(row["cell"])
    present = rows.loc[rows["property"] == family, "cell"].astype(str)
    if (present == cell).any():
        return [(family, cell)]
    if family == "root_n_rate":
        label = cell.rsplit("__", 1)[0] if "__" in cell else ""
        prefix = f"{label}__" if label else ""
        ladder = rows.loc[rows["property"] == "root_n_and_efficiency", "cell"].astype(str)
        return [
            ("root_n_and_efficiency", name)
            for name in sorted(set(ladder))
            if name.startswith(prefix)
        ]
    if family == property_verdicts.CONTRACTION_FAMILY and cell.startswith("rate_"):
        prefix = f"{cell.removeprefix('rate_')}_n"
        return [(family, name) for name in sorted(set(present)) if name.startswith(prefix)]
    if family == property_verdicts.ONE_WRONG_FAMILY and cell.endswith("__growth"):
        setting, estimand, _kind = cell.split("__")
        pattern = re.compile(rf"{re.escape(setting)}_n([0-9]+)__{re.escape(estimand)}__first_order")
        rungs = sorted(
            (int(match.group(1)), name)
            for name in set(present)
            if (match := pattern.fullmatch(name))
        )
        return [(family, rungs[0][1]), (family, rungs[-1][1])] if rungs else []
    return []


# ---- the accounting -------------------------------------------------------------------------


def _rule_for(family: str, rules: Sequence[FamilyRule], local: frozenset[str]) -> FamilyRule | None:
    if family in local:
        return None
    for rule in rules:
        if family in rule.families:
            return rule
    return None


def _worst(clause: Clause, ctx: Context, row: pd.Series) -> tuple[Leg, bool]:
    """A clause's leg: its complete case, and its least favourable candidate."""
    complete = clause.read(ctx, row)
    candidates = clause.candidates(ctx, row)
    failing = [item for item in candidates if not item[1].passed]
    label, worst = min(failing or candidates, key=lambda item: item[1].slack)
    passed = complete.passed and not failing
    return Leg(clause.name, complete.slack, worst.slack, label, passed), complete.passed


def _gate(exceeded: bool, legs: Sequence[tuple[Leg, bool]]) -> tuple[bool, str]:
    """RM39 a2's gate: the bound first, then the clauses under their worst cases.

    A clause that holds at the complete cases and fails a candidate is ``worst case fails``;
    a cell whose failing clauses all fail at the complete cases is ``complete case fails``.
    """
    if exceeded:
        return False, BOUND_EXCEEDED
    failing = [(leg, complete) for leg, complete in legs if not leg.passed]
    if not failing:
        return True, WITHIN_BOUND
    if any(complete for _, complete in failing):
        return False, WORST_CASE_FAILS
    return False, COMPLETE_CASE_FAILS


def _account_row(
    payload: tuple[Context, pd.Series, FamilyRule | None, float, list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    ctx, row, rule, bound, cells = payload
    failed = ctx.k(row)
    requested = sum(count for count, _ in cells) if cells else failed
    if math.isfinite(float(row.get("replicates", math.nan))):
        requested = int(row["replicates"]) + failed
    allowance = sum(allowed(bound, count) for count, _ in cells)
    exceeded = any(lost > allowed(bound, count) for count, lost in cells)
    bound_leg = Leg(BOUND_LEG, float(failed), float(allowance), "", not exceeded)
    if rule is None or not rule.accountable(row):
        legs = [(bound_leg, True), (Leg(NOT_ACCOUNTABLE_LEG, math.nan, math.nan, "", False), True)]
        gate = (False, BOUND_EXCEEDED if exceeded else NOT_ACCOUNTABLE)
    else:
        clause_legs = [
            _worst(clause, ctx, row) for clause in rule.clauses if clause.applies(ctx, row)
        ]
        legs = [(bound_leg, True), *clause_legs]
        gate = _gate(exceeded, clause_legs)
    return [
        {
            "property": str(row["property"]),
            "cell": str(row["cell"]),
            "role": str(row["role"]),
            "requested": requested,
            "failed": failed,
            "rate": failed / requested if requested else math.nan,
            "bound": bound,
            "allowed": allowance,
            "leg": leg.name,
            "complete_case": leg.complete_case,
            "worst_case": leg.worst_case,
            "imputation": leg.imputation,
            "leg_passed": leg.passed,
            "gate_passed": gate[0],
            "reason": gate[1],
        }
        for leg, _ in legs
    ]


def property_accounting(
    rows: pd.DataFrame,
    records: pd.DataFrame | None,
    summary: pd.DataFrame,
    *,
    record: StudyRecord,
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
    local: frozenset[str] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """The accounting rows of every summary row whose verdict lost a replicate.

    A row is red when a cell it reads lost more replicates than ``FAILURE_BOUND`` allows, or
    when a clause of its family's :class:`FamilyRule` that holds on the complete cases fails
    under a candidate of its statistic type.  A family with no shared rule, or one the study
    computes itself (:data:`STUDY_LOCAL`), is not accountable: a row of it that lost a
    replicate is red.  A diagnostic row gets no accounting row.  The accounting only turns
    rows red; :func:`correct_properties` applies it.

    Parameters
    ----------
    rows : pandas.DataFrame
        The stamped surviving replication rows.
    records : pandas.DataFrame or None
        The property failure records.
    summary : pandas.DataFrame
        The complete-case summary, as the study's ``summarize_properties`` returns it.
    record : StudyRecord
        The study.
    module : module or None
        The property module, which declares ``FAILURE_BOUND`` and, for an efficiency claim,
        ``EFFICIENCY_RATIO_BAND``.
    rules : Sequence of FamilyRule or None
        The shared rules; :data:`tests.studies.evidence.property_verdicts.FAMILY_RULES` by
        default.
    local : frozenset of str or None
        The families the study computes itself; its :data:`STUDY_LOCAL` entry by default.
    n_jobs : int
        Worker processes; rows run in parallel and candidates in sequence.

    Returns
    -------
    pandas.DataFrame
        The rows of :data:`PROPERTY_ACCOUNTING_COLUMNS`, empty without a record.
    """
    from tests.studies.evidence import property_verdicts

    columns = list(PROPERTY_ACCOUNTING_COLUMNS)
    if records is None or records.empty:
        return pd.DataFrame(columns=columns)
    rules = property_verdicts.FAMILY_RULES if rules is None else rules
    local = STUDY_LOCAL.get(record.slug, frozenset()) if local is None else local
    bound = float(getattr(module, "FAILURE_BOUND", 0.0))
    if not 0.0 <= bound <= MAXIMUM_BOUND:
        raise ValueError(f"FAILURE_BOUND {bound} is outside [0, {MAXIMUM_BOUND}]")
    lost = counts(records)
    ctx = Context(
        record,
        rows,
        summary,
        lost,
        {"efficiency_band": getattr(module, "EFFICIENCY_RATIO_BAND", None)},
    )
    requested = {
        (str(family), str(cell)): int(group["requested_replicates"].iloc[0])
        for (family, cell), group in rows.groupby(["property", "cell"], sort=True)
    }
    scored = summary.loc[summary["role"] != property_verdicts.DIAGNOSTIC_ROLE]
    touched_families = set()
    reads: dict[Any, list[tuple[str, str]]] = {}
    for index, row in scored.iterrows():
        reads[index] = sources(row, rows)
        if ctx.k(row) > 0:
            touched_families.add(str(row["property"]))
    payloads = []
    for index, row in scored.iterrows():
        rule = _rule_for(str(row["property"]), rules, local)
        grouped = rule is not None and rule.joint == "group"
        if ctx.k(row) == 0 and not (grouped and str(row["property"]) in touched_families):
            continue
        cells = [(requested[key], lost.get(key, 0)) for key in reads[index]]
        if cells and len(cells) == 1 and math.isfinite(float(row["failed_replicates"])):
            # A cell row's own verdict may read fewer replicates than the cell holds: a
            # contraction rung at its verdict budget publishes its own requested and lost.
            cells = [(int(row["replicates"]) + ctx.k(row), ctx.k(row))]
        payloads.append(((ctx, row, rule, bound, cells),))
    out = [
        entry
        for entries in map_parallel(_account_row, payloads, n_jobs=n_jobs)
        for entry in entries
    ]
    return pd.DataFrame.from_records(out, columns=columns)


def correct_properties(
    summary: pd.DataFrame,
    accounting: pd.DataFrame,
    *,
    rules: Sequence[FamilyRule] | None = None,
) -> pd.DataFrame:
    """``properties.csv`` with each accounting-red row's verdicts set false.

    A red clause sets its targets false: ``passed`` (and the row's ``property_passed``) for a
    row clause, the family's ``property_passed`` for a joint one.  A row whose cell exceeded
    its bound sets its own verdicts, and the family's joint verdict where the rule's joint
    clause follows its cells.  A not-accountable row sets its own verdicts and the family's
    ``property_passed``, because no rule says which rows its joint claim spans.  A summary row
    that published no failure count, such as a growth row, takes the accounted count.  No
    column is added, so a summary with no failure comes back unchanged.

    Parameters
    ----------
    summary : pandas.DataFrame
        The complete-case summary.
    accounting : pandas.DataFrame
        Its :func:`property_accounting` rows.
    rules : Sequence of FamilyRule or None
        The shared rules; :data:`tests.studies.evidence.property_verdicts.FAMILY_RULES` by
        default.

    Returns
    -------
    pandas.DataFrame
        The corrected summary.
    """
    from tests.studies.evidence import property_verdicts

    if accounting.empty:
        return summary
    rules = property_verdicts.FAMILY_RULES if rules is None else rules
    out = summary.copy()
    for (family, cell), group in accounting.groupby(["property", "cell"], sort=False):
        mask = (out["property"] == family) & (out["cell"] == cell)
        if not mask.any():
            raise ValueError(f"the accounting names {family}/{cell}, which the summary lacks")
        failed = int(group["failed"].iloc[0])
        if failed and not np.isfinite(out.loc[mask, "failed_replicates"].astype(float)).all():
            out.loc[mask, "failed_replicates"] = failed
        if bool(group["gate_passed"].iloc[0]):
            continue
        whole = out["property"] == family
        reason = str(group["reason"].iloc[0])
        rule = (
            None
            if (group["leg"] == NOT_ACCOUNTABLE_LEG).any()
            else _rule_for(family, rules, frozenset())
        )
        if rule is None:
            out.loc[mask, "passed"] = False
            out.loc[whole, "property_passed"] = False
            continue
        joint = whole if rule.joint == "group" else mask
        if reason == BOUND_EXCEEDED:
            out.loc[mask, "passed"] = False
            out.loc[joint, "property_passed"] = False
            continue
        for name in group.loc[~group["leg_passed"].astype(bool), "leg"]:
            if name == BOUND_LEG:
                continue
            if rule.clause(str(name)).target == "passed":
                out.loc[mask, "passed"] = False
            out.loc[joint, "property_passed"] = False
    return out


def rule_verdicts(rule: FamilyRule, summary: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """A rule's complete-case verdicts, read off a summary's columns alone.

    The reproduction and differential tests compare these with the study's own ``passed``
    and ``property_passed`` (F10).

    Parameters
    ----------
    rule : FamilyRule
        The rule.
    summary : pandas.DataFrame
        A summary the study's own verdicts computed.
    ctx : Context
        The context its clauses read.

    Returns
    -------
    pandas.DataFrame
        ``property``, ``cell``, ``passed`` and ``property_passed`` per accountable row.
    """
    from tests.studies.evidence import property_verdicts

    out = []
    for family in sorted(rule.families):
        rows = summary.loc[
            (summary["property"] == family) & (summary["role"] != property_verdicts.DIAGNOSTIC_ROLE)
        ]
        verdicts = []
        for _, row in rows.iterrows():
            if not rule.accountable(row):
                continue
            judged = [
                (clause, clause.read(ctx, row))
                for clause in rule.clauses
                if clause.applies(ctx, row)
            ]
            own = all(value.passed for clause, value in judged if clause.target == "passed")
            joint = all(value.passed for clause, value in judged if clause.target != "passed")
            verdicts.append([family, str(row["cell"]), own, joint])
        if rule.joint == "group" and verdicts:
            together = all(own and joint for _, _, own, joint in verdicts)
            verdicts = [[family, cell, own, together] for family, cell, own, _ in verdicts]
        else:
            verdicts = [[family, cell, own, own and joint] for family, cell, own, joint in verdicts]
        out.extend(verdicts)
    return pd.DataFrame(out, columns=["property", "cell", "passed", "property_passed"])


#: The perturbations the differential test applies to committed rows (F10).  Each moves the
#: summary across thresholds a complete run never reaches, so a rule that drops a conjunct or
#: moves a threshold disagrees with the study's own verdict somewhere.
PERTURBATIONS = ("none", "se_low", "se_high", "bias")


def perturbed(rows: pd.DataFrame, kind: str, *, alpha: float = 0.05) -> pd.DataFrame:
    """Replication rows moved by one :data:`PERTURBATIONS` entry, kept coherent.

    ``se_low`` scales every standard error by 0.7 and ``se_high`` by 1.45.  ``bias`` shifts
    each cell's estimates by 0.35 of its own spread.  Where the standard error is finite and
    positive, ``covered`` and ``rejected`` are recomputed from a Wald interval at ``alpha``,
    so coverage and rejection move with the change.  The rows are no study's output: the test
    reads only whether a rule and the study's own code agree on them.

    Parameters
    ----------
    rows : pandas.DataFrame
        The replication rows.
    kind : str
        One of :data:`PERTURBATIONS`.
    alpha : float
        The level of the recomputed interval and test.

    Returns
    -------
    pandas.DataFrame
        A perturbed copy.
    """
    from scipy.stats import norm

    if kind not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {kind!r}")
    out = rows.copy()
    if kind == "none":
        return out
    estimates = out["estimate"].astype(float)
    std_errors = out["std_error"].astype(float)
    if kind == "se_low":
        std_errors = std_errors * 0.7
    elif kind == "se_high":
        std_errors = std_errors * 1.45
    else:
        spread = out.groupby(["property", "cell"])["estimate"].transform("std").fillna(0.0)
        estimates = estimates + 0.35 * spread
    out["estimate"] = estimates
    out["std_error"] = std_errors
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    usable = np.isfinite(std_errors) & (std_errors > 0.0)
    errors = (estimates - out["truth"].astype(float)).abs()
    out.loc[usable, "covered"] = (errors <= critical * std_errors)[usable].astype(int)
    out.loc[usable, "rejected"] = ((estimates / std_errors).abs() > critical)[usable].astype(int)
    return out


def differential(
    rules: Sequence[FamilyRule],
    summary: pd.DataFrame,
    ctx: Context,
    *,
    local: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """The accountable rows where a rule's verdict and the study's own disagree.

    Parameters
    ----------
    rules : Sequence of FamilyRule
        The rules to hold to the summary.
    summary : pandas.DataFrame
        A summary the study's own ``summarize_properties`` computed.
    ctx : Context
        The context the clauses read.
    local : frozenset of str
        The study's :data:`STUDY_LOCAL` families, which no rule is held to.

    Returns
    -------
    pandas.DataFrame
        ``property``, ``cell``, the rule's ``passed`` and ``property_passed`` and the
        study's, one row per disagreement.
    """
    present = set(summary["property"].astype(str)) - local
    frames = [rule_verdicts(rule, summary, ctx) for rule in rules if rule.families & present]
    columns = ["property", "cell", "passed", "property_passed"]
    if not frames:
        return pd.DataFrame(columns=[*columns, "study_passed", "study_property_passed"])
    verdicts = pd.concat(frames, ignore_index=True)
    merged = verdicts.loc[~verdicts["property"].isin(local)].merge(
        summary[columns].rename(
            columns={"passed": "study_passed", "property_passed": "study_property_passed"}
        ),
        on=["property", "cell"],
        validate="one_to_one",
    )
    wrong = (merged["passed"] != merged["study_passed"].astype(bool)) | (
        merged["property_passed"] != merged["study_property_passed"].astype(bool)
    )
    return merged.loc[wrong].reset_index(drop=True)
