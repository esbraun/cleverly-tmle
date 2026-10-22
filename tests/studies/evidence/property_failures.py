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

A run with a record also publishes ``property-failure-accounting.csv`` (:func:`property_accounting`)
and ``property-failure-witnesses.parquet``, the draw-level witness of every key of each cell that
lost a replicate (rev 3 D3'f).
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import itertools
import json
import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

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
from tests.studies.evidence.inference import (
    Interval,
    clopper_pearson,
    percentile_interval,
    student_interval,
)
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.properties import (
    absolute_bias_bound,
    log_slope,
    mean_difference_interval,
    paired,
    paired_spread_ratio_interval,
    rate_draws,
    rate_observed,
    rate_samples,
    ratio_intervals,
    spread_values,
    standardized_bias,
)
from tests.studies.evidence.registry import Margins, StudyRecord
from tests.studies.evidence.replication import WITNESS_COLUMNS, Recording
from tests.studies.evidence.seeds import stream_seed

#: The published record file and the published accounting file of the property phase.  Each
#: exists only when a property fit failed, so a study without one publishes exactly the
#: artifacts it did before.
PROPERTY_FAILURES = "property-failures.csv"
PROPERTY_ACCOUNTING = "property-failure-accounting.csv"

#: The manifest configuration key that records the property bound a run applied (rev 3 M1).
PROPERTY_BOUND_KEY = "property_failure_bound"

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


def raised_after_failures(here: Path, bound: float) -> bool:
    """Whether a property run would raise its bound after a committed run lost replicates.

    A committed ``property-failures.csv`` shows that the committed property run lost a
    replicate.  A re-run draws the same seeds and meets the same failures, so a bound above
    the one that run applied would buy the pass (rev 3 section 6).  The applied bound is the
    committed manifest's :data:`PROPERTY_BOUND_KEY`, and 0 when the key is absent.

    Parameters
    ----------
    here : pathlib.Path
        The study's artifact directory.
    bound : float
        The module's ``FAILURE_BOUND`` for this run.

    Returns
    -------
    bool
        ``True`` when the committed run recorded a property failure and ``bound`` exceeds the
        bound it applied.
    """
    records = read(here / PROPERTY_FAILURES)
    if records is None or records.empty:
        return False
    manifest = here / "manifest.json"
    configuration = (
        (json.loads(manifest.read_text()).get("configuration") or {}) if manifest.exists() else {}
    )
    return float(bound) > float(configuration.get(PROPERTY_BOUND_KEY, 0.0))


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


# ---- the accounting (RM39-plan 4.5, RM39b2-plan section 7, rev 3) -----------------------------

#: The columns of ``property-failure-accounting.csv``: one row per cell and clause, after one
#: ``failure bound`` row, then one witness row per lost replicate of the cell.
#: ``complete_case`` and ``worst_case`` hold the clause's slack, the distance by which it clears
#: its threshold, so a negative value fails.  ``imputation`` names the leg's label (``support``
#: or ``range``, D3'e) and its least favourable candidate.  A witness row holds the lost draw's
#: statistic in ``complete_case`` and its mid-rank quantile among the survivors' in
#: ``worst_case``.
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

#: The witness values a run with a property failure publishes beside its records: the
#: statistic of every key of each cell that lost a replicate (D3'f).
PROPERTY_WITNESSES = "property-failure-witnesses.parquet"

#: The text columns of :data:`PROPERTY_WITNESSES`, which a file with no failure leaves empty.
WITNESS_DTYPE = {"property": str, "cell": str, "statistic": str}

#: The largest ``FAILURE_BOUND`` a property module may declare (RM39-plan a2 Q3).
MAXIMUM_BOUND = 0.05

#: The leg every accounted cell writes first, the leg of a clause no candidate set accounts
#: for, and the reported row of a lost replicate's draw-level witness.
BOUND_LEG = "failure bound"
NOT_ACCOUNTABLE_LEG = "failures not accountable"
#: The ``imputation`` of a clause leg of a not-accountable cell: no candidate set exists, so
#: the leg gives the complete case alone and its ``worst_case`` is empty.
COMPLETE_CASE_ONLY = "complete case only"
WITNESS_LEG = "informative-failure witness"

#: The two leg labels (D3'e).  A ``support`` leg fills a bounded variable over its full
#: support and needs no assumption.  A ``range`` leg rests on B2's assumption that a lost value
#: lies in the survivors' range, with the margin's scale held at its complete-case value.
SUPPORT = "support"
RANGE = "range"

#: How a family's rows share a joint verdict (rev 3 section 5.0).  ``row``: each row's
#: ``property_passed`` is its own.  ``pair``, ``label`` and ``family``: the members of one
#: declared group share one, which holds when every member passes and every
#: ``property_passed`` clause of every member holds.
GROUPS = ("row", "pair", "label", "family")

#: Rounding slack on ``bound * requested``, as :data:`tests.studies.evidence.failures` keeps.
_FLOOR_SLACK = 1e-9


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
    groups : dict
        The module's ``PROPERTY_GROUPS``: each family's declared groups of cells, for a rule
        whose ``group`` is ``pair`` or ``label`` (rev 3 L5).  Pairs are never inferred from
        cell names.
    """

    record: StudyRecord
    rows: pd.DataFrame
    summary: pd.DataFrame
    lost: dict[tuple[str, str], int] = field(default_factory=dict)
    groups: Mapping[str, Sequence[Sequence[str]]] = field(default_factory=dict)

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

    def lost_in(self, family: str, cell: str) -> int:
        """The replicates the records cost one cell."""
        return self.lost.get((family, cell), 0)

    def members(self, family: str, cell: str, group: str) -> tuple[str, ...]:
        """The cells that share ``cell``'s joint verdict under a rule of ``group``."""
        return group_members(self.summary, self.groups, family, cell, group)


def group_members(
    summary: pd.DataFrame,
    groups: Mapping[str, Sequence[Sequence[str]]],
    family: str,
    cell: str,
    group: str,
) -> tuple[str, ...]:
    """The cells of ``family`` that share ``cell``'s joint verdict.

    A ``row`` rule's group is the cell alone.  A declared group (``PROPERTY_GROUPS``) is read
    as declared.  An undeclared ``pair`` or ``family`` group is the whole family: a family
    with one pair is that pair.  A ``label`` rule needs a declaration.

    Parameters
    ----------
    summary : pandas.DataFrame
        The summary the family's cells are read from.
    groups : Mapping
        The module's ``PROPERTY_GROUPS``.
    family : str
        The property family.
    cell : str
        One cell of it.
    group : str
        The rule's :data:`GROUPS` entry.

    Returns
    -------
    tuple of str
        The member cells, ``cell`` included.

    Raises
    ------
    ValueError
        When a declared family does not place the cell, or a ``label`` rule has no
        declaration.
    """
    from tests.studies.evidence import property_verdicts

    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")
    if group == "row":
        return (cell,)
    declared = groups.get(family)
    if declared:
        for members in declared:
            if cell in members:
                return tuple(members)
        raise ValueError(f"{family}/{cell} is in no group the module declares for {family}")
    if group == "label":
        raise ValueError(f"{family} groups by label, and the module declares no label groups")
    selected = summary.loc[
        (summary["property"] == family) & (summary["role"] != property_verdicts.DIAGNOSTIC_ROLE),
        "cell",
    ]
    return tuple(sorted(str(name) for name in selected))


Judge = Callable[[Context, pd.Series, Interval], Judged]
Applies = Callable[[Context, pd.Series], bool]
Reads = Callable[[Context, pd.Series], Sequence[tuple[str, str]]]


@dataclass(frozen=True)
class Clause:
    """One clause of a family's verdict, and the candidate set its statistic type declares.

    Parameters
    ----------
    name : str
        The leg name the accounting file publishes.
    statistic : str
        The statistic type (rev 3 section 4): ``S1``, ``S1f`` and ``S2`` to ``S15``.
    applies : Callable
        Whether the clause is part of a summary row's verdict.
    read : Callable
        The complete-case value, read off the summary row.
    candidates : Callable or None
        ``(label, value)`` for each candidate imputation of the lost replicates.  ``None`` for
        a clause that no candidate set accounts for.
    target : str
        ``passed`` when the clause feeds the row's own verdict, ``property_passed`` when it
        feeds the joint verdict alone.
    imputation : str
        :data:`SUPPORT` or :data:`RANGE`: whether the leg rests on the range assumption.
    accountable : bool
        ``False`` for a clause with a stated reason to have no candidate set (rev 3 section
        2): a row of it that lost a replicate is red at any bound (R3-1).
    reads : Callable or None
        Further ``(property, cell)`` pairs the clause reads beyond the row's own cell, such as
        the other arm of a pair or another row's endpoint.  A loss in any of them makes the
        row accounted.
    """

    name: str
    statistic: str
    applies: Applies
    read: Callable[[Context, pd.Series], Judged]
    candidates: Callable[[Context, pd.Series], list[tuple[str, Judged]]] | None
    target: str = "passed"
    imputation: str = RANGE
    accountable: bool = True
    reads: Reads | None = None


@dataclass(frozen=True)
class FamilyRule:
    """The declared clause tree of one or more property families.

    A shared rule sits in :data:`tests.studies.evidence.property_verdicts.FAMILY_RULES`.  A
    module that adds a conjunct, or computes a family itself, declares its own in
    ``PROPERTY_RULES``, which takes precedence (rev 3 D3'a).

    Parameters
    ----------
    families : frozenset of str
        The property families the rule reads.
    shared : str
        The function that computes the verdict.
    clauses : tuple of Clause
        Every clause, each with the rows it applies to.  A row's verdict is the conjunction of
        its ``passed`` clauses.
    group : str
        How the rows share ``property_passed`` (:data:`GROUPS`).
    sources : Callable or None
        The replication cells a summary row with no cell of its own reads; :func:`sources`
        by default.
    """

    families: frozenset[str]
    shared: str
    clauses: tuple[Clause, ...]
    group: str = "row"
    sources: Callable[[pd.Series, pd.DataFrame], list[tuple[str, str]]] | None = None

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"unknown group {self.group!r}; expected one of {GROUPS}")
        names = [clause.name for clause in self.clauses]
        if len(set(names)) != len(names):
            raise ValueError(f"{sorted(self.families)} names a clause twice: {names}")
        for clause in self.clauses:
            if clause.target not in {"passed", "property_passed"}:
                raise ValueError(f"{clause.name}: unknown target {clause.target!r}")
            if clause.imputation not in {SUPPORT, RANGE}:
                raise ValueError(f"{clause.name}: unknown imputation {clause.imputation!r}")
            if clause.accountable and clause.candidates is None:
                raise ValueError(f"{clause.name}: an accountable clause needs candidates")

    def clause(self, name: str) -> Clause:
        """The clause named ``name``."""
        for clause in self.clauses:
            if clause.name == name:
                return clause
        raise KeyError(f"{sorted(self.families)} has no clause {name!r}")


def rule_for(
    family: str,
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
) -> FamilyRule | None:
    """The rule that declares ``family``: the module's own first, then the shared one.

    Parameters
    ----------
    family : str
        The property family.
    module : module or None
        The property module, whose ``PROPERTY_RULES`` take precedence (rev 3 D3'a).
    rules : Sequence of FamilyRule or None
        The shared rules; :data:`tests.studies.evidence.property_verdicts.FAMILY_RULES` by
        default.

    Returns
    -------
    FamilyRule or None
        The rule, or ``None`` when no rule declares the family.
    """
    from tests.studies.evidence import property_verdicts

    for rule in getattr(module, "PROPERTY_RULES", ()) or ():
        if family in rule.families:
            return rule
    for rule in property_verdicts.FAMILY_RULES if rules is None else rules:
        if family in rule.families:
            return rule
    return None


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


def upper_at_or_above_judge(threshold: Callable[[Context], float]) -> Judge:
    """The upper endpoint at or above a threshold: an interval that still reaches it.

    A control that must read no reduction passes on this judge, because a ratio interval
    whose upper endpoint falls below the threshold has found a reduction.
    """
    return lambda ctx, row, iv: Judged(iv.high - threshold(ctx), bool(iv.high >= threshold(ctx)))


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


def cells(*names: str) -> Applies:
    """A clause that applies to rows of the named cells."""
    return lambda ctx, row: str(row["cell"]) in names


def every(ctx: Context, row: pd.Series) -> bool:
    """A clause that applies to every row of its family."""
    return True


# ---- candidate imputations ------------------------------------------------------------------


def _errors(ctx: Context, group: pd.DataFrame, truth: float | None = None) -> np.ndarray:
    estimates = group["estimate"].to_numpy(dtype=float)
    if truth is not None:
        return estimates - truth
    if ctx.record.truth_varies_by_replicate:
        return estimates - group["truth"].to_numpy(dtype=float)
    return estimates - float(group["truth"].iloc[0])


def _spread_values(group: pd.DataFrame, truth_varies: bool) -> np.ndarray:
    return spread_values(group, truth_varies=truth_varies)


def _requested(group: pd.DataFrame) -> int:
    return int(group["requested_replicates"].iloc[0])


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


def _completed(group: pd.DataFrame, estimates: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """Every requested replicate of one arm, ordered by id: survivors real, fills after.

    ``estimates`` and ``truth`` hold the survivors in the group's order, then one value per
    lost id in ascending id order.  The frame carries complete stamps, so the paired helpers
    read it as a complete arm (b1 M2's aligned construction).
    """
    requested = _requested(group)
    survived = [int(value) for value in group["replicate"]]
    missing = sorted(set(range(requested)) - set(survived))
    frame = pd.DataFrame(
        {
            "property": group["property"].iloc[0],
            "cell": group["cell"].iloc[0],
            "role": group["role"].iloc[0] if "role" in group else "positive",
            "replicate": survived + missing,
            "n": int(group["n"].iloc[0]),
            "requested_replicates": requested,
            "failed_replicates": 0,
            "truth": truth,
            "estimate": estimates,
            "std_error": np.concatenate(
                [group["std_error"].to_numpy(dtype=float), np.ones(len(missing))]
            ),
        }
    )
    return frame.sort_values("replicate", ignore_index=True)


def se_ratio_sets(
    group: pd.DataFrame, k: int, *, truth_varies: bool
) -> list[tuple[str, pd.DataFrame]]:
    """The S4 candidate frames of a cell that lost ``k`` replicates (RM39a2's SE-ratio set).

    The SEs at the survivor maximum with the spread values at their mean, then the SEs at the
    survivor minimum with the spread values at each vertex split.  The numerator and the
    denominator read disjoint fills, the mean SE is linear in its fills and the standard
    deviation is convex in its own, so the set holds both extremes of the point ratio (G7).

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


@dataclass(frozen=True)
class Option:
    """One candidate fill of one cell: its label, its bootstrap draws and its point value."""

    label: str
    draws: np.ndarray
    point: float


def prune(options: Sequence[Option], weight: float, direction: int) -> list[Option]:
    """The options no other option dominates in one adverse direction (rev 3 M5).

    A combined statistic ``sum_r w_r X_r`` is monotone in each cell's draw vector.  For
    ``direction = -1`` the adverse side is a small combination: an option whose ``w X`` is at
    or above another's on every draw and at the point can never be the worst, so it goes.
    ``direction = +1`` mirrors it.  Exact: the worst combination survives.

    Parameters
    ----------
    options : Sequence of Option
        One cell's candidate fills, evaluated with common random numbers.
    weight : float
        The cell's weight in the combined statistic.
    direction : int
        ``-1`` to keep the options that can push the combination down, ``+1`` up.

    Returns
    -------
    list of Option
        The undominated options, in their original order.
    """
    if len(options) <= 1:
        return list(options)
    if weight == 0.0:
        return [options[0]]
    sign = -direction * float(np.sign(weight))
    # ``score`` is large where the option is adverse.
    scores = [(sign * option.draws, sign * option.point) for option in options]
    order = sorted(
        range(len(options)), key=lambda index: float(np.mean(scores[index][0])), reverse=True
    )
    kept: list[int] = []
    for index in order:
        draws, point = scores[index]
        if any(np.all(scores[other][0] >= draws) and scores[other][1] >= point for other in kept):
            continue
        kept.append(index)
    return [options[index] for index in sorted(kept)]


def _combinations(
    per_cell: Sequence[Sequence[Option]], weights: Sequence[float]
) -> list[tuple[Option, ...]]:
    """Every combination of the pruned options, in both adverse directions."""
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[Option, ...]] = []
    for direction in (-1, 1):
        pruned = [
            prune(options, weight, direction)
            for options, weight in zip(per_cell, weights, strict=True)
        ]
        for chosen in itertools.product(*pruned):
            key = tuple(id(option) for option in chosen)
            if key not in seen:
                seen.add(key)
                out.append(chosen)
    return out


# ---- statistic types (rev 3 section 4) ----------------------------------------------------


def bias_clause(
    name: str,
    *,
    kind: str,
    applies: Applies,
    side: Callable[[Context, pd.Series], int] | None = None,
    margin: Callable[[Context, pd.Series], float] | None = None,
    strict: bool = True,
    prefix: str = "bias",
    truth: Callable[[Context, pd.Series], float] | None = None,
    statistic: str = "S1",
    target: str = "passed",
) -> Clause:
    """S1: a bias interval against a margin, at every vertex split with the scale held.

    ``kind`` is ``equivalent`` (the interval inside ``[-m, m]``) or ``beyond`` (wholly beyond
    ``m`` on one side).  The side is ``side(ctx, row)`` when declared, else the complete-case
    side, the sign of the interval's centre (m14).  The margin is ``margin(ctx, row)``, by
    default the row's ``<prefix>_margin``, the standardized margin times the complete-case
    spread held fixed (B2).  A fixed threshold is the ``S1f`` form.  The candidates fill the
    lost errors at each split ``j`` between the survivor minimum and maximum.  A Student upper
    endpoint is convex in the fills and a lower one concave, so the set is exact (section 2),
    and a candidate must clear on the complete case's side (L6).

    Parameters
    ----------
    name : str
        The leg name.
    kind : str
        ``equivalent`` or ``beyond``.
    applies : Callable
        The rows the clause is part of.
    side : Callable or None
        A declared side, ``+1`` or ``-1``.
    margin : Callable or None
        The margin or fixed threshold; ``row[f"{prefix}_margin"]`` by default.
    strict : bool
        Whether ``beyond`` needs a strict inequality.
    prefix : str
        The interval the row publishes, ``<prefix>_ci_lower`` and ``<prefix>_ci_upper``.
    truth : Callable or None
        The truth the errors are measured against; the cell's own by default.
    statistic : str
        ``S1``, or ``S1f`` for a fixed threshold.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S1 clause.
    """
    if kind not in {"equivalent", "beyond"}:
        raise ValueError(f"unknown bias clause kind {kind!r}")

    def threshold(ctx: Context, row: pd.Series) -> float:
        return float(row[f"{prefix}_margin"]) if margin is None else float(margin(ctx, row))

    def interval(row: pd.Series) -> Interval:
        return Interval(float(row[f"{prefix}_ci_lower"]), float(row[f"{prefix}_ci_upper"]))

    def sign_of(ctx: Context, row: pd.Series) -> int:
        if side is not None:
            return 1 if side(ctx, row) > 0 else -1
        complete = interval(row)
        return 1 if (complete.low + complete.high) / 2.0 >= 0.0 else -1

    def judge(m: float, sign: int, iv: Interval) -> Judged:
        if kind == "equivalent":
            return Judged(m - max(-iv.low, iv.high), bool(iv.within(-m, m)))
        if sign > 0:
            return Judged(iv.low - m, bool(iv.low > m if strict else iv.low >= m))
        return Judged(-m - iv.high, bool(iv.high < -m if strict else iv.high <= -m))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(threshold(ctx, row), sign_of(ctx, row), interval(row))

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        group = ctx.cell(row["property"], row["cell"])
        errors = _errors(ctx, group, None if truth is None else truth(ctx, row))
        k = ctx.k(row)
        level = ctx.margins.confidence_level
        m, sign = threshold(ctx, row), sign_of(ctx, row)
        return [
            (
                f"j={j} at min",
                judge(m, sign, student_interval(vertex(errors, k, j), confidence_level=level)),
            )
            for j in range(k + 1)
        ]

    return Clause(name, statistic, applies, read, candidates, target=target)


def proportion_clause(
    name: str, *, statistic: str, judge: Judge, applies: Applies, target: str = "passed"
) -> Clause:
    """S2 (coverage) or S3 (rejection): the exact interval at every split of the lost replicates.

    A Clopper-Pearson endpoint is monotone in the success count, and an indicator's support is
    ``{0, 1}``, so the set needs no range assumption (``support``).

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
    target : str
        The verdict the clause feeds.

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

    return Clause(name, statistic, applies, read, candidates, target=target, imputation=SUPPORT)


def se_point_clause(
    name: str,
    *,
    band: Callable[[Context], tuple[float, float]],
    applies: Applies,
    of: Callable[[Context, pd.Series], str] | None = None,
    target: str = "passed",
) -> Clause:
    """S4: the point SE ratio inside a band, over the RM39a2 SE-ratio set (exact, G7).

    Parameters
    ----------
    name : str
        The leg name.
    band : Callable
        The band the ratio must sit inside.
    applies : Callable
        The rows the clause is part of.
    of : Callable or None
        The cell of the same family whose ratio the clause reads, for a cross-row read such
        as a two-size verdict written to both sizes; the row's own cell by default.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S4 clause.
    """

    def judge(ctx: Context, ratio: float) -> Judged:
        low, high = band(ctx)
        return Judged(min(ratio - low, high - ratio), bool(low <= ratio <= high))

    def cell_of(ctx: Context, row: pd.Series) -> str:
        return str(row["cell"]) if of is None else of(ctx, row)

    def read(ctx: Context, row: pd.Series) -> Judged:
        if of is None:
            return judge(ctx, float(row["se_ratio"]))
        summary = ctx.summary
        selected = summary.loc[
            (summary["property"] == row["property"]) & (summary["cell"] == cell_of(ctx, row)),
            "se_ratio",
        ]
        return judge(ctx, float(selected.iloc[0]))

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        return [(str(row["property"]), cell_of(ctx, row))]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        varies = ctx.record.truth_varies_by_replicate
        family, cell = str(row["property"]), cell_of(ctx, row)
        group = ctx.cell(family, cell)
        k = ctx.k(row) if of is None else ctx.lost_in(family, cell)
        return [
            (label, judge(ctx, _se_ratio(frame, varies)))
            for label, frame in se_ratio_sets(group, k, truth_varies=varies)
        ]

    return Clause(
        name,
        "S4",
        applies,
        read,
        candidates,
        target=target,
        reads=None if of is None else reads,
    )


def se_interval_clause(
    name: str,
    *,
    which: str,
    judge: Judge,
    seed: Callable[[Context, pd.Series], int],
    applies: Applies,
    truth_varies: Callable[[Context], bool] | None = None,
    columns: str | None = None,
    target: str = "passed",
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
        The interval :func:`~tests.studies.evidence.properties.ratio_intervals` returns.
    judge : Callable
        The clause's rule on the interval.
    seed : Callable
        The verdict's own resampling seed for the row.
    applies : Callable
        The rows the clause is part of.
    truth_varies : Callable or None
        Whether the spread reads each row's error; the study's flag by default.
    columns : str or None
        The prefix of the summary columns the interval is published in; ``which`` by default.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S5 clause.
    """
    published = which if columns is None else columns

    def varies(ctx: Context) -> bool:
        return ctx.record.truth_varies_by_replicate if truth_varies is None else truth_varies(ctx)

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(float(row[f"{published}_ci_lower"]), float(row[f"{published}_ci_upper"]))
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

    return Clause(name, "S5", applies, read, candidates, target=target)


def rung_options(
    values: np.ndarray, k: int, statistic: str, truth: float
) -> list[tuple[str, np.ndarray]]:
    """The S6 candidate fills of one rung that lost ``k`` replicates (RM39-plan 4.5, G3, G4).

    Every vertex split between the survivor minimum and maximum, plus the survivor mean for a
    spread or a mean SE, and for ``bias`` the fill nearest a zero mean.  The vertex splits hold
    the maximum of a convex spread and both extremes of a mean; the mean fill holds the
    spread's minimum, and the zero-mean fill the minimum of ``|mean - truth|``.

    Parameters
    ----------
    values : numpy.ndarray
        The rung's surviving values (:func:`~tests.studies.evidence.properties.rate_samples`).
    k : int
        The rung's lost replicates.
    statistic : str
        ``spread``, ``reported`` or ``bias``.
    truth : float
        The rung's truth.

    Returns
    -------
    list of (str, numpy.ndarray)
        Each fill's label and its completed values.
    """
    if k == 0:
        return [("survivors", values)]
    out = [(f"j={j} at min", vertex(values, k, j)) for j in range(k + 1)]
    if statistic == "bias":
        total = len(values) + k
        centre = (total * truth - float(np.sum(values))) / k
        centre = float(np.clip(centre, np.min(values), np.max(values)))
        out.append(("nearest a zero mean", filled(values, k, centre)))
    else:
        out.append(("at survivor mean", filled(values, k, float(np.mean(values)))))
    return out


def rate_slope_clause(
    name: str,
    *,
    judge: Judge,
    ladder: Callable[[pd.Series], tuple[str, str, str, tuple[str, ...]]],
    applies: Applies,
    target: str = "passed",
) -> Clause:
    """S6: a fitted rate slope with each lost rung at every candidate fill (G3, G4, M5).

    ``ladder(row)`` gives the ladder's property, the cell prefix of its rungs, the statistic
    (``spread``, ``reported`` or ``bias``) and the seed labels of
    :func:`~tests.studies.evidence.properties.rate`.  Each lost rung takes the
    :func:`rung_options` set.  Each option's rung draws come from the rung's own stream once,
    so options share common random numbers.  The slope draw is a fixed weighted sum of the
    rung draws, so :func:`prune` drops the options another dominates in each direction, and
    the product runs over the survivors.  The point slope is exact; the bootstrap interval is
    enumerated.

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
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S6 clause.
    """

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(ctx, row, Interval(float(row["slope_ci_lower"]), float(row["slope_ci_upper"])))

    def rungs(ctx: Context, row: pd.Series) -> list[tuple[int, str, pd.DataFrame]]:
        family, prefix, _, _ = ladder(row)
        selected = ctx.rows.loc[
            (ctx.rows["property"] == family) & ctx.rows["cell"].astype(str).str.startswith(prefix)
        ]
        out = [
            (int(group["n"].iloc[0]), str(cell), group)
            for cell, group in selected.groupby("cell", sort=True)
        ]
        out.sort(key=lambda item: item[0])
        return out

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        family = ladder(row)[0]
        return [(family, cell) for _, cell, _ in rungs(ctx, row)]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family, _, statistic, labels = ladder(row)
        varies = ctx.record.truth_varies_by_replicate
        ladder_rows = rungs(ctx, row)
        sizes = np.array([size for size, _, _ in ladder_rows], dtype=float)
        samples, truths = rate_samples([group for _, _, group in ladder_rows], statistic, varies)
        children = np.random.SeedSequence(stream_seed(ctx.record, *labels)).spawn(len(samples))
        replicates = ctx.margins.bootstrap_replicates
        per_rung: list[list[Option]] = []
        for index, (_, cell, _) in enumerate(ladder_rows):
            k = ctx.lost_in(family, cell)
            per_rung.append(
                [
                    Option(
                        label if not k else f"{cell} {label}",
                        rate_draws(
                            values,
                            truths[index],
                            statistic,
                            replicates=replicates,
                            seed=children[index],
                        ),
                        rate_observed(values, truths[index], statistic),
                    )
                    for label, values in rung_options(samples[index], k, statistic, truths[index])
                ]
            )
        logs = np.log(sizes)
        weights = list(logs - logs.mean())
        out = []
        for chosen in _combinations(per_rung, weights):
            draws = np.column_stack([option.draws for option in chosen])
            interval = percentile_interval(
                log_slope(sizes, draws), confidence_level=ctx.margins.confidence_level
            )
            label = "; ".join(
                option.label
                for option, (_, cell, _) in zip(chosen, ladder_rows, strict=True)
                if ctx.lost_in(family, cell)
            )
            out.append((label or "survivors", judge(ctx, row, interval)))
        return out

    return Clause(name, "S6", applies, read, candidates, target=target, reads=reads)


def _pair_arms(ctx: Context, row: pd.Series, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The positive and control arms of the row's group, from the summary's roles."""
    family = str(row["property"])
    members = ctx.members(family, str(row["cell"]), group)
    selected = ctx.summary.loc[
        (ctx.summary["property"] == family) & ctx.summary["cell"].isin(members)
    ]
    positive = selected.loc[selected["role"] == "positive", "cell"]
    control = selected.loc[selected["role"] == "control", "cell"]
    if len(positive) != 1 or len(control) != 1:
        raise ValueError(f"{family} group {members} is not one positive and one control cell")
    return ctx.cell(family, positive.iloc[0]), ctx.cell(family, control.iloc[0])


def coverage_gain_clause(
    name: str,
    *,
    floor: float,
    seed: Callable[[Context, pd.Series], int],
    applies: Applies,
    group: str = "pair",
) -> Clause:
    """S7: the paired coverage gain under the aligned fill (G5).

    A replicate the positive arm lost is filled uncovered, one the control lost is filled
    covered, and a replicate an arm kept keeps its real value (b1 M2).  Under the verdict's
    seed each bootstrap draw is a mean of the paired differences, monotone in each fill, so
    the one candidate is exact.  It feeds the group's ``property_passed`` alone.

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
    group : str
        The rule's group, which says which cells form the pair.

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

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        family = str(row["property"])
        return [(family, cell) for cell in ctx.members(family, str(row["cell"]), group)]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        positive, control = _pair_arms(ctx, row, group)
        requested = _requested(positive)
        kept = [int(value) for value in positive["replicate"]]
        ids = kept + sorted(set(range(requested)) - set(kept))
        positive_hits = dict(zip(kept, positive["covered"].to_numpy(dtype=float), strict=True))
        control_hits = dict(
            zip(
                (int(value) for value in control["replicate"]),
                control["covered"].to_numpy(dtype=float),
                strict=True,
            )
        )
        differences = np.array(
            [positive_hits.get(index, 0.0) - control_hits.get(index, 1.0) for index in ids]
        )
        interval = mean_difference_interval(
            differences,
            replicates=ctx.margins.bootstrap_replicates,
            confidence_level=ctx.margins.confidence_level,
            seed=seed(ctx, row),
        )
        return [("lost positive arm uncovered, lost control arm covered", judge(interval))]

    return Clause(
        name,
        "S7",
        applies,
        read,
        candidates,
        target="property_passed",
        imputation=SUPPORT,
        reads=reads,
    )


SpreadStatistic = Callable[[Context, pd.Series, pd.DataFrame, pd.DataFrame], Interval]


def spread_ratio_clause(
    name: str,
    *,
    cells: tuple[str, str],
    side: str,
    judge: Judge,
    seed: Callable[[Context, pd.Series], int],
    applies: Applies,
    statistic: SpreadStatistic | None = None,
    columns: tuple[str | None, str] = ("spread_ratio_ci_lower", "spread_ratio_ci_upper"),
    target: str = "passed",
) -> Clause:
    """S9: a paired spread ratio, pushed toward the clause's boundary.

    For an ``upper`` clause the numerator's lost values sit at each vertex split and the
    denominator's at the survivor mean; a ``lower`` clause reverses the roles.  ``side`` names
    the direction of the fills, which push the ratio up or down, and it does not name the
    endpoint the judge reads: a ``lower`` clause may be judged on the upper endpoint, as a
    control whose ratio must not fall below its boundary is.  The arms are
    completed by replicate (b1 M2).  ``statistic`` is the verdict's own statistic function
    (G6); :func:`~tests.studies.evidence.properties.paired_spread_ratio_interval` by default.
    The point ratio is exact; the bootstrap interval is enumerated.

    Parameters
    ----------
    name : str
        The leg name.
    cells : tuple of str
        The numerator and denominator cells of the row's family.
    side : str
        ``upper`` or ``lower``: the direction the fills push the ratio, ``upper`` for fills
        that raise it and ``lower`` for fills that lower it.
    judge : Callable
        The rule on the ratio interval.
    seed : Callable
        The verdict's own resampling seed.
    applies : Callable
        The rows the clause is part of.
    statistic : Callable or None
        ``statistic(ctx, row, numerator, denominator)``, the interval the verdict reads.
    columns : tuple
        The summary columns of the published interval; ``None`` for an endpoint the verdict
        does not publish.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S9 clause.
    """
    if side not in {"upper", "lower"}:
        raise ValueError(f"unknown spread ratio side {side!r}")

    def default(
        ctx: Context, row: pd.Series, numerator: pd.DataFrame, den: pd.DataFrame
    ) -> Interval:
        return paired_spread_ratio_interval(
            numerator,
            den,
            replicates=ctx.margins.bootstrap_replicates,
            confidence_level=ctx.margins.confidence_level,
            seed=seed(ctx, row),
        ).interval

    evaluate = default if statistic is None else statistic

    def read(ctx: Context, row: pd.Series) -> Judged:
        low, high = columns
        iv = Interval(math.nan if low is None else float(row[low]), float(row[high]))
        return judge(ctx, row, iv)

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        return [(str(row["property"]), cell) for cell in cells]

    def arm(group: pd.DataFrame, spread: bool) -> list[tuple[str, pd.DataFrame]]:
        k = _requested(group) - len(group)
        values = group["estimate"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        fill_truth = np.full(k, float(np.mean(truth)))
        if spread:
            options = [(f"j={j} at min", vertex(values, k, j)) for j in range(k + 1)]
        else:
            options = [("at mean", filled(values, k, float(np.mean(values))))]
        return [
            (label, _completed(group, option, np.concatenate([truth, fill_truth])))
            for label, option in options
        ]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family = str(row["property"])
        numerator = ctx.cell(family, cells[0])
        denominator = ctx.cell(family, cells[1])
        out = []
        for numerator_label, numerator_frame in arm(numerator, side == "upper"):
            for denominator_label, denominator_frame in arm(denominator, side == "lower"):
                interval = evaluate(ctx, row, numerator_frame, denominator_frame)
                out.append(
                    (
                        f"numerator {numerator_label}; denominator {denominator_label}",
                        judge(ctx, row, interval),
                    )
                )
        return out

    return Clause(name, "S9", applies, read, candidates, target=target, reads=reads)


def extra_mean_clause(
    name: str,
    *,
    value: Callable[[pd.DataFrame], np.ndarray],
    summary_column: str,
    side: str,
    limit: float,
    applies: Applies,
    support: tuple[float, float] | None = None,
    strict: bool = False,
    target: str = "passed",
) -> Clause:
    """S10: the mean of a per-row value, with the lost rows at their adverse fill.

    ``value(group)`` derives the per-row value from a cell's rows, as the verdict does.  A
    ``cap`` clause needs the mean at or below ``limit``; a ``floor`` clause at or above.  A
    bounded value (``support``) fills the lost rows at the adverse support bound, a share at 1
    for a cap, with no range assumption.  Otherwise the lost rows sit at the survivor maximum
    for a cap or the minimum for a floor.  Exact: a mean is linear in its fills.

    Parameters
    ----------
    name : str
        The leg name.
    value : Callable
        The per-row value of a cell's rows.
    summary_column : str
        The summary column that publishes the complete-case mean.
    side : str
        ``cap`` or ``floor``.
    limit : float
        The declared cap or floor.
    applies : Callable
        The rows the clause is part of.
    support : tuple of float or None
        The value's bounded support, when it has one.
    strict : bool
        Whether the inequality is strict.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S10 clause.
    """
    if side not in {"cap", "floor"}:
        raise ValueError(f"unknown extra mean side {side!r}")

    def judge(mean: float) -> Judged:
        if side == "cap":
            return Judged(limit - mean, bool(mean < limit if strict else mean <= limit))
        return Judged(mean - limit, bool(mean > limit if strict else mean >= limit))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(float(row[summary_column]))

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        values = np.asarray(value(ctx.cell(row["property"], row["cell"])), dtype=float)
        if support is not None:
            fill = support[1] if side == "cap" else support[0]
            label = f"lost rows at the support bound {fill:g}"
        else:
            fill = float(np.max(values) if side == "cap" else np.min(values))
            label = "lost rows at survivor max" if side == "cap" else "lost rows at survivor min"
        return [(label, judge(float(np.mean(filled(values, ctx.k(row), fill)))))]

    return Clause(
        name,
        "S10",
        applies,
        read,
        candidates,
        target=target,
        imputation=SUPPORT if support is not None else RANGE,
    )


def displacement_bounds(
    left: pd.DataFrame, right: pd.DataFrame, *, truth_varies: bool
) -> tuple[float, float, float]:
    """The attainable ``mean(right) - mean(left)`` over complete arms, and the held spread.

    Each arm's lost values lie in its survivors' range, and a surviving value is real.  The
    held spread is the complete-case one, ``left``'s spread over the replicates both arms kept,
    as :func:`~tests.studies.evidence.properties.paired_displacement` reads it (B2).

    Parameters
    ----------
    left : pandas.DataFrame
        The positive arm's surviving rows.
    right : pandas.DataFrame
        The control arm's surviving rows.
    truth_varies : bool
        Whether the arms read each row's error.

    Returns
    -------
    tuple of float
        The lowest and highest attainable difference of means, and the held spread.
    """
    kept_left, _ = paired(left, right)
    held = float(
        pd.Series(spread_values(kept_left.sort_values("replicate"), truth_varies=truth_varies)).std(
            ddof=1
        )
    )
    ends = []
    for arm in (left, right):
        values = spread_values(arm, truth_varies=truth_varies)
        total = _requested(arm)
        k = total - len(values)
        ends.append(
            (
                (float(np.sum(values)) + k * float(np.min(values))) / total,
                (float(np.sum(values)) + k * float(np.max(values))) / total,
            )
        )
    (left_low, left_high), (right_low, right_high) = ends
    return right_low - left_high, right_high - left_low, held


def displacement_clause(
    name: str,
    *,
    column: str,
    labels: Sequence[str],
    arms: tuple[str, str],
    threshold: float,
    truth_varies: bool = False,
    applies: Applies = every,
) -> Clause:
    """S12: the least displacement over labels, with each arm's lost values in its range.

    ``D = abs(mean(control) - mean(positive)) / sd(positive)`` with the spread held at its
    complete-case value (B2).  The attainable difference of means is an interval
    (:func:`displacement_bounds`); ``D`` is smallest at the end nearer zero, and zero when the
    interval holds zero.  The family clause reads the minimum over labels (section 2), and
    feeds ``property_passed``.  Exact under the held scale (L7).

    Parameters
    ----------
    name : str
        The leg name.
    column : str
        The summary column the joint displacement is published in.
    labels : Sequence of str
        The labels the family reports; cells are ``f"{label}__{arm}"``.
    arms : tuple of str
        The positive and control suffixes, in that order.
    threshold : float
        The declared minimum displacement.
    truth_varies : bool
        Whether each arm reads its errors, as the verdict does.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S12 clause.
    """

    def judge(value: float) -> Judged:
        return Judged(value - threshold, bool(value >= threshold))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(float(row[column]))

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        family = str(row["property"])
        return [(family, f"{label}__{arm}") for label in labels for arm in arms]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family = str(row["property"])
        least: tuple[float, str] | None = None
        for label in labels:
            low, high, held = displacement_bounds(
                ctx.cell(family, f"{label}__{arms[0]}"),
                ctx.cell(family, f"{label}__{arms[1]}"),
                truth_varies=truth_varies,
            )
            moved = 0.0 if low <= 0.0 <= high else min(abs(low), abs(high))
            value = moved / held
            if least is None or value < least[0]:
                least = (value, label)
        assert least is not None
        return [(f"{least[1]}: means at the ends nearest each other", judge(least[0]))]

    return Clause(name, "S12", applies, read, candidates, target="property_passed", reads=reads)


RatioStatistic = Callable[[pd.DataFrame, pd.DataFrame], float]


def rmse_ratio_clause(
    name: str,
    *,
    paths: Sequence[tuple[str, str]],
    statistic: RatioStatistic,
    threshold: float,
    column: str = "rmse_ratio",
    applies: Applies = every,
) -> Clause:
    """S13: the largest RMSE ratio over paths, its numerator at its worst and denominator at best.

    Per path the numerator arm's lost errors sit at the survivor end farther from zero, which
    maximizes its RMSE, and the denominator arm's lost estimates sit at the truth, 4.5's
    reference best case, which minimizes its RMSE with no range assumption on that arm.  Exact
    when the truth lies in the denominator's survivor range, conservative otherwise.  The
    group clause reads the maximum over paths and feeds ``property_passed``.

    Parameters
    ----------
    name : str
        The leg name.
    paths : Sequence of tuple of str
        Each ``(numerator cell, denominator cell)`` pair.
    statistic : Callable
        The verdict's own ratio of two arms' rows.
    threshold : float
        What every ratio must stay at or below.
    column : str
        The summary column each numerator row publishes its ratio in.
    applies : Callable
        The rows the clause is part of.

    Returns
    -------
    Clause
        The S13 clause.
    """

    def judge(value: float) -> Judged:
        return Judged(threshold - value, bool(value <= threshold))

    def read(ctx: Context, row: pd.Series) -> Judged:
        summary = ctx.summary
        numerators = [numerator for numerator, _ in paths]
        selected = summary.loc[
            (summary["property"] == row["property"]) & summary["cell"].isin(numerators), column
        ]
        return judge(float(selected.astype(float).max()))

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        family = str(row["property"])
        return list(dict.fromkeys((family, cell) for path in paths for cell in path))

    def complete(group: pd.DataFrame, *, worst: bool) -> pd.DataFrame:
        truth = group["truth"].to_numpy(dtype=float)
        estimates = group["estimate"].to_numpy(dtype=float)
        k = _requested(group) - len(group)
        fill_truth = np.full(k, float(np.mean(truth)))
        if worst:
            errors = estimates - truth
            far = max(float(np.min(errors)), float(np.max(errors)), key=abs)
            fills = fill_truth + far
        else:
            fills = fill_truth
        return _completed(
            group, np.concatenate([estimates, fills]), np.concatenate([truth, fill_truth])
        )

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family = str(row["property"])
        worst = max(
            (
                statistic(
                    complete(ctx.cell(family, numerator), worst=True),
                    complete(ctx.cell(family, denominator), worst=False),
                ),
                numerator,
            )
            for numerator, denominator in paths
        )
        return [
            (
                f"{worst[1]}: numerator at its farther end, denominator at truth",
                judge(worst[0]),
            )
        ]

    return Clause(name, "S13", applies, read, candidates, target="property_passed", reads=reads)


def error_bias_bound(errors: np.ndarray) -> float:
    """The S14 statistic of a cell's errors: :func:`absolute_bias_bound` of their mean and its
    Monte Carlo standard error, as the verdict reads them off the summary.

    Parameters
    ----------
    errors : numpy.ndarray
        The cell's errors.

    Returns
    -------
    float
        ``|mean| + 2 sd / sqrt(R)``.
    """
    errors = np.asarray(errors, dtype=float)
    return absolute_bias_bound(
        float(np.mean(errors)), float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
    )


def _equal_fill_bound(errors: np.ndarray, k: int) -> tuple[float, float]:
    """The minimum of :func:`error_bias_bound` over equal fills in the survivors' range.

    The bound is convex and symmetric in the fills, so its minimum over the box sits at equal
    fills, and over those it is a convex function of one variable.
    """
    low, high = float(np.min(errors)), float(np.max(errors))
    if k == 0 or low == high:
        return error_bias_bound(filled(errors, k, low)), low
    grid = np.linspace(low, high, 257)
    values = [error_bias_bound(filled(errors, k, float(c))) for c in grid]
    best = int(np.argmin(values))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    found = minimize_scalar(
        lambda c: error_bias_bound(filled(errors, k, float(c))),
        bounds=(float(left), float(right)),
        method="bounded",
        options={"xatol": 1e-12 * max(1.0, high - low)},
    )
    centre = float(found.x)
    value = error_bias_bound(filled(errors, k, centre))
    if values[best] < value:
        return float(values[best]), float(grid[best])
    return value, centre


def bias_bound_clause(
    name: str,
    *,
    small: str,
    large: str,
    kind: str,
    column: str,
    tolerance: float | None = None,
    applies: Applies = every,
    target: str = "passed",
) -> Clause:
    """S14: a two-size absolute-bias bound, the large size at its worst, the small at its best.

    ``B = abs(mean error) + 2 sd / sqrt(R)`` (:func:`error_bias_bound`) is convex and
    symmetric in the fills.  Its maximum over the box sits at a vertex, and symmetry reduces
    the vertices to splits; its minimum sits at equal fills, found by a one-dimensional convex
    search (p22).  ``kind`` is ``tolerance`` (``B_large < tolerance``) or ``below small``
    (``B_large < B_small``).  Exact.

    Parameters
    ----------
    name : str
        The leg name.
    small : str
        The smaller size's cell.
    large : str
        The larger size's cell.
    kind : str
        ``tolerance`` or ``below small``.
    column : str
        The summary column that publishes each cell's bound.
    tolerance : float or None
        The tolerance of the ``tolerance`` kind.
    applies : Callable
        The rows the clause is part of.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S14 clause.
    """
    if kind not in {"tolerance", "below small"}:
        raise ValueError(f"unknown bias bound kind {kind!r}")
    if (kind == "tolerance") != (tolerance is not None):
        raise ValueError("a tolerance clause needs a tolerance, and only it takes one")

    def judge(large_bound: float, small_bound: float) -> Judged:
        limit = small_bound if tolerance is None else float(tolerance)
        return Judged(limit - large_bound, bool(large_bound < limit))

    def bound_of(summary: pd.DataFrame, family: str, cell: str) -> float:
        return float(
            summary.loc[(summary["property"] == family) & (summary["cell"] == cell), column].iloc[0]
        )

    def read(ctx: Context, row: pd.Series) -> Judged:
        family = str(row["property"])
        return judge(bound_of(ctx.summary, family, large), bound_of(ctx.summary, family, small))

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        family = str(row["property"])
        return [(family, small), (family, large)] if kind != "tolerance" else [(family, large)]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family = str(row["property"])
        large_errors = _errors(ctx, ctx.cell(family, large))
        k_large = ctx.lost_in(family, large)
        worst_large = max(
            (error_bias_bound(vertex(large_errors, k_large, j)), j) for j in range(k_large + 1)
        )
        if kind == "tolerance":
            small_bound, small_label = math.nan, ""
        else:
            small_errors = _errors(ctx, ctx.cell(family, small))
            small_bound, centre = _equal_fill_bound(small_errors, ctx.lost_in(family, small))
            small_label = f"; small at equal fill {centre:.6g}"
        return [
            (
                f"large at j={worst_large[1]} at min{small_label}",
                judge(worst_large[0], small_bound),
            )
        ]

    return Clause(name, "S14", applies, read, candidates, target=target, reads=reads)


def _equal_fill_t(values: np.ndarray, k: int, *, largest: bool) -> float:
    """The equal fill ``c`` in the survivors' range that makes ``mean / sd`` largest or least.

    ``t`` is quasi-concave where it is non-negative and quasi-convex where it is negative, so
    with symmetry its extremes over the box sit at equal fills or at vertex splits (section
    2).  The search over ``c`` reads a closed form on a fine grid, then refines the best cell.
    """
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        return low
    total, squares, count = float(np.sum(values)), float(np.sum(values**2)), len(values) + k

    def t_of(c: np.ndarray) -> np.ndarray:
        mean = (total + k * c) / count
        variance = (squares + k * c * c - count * mean * mean) / (count - 1)
        return mean / np.sqrt(np.maximum(variance, np.finfo(float).tiny))

    sign = 1.0 if largest else -1.0
    grid = np.linspace(low, high, 4097)
    scores = sign * t_of(grid)
    best = int(np.argmax(scores))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    found = minimize_scalar(
        lambda c: -sign * float(t_of(np.array([c]))[0]),
        bounds=(float(left), float(right)),
        method="bounded",
        options={"xatol": 1e-12 * max(1.0, high - low)},
    )
    centre = float(found.x)
    return (
        centre if sign * float(t_of(np.array([centre]))[0]) >= scores[best] else float(grid[best])
    )


def growth_options(
    values: np.ndarray, k: int, *, equal_fill: bool = True
) -> list[tuple[str, np.ndarray]]:
    """The S15 candidate fills of one rung that lost ``k`` replicates (H1).

    Every vertex split, the equal fill that makes ``mean / sd`` largest and the one that makes
    it least, each by a one-dimensional search in the survivors' range, and the fill nearest a
    zero mean when the attainable mean interval holds zero.  ``equal_fill=False`` is the
    deliberate mutation of p30: without the equal fills the set is not sound.

    Parameters
    ----------
    values : numpy.ndarray
        The rung's surviving errors.
    k : int
        The rung's lost replicates.
    equal_fill : bool
        Whether to include the two equal-fill extremes.

    Returns
    -------
    list of (str, numpy.ndarray)
        Each fill's label and its completed errors.
    """
    if k == 0:
        return [("survivors", values)]
    out = [(f"j={j} at min", vertex(values, k, j)) for j in range(k + 1)]
    if equal_fill:
        for largest, word in ((True, "largest"), (False, "least")):
            centre = _equal_fill_t(values, k, largest=largest)
            out.append((f"equal fill {centre:.6g}, t {word}", filled(values, k, centre)))
    centre = -float(np.sum(values)) / k
    if float(np.min(values)) <= centre <= float(np.max(values)):
        out.append(("nearest a zero mean", filled(values, k, centre)))
    return out


def growth_clause(
    name: str,
    *,
    rungs: Callable[[Context, pd.Series], tuple[tuple[str, int], tuple[str, int]]],
    sign: Callable[[Context, pd.Series], float],
    seed: Callable[[Context, pd.Series, int], int],
    draws: Callable[[Context, np.ndarray, int], np.ndarray],
    judge: Judge,
    applies: Applies,
    columns: tuple[str, str] = ("growth_ci_lower", "growth_ci_upper"),
    equal_fill: bool = True,
    target: str = "passed",
) -> Clause:
    """S15: a signed growth of standardized bias between two rungs (H1).

    ``G = s (t_large - t_small)`` with ``t = mean / sd`` of a rung's errors.  Each lost rung
    takes :func:`growth_options`; each option's draws come from the rung's own stream
    (``draws(ctx, values, seed)``), so options share common random numbers.  :func:`prune` drops
    the dominated options in each direction, and the product runs over the rest.  The point
    is exact; the bootstrap interval is enumerated.

    Parameters
    ----------
    name : str
        The leg name.
    rungs : Callable
        The ``(cell, size)`` of the smallest and the largest rung.
    sign : Callable
        The declared sign ``s``.
    seed : Callable
        The verdict's own seed of one rung, by size.
    draws : Callable
        The verdict's own per-rung bootstrap draws of ``t``.
    judge : Callable
        The rule on the growth interval.
    applies : Callable
        The rows the clause is part of.
    columns : tuple of str
        The summary columns of the published interval.
    equal_fill : bool
        Whether the candidate set includes the equal fills (the p30 mutation turns it off).
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The S15 clause.
    """

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judge(ctx, row, Interval(float(row[columns[0]]), float(row[columns[1]])))

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        family = str(row["property"])
        return [(family, cell) for cell, _ in rungs(ctx, row)]

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        family = str(row["property"])
        s = float(sign(ctx, row))
        per_rung: list[list[Option]] = []
        for cell, size in rungs(ctx, row):
            values = _errors(ctx, ctx.cell(family, cell))
            k = ctx.lost_in(family, cell)
            per_rung.append(
                [
                    Option(
                        label if not k else f"{cell} {label}",
                        draws(ctx, option, seed(ctx, row, size)),
                        standardized_bias(option),
                    )
                    for label, option in growth_options(values, k, equal_fill=equal_fill)
                ]
            )
        out = []
        for small, large in _combinations(per_rung, [-s, s]):
            interval = percentile_interval(
                s * (large.draws - small.draws), confidence_level=ctx.margins.confidence_level
            )
            labels = [option.label for option in (small, large) if option.label != "survivors"]
            out.append(("; ".join(labels) or "survivors", judge(ctx, row, interval)))
        return out

    return Clause(name, "S15", applies, read, candidates, target=target, reads=reads)


def unaccountable_clause(
    name: str,
    *,
    statistic: str,
    read: Callable[[Context, pd.Series], Judged],
    applies: Applies,
    reads: Reads | None = None,
    target: str = "passed",
) -> Clause:
    """A clause with a stated reason to have no candidate set (rev 3 section 2, R3-1).

    A sup-norm identity with a fixed tolerance, or the fit's own numerical success.  The read
    reproduces the verdict for the differential test.  A row of it that lost a replicate reads
    ``failures not accountable`` at any bound.

    Parameters
    ----------
    name : str
        The leg name.
    statistic : str
        What kind of claim it is, as the declared-rules table names it.
    read : Callable
        The complete-case verdict, read off the summary row.
    applies : Callable
        The rows the clause is part of.
    reads : Callable or None
        Other cells the verdict reads.
    target : str
        The verdict the clause feeds.

    Returns
    -------
    Clause
        The clause, marked not accountable.
    """
    return Clause(
        name, statistic, applies, read, None, target=target, accountable=False, reads=reads
    )


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


def _worst(clause: Clause, ctx: Context, row: pd.Series) -> tuple[Leg, bool]:
    """A clause's leg: its complete case, and its least favourable candidate.

    The leg's ``imputation`` names its label (D3'e) and the candidate.
    """
    assert clause.candidates is not None
    complete = clause.read(ctx, row)
    candidates = clause.candidates(ctx, row)
    if not candidates:
        raise ValueError(f"{clause.name}: no candidate for {row['property']}/{row['cell']}")
    failing = [item for item in candidates if not item[1].passed]
    label, worst = min(failing or candidates, key=lambda item: item[1].slack)
    passed = complete.passed and not failing
    return (
        Leg(clause.name, complete.slack, worst.slack, f"{clause.imputation}: {label}", passed),
        complete.passed,
    )


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


def _unique(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(pairs))


def read_cells(ctx: Context, rule: FamilyRule | None, row: pd.Series) -> list[tuple[str, str]]:
    """Every replication cell a summary row's verdict reads under its rule.

    The row's own :func:`sources`, then each applying clause's further reads, such as the
    other arm of a pair or another row's endpoint.

    Parameters
    ----------
    ctx : Context
        The accounting context.
    rule : FamilyRule or None
        The row's rule.
    row : pandas.Series
        The summary row.

    Returns
    -------
    list of tuple
        The ``(property, cell)`` pairs, in order and without repeats.
    """
    own = (rule.sources or sources) if rule is not None else sources
    out = list(own(row, ctx.rows))
    if rule is not None:
        for clause in rule.clauses:
            if clause.reads is not None and clause.applies(ctx, row):
                out.extend(clause.reads(ctx, row))
    return _unique(out)


@dataclass(frozen=True)
class _Accounted:
    """One summary row to account: its rule, its own cells, and every cell it reads."""

    row: pd.Series
    rule: FamilyRule | None
    own: list[tuple[str, str]]
    cells: list[tuple[tuple[str, str], int, int]]


def _account_row(ctx: Context, item: _Accounted, bound: float) -> list[dict[str, Any]]:
    row, rule = item.row, item.rule
    failed = ctx.k(row)
    own = [(requested, lost) for key, requested, lost in item.cells if key in item.own]
    requested = sum(count for count, _ in own) if own else failed
    if math.isfinite(float(row.get("replicates", math.nan))):
        requested = int(row["replicates"]) + failed
    lost_read = sum(lost for _, _, lost in item.cells)
    allowance = sum(allowed(bound, count) for _, count, _ in item.cells)
    exceeded = any(lost > allowed(bound, count) for _, count, lost in item.cells)
    bound_leg = Leg(BOUND_LEG, float(lost_read), float(allowance), "", not exceeded)
    applying = [] if rule is None else [c for c in rule.clauses if c.applies(ctx, row)]
    unaccountable = [clause.name for clause in applying if not clause.accountable]
    if rule is None or not applying or unaccountable:
        named = "; ".join(unaccountable) if unaccountable else "no declared clause"
        # Each applying clause still reads its complete case, so a reader can tell a cell
        # that its complete case already fails from one that the accounting alone turned red.
        complete = [(clause, clause.read(ctx, row)) for clause in applying]
        legs = [
            (bound_leg, True),
            (Leg(NOT_ACCOUNTABLE_LEG, math.nan, math.nan, named, False), True),
            *(
                (Leg(clause.name, value.slack, math.nan, COMPLETE_CASE_ONLY, value.passed), True)
                for clause, value in complete
            ),
        ]
        gate = (False, BOUND_EXCEEDED if exceeded else NOT_ACCOUNTABLE)
    else:
        clause_legs = [_worst(clause, ctx, row) for clause in applying]
        legs = [(bound_leg, True), *clause_legs]
        gate = _gate(exceeded, clause_legs)
    shared = {
        "property": str(row["property"]),
        "cell": str(row["cell"]),
        "role": str(row["role"]),
        "requested": requested,
        "failed": failed,
        "rate": failed / requested if requested else math.nan,
        "bound": bound,
        "allowed": allowance,
        "gate_passed": gate[0],
        "reason": gate[1],
    }
    return [
        {
            **shared,
            "leg": leg.name,
            "complete_case": leg.complete_case,
            "worst_case": leg.worst_case,
            "imputation": leg.imputation,
            "leg_passed": leg.passed,
        }
        for leg, _ in legs
    ]


def _account_payload(payload: tuple[Context, _Accounted, float]) -> list[dict[str, Any]]:
    ctx, item, bound = payload
    return _account_row(ctx, item, bound)


def _mid_rank(value: float, survivors: np.ndarray) -> float:
    """The mid-rank quantile of ``value`` among ``survivors``, or ``nan`` without one."""
    survivors = survivors[np.isfinite(survivors)]
    if not math.isfinite(value) or survivors.size == 0:
        return math.nan
    below = float(np.sum(survivors < value))
    tied = float(np.sum(survivors == value))
    return (below + 0.5 * tied) / float(survivors.size)


def witness_rows(
    lost: pd.DataFrame,
    witnesses: pd.DataFrame | None,
    family: str,
    cell: str,
) -> list[Leg]:
    """The reported witness legs of one cell's lost replicates (D3'f).

    Each lost replicate's statistic sits in ``complete_case`` and its mid-rank quantile among
    the cell's surviving replicates' in ``worst_case``.  The leg is reported, never gated, so it
    always passes.  A run that published no witness value reports ``nan``.

    Parameters
    ----------
    lost : pandas.DataFrame
        :func:`expand` of the records.
    witnesses : pandas.DataFrame or None
        The run's witness values, with
        :data:`~tests.studies.evidence.replication.WITNESS_COLUMNS`.
    family : str
        The property family.
    cell : str
        The cell.

    Returns
    -------
    list of Leg
        One leg per lost replicate, by replicate.
    """
    replicates = sorted(
        int(value)
        for value in lost.loc[(lost["property"] == family) & (lost["cell"] == cell), "replicate"]
    )
    if not replicates:
        return []
    values: dict[int, float] = {}
    name = "no witness values"
    if witnesses is not None and not witnesses.empty:
        own = witnesses.loc[(witnesses["property"] == family) & (witnesses["cell"] == cell)]
        values = {int(r): float(v) for r, v in zip(own["replicate"], own["value"], strict=True)}
        if not own.empty:
            name = str(own["statistic"].iloc[0])
    gone = set(replicates)
    survivors = np.array([v for r, v in values.items() if r not in gone], dtype=float)
    out = []
    for replicate in replicates:
        value = values.get(replicate, math.nan)
        out.append(
            Leg(
                WITNESS_LEG,
                value,
                _mid_rank(value, survivors),
                f"replicate {replicate}: {name}",
                True,
            )
        )
    return out


def module_groups(module: Any) -> Mapping[str, Sequence[Sequence[str]]]:
    """A module's ``PROPERTY_GROUPS`` declaration, or none.

    The declaration is optional: an undeclared ``pair`` or ``family`` group is the whole
    family, and a ``label`` rule without one raises in :func:`group_members`, so its absence
    cannot drop a conjunct silently.

    Parameters
    ----------
    module : module or None
        The property module.

    Returns
    -------
    Mapping
        Each family's declared groups of cells.
    """
    return getattr(module, "PROPERTY_GROUPS", None) or {}


def summary_rows(module: Any, rows: pd.DataFrame) -> pd.DataFrame:
    """The rows a module's ``summarize_properties`` summarizes, from the committed rows.

    A module whose summary reads its rows on another scale declares ``SUMMARY_ROWS``, the
    function it applies first.  The clauses' candidates must read that scale, or a ``k = 0``
    candidate would not reproduce the complete case (F18).  Without the declaration the rows
    come back unchanged.

    Parameters
    ----------
    module : module or None
        The property module.
    rows : pandas.DataFrame
        The stamped surviving replication rows.

    Returns
    -------
    pandas.DataFrame
        ``SUMMARY_ROWS(rows)``, or ``rows``.
    """
    transform = getattr(module, "SUMMARY_ROWS", None)
    return rows if transform is None else transform(rows)


def module_context(
    record: StudyRecord,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    lost: dict[tuple[str, str], int],
    module: Any,
) -> Context:
    """The :class:`Context` of a module: its :func:`summary_rows` and its ``PROPERTY_GROUPS``.

    Parameters
    ----------
    record : StudyRecord
        The study.
    rows : pandas.DataFrame
        The stamped surviving replication rows, as committed.
    summary : pandas.DataFrame
        The complete-case summary.
    lost : dict
        The replicates the records cost each ``(property, cell)``.
    module : module or None
        The property module.

    Returns
    -------
    Context
        The context every clause of the module reads.
    """
    return Context(record, summary_rows(module, rows), summary, lost, module_groups(module))


def exhausted(
    rows: pd.DataFrame, records: pd.DataFrame | None
) -> dict[tuple[str, str], tuple[int, int]]:
    """The cells that the records left with fewer than two surviving replicates.

    A spread needs two survivors, so the study's summary refuses such a cell
    (:func:`~tests.studies.evidence.properties.require_accounted`).  The cell lost at least
    ``R - 1`` of its ``R`` replicates, which every bound up to :data:`MAXIMUM_BOUND` exceeds
    for ``R >= 2``.  So its verdict is ``bound exceeded`` without a summary.

    Parameters
    ----------
    rows : pandas.DataFrame
        The stamped surviving replication rows.
    records : pandas.DataFrame or None
        The property failure records.

    Returns
    -------
    dict
        ``(requested, lost)`` for each such ``(property, cell)``.
    """
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for (family, cell), lost in counts(records).items():
        group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        if len(group) >= 2:
            continue
        requested = int(group["requested_replicates"].iloc[0]) if len(group) else lost
        if requested != len(group) + lost:
            raise ValueError(
                f"{family}/{cell} requested {requested} replicates, kept {len(group)} and lost "
                f"{lost}"
            )
        out[(family, cell)] = (requested, lost)
    return out


def set_aside(
    rows: pd.DataFrame,
    records: pd.DataFrame | None,
    families: Collection[str],
    *,
    bound: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    """Remove the families of exhausted cells from a run, and read each of their cells red.

    A family's summary can read every cell of it, so a family with an exhausted cell
    (:func:`exhausted`) is not summarized.  Each of its cells gets a summary row that is red
    on ``passed`` and ``property_passed``, and an accounting row whose gate is
    ``bound exceeded``.  The bound leg gives the family's lost replicates and its allowance,
    because each cell's verdict reads the family.

    Parameters
    ----------
    rows : pandas.DataFrame
        The stamped surviving replication rows.
    records : pandas.DataFrame or None
        The property failure records.
    families : Collection of str
        The families to set aside.
    bound : float
        The module's ``FAILURE_BOUND``.

    Returns
    -------
    tuple
        The kept rows, the kept records (each cell of a set-aside family removed from its
        ``scenarios``), the set-aside summary rows, and their accounting rows.
    """
    if not families:
        empty = pd.DataFrame(columns=list(PROPERTY_ACCOUNTING_COLUMNS))
        return rows, records, pd.DataFrame(), empty
    aside = rows["property"].astype(str).isin(families)
    kept_records = records
    if records is not None and not records.empty:
        scenarios = [
            ";".join(
                name
                for name in str(value).split(";")
                if name and name.split("/")[0] not in families
            )
            for value in records["scenarios"]
        ]
        kept_records = records.assign(scenarios=scenarios)
        kept_records = kept_records.loc[kept_records["scenarios"] != ""].reset_index(drop=True)
    lost = counts(records)
    cells: dict[tuple[str, str], tuple[str, int, int]] = {}
    for (family, cell), group in rows.loc[aside].groupby(["property", "cell"], sort=True):
        key = (str(family), str(cell))
        requested = int(group["requested_replicates"].iloc[0])
        cells[key] = (str(group["role"].iloc[0]), requested, lost.get(key, 0))
    for key, count in lost.items():
        if key[0] in families and key not in cells:
            cells[key] = ("positive", count, count)
    summary, accounting = [], []
    for family in sorted(families):
        members = {key: value for key, value in cells.items() if key[0] == family}
        read = sum(failed for _, _, failed in members.values())
        allowance = sum(allowed(bound, requested) for _, requested, _ in members.values())
        for (_, cell), (role, requested, failed) in sorted(members.items()):
            summary.append(
                {
                    "property": family,
                    "cell": cell,
                    "role": role,
                    "replicates": requested - failed,
                    "failed_replicates": failed,
                    "passed": False,
                    "property_passed": False,
                }
            )
            accounting.append(
                {
                    "property": family,
                    "cell": cell,
                    "role": role,
                    "requested": requested,
                    "failed": failed,
                    "rate": failed / requested if requested else math.nan,
                    "bound": bound,
                    "allowed": allowance,
                    "leg": BOUND_LEG,
                    "complete_case": float(read),
                    "worst_case": float(allowance),
                    "imputation": "",
                    "leg_passed": False,
                    "gate_passed": False,
                    "reason": BOUND_EXCEEDED,
                }
            )
    return (
        rows.loc[~aside].reset_index(drop=True),
        kept_records,
        pd.DataFrame(summary),
        pd.DataFrame(accounting, columns=list(PROPERTY_ACCOUNTING_COLUMNS)),
    )


def property_accounting(
    rows: pd.DataFrame,
    records: pd.DataFrame | None,
    summary: pd.DataFrame,
    *,
    record: StudyRecord,
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
    witnesses: pd.DataFrame | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """The accounting rows of every summary row whose verdict read a lost replicate.

    A row is red when a cell it reads lost more replicates than ``FAILURE_BOUND`` allows, or
    when a clause of its family's :class:`FamilyRule` that holds on the complete cases fails
    under a candidate of its statistic type.  The rule is the module's ``PROPERTY_RULES`` entry,
    else the shared one (:func:`rule_for`).  A row that a not-accountable clause reads, or whose
    family declares no rule, is red at any bound (R3-1).  A diagnostic row gets no accounting
    row.  After a row's legs come the witness legs of its own cells' lost replicates (D3'f).
    The accounting only turns rows red; :func:`correct_properties` applies it.

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
        The property module: its ``FAILURE_BOUND``, ``PROPERTY_RULES``, ``PROPERTY_GROUPS``
        and ``SUMMARY_ROWS``.  ``None`` means a bound of zero and the shared rules alone.
    rules : Sequence of FamilyRule or None
        The shared rules; :data:`tests.studies.evidence.property_verdicts.FAMILY_RULES` by
        default.
    witnesses : pandas.DataFrame or None
        The run's witness values (:data:`PROPERTY_WITNESSES`).
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
    bound = 0.0 if module is None else float(module.FAILURE_BOUND)
    if not 0.0 <= bound <= MAXIMUM_BOUND:
        raise ValueError(f"FAILURE_BOUND {bound} is outside [0, {MAXIMUM_BOUND}]")
    lost = counts(records)
    ctx = module_context(record, rows, summary, lost, module)
    requested = {
        (str(family), str(cell)): int(group["requested_replicates"].iloc[0])
        for (family, cell), group in rows.groupby(["property", "cell"], sort=True)
    }
    scored = summary.loc[summary["role"] != property_verdicts.DIAGNOSTIC_ROLE]
    items: list[_Accounted] = []
    for _, row in scored.iterrows():
        rule = rule_for(str(row["property"]), module, rules)
        if rule is not None:
            # A rule's group must resolve before its legs run: a label rule needs its
            # declaration, and correct_properties reads the same members.
            ctx.members(str(row["property"]), str(row["cell"]), rule.group)
        own = (rule.sources or sources) if rule is not None else sources
        own_cells = _unique(own(row, rows))
        read = read_cells(ctx, rule, row)
        if not any(lost.get(key, 0) for key in read) and ctx.k(row) == 0:
            continue
        cells = [(key, requested[key], lost.get(key, 0)) for key in read]
        if len(own_cells) == 1 and math.isfinite(float(row["failed_replicates"])):
            # A cell row's own verdict may read fewer replicates than the cell holds: a
            # contraction rung at its verdict budget publishes its own requested and lost.
            key = own_cells[0]
            cells = [
                (name, int(row["replicates"]) + ctx.k(row), ctx.k(row))
                if name == key
                else (name, count, gone)
                for name, count, gone in cells
            ]
        items.append(_Accounted(row, rule, own_cells, cells))
    entries = map_parallel(
        _account_payload, [((ctx, item, bound),) for item in items], n_jobs=n_jobs
    )
    expanded = expand(records)
    out: list[dict[str, Any]] = []
    for item, legs in zip(items, entries, strict=True):
        out.extend(legs)
        first = legs[0]
        for family, cell in item.own:
            for leg in witness_rows(expanded, witnesses, family, cell):
                out.append(
                    {
                        **first,
                        "leg": leg.name,
                        "complete_case": leg.complete_case,
                        "worst_case": leg.worst_case,
                        "imputation": leg.imputation,
                        "leg_passed": leg.passed,
                    }
                )
    return pd.DataFrame.from_records(out, columns=columns)


def correct_properties(
    summary: pd.DataFrame,
    accounting: pd.DataFrame,
    *,
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
) -> pd.DataFrame:
    """``properties.csv`` with each accounting-red row's verdicts set false.

    The rule is resolved as :func:`property_accounting` resolves it.  A red row sets its own
    ``passed`` false when a failing leg targets it, or when its bound was exceeded or its
    failures are not accountable.  Any red row sets ``property_passed`` false on every member
    of its group (:func:`group_members`), because a group's joint verdict needs every member
    and every joint clause.  A family with no rule has no declared group, so a red row of it
    sets the whole family's ``property_passed`` false.  A summary row that published no failure
    count, such as a growth row, takes the accounted count.  No column is added, so a summary
    with no failure comes back unchanged.

    Parameters
    ----------
    summary : pandas.DataFrame
        The complete-case summary.
    accounting : pandas.DataFrame
        Its :func:`property_accounting` rows.
    module : module or None
        The property module, for its ``PROPERTY_RULES`` and ``PROPERTY_GROUPS``.
    rules : Sequence of FamilyRule or None
        The shared rules; :data:`tests.studies.evidence.property_verdicts.FAMILY_RULES` by
        default.

    Returns
    -------
    pandas.DataFrame
        The corrected summary.
    """
    if accounting.empty:
        return summary
    groups = module_groups(module)
    out = summary.copy()
    for (family, cell), group in accounting.groupby(["property", "cell"], sort=False):
        family, cell = str(family), str(cell)
        mask = (out["property"] == family) & (out["cell"] == cell)
        if not mask.any():
            raise ValueError(f"the accounting names {family}/{cell}, which the summary lacks")
        failed = int(group["failed"].iloc[0])
        if failed and not np.isfinite(out.loc[mask, "failed_replicates"].astype(float)).all():
            out.loc[mask, "failed_replicates"] = failed
        if bool(group["gate_passed"].iloc[0]):
            continue
        rule = rule_for(family, module, rules)
        if rule is None:
            out.loc[mask, "passed"] = False
            out.loc[out["property"] == family, "property_passed"] = False
            continue
        members = group_members(summary, groups, family, cell, rule.group)
        joint = (out["property"] == family) & out["cell"].astype(str).isin(members)
        reason = str(group["reason"].iloc[0])
        legs = group.loc[group["leg"] != WITNESS_LEG]
        whole = reason == BOUND_EXCEEDED or (legs["leg"] == NOT_ACCOUNTABLE_LEG).any()
        failing = [
            str(name)
            for name in legs.loc[~legs["leg_passed"].astype(bool), "leg"]
            if name not in {BOUND_LEG, NOT_ACCOUNTABLE_LEG}
        ]
        if whole or any(rule.clause(name).target == "passed" for name in failing):
            out.loc[mask, "passed"] = False
        out.loc[joint, "property_passed"] = False
    return out


def rule_verdicts(rule: FamilyRule, summary: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """A rule's complete-case verdicts, read off a summary's columns alone.

    A row's ``passed`` is the conjunction of its applying ``passed`` clauses.  Its
    ``property_passed`` holds when every member of its group (:func:`group_members`) passes
    and every applying ``property_passed`` clause of every member holds.  A row no clause
    applies to is not a verdict row of the rule and is left out.  The reproduction and
    differential tests compare these with the study's own columns (F10).

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
        ``property``, ``cell``, ``passed`` and ``property_passed`` per verdict row.
    """
    out = []
    for family in sorted(rule.families):
        out.extend(_combined(rule, family, _judged(rule, family, summary, ctx), summary, ctx))
    return pd.DataFrame(out, columns=["property", "cell", "passed", "property_passed"])


def _judged(
    rule: FamilyRule, family: str, summary: pd.DataFrame, ctx: Context
) -> dict[str, list[tuple[Clause, Judged]]]:
    """Each verdict row's applying clauses of ``family`` and their complete-case values."""
    from tests.studies.evidence import property_verdicts

    rows = summary.loc[
        (summary["property"] == family) & (summary["role"] != property_verdicts.DIAGNOSTIC_ROLE)
    ]
    out: dict[str, list[tuple[Clause, Judged]]] = {}
    for _, row in rows.iterrows():
        judged = [
            (clause, clause.read(ctx, row)) for clause in rule.clauses if clause.applies(ctx, row)
        ]
        if judged:
            out[str(row["cell"])] = judged
    return out


def _combined(
    rule: FamilyRule,
    family: str,
    judged: Mapping[str, Sequence[tuple[Clause, Judged]]],
    summary: pd.DataFrame,
    ctx: Context,
    *,
    dropped: str | None = None,
) -> list[list[Any]]:
    """``[family, cell, passed, property_passed]`` per verdict row, without clause ``dropped``."""
    verdicts: dict[str, tuple[bool, bool]] = {}
    for cell, values in judged.items():
        kept = [(clause, value) for clause, value in values if clause.name != dropped]
        own = all(value.passed for clause, value in kept if clause.target == "passed")
        joint = all(value.passed for clause, value in kept if clause.target != "passed")
        verdicts[cell] = (own, own and joint)
    out = []
    for cell, (own, _) in verdicts.items():
        members = group_members(summary, ctx.groups, family, cell, rule.group)
        together = all(verdicts[name][1] for name in members if name in verdicts)
        out.append([family, cell, own, together])
    return out


def read_witnesses(path: Path) -> pd.DataFrame | None:
    """The witness values in ``path``, or ``None`` when the file does not exist.

    Parameters
    ----------
    path : pathlib.Path
        A ``property-failure-witnesses.parquet``.

    Returns
    -------
    pandas.DataFrame or None
        The values, with :data:`~tests.studies.evidence.replication.WITNESS_COLUMNS`.
    """
    if not path.exists():
        return None
    published = read_rows(path).astype(WITNESS_DTYPE)
    return published.loc[:, list(WITNESS_COLUMNS)]


#: The perturbations the differential test applies to committed rows (F10, rev 3 section 8).
#: Each moves the summary across thresholds a complete run never reaches, so a rule that drops
#: a conjunct or moves a threshold disagrees with the study's own verdict somewhere.
#: :func:`extra_perturbations` adds, per extra column and per cell, a scaling and a shift (L10).
PERTURBATIONS = ("none", "se_low", "se_high", "bias", "contract", "pair_collapse", "affine")

#: The standard-error factors of ``se/<f>``.
SE_FACTORS = (0.05, 0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 2.0, 20.0)
#: The shifts of ``shift/<z>``, in each cell's own spread.
SHIFT_SIZES = (0.15, 0.7, 1.0, 1.5, 3.0)
#: The coverage shares of ``coverage/<p>`` and the rejection shares of ``rejection/<p>``.
COVERAGE_TARGETS = (0.5, 0.8, 0.9, 0.925, 0.94, 0.96, 0.99)
REJECTION_TARGETS = (0.01, 0.04, 0.06, 0.1, 0.5, 0.8, 0.95)
#: ``two_point/<g>/<r>`` moves the spread by ``g`` and the SE ratio by ``r``.  Two-point errors
#: have the least sampling variance of a spread, so the intervals of an SE ratio and of an
#: efficiency ratio narrow, and their product can leave a band that each of them stays in.
TWO_POINT_STEPS = ((1.06, 1.05), (0.94, 0.95), (1.08, 1.03), (0.92, 0.97))
#: The spreads and SE ratios of :func:`targeted_perturbations`.  A per-module band sits at its
#: own place, so the grid steps by 3 per cent on each side of 1, out to 1.3 and its inverse.
TARGETED_SPREADS = tuple(
    float(f"{value:.4g}")
    for value in (*np.geomspace(1.03, 1.3, 9), *np.geomspace(0.97, 1 / 1.3, 9))
)
TARGETED_RATIOS = (1.0, *TARGETED_SPREADS)
#: The mean-error factors of ``centre/<role>/<c>`` and the SE factors of ``se_role/<role>/<f>``.
CENTRE_FACTORS = (0.9, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, 1.5, 3.0)
ROLE_SE_FACTORS = (0.5, 0.25, 0.8, 1.25, 2.0, 4.0)
ROLE_SPREADS = (2.0, 4.0, 10.0, 0.5)
#: The SE factor that takes a reported variance below a rounding-level floor.  A split variance
#: share near 1e-2 falls under a floor of 1e-8 once each standard error grows by 1e5.
ROUNDING_SE_FACTOR = 1.0e5
#: The amplitude of each ``mixed/<k>`` draw, and the seed root of the draws.
MIXED_AMPLITUDES = tuple(float(value) for value in np.geomspace(0.1, 1.0, 16))
MIXED_SEED = 20_261_015

#: The exponent of ``contract``: each cell's errors scale by ``(n / n_min) ** CONTRACT_POWER``
#: within its family, which pushes a contracting slope across zero.
CONTRACT_POWER = 1.5

#: The seed root of the ``affine`` perturbation's per-cell maps.
AFFINE_SEED = 20_261_014

#: The factor and the additive step of each extra column's per-cell perturbation.  The step is
#: in units of the cell's largest absolute value of the column, and at least one.
EXTRA_SCALE = 4.0
EXTRA_SHIFT = 4.0

#: The kinds of one extra column and one cell: two that raise the values and one that sets them
#: down until their mean is their smallest finite value, the floor of a count or a share.
EXTRA_KINDS = ("scale", "shift", "floor")


def cell_key(*labels: str) -> int:
    """A stable 64-bit integer of ``labels``, for a seed that names a cell.

    Parameters
    ----------
    *labels : str
        The names, such as a family and a cell.

    Returns
    -------
    int
        The first eight bytes of their BLAKE2b digest.
    """
    digest = hashlib.blake2b("\x1f".join(labels).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _retested(out: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """``covered`` and ``rejected`` recomputed from a Wald interval where the SE is usable."""
    from scipy.stats import norm

    estimates = out["estimate"].astype(float)
    std_errors = out["std_error"].astype(float)
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    usable = np.isfinite(std_errors) & (std_errors > 0.0)
    errors = (estimates - out["truth"].astype(float)).abs()
    out.loc[usable, "covered"] = (errors <= critical * std_errors)[usable].astype(int)
    out.loc[usable, "rejected"] = ((estimates / std_errors).abs() > critical)[usable].astype(int)
    return out


def _partner(cell: str, positives: Sequence[str]) -> str:
    """The positive cell whose name shares the longest ``__``-separated prefix with ``cell``."""

    def shared(other: str) -> int:
        count = 0
        for left, right in zip(cell.split("__"), other.split("__"), strict=False):
            if left != right:
                break
            count += 1
        return count

    return max(sorted(positives), key=shared)


def perturbed(rows: pd.DataFrame, kind: str, *, alpha: float = 0.05) -> pd.DataFrame:
    """Replication rows moved by one perturbation kind.

    | kind | change |
    | --- | --- |
    | ``se_low``, ``se_high`` | every standard error times 0.7 or 1.45 |
    | ``bias`` | each cell's estimates shifted by 0.35 of its own spread |
    | ``contract`` | each cell's errors times ``(n / n_min) ** CONTRACT_POWER`` in its family |
    | ``pair_collapse`` | each control cell's estimates and SEs set to its partner positive cell's, by replicate |
    | ``affine`` | each cell's errors times ``a`` plus ``b`` of its spread, ``a`` and ``b`` seeded by the cell's name |

    For these named kinds, ``covered`` and ``rejected`` are recomputed from a Wald interval at
    ``alpha`` where the standard error is finite and positive, so coverage and rejection move
    with the change.  :data:`SE_FACTORS` and the graded kinds say which of those recompute.  The
    rows are no study's output: the test reads only whether a rule and the study's own code
    agree on them.

    Parameters
    ----------
    rows : pandas.DataFrame
        The replication rows.
    kind : str
        One of :data:`PERTURBATIONS`, of :func:`binding_perturbations` or of
        :func:`extra_perturbations`.
    alpha : float
        The level of the recomputed interval and test.

    Returns
    -------
    pandas.DataFrame
        A perturbed copy.
    """
    if kind.startswith("extra|"):
        return _extra(rows, kind)
    if "+" in kind:
        out = rows
        for part in kind.split("+"):
            out = perturbed(out, part, alpha=alpha)
        return out
    if kind.partition("/")[0] in {"centre", "se_role", "coverage_role", "spread_role"}:
        return _role_kind(rows, kind, alpha)
    name, _, level = kind.partition("/")
    graded = name in {"se", "coverage"} and re.fullmatch(r"\d+(\.\d+)?(e-?\d+)?", level)
    if graded or kind in binding_perturbations() or kind in _targeted():
        return _graded(rows, kind, alpha)
    if kind not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {kind!r}")
    out = rows.copy()
    if kind == "none":
        return out
    estimates = out["estimate"].astype(float)
    truth = out["truth"].astype(float)
    if kind == "se_low":
        out["std_error"] = out["std_error"].astype(float) * 0.7
    elif kind == "se_high":
        out["std_error"] = out["std_error"].astype(float) * 1.45
    elif kind == "bias":
        spread = out.groupby(["property", "cell"])["estimate"].transform("std").fillna(0.0)
        out["estimate"] = estimates + 0.35 * spread
    elif kind == "contract":
        n = out["n"].astype(float)
        smallest = n.groupby(out["property"]).transform("min")
        out["estimate"] = truth + (estimates - truth) * (n / smallest) ** CONTRACT_POWER
    elif kind == "affine":
        spread = out.groupby(["property", "cell"])["estimate"].transform("std").fillna(0.0)
        scale = pd.Series(1.0, index=out.index)
        shift = pd.Series(0.0, index=out.index)
        for (family, cell), index in out.groupby(["property", "cell"]).groups.items():
            rng = np.random.default_rng([AFFINE_SEED, cell_key(str(family), str(cell))])
            scale.loc[index] = rng.uniform(0.6, 1.6)
            shift.loc[index] = rng.normal(0.0, 0.6)
        out["estimate"] = truth + scale * (estimates - truth) + shift * spread
    else:
        for family, selected in out.groupby("property", sort=False):
            positives = sorted(set(selected.loc[selected["role"] == "positive", "cell"]))
            controls = sorted(set(selected.loc[selected["role"] == "control", "cell"]))
            if not positives:
                continue
            for cell in controls:
                source = selected.loc[selected["cell"] == _partner(cell, positives)]
                by_replicate = source.set_index("replicate")
                target = out.index[(out["property"] == family) & (out["cell"] == cell)]
                replicates = out.loc[target, "replicate"]
                found = replicates.isin(by_replicate.index)
                chosen = target[found.to_numpy()]
                for column in ("estimate", "std_error"):
                    out.loc[chosen, column] = by_replicate.loc[replicates[found], column].to_numpy(
                        dtype=float
                    )
    return _retested(out, alpha)


def binding_perturbations() -> tuple[str, ...]:
    """The graded :func:`perturbed` kinds the differential adds to :data:`PERTURBATIONS`.

    The named perturbations move every cell one way, so two clauses that read the same change
    fail together and neither decides a verdict alone.  The clause-binding test needs, for
    each clause, a perturbation under which that clause alone decides a verdict (rev 3
    section 8).  These kinds move one quantity at a time, or move the cells of a family apart.

    | kind | change | ``covered``, ``rejected`` |
    | --- | --- | --- |
    | ``se/<f>`` | every standard error times ``f`` | kept |
    | ``shift/<z>`` | each cell's estimates shifted by ``z`` of its own spread | recomputed |
    | ``coverage/<p>`` | each cell's ``covered`` flipped on its first replicates to a share ``p`` | ``rejected`` kept |
    | ``rejection/<p>`` | each cell's ``rejected`` flipped on its first replicates to a share ``p`` | ``covered`` kept |
    | ``control_concentrated`` | positives centred on the truth; each control's spread times 0.05, its bias 0.6 of that spread | recomputed |
    | ``control_diffuse`` | positives centred on the truth; each control's spread times 10, its bias 0.1 of that spread | recomputed |
    | ``positive_diffuse`` | positives centred on the truth and their spread times 1,000 | recomputed |
    | ``dilate/even``, ``dilate/odd`` | positives centred on the truth; the cells at even or odd places of each family's sorted cells dilated 10 times about their mean | recomputed |
    | ``two_point/<g>/<r>`` | each cell's errors set to their mean plus or minus ``g`` times their spread, by side, and its standard errors times ``g r`` | kept |
    | ``mixed/<k>`` | each cell's own seeded shift, dilation and SE scale at amplitude ``MIXED_AMPLITUDES[k]`` | recomputed on even ``k``; flipped at random on odd ``k`` |

    Returns
    -------
    tuple of str
        Every ``se/``, ``coverage/``, ``rejection/`` and ``mixed/`` kind, and the two control
        reshapes.
    """
    return (
        *(f"se/{factor:g}" for factor in SE_FACTORS),
        *(f"shift/{size:g}" for size in SHIFT_SIZES),
        *(f"coverage/{share:g}" for share in COVERAGE_TARGETS),
        *(f"rejection/{share:g}" for share in REJECTION_TARGETS),
        "control_concentrated",
        "control_diffuse",
        "positive_diffuse",
        "dilate/even",
        "dilate/odd",
        *(f"two_point/{spread:g}/{ratio:g}" for spread, ratio in TWO_POINT_STEPS),
        *(f"mixed/{draw}" for draw in range(len(MIXED_AMPLITUDES))),
    )


def targeted_perturbations() -> tuple[str, ...]:
    """The ``two_point/<g>/<r>`` grid that the binding search tries per subject.

    A clause with a per-module threshold, such as an efficiency band, binds only where a
    perturbation reaches that module's own threshold.  The search runs this grid on a subject
    after :data:`PERTURBATIONS` and :func:`binding_perturbations` leave a clause of it unbound.
    ``g`` moves the empirical spread, ``g r`` the reported standard error, and ``r`` the SE
    ratio, so two of the three can sit inside their bands while the third leaves.  The kinds
    are ordered by their distance from no change, so the search finds the smallest witness.

    Returns
    -------
    tuple of str
        Every ``two_point/<g>/<r>`` with ``g`` in :data:`TARGETED_SPREADS` and ``r`` in
        :data:`TARGETED_RATIOS`.
    """
    return _targeted()


@functools.cache
def _targeted() -> tuple[str, ...]:
    grid = [(g, r) for g in TARGETED_SPREADS for r in TARGETED_RATIOS]
    grid.sort(key=lambda pair: (abs(np.log(pair[0])) + abs(np.log(pair[1])), pair))
    return tuple(f"two_point/{g:g}/{r:g}" for g, r in grid)


def composed_perturbations() -> tuple[str, ...]:
    """The role-restricted and composed kinds the binding search tries last.

    A clause can share its row with a clause that already fails, or with a clause that reads
    the same quantity.  It then decides a verdict alone only under a change that repairs the
    other clause, or that moves one role's cells and not the other's.

    | kind | change | ``covered``, ``rejected`` |
    | --- | --- | --- |
    | ``centre/<role>/<c>`` | each cell of the role keeps its spread, and its mean error times ``c`` | recomputed for the role |
    | ``se_role/<role>/<f>`` | each standard error of the role times ``f``. At 1e5 it puts a rounding-level variance floor out of reach | kept |
    | ``coverage_role/<role>/<p>`` | ``coverage/<p>`` on the role's cells alone | ``rejected`` kept |
    | ``spread_role/<role>/<g>`` | each cell of the role dilated ``g`` times about its mean, and its standard errors times ``g`` | recomputed for the role |
    | ``<a>+<b>`` | kind ``a``, then kind ``b`` | as each kind says |

    Returns
    -------
    tuple of str
        The coverage, centre and spread kinds of each role, then an SE scale of every cell and of each role, each
        followed by a coverage share, then ``se_role/<role>/<f>`` at :data:`ROUNDING_SE_FACTOR` for each role.
    """
    return _composed()


@functools.cache
def _composed() -> tuple[str, ...]:
    roles = ("control", "positive")
    return (
        *(f"coverage_role/{role}/{p:g}" for p in COVERAGE_TARGETS for role in roles),
        *(f"centre/{role}/{c:g}" for c in CENTRE_FACTORS for role in roles),
        *(f"spread_role/{role}/{g:g}" for g in ROLE_SPREADS for role in roles),
        *(
            f"se/{f:g}+coverage/{p:g}"
            for f in sorted(TARGETED_SPREADS, key=lambda value: abs(np.log(value)))
            for p in COVERAGE_TARGETS
        ),
        *(
            f"se_role/{role}/{f:g}+coverage/{p:g}"
            for f in ROLE_SE_FACTORS
            for role in roles
            for p in COVERAGE_TARGETS
        ),
        *(f"se_role/{role}/{ROUNDING_SE_FACTOR:g}" for role in roles),
    )


def _role_kind(rows: pd.DataFrame, kind: str, alpha: float) -> pd.DataFrame:
    """A ``centre``, ``spread_role``, ``se_role`` or ``coverage_role`` kind on a copy."""
    name, role, level = kind.split("/")
    out = rows.copy()
    chosen = (out["role"] == role).to_numpy()
    if name == "coverage_role":
        flipped = _graded(out, f"coverage/{level}", alpha)["covered"].to_numpy()
        values = out["covered"].to_numpy().copy()
        values[chosen] = flipped[chosen]
        out["covered"] = values
        return out
    if name == "se_role":
        scaled = out["std_error"].astype(float).to_numpy().copy()
        scaled[chosen] *= float(level)
        out["std_error"] = scaled
        return out
    errors = out["estimate"].astype(float) - out["truth"].astype(float)
    centre = errors.groupby([out["property"], out["cell"]]).transform("mean")
    if name == "spread_role":
        moved = (centre + float(level) * (errors - centre)).to_numpy()
        scaled = out["std_error"].astype(float).to_numpy().copy()
        scaled[chosen] *= float(level)
        out["std_error"] = scaled
    else:
        moved = (errors - centre + float(level) * centre).to_numpy()
    estimates = out["estimate"].astype(float).to_numpy().copy()
    estimates[chosen] = out["truth"].astype(float).to_numpy()[chosen] + moved[chosen]
    out["estimate"] = estimates
    retested = _retested(out.copy(), alpha)
    for column in ("covered", "rejected"):
        values = out[column].to_numpy().copy()
        values[chosen] = retested[column].to_numpy()[chosen]
        out[column] = values
    return out


def _graded(rows: pd.DataFrame, kind: str, alpha: float) -> pd.DataFrame:
    """One :func:`binding_perturbations` kind applied to a copy of ``rows``."""
    out = rows.copy()
    name, _, level = kind.partition("/")
    if name == "se":
        out["std_error"] = out["std_error"].astype(float) * float(level)
        return out
    if name == "shift":
        spread = out.groupby(["property", "cell"])["estimate"].transform("std").fillna(0.0)
        out["estimate"] = out["estimate"].astype(float) + float(level) * spread
        return _retested(out, alpha)
    if name in {"coverage", "rejection"}:
        column = "covered" if name == "coverage" else "rejected"
        values = out[column].to_numpy(dtype=float).copy()
        order = out.sort_values(["property", "cell", "replicate"]).index
        for _, index in out.loc[order].groupby(["property", "cell"], sort=True).indices.items():
            positions = out.index.get_indexer(order[index])
            finite = positions[np.isfinite(values[positions])]
            wanted = round(float(level) * finite.size)
            ones = finite[values[finite] == 1.0]
            zeros = finite[values[finite] == 0.0]
            if ones.size > wanted:
                values[ones[: ones.size - wanted]] = 0.0
            else:
                values[zeros[: wanted - ones.size]] = 1.0
        out[column] = values.astype(out[column].dtype)
        return out
    if name == "mixed":
        return _mixed(out, int(level), alpha)
    if name in {"positive_diffuse", "dilate", "two_point"}:
        return _dilated(out, name, level, alpha)
    if name not in {"control_concentrated", "control_diffuse"}:
        raise ValueError(f"unknown perturbation {kind!r}")
    spread_factor, bias = (0.05, 0.6) if name == "control_concentrated" else (10.0, 0.1)
    errors = out["estimate"].astype(float) - out["truth"].astype(float)
    keys = [out["property"], out["cell"]]
    centre = errors.groupby(keys).transform("mean")
    spread = errors.groupby(keys).transform("std").fillna(0.0)
    sign = np.where(centre < 0.0, -1.0, 1.0)
    control = (out["role"] == "control").to_numpy()
    positive = (out["role"] == "positive").to_numpy()
    moved = errors.to_numpy().copy()
    moved[positive] = (errors - centre).to_numpy()[positive]
    reshaped = spread_factor * (errors - centre) + sign * bias * spread_factor * spread
    moved[control] = reshaped.to_numpy()[control]
    out["estimate"] = out["truth"].astype(float) + moved
    return _retested(out, alpha)


def _dilated(out: pd.DataFrame, name: str, level: str, alpha: float) -> pd.DataFrame:
    """``positive_diffuse``, ``dilate/<even|odd>`` or ``two_point/<g>/<r>`` on a copy."""
    errors = out["estimate"].astype(float) - out["truth"].astype(float)
    keys = [out["property"], out["cell"]]
    centre = errors.groupby(keys).transform("mean")
    positive = out["role"] == "positive"
    if name == "two_point":
        spread, ratio = (float(value) for value in level.split("/"))
        side = np.where(errors < centre, -1.0, 1.0)
        width = errors.groupby(keys).transform("std").fillna(0.0)
        out["estimate"] = out["truth"].astype(float) + centre + spread * side * width
        out["std_error"] = out["std_error"].astype(float) * spread * ratio
        return out
    if name == "positive_diffuse":
        chosen = positive
    else:
        place = out.groupby("property")["cell"].transform(
            lambda cells: cells.map({cell: i for i, cell in enumerate(sorted(set(cells)))})
        )
        chosen = place % 2 == (0 if level == "even" else 1)
    kept = centre.where(~positive, 0.0)
    wide = 1_000.0 if name == "positive_diffuse" else 10.0
    factor = pd.Series(np.where(chosen, wide, 1.0), index=out.index)
    out["estimate"] = out["truth"].astype(float) + kept + factor * (errors - centre)
    return _retested(out, alpha)


def _mixed(out: pd.DataFrame, draw: int, alpha: float) -> pd.DataFrame:
    """One seeded mixture: each cell's own shift, dilation, SE scale and indicator flips.

    Each cell draws, from a stream of ``(MIXED_SEED, draw, cell)``, whether and how far to
    shift its estimates (in its own spread), to dilate them about their mean, and to scale its
    standard errors, at the draw's amplitude.  Even draws recompute ``covered`` and
    ``rejected`` from a Wald interval.  Odd draws keep them and flip a drawn share of each
    independently in some cells, so a coverage clause and a rejection or spread clause can
    fail apart.
    """
    amplitude = MIXED_AMPLITUDES[draw]
    estimates = out["estimate"].astype(float).to_numpy().copy()
    std_errors = out["std_error"].astype(float).to_numpy().copy()
    covered = out["covered"].to_numpy(dtype=float).copy()
    rejected = out["rejected"].to_numpy(dtype=float).copy()
    for (family, cell), index in out.groupby(["property", "cell"], sort=True).indices.items():
        rng = np.random.default_rng([MIXED_SEED, draw, cell_key(str(family), str(cell))])
        values = estimates[index]
        finite = values[np.isfinite(values)]
        centre = float(finite.mean()) if finite.size else 0.0
        spread = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        shift = rng.normal(0.0, amplitude) if rng.random() < 0.5 else 0.0
        dilation = math.exp(rng.normal(0.0, amplitude)) if rng.random() < 0.5 else 1.0
        scale = math.exp(rng.normal(0.0, amplitude)) if rng.random() < 0.5 else 1.0
        estimates[index] = centre + dilation * (values - centre) + shift * spread
        std_errors[index] = std_errors[index] * scale
        for indicator in (covered, rejected):
            if draw % 2 and rng.random() < 0.5:
                flip = rng.random(index.size) < rng.uniform(0.0, 0.15 * amplitude)
                chosen = index[flip & np.isfinite(indicator[index])]
                indicator[chosen] = 1.0 - indicator[chosen]
    out["estimate"] = estimates
    out["std_error"] = std_errors
    if draw % 2:
        out["covered"] = covered.astype(out["covered"].dtype)
        out["rejected"] = rejected.astype(out["rejected"].dtype)
        return out
    return _retested(out, alpha)


def _extra_columns(rows: pd.DataFrame) -> list[str]:
    from tests.studies.evidence.properties import REPLICATE_COLUMNS

    return [
        column
        for column in rows.columns
        if column not in REPLICATE_COLUMNS
        and pd.api.types.is_numeric_dtype(rows[column])
        and not pd.api.types.is_bool_dtype(rows[column])
    ]


def extra_perturbations(rows: pd.DataFrame) -> tuple[str, ...]:
    """Per numeric extra column and per cell, a scaling, a shift and a floor (rev 3 L10).

    An extra column is any numeric column beyond
    :data:`~tests.studies.evidence.properties.REPLICATE_COLUMNS`, such as the bootstrap counts
    a cap reads.  The scaling multiplies the cell's values by :data:`EXTRA_SCALE`.  A scaling
    alone cannot move a column that is zero on every row, so the shift adds
    :data:`EXTRA_SHIFT` times the cell's largest absolute value, at least one.  A column with
    no finite value in a cell gives that cell none.

    The scaling and the shift only raise a non-negative share or count, so neither can flip a
    floor clause such as a share of indicator rows at least one half.  The floor kind moves the
    cell's values down by the gap between their mean and their smallest finite value.  The cell
    mean then equals the observed minimum, which is zero for an indicator, so a share falls to
    its floor and no clause needs a threshold of the subject to flip.  The floor kind has the
    least side effect on other subjects.  It keeps the spread of the column, so a variance, a
    covariance or a ratio that another statistic reads stays finite.  It leaves a constant
    column unchanged, such as the bootstrap-failure counts that a cap reads when they are zero,
    and such as a convergence flag that is one on every row.  A zeroing or a collapse to the
    minimum would remove the spread, and a fractional lowering would need a factor per clause.
    A lost (non-finite) row stays lost.

    Parameters
    ----------
    rows : pandas.DataFrame
        The replication rows.

    Returns
    -------
    tuple of str
        One :func:`perturbed` kind per perturbation, ``extra|column|property|cell|scale``,
        ``...|shift`` or ``...|floor``.
    """
    out: list[str] = []
    for column in _extra_columns(rows):
        finite = rows[column].astype(float).map(math.isfinite)
        kept = rows.loc[finite, ["property", "cell"]].drop_duplicates()
        for family, cell in sorted(kept.itertuples(index=False, name=None)):
            out.extend(f"extra|{column}|{family}|{cell}|{how}" for how in EXTRA_KINDS)
    return tuple(out)


def _extra(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One :func:`extra_perturbations` kind applied to a copy of ``rows``."""
    _, column, family, cell, how = kind.split("|")
    if column not in _extra_columns(rows) or how not in EXTRA_KINDS:
        raise ValueError(f"unknown perturbation {kind!r}")
    out = rows.copy()
    out[column] = out[column].astype(float)
    index = (out["property"] == family) & (out["cell"] == cell)
    values = out.loc[index, column]
    if how == "scale":
        out.loc[index, column] = values * EXTRA_SCALE
    elif how == "floor":
        finite = values[np.isfinite(values)]
        out.loc[index, column] = values - (float(finite.mean()) - float(finite.min()))
    else:
        out.loc[index, column] = values + EXTRA_SHIFT * max(1.0, float(np.nanmax(np.abs(values))))
    return out


def declared_rules(
    families: Sequence[str],
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
) -> list[FamilyRule]:
    """The distinct rules that ``families`` resolve to, in first-use order.

    Parameters
    ----------
    families : Sequence of str
        Property families.
    module : module or None
        The property module, whose ``PROPERTY_RULES`` take precedence.
    rules : Sequence of FamilyRule or None
        The shared rules.

    Returns
    -------
    list of FamilyRule
        One entry per rule; an undeclared family adds none.
    """
    out: list[FamilyRule] = []
    for family in families:
        rule = rule_for(family, module, rules)
        if rule is not None and all(rule is not seen for seen in out):
            out.append(rule)
    return out


def differential(
    summary: pd.DataFrame,
    ctx: Context,
    *,
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
) -> pd.DataFrame:
    """The verdict rows where a declared rule and the study's own verdict disagree.

    Each family resolves to its rule as the accounting resolves it.  The rule reads only the
    family it is resolved for, so a shared rule another family of the module overrides is not
    held to that family.

    Parameters
    ----------
    summary : pandas.DataFrame
        A summary the study's own ``summarize_properties`` computed.
    ctx : Context
        The context the clauses read.
    module : module or None
        The property module.
    rules : Sequence of FamilyRule or None
        The shared rules.

    Returns
    -------
    pandas.DataFrame
        ``property``, ``cell``, the rule's ``passed`` and ``property_passed`` and the
        study's, one row per disagreement.
    """
    columns = ["property", "cell", "passed", "property_passed"]
    frames = []
    for family in sorted(set(summary["property"].astype(str))):
        rule = rule_for(family, module, rules)
        if rule is None:
            continue
        verdicts = rule_verdicts(rule, summary, ctx)
        frames.append(verdicts.loc[verdicts["property"] == family])
    if not frames:
        return pd.DataFrame(columns=[*columns, "study_passed", "study_property_passed"])
    verdicts = pd.concat(frames, ignore_index=True)
    merged = verdicts.merge(
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


def clause_binding(
    summary: pd.DataFrame,
    ctx: Context,
    *,
    module: Any = None,
    rules: Sequence[FamilyRule] | None = None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """The ``(family, clause)`` pairs that apply on a summary, and those that decide a verdict.

    A clause is decisive when the rule without it gives some verdict row a different
    ``passed`` or ``property_passed``: a rule that dropped it would then disagree with the
    study somewhere, so the differential test binds it.  The clause-binding test requires
    each applying clause to be decisive under some perturbation (rev 3 section 8).

    Parameters
    ----------
    summary : pandas.DataFrame
        A summary the study's own ``summarize_properties`` computed.
    ctx : Context
        The context the clauses read.
    module : module or None
        The property module.
    rules : Sequence of FamilyRule or None
        The shared rules.

    Returns
    -------
    tuple of set
        The applying pairs and the decisive pairs.
    """
    used: set[tuple[str, str]] = set()
    decisive: set[tuple[str, str]] = set()
    for family in sorted(set(summary["property"].astype(str))):
        rule = rule_for(family, module, rules)
        if rule is None:
            continue
        judged = _judged(rule, family, summary, ctx)
        names = {clause.name for values in judged.values() for clause, _ in values}
        used.update((family, name) for name in names)
        full = _combined(rule, family, judged, summary, ctx)
        for name in sorted(names):
            if _combined(rule, family, judged, summary, ctx, dropped=name) != full:
                decisive.add((family, name))
    return used, decisive


def unbound(used: set[tuple[str, str]], decisive: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """The applying ``(family, clause)`` pairs that no perturbation made decisive.

    Parameters
    ----------
    used : set of tuple
        The union of :func:`clause_binding`'s applying pairs over every perturbation.
    decisive : set of tuple
        The union of its decisive pairs.

    Returns
    -------
    list of tuple
        The unbound pairs, sorted.  The clause-binding test fails on any.
    """
    return sorted(used - decisive)


# ---- the verdict-writer check (rev 3 section 8, L9) ------------------------------------------

#: The summary columns a verdict writes.
VERDICT_COLUMNS = frozenset({"passed", "property_passed"})

#: The package whose functions are the shared verdicts, which declare their rules themselves.
SHARED_PACKAGE = "tests.studies.evidence"


def verdict_writes(source: str) -> list[int]:
    """The lines of ``source`` that assign a verdict column.

    A write is an assignment to a subscript that names ``passed`` or ``property_passed``,
    such as ``summary.loc[mask, "passed"] = ...``, or a call with one of them as a keyword,
    such as ``frame.assign(passed=...)``.  The check cannot tell which family a dynamic mask
    writes (L9), so it reports the write and nothing more.

    Parameters
    ----------
    source : str
        A module's source.

    Returns
    -------
    list of int
        The line numbers, sorted.
    """
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Subscript) and any(
                    isinstance(leaf, ast.Constant) and leaf.value in VERDICT_COLUMNS
                    for leaf in ast.walk(sub.slice)
                ):
                    lines.add(target.lineno)
        if isinstance(node, ast.Call) and any(
            keyword.arg in VERDICT_COLUMNS for keyword in node.keywords
        ):
            lines.add(node.lineno)
    return sorted(lines)


def _imported(source: str) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
            out.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
    return [name for name in out if name.startswith("tests.studies.")]


def _module_source(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        # ``from package.module import name`` lists ``package.module.name``, which is an
        # attribute rather than a module.
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin).read_text()


def verdict_writers(
    name: str, *, source: Callable[[str], str | None] = _module_source
) -> list[str]:
    """The modules of ``name``'s ``tests.studies`` imports, itself included, that write a verdict.

    The shared verdict functions of :data:`SHARED_PACKAGE` are left out: their families
    resolve to the shared rules.  A helper that writes a verdict for its importer is the
    importer's write, so the importer has to declare the rule.

    Parameters
    ----------
    name : str
        A property module's import name.
    source : Callable
        The source of a module name, or ``None`` when it is no Python module.

    Returns
    -------
    list of str
        The writing modules, sorted.
    """
    seen: set[str] = set()
    stack = [name]
    out: list[str] = []
    while stack:
        current = stack.pop()
        if current in seen or current.startswith(SHARED_PACKAGE):
            continue
        seen.add(current)
        text = source(current)
        if text is None:
            continue
        if verdict_writes(text):
            out.append(current)
        stack.extend(_imported(text))
    return sorted(out)


def undeclared_verdict_writers(
    module: Any, *, source: Callable[[str], str | None] = _module_source
) -> list[str]:
    """The verdict writers of a module that declares no ``PROPERTY_RULES`` (rev 3 section 8).

    Parameters
    ----------
    module : module
        A property module.
    source : Callable
        As for :func:`verdict_writers`.

    Returns
    -------
    list of str
        Empty when the module declares its rules or nothing it imports writes a verdict.
    """
    if hasattr(module, "PROPERTY_RULES"):
        return []
    return verdict_writers(module.__name__, source=source)
