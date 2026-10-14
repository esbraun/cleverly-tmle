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

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from tests.studies.evidence.failures import FAILURE_COLUMNS
from tests.studies.evidence.replication import Recording

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
