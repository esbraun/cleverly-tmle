"""What the committed history already separates, read with ``git show`` and pandas alone.

This module fits no estimator.  It compares each replication row that two commits both
publish, and it reproduces two tables in RM18 of ``docs/roadmap.md``:

* "What the committed history already separates".  Three point-treatment studies changed
  runtime between an older commit and :data:`AFTER`, and nothing else that feeds their rows
  changed.  The DR-TMLE contraction slope is also refitted on the first 800 replications of
  every rung, which is the budget both commits share.
* The single-fold control table in "What the pooled update found".  The in-sample control of
  four cross-fitted longitudinal studies has no outer split, so the pooled update does not
  reach its code path, and any row it moved moved with the runtime.

It reads git history, so it runs in a full clone only.  CI checks out a shallow clone.

    python -m tests.diagnostics.rm18_runtime.row_drift
"""

from __future__ import annotations

import argparse
import io
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent

#: The commit every comparison reads as the newer side.  It is the merge that carries the
#: pooled update, and its ``tests/canonical`` is identical to the declaration commit's.
AFTER = "0e03a15"

#: The commit whose parent holds the longitudinal rows the pooled regeneration replaced.
POOLED_REGENERATION = "656674c"

PRIMARY_KEYS = ("implementation", "scenario", "replicate", "n", "estimand")
PROPERTY_KEYS = ("property", "cell", "role", "replicate")

#: The threshold the history table counts a moved row against.
MOVED = 1e-6

#: The first replications of each contraction rung that both DR-TMLE commits publish.
SHARED_RUNG_BUDGET = 800

LADDER = "double_robust_contraction"

Subset = Callable[[pd.DataFrame], "pd.Series[bool]"]


@dataclass(frozen=True)
class Comparison:
    """One artefact of one study, read at two commits and restricted to one row subset."""

    section: str
    study: str
    artifact: str
    older: str
    keys: Sequence[str]
    subset: str = "all"
    where: Subset | None = None
    newer: str = AFTER


def _is(column: str, value: str) -> Subset:
    return lambda frame: frame[column] == value


def _is_not(column: str, value: str) -> Subset:
    return lambda frame: frame[column] != value


def _in_sample_control(frame: pd.DataFrame) -> pd.Series[bool]:
    return frame["cell"].str.contains("in_sample_control", regex=False)


HISTORY = "history"
POOLED_CONTROL = "pooled_in_sample_control"

COMPARISONS = (
    Comparison(HISTORY, "multi_arm_drtmle", "replicates.csv.gz", "6933968", PRIMARY_KEYS),
    Comparison(HISTORY, "multi_arm_drtmle", "property-replicates.csv.gz", "6933968", PROPERTY_KEYS),
    Comparison(HISTORY, "multi_arm_ctmle_selector", "replicates.csv.gz", "2049349", PRIMARY_KEYS),
    Comparison(
        HISTORY,
        "multi_arm_ctmle_selector",
        "property-replicates.csv.gz",
        "2049349",
        PROPERTY_KEYS,
    ),
    Comparison(
        HISTORY,
        "drtmle",
        "replicates.csv.gz",
        "99d238c",
        PRIMARY_KEYS,
        "implementation == cleverly",
        _is("implementation", "cleverly"),
    ),
    Comparison(
        HISTORY,
        "drtmle",
        "replicates.csv.gz",
        "99d238c",
        PRIMARY_KEYS,
        "implementation == drtmle-r",
        _is("implementation", "drtmle-r"),
    ),
    Comparison(HISTORY, "drtmle", "property-replicates.csv.gz", "99d238c", PROPERTY_KEYS),
    Comparison(
        HISTORY,
        "drtmle",
        "property-replicates.csv.gz",
        "99d238c",
        PROPERTY_KEYS,
        f"property == {LADDER}",
        _is("property", LADDER),
    ),
    Comparison(
        HISTORY,
        "drtmle",
        "property-replicates.csv.gz",
        "99d238c",
        PROPERTY_KEYS,
        f"property != {LADDER}",
        _is_not("property", LADDER),
    ),
    *(
        Comparison(
            POOLED_CONTROL,
            study,
            "property-replicates.csv.gz",
            f"{POOLED_REGENERATION}^",
            PROPERTY_KEYS,
            "cell contains in_sample_control",
            _in_sample_control,
        )
        for study in (
            "lmtp_ltmle",
            "lmtp_ltmle_survival",
            "lmtp_ltmle_competing_crossfit",
            "categorical_ltmle_crossfit",
        )
    ),
)


def committed(revision: str, path: str) -> pd.DataFrame:
    """One committed table at one revision, read from the object store and not the tree."""
    raw = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    compression = "gzip" if path.endswith(".gz") else None
    return pd.read_csv(io.BytesIO(raw), compression=compression)


def drift(
    older: pd.DataFrame,
    newer: pd.DataFrame,
    keys: Sequence[str],
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """How far the rows both frames publish moved, and how many rows only one publishes.

    A row "differs" when its estimate or its standard error is not bit-identical, or is
    missing on one side only.  A row "moved" when its estimate changed by more than
    :data:`MOVED`.  With a ``tolerance``, a row is also counted when its estimate or its
    standard error changed by more than that, or is missing on one side only.
    """
    merged = older.merge(
        newer, on=list(keys), suffixes=("_older", "_newer"), how="outer", indicator=True
    )
    shared = merged.loc[merged["_merge"] == "both"]
    estimate = (shared["estimate_older"] - shared["estimate_newer"]).abs()
    std_error = (shared["std_error_older"] - shared["std_error_newer"]).abs()
    missing = (shared["estimate_older"].isna() != shared["estimate_newer"].isna()) | (
        shared["std_error_older"].isna() != shared["std_error_newer"].isna()
    )
    tolerated = (
        {}
        if tolerance is None
        else {
            "rows_outside_tolerance": int(
                ((estimate > tolerance) | (std_error > tolerance) | missing).sum()
            )
        }
    )
    return {
        "rows_compared": len(shared),
        "rows_only_older": int((merged["_merge"] == "left_only").sum()),
        "rows_only_newer": int((merged["_merge"] == "right_only").sum()),
        "max_abs_estimate_change": float(estimate.max()) if len(shared) else float("nan"),
        "max_abs_std_error_change": float(std_error.max()) if len(shared) else float("nan"),
        "rows_differing": int(((estimate > 0.0) | (std_error > 0.0) | missing).sum()),
        "rows_moved_over_1e-6": int((estimate > MOVED).sum()),
        "covered_flags_changed": int((shared["covered_older"] != shared["covered_newer"]).sum()),
        **tolerated,
    }


def _records(comparison: Comparison) -> list[dict[str, Any]]:
    path = f"tests/canonical/{comparison.study}/{comparison.artifact}"
    older, newer = committed(comparison.older, path), committed(comparison.newer, path)
    if comparison.where is not None:
        older = older.loc[comparison.where(older)]
        newer = newer.loc[comparison.where(newer)]
    identity = {
        "section": comparison.section,
        "study": comparison.study,
        "artifact": comparison.artifact,
        "subset": comparison.subset,
        "older": comparison.older,
        "newer": comparison.newer,
    }
    return [
        {**identity, "statistic": name, "value": value}
        for name, value in drift(older, newer, comparison.keys).items()
    ]


def _contraction_records() -> list[dict[str, Any]]:
    """The DR-TMLE slope rule refitted where both commits publish the same rung budget."""
    from tests.studies.canonical_drtmle import STUDY
    from tests.studies.evidence.property_verdicts import contraction_rates

    path = "tests/canonical/drtmle/property-replicates.csv.gz"
    columns = committed(AFTER, "tests/canonical/drtmle/properties.csv").columns
    older = committed("99d238c", path)
    newer = committed(AFTER, path)
    readings = (
        ("99d238c", f"{LADDER}, as committed", older),
        (
            AFTER,
            f"{LADDER}, replicate < {SHARED_RUNG_BUDGET}",
            newer.loc[newer["replicate"] < SHARED_RUNG_BUDGET],
        ),
        (AFTER, f"{LADDER}, as committed", newer),
    )
    records = []
    for revision, subset, rows in readings:
        for rate in contraction_rates(rows, STUDY, columns):
            for statistic in ("slope", "slope_ci_lower", "slope_ci_upper"):
                records.append(
                    {
                        "section": "contraction_slope",
                        "study": "drtmle",
                        "artifact": "property-replicates.csv.gz",
                        "subset": f"{subset}, cell {rate['cell']}",
                        "older": revision,
                        "newer": revision,
                        "statistic": statistic,
                        "value": float(rate[statistic]),
                    }
                )
    return records


def row_drift() -> pd.DataFrame:
    """Every comparison as long-format rows, one statistic per row."""
    records = [record for comparison in COMPARISONS for record in _records(comparison)]
    return pd.DataFrame([*records, *_contraction_records()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=HERE / "row-drift.csv")
    arguments = parser.parse_args()
    frame = row_drift()
    write_csv(frame, arguments.output)
    wide = frame.pivot_table(
        index=["section", "study", "artifact", "subset", "older"],
        columns="statistic",
        values="value",
        aggfunc="first",
        sort=False,
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(wide.to_string())
    print(f"wrote {len(frame)} rows to {arguments.output}")


if __name__ == "__main__":
    main()
