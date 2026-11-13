"""The interval audit of bounded means: committed rows read under the shipped logit interval.

The package forms the interval of an unclustered mean of a yes/no outcome on the logit scale,
and withholds it when an arm the mean reads holds one outcome value (``constant_arm_cell_plugin``)
or when the mean sits on or outside its range (``out_of_range_plugin``).  A contrast that reads
such a mean inherits the withholding.  The committed rows of the registered studies were written
under the identity interval.

The logit interval and its test are deterministic functions of ``estimate``, ``std_error`` and
the null, so each bounded-mean row is recomputed here with no refit.  The constant-cell trigger
reads the data, and each study's draws are deterministic in their seeds, so :data:`AUDITS` names a
scan that regenerates every committed replicate's sample, fits nothing, and lists the rows whose
estimate reads a constant arm cell.  Each audited study commits ``interval-audit.parquet`` beside
its rows, in the Parquet row format of its other row files: one record per bounded-mean row and per withheld row, with the committed and shipped
``covered`` and ``rejected``.  ``tests/unit/test_interval_audit.py`` recomputes every shipped
column from the record and every committed column from the study's row files.

A study with audited property cells also commits ``interval-audit-properties.csv``: its own
property summary of those cells under the shipped interval.  The committed ``properties.csv`` and
``performance-tests.csv`` stay as the record of the identity interval, and the red-cell ledger
reads a cell that passes there and fails under the shipped interval from this file.  Write or refresh an audit file with::

    python -m tests.studies.evidence.interval_audit canonical-stratified-tmle
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from tests.studies.evidence.manifest import read_rows, write_rows
from tests.studies.evidence.registry import StudyRecord, registered

#: The audit file's name inside a study's artefact directory.
AUDIT_FILE = "interval-audit.parquet"

#: The committed values a record is computed from.  The file leaves them out, and :func:`read`
#: joins them back from the study's row files, so each record stays tied to the committed rows.
VALUES = ("estimate", "std_error", "truth", "covered_committed", "rejected_committed")

#: The text columns of an audit record.
TEXT = ("source", "implementation", "scenario", "estimand", "property", "cell", "trigger", "scale")

#: The columns of an audit record, in order.
COLUMNS = (
    "source",
    "implementation",
    "scenario",
    "estimand",
    "property",
    "cell",
    "replicate",
    "estimate",
    "std_error",
    "truth",
    "covered_committed",
    "rejected_committed",
    "trigger",
    "scale",
    "covered_shipped",
    "rejected_shipped",
)

#: The property verdicts of the audited cells under the shipped interval, beside the audit.
PROPERTIES_FILE = "interval-audit-properties.csv"

#: The withheld triggers, as the two parameter statuses name them.
CONSTANT_ARM_CELL = "constant_arm_cell"
OUT_OF_RANGE = "out_of_range"

#: A scan's record of a withheld row: ``("truth", scenario, estimand, replicate)`` or
#: ``("property", property, cell, replicate)``.
Withheld = tuple[str, str, str, int]


@dataclass(frozen=True)
class Audit:
    """What one study's audit reads.

    Parameters
    ----------
    truth : mapping of str to tuple of str
        Scenario to the bounded-mean estimands of the package implementation's truth rows.
    cells : tuple of str
        The ``property/cell`` names whose rows are a bounded mean under the package interval.
    scan : callable or None
        Returns every row the constant-arm-cell trigger withholds, means and the contrasts that
        read them alike, from the regenerated samples.
    """

    truth: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cells: tuple[str, ...] = ()
    scan: Callable[[], list[Withheld]] | None = None


def logit_interval(estimate: Any, std_error: Any, *, alpha: float = 0.05) -> tuple[Any, Any]:
    """The shipped interval of a bounded mean: the logit-scale normal interval mapped back.

    Parameters
    ----------
    estimate : array-like
        Estimates strictly inside ``(0, 1)``.
    std_error : array-like
        Their influence-curve standard errors.
    alpha : float, default=0.05
        One minus the level.

    Returns
    -------
    tuple of numpy.ndarray
        The lower and upper endpoints.
    """
    p = np.asarray(estimate, dtype=float)
    half = stats.norm.ppf(1.0 - alpha / 2.0) * np.asarray(std_error, dtype=float) / (p * (1 - p))
    return expit(logit(p) - half), expit(logit(p) + half)


def logit_rejects(estimate: Any, std_error: Any, null: float, *, alpha: float = 0.05) -> Any:
    """Whether the shipped test of a bounded mean rejects ``null``; a null on a bound always does."""
    p = np.asarray(estimate, dtype=float)
    if null <= 0.0 or null >= 1.0:
        return np.ones(len(p), dtype=bool)
    statistic = (logit(p) - logit(null)) * p * (1 - p) / np.asarray(std_error, dtype=float)
    return np.abs(statistic) > stats.norm.ppf(1.0 - alpha / 2.0)


def shipped(audit: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """``trigger``, ``scale``, ``covered_shipped`` and ``rejected_shipped`` of audit records.

    A record whose ``trigger`` is already ``constant_arm_cell`` keeps it.  Every other record is
    withheld as ``out_of_range`` when its estimate lies on or outside ``(0, 1)``, and otherwise
    takes the logit interval.  A withheld row is neither covered nor rejected, as the committed
    rows record a ``nan`` interval and p-value.  ``rejected`` is read against the default null 0
    only where the committed row recorded one.
    """
    out = audit.copy()
    p = out["estimate"].to_numpy(dtype=float)
    se = out["std_error"].to_numpy(dtype=float)
    truth = out["truth"].to_numpy(dtype=float)
    constant = out["trigger"].fillna("").to_numpy(dtype=str) == CONSTANT_ARM_CELL
    outside = ~((p > 0.0) & (p < 1.0)) & ~constant
    inside = ~(constant | outside)
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = logit_interval(np.where(inside, p, 0.5), se, alpha=alpha)
        rejects = logit_rejects(np.where(inside, p, 0.5), se, 0.0, alpha=alpha)
    out["trigger"] = np.where(constant, CONSTANT_ARM_CELL, np.where(outside, OUT_OF_RANGE, ""))
    out["scale"] = np.where(inside, "range_logit", "withheld")
    out["covered_shipped"] = (inside & (low <= truth) & (truth <= high)).astype(int)
    recorded = out["rejected_committed"].notna().to_numpy()
    rejected = (inside & rejects).astype(float)
    out["rejected_shipped"] = np.where(recorded, rejected, np.nan)
    return out


def _truth_records(record: StudyRecord, audit: Audit, withheld: set[Withheld]) -> pd.DataFrame:
    rows = read_rows(record.artifacts / "replicates.parquet")
    rows = rows.loc[rows["implementation"] == record.implementation]
    keep = np.zeros(len(rows), dtype=bool)
    for scenario, names in audit.truth.items():
        keep |= (rows["scenario"] == scenario).to_numpy() & rows["estimand"].isin(names).to_numpy()
    flagged = {
        (scenario, name, replicate)
        for kind, scenario, name, replicate in withheld
        if kind == "truth"
    }
    keys = list(zip(rows["scenario"], rows["estimand"], rows["replicate"], strict=True))
    hit = np.array([key in flagged for key in keys], dtype=bool)
    if len(flagged) != int(hit.sum()):
        raise ValueError(f"{record.slug}: a withheld truth row has no committed row")
    rows = rows.loc[keep | hit]
    hit = hit[keep | hit]
    return pd.DataFrame(
        {
            "source": "truth",
            "implementation": rows["implementation"].to_numpy(),
            "scenario": rows["scenario"].to_numpy(),
            "estimand": rows["estimand"].to_numpy(),
            "property": "",
            "cell": "",
            "replicate": rows["replicate"].to_numpy(dtype=int),
            "estimate": rows["estimate"].to_numpy(dtype=float),
            "std_error": rows["std_error"].to_numpy(dtype=float),
            "truth": rows["truth"].to_numpy(dtype=float),
            "covered_committed": rows["covered"].to_numpy(dtype=int),
            "rejected_committed": np.nan,
            "trigger": np.where(hit, CONSTANT_ARM_CELL, ""),
        }
    )


def _property_records(record: StudyRecord, audit: Audit, withheld: set[Withheld]) -> pd.DataFrame:
    path = record.artifacts / "property-replicates.parquet"
    if not audit.cells and not any(kind == "property" for kind, *_ in withheld):
        return pd.DataFrame(columns=COLUMNS)
    rows = read_rows(path)
    names = rows["property"] + "/" + rows["cell"]
    keep = names.isin(audit.cells).to_numpy()
    flagged = {
        (prop, cell, replicate) for kind, prop, cell, replicate in withheld if kind == "property"
    }
    keys = list(zip(rows["property"], rows["cell"], rows["replicate"], strict=True))
    hit = np.array([key in flagged for key in keys], dtype=bool)
    if len(flagged) != int(hit.sum()):
        raise ValueError(f"{record.slug}: a withheld property row has no committed row")
    rows = rows.loc[keep | hit]
    hit = hit[keep | hit]
    return pd.DataFrame(
        {
            "source": "property",
            "implementation": record.implementation,
            "scenario": "",
            "estimand": "",
            "property": rows["property"].to_numpy(),
            "cell": rows["cell"].to_numpy(),
            "replicate": rows["replicate"].to_numpy(dtype=int),
            "estimate": rows["estimate"].to_numpy(dtype=float),
            "std_error": rows["std_error"].to_numpy(dtype=float),
            "truth": rows["truth"].to_numpy(dtype=float),
            "covered_committed": rows["covered"].to_numpy(dtype=int),
            "rejected_committed": rows["rejected"].to_numpy(dtype=float),
            "trigger": np.where(hit, CONSTANT_ARM_CELL, ""),
        }
    )


def build(record: StudyRecord, audit: Audit) -> pd.DataFrame:
    """The audit records of one study, shipped columns included, in file order."""
    withheld = set(audit.scan()) if audit.scan is not None else set()
    parts = [
        frame
        for frame in (
            _truth_records(record, audit, withheld),
            _property_records(record, audit, withheld),
        )
        if len(frame)
    ]
    frame = pd.concat(parts, ignore_index=True)
    frame = shipped(frame, alpha=record.margins.alpha)
    return (
        frame.loc[:, list(COLUMNS)]
        .sort_values(
            ["source", "scenario", "estimand", "property", "cell", "replicate"], kind="stable"
        )
        .reset_index(drop=True)
    )


def read(record: StudyRecord) -> pd.DataFrame:
    """A study's committed audit file, with the committed values of each record joined back."""
    frame = read_rows(record.artifacts / AUDIT_FILE)
    frame[list(TEXT)] = frame[list(TEXT)].fillna("").astype(str)
    parts = []
    truth = frame.loc[frame["source"] == "truth"]
    if len(truth):
        rows = read_rows(record.artifacts / "replicates.parquet")
        rows = rows.rename(columns={"covered": "covered_committed"})
        rows["rejected_committed"] = np.nan
        keys = ["implementation", "scenario", "estimand", "replicate"]
        parts.append(truth.merge(rows[keys + list(VALUES)], on=keys, how="left", validate="1:1"))
    prop = frame.loc[frame["source"] == "property"]
    if len(prop):
        rows = read_rows(record.artifacts / "property-replicates.parquet")
        rows = rows.rename(
            columns={"covered": "covered_committed", "rejected": "rejected_committed"}
        )
        keys = ["property", "cell", "replicate"]
        parts.append(prop.merge(rows[keys + list(VALUES)], on=keys, how="left", validate="1:1"))
    joined = pd.concat(parts).sort_index()
    return joined.loc[:, list(COLUMNS)]


def coverage(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coverage by row group: truth rows by scenario and estimand, property rows by cell."""
    key = np.where(
        frame["source"] == "truth",
        frame["scenario"] + "/" + frame["estimand"],
        frame["property"] + "/" + frame["cell"],
    )
    return frame.groupby(key)[column].mean()


# ------------------------------------------------------------------ the audited studies


def _stratified_scan() -> list[Withheld]:
    """Constant arm cells of ``canonical-stratified-tmle``, in its truth and property samples."""
    from cleverly.inference.influence import constant_outcome_cells
    from tests.studies import canonical_stratified_tmle as primary
    from tests.studies import stratified_tmle_properties as properties

    def cells(frame: pd.DataFrame) -> Iterator[tuple[int, int]]:
        for stratum in (0, 1, 2):
            rows = (frame["V"] == stratum).to_numpy()
            for arm in constant_outcome_cells(frame["Y"].to_numpy(), frame["A"].to_numpy(), rows):
                yield stratum, int(arm)

    out: list[Withheld] = []
    record = primary.STUDY
    for scenario in record.scenarios:
        for replicate in range(record.replicates):
            frame, _ = primary.draw_scenario(scenario, record.n, replicate)
            for stratum, arm in cells(frame):
                for name in (f"ey[{arm}][V={stratum}]", f"ate[V={stratum}]"):
                    out.append(("truth", scenario, name, replicate))
    declared = {
        f"{family}/{cell}" for family, names in record.property_cells.items() for cell in names
    }
    for key in properties._payloads():
        _, _, replicate, *_ = key[0]
        for stratum, arm in cells(properties._draw(key)):
            for name in properties._cells_of(key):
                prop, cell = name.split("/", 1)
                label = cell.split("__", 1)[0]
                reads = label.startswith(f"v{stratum}_") and label != f"v{stratum}_ey{1 - arm}"
                kind = cell.split("__", 1)[1]
                # The derived controls build their own intervals and the band is the band's.
                package = kind in {
                    "correctly_specified",
                    "both_correct",
                    "outcome_correct",
                    "treatment_correct",
                    "both_wrong",
                    "stratified",
                }
                if reads and package and name in declared:
                    out.append(("property", prop, cell, replicate))
    return out


def _stratified_audit() -> Audit:
    means = tuple(f"ey[{arm}][V={stratum}]" for stratum in (0, 1, 2) for arm in (1, 0))
    cells = tuple(
        f"interval_calibration/v{stratum}_ey{arm}__correctly_specified"
        for stratum in (0, 1, 2)
        for arm in (1, 0)
    )
    return Audit(truth={"stratified_binary": means}, cells=cells, scan=_stratified_scan)


def _composite_audit() -> Audit:
    from tests.studies import composite_drtmle_properties as design

    # No scan: the smallest arm of the three-arm law holds hundreds of rows at a mean near
    # 0.69, so a constant arm cell has probability far below 1e-30 per sample.
    binary = ("ey0", "ey1")
    return Audit(
        truth={
            "binary_observational_mar": binary,
            "binary_mar_outcome_and_treatment": binary,
            design.THREE_ARM: ("ey[high]", "ey[low]", "ey[mid]"),
        }
    )


def _pricing_audit() -> Audit:
    # No scan: a longitudinal level whose last-node regression reads a constant outcome takes
    # ``constant_node_plugin``, which the committed rows already record.  The range check
    # reads the committed estimate.
    record = record_for("mtp-pricing")
    return Audit(
        truth={
            scenario: tuple(name for name in names if name.startswith("ey_regimen["))
            for scenario, names in record.scenarios.items()
        }
    )


def _learned_rule_audit() -> Audit:
    # No scan: every arm of the law has probability at least 0.166 on its support, so each arm
    # holds hundreds of rows at n = 2,000 and a constant arm cell is negligible.  The derived
    # controls build their own intervals and stay out.
    return Audit(
        truth={"three_arms": ("ey_learned_rule[learned rule]",)},
        cells=(
            "interval_calibration/three_arms__declared",
            "conservative_coverage/missing_quadratic__declared",
            "targeting_necessity/drop_w1__targeted",
        ),
    )


#: Study slug to the factory of its audit declaration.
AUDITS: dict[str, Callable[[], Audit]] = {
    "canonical-stratified-tmle": _stratified_audit,
    "composite-missing-drtmle": _composite_audit,
    "mtp-pricing": _pricing_audit,
    "learned-rule-arms": _learned_rule_audit,
}


def shipped_property_rows(record: StudyRecord, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The study's property rows with ``covered`` and ``rejected`` read under the shipped interval.

    Only the audited records change.  ``rejected`` changes only where the committed row recorded
    one, as :func:`shipped` reads it.
    """
    frame = read(record) if frame is None else frame
    rows = read_rows(record.artifacts / "property-replicates.parquet")
    audited = frame.loc[frame["source"] == "property"]
    keys = ["property", "cell", "replicate"]
    merged = rows[keys].merge(
        audited[[*keys, "covered_shipped", "rejected_shipped"]], on=keys, how="left"
    )
    hit = merged["covered_shipped"].notna().to_numpy()
    rows.loc[hit, "covered"] = merged.loc[hit, "covered_shipped"].to_numpy(dtype=int)
    reject = hit & merged["rejected_shipped"].notna().to_numpy()
    rows.loc[reject, "rejected"] = merged.loc[reject, "rejected_shipped"].to_numpy()
    return rows


def shipped_properties(record: StudyRecord) -> pd.DataFrame:
    """The study's own property summary of the audited cells under the shipped interval.

    The summary is the study module's ``summarize_properties`` of the rows that
    :func:`shipped_property_rows` returns, so every rule, clause and band is the registered one.
    """
    frame = read(record)
    summary = record.properties().summarize_properties(shipped_property_rows(record, frame))
    audited = frame.loc[frame["source"] == "property", ["property", "cell"]].drop_duplicates()
    return summary.merge(audited, on=["property", "cell"]).reset_index(drop=True)


def read_properties(record: StudyRecord) -> pd.DataFrame | None:
    """A study's committed shipped-interval property verdicts, or ``None`` when it has none."""
    path = record.artifacts / PROPERTIES_FILE
    return pd.read_csv(path) if path.exists() else None


def record_for(slug: str) -> StudyRecord:
    """The registered study of ``slug``."""
    for record in registered():
        if record.slug == slug:
            return record
    raise KeyError(slug)


def write(slug: str) -> Path:
    """Build and write one study's audit file; its path."""
    record = record_for(slug)
    frame = build(record, AUDITS[slug]()).drop(columns=list(VALUES))
    path = record.artifacts / AUDIT_FILE
    write_rows(frame, path, dtype=dict.fromkeys(TEXT, str))
    if (frame["source"] == "property").any():
        shipped_properties(record).to_csv(
            record.artifacts / PROPERTIES_FILE, index=False, lineterminator="\n"
        )
    return path


def main(argv: list[str] | None = None) -> None:
    """Write the audit file of each named study."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("studies", nargs="+", choices=sorted(AUDITS))
    for slug in parser.parse_args(argv).studies:
        print(write(slug))


if __name__ == "__main__":
    main()
