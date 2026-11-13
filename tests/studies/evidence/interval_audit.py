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

The committed ``properties.csv`` and ``performance-tests.csv`` stay as the record of the
identity interval.  A study rerun under the shipped interval records it in its own rows and has
no audit: ``learned-rule-arms`` left this list when its conservative coverage cell went red on
logit and was rerun under the shipped interval, so the screen reads its rows directly.  Write
or refresh an audit file with::

    python -m tests.studies.evidence.interval_audit canonical-stratified-tmle

The screen (``tests/canonical/interval-audit-screen.csv``) holds one row per registered study.
It recomputes, from the committed rows, every group that could be a bounded mean (a superset),
counts the groups whose coverage moves under the shipped interval, and names the groups whose
exact coverage interval changes side of a coverage gate.  Each such flip is audited here or
explained in :data:`SCREEN_REASONS`, and the fast tier recomputes every row, so a new study or
a new cell cannot pass unscreened.  The screen also counts each study's
``simultaneous_coverage/*__pointwise_joint_control`` cells: they are not gated, and their
pointwise reading moves to the logit scale for a bounded-mean family on the next run.  Refresh
it with::

    python -m tests.studies.evidence.interval_audit --screen
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from scipy.special import expit, logit

from tests.studies.evidence import properties as properties_module
from tests.studies.evidence.manifest import read_rows, write_rows
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

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


def logit_interval(
    estimate: Any, std_error: Any, *, alpha: float = 0.05, margin: Any = 0.0
) -> tuple[Any, Any]:
    """The shipped interval of a bounded mean: the logit-scale normal interval mapped back.

    A fit under a declared stopping rule adds its remaining score ``margin`` to the half-width
    on the level scale, which the logit map carries by the slope that carries the standard
    error (:meth:`~cleverly.inference.ParameterEstimate.residual_margin`).

    Parameters
    ----------
    estimate : array-like
        Estimates strictly inside ``(0, 1)``.
    std_error : array-like
        Their influence-curve standard errors.
    alpha : float, default=0.05
        One minus the level.
    margin : array-like, default=0.0
        The absolute remaining score of a fit under a declared stopping rule, else zero.

    Returns
    -------
    tuple of numpy.ndarray
        The lower and upper endpoints.
    """
    p = np.asarray(estimate, dtype=float)
    level = stats.norm.ppf(1.0 - alpha / 2.0) * np.asarray(std_error, dtype=float)
    half = (level + np.asarray(margin, dtype=float)) / (p * (1 - p))
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


#: The cell kinds of ``canonical-stratified-tmle`` whose rows carry the package interval.  The
#: derived calibration controls, the joint band cells and the hand-built ``marginal_fluctuation``
#: control form their own intervals, and the study's role metadata does not separate a
#: package-built control (``both_wrong``) from a hand-built one, so the kinds are declared.
STRATIFIED_PACKAGE_KINDS = frozenset(
    {
        "correctly_specified",
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
        "stratified",
    }
)


def _stratified_cells(frame: pd.DataFrame) -> Iterator[tuple[int, int]]:
    """The ``(stratum, arm)`` cells of one stratified sample that hold one outcome value."""
    from cleverly.inference.influence import constant_outcome_cells

    for stratum in (0, 1, 2):
        rows = (frame["V"] == stratum).to_numpy()
        for arm in constant_outcome_cells(frame["Y"].to_numpy(), frame["A"].to_numpy(), rows):
            yield stratum, int(arm)


def stratified_truth_withheld(scenario: str, replicate: int) -> list[Withheld]:
    """The truth rows one replicate of ``canonical-stratified-tmle`` withholds."""
    from tests.studies import canonical_stratified_tmle as primary

    frame, _ = primary.draw_scenario(scenario, primary.STUDY.n, replicate)
    return [
        ("truth", scenario, name, replicate)
        for stratum, arm in _stratified_cells(frame)
        for name in (f"ey[{arm}][V={stratum}]", f"ate[V={stratum}]")
    ]


def stratified_property_withheld(key: Any) -> list[Withheld]:
    """The property rows one payload ``key`` of ``canonical-stratified-tmle`` withholds.

    A row is withheld when its label reads a constant cell's stratum and arm (a mean reads its
    own arm, a contrast both), its kind is in :data:`STRATIFIED_PACKAGE_KINDS`, and the study
    declares the cell.
    """
    from tests.studies import canonical_stratified_tmle as primary
    from tests.studies import stratified_tmle_properties as properties

    declared = {
        f"{family}/{cell}"
        for family, names in primary.STUDY.property_cells.items()
        for cell in names
    }
    _, _, replicate, *_ = key[0]
    out: list[Withheld] = []
    for stratum, arm in _stratified_cells(properties._draw(key)):
        for name in properties._cells_of(key):
            prop, cell = name.split("/", 1)
            label, kind = cell.split("__", 1)
            reads = label.startswith(f"v{stratum}_") and label != f"v{stratum}_ey{1 - arm}"
            if reads and kind in STRATIFIED_PACKAGE_KINDS and name in declared:
                out.append(("property", prop, cell, replicate))
    return out


def _stratified_scan() -> list[Withheld]:
    """Constant arm cells of ``canonical-stratified-tmle``, in its truth and property samples."""
    from tests.studies import canonical_stratified_tmle as primary
    from tests.studies import stratified_tmle_properties as properties

    record = primary.STUDY
    out: list[Withheld] = []
    for scenario in record.scenarios:
        for replicate in range(record.replicates):
            out += stratified_truth_withheld(scenario, replicate)
    for key in properties._payloads():
        out += stratified_property_withheld(key)
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


def _stratified_drtmle_foldwise_audit() -> Audit:
    # No scan: the constant-arm-cell trigger is applied by the point-treatment TMLE only, and a
    # DR-TMLE fit reports no such status. The contraction ladder's cells read the marginal ate, a
    # difference, whose interval is unchanged.
    means = tuple(f"ey[{arm}][V={stratum}]" for stratum in (0, 1, 2) for arm in (1, 0))
    return Audit(truth={"stratified_both_correct": means})


def _pricing_audit() -> Audit:
    # No scan: a longitudinal level whose last-node regression reads a constant outcome takes
    # ``constant_node_plugin``, which the committed rows already record.  The range check
    # reads the committed estimate.  The package cells of ``interval_calibration`` are levels
    # of the binary outcome too; the ``revenue`` cell's outcome is continuous.
    record = record_for("mtp-pricing")
    return Audit(
        truth={
            scenario: tuple(name for name in names if name.startswith("ey_regimen["))
            for scenario, names in record.scenarios.items()
        },
        cells=tuple(
            f"interval_calibration/{cell}"
            for cell in record.property_cells["interval_calibration"]
            if cell.endswith("__correctly_specified") and not cell.startswith("revenue_")
        ),
    )


#: Study slug to the factory of its audit declaration.
AUDITS: dict[str, Callable[[], Audit]] = {
    "canonical-stratified-tmle": _stratified_audit,
    "composite-missing-drtmle": _composite_audit,
    "mtp-pricing": _pricing_audit,
    "stratified-drtmle-foldwise": _stratified_drtmle_foldwise_audit,
}


# ------------------------------------------------------------------ the screen

#: The screen of every registered study, one row each, beside the audits.
SCREEN_FILE = ROOT / "tests" / "canonical" / "interval-audit-screen.csv"

#: The columns of the screen, in order.
SCREEN_COLUMNS = (
    "study",
    "disposition",
    "candidate_groups",
    "moved_groups",
    "max_coverage_change",
    "flips",
    "crossings",
    "pointwise_joint_controls",
    "reason",
)

#: Studies whose committed rows are declared to carry the shipped interval already, so the screen
#: does not recompute them: a rerun replaces their rows, and recomputing the first run's rows would
#: describe a record the rerun withdraws. A study leaves when its rerun is committed, because its
#: rows then carry the shipped interval and the ordinary screen reads them with no change.
RERUN: dict[str, str] = {}

#: Why a study outside :data:`AUDITS` holds no verdict the shipped interval can move, for each
#: study whose screen row records a flip.  The screen treats every positive row group with
#: estimates and truths in ``[0, 1]`` as a bounded mean, a superset, so a flip names a group to
#: explain rather than a moved verdict.
SCREEN_REASONS: dict[str, str] = {
    "canonical-multi-arm-ctmle-selector": (
        "the flagged cells read ate[medium vs high], a difference, whose interval is unchanged"
    ),
    "canonical-multi-arm-drtmle": (
        "root_n_and_efficiency/n_500 reads ate[medium vs high], a difference, whose interval is "
        "unchanged"
    ),
    "canonical-stratified-incremental-msm": (
        "logit_v0_w is a working-model coefficient of the logit MSM, whose interval is unchanged"
    ),
    "ctmle-oat-per-arm-strata-missing": (
        "root_n_and_efficiency/n_2000 and n_4000 read the marginal ate, a difference, whose "
        "interval is unchanged"
    ),
    "option-patience": (
        "late_confounder_stop/lc3_ordered reads an exclusion share in its covered column, not an "
        "interval; the selector path withholds its interval (the study module states it)"
    ),
    "score-assessment": (
        "the flagged cell is a calibration slope of a CATE score, not a package interval of a mean"
    ),
    "targeting-curves": (
        "the flagged cells are band and type I rows of a targeting curve with truth 0, not "
        "package intervals of a mean"
    ),
}


def bounded_heads() -> frozenset[str]:
    """The parameter heads of a bounded level: the package's mean targets and longitudinal levels.

    The point-treatment heads are the built-in targets declared ``bounded="mean"``; the
    longitudinal heads are every name ``_level_head`` gives.  A truth row whose estimand starts
    with one of them is a screen candidate.
    """
    from itertools import product

    from cleverly.longitudinal.estimator import _level_head
    from cleverly.targets.builtin import BUILTIN_TARGETS

    heads = {target.name for target in BUILTIN_TARGETS if target.bounded == "mean"}
    for survival, competing, complement in product((False, True), repeat=3):
        try:
            heads.add(_level_head(survival=survival, competing=competing, complement=complement))
        except Exception:  # a combination with no level, such as a complement of a mean
            continue
    return frozenset(heads)


def _exact_interval(count: int, total: int, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low = 0.0 if count == 0 else float(stats.beta.ppf(tail, count, total - count + 1))
    high = 1.0 if count == total else float(stats.beta.ppf(1.0 - tail, count + 1, total - count))
    return low, high


def _screen_groups(
    record: StudyRecord, heads: frozenset[str], property_rows: pd.DataFrame | None = None
) -> Iterator[tuple[str, Any, str]]:
    """Each candidate group of the study: its key, its rows and ``"truth"`` or ``"property"``.

    ``property_rows`` are the committed property rows when the caller has read them already. A
    property group keeps their index, so a caller can write a recomputed column back.
    """
    path = record.artifacts / "replicates.parquet"
    if path.exists():
        rows = read_rows(path)
        rows = rows.loc[rows["implementation"] == record.implementation]
        for (scenario, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
            if str(estimand).split("[", 1)[0] in heads:
                yield f"{scenario}/{estimand}", group, "truth"
    path = record.artifacts / "property-replicates.parquet"
    if property_rows is None and path.exists():
        property_rows = read_rows(path)
    if property_rows is not None:
        rows = property_rows.loc[property_rows["role"] == "positive"]
        for (prop, cell), group in rows.groupby(["property", "cell"], sort=True):
            yield f"{prop}/{cell}", group, "property"


def _crosses(before: tuple[float, float], after: tuple[float, float], margins: Any) -> bool:
    """Whether a moved exact coverage interval crosses a threshold a coverage verdict reads.

    The thresholds are the lower end against the coverage floor, the lower end against the
    calibration band's lower end, and the interval inside the calibration band. A cell whose
    rule reads only another threshold, such as a share floor of a cell whose ``covered`` holds
    an event, is not re-judged. With this filter replaced by ``True`` on 2026-11-13, the own
    verdicts of every registered study moved only where the screen flips, and at the two
    share cells of ``isotonic-binding`` (``binding_witness`` and ``material_move``), whose
    ``covered`` is an event with a placeholder standard error, not an interval.
    """
    floor, band = margins.coverage_floor, margins.calibration_coverage

    def sides(interval: tuple[float, float]) -> tuple[bool, bool, bool]:
        low, high = interval
        return low >= floor, low >= band[0], band[0] <= low and high <= band[1]

    return sides(before) != sides(after)


@contextmanager
def _shared_ratio_draws() -> Iterator[None]:
    """Reuse the bootstrap draws of a spread ratio across calls on the same rows.

    The draws read the estimates, standard errors and truths and never ``covered``, so a
    summary of the rows before and after the screen's recomputation draws them once.
    """
    shipped = properties_module.ratio_draws
    cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}

    def shared(group: pd.DataFrame, **options: Any) -> dict[str, np.ndarray]:
        rows = group.drop(columns=["covered"], errors="ignore")
        key = (
            pd.util.hash_pandas_object(rows, index=True).to_numpy().tobytes(),
            tuple(rows.columns),
            tuple(sorted(options.items())),
        )
        if key not in cache:
            cache[key] = shipped(group, **options)
        return cache[key]

    properties_module.ratio_draws = shared
    try:
        yield
    finally:
        properties_module.ratio_draws = shipped


def _moved_verdicts(
    record: StudyRecord,
    committed: pd.DataFrame,
    recomputed: pd.DataFrame,
    families: Iterable[str],
) -> list[str]:
    """The ``property/cell`` keys of ``families`` whose own verdict changes with ``covered``.

    The study's ``summarize_properties`` judges each cell against what its own rule reads: the
    coverage floor, the calibration band or its lower end, a floor the study declares, or a
    comparison with another cell. It runs on the committed rows and on the rows with the
    shipped interval's ``covered``, and a cell whose ``passed`` differs is a flip. It reads the
    rows of ``families`` alone, or every row when a family's rule needs another family.
    """
    summarize = record.properties().summarize_properties
    columns = ["property", "cell", "passed"]
    within = committed["property"].isin(list(families)).to_numpy()
    with _shared_ratio_draws():
        try:
            before = summarize(committed.loc[within].copy())
            after = summarize(recomputed.loc[within].copy())
        except Exception:  # a family whose rule reads another family's rows
            before, after = summarize(committed.copy()), summarize(recomputed.copy())
    before = before.loc[:, columns].set_index(["property", "cell"])
    after = after.loc[:, columns].set_index(["property", "cell"])
    joined = before.join(after, how="outer", lsuffix="_before", rsuffix="_after")
    changed = joined["passed_before"].astype(str) != joined["passed_after"].astype(str)
    return [f"{prop}/{cell}" for prop, cell in joined.index[changed]]


#: The suffix of a stopping-rule study's cell whose ``covered`` column reads the residual, not an
#: interval (``tests/studies/_option_stopping_cells.py``).
WITNESS_SUFFIX = "__witness"
#: The suffix of a stopping-rule study's cell that reads the widened interval.
WIDENED_SUFFIX = "__widened"


def _stopping_rule_study(record: StudyRecord) -> bool:
    """Whether the study's property rows carry the residual of a declared stopping rule."""
    path = record.artifacts / "property-replicates.parquet"
    return path.exists() and "residual" in pq.read_schema(path).names


def _screen_margin(key: str, group: Any, alpha: float, *, stopping_rule: bool) -> Any:
    """The remaining-score margin each committed row's interval carried, on the level scale.

    A stopping-rule study's property row carries the residual in standard errors, and its
    widened cell reads ``|residual| * se``. Its truth row carries the identity-scale interval,
    whose half-width beyond ``z * se`` is the margin; it is set to zero below ``1e-9 * se``,
    where it is roundoff. Every other row carried none: an identity half-width beyond
    ``z * se`` there is a t or cluster reference, which the screen does not model.
    """
    se = group["std_error"].to_numpy(dtype=float)
    if not stopping_rule:
        return np.zeros_like(se)
    if "residual" in group.columns:
        if not key.endswith(WIDENED_SUFFIX):
            return np.zeros_like(se)
        return np.abs(group["residual"].to_numpy(dtype=float)) * se
    if {"ci_lower", "ci_upper", "inference_scale"} <= set(group.columns) and (
        group["inference_scale"] == "identity"
    ).all():
        half = (
            group["ci_upper"].to_numpy(dtype=float) - group["ci_lower"].to_numpy(dtype=float)
        ) / 2
        extra = half - stats.norm.ppf(1.0 - alpha / 2.0) * se
        return np.where(extra > 1e-9 * se, extra, 0.0)
    return np.zeros_like(se)


def _pointwise_joint_controls(record: StudyRecord) -> int:
    path = record.artifacts / "property-replicates.parquet"
    if not path.exists():
        return 0
    cells = read_rows(path)["cell"].drop_duplicates()
    return int(cells.str.endswith("__pointwise_joint_control").sum())


def screen_row(record: StudyRecord, heads: frozenset[str] | None = None) -> dict[str, Any]:
    """One study's screen row, recomputed from its committed rows.

    A candidate is a package truth-row group whose estimand head is a bounded level
    (:func:`bounded_heads`), or a positive property cell, whose estimates and truths all lie in
    ``[0, 1]``.  The screen reads each candidate under the shipped logit interval with the
    committed estimate and standard error, as :func:`shipped` does, and counts the candidates
    whose coverage moves.  Each flip is judged against what its own verdict reads. A truth-row
    group flips when its exact coverage interval, at the study's confidence level, changes side
    of the coverage floor, which is all its verdict reads (``performance.py``). A property cell
    is a crossing when its exact interval crosses the coverage floor or the calibration band's
    lower end, or enters or leaves the band (:func:`_crosses`). The families with a crossing are
    then judged by the study's own ``summarize_properties`` with every moved group carrying the
    shipped ``covered``, and each cell whose ``passed`` changes is a flip
    (:func:`_moved_verdicts`). A crossing that is not a flip is a cell whose verdict does not
    move. Every audited flip is audited, and every other flip needs a reason in
    :data:`SCREEN_REASONS`.
    """
    slug = record.slug
    if slug in RERUN:
        return {
            "study": slug,
            "disposition": "rerun",
            "candidate_groups": 0,
            "moved_groups": 0,
            "max_coverage_change": 0.0,
            "flips": "",
            "crossings": "",
            "pointwise_joint_controls": _pointwise_joint_controls(record),
            "reason": RERUN[slug],
        }
    heads = bounded_heads() if heads is None else heads
    margins = record.margins
    candidates = moved = 0
    largest = 0.0
    flips: list[str] = []
    stopping_rule = _stopping_rule_study(record)
    path = record.artifacts / "property-replicates.parquet"
    committed_rows = read_rows(path) if path.exists() else None
    property_rows = None if committed_rows is None else committed_rows.copy()
    crossings: list[str] = []
    families: set[str] = set()
    for key, group, kind in _screen_groups(record, heads, committed_rows):
        if stopping_rule and key.endswith(WITNESS_SUFFIX):
            continue
        estimate = group["estimate"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        se = group["std_error"].to_numpy(dtype=float)
        finite = np.isfinite(estimate) & np.isfinite(truth) & np.isfinite(se)
        if not finite.any():
            continue
        margin = _screen_margin(key, group, margins.alpha, stopping_rule=stopping_rule)[finite]
        estimate, truth, se = estimate[finite], truth[finite], se[finite]
        committed = group["covered"].to_numpy(dtype=int)[finite]
        unit = (estimate >= 0.0) & (estimate <= 1.0) & (truth >= 0.0) & (truth <= 1.0)
        if not (unit & (se > 0.0)).all():
            continue
        candidates += 1
        inside = (estimate > 0.0) & (estimate < 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            low, high = logit_interval(
                np.where(inside, estimate, 0.5), se, alpha=margins.alpha, margin=margin
            )
        covered = (inside & (low <= truth) & (truth <= high)).astype(int)
        if (covered == committed).all():
            continue
        moved += 1
        total = len(covered)
        largest = max(largest, abs(float(covered.mean()) - float(committed.mean())))
        before = _exact_interval(int(committed.sum()), total, margins.confidence_level)
        after = _exact_interval(int(covered.sum()), total, margins.confidence_level)
        if kind == "truth":
            # A truth row's verdict reads the coverage floor alone (performance.py).
            floor = margins.coverage_floor
            if (before[0] >= floor) != (after[0] >= floor):
                flips.append(key)
        else:
            assert property_rows is not None
            property_rows.loc[group.index[finite], "covered"] = covered
            if _crosses(before, after, margins):
                crossings.append(key)
                families.add(str(group["property"].iloc[0]))
    if families:
        assert committed_rows is not None and property_rows is not None
        flips.extend(_moved_verdicts(record, committed_rows, property_rows, sorted(families)))
    if slug in AUDITS:
        disposition, reason = "audited", AUDIT_FILE
    elif flips:
        disposition, reason = "declared", SCREEN_REASONS.get(slug, "")
    else:
        disposition, reason = "neutral", ""
    return {
        "study": slug,
        "disposition": disposition,
        "candidate_groups": candidates,
        "moved_groups": moved,
        "max_coverage_change": round(largest, 6),
        "flips": ";".join(flips),
        "crossings": ";".join(crossings),
        "pointwise_joint_controls": _pointwise_joint_controls(record),
        "reason": reason,
    }


def write_screen() -> Path:
    """Recompute the screen of every registered study and write it; its path."""
    heads = bounded_heads()
    frame = pd.DataFrame(
        [screen_row(record, heads) for record in sorted(registered(), key=lambda r: r.slug)],
        columns=list(SCREEN_COLUMNS),
    )
    frame.to_csv(SCREEN_FILE, index=False, lineterminator="\n")
    return SCREEN_FILE


def read_screen() -> pd.DataFrame:
    """The committed screen, with empty text cells as ``""``."""
    frame = pd.read_csv(SCREEN_FILE, keep_default_na=False)
    return frame


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
    return path


def main(argv: list[str] | None = None) -> None:
    """Write the audit file of each named study."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("studies", nargs="*", choices=sorted(AUDITS))
    parser.add_argument("--screen", action="store_true", help="rewrite the screen of every study")
    arguments = parser.parse_args(argv)
    for slug in arguments.studies:
        print(write(slug))
    if arguments.screen:
        print(write_screen())


if __name__ == "__main__":
    main()
