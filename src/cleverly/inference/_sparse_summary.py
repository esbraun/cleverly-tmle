"""The printed summary of the influence report (X89 T3).

One row per estimate: how many equal contributions carry its spread, the shape of its
contributions, how noisy its standard error is, and the coverage the shape is predicted to
cost a Wald interval. The numbers are read from the report; the flag is the report's own.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    note,
    short_repr,
    vocabulary,
)
from .._summary._render import count_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sparse import InfluenceReport, InfluenceRow

__all__ = ["influence_repr", "influence_summary"]


def _number(value: float, decimals: int) -> Cell:
    if not math.isfinite(value):
        return Cell(None, vocabulary.MISSING)
    text = f"{value:.{decimals}f}"
    return Cell(float(value), text[1:] if text.strip("-0.") == "" else text)


def _units(report: InfluenceReport) -> tuple[str, str]:
    return ("cluster", "clusters") if report.clustered else ("row", "rows")


def _row(row: InfluenceRow) -> tuple[Cell, ...]:
    kish = (
        Cell(float(row.kish), f"{row.kish:,.1f}")
        if math.isfinite(row.kish)
        else Cell(None, vocabulary.MISSING)
    )
    return (
        Cell(row.name, row.name),
        kish,
        _number(row.skewness, 2),
        _number(row.variance_rel_sd, 3),
        _number(row.edgeworth_loss, 3),
        Cell(row.sparse, vocabulary.INFLUENCE_FLAG if row.sparse else vocabulary.MISSING),
    )


def _table(report: InfluenceReport) -> SummaryTable:
    _, units = _units(report)
    if not report.rows:
        return SummaryTable(("estimates",), ((Cell(None, "none reported"),),))
    columns = (
        "estimate",
        f"effective {units}",
        "skewness",
        "SE uncertainty",
        "coverage loss",
        "flag",
    )
    return SummaryTable(columns, tuple(_row(row) for row in report.rows))


def _reading(report: InfluenceReport, threshold: float) -> str:
    if not report.rows:
        return vocabulary.INFLUENCE_READINGS["empty"]
    flagged = [row for row in report.rows if row.sparse]
    if not flagged:
        return vocabulary.INFLUENCE_READINGS["none"].format(threshold=f"{threshold:g}")
    names = [row.name for row in flagged]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    value = max(row.variance_rel_sd for row in flagged)
    return vocabulary.INFLUENCE_READINGS["one" if len(flagged) == 1 else "several"].format(
        names=joined, value=f"{value:.3f}", threshold=f"{threshold:g}"
    )


def _notes(report: InfluenceReport) -> list[Note]:
    unit, units = _units(report)
    notes = []
    if report.rows:
        level = 100.0 * (1.0 - report.alpha)
        notes.append(
            note(
                "influence_terms",
                vocabulary.NOTE_TEXTS["influence_terms"].format(
                    unit=unit, units=units, level=f"{level:g}"
                ),
            )
        )
    if report.clustered:
        notes.append(note("cluster_unit", vocabulary.NOTE_TEXTS["cluster_unit"]))
    if report.warned:
        notes.append(
            note("sparse_warning_pointer", vocabulary.NOTE_TEXTS["sparse_warning_pointer"])
        )
    return notes


def _sections(report: InfluenceReport, threshold: float) -> list[SummarySection]:
    _, units = _units(report)
    lines = tuple(
        line.format(units=units, threshold=f"{threshold:g}")
        for line in vocabulary.INFLUENCE_DEFINITION_LINES
    )
    sections = [SummarySection(vocabulary.INFLUENCE_DEFINITIONS, lines)]
    if report.rows:
        kurtosis = tuple(
            vocabulary.INFLUENCE_KURTOSIS_LINE.format(
                name=row.name, kurtosis=_number(row.kurtosis, 2).text
            )
            for row in report.rows
        )
        sections.append(SummarySection(vocabulary.INFLUENCE_KURTOSIS, kurtosis))
    bootstrap = tuple(
        vocabulary.INFLUENCE_BOOTSTRAP_LINE.format(
            name=row.name,
            z0=f"{row.bca_z0:.3f}",
            acceleration=f"{row.bca_acceleration:.4f}",
        )
        for row in report.rows
        if math.isfinite(row.bca_z0) or math.isfinite(row.bca_acceleration)
    )
    if bootstrap:
        sections.append(SummarySection(vocabulary.INFLUENCE_BOOTSTRAP, bootstrap))
    return sections


def influence_summary(report: InfluenceReport, detail: Detail) -> Summary:
    """The printed summary of an influence report.

    Parameters
    ----------
    report : InfluenceReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds the definitions of the measures and, with a bootstrap, its shape.

    Returns
    -------
    Summary
        The summary.
    """
    from .sparse import SPARSE_INFLUENCE_THRESHOLD

    _, units = _units(report)
    size = report.rows[0].n if report.rows else None
    label = vocabulary.CLUSTERS if report.clustered else vocabulary.ROWS
    header = [
        (label, vocabulary.MISSING if size is None else count_text(size)),
        (vocabulary.ESTIMATES_MEASURED, count_text(len(report.rows))),
    ]
    return Summary.build(
        title=vocabulary.INFLUENCE_TITLE.format(units=units),
        header=header,
        tables=[_table(report)],
        notes=_notes(report),
        reading=_reading(report, SPARSE_INFLUENCE_THRESHOLD),
        sections=_sections(report, SPARSE_INFLUENCE_THRESHOLD) if detail == "full" else (),
    )


def influence_repr(report: InfluenceReport) -> str:
    """A one-line repr of an influence report.

    Parameters
    ----------
    report : InfluenceReport
        The report.

    Returns
    -------
    str
        Such as ``InfluenceReport(ate, effective rows=185.5)``, or the count of estimates,
        the fewest effective rows and the count flagged when there are several.
    """
    _, units = _units(report)
    rows = report.rows
    if not rows:
        return short_repr("InfluenceReport", [("", "no estimates")])
    if len(rows) == 1:
        return short_repr(
            "InfluenceReport", [("", rows[0].name), (f"effective {units}", f"{rows[0].kish:.1f}")]
        )
    fewest = min(row.kish for row in rows)
    return short_repr(
        "InfluenceReport",
        [
            ("", f"{len(rows)} estimates"),
            (f"fewest effective {units}", f"{fewest:.1f}"),
            ("flagged", len(report.sparse)),
        ],
    )
