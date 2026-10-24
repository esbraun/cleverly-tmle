"""The printed summary of the weight report (X89 T3).

The header names the weight column, how the fit treats it and how many rows' worth of
information the weights leave. The table always prints, so constant weights still show
their design effect of one. The numbers are read from the report.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

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
    from .weighting import WeightReport

__all__ = ["weight_repr", "weight_summary"]


def _number(value: float, text: str) -> Cell:
    if not math.isfinite(value):
        return Cell(None, vocabulary.MISSING)
    return Cell(float(value), text)


def _cell(key: str, value: Any) -> Cell:
    if key == "n_zero":
        return Cell(int(value), count_text(value))
    if key in {"top_1pct", "top_5pct"}:
        return _number(value, f"{value:.3f}")
    if key in {"min_ratio", "max_ratio"}:
        return _number(value, f"{value:.3g}")
    return _number(value, f"{value:.2f}")


def _table(report: WeightReport) -> SummaryTable:
    values = report.to_dict()
    rows = tuple(
        (Cell(label, label), _cell(key, values[key]))
        for key, label in vocabulary.WEIGHT_MEASURES.items()
    )
    return SummaryTable(("measure", "value"), rows)


def _reading(report: WeightReport) -> str | None:
    from .weighting import CONCENTRATED_DESIGN_EFFECT

    if not report.is_weighted:
        return vocabulary.WEIGHT_READINGS["constant"]
    if report.design_effect > CONCENTRATED_DESIGN_EFFECT:
        return vocabulary.WEIGHT_READINGS["dominant"].format(limit=CONCENTRATED_DESIGN_EFFECT)
    return None


def _notes(report: WeightReport) -> list[Note]:
    if not report.is_weighted:
        return []
    texts = vocabulary.NOTE_TEXTS
    notes = [note("weights_population", texts["weights_population"])]
    if report.estimated:
        notes.append(note("weights_estimated", texts["weights_estimated"]))
    else:
        notes.append(note("weights_fixed", texts["weights_fixed"]))
    if report.n_zero:
        notes.append(note("zero_weights", texts["zero_weights"].format(count=report.n_zero)))
    source = vocabulary.CTMLE_CURVE_SOURCE
    notes.append(
        note("outcome_adaptive_weighted", texts["outcome_adaptive_weighted"].format(source=source))
    )
    return notes


def weight_summary(report: WeightReport, detail: Detail) -> Summary:
    """The printed summary of a weight report.

    Parameters
    ----------
    report : WeightReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds the estimand the weights define and what the standard error reads.

    Returns
    -------
    Summary
        The summary.
    """
    effective = (
        f"{report.effective_n:,.1f}" if math.isfinite(report.effective_n) else vocabulary.MISSING
    )
    header = [
        (vocabulary.WEIGHT_COLUMN, report.name or vocabulary.MISSING),
        (vocabulary.WEIGHT_KIND, str(report.kind)),
        (vocabulary.ROWS, count_text(report.n)),
        (vocabulary.WEIGHTS_TREATED, vocabulary.WEIGHT_TREATMENT[report.estimated]),
        (vocabulary.EFFECTIVE_ROWS, effective),
    ]
    sections = []
    if detail == "full":
        lines = (
            vocabulary.WEIGHT_ESTIMAND_LINES
            if report.is_weighted
            else (vocabulary.WEIGHT_READINGS["constant"],)
        )
        sections.append(SummarySection(vocabulary.WEIGHT_ESTIMAND, lines))
    return Summary.build(
        title=vocabulary.WEIGHT_TITLE,
        header=header,
        header_split=3,
        tables=[_table(report)],
        notes=_notes(report),
        reading=_reading(report),
        sections=sections,
    )


def weight_repr(report: WeightReport) -> str:
    """A one-line repr of a weight report.

    Parameters
    ----------
    report : WeightReport
        The report.

    Returns
    -------
    str
        Such as ``WeightReport(w, effective rows=272.3 of 300)``.
    """
    return short_repr(
        "WeightReport",
        [
            ("", report.name or "constant weights"),
            ("effective rows", f"{report.effective_n:.1f} of {report.n}"),
        ],
    )
