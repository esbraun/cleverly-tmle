"""The printed report of a :class:`~cleverly.targeting_curves.TargetingCurve`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import Cell, Summary, SummarySection, SummaryTable, check_detail, note, vocabulary
from ._render import count_text, decimals_for, number_text, share_text

__all__ = ["targeting_curve_summary"]

_INTERVAL = ("estimate", "std err", "lower", "upper")
_AREAS = {"autoc": "AUTOC", "qini": "Qini"}


def _source(curve: Any) -> str:
    if curve.route == "split":
        return (
            f"the curve refit on {count_text(curve.n_train)} training rows, evaluated on the "
            f"other {count_text(curve.n_eval)}"
        )
    return f"{curve.score}, a column fitted on other rows"


def _ranked(curve: Any) -> str:
    return "the refit curve" if curve.route == "split" else curve.score


def _withheld(row: dict[str, Any]) -> bool:
    """Whether the row's interval is withheld; ``TOC(1)`` is 0 by definition and has none."""
    anchor = row["quantity"] == "toc" and float(row["q"]) >= 1.0
    return not anchor and not math.isfinite(float(row["std_error"]))


def _label(row: dict[str, Any], text: str) -> str:
    return text + vocabulary.NO_INFERENCE_MARK if _withheld(row) else text


def _cell(value: Any, places: int) -> Cell:
    number = float(value)
    return Cell(number if math.isfinite(number) else None, number_text(number, places))


def _places(rows: Sequence[dict[str, Any]]) -> int:
    spreads = [float(row["std_error"]) for row in rows]
    return decimals_for(np.asarray([abs(v) for v in spreads if math.isfinite(v)] or [np.nan]))


def _area_table(curve: Any, level: str, places: int) -> SummaryTable:
    rows = []
    for quantity in ("autoc", "qini"):
        row = curve.area(quantity)
        rows.append(
            (
                Cell(quantity, _label(row, _AREAS[quantity])),
                *(_cell(row[key], places) for key in ("estimate", "std_error", "lower", "upper")),
            )
        )
    return SummaryTable(("area", *_INTERVAL), tuple(rows), title=f"Areas with {level} intervals")


def _toc_table(curve: Any, level: str, places: int) -> SummaryTable:
    columns = ("q", *_INTERVAL, "band lower", "band upper", "rows in top")
    keys = ("estimate", "std_error", "lower", "upper", "band_lower", "band_upper")
    rows = []
    for row in curve.rows("toc"):
        q = float(row["q"])
        rows.append(
            (
                Cell(q, _label(row, f"{q:g}")),
                *(_cell(row[key], places) for key in keys),
                Cell(float(row["rows_in_top"]), count_text(float(row["rows_in_top"]))),
            )
        )
    return SummaryTable(columns, tuple(rows), title=f"TOC with {level} intervals and band")


def _line(row: dict[str, Any], label: str, places: int) -> str:
    estimate = number_text(float(row["estimate"]), places)
    if _withheld(row):
        return f"{label}: {estimate} {vocabulary.NO_INFERENCE_MARK}"
    if not math.isfinite(float(row["std_error"])):
        return f"{label}: {estimate}"
    return (
        f"{label}: {estimate} (std err {number_text(float(row['std_error']), places)}; "
        f"{number_text(float(row['lower']), places)} to {number_text(float(row['upper']), places)})"
    )


def _sections(curve: Any, places: int) -> list[SummarySection]:
    sections = [
        SummarySection(
            "Cumulative gain",
            tuple(_line(row, f"q = {row['q']:g}", places) for row in curve.rows("gain")),
        )
    ]
    groups = curve.rows("group_effect")
    if groups:
        lines = []
        previous = 0.0
        for row in groups:
            mean = number_text(float(row["mean_score"]), places)
            lines.append(
                _line(row, f"ranks {previous:g} to {row['q']:g}, mean score {mean}", places)
            )
            previous = float(row["q"])
        sections.append(SummarySection("Group effects", tuple(lines)))
    names = [curve.score]
    if curve.compare is not None:
        difference = f"{curve.score} - {curve.compare}"
        names += [curve.compare, difference]
        lines = [
            _line(curve.area(quantity, name), f"{_AREAS[quantity]} of {name}", places)
            for name in (curve.compare, difference)
            for quantity in ("autoc", "qini")
        ]
        lines += [
            _line(row, f"TOC of {difference} at q = {row['q']:g}", places)
            for row in curve.rows("toc", difference)
        ]
        sections.append(SummarySection(f"Comparison with {curve.compare}", tuple(lines)))
    lines = []
    for name in names:
        for quantity in ("autoc", "qini"):
            p = float(curve.area(quantity, name)["p_value"])
            text = f"{p:.3g}" if math.isfinite(p) else vocabulary.MISSING
            lines.append(f"{_AREAS[quantity]} of {name}: {text}")
    sections.append(SummarySection("P-values against 0", tuple(lines)))
    return sections


def _notes(curve: Any, level: str) -> list[Any]:
    notes = []
    if curve.route == "split":
        notes.append(
            note(
                "targeting_split",
                vocabulary.NOTE_TEXTS["targeting_split"].format(
                    n_train=count_text(curve.n_train), n_eval=count_text(curve.n_eval)
                ),
            )
        )
    else:
        notes.append(
            note(
                "targeting_external",
                vocabulary.NOTE_TEXTS["targeting_external"].format(column=curve.score),
            )
        )
    if math.isfinite(float(curve.critical_value)):
        notes.append(
            note(
                "targeting_band",
                vocabulary.NOTE_TEXTS["targeting_band"].format(
                    level=level,
                    points=count_text(sum(1 for q in curve.grid if q < 1.0)),
                    critical=f"{curve.critical_value:.3g}",
                ),
            )
        )
    # The gain rows (the frame, and the full report) carry the gain's own interval (plan 5.1).
    notes.append(note("targeting_gain_interval", vocabulary.NOTE_TEXTS["targeting_gain_interval"]))
    shown = curve.rows("toc") + curve.rows("autoc") + curve.rows("qini")
    if any(_withheld(row) for row in shown):
        notes.append(
            note(
                "targeting_withheld",
                vocabulary.NOTE_TEXTS["targeting_withheld"].format(
                    mark=vocabulary.NO_INFERENCE_MARK
                ),
            )
        )
    if curve.status == "experimental":
        reason = " ".join(
            part[:1].upper() + part[1:] + "." for part in curve.status_reason.split("; ") if part
        )
        notes.append(
            note(
                "targeting_experimental",
                vocabulary.NOTE_TEXTS["targeting_experimental"].format(reason=reason),
            )
        )
    return notes


def targeting_curve_summary(curve: Any, *, detail: str = "short") -> Summary:
    """Build the summary of a targeting curve.

    Parameters
    ----------
    curve : TargetingCurve
        The curve.
    detail : {"short", "full"}
        ``"full"`` adds the gain, the group effects, the comparison and the p-values of the
        areas.

    Returns
    -------
    Summary
        The report.
    """
    detail = check_detail(detail)
    level = share_text(1.0 - curve.alpha)
    header = [
        (
            vocabulary.QUESTION,
            f"how much treating by {_ranked(curve)} first gains over treating at random",
        ),
        (vocabulary.ESTIMAND, vocabulary.TARGETING_ESTIMAND),
        (vocabulary.TARGETING_SOURCE, _source(curve)),
        (vocabulary.ROWS, count_text(curve.n_eval)),
        (
            vocabulary.METHOD,
            f"TMLE, half-sample bootstrap with {count_text(curve.n_half_samples)} draws",
        ),
        (vocabulary.TARGETING_ATE, number_text(curve.ate, 4)),
        (vocabulary.CURVE_STATUS, curve.status),
    ]
    if curve.compare is not None:
        header.insert(1, (vocabulary.TARGETING_COMPARE, curve.compare))
    places = _places(curve.rows("toc") + curve.rows("autoc") + curve.rows("qini"))
    return Summary.build(
        title=vocabulary.TARGETING_TITLE,
        header=header,
        tables=[_area_table(curve, level, places), _toc_table(curve, level, places)],
        notes=_notes(curve, level),
        sections=_sections(curve, places) if detail == "full" else [],
    )
