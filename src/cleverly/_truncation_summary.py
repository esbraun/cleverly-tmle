"""The printed summary of ``truncation_curve()`` (X89 T3).

The curve re-estimates at other limits on the probabilities that weight the estimate. The
summary prints one row per estimand and limits: the estimate, its change from the fitted
estimate and how much the limits capped, with the fitted limits marked, and a reading of the
largest change, in standard errors when the curve carries one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ._summary import Cell, Detail, Summary, SummarySection, SummaryTable, note, vocabulary
from ._summary._render import count_text, decimals_for, number_text
from .utils.frames import as_frame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .assessment import TruncationCurve

__all__ = ["curve_columns", "curve_repr", "curve_summary"]

#: The inferential spread column. A fit without inference publishes a diagnostic spread
#: under another name, which the reading does not scale by.
_STANDARD_ERROR = "std_err"


def curve_columns(curve: TruncationCurve) -> dict[str, list[Any]]:
    """The curve's columns as lists, whatever the frame's backend.

    Parameters
    ----------
    curve : TruncationCurve
        The curve.

    Returns
    -------
    dict of str to list
        Column name to its values.
    """
    return dict(as_frame(curve.to_frame()).to_dict(as_series=False))


def _lower(columns: dict[str, list[Any]]) -> list[float]:
    return [float(value) for value in columns.get("lower_bound", columns.get("bound", []))]


def _limit_decimals(values: list[float]) -> int:
    return max(3, decimals_for(value for value in values if value > 0))


def _limits(lower: float, upper: float, decimals: int) -> str:
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _largest(columns: dict[str, list[Any]], names: list[str]) -> tuple[int, float | None]:
    """The row whose estimate moves most, and that move in fitted standard errors."""
    changes = [abs(float(value)) for value in columns["delta_from_fitted"]]
    index = max(range(len(changes)), key=changes.__getitem__)
    spreads = columns.get(_STANDARD_ERROR)
    if spreads is None:
        return index, None
    fitted = {
        name: float(spread)
        for name, spread, is_fitted in zip(names, spreads, columns["is_fitted_bound"], strict=True)
        if is_fitted and _finite(spread)
    }
    name = names[index]
    if name not in fitted or fitted[name] <= 0:
        return index, None
    return index, changes[index] / fitted[name]


def curve_summary(curve: TruncationCurve, detail: Detail) -> Summary:
    """The printed summary of a truncation curve.

    Parameters
    ----------
    curve : TruncationCurve
        The curve.
    detail : {"short", "full"}
        ``"full"`` adds every limits pair tried, with both limits to six significant figures.

    Returns
    -------
    Summary
        The summary.
    """
    columns = curve_columns(curve)
    names = [str(name) for name in columns["estimand"]]
    lower = _lower(columns)
    upper = [float(value) for value in columns["upper_bound"]]
    fitted_lower = [float(value) for value in columns["fitted_lower_bound"]]
    fitted_upper = [float(value) for value in columns["fitted_upper_bound"]]
    limit_decimals = _limit_decimals(lower + fitted_lower)
    spreads = columns.get(_STANDARD_ERROR) or columns.get("plugin_std_err")
    decimals = decimals_for(spreads) if spreads is not None else 4
    # A change far below the estimate's spread still prints its first figures.
    moves = [abs(float(value)) for value in columns["delta_from_fitted"]]
    change_decimals = max(decimals, decimals_for(moves) if any(moves) else decimals)
    several = len(dict.fromkeys(names)) > 1
    longitudinal = "truncated_score_cells" in columns

    rows = []
    for index, name in enumerate(names):
        marked = columns["is_fitted_bound"][index]
        limits = _limits(lower[index], upper[index], limit_decimals)
        if longitudinal:
            truncated = columns["truncated_score_cells"][index]
            evaluated = columns["evaluated_score_cells"][index]
            capped = Cell(int(truncated), f"{count_text(truncated)} of {count_text(evaluated)}")
        else:
            fraction = float(columns["truncated_fraction"][index])
            capped = Cell(fraction, f"{fraction:.1%}")
        cells = (
            Cell(name, name),
            Cell(
                (lower[index], upper[index]),
                limits + (vocabulary.CURVE_FITTED_MARK if marked else ""),
            ),
            Cell(float(columns["psi"][index]), number_text(columns["psi"][index], decimals)),
            Cell(
                float(columns["delta_from_fitted"][index]),
                number_text(columns["delta_from_fitted"][index], change_decimals),
            ),
            capped,
        )
        rows.append(cells if several else cells[1:])
    table_columns = vocabulary.CURVE_COLUMNS if several else vocabulary.CURVE_COLUMNS[1:]

    pairs = list(dict.fromkeys(zip(lower, upper, strict=True)))
    if several:
        header = [
            (vocabulary.CURVE_ESTIMATES, count_text(len(dict.fromkeys(names)))),
            (vocabulary.CURVE_LIMITS_TRIED, count_text(len(pairs))),
        ]
    else:
        fitted_psi = columns["fitted_psi"][0]
        header = [
            (vocabulary.CURVE_ESTIMATE, names[0]),
            (
                vocabulary.CURVE_FITTED_LIMITS,
                _limits(fitted_lower[0], fitted_upper[0], limit_decimals),
            ),
            (vocabulary.CURVE_FITTED_ESTIMATE, number_text(fitted_psi, decimals)),
            (vocabulary.CURVE_LIMITS_TRIED, count_text(len(pairs))),
        ]

    index, ratio = _largest(columns, names)
    change = f"{abs(float(columns['delta_from_fitted'][index])):.3g}"
    readings = vocabulary.CURVE_READINGS
    key = ("spread" if ratio is not None else "plain") + ("_named" if several else "")
    reading = readings[key].format(
        change=change,
        ratio="-" if ratio is None else f"{ratio:.2g}",
        name=names[index],
    )

    notes = []
    if any(columns["is_fitted_bound"]):
        notes.append(note("fitted_limits", vocabulary.NOTE_TEXTS["fitted_limits"]))
    code = "cells_capped" if longitudinal else "units_capped"
    notes.append(note(code, vocabulary.NOTE_TEXTS[code]))
    if ratio is None and _STANDARD_ERROR not in columns:
        notes.append(note("no_spread", vocabulary.NOTE_TEXTS["no_spread"]))

    sections: tuple[SummarySection, ...] = ()
    if detail == "full":
        lines = tuple(
            vocabulary.CURVE_GRID_LINE.format(
                name=name,
                lower=f"{lower[index]:.6g}",
                upper=f"{upper[index]:.6g}",
                fitted=vocabulary.CURVE_FITTED_WORDS if columns["is_fitted_bound"][index] else "",
            )
            for index, name in enumerate(names)
        )
        sections = (SummarySection(vocabulary.CURVE_LIMITS_TRIED, lines),)
    return Summary.build(
        title=vocabulary.CURVE_TITLE,
        header=header,
        tables=[SummaryTable(table_columns, tuple(rows))],
        notes=notes,
        reading=reading,
        sections=sections,
    )


def curve_repr(curve: TruncationCurve) -> str:
    """A one-line repr of a truncation curve.

    Parameters
    ----------
    curve : TruncationCurve
        The curve.

    Returns
    -------
    str
        Such as ``TruncationCurve(8 rows, ate)``.
    """
    columns = curve_columns(curve)
    names = list(dict.fromkeys(str(name) for name in columns["estimand"]))
    shown = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
    return f"TruncationCurve({len(columns['estimand'])} rows, {shown})"
