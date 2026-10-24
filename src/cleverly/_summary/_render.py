"""Text and HTML rendering of a summary, and the number formats its cells use."""

from __future__ import annotations

import html
import math
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from . import Note, SummaryTable, _Parts

#: The space between the two header columns.
_GAP = 2
#: The space between two table columns.
_COLUMN_GAP = 3


# ----------------------------------------------------------------- numbers


def decimals_for(spreads: Any) -> int:
    """The decimals an estimates table prints, from the smallest spread it shows.

    Enough decimals to show two significant figures of the smallest finite positive
    spread, kept between 2 and 6, so an estimate, its interval and its spread share one
    number of decimals. Four when no spread is finite.

    Parameters
    ----------
    spreads : iterable of float
        The spreads the table prints.

    Returns
    -------
    int
        The number of decimals.
    """
    values = [float(value) for value in spreads if _positive(value)]
    if not values:
        return 4
    smallest = min(values)
    return max(2, min(6, 1 - math.floor(math.log10(smallest))))


def _positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def number_text(value: Any, decimals: int) -> str:
    """A number at fixed decimals, ``-`` when it is missing or not finite.

    Parameters
    ----------
    value : float or None
        The number.
    decimals : int
        Decimals to print.

    Returns
    -------
    str
        The text.
    """
    if value is None:
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "-" if math.isnan(number) else ("inf" if number > 0 else "-inf")
    text = f"{number:.{decimals}f}"
    # A value that rounds to zero prints without a sign, so a summary does not show -0.00.
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text


def pvalue_text(value: Any) -> str:
    """A p-value at 4 decimals with the floor ``<0.0001``.

    Parameters
    ----------
    value : float or None
        The p-value.

    Returns
    -------
    str
        The text.
    """
    if value is None:
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "-"
    if number < 1e-4:
        return "<0.0001"
    return f"{number:.4f}"


def count_text(value: Any) -> str:
    """A count with thousands separators.

    Parameters
    ----------
    value : int or float
        The count.

    Returns
    -------
    str
        The text, such as ``"2,000"``. A fractional value prints one decimal.
    """
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}"


def share_text(value: Any) -> str:
    """A share at 3 decimals.

    Parameters
    ----------
    value : float
        The share.

    Returns
    -------
    str
        The text.
    """
    return f"{float(value):.3f}"


# ----------------------------------------------------------------- text


def render_text(parts: _Parts, width: int) -> str:
    """The plain-text form of a summary.

    Parameters
    ----------
    parts : _Parts
        The summary's parts.
    width : int
        The line width the header, notes and sections wrap at.

    Returns
    -------
    str
        The text.
    """
    tables = [_table_lines(table, width) for table in parts.tables]
    rule_width = max([width, *(len(line) for head, body in tables for line in (*head, *body))])
    lines = [parts.title.center(rule_width).rstrip(), "=" * rule_width]
    if parts.header:
        lines.extend(_header_lines(parts.header, rule_width, parts.split))
        lines.append("=" * rule_width)
    for index, (head, body) in enumerate(tables):
        if index:
            lines.append("=" * rule_width)
        lines.extend([*head, "-" * rule_width, *body])
    if tables:
        lines.append("=" * rule_width)
    if parts.reading:
        lines.extend(textwrap.wrap(parts.reading, width))
    lines.extend(_note_lines(parts.notes, width))
    for section in parts.sections:
        lines.append("")
        lines.append(section.title)
        lines.append("-" * len(section.title))
        for line in section.lines:
            lines.extend(textwrap.wrap(line, width, subsequent_indent="  ") or [""])
    return "\n".join(lines)


def _default_split(count: int) -> int:
    """How many header facts go in the left column when the summary names no split."""
    return (count + 1) // 2


def _header_lines(
    header: tuple[tuple[str, str], ...], width: int, split: int | None = None
) -> list[str]:
    half = (width - _GAP) // 2
    if split is None:
        split = _default_split(len(header))
    left_facts, right_facts = header[:split], header[split:]
    left = _column_lines(left_facts, half, align="left")
    right = _column_lines(right_facts, width - _GAP - half, align="right")
    rows = max(len(left), len(right))
    left += [""] * (rows - len(left))
    right += [""] * (rows - len(right))
    return [(f"{a:<{half}}" + " " * _GAP + b).rstrip() for a, b in zip(left, right, strict=True)]


def _column_lines(facts: tuple[tuple[str, str], ...], width: int, *, align: str) -> list[str]:
    """One header column. The left column aligns its values after its widest label; the
    right column right-aligns each value, so each fact has the room its own label leaves."""
    if not facts:
        return []
    label_width = max(len(label) for label, _ in facts) + 3
    lines: list[str] = []
    for label, value in facts:
        own = label_width if align == "left" else len(label) + 3
        room = max(8, width - own)
        chunks = textwrap.wrap(value, room) or [""]
        for index, chunk in enumerate(chunks):
            head = f"{label}:" if index == 0 else ""
            if align == "left":
                lines.append(f"{head:<{own}}{chunk}")
            else:
                lines.append(f"{head:<{own}}{chunk:>{width - own}}")
    return lines


def _is_text_column(table: SummaryTable, index: int) -> bool:
    """Whether a column holds words rather than numbers, which read left-aligned."""
    values = [row[index].value for row in table.rows]
    return bool(values) and all(isinstance(value, str) for value in values)


def _table_lines(table: SummaryTable, width: int) -> tuple[list[str], list[str]]:
    """The lines above the ``-`` rule (title and labels) and the row lines below it.

    Numbers never wrap. A last column of words wraps to keep the table within ``width``.
    """
    texts = [[cell.text for cell in row] for row in table.rows]
    count = len(table.columns)
    text_columns = [index > 0 and _is_text_column(table, index) for index in range(count)]
    widths = [
        max([len(label), *(len(row[index]) for row in texts)])
        for index, label in enumerate(table.columns)
    ]
    # Number columns get a statsmodels-like minimum, so a table of short numbers still reads.
    widths = [
        width_ if index == 0 or text_columns[index] else max(width_, 8)
        for index, width_ in enumerate(widths)
    ]
    last = count - 1
    if last > 0 and text_columns[last]:
        room = width - sum(widths[:last]) - _COLUMN_GAP * last
        widths[last] = max(min(widths[last], room), 20)

    def line(cells: list[str]) -> str:
        out = [f"{cells[0]:<{widths[0]}}"]
        for index, (cell, width_) in enumerate(zip(cells[1:], widths[1:], strict=True), 1):
            out.append(f"{cell:<{width_}}" if text_columns[index] else f"{cell:>{width_}}")
        return (" " * _COLUMN_GAP).join(out).rstrip()

    body: list[str] = []
    for row in texts:
        if last > 0 and text_columns[last] and len(row[last]) > widths[last]:
            chunks = textwrap.wrap(row[last], widths[last]) or [""]
            body.append(line([*row[:last], chunks[0]]))
            blank = [""] * last
            body.extend(line([*blank, chunk]) for chunk in chunks[1:])
        else:
            body.append(line(row))
    head = [table.title] if table.title else []
    return [*head, line(list(table.columns))], body


def _note_lines(notes: tuple[Note, ...], width: int) -> list[str]:
    lines: list[str] = []
    number = 0
    for kind, heading in (("warning", "Warnings:"), ("note", "Notes:")):
        chosen = [item for item in notes if item.kind == kind]
        if not chosen:
            continue
        lines.append(heading)
        for item in chosen:
            number += 1
            marker = f"[{number}] "
            lines.extend(
                textwrap.wrap(
                    item.text,
                    width,
                    initial_indent=marker,
                    subsequent_indent=" " * len(marker),
                )
            )
    return lines


# ----------------------------------------------------------------- HTML


def render_html(parts: _Parts) -> str:
    """The HTML form of a summary: a ``div`` of tables, an ordered list and sections.

    Every string is escaped, so a user's column name or estimand name renders as text.

    Parameters
    ----------
    parts : _Parts
        The summary's parts.

    Returns
    -------
    str
        The fragment.
    """
    esc = html.escape
    out = ['<div class="cleverly-summary">', f"<h4>{esc(parts.title)}</h4>"]
    if parts.header:
        split = _default_split(len(parts.header)) if parts.split is None else parts.split
        left, right = parts.header[:split], parts.header[split:]
        out.append("<table>")
        for index in range(max(len(left), len(right))):
            cells = []
            for side in (left, right):
                if index < len(side):
                    label, value = side[index]
                    cells.append(
                        f'<th style="text-align:left">{esc(label)}:</th>'
                        f'<td style="text-align:left">{esc(value)}</td>'
                    )
                else:
                    cells.append("<th></th><td></td>")
            out.append(f"<tr>{''.join(cells)}</tr>")
        out.append("</table>")
    for table in parts.tables:
        out.append("<table>")
        if table.title:
            out.append(f"<caption>{esc(table.title)}</caption>")
        heads = "".join(
            f'<th style="text-align:{"left" if index == 0 else "right"}">{esc(label)}</th>'
            for index, label in enumerate(table.columns)
        )
        out.append(f"<thead><tr>{heads}</tr></thead><tbody>")
        for row in table.rows:
            row_cells = "".join(
                f'<td style="text-align:{"left" if index == 0 else "right"}">{esc(cell.text)}</td>'
                for index, cell in enumerate(row)
            )
            out.append(f"<tr>{row_cells}</tr>")
        out.append("</tbody></table>")
    if parts.reading:
        out.append(f"<p>{esc(parts.reading)}</p>")
    number = 0
    for kind, heading in (("warning", "Warnings:"), ("note", "Notes:")):
        chosen = [item for item in parts.notes if item.kind == kind]
        if not chosen:
            continue
        out.append(f"<p>{heading}</p>")
        out.append(f'<ol start="{number + 1}">')
        for item in chosen:
            number += 1
            out.append(f"<li>{esc(item.text)}</li>")
        out.append("</ol>")
    for section in parts.sections:
        out.append(f"<p><b>{esc(section.title)}</b></p><ul>")
        out.extend(f"<li>{esc(line)}</li>" for line in section.lines)
        out.append("</ul>")
    out.append("</div>")
    return "\n".join(out)
