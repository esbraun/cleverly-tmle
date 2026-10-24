"""Text and HTML rendering of a summary, and the number formats its cells use."""

from __future__ import annotations

import html
import math
import re
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
    # A value with a word too long for its half, such as a long column name, takes a row of
    # the full width below the two columns, so no name is broken across lines.
    wide = tuple(
        fact
        for fact in header
        if _longest_word(fact[1]) > _room(fact, header[:split] if fact in left_facts else (), half)
    )
    left_facts = tuple(fact for fact in left_facts if fact not in wide)
    right_facts = tuple(fact for fact in right_facts if fact not in wide)
    left = _column_lines(left_facts, half, align="left")
    right = _column_lines(right_facts, width - _GAP - half, align="right")
    rows = max(len(left), len(right))
    left += [""] * (rows - len(left))
    right += [""] * (rows - len(right))
    lines = [(f"{a:<{half}}" + " " * _GAP + b).rstrip() for a, b in zip(left, right, strict=True)]
    return lines + _column_lines(wide, width, align="left", whole_words=True)


def _longest_word(value: str) -> int:
    return max((len(word) for word in value.split()), default=0)


def _name_pieces(name: str) -> list[str]:
    """A row name cut where it may wrap: at its spaces and after each ``[``.

    A time label such as ``@ t=2`` stays whole.
    """
    return [piece for piece in re.split(r"(\s+)(?!t=)|(?<=\[)", name) if piece]


def _wrap_name(name: str, width: int) -> list[str]:
    """A row name wrapped to ``width`` at its spaces and after each ``[``."""
    lines: list[str] = []
    current = ""
    space = False
    for piece in _name_pieces(name):
        if piece.isspace():
            space = True
            continue
        candidate = current + (" " if space and current else "") + piece
        if current and len(candidate) > width:
            lines.append(current)
            current = piece
        else:
            current = candidate
        space = False
    if current:
        lines.append(current)
    return lines or [""]


def _room(fact: tuple[str, str], column: tuple[tuple[str, str], ...], width: int) -> int:
    """The characters a fact's value has in its half of the header."""
    labels = [label for label, _ in column] or [fact[0]]
    own = max(len(label) for label in labels) + 3
    return max(8, width - own)


def _column_lines(
    facts: tuple[tuple[str, str], ...], width: int, *, align: str, whole_words: bool = False
) -> list[str]:
    """One header column. The left column aligns its values after its widest label; the
    right column right-aligns each value, so each fact has the room its own label leaves."""
    if not facts:
        return []
    label_width = max(len(label) for label, _ in facts) + 3
    lines: list[str] = []
    for label, value in facts:
        own = label_width if align == "left" else len(label) + 3
        room = max(8, width - own)
        chunks = textwrap.wrap(value, room, break_long_words=not whole_words) or [""]
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

    Numbers never wrap. A last column of words wraps to keep the table within ``width``,
    and so does a first column of row names when the table still passes it, at the
    spaces of a long name such as ``ate_regimen[always vs never]`` and after its ``[``.
    """
    texts = [[cell.text for cell in row] for row in table.rows]
    count = len(table.columns)
    text_columns = [index > 0 and _is_text_column(table, index) for index in range(count)]
    natural = [
        max([len(label), *(len(row[index]) for row in texts)])
        for index, label in enumerate(table.columns)
    ]
    last = count - 1
    wraps = last > 0 and text_columns[last]

    def fitted(gap: int, minimum: int, floor: int) -> list[int]:
        # Number columns get a statsmodels-like minimum, so a table of short numbers reads.
        widths_ = [
            width_ if index == 0 or text_columns[index] else max(width_, minimum)
            for index, width_ in enumerate(natural)
        ]
        if wraps:
            room = width - sum(widths_[:last]) - gap * last
            widths_[last] = min(widths_[last], max(room, floor))
        return widths_

    # Narrow a table that would pass the width: first drop the number minimum, then close
    # the gap, then let a last column of words wrap narrower. A table that still passes it
    # wraps its row names, and one that passes it after that prints wider, and the rules
    # widen with it.
    steps = ((_COLUMN_GAP, 8, 20), (_COLUMN_GAP, 0, 20), (_COLUMN_GAP - 1, 0, 20))
    for gap, minimum, floor in (*steps, (_COLUMN_GAP - 1, 0, 12)):
        widths = fitted(gap, minimum, floor)
        if sum(widths) + gap * last <= width:
            break
    wrapped: set[int] = {last} if wraps else set()
    if last > 0 and sum(widths) + gap * last > width:
        words = max((len(piece) for row in texts for piece in _name_pieces(row[0])), default=0)
        room = width - sum(widths[1:]) - gap * last
        if words < widths[0] and room < widths[0]:
            widths[0] = max(room, words, len(table.columns[0]))
            wrapped.add(0)

    def line(cells: list[str]) -> str:
        out = [f"{cells[0]:<{widths[0]}}"]
        for index, (cell, width_) in enumerate(zip(cells[1:], widths[1:], strict=True), 1):
            out.append(f"{cell:<{width_}}" if text_columns[index] else f"{cell:>{width_}}")
        return (" " * gap).join(out).rstrip()

    body: list[str] = []
    for row in texts:
        chunks = [
            [cell]
            if index not in wrapped or len(cell) <= widths[index]
            else _wrap_name(cell, widths[index])
            if index == 0
            else (textwrap.wrap(cell, widths[index]) or [""])
            for index, cell in enumerate(row)
        ]
        depth = max(len(column) for column in chunks)
        for depth_index in range(depth):
            body.append(
                line(
                    [column[depth_index] if depth_index < len(column) else "" for column in chunks]
                )
            )
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
        align = [
            "left" if index == 0 or _is_text_column(table, index) else "right"
            for index in range(len(table.columns))
        ]
        heads = "".join(
            f'<th style="text-align:{align[index]}">{esc(label)}</th>'
            for index, label in enumerate(table.columns)
        )
        out.append(f"<thead><tr>{heads}</tr></thead><tbody>")
        for row in table.rows:
            row_cells = "".join(
                f'<td style="text-align:{align[index]}">{esc(cell.text)}</td>'
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
