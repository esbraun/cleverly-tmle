"""Compare a notebook's stored text outputs with a fresh in-process run of its cells.

The execution stamp (:mod:`tests.notebooks`) ties stored outputs to the code cells, and the
narration check ties the prose to the stored outputs.  Neither sees a library change that moves a
published number while every code cell keeps its bytes: the stored output then describes code
that no longer prints it.  The fast tier already runs every non-refit cell of each tutorial
notebook at its documented size, so this module compares what those cells print with what the
notebook stores, at no extra execution cost.

What is compared, per code cell: the ``stdout`` stream and each ``text/plain`` payload of a
display or a last expression, in order.  Left out: images; the ``text/plain`` figure repr stored
beside an image, whose pixel size follows ``figure.dpi``; ``stderr``, which carries warnings with
kernel file names and which pytest captures; and the ``cleverly <version>`` line, which
:func:`tests.unit.test_documentation_runtime.test_every_notebook_was_executed_on_the_current_version`
guards on its own.

How two texts compare: whitespace runs collapse, and the text between number tokens must be
equal.  An integer, such as a count of rows, must be equal.  Two decimals agree when they differ by at most one unit in the last place that both print,
and a cell may hold at most one such last-place difference.  So a single rounding change from a
library release passes, while a result that moved, even by one unit in several places at once,
fails.  When a dependency release moves an output, the response is a re-execution that reads the
difference and updates the prose, never a pin.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

#: A signed decimal or integer that touches no letter, digit, underscore or dot.  The lookarounds
#: keep a hex digest such as ``4f60ba6043a32341``, an argument such as ``C=1000000.0`` inside a
#: name, and a version such as ``1.5.4`` whole, so they compare as text.
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?(?![\w.]|\.\d)")

_WHITESPACE = re.compile(r"\s+")

#: The ``text/plain`` reprs of a matplotlib figure or axes.
FIGURE_REPRS = ("<Figure", "Figure(", "<Axes", "<matplotlib.")

#: The version line that every ``setup`` cell prints.
VERSION_LINE = re.compile(r"^cleverly \d+\.\d+\.\d+\S*$", re.MULTILINE)


def _joined(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def is_figure_repr(text: str) -> bool:
    """Whether ``text`` is the plain-text repr of a matplotlib figure or axes."""
    return text.lstrip().startswith(FIGURE_REPRS)


def stored_cell_text(outputs: Iterable[Mapping[str, Any]]) -> str:
    """The comparable text a kernel stored for one code cell: ``stdout`` and plain-text payloads."""
    parts = []
    for output in outputs:
        kind = output.get("output_type")
        if kind == "stream":
            if output.get("name") == "stdout":
                parts.append(_joined(output.get("text", "")))
            continue
        data = output.get("data", {})
        if "text/plain" not in data:
            continue
        text = _joined(data["text/plain"])
        if is_figure_repr(text) and any(mime.startswith("image/") for mime in data):
            continue
        parts.append(text + "\n")
    return "".join(parts)


def _normalized(text: str) -> str:
    return _WHITESPACE.sub(" ", VERSION_LINE.sub("cleverly <version>", text)).strip()


def _last_place(token: str) -> float:
    mantissa, _, exponent = token.lower().lstrip("+-").partition("e")
    places = len(mantissa.partition(".")[2])
    return float(f"1e{(int(exponent) if exponent else 0) - places}")


def _is_rounded(token: str) -> bool:
    """Whether ``token`` is a rounded real number rather than a count: it has a point or exponent."""
    return "." in token or "e" in token.lower()


def text_moves(stored: str, fresh: str) -> list[str]:
    """Each difference between a stored and a fresh cell text that is more than one rounding.

    Returns an empty list when the texts agree under the rule in the module docstring.
    """
    left, right = _normalized(stored), _normalized(fresh)
    if left == right:
        return []
    if NUMBER.split(left) != NUMBER.split(right):
        return ["the text around the numbers differs"]
    moves = []
    rounding = 0
    for old, new in zip(NUMBER.findall(left), NUMBER.findall(right), strict=True):
        if old == new:
            continue
        if not (_is_rounded(old) and _is_rounded(new)):
            moves.append(f"{old} -> {new}")
            continue
        unit = max(_last_place(old), _last_place(new))
        if abs(float(old) - float(new)) <= unit * (1.0 + 1e-9):
            rounding += 1
            if rounding > 1:
                moves.append(f"{old} -> {new} (a second last-place change in one cell)")
        else:
            moves.append(f"{old} -> {new}")
    return moves


def unreproduced(
    stored: Mapping[str, str], fresh: Mapping[str, str], exempt: Mapping[str, str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Cells whose fresh text moved, and ``exempt`` entries that no longer need their exemption.

    ``stored`` and ``fresh`` map a cell id to its text; only cells present in ``fresh`` (the
    cells the run executed) are compared.  An ``exempt`` cell is not reported as moved; it is
    reported as stale when it was executed and reproduces, or when it no longer exists.
    """
    moved = {}
    stale = []
    for cell, text in fresh.items():
        moves = text_moves(stored.get(cell, ""), text)
        if cell in exempt:
            if not moves:
                stale.append(cell)
        elif moves:
            moved[cell] = moves
    stale.extend(sorted(cell for cell in exempt if cell not in stored))
    return moved, stale
