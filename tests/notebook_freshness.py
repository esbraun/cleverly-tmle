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
equal.  An integer, such as a count of rows, must be equal.  Two decimals agree when they differ
by at most :data:`RELATIVE_TOLERANCE` (1e-4) of their size.  Cells that read a learner fit print
sums whose sixth significant figure changes with the machine's BLAS, CPU and Python build (CI
printed an effective size of 5325.445 where the stored run printed 5325.470), and this tolerance
absorbs that.  Two decimals also agree when they differ by at most one unit in the last place
that both print, and a cell may hold at most one such last-place difference beyond the
tolerance.  So a rounding change from a library release passes, while a result that moved, even
by one unit in several places at once, fails; a re-estimated effect moves in its second or third
figure, far beyond 1e-4.  When a dependency release moves an output, the response is a re-execution that reads the
difference and updates the prose, never a pin.

A cell whose learner amplifies that machine noise is declared in its callback's
``HARDWARE_SENSITIVE_CELLS``, with the reason.  Such a cell's decimals agree within
:data:`HARDWARE_TOLERANCE` (1e-2) of their size or one unit in the last place, with no limit on
how many differ.  Histogram gradient boosting is the known case: its histograms are float32 sums
whose rounding depends on the CPU's vector instructions, and a different split cascades through
the trees.  ``dose-response`` printed 359.4 on one CI runner where its stored run printed 360.1,
a 0.2% change, while the same commit reproduced exactly on 3.12 and 3.13 elsewhere.  Counts must
still be equal, and a move of a percent or more still fails.
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

#: The fraction of its size within which a decimal agrees (module docstring).
RELATIVE_TOLERANCE = 1e-4

#: The fraction of its size within which a decimal of a ``HARDWARE_SENSITIVE_CELLS`` cell agrees.
HARDWARE_TOLERANCE = 1e-2

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


def text_moves(stored: str, fresh: str, *, hardware_sensitive: bool = False) -> list[str]:
    """Each difference between a stored and a fresh cell text that is more than one rounding.

    Returns an empty list when the texts agree under the rule in the module docstring.
    ``hardware_sensitive`` applies the looser rule of a ``HARDWARE_SENSITIVE_CELLS`` cell.
    """
    tolerance = HARDWARE_TOLERANCE if hardware_sensitive else RELATIVE_TOLERANCE
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
        gap = abs(float(old) - float(new))
        if gap <= tolerance * max(abs(float(old)), abs(float(new))):
            continue
        unit = max(_last_place(old), _last_place(new))
        if gap <= unit * (1.0 + 1e-9):
            rounding += 1
            if rounding > 1 and not hardware_sensitive:
                moves.append(f"{old} -> {new} (a second last-place change in one cell)")
        else:
            moves.append(f"{old} -> {new}")
    return moves


def unreproduced(
    stored: Mapping[str, str],
    fresh: Mapping[str, str],
    exempt: Mapping[str, str],
    hardware_sensitive: Mapping[str, str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Cells whose fresh text moved, and ``exempt`` entries that no longer need their exemption.

    ``stored`` and ``fresh`` map a cell id to its text; only cells present in ``fresh`` (the
    cells the run executed) are compared.  An ``exempt`` cell is not reported as moved; it is
    reported as stale when it was executed and reproduces, or when it no longer exists.  A
    ``hardware_sensitive`` cell is compared under the looser rule, and is reported as stale only
    when it no longer exists, since it reproduces exactly on the machine that stored it.
    """
    loose = hardware_sensitive or {}
    moved = {}
    stale = []
    for cell, text in fresh.items():
        moves = text_moves(stored.get(cell, ""), text, hardware_sensitive=cell in loose)
        if cell in exempt:
            if not moves:
                stale.append(cell)
        elif moves:
            moved[cell] = moves
    stale.extend(sorted(cell for cell in exempt if cell not in stored))
    stale.extend(sorted(cell for cell in loose if cell not in stored))
    return moved, stale
