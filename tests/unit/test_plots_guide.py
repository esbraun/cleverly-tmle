"""The plots guide names every plot function, and only those, in its question table.

``docs/user-guide/plots.md`` borrows its questions from the ``cleverly.plotting`` module
docstring, which owns the list. A plot added to ``__all__`` without a row, or a row left
for a removed plot, fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import cleverly.plotting as plotting

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "plots.md"

#: Exports that are not plots: the notes reader, the style manager and the colours.
NOT_PLOTS = {"notes", "style", "PALETTE"}


def _table_functions() -> list[str]:
    text = GUIDE.read_text()
    section = text.split("## Pick the plot by question", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| [^|]+ \| `(\w+)` \|", section, flags=re.M)


def test_the_plots_guide_has_one_row_per_plot_function() -> None:
    rows = _table_functions()
    assert len(rows) == len(set(rows))
    assert set(rows) == set(plotting.__all__) - NOT_PLOTS


def test_the_plots_guide_borrows_the_module_docstrings_functions() -> None:
    # The module docstring owns the question list; both must name the same plots.
    owned = set(re.findall(r":func:`(\w+)`", plotting.__doc__ or "")) - NOT_PLOTS
    assert set(_table_functions()) == owned
