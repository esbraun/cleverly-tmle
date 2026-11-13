"""Style and layout checks for the figures of ``cleverly.plotting`` that use header strips.

The love plot and the zipper put their header and notes on strips with the axis turned
off, so the checks of ``test_plotting_core`` (which ask every axes for an x label) do not
apply to them as they stand. These are the same rules, read on the panels only, with two
additions: a panel may take its y label from the first panel of its row, and every visible
text must lie inside the figure once it is drawn. Each check has a control figure in the
test modules that must fail it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import to_hex

from cleverly.plotting import PALETTE, _core
from tests.unit.test_plotting_core import (
    _colors,
    _is_grey,
    assert_no_note_text,
    assert_short_title,
)


def assert_style(fig: Any) -> None:
    """Opaque white face, a title, labelled panels, palette colours, at most three per panel.

    Parameters
    ----------
    fig : Figure
        A figure a plot drew.
    """
    assert to_hex(fig.get_facecolor(), keep_alpha=True) == "#ffffffff"
    assert fig.axes[0].get_title(loc="left"), "the figure has no title"
    assert_no_note_text(fig)
    assert_short_title(fig)
    fig.canvas.draw()
    panels = [ax for ax in fig.axes if ax.axison]
    assert panels, "the figure has no panel"
    positions = [tuple(np.round(ax.get_position().bounds, 6)) for ax in panels]
    assert len(positions) == len(set(positions)), "two axes overlay each other (a twin axis)"
    rows: dict[float, list[Any]] = {}
    for ax in panels:
        rows.setdefault(round(ax.get_position().y0, 3), []).append(ax)
    for ax in panels:
        assert ax.get_xlabel() or ax.get_shared_x_axes().get_siblings(ax) != [ax], (
            "a panel with no x label"
        )
        ticks = [text.get_text() for text in ax.get_yticklabels() if text.get_visible()]
        row_label = any(
            other.get_ylabel()
            or (other.yaxis.get_tick_params()["labelleft"] and any(other.get_yticklabels()))
            for other in rows[round(ax.get_position().y0, 3)]
        )
        assert ax.get_ylabel() or any(ticks) or row_label, (
            "a panel with neither a y label, row labels, nor a labelled panel in its row"
        )
        used = set()
        for artist in [*ax.lines, *ax.collections, *ax.patches]:
            for color in np.atleast_2d(_colors(artist)):
                if color.size == 4 and color[3] == 0.0:
                    continue
                hexed = to_hex(color)
                if _is_grey(hexed):
                    continue
                assert hexed in PALETTE, f"{hexed} is not a palette colour"
                used.add(hexed)
        assert len(used) <= _core.MAX_SERIES


def assert_text_inside(fig: Any) -> None:
    """Every visible text of the figure lies inside it once drawn, to within a pixel.

    Parameters
    ----------
    fig : Figure
        A figure a plot drew.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for ax in fig.axes:
        texts = [*ax.texts, ax.title, ax._left_title]
        if ax.axison:
            texts += [ax.xaxis.label, ax.yaxis.label]
            # Only the ticks the axis draws: the tick list also keeps labels outside the view.
            for axis in (ax.xaxis, ax.yaxis):
                texts += [tick.label1 for tick in axis._update_ticks()]
        legend = ax.get_legend()
        if legend is not None:
            texts += list(legend.get_texts())
        for text in texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            extent = text.get_window_extent(renderer)
            where = f"{text.get_text()[:40]!r} at {extent.bounds}"
            assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, f"{where} runs off"
            assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, f"{where} runs off"
