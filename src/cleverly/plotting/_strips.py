"""A grid of panels under a full-width header strip.

The plots with more than one panel across (the love plot, the zipper) put the title,
subtitle and key on a strip above the grid. The strip and the grid sit in their own
subfigures, so long row labels in the first column cannot push the header right, and a long
header cannot narrow the panels. :func:`fit_height` then sizes the figure from what each
part holds: the panels keep the height the plot asked for, and the header gets exactly the
room its text needs.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: First guess, in inches, at the header height; :func:`fit_height` corrects it.
_HEAD_GUESS = 0.9
#: The least height of the strip's own (empty) axes, in inches.
_STRIP = 0.02
#: The attribute that marks a figure whose size the caller chose, which keeps its height.
_FIXED = "_cleverly_fixed_size"


def strip_figure(
    panels: int,
    columns: int,
    *,
    panel: tuple[float, float],
    left: float = 0.0,
    sharey: bool = False,
) -> tuple[Figure, Axes, list[Axes]]:
    """A grid of panels with a full-width strip above for the header.

    The strip has no axis lines, ticks or face; only the text drawn on it shows. The
    figure's axes are, in order, the header strip and the panels.

    Parameters
    ----------
    panels : int
        Panels to draw.
    columns : int
        Panels per row.
    panel : tuple of float
        One panel's width and the height of its plotting area, in inches.
    left : float, default=0.0
        Inches added to the figure's width for row labels left of the first column, so long
        labels do not narrow the panels.
    sharey : bool, default=False
        Whether the panels share their y axis.

    Returns
    -------
    tuple of Figure, Axes and list of Axes
        The figure, the header strip and the panels in reading order.
    """
    plt = _core.pyplot()
    rows = math.ceil(panels / columns)
    width = max(6.4, panel[0] * columns + left + 0.8)
    middle_height = (panel[1] + 0.7) * rows
    chosen = _core.label_options().figsize
    if chosen is not None:
        width, total = chosen
        middle_height = max(total - _HEAD_GUESS, 0.5)
    fig = plt.figure(figsize=(width, _HEAD_GUESS + middle_height), layout="constrained")
    setattr(fig, _FIXED, chosen is not None)
    fig.set_facecolor(_core.FACE)
    top, middle = fig.subfigures(2, 1, height_ratios=[_HEAD_GUESS, middle_height])
    head = top.add_subplot()
    grid = middle.add_gridspec(rows, columns)
    axes: list[Axes] = []
    for index in range(panels):
        share = axes[0] if sharey and axes else None
        axes.append(middle.add_subplot(grid[index // columns, index % columns], sharey=share))
    for part in (top, middle):
        part.set_facecolor(_core.FACE)
    for axis in axes:
        _core.style_axes(axis)
    head.set_axis_off()
    return fig, head, axes


def _inches(fig: Figure, pixels: float) -> float:
    return float(pixels) / float(fig.dpi)


def fit_height(fig: Figure, axes: list[Axes], columns: int, panel_height: float) -> None:
    """Resize the figure so the header gets the room its text needs.

    Each row of panels gets ``panel_height`` inches of plotting area. Call it last: the
    header is wrapped to the figure's width, which does not change, so a few layout passes
    settle the heights. A figure whose size the caller chose keeps it: the header takes the
    room its text needs and the panels the rest.

    Parameters
    ----------
    fig : Figure
        A figure from :func:`strip_figure`, fully drawn on.
    axes : list of Axes
        Its panels, in reading order.
    columns : int
        Panels per row.
    panel_height : float
        The plotting-area height of one row, in inches.
    """
    head = fig.axes[0]
    top: Any = head.get_figure()
    spec: Any = top._subplotspec.get_gridspec()
    first_column = axes[::columns]
    for _ in range(4):
        with warnings.catch_warnings():
            # A first guess too small for the header collapses the layout for one pass; the
            # text heights measured on it are still right, and the next pass corrects it.
            warnings.filterwarnings("ignore", message="constrained_layout not applied")
            fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        bounds = head.get_tightbbox(renderer)
        box = head.get_window_extent(renderer)
        text = _inches(fig, bounds.height - box.height) if bounds is not None else 0.0
        head_height = text + _STRIP + 0.12
        middle: Any = axes[0].get_figure()
        middle_height = _inches(fig, middle.bbox.height)
        drawn = sum(_inches(fig, axis.get_window_extent(renderer).height) for axis in first_column)
        middle_height += panel_height * len(first_column) - drawn
        if getattr(fig, _FIXED, False):
            total = float(fig.get_figheight())
            spec.set_height_ratios([head_height, max(total - head_height, 0.1)])
            return
        ratios = [head_height, middle_height]
        total = sum(ratios)
        if (
            abs(total - float(fig.get_figheight())) < 0.02
            and abs(drawn - panel_height * len(first_column)) < 0.02
        ):
            return
        spec.set_height_ratios(ratios)
        fig.set_figheight(total)


__all__ = ["fit_height", "strip_figure"]
