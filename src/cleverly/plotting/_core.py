"""The style, layout and honesty helpers every plot shares.

Nothing here imports matplotlib at module import time: :func:`pyplot` does, the first time a
plot runs, so ``import cleverly`` and ``import cleverly.plotting`` stay free of it.

Every property a plot relies on is set directly on the figure, axes and artists. Settings
read from ``rcParams`` at draw or save time would be lost once an ``rc_context`` exits, and
the caller's global settings are never changed. :func:`style` is the one context manager, for
callers who want their own figures to match.
"""

from __future__ import annotations

import math
import textwrap
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from .._summary._render import decimals_for, number_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The categorical palette, in its fixed slot order: blue, orange, aqua. These three slots
#: pass every colour-vision check for every pair against the white figure face; the
#: validator's output is ``palette-validation.txt`` beside this module. Slot 3 sits below 3:1
#: contrast on white, so a series in it always carries a direct label.
PALETTE: Final[tuple[str, str, str]] = ("#2a78d6", "#eb6834", "#1baf7a")
#: At most this many coloured entities share one axes; more become small multiples.
MAX_SERIES: Final = 3

#: Text, reference and grid inks. Text never wears a series colour.
INK: Final = "#1f1f1e"
SECONDARY_INK: Final = "#52514e"
REFERENCE: Final = "#8a8985"
GRID: Final = "#e8e7e3"
CONTEXT: Final = "#c9c8c3"
FACE: Final = "#ffffff"

#: The ``gid`` each kind of artist carries, which the tests and later plots read.
GID_INTERVAL: Final = "cleverly-interval"
GID_BAND: Final = "cleverly-band"
GID_POINT: Final = "cleverly-point"
GID_RANGE: Final = "cleverly-range"
GID_REFERENCE: Final = "cleverly-reference"
GID_MARKER: Final = "cleverly-marker"
GID_SUBTITLE: Final = "cleverly-subtitle"
GID_NOTES: Final = "cleverly-notes"
GID_SERIES_LABEL: Final = "cleverly-series-label"
GID_SWATCH: Final = "cleverly-swatch"
GID_TRUTH: Final = "cleverly-truth"
#: The figure-level title and key that :func:`figure_header` writes.
GID_TITLE: Final = "cleverly-title"
GID_KEY: Final = "cleverly-key"

#: The key entry of a known true value, which only simulated data have.
TRUTH_LABEL: Final = "True value (simulated data only)"

#: Point sizes of the text roles.
TITLE_SIZE: Final = 12.0
SUBTITLE_SIZE: Final = 10.0
LABEL_SIZE: Final = 10.0
TICK_SIZE: Final = 9.0
NOTE_SIZE: Final = 8.5

#: The default figure size, in inches.
FIGSIZE: Final = (7.0, 4.5)
#: Alpha of a pointwise ribbon and of a simultaneous band behind it.
INTERVAL_ALPHA: Final = 0.25
BAND_ALPHA: Final = 0.12

#: What the caller is told when matplotlib is missing.
INSTALL_HINT: Final = (
    "cleverly.plotting needs matplotlib, which is not installed. Install it with "
    'pip install "cleverly[plot]".'
)


def pyplot() -> Any:
    """Return ``matplotlib.pyplot``, or raise an ``ImportError`` that names the extra.

    Returns
    -------
    module
        ``matplotlib.pyplot``.

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - exercised by the minimal-install job
        raise ImportError(INSTALL_HINT) from error
    return plt


@contextmanager
def style() -> Iterator[None]:
    """Draw the caller's own matplotlib figures in the cleverly style.

    The plots in :mod:`cleverly.plotting` set their style on each figure and need no
    context. This manager is for a caller who wants a figure of their own to match them.
    It changes ``rcParams`` only while the block runs.

    Yields
    ------
    None
        Nothing; draw inside the block.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from cleverly.plotting import style
    >>> with style():
    ...     fig, ax = plt.subplots()
    ...     line, = ax.plot([0, 1], [0, 1])
    >>> line.get_color()
    '#2a78d6'
    >>> plt.close(fig)
    """
    plt = pyplot()
    from cycler import cycler

    settings = {
        "axes.prop_cycle": cycler(color=list(PALETTE)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SECONDARY_INK,
        "axes.labelcolor": INK,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "text.color": INK,
        "figure.facecolor": FACE,
        "axes.facecolor": FACE,
        "savefig.facecolor": FACE,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": TICK_SIZE,
    }
    with plt.rc_context(settings):
        yield


# ----------------------------------------------------------------------- figures and axes


def new_figure(
    nrows: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    height_ratios: Sequence[float] | None = None,
    sharex: bool = False,
    ncols: int = 1,
) -> tuple[Figure, list[Axes]]:
    """A figure in the cleverly style with its axes, top to bottom then left to right.

    Parameters
    ----------
    nrows : int, default=1
        Rows of panels.
    figsize : tuple of float or None, default=None
        Width and height in inches; :data:`FIGSIZE` when omitted.
    height_ratios : sequence of float or None, default=None
        Relative panel heights.
    sharex : bool, default=False
        Whether the panels share one x axis.
    ncols : int, default=1
        Columns of panels.

    Returns
    -------
    tuple of Figure and list of Axes
        The figure and its axes in reading order.
    """
    plt = pyplot()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or FIGSIZE,
        sharex=sharex,
        squeeze=False,
        layout="constrained",
        gridspec_kw={"height_ratios": list(height_ratios)} if height_ratios else None,
    )
    fig.set_facecolor(FACE)
    flat = [axis for row in axes for axis in row]
    for axis in flat:
        style_axes(axis)
    return fig, flat


def figure_for(ax: Axes | None, **kwargs: Any) -> tuple[Figure, Axes]:
    """The caller's axes and its figure, or a new one-panel figure.

    Parameters
    ----------
    ax : Axes or None
        The caller's axes.
    **kwargs
        Passed to :func:`new_figure` when ``ax`` is ``None``.

    Returns
    -------
    tuple of Figure and Axes
        The figure and the axes to draw on.
    """
    if ax is None:
        fig, (axis,) = new_figure(**kwargs)
        return fig, axis
    style_axes(ax)
    return ax.figure, ax  # type: ignore[return-value]


def style_axes(ax: Axes, *, grid: str = "y") -> None:
    """Set the face, spines, ticks and grid of one axes.

    Parameters
    ----------
    ax : Axes
        The axes.
    grid : {"y", "x", "none"}, default="y"
        Which light grid to draw.
    """
    ax.set_facecolor(FACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SECONDARY_INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=SECONDARY_INK, labelsize=TICK_SIZE, labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.grid(False)
    if grid in ("x", "y"):
        ax.grid(True, axis="y" if grid == "y" else "x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def wrap(text: str, ax: Axes, size: float) -> str:
    """Wrap ``text`` to the width of the figure that holds ``ax``, at ``size`` points.

    Parameters
    ----------
    text : str
        The text.
    ax : Axes
        Any axes of the figure.
    size : float
        Font size in points.

    Returns
    -------
    str
        The text with line breaks.
    """
    figure: Any = ax.figure
    while not hasattr(figure, "get_figwidth"):  # a subfigure: read its root figure
        figure = figure.figure
    width_inches = float(figure.get_figwidth()) * 0.94
    characters = max(30, int(width_inches * 72.0 / (size * 0.52)))
    return "\n".join(
        textwrap.fill(part, characters, break_on_hyphens=False) for part in text.split("\n")
    )


def header(
    ax: Axes,
    title: str,
    subtitle: str | None = None,
    *,
    key_sources: Sequence[Axes] | None = None,
    key_columns: int = 3,
) -> None:
    """Write the title (the question) and the subtitle (the estimand or headline number).

    The title goes on ``ax``, left aligned, the subtitle under it, and the key between the
    subtitle and the axes, so the layout engine reserves room for all three and the key never
    covers data. The key is drawn here by :func:`legend` unless one is already on ``ax``. Pass
    the top panel of a multi-panel plot, after every labelled artist is drawn.

    Parameters
    ----------
    ax : Axes
        The top axes.
    title : str
        The title.
    subtitle : str or None, default=None
        One line, wrapped when long.
    key_sources : sequence of Axes or None, default=None
        The panels whose labelled artists the key lists; ``ax`` alone when omitted.
    key_columns : int, default=3
        The most columns the key may take, for a plot with text beside its axes.
    """
    if ax.get_legend() is None:
        legend(ax, sources=key_sources, max_columns=key_columns)
    key = ax.get_legend()
    key_height = 0.0
    if key is not None:
        names = [text.get_text() for text in key.get_texts()]
        rows = math.ceil(len(names) / _key_columns(ax, names, key_columns))
        key_height = rows * TICK_SIZE * 1.45 + 4.0
    sub_lines = 0
    if subtitle:
        subtitle = wrap(subtitle, ax, SUBTITLE_SIZE)
        sub_lines = subtitle.count("\n") + 1
    pad = 6.0 + key_height + sub_lines * SUBTITLE_SIZE * 1.35
    ax.set_title(
        wrap(title, ax, TITLE_SIZE),
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=INK,
        pad=pad,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 4.0 + key_height),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=SUBTITLE_SIZE,
            color=SECONDARY_INK,
            gid=GID_SUBTITLE,
        )


def subtitle_text(ax: Axes) -> str | None:
    """The subtitle :func:`header` wrote on ``ax``, or :func:`figure_header` on its figure.

    The subtitle is returned as one line, or ``None`` when there is none.

    Parameters
    ----------
    ax : Axes
        The top axes of a plot.

    Returns
    -------
    str or None
        The subtitle with its line breaks joined by spaces.
    """
    for text in [*ax.texts, *_root(ax).texts]:
        if text.get_gid() == GID_SUBTITLE:
            return " ".join(text.get_text().split())
    return None


def notes(ax: Axes, lines: Sequence[str]) -> None:
    """Write the note lines under the bottom axes, below its axis label.

    Each line is one statement the reader needs to read the figure correctly: what an
    interval is, why one is missing, that a quantity is descriptive or experimental. The
    block starts at the figure's left edge and is wrapped to the figure's width, so long row
    labels cannot push it off the page.

    Parameters
    ----------
    ax : Axes
        The bottom axes.
    lines : sequence of str
        The notes, in order. Duplicates are dropped.
    """
    distinct = list(dict.fromkeys(line for line in lines if line))
    if not distinct:
        return
    text = "\n".join(wrap(line, ax, NOTE_SIZE) for line in distinct)
    below = ax.xaxis.label if ax.get_xlabel() else ax
    ax.annotate(
        text,
        xy=(0.015, 0.0),
        xycoords=("figure fraction", below),
        xytext=(0.0, -6.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=NOTE_SIZE,
        color=SECONDARY_INK,
        gid=GID_NOTES,
        annotation_clip=False,
    )


def note_text(fig: Figure) -> str:
    """Every note line of a figure, joined by spaces.

    Parameters
    ----------
    fig : Figure
        A figure a plot drew.

    Returns
    -------
    str
        The notes, or an empty string.
    """
    texts = [*(text for axis in fig.axes for text in axis.texts), *fig.texts]
    parts = [text.get_text() for text in texts if text.get_gid() == GID_NOTES]
    return " ".join(" ".join(parts).split())


# ----------------------------------------------------------------------- figure-level text
#
# header() and notes() write on one axes, so the layout engine counts the text's width
# against that axes' column. In a figure with several columns of panels a long title or
# note then squeezes its column and pushes the next one off the page. The figure-level
# pair below writes the same roles as figure texts anchored at the figure's left edge,
# leaves them out of the layout, and reserves their height through the layout rect.

#: Left edge of figure-level text, in figure fraction (the edge notes() uses too).
_FIGURE_LEFT: Final = 0.015
#: Space between the figure's edge and its text, and between text blocks, in points.
_EDGE_PAD: Final = 6.0
_GAP: Final = 4.0


def _root(ax: Any) -> Any:
    """The root figure of an axes or a subfigure."""
    figure: Any = ax.figure if hasattr(ax, "figure") else ax
    while not hasattr(figure, "get_figwidth"):
        figure = figure.figure
    return figure


def _text_height(text: str, size: float) -> float:
    """The height in points of ``text`` at ``size``, at matplotlib's 1.2 line spacing."""
    lines = text.count("\n") + 1
    return size * (1.0 + 1.2 * (lines - 1)) + 0.3 * size


def _reserve(fig: Figure, *, top: float | None = None, bottom: float | None = None) -> None:
    """Keep the top or bottom band of ``fig`` free of panels, in figure fraction.

    The band of the side not named is kept as it is, so the header and the notes can be
    written in either order.
    """
    engine: Any = fig.get_layout_engine()
    settings = engine.get() if engine is not None else {}
    rect = settings.get("rect")
    if rect is not None and type(engine).__name__ == "ConstrainedLayoutEngine":
        left, low, width, height = (float(value) for value in rect)
        high = low + height
        low = bottom if bottom is not None else low
        high = 1.0 - top if top is not None else high
        engine.set(rect=(left, low, width, high - low))
    elif rect is not None:  # the tight layout engine reads (left, bottom, right, top)
        left, low, right, high = (float(value) for value in rect)
        low = bottom if bottom is not None else low
        high = 1.0 - top if top is not None else high
        engine.set(rect=(left, low, right, high))
    else:
        settings = {}
        if top is not None:
            settings["top"] = 1.0 - top
        if bottom is not None:
            settings["bottom"] = bottom
        fig.subplots_adjust(**settings)


def figure_header(
    fig: Figure,
    title: str,
    subtitle: str | None = None,
    *,
    key_sources: Sequence[Axes] | None = None,
) -> None:
    """Write the title, subtitle and key of a figure with several columns of panels.

    The figure-level counterpart of :func:`header`, with the same roles and inks. The title,
    the subtitle and the key (the labelled artists of ``key_sources``, as :func:`legend`
    collects them) are written at the figure's left edge, wrapped to the figure's width and
    left out of the layout, and the layout engine's ``rect`` is lowered at the top by their
    height, so no panel is squeezed by text wider than its column. Use it, with
    :func:`figure_notes`, for a figure the plot created; :func:`header` stays the call for
    one axes. Call it after every labelled artist is drawn.

    Parameters
    ----------
    fig : Figure
        The figure; a constrained-layout figure from :func:`new_figure`, or any figure.
    title : str
        The title (the question).
    subtitle : str or None, default=None
        The estimand or the headline number.
    key_sources : sequence of Axes or None, default=None
        The panels whose labelled artists the key lists; every axes of ``fig`` when omitted.
    """
    height = float(fig.get_figheight()) * 72.0
    cursor = _EDGE_PAD

    def place(text: str, size: float, **style: Any) -> float:
        wrapped = wrap(text, fig, size)  # type: ignore[arg-type]  # a figure has .figure too
        fig.text(
            _FIGURE_LEFT,
            1.0 - cursor / height,
            wrapped,
            ha="left",
            va="top",
            fontsize=size,
            **style,
        ).set_in_layout(False)
        return _text_height(wrapped, size)

    cursor += place(title, TITLE_SIZE, fontweight="bold", color=INK, gid=GID_TITLE) + _GAP
    if subtitle:
        cursor += place(subtitle, SUBTITLE_SIZE, color=SECONDARY_INK, gid=GID_SUBTITLE) + _GAP
    handles, labels = _key_entries(key_sources if key_sources is not None else fig.axes)
    if handles:
        columns = min(len(handles), 3)
        key = fig.legend(
            handles,
            labels,
            handler_map=_key_handlers(handles),
            loc="upper left",
            bbox_to_anchor=(_FIGURE_LEFT, 1.0 - cursor / height),
            bbox_transform=fig.transFigure,
            ncol=columns,
            frameon=False,
            fontsize=TICK_SIZE,
            labelcolor=INK,
            borderaxespad=0.0,
            borderpad=0.0,
            handlelength=1.8,
            columnspacing=1.4,
        )
        key.set_in_layout(False)
        key.set_gid(GID_KEY)
        cursor += math.ceil(len(handles) / columns) * TICK_SIZE * 1.45 + _GAP
    _reserve(fig, top=(cursor + _GAP) / height)


def figure_notes(fig: Figure, lines: Sequence[str]) -> None:
    """Write the note lines at the bottom of a figure with several columns of panels.

    The figure-level counterpart of :func:`notes`: the same roles and ink, written at the
    figure's left edge, wrapped to the figure's width and left out of the layout, with the
    layout engine's ``rect`` raised at the bottom by their height. :func:`note_text` reads
    them.

    Parameters
    ----------
    fig : Figure
        The figure.
    lines : sequence of str
        The notes, in order. Duplicates are dropped.
    """
    distinct = list(dict.fromkeys(line for line in lines if line))
    if not distinct:
        return
    height = float(fig.get_figheight()) * 72.0
    characters = max(30, int(float(fig.get_figwidth()) * 0.94 * 72.0 / (NOTE_SIZE * 0.52)))
    text = "\n".join(
        "\n".join(
            textwrap.fill(part, characters, break_on_hyphens=False) for part in line.split("\n")
        )
        for line in distinct
    )
    fig.text(
        _FIGURE_LEFT,
        _EDGE_PAD / height,
        text,
        ha="left",
        va="bottom",
        fontsize=NOTE_SIZE,
        color=SECONDARY_INK,
        gid=GID_NOTES,
    ).set_in_layout(False)
    _reserve(fig, bottom=(_EDGE_PAD + _text_height(text, NOTE_SIZE) + 2.0 * _GAP) / height)


def title_text(fig: Figure) -> str:
    """The title of a figure a plot drew, as one line.

    Parameters
    ----------
    fig : Figure
        The figure.

    Returns
    -------
    str
        The :func:`figure_header` title, else the first axes' left title, else ``""``.
    """
    for text in fig.texts:
        if text.get_gid() == GID_TITLE:
            return " ".join(text.get_text().split())
    if fig.axes:
        return " ".join(fig.axes[0].get_title(loc="left").split())
    return ""


def key_labels(fig: Figure) -> list[str]:
    """Every key entry of a figure: the axes keys of :func:`legend`, then the figure key.

    Parameters
    ----------
    fig : Figure
        The figure.

    Returns
    -------
    list of str
        The entries in drawing order.
    """
    keys = [axis.get_legend() for axis in fig.axes]
    found = [key for key in keys if key is not None] + list(fig.legends)
    return [text.get_text() for key in found for text in key.get_texts()]


def reference_line(
    ax: Axes,
    value: float,
    *,
    orientation: str = "horizontal",
    label: str | None = None,
    kind: str = "reference",
) -> Any:
    """A neutral grey line: a dashed reference value, or a dotted marked setting.

    A reference (``kind="reference"``) is a value the estimate is read against: no effect,
    the ATE, random targeting, a threshold. A marker (``kind="marker"``) points at a value
    of the plot's own axis: the setting the fit used, a breakdown point. The two differ in
    dash pattern and ink, so a reader can tell them apart in grey scale.

    Parameters
    ----------
    ax : Axes
        The axes.
    value : float
        Where the line sits.
    orientation : {"horizontal", "vertical"}, default="horizontal"
        A horizontal line at ``y = value`` or a vertical one at ``x = value``.
    label : str or None, default=None
        The legend entry.
    kind : {"reference", "marker"}, default="reference"
        Which of the two line kinds to draw.

    Returns
    -------
    Line2D
        The line, with gid ``"cleverly-reference"`` or ``"cleverly-marker"``.
    """
    draw = ax.axhline if orientation == "horizontal" else ax.axvline
    line = draw(
        value,
        color=REFERENCE if kind == "reference" else SECONDARY_INK,
        linestyle=(0, (4, 3)) if kind == "reference" else (0, (1, 2)),
        linewidth=1.2,
        zorder=1.5,
        label=label if label is not None else "_nolegend_",
    )
    line.set_gid(GID_REFERENCE if kind == "reference" else GID_MARKER)
    return line


def truth_mark(ax: Axes, x: float, y: float, *, transform: Any = None) -> Any:
    """A known true value at one point: an ink cross, in no series colour.

    A true value exists only for simulated data. The cross differs from an estimate's
    round marker and from the dashed reference line, and carries its own key entry,
    :data:`TRUTH_LABEL`.

    Parameters
    ----------
    ax : Axes
        The axes.
    x : float
        The x position, in the coordinates of ``transform``.
    y : float
        The y position, in the coordinates of ``transform``.
    transform : Transform or None, default=None
        The transform of the estimate the mark sits beside, such as a sideways shift;
        ``ax.transData`` when omitted.

    Returns
    -------
    Line2D
        The mark, with gid ``"cleverly-truth"``.
    """
    extra = {} if transform is None else {"transform": transform}
    (line,) = ax.plot(
        [x],
        [y],
        marker="x",
        markersize=8.0,
        markeredgewidth=2.0,
        color=INK,
        linestyle="none",
        label=TRUTH_LABEL,
        gid=GID_TRUTH,
        zorder=4,
        **extra,
    )
    return line


def truth_line(ax: Axes, x: Sequence[float], y: Sequence[float]) -> Any:
    """A known true curve: a thin ink dash-dot line, in no series colour.

    Parameters
    ----------
    ax : Axes
        The axes.
    x : sequence of float
        The x values.
    y : sequence of float
        The true value at each x.

    Returns
    -------
    Line2D
        The line, with gid ``"cleverly-truth"`` and the key entry :data:`TRUTH_LABEL`.
    """
    (line,) = ax.plot(
        list(x),
        list(y),
        color=INK,
        linewidth=1.4,
        linestyle=(0, (6, 2, 1, 2)),
        label=TRUTH_LABEL,
        gid=GID_TRUTH,
        zorder=2.5,
    )
    return line


def legend(ax: Axes, *, sources: Sequence[Axes] | None = None, max_columns: int = 3) -> None:
    """The key of interval kinds and reference lines, in one row above ``ax``.

    Series are labelled directly, so the key holds only labelled artists, which the plots
    give to interval kinds and reference lines alone. It sits between the subtitle and the
    axes, where it cannot cover data; call it before :func:`header`.

    Parameters
    ----------
    ax : Axes
        The top axes.
    sources : sequence of Axes or None, default=None
        The axes whose labelled artists the key lists; ``ax`` alone when omitted.
    max_columns : int, default=3
        The most columns the key may take.
    """
    handles, labels = _key_entries(sources or [ax])
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        handler_map=_key_handlers(handles),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=_key_columns(ax, labels, max_columns),
        frameon=False,
        fontsize=TICK_SIZE,
        labelcolor=INK,
        borderaxespad=0.2,
        borderpad=0.0,
        handlelength=1.8,
        columnspacing=1.4,
    )


#: How much wider an axes is after the layout engine runs than the default subplot it
#: starts as (0.775 of the figure), which is the width :func:`_key_columns` reads.
_LAYOUT_GROWTH: Final = 1.1


def _key_columns(ax: Axes, labels: Sequence[str], most: int = 3) -> int:
    """The most key columns, up to ``most``, whose widest row fits over the axes.

    A key wider than its axes runs past the figure. The label widths are measured at the
    key's font size, in points, and compared with the axes width in points.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    fig = ax.get_figure(root=True)
    inches = float(fig.get_size_inches()[0]) if fig is not None else FIGSIZE[0]
    width = ax.get_position().width * inches * 72.0 * _LAYOUT_GROWTH
    font = FontProperties(size=TICK_SIZE)
    measure = TextToPath()
    text = [measure.get_text_width_height_descent(label, font, ismath=False)[0] for label in labels]
    handle = TICK_SIZE * (1.8 + 0.8)
    gap = TICK_SIZE * 1.4
    for columns in range(min(len(labels), most), 1, -1):
        widths = [
            max(text[i] for i in range(c, len(labels), columns)) + handle for c in range(columns)
        ]
        if sum(widths) + gap * (columns - 1) <= width:
            return columns
    return 1


def _key_entries(sources: Sequence[Axes]) -> tuple[list[Any], list[str]]:
    """The labelled artists of ``sources``, one handle per distinct label, in order."""
    unique: dict[str, Any] = {}
    for axis in sources:
        found, names = axis.get_legend_handles_labels()
        for handle, name in zip(found, names, strict=True):
            unique[name] = handle  # first position, last handle, as dict(zip()) gave
    return list(unique.values()), list(unique.keys())


def end_label(ax: Axes, x: float, y: float, text: str, color: str) -> None:
    """A direct label at the end of a series: a swatch in the series colour, text in ink.

    Parameters
    ----------
    ax : Axes
        The axes.
    x : float
        The series' last drawn x, in data coordinates.
    y : float
        The series' last drawn y, in data coordinates.
    text : str
        The series name.
    color : str
        The series colour, used for the swatch only.
    """
    ax.annotate(
        "\u25a0",
        xy=(x, y),
        xytext=(6.0, 0.0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=TICK_SIZE,
        color=color,
        annotation_clip=False,
        gid=GID_SWATCH,
    )
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(17.0, 0.0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=TICK_SIZE,
        color=INK,
        gid=GID_SERIES_LABEL,
        annotation_clip=False,
    )


def corner_label(ax: Axes, y: float, text: str, color: str, *, va: str = "top") -> None:
    """A series label inside the axes at the left edge: a coloured swatch, then ink text.

    Parameters
    ----------
    ax : Axes
        The axes.
    y : float
        Height in axes fraction.
    text : str
        The series name.
    color : str
        The series colour.
    va : {"top", "bottom"}, default="top"
        Vertical alignment of the label at ``y``.
    """
    box = {"boxstyle": "square,pad=0.15", "facecolor": FACE, "edgecolor": "none", "alpha": 0.85}
    ax.annotate(
        "\u25a0",
        xy=(0.012, y),
        xycoords="axes fraction",
        ha="left",
        va=va,
        fontsize=TICK_SIZE,
        color=color,
        gid=GID_SWATCH,
    )
    ax.annotate(
        text,
        xy=(0.012, y),
        xycoords="axes fraction",
        xytext=(11.0, 0.0),
        textcoords="offset points",
        ha="left",
        va=va,
        fontsize=TICK_SIZE,
        color=INK,
        bbox=box,
        gid=GID_SERIES_LABEL,
    )


def series_color(index: int) -> str:
    """The colour of the ``index``-th coloured entity on one axes, in fixed slot order.

    Parameters
    ----------
    index : int
        Zero-based position of the entity (regimen, arm, score or method).

    Returns
    -------
    str
        The palette slot.

    Raises
    ------
    ValueError
        For a fourth entity: four or more become small multiples, one panel each.
    """
    if not 0 <= index < MAX_SERIES:
        raise ValueError(
            f"at most {MAX_SERIES} coloured series share one axes; draw more as small "
            "multiples, one panel per series"
        )
    return PALETTE[index]


def ribbon(
    ax: Axes,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    color: str,
    label: str,
    band: bool = False,
    step: bool = False,
) -> Any:
    """A pointwise ribbon, or the lighter simultaneous band behind it.

    Draw a ribbon only over a curve the fit evaluates on a fine grid, or as a step ribbon
    over a discrete-time curve that reports every node (honesty rule 2). Over a finite grid
    use :func:`brackets`.

    A step ribbon holds each node's limits until the next node, so the last node's limits
    would span no width and draw nothing. They are marked at the last node with
    :func:`brackets` instead, which claims nothing past the last node; a plot that draws a
    step ribbon says so in its notes.

    Parameters
    ----------
    ax : Axes
        The axes.
    x : sequence of float
        Positions.
    lower : sequence of float
        Lower limits at ``x``.
    upper : sequence of float
        Upper limits at ``x``.
    color : str
        The series colour.
    label : str
        The key entry, from :func:`pointwise_label` or :func:`band_label`.
    band : bool, default=False
        Whether this is a simultaneous band (lighter, drawn behind) rather than pointwise.
    step : bool, default=False
        Whether the curve is constant between nodes (``step="post"``).

    Returns
    -------
    PolyCollection
        The ribbon, with gid :data:`GID_BAND` or :data:`GID_INTERVAL`.
    """
    collection = ax.fill_between(
        list(x),
        list(lower),
        list(upper),
        color=color,
        alpha=BAND_ALPHA if band else INTERVAL_ALPHA,
        linewidth=0.0,
        step="post" if step else None,
        label=label,
        zorder=1.0 if band else 1.2,
    )
    collection.set_gid(GID_BAND if band else GID_INTERVAL)
    if step:
        last = [list(x)[-1], list(lower)[-1], list(upper)[-1]]
        if all(finite(value) for value in last):
            brackets(
                ax, [last[0]], [last[1]], [last[2]], color=color, label="_nolegend_", band=band
            )
    return collection


def brackets(
    ax: Axes,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    color: str,
    label: str,
    band: bool = False,
) -> Any:
    """Interval or band limits as marks at grid points only, never joined between them.

    Parameters
    ----------
    ax : Axes
        The axes.
    x : sequence of float
        The grid points.
    lower : sequence of float
        Lower limits at the grid points.
    upper : sequence of float
        Upper limits at the grid points.
    color : str
        The series colour.
    label : str
        The key entry.
    band : bool, default=False
        Whether these are simultaneous-band limits (open bracket marks) rather than a
        pointwise interval (a solid bar).

    Returns
    -------
    Artist
        The drawn collection or container, with gid :data:`GID_BAND` or :data:`GID_INTERVAL`.
    """
    if band:
        (line,) = ax.plot(
            [*x, *x],
            [*lower, *upper],
            linestyle="none",
            marker="_",
            markersize=12,
            markeredgewidth=1.6,
            color=color,
            label=label,
            zorder=2.5,
        )
        line.set_gid(GID_BAND)
        _BRACKET_GLYPHS[line] = True
        return line
    collection = ax.vlines(
        list(x), list(lower), list(upper), color=color, linewidth=2.0, label=label
    )
    collection.set_gid(GID_INTERVAL)
    _BRACKET_GLYPHS[collection] = False
    return collection


#: The artists :func:`brackets` drew, with whether each is a band. The key draws their glyphs
#: as the marks look on the axes: matplotlib's default would draw both a band's caps and an
#: interval's bar as one short horizontal dash. Weak keys let a closed figure's artists go.
_BRACKET_GLYPHS: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
#: The ``gid`` of the key glyphs of :func:`brackets`' marks.
GID_KEY_BAND: Final = "cleverly-key-band"
GID_KEY_INTERVAL: Final = "cleverly-key-interval"


def _key_handlers(handles: Sequence[Any]) -> dict[Any, Any]:
    """A legend handler map that draws each bracket handle as the mark it stands for.

    A band's key glyph is its two caps, one above the other with nothing between them; an
    interval's is one vertical bar of the same height. They differ by shape, so the two
    entries of one colour read apart at a glance.
    """
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D

    class _Glyph(HandlerBase):
        def __init__(self, band: bool) -> None:
            super().__init__()
            self.band = band

        def create_artists(
            self,
            legend: Any,
            orig_handle: Any,
            xdescent: float,
            ydescent: float,
            width: float,
            height: float,
            fontsize: float,
            trans: Any,
        ) -> list[Any]:
            from matplotlib.colors import to_hex

            color = orig_handle.get_color()
            if not isinstance(color, str):
                # a line collection holds one colour per segment; brackets() gives one
                color = to_hex(next(iter(color)), keep_alpha=True)
            middle = width / 2.0 - xdescent
            low, high = -ydescent - 0.25 * height, -ydescent + 1.25 * height
            if self.band:
                half = 0.3 * width
                pieces = [
                    Line2D([middle - half, middle + half], [y, y], color=color, linewidth=1.6)
                    for y in (low, high)
                ]
                gid = GID_KEY_BAND
            else:
                pieces = [Line2D([middle, middle], [low, high], color=color, linewidth=2.0)]
                gid = GID_KEY_INTERVAL
            for piece in pieces:
                piece.set_transform(trans)
                piece.set_gid(gid)
            return pieces

    found: dict[Any, Any] = {}
    for handle in handles:
        band = _BRACKET_GLYPHS.get(handle)
        if band is not None:
            found[handle] = _Glyph(band)
    return found


def log_axis(ax: Axes, which: str) -> None:
    """A log scale whose tick labels are plain numbers (``0.5``, ``1``, ``2``).

    Call it after the data are drawn: an axis that spans less than a factor of ten gets
    evenly spaced round ticks, one that spans up to :data:`_LOG_DENSE_DECADES` powers of ten
    gets ticks at 1, 2 and 5 times each power, and a wider one ticks at powers of ten alone,
    at most :data:`_LOG_MOST_TICKS` of them. A number from 0.001 to 9999 is printed as
    it is, a smaller or larger power of ten as ``10`` with an exponent.

    Parameters
    ----------
    ax : Axes
        The axes.
    which : {"x", "y"}
        The axis.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    low, high = ax.get_xlim() if which == "x" else ax.get_ylim()
    decades = math.log10(max(high, low) / min(high, low)) if low > 0 and high > 0 else math.inf
    if decades < 1.0:
        axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    elif decades <= _LOG_DENSE_DECADES:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    else:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=_LOG_MOST_TICKS))
    axis.set_major_formatter(FuncFormatter(lambda value, _: log_tick_text(value)))
    axis.set_minor_formatter(NullFormatter())


#: The widest span, in powers of ten, that a log axis ticks at 1, 2 and 5 times each power.
_LOG_DENSE_DECADES: Final = 2.0
#: The most ticks a log axis wider than that carries.
_LOG_MOST_TICKS: Final = 7


def log_tick_text(value: float) -> str:
    """A log axis tick label: the number from 0.001 to 9999, a power of ten beyond.

    Parameters
    ----------
    value : float
        The tick's value.

    Returns
    -------
    str
        ``"0.5"`` or ``"20"`` as written, ``"$10^{-6}$"`` for a power of ten outside that
        range, and the number in ``g`` form for any other value outside it.
    """
    if value <= 0.0 or not math.isfinite(value):
        return f"{value:g}"
    if 1e-3 <= value < 1e4:
        return f"{value:g}"
    power = math.log10(value)
    if abs(power - round(power)) < 1e-9:
        return f"$10^{{{round(power)}}}$"
    return f"{value:g}"


# ----------------------------------------------------------------------- interval words


def level_text(level: float) -> str:
    """``0.95`` as ``"95%"``.

    Parameters
    ----------
    level : float
        A coverage level.

    Returns
    -------
    str
        The level as a percentage.
    """
    return f"{level * 100:g}%"


def pointwise_label(level: float) -> str:
    """The legend entry of a pointwise interval.

    Parameters
    ----------
    level : float
        The coverage level.

    Returns
    -------
    str
        For example ``"95% pointwise interval"``.
    """
    return f"{level_text(level)} pointwise interval"


def interval_label(level: float) -> str:
    """The legend entry of one estimate's confidence interval.

    Parameters
    ----------
    level : float
        The coverage level.

    Returns
    -------
    str
        For example ``"95% confidence interval"``.
    """
    return f"{level_text(level)} confidence interval"


def band_label(level: float, kind: str) -> str:
    """The legend entry of a simultaneous band.

    Parameters
    ----------
    level : float
        The coverage level.
    kind : {"multiplier", "bootstrap sup-t"}
        How the band was computed.

    Returns
    -------
    str
        For example ``"95% simultaneous band (multiplier)"``.
    """
    return f"{level_text(level)} simultaneous band ({kind})"


def format_values(values: Sequence[float], spreads: Sequence[float]) -> list[str]:
    """Numbers at the decimals the printed summary would use for these spreads.

    Parameters
    ----------
    values : sequence of float
        The numbers.
    spreads : sequence of float
        The spreads that set the decimals (standard errors or interval half-widths).

    Returns
    -------
    list of str
        The formatted numbers.
    """
    decimals = decimals_for(spreads)
    return [number_text(value, decimals) for value in values]


# ----------------------------------------------------------------------- honesty


@dataclass(frozen=True)
class Interval:
    """What one estimate lets a plot draw.

    Parameters
    ----------
    psi : float
        The estimate.
    lower : float or None
        The interval's lower limit, or ``None`` when the estimate supplies no inference.
    upper : float or None
        The interval's upper limit, or ``None`` when the estimate supplies no inference.
    level : float
        The coverage level.
    std_error : float
        The spread that sets the printed decimals; the plug-in spread when no interval is
        drawn.
    notes : tuple of str
        Why no interval is drawn, or the experimental label.
    experimental : bool
        Whether the estimate carries the experimental label.
    """

    psi: float
    lower: float | None
    upper: float | None
    level: float
    std_error: float
    notes: tuple[str, ...] = field(default=())
    experimental: bool = False

    @property
    def drawn(self) -> bool:
        """Whether an interval is drawn."""
        return self.lower is not None and self.upper is not None


def estimate_interval(estimate: Any, name: str | None = None) -> Interval:
    """Read what a :class:`~cleverly.inference.ParameterEstimate` lets a plot draw.

    An estimate whose inference status supplies no inference gives its point only, and the
    note carries the status's plain reason (honesty rule 1). An experimental estimate
    carries its label and the plain support reason (rule 3).

    Parameters
    ----------
    estimate : ParameterEstimate
        The estimate.
    name : str or None, default=None
        The row name the notes refer to; the estimate's own name when omitted.

    Returns
    -------
    Interval
        The point, the interval or ``None``, and the notes.
    """
    from .._inference_status import status_record
    from .._summary._estimates import plain_support_reason

    label = name or str(estimate.name)
    level = 1.0 - float(estimate.alpha)
    spread = float(estimate.plugin_std_error)
    found: list[str] = []
    experimental = getattr(estimate.support, "label", "supported") == "experimental"
    if experimental:
        found.append(f"{label}: experimental. {plain_support_reason(estimate.support.reason)}")
    if not estimate.supplies_inference:
        record = status_record(str(estimate.inference))
        found.append(f"{label}: no interval is drawn. {record.plain_reason}")
        return Interval(float(estimate.psi), None, None, level, spread, tuple(found), experimental)
    low, high = estimate.ci
    return Interval(
        float(estimate.psi), float(low), float(high), level, spread, tuple(found), experimental
    )


#: Plain words for the estimand stems a summary prints, keyed by the alias's first token.
_ESTIMAND_WORDS: Final = {
    "ate": "Average effect",
    "att": "Average effect among the treated",
    "atc": "Average effect among the untreated",
    "rr": "Risk ratio",
    "or": "Odds ratio",
    "ey": "Mean outcome",
    "ey0": "Mean outcome untreated",
    "ey1": "Mean outcome treated",
    "ey_obs": "Mean outcome as treated",
    "par": "Population attributable effect",
    "paf": "Population attributable fraction",
}


def estimand_words(name: str) -> str:
    """An estimand alias in words, with the alias kept in parentheses.

    Parameters
    ----------
    name : str
        The alias a result reports, such as ``"ate"`` or ``"ate[high vs low]"``.

    Returns
    -------
    str
        For example ``"Average effect (ate)"``, or the alias alone when no words are known.
    """
    import re

    stem = re.split(r"[\[_]", name, maxsplit=1)[0]
    words = _ESTIMAND_WORDS.get(name) or _ESTIMAND_WORDS.get(stem)
    return f"{words} ({name})" if words else name


def null_for(scale: str) -> float | None:
    """The no-effect value of an estimate on ``scale``, or ``None`` for a level.

    Parameters
    ----------
    scale : str
        The estimate's scale: ``"difference"``, ``"ratio"``, ``"level"`` or ``"fraction"``.

    Returns
    -------
    float or None
        ``0`` for a difference or a fraction (the share an effect explains), ``1`` for a
        ratio, ``None`` for a level.
    """
    if scale in ("difference", "fraction"):
        return 0.0
    if scale == "ratio":
        return 1.0
    return None


def finite(value: Any) -> bool:
    """Whether ``value`` is a finite number.

    Parameters
    ----------
    value : Any
        The value.

    Returns
    -------
    bool
        ``True`` for a finite float.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "BAND_ALPHA",
    "CONTEXT",
    "FACE",
    "FIGSIZE",
    "GID_BAND",
    "GID_INTERVAL",
    "GID_KEY",
    "GID_KEY_BAND",
    "GID_KEY_INTERVAL",
    "GID_MARKER",
    "GID_NOTES",
    "GID_POINT",
    "GID_RANGE",
    "GID_REFERENCE",
    "GID_SERIES_LABEL",
    "GID_SUBTITLE",
    "GID_SWATCH",
    "GID_TITLE",
    "GID_TRUTH",
    "GRID",
    "INK",
    "INSTALL_HINT",
    "INTERVAL_ALPHA",
    "MAX_SERIES",
    "PALETTE",
    "REFERENCE",
    "SECONDARY_INK",
    "TRUTH_LABEL",
    "Interval",
    "band_label",
    "brackets",
    "corner_label",
    "end_label",
    "estimand_words",
    "estimate_interval",
    "figure_for",
    "figure_header",
    "figure_notes",
    "finite",
    "format_values",
    "header",
    "interval_label",
    "key_labels",
    "legend",
    "level_text",
    "log_axis",
    "log_tick_text",
    "new_figure",
    "note_text",
    "notes",
    "null_for",
    "pointwise_label",
    "pyplot",
    "reference_line",
    "ribbon",
    "series_color",
    "style",
    "style_axes",
    "subtitle_text",
    "title_text",
    "truth_line",
    "truth_mark",
    "wrap",
]
