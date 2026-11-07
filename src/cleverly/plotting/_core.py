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
    width_inches = float(figure.get_figwidth()) * 0.92
    characters = max(30, int(width_inches * 72.0 / (size * 0.52)))
    return "\n".join(textwrap.fill(part, characters) for part in text.split("\n"))


def header(ax: Axes, title: str, subtitle: str | None = None) -> None:
    """Write the title (the question) and the subtitle (the estimand or headline number).

    The title goes on ``ax``, left aligned, the subtitle under it, and the key that
    :func:`legend` placed (if any) between the subtitle and the axes, so the layout engine
    reserves room for all three and the key never covers data. Call :func:`legend` first.
    Pass the top panel of a multi-panel plot.

    Parameters
    ----------
    ax : Axes
        The top axes.
    title : str
        The title.
    subtitle : str or None, default=None
        One line, wrapped when long.
    """
    key = ax.get_legend()
    key_height = 0.0
    if key is not None:
        rows = math.ceil(len(key.get_texts()) / 3)
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
            gid="cleverly-subtitle",
        )


def subtitle_text(ax: Axes) -> str | None:
    """The subtitle :func:`header` wrote on ``ax``, as one line, or ``None``.

    Parameters
    ----------
    ax : Axes
        The top axes of a plot.

    Returns
    -------
    str or None
        The subtitle with its line breaks joined by spaces.
    """
    for text in ax.texts:
        if text.get_gid() == "cleverly-subtitle":
            return " ".join(text.get_text().split())
    return None


def notes(ax: Axes, lines: Sequence[str]) -> None:
    """Write the note lines under the bottom axes, below its axis label.

    Each line is one statement the reader needs to read the figure correctly: what an
    interval is, why one is missing, that a quantity is descriptive or experimental.

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
    ax.annotate(
        text,
        xy=(0.0, 0.0),
        xycoords=("axes fraction", ax.xaxis.label) if ax.get_xlabel() else "axes fraction",
        xytext=(0.0, -6.0 if ax.get_xlabel() else -28.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=NOTE_SIZE,
        color=SECONDARY_INK,
        gid="cleverly-notes",
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
    parts = [
        text.get_text()
        for axis in fig.axes
        for text in axis.texts
        if text.get_gid() == "cleverly-notes"
    ]
    return " ".join(" ".join(parts).split())


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
    line.set_gid(f"cleverly-{kind}")
    return line


def legend(ax: Axes, *, sources: Sequence[Axes] | None = None) -> None:
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
    """
    handles: list[Any] = []
    labels: list[str] = []
    for axis in sources or [ax]:
        found, names = axis.get_legend_handles_labels()
        handles.extend(found)
        labels.extend(names)
    if not handles:
        return
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=min(len(unique), 3),
        frameon=False,
        fontsize=TICK_SIZE,
        labelcolor=INK,
        borderaxespad=0.2,
        borderpad=0.0,
        handlelength=1.8,
        columnspacing=1.4,
    )


def end_label(ax: Axes, x: float, y: float, text: str, color: str) -> None:
    """A direct label at the end of a series: a swatch in the series colour, text in ink.

    Parameters
    ----------
    ax : Axes
        The axes.
    x, y : float
        The series' last drawn point, in data coordinates.
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
        gid="cleverly-swatch",
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
        gid="cleverly-series-label",
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
        gid="cleverly-swatch",
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
        gid="cleverly-series-label",
    )


def log_axis(ax: Axes, which: str) -> None:
    """A log scale whose tick labels are plain numbers (``0.5``, ``1``, ``2``).

    Call it after the data are drawn: an axis that spans less than a factor of ten gets
    evenly spaced round ticks, and a wider one gets ticks at 1, 2 and 5 times each power of
    ten.

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
    narrow = low > 0 and high > 0 and math.log10(max(high, low) / min(high, low)) < 1.0
    if narrow:
        axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    else:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


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
    lower, upper : float or None
        The interval, or ``None`` when the estimate supplies no inference.
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


def null_for(scale: str) -> float | None:
    """The no-effect value of an estimate on ``scale``, or ``None`` for a level.

    Parameters
    ----------
    scale : str
        The estimate's scale: ``"difference"``, ``"ratio"``, ``"level"`` or ``"fraction"``.

    Returns
    -------
    float or None
        ``0`` for a difference, ``1`` for a ratio, ``None`` otherwise.
    """
    if scale == "difference":
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
    "GRID",
    "INK",
    "INSTALL_HINT",
    "INTERVAL_ALPHA",
    "MAX_SERIES",
    "PALETTE",
    "REFERENCE",
    "SECONDARY_INK",
    "Interval",
    "band_label",
    "corner_label",
    "end_label",
    "estimate_interval",
    "figure_for",
    "finite",
    "format_values",
    "header",
    "interval_label",
    "legend",
    "level_text",
    "log_axis",
    "new_figure",
    "note_text",
    "notes",
    "null_for",
    "pointwise_label",
    "pyplot",
    "reference_line",
    "style",
    "style_axes",
    "subtitle_text",
    "wrap",
]
