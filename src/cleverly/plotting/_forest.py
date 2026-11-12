"""V1: the forest plot of estimates with their intervals.

One row per estimate, the point and its interval, the no-effect line, and the numbers in a
column on the right, formatted as the printed summary formats them. Estimates on different
scales go in separate panels, because one axes carries one scale: differences around 0,
ratios on a log axis around 1, and arm means with no null line.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from . import _core, _truth

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The panel order and the x-axis words of each scale.
_SCALE_ORDER = ("difference", "ratio", "level", "fraction")
#: The default title, a question as every plot's title is.
TITLE = "Estimates and intervals"
#: How far, as a share of the drawn data's span, the no-effect value may lie from the data and
#: still widen the axis to show it, so the rows keep at least half the axis. A null further
#: out would squeeze them into a corner.
_NULL_REACH = 1.0


@dataclass(frozen=True)
class _Row:
    label: str
    interval: _core.Interval
    scale: str
    outcome: str | None
    #: Whether the scale's no-effect value exists under the estimate's transform.
    null_defined: bool = True
    #: Which item the row came from; the numbers of one item share their decimals.
    source: int = 0
    #: The name the fit reports the estimate under, which ``truth=`` keys.
    estimand: str = ""


def _null_defined(estimate: Any) -> bool:
    """Whether the default no-effect value maps through the estimate's transform."""
    import numpy as np

    transform = getattr(estimate, "transform", None)
    if transform is None:
        return True
    default = 1.0 if str(estimate.scale) == "ratio" else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return bool(np.isfinite(float(transform.forward(default))))


def _row(label: str, estimate: Any, outcome: str | None, estimand: str | None = None) -> _Row:
    return _Row(
        label,
        _core.estimate_interval(estimate, label),
        str(estimate.scale),
        outcome,
        _null_defined(estimate),
        estimand=str(estimand if estimand is not None else getattr(estimate, "name", label)),
    )


def _is_estimate(item: Any) -> bool:
    return hasattr(item, "influence_curve") and hasattr(item, "psi") and hasattr(item, "scale")


def _outcome(result: Any) -> str | None:
    data = getattr(result, "data", None)
    name = getattr(data, "outcome_name", None)
    return None if name is None else str(name)


def _method(result: Any) -> str:
    from .._summary import vocabulary

    return vocabulary.method_short_name(str(getattr(result, "fitted_method", "tmle")))


def _result_rows(result: Any, prefix: str | None) -> list[_Row]:
    estimates = result.estimates
    outcome = _outcome(result)
    rows = []
    for name, estimate in estimates.items():
        label = str(name) if prefix is None else f"{prefix}: {name}"
        rows.append(_row(label, estimate, outcome, str(name)))
    return rows


def _expand(item: Any, label: str | None, several: bool) -> list[_Row]:
    """The rows one argument contributes."""
    if _is_estimate(item):
        name = label or str(item.name)
        return [_row(name, item, None)]
    if hasattr(item, "results") and hasattr(item, "levels"):  # TMLEResultSet
        levels = item.levels
        if len(levels) == 1:
            return _expand(item.results[levels[0]], label, several)
        rows: list[_Row] = []
        for level in levels:
            stem = f"{item.intermediate_name} = {level:g}"
            level_prefix = stem if label is None else f"{label}, {stem}"
            rows.extend(_result_rows(item.results[level], level_prefix))
        return rows
    if hasattr(item, "estimates") and isinstance(item.estimates, Mapping):
        prefix = label if label is not None else (_method(item) if several else None)
        if len(item.estimates) == 1 and prefix is not None:
            ((name, estimate),) = item.estimates.items()
            return [_row(prefix, estimate, _outcome(item), str(name))]
        return _result_rows(item, prefix)
    if isinstance(item, Mapping):
        rows = []
        for key, value in item.items():
            rows.extend(_expand(value, str(key), True))
        return rows
    raise TypeError(
        "forest() takes fitted results, result sets, ParameterEstimate objects or a mapping "
        f"of labels to them; got {type(item).__name__}"
    )


def _axis_label(
    scale: str, outcome_label: str | None, outcome: str | None, units: str | None
) -> str:
    """The x label of one scale's panel.

    ``outcome_label=`` names the whole quantity (``"Difference in orders in month 3"``), as
    in :func:`cleverly.plotting.curve`; the outcome column follows ``"mean"`` without it.
    """
    unit = f" ({units})" if units else ""
    if scale == "ratio":
        return "Ratio (log scale)"
    if scale == "fraction":
        return "Share of the mean outcome"
    difference = scale == "difference"
    if outcome_label or outcome:
        words = _core.quantity(outcome_label, str(outcome), units, difference=difference)
        return " ".join(words).strip()
    return f"Difference{unit}" if difference else f"Mean outcome{unit}"


def _question(items: Sequence[Any]) -> str | None:
    """The summary's Question fact for a single fitted result, else ``None``."""
    if len(items) != 1:
        return None
    (item,) = items
    if hasattr(item, "results") and hasattr(item, "levels") and len(item.levels) == 1:
        item = item.results[item.levels[0]]
    summary = getattr(item, "summary", None)
    if summary is None or _is_estimate(item) or not hasattr(item, "estimates"):
        return None
    question = summary().facts.get("Question")
    if not question:
        return None
    text = " ".join(str(question).split())
    # The summary defines its rows in a table below; the figure has no such table.
    text = text.removesuffix("; each row is defined below")
    if _starts_with_a_name(text, item):
        # A question that opens with a column name keeps the name as the data spell it.
        return text
    return text[:1].upper() + text[1:]


def _starts_with_a_name(text: str, result: Any) -> bool:
    """Whether ``text`` opens with a name from the data rather than an English word."""
    first = text.split(" ", 1)[0]
    data = getattr(result, "data", None)
    names = {
        str(name)
        for name in (
            getattr(data, "outcome_name", None),
            getattr(data, "treatment_name", None),
        )
        if name is not None
    }
    return first in names or not first.isalpha()


@_core.labelled
def forest(
    *items: Any,
    labels: Sequence[str] | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    outcome_label: str | None = None,
    units: str | None = None,
    truth: float | Mapping[str, float] | None = None,
    include_null: bool | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot estimates with their intervals, one row each (a forest plot).

    Use it to compare several estimates, contrasts or methods on one scale and to see at a
    glance which intervals cross the no-effect line. Each row draws the estimate and its
    confidence interval; the numbers are printed in a column on the right with the decimals
    the printed summary uses. Estimates on different scales go in separate panels:
    differences around 0, ratios on a log axis around 1, and arm means with no null line.
    The no-effect line is drawn when it lies near the intervals; when it lies far from them,
    the axis keeps to the intervals and :func:`~cleverly.plotting.notes` says on which side
    no effect lies.

    An estimate whose inference the package withholds is drawn as a hollow point with no
    interval, and its value column reads "(no interval)"; :func:`~cleverly.plotting.notes`
    gives the reason.

    Parameters
    ----------
    *items : TMLEResult, TMLEResultSet, LongitudinalResult, ParameterEstimate or mapping
        What to draw. A fitted result contributes every estimate it reports; a mapping
        contributes each value under its key as the label.
    labels : sequence of str or None, default=None
        One label per item. A result's default label is its method's short name when
        several items are drawn, and nothing when one is.
    title : str, False or None, default=None
        The title; a short description of the figure when omitted, and none when ``False``.
    subtitle : str, False or None, default=None
        The line under the title; the plot's own, which names the estimate and its headline
        number, when omitted, and none when ``False``.
    xlabel : str, sequence of str or None, default=None
        The x-axis label. A string replaces the first labelled panel's label and the same
        label wherever it repeats; a sequence gives each labelled panel its own, in reading
        order.
    ylabel : str, sequence of str or None, default=None
        The y-axis label, given as for ``xlabel``.
    legend : {"top", "right", "bottom"} or False, default="top"
        Where the key goes: above the axes, right of them, or under the figure; ``False``
        draws no key.
    figsize : tuple of float or None, default=None
        Width and height in inches of a figure the plot creates.
    outcome_label : str or None, default=None
        Words for the quantity on the axis of differences and means, such as ``"orders in
        the third month"``: the labels read ``"Difference in orders in the third month"``
        and ``"orders in the third month"``. Without it they read ``"Difference in mean"``
        and ``"Mean"`` the outcome column's name when every row shares one.
    units : str or None, default=None
        Units of the outcome, added to the axis label of differences and means.
    truth : float, mapping of str to float, or None, default=None
        The known true values of simulated data, drawn as an ink cross on each row it
        belongs to and named "True value (simulated data only)" in the key. Real data have
        none. A number is the true value of every row, and needs rows that all estimate
        one quantity (one estimate, or one estimate from each of several fits). A mapping
        gives the value of each estimate by the name the fit reports (``"ate"``) or by the
        row's label; a row it does not name gets no cross. A forest draws every estimate
        of each fit it is given, so a name that matches no row is refused.
    include_null : bool or None, default=None
        Whether the axis reaches the no-effect value (0 for a difference, 1 for a ratio) and draws
        its line. ``None`` draws it when it lies no further from the drawn intervals than their own
        span (a ratio's measured on the log axis), so the rows keep at least half the axis;
        otherwise it leaves no effect off the axis and :func:`~cleverly.plotting.notes` says on
        which side it lies. A panel with a row that has no finite interval always draws it under
        ``None``. ``True`` draws it wherever the estimates' scale has a no-effect value (a transform
        with no finite null draws no line); ``False`` leaves it off the axis unless the intervals
        cross it.
    ax : Axes or None, default=None
        Axes to draw on. Only a plot with one scale can draw on a caller's axes.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    ValueError
        When ``labels`` does not match ``items``, when nothing is given, or when ``ax`` is
        passed for estimates on more than one scale.
    CapabilityError
        When ``truth`` names no drawn row, or is one number for rows that estimate
        different quantities.

    See Also
    --------
    cleverly.estimators.TMLEResult.plot : This plot for one fit.
    cleverly.plotting.sensitivity : One estimate across a declared sensitivity parameter.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes give the reason for each
    withheld interval, which rows are experimental and why, and on which side a no-effect
    line left off the axis lies.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import forest
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    ... )
    >>> learners = dict(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     cross_fit=False,
    ...     random_state=0,
    ... )
    >>> tmle = study.identify(ATE()).estimate(**learners)
    >>> fig = forest(tmle["ate"], labels=["TMLE"])
    >>> fig.axes[0].get_title(loc="left")
    'Estimates and intervals'
    >>> [text.get_text() for text in fig.axes[0].get_yticklabels()]
    ['TMLE']
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    if not items:
        raise ValueError("forest() needs at least one estimate or result")
    if labels is not None and len(labels) != len(items):
        raise ValueError(f"labels has {len(labels)} entries for {len(items)} items")
    several = len(items) > 1
    rows: list[_Row] = []
    for index, item in enumerate(items):
        found = _expand(item, None if labels is None else labels[index], several)
        rows.extend(dataclasses.replace(row, source=index) for row in found)
    scales = sorted({row.scale for row in rows}, key=_scale_rank)
    if ax is not None and len(scales) > 1:
        raise ValueError(
            "these estimates are on more than one scale, which forest() draws as separate "
            "panels; omit ax= or pass the estimates of one scale"
        )
    groups = [[row for row in rows if row.scale == scale] for scale in scales]
    for scale, group in zip(scales, groups, strict=True):
        levels = {round(row.interval.level, 12) for row in group}
        if len(levels) > 1:
            shown = ", ".join(_core.level_text(level) for level in sorted(levels))
            raise ValueError(
                f"the {scale} estimates use different confidence levels ({shown}), which one "
                "panel cannot label; refit at one level or plot them separately"
            )
    texts = _number_texts(rows)
    if ax is None:
        heights = [max(1.0, 0.42 * len(group) + 0.9) for group in groups]
        fig, axes = _core.new_figure(
            len(groups),
            figsize=(7.0, sum(heights) + 1.4 + 0.25 * len(groups)),
            height_ratios=heights,
        )
    else:
        fig, axis = _core.figure_for(ax)
        axes = [axis]
    truths = None if truth is None else _truth_values(truth, rows)
    off_axis: list[str] = []
    clamps: list[tuple[Axes, str, float, float, bool]] = []
    at = 0
    for axis, scale, group in zip(axes, scales, groups, strict=True):
        mine = None if truths is None else truths[at : at + len(group)]
        at += len(group)
        note, keep_off = _panel(
            axis,
            scale,
            group,
            outcome_label,
            units,
            [texts[id(row)] for row in group],
            include_null=include_null,
            truths=[] if mine is None else [value for value in mine if value is not None],
        )
        if keep_off is not None:
            clamps.append((axis, *keep_off))
        if note is not None:
            off_axis.append(
                note[:1].upper() + note[1:] if len(scales) == 1 else f"On the {scale} panel, {note}"
            )
    if truths is not None:
        _draw_truth(truths, axes, groups)
    for axis, side, null, edge, log in clamps:
        _keep_off(axis, side, null, edge, log=log)
    question = subtitle or _question(items)
    # The number column sits right of the axes, so the key keeps to two columns.
    _core.header(axes[0], title or TITLE, question, key_sources=axes, key_columns=2)
    _core.add_notes(axes[-1], [*_merged_notes(rows), *off_axis])
    return fig


def _truth_values(truth: Any, rows: Sequence[_Row]) -> list[float | None]:
    """The true value of each row, in order, or ``None`` where ``truth`` gives none."""
    if isinstance(truth, Mapping):
        if not truth:
            raise _truth.refuse(
                "truth= is an empty mapping, so it gives no true value to draw",
                "pass the true values, or leave truth= out",
            )
        known = {name for row in rows for name in (row.estimand, row.label)}
        unknown = [str(key) for key in truth if str(key) not in known]
        if unknown:
            raise _truth.refuse(
                f"truth= names {unknown}, which no row of this plot estimates; the rows are "
                f"{list(dict.fromkeys(row.estimand for row in rows))}",
                "key truth= by the names the fits report or by the row labels",
            )
        keyed = {str(key): value for key, value in truth.items()}
        found: list[float | None] = []
        for row in rows:
            value = keyed.get(row.label, keyed.get(row.estimand))
            found.append(None if value is None else _truth.number(value, row.label))
        return found
    estimands = list(dict.fromkeys(row.estimand for row in rows))
    if len(estimands) > 1:
        raise _truth.refuse(
            f"truth= is one number, and the rows of this plot estimate {len(estimands)} "
            f"quantities: {estimands}",
            "pass a mapping of each estimate's name to its true value",
        )
    value = _truth.number(truth, estimands[0])
    return [value] * len(rows)


def _draw_truth(
    values: Sequence[float | None], axes: Sequence[Axes], groups: Sequence[Sequence[_Row]]
) -> None:
    """A cross at each row's true value, on the row's own panel."""
    at = 0
    for axis, group in zip(axes, groups, strict=True):
        mine = values[at : at + len(group)]
        at += len(group)
        positions = list(range(len(group)))[::-1]
        drawn = [(y, value) for y, value in zip(positions, mine, strict=True) if value is not None]
        _truth.check_log(axis, [value for _, value in drawn], axis="x")
        for y, value in drawn:
            _core.truth_mark(axis, value, y)
        _truth.include(axis, [value for _, value in drawn], axis="x")


def _merged_notes(rows: Sequence[_Row]) -> list[str]:
    """Each row's notes, with rows that share a statement named together once."""
    names: dict[str, list[str]] = {}
    for row in rows:
        for note in row.interval.notes:
            statement = note.removeprefix(f"{row.label}: ")
            if statement == note:
                names.setdefault(note, [])
            else:
                names.setdefault(statement, []).append(row.label)
    return [
        f"{', '.join(dict.fromkeys(labels))}: {statement}" if labels else statement
        for statement, labels in names.items()
    ]


def _scale_rank(scale: str) -> int:
    return _SCALE_ORDER.index(scale) if scale in _SCALE_ORDER else len(_SCALE_ORDER)


def _number_texts(rows: Sequence[_Row]) -> dict[int, str]:
    """Each row's estimate and interval text, with decimals shared across one item's rows.

    The printed summary takes its decimals over a result's whole estimates table, so the rows
    of one item share them across panels.
    """
    nan = float("nan")
    found: dict[int, str] = {}
    for source in sorted({row.source for row in rows}):
        group = [row for row in rows if row.source == source]
        spreads = [row.interval.std_error for row in group]
        values = [row.interval.psi for row in group]
        lowers = [nan if row.interval.lower is None else row.interval.lower for row in group]
        uppers = [nan if row.interval.upper is None else row.interval.upper for row in group]
        texts = _core.format_values([*values, *lowers, *uppers], spreads)
        count = len(group)
        for index, row in enumerate(group):
            if row.interval.drawn:
                text = f"{texts[index]} [{texts[count + index]}, {texts[2 * count + index]}]"
            else:
                text = f"{texts[index]} (no interval)"
            found[id(row)] = text
    return found


def _panel(
    ax: Axes,
    scale: str,
    rows: Sequence[_Row],
    outcome_label: str | None,
    units: str | None,
    texts: Sequence[str],
    *,
    include_null: bool | None,
    truths: Sequence[float],
) -> tuple[str | None, tuple[str, float, float, bool] | None]:
    """One scale's rows on one axes.

    Returns the note of a null left off the axis, and what :func:`_keep_off` needs to keep
    it off: the side, the null and the near data edge (logged for a ratio), and whether the
    axis is logged.
    """
    _core.style_axes(ax, grid="x")
    color = _core.PALETTE[0]
    positions = list(range(len(rows)))[::-1]
    level = rows[0].interval.level
    drawn_label = _core.interval_label(level)
    for y, row in zip(positions, rows, strict=True):
        interval = row.interval
        if interval.lower is not None and interval.upper is not None:
            ax.hlines(
                y,
                interval.lower,
                interval.upper,
                color=color,
                linewidth=2.0,
                label=drawn_label,
                gid="cleverly-interval",
            )
            drawn_label = "_nolegend_"
        ax.plot(
            [interval.psi],
            [y],
            marker="o",
            markersize=7,
            color=color,
            markerfacecolor=color if interval.drawn else _core.FACE,
            markeredgewidth=1.5,
            linestyle="none",
            gid="cleverly-point",
            zorder=3,
        )
    null = _core.null_for(scale)
    off_axis = None
    keep_off = None
    if null is not None and all(row.null_defined for row in rows):
        placed = _null_side(scale, null, rows, truths, include_null)
        if placed is None:
            _core.reference_line(ax, null, orientation="vertical", label="No effect")
        else:
            side, edge = placed
            # Only rows that all have an interval can say none reaches the null.
            reach = " and no interval here reaches it" if _all_intervals(rows) else ""
            off_axis = (
                f"no effect ({null:g}) lies off the axis to the {side}, so its line is not "
                f"drawn{reach}."
            )
            keep_off = (side, null, edge, scale == "ratio")
    if scale == "ratio":
        _core.log_axis(ax, "x")
    ax.set_yticks(positions, [row.label for row in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.spines["left"].set_visible(False)
    outcomes = {row.outcome for row in rows}
    outcome = outcomes.pop() if len(outcomes) == 1 else None
    ax.set_xlabel(_axis_label(scale, outcome_label, outcome, units))
    _numbers(ax, rows, positions, texts)
    return off_axis, keep_off


def _all_intervals(rows: Sequence[_Row]) -> bool:
    """Whether every row has a finite interval: a pair of finite limits."""
    return all(
        _core.finite(row.interval.lower) and _core.finite(row.interval.upper) for row in rows
    )


def _keep_off(ax: Axes, side: str, null: float, edge: float, *, log: bool) -> None:
    """Move the near x limit between the null and the data, so the null is off the axis.

    The autoscale margin, or the room a true value's cross takes, can reach past a null
    that sits close to the data; the limit then goes halfway from the data's edge to the
    null, or to the autoscale's own limit when that already leaves the null out. ``edge``
    and ``null`` are on the drawn scale (logged for a ratio).
    """
    import math

    low, high = ax.get_xlim()
    point = (lambda v: math.log(v)) if log else (lambda v: v)
    back = (lambda v: math.exp(v)) if log else (lambda v: v)
    where = 0.0 if log else null
    middle = back(where + 0.5 * (edge - where))
    if side == "left" and point(low) <= where:
        ax.set_xlim(middle, high)
    elif side == "right" and point(high) >= where:
        ax.set_xlim(low, middle)


def _null_side(
    scale: str,
    null: float,
    rows: Sequence[_Row],
    truths: Sequence[float],
    include_null: bool | None,
) -> tuple[str, float] | None:
    """``None`` when the axis shows the no-effect value, else its side and the data's edge.

    The span is measured on the drawn scale (logged for a ratio) over every point, interval
    limit and true value, and the edge is the data's limit nearest the null on that scale.
    A null inside that span is always shown, whatever ``include_null``. Under the automatic
    rule a panel with a row that has no finite interval always shows it: a point alone
    says nothing about how far the null is.
    """
    import math

    drawn = [row.interval.psi for row in rows]
    drawn += [v for row in rows for v in (row.interval.lower, row.interval.upper) if v is not None]
    drawn += list(truths)
    found = [float(v) for v in drawn if _core.finite(v)]
    if scale == "ratio":
        found = [math.log(v) for v in found if v > 0]
        null = 0.0
    if not found or include_null is True:
        return None
    if include_null is None and not _all_intervals(rows):
        return None
    low, high = min(found), max(found)
    if low <= null <= high:
        return None
    gap = low - null if null < low else null - high
    if include_null is None and gap <= _NULL_REACH * (high - low):
        return None
    return ("left", low) if null < low else ("right", high)


def _numbers(
    ax: Axes, rows: Sequence[_Row], positions: Sequence[int], texts: Sequence[str]
) -> None:
    """The estimate and interval of each row, in a text column right of the axes."""
    level = _core.level_text(rows[0].interval.level)
    any_interval = any(row.interval.drawn for row in rows)
    ax.annotate(
        f"estimate [{level} CI]" if any_interval else "estimate",
        xy=(1.02, 1.0),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=_core.TICK_SIZE,
        color=_core.SECONDARY_INK,
        gid="cleverly-number-header",
    )
    for y, text in zip(positions, texts, strict=True):
        ax.annotate(
            text,
            xy=(1.02, y),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="center",
            fontsize=_core.TICK_SIZE,
            color=_core.INK,
            gid="cleverly-number",
            annotation_clip=False,
        )


__all__ = ["forest"]
