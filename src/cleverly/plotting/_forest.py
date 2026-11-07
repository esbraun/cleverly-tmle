"""V1: the forest plot of estimates with their intervals.

One row per estimate, the point and its interval, the no-effect line, and the numbers in a
column on the right, formatted as the printed summary formats them. Estimates on different
scales go in separate panels, because one axes carries one scale: differences around 0,
ratios on a log axis around 1, and arm means with no null line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The panel order and the x-axis words of each scale.
_SCALE_ORDER = ("difference", "ratio", "level", "fraction")


@dataclass(frozen=True)
class _Row:
    label: str
    interval: _core.Interval
    scale: str
    outcome: str | None


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
        rows.append(
            _Row(label, _core.estimate_interval(estimate, label), str(estimate.scale), outcome)
        )
    return rows


def _expand(item: Any, label: str | None, several: bool) -> list[_Row]:
    """The rows one argument contributes."""
    if _is_estimate(item):
        name = label or str(item.name)
        return [_Row(name, _core.estimate_interval(item, name), str(item.scale), None)]
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
            return [
                _Row(
                    prefix,
                    _core.estimate_interval(estimate, prefix),
                    str(estimate.scale),
                    _outcome(item),
                )
            ]
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


def _axis_label(scale: str, outcome: str | None, units: str | None) -> str:
    unit = f" ({units})" if units else ""
    if scale == "ratio":
        return "Ratio (log scale)"
    if scale == "difference":
        return f"Difference in mean {outcome}{unit}" if outcome else f"Difference{unit}"
    if scale == "fraction":
        return "Share of the mean outcome"
    return f"Mean {outcome}{unit}" if outcome else f"Mean outcome{unit}"


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
    return text[:1].upper() + text[1:]


def forest(
    *items: Any,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    units: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot estimates with their intervals, one row each (a forest plot).

    Use it to compare several estimates, contrasts or methods on one scale and to see at a
    glance which intervals cross the no-effect line. Each row draws the estimate and its
    confidence interval; the numbers are printed in a column on the right with the decimals
    the printed summary uses. Estimates on different scales go in separate panels:
    differences around 0, ratios on a log axis around 1, and arm means with no null line.

    An estimate whose inference the package withholds is drawn as a point with no interval,
    and the note under the figure gives the reason. An experimental estimate says so in the
    note.

    Parameters
    ----------
    *items : TMLEResult, TMLEResultSet, LongitudinalResult, ParameterEstimate or mapping
        What to draw. A fitted result contributes every estimate it reports; a mapping
        contributes each value under its key as the label.
    labels : sequence of str or None, default=None
        One label per item. A result's default label is its method's short name when
        several items are drawn, and nothing when one is.
    title : str or None, default=None
        The title; ``"Estimated effects"`` when omitted.
    subtitle : str or None, default=None
        The line under the title; the printed summary's question for a single result.
    units : str or None, default=None
        Units of the outcome, added to the axis label of differences and means.
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

    See Also
    --------
    cleverly.estimators.TMLEResult.plot : This plot for one fit.
    cleverly.plotting.sensitivity : One estimate across a declared sensitivity parameter.

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
    'Estimated effects'
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
        rows.extend(_expand(item, None if labels is None else labels[index], several))
    scales = sorted({row.scale for row in rows}, key=_scale_rank)
    if ax is not None and len(scales) > 1:
        raise ValueError(
            "these estimates are on more than one scale, which forest() draws as separate "
            "panels; omit ax= or pass the estimates of one scale"
        )
    groups = [[row for row in rows if row.scale == scale] for scale in scales]
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
    for axis, scale, group in zip(axes, scales, groups, strict=True):
        _panel(axis, scale, group, units)
    question = subtitle if subtitle is not None else _question(items)
    _core.legend(axes[0], sources=axes)
    _core.header(axes[0], title or "Estimated effects", question)
    _core.notes(axes[-1], [note for row in rows for note in row.interval.notes])
    return fig


def _scale_rank(scale: str) -> int:
    return _SCALE_ORDER.index(scale) if scale in _SCALE_ORDER else len(_SCALE_ORDER)


def _panel(ax: Axes, scale: str, rows: Sequence[_Row], units: str | None) -> None:
    """One scale's rows on one axes."""
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
    if null is not None:
        _core.reference_line(ax, null, orientation="vertical", label="No effect")
    if scale == "ratio":
        _core.log_axis(ax, "x")
    ax.set_yticks(positions, [row.label for row in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.spines["left"].set_visible(False)
    outcomes = {row.outcome for row in rows}
    outcome = outcomes.pop() if len(outcomes) == 1 else None
    ax.set_xlabel(_axis_label(scale, outcome, units))
    _numbers(ax, rows, positions)


def _numbers(ax: Axes, rows: Sequence[_Row], positions: Sequence[int]) -> None:
    """The estimate and interval of each row, in a text column right of the axes."""
    spreads = [row.interval.std_error for row in rows]
    values = [row.interval.psi for row in rows]
    nan = float("nan")
    lowers = [nan if row.interval.lower is None else row.interval.lower for row in rows]
    uppers = [nan if row.interval.upper is None else row.interval.upper for row in rows]
    texts = _core.format_values([*values, *lowers, *uppers], spreads)
    count = len(rows)
    level = _core.level_text(rows[0].interval.level)
    ax.annotate(
        f"estimate [{level} CI]",
        xy=(1.02, 1.0),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=_core.TICK_SIZE,
        color=_core.SECONDARY_INK,
        gid="cleverly-number-header",
    )
    for index, (y, row) in enumerate(zip(positions, rows, strict=True)):
        if row.interval.drawn:
            text = f"{texts[index]} [{texts[count + index]}, {texts[2 * count + index]}]"
        else:
            text = f"{texts[index]} (no interval)"
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
