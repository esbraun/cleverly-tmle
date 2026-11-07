"""V18: the confidence intervals of a repeated-sampling study, one per replication, sorted.

Each replication of a :class:`~cleverly.validation.CoverageStudy` reports an interval for
a quantity whose true value the study knows. The zipper plot draws every interval as a
horizontal line, measured from that true value and sorted by where its centre falls, so
the intervals that miss the truth gather at the two ends. It shows what a coverage
number hides: whether the misses fall on one side (bias) or on both (too narrow an
interval).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Panels per row.
_COLUMNS = 3


def _on_log_scale(summary: Any) -> bool:
    """Whether the study measured this estimand on the log scale, which is a ratio."""
    shifted = summary.inference_estimates
    if shifted is None:
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.asarray(summary.estimates, dtype=float))
    return bool(np.allclose(shifted, logs, equal_nan=True))


def _positions(records: Sequence[Any], *, ratio: bool) -> dict[str, Any]:
    """Each interval's limits and centre, measured from its own true value.

    A difference is measured as limit minus truth (the truth sits at 0) and a ratio as
    limit over truth (the truth sits at 1, and centres are geometric means).
    """
    from ..exceptions import DataError

    lower = np.asarray([record.lower for record in records], dtype=float)
    upper = np.asarray([record.upper for record in records], dtype=float)
    truth = np.asarray([record.truth for record in records], dtype=float)
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise DataError(
            "the study's records hold no finite interval limits, so there is no interval to draw"
        )
    if ratio:
        low, high = lower / truth, upper / truth
        centre = np.sqrt(low * high)
        reference = 1.0
    else:
        low, high = lower - truth, upper - truth
        centre = (low + high) / 2.0
        reference = 0.0
    contains = (low <= reference) & (reference <= high)
    flagged = np.asarray([record.covered for record in records], dtype=bool)
    if not np.array_equal(contains, flagged):
        raise DataError(
            "a replication's interval limits disagree with its recorded coverage, so the "
            "figure would draw a different verdict than the study reported"
        )
    return {"low": low, "high": high, "centre": centre, "covered": contains}


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def strip_figure(
    panels: int,
    columns: int,
    *,
    panel: tuple[float, float],
    extra_height: float,
    sharey: bool = False,
) -> tuple[Figure, Axes, list[Axes], Axes]:
    """A grid of panels with a full-width strip above for the header and one below for notes.

    The title, subtitle and key sit on the top strip and the notes on the bottom one, so a
    long line never widens the first panel. Shared with the other multi-panel plots of this
    package that have more than one panel across.

    Parameters
    ----------
    panels : int
        Panels to draw.
    columns : int
        Panels per row.
    panel : tuple of float
        One panel's width and height in inches.
    extra_height : float
        Inches added for the header and notes.
    sharey : bool, default=False
        Whether the panels share their y axis.

    Returns
    -------
    tuple of Figure, Axes, list of Axes and Axes
        The figure, the header strip, the panels in reading order and the notes strip.
    """
    rows = math.ceil(panels / columns)
    fig, grid = _core.new_figure(
        rows + 2,
        ncols=columns,
        figsize=(max(5.5, panel[0] * columns + 0.8), extra_height + panel[1] * rows),
        height_ratios=[0.02] + [panel[1]] * rows + [0.02],
    )
    spec: Any = grid[0].get_gridspec()
    for axis in grid:
        axis.remove()
    head = fig.add_subplot(spec[0, :])
    axes: list[Axes] = []
    for index in range(panels):
        share = axes[0] if sharey and axes else None
        axes.append(fig.add_subplot(spec[1 + index // columns, index % columns], sharey=share))
    foot = fig.add_subplot(spec[rows + 1, :])
    for axis in axes:
        _core.style_axes(axis)
    for strip in (head, foot):
        strip.set_xticks([])
        strip.set_yticks([])
        strip.patch.set_visible(False)
        for spine in strip.spines.values():
            spine.set_visible(False)
    return fig, head, axes, foot


def zipper(
    study: Any,
    *,
    estimand: str | Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot every replication's interval, sorted by centre, with the misses coloured.

    Use it to see how a study's coverage was lost. Each line is one replication's
    interval, drawn relative to the true value (the dashed line) and sorted by its
    centre. Intervals that miss are orange. Misses piled at one end mean the estimator
    is biased; misses split between the ends mean the intervals are too narrow. The
    subtitle gives the coverage and the level it was aimed at.

    A difference is drawn as the limit minus the true value, a ratio as the limit divided
    by it on a log axis. Both are drawn from the limits each record holds, and the figure
    refuses a study whose limits disagree with the coverage it recorded.

    Parameters
    ----------
    study : StudyResult
        A finished coverage study.
    estimand : str, sequence of str or None, default=None
        The estimands to draw, one panel each; every estimand of the study when omitted.
    labels : dict or None, default=None
        Words for each estimand, keyed by its alias.
    title : str or None, default=None
        The title.

    Returns
    -------
    Figure
        One panel per estimand, up to three across.

    Raises
    ------
    DataError
        When the study's records hold no finite interval limits, or the limits disagree
        with the recorded coverage.
    KeyError
        When ``estimand`` names an estimand the study did not measure.

    See Also
    --------
    cleverly.validation.CoverageStudy : The study that produces the records.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import nonlinear_dgp
    >>> from cleverly.estimators import TMLE
    >>> from cleverly.plotting import zipper
    >>> from cleverly.validation import CoverageStudy
    >>> study = CoverageStudy(
    ...     dgp=nonlinear_dgp(),
    ...     estimator=lambda: TMLE(
    ...         outcome_learner=LinearRegression(),
    ...         treatment_learner=LogisticRegression(max_iter=1000),
    ...         cross_fit=False,
    ...     ),
    ...     n=200,
    ...     n_replicates=12,
    ...     estimands=("ate",),
    ...     seed=0,
    ... ).run()
    >>> fig = zipper(study)
    >>> fig.axes[0].get_title(loc="left")
    'Do the intervals cover the true value?'
    >>> plt.close(fig)
    """
    names = (
        list(study.summaries)
        if estimand is None
        else ([estimand] if isinstance(estimand, str) else list(estimand))
    )
    for name in names:
        if name not in study.summaries:
            raise KeyError(f"the study did not measure {name!r}; it did {list(study.summaries)}")
    columns = min(len(names), _COLUMNS)
    fig, head, axes, foot = strip_figure(len(names), columns, panel=(3.3, 3.0), extra_height=1.2)
    coverage: list[str] = []
    found: list[str] = []
    for axis, name in zip(axes, names, strict=False):
        records = [record for record in study.replications if record.estimand == name]
        ratio = _on_log_scale(study.summaries[name])
        place = _positions(records, ratio=ratio)
        order = np.argsort(place["centre"], kind="stable")
        height = np.arange(1, len(order) + 1)
        low, high, hit = place["low"][order], place["high"][order], place["covered"][order]
        for mask, color, text, gid in (
            (hit, _core.PALETTE[0], "Covers the true value", "cleverly-zipper-covered"),
            (~hit, _core.PALETTE[1], "Misses the true value", "cleverly-zipper-missed"),
        ):
            if mask.any():
                axis.hlines(
                    height[mask],
                    low[mask],
                    high[mask],
                    colors=color,
                    linewidth=1.2,
                    alpha=0.6 if text.startswith("Covers") else 1.0,
                    label=text,
                    gid=gid,
                    zorder=2.0,
                )
        reference = 1.0 if ratio else 0.0
        _core.reference_line(axis, reference, orientation="vertical", label="True value")
        axis.set_ylim(0.0, len(order) + 1.0)
        axis.set_yticks([])
        axis.set_ylabel("Replications, sorted by interval centre" if axis is axes[0] else "")
        shown = (labels or {}).get(name, name)
        if ratio:
            _core.log_axis(axis, "x")
            axis.set_xlabel(f"{shown}\n(limit as a multiple of the true value)")
        else:
            axis.set_xlabel(f"{shown}\n(limit minus the true value)")
        coverage.append(f"{shown} {_percent(float(np.mean(place['covered'])))}")
        below = int(np.sum(place["high"] < reference))
        above = int(np.sum(place["low"] > reference))
        found.append(f"{shown}: {below} intervals fall wholly below the true value, {above} above.")
    _core.legend(head, sources=axes[:1])
    _core.header(
        head,
        title or "Do the intervals cover the true value?",
        f"Coverage over {study.n_replicates} replications of n = {study.n}: "
        + ", ".join(coverage)
        + f" (aimed at {_percent(1.0 - float(study.alpha))})",
    )
    if study._diagnostic:
        found.append(
            f"These intervals are {study._diagnostic_noun}s, not confidence intervals; "
            "the figure shows what the study measured."
        )
    if any(study.summaries[name].support == "experimental" for name in names):
        found.append("Some estimates in this study are experimental.")
    if study.n_failed:
        found.append(f"{study.n_failed} replications failed and are not drawn.")
    _core.notes(foot, found)
    return fig


__all__ = ["zipper"]
