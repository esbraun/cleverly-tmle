"""V10 and V11: the weights a fit divided by, and whether weighting balanced the covariates.

:func:`weights` shows, node by node, how the weights of a longitudinal fit are spread and
where the fit's limits on the cumulative probabilities replaced a probability. It shows at
which decision point positivity breaks down, which one summary number hides.
:func:`balance` is the love plot: each covariate's standardized mean difference between
treatment groups before and after weighting, the check of the treatment model on the
covariates the reader knows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core
from ._zipper import strip_figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

#: Nodes whose weight is the data's weight over the cumulative probability alone, so the
#: bound on that probability caps it. A policy or modified-treatment node divides a varying
#: numerator by it, and a tilt node's factor is never bounded.
_CAPPED_KINDS = ("label", "rule")
#: Series at one axes: the palette's three slots.
_BOX_WIDTH = 0.7

# ---------------------------------------------------------------------------- V10


def _box(values: Any) -> dict[str, Any]:
    """The box statistics of the used weights: the middle half, the median, the 1st to 99th."""
    low, q1, median, q3, high = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "med": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "whislo": float(low),
        "whishi": float(high),
        "fliers": [],
    }


def _percent_text(share: float) -> str:
    if share == 0.0:
        return "0%"
    if share < 0.01:
        return "<1%"
    return f"{100.0 * share:.0f}%"


def weights(
    result: Any,
    *,
    regimen: str | None = None,
    horizon: int | None = None,
    repeat: int = 1,
    labels: dict[str, str] | None = None,
    time_label: str = "Time point",
    title: str | None = None,
) -> Figure:
    """Plot how the weights are spread at each time point, and how many units they leave.

    Use it to see at which decision point the weights grow too large. The top panel shows,
    for each time point, the weight each unit still followed carried: the box spans the
    middle half, the whiskers the 1st to 99th percentile, the cross is the largest. The
    dotted line marks the largest weight the fit's limit on the cumulative probability
    allows, with the share of units the limit moved printed above. The bottom panel shows
    the effective sample size, the number of equally weighted units that would carry the
    same information, against the number still followed. A gap that widens at a time point
    is where the regimen loses its support.

    A fit with several horizons draws the largest, a fit with several cross-fitting repeats
    the first, and the note says which. A time point whose weights the limit does not cap
    (an incremental or modified policy, or data weights other than one) has no dotted line.

    Parameters
    ----------
    result : LongitudinalResult
        A fitted inverse-weighting or targeted longitudinal result.
    regimen : str or None, default=None
        One regimen's label; every regimen of the result, up to three, when omitted.
    horizon : int or None, default=None
        The horizon of a curve fit; the largest when omitted.
    repeat : int, default=1
        The cross-fitting repeat, counted from one.
    labels : dict or None, default=None
        Words for each regimen, keyed by its label.
    time_label : str, default="Time point"
        Words for a node on the x axis.
    title : str or None, default=None
        The title.

    Returns
    -------
    Figure
        Two panels with a shared x axis.

    Raises
    ------
    CapabilityError
        When the result is a g-computation or parametric g-formula fit, which has no weights.
    DataError
        When the result holds more than three regimens and none is named, or the regimen,
        horizon or repeat is not in the result.

    See Also
    --------
    cleverly.longitudinal.node_weights : The same weights as numbers.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import LTMLE
    >>> from cleverly.plotting import weights
    >>> frame, _ = make_longitudinal(n=200, seed=0)
    >>> result = LTMLE(
    ...     {"always": 1, "never": 0},
    ...     n_folds=1,
    ...     random_state=0,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> fig = weights(result)
    >>> fig.axes[0].get_title(loc="left")
    'Where do the weights grow too large?'
    >>> plt.close(fig)
    """
    from ..exceptions import DataError
    from ..longitudinal.weights import node_weights

    fits = result.repeats[repeat - 1].fits if 1 <= repeat <= len(result.repeats) else {}
    names = (
        [regimen]
        if regimen is not None
        else list(dict.fromkeys(f.regimen.label for f in fits.values()))
    )
    if len(names) > _core.MAX_SERIES:
        raise DataError(
            f"the result holds {len(names)} regimens and one panel draws at most "
            f"{_core.MAX_SERIES}; name one with regimen="
        )
    series = [node_weights(result, regimen=name, horizon=horizon, repeat=repeat) for name in names]
    first = series[0]
    times = first.times
    fig, (top, bottom) = _core.new_figure(
        2, figsize=(7.0, 6.2), sharex=True, height_ratios=[3.0, 2.0]
    )
    count = len(series)
    step = _BOX_WIDTH / count
    capped_somewhere = False
    unit = all(item.unit_weights for item in series)
    for index, item in enumerate(series):
        color = _core.PALETTE[index]
        shown = (labels or {}).get(item.regimen, item.regimen)
        offset = (index - (count - 1) / 2.0) * step
        stats, positions, peaks = [], [], []
        for node, time in enumerate(times):
            used = item.used[item.followed[:, node], node]
            used = used[np.isfinite(used) & (used > 0)]
            if not used.size:
                continue
            stats.append(_box(used))
            positions.append(time + offset)
            peaks.append(float(used.max()))
        if stats:
            top.bxp(
                stats,
                positions=positions,
                widths=step * 0.85,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color, "linewidth": 1.0},
                medianprops={"color": color, "linewidth": 1.6},
                whiskerprops={"color": color, "linewidth": 1.0},
                capprops={"color": color, "linewidth": 1.0},
                manage_ticks=False,
            )
            top.scatter(
                positions,
                peaks,
                marker="x",
                s=22,
                color=color,
                linewidths=1.2,
                zorder=3,
                gid="cleverly-weight-max",
            )
        for node, time in enumerate(times):
            kind = item.node_kinds[node]
            if kind not in _CAPPED_KINDS or not item.unit_weights or item.bounds[0] <= 0.0:
                continue
            capped_somewhere = True
            limit = 1.0 / item.bounds[0]
            line = top.plot(
                [time + offset - step * 0.5, time + offset + step * 0.5],
                [limit, limit],
                color=_core.SECONDARY_INK,
                linestyle=(0, (1, 2)),
                linewidth=1.4,
                zorder=2.5,
                label=f"Largest weight the limit allows ({limit:g})"
                if len({item.bounds for item in series}) == 1
                else "Largest weight the limit allows",
                gid="cleverly-marker",
            )[0]
            del line
            share = item.share_beyond_bounds[node]
            top.annotate(
                _percent_text(float(share)),
                xy=(time + offset, limit),
                xytext=(0.0, 3.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=_core.TICK_SIZE - 1.0,
                color=_core.SECONDARY_INK,
                gid="cleverly-tail-share",
            )
        effective = item.effective_n
        followers = item.followed.sum(axis=0).astype(float)
        bottom.plot(
            times,
            followers,
            marker="o",
            markerfacecolor=_core.FACE,
            markeredgecolor=_core.CONTEXT,
            color=_core.CONTEXT,
            linewidth=1.0,
            label="Units still followed",
            gid="cleverly-followed",
        )
        bottom.plot(
            times,
            effective,
            marker="o",
            color=color,
            linewidth=1.8,
            gid="cleverly-effective-n",
        )
        if count > 1:
            _core.end_label(bottom, times[-1], float(effective[-1]), shown, color)
    _core.log_axis(top, "y")
    top.set_ylabel("Weight each unit carried (log scale)")
    ceiling = max(
        [top.get_ylim()[1]]
        + [1.0 / item.bounds[0] * 1.6 for item in series if capped_somewhere and item.bounds[0] > 0]
    )
    top.set_ylim(top.get_ylim()[0], ceiling)
    bottom.set_ylabel("Effective sample size")
    bottom.set_xlabel(time_label)
    bottom.set_xticks(list(times))
    bottom.set_xlim(times[0] - 0.7, times[-1] + 0.7)
    bottom.set_ylim(0.0, max(float(item.followed.sum(axis=0).max()) for item in series) * 1.1)
    _core.legend(top, sources=[top, bottom])
    only = series[0]
    cause = f", cause {only.cause}" if only.cause is not None else ""
    shown_names = ", ".join((labels or {}).get(item.regimen, item.regimen) for item in series)
    _core.header(
        top,
        title or "Where do the weights grow too large?",
        f"Weights used at each {time_label.lower()} by units still followed, regimen {shown_names}",
    )
    found = [
        f"Horizon {only.horizon}{cause}, cross-fitting repeat {only.repeat} of {only.n_repeats}.",
        "Boxes span the middle half of the weights, whiskers the 1st to 99th percentile, "
        "crosses the largest weight.",
    ]
    if capped_somewhere:
        found.append(
            "The percentage above each box is the share of units whose cumulative probability "
            "the limit replaced."
        )
    skipped = sorted({kind for item in series for kind in item.node_kinds} - set(_CAPPED_KINDS))
    if skipped:
        found.append(
            "No limit line is drawn at a time point of kind "
            + ", ".join(skipped)
            + ", whose weight the limit does not cap."
        )
    if not unit:
        found.append("The data's weights multiply the limit, so no limit line is drawn.")
    found.append("Descriptive: the weights are read from the fit, with no interval.")
    _core.notes(bottom, found)
    return fig


# ---------------------------------------------------------------------------- V11


def _balance_object(source: Any) -> Any:
    from ..sensitivity.balance import CovariateBalance, covariate_balance

    return source if isinstance(source, CovariateBalance) else covariate_balance(source)


def balance(
    source: Any,
    *,
    max_covariates: int = 30,
    treatment_label: str | None = None,
    labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot each covariate's absolute standardized difference before and after weighting.

    Use it to check the treatment model on the covariates you know. Each row is a
    covariate: the hollow marker is the absolute difference in its mean between the two
    treatment groups, in pooled standard deviations, and the filled marker is the same
    difference after weighting each unit by the inverse of the probability its own arm was
    given. Weighting has worked on a covariate when the filled marker sits left of the
    dashed line, at 0.1, the usual threshold for a small difference. Covariates are sorted
    by their difference before weighting.

    With more than two arms each arm is compared with all the others, one panel per arm.
    A categorical covariate has a row per level. The figure is descriptive: it carries no
    interval or test.

    Parameters
    ----------
    source : CovariateBalance or TMLEResult
        The balance, or a fitted point-treatment result to compute it from.
    max_covariates : int, default=30
        The most covariates to draw, the ones with the largest difference before weighting.
    treatment_label : str or None, default=None
        Words for the treatment in the subtitle; its column name when omitted.
    labels : dict or None, default=None
        Words for each covariate, keyed by its name.
    title : str or None, default=None
        The title.

    Returns
    -------
    Figure
        One panel per contrast, up to three across.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism.
    DataError
        When the treatment is continuous.

    See Also
    --------
    cleverly.sensitivity.covariate_balance : The same differences as numbers.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import balance
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     cross_fit=False,
    ...     random_state=0,
    ... )
    >>> fig = balance(result)
    >>> fig.axes[0].get_title(loc="left")
    'Covariate balance before and after weighting'
    >>> plt.close(fig)
    """
    summary = _balance_object(source)
    contrasts = summary.contrasts
    covariates = list(dict.fromkeys(row.covariate for row in summary.rows))
    largest = {
        name: max(
            (
                abs(row.smd_unweighted)
                for row in summary.rows
                if row.covariate == name and np.isfinite(row.smd_unweighted)
            ),
            default=0.0,
        )
        for name in covariates
    }
    order = sorted(covariates, key=lambda name: (-largest[name], covariates.index(name)))
    shown = order[:max_covariates]
    hidden = len(order) - len(shown)
    columns = min(len(contrasts), 3)
    fig, head, axes, foot = strip_figure(
        len(contrasts),
        columns,
        panel=(3.6, 0.27 * len(shown) + 1.1),
        extra_height=1.5,
        sharey=True,
    )
    ceiling = max(
        [summary.threshold * 1.6]
        + [
            abs(value)
            for row in summary.rows
            if row.covariate in shown
            for value in (row.smd_unweighted, row.smd_weighted)
            if np.isfinite(value)
        ]
    )
    heights = np.arange(len(shown))[::-1] + 0.0
    flat_ink = _core.INK
    for axis, contrast in zip(axes, contrasts, strict=True):
        rows = {row.covariate: row for row in summary.rows_of(contrast)}
        before = np.asarray([abs(rows[name].smd_unweighted) for name in shown])
        after = np.asarray([abs(rows[name].smd_weighted) for name in shown])
        both = np.isfinite(before) & np.isfinite(after)
        axis.hlines(
            heights[both], after[both], before[both], color=_core.CONTEXT, linewidth=1.2, zorder=1
        )
        axis.scatter(
            before,
            heights,
            s=34,
            facecolors=_core.FACE,
            edgecolors=_core.PALETTE[1],
            linewidths=1.6,
            zorder=3,
            gid="cleverly-balance-before",
        )
        axis.scatter(
            after,
            heights,
            s=34,
            color=_core.PALETTE[0],
            zorder=4,
            gid="cleverly-balance-after",
        )
        _core.reference_line(
            axis,
            summary.threshold,
            orientation="vertical",
            label=f"Threshold, {summary.threshold:g}",
        )
        axis.set_xlim(0.0, ceiling * 1.08)
        axis.set_yticks(heights)
        axis.set_yticklabels(
            [(labels or {}).get(name, name) for name in shown],
            fontsize=_core.TICK_SIZE,
            color=flat_ink,
        )
        axis.set_ylim(-0.7, len(shown) + 0.75)
        if axis is not axes[0]:
            axis.tick_params(labelleft=False)
        axis.grid(False)
        axis.grid(True, axis="x", color=_core.GRID, linewidth=0.8)
        axis.set_xlabel("Absolute standardized mean difference")
        if len(contrasts) > 1:
            axis.set_title(
                contrast.replace(" vs the other arms", ""),
                loc="left",
                fontsize=_core.SUBTITLE_SIZE,
                color=_core.INK,
            )
    # Direct labels of the two markers above the top row of the first panel, each over its own
    # marker and on its own line, so they never meet.
    top_row = {row.covariate: row for row in summary.rows_of(contrasts[0])}[shown[0]]
    reach = axes[0].get_xlim()[1]
    items = [
        (abs(top_row.smd_unweighted), "Before weighting"),
        (abs(top_row.smd_weighted), "After weighting"),
    ]
    items.sort(key=lambda item: item[0])
    for position, (x, text) in enumerate(items):
        if not np.isfinite(x):
            continue
        near_left = x < 0.45 * reach
        axes[0].annotate(
            text,
            xy=(x, len(shown) - 1 + 0.55 + 0.7 * position),
            xytext=(-3.0 if near_left else 3.0, 0.0),
            textcoords="offset points",
            ha="left" if near_left else "right",
            va="center",
            fontsize=_core.TICK_SIZE,
            color=_core.INK,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": _core.FACE, "edgecolor": "none"},
            gid="cleverly-series-label",
        )
    axes[0].set_ylabel("")
    _core.legend(head, sources=axes[:1])
    name = treatment_label or summary.treatment
    finite = [row for row in summary.rows if np.isfinite(row.smd_weighted)]
    worst = max(finite, key=lambda row: abs(row.smd_weighted)) if finite else None
    pieces = [f"Treatment {name}" if len(contrasts) > 1 else contrasts[0]]
    if worst is not None:
        pieces.append(
            f"largest after weighting {(labels or {}).get(worst.covariate, worst.covariate)} "
            f"{abs(worst.smd_weighted):.3f}"
        )
    _core.header(
        head,
        title or "Covariate balance before and after weighting",
        "; ".join(pieces),
    )
    found = [
        "Descriptive: these differences carry no interval or test. Each is a difference in "
        "group means over the standard deviation of the two groups together.",
        "After weighting, each unit counts as the inverse of the probability the fit gave its own "
        "arm"
        + (
            ""
            if summary.limits is None
            else f", within [{summary.limits[0]:g}, {summary.limits[1]:g}]"
        )
        + (
            ""
            if summary.n_repeats == 1
            else f", from the first of {summary.n_repeats} repeats' out-of-fold predictions"
        )
        + ".",
    ]
    if hidden:
        found.append(
            f"{hidden} covariates with the smallest differences before weighting are not drawn."
        )
    undefined = sorted(
        {row.covariate for row in summary.rows if not np.isfinite(row.smd_unweighted)}
    )
    if undefined:
        found.append(f"No difference is drawn for {', '.join(undefined)}, which does not vary.")
    _core.notes(foot, found)
    return fig


__all__: Sequence[str] = ["balance", "weights"]
