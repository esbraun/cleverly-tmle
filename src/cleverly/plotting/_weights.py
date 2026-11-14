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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import _core
from ._strips import fit_height, strip_figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Nodes whose weight is the data's weight over the cumulative probability alone, so the
#: bound on that probability caps it. A policy or modified-treatment node divides a varying
#: numerator by it, and a tilt node's factor is never bounded. A held node repeats the
#: baseline decision and adds no treatment factor.
_CAPPED_KINDS = ("label", "rule", "held")
#: Plain words for each node kind whose weight the limit does not cap, as the note names it.
_UNCAPPED_WORDS = {
    "policy": "a policy that gives each treatment a set probability",
    "tilt": "an incremental policy that scales the odds of treatment",
    "mtp": "a modified treatment policy that shifts each unit's treatment",
}
#: The width one time point's boxes share.
_BOX_WIDTH = 0.7
#: The least vertical distance between two end labels, in points.
_LABEL_GAP = 11.0

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


def _capped(kinds: Sequence[str], node: int) -> bool:
    """Whether the limit caps the weight at ``node``: every node up to it is a label or rule.

    The weight at a node is a product over the nodes before it, so a policy or tilt factor
    at an earlier node carries into it, and ``1 / bound`` is no longer its largest value.
    """
    return all(kind in _CAPPED_KINDS for kind in kinds[: node + 1])


def _end_labels(ax: Axes, x: float, items: list[tuple[float, str, str]]) -> None:
    """End labels at ``x``, moved apart vertically so no two overlap.

    ``items`` holds each series' last value, its label and its colour. Labels closer than
    :data:`_LABEL_GAP` points are spread around their mean; the axes' size after layout
    sets the conversion, so call this after everything else on the figure is drawn.
    """
    if not items:
        return
    fig: Any = ax.get_figure(root=True)
    fig.draw_without_rendering()
    low, high = ax.get_ylim()
    points = ax.bbox.height * 72.0 / fig.dpi
    gap = _LABEL_GAP * (high - low) / max(points, 1.0)
    ordered = sorted(items, key=lambda item: item[0])
    placed = [ordered[0][0]]
    for value, _, _ in ordered[1:]:
        placed.append(max(value, placed[-1] + gap))
    # Centre the moved block on the values it labels, so it does not drift upward.
    shift = float(np.mean([item[0] for item in ordered]) - np.mean(placed))
    shift = max(shift, low - placed[0])
    for (_, text, color), y in zip(ordered, placed, strict=True):
        _core.end_label(ax, x, y + shift, text, color)


@_core.labelled
def weights(
    result: Any,
    *,
    regimens: Sequence[str] | None = None,
    horizon: int | None = None,
    cause: str | None = None,
    repeat: int = 1,
    labels: dict[str, str] | None = None,
    time_label: str = "Time point",
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot how the weights are spread at each time point, and how many units they leave.

    Use it to see at which decision point the weights grow too large. The top panel shows,
    for each time point, the weight each unit still followed carried: the box spans the
    middle half, the whiskers the 1st to 99th percentile, the cross is the largest. The
    dotted line marks the largest weight the fit's limit on the cumulative probability
    allows, with the share of units the limit moved printed above. The bottom panel shows
    the effective sample size (solid), the number of equally weighted units that would
    carry the same information, against the number still followed (dashed). A gap that
    widens at a time point is where the regimen loses its support.

    A fit with several horizons draws the largest, a fit with several cross-fitting repeats the
    first, and :func:`~cleverly.plotting.notes` says which. A time point whose weight the limit does
    not cap has no dotted line: one with an incremental or modified policy at it or at an earlier
    time point, whose factor the weight carries, or any time point when the data's weights are not
    all one. A log axis cannot show a weight of zero, which a modified treatment policy at that or
    an earlier time point gives a unit when the shifted treatment cannot produce the one it
    received; :func:`~cleverly.plotting.notes` counts the units still followed with a weight of zero
    at each time point, and the boxes describe the other weights.

    Parameters
    ----------
    result : LongitudinalResult
        A fitted inverse-weighting or targeted longitudinal result.
    regimens : sequence of str or None, default=None
        The labels of the regimens to draw, at most three, in the order given; every
        regimen of the result, up to three, when omitted.
    horizon : int or None, default=None
        The horizon of a curve fit; the largest when omitted.
    cause : str or None, default=None
        The cause, under competing events; the first when omitted.
    repeat : int, default=1
        The cross-fitting repeat, counted from one.
    labels : dict or None, default=None
        Words for each regimen, keyed by its label.
    time_label : str, default="Time point"
        Words for a node on the x axis.
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

    Returns
    -------
    Figure
        Two panels with a shared x axis.

    Raises
    ------
    CapabilityError
        When the result is not a longitudinal fit, or is a g-computation or parametric
        g-formula fit, which has no weights.
    DataError
        When the result holds more than three regimens and ``regimens`` is omitted, when
        ``regimens`` is a single string, empty, repeats a label or names more than three,
        or when a regimen, the cause, the horizon or the repeat is not in the result.

    See Also
    --------
    cleverly.longitudinal.node_weights : The same weights as numbers.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say which horizon and repeat
    are drawn, where no limit line applies and why, and how many units carry a weight of
    zero.

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
    'Weights over time'
    >>> plt.close(fig)
    """
    from .._refusal import NO_ALTERNATIVE, RefusalReason
    from ..exceptions import CapabilityError, DataError
    from ..longitudinal.estimator import LongitudinalResult
    from ..longitudinal.weights import node_weights
    from ..survival.result import ContinuousSurvivalResult

    if isinstance(result, ContinuousSurvivalResult):
        raise CapabilityError(
            "weights() does not yet draw a continuous-time survival fit's weights, the inverse "
            "of each row's treatment probability times its probability of staying uncensored "
            "over time.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X100",
        )
    if not isinstance(result, LongitudinalResult):
        point_fit = hasattr(result, "nuisance") and hasattr(result, "repeats")
        raise CapabilityError(
            "weights() draws the cumulative inverse-probability weights of a longitudinal fit "
            f"at each time point; this {type(result).__name__} "
            + (
                "fits its treatment once, so it has no weights over time"
                if point_fit
                else "has no sequential weights"
            )
            + ".",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    repeats = result.repeats
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= len(repeats):
        raise DataError(f"repeat must be an integer from 1 to {len(repeats)}; got {repeat!r}")
    fits = repeats[repeat - 1].fits
    held = list(dict.fromkeys(f.regimen.label for f in fits.values()))
    if regimens is None:
        names = held
        if len(names) > _core.MAX_SERIES:
            raise DataError(
                f"the result holds {len(names)} regimens and one panel draws at most "
                f"{_core.MAX_SERIES}; choose up to {_core.MAX_SERIES} of {held} with regimens="
            )
    else:
        names = _chosen_regimens(regimens, held)
    series = [
        node_weights(result, regimen=name, horizon=horizon, cause=cause, repeat=repeat)
        for name in names
    ]
    first = series[0]
    times = first.times
    fig, (top, bottom) = _core.new_figure(
        2, figsize=(7.0, 6.4), sharex=True, height_ratios=[3.0, 2.0]
    )
    count = len(series)
    step = _BOX_WIDTH / count
    capped_somewhere = False
    limits = {round(1.0 / item.bounds[0], 12) for item in series if item.bounds[0] > 0.0}
    shown_names = [(labels or {}).get(item.regimen, item.regimen) for item in series]
    ends: list[tuple[float, str, str]] = []
    zeros: list[_Zeros] = []
    for index, item in enumerate(series):
        color = _core.series_color(index)
        shown = shown_names[index]
        offset = (index - (count - 1) / 2.0) * step
        stats, positions, peaks = [], [], []
        for node, time in enumerate(times):
            used = item.used[item.followed[:, node], node]
            zero = int(np.sum(used == 0.0))
            if zero:
                zeros.append(
                    _Zeros(
                        shown,
                        time,
                        zero,
                        int(used.size),
                        shifted="mtp" in item.node_kinds[: node + 1],
                        unit_weights=item.unit_weights,
                    )
                )
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
            if count > 1:
                # Identity never rests on colour alone: each box carries its regimen's name.
                for position, box in zip(positions, stats, strict=True):
                    top.annotate(
                        shown,
                        xy=(position, box["whislo"]),
                        xytext=(0.0, -3.0),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=_core.TICK_SIZE - 1.0,
                        color=_core.INK,
                        gid=_core.GID_SERIES_LABEL,
                    )
        for node, time in enumerate(times):
            share = float(item.share_beyond_bounds[node])
            if (
                not _capped(item.node_kinds, node)
                or not item.unit_weights
                or item.bounds[0] <= 0.0
                or not np.isfinite(share)
            ):
                continue
            capped_somewhere = True
            limit = 1.0 / item.bounds[0]
            top.plot(
                [time + offset - step * 0.5, time + offset + step * 0.5],
                [limit, limit],
                color=_core.SECONDARY_INK,
                linestyle=(0, (1, 2)),
                linewidth=1.4,
                zorder=2.5,
                label=f"Largest weight the limit allows ({limit:.3g})"
                if len(limits) == 1
                else "Largest weight the limit allows",
                gid=_core.GID_MARKER,
            )
            top.annotate(
                _percent_text(share),
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
            markersize=4,
            markerfacecolor=_core.FACE,
            markeredgecolor=color,
            color=color,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
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
            ends.append((float(followers[-1]), f"{shown}: units still followed", color))
            ends.append((float(effective[-1]), f"{shown}: effective sample size", color))
        else:
            ends.append((float(followers[-1]), "Units still followed", color))
            ends.append((float(effective[-1]), "Effective sample size", color))
    _core.log_axis(top, "y")
    top.set_ylabel("Weight each unit carried (log scale)")
    ceiling = max(
        [top.get_ylim()[1]]
        + [1.0 / item.bounds[0] * 1.6 for item in series if capped_somewhere and item.bounds[0] > 0]
    )
    top.set_ylim(top.get_ylim()[0] / (1.25 if count > 1 else 1.0), ceiling)
    bottom.set_ylabel("Units")
    bottom.set_xlabel(time_label)
    bottom.set_xticks(list(times))
    bottom.set_xlim(times[0] - 0.7, times[-1] + 0.7)
    bottom.set_ylim(0.0, max(float(item.followed.sum(axis=0).max()) for item in series) * 1.1)
    cause_words = f", cause {first.cause}" if first.cause is not None else ""
    _core.header(
        top,
        title or "Weights over time",
        f"Weights used at each {time_label.lower()} by units still followed, regimen "
        + ", ".join(shown_names),
        key_sources=[top],
    )
    where = "Horizon" if first.reports_curve else "Final time point"
    if first.n_repeats > 1:
        source = f", cross-fitting repeat {first.repeat} of {first.n_repeats}"
    elif first.cross_fitted:
        source = ", out-of-fold probabilities"
    else:
        source = ", in-sample probabilities (the fit was not cross-fitted)"
    found = [
        f"{where} {first.horizon}{cause_words}{source}.",
        "Boxes span the middle half of the weights, whiskers the 1st to 99th percentile, "
        "crosses the largest weight. In the lower panel, solid lines are the effective "
        "sample size and dashed lines the units still followed.",
    ]
    if capped_somewhere:
        found.append(
            "The percentage above each dotted line is the share of units whose cumulative "
            "probability the limit replaced."
        )
    if zeros:
        found.append(_zero_text(zeros, time_label, named=count > 1))
    uncapped = list(
        dict.fromkeys(
            kind for item in series for kind in item.node_kinds if kind not in _CAPPED_KINDS
        )
    )
    if uncapped and all(item.unit_weights for item in series):
        words = [_UNCAPPED_WORDS.get(kind, kind) for kind in uncapped]
        joined = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]
        point = time_label.lower()
        found.append(
            f"No limit line is drawn where {joined} acts at that {point} or an earlier one: "
            "the weight carries the policy's factor, which the limit does not cap."
        )
    if not all(item.unit_weights for item in series):
        found.append("The data's weights multiply the limit, so no limit line is drawn.")
    found.append("Descriptive: the weights are read from the fit, with no interval.")
    _core.add_notes(bottom, found)
    _end_labels(bottom, float(times[-1]), ends)
    return fig


def _chosen_regimens(regimens: Sequence[str], held: Sequence[str]) -> list[str]:
    """The regimens ``regimens=`` names, checked against the result's own labels."""
    from ..exceptions import DataError

    if isinstance(regimens, str):
        raise DataError(
            f"regimens= takes a list of regimen labels; got the string {regimens!r}. "
            f"Pass [{regimens!r}] to draw that one regimen"
        )
    names = [str(name) for name in regimens]
    if not names:
        raise DataError(f"regimens= names no regimen; choose up to {_core.MAX_SERIES} of {held}")
    if len(set(names)) != len(names):
        raise DataError(f"regimens= repeats a label: {names}")
    unknown = [name for name in names if name not in held]
    if unknown:
        raise DataError(
            f"regimens= names {unknown}, which this result does not hold; it holds {held}"
        )
    if len(names) > _core.MAX_SERIES:
        raise DataError(
            f"regimens= names {len(names)} regimens and one panel draws at most "
            f"{_core.MAX_SERIES}; choose fewer of {held}"
        )
    return names


@dataclass(frozen=True)
class _Zeros:
    """The units still followed with a weight of zero at one time point of one regimen."""

    name: str
    time: int
    count: int
    followed: int
    #: Whether a modified treatment policy acts at this or an earlier time point.
    shifted: bool
    unit_weights: bool


def _zero_text(zeros: Sequence[_Zeros], time_label: str, *, named: bool) -> str:
    """The note that counts the followed units whose weight is zero, which a log axis drops.

    A follower's weight is zero only where a modified treatment policy at this or an
    earlier time point gives its treatment a ratio of zero, or where its data weight is
    zero (``Plan.support`` in ``cleverly.longitudinal.regimen`` keeps no other follower whose
    factor is zero). The note names that cause only for a time point where such a policy
    acts, and names none elsewhere.
    """
    point = time_label.lower()

    def where(item: _Zeros) -> str:
        return (f"{item.name}, " if named else "") + f"{point} {item.time}"

    text = (
        "Units still followed with a weight of zero, which a log axis cannot show: "
        + "; ".join(f"{where(item)}: {item.count} of {item.followed}" for item in zeros)
        + "."
    )
    shifted = [item for item in zeros if item.shifted]
    if shifted:
        at = "" if len(shifted) == len(zeros) else f" ({', '.join(map(where, shifted))})"
        text += (
            f" A modified treatment policy acts at this or an earlier {point}{at}; it gives a "
            "unit a weight of zero when the shifted treatment cannot produce the one the unit "
            "received"
            + (
                ""
                if all(item.unit_weights for item in shifted)
                else ", and so does a data weight of zero"
            )
            + "."
        )
    return (
        text + " The boxes and crosses describe the other weights; the lower panel counts these "
        "units among those still followed."
    )


# ---------------------------------------------------------------------------- V11


def _balance_object(source: Any) -> Any:
    from ..sensitivity.balance import CovariateBalance, covariate_balance

    return source if isinstance(source, CovariateBalance) else covariate_balance(source)


def _shown_covariates(summary: Any, max_covariates: int) -> tuple[list[str], int]:
    """The covariates to draw, in drawing order from the top, and how many are left out.

    Every covariate whose difference after weighting is above the threshold in some panel
    is kept, and so is the largest after weighting; the rest of the room goes to the
    covariates with the largest difference before or after weighting. Drawn rows are sorted
    by their largest difference before weighting.
    """
    covariates = list(dict.fromkeys(row.covariate for row in summary.rows))

    def largest(name: str, attribute: str) -> float:
        values = [
            abs(getattr(row, attribute))
            for row in summary.rows
            if row.covariate == name and np.isfinite(getattr(row, attribute))
        ]
        return max(values, default=0.0)

    before = {name: largest(name, "smd_unweighted") for name in covariates}
    after = {name: largest(name, "smd_weighted") for name in covariates}
    kept = [name for name in covariates if after[name] > summary.threshold]
    if covariates:
        kept.append(max(covariates, key=lambda name: after[name]))
    ranked = sorted(covariates, key=lambda name: -max(before[name], after[name]))
    for name in ranked:
        if len(set(kept)) >= max_covariates:
            break
        kept.append(name)
    chosen = set(kept)
    order = sorted(
        (name for name in covariates if name in chosen),
        key=lambda name: (-before[name], covariates.index(name)),
    )
    return order, len(covariates) - len(order)


@_core.labelled
def balance(
    source: Any,
    *,
    max_covariates: int = 30,
    treatment_label: str | None = None,
    labels: dict[str, str] | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot each covariate's absolute standardized difference before and after weighting.

    Use it to check the treatment model on the covariates you know. Each row is a
    covariate: the hollow marker is the absolute difference in its mean between the two
    groups before weighting, in standard deviations, and the filled marker is the same
    difference after weighting each unit with the fit's own weights. Weighting has worked
    on a covariate when the filled marker sits left of the dashed line, at 0.1, the usual
    threshold for a small difference. Covariates are sorted by their difference before
    weighting.

    The weights and the standard deviation follow the estimand, as
    :func:`~cleverly.sensitivity.covariate_balance` describes: for the ATE, each unit
    weighted by the inverse of the probability of its own arm, and for the ATT or ATC, the
    group the effect is among left as it is. With more than two arms and the ATE, each arm
    is compared with all the others, one panel per arm. A categorical covariate has a row
    per level. The figure is descriptive: it carries no interval or test.

    Parameters
    ----------
    source : CovariateBalance or TMLEResult
        The balance, or a fitted point-treatment result to compute it from with the default
        estimand.
    max_covariates : int, default=30
        The most covariates to draw. Every covariate above the threshold after weighting is
        drawn, then those with the largest difference before or after weighting.
    treatment_label : str or None, default=None
        Words for the treatment in the subtitle and the panel titles, such as ``"Offer"``
        (read ``"Offer = 1 vs Offer = 0"``); its column name when omitted.
    labels : dict or None, default=None
        Words for each covariate, keyed by its name.
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

    Returns
    -------
    Figure
        One panel per contrast, up to three across.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism, or is not a point-treatment fit.
    DataError
        When the treatment is continuous.
    ValueError
        When ``max_covariates`` is less than one.

    See Also
    --------
    cleverly.sensitivity.covariate_balance : The same differences as numbers.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes give the standardized
    difference and the weights used, and name the covariates not drawn.

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
    from .._summary import vocabulary
    from .._summary._balance import descriptive_text, rest_text, source_text, weights_text
    from .._summary._render import number_text

    if isinstance(max_covariates, bool) or not isinstance(max_covariates, int):
        raise ValueError(f"max_covariates must be an integer of at least 1; got {max_covariates!r}")
    if max_covariates < 1:
        raise ValueError(f"max_covariates must be at least 1; got {max_covariates!r}")
    summary = _balance_object(source)
    contrasts = summary.contrasts

    def said(contrast: str) -> str:
        # Each arm reads "<treatment column> = <arm>"; treatment_label= names the column.
        if not treatment_label:
            return contrast
        return contrast.replace(f"{summary.treatment} = ", f"{treatment_label} = ")

    shown, hidden = _shown_covariates(summary, max_covariates)
    columns = min(len(contrasts), 3)
    panel_height = 0.27 * len(shown) + 0.45
    longest = max(len((labels or {}).get(name, name)) for name in shown)
    fig, head, axes = strip_figure(
        len(contrasts),
        columns,
        panel=(3.6, panel_height),
        left=longest * _core.TICK_SIZE * 0.55 / 72.0,
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
            color=_core.INK,
        )
        axis.set_ylim(-0.7, len(shown) + 0.75)
        if axis is not axes[0]:
            axis.tick_params(labelleft=False)
        axis.grid(False)
        axis.grid(True, axis="x", color=_core.GRID, linewidth=0.8)
        axis.set_xlabel("Absolute standardized mean difference")
        if len(contrasts) > 1:
            axis.set_title(
                said(contrast), loc="left", fontsize=_core.SUBTITLE_SIZE, color=_core.INK
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
            gid=_core.GID_SERIES_LABEL,
        )
    axes[0].set_ylabel("")
    name = treatment_label or summary.treatment
    finite = [row for row in summary.rows if np.isfinite(row.smd_weighted)]
    worst = max(finite, key=lambda row: abs(row.smd_weighted)) if finite else None
    pieces = [
        (f"Treatment {name}" if len(contrasts) > 1 else said(contrasts[0]))
        + ", weights for "
        # the printed summary keeps the alias; the figure carries the words alone
        + vocabulary.BALANCE_ESTIMAND_WORDS[summary.estimand].rsplit(" (", 1)[0]
    ]
    if worst is not None:
        pieces.append(
            f"largest after weighting {(labels or {}).get(worst.covariate, worst.covariate)} "
            f"{number_text(abs(worst.smd_weighted), 3)}"
        )
    _core.header(
        head,
        title or "Covariate balance before and after weighting",
        "; ".join(pieces),
        key_sources=axes[:1],
    )
    found = [descriptive_text(summary), f"{weights_text(summary)} {source_text(summary)}"]
    rest = rest_text(summary, "panel")
    if rest is not None:
        found.append(rest)
    if hidden:
        found.append(
            f"{hidden} covariates are not drawn. Every covariate above the threshold after "
            "weighting is drawn, then those with the largest difference before or after."
        )
    undefined = sorted(
        {row.covariate for row in summary.rows if not np.isfinite(row.smd_unweighted)}
    )
    if undefined:
        found.append(f"No difference is drawn for {', '.join(undefined)}, which does not vary.")
    _core.add_notes(fig, found)
    fit_height(fig, axes, columns, panel_height)
    return fig


__all__: Sequence[str] = ["balance", "weights"]
