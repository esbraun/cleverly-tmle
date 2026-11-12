"""V12: one estimate across a declared sensitivity parameter.

Three analyses share this family, because each asks whether the conclusion depends on a
number the analyst chose: the probability limits that cap extreme weights
(:class:`~cleverly.assessment.TruncationCurve`), the strength of unmeasured confounding in the
marginal sensitivity model (:class:`~cleverly.sensitivity.MarginalSensitivityBounds`), and the
departure from outcomes missing at random (:class:`~cleverly.sensitivity.MissingnessTilt`).

The parameter is a finite grid, so every value is drawn at its grid point and nothing is drawn
between them except a thin guide line through the points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _column(frame: Any, name: str) -> list[Any]:
    """One column of a pandas, polars or pyarrow frame, as a Python list."""
    values = frame[name]
    for method in ("to_list", "tolist"):
        reader = getattr(values, method, None)
        if reader is not None:
            return list(reader())
    return list(values)


def _columns(frame: Any) -> tuple[str, ...]:
    names = getattr(frame, "columns", None)
    if names is None:
        names = frame.column_names
    return tuple(str(name) for name in names)


def _choose(estimands: Sequence[str], estimand: str | None, what: str) -> str:
    """The estimand to draw: the caller's, or the first one the object reports."""
    available = list(dict.fromkeys(estimands))
    if not available:
        raise ValueError(f"this {what} holds no rows to plot")
    if estimand is None:
        return available[0]
    if estimand not in available:
        raise ValueError(f"estimand {estimand!r} is not in this {what}; it holds {available}")
    return estimand


def _others(estimands: Sequence[str], chosen: str) -> str | None:
    rest = [name for name in dict.fromkeys(estimands) if name != chosen]
    if not rest:
        return None
    return f"This figure shows {chosen}; pass estimand= to draw {', '.join(rest)}."


def _with_units(label: str, units: str | None) -> str:
    """``label`` with the outcome's units appended in parentheses, when given."""
    return f"{label} ({units})" if units else label


def _warnings(obj: Any) -> list[str]:
    """The warnings the object's printed summary carries, so the figure says the same."""
    summary = getattr(obj, "summary", None)
    if summary is None:
        return []
    return [note.text for note in summary().notes if note.kind == "warning"]


def _points(
    ax: Axes,
    x: Sequence[float],
    psi: Sequence[float],
    lower: Sequence[float | None],
    upper: Sequence[float | None],
    level: float,
) -> None:
    """Points joined by a thin guide line, with error bars where an interval exists."""
    color = _core.PALETTE[0]
    ax.plot(x, psi, color=color, linewidth=1.0, alpha=0.6, gid="cleverly-guide", zorder=2)
    label = _core.interval_label(level)
    for xi, yi, lo, hi in zip(x, psi, lower, upper, strict=True):
        has = lo is not None and hi is not None and _core.finite(lo) and _core.finite(hi)
        if has and lo is not None and hi is not None:
            ax.vlines(xi, lo, hi, color=color, linewidth=2.0, label=label, gid="cleverly-interval")
            label = "_nolegend_"
        ax.plot(
            [xi],
            [yi],
            marker="o",
            markersize=6.5,
            color=color,
            markerfacecolor=color if has else _core.FACE,
            markeredgewidth=1.5,
            linestyle="none",
            gid="cleverly-point",
            zorder=3,
        )


def _keep_inside(ax: Axes, value: float, *, log: bool) -> None:
    """Widen the y limits so ``value`` sits clear of both axes edges, not on a spine."""
    low, high = ax.get_ylim()
    if log:
        lo, hi, v = math.log(low), math.log(high), math.log(value)
    else:
        lo, hi, v = low, high, value
    margin = 0.05 * (hi - lo)
    lo, hi = min(lo, v - margin), max(hi, v + margin)
    ax.set_ylim(*((math.exp(lo), math.exp(hi)) if log else (lo, hi)))


def _null_axis(ax: Axes, null: float | None, ratio: bool) -> None:
    if ratio:
        _core.log_axis(ax, "y")
    if null is not None:
        _core.reference_line(ax, null, label="No effect")


# ----------------------------------------------------------------------- truncation


def truncation(
    curve: Any,
    *,
    estimand: str | None = None,
    outcome_label: str | None = None,
    units: str | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the estimate against the probability limits that cap extreme weights.

    Parameters
    ----------
    curve : TruncationCurve
        From ``result.diagnostics.truncation_curve()``.
    estimand : str or None, default=None
        Which estimate to draw; the first the curve holds when omitted.
    outcome_label : str or None, default=None
        The outcome's quantity in your words, such as ``"order share"``; the y axis then
        reads "Average effect on order share". The estimand's plain name when omitted.
    units : str or None, default=None
        Units of the outcome, added to the y axis in parentheses, such as ``"points"``.
    title : str or None, default=None
        The title.

    Returns
    -------
    Figure
        Two panels with a shared x axis: the estimate, and the share of units capped.

    Raises
    ------
    CapabilityError
        For a longitudinal curve, whose limits are cumulative and are not drawn by this
        plot.
    ValueError
        For a curve that does not record its scale and interval level, or for an unknown
        estimand.
    """
    frame = curve.to_frame()
    columns = _columns(frame)
    if "bound" not in columns or "truncated_fraction" not in columns:
        from .._refusal import RefusalReason
        from ..exceptions import CapabilityError

        raise CapabilityError(
            "this truncation curve is from a longitudinal fit, whose limits bound a "
            "cumulative probability; this plot draws only a point-treatment curve",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="curve.to_frame() holds the longitudinal curve's rows",
            roadmap="X92",
        )
    missing = [name for name in ("scale", "level") if name not in columns]
    if missing:
        raise ValueError(
            f"this truncation curve does not record its {' and '.join(missing)}, which the "
            "plot needs to draw the right null line and label the intervals; recompute it "
            "with result.diagnostics.truncation_curve()"
        )
    estimands = [str(value) for value in _column(frame, "estimand")]
    chosen = _choose(estimands, estimand, "truncation curve")
    rows = [index for index, name in enumerate(estimands) if name == chosen]
    # The guide line joins points in row order, so rows go in bound order; every array
    # below is picked through ``rows``, which keeps each estimate with its interval.
    bounds = _column(frame, "bound")
    rows.sort(key=lambda index: float(bounds[index]))

    def pick(name: str) -> list[Any]:
        if name not in columns:
            return [None] * len(rows)
        values = _column(frame, name)
        return [values[index] for index in rows]

    scale = str(pick("scale")[0])
    level = float(pick("level")[0])
    bound = [float(value) for value in pick("bound")]
    upper_bound = [float(value) for value in pick("upper_bound")]
    psi = [float(value) for value in pick("psi")]
    has_ci = "ci_lower" in columns
    lower = pick("ci_lower")
    upper = pick("ci_upper")
    capped = [float(value) for value in pick("truncated_fraction")]
    fitted = [bool(value) for value in pick("is_fitted_bound")]
    fitted_bound = float(pick("fitted_lower_bound")[0])
    fitted_upper = float(pick("fitted_upper_bound")[0])

    fig, (top, bottom) = _core.new_figure(
        2, figsize=(7.0, 5.4), height_ratios=(3.0, 1.3), sharex=True
    )
    _points(top, bound, psi, lower, upper, level)
    for axis in (top, bottom):
        _core.log_axis(axis, "x")
        _core.reference_line(
            axis,
            fitted_bound,
            orientation="vertical",
            label="Limit the fit used" if axis is top else None,
            kind="marker",
        )
    _null_axis(top, _core.null_for(scale), scale == "ratio")
    top.set_ylabel(_with_units(_core.estimand_axis_label(chosen, outcome_label), units))
    bottom.plot(
        bound,
        [100.0 * value for value in capped],
        marker="o",
        markersize=5,
        color=_core.PALETTE[0],
        linewidth=1.2,
        gid="cleverly-capped",
    )
    bottom.set_ylabel("Units capped (%)")
    bottom.set_ylim(bottom=0.0)
    symmetric = all(abs(hi - (1.0 - lo)) < 1e-12 for lo, hi in zip(bound, upper_bound, strict=True))
    one_sided = all(hi == 1.0 for hi in upper_bound)
    decimals = [fitted_bound / 10.0]
    low_text, high_text = _core.format_values([fitted_bound, fitted_upper], decimals * 2)
    if symmetric:
        bottom.set_xlabel(
            "Lower limit; each arm's probability capped to [limit, 1 - limit] (log scale)"
        )
        used = f"the fit used [{low_text}, {high_text}]"
    elif one_sided:
        bottom.set_xlabel("Lower limit on the probability in the weight (log scale)")
        used = f"the fit used {low_text}"
    else:
        bottom.set_xlabel("Lower limit of each declared pair of limits (log scale)")
        used = f"the fit used [{low_text}, {high_text}]"
    _core.header(
        top,
        title or "Estimate across probability limits",
        f"{_core.estimand_words(chosen)} re-estimated at each limit; {used}",
    )
    found = []
    supports = pick("support")
    reasons = pick("support_reason")
    experimental = [
        str(reason)
        for label, reason in zip(supports, reasons, strict=True)
        if label == "experimental"
    ]
    if experimental:
        from .._summary._estimates import plain_support_reason

        found.append(f"{chosen}: experimental. {plain_support_reason(experimental[0])}")
    if not has_ci:
        found.append(
            "No interval is drawn: the fit's inference status supplies none, so the curve "
            "shows estimates only."
        )
    if not any(fitted):
        found.append("The limit the fit used is not on this grid.")
    _core.add_notes(bottom, [*found, *_warnings(curve), _others(estimands, chosen) or ""])
    return fig


# ----------------------------------------------------------------------- marginal bounds


def marginal(
    bounds: Any,
    *,
    outcome_label: str | None = None,
    units: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the marginal-sensitivity-model range of the estimate against gamma.

    Parameters
    ----------
    bounds : MarginalSensitivityBounds
        From ``result.sensitivity.marginal_bounds()``.
    outcome_label : str or None, default=None
        The outcome's quantity in your words, such as ``"order share"``, named with the
        estimand on the y axis.
    units : str or None, default=None
        Units of the outcome, added to the y axis in parentheses, such as ``"points"``.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    Figure
        One panel: at each gamma, the range of the estimate as a thick bar and its interval as
        thin whiskers.
    """
    fig, ax = _core.figure_for(ax)
    color = _core.PALETTE[0]
    level = float(bounds.level)
    range_label = "Range allowed by the model"
    limit_label = f"{_core.level_text(level)} interval for the range"
    for row in bounds.rows:
        ax.vlines(
            row.gamma,
            row.lower_limit,
            row.upper_limit,
            color=color,
            linewidth=1.2,
            label=limit_label,
            gid="cleverly-interval",
        )
        ax.vlines(
            row.gamma,
            row.lower,
            row.upper,
            color=color,
            linewidth=7.0,
            alpha=0.55,
            label=range_label,
            gid="cleverly-range",
        )
        if math.isclose(row.lower, row.upper, rel_tol=1e-9, abs_tol=1e-12):
            ax.plot(
                [row.gamma], [row.lower], marker="o", color=color, markersize=6, linestyle="none"
            )
        range_label = limit_label = "_nolegend_"
    ratio = bounds.scale == "log" or (bounds.null is not None and float(bounds.null) == 1.0)
    _null_axis(ax, bounds.null, ratio)
    if bounds.null is not None:
        _keep_inside(ax, float(bounds.null), log=ratio)
    words = _core.estimand_words(str(bounds.estimand))
    if words != str(bounds.estimand):
        words = "the " + words[:1].lower() + words[1:]
    subtitle = _core.estimand_words(str(bounds.estimand))
    if bounds.null is not None:
        subtitle += ": " + _breakdowns(bounds, level)
    ax.set_xlabel("gamma, the largest factor by which hidden confounding may change treatment odds")
    if outcome_label:
        named = _core.estimand_axis_label(str(bounds.estimand), outcome_label)
        ax.set_ylabel(_with_units(f"Range of the {_core.mid_phrase(named)}", units))
    else:
        ax.set_ylabel(_with_units(f"Range of {words}", units))
    _core.header(ax, title or "Bounds under hidden confounding", subtitle)
    _core.add_notes(ax, _warnings(bounds))
    return fig


def _breakdowns(bounds: Any, level: float) -> str:
    """Where the range, and where its interval, first include no effect on the grid.

    The two differ: the interval is wider than the range, so it can include no effect at a
    smaller gamma, or on a grid where the range never does. Both are named, as the printed
    summary's breakdown table names both.
    """
    last = max(row.gamma for row in bounds.rows)
    interval = f"its {_core.level_text(level)} interval"
    range_at, limit_at = bounds.breakdown, bounds.limit_breakdown
    if range_at is None and limit_at is None:
        return f"range and {interval} clear of no effect up to gamma = {last:g}"
    if range_at is None:
        return (
            f"range clear of no effect up to gamma = {last:g}; {interval} reaches it at "
            f"gamma = {limit_at:g}"
        )
    if limit_at is None:  # pragma: no cover - the interval holds the range
        return f"range reaches no effect at gamma = {range_at:g}"
    return f"range reaches no effect at gamma = {range_at:g}, {interval} at gamma = {limit_at:g}"


# ----------------------------------------------------------------------- missingness tilt


def missingness(
    tilt: Any,
    *,
    estimand: str | None = None,
    outcome_label: str | None = None,
    units: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the estimate against the departure from outcomes missing at random.

    Parameters
    ----------
    tilt : MissingnessTilt
        From ``result.sensitivity.missingness()``.
    estimand : str or None, default=None
        Which estimate to draw; the first the tilt holds when omitted.
    outcome_label : str or None, default=None
        The outcome's quantity in your words, such as ``"order share"``, named with the
        estimand on the y axis.
    units : str or None, default=None
        Units of the outcome, added to the y axis in parentheses, such as ``"points"``.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    Figure
        One panel: the estimate at each tilt, with its interval where the tilt reports one.
    """
    from .._inference_status import status_record, supplies_inference
    from .._summary import vocabulary

    columns = tilt.columns
    estimands = [str(value) for value in tilt.column("estimand")]
    chosen = _choose(estimands, estimand, "missingness tilt")
    rows = [index for index, name in enumerate(estimands) if name == chosen]
    # A gamma grid may arrive in any order and the guide line joins points in row order,
    # so rows go in gamma order; every array below is picked through ``rows``, which keeps
    # each estimate with its interval.
    gammas = tilt.column("gamma")
    rows.sort(key=lambda index: float(gammas[index]))

    def pick(name: str) -> list[Any]:
        if name not in columns:
            return [None] * len(rows)
        values = tilt.column(name)
        return [values[index] for index in rows]

    gamma = [float(value) for value in pick("gamma")]
    psi = [float(value) for value in pick("psi")]
    lower = [_or_none(value) for value in pick("ci_lower")]
    upper = [_or_none(value) for value in pick("ci_upper")]
    scale = dict(tilt.scales).get(chosen)
    if scale is None:
        scale = "ratio" if chosen in tilt.ratios else "level"
    status = dict(tilt.statuses).get(chosen, "influence_curve")
    if not supplies_inference(status):  # honesty rule 1: no interval the status withholds
        lower = [None] * len(lower)
        upper = [None] * len(upper)
    fig, axis = _core.figure_for(ax)
    level = float(tilt.level)
    _points(axis, gamma, psi, lower, upper, level)
    _null_axis(axis, _core.null_for(scale), scale == "ratio")
    _core.reference_line(
        axis,
        0.0,
        orientation="vertical",
        label="Missing at random (the main analysis)",
        kind="marker",
    )
    direction = (
        f"above 0, missing {tilt.outcome} values higher"
        if tilt.arm_gamma is None
        else "each arm's tilt is gamma times its factor"
    )
    axis.set_xlabel(f"gamma, log-odds tilt of missing outcomes ({direction})")
    axis.set_ylabel(_with_units(_core.estimand_axis_label(chosen, outcome_label), units))
    _core.header(
        axis,
        title or "Estimate across missingness assumptions",
        f"{_core.estimand_words(chosen)}; {tilt.n_missing} of {tilt.n} outcomes missing",
    )
    found = []
    if not supplies_inference(status):
        found.append(f"{chosen}: no interval is drawn. {status_record(status).plain_reason}")
    else:
        at_mar = [lo for g, lo in zip(gamma, lower, strict=True) if g == 0.0]
        off_mar = [lo for g, lo in zip(gamma, lower, strict=True) if g != 0.0]
        if any(lo is None for lo in off_mar) and at_mar and at_mar[0] is not None:
            found.append(
                f"Hollow points have no interval: off gamma = 0 the fit reports only a "
                f"plug-in spread for {chosen}, which is not a confidence interval."
            )
    if tilt.arm_gamma is not None:
        found.append(vocabulary.NOTE_TEXTS["tilt_direction_arms"])
    _core.add_notes(axis, [*found, *_warnings(tilt), _others(estimands, chosen) or ""])
    return fig


def _or_none(value: Any) -> float | None:
    return float(value) if _core.finite(value) else None


# ----------------------------------------------------------------------- dispatch


@_core.labelled
def sensitivity(obj: Any, **kwargs: Any) -> Figure:
    """Plot one estimate across a declared sensitivity parameter.

    Use it to see whether the conclusion changes as a number the analyst chose changes: the
    probability limits that cap extreme weights, the strength of hidden confounding gamma in the
    marginal sensitivity model, or the departure gamma from outcomes missing at random. Each
    value is drawn at its grid point with its interval where the analysis reports one; a
    grey dashed line marks no effect, and the fitted or main-analysis value is marked.

    Parameters
    ----------
    obj : TruncationCurve, MarginalSensitivityBounds or MissingnessTilt
        The analysis to draw.
    **kwargs
        ``estimand=`` (truncation and missingness), ``outcome_label=``, ``units=``, ``ax=``
        (bounds and missingness), and the formatting keywords ``title=``, ``subtitle=``,
        ``xlabel=``, ``ylabel=``, ``legend=`` and ``figsize=``, as
        :func:`~cleverly.plotting.forest` describes them.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    TypeError
        When ``obj`` is not one of the three analyses.

    See Also
    --------
    cleverly.plotting.forest : The estimates of a fit with their intervals.
    cleverly.assessment.TruncationCurve.plot : This plot for a truncation curve.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes name the estimands not drawn,
    say when no validation study licenses an interval (read it as descriptive), and give the
    sign convention of the sensitivity parameter.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import sensitivity
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
    >>> fig = sensitivity(result.diagnostics.truncation_curve())
    >>> fig.axes[1].get_ylabel()
    'Units capped (%)'
    >>> plt.close(fig)
    """
    from ..assessment import TruncationCurve
    from ..sensitivity._objects import MissingnessTilt
    from ..sensitivity.marginal import MarginalSensitivityBounds

    if isinstance(obj, TruncationCurve):
        return truncation(obj, **kwargs)
    if isinstance(obj, MarginalSensitivityBounds):
        return marginal(obj, **kwargs)
    if isinstance(obj, MissingnessTilt):
        return missingness(obj, **kwargs)
    raise TypeError(
        "sensitivity() takes a TruncationCurve, MarginalSensitivityBounds or MissingnessTilt; "
        f"got {type(obj).__name__}"
    )


__all__ = ["marginal", "missingness", "sensitivity", "truncation"]
