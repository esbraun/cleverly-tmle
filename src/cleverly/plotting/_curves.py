"""V2 to V6: result curves.

One function, :func:`curve`, dispatches on the object it is given:

* a :class:`~cleverly.longitudinal.estimator.LongitudinalResult` gives the survival, risk or
  mean-outcome curve over time (V2) or the restricted mean up to each horizon (V3);
* a point-treatment fit of per-level counterfactual means gives the means against the level,
  with the working model beside them when a marginal structural model fit is passed (V4);
* a point-treatment fit of a family of policies gives the effect against the policy
  parameter (V5);
* an :class:`~cleverly.effect_curves.EffectCurveResult` gives the conditional effect over one
  modifier (V6).

Every figure draws numbers the object already holds. A band is drawn only when the fit
computed one: as a ribbon over a step curve that the fit reports at every node, as bracket
marks at the grid points otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Marker and dash pattern of the first three series, so identity never rests on colour.
_MARKERS = ("o", "s", "^")
_DASHES: tuple[Any, ...] = ("-", (0, (5, 2)), (0, (1, 1.6)))
#: Inches between neighbouring series at a shared x, so their error bars do not overlap.
_DODGE = 0.07


# ----------------------------------------------------------------------- frame helpers


def _columns(frame: Any) -> tuple[str, ...]:
    names = getattr(frame, "columns", None)
    if names is None:
        names = frame.column_names
    return tuple(str(name) for name in names)


def _column(frame: Any, name: str) -> list[Any]:
    values = frame[name]
    for method in ("to_list", "tolist"):
        reader = getattr(values, method, None)
        if reader is not None:
            return list(reader())
    return list(values)


def _finite_pair(low: Any, high: Any) -> bool:
    return _core.finite(low) and _core.finite(high)


def _offset(ax: Axes, index: int, count: int) -> Any:
    """``ax.transData`` shifted sideways by the series' place, in inches."""
    from matplotlib.transforms import ScaledTranslation

    shift = (index - (count - 1) / 2.0) * _DODGE
    return ax.transData + ScaledTranslation(shift, 0.0, ax.figure.dpi_scale_trans)


def _pad(ax: Axes, values: Sequence[float]) -> None:
    """Set the y limits to the drawn values plus a margin.

    Artists drawn through a shifted transform do not feed the autoscale, so the limits are
    set from the numbers drawn.
    """
    found = [float(v) for v in values if _core.finite(v)]
    if not found:
        return
    low, high = min(found), max(found)
    margin = 0.08 * ((high - low) or max(abs(high), 1.0) * 0.1)
    ax.set_ylim(low - margin, high + margin)


def _wrap_axis_label(text: str, width: int = 26) -> str:
    """Break a long axis label so it fits the height of its panel."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text


def _spread(values: Sequence[float], gap: float) -> list[float]:
    """Move label heights apart until neighbours are at least ``gap`` apart."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous: float | None = None
    for i in order:
        value = values[i] if previous is None else max(values[i], previous + gap)
        placed[i] = value
        previous = value
    return placed


def _support_notes(estimates: Sequence[Any], what: str) -> list[str]:
    """One note per distinct reason the fit's estimates are labelled experimental (rule 3)."""
    from .._summary._estimates import plain_support_reason

    reasons: dict[str, None] = {}
    for estimate in estimates:
        if getattr(estimate.support, "label", "supported") == "experimental":
            reasons[plain_support_reason(estimate.support.reason)] = None
    return [f"The {what} is experimental. {reason}" for reason in reasons]


def _withheld_notes(estimates: Sequence[Any]) -> list[str]:
    """One note per distinct reason an estimate's interval is withheld (rule 1)."""
    from .._inference_status import status_record

    reasons: dict[str, None] = {}
    for estimate in estimates:
        if not estimate.supplies_inference:
            reasons[status_record(str(estimate.inference)).plain_reason] = None
    return [f"No interval is drawn where the fit withholds it. {reason}" for reason in reasons]


def _band_note(support: Any, what: str) -> list[str]:
    from .._summary._estimates import plain_support_reason

    if getattr(support, "label", "supported") != "experimental":
        return []
    reason = plain_support_reason(support.reason) if getattr(support, "reason", "") else ""
    return [f"The {what} is experimental. {reason}".strip()]


# ----------------------------------------------------------------------- dispatch


def curve(
    obj: Any,
    *,
    scale: str | None = None,
    measure: str = "curve",
    band: str | None = None,
    cause: str | None = None,
    regimens: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    msm: Any = None,
    means: Any = None,
    levels: Mapping[str, Any] | None = None,
    modifier: str | None = None,
    by: str | None = None,
    at: Mapping[str, Any] | None = None,
    alpha: float = 0.05,
    outcome_label: str | None = None,
    treatment_label: str | None = None,
    units: str | None = None,
    time_label: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a result as a curve: over time, over a treatment level, or over a modifier.

    The object decides the plot. A longitudinal fit with a report at each time point draws
    its survival, risk or mean-outcome curve for each regimen, the difference between
    regimens in a second panel, and the number of units still following each regimen under
    observation below. A point-treatment fit of a family of policies (a sequence of shifts,
    scale factors or incremental odds multipliers) draws the effect against the policy's
    parameter. A point-treatment fit of per-level means draws the means against the
    treatment level, and the working model of a marginal structural model fit beside them.
    A conditional effect curve draws the effect over one modifier.

    A band is drawn only when the fit computed one. A fit that reports every time point
    gets step ribbons, because the curve is constant between time points; a fit that
    reports only some horizons gets points and bracket marks at those horizons. A result
    that withholds inference draws points only and the note under the figure gives the
    reason; an experimental result or band says so there.

    Parameters
    ----------
    obj : LongitudinalResult, TMLEResult or EffectCurveResult
        What to draw.
    scale : {"risk", "survival", "mean"} or None, default=None
        Longitudinal fits: the view of the outcome. ``None`` is ``"risk"`` for a survival
        outcome and ``"mean"`` for a repeated one. The levels and differences are read from
        ``obj.curve(scale=...)``.
    measure : {"curve", "rmst", "rmtl"}, default="curve"
        Longitudinal fits: the curve over time, or the restricted mean survival time or time
        lost up to each horizon (points joined by a thin line, pointwise intervals only).
    band : {"multiplier", "bootstrap", "none"} or None, default=None
        Longitudinal fits: which simultaneous band to draw. ``None`` draws the multiplier
        band when the fit holds one, else the bootstrap band; the other is named in the note.
    cause : str or None, default=None
        Longitudinal fits with several causes: the cause to draw.
    regimens : sequence of str or None, default=None
        Longitudinal fits: the regimens to draw, at most three.
    labels : mapping of str to str or None, default=None
        Display names for regimens.
    msm : TMLEResult or None, default=None
        Per-level means: the point-treatment fit of a working model to draw beside them.
    means : TMLEResult or None, default=None
        Working model fit: the fit of per-level means to draw beside it.
    levels : mapping or None, default=None
        Per-level means: the treatment level of each regime name, when the regime names are
        not the levels themselves.
    modifier : str or None, default=None
        Conditional effect curve: the modifier on the x axis; the only continuous modifier
        the curve reads when omitted.
    by : str or None, default=None
        Conditional effect curve: a categorical modifier, drawn as one panel per level.
    at : mapping or None, default=None
        Conditional effect curve: the value at which each other modifier the curve reads is
        held.
    alpha : float, default=0.05
        Conditional effect curve: the pointwise interval holds at ``1 - alpha``.
    outcome_label : str or None, default=None
        Words for the outcome on the y axis.
    treatment_label : str or None, default=None
        Words for the treatment on the x axis.
    units : str or None, default=None
        Units of the outcome or of time, added to the axis label.
    time_label : str or None, default=None
        Words for the time axis of a longitudinal fit.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on. Only a one-panel plot can use it.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    TypeError
        When ``obj`` is not a result this function draws.
    ValueError
        When an argument does not match the object: too many regimens or contrasts for one
        axes, an unknown modifier, a held value that is missing, or a policy family with no
        single parameter.
    CapabilityError
        When the fit refuses the view (for example the survival view of a fit with several
        causes), with the fit's own reason.

    See Also
    --------
    cleverly.plotting.forest : The estimates of one or several fits as a forest plot.
    cleverly.longitudinal.estimator.LongitudinalResult.plot : This plot for a longitudinal fit.
    cleverly.effect_curves.EffectCurveResult.plot : This plot for a conditional effect curve.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import make_longitudinal_survival
    >>> from cleverly.longitudinal import LTMLE
    >>> from cleverly.plotting import curve
    >>> frame, _ = make_longitudinal_survival(n=300, seed=0)
    >>> result = LTMLE(
    ...     {"always": 1, "never": 0},
    ...     n_folds=1,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ... ).fit(
    ...     frame,
    ...     outcome=["Y1", "Y2"],
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> fig = curve(result, scale="survival")
    >>> fig.axes[0].get_title(loc="left")
    'Survival over time'
    >>> plt.close(fig)
    """
    from ..effect_curves import EffectCurveResult
    from ..longitudinal.estimator import LongitudinalResult

    if isinstance(obj, LongitudinalResult):
        if measure not in ("curve", "rmst", "rmtl"):
            raise ValueError(f"measure must be 'curve', 'rmst' or 'rmtl'; got {measure!r}")
        if measure == "curve":
            return _over_time(
                obj,
                scale=scale,
                band=band,
                cause=cause,
                regimens=regimens,
                labels=labels,
                outcome_label=outcome_label,
                units=units,
                time_label=time_label,
                title=title,
            )
        return _restricted(
            obj,
            measure=measure,
            cause=cause,
            regimens=regimens,
            labels=labels,
            units=units,
            time_label=time_label,
            title=title,
        )
    if isinstance(obj, EffectCurveResult):
        return _effect_curve(
            obj,
            modifier=modifier,
            by=by,
            at=at,
            alpha=alpha,
            outcome_label=outcome_label,
            treatment_label=treatment_label,
            units=units,
            title=title,
            ax=ax,
        )
    keys = getattr(obj, "parameter_keys", None)
    if isinstance(keys, Mapping) and hasattr(obj, "estimates"):
        axes = {str(key.axis) for key in keys.values()}
        if axes == {"msm"}:
            if means is None:
                raise ValueError(
                    "this is the fit of a working model; pass means= the fit of the "
                    "per-level counterfactual means to draw it against"
                )
            return _dose(
                means,
                obj,
                levels=levels,
                outcome_label=outcome_label,
                treatment_label=treatment_label,
                units=units,
                title=title,
                ax=ax,
            )
        if axes == {"regime"}:
            return _dose(
                obj,
                msm,
                levels=levels,
                outcome_label=outcome_label,
                treatment_label=treatment_label,
                units=units,
                title=title,
                ax=ax,
            )
        if axes <= {"policy", "ipsi"}:
            return _policies(
                obj,
                outcome_label=outcome_label,
                treatment_label=treatment_label,
                units=units,
                title=title,
                ax=ax,
            )
    raise TypeError(
        "curve() draws a longitudinal fit, a point-treatment fit of per-level means, of a "
        "working model or of a family of policies, or a conditional effect curve; "
        f"got {type(obj).__name__}"
    )


# ----------------------------------------------------------------------- V2: over time


@dataclass(frozen=True)
class _Point:
    parameter: str
    regimen: str
    x: float
    node: int
    psi: float
    lower: float | None
    upper: float | None
    level: bool
    estimate: Any


def _read_curve(result: Any, view: str, cause: str | None) -> tuple[list[_Point], str | None]:
    """The curve's rows from ``result.curve(scale=view)``, for one cause."""
    frame = result.curve(scale=view)
    names = _columns(frame)
    parameters = [str(value) for value in _column(frame, "parameter")]
    regimen = [str(value) for value in _column(frame, "regimen")]
    times = _column(frame, "time")
    psi = _column(frame, "psi")
    scales = [str(value) for value in _column(frame, "scale")]
    low = _column(frame, "ci_lower") if "ci_lower" in names else [None] * len(psi)
    high = _column(frame, "ci_upper") if "ci_upper" in names else [None] * len(psi)
    causes = [str(value) for value in _column(frame, "cause")] if "cause" in names else None
    chosen = cause
    if causes is not None:
        distinct = list(dict.fromkeys(causes))
        if chosen is None:
            if len(distinct) > 1:
                raise ValueError(
                    f"this fit has several causes {distinct}; pass cause= to draw one of them"
                )
            chosen = distinct[0]
        elif chosen not in distinct:
            raise ValueError(f"cause {chosen!r} is not in this fit; it reports {distinct}")
    index = result.parameter_index or {}
    rows = []
    for i, name in enumerate(parameters):
        if causes is not None and causes[i] != chosen:
            continue
        estimate = result.estimates[name]
        drawn = estimate.supplies_inference and _finite_pair(low[i], high[i])
        rows.append(
            _Point(
                name,
                regimen[i],
                float(times[i]),
                int(index[name][2]),
                float(psi[i]),
                float(low[i]) if drawn else None,
                float(high[i]) if drawn else None,
                scales[i] == "level",
                estimate,
            )
        )
    return rows, chosen


def _select(rows: list[_Point], regimens: Sequence[str] | None) -> tuple[list[str], list[str]]:
    """The level regimens to draw and the contrast labels that go with them."""
    found = list(dict.fromkeys(row.regimen for row in rows if row.level))
    chosen = found if regimens is None else list(regimens)
    unknown = [name for name in chosen if name not in found]
    if unknown:
        raise ValueError(f"unknown regimen {unknown}; this fit reports {found}")
    if len(chosen) > _core.MAX_SERIES:
        raise ValueError(
            f"this fit has {len(chosen)} regimens and one axes holds at most "
            f"{_core.MAX_SERIES} coloured series; pass regimens= to draw up to "
            f"{_core.MAX_SERIES} of them"
        )
    contrasts = list(dict.fromkeys(row.regimen for row in rows if not row.level))
    if regimens is not None:
        wanted = {f"{a} vs {b}" for a in chosen for b in chosen if a != b}
        contrasts = [name for name in contrasts if name in wanted]
    if len(contrasts) > _core.MAX_SERIES:
        raise ValueError(
            f"this fit has {len(contrasts)} contrasts and one axes holds at most "
            f"{_core.MAX_SERIES} series; pass regimens= to select fewer"
        )
    return chosen, contrasts


def _series(rows: list[_Point], regimen: str, *, level: bool) -> list[_Point]:
    return sorted((r for r in rows if r.regimen == regimen and r.level == level), key=lambda r: r.x)


def _band_choice(result: Any, band: str | None) -> tuple[Any, str | None, list[str]]:
    """The band to draw (rule 2), its kind word, and the notes about the other band."""
    if band not in (None, "multiplier", "bootstrap", "none"):
        raise ValueError(f"band must be 'multiplier', 'bootstrap' or 'none'; got {band!r}")
    multiplier = result.simultaneous
    bootstrap = result.bootstrap_band
    if band == "multiplier" and multiplier is None:
        raise ValueError("this fit holds no multiplier band")
    if band == "bootstrap" and bootstrap is None:
        raise ValueError("this fit holds no bootstrap band")
    if band == "none":
        return None, None, []
    if band == "bootstrap" or (band is None and multiplier is None):
        chosen, kind, other = bootstrap, "bootstrap sup-t", multiplier
        other_name, hint = "multiplier band", "multiplier"
    else:
        chosen, kind, other = multiplier, "multiplier", bootstrap
        other_name, hint = "bootstrap sup-t band", "bootstrap"
    if chosen is None:
        return None, None, []
    notes = []
    if other is not None:
        notes.append(f"The fit also holds a {other_name}; pass band={hint!r} to draw it.")
    return chosen, kind, notes


def _view_band(
    chosen: Any,
    row: _Point,
    psi_risk: Mapping[str, float],
    view: str,
) -> tuple[float, float] | None:
    """The band limits at one row, on the requested view.

    The fit holds its bands on the risk (or mean) scale. On the survival view the limits
    are the risk limits mirrored about the estimate: the distance from the risk estimate
    to each risk limit, swapped to the other side of the survival estimate the fit's own
    ``curve(scale="survival")`` reports. That is the map for a level and for a contrast
    alike, and no ``1 - x`` is applied here.
    """
    pair = chosen.bands.get(row.parameter)
    if pair is None:
        return None
    low, high = float(pair[0]), float(pair[1])
    if view != "survival":
        return low, high
    anchor = psi_risk[row.parameter]
    return row.psi - (high - anchor), row.psi + (anchor - low)


def _nodes_reported(result: Any, rows: list[_Point]) -> bool:
    """Whether the fit reports every node (rule 2's exception for step ribbons)."""
    reported = {row.node for row in rows if row.level}
    return reported == set(range(1, int(result.data.n_times) + 1))


def _counts(
    result: Any, regimens: Sequence[str], nodes: Mapping[int, float]
) -> dict[str, dict[int, int]]:
    """Units following each regimen and still observed, per regimen and node."""
    from ..validation.longitudinal import _longitudinal_stagewise

    table = _longitudinal_stagewise(result)
    found: dict[str, dict[int, int]] = {}
    for name in regimens:
        mine = [r for r in table.rows if r.regimen == name]
        per: dict[int, int] = {}
        for node in nodes:
            exact = [r for r in mine if r.time == node and r.horizon == node]
            same = [r for r in mine if r.time == node]
            pick = exact or sorted(same, key=lambda r: -(r.horizon or 0))
            if pick:
                per[node] = int(pick[0].n_followed)
        found[name] = per
    return found


def _draw_line(
    ax: Axes,
    points: list[_Point],
    *,
    color: str,
    index: int,
    count: int,
    step: bool,
    pointwise: bool,
    limits: Mapping[str, tuple[float, float]] | None,
    level: float,
    band_text: str | None,
    first: dict[str, bool],
) -> None:
    """One series: the curve, its pointwise interval and its band."""
    x = np.array([p.x for p in points])
    y = np.array([p.psi for p in points])
    style = _DASHES[index % len(_DASHES)]
    has_interval = pointwise and all(p.lower is not None for p in points)
    low = np.array([p.lower if p.lower is not None else np.nan for p in points])
    high = np.array([p.upper if p.upper is not None else np.nan for p in points])
    box = None
    if limits is not None and all(p.parameter in limits for p in points):
        box = np.array([limits[p.parameter] for p in points])

    def tag(key: str, text: str) -> str:
        if first[key]:
            first[key] = False
            return text
        return "_nolegend_"

    if step:
        if box is not None:
            ax.fill_between(
                x,
                box[:, 0],
                box[:, 1],
                step="post",
                color=color,
                alpha=_core.BAND_ALPHA,
                linewidth=0,
                label=tag("band", band_text or ""),
                gid="cleverly-band",
            )
        if has_interval:
            ax.fill_between(
                x,
                low,
                high,
                step="post",
                color=color,
                alpha=_core.INTERVAL_ALPHA,
                linewidth=0,
                label=tag("pointwise", _core.pointwise_label(level)),
                gid="cleverly-ribbon",
            )
        ax.step(
            x, y, where="post", color=color, linewidth=2.0, linestyle=style, gid="cleverly-curve"
        )
    else:
        shift = _offset(ax, index, count)
        if box is not None:
            ax.vlines(
                x,
                box[:, 0],
                box[:, 1],
                color=color,
                linewidth=1.0,
                alpha=0.55,
                transform=shift,
                label=tag("band", band_text or ""),
                gid="cleverly-band",
            )
            for edge in (0, 1):
                ax.plot(
                    x,
                    box[:, edge],
                    marker="_",
                    markersize=9,
                    color=color,
                    alpha=0.55,
                    linestyle="none",
                    transform=shift,
                    gid="cleverly-band-cap",
                )
        if has_interval:
            ax.vlines(
                x,
                low,
                high,
                color=color,
                linewidth=2.2,
                transform=shift,
                label=tag("pointwise", _core.pointwise_label(level)),
                gid="cleverly-interval",
            )
    ax.plot(
        x,
        y,
        marker=_MARKERS[index % len(_MARKERS)],
        markersize=5.5 if step else 7,
        color=color,
        markerfacecolor=color,
        linestyle="none",
        transform=ax.transData if step else _offset(ax, index, count),
        gid="cleverly-point",
        zorder=3,
    )


def _extent(
    rows: list[_Point],
    names: Sequence[str],
    limits: Mapping[str, tuple[float, float]] | None,
    *,
    level_rows: bool,
) -> list[float]:
    """Every y value the series ``names`` draw: estimates, intervals and band limits."""
    found: list[float] = []
    for row in rows:
        if row.regimen not in names or row.level != level_rows:
            continue
        found.append(row.psi)
        for value in (row.lower, row.upper):
            if value is not None:
                found.append(value)
        if limits is not None and row.parameter in limits:
            found.extend(limits[row.parameter])
    return found


def _label_ends(ax: Axes, ends: list[tuple[float, float, str, str]]) -> None:
    """Direct labels at the right end of each series, moved apart where they would touch."""
    low, high = ax.get_ylim()
    heights = _spread([y for _, y, _, _ in ends], 0.075 * (high - low))
    for (x, _, text, color), y in zip(ends, heights, strict=True):
        _core.end_label(ax, x, y, text, color)


def _over_time(
    result: Any,
    *,
    scale: str | None,
    band: str | None,
    cause: str | None,
    regimens: Sequence[str] | None,
    labels: Mapping[str, str] | None,
    outcome_label: str | None,
    units: str | None,
    time_label: str | None,
    title: str | None,
) -> Figure:
    view = scale or ("mean" if result.data.is_repeated else "risk")
    rows, chosen_cause = _read_curve(result, view, cause)
    chosen, contrasts = _select(rows, regimens)
    name_of = (lambda s: str(labels.get(s, s))) if labels else str
    step = _nodes_reported(result, rows)
    level = 1.0 - float(rows[0].estimate.alpha)
    band_obj, band_kind, band_notes = _band_choice(result, band)
    psi_risk: dict[str, float] = {}
    if band_obj is not None or result.isotonic_moves:
        base = rows if view != "survival" else _read_curve(result, "risk", chosen_cause)[0]
        psi_risk = {r.parameter: r.psi for r in base}
    in_play = [
        r
        for r in rows
        if (r.level and r.regimen in chosen) or (not r.level and r.regimen in contrasts)
    ]
    limits: dict[str, tuple[float, float]] | None = None
    if band_obj is not None:
        found = {}
        for row in in_play:
            pair = _view_band(band_obj, row, psi_risk, view)
            if pair is not None:
                found[row.parameter] = pair
        limits = found or None
    band_text = (
        None if band_obj is None else _core.band_label(1.0 - float(band_obj.alpha), band_kind or "")
    )

    tall = [3.4] + ([1.7] if contrasts else []) + [0.45 + 0.3 * len(chosen)]
    fig, panels = _core.new_figure(
        len(tall), figsize=(7.2, 1.2 + sum(tall)), height_ratios=tall, sharex=True
    )
    top = panels[0]
    count_ax = panels[-1]
    count_ax.grid(False)
    first = {"band": True, "pointwise": True}
    ends: list[tuple[float, float, str, str]] = []
    for i, regimen in enumerate(chosen):
        points = _series(rows, regimen, level=True)
        _draw_line(
            top,
            points,
            color=_core.PALETTE[i],
            index=i,
            count=len(chosen),
            step=step,
            pointwise=True,
            limits=limits,
            level=level,
            band_text=band_text,
            first=first,
        )
        ends.append((points[-1].x, points[-1].psi, name_of(regimen), _core.PALETTE[i]))
    _mark_isotonic(top, result, rows, chosen, view, psi_risk, chosen_cause)
    _pad(top, _extent(rows, chosen, limits, level_rows=True))
    if contrasts:
        diff = panels[1]
        for i, contrast in enumerate(contrasts):
            points = _series(rows, contrast, level=False)
            _draw_line(
                diff,
                points,
                color=_core.SECONDARY_INK,
                index=i,
                count=len(contrasts),
                step=step,
                pointwise=True,
                limits=limits,
                level=level,
                band_text=band_text,
                first=first,
            )
        _core.reference_line(diff, 0.0, label="No difference")
        _pad(diff, [0.0, *_extent(rows, contrasts, limits, level_rows=False)])
        diff.set_ylabel(
            _wrap_axis_label(_measure_word(view, outcome_label, units, difference=True))
        )
        diff_ends = []
        for contrast in contrasts:
            points = _series(rows, contrast, level=False)
            diff_ends.append((points[-1].x, points[-1].psi, name_of(contrast), _core.SECONDARY_INK))
        _core.legend(top, sources=[diff, top])
    else:
        diff_ends = []
        _core.legend(top, sources=[top])
    top.set_ylabel(_wrap_axis_label(_measure_word(view, outcome_label, units, difference=False)))
    xs = sorted({r.x for r in in_play})
    span = (xs[-1] - xs[0]) or 1.0
    left, right = xs[0] - 0.06 * span, xs[-1] + (0.24 if span else 0.4) * span
    top.set_xlim(left, right)
    for axis in panels:
        axis.set_xlim(left, right)
    nudge = 0.0 if step else 0.04 * span
    _label_ends(top, [(x + nudge, y, text, color) for x, y, text, color in ends])
    if contrasts:
        _label_ends(panels[1], [(x + nudge, y, t, c) for x, y, t, c in diff_ends])
    _count_row(count_ax, result, rows, chosen, name_of, time_label)
    heading = {
        "risk": "Risk over time",
        "survival": "Survival over time",
        "mean": "Mean outcome over time",
    }[view]
    suffix = f" ({chosen_cause})" if chosen_cause else ""
    _core.header(top, title or heading, _over_time_subtitle(view, suffix, step, contrasts))
    note_list = list(band_notes)
    note_list += _withheld_notes([r.estimate for r in in_play])
    note_list += _support_notes([r.estimate for r in in_play], "curve")
    if band_obj is not None:
        note_list += _band_note(band_obj.support, "simultaneous band")
        if view == "survival":
            note_list.append(
                "The survival band mirrors the fit's risk band about the survival estimate."
            )
    if not step:
        horizons = ", ".join(f"{r.x:g}" for r in _series(rows, chosen[0], level=True))
        note_list.append(
            f"This fit reports only the horizons {horizons}, so the curve is drawn at those "
            "horizons and nothing is drawn between them."
        )
    if result.isotonic_moves:
        note_list.append(
            "Hollow markers are the raw estimates the monotone projection moved; the drawn "
            "curve is the projected one."
        )
    _core.notes(count_ax, note_list)
    return fig


def _over_time_subtitle(view: str, suffix: str, step: bool, contrasts: list[str]) -> str:
    base = {
        "risk": "Cumulative risk under each regimen",
        "survival": "Chance of staying event-free under each regimen",
        "mean": "Mean outcome under each regimen",
    }[view]
    tail = ", and their difference below" if contrasts else ""
    return f"{base}{suffix}{tail}"


def _measure_word(view: str, outcome: str | None, units: str | None, *, difference: bool) -> str:
    unit = f" ({units})" if units else ""
    if view == "mean":
        name = outcome or "outcome"
        return f"Difference in mean {name}{unit}" if difference else f"Mean {name}{unit}"
    word = "risk" if view == "risk" else "survival"
    if difference:
        return f"Difference in {word}"
    return "Cumulative risk" if view == "risk" else "Survival probability"


def _mark_isotonic(
    ax: Axes,
    result: Any,
    rows: list[_Point],
    chosen: Sequence[str],
    view: str,
    psi_risk: Mapping[str, float],
    cause: str | None,
) -> None:
    """Hollow markers at the raw estimates the isotonic projection moved (rule 6)."""
    if not result.isotonic_moves or view == "mean":
        return
    index = result.parameter_index or {}
    lookup = {index[r.parameter]: r for r in rows if r.level}
    scaler = result.scaler
    for position, name in enumerate(chosen):
        raw_x, raw_y = [], []
        for label, move_cause, horizon, before, _after in result.isotonic_moves:
            if label != name or (move_cause is not None and move_cause != cause):
                continue
            row = lookup.get((label, move_cause, horizon))
            if row is None:
                continue
            raw_risk = float(scaler.unscale_level(before))
            shift = raw_risk - psi_risk[row.parameter]
            raw_x.append(row.x)
            raw_y.append(row.psi - shift if view == "survival" else row.psi + shift)
        if raw_x:
            ax.plot(
                raw_x,
                raw_y,
                marker=_MARKERS[position],
                markersize=8,
                markerfacecolor=_core.FACE,
                markeredgecolor=_core.PALETTE[position],
                markeredgewidth=1.6,
                linestyle="none",
                gid="cleverly-raw",
                zorder=4,
            )


def _count_row(
    ax: Axes,
    result: Any,
    rows: list[_Point],
    chosen: Sequence[str],
    name_of: Any,
    time_label: str | None,
) -> None:
    """The number following each regimen and still observed, under the time axis."""
    nodes: dict[int, float] = {}
    for row in rows:
        if row.level and row.regimen in chosen:
            nodes[row.node] = row.x
    counts = _counts(result, chosen, nodes)
    ax.set_ylim(len(chosen) - 0.4, -0.6)
    ax.set_yticks(range(len(chosen)), [name_of(name) for name in chosen])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for position, name in enumerate(chosen):
        for node, x in nodes.items():
            number = counts[name].get(node)
            ax.annotate(
                "-" if number is None else f"{number:,}",
                xy=(x, position),
                ha="center",
                va="center",
                fontsize=_core.TICK_SIZE,
                color=_core.INK,
                gid="cleverly-count",
            )
    ax.annotate(
        "Following the regimen, uncensored",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, 3.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=_core.TICK_SIZE,
        color=_core.SECONDARY_INK,
        gid="cleverly-count-caption",
    )
    grid = result.data.time_grid
    default = "Follow-up time" if grid is not None else "Follow-up time point"
    ax.set_xlabel(time_label or default)
    ticks = sorted({row.x for row in rows if row.level})
    if len(ticks) <= 12:
        ax.set_xticks(ticks)
    else:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=grid is None))


# ----------------------------------------------------------------------- V3: restricted means


def _restricted(
    result: Any,
    *,
    measure: str,
    cause: str | None,
    regimens: Sequence[str] | None,
    labels: Mapping[str, str] | None,
    units: str | None,
    time_label: str | None,
    title: str | None,
) -> Figure:
    grid = result.data.time_grid
    n_times = int(result.data.n_times)
    horizons: list[Any] = (
        list(range(2, n_times + 2)) if grid is None else [float(value) for value in grid[1:]]
    )
    found = sorted({key[0] for key in result._levels()})
    chosen = found if regimens is None else list(regimens)
    unknown = [name for name in chosen if name not in found]
    if unknown:
        raise ValueError(f"unknown regimen {unknown}; this fit reports {found}")
    if len(chosen) > _core.MAX_SERIES:
        raise ValueError(
            f"this fit has {len(chosen)} regimens and one axes holds at most "
            f"{_core.MAX_SERIES} coloured series; pass regimens= to draw up to "
            f"{_core.MAX_SERIES} of them"
        )
    reference = str(result.config.reference)
    versus = [name for name in chosen if name != reference] if reference in chosen else []
    compute = result.rmst if measure == "rmst" else result.rmtl
    extra: dict[str, Any] = {} if measure == "rmst" else {"cause": cause}
    levels = {name: [compute(name, h, **extra) for h in horizons] for name in chosen}
    contrasts = {
        name: [compute(name, h, versus=reference, **extra) for h in horizons] for name in versus
    }
    name_of = (lambda s: str(labels.get(s, s))) if labels else str
    unit = f" ({units})" if units else ""
    word = "survival time" if measure == "rmst" else "time lost"
    heights = [3.4] + ([1.7] if contrasts else [])
    fig, panels = _core.new_figure(
        len(heights), figsize=(7.2, 1.7 + sum(heights)), height_ratios=heights, sharex=True
    )
    top = panels[0]
    ends = []
    notes: list[str] = []
    all_estimates: list[Any] = []

    def draw(
        ax: Axes, series: list[Any], color: str, index: int, count: int, *, first: bool
    ) -> None:
        xs = np.array([float(h) for h in horizons])
        intervals = [_core.estimate_interval(e) for e in series]
        ys = np.array([i.psi for i in intervals])
        shift = _offset(ax, index, count)
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7, gid="cleverly-guide", zorder=2)
        label = _core.interval_label(intervals[0].level).replace("confidence", "pointwise")
        for x, interval in zip(xs, intervals, strict=True):
            if interval.drawn and interval.lower is not None and interval.upper is not None:
                ax.vlines(
                    x,
                    interval.lower,
                    interval.upper,
                    color=color,
                    linewidth=2.2,
                    transform=shift,
                    label=label if first else "_nolegend_",
                    gid="cleverly-interval",
                )
                first = False
        ax.plot(
            xs,
            ys,
            marker=_MARKERS[index % 3],
            markersize=6.5,
            color=color,
            linestyle="none",
            transform=shift,
            gid="cleverly-point",
            zorder=3,
        )
        spans[ax].extend(
            value
            for interval in intervals
            for value in (interval.psi, interval.lower, interval.upper)
            if value is not None
        )

    spans: dict[Any, list[float]] = {panel: [] for panel in panels}
    for i, name in enumerate(chosen):
        draw(top, levels[name], _core.PALETTE[i], i, len(chosen), first=i == 0)
        all_estimates.extend(levels[name])
        ends.append((float(horizons[-1]), levels[name][-1].psi, name_of(name), _core.PALETTE[i]))
    top.set_ylabel(_wrap_axis_label(f"Restricted mean {word}{unit}"))
    if contrasts:
        diff = panels[1]
        diff_ends = []
        for i, name in enumerate(versus):
            draw(diff, contrasts[name], _core.SECONDARY_INK, i, len(versus), first=i == 0)
            all_estimates.extend(contrasts[name])
            diff_ends.append(
                (
                    float(horizons[-1]),
                    contrasts[name][-1].psi,
                    f"{name_of(name)} vs {name_of(reference)}",
                    _core.SECONDARY_INK,
                )
            )
        _core.reference_line(diff, 0.0, label="No difference")
        diff.set_ylabel(_wrap_axis_label(f"Difference in restricted mean {word}{unit}"))
    span = float(horizons[-1]) - float(horizons[0]) or 1.0
    left, right = float(horizons[0]) - 0.06 * span, float(horizons[-1]) + 0.3 * span
    for axis in panels:
        axis.set_xlim(left, right)
        _pad(axis, [0.0, *spans[axis]] if axis is not top else spans[axis])
    nudge = 0.04 * span
    _label_ends(top, [(x + nudge, y, t, c) for x, y, t, c in ends])
    if contrasts:
        _label_ends(panels[1], [(x + nudge, y, t, c) for x, y, t, c in diff_ends])
    panels[-1].set_xlabel(time_label or "Restriction time (up to this time point)")
    panels[-1].set_xticks([float(h) for h in horizons])
    _core.legend(top, sources=panels[::-1])
    heading = (
        "Restricted mean survival up to each horizon"
        if measure == "rmst"
        else "Restricted mean time lost up to each horizon"
    )
    _core.header(
        top,
        title or heading,
        f"Average {word} accumulated before each restriction time, under each regimen",
    )
    notes.extend(_withheld_notes(all_estimates))
    notes.extend(_support_notes(all_estimates, "estimate"))
    notes.append(
        "Each point is a separate restricted mean with its own pointwise interval; the fit "
        "computes no band across horizons, and the thin line only joins the points."
    )
    _core.notes(panels[-1], notes)
    return fig


# ----------------------------------------------------------------------- shared point plots


def _numeric(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{what} {value!r} is not a number, so it has no place on a numeric axis; pass "
            "levels= to give each regime its numeric treatment level"
        ) from None


def _points_at(
    ax: Axes,
    xs: Sequence[float],
    estimates: Sequence[Any],
    *,
    color: str,
    marker: str,
    index: int,
    count: int,
    interval_text: str,
    box: Sequence[tuple[float, float] | None] | None = None,
    guide: bool = True,
    intervals: Sequence[tuple[float | None, float | None]] | None = None,
    band_text: str | None = None,
    extent: list[float] | None = None,
) -> list[str]:
    """Points with pointwise error bars and, where given, band brackets; returns the notes."""
    shift = _offset(ax, index, count)
    xs = list(xs)
    if intervals is None:
        read = [_core.estimate_interval(e) for e in estimates]
        ys = [r.psi for r in read]
        pairs = [(r.lower, r.upper) for r in read]
        found: list[str] = []
        found += _withheld_notes(list(estimates))
        found += _support_notes(list(estimates), "estimate")
    else:
        ys = [float(e) for e in estimates]
        pairs = list(intervals)
        found = []
    if extent is not None:
        extent.extend(ys)
        extent.extend(v for pair in pairs for v in pair if v is not None)
        extent.extend(v for pair in box or () if pair is not None for v in pair)
    order = np.argsort(xs)
    if guide:
        ax.plot(
            np.asarray(xs)[order],
            np.asarray(ys)[order],
            color=color,
            linewidth=1.0,
            alpha=0.7,
            gid="cleverly-guide",
            zorder=2,
        )
    first = True
    for x, (low, high) in zip(xs, pairs, strict=True):
        if low is not None and high is not None and _finite_pair(low, high):
            ax.vlines(
                x,
                low,
                high,
                color=color,
                linewidth=2.2,
                transform=shift,
                label=interval_text if first else "_nolegend_",
                gid="cleverly-interval",
            )
            first = False
    if box is not None:
        shown = [(x, pair) for x, pair in zip(xs, box, strict=True) if pair is not None]
        named = band_text
        for x, (low, high) in shown:
            ax.vlines(
                x,
                low,
                high,
                color=color,
                linewidth=1.0,
                alpha=0.55,
                transform=shift,
                label=named if named else "_nolegend_",
                gid="cleverly-band",
            )
            named = None
            for edge in (low, high):
                ax.plot(
                    [x],
                    [edge],
                    marker="_",
                    markersize=9,
                    color=color,
                    alpha=0.55,
                    linestyle="none",
                    transform=shift,
                    gid="cleverly-band-cap",
                )
    drawn = [low is not None and high is not None for low, high in pairs]
    for x, y, has in zip(xs, ys, drawn, strict=True):
        ax.plot(
            [x],
            [y],
            marker=marker,
            markersize=7,
            color=color,
            markerfacecolor=color if has else _core.FACE,
            markeredgewidth=1.5,
            linestyle="none",
            transform=shift,
            gid="cleverly-point",
            zorder=3,
        )
    return found


# ----------------------------------------------------------------------- V4: means by level


def _regime_levels(means: Any, levels: Mapping[str, Any] | None) -> dict[str, Any]:
    found: dict[str, Any] = {}
    estimand: Any = getattr(getattr(means, "identified_effect", None), "estimand", None)
    for regimen in getattr(estimand, "regimens", ()):
        if hasattr(regimen, "level") and getattr(regimen, "name", None):
            found[str(regimen.name)] = regimen.level
    if levels:
        found.update({str(key): value for key, value in levels.items()})
    return found


def _dose(
    means: Any,
    msm_fit: Any,
    *,
    levels: Mapping[str, Any] | None,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    title: str | None,
    ax: Axes | None,
) -> Figure:
    from ..msm import working_model_means

    level_of = _regime_levels(means, levels)
    entries = []
    for alias, key in means.parameter_keys.items():
        name = str(key.value)
        if name not in level_of:
            raise ValueError(
                f"the treatment level of regime {name!r} is not known; pass "
                f"levels={{{name!r}: <level>}}"
            )
        entries.append(
            (_numeric(level_of[name], "the treatment level"), means.estimates[alias], alias)
        )
    entries.sort(key=lambda item: item[0])
    xs = [e[0] for e in entries]
    estimates = [e[1] for e in entries]
    fig, axis = _core.figure_for(ax, figsize=(7.8, 4.8))
    outcome = outcome_label or str(getattr(means.data, "outcome_name", "outcome"))
    treatment = treatment_label or str(getattr(means.data, "treatment_name", "treatment"))
    unit = f" ({units})" if units else ""
    level = 1.0 - float(estimates[0].alpha)
    bands = means.simultaneous
    box = None
    if bands is not None:
        box = [bands.bands.get(e[2]) for e in entries]
    count = 2 if msm_fit is not None else 1
    extent: list[float] = []
    notes = _points_at(
        axis,
        xs,
        estimates,
        color=_core.PALETTE[0],
        marker="o",
        index=0,
        count=count,
        interval_text=_core.interval_label(level),
        box=box,
        band_text=None
        if bands is None
        else _core.band_label(1.0 - float(bands.alpha), "multiplier"),
        extent=extent,
    )
    _core.corner_label(axis, 0.97, "Mean outcome at each level", _core.PALETTE[0])
    if bands is not None:
        notes += _band_note(bands.support, "simultaneous band")
    subtitle = "Mean outcome if everyone were given each level"
    if msm_fit is not None:
        fitted = working_model_means(msm_fit)
        mx = [_numeric(label, "the working model's level") for label in fitted.levels]
        pairs: list[tuple[float | None, float | None]] = (
            [(None, None)] * len(mx)
            if fitted.lower is None or fitted.upper is None
            else [(float(lo), float(hi)) for lo, hi in zip(fitted.lower, fitted.upper, strict=True)]
        )
        notes += _points_at(
            axis,
            mx,
            [float(v) for v in fitted.estimate],
            color=_core.PALETTE[1],
            marker="D",
            index=1,
            count=2,
            interval_text=(f"{_core.level_text(1.0 - fitted.alpha)} interval of the working model"),
            intervals=pairs,
            extent=extent,
        )
        _core.corner_label(axis, 0.88, "Working model", _core.PALETTE[1])
        coefficients = [msm_fit.estimates[k] for k in msm_fit.estimates]
        notes += _withheld_notes(coefficients)
        notes += _support_notes(coefficients, "working-model fit")
        notes.append(
            f"The working model has the terms {', '.join(fitted.terms)} and the "
            f"{fitted.link} link. "
            "It is evaluated at the declared levels, and the thin line only joins them. Its "
            "pointwise interval is built on the link scale and mapped back."
        )
        subtitle += ", and the working model fitted to those means"
    axis.set_xticks(xs)
    _pad(axis, extent)
    axis.set_xlabel(f"{treatment}")
    axis.set_ylabel(f"Mean {outcome}{unit}")
    _core.legend(axis)
    _core.header(axis, title or "How does the mean outcome change with the dose?", subtitle)
    _core.notes(axis, notes)
    return fig


# ----------------------------------------------------------------------- V5: policies


def _policy_axis(estimand: Any) -> tuple[dict[str, float], str, bool, str] | None:
    """Policy name to its one scalar parameter, the axis words, log axis, and a cap note."""
    from ..interventions import Incremental, Scale, Shift

    declared: Any = getattr(estimand, "policies", None) or getattr(estimand, "interventions", ())
    policies = list(declared)
    if not policies:
        return None
    kinds = {type(policy) for policy in policies}
    if len(kinds) != 1:
        return None
    (kind,) = kinds
    if kind is Shift or kind is Scale:
        caps = {policy.cap for policy in policies}
        if len(caps) != 1:
            return None
        cap = next(iter(caps))
        cap_note = "" if cap is None else f"; every policy is capped at {cap}"
        if kind is Shift:
            values = {str(p.name): float(p.delta) for p in policies}
            return values, "Shift added to the treatment", False, cap_note
        values = {str(p.name): float(p.factor) for p in policies}
        return values, "Factor multiplying the treatment", False, cap_note
    if kind is Incremental:
        if any(isinstance(p.delta, (Mapping, tuple, list)) for p in policies):
            return None
        values = {str(p.name): float(p.delta) for p in policies}
        if any(v <= 0 for v in values.values()):
            return None
        return values, "Multiplier on the odds of treatment (log scale)", True, ""
    return None


def _policies(
    result: Any,
    *,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    title: str | None,
    ax: Axes | None,
) -> Figure:
    from ._forest import forest

    estimand = getattr(getattr(result, "identified_effect", None), "estimand", None)
    axis_spec = _policy_axis(estimand)
    if axis_spec is None:
        # No single scalar parameter orders these policies, so there is no axis to draw them
        # against: they are separate estimates, and a forest plot shows them as such.
        return forest(result, units=units)
    values, axis_words, log, cap_note = axis_spec
    entries = []
    for alias, key in result.parameter_keys.items():
        entries.append((values[str(key.value)], result.estimates[alias], alias, key))
    entries.sort(key=lambda item: item[0])
    xs = [e[0] for e in entries]
    estimates = [e[1] for e in entries]
    contrast = estimates[0].scale == "difference"
    reference = entries[0][3].reference
    fig, axis = _core.figure_for(ax, figsize=(7.0, 4.4))
    outcome = outcome_label or str(getattr(result.data, "outcome_name", "outcome"))
    treatment = treatment_label or str(getattr(result.data, "treatment_name", "treatment"))
    unit = f" ({units})" if units else ""
    level = 1.0 - float(estimates[0].alpha)
    bands = result.simultaneous
    box = None if bands is None else [bands.bands.get(e[2]) for e in entries]
    extent: list[float] = [0.0] if contrast else []
    notes = _points_at(
        axis,
        xs,
        estimates,
        color=_core.PALETTE[0],
        marker="o",
        index=0,
        count=1,
        interval_text=_core.interval_label(level),
        box=box,
        band_text=None
        if bands is None
        else _core.band_label(1.0 - float(bands.alpha), "multiplier"),
        extent=extent,
    )
    _pad(axis, extent)
    if bands is not None:
        notes += _band_note(bands.support, "simultaneous band")
        notes.append(
            f"The {_core.level_text(1.0 - float(bands.alpha))} simultaneous band covers the "
            "policies of this one fit together; it is marked at those policies only."
        )
    if contrast:
        _core.reference_line(axis, 0.0, label="No effect")
        axis.set_ylabel(f"Difference in mean {outcome}{unit}")
        subtitle = f"Each policy compared with {reference}{cap_note}"
        heading = "How does the effect change as the policy shifts?"
    else:
        axis.set_ylabel(f"Mean {outcome}{unit}")
        subtitle = f"Mean outcome under each policy{cap_note}"
        heading = "How does the mean outcome change as the policy shifts?"
    if log:
        _core.log_axis(axis, "x")
        axis.set_xticks(xs, [f"{x:g}" for x in xs])
        axis.xaxis.set_minor_formatter(_null_formatter())
    axis.set_xlabel(f"{axis_words} ({treatment})" if not log else axis_words)
    _core.legend(axis)
    _core.header(axis, title or heading, subtitle)
    _core.notes(axis, notes)
    return fig


def _null_formatter() -> Any:
    from matplotlib.ticker import NullFormatter

    return NullFormatter()


# ----------------------------------------------------------------------- V6: effect curve


def _observed(result: Any, name: str) -> np.ndarray | None:
    data: Any = getattr(getattr(result, "fit", None), "data", None)
    names = list(getattr(data, "covariate_names", ()) or ())
    if name not in names:
        return None
    return np.asarray(data.covariates, dtype=np.float64)[:, names.index(name)]


def _level_rows(result: Any, name: str, level: Any, reference: Any) -> np.ndarray | None:
    """Which fitted rows have ``name`` at ``level``, read from the fit's indicator columns."""
    data = getattr(getattr(result, "fit", None), "data", None)
    names = list(getattr(data, "covariate_names", ()) or ())
    columns = np.asarray(getattr(data, "covariates", np.empty((0, 0))), dtype=float)
    stem = f"{name}__"
    dummies = [(n, i) for i, n in enumerate(names) if n.startswith(stem)]
    if not dummies:
        return None
    if level == reference:
        return np.asarray(np.all([columns[:, i] == 0 for _, i in dummies], axis=0))
    match = [i for n, i in dummies if n == f"{stem}{level}"]
    return None if not match else np.asarray(columns[:, match[0]] == 1)


def _effect_curve(
    result: Any,
    *,
    modifier: str | None,
    by: str | None,
    at: Mapping[str, Any] | None,
    alpha: float,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    title: str | None,
    ax: Axes | None,
) -> Figure:
    from .._inference_status import supplies_inference
    from .._summary import vocabulary
    from .._summary._estimates import reason_kind

    mods = result.modifiers
    basis = getattr(result.model, "basis", None)
    read = tuple(getattr(basis, "columns", None) or mods.columns)
    continuous = [c for c in read if c in mods.continuous]
    if modifier is None:
        if len(continuous) != 1:
            raise ValueError(
                f"the curve reads the continuous modifiers {continuous}; pass modifier= to "
                "choose the one on the x axis and at= to hold the others"
            )
        modifier = continuous[0]
    if modifier not in mods.continuous:
        raise ValueError(f"modifier {modifier!r} is not a continuous modifier of this curve")
    if by is not None:
        if by not in mods.categorical:
            raise ValueError(f"by={by!r} is not a categorical modifier of this curve")
        if by not in read:
            raise ValueError(
                f"the curve does not read {by!r}, so its panels would all be the same curve"
            )
    held = dict(at or {})
    unknown = [name for name in held if name not in mods.columns]
    if unknown:
        raise ValueError(f"at= names {unknown}, which are not modifiers {list(mods.columns)}")
    missing = [c for c in read if c not in held and c not in (modifier, by)]
    if missing:
        raise ValueError(
            f"the curve also reads {missing}; hold each at a value with at={{...}}, because "
            f"the plot draws the curve over {modifier!r} alone"
        )
    seen = _observed(result, modifier)
    domain = mods.domain(modifier)
    if seen is None and domain is None:
        raise ValueError(f"cannot find the range of {modifier!r}: declare a domain for it")
    low = float(np.min(seen)) if seen is not None else float(domain[0])
    high = float(np.max(seen)) if seen is not None else float(domain[1])
    if domain is not None:
        low, high = max(low, float(domain[0])), min(high, float(domain[1]))
    grid = np.linspace(low, high, 101)
    panels_levels: list[Any] = list(mods.levels_of(by)) if by is not None else [None]
    ratio = result.scale != "difference"
    transform = np.exp if ratio else (lambda v: v)
    projection = result.fitted_method != "ep"
    drawn_interval = projection and supplies_inference(result.inference_status)

    def evaluate(level: Any) -> dict[str, np.ndarray]:
        frame: dict[str, Any] = {modifier: grid}
        for name, value in held.items():
            if name in read:
                frame[name] = np.full(grid.size, value)
        if by is not None:
            frame[by] = np.full(grid.size, level)
        found: dict[str, np.ndarray] = {}
        if projection:
            table = result.predict(frame, interval=True, alpha=alpha)
            found["estimate"] = np.asarray(_column(table, "estimate"), dtype=float)
            found["lower"] = np.asarray(_column(table, "ci_lower"), dtype=float)
            found["upper"] = np.asarray(_column(table, "ci_upper"), dtype=float)
        else:
            found["estimate"] = np.asarray(result.predict(frame), dtype=float)
        return found

    predictions = {level: evaluate(level) for level in panels_levels}
    rows = len(panels_levels)
    if ax is not None and rows > 1:
        raise ValueError("by= draws one panel per level, which needs its own figure; omit ax=")
    if ax is None:
        fig, panels = _core.new_figure(rows, figsize=(7.4, 2.4 + 2.6 * rows), sharex=True)
    else:
        fig, only = _core.figure_for(ax)
        panels = [only]
    outcome = outcome_label or str(
        getattr(getattr(result.fit, "data", None), "outcome_name", "outcome")
    )
    unit = f" ({units})" if units else ""
    average = _curve_average(result)
    band = result.band
    notes: list[str] = []
    for panel, level in zip(panels, panels_levels, strict=True):
        if by is not None:
            for other in panels_levels:
                if other != level:
                    panel.plot(
                        grid,
                        transform(predictions[other]["estimate"]),
                        color=_core.CONTEXT,
                        linewidth=1.2,
                        gid="cleverly-context",
                        zorder=1,
                    )
        data = predictions[level]
        color = _core.PALETTE[0]
        if drawn_interval:
            panel.fill_between(
                grid,
                transform(data["lower"]),
                transform(data["upper"]),
                color=color,
                alpha=_core.INTERVAL_ALPHA,
                linewidth=0,
                label=_core.pointwise_label(1.0 - alpha),
                gid="cleverly-ribbon",
            )
        panel.plot(
            grid,
            transform(data["estimate"]),
            color=color,
            linewidth=2.2,
            gid="cleverly-curve",
            zorder=3,
        )
        _core.reference_line(panel, 1.0 if ratio else 0.0, label="No effect")
        panel.axhline(
            transform(average),
            color=_core.SECONDARY_INK,
            linewidth=1.0,
            linestyle=(0, (1, 2)),
            label="Average of the curve",
            gid="cleverly-average",
            zorder=1.5,
        )
        if band is not None:
            _band_marks(panel, band, modifier, by, level, held, transform, color)
        _rug(panel, result, modifier, by, level, mods)
        if by is not None:
            _core.corner_label(panel, 0.97, f"{by} = {level}", color)
        panel.set_ylabel(
            "Conditional risk ratio (log scale)" if ratio else f"Effect on {outcome}{unit}"
        )
        if ratio:
            _core.log_axis(panel, "y")
    if len(panels) > 1:
        shared = [panel.get_ylim() for panel in panels]
        for panel in panels:
            panel.set_ylim(min(low for low, _ in shared), max(high for _, high in shared))
    panels[-1].set_xlabel(modifier)
    _core.legend(panels[0], sources=[panels[0]])
    if band is not None:
        matching = _matching_band_points(band, modifier, by, panels_levels[0], held)
        if not matching.any():
            notes.append(
                "The fit's band is declared at other values of the held modifiers, so it is not "
                "drawn here."
            )
        else:
            notes.append(
                f"The band holds jointly at {_core.level_text(1.0 - float(band.alpha))} over the "
                f"{int(np.asarray(band.estimate).size)} declared grid points, with critical value "
                f"{band.critical_value:.3f}. No registered study measures its joint coverage, so "
                "it is experimental. Brackets mark it at those points only; nothing is drawn "
                "between them."
            )
    if not projection:
        notes.append(
            "The EP curve carries no interval: its source proves no limit law for the curve, "
            "so only the estimate is drawn."
        )
    elif not drawn_interval:
        from .._inference_status import status_record

        notes.append(
            f"No interval is drawn. {status_record(str(result.inference_status)).plain_reason}"
        )
    if result.status == "experimental":
        notes.append(
            vocabulary.EXPERIMENTAL_REASONS[reason_kind(result.status_reason)].format(
                names="the curve"
            )
        )
    notes.append(
        "The dotted line is the average of the curve over the fitted rows"
        + (", weighted by the data's weights." if _has_weights(result) else ".")
    )
    if held:
        shown = ", ".join(
            f"{name} = {value:g}" if isinstance(value, (int, float)) else f"{name} = {value}"
            for name, value in held.items()
        )
        notes.append(f"The other modifiers are held at {shown}.")
    notes.append(
        "The curve is the projection of the conditional effect onto the declared model; it "
        "equals the conditional effect only where that effect lies in the model."
        if projection
        else "The curve is the EP learner's estimate of the conditional effect."
    )
    _core.header(
        panels[0],
        title or f"Conditional effect by {modifier}",
        _effect_subtitle(result, ratio),
    )
    _core.notes(panels[-1], notes)
    return fig


def _effect_subtitle(result: Any, ratio: bool) -> str:
    word = "risk ratio" if ratio else "average treatment effect"
    method = "projected onto the declared model" if result.fitted_method != "ep" else "EP learner"
    return f"Conditional {word}, {method}"


def _has_weights(result: Any) -> bool:
    data = getattr(getattr(result, "fit", None), "data", None)
    weights = getattr(data, "weights", None)
    if weights is None or np.asarray(weights).shape != np.asarray(result.in_sample).shape:
        return False
    return bool(np.ptp(np.asarray(weights, dtype=float)) > 0.0)


def _curve_average(result: Any) -> float:
    """The average of the curve over the fitted rows, weighted when the fit has weights."""
    values = np.asarray(result.in_sample, dtype=float)
    if _has_weights(result):
        weights = np.asarray(result.fit.data.weights, dtype=float)
        return float(np.average(values, weights=weights))
    return float(np.mean(values))


def _matching_band_points(
    band: Any, modifier: str, by: str | None, level: Any, held: Mapping[str, Any]
) -> np.ndarray:
    size = int(np.asarray(band.estimate).size)
    mask = np.ones(size, dtype=bool)
    for name, values in band.grid.items():
        array = np.asarray(values)
        if name == modifier:
            continue
        if name == by:
            mask &= array == level
        elif name in held:
            wanted = held[name]
            if isinstance(wanted, (int, float)) and array.dtype.kind in "fiu":
                mask &= np.isclose(array.astype(float), float(wanted))
            else:
                mask &= array == wanted
    return mask


def _band_marks(
    ax: Axes,
    band: Any,
    modifier: str,
    by: str | None,
    level: Any,
    held: Mapping[str, Any],
    transform: Any,
    color: str,
) -> None:
    """Bracket marks of the declared band at its grid points that match the held values."""
    if modifier not in band.grid:
        return
    mask = _matching_band_points(band, modifier, by, level, held)
    if not mask.any():
        return
    x = np.asarray(band.grid[modifier], dtype=float)[mask]
    low = transform(np.asarray(band.lower, dtype=float)[mask])
    high = transform(np.asarray(band.upper, dtype=float)[mask])
    ax.vlines(
        x,
        low,
        high,
        color=color,
        linewidth=1.2,
        alpha=0.6,
        label=_core.band_label(1.0 - float(band.alpha), "multiplier"),
        gid="cleverly-band",
    )
    for edge in (low, high):
        ax.plot(
            x,
            edge,
            marker="_",
            markersize=9,
            color=color,
            alpha=0.6,
            linestyle="none",
            gid="cleverly-band-cap",
        )


def _rug(ax: Axes, result: Any, modifier: str, by: str | None, level: Any, mods: Any) -> None:
    """Short ticks along the bottom edge at the fitted rows' modifier values."""
    from matplotlib.transforms import blended_transform_factory

    values = _observed(result, modifier)
    if values is None:
        return
    if by is not None:
        mask = _level_rows(result, by, level, mods.levels_of(by)[0])
        if mask is not None:
            values = values[mask]
    if values.size > 20000:
        values = values[:: math.ceil(values.size / 20000)]
    ax.vlines(
        values,
        0.0,
        0.035,
        transform=blended_transform_factory(ax.transData, ax.transAxes),
        color=_core.SECONDARY_INK,
        linewidth=0.5,
        alpha=0.35,
        gid="cleverly-rug",
    )
