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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import _core, _truth

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Marker and dash pattern of the first three series, so identity never rests on colour.
_MARKERS = ("o", "s", "^")
_DASHES: tuple[Any, ...] = ("-", (0, (5, 2)), (0, (1, 1.6)))
#: Inches between neighbouring series at a shared x, so their error bars and band brackets
#: do not overlap: wider than a band bracket (:func:`_core.brackets`, 12 points).
_DODGE = 0.19
#: Points right of its estimate at which a true value's cross is drawn where the estimate
#: carries a bar and brackets, so the cross does not sit on them: the bracket's half width
#: (6 points), the cross's half width (4) and a gap.
_TRUTH_NUDGE = 11.0
#: The dodge between series when crosses are drawn: room for a cross between neighbours.
_TRUTH_DODGE = 0.33


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


def _offset(ax: Axes, index: int, count: int, gap: float = _DODGE) -> Any:
    """``ax.transData`` shifted sideways by the series' place, in inches."""
    from matplotlib.transforms import ScaledTranslation

    shift = (index - (count - 1) / 2.0) * gap
    return ax.transData + ScaledTranslation(shift, 0.0, ax.figure.dpi_scale_trans)


#: Points the pointwise bar and the band's caps sit either side of their estimate.
_BAND_SPLIT = 6.0


def _nudged(ax: Axes, transform: Any, points: float) -> Any:
    """``transform`` moved ``points`` to the right, for a mark drawn beside its estimate."""
    from matplotlib.transforms import ScaledTranslation

    return transform + ScaledTranslation(points / 72.0, 0.0, ax.figure.dpi_scale_trans)


def _beside(ax: Axes, transform: Any) -> Any:
    """``transform`` moved :data:`_TRUTH_NUDGE` points right, for a cross beside a bar.

    Only the drawing moves: the cross's data stay at the estimate's x.
    """
    from matplotlib.transforms import ScaledTranslation

    return transform + ScaledTranslation(_TRUTH_NUDGE / 72.0, 0.0, ax.figure.dpi_scale_trans)


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


def _wrap_axis_label(prefix: str, phrase: str, width: int = 26) -> str:
    """An axis label that fits the height of its panel, breaking only where it must.

    The label is ``prefix`` and then ``phrase`` (the quantity, such as ``"order share"``).
    A label that fits stays on one line; otherwise the break goes between the two, so the
    phrase is never split while it fits a line of its own.
    """
    import textwrap

    text = f"{prefix} {phrase}".strip()
    if len(text) <= width:
        return text
    if prefix and len(phrase) <= width:
        return f"{prefix}\n{phrase}"
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


@_core.labelled
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
    alpha: float | None = None,
    outcome_label: str | None = None,
    treatment_label: str | None = None,
    units: str | None = None,
    time_label: str | None = None,
    truth: Any = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
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
    A conditional effect curve draws the effect over one modifier. A smoothed dose-response
    curve draws one series per declared bandwidth, at most three, with its pointwise
    intervals and the band over every declared pair.

    A band is drawn only when the fit computed one. A fit that reports every time point
    gets step ribbons, because the curve is constant between time points, and the last time
    point, where a step has no width, gets a bar and bracket marks; a fit that reports only
    some horizons gets points and bracket marks at those horizons. A result
    that withholds inference draws points only, and :func:`~cleverly.plotting.notes` gives
    the reason.

    Parameters
    ----------
    obj : LongitudinalResult, TMLEResult, EffectCurveResult or DoseResponseResult
        What to draw.
    scale : {"risk", "survival", "mean"} or None, default=None
        Longitudinal fits: the view of the outcome. ``None`` is ``"risk"`` for a survival
        outcome and ``"mean"`` for a repeated one. The levels and differences are read from
        ``obj.curve(scale=...)``.
    measure : {"curve", "rmst", "rmtl"}, default="curve"
        Longitudinal fits: the curve over time, or the restricted mean survival time or time
        lost up to each horizon (points joined by a thin line, pointwise intervals only).
    band : {"multiplier", "bootstrap", "none"}, False or None, default=None
        Which simultaneous band to draw. ``None`` draws the multiplier band when the fit holds one,
        else the bootstrap band; :func:`~cleverly.plotting.notes` names the other. ``"none"`` (or
        ``False``) draws no band, so a figure need not show a band its page does not discuss. A
        point-treatment fit of per-level means or of policies, and a conditional effect curve, hold
        a multiplier band at most.
    cause : str or None, default=None
        Longitudinal fits with several causes: the cause to draw.
    regimens : sequence of str or None, default=None
        Longitudinal fits: the regimens to draw, at most three.
    labels : mapping of str to str or None, default=None
        Display names for regimens.
    msm : TMLEResult or None, default=None
        Per-level means (a fit of ``RegimeMean`` or ``CounterfactualMean``): the
        point-treatment fit of a working model to draw beside them.
    means : TMLEResult or None, default=None
        Working model fit: the fit of per-level means to draw beside it.
    levels : mapping or None, default=None
        Per-level means: the numeric treatment level of each regime name or arm label, when
        the names are not the levels themselves. It names every regime or arm the fit
        reports and no other. The working model's levels are mapped the same way, so
        labelled arms such as ``"low"`` and ``"high"`` need it once.
    modifier : str or None, default=None
        Conditional effect curve: the modifier on the x axis; the only continuous modifier
        the curve reads when omitted.
    by : str or None, default=None
        Conditional effect curve: a categorical modifier, drawn as one panel per level.
        Dose-response curve with baseline strata: the stratum label to draw, such as
        ``"V=1"``.
    at : mapping or None, default=None
        Conditional effect curve: the value at which each other modifier the curve reads is
        held.
    alpha : float or None, default=None
        Conditional effect curve: the pointwise interval holds at ``1 - alpha``. ``None``
        uses the level of the curve's band when it has one, so the two are read at one
        level, and the level of the fit's coefficients otherwise.
    outcome_label : str or None, default=None
        Words for the quantity on the y axis of a mean or a difference in means, such as
        ``"order share"``: the label reads ``"order share"`` and ``"Difference in order
        share"``. Without it the label is the mean of the outcome column, ``"Mean Y"`` and
        ``"Difference in mean Y"``. A conditional effect curve reads ``"Effect on"`` the
        words.
    treatment_label : str or None, default=None
        Words for the treatment on the x axis.
    units : str or None, default=None
        Units of the quantity on the y axis, added to its label: the outcome of a mean
        view, a dose, policy or effect plot, or time for a restricted mean. A risk or a
        survival probability has none, and this is ignored there.
    time_label : str or None, default=None
        Words for the time axis of a longitudinal fit, with its units, such as
        ``"Days since enrolment"``.
    truth : mapping, sequence, callable or None, default=None
        The known true values of simulated data, drawn as ink crosses beside the estimates
        and named "True value (simulated data only)" in the key. Real data have none. A
        mapping of estimate name to value (``{"ate_regimen[a vs b @ t=1]": 0.1}``) works
        on every plot of named estimates; a name the fit reports but the plot does not
        draw is passed over. A figure of one series also takes one value per x level, as
        a sequence in increasing order of the level or a mapping of level to value. A
        conditional effect curve takes a function of the modifier's values, drawn as a
        dash-dot line. Values are on the scale the plot draws.
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
        causes), with the fit's own reason, or when ``truth`` does not match what the plot
        draws.

    See Also
    --------
    cleverly.plotting.forest : The estimates of one or several fits as a forest plot.
    cleverly.longitudinal.estimator.LongitudinalResult.plot : This plot for a longitudinal fit.
    cleverly.effect_curves.EffectCurveResult.plot : This plot for a conditional effect curve.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say which estimates or bands
    are experimental, why an interval is withheld, which band the fit also holds, the values
    other modifiers are held at, which horizons a fit reports, and what a working model or
    projection means for the drawn curve.

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
    from ..estimators.dose_response import DoseResponseResult
    from ..longitudinal.estimator import LongitudinalResult

    if band is False:
        band = "none"
    if isinstance(obj, DoseResponseResult):
        return _dose_response(
            obj,
            band=band,
            by=by,
            outcome_label=outcome_label,
            treatment_label=treatment_label,
            units=units,
            truth=truth,
            title=title or None,
            ax=ax,
        )
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
                truth=truth,
                title=title or None,
            )
        return _restricted(
            obj,
            measure=measure,
            cause=cause,
            regimens=regimens,
            labels=labels,
            units=units,
            time_label=time_label,
            truth=truth,
            title=title or None,
        )
    if isinstance(obj, EffectCurveResult):
        return _effect_curve(
            obj,
            band_choice=band,
            modifier=modifier,
            by=by,
            at=at,
            alpha=alpha,
            outcome_label=outcome_label,
            treatment_label=treatment_label,
            units=units,
            truth=truth,
            title=title or None,
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
                band=band,
                levels=levels,
                outcome_label=outcome_label,
                treatment_label=treatment_label,
                units=units,
                truth=truth,
                title=title or None,
                ax=ax,
            )
        if axes == {"regime"} or (axes == {"arm"} and _level_means(keys.values())):
            return _dose(
                obj,
                msm,
                band=band,
                levels=levels,
                outcome_label=outcome_label,
                treatment_label=treatment_label,
                units=units,
                truth=truth,
                title=title or None,
                ax=ax,
            )
        if axes <= {"policy", "ipsi"}:
            return _policies(
                obj,
                band=band,
                outcome_label=outcome_label,
                treatment_label=treatment_label,
                units=units,
                truth=truth,
                title=title or None,
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


def _select(
    rows: list[_Point],
    regimens: Sequence[str] | None,
    *,
    order: Sequence[str] | None = None,
    sides: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """The regimens to draw and the contrast labels that go with them.

    The regimens follow ``order`` (the fit's own, :func:`_regimen_order`) when given, so
    every plot of one result lists and colours them alike. A fit that reports contrasts
    and no level of any regimen (a regime contrast) chooses among ``sides``, the regimens
    its contrasts compare; those regimens get a row of counts and no curve of their own.
    """
    reported = list(dict.fromkeys(row.regimen for row in rows if row.level))
    contrast_only = not reported
    if contrast_only:
        reported = list(sides)
    found = [name for name in order or () if name in reported] or reported
    chosen = found if regimens is None else list(regimens)
    unknown = [name for name in chosen if name not in found]
    if unknown:
        raise ValueError(f"unknown regimen {unknown}; this fit reports {found}")
    if not contrast_only and len(chosen) > _core.MAX_SERIES:
        raise ValueError(
            f"this fit has {len(chosen)} regimens and one axes holds at most "
            f"{_core.MAX_SERIES} coloured series; pass regimens= to draw up to "
            f"{_core.MAX_SERIES} of them"
        )
    contrasts = list(dict.fromkeys(row.regimen for row in rows if not row.level))
    if regimens is not None:
        wanted = {f"{a} vs {b}" for a in chosen for b in chosen if a != b}
        contrasts = [name for name in contrasts if name in wanted]
    if contrast_only and not contrasts:
        raise ValueError(
            f"regimens={chosen} selects no contrast of this fit, because none of its "
            f"contrasts is between those regimens; pass at least two of {found}"
        )
    if len(contrasts) > _core.MAX_SERIES:
        raise ValueError(
            f"this fit has {len(contrasts)} contrasts and one axes holds at most "
            f"{_core.MAX_SERIES} series; pass regimens= to select fewer"
        )
    return chosen, contrasts


def _series(rows: list[_Point], regimen: str, *, level: bool) -> list[_Point]:
    return sorted((r for r in rows if r.regimen == regimen and r.level == level), key=lambda r: r.x)


def _point_band(held: Any, band: str | None) -> Any:
    """The multiplier band ``held`` by a fit to draw, or ``None`` for ``"none"``.

    A point-treatment fit and a conditional effect curve hold a multiplier band at most, so
    ``"bootstrap"`` is refused, as is ``"multiplier"`` when the fit holds none.
    """
    if band not in (None, "multiplier", "bootstrap", "none"):
        raise ValueError(f"band must be 'multiplier', 'bootstrap' or 'none'; got {band!r}")
    if band == "bootstrap":
        raise ValueError("this fit holds no bootstrap band; its band is the multiplier band")
    if band == "multiplier" and held is None:
        raise ValueError("this fit holds no multiplier band")
    return None if band == "none" else held


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
    notes = [f"The simultaneous band drawn is the {kind} band."]
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
    """Whether the fit reports every node (rule 2's exception for step ribbons).

    The levels decide, or the contrasts of a fit that reports no level.
    """
    reported = {row.node for row in rows if row.level} or {row.node for row in rows}
    return reported == set(range(1, int(result.data.n_times) + 1))


def _contrast_sides(result: Any, rows: list[_Point]) -> list[str]:
    """The regimens the fit's contrasts compare, in the fit's own regimen order."""
    labels = [str(regimen.label) for regimen in result.config.regimens]
    labels += [str(plan.label) for plan in getattr(result.config, "separable", ())]
    named = {row.regimen for row in rows if not row.level}
    return [
        name
        for name in labels
        if any(
            contrast in named
            for other in labels
            for contrast in (f"{name} vs {other}", f"{other} vs {name}")
        )
    ]


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


def _regimen_order(result: Any) -> list[str]:
    """The fit's regimens in its own reporting order, the one source of series colours."""
    return list(dict.fromkeys(str(key[0]) for key in result._levels()))


def _slots(chosen: Sequence[str], order: Sequence[str]) -> dict[str, int]:
    """The palette slot of each drawn regimen.

    A regimen keeps its slot in the fit's order across every plot of one result (style rule
    of plan 4.2); a fit with more regimens than slots numbers the drawn ones in turn.
    """
    if len(order) <= _core.MAX_SERIES and all(name in order for name in chosen):
        return {name: order.index(name) for name in chosen}
    return {name: i for i, name in enumerate(chosen)}


def _draw_points(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    has_interval: Sequence[bool],
    *,
    color: str,
    marker: str,
    transform: Any,
    size: float = 7.0,
) -> None:
    """The estimates: filled where an interval is drawn, hollow where none is (rule 1)."""
    for x, y, has in zip(xs, ys, has_interval, strict=True):
        ax.plot(
            [x],
            [y],
            marker=marker,
            markersize=size,
            color=color,
            markerfacecolor=color if has else _core.FACE,
            markeredgewidth=1.5,
            linestyle="none",
            transform=transform,
            gid=_core.GID_POINT,
            zorder=3,
        )


def _dodged_brackets(
    ax: Axes,
    xs: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    color: str,
    label: str,
    band: bool,
    transform: Any,
) -> None:
    """:func:`_core.brackets` at the finite limits, shifted sideways with the series."""
    keep = [i for i in range(len(xs)) if _finite_pair(lower[i], upper[i])]
    if not keep:
        return
    artist = _core.brackets(
        ax,
        [float(xs[i]) for i in keep],
        [float(lower[i]) for i in keep],
        [float(upper[i]) for i in keep],
        color=color,
        label=label,
        band=band,
    )
    artist.set_transform(transform)


def _held_steps(
    x: Sequence[float], low: Sequence[float], high: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    """Step-ribbon inputs in which a node before a missing one still holds to the next node.

    ``fill_between(step="post")`` breaks at a missing value and drops the step that leads
    into it, so a node followed by a missing node would span no width. Its limits are
    repeated at the next node's position before the gap.
    """
    xs: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for i, (position, lo, hi) in enumerate(zip(x, low, high, strict=True)):
        xs.append(position)
        lows.append(lo)
        highs.append(hi)
        if i + 1 < len(x) and _finite_pair(lo, hi) and not _finite_pair(low[i + 1], high[i + 1]):
            xs.append(x[i + 1])
            lows.append(lo)
            highs.append(hi)
    return xs, lows, highs


def _draw_line(
    ax: Axes,
    points: list[_Point],
    *,
    color: str,
    index: int,
    dodge: tuple[int, int],
    step: bool,
    limits: Mapping[str, tuple[float, float]] | None,
    level: float,
    band_text: str | None,
    first: dict[str, bool],
    spots: list[_truth.Spot],
    gap: float = _DODGE,
) -> None:
    """One series: the curve, its pointwise interval and its band.

    A node without an interval leaves a gap in the ribbon (or no bar) and a hollow marker;
    the other nodes keep theirs. A true value's cross sits beside each node drawn with a bar
    and brackets (every node of a point curve, the last of a step curve), not on them.
    """
    x = [p.x for p in points]
    y = [p.psi for p in points]
    style = _DASHES[index % len(_DASHES)]
    low = [p.lower if p.lower is not None else math.nan for p in points]
    high = [p.upper if p.upper is not None else math.nan for p in points]
    has = [_finite_pair(lo, hi) for lo, hi in zip(low, high, strict=True)]
    box = None
    if limits is not None and any(p.parameter in limits for p in points):
        box = (
            [limits[p.parameter][0] if p.parameter in limits else math.nan for p in points],
            [limits[p.parameter][1] if p.parameter in limits else math.nan for p in points],
        )

    def tag(key: str, text: str) -> str:
        if first[key]:
            first[key] = False
            return text
        return "_nolegend_"

    if step:
        # The ribbons mark the last node with a bar and brackets; those, and the last point,
        # move sideways with the series so that several series' marks there do not overlap.
        from matplotlib.collections import PolyCollection

        before = (len(ax.lines), len(ax.collections))
        if box is not None:
            _core.ribbon(
                ax,
                *_held_steps(x, box[0], box[1]),
                color=color,
                label=tag("band", band_text or ""),
                band=True,
                step=True,
            )
        if any(has):
            _core.ribbon(
                ax,
                *_held_steps(x, low, high),
                color=color,
                label=tag("pointwise", _core.pointwise_label(level)),
                step=True,
            )
        ends = _offset(ax, *dodge, gap=gap)
        for artist in [*ax.lines[before[0] :], *ax.collections[before[1] :]]:
            if not isinstance(artist, PolyCollection):
                artist.set_transform(ends)
        ax.step(
            x, y, where="post", color=color, linewidth=2.0, linestyle=style, gid="cleverly-curve"
        )
        _draw_points(
            ax,
            x[:-1],
            y[:-1],
            has[:-1],
            color=color,
            marker=_MARKERS[index % len(_MARKERS)],
            transform=ax.transData,
            size=5.5,
        )
        _draw_points(
            ax,
            x[-1:],
            y[-1:],
            has[-1:],
            color=color,
            marker=_MARKERS[index % len(_MARKERS)],
            transform=ends,
            size=5.5,
        )
        last = len(points) - 1
        spots.extend(
            _truth.Spot(
                ax, p.parameter, p.x, _beside(ax, ends) if i == last else ax.transData, p.regimen
            )
            for i, p in enumerate(points)
        )
        return
    transform = _offset(ax, *dodge, gap=gap)
    beside = _beside(ax, transform)
    spots.extend(_truth.Spot(ax, p.parameter, p.x, beside, p.regimen) for p in points)
    if box is not None:
        _dodged_brackets(
            ax,
            x,
            box[0],
            box[1],
            color=color,
            label=tag("band", band_text or ""),
            band=True,
            transform=transform,
        )
    if any(has):
        _dodged_brackets(
            ax,
            x,
            low,
            high,
            color=color,
            label=tag("pointwise", _core.pointwise_label(level)),
            band=False,
            transform=transform,
        )
    _draw_points(
        ax,
        x,
        y,
        has,
        color=color,
        marker=_MARKERS[index % len(_MARKERS)],
        transform=transform,
        size=7.0,
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


def _label_ends(ax: Axes, ends: list[tuple[float, float, str, str]], *, pad: float = 0.0) -> None:
    """Direct labels at the right end of each series, moved apart where they would touch.

    ``pad`` moves them a further number of points right, past the crosses of true values.
    """
    low, high = ax.get_ylim()
    heights = _spread([y for _, y, _, _ in ends], 0.075 * (high - low))
    for (x, _, text, color), y in zip(ends, heights, strict=True):
        _core.end_label(ax, x, y, text, color, pad=pad)


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
    truth: Any,
    title: str | None,
) -> Figure:
    view = scale or ("mean" if result.data.is_repeated else "risk")
    rows, chosen_cause = _read_curve(result, view, cause)
    order = _regimen_order(result)
    has_levels = any(row.level for row in rows)
    sides = [] if has_levels else _contrast_sides(result, rows)
    chosen, contrasts = _select(rows, regimens, order=order or sides, sides=sides)
    # The regimens drawn as curves; a fit of contrasts alone draws its difference panel only.
    drawn = chosen if has_levels else []
    slot = _slots(drawn, order)
    name_of = (lambda s: str(labels.get(s, s))) if labels else str
    step = _nodes_reported(result, rows)
    level = 1.0 - float(rows[0].estimate.alpha)
    band_obj, _, band_notes = _band_choice(result, band)
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
    band_text = None if band_obj is None else _core.band_label(1.0 - float(band_obj.alpha))

    tall = (
        ([3.4] if has_levels else [])
        + ([1.7 if has_levels else 3.4] if contrasts else [])
        + [0.45 + 0.3 * len(chosen)]
    )
    fig, panels = _core.new_figure(
        len(tall), figsize=(7.2, 1.2 + sum(tall)), height_ratios=tall, sharex=True
    )
    top = panels[0]
    count_ax = panels[-1]
    count_ax.grid(False)
    first = {"band": True, "pointwise": True}
    spots: list[_truth.Spot] = []
    ends: list[tuple[float, float, str, str]] = []
    # Crosses of true values sit right of their estimates' bars, so neighbours move apart.
    gap = _DODGE if truth is None else _TRUTH_DODGE
    for i, regimen in enumerate(drawn):
        points = _series(rows, regimen, level=True)
        color = _core.series_color(slot[regimen])
        _draw_line(
            top,
            points,
            color=color,
            index=slot[regimen],
            dodge=(i, len(drawn)),
            step=step,
            limits=limits,
            level=level,
            band_text=band_text,
            first=first,
            spots=spots,
            gap=gap,
        )
        ends.append((points[-1].x, points[-1].psi, name_of(regimen), color))
    marked = _mark_isotonic(top, result, rows, drawn, slot, view, psi_risk, chosen_cause)
    if has_levels:
        _pad(top, _extent(rows, chosen, limits, level_rows=True))
        top.set_ylabel(
            _wrap_axis_label(*_measure_word(view, outcome_label, units, difference=False), 40)
        )
    diff_ends: list[tuple[float, float, str, str]] = []
    diff = panels[1] if has_levels else panels[0]
    if contrasts:
        # Each difference takes the colour of its plan (the side compared with the
        # reference), which keeps it apart from the grey no-difference line. Beside the
        # regimens' curves it is the plan's own curve colour above; differences that share a
        # plan there stay grey, since numbering them in turn would give one a curve's colour
        # it does not belong to.
        plan_slots = (
            _plan_slots(contrasts, sides) if not has_levels else _drawn_plan_slots(contrasts, slot)
        )
        for i, contrast in enumerate(contrasts):
            points = _series(rows, contrast, level=False)
            index = i if plan_slots is None else plan_slots[contrast]
            color = _core.SECONDARY_INK if plan_slots is None else _core.series_color(index)
            _draw_line(
                diff,
                points,
                color=color,
                index=index,
                dodge=(i, len(contrasts)),
                step=step,
                limits=limits,
                level=level,
                band_text=band_text,
                first=first,
                spots=spots,
                gap=gap,
            )
            diff_ends.append((points[-1].x, points[-1].psi, name_of(contrast), color))
        _core.reference_line(diff, 0.0, label="No difference")
        _pad(diff, [0.0, *_extent(rows, contrasts, limits, level_rows=False)])
        # The difference panel is half the height of the curves above it, or the whole
        # figure when the fit reports differences alone.
        diff.set_ylabel(
            _wrap_axis_label(
                *_measure_word(view, outcome_label, units, difference=True),
                40 if not has_levels else 26,
            )
        )
    xs = sorted({r.x for r in in_play})
    span = (xs[-1] - xs[0]) or 1.0
    left, right = xs[0] - 0.06 * span, xs[-1] + (0.24 if span else 0.4) * span
    for axis in panels:
        axis.set_xlim(left, right)
    nudge = 0.03 * span
    if has_levels:
        _label_ends(
            top,
            [(x + nudge, y, text, color) for x, y, text, color in ends],
            pad=_truth_pad(truth, len(drawn), gap),
        )
    if contrasts:
        _label_ends(
            diff,
            [(x + nudge, y, t, c) for x, y, t, c in diff_ends],
            pad=_truth_pad(truth, len(contrasts), gap),
        )
    _count_row(count_ax, result, rows, chosen, name_of, time_label)
    heading = {
        "risk": "Risk over time",
        "survival": "Survival over time",
        "mean": "Mean outcome over time",
    }[view]
    _truth.draw(truth, spots, known=list(result.estimates))
    if not has_levels:
        heading = {
            "risk": "Difference in risk over time",
            "survival": "Difference in survival over time",
            "mean": "Difference in mean outcome over time",
        }[view]
    suffix = f" ({chosen_cause})" if chosen_cause else ""
    _core.header(
        top,
        title or heading,
        _over_time_subtitle(view, suffix, contrasts, levels=has_levels),
        key_sources=panels[:-1],
    )
    note_list = []
    if not has_levels:
        note_list.append(
            "The fit estimates only the differences between regimens, so no regimen's own "
            "curve is drawn."
        )
    note_list += band_notes
    note_list += _withheld_notes([r.estimate for r in in_play])
    gaps = sorted({r.x for r in in_play if r.estimate.supplies_inference and r.lower is None})
    if gaps:
        note_list.append(
            "The fit reports no finite interval at time "
            + ", ".join(f"{x:g}" for x in gaps)
            + ", so none is drawn there and the marker is hollow."
        )
    note_list += _support_notes([r.estimate for r in in_play], "curve")
    if band_obj is not None:
        note_list += _band_note(band_obj.support, "simultaneous band")
        if view == "survival":
            note_list.append(
                "The survival band mirrors the fit's risk band about the survival estimate."
            )
    if step:
        note_list.append(_step_note(any(r.lower is not None for r in in_play), limits is not None))
    else:
        horizons = ", ".join(f"{x:g}" for x in sorted({r.x for r in in_play}))
        note_list.append(
            f"This fit reports only the horizons {horizons}, so the curve is drawn at those "
            "horizons and nothing is drawn between them."
        )
    if marked:
        note_list.append(
            "Hollow markers without an interval are the raw estimates the monotone projection "
            "moved; the drawn curve is the projected one."
        )
    elif result.isotonic_moves and result.n_repeats > 1 and view != "mean":
        note_list.append(
            "The monotone projection moved some estimates. The raw values are not marked, "
            f"because they belong to one of the {result.n_repeats} repeats this curve combines."
        )
    _core.add_notes(count_ax, note_list)
    return fig


def _plan_slots(contrasts: Sequence[str], sides: Sequence[str]) -> dict[str, int]:
    """The palette slot of each difference of a fit of differences alone: its plan's slot.

    The plan is the side named first (``"always"`` in ``"always vs never"``) and keeps its
    place in the fit's regimen order. Differences that share a plan, or more plans than
    slots, are numbered in turn instead, so no two differences share a colour.
    """
    plans = [contrast.split(" vs ", 1)[0] for contrast in contrasts]
    if len(set(plans)) == len(plans):
        slots = _slots(plans, sides)
        if len(set(slots.values())) == len(plans) and max(slots.values()) < _core.MAX_SERIES:
            return {c: slots[p] for c, p in zip(contrasts, plans, strict=True)}
    return {contrast: i for i, contrast in enumerate(contrasts)}


def _drawn_plan_slots(contrasts: Sequence[str], slot: Mapping[str, int]) -> dict[str, int] | None:
    """Each difference's plan's slot among the drawn curves, or ``None`` to draw them grey.

    ``None`` when a plan is not drawn as a curve or two differences share one.
    """
    plans = [contrast.split(" vs ", 1)[0] for contrast in contrasts]
    if len(set(plans)) != len(plans) or any(plan not in slot for plan in plans):
        return None
    return {c: slot[p] for c, p in zip(contrasts, plans, strict=True)}


def _truth_pad(truth: Any, count: int, gap: float) -> float:
    """Points the end labels move right so they clear the last node's crosses."""
    if truth is None:
        return 0.0
    rightmost = (count - 1) / 2.0 * gap * 72.0
    return rightmost + _TRUTH_NUDGE + 4.0


def _step_note(interval: bool, band: bool) -> str:
    """What a step curve's last time point shows, where its ribbons have no width."""
    text = "Each estimate holds until the next time point."
    if interval and band:
        return f"{text} At the last one the interval is drawn as a bar and the band as brackets."
    if interval:
        return f"{text} At the last one the interval is drawn as a bar."
    if band:
        return f"{text} At the last one the band is drawn as brackets."
    return text


def _over_time_subtitle(
    view: str, suffix: str, contrasts: list[str], *, levels: bool = True
) -> str:
    if not levels:
        base = {
            "risk": "Difference in cumulative risk between regimens",
            "survival": "Difference in the chance of staying event-free between regimens",
            "mean": "Difference in mean outcome between regimens",
        }[view]
        return f"{base}{suffix}"
    base = {
        "risk": "Cumulative risk under each regimen",
        "survival": "Chance of staying event-free under each regimen",
        "mean": "Mean outcome under each regimen",
    }[view]
    tail = ", and their difference below" if contrasts else ""
    return f"{base}{suffix}{tail}"


def _measure_word(
    view: str, outcome: str | None, units: str | None, *, difference: bool
) -> tuple[str, str]:
    """The y label of an over-time panel, as the prefix and the quantity it names."""
    if view == "mean":
        return _core.quantity(outcome, "outcome", units, difference=difference)
    word = "risk" if view == "risk" else "survival"
    if difference:
        return "Difference in", word
    return "", "Cumulative risk" if view == "risk" else "Survival probability"


def _mark_isotonic(
    ax: Axes,
    result: Any,
    rows: list[_Point],
    chosen: Sequence[str],
    slot: Mapping[str, int],
    view: str,
    psi_risk: Mapping[str, float],
    cause: str | None,
) -> bool:
    """Hollow markers at the raw estimates the isotonic projection moved (rule 6).

    Returns whether any marker was drawn. A fit of several repeats records the moves of one
    repeat, whose raw values are not the combined curve's, so none is drawn for it.
    """
    if not result.isotonic_moves or view == "mean" or result.n_repeats > 1:
        return False
    index = result.parameter_index or {}
    lookup = {index[r.parameter]: r for r in rows if r.level}
    scaler = result.scaler
    marked = False
    for name in chosen:
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
                marker=_MARKERS[slot[name] % len(_MARKERS)],
                markersize=8,
                color=_core.series_color(slot[name]),
                markerfacecolor=_core.FACE,
                markeredgecolor=_core.series_color(slot[name]),
                markeredgewidth=1.6,
                linestyle="none",
                gid="cleverly-raw",
                zorder=4,
            )
            marked = True
    return marked


def _count_row(
    ax: Axes,
    result: Any,
    rows: list[_Point],
    chosen: Sequence[str],
    name_of: Any,
    time_label: str | None,
) -> None:
    """The number following each regimen and still observed, under the time axis.

    The time points are those of the levels, or of the contrasts of a fit that reports no
    level.
    """
    has_levels = any(row.level for row in rows)
    timed = [row for row in rows if row.level == has_levels]
    nodes: dict[int, float] = {}
    for row in timed:
        if not has_levels or row.regimen in chosen:
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
    ticks = sorted({row.x for row in timed})
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
    truth: Any,
    title: str | None,
) -> Figure:
    grid = result.data.time_grid
    n_times = int(result.data.n_times)
    horizons: list[Any] = (
        list(range(2, n_times + 2)) if grid is None else [float(value) for value in grid[1:]]
    )
    found = _regimen_order(result)
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
    slot = _slots(chosen, found)
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
    spans: dict[Any, list[float]] = {panel: [] for panel in panels}
    keyed = {"pointwise": False}
    spots: list[_truth.Spot] = []

    def draw(
        ax: Axes,
        series: list[Any],
        color: str,
        style: int,
        dodge: tuple[int, int],
        name: str,
    ) -> None:
        xs = [float(h) for h in horizons]
        intervals = [_core.estimate_interval(e) for e in series]
        ys = [i.psi for i in intervals]
        shift = _offset(ax, *dodge)
        spots.extend(_truth.Spot(ax, None, x, shift, name) for x in xs)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.0,
            alpha=0.7,
            transform=shift,
            gid="cleverly-guide",
            zorder=2,
        )
        has = [i.drawn for i in intervals]
        if any(has):
            label = (
                "_nolegend_" if keyed["pointwise"] else _core.pointwise_label(intervals[0].level)
            )
            keyed["pointwise"] = True
            _dodged_brackets(
                ax,
                xs,
                [math.nan if i.lower is None else i.lower for i in intervals],
                [math.nan if i.upper is None else i.upper for i in intervals],
                color=color,
                label=label,
                band=False,
                transform=shift,
            )
        _draw_points(
            ax, xs, ys, has, color=color, marker=_MARKERS[style % 3], transform=shift, size=6.5
        )
        spans[ax].extend(
            value
            for interval in intervals
            for value in (interval.psi, interval.lower, interval.upper)
            if value is not None
        )

    for i, name in enumerate(chosen):
        color = _core.series_color(slot[name])
        draw(top, levels[name], color, slot[name], (i, len(chosen)), name)
        all_estimates.extend(levels[name])
        ends.append((float(horizons[-1]), levels[name][-1].psi, name_of(name), color))
    top.set_ylabel(_wrap_axis_label("", f"Restricted mean {word}{unit}"))
    diff_ends = []
    if contrasts:
        diff = panels[1]
        for i, name in enumerate(versus):
            draw(diff, contrasts[name], _core.SECONDARY_INK, i, (i, len(versus)), f"{name} vs")
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
        diff.set_ylabel(_wrap_axis_label("Difference in", f"restricted mean {word}{unit}"))
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
    _truth.draw(truth, spots)
    heading = (
        "Restricted mean survival up to each horizon"
        if measure == "rmst"
        else "Restricted mean time lost up to each horizon"
    )
    _core.header(
        top,
        title or heading,
        f"Average {word} accumulated before each restriction time, under each regimen",
        key_sources=panels,
    )
    notes.extend(_withheld_notes(all_estimates))
    notes.extend(_support_notes(all_estimates, "estimate"))
    notes.append(
        "Each point is a separate restricted mean with its own pointwise interval; the fit "
        "computes no band across horizons, and the thin line only joins the points."
    )
    _core.add_notes(panels[-1], notes)
    return fig


# ----------------------------------------------------------------------- shared point plots


def _numeric(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        from .._refusal import RefusalReason
        from ..exceptions import CapabilityError

        raise CapabilityError(
            f"{what} {value!r} is not a number, so it has no place on a numeric axis",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="pass levels= to give each regime or arm its numeric treatment level",
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
    guide_through: Sequence[tuple[float, float]] = (),
    spots: list[_truth.Spot] | None = None,
    names: Sequence[str | None] | None = None,
) -> list[str]:
    """Points with pointwise error bars and, where given, band brackets; returns the notes.

    The guide line, the bars, the brackets and the points share one sideways shift, so the
    line runs through the markers. ``guide_through`` adds points the guide passes through
    that are drawn elsewhere (a reference policy).
    """
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
    if guide:
        joined = sorted([*zip(xs, ys, strict=True), *guide_through])
        ax.plot(
            [x for x, _ in joined],
            [y for _, y in joined],
            color=color,
            linewidth=1.0,
            alpha=0.7,
            transform=shift,
            gid="cleverly-guide",
            zorder=2,
        )
    low = [math.nan if lo is None else float(lo) for lo, _ in pairs]
    high = [math.nan if hi is None else float(hi) for _, hi in pairs]
    has = [_finite_pair(lo, hi) for lo, hi in zip(low, high, strict=True)]
    banded = box is not None and any(pair is not None for pair in box)
    if any(has):
        # Beside a band's caps the pointwise bar moves left, so neither hides the other nor
        # the marker between them.
        bar_shift = _nudged(ax, shift, -_BAND_SPLIT) if banded else shift
        _dodged_brackets(
            ax, xs, low, high, color=color, label=interval_text, band=False, transform=bar_shift
        )
    if banded and box is not None:
        _dodged_brackets(
            ax,
            xs,
            [math.nan if pair is None else float(pair[0]) for pair in box],
            [math.nan if pair is None else float(pair[1]) for pair in box],
            color=color,
            label=band_text or "_nolegend_",
            band=True,
            transform=_nudged(ax, shift, _BAND_SPLIT) if any(has) else shift,
        )
    _draw_points(ax, xs, ys, has, color=color, marker=marker, transform=shift)
    if spots is not None:
        named = list(names) if names is not None else [None] * len(xs)
        spots.extend(
            _truth.Spot(ax, name, x, shift, marker) for name, x in zip(named, xs, strict=True)
        )
    return found


# ----------------------------------------------------------------------- V4: means by level


def _level_means(keys: Any) -> bool:
    """Whether the keys of a fit on the treatment-arm axis are each a per-level mean.

    A fit of ``CounterfactualMean`` reports ``ey[level]`` on the ``"arm"`` axis; an average
    effect on the same axis is a contrast against a reference and is not a mean.
    """
    found = list(keys)
    return bool(found) and all(
        key.estimand == "ey"
        and key.reference is None
        and key.stratum is None
        and key.value is not None
        for key in found
    )


def _regime_levels(means: Any, levels: Mapping[str, Any] | None) -> dict[str, Any]:
    found: dict[str, Any] = {}
    estimand: Any = getattr(getattr(means, "identified_effect", None), "estimand", None)
    for regimen in getattr(estimand, "regimens", ()):
        if hasattr(regimen, "level") and getattr(regimen, "name", None):
            found[str(regimen.name)] = regimen.level
    if levels:
        found.update({str(key): value for key, value in levels.items()})
    return found


def _check_levels(means: Any, levels: Mapping[str, Any] | None) -> None:
    """Refuse a ``levels=`` that names other regimes or arms, or only some of them.

    A key that matches nothing would be passed over in silence, and a partial mapping would
    draw some means at mapped places and the rest at their own labels.
    """
    if levels is None:
        return
    names = list(dict.fromkeys(str(key.value) for key in means.parameter_keys.values()))
    given = [str(key) for key in levels]
    unknown = [name for name in given if name not in names]
    missing = [name for name in names if name not in given]
    if unknown or missing:
        problem = []
        if unknown:
            problem.append(f"names {unknown}, which this fit does not report")
        if missing:
            problem.append(f"leaves out {missing}")
        raise _truth.refuse(
            f"levels= {' and '.join(problem)}; the fit reports {names}",
            "pass levels= with one numeric level for each of them",
        )


def _model_levels(labels: Sequence[Any], levels: Mapping[str, Any] | None) -> list[float]:
    """The working model's levels on the numeric axis.

    ``levels=`` maps a level the way it maps the per-level means, so a model fitted at
    labelled arms (``"low"``, ``"high"``) lands at the same places as their means.
    """
    mapped = {str(key): value for key, value in (levels or {}).items()}
    return [
        _numeric(mapped.get(str(label), label), "the working model's level") for label in labels
    ]


def _same_data(first: Any, second: Any) -> bool:
    """Whether two point-treatment fits read the same outcome and treatment columns."""
    for name in ("outcome", "treatment"):
        left = np.asarray(getattr(first, name))
        right = np.asarray(getattr(second, name))
        if left.shape != right.shape:
            return False
        try:
            same = np.array_equal(left.astype(float), right.astype(float), equal_nan=True)
        except (TypeError, ValueError):
            # Labelled arms: compared as they are.
            same = bool(np.all(left == right))
        if not same:
            return False
    return True


def _level_title(means: Any, msm_fit: Any) -> str:
    """The question of a per-level means plot, in dose words only for a declared dose.

    Per-level means compare arms, contact counts or doses alike, so the words are neutral
    unless a fit declares its treatment continuous.
    """
    fits = (means, msm_fit)
    dose = any(
        bool(getattr(getattr(fit, "data", None), "is_continuous_treatment", False)) for fit in fits
    )
    return f"Mean outcome by {'dose' if dose else 'treatment level'}"


def _dose(
    means: Any,
    msm_fit: Any,
    *,
    band: str | None,
    levels: Mapping[str, Any] | None,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    truth: Any,
    title: str | None,
    ax: Axes | None,
) -> Figure:
    from ..msm import working_model_means

    _check_levels(means, levels)
    level_of = _regime_levels(means, levels)
    entries = []
    for alias, key in means.parameter_keys.items():
        name = str(key.value)
        if name not in level_of and key.axis == "arm":
            # A per-level mean is keyed by the treatment value itself.
            level_of[name] = key.value
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
    fitted = None
    if msm_fit is not None:
        fitted = working_model_means(msm_fit)
        mx = _model_levels(fitted.levels, levels)
        missing = [x for x in mx if not any(math.isclose(x, level) for level in xs)]
        if missing:
            raise ValueError(
                f"the working model is fitted at the levels {mx}, and the per-level means "
                f"report no mean at {missing}; fit both on the same data and levels"
            )
        if not _same_data(means.data, msm_fit.data):
            raise ValueError(
                "the working model and the per-level means were fitted on different data "
                "(their outcome or treatment columns differ); fit both on the same data"
            )
    fig, axis = _core.figure_for(ax, figsize=(7.8, 4.8))
    outcome = str(getattr(means.data, "outcome_name", "outcome"))
    treatment = treatment_label or str(getattr(means.data, "treatment_name", "treatment"))
    level = 1.0 - float(estimates[0].alpha)
    bands = _point_band(means.simultaneous, band)
    box = None
    if bands is not None:
        box = [bands.bands.get(e[2]) for e in entries]
    count = 2 if fitted is not None else 1
    extent: list[float] = []
    means_color = _core.series_color(0)
    spots: list[_truth.Spot] = []
    notes = _points_at(
        axis,
        xs,
        estimates,
        color=means_color,
        marker="o",
        index=0,
        count=count,
        interval_text=(
            _core.interval_label(level)
            if bands is None and fitted is None
            else _core.pointwise_label(level)
        ),
        box=box,
        band_text=None if bands is None else _core.band_label(1.0 - float(bands.alpha)),
        extent=extent,
        spots=spots,
        names=[e[2] for e in entries],
    )
    ends = [(xs[-1], float(estimates[-1].psi), "Mean at each level", means_color)]
    if bands is not None:
        notes += _band_note(bands.support, "simultaneous band")
    subtitle = "Mean outcome if everyone were given each level"
    if fitted is not None:
        model_color = _core.series_color(1)
        pairs: list[tuple[float | None, float | None]] = (
            [(None, None)] * len(mx)
            if fitted.lower is None or fitted.upper is None
            else [(float(lo), float(hi)) for lo, hi in zip(fitted.lower, fitted.upper, strict=True)]
        )
        notes += _points_at(
            axis,
            mx,
            [float(v) for v in fitted.estimate],
            color=model_color,
            marker="D",
            index=1,
            count=2,
            # The key lists interval kinds and the end labels name the series, so the
            # model's interval shares the means' entry when the two levels agree.
            interval_text=(
                "_nolegend_"
                if math.isclose(fitted.alpha, float(estimates[0].alpha))
                else _core.pointwise_label(1.0 - fitted.alpha)
            ),
            intervals=pairs,
            extent=extent,
        )
        last = int(np.argmax(mx))
        ends.append((mx[last], float(fitted.estimate[last]), "Working model", model_color))
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
    span = (max(xs) - min(xs)) or 1.0
    axis.set_xlim(min(xs) - 0.08 * span, max(xs) + 0.42 * span)
    _pad(axis, extent)
    _label_ends(axis, [(x + 0.05 * span, y, t, c) for x, y, t, c in ends])
    axis.set_xlabel(f"{treatment}")
    axis.set_ylabel(
        " ".join(_core.quantity(outcome_label, outcome, units, difference=False)).strip()
    )
    _truth.draw(truth, spots, known=list(means.estimates))
    _core.header(axis, title or _level_title(means, msm_fit), subtitle)
    _core.add_notes(axis, notes)
    return fig


def _dose_response(
    result: Any,
    *,
    band: str | None,
    by: str | None,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    truth: Any,
    title: str | None,
    ax: Axes | None,
) -> Figure:
    """The smoothed dose-response curve: one series per bandwidth, the band as brackets."""
    strata = sorted({cell.stratum for cell in result.pair_support if cell.stratum is not None})
    if strata and by is None:
        raise ValueError(f"this fit has baseline strata; pass by= one of {strata}")
    if by is not None and by not in strata:
        raise ValueError(f"by= must be one of the fit's strata {strata}; got {by!r}")
    bandwidths = list(result.curve.bandwidths)
    if len(bandwidths) > len(_MARKERS):
        raise ValueError(
            f"this fit declares {len(bandwidths)} bandwidths and one axes draws at most "
            f"{len(_MARKERS)}; fit a declaration with fewer bandwidths to draw it"
        )
    if band not in (None, "multiplier", "none"):
        raise ValueError(f"a dose-response curve holds a multiplier band at most; got {band!r}")
    fig, axis = _core.figure_for(ax, figsize=(7.8, 4.8))
    level = 1.0 - float(result.alpha)
    draw_band = band != "none"
    extent: list[float] = []
    spots: list[_truth.Spot] = []
    notes: list[str] = []
    ends = []
    xs_all: list[float] = []
    for index, h in enumerate(bandwidths):
        cells = [c for c in result.pair_support if c.bandwidth == h and c.stratum == by]
        xs = [c.dose for c in cells]
        xs_all.extend(xs)
        estimates = [result.estimates[c.name] for c in cells]
        box = [result.band.bands[c.name] for c in cells] if draw_band else None
        color = _core.series_color(index)
        found = _points_at(
            axis,
            xs,
            estimates,
            color=color,
            marker=_MARKERS[index],
            index=index,
            count=len(bandwidths),
            interval_text=_core.pointwise_label(level) if index == 0 else "_nolegend_",
            box=box,
            band_text=_core.band_label(level) if index == 0 else None,
            extent=extent,
            spots=spots,
            names=[c.name for c in cells],
        )
        notes += [note for note in found if note not in notes]
        ends.append((xs[-1], float(estimates[-1].psi), f"bandwidth {h:g}", color))
    span = (max(xs_all) - min(xs_all)) or 1.0
    axis.set_xticks(sorted(set(xs_all)))
    axis.set_xlim(min(xs_all) - 0.08 * span, max(xs_all) + 0.42 * span)
    _pad(axis, extent)
    _label_ends(axis, [(x + 0.05 * span, y, t, c) for x, y, t, c in ends])
    axis.set_xlabel(treatment_label or "Dose")
    axis.set_ylabel(
        " ".join(_core.quantity(outcome_label, "outcome", units, difference=False)).strip()
    )
    low = [c.name for c in result.pair_support if c.below_floor and c.stratum == by]
    if low:
        notes.append(
            f"{len(low)} point(s) rest on few nearby rows; their intervals carry the "
            "low-support label."
        )
    notes.append(
        "Each point is the curve averaged over a window of doses around it (a kernel of the "
        "named bandwidth), not the mean at that exact dose. The band holds for every drawn "
        "point at once, so it also holds at a bandwidth chosen after looking at this figure; "
        "a pointwise interval does not."
    )
    _truth.draw(truth, spots, known=list(result.estimates))
    subtitle = "Mean outcome if everyone were given each dose, smoothed over nearby doses"
    if by is not None:
        subtitle += f", within {by}"
    _core.header(axis, title or "Smoothed dose-response curve", subtitle)
    _core.add_notes(axis, notes)
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
    band: str | None,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    truth: Any,
    title: str | None,
    ax: Axes | None,
) -> Figure:
    from ._forest import forest

    estimand = getattr(getattr(result, "identified_effect", None), "estimand", None)
    axis_spec = _policy_axis(estimand)
    if axis_spec is None:
        # No single scalar parameter orders these policies, so there is no axis to draw them
        # against: they are separate estimates, and a forest plot shows them as such.
        return forest(
            result,
            title=title,
            subtitle="No single parameter orders these policies, so each is drawn alone",
            units=units,
            truth=truth,
            ax=ax,
        )
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
    outcome = str(getattr(result.data, "outcome_name", "outcome"))
    treatment = treatment_label or str(getattr(result.data, "treatment_name", "treatment"))
    level = 1.0 - float(estimates[0].alpha)
    bands = _point_band(result.simultaneous, band)
    box = None if bands is None else [bands.bands.get(e[2]) for e in entries]
    extent: list[float] = [0.0] if contrast else []
    color = _core.series_color(0)
    spots: list[_truth.Spot] = []
    # The effect form reports each policy against the reference, whose own effect is zero
    # by definition and is not an estimate: it is drawn on the null line with no interval,
    # and the guide runs through it.
    anchor = None
    if contrast and reference is not None and str(reference) in values:
        anchor = (values[str(reference)], 0.0)
    notes = _points_at(
        axis,
        xs,
        estimates,
        color=color,
        marker="o",
        index=0,
        count=1,
        interval_text=(
            _core.interval_label(level) if bands is None else _core.pointwise_label(level)
        ),
        box=box,
        band_text=None if bands is None else _core.band_label(1.0 - float(bands.alpha)),
        extent=extent,
        guide_through=() if anchor is None else (anchor,),
        spots=spots,
        names=[e[2] for e in entries],
    )
    if anchor is not None:
        axis.plot(
            [anchor[0]],
            [anchor[1]],
            marker="o",
            markersize=7,
            color=color,
            markerfacecolor=_core.FACE,
            markeredgewidth=1.5,
            linestyle="none",
            gid="cleverly-reference-policy",
            zorder=3,
        )
        axis.annotate(
            "reference",
            xy=anchor,
            xytext=(6.0, -6.0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=_core.TICK_SIZE,
            color=_core.SECONDARY_INK,
            gid="cleverly-reference-policy-label",
        )
        notes.append(
            f"The open marker at {anchor[0]:g} is the reference policy {reference}: its "
            "effect is zero by definition, so it has no interval."
        )
    elif contrast and reference is not None:
        notes.append(
            f"The reference policy {reference} is not one of the declared policies, so it "
            "has no place on this axis."
        )
    shown = [*xs, *(() if anchor is None else (anchor[0],))]
    if not log:
        span = (max(shown) - min(shown)) or 1.0
        axis.set_xlim(min(shown) - 0.08 * span, max(shown) + 0.08 * span)
    _pad(axis, extent)
    if anchor is not None:
        # Room for the reference label under the marker.
        low, high = axis.get_ylim()
        axis.set_ylim(min(low, anchor[1] - 0.12 * (high - low)), high)
    if bands is not None:
        notes += _band_note(bands.support, "simultaneous band")
        notes.append(
            f"The {_core.level_text(1.0 - float(bands.alpha))} simultaneous band covers the "
            "policies of this one fit together; it is marked at those policies only."
        )
    if contrast:
        _core.reference_line(axis, 0.0, label="No effect")
        axis.set_ylabel(" ".join(_core.quantity(outcome_label, outcome, units, difference=True)))
        subtitle = f"Each policy compared with {reference}{cap_note}"
        heading = "Effect by policy"
    else:
        axis.set_ylabel(
            " ".join(_core.quantity(outcome_label, outcome, units, difference=False)).strip()
        )
        subtitle = f"Mean outcome under each policy{cap_note}"
        heading = "Mean outcome by policy"
    ticks = sorted(set(shown))
    if log:
        _core.log_axis(axis, "x")
        axis.set_xticks(ticks, [f"{x:g}" for x in ticks])
        axis.xaxis.set_minor_formatter(_null_formatter())
    else:
        axis.set_xticks(ticks)
    if log:
        axis_words = axis_words.replace("odds of treatment", f"odds of {treatment}")
    axis.set_xlabel(f"{axis_words} ({treatment})" if not log else axis_words)
    _truth.draw(truth, spots, known=list(result.estimates))
    _core.header(axis, title or heading, subtitle)
    _core.add_notes(axis, notes)
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
    band_choice: str | None,
    modifier: str | None,
    by: str | None,
    at: Mapping[str, Any] | None,
    alpha: float | None,
    outcome_label: str | None,
    treatment_label: str | None,
    units: str | None,
    truth: Any,
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
    if truth is not None and not callable(truth):
        raise _truth.refuse(
            "truth= for a conditional effect curve is a function of the modifier, because "
            "the curve is drawn over a grid of its values and not at a set of levels",
            "pass truth=lambda x: <the true effect at modifier values x>",
        )
    if truth is not None and by is not None:
        raise _truth.refuse(
            "truth= draws one true curve, and by= draws one panel per level, each with its "
            "own true curve",
            f"draw one level at a time with at={{{by!r}: <level>}}",
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
    band = _point_band(result.band, band_choice)
    level_alpha = _curve_alpha(result) if alpha is None else float(alpha)

    def evaluate(level: Any) -> dict[str, np.ndarray]:
        frame: dict[str, Any] = {modifier: grid}
        for name, value in held.items():
            if name in read:
                frame[name] = np.full(grid.size, value)
        if by is not None:
            frame[by] = np.full(grid.size, level)
        found: dict[str, np.ndarray] = {}
        if projection:
            table = result.predict(frame, interval=True, alpha=level_alpha)
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
    # The caller's words go mid-phrase, after "Effect on"; a column name stays as written.
    outcome = (
        _core.mid_phrase(outcome_label)
        if outcome_label
        else str(getattr(getattr(result.fit, "data", None), "outcome_name", "outcome"))
    )
    unit = f" ({units})" if units else ""
    binary = _binary_outcome(result)
    average, average_note = _average_effect(result, binary, several_panels=rows > 1)
    notes: list[str] = []
    color = _core.series_color(0)
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
        if drawn_interval:
            _core.ribbon(
                panel,
                grid.tolist(),
                np.asarray(transform(data["lower"])).tolist(),
                np.asarray(transform(data["upper"])).tolist(),
                color=color,
                label=_core.pointwise_label(1.0 - level_alpha),
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
        if average is not None:
            panel.axhline(
                average,
                color=_core.SECONDARY_INK,
                linewidth=1.0,
                linestyle=(0, (1, 2)),
                label="Average effect (all rows)" if rows > 1 else "Average effect",
                gid="cleverly-average",
                zorder=1.5,
            )
        if band is not None:
            _band_marks(panel, band, modifier, by, level, held, transform, color)
        _rug(panel, result, modifier, by, level, mods)
        if by is not None:
            _core.corner_label(panel, 0.97, f"{by} = {level}", color)
        if ratio:
            panel.set_ylabel(f"Conditional {_ratio_word(binary)} (log scale)")
        else:
            panel.set_ylabel(f"Effect on {outcome}{unit}")
        if ratio:
            _core.log_axis(panel, "y")
    if truth is not None:
        try:
            true_values = np.asarray(truth(grid.copy()), dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise _truth.refuse(
                f"truth= could not be evaluated at the modifier's values: {error}",
                "pass a function that takes an array of modifier values and returns one "
                "number per value",
            ) from None
        if true_values.size != grid.size or not np.all(np.isfinite(true_values)):
            raise _truth.refuse(
                f"truth= returned {true_values.size} values for {grid.size} modifier values, "
                "or a value that is not finite",
                "pass a function that returns one finite number per modifier value",
            )
        _truth.check_log(panels[0], true_values.tolist())
        _core.truth_line(panels[0], grid.tolist(), true_values.tolist())
    if len(panels) > 1:
        shared = [panel.get_ylim() for panel in panels]
        for panel in panels:
            panel.set_ylim(min(low for low, _ in shared), max(high for _, high in shared))
    panels[-1].set_xlabel(modifier)
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
    notes.append(average_note)
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
        _effect_subtitle(result, ratio, binary),
        key_sources=[panels[0]],
    )
    _core.add_notes(panels[-1], notes)
    return fig


def _curve_alpha(result: Any) -> float:
    """The level the pointwise interval is drawn at when ``alpha=`` is omitted.

    The band's own level when the curve has a band, so the two are read at one level, else
    the level of the coefficients the curve is built from.
    """
    if result.band is not None:
        return float(result.band.alpha)
    estimates = getattr(getattr(result, "fit", None), "estimates", None) or {}
    for estimate in getattr(estimates, "values", lambda: ())():
        return float(estimate.alpha)
    return 0.05


def _binary_outcome(result: Any) -> bool:
    from .._summary._data import binary_outcome

    data = getattr(getattr(result, "fit", None), "data", None)
    outcome = getattr(data, "outcome", None)
    return outcome is not None and binary_outcome(outcome)


def _contrast_arms(result: Any) -> str | None:
    """``"t vs r"`` for a curve that contrasts two arms of a treatment with more than two."""
    fit: Any = getattr(result, "fit", None)
    contrast = getattr(getattr(fit, "nuisance", None), "effect_modification", None)
    arms = tuple(getattr(contrast, "arms", ()) or ())
    pair = getattr(contrast, "pair", None)
    if len(arms) <= 2 or pair is None:
        return None
    label = getattr(fit.data, "arm_label", str)
    return f"{label(pair[0])} vs {label(pair[1])}"


def _ratio_word(binary: bool) -> str:
    """A ratio of two arm means: a risk ratio when the outcome is binary."""
    return "risk ratio" if binary else "ratio of means"


def _effect_subtitle(result: Any, ratio: bool, binary: bool) -> str:
    word = _ratio_word(binary) if ratio else "average treatment effect"
    method = "projected onto the declared model" if result.fitted_method != "ep" else "EP learner"
    arms = _contrast_arms(result)
    return f"Conditional {word}{f' of {arms}' if arms else ''}, {method}"


def _has_weights(result: Any) -> bool:
    data = getattr(getattr(result, "fit", None), "data", None)
    weights = getattr(data, "weights", None)
    if weights is None or np.asarray(weights).shape != np.asarray(result.in_sample).shape:
        return False
    return bool(np.ptp(np.asarray(weights, dtype=float)) > 0.0)


def _average_effect(result: Any, binary: bool, *, several_panels: bool) -> tuple[float | None, str]:
    """The average effect to draw as a reference line, or ``None``, and the note that says so.

    The curve holds no estimate of the average effect. The mean of the curve over the fitted
    rows equals the fit's plug-in average effect in one case only: a difference-scale
    projection whose model spans the constant, because the projection's normal equations
    then make the (weighted) mean of the fitted curve equal the mean of the per-row effect
    predictions it projects. Anywhere else the mean of the curve is another number, and no
    line is drawn.
    """
    if result.fitted_method == "ep":
        return None, (
            "No average-effect line is drawn: the EP learner's curve is not fitted so that its "
            "mean over the rows equals the average effect."
        )
    if result.scale != "difference":
        return None, (
            f"No average-effect line is drawn: the mean of a log-ratio curve is not the overall "
            f"{_ratio_word(binary)}, which is the ratio of the two average "
            f"{'risks' if binary else 'outcomes'}."
        )
    contrast = getattr(getattr(result.fit, "nuisance", None), "effect_modification", None)
    design = getattr(contrast, "design", None)
    values = np.asarray(result.in_sample, dtype=float)
    if design is None or np.asarray(design).shape[0] != values.size:
        return None, (
            "No average-effect line is drawn: the curve does not carry the model at its fitted "
            "rows, so it cannot show that its mean is the average effect."
        )
    phi = np.asarray(design, dtype=float)
    ones = np.ones(phi.shape[0])
    solution = np.linalg.lstsq(phi, ones, rcond=None)[0]
    if float(np.max(np.abs(phi @ solution - ones))) > 1e-8:
        return None, (
            "No average-effect line is drawn: the model has no constant term, so the mean of "
            "the curve over the fitted rows need not equal the average effect."
        )
    weighted = _has_weights(result)
    if weighted:
        value = float(np.average(values, weights=np.asarray(result.fit.data.weights, dtype=float)))
    else:
        value = float(np.mean(values))
    rows = "all fitted rows" if several_panels else "the fitted rows"
    return value, (
        f"The dotted line is the mean of the curve over {rows}"
        + (", weighted by the data's weights" if weighted else "")
        + (", drawn alike in every panel" if several_panels else "")
        + ". Because the model includes a constant term, it equals the fit's plug-in average "
        "effect; no interval is drawn for it."
    )


def _declared_value(array: np.ndarray, wanted: float) -> float:
    """The grid value that ``wanted`` names: itself when declared, else one within rounding.

    Two declared values can be close in relative terms and still be different grid points,
    so no loose tolerance applies. A value typed in decimal that differs from a declared
    value only in the last bits (``0.3`` for ``0.1 + 0.2``) names it when exactly one
    declared value is that close; otherwise ``wanted`` names nothing.
    """
    values = np.unique(array.astype(float))
    if np.any(values == wanted):
        return wanted
    close = [v for v in values if math.isclose(v, wanted, rel_tol=1e-12, abs_tol=1e-12)]
    return float(close[0]) if len(close) == 1 else math.nan


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
                mask &= array.astype(float) == _declared_value(array, float(wanted))
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
    _core.brackets(
        ax,
        np.asarray(band.grid[modifier], dtype=float)[mask].tolist(),
        transform(np.asarray(band.lower, dtype=float)[mask]).tolist(),
        transform(np.asarray(band.upper, dtype=float)[mask]).tolist(),
        color=color,
        label=_core.band_label(1.0 - float(band.alpha)),
        band=True,
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
