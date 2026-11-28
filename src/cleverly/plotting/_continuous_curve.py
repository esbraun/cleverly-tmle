"""The targeted risks of a continuous-time survival fit, at its target times only.

:func:`~cleverly.plotting.curve` hands a
:class:`~cleverly.survival.ContinuousSurvivalResult` or a
:class:`~cleverly.survival.ContinuousLongitudinalResult` here. The risks and their difference are
read from ``result.curve(scale)``, with each interval and band limit as that frame reports
it, and the restricted means from ``result.rmst`` and ``result.rmtl``. A risk between two
target times solves no score equation of the fit, so nothing is drawn between target times:
each estimate is a marker with its interval bar and band brackets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from . import _core, _truth
from ._curves import (
    _DODGE,
    _TRUTH_DODGE,
    _band_note,
    _draw_line,
    _extent,
    _finite_pair,
    _label_ends,
    _pad,
    _Point,
    _support_notes,
    _truth_pad,
    _withheld_notes,
    _wrap_axis_label,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

    from ..survival.longitudinal_result import ContinuousLongitudinalResult
    from ..survival.result import ContinuousSurvivalResult

    #: The fits whose risks this module draws at their target times.
    _Fit: TypeAlias = ContinuousSurvivalResult | ContinuousLongitudinalResult


# --------------------------------------------------------------------------- readers


def curve_frame(result: _Fit, view: str) -> Any:
    """The fit's own curve on ``view``: the rows, intervals and band limits it reports."""
    return result.curve(scale=view)  # type: ignore[arg-type]


def _values(frame: Any, name: str) -> list[Any]:
    column = frame[name]
    return list(column.to_list() if hasattr(column, "to_list") else column)


def curve_rows(
    result: _Fit, view: str
) -> tuple[list[_Point], dict[str, str | None], dict[str, tuple[float, float]] | None]:
    """Each row of ``result.curve(view)`` as a point, its cause, and its band limits."""
    frame = curve_frame(result, view)
    names = list(frame.columns)
    parameters = [str(v) for v in _values(frame, "parameter")]
    arms = [str(v) for v in _values(frame, "arm")]
    times = [float(v) for v in _values(frame, "time")]
    psi = [float(v) for v in _values(frame, "psi")]
    low = _values(frame, "ci_lower")
    high = _values(frame, "ci_upper")
    scales = [str(v) for v in _values(frame, "scale")]
    causes: list[str | None] = (
        [str(v) for v in _values(frame, "cause")] if "cause" in names else [None] * len(parameters)
    )
    banded = "band_lower" in names
    band_low = _values(frame, "band_lower") if banded else [None] * len(parameters)
    band_high = _values(frame, "band_upper") if banded else [None] * len(parameters)
    target = {t: k for k, t in enumerate(result.horizons)}
    rows: list[_Point] = []
    cause_of: dict[str, str | None] = {}
    limits: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(parameters):
        estimate = result.estimates[name]
        drawn = estimate.supplies_inference and _finite_pair(low[i], high[i])
        rows.append(
            _Point(
                name,
                arms[i],
                times[i],
                target[times[i]],
                psi[i],
                float(low[i]) if drawn else None,
                float(high[i]) if drawn else None,
                scales[i] == "level",
                estimate,
            )
        )
        cause_of[name] = causes[i]
        if banded and _finite_pair(band_low[i], band_high[i]):
            limits[name] = (float(band_low[i]), float(band_high[i]))
    return rows, cause_of, (limits if banded else None)


def bootstrap_limits(
    result: _Fit, rows: Sequence[_Point], view: str
) -> dict[str, tuple[float, float]]:
    """The bootstrap band's limits at each row, mirrored on the survival view as
    ``result.curve(scale='survival')`` mirrors the multiplier band's."""
    bands = result.bootstrap_band.bands if result.bootstrap_band is not None else {}
    limits: dict[str, tuple[float, float]] = {}
    for row in rows:
        held = bands.get(row.parameter)
        if held is None or not _finite_pair(held[0], held[1]):
            continue
        low, high = float(held[0]), float(held[1])
        if view == "survival":
            low, high = (1.0 - high, 1.0 - low) if row.level else (-high, -low)
        limits[row.parameter] = (low, high)
    return limits


def restricted_rows(result: _Fit, measure: str) -> tuple[list[_Point], dict[str, str | None]]:
    """Each arm's restricted mean and their difference at every restricted horizon."""
    reference, treated = result.arms
    single = len(result.causes) == 1
    causes: list[str | None] = [None] if measure == "rmst" or single else list(result.causes)
    rows: list[_Point] = []
    cause_of: dict[str, str | None] = {}
    for cause in causes:
        for k, tau in enumerate(result.restricted_horizons):

            def read(
                arm: str, versus: str | None = None, tau: float = tau, cause: Any = cause
            ) -> Any:
                if measure == "rmst":
                    return result.rmst(arm, tau, versus=versus)
                return result.rmtl(arm, tau, cause, versus=versus)

            pairs = [(arm, read(arm), True) for arm in result.arms]
            pairs.append((f"{treated} vs {reference}", read(treated, reference), False))
            for label, estimate, level in pairs:
                low, high = estimate.ci
                drawn = estimate.supplies_inference and _finite_pair(low, high)
                rows.append(
                    _Point(
                        estimate.name,
                        label,
                        float(tau),
                        k,
                        float(estimate.psi),
                        float(low) if drawn else None,
                        float(high) if drawn else None,
                        level,
                        estimate,
                    )
                )
                cause_of[estimate.name] = cause if not single else None
    return rows, cause_of


# --------------------------------------------------------------------------- drawing


def _y_words(measure: str, view: str, several: bool, units: str | None) -> tuple[str, str]:
    if measure == "rmst":
        word = "Restricted mean survival time"
    elif measure == "rmtl":
        word = "Restricted mean time lost"
    elif view == "survival":
        word = "Survival probability"
    else:
        word = "Cumulative incidence" if several else "Cumulative risk"
    if measure != "curve" and units:
        word = f"{word} ({units})"
    return word, f"Difference in {word[:1].lower()}{word[1:]}"


def curve(
    result: _Fit,
    *,
    scale: str | None,
    measure: str,
    band: str | None,
    cause: str | None,
    regimens: Sequence[str] | None,
    labels: Mapping[str, str] | None,
    units: str | None,
    time_label: str | None,
    truth: Any,
    title: str | None,
) -> Figure:
    """Markers, interval bars and band brackets at the target times, levels above, the
    difference below, one column per cause."""
    from ..exceptions import DataError

    if measure not in ("curve", "rmst", "rmtl"):
        raise ValueError(f"measure must be 'curve', 'rmst' or 'rmtl'; got {measure!r}")
    if band not in (None, "multiplier", "bootstrap", "none"):
        raise ValueError(f"band must be 'multiplier', 'bootstrap' or 'none'; got {band!r}")
    if band == "bootstrap" and result.bootstrap_band is None:
        raise ValueError(
            "this fit holds no bootstrap band; fit it with n_bootstrap= and simultaneous=True, "
            "and a bootstrap whose refits more than 1% failed holds none"
        )
    if band == "multiplier" and result.simultaneous is None:
        raise ValueError("this fit holds no multiplier band; fit it with simultaneous=True")
    # The multiplier band is drawn unless band='bootstrap' asks for the bootstrap band.
    drawn_band: Any = result.bootstrap_band if band == "bootstrap" else result.simultaneous
    single = len(result.causes) == 1
    view = scale or "risk"
    limits: dict[str, tuple[float, float]] | None = None
    if measure == "curve":
        rows, cause_of, limits = curve_rows(result, view)
        if band == "none":
            limits = None
        elif band == "bootstrap":
            limits = bootstrap_limits(result, rows, view)
    else:
        if scale is not None:
            raise ValueError(
                f"scale= sets the view of measure='curve'; leave it out with measure={measure!r}"
            )
        if measure == "rmst" and cause is not None:
            raise ValueError("cause= does not apply to the restricted mean survival time")
        if not result.restricted_horizons:
            # The fit's own refusal, which names restricted_horizons=.
            arm = result.arms[0]
            tau = result.horizons[-1]
            if measure == "rmst":
                result.rmst(arm, tau)
            else:
                result.rmtl(arm, tau, None if single else result.causes[0])
        rows, cause_of = restricted_rows(result, measure)
    several = not single and measure != "rmst"
    held = list(dict.fromkeys(cause_of.values()))
    if cause is not None:
        if single or cause not in held:
            raise DataError(f"cause {cause!r} is not in this fit; it reports {list(result.causes)}")
        columns: list[str | None] = [cause]
    else:
        columns = held
    if len(columns) > _core.MAX_SERIES:
        raise ValueError(
            f"this fit has {len(columns)} causes and the figure draws at most "
            f"{_core.MAX_SERIES} columns; pass cause= to draw one"
        )
    order = list(result.arms)
    reference = order[0]
    chosen = order if regimens is None else [str(name) for name in regimens]
    unknown = [name for name in chosen if name not in order]
    if unknown or not chosen or len(set(chosen)) != len(chosen):
        raise DataError(f"regimens= names arms of this fit, {order}; got {list(chosen)}")
    # One difference per other arm against the reference (two arms on a baseline treatment,
    # one per regime on a longitudinal fit), drawn when both of its arms are chosen.
    compared = [arm for arm in order[1:] if arm in chosen] if reference in chosen else []
    contrasts = [f"{arm} vs {reference}" for arm in compared]
    slot = {name: order.index(name) for name in order}
    name_of = (lambda s: str(labels.get(s, s))) if labels else str
    in_play = [
        row
        for row in rows
        if cause_of[row.parameter] in columns
        and ((row.level and row.regimen in chosen) or (not row.level and row.regimen in contrasts))
    ]
    level = 1.0 - float(rows[0].estimate.alpha)
    band_text = None
    if limits is not None:
        band_text = _core.band_label(1.0 - float(drawn_band.alpha))
    ncols = len(columns)
    nrows = 2 if contrasts else 1
    heights = [3.2, 1.8][:nrows]
    fig, axes = _core.new_figure(
        nrows,
        ncols=ncols,
        figsize=(7.2 if ncols == 1 else 3.7 * ncols + 0.8, 1.6 + sum(heights)),
        height_ratios=heights,
    )
    first = {"band": True, "pointwise": True}
    spots: list[_truth.Spot] = []
    gap = _DODGE if truth is None else _TRUTH_DODGE
    level_word, diff_word = _y_words(measure, view, several, units)
    xs = sorted({row.x for row in in_play})
    span = (xs[-1] - xs[0]) or max(abs(xs[-1]), 1.0)
    left, right = xs[0] - 0.12 * span, xs[-1] + 0.3 * span
    for j, column in enumerate(columns):
        top = axes[j]
        mine = [row for row in in_play if cause_of[row.parameter] == column]
        ends: list[tuple[float, float, str, str]] = []
        for i, arm in enumerate(chosen):
            points = sorted((r for r in mine if r.level and r.regimen == arm), key=lambda r: r.x)
            color = _core.series_color(slot[arm])
            _draw_line(
                top,
                points,
                color=color,
                index=slot[arm],
                dodge=(i, len(chosen)),
                step=False,
                limits=limits,
                level=level,
                band_text=band_text,
                first=first,
                spots=spots,
                gap=gap,
            )
            ends.append((points[-1].x, points[-1].psi, name_of(arm), color))
        _pad(top, _extent(mine, chosen, limits, level_rows=True))
        if j == 0:
            top.set_ylabel(_wrap_axis_label("", level_word, 34))
        if ncols > 1:
            top.set_title(
                f"Cause {column}", loc="left", fontsize=_core.LABEL_SIZE, color=_core.INK, pad=4.0
            )
        panels = [top]
        diff_ends: list[tuple[float, float, str, str]] = []
        if contrasts:
            diff = axes[ncols + j]
            for i, (arm, contrast) in enumerate(zip(compared, contrasts, strict=True)):
                points = sorted(
                    (r for r in mine if not r.level and r.regimen == contrast), key=lambda r: r.x
                )
                color = _core.series_color(slot[arm])
                _draw_line(
                    diff,
                    points,
                    color=color,
                    index=slot[arm],
                    dodge=(i, len(contrasts)),
                    step=False,
                    limits=limits,
                    level=level,
                    band_text=band_text,
                    first=first,
                    spots=spots,
                    gap=gap,
                )
                diff_ends.append((points[-1].x, points[-1].psi, name_of(contrast), color))
            _core.reference_line(diff, 0.0, label="No difference" if j == 0 else "_nolegend_")
            _pad(diff, [0.0, *_extent(mine, contrasts, limits, level_rows=False)])
            if j == 0:
                diff.set_ylabel(_wrap_axis_label("", diff_word, 26))
            panels.append(diff)
        for axis in panels:
            axis.set_xlim(left, right)
            axis.set_xticks(xs)
            axis.set_xticklabels([f"{x:g}" for x in xs])
        panels[-1].set_xlabel(time_label or "Time")
        nudge = 0.03 * span
        if j == ncols - 1:
            _label_ends(
                top,
                [(x + nudge, y, text, c) for x, y, text, c in ends],
                pad=_truth_pad(truth, len(chosen), gap),
            )
            if contrasts:
                _label_ends(
                    axes[ncols + j],
                    [(x + nudge, y, text, c) for x, y, text, c in diff_ends],
                    pad=_truth_pad(truth, len(contrasts), gap),
                )
    _truth.draw(truth, spots, known=list(result.estimates))
    heading = {
        "rmst": "Restricted mean survival time",
        "rmtl": "Restricted mean time lost",
    }.get(measure) or (
        "Survival at the target times" if view == "survival" else "Risk at the target times"
    )
    what = {
        "rmst": "Each arm's restricted mean survival time at each restricted horizon",
        "rmtl": "Each arm's restricted mean time lost at each restricted horizon",
    }.get(measure) or (
        "Each arm's chance of staying event-free at each target time"
        if view == "survival"
        else "Each arm's cumulative "
        + ("incidence of each cause" if several else "risk")
        + " at each target time"
    )
    subtitle = what + (f" (cause {columns[0]})" if several and ncols == 1 else "")
    subtitle += ", and their difference below" if contrasts else ""
    if ncols == 1:
        _core.header(axes[0], title or heading, subtitle, key_sources=axes)
    else:
        _core.figure_header(fig, title or heading, subtitle, key_sources=axes)
    found = _notes(result, measure, view, in_play, limits, xs, bootstrap=band == "bootstrap")
    _core.add_notes(axes[-1] if ncols == 1 else fig, found)
    return fig


def _notes(
    result: _Fit,
    measure: str,
    view: str,
    rows: list[_Point],
    limits: Mapping[str, tuple[float, float]] | None,
    xs: Sequence[float],
    *,
    bootstrap: bool = False,
) -> list[str]:
    times = ", ".join(f"{x:g}" for x in xs)
    found: list[str] = []
    if measure == "curve":
        found.append(
            f"The fit targets the times {times} only, so each estimate is drawn at its target "
            "time and nothing is drawn between them: a risk between target times solves no "
            "score equation of the fit and is not estimated."
        )
    else:
        found.append(
            f"The fit targets the restricted means at {times} only (restricted_horizons=), so "
            "nothing is drawn at other times. No simultaneous band covers a restricted mean, "
            "so each has its pointwise interval only."
        )
    labels = {
        x: " and ".join(dict.fromkeys(r.estimate.support.label for r in rows if r.x == x))
        for x in xs
    }
    found.append("Support by time: " + ", ".join(f"{x:g} {labels[x]}" for x in xs) + ".")
    found += _support_notes([r.estimate for r in rows], "estimate")
    found += _withheld_notes([r.estimate for r in rows])
    gaps = sorted({r.x for r in rows if r.estimate.supplies_inference and r.lower is None})
    if gaps:
        found.append(
            "The fit reports no finite interval at time "
            + ", ".join(f"{x:g}" for x in gaps)
            + ", so none is drawn there and the marker is hollow."
        )
    if measure == "curve":
        if limits is not None and bootstrap and result.bootstrap_band is not None:
            found.append(
                "The simultaneous band drawn is the studentized bootstrap band over every risk "
                "and difference, drawn as brackets at each target time. It is formed on the "
                "estimates' own scale, not the logit, so a small risk's band can fall below 0."
            )
            found += _band_note(result.bootstrap_band.support, "simultaneous band")
        elif limits is not None:
            band = result.simultaneous
            found.append(
                "The simultaneous band drawn is the multiplier band over every risk and "
                "difference the fit reports, drawn as brackets at each target time."
            )
            found += _band_note(band.support, "simultaneous band")
            if result.bootstrap_band is not None:
                found.append(
                    "The fit also holds a bootstrap band; pass band='bootstrap' to draw it."
                )
        elif result.simultaneous is None:
            found.append("The fit holds no simultaneous band; fit with simultaneous=True for one.")
        else:
            found.append("No band is drawn (band='none'); the fit holds a multiplier band.")
        if view == "survival":
            found.append(
                "On the survival scale each interval and band limit is the risk's, mirrored "
                "about the survival estimate, as result.curve(scale='survival') reports it."
            )
    return found
