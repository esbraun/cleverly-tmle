"""V7: how much treating by a priority score first gains over treating at random.

Three views of one :class:`~cleverly.targeting_curves.TargetingCurve`. The TOC view draws the
targeting operator characteristic at each grid point against 0, with its pointwise intervals
and the sup-t band as bracket marks at the grid points (the band holds there only). The gain
view draws the cumulative gain against the random-targeting line ``q * ATE``; the interval on
each gain is the TOC interval scaled by ``q``, the uncertainty of the gap to that line, and a
note says so. The difference view draws ``TOC_A(q) - TOC_B(q)`` of a paired comparison with
pointwise intervals against 0 and no band.

The curve is estimated at the grid points only, so nothing is drawn between them except a thin
guide line through the points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["targeting"]

_VIEWS = ("toc", "gain", "difference")

#: Half the width of a band bracket, in units of the x axis (treated fractions).
_BRACKET = 0.012


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _withheld(row: dict[str, Any]) -> bool:
    anchor = row["quantity"] == "toc" and float(row["q"]) >= 1.0
    return not anchor and not _finite(row["std_error"])


def _points(
    ax: Axes,
    q: Sequence[float],
    y: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    hollow: Sequence[bool],
    label: str,
    *,
    origin: bool = False,
) -> None:
    """Points on a thin guide line, with an error bar where an interval exists.

    ``origin`` starts the guide line at ``(0, 0)``, where the gain is 0 by definition; no
    point is drawn there.
    """
    color = _core.PALETTE[0]
    guide_x = [0.0, *q] if origin else list(q)
    guide_y = [0.0, *y] if origin else list(y)
    ax.plot(guide_x, guide_y, color=color, linewidth=1.0, alpha=0.6, gid="cleverly-guide", zorder=2)
    for xi, yi, lo, hi, empty in zip(q, y, lower, upper, hollow, strict=True):
        if _finite(lo) and _finite(hi):
            ax.vlines(xi, lo, hi, color=color, linewidth=2.0, label=label, gid="cleverly-interval")
            label = "_nolegend_"
        ax.plot(
            [xi],
            [yi],
            marker="o",
            markersize=6.0,
            color=color,
            markerfacecolor=_core.FACE if empty else color,
            markeredgewidth=1.5,
            linestyle="none",
            gid="cleverly-point",
            zorder=3,
        )


def _brackets(
    ax: Axes, q: Sequence[float], lower: Sequence[float], upper: Sequence[float], label: str
) -> None:
    """The band as a bracket at each grid point: two short ticks joined by a thin line."""
    for xi, lo, hi in zip(q, lower, upper, strict=True):
        if not (_finite(lo) and _finite(hi)):
            continue
        for edge in (lo, hi):
            ax.hlines(
                edge,
                xi - _BRACKET,
                xi + _BRACKET,
                color=_core.SECONDARY_INK,
                linewidth=1.2,
                label=label,
                gid="cleverly-band",
                zorder=2.5,
            )
            label = "_nolegend_"
        ax.vlines(
            xi,
            lo,
            hi,
            color=_core.SECONDARY_INK,
            linewidth=0.6,
            gid="cleverly-band",
            zorder=2.4,
        )


def _experimental_reasons(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The reasons the drawn rows read experimental, without the withheld-interval reasons."""
    reasons: dict[str, None] = {}
    for row in rows:
        if row["status"] != "experimental":
            continue
        for part in str(row["status_reason"]).split("; "):
            if part and not part.startswith("no interval: "):
                reasons[part] = None
    return list(reasons)


def _gain_interval(
    curve: Any, name: str, rows: Sequence[dict[str, Any]]
) -> tuple[list[float], list[float], list[float]]:
    """The gain at each grid point and the interval the gain view draws on it.

    The gap to the random line is ``q * TOC(q)``, so the drawn interval is the TOC interval
    scaled by ``q`` and placed on the gain (plan 5.1). The gain's own interval, which also
    carries the uncertainty of the average effect, stays in the frame.
    """
    gains = {float(row["q"]): float(row["estimate"]) for row in curve.rows("gain", name)}
    y, low, high = [], [], []
    for row in rows:
        q = float(row["q"])
        toc = float(row["estimate"])
        y.append(gains[q])
        low.append(gains[q] - q * (toc - float(row["lower"])))
        high.append(gains[q] + q * (float(row["upper"]) - toc))
    return y, low, high


def _random_line(ax: Axes, ate: float) -> None:
    """The gain of treating at random, ``q * ATE``, as a dashed reference."""
    (line,) = ax.plot(
        [0.0, 1.0],
        [0.0, ate],
        color=_core.REFERENCE,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=1.5,
        label="Treating at random",
    )
    line.set_gid("cleverly-reference")


def _subtitle(curve: Any, name: str) -> str:
    area = curve.area("autoc", name)
    level = _core.level_text(1.0 - curve.alpha)
    shown = _core.format_values(
        [area["estimate"], area["lower"], area["upper"]],
        [area["std_error"]] if _finite(area["std_error"]) else [],
    )
    if _finite(area["lower"]):
        text = f"AUTOC {shown[0]} ({level} CI {shown[1]} to {shown[2]})"
    else:
        text = f"AUTOC {shown[0]}, no interval"
    if curve.route == "split":
        share = curve.n_train / (curve.n_train + curve.n_eval)
        return (
            f"{text}; the curve fitted on a {share:.0%} training split, evaluated on the "
            f"other {curve.n_eval:,} rows"
        )
    return f"{text}; {curve.n_eval:,} rows"


def targeting(
    curve: Any,
    *,
    view: str = "toc",
    score: str | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Plot how much treating by a score first gains over treating at random.

    Parameters
    ----------
    curve : TargetingCurve
        From :meth:`cleverly.study.IdentifiedEffect.targeting_curve` or
        ``curve.diagnostics.targeting_curve()``.
    view : {"toc", "gain", "difference"}, default="toc"
        ``"toc"``: the TOC against 0 with pointwise intervals and the band at the grid
        points. ``"gain"``: the cumulative gain against the random-targeting line.
        ``"difference"``: the TOC of the first score minus the second's, of a curve fitted
        with ``compare=``.
    score : str or None, default=None
        Which score to draw on the TOC and gain views; the primary score when omitted.
    ax : Axes or None, default=None
        The axes to draw on; a new figure when omitted.
    title : str or None, default=None
        The title.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    ValueError
        For an unknown view or score, or the difference view of a curve with no comparison.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import CausalStudy, ConditionalContrast, PointTreatment
    >>> from cleverly.bases import Modifiers
    >>> from cleverly.plotting import targeting
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> w1, w2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    >>> a = rng.binomial(1, 0.5, n)
    >>> y = w2 + a * (1 + w1) + rng.normal(0, 0.5, n)
    >>> frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "S": w1})
    >>> study = CausalStudy(
    ...     frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    ... )
    >>> effect = study.identify(ConditionalContrast(Modifiers(["W1"])))
    >>> curve = effect.targeting_curve(
    ...     "S",
    ...     computed_from=["W1"],
    ...     score_kind="external",
    ...     n_half_samples=200,
    ...     random_state=0,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(),
    ...     outcome_by_arm=True,
    ...     q_bounds=(-5.0, 6.0),
    ... )
    >>> fig = targeting(curve, view="gain")
    >>> fig.axes[0].get_xlabel()
    'Fraction treated, highest scores first'
    """
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}; got {view!r}")
    if view == "difference":
        if curve.compare is None:
            raise ValueError(
                "the difference view needs a curve fitted with compare=, which this one is not"
            )
        name = f"{curve.score} - {curve.compare}"
    else:
        name = curve.score if score is None else score
        if name not in (curve.score, curve.compare):
            raise ValueError(f"score {name!r} is not in this curve")
    rows = curve.rows("toc", name)
    level = 1.0 - curve.alpha
    q = [float(row["q"]) for row in rows]
    toc = [float(row["estimate"]) for row in rows]
    lower = [float(row["lower"]) for row in rows]
    upper = [float(row["upper"]) for row in rows]
    hollow = [_withheld(row) for row in rows]
    fig, axis = _core.figure_for(ax)
    lines: list[str] = []
    if view == "gain":
        y, low, high = _gain_interval(curve, name, rows)
        _random_line(axis, curve.ate)
        _points(
            axis,
            q,
            y,
            low,
            high,
            hollow,
            f"{_core.level_text(level)} interval of the gap to random",
            origin=True,
        )
        axis.set_ylabel("Cumulative gain")
        from .._summary import vocabulary

        lines.append(vocabulary.NOTE_TEXTS["targeting_gain_interval"])
    else:
        _core.reference_line(
            axis, 0.0, label="Treating at random" if view == "toc" else "No difference"
        )
        _points(axis, q, toc, lower, upper, hollow, _core.pointwise_label(level))
        if view == "toc":
            band_lower = [float(row["band_lower"]) for row in rows]
            band_upper = [float(row["band_upper"]) for row in rows]
            if any(_finite(value) for value in band_lower):
                _brackets(
                    axis, q, band_lower, band_upper, _core.band_label(level, "bootstrap sup-t")
                )
                lines.append("The band holds jointly at the grid points only.")
            axis.set_ylabel("TOC: effect in the top fraction minus the average")
        else:
            axis.set_ylabel(f"TOC of {curve.score} minus TOC of {curve.compare}")
            lines.append("Pointwise intervals only; no band is computed for the difference.")
    axis.set_xlim(0.0, 1.03)
    axis.set_xlabel("Fraction treated, highest scores first")
    reasons = [
        str(row["status_reason"]).split("no interval: ", 1)[1].split(";")[0]
        for row in rows
        if _withheld(row) and "no interval: " in str(row["status_reason"])
    ]
    if any(hollow):
        why = "; ".join(dict.fromkeys(reasons)) or "the interval is withheld"
        lines.append(f"Hollow points have no interval: {why}.")
    reasons = _experimental_reasons(rows + curve.rows("autoc", name) + curve.rows("qini", name))
    if reasons:
        lines.append(f"Experimental: {'; '.join(reasons)}.")
    _core.legend(axis)
    question = (
        f"Does {curve.score} rank the effect better than {curve.compare}?"
        if view == "difference"
        else f"How much does treating by {name} first gain over treating at random?"
    )
    _core.header(
        axis,
        title or question,
        _subtitle(curve, name),
    )
    _core.notes(axis, lines)
    return fig
