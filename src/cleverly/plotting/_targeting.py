"""V7: how much treating by a priority score first gains over treating at random.

Three views of one :class:`~cleverly.targeting_curves.TargetingCurve`. The TOC view draws the
targeting operator characteristic at each grid point against 0, with its pointwise intervals and the
sup-t band as bracket marks at the grid points (the band holds there only). The gain view draws the
cumulative gain against the random-targeting line ``q * ATE``; the interval on each gain is the TOC
interval scaled by ``q``, the uncertainty of the gap to that line, and
:func:`~cleverly.plotting.notes` says so. The difference view draws ``TOC_A(q) - TOC_B(q)`` of a
paired comparison with pointwise intervals against 0 and no band.

The curve is estimated at the grid points only, so nothing is drawn between them except a thin
guide line through the points.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["targeting"]

_VIEWS = ("toc", "gain", "difference")


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
            ax.vlines(xi, lo, hi, color=color, linewidth=2.0, label=label, gid=_core.GID_INTERVAL)
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
            gid=_core.GID_POINT,
            zorder=3,
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
    line.set_gid(_core.GID_REFERENCE)


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


@_core.labelled
def targeting(
    curve: Any,
    *,
    view: str = "toc",
    score: str | None = None,
    ax: Axes | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    outcome_label: str | None = None,
    labels: Mapping[str, str] | None = None,
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
        The outcome's quantity in your words, such as ``"order share"``, for the y axis; the
        axis says "the mean outcome" or "effect" when omitted.
    labels : mapping of str to str or None, default=None
        Words for each score in the title and the y axis, keyed by the score's column name,
        such as ``{"S": "the predicted effect"}``; the column name when omitted.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    ValueError
        For an unknown view or score, or the difference view of a curve with no comparison.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes mark the experimental top
    fractions, say that each interval is for the gap to the random line, and say where the
    band holds.

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
    >>> fig = targeting(curve, view="gain", outcome_label="Y", labels={"S": "the score"})
    >>> fig.axes[0].get_ylabel()
    'Change in Y from treating the top fraction'
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
    words = dict(labels or {})
    first = words.get(curve.score, curve.score)
    second = words.get(curve.compare, curve.compare) if curve.compare is not None else None
    said = words.get(name, name)
    outcome = _core.mid_phrase(outcome_label) if outcome_label else "the mean outcome"
    effect = f"Effect on {_core.mid_phrase(outcome_label)}" if outcome_label else "Effect"
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
        axis.set_ylabel(f"Change in {outcome} from treating the top fraction")
        lines.append(
            "Each interval is for the gap to the random line. The interval of the gain itself, "
            "which adds the uncertainty of the average effect, is in to_frame()."
        )
    else:
        _core.reference_line(
            axis, 0.0, label="Treating at random" if view == "toc" else "No difference"
        )
        _points(axis, q, toc, lower, upper, hollow, _core.pointwise_label(level))
        if any(xi >= 1.0 for xi in q):
            lines.append("TOC(1) is 0 by definition, so its point has no interval.")
        if view == "toc":
            band_lower = [float(row["band_lower"]) for row in rows]
            band_upper = [float(row["band_upper"]) for row in rows]
            drawn = [
                (xi, lo, hi)
                for xi, lo, hi in zip(q, band_lower, band_upper, strict=True)
                if _finite(lo) and _finite(hi)
            ]
            if drawn:
                _core.brackets(
                    axis,
                    [xi for xi, _, _ in drawn],
                    [lo for _, lo, _ in drawn],
                    [hi for _, _, hi in drawn],
                    color=_core.SECONDARY_INK,
                    label=_core.band_label(level),
                    band=True,
                )
                lines.append("The band holds jointly at the grid points only.")
            axis.set_ylabel(f"{effect} in the top fraction minus the average")
        else:
            axis.set_ylabel(f"TOC by {first} minus TOC by {second}")
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
        f"Ranking by {first} against {second}"
        if view == "difference"
        else f"Gain from treating by {said} first"
    )
    _core.header(
        axis,
        title or question,
        _subtitle(curve, name),
    )
    _core.add_notes(axis, lines)
    return fig
