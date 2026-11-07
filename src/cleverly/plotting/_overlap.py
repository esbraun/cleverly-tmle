"""V9: the predicted probability of treatment, by the treatment each unit received.

Mirrored histograms on one probability axis, treated above and untreated below, show where
one group has no counterparts in the other: the overlap check every weighting or doubly
robust analysis needs. The limits the fit put on the probabilities are drawn as vertical
lines with the share of units beyond each. With more than two arms each arm gets its own
panel, that arm against the rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Histogram bins over [0, 1].
_BINS = 40


def _share_text(share: float) -> str:
    if share == 0.0:
        return "0%"
    if share < 0.001:
        return "<0.1%"
    return f"{100.0 * share:.1f}%"


def _mirror(
    ax: Axes,
    raw: Any,
    weights: Any,
    inside: Any,
    *,
    names: tuple[str, str],
    bounds: tuple[float, float] | None,
) -> None:
    """One arm's probability, rows of that arm above the axis and the rest below."""
    edges = np.linspace(0.0, 1.0, _BINS + 1)
    total = float(weights.sum())
    for sign, mask, name, color in (
        (1.0, inside, names[0], _core.PALETTE[0]),
        (-1.0, ~inside, names[1], _core.PALETTE[1]),
    ):
        counts, _ = np.histogram(raw[mask], bins=edges, weights=weights[mask])
        share = sign * counts / total
        ax.bar(
            edges[:-1],
            share,
            width=np.diff(edges),
            align="edge",
            color=color,
            edgecolor=_core.FACE,
            linewidth=0.6,
            gid=f"cleverly-overlap-{'above' if sign > 0 else 'below'}",
        )
        _core.corner_label(
            ax, 0.97 if sign > 0 else 0.03, name, color, va="top" if sign > 0 else "bottom"
        )
    ax.axhline(0.0, color=_core.SECONDARY_INK, linewidth=0.8)
    peak = float(np.max(np.abs(ax.get_ylim())))
    ax.set_ylim(-peak * 1.15, peak * 1.15)
    ax.yaxis.set_major_formatter(_absolute_percent())
    ax.set_xlim(0.0, 1.0)
    if bounds is None:
        return
    low, high = bounds
    below = float(weights[raw < low].sum()) / total
    above = float(weights[raw > high].sum()) / total
    for value, share, side in ((low, below, "below"), (high, above, "above")):
        _core.reference_line(
            ax, value, orientation="vertical", label="Limits the fit used", kind="marker"
        )
        ax.annotate(
            f"{_share_text(share)} {side}",
            xy=(value, 0.5),
            xycoords=("data", "axes fraction"),
            xytext=(4.0 if side == "below" else -4.0, 3.0),
            textcoords="offset points",
            ha="left" if side == "below" else "right",
            va="bottom",
            fontsize=_core.TICK_SIZE,
            color=_core.SECONDARY_INK,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": _core.FACE, "edgecolor": "none"},
            gid="cleverly-tail-share",
        )


def _absolute_percent() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _: f"{abs(value) * 100:.0f}%")


def overlap(
    result: Any,
    *,
    treatment_label: str | None = None,
    labels: dict[Any, str] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the predicted probability of treatment for treated and untreated units.

    Use it to check overlap: whether every kind of unit had a real chance of each
    treatment. Where one group's histogram has mass and the other's has none, the
    comparison rests on extrapolation, and units near 0 or 1 get large weights. The limits
    the fit put on the probabilities are drawn as dashed lines, with the share of units
    beyond each. Bars are shares of all units, weighted by the data's weights.

    The probabilities are the out-of-fold predictions before the limits were applied. A fit
    with several cross-fitting repeats draws the first repeat, and the note says so.

    Parameters
    ----------
    result : TMLEResult
        A fitted point-treatment result with an estimated treatment mechanism.
    treatment_label : str or None, default=None
        Words for the treatment; its column name when omitted.
    labels : dict or None, default=None
        Words for each arm, keyed by the arm's level as in the data.
    title : str or None, default=None
        The title.

    Returns
    -------
    Figure
        One panel for a two-arm treatment, one per arm for more arms.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism.
    DataError
        When the treatment is continuous, which has no probability per arm.

    See Also
    --------
    cleverly.sensitivity.positivity_report : The same overlap as numbers.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import overlap
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
    >>> fig = overlap(result)
    >>> fig.axes[0].get_xlabel()
    'Predicted probability of A = 1'
    >>> plt.close(fig)
    """
    from ..sensitivity.positivity import positivity_report

    positivity_report(result)  # the report's own refusals: no mechanism, a continuous dose
    data = result.data
    name = treatment_label or str(data.treatment_name)
    propensity = result.repeats[0].nuisance.propensity
    raw_all = np.asarray(propensity.values, dtype=float)
    treatment = np.asarray(data.treatment, dtype=float)
    weights_all = np.asarray(data.weights, dtype=float)
    recorded = np.isfinite(treatment)
    arms = tuple(float(arm) for arm in propensity.arms)
    bounds = getattr(result.config, "g_bounds", None)
    bound_pair = None if bounds is None else (float(bounds[0]), float(bounds[1]))

    def arm_words(code: float) -> str:
        level = data.arm_label(code)
        if labels is not None and level in labels:
            return labels[level]
        return f"{name} = {level}"

    binary = len(arms) == 2
    shown = (arms[1],) if binary else arms
    figure_height = 4.5 if binary else 1.0 + 2.4 * len(shown)
    fig, axes = _core.new_figure(len(shown), figsize=(7.0, figure_height), sharex=True)
    for axis, arm in zip(axes, shown, strict=True):
        raw = raw_all[recorded, propensity.column_for(arm)]
        inside = treatment[recorded] == arm
        rest = arm_words(arms[0]) if binary else "every other arm"
        _mirror(
            axis,
            raw,
            weights_all[recorded],
            inside,
            names=(f"Units with {arm_words(arm)}", f"Units with {rest}"),
            bounds=bound_pair,
        )
        axis.set_ylabel("Share of units")
        axis.set_xlabel(f"Predicted probability of {arm_words(arm)}")
    _core.legend(axes[0])
    _core.header(
        axes[0],
        title or "Predicted probability of treatment, by group",
        "Out-of-fold predictions before the limits were applied"
        + ("" if float(np.ptp(weights_all)) == 0.0 else "; bars weighted by the data's weights"),
    )
    found = []
    if len(result.repeats) > 1:
        found.append(
            f"The fit has {len(result.repeats)} cross-fitting repeats; this figure shows the first."
        )
    if bound_pair is not None:
        found.append(
            "The limits are those used for the average effect; the share of units beyond "
            "each is printed beside it."
        )
    _core.notes(axes[-1], found)
    return fig


__all__ = ["overlap"]
