"""V9: the predicted probability of treatment, by the treatment each unit received.

Mirrored histograms on one probability axis, treated above and untreated below, show where
one group has no counterparts in the other: the overlap check every weighting or doubly
robust analysis needs. The limits the fit put on the probabilities are drawn as vertical
lines with the share of units beyond each. With more than two arms each arm gets its own
panel, that arm against the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

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
    bounds: list[tuple[tuple[float, float], str]],
    peak: float | None = None,
) -> float:
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
        ax.annotate(
            name,
            xy=(0.5, 0.98 if sign > 0 else 0.02),
            xycoords="axes fraction",
            ha="center",
            va="top" if sign > 0 else "bottom",
            fontsize=_core.TICK_SIZE,
            color=_core.INK,
            bbox={"boxstyle": "square,pad=0.2", "facecolor": _core.FACE, "edgecolor": color},
            gid=_core.GID_SERIES_LABEL,
        )
    ax.axhline(0.0, color=_core.SECONDARY_INK, linewidth=0.8)
    top = float(np.max(np.abs(ax.get_ylim()))) if peak is None else peak
    ax.set_ylim(-top * 1.2, top * 1.2)
    ax.yaxis.set_major_formatter(_absolute_percent())
    ax.set_xlim(0.0, 1.0)
    for index, ((low, high), label) in enumerate(bounds):
        below = float(weights[raw < low].sum()) / total
        above = float(weights[raw > high].sum()) / total
        for value, share, side in ((low, below, "below"), (high, above, "above")):
            line = _core.reference_line(
                ax, value, orientation="vertical", label=label, kind="marker"
            )
            if index:
                line.set_linestyle((0, (5, 2, 1, 2)))
            ax.annotate(
                f"{_share_text(share)} {side}",
                xy=(value, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(3.0 if side == "below" else -3.0, 3.0 + 11.0 * index),
                textcoords="offset points",
                ha="left" if side == "below" else "right",
                va="bottom",
                fontsize=_core.TICK_SIZE,
                color=_core.SECONDARY_INK,
                gid="cleverly-tail-share",
            )
    return top


def _absolute_percent() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _: f"{round(abs(value) * 100, 2):g}%")


@_core.labelled
def overlap(
    result: Any,
    *,
    treatment_label: str | None = None,
    labels: dict[Any, str] | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot the predicted probability of treatment for treated and untreated units.

    Use it to check overlap: whether every kind of unit had a real chance of each
    treatment. Where one group's histogram has mass and the other's has none, the
    comparison rests on extrapolation, and units near 0 or 1 get large weights. The limits
    the fit put on the probabilities are drawn as dotted lines, with the share of units
    beyond each. Bars are shares of all units, weighted by the data's weights.

    The probabilities are the out-of-fold predictions before the limits were applied. A fit
    with several cross-fitting repeats draws the first repeat, and
    :func:`~cleverly.plotting.notes` says so.

    Parameters
    ----------
    result : TMLEResult
        A fitted point-treatment result with an estimated treatment mechanism.
    treatment_label : str or None, default=None
        Words for the treatment in the axis label: ``"Predicted probability of smoking =
        1"``, or with ``labels=`` ``"Predicted probability of smoking: smoking mothers"``.
        The column name when omitted, or ``"being in the group"`` beside ``labels=``.
    labels : dict or None, default=None
        Words for each arm, keyed by the arm's level as in the data, such as
        ``{1: "offered", 0: "not offered"}``. Each names its group of units on its own.
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
        One panel for a two-arm treatment, one per arm for more arms.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism, or is not a point-treatment fit.
    DataError
        When the treatment is continuous, which has no probability per arm.

    See Also
    --------
    cleverly.sensitivity.positivity_report : The same overlap as numbers.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say which cross-fitting repeat
    is drawn and that the share of units beyond each limit is printed at its foot.

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

    if not hasattr(result, "repeats") or not hasattr(result, "nuisance"):
        from .._refusal import NO_ALTERNATIVE, RefusalReason
        from ..exceptions import CapabilityError

        raise CapabilityError(
            f"overlap() draws a point-treatment fit's predicted probability of treatment; a "
            f"{type(result).__name__} has one treatment probability per time point, which this "
            "plot does not draw",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X92",
        )
    positivity_report(result)  # the report's own refusals: no mechanism, a continuous dose
    data = result.data
    name = treatment_label or str(data.treatment_name)
    propensity = result.repeats[0].nuisance.propensity
    raw_all = np.asarray(propensity.values, dtype=float)
    treatment = np.asarray(data.treatment, dtype=float)
    weights_all = np.asarray(data.weights, dtype=float)
    recorded = np.isfinite(treatment)
    arms = tuple(float(arm) for arm in propensity.arms)
    bounds = _limits(result)

    def labelled(code: float) -> str | None:
        level = data.arm_label(code)
        return None if labels is None or level not in labels else str(labels[level])

    def group(code: float) -> str:
        # A caller's label names the group on its own ("offered", "an offer"), so it is
        # not fitted into a sentence; an unlabelled arm reads as its column and level.
        own = labelled(code)
        return own if own is not None else f"Units with {name} = {data.arm_label(code)}"

    def axis_words(code: float) -> str:
        # A caller's label names a group of units, so the axis names the group after a
        # colon: the treatment's words lead when given, being in the group otherwise.
        own = labelled(code)
        if own is not None and treatment_label and binary:
            # one panel: the treatment's words name the probability, and the key the groups
            return f"Predicted probability of {treatment_label}"
        if own is not None:
            lead = treatment_label or "being in the group"
            return f"Predicted probability of {lead}: {_core.mid_phrase(own)}"
        return f"Predicted probability of {name} = {data.arm_label(code)}"

    binary = len(arms) == 2
    shown = (arms[1],) if binary else arms
    figure_height = 4.5 if binary else 1.0 + 2.4 * len(shown)
    fig, axes = _core.new_figure(len(shown), figsize=(7.0, figure_height), sharex=True)
    edges = np.linspace(0.0, 1.0, _BINS + 1)
    weights = weights_all[recorded]
    peak = 0.0
    for arm in shown:
        raw = raw_all[recorded, propensity.column_for(arm)]
        inside = treatment[recorded] == arm
        for mask in (inside, ~inside):
            counts, _ = np.histogram(raw[mask], bins=edges, weights=weights[mask])
            peak = max(peak, float(counts.max(initial=0.0)) / float(weights.sum()))
    for axis, arm in zip(axes, shown, strict=True):
        raw = raw_all[recorded, propensity.column_for(arm)]
        inside = treatment[recorded] == arm
        rest = group(arms[0]) if binary else "Units in every other arm"
        _mirror(
            axis,
            raw,
            weights,
            inside,
            names=(group(arm), rest),
            bounds=bounds,
            peak=peak,
        )
        axis.set_ylabel("Share of units")
        axis.set_xlabel(axis_words(arm))
    predictions = (
        "Out-of-fold predictions"
        if getattr(result.config, "cross_fit", True)
        else "In-sample predictions (not cross-fitted)"
    )
    weighted = float(np.ptp(weights_all)) != 0.0
    _core.header(
        axes[0],
        title or "Predicted probability of treatment, by group",
        f"{predictions}, before the limits were applied"
        + ("; bars weighted by the data's weights" if weighted else ""),
    )
    found = []
    if len(result.repeats) > 1:
        found.append(
            f"The fit has {len(result.repeats)} cross-fitting repeats; this figure shows the first."
        )
    if bounds:
        found.append(
            "The share of units beyond each limit is printed at its foot"
            + (", weighted by the data's weights." if weighted else ".")
        )
    if not binary:
        found.append(
            "Each panel compares one arm with the rest; blue is that arm and orange the rest "
            "in every panel."
        )
    _core.add_notes(axes[-1], found)
    return fig


def _limits(result: Any) -> list[tuple[tuple[float, float], str]]:
    """The limit pairs the reported estimates used, each with its key entry.

    The ATT and ATC divide by the other arm's probability under their own, conditional
    limits; every other estimate uses the fit's marginal pair.
    """
    config = result.config
    stems = {str(name).split("[")[0].split("_")[0] for name in result.estimates}
    conditional = stems & {"att", "atc"}
    marginal = stems - {"att", "atc"}
    found: list[tuple[tuple[float, float], str]] = []
    words = " and ".join(
        {"att": "the treated", "atc": "the untreated"}[stem]
        for stem in sorted(conditional, reverse=True)
    )
    marginal_pair = getattr(config, "g_bounds", None)
    conditional_pair = getattr(config, "g_bounds_conditional", None)
    if conditional_pair is not None and conditional:
        conditional_pair = (float(conditional_pair[0]), float(conditional_pair[1]))
    else:
        conditional_pair = None
    if marginal and marginal_pair is not None:
        marginal_pair = (float(marginal_pair[0]), float(marginal_pair[1]))
        # The marginal pair serves every estimate but the ATT and ATC (arm means, ratios,
        # the average effect); it is named as the rest only when those are drawn too.
        own = conditional_pair is not None and conditional_pair != marginal_pair
        text = (
            f"Probability limits used for every estimate but the effect among {words}"
            if own
            else "Probability limits the fit used"
        )
        found.append((marginal_pair, text))
    if conditional_pair is not None and (not found or found[0][0] != conditional_pair):
        found.append((conditional_pair, f"Probability limits used for the effect among {words}"))
    return found


__all__ = ["overlap"]
