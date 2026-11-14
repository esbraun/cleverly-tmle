"""V9: the predicted probability of treatment, by the treatment each unit received.

Mirrored histograms on one probability axis, treated above and untreated below, show where
one group has no counterparts in the other: the overlap check every weighting or doubly
robust analysis needs. The limits the fit put on the probabilities are drawn as vertical
lines with the share of units beyond each. With more than two arms each arm gets its own
panel, that arm against the rest. Fits over time and adaptive TMLE fits are drawn by
:mod:`cleverly.plotting._overlap_time`, which reuses the same mirrored panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

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


class Limit(NamedTuple):
    """A pair of probability limits drawn as dotted lines, with the share beyond each.

    Either end may be ``None``: a floor alone, or a ceiling alone. ``rows`` restricts the
    printed shares to those rows (all drawn rows when ``None``), as a summary that counts
    only some rows does.
    """

    low: float | None
    high: float | None
    label: str
    rows: Any = None


def peak_share(raw: Any, weights: Any, inside: Any) -> float:
    """The tallest bar of either group, as a share of the drawn rows' weight."""
    edges = np.linspace(0.0, 1.0, _BINS + 1)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    peak = 0.0
    for mask in (inside, ~inside):
        counts, _ = np.histogram(raw[mask], bins=edges, weights=weights[mask])
        peak = max(peak, float(counts.max(initial=0.0)) / total)
    return peak


def _mirror(
    ax: Axes,
    raw: Any,
    weights: Any,
    inside: Any,
    *,
    names: tuple[str, str | None],
    bounds: Sequence[Limit],
    peak: float | None = None,
) -> float:
    """One arm's probability, rows of that arm above the axis and the rest below.

    With no second name only the rows ``inside`` are drawn, above the axis.
    """
    edges = np.linspace(0.0, 1.0, _BINS + 1)
    total = float(weights.sum())
    groups = [(1.0, inside, names[0], _core.PALETTE[0])]
    if names[1] is not None:
        groups.append((-1.0, ~inside, names[1], _core.PALETTE[1]))
    for sign, mask, name, color in groups:
        counts, _ = np.histogram(raw[mask], bins=edges, weights=weights[mask])
        share = sign * counts / total if total > 0.0 else counts * 0.0
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
            str(name),
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
    top = top if top > 0.0 else 1.0
    ax.set_ylim(-top * 1.2 if names[1] is not None else 0.0, top * 1.2)
    ax.yaxis.set_major_formatter(_absolute_percent())
    ax.set_xlim(0.0, 1.0)
    for index, limit in enumerate(bounds):
        among = np.ones(raw.shape, dtype=bool) if limit.rows is None else np.asarray(limit.rows)
        among_total = float(weights[among].sum())
        ends = []
        if limit.low is not None:
            ends.append((float(limit.low), raw < limit.low, "below"))
        if limit.high is not None:
            ends.append((float(limit.high), raw > limit.high, "above"))
        for value, beyond, side in ends:
            share = float(weights[among & beyond].sum()) / among_total if among_total else 0.0
            line = _core.reference_line(
                ax, value, orientation="vertical", label=limit.label, kind="marker"
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
    regimen: str | None = None,
    nodes: Sequence[int] | None = None,
    censoring: bool = False,
    horizon: int | None = None,
    cause: str | None = None,
    time_label: str = "Time point",
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

    Fits beyond one point treatment draw the probabilities they hold:

    * A survival fit of a treatment given once at baseline draws the baseline treatment
      probability per arm. A collaborative fit draws, per arm, the treatment model its
      search selected for that arm at the last horizon. The fit's ``g_bounds`` limit the
      product of the treatment probability and the probability of staying under
      observation, not the treatment probability alone, so ``censoring=True`` adds that
      product for each regimen at the chosen horizon, with its limit.
    * A fit with a treatment at several time points draws one panel per time point: the
      probability of the regimen's treatment among the units whose history up to that time
      point is observed and followed the regimen, those who then took the regimen's
      treatment above and those who did not below. ``censoring=True`` adds each time
      point's probability of staying under observation among those who took it. The
      limits apply to the cumulative product, which :func:`~cleverly.plotting.weights`
      draws, so no limit line is drawn here.
    * A structural nested mean fit draws each time point's estimated probability of each
      treatment level given the history, with the margin from 0 and 1 its positivity
      summary counts.
    * An adaptive TMLE fit with external data draws the treatment probability on the trial
      rows, and the probability of being in the trial for trial against external rows,
      with the floor its positivity warning checks.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult or StructuralNestedMeanResult or AdaptiveTMLEResult
        A fitted result with an estimated treatment mechanism.
    treatment_label : str or None, default=None
        Words for the treatment in the axis label: ``"Predicted probability of smoking =
        1"``, or with ``labels=`` ``"Predicted probability of smoking: smoking mothers"``.
        The column name when omitted, or ``"being in the group"`` beside ``labels=``.
    labels : dict or None, default=None
        Words for each arm, keyed by the arm's level as in the data, such as
        ``{1: "offered", 0: "not offered"}``. Each names its group of units on its own. On
        a fit with a treatment at several time points, words for each regimen keyed by its
        label.
    regimen : str or None, default=None
        On a fit with a treatment at several time points, the one regimen to draw; every
        regimen of the result, one block of panels each, when omitted. A result with more
        than three regimens needs it.
    nodes : sequence of int or None, default=None
        On a fit with a treatment at several time points, the time points to draw, counted
        from one; every one when omitted.
    censoring : bool, default=False
        On a longitudinal or survival fit with censoring, also draw the probability of
        staying under observation: the cumulative product with the treatment probability
        for a baseline treatment, each time point's probability for a treatment over time.
    horizon : int or None, default=None
        The horizon of the cumulative product ``censoring=True`` draws for a baseline
        treatment; the largest when omitted.
    cause : str or None, default=None
        The cause of that product under competing events; the first when omitted.
    time_label : str, default="Time point"
        Words for a time point in the panel titles.
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
        One panel for a two-arm treatment, one per arm for more arms; one per time point,
        or per time point and level, for a treatment over time.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism; when a time point's treatment is a
        dose or a policy whose weight is a density ratio, which has no probability per arm;
        or for a continuous-time survival fit, whose probabilities this plot does not
        draw yet.
    DataError
        When the treatment is continuous, which has no probability per arm; when the
        regimen, a time point, the horizon or the cause is not in the result; or when a fit
        over time holds more than three regimens and ``regimen`` is omitted.
    ValueError
        When ``censoring=True`` and the fit has no censoring, or a keyword for a fit over
        time is given for one that is not.

    See Also
    --------
    cleverly.sensitivity.positivity_report : The same overlap as numbers.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say which cross-fitting repeat
    is drawn and that the share of units beyond each limit is printed at its foot. For a fit
    over time they also say which regimen and horizon are drawn, which units each panel
    holds, and where the fit's limits apply instead.

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
    from ._overlap_time import over_time

    drawn = over_time(
        result,
        treatment_label=treatment_label,
        labels=labels,
        regimen=regimen,
        nodes=nodes,
        censoring=censoring,
        horizon=horizon,
        cause=cause,
        time_label=time_label,
    )
    if drawn is not None:
        return drawn
    if not hasattr(result, "repeats") or not hasattr(result, "nuisance"):
        from .._refusal import NO_ALTERNATIVE, RefusalReason
        from ..exceptions import CapabilityError

        gap = _unread_overlap(result)
        if gap is None:
            raise CapabilityError(
                "overlap() draws a point-treatment fit's predicted probability of treatment; "
                + _not_a_propensity(result)
                + ".",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        raise CapabilityError(
            "overlap() draws a point-treatment fit's predicted probability of treatment; "
            + gap
            + ", which this plot does not draw yet.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X100",
        )
    if regimen is not None or nodes is not None or censoring or horizon is not None or cause:
        raise ValueError(
            "regimen=, nodes=, censoring=, horizon= and cause= apply to a fit over time; this "
            f"{type(result).__name__} fits its treatment once"
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


def _limits(result: Any) -> list[Limit]:
    """The limit pairs the reported estimates used, each with its key entry.

    The ATT and ATC divide by the other arm's probability under their own, conditional
    limits; every other estimate uses the fit's marginal pair.
    """
    config = result.config
    stems = {str(name).split("[")[0].split("_")[0] for name in result.estimates}
    conditional = stems & {"att", "atc"}
    marginal = stems - {"att", "atc"}
    found: list[tuple[tuple[float, float], str]] = []  # (pair, label), then as Limit
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
    return [Limit(pair[0], pair[1], label) for pair, label in found]


__all__ = ["overlap"]


def _unread_overlap(result: Any) -> str | None:
    """What a fit X100 will draw holds in place of one propensity, or ``None``."""
    from ..survival.result import ContinuousSurvivalResult

    if isinstance(result, ContinuousSurvivalResult):
        return "a continuous-time survival fit does not keep each row's treatment probability"
    return None


def _not_a_propensity(result: Any) -> str:
    """Why a result outside X100 has no propensity this plot can draw, in words."""
    from ..effect_curves import EffectCurveResult
    from ..estimators.dose_response import DoseResponseResult
    from ..longitudinal.estimator import LongitudinalResult

    if isinstance(result, LongitudinalResult):
        return (
            "a g-computation or parametric g-formula fit estimates no treatment mechanism; "
            "fit SequentialIPW or LTMLE for treatment probabilities"
        )
    if isinstance(result, DoseResponseResult):
        return "a dose-response fit has a continuous treatment, which has no probability per arm"
    if isinstance(result, EffectCurveResult):
        return "an effect curve is a summary of a fit; pass the fit it came from"
    return f"this {type(result).__name__} is not a point-treatment fit"
