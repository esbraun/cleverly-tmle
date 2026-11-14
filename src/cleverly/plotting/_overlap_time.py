"""V9 beyond one point treatment: the treatment probabilities a fit over time holds.

:func:`~cleverly.plotting.overlap` hands these fits here:

* a survival fit of a treatment given once at baseline: the baseline treatment probability
  per arm, and with ``censoring=True`` the cumulative product of treatment and retention
  probabilities that the fit's ``g_bounds`` limit;
* a fit with a treatment at several time points: one panel per time point, the probability
  of the regimen's treatment among the units whose history up to that point followed the
  regimen, mirrored by whether they then took it;
* a structural nested mean fit: each time point's class probabilities, with the margin its
  positivity summary counts;
* an adaptive TMLE fit with external data: the treatment probability on the trial rows and
  the trial-membership probability, with the floor its positivity warning checks.

Every value is one the fit stored; nothing is refitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core
from ._overlap import Limit, _mirror, peak_share

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Node kinds whose factor is a density ratio or a policy's own probability, not the
#: probability of one assigned treatment.
_RATIO_KINDS = {
    "policy": "a policy that gives each treatment a set probability",
    "tilt": "an incremental policy that scales the odds of treatment",
    "mtp": "a modified treatment policy that shifts each unit's treatment",
}
#: The most panels in one row of a per-time-point figure.
_COLUMNS = 3
#: The prefix of a hazard fit's internal per-level keys in ``Mechanism.treatment``.
_INTERNAL = "\x00"


def over_time(
    result: Any,
    *,
    treatment_label: str | None,
    labels: dict[Any, str] | None,
    regimen: str | None,
    nodes: Sequence[int] | None,
    censoring: bool,
    horizon: int | None,
    cause: str | None,
    time_label: str,
) -> Figure | None:
    """Draw ``result`` when it is a fit over time this module covers, else ``None``."""
    from ..estimators.atmle_result import AdaptiveTMLEResult
    from ..longitudinal.estimator import LongitudinalResult
    from ..longitudinal.structural import StructuralNestedMeanResult

    if isinstance(result, LongitudinalResult):
        if result.config.estimator in ("gcomp", "parametric_gformula"):
            return None
        if bool(getattr(result.data, "treatment_held", False)):
            _unused(result, nodes=nodes, regimen=regimen)
            return baseline(
                result,
                treatment_label=treatment_label,
                labels=labels,
                censoring=censoring,
                horizon=horizon,
                cause=cause,
            )
        _unused(result, horizon=horizon, cause=cause)
        return per_node(
            result,
            regimen=regimen,
            nodes=nodes,
            censoring=censoring,
            labels=labels,
            time_label=time_label,
        )
    if isinstance(result, StructuralNestedMeanResult):
        _unused(result, regimen=regimen, horizon=horizon, cause=cause)
        if censoring:
            raise ValueError(
                "censoring=True draws retention, which a structural nested mean does not model"
            )
        return structural(result, nodes=nodes, time_label=time_label)
    if isinstance(result, AdaptiveTMLEResult):
        _unused(result, regimen=regimen, nodes=nodes, horizon=horizon, cause=cause)
        if censoring:
            raise ValueError(
                "censoring=True draws retention, which an adaptive TMLE fit does not model"
            )
        return adaptive(result, treatment_label=treatment_label, labels=labels)
    return None


def _unused(result: Any, **given: Any) -> None:
    named = [f"{key}=" for key, value in given.items() if value is not None]
    if named:
        raise ValueError(
            f"{', '.join(named)} does not apply to this {type(result).__name__}; "
            "see overlap()'s parameters for the fits each keyword selects in"
        )


def _predictions(n_folds: int) -> str:
    return "Out-of-fold predictions" if n_folds > 1 else "In-sample predictions (not cross-fitted)"


def _grid(panels: int, rows_per_block: int, blocks: int, columns: int) -> tuple[Figure, list[Axes]]:
    """A grid of ``blocks`` row blocks, each of ``rows_per_block`` rows of ``columns``."""
    nrows = rows_per_block * blocks
    fig, axes = _core.new_figure(
        nrows,
        ncols=columns,
        figsize=(3.3 * columns + 0.6, 2.7 * nrows + 1.9),
        sharex=False,
    )
    return fig, axes


def _drop_unused(fig: Figure, axes: list[Axes], used: set[int]) -> list[Axes]:
    for index, axis in enumerate(axes):
        if index not in used:
            fig.delaxes(axis)
    return [axis for index, axis in enumerate(axes) if index in used]


def _panel_title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=_core.LABEL_SIZE, color=_core.INK, pad=4.0)


def _level(level: object) -> str:
    """A treatment level as the data spell it, with a whole float shown as an integer."""
    try:
        number = float(str(level))
    except ValueError:
        return str(level)
    return str(int(number)) if number.is_integer() else str(level)


def _repeats_note(count: int) -> list[str]:
    if count > 1:
        return [f"The fit has {count} cross-fitting draws; this figure shows the first."]
    return []


# ------------------------------------------------------------------- baseline treatment


def baseline_probability(result: Any) -> tuple[Any, tuple[object, ...]]:
    """The node-1 probability matrix of a held fit: out of fold, unbounded, one column per level.

    Independent of the declared regimens: it is the treatment model at the observed
    history, ``Mechanism.treatment_observed[0]``.
    """
    mechanism = result.mechanism
    return (
        np.asarray(mechanism.treatment_observed[0], dtype=float),
        tuple(result.data.treatment_levels[0]),
    )


def selected_probability(result: Any, label: str) -> Any:
    """The probability of ``label``'s arm from the treatment model selected for that arm."""
    return np.asarray(result.mechanism.treatment[0][label], dtype=float)


def _regimen_fits(result: Any, horizon: int | None, cause: str | None) -> list[Any]:
    from ..longitudinal.weights import _select

    fits = result.repeats[0].fits
    labels = [
        label
        for label in dict.fromkeys(fit.regimen.label for fit in fits.values())
        if not str(label).startswith(_INTERNAL)
    ]
    return [_select(fits, label, horizon, cause) for label in labels]


def baseline(
    result: Any,
    *,
    treatment_label: str | None,
    labels: dict[Any, str] | None,
    censoring: bool,
    horizon: int | None,
    cause: str | None,
) -> Figure:
    """A survival fit of a treatment given once at baseline."""
    data = result.data
    if censoring and not data.has_censoring:
        raise ValueError("censoring=True draws retention, and this fit declares no censoring")
    name = treatment_label or str(data.treatment_names[0])
    codes = np.asarray(data.treatment[:, 0], dtype=float)
    present = np.isfinite(codes)
    weights = np.asarray(data.weights, dtype=float)
    selection = getattr(result, "selection", None)
    fits = _regimen_fits(result, horizon, cause)

    def group(level: object) -> str:
        if labels is not None and level in labels:
            return str(labels[level])
        return f"Units with {name} = {_level(level)}"

    # Each treatment panel: (values, received mask, names, axis words).
    panels: list[tuple[Any, Any, tuple[str, str], str]] = []
    levels = tuple(data.treatment_levels[0])
    if selection is not None:
        for fit in fits:
            code = float(fit.assignment[0, 0])
            level = levels[int(code)]
            values = selected_probability(result, fit.regimen.label)
            panels.append(
                (
                    values,
                    codes == code,
                    (group(level), "Units in every other arm"),
                    f"Selected model's probability of {name} = {_level(level)}",
                )
            )
    else:
        matrix, levels = baseline_probability(result)
        shown = (1,) if len(levels) == 2 else tuple(range(len(levels)))
        for column in shown:
            rest = group(levels[0]) if len(levels) == 2 else "Units in every other arm"
            panels.append(
                (
                    matrix[:, column],
                    codes == float(column),
                    (group(levels[column]), rest),
                    f"Predicted probability of {name} = {_level(levels[column])}",
                )
            )
    retention: list[tuple[Any, Any, str]] = []
    if censoring:
        for fit in fits:
            code = float(fit.assignment[0, 0])
            retention.append(
                (
                    np.asarray(fit.cumulative_unbounded[:, fit.horizon - 1], dtype=float),
                    codes == code,
                    fit.regimen.label,
                )
            )
    count = len(panels) + len(retention)
    fig, axes = _core.new_figure(count, figsize=(7.0, 1.4 + 2.9 * count))
    rows = present
    peak = max(
        peak_share(values[rows], weights[rows], inside[rows]) for values, inside, _, _ in panels
    )
    for axis, (values, inside, names, words) in zip(axes, panels, strict=False):
        _mirror(axis, values[rows], weights[rows], inside[rows], names=names, bounds=(), peak=peak)
        axis.set_ylabel("Share of units")
        axis.set_xlabel(words)
    pair = (float(result.config.g_bounds[0]), float(result.config.g_bounds[1]))
    limit = Limit(
        pair[0] if pair[0] > 0.0 else None,
        pair[1] if pair[1] < 1.0 else None,
        "Limit the fit put on the cumulative probability",
    )
    for axis, (values, inside, label) in zip(axes[len(panels) :], retention, strict=True):
        followers = rows & inside
        _mirror(
            axis,
            values[followers],
            weights[followers],
            np.ones(int(followers.sum()), dtype=bool),
            names=(f"Units given regimen {label}'s treatment", None),
            bounds=[limit] if limit.low is not None or limit.high is not None else [],
        )
        axis.set_ylabel("Share of units")
        axis.set_xlabel(f"Probability of {label}'s treatment and staying observed")
    predictions = _predictions(int(getattr(result.config, "n_folds", 1)))
    if selection is not None:
        last = max(fit.horizon for fit in fits)
        heading = f"Treatment probability, selection at horizon {last}"
        subtitle = f"{predictions} of each arm's selected treatment model, before any limit"
    else:
        heading = "Predicted probability of treatment, by group"
        subtitle = f"{predictions} at baseline, before any limit"
    _core.header(axes[0], heading, subtitle, key_sources=axes)
    found = _repeats_note(len(result.repeats))
    if selection is not None:
        found.append(
            "The collaborative search selects a treatment model for each arm at each horizon; "
            f"this figure draws each arm's selection at the last horizon ({last}), which "
            "result.mechanism holds. Other horizons' selections are not drawn."
        )
    found.append(
        "The fit's g_bounds limit the product of the treatment probability and the "
        "probability of staying under observation, not the treatment probability alone, so "
        "no limit is drawn on it"
        + ("." if censoring else "; pass censoring=True to draw that product.")
    )
    if censoring:
        first = fits[0]
        cause_words = f", cause {first.cause}" if first.cause is not None else ""
        found.append(
            f"The lower panels draw, for the units given each regimen's treatment, the "
            f"cumulative probability up to horizon {first.horizon}{cause_words} before the "
            "limit; the share beyond the limit is printed at its foot."
        )
    _core.add_notes(axes[-1], found)
    return fig


# ------------------------------------------------------------------- per time point


def node_population(data: Any, assignment: Any, time: int) -> Any:
    """The rows the regimen's treatment probability at ``time`` is a statement about.

    Rows whose history before the treatment at ``time`` is observed, followed the regimen
    and is event-free: ``LongitudinalData.at_risk``.
    """
    return np.asarray(data.at_risk(assignment, time), dtype=bool)


def _refuse_ratio_nodes(result: Any, fit: Any, times: Sequence[int]) -> None:
    from .._refusal import RefusalReason
    from ..exceptions import CapabilityError
    from ..validation.longitudinal import _node_kind

    data = result.data
    for time in times:
        kind = data.node_kind(time)
        if kind != "categorical":
            held = (
                "a dose, which has a density"
                if kind == "continuous"
                else "a treatment with a dose component, whose dose has a density"
            )
            raise CapabilityError(
                f"overlap() draws the probability of the regimen's treatment at each time "
                f"point; time point {time} holds {held}, not a probability per arm. "
                "weights() draws the weights the fit used.",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        policy = _node_kind(fit, time)
        if policy in _RATIO_KINDS:
            raise CapabilityError(
                f"overlap() draws the probability of the regimen's treatment at each time "
                f"point; at time point {time} the regimen is {_RATIO_KINDS[policy]}, whose "
                "weight is a ratio of probabilities or densities, not one probability. "
                "weights() draws the weights the fit used.",
                reason=RefusalReason.INVALID_DECLARATION,
            )


def _chosen_nodes(nodes: Sequence[int] | None, count: int) -> list[int]:
    from ..exceptions import DataError

    if nodes is None:
        return list(range(1, count + 1))
    if isinstance(nodes, (str, int)):
        raise DataError(f"nodes= takes a list of time points; got {nodes!r}")
    chosen = [int(node) for node in nodes]
    if not chosen or len(set(chosen)) != len(chosen):
        raise DataError(f"nodes= must name distinct time points; got {list(nodes)}")
    outside = [node for node in chosen if not 1 <= node <= count]
    if outside:
        raise DataError(f"nodes= names {outside}; the fit has time points 1 to {count}")
    return chosen


def _regimens(fits: dict[str, Any], regimen: str | None) -> tuple[list[str], list[str]]:
    """The regimens to draw: the one named, else every one up to three; and all held."""
    from ..exceptions import DataError

    held = [str(label) for label in dict.fromkeys(fit.regimen.label for fit in fits.values())]
    if regimen is not None:
        if regimen not in held:
            raise DataError(f"the result holds no regimen {regimen!r}; it holds {held}")
        return [regimen], held
    if len(held) > _core.MAX_SERIES:
        raise DataError(
            f"the result holds {len(held)} regimens and the figure draws at most "
            f"{_core.MAX_SERIES}; name one of {held} with regimen="
        )
    return held, held


def per_node(
    result: Any,
    *,
    regimen: str | None,
    nodes: Sequence[int] | None,
    censoring: bool,
    labels: dict[Any, str] | None,
    time_label: str,
) -> Figure:
    """A longitudinal fit with a treatment at several time points.

    One block of panels per regimen, each block one panel per time point (and one more
    row of retention panels with ``censoring=True``): the population and the probability
    both depend on the regimen, so none is left out by default.
    """
    from ..longitudinal.weights import _select

    data = result.data
    if censoring and not data.has_censoring:
        raise ValueError("censoring=True draws retention, and this fit declares no censoring")
    fits = result.repeats[0].fits
    drawn_regimens, held = _regimens(fits, regimen)
    times = _chosen_nodes(nodes, data.n_times)
    chosen = [_select(fits, label, None, None) for label in drawn_regimens]
    for fit in chosen:
        _refuse_ratio_nodes(result, fit, times)
    mechanism = result.mechanism
    weights = np.asarray(data.weights, dtype=float)
    columns = min(len(times), _COLUMNS)
    block = math.ceil(len(times) / columns)
    per_regimen = block * (2 if censoring else 1)
    fig, axes = _grid(len(times), per_regimen, len(chosen), columns)
    used: set[int] = set()
    point = time_label.lower()
    several = len(chosen) > 1
    shown = [str((labels or {}).get(label, label)) for label in drawn_regimens]
    for number, (label, fit, words) in enumerate(zip(drawn_regimens, chosen, shown, strict=True)):
        assignment = np.asarray(fit.assignment, dtype=float)
        first = number * per_regimen * columns
        lead = f"{words}, " if several else ""
        for index, time in enumerate(times):
            position = first + index
            axis = axes[position]
            used.add(position)
            rows = node_population(data, assignment, time)
            took = np.asarray(data.treatment[:, time - 1] == assignment[:, time - 1], dtype=bool)
            values = np.asarray(mechanism.treatment[time - 1][label], dtype=float)
            _mirror(
                axis,
                values[rows],
                weights[rows],
                took[rows],
                names=("Took the regimen's treatment", "Took another"),
                bounds=(),
            )
            _panel_title(axis, f"{lead}{time_label} {time}")
            axis.set_xlabel("Probability of the regimen's treatment")
            axis.set_ylabel("Share of units" if index % columns == 0 else "")
            if censoring:
                offset = first + block * columns + index
                used.add(offset)
                retained = np.asarray(mechanism.censoring[time - 1][label], dtype=float)
                followers = rows & took
                _mirror(
                    axes[offset],
                    retained[followers],
                    weights[followers],
                    np.ones(int(followers.sum()), dtype=bool),
                    names=("Took the regimen's treatment", None),
                    bounds=(),
                )
                _panel_title(axes[offset], f"{lead}{time_label} {time}: staying observed")
                axes[offset].set_xlabel("Probability of staying under observation")
                axes[offset].set_ylabel("Share of units" if index % columns == 0 else "")
    drawn = _drop_unused(fig, axes, used)
    predictions = _predictions(int(getattr(result.config, "n_folds", 1)))
    _core.figure_header(
        fig,
        "Treatment probability at each time point",
        f"Regimen {', '.join(shown)}: units still on it at each {point}; {predictions.lower()}",
        key_sources=drawn,
    )
    found = _repeats_note(len(result.repeats))
    if several:
        found.append(
            "Each block of panels is one regimen, named in its panel titles; regimen= draws one."
        )
    elif len(held) > 1:
        found.append(
            f"The figure draws regimen {shown[0]}, as regimen= asked; the fit holds {held}."
        )
    found.append(
        f"Each panel holds the units whose history up to that {point} is observed, followed "
        f"the regimen and is event-free; above, those who then took the regimen's treatment, "
        "below, those who did not. Bars are shares of those units."
    )
    if censoring:
        found.append(
            "The retention panels draw each unit's probability of staying under observation "
            f"after the {point}, among those who took the regimen's treatment there."
        )
    found.append(
        "The fit's g_bounds limit the cumulative product of these probabilities over time "
        "points, not each one, so no limit is drawn; weights() draws that product."
    )
    _core.add_notes(fig, found)
    return fig


# ------------------------------------------------------------------- structural nested mean


def structural_margin(config: Any) -> float:
    """The distance from 0 and 1 that the fit's ``extreme_share`` counts."""
    return float(config.positivity_margin)


def structural(result: Any, *, nodes: Sequence[int] | None, time_label: str) -> Figure:
    """A structural nested mean fit: each node's class probabilities."""
    from .._refusal import RefusalReason
    from ..exceptions import CapabilityError

    data = result.data
    config = result.config
    probabilities = result.diagnostics.class_probabilities
    times = _chosen_nodes(nodes, len(config.nodes))
    dose = [time for time in times if config.nodes[time - 1].continuous]
    if dose:
        raise CapabilityError(
            f"overlap() draws class probabilities; time point {dose[0]} holds a dose, which has "
            "none (its extreme share is nan)"
            + ("; choose categorical time points with nodes=" if nodes is None else "")
            + ".",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    margin = structural_margin(config)
    panels = [(time, column) for time in times for column in range(config.nodes[time - 1].d)]
    columns = min(len(panels), _COLUMNS)
    block = math.ceil(len(panels) / columns)
    fig, axes = _grid(len(panels), block, 1, columns)
    weights = np.asarray(data.weights, dtype=float)
    for index, (time, column) in enumerate(panels):
        node = config.nodes[time - 1]
        codes = np.asarray(data.node_codes(time), dtype=float)
        rows = np.isfinite(codes)
        values = np.asarray(probabilities[time - 1][:, column], dtype=float)
        level = node.levels[column]
        _mirror(
            axes[index],
            values[rows],
            weights[rows],
            (codes == float(column + 1))[rows],
            names=(f"{node.treatment} = {_level(level)}", "Any other level"),
            bounds=[Limit(margin, 1.0 - margin, "Margin the positivity summary counts")],
        )
        _panel_title(axes[index], f"{time_label} {time}")
        axes[index].set_xlabel(f"Probability of {node.treatment} = {_level(level)}")
        axes[index].set_ylabel("Share of units" if index % columns == 0 else "")
    drawn = _drop_unused(fig, axes, set(range(len(panels))))
    _core.figure_header(
        fig,
        "Treatment probability at each time point",
        f"{_predictions(int(config.n_folds))} given the history, the centring of g-estimation",
        key_sources=drawn,
    )
    found = [
        f"Dotted lines mark the margin {margin:g} from 0 and 1 that diagnostics.extreme_share "
        "counts; the share of units beyond each is printed at its foot. On a two-level time "
        "point the two shares add up to its extreme_share.",
        "The fit uses these probabilities to centre the treatment in its estimating equation, "
        "not as inverse weights, so no weight limit applies.",
    ]
    if any(config.nodes[time - 1].d > 1 for time in times):
        found.append(
            "A time point with more than two levels has one panel per level other than the "
            "reference, that level against the rest. Its printed shares are per level: "
            "extreme_share also counts the reference level's probability and counts a unit "
            "once however many of its levels are near 0 or 1, so the shares need not add up "
            "to it."
        )
    _core.add_notes(fig, found)
    return fig


# ------------------------------------------------------------------- adaptive TMLE


def adaptive_probabilities(fit: Any) -> tuple[Any, Any]:
    """The stored cross-fitted ``g(1 | W)`` and ``pi(W)`` on each row, before the limits."""
    return (
        np.asarray(fit.treatment_probability, dtype=float),
        np.asarray(fit.source_probability, dtype=float),
    )


def adaptive(result: Any, *, treatment_label: str | None, labels: dict[Any, str] | None) -> Figure:
    """An adaptive TMLE fit: the treatment and the trial-membership probability."""
    from ..estimators.atmle import SOURCE_FLOOR

    fit = result.fit
    data = result.data
    name = treatment_label or str(data.treatment_name)
    treatment = np.asarray(data.treatment, dtype=float)
    source = np.asarray(data.source, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    g, membership = adaptive_probabilities(fit)
    low, high = fit.probability_bounds
    clip = Limit(
        float(low) if low > 0.0 else None,
        float(high) if high < 1.0 else None,
        "Limits the fit put on the probabilities",
    )
    trial = source == 1.0
    external = ~trial

    def group(code: float) -> str:
        level = data.arm_label(code)
        if labels is not None and level in labels:
            return f"Trial: {labels[level]}"
        return f"Trial units with {name} = {level}"

    fig, (top, bottom) = _core.new_figure(2, figsize=(7.0, 7.4))
    _mirror(
        top,
        g[trial],
        weights[trial],
        (treatment == 1.0)[trial],
        names=(group(1.0), group(0.0)),
        bounds=[clip],
    )
    top.set_ylabel("Share of trial units")
    top.set_xlabel(f"Predicted probability of {name} = {data.arm_label(1.0)}")
    _mirror(
        bottom,
        membership,
        weights,
        trial,
        names=("Trial units", "External units"),
        bounds=[
            clip,
            Limit(SOURCE_FLOOR, None, "Floor the positivity warning checks", rows=external),
        ],
    )
    bottom.set_ylabel("Share of units")
    bottom.set_xlabel("Predicted probability of being in the trial")
    predictions = _predictions(int(fit.folds.n_folds))
    _core.header(
        top,
        "Treatment and trial-membership probability",
        f"{predictions}, before the limits were applied",
        key_sources=[top, bottom],
    )
    weighted = float(np.ptp(weights)) != 0.0
    found = [
        "The upper panel draws the treatment probability the fit estimated on the combined "
        "data, read on the trial rows.",
        f"The share of external units below the floor ({SOURCE_FLOOR:g}) is printed at its "
        "foot. "
        + (
            "It is weighted by the data's weights; the fit's summary and its positivity "
            "warning count the external units unweighted."
            if weighted
            else "It is the share the fit's summary reports and its positivity warning reads."
        ),
        "These are the initial trial-membership probabilities; targeting then moves them, so "
        "the values the estimate used differ.",
    ]
    if weighted:
        found.append("Bars and shares are weighted by the data's weights.")
    _core.add_notes(bottom, found)
    return fig


__all__ = ["over_time"]
