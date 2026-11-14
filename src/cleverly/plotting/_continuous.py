"""Overlap, weights and learners of a continuous-time survival fit, read off its record.

:class:`~cleverly.survival.ContinuousSurvivalRecord` keeps each row's estimated probability
of the treated arm, its estimated probability of staying uncensored just before each target
time under each arm, the floor the targeting weights applied, and the hazard and treatment
library selections. :func:`~cleverly.plotting.overlap`, :func:`~cleverly.plotting.weights`
and :func:`~cleverly.plotting.learners` hand such a fit here. Every value is one the fit
stored, before any floor; nothing is refitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core
from ._overlap import Limit, _mirror

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

    from ..survival.diagnostics import ContinuousSurvivalNuisances, NuisanceSelection
    from ..survival.result import ContinuousSurvivalResult

#: The width one target time's boxes share.
_BOX_WIDTH = 0.7


# --------------------------------------------------------------------------- readers


def treatment_probabilities(record: Any) -> Any:
    """``(2, n)`` each row's probability of each arm, in arm-code order, as fitted.

    Row 0 is the reference arm, :math:`1 - \\hat\\pi(1 \\mid L)`, and row 1 the treated
    arm, :math:`\\hat\\pi(1 \\mid L)`, both before any floor.
    """
    p1 = np.asarray(record.p1, dtype=float)
    return np.stack([1.0 - p1, p1])


def retention_before(record: Any) -> Any:
    """``(2, n, K)`` each row's probability of staying uncensored just before each target.

    :math:`\\hat S_c(t_k^- \\mid a, L)`, axis 0 the arm code, before any floor.
    """
    return np.asarray(record.retention_before, dtype=float)


def selections(source: Any) -> ContinuousSurvivalNuisances:
    """The library selections of a fit, or the report that already holds them."""
    from ..survival.diagnostics import nuisance_report
    from ..survival.result import ContinuousSurvivalResult

    if isinstance(source, ContinuousSurvivalResult):
        return nuisance_report(source)
    return source


def chosen_index(row: NuisanceSelection) -> int:
    """The position of the candidate the selection chose."""
    return row.members.index(row.chosen)


def _predictions(record: Any, noun: str) -> str:
    if record.cross_fitted:
        return f"Out-of-fold {noun}"
    return f"In-sample {noun} (the fit was not cross-fitted)"


def _target_index(result: ContinuousSurvivalResult, horizon: float | None) -> int:
    from ..exceptions import DataError

    times = np.asarray(result.horizons, dtype=float)
    if horizon is None:
        return int(times.size - 1)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, float)):
        raise DataError(f"horizon= takes one of the fit's target times; got {horizon!r}")
    match = np.flatnonzero(np.isclose(times, float(horizon), rtol=1e-12, atol=0.0))
    if not match.size:
        held = [f"{t:g}" for t in times]
        raise DataError(f"the fit has no target time {horizon!r}; it has {held}")
    return int(match[0])


# --------------------------------------------------------------------------- overlap


def overlap(
    result: ContinuousSurvivalResult,
    *,
    treatment_label: str | None,
    labels: dict[Any, str] | None,
    censoring: bool,
    horizon: float | None,
) -> Figure:
    """The treated arm's probability, mirrored by received arm, and with ``censoring=True``
    each arm's product with the probability of staying uncensored, with the floor."""
    if horizon is not None and not censoring:
        raise ValueError(
            "horizon= chooses the target time of the panels censoring=True adds; pass "
            "censoring=True with it"
        )
    k = _target_index(result, horizon)
    record = result.record
    arm = np.asarray(record.arm)
    pi = treatment_probabilities(record)
    reference, treated = result.arms
    floor = float(record.weight_floor)
    ones = np.ones(arm.size, dtype=float)

    def group(label: str) -> str:
        if labels is not None and label in labels:
            return str(labels[label])
        return f"Units given {label}"

    count = 3 if censoring else 1
    fig, axes = _core.new_figure(count, figsize=(7.0, 1.4 + 2.9 * count))
    _mirror(
        axes[0],
        pi[1],
        ones,
        arm == 1,
        names=(group(treated), group(reference)),
        bounds=(),
    )
    axes[0].set_ylabel("Share of units")
    axes[0].set_xlabel(f"Predicted probability of {treatment_label or treated}")
    time = float(result.horizons[k])
    if censoring:
        product = pi[:, :, None] * retention_before(record)
        limit = Limit(floor, None, "Floor the fit put on this product")
        for axis, code in zip(axes[1:], (1, 0), strict=True):
            label = result.arms[code]
            rows = arm == code
            _mirror(
                axis,
                product[code, rows, k],
                ones[rows],
                np.ones(int(rows.sum()), dtype=bool),
                names=(group(label), None),
                bounds=[limit],
            )
            axis.set_ylabel("Share of units")
            axis.set_xlabel(f"Probability of {label} and staying uncensored just before {time:g}")
    _core.header(
        axes[0],
        "Predicted probability of treatment, by arm",
        f"{_predictions(record, 'predictions')} of the treatment model, before any floor",
        key_sources=axes,
    )
    found = [
        "The fit's floor bounds the product of the treatment probability and the "
        "probability of staying uncensored, not the treatment probability alone, so no floor "
        "is drawn on it" + ("." if censoring else "; pass censoring=True to draw that product.")
    ]
    if censoring:
        found.append(
            f"The lower panels draw, for the units given each arm, that product just before "
            f"target time {time:g}, before the floor ({floor:.3g}); the share below the floor "
            "is printed at its foot. It counts every unit given the arm, followed to that time "
            "or not, so it differs from result.record.bounded_share, which counts the times "
            "each unit was followed."
        )
    _core.add_notes(axes[-1], found)
    return fig


# --------------------------------------------------------------------------- weights


def weights(
    result: ContinuousSurvivalResult,
    *,
    regimens: Sequence[str] | None,
    horizon: int | None,
    cause: str | None,
    repeat: int,
    labels: dict[str, str] | None,
    time_label: str,
    counterfactual: bool,
    title: str | None,
) -> Figure:
    """Boxes of :math:`1 / (\\hat\\pi(a \\mid L)\\hat S_c(t_k^- \\mid a, L))` per target time."""
    from ._weights import _box, _chosen_regimens, _percent_text

    named = [key for key, value in (("horizon", horizon), ("cause", cause)) if value is not None]
    if named:
        raise ValueError(
            f"{'= and '.join(named)}= does not apply to a continuous-time survival fit: its "
            "weights do not depend on the cause, and every target time is drawn"
        )
    if repeat != 1:
        raise ValueError(
            "repeat= does not apply to a continuous-time survival fit, which has one draw"
        )
    record = result.record
    held = list(result.arms)
    names = held if regimens is None else _chosen_regimens(regimens, held)
    arm = np.asarray(record.arm)
    product = treatment_probabilities(record)[:, :, None] * retention_before(record)
    floor = float(record.weight_floor)
    times = tuple(float(t) for t in result.horizons)
    slots = np.arange(len(times), dtype=float)
    fig, (top,) = _core.new_figure(1, figsize=(7.0, 4.8))
    count = len(names)
    step = _BOX_WIDTH / count
    shown_names = [(labels or {}).get(name, name) for name in names]
    limit = 1.0 / floor
    for index, name in enumerate(names):
        code = held.index(name)
        color = _core.series_color(index)
        shown = shown_names[index]
        offset = (index - (count - 1) / 2.0) * step
        rows = arm != code if counterfactual else arm == code
        if not rows.any():
            continue
        cells = product[code, rows, :]
        stats, positions, peaks = [], [], []
        for k, slot in enumerate(slots):
            used = 1.0 / cells[:, k]
            used = used[np.isfinite(used) & (used > 0)]
            if not used.size:
                continue
            stats.append(_box(used))
            positions.append(slot + offset)
            peaks.append(float(used.max()))
        top.bxp(
            stats,
            positions=positions,
            widths=step * 0.85,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color, "linewidth": 1.0},
            medianprops={"color": color, "linewidth": 1.6},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
            manage_ticks=False,
        )
        top.scatter(
            positions,
            peaks,
            marker="x",
            s=22,
            color=color,
            linewidths=1.2,
            zorder=3,
            gid="cleverly-weight-max",
        )
        if count > 1:
            # Identity never rests on colour alone: each box carries its arm's name.
            for position, box in zip(positions, stats, strict=True):
                top.annotate(
                    shown,
                    xy=(position, box["whislo"]),
                    xytext=(0.0, -3.0),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=_core.TICK_SIZE - 1.0,
                    color=_core.INK,
                    gid=_core.GID_SERIES_LABEL,
                )
        for k, slot in enumerate(slots):
            share = float(np.mean(cells[:, k] < floor))
            top.plot(
                [slot + offset - step * 0.5, slot + offset + step * 0.5],
                [limit, limit],
                color=_core.SECONDARY_INK,
                linestyle=(0, (1, 2)),
                linewidth=1.4,
                zorder=2.5,
                label=f"Largest weight the floor allows ({limit:.3g})",
                gid=_core.GID_MARKER,
            )
            top.annotate(
                _percent_text(share),
                xy=(slot + offset, limit),
                xytext=(0.0, 3.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=_core.TICK_SIZE - 1.0,
                color=_core.SECONDARY_INK,
                gid="cleverly-tail-share",
            )
    _core.log_axis(top, "y")
    top.set_ylabel("Weight before the floor (log scale)")
    low, high = top.get_ylim()
    top.set_ylim(low / (1.25 if count > 1 else 1.0), max(high, limit * 1.6))
    top.set_xticks(list(slots))
    top.set_xticklabels([f"{t:g}" for t in times])
    top.set_xlim(-0.7, len(times) - 0.3)
    top.set_xlabel(time_label)
    which = "it did not receive" if counterfactual else "it received"
    _core.header(
        top,
        title or "Weights over time",
        f"Each unit's weight under the arm {which}, just before each target time: "
        + ", ".join(shown_names),
        key_sources=[top],
    )
    population = "given the other arm" if counterfactual else "given that arm"
    found = [
        f"{_predictions(record, 'probabilities')}.",
        "Each weight is 1 / (pi(a | L) S_c(t- | a, L)) at a target time t, before the floor: "
        "the inverse of the probability of the arm times the probability of staying "
        "uncensored just before t. The fit's weights at the observed times up to t are at "
        "most this, because the probability of staying uncensored only falls.",
        "Boxes span the middle half of the weights, whiskers the 1st to 99th percentile, "
        "crosses the largest weight.",
        f"Each box holds every unit {population}, followed to that time or not: the fit's "
        "record keeps no follow-up times, so no effective sample size among the units still "
        "followed is drawn.",
        f"The percentage above each dotted line is the share of those units whose product "
        f"the floor ({floor:.3g}) raises. result.record.bounded_share counts only the times "
        "each unit was followed under the arm it received"
        + (
            ", and result.record.last_time_floor_share counts both arms at the last target time."
            if counterfactual
            else "."
        ),
        "Descriptive: the weights are read from the fit, with no interval.",
    ]
    if counterfactual:
        found.insert(
            1,
            "Each box is labelled by the arm whose weight it draws, over the units that "
            "received the other arm.",
        )
    _core.add_notes(top, found)
    return fig


# --------------------------------------------------------------------------- learners


def _role_title(role: str) -> str:
    if role == "treatment":
        return "Treatment model"
    return role[:1].upper() + role[1:]


def learners(source: Any) -> Figure:
    """One panel per hazard and the treatment model: each candidate's loss, the chosen marked."""
    from ._diagnostics import _finish

    report = selections(source)
    rows = [row for row in report.rows if row.members]
    supplied = [row for row in report.rows if not row.members]
    sizes = [len(row.members) for row in rows]
    fig, axes = _core.new_figure(
        len(rows),
        figsize=(7.0, 1.9 + sum(0.42 * size + 1.0 for size in sizes)),
        height_ratios=[size + 1.5 for size in sizes],
    )
    unscored = False
    labelled = False
    for axis, row in zip(axes, rows, strict=True):
        losses = np.asarray(row.losses, dtype=float)
        positions = np.arange(len(row.members), dtype=float)
        chosen = chosen_index(row)
        finite = np.isfinite(losses)
        others = finite & (positions != chosen)
        axis.scatter(
            losses[others],
            positions[others],
            s=42,
            color=_core.PALETTE[0],
            edgecolor=_core.FACE,
            linewidth=0.8,
            zorder=3,
            gid="cleverly-learner-loss",
        )
        if finite[chosen]:
            axis.scatter(
                [losses[chosen]],
                [positions[chosen]],
                s=130,
                color=_core.PALETTE[0],
                edgecolor=_core.INK,
                linewidth=1.3,
                zorder=4,
                label="Chosen candidate" if not labelled else "_nolegend_",
                gid="cleverly-learner-chosen",
            )
            labelled = True
        for position, value in zip(positions, losses, strict=True):
            if math.isfinite(float(value)):
                axis.annotate(
                    f"{value:.2f}",
                    xy=(float(value), position),
                    xytext=(7.0, 0.0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=_core.TICK_SIZE,
                    color=_core.INK,
                    gid="cleverly-value-label",
                )
        ticks = []
        for index, member in enumerate(row.members):
            if index == chosen and len(row.members) == 1:
                ticks.append(f"{member} (chosen, not cross-validated)")
                unscored = True
            elif index == chosen:
                ticks.append(f"{member} (chosen)")
            elif not finite[index]:
                ticks.append(f"{member} (no finite loss)")
            else:
                ticks.append(str(member))
        axis.set_yticks(positions)
        axis.set_yticklabels(ticks)
        axis.set_ylim(len(row.members) - 0.4, -0.6)
        if finite.any():
            low, high = float(losses[finite].min()), float(losses[finite].max())
            pad = 0.12 * (high - low) if high > low else 0.05 * max(abs(high), 1.0)
            axis.set_xlim(low - pad, high + 2.2 * pad)
        else:
            axis.set_xticks([])
        axis.set_xlabel(
            "Cross-validated log loss"
            if row.role == "treatment"
            else "Cross-validated negative log partial likelihood"
        )
        axis.set_title(
            _role_title(row.role), loc="left", fontsize=_core.LABEL_SIZE, color=_core.INK, pad=4.0
        )
        _core.style_axes(axis, grid="x")
    subtitle = "Each candidate's cross-validated loss; lower is better"
    if report.cross_fitted:
        subtitle += f". First fold's selection of {report.n_folds}"
    found = [
        "Descriptive: each candidate's cross-validated loss, summed over the held-out rows; "
        "no interval is drawn. Each selection keeps one candidate, so the chosen one is "
        "marked and no weights are drawn. Panels do not share a scale."
    ]
    if report.cross_fitted:
        found.append(
            f"The fit selected each model inside each of its {report.n_folds} cross-fitting "
            "folds; result.record keeps the first fold's selections, which this figure draws."
        )
    if unscored:
        found.append(
            "A one-member library is not cross-validated: its candidate is chosen with no "
            "loss, so no point is drawn."
        )
    if supplied:
        found.append(
            "The treatment model is the supplied treatment_learner, which was not selected, "
            "so it has no panel."
        )
    _finish(fig, axes, "Hazard and treatment model selection", subtitle, found, owned=True)
    return fig
