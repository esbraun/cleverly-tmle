"""V15 and V16 for fits over time: learners and calibration per time point and role.

:func:`~cleverly.plotting.learners` and :func:`~cleverly.plotting.calibration` hand these
sources here:

* a survival collaborative fit's selection paths (:class:`SurvivalSelection`), one panel per
  arm and horizon;
* a longitudinal fit's nuisance report (``result.diagnostics.nuisance_models()``): the Super
  Learner weights and risks of each role across time points, and the calibration table of
  each probability model at each time point.

Every value is one the fit stored; nothing is refitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Words for each nuisance role of a longitudinal report.
_ROLE_WORDS = {
    "treatment": "Treatment model",
    "censoring": "Censoring model",
    "competing": "Competing-event model",
    "hazard": "Hazard model",
    "outcome": "Outcome regression",
    "pseudo_outcome": "Pseudo-outcome regression",
}
#: The most calibration panels in one row.
_COLUMNS = 4
#: The message of a source with no Super Learner, shared with the point path.
NO_SUPER_LEARNER = (
    "no nuisance model here was fitted by a Super Learner, so there are no candidate "
    "risks or weights to draw; pass a SuperLearner as a learner, or a "
    "SuperLearnerDiagnostics (a fitted SuperLearner's diagnostics_)"
)


def role_words(role: str) -> str:
    """A nuisance role in words."""
    return _ROLE_WORDS.get(str(role), str(role).replace("_", " ").capitalize())


# ------------------------------------------------------------------- the rows to draw


class Chosen:
    """The rows of one longitudinal report to draw, with the coordinates chosen."""

    def __init__(
        self,
        rows: list[Any],
        regimen: str | None,
        regimens: list[str],
        horizon: int | None,
        cause: str | None,
        causes: list[str],
    ) -> None:
        self.rows = rows
        self.regimen = regimen
        self.regimens = regimens
        self.horizon = horizon
        self.cause = cause
        self.causes = causes

    def title(self, role: str) -> str:
        """A role in words, with its cause on a fit with several causes."""
        words = role_words(role)
        return f"{words}, cause {self.cause}" if self.cause_specific(role) else words

    def cause_specific(self, role: str) -> bool:
        """Whether the drawn rows of ``role`` belong to the chosen cause alone."""
        return len(self.causes) > 1 and any(
            row.cause is not None for row in self.rows if str(row.role) == role
        )

    def notes(self) -> list[str]:
        """Which regimen and cause are drawn, and how to choose others."""
        found = []
        if len(self.causes) > 1:
            found.append(
                f"The models of each cause are drawn for cause {self.cause}; pass cause= for "
                f"another of {self.causes}. Models shared by every cause are always drawn."
            )
        if self.regimen is None:
            return found
        where = f"regimen {self.regimen}"
        if self.horizon is not None:
            where += f", horizon {self.horizon}"
        if self.cause is not None:
            where += f", cause {self.cause}"
        found.append(f"The outcome regressions drawn are those of {where}.")
        if len(self.regimens) > 1:
            found.append(f"Pass regimen= for another of {self.regimens}.")
        return found


def choose_rows(report: Any, regimen: str | None, horizon: int | None, cause: str | None) -> Chosen:
    """The rows of one regimen, horizon and cause, with the models every regimen shares.

    The defaults are the first regimen, the first cause and the largest horizon; the cause
    and horizon defaults are the ones :func:`~cleverly.longitudinal.node_weights` takes. A
    model of one cause (the hazard of that cause on a competing-risks fit) is drawn only for
    the chosen cause, so two causes' models never share a panel.
    """
    from ..exceptions import DataError

    rows = list(report.rows)
    causes = [str(name) for name in dict.fromkeys(row.cause for row in rows) if name is not None]
    if cause is not None and cause not in causes:
        raise DataError(f"the report holds no cause {cause!r}; it holds {causes}")
    chosen_cause = cause if cause is not None else (causes[0] if causes else None)
    rows = [row for row in rows if row.cause is None or row.cause == chosen_cause]
    outcome = [row for row in rows if row.regimen is not None]
    regimens = [str(label) for label in dict.fromkeys(row.regimen for row in outcome)]
    if regimen is not None and regimen not in regimens:
        raise DataError(f"the report holds no regimen {regimen!r}; it holds {regimens}")
    label = regimen if regimen is not None else (regimens[0] if regimens else None)
    pool = [row for row in outcome if row.regimen == label]
    horizons = sorted({int(row.horizon) for row in pool if row.horizon is not None})
    if horizon is not None and horizon not in horizons:
        raise DataError(f"the regimen {label!r} has no horizon {horizon!r}; it has {horizons}")
    chosen_horizon = horizon if horizon is not None else (horizons[-1] if horizons else None)
    if horizons:
        pool = [row for row in pool if row.horizon == chosen_horizon]
    mechanism = [row for row in rows if row.regimen is None]
    return Chosen(mechanism + pool, label, regimens, chosen_horizon, chosen_cause, causes)


def _by_role(rows: Sequence[Any]) -> dict[str, list[Any]]:
    roles: dict[str, list[Any]] = {}
    for row in rows:
        roles.setdefault(str(row.role), []).append(row)
    for members in roles.values():
        members.sort(key=lambda row: int(row.time))
    return roles


def _repeats(report: Any) -> list[str]:
    count = int(getattr(report, "n_repeats", 1))
    if count > 1:
        drawn = int(getattr(report, "reported_repeat", 1))
        return [f"The fit combined {count} cross-fitting draws; the figure describes draw {drawn}."]
    return []


# ------------------------------------------------------------------- learners (C2)


def node_learners(
    report: Any,
    *,
    regimen: str | None,
    horizon: int | None,
    cause: str | None,
    time_label: str,
) -> Figure:
    """Each role's Super Learner weights (top) and candidate risks (bottom) by time point."""
    chosen = choose_rows(report, regimen, horizon, cause)
    fitted = [
        row
        for row in chosen.rows
        if row.model is not None and getattr(row.model, "learner_risks", None)
    ]
    if not fitted:
        raise ValueError(NO_SUPER_LEARNER)
    roles = _by_role(fitted)
    count = len(roles)
    fig, axes = _core.new_figure(
        2, ncols=count, figsize=(3.6 * count + 0.8, 6.6), height_ratios=[1.0, 1.0]
    )
    tops, bottoms = axes[:count], axes[count:]
    gaps = False
    grey: list[str] = []
    point = time_label.lower()
    # One colour per candidate across every panel: the three with the largest mean weight
    # over all the fitted models are coloured, the rest grey.
    every = list(dict.fromkeys(name for row in fitted for name in row.model.learner_risks))
    mean_weight = {
        name: float(np.mean([float(row.model.learner_weights.get(name, 0.0)) for row in fitted]))
        for name in every
    }
    colored = sorted(every, key=lambda name: -mean_weight[name])[: _core.MAX_SERIES]
    for top, bottom, (role, members) in zip(tops, bottoms, roles.items(), strict=True):
        names = list(dict.fromkeys(name for row in members for name in row.model.learner_risks))
        times = [int(row.time) for row in members]
        span = range(min(times), max(times) + 1)
        gaps = gaps or len(times) != len(span)
        base = np.zeros(len(members))
        for name in names:
            color = _core.series_color(colored.index(name)) if name in colored else _core.REFERENCE
            if name not in colored and name not in grey:
                grey.append(name)
            heights = np.asarray(
                [float(row.model.learner_weights.get(name, 0.0)) for row in members], dtype=float
            )
            label = name if name in colored else "Other candidates"
            bars = top.bar(
                times,
                heights,
                bottom=base,
                width=0.6,
                color=color,
                edgecolor=_core.FACE,
                linewidth=0.6,
            )
            for bar in bars:
                bar.set_gid("cleverly-learner-weight")
                bar.set_url(name)
            # One key entry per candidate (one for every grey one), from its first bar.
            keyed = {text for axis in tops for text in axis.get_legend_handles_labels()[1]}
            if bars.patches and label not in keyed:
                bars.patches[0].set_label(label)
            base = base + heights
            risks = np.asarray(
                [float(row.model.learner_risks.get(name, float("nan"))) for row in members],
                dtype=float,
            )
            (line,) = bottom.plot(
                times,
                risks,
                marker="o",
                markersize=4,
                color=color,
                linewidth=1.4 if name in colored else 0.9,
                gid="cleverly-learner-risk",
                label="_nolegend_",
            )
            line.set_url(name)
        top.set_ylim(0.0, 1.12)
        top.set_ylabel("Ensemble weight")
        top.set_xticks(list(span))
        _panel_title(top, chosen.title(role))
        loss = getattr(members[0].model, "learner_loss", None)
        bottom.set_ylabel("Cross-validated risk" + (f" ({_loss_words(loss)})" if loss else ""))
        bottom.set_xticks(list(span))
        bottom.set_xlabel(time_label)
    drawn_regimen = f"; regimen {chosen.regimen}" if chosen.regimen is not None else ""
    _core.figure_header(
        fig,
        "Learner weights at each time point",
        f"Super Learner weight (top) and each candidate's cross-validated risk (bottom), by "
        f"{point}{drawn_regimen}. Lower risk is better",
        key_sources=tops,
    )
    found = [
        "Descriptive: each weight and risk is the mean over the model's Super Learner fits, "
        "one per cross-fitting fold; no interval is drawn.",
        *_repeats(report),
        *chosen.notes(),
    ]
    if grey:
        found.append(
            "Grey bars and lines are the candidates beyond the three with the largest mean "
            "weight: " + ", ".join(grey) + "."
        )
    if gaps:
        found.append(f"A {point} with no bar fitted no Super Learner for that role there.")
    _core.add_notes(fig, found)
    return fig


def _loss_words(loss: str) -> str:
    from ._diagnostics import _loss_text

    return _loss_text(loss)


def _panel_title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=_core.LABEL_SIZE, color=_core.INK, pad=4.0)


# ------------------------------------------------------------------- calibration (C5)


def node_calibration(
    report: Any,
    *,
    regimen: str | None,
    horizon: int | None,
    cause: str | None,
    weighted: bool | None,
    time_label: str,
) -> Figure:
    """One calibration panel per probability model and time point."""
    from .._refusal import RefusalReason
    from ..exceptions import CapabilityError
    from ._diagnostics import (
        _Context,
        _finish,
        _probability_panel,
        _probability_text,
        _square_limits,
    )

    chosen = choose_rows(report, regimen, horizon, cause)
    rows = [
        row
        for row in chosen.rows
        if row.model is not None
        and getattr(row.model, "kind", None) == "probability"
        and len(row.model.calibration.get("n", ()))
    ]
    if not rows:
        # The point path's refusal, word for word: the report holds no table to draw.
        raise CapabilityError(
            "no nuisance model here reports a calibration table of predicted probabilities",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    roles = _by_role(rows)
    columns = min(_COLUMNS, max(len(members) for members in roles.values()))
    layout: list[tuple[int, Any]] = []
    row_index = 0
    for members in roles.values():
        for index, member in enumerate(members):
            layout.append((row_index * columns + index, member))
        row_index += math.ceil(len(members) / columns)
    fig, axes = _core.new_figure(
        row_index, ncols=columns, figsize=(3.1 * columns + 0.6, 3.0 * row_index + 2.1)
    )
    values = np.concatenate(
        [
            np.asarray(member.model.calibration[column], dtype=float)
            for _, member in layout
            for column in ("mean_predicted", "mean_observed")
        ]
    )
    low, high = _square_limits(values)
    used = {position for position, _ in layout}
    point = time_label.lower()
    for position, member in layout:
        _probability_panel(axes[position], member.model, _core.PALETTE[0], low, high)
        _panel_title(axes[position], f"{chosen.title(member.role)}, {point} {member.time}")
    for index, axis in enumerate(axes):
        if index not in used:
            fig.delaxes(axis)
    drawn = [axes[position] for position, _ in layout]
    bases = {member.evaluation for _, member in layout}
    context = _Context(
        evaluation=bases.pop() if len(bases) == 1 else None,
        n_repeats=int(getattr(report, "n_repeats", 1)),
        reported_repeat=int(getattr(report, "reported_repeat", 1)),
        weighted=weighted,
    )
    _, found = _probability_text([member.model for _, member in layout], context)
    basis = {
        "out_of_fold": "out-of-fold predictions",
        "in_sample": "in-sample predictions (the fit used one fold)",
    }.get(str(context.evaluation), "predictions")
    _finish(
        fig,
        drawn,
        "Calibration at each time point",
        f"Observed rate against mean predicted probability, by model and {point}; {basis}"
        + (f"; regimen {chosen.regimen}" if chosen.regimen is not None else ""),
        [*found, *chosen.notes()],
        owned=True,
    )
    return fig


# ------------------------------------------------------------------- survival CTMLE (C1)


def chosen_position(selection: Any) -> int:
    """The path position the cross-validated risk chose: ``selection.selected``."""
    return int(selection.selected)


def selection_paths(selections: dict[tuple[str, int], Any]) -> Figure:
    """One panel per arm and horizon: the cross-validated risk along the selection path."""
    from matplotlib.ticker import FixedLocator

    from ._diagnostics import _finish

    arms = list(dict.fromkeys(arm for arm, _ in selections))
    horizons = sorted({horizon for _, horizon in selections})
    fig, axes = _core.new_figure(
        len(arms),
        ncols=len(horizons),
        figsize=(3.8 * len(horizons) + 0.8, 3.3 * len(arms) + 2.0),
    )
    used: set[int] = set()
    penalized = False
    last = None
    for row, arm in enumerate(arms):
        for column, horizon in enumerate(horizons):
            position = row * len(horizons) + column
            selection = selections.get((arm, horizon))
            if selection is None:
                continue
            used.add(position)
            axis = axes[position]
            risk = np.asarray(selection.cv_risk, dtype=float)
            steps = np.arange(len(risk), dtype=float)
            chosen = chosen_position(selection)
            penalized = penalized or bool(selection.penalized)
            last = int(selection.diagnostic_horizon)
            axis.plot(
                steps,
                risk,
                color=_core.PALETTE[0],
                marker="o",
                markersize=5,
                linewidth=1.6,
                markeredgecolor=_core.FACE,
                markeredgewidth=0.8,
                zorder=3,
                gid="cleverly-ctmle-risk",
                label="_nolegend_",
            )
            axis.scatter(
                [steps[chosen]],
                [risk[chosen]],
                s=120,
                color=_core.PALETTE[0],
                edgecolor=_core.INK,
                linewidth=1.2,
                zorder=4,
                gid="cleverly-ctmle-chosen",
            )
            _core.reference_line(
                axis,
                float(steps[chosen]),
                orientation="vertical",
                label="Chosen position",
                kind="marker",
            )
            axis.xaxis.set_major_locator(FixedLocator(list(steps)))
            axis.set_xticklabels(
                list(selection.move_labels),
                rotation=30,
                ha="right",
                fontsize=_core.TICK_SIZE - 1.0,
            )
            finite = risk[np.isfinite(risk)]
            if finite.size:
                spread = float(finite.max() - finite.min()) or abs(float(finite.max())) or 1.0
                axis.set_ylim(
                    float(finite.min()) - 0.15 * spread, float(finite.max()) + 0.15 * spread
                )
            _panel_title(axis, f"{arm}, horizon {horizon}")
            axis.set_ylabel(
                ("Cross-validated criterion" if selection.penalized else "Cross-validated risk")
                if column == 0
                else ""
            )
    for index, axis in enumerate(axes):
        if index not in used:
            fig.delaxes(axis)
    drawn = [axes[index] for index in sorted(used)]
    found = [
        "Descriptive: the cross-validated "
        + ("criterion" if penalized else "risk")
        + " of each move count, summed over the selection folds; no interval is drawn.",
    ]
    if penalized:
        found.append(
            "The criterion adds the source's printed penalty (its equation 24) to the loss."
        )
    found.append(
        "Each tick names the move that position adds: a covariate in the treatment model "
        "(g1) or a term in the censoring model (g2); M0 is the starting model."
    )
    if last is not None:
        found.append(f"result.mechanism holds each arm's selection at horizon {last}.")
    _finish(
        fig,
        drawn,
        "Collaborative selection path",
        "Cross-validated "
        + ("criterion" if penalized else "risk")
        + " of each position along each arm's search; chosen position marked. Lower is better",
        found,
        owned=True,
    )
    return fig


__all__ = ["choose_rows", "node_calibration", "node_learners", "selection_paths"]
