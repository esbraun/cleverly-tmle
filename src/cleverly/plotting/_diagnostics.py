"""V8, V15, V16, V17 and V19: diagnostics that read only what a fit already holds.

* :func:`calibration`: do the predicted effects (a fitted effect curve) or probabilities
  (the treatment, outcome and other probability models) match what was measured, bin by bin?
* :func:`learners`: which candidate learners carry a Super Learner, and how sharp was a
  collaborative fit's choice of treatment model?
* :func:`repeats`: does the estimate depend on how the cross-fitting split was drawn?
* :func:`bootstrap`: is the bootstrap distribution close to normal on the scale the Wald
  interval is built on?
* :func:`influence`: how few units carry the variance of the influence values?

Every quantity is descriptive and none carries an interval the object does not hold;
:func:`~cleverly.plotting.notes` says so for each figure. Nothing is refitted and nothing is drawn
at random.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._refusal import RefusalReason
from ..exceptions import CapabilityError
from . import _core
from ._forest import _axis_label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The loss words of a Super Learner and of a collaborative selection.
_LOSS_WORDS = {
    "mse": "squared error",
    "squared": "squared error",
    "neg_log_likelihood": "negative log-likelihood",
    "loglik": "negative log-likelihood",
}

#: The treatment roles whose propensity panel is not a fitted estimate of the treatment law.
_WORKING = "collaborative_working_model"
_DECLARED = "declared_treatment_law"


def _loss_text(loss: str) -> str:
    return _LOSS_WORDS.get(str(loss), str(loss).replace("_", " "))


def _percent(share: float) -> str:
    """A share as a percentage with one decimal below 10%, none above."""
    value = 100.0 * share
    return f"{value:.1f}%" if value < 10.0 else f"{value:.0f}%"


def _square(ax: Axes) -> None:
    """Equal axis scales, so the 45-degree line is at 45 degrees, centred in its cell."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")


def _square_limits(values: Any) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.06 * (high - low) if high > low else 0.05
    return low - pad, high + pad


def _diagonal(ax: Axes, low: float, high: float, label: str = "Perfect calibration") -> Any:
    """The 45-degree line, in the reference style."""
    (line,) = ax.plot(
        [low, high],
        [low, high],
        color=_core.REFERENCE,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=1.5,
        label=label,
        gid=_core.GID_REFERENCE,
    )
    return line


def _bins(ax: Axes, x: Any, y: Any, rows: Any, color: str) -> None:
    """Unjoined bin points whose area grows with the rows in the bin."""
    counts = np.asarray(rows, dtype=float)
    top = float(np.nanmax(counts)) if counts.size and np.nanmax(counts) > 0 else 1.0
    ax.scatter(
        x,
        y,
        s=30.0 + 170.0 * counts / top,
        color=color,
        edgecolor=_core.FACE,
        linewidth=0.8,
        zorder=3,
        gid="cleverly-calibration-points",
    )


def _finish(
    fig: Figure,
    axes: Sequence[Axes],
    title: str,
    subtitle: str | None,
    found: Sequence[str],
    *,
    owned: bool,
) -> None:
    """The title, subtitle and key: at the figure's edge on a figure the plot made.

    A figure the plot created takes the figure-level header, anchored at the figure's left
    edge and wrapped to its width, so a square panel or a column of panels never pushes it
    past the edge. On the caller's axes it goes on that axes. The notes are recorded, not
    drawn.
    """
    if owned:
        _core.figure_header(fig, title, subtitle, key_sources=axes)
        _core.add_notes(fig, found)
    else:
        _core.header(axes[0], title, subtitle, key_sources=axes)
        _core.add_notes(axes[-1], found)


# ------------------------------------------------------------------------- calibration


def _curve(calibration: Any, title: str | None, ax: Axes | None) -> Figure:
    bins = calibration.bins
    x = np.asarray(bins["mean_prediction"], dtype=float)
    y = np.asarray(bins["pseudo_outcome_contrast"], dtype=float)
    rows = np.asarray(bins["rows"], dtype=float) if "rows" in bins else np.ones_like(x)
    ratio = str(calibration.scale) != "difference"
    kept = np.isfinite(x) & np.isfinite(y)
    gap = (
        "no log ratio (a mean doubly robust pseudo-outcome that is not positive)"
        if ratio
        else "no finite contrast"
    )
    if not kept.any():
        raise ValueError(f"no bin of this calibration can be drawn: every bin has {gap}")
    fig, axis = _core.figure_for(ax, figsize=(5.2, 5.6))
    low, high = _square_limits(np.concatenate([x[kept], y[kept]]))
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    _diagonal(axis, low, high)
    _bins(axis, x[kept], y[kept], rows[kept], _core.PALETTE[0])
    _square(axis)
    axis.grid(True, axis="both", color=_core.GRID, linewidth=0.8)
    axis.set_xlabel(
        "Mean predicted log ratio in the bin" if ratio else "Mean predicted effect in the bin"
    )
    axis.set_ylabel(
        "Measured log ratio (doubly robust contrast)"
        if ratio
        else "Measured effect (doubly robust contrast)"
    )
    numbers = f"{float(calibration.slope):.2f} and intercept {float(calibration.intercept):.2f}"
    subtitle = (
        f"Calibration slope {numbers} from the one-step log-ratio loss "
        "(1 and 0 for a calibrated curve)"
        if ratio
        else f"Linear calibration slope {numbers} (1 and 0 for a calibrated curve)"
    )
    found = [
        f"Descriptive: bins of out-of-fold refits ({int(calibration.n_folds)} folds); point "
        "area shows the rows in the bin; no interval is drawn.",
        "The slope and intercept minimise the one-step log-ratio loss on (1, prediction), and "
        "each bin is the log of the ratio of its mean doubly robust pseudo-outcomes; the "
        "slope is not the slope of the drawn bins."
        if ratio
        else "The slope and intercept come from the least-squares projection of the doubly "
        "robust pseudo-outcome on (1, prediction); they are not the slope of the drawn bins.",
    ]
    dropped = int((~kept).sum())
    if dropped:
        found.append(
            f"{dropped} bin{' has' if dropped == 1 else 's have'} {gap} and "
            f"{'is' if dropped == 1 else 'are'} not drawn."
        )
    _finish(
        fig,
        [axis],
        title or "Predicted and measured effects",
        subtitle,
        found,
        owned=ax is None,
    )
    return fig


@dataclass(frozen=True)
class _Context:
    """What the source says about the reports beyond the reports themselves."""

    evaluation: str | None = None
    role: str | None = None
    n_repeats: int = 1
    reported_repeat: int = 1
    #: ``True`` or ``False`` when the data are known, ``None`` for a bare report.
    weighted: bool | None = None


def _is_propensity(name: str) -> bool:
    return name == "propensity" or name.startswith("propensity[")


def _model_words(name: str, role: str | None = None) -> str:
    """A nuisance report's name in words, with the propensity's role where it changes it."""
    arm = re.fullmatch(r"propensity\[(.*)\]", str(name))
    if _is_propensity(str(name)):
        stem = {
            _WORKING: "Collaborative working treatment model",
            _DECLARED: "Declared treatment probabilities",
        }.get(str(role), "Treatment model")
        return f"{stem}, arm {arm[1]} (one vs rest)" if arm else stem
    return {
        "outcome": "Outcome model",
        "missingness": "Missingness model (outcome observed)",
        "intermediate": "Intermediate model",
    }.get(str(name), str(name))


def _probability_models(source: Any) -> tuple[list[Any], _Context]:
    """The reports to draw and what the source states about them."""
    weighted: bool | None = None
    _refuse_unread(source, "calibration")
    if hasattr(source, "diagnostics") and hasattr(source.diagnostics, "nuisance_models"):
        weights = getattr(getattr(source, "data", None), "weights", None)
        if weights is not None:
            values = np.asarray(weights, dtype=float)
            weighted = bool(values.size and not np.allclose(values, values[0]))
        source = source.diagnostics.nuisance_models()
        _refuse_unread(source, "calibration")
        if not hasattr(source, "models"):
            raise ValueError(
                "calibration() draws the point-treatment nuisance reports; this result "
                "reports its nuisance models another way"
            )
    if hasattr(source, "models"):
        models = list(source.models)
        context = _Context(
            evaluation=getattr(source, "evaluation", None),
            role=getattr(source, "treatment_role", None),
            n_repeats=int(getattr(source, "n_repeats", 1)),
            reported_repeat=int(getattr(source, "reported_repeat", 1)),
            weighted=weighted,
        )
    else:
        models, context = [source], _Context(evaluation=getattr(source, "evaluation", None))
    drawn = [
        model
        for model in models
        if getattr(model, "kind", None) == "probability" and len(model.calibration.get("n", ()))
    ]
    if not drawn:
        mean_only = bool(models) and all(
            getattr(model, "kind", None) == "conditional mean" for model in models
        )
        raise CapabilityError(
            "no probability model with a calibration table to draw: a conditional-mean "
            "(continuous outcome) model has no probability to calibrate"
            if mean_only
            else "no nuisance model here reports a calibration table of predicted probabilities",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return drawn, context


def _slope_text(model: Any) -> str:
    slope = float(model.metrics.get("calibration_slope", float("nan")))
    if math.isfinite(slope):
        return f"{slope:.2f}"
    omission = getattr(model, "calibration_omission", None)
    return f"not available ({omission})" if omission else "not available"


def _probability_panel(ax: Axes, model: Any, color: str, low: float, high: float) -> None:
    table = model.calibration
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    _diagonal(ax, low, high)
    _bins(ax, table["mean_predicted"], table["mean_observed"], table["n"], color)
    _square(ax)
    ax.grid(True, axis="both", color=_core.GRID, linewidth=0.8)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rate")


def _probability_text(models: list[Any], context: _Context) -> tuple[str, list[str]]:
    """The subtitle and the notes of the probability calibration figure."""
    bases = {getattr(model, "evaluation", None) or context.evaluation for model in models}
    basis = bases.pop() if len(bases) == 1 else None
    role = context.role
    sloped = [
        model
        for model in models
        if not (_is_propensity(model.name) and role in (_WORKING, _DECLARED))
    ]
    slopes = "; ".join(f"{_model_words(model.name, role)} {_slope_text(model)}" for model in sloped)
    if basis == "in_sample":
        subtitle = "In-sample predictions (the fit used one fold): no calibration slope is shown"
    elif not slopes:
        subtitle = "No calibration slope is shown for these models"
    elif basis == "out_of_fold":
        subtitle = f"Out-of-fold calibration slope (1 = calibrated): {slopes}"
    else:
        subtitle = f"Calibration slope (1 = calibrated): {slopes}"
    where = {
        "out_of_fold": "cross-fitted (out-of-fold) predictions",
        "in_sample": "in-sample predictions (the fit used one fold)",
    }.get(str(basis), "predictions whose evaluation basis the report does not record")
    found = [
        f"Descriptive: bins of {where}; no interval is drawn. Point area shows the rows in the bin."
    ]
    if basis == "in_sample":
        found.append(
            "In sample, an unpenalized logistic model with an intercept has a calibration "
            "slope of 1 by construction, so no slope is shown and the bins are not evidence "
            "of calibration on new data."
        )
    elif basis is None:
        found.append(
            "Whether the slope is out of fold is not recorded here; plot the fit, or its "
            "diagnostics.nuisance_models(), to see."
        )
    if any(_is_propensity(model.name) for model in models):
        if role == _WORKING:
            found.append(
                "The collaborative search chose this treatment model for targeting; it is not "
                "an estimate of the treatment law, so its calibration is not a finding and no "
                "slope is shown."
            )
        elif role == _DECLARED:
            found.append(
                "The treatment probabilities are the ones the design declares, not a fitted "
                "model: the panel compares that declaration with the treatment observed, and "
                "no slope is shown."
            )
    if context.weighted is None:
        found.append("The observed rate is weighted when the data carry weights.")
    elif context.weighted:
        found.append("The observed rate and the bins are weighted by the data's weights.")
    if context.n_repeats > 1:
        found.append(
            f"The fit combined {context.n_repeats} cross-fitting draws; the bins describe "
            f"draw {context.reported_repeat}."
        )
    return subtitle, found


@_core.labelled
def calibration(
    source: Any,
    *,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot whether predictions match what was measured, bin by bin.

    Use it to see where along the range of predictions a model over- or understates, which
    a single slope hides. Two kinds of source are drawn.

    A ``CurveCalibration`` (from ``curve.diagnostics.calibration()``) plots each bin's
    measured effect, a doubly robust contrast, against the bin's mean predicted effect. The
    calibration slope and intercept are in the subtitle. On the difference scale they come
    from the least-squares projection of the doubly robust pseudo-outcome on the prediction,
    and on the log-ratio scale they minimise the one-step log-ratio loss; either way they are
    not the slope of the drawn bins.

    A :class:`~cleverly.validation.NuisanceModelReport`, a fit's
    ``result.diagnostics.nuisance_models()`` or the result itself plots the observed rate
    against the mean predicted probability of each probability model (the treatment model,
    a binary outcome model, and the missingness and intermediate models a fit has), with
    point area showing the rows in the bin. The logistic calibration slope of out-of-fold
    predictions is in the subtitle. An in-sample fit shows no slope, because an unpenalized
    logistic model with an intercept has an in-sample slope of 1 by construction, and a
    collaborative working treatment model or a declared treatment law shows none either,
    because neither is an estimate of the treatment law.

    Bins are descriptive: no interval is drawn.

    Parameters
    ----------
    source : CurveCalibration or NuisanceModelReport or NuisanceDiagnostics or TMLEResult
        What to plot.
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
        Axes to draw on. Only for a single panel.

    Returns
    -------
    Figure
        One panel per probability model, or one panel for a curve.

    Raises
    ------
    ValueError
        When the source holds no probability model with a calibration table, a curve
        calibration has no finite bin, or ``ax`` is given for more than one panel.
    CapabilityError
        When the source is a longitudinal, continuous-time survival or adaptive TMLE fit, or
        its nuisance report, whose models this plot does not draw yet.

    See Also
    --------
    cleverly.plotting.learners : Which learners carry the models.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say the bins are descriptive,
    which predictions they use, and how the slope and intercept are computed.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.plotting import calibration
    >>> rng = np.random.default_rng(0)
    >>> w = rng.normal(size=(300, 2))
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-0.6 * w[:, 0])))
    >>> y = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * w[:, 1] + 0.8 * a - 0.3))))
    >>> frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    >>> study = CausalStudy(
    ...     frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LogisticRegression(max_iter=1000),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     n_folds=3,
    ...     random_state=0,
    ... )
    >>> fig = calibration(result)
    >>> fig.texts[0].get_text()
    'Calibration of predicted probabilities'
    >>> plt.close(fig)
    """
    if hasattr(source, "bins") and hasattr(source, "slope") and hasattr(source, "intercept"):
        return _curve(source, title or None, ax)
    models, context = _probability_models(source)
    count = len(models)
    if ax is not None and count > 1:
        raise ValueError(
            "ax= takes one panel, but this source has several probability models; "
            "pass one NuisanceModelReport"
        )
    if count == 1:
        fig, axis = _core.figure_for(ax, figsize=(5.2, 5.6))
        axes = [axis]
    else:
        ncols = min(count, 3)
        nrows = math.ceil(count / ncols)
        fig, axes = _core.new_figure(
            nrows, ncols=ncols, figsize=(3.4 * ncols + 0.4, 3.1 * nrows + 2.1)
        )
        for extra in axes[count:]:
            fig.delaxes(extra)
        axes = axes[:count]
    values = np.concatenate(
        [
            np.asarray(model.calibration[column], dtype=float)
            for model in models
            for column in ("mean_predicted", "mean_observed")
        ]
    )
    low, high = _square_limits(values)
    for axis, model in zip(axes, models, strict=True):
        # One model per panel: the panel title names it, and one colour serves every panel.
        _probability_panel(axis, model, _core.PALETTE[0], low, high)
        words = _model_words(model.name, context.role)
        if ax is None:
            axis.set_title(words, loc="left", fontsize=_core.LABEL_SIZE, color=_core.INK, pad=4.0)
        else:
            _core.corner_label(axis, 0.97, words, _core.PALETTE[0])
    subtitle, found = _probability_text(models, context)
    heading = title or "Calibration of predicted probabilities"
    _finish(fig, axes, heading, subtitle, found, owned=ax is None)
    return fig


# --------------------------------------------------------------------------- learners


@dataclass(frozen=True)
class _Library:
    """One Super Learner library to draw: the candidates' risks and weights."""

    label: str | None
    names: tuple[str, ...]
    risks: np.ndarray
    weights: np.ndarray
    loss: str | None


def _library_rows(fig: Figure, axes: list[Axes], libraries: list[_Library]) -> None:
    for row, library in enumerate(libraries):
        left, right = axes[2 * row], axes[2 * row + 1]
        right.sharey(left)
        order = np.argsort(library.risks, kind="stable")
        count = len(library.names)
        positions = np.arange(count)
        for axis, values, gid in (
            (left, library.risks[order], "cleverly-learner-risk"),
            (right, library.weights[order], "cleverly-learner-weight"),
        ):
            bars = axis.barh(
                positions,
                np.nan_to_num(values, nan=0.0),
                height=0.62,
                color=_core.PALETTE[0],
                edgecolor=_core.FACE,
                linewidth=0.6,
            )
            for bar in bars:
                bar.set_gid(gid)
            finite = values[np.isfinite(values)]
            top = float(finite.max()) if finite.size else 1.0
            axis.set_xlim(0.0, top * 1.25 if top > 0 else 1.0)
            for position, value in zip(positions, values, strict=True):
                axis.annotate(
                    "not scored"
                    if not math.isfinite(float(value))
                    else f"{value:.2f}"
                    if gid.endswith("weight")
                    else f"{value:.3g}",
                    xy=(float(np.nan_to_num(value, nan=0.0)), position),
                    xytext=(4.0, 0.0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=_core.TICK_SIZE,
                    color=_core.INK,
                    gid="cleverly-value-label",
                )
            _core.style_axes(axis, grid="x")
        left.set_yticks(positions)
        left.set_yticklabels([library.names[index] for index in order])
        left.set_ylim(count - 0.4, -0.6)
        left.set_ylabel(library.label or "Candidate learner")
        right.tick_params(labelleft=False)
        right.set_xlim(0.0, 1.18)
        right.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        loss = f" ({_loss_text(library.loss)})" if library.loss else ""
        left.set_xlabel(f"Cross-validated risk{loss}")
        right.set_xlabel("Ensemble weight")


def _libraries(libraries: list[_Library], title: str | None, extra: Sequence[str] = ()) -> Figure:
    counts = [len(library.names) for library in libraries]
    height = 1.9 + sum(0.42 * count + 0.9 for count in counts)
    fig, axes = _core.new_figure(
        len(libraries), ncols=2, figsize=(7.0, height), height_ratios=[c + 1.5 for c in counts]
    )
    _library_rows(fig, axes, libraries)

    def lead(library: _Library) -> tuple[str, str]:
        best = library.names[int(np.nanargmin(library.risks))]
        heavy = library.names[int(np.nanargmax(library.weights))]
        return best, heavy

    if len(libraries) == 1:
        best, heavy = lead(libraries[0])
        subtitle = (
            f"Lower risk is better. Lowest cross-validated risk: {best}. Largest weight: {heavy}"
        )
    else:
        subtitle = "Lower risk is better. " + "; ".join(
            f"{library.label}: lowest risk {lead(library)[0]}, largest weight {lead(library)[1]}"
            for library in libraries
        )
    found = [
        "Descriptive: cross-validated risk of each candidate. The ensemble's own risk is not "
        "drawn, because it is scored on the predictions its weights were fitted to and so "
        "flatters it.",
        *extra,
    ]
    _finish(fig, axes, title or "Learner risks and weights", subtitle, found, owned=True)
    return fig


def _from_reports(models: Sequence[Any], role: str | None) -> Figure:
    libraries = [
        _Library(
            label=_model_words(model.name, role),
            names=tuple(str(name) for name in model.learner_risks),
            risks=np.asarray(list(model.learner_risks.values()), dtype=float),
            weights=np.asarray(
                [
                    float(model.learner_weights.get(name, float("nan")))
                    for name in model.learner_risks
                ],
                dtype=float,
            ),
            loss=getattr(model, "learner_loss", None),
        )
        for model in models
        if getattr(model, "learner_risks", None)
    ]
    if not libraries:
        raise ValueError(
            "no nuisance model here was fitted by a Super Learner, so there are no candidate "
            "risks or weights to draw; pass a SuperLearner as a learner, or a "
            "SuperLearnerDiagnostics (a fitted SuperLearner's diagnostics_)"
        )
    return _libraries(
        libraries,
        None,
        [
            "Each risk and weight is the mean over the model's Super Learner fits, one per "
            "cross-fitting fold."
        ],
    )


def _ctmle(selection: Any, ax: Axes | None) -> Figure:
    risk = np.asarray(selection.cv_risk, dtype=float)
    steps = np.arange(len(risk), dtype=float)
    chosen = int(selection.selected)
    fig, axis = _core.figure_for(ax)
    (line,) = axis.plot(
        steps,
        risk,
        color=_core.PALETTE[0],
        marker="o",
        markersize=6,
        linewidth=1.8,
        markeredgecolor=_core.FACE,
        markeredgewidth=0.8,
        zorder=3,
        gid="cleverly-ctmle-risk",
    )
    line.set_label("_nolegend_")
    axis.scatter(
        [steps[chosen]],
        [risk[chosen]],
        s=150,
        color=_core.PALETTE[0],
        edgecolor=_core.INK,
        linewidth=1.2,
        zorder=4,
        gid="cleverly-ctmle-chosen",
    )
    _core.reference_line(
        axis, float(steps[chosen]), orientation="vertical", label="Chosen candidate", kind="marker"
    )
    from matplotlib.ticker import MaxNLocator

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    spread = float(np.nanmax(risk) - np.nanmin(risk))
    axis.set_ylim(
        float(np.nanmin(risk)) - 0.15 * (spread or 1.0),
        float(np.nanmax(risk)) + 0.15 * (spread or 1.0),
    )
    loss = _loss_text(selection.loss)
    penalized = bool(getattr(selection, "penalized", False))
    axis.set_xlabel("Candidate along the search path (0 = first)")
    axis.set_ylabel(
        f"Cross-validated criterion\n({loss} + penalty)"
        if penalized
        else f"Cross-validated risk ({loss})"
    )
    labels = tuple(getattr(selection, "labels", ()))
    pieces = [f"The search chose candidate {chosen}"]
    if str(getattr(selection, "strategy", "")) == "learners" and chosen < len(labels):
        pieces.append(f"({labels[chosen]})")
    elif chosen < len(selection.path):
        covariates = len(selection.path[chosen])
        pieces.append(
            f"with {covariates} covariate{'s' if covariates != 1 else ''} in the treatment model"
        )
    subtitle = " ".join(pieces) + ". Lower is better"
    found = [
        "Descriptive: the cross-validated "
        + ("criterion" if penalized else "risk")
        + " of each candidate; no interval is drawn."
    ]
    if penalized:
        found.append("The criterion adds the selection's variance and bias penalty to the loss.")
    if chosen < len(labels):
        found.append(f"Chosen candidate: {labels[chosen]}")
    stopped = getattr(selection, "stopped_at", None)
    if stopped is not None:
        patience = getattr(selection, "patience", None)
        found.append(
            f"The search stopped early at candidate {int(stopped)} (patience={patience}), so "
            "later candidates were not scored."
        )
    final = getattr(selection, "final", None)
    if final is not None and int(final) != chosen:
        found.append(
            f"The final step then reported candidate {int(final)}, not the one the "
            "cross-validated risk chose."
        )
    _finish(
        fig,
        [axis],
        "Treatment model selection",
        subtitle,
        found,
        owned=ax is None,
    )
    return fig


@_core.labelled
def learners(
    source: Any,
    *,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot which learners carry a fit.

    A :class:`~cleverly.learners.SuperLearnerDiagnostics` draws two panels with the
    candidates in the same order: each candidate's cross-validated risk (lower is better)
    and its weight in the ensemble. Use it to see whether one learner dominates and which
    ones fail. The ensemble's own risk is not drawn, because the stored value is scored on
    the predictions its weights were fitted to, so it flatters the ensemble.

    A fitted result, its ``result.diagnostics.nuisance_models()`` or one
    :class:`~cleverly.validation.NuisanceModelReport` draws the same two panels for every
    nuisance model a Super Learner fitted, one row of panels per model, from the risks and
    weights the reports hold (each the mean over the model's cross-fitting folds).

    A :class:`~cleverly.estimators.CTMLESelection` (or a collaborative result that holds
    one) draws the cross-validated risk along the candidate sequence with the chosen
    candidate marked, which shows whether the choice was sharp.

    Parameters
    ----------
    source : SuperLearnerDiagnostics or CTMLESelection or NuisanceModelReport or TMLEResult
        What to plot. A fitted ``SuperLearner`` holds the first as ``diagnostics_``. A fit's
        ``NuisanceDiagnostics`` is accepted too.
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
        Axes to draw on. Only for a collaborative selection, which is one panel.

    Returns
    -------
    Figure
        Two panels per Super Learner, one for a collaborative selection.

    Raises
    ------
    ValueError
        When the source holds nothing to draw (no nuisance model was fitted by a Super
        Learner), or ``ax`` is given for the two-panel plot.
    TypeError
        When the source is none of the above.
    CapabilityError
        When the source is a longitudinal, continuous-time survival or adaptive TMLE fit, or
        its nuisance report, whose learners this plot does not draw yet.

    See Also
    --------
    cleverly.plotting.calibration : Whether the models' probabilities are calibrated.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say the risks are descriptive
    and what the criterion adds to the loss.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> from cleverly.learners import SuperLearner
    >>> from cleverly.plotting import learners
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(120, 2))
    >>> y = x[:, 0] + rng.normal(size=120)
    >>> ensemble = SuperLearner(
    ...     library=[("ols", LinearRegression()), ("ridge", Ridge(alpha=5.0))],
    ...     task="regression",
    ...     n_folds=3,
    ... ).fit(x, y)
    >>> fig = learners(ensemble.diagnostics_)
    >>> fig.texts[0].get_text()
    'Learner risks and weights'
    >>> plt.close(fig)
    """
    role = None
    _refuse_unread(source, "learners")
    if hasattr(source, "ctmle_selection") or hasattr(source, "diagnostics"):
        selection = getattr(source, "ctmle_selection", None)
        if selection is not None and hasattr(selection, "cv_risk") and hasattr(selection, "path"):
            return _ctmle(selection, ax)
        if not hasattr(getattr(source, "diagnostics", None), "nuisance_models"):
            raise TypeError(
                f"learners() cannot read the nuisance models of {type(source).__name__}"
            )
        source = source.diagnostics.nuisance_models()
        _refuse_unread(source, "learners")
    if hasattr(source, "selected") and hasattr(source, "cv_risk") and hasattr(source, "path"):
        return _ctmle(source, ax)
    if ax is not None:
        raise ValueError("learners() draws two panels per Super Learner; ax= is not accepted")
    if hasattr(source, "names") and hasattr(source, "cv_risk") and hasattr(source, "weights"):
        library = _Library(
            None,
            tuple(str(name) for name in source.names),
            np.asarray(source.cv_risk, dtype=float),
            np.asarray(source.weights, dtype=float),
            str(source.loss),
        )
        extra = []
        if source.failed:
            extra.append("Dropped because fitting failed: " + ", ".join(map(str, source.failed)))
        return _libraries([library], None, extra)
    if hasattr(source, "models"):
        role = getattr(source, "treatment_role", None)
        return _from_reports(list(source.models), role)
    if hasattr(source, "learner_risks") and hasattr(source, "learner_weights"):
        return _from_reports([source], role)
    raise TypeError(
        "learners() takes SuperLearnerDiagnostics, a CTMLESelection, a NuisanceModelReport, "
        f"a point-treatment fit or its nuisance_models(); got {type(source).__name__}"
    )


def _refuse_unread(source: Any, plot: str) -> None:
    """Refuse the fits and reports whose nuisance models this plot does not draw yet."""
    from .._refusal import RefusalReason
    from ..estimators.atmle_result import AdaptiveTMLENuisanceReport, AdaptiveTMLEResult
    from ..exceptions import CapabilityError
    from ..longitudinal.estimator import LongitudinalResult
    from ..survival.result import ContinuousSurvivalResult
    from ..validation.longitudinal import LongitudinalNuisanceDiagnostics

    if isinstance(source, (LongitudinalResult, LongitudinalNuisanceDiagnostics)):
        what = "a longitudinal fit's nuisance models, one per node and role"
    elif isinstance(source, ContinuousSurvivalResult):
        if plot == "calibration":
            raise CapabilityError(
                "calibration() draws a nuisance report's calibration table; a continuous-time "
                "survival fit keeps no nuisance report, only its library selections in "
                "result.record.",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        what = "a continuous-time survival fit's hazard and treatment library selections"
    elif isinstance(source, (AdaptiveTMLEResult, AdaptiveTMLENuisanceReport)):
        what = "an adaptive TMLE fit's nuisance models"
    else:
        return
    raise CapabilityError(
        f"{plot}() does not yet draw {what}.",
        reason=RefusalReason.IMPLEMENTATION_GAP,
        alternative=(
            "Read them in result.record."
            if isinstance(source, ContinuousSurvivalResult)
            else "Read them with the fit's diagnostics.nuisance_models()."
        ),
        roadmap="X100",
    )


# ---------------------------------------------------------------------------- repeats


def _draw_values(draw: Any) -> Mapping[str, float]:
    """Each estimate's value in one cross-fitting draw.

    A point-treatment draw holds them as ``psi``; a longitudinal draw holds its whole
    estimate mapping (``LongitudinalRepeat.estimates``).
    """
    if hasattr(draw, "psi"):
        return dict(draw.psi)
    return {str(key): float(value.psi) for key, value in draw.estimates.items()}


def _estimate_of(result: Any, name: str | None) -> tuple[str, Any]:
    estimates = result.estimates
    if name is None:
        name = next(iter(estimates))
    if name not in estimates:
        raise ValueError(f"estimand {name!r} is not in this result; it holds {list(estimates)}")
    return str(name), estimates[name]


def _spread_word(estimate: Any) -> str:
    """The name the summary prints the estimate's spread under: never "standard error"
    for an estimate whose inference is withheld."""
    if estimate.supplies_inference:
        return "standard error"
    from .._summary.vocabulary import SPREAD_LABELS
    from ..inference.influence import spread_name

    return SPREAD_LABELS[spread_name("std_err", estimate.inference)]


@_core.labelled
def repeats(
    result: Any,
    *,
    name: str | None = None,
    outcome_label: str | None = None,
    units: str | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the estimate from each cross-fitting repeat against the reported estimate.

    A fit run with ``repeats=R`` draws ``R`` independent fold splits and reports their
    combined estimate. Use the plot to see whether the answer depends on how the data were
    split: each repeat is a point, the reported estimate is a line and its confidence
    interval is a shaded span. The repeats are descriptive points; none carries an
    interval of its own. When the fit withholds the interval, no span is drawn and the
    subtitle names the plug-in spread as the summary does, not as a standard error.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        A fit with at least two cross-fitting repeats.
    name : str or None, default=None
        The estimand to draw; the first one when omitted.
    outcome_label : str or None, default=None
        Words for the quantity in the axis label, such as ``"transition score"``, read as
        ``"Difference in transition score"``; ``"Difference in mean"`` the outcome column's
        name when omitted.
    units : str or None, default=None
        Units of the outcome, for the axis label.
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
        Axes to draw on.

    Returns
    -------
    Figure
        One panel with a point per repeat.

    Raises
    ------
    ValueError
        When the fit has one repeat, the estimand is unknown, or a repeat does not hold the
        estimand.
    CapabilityError
        When the result was not fitted with repeated cross-fitting splits.

    See Also
    --------
    cleverly.estimators.TMLEResult.repeat_spread : The spread of the repeats as numbers.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say the points are descriptive
    and which estimates are experimental.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import repeats
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     n_folds=3,
    ...     repeats=3,
    ...     q_bounds=(-10.0, 10.0),
    ...     random_state=0,
    ... )
    >>> fig = repeats(result)
    >>> fig.texts[0].get_text()
    'Estimate across repeated splits'
    >>> plt.close(fig)
    """
    if not hasattr(result, "repeats"):
        from .._refusal import RefusalReason
        from ..exceptions import CapabilityError

        raise CapabilityError(
            "repeats() draws the estimate of each cross-fitting split of a fit run with "
            f"repeats=; this {type(result).__name__} has no repeated splits.",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    draws = tuple(result.repeats)
    if len(draws) < 2:
        raise ValueError(
            "this fit has one cross-fitting repeat, so there is no spread to show; fit with "
            "repeats=2 or more"
        )
    label, estimate = _estimate_of(result, name)
    per_draw = [_draw_values(draw) for draw in draws]
    if any(label not in values for values in per_draw):
        raise ValueError(f"the repeats do not hold a per-repeat estimate of {label!r}")
    values = np.asarray([float(values[label]) for values in per_draw])
    shown = _core.estimate_interval(estimate, label)
    fig, axis = _core.figure_for(ax)
    rows = np.arange(1, len(values) + 1, dtype=float)
    (points,) = axis.plot(
        values,
        rows,
        linestyle="none",
        marker="o",
        markersize=8,
        color=_core.PALETTE[0],
        markeredgecolor=_core.FACE,
        markeredgewidth=0.8,
        zorder=3,
        gid="cleverly-repeat-points",
    )
    points.set_label("_nolegend_")
    learned = getattr(getattr(result, "config", None), "parameter_axis", None) == "learned_rule"
    combine = "mean" if learned else "median"
    if shown.drawn:
        assert shown.lower is not None and shown.upper is not None
        span = axis.axvspan(
            shown.lower,
            shown.upper,
            color=_core.PALETTE[0],
            alpha=_core.INTERVAL_ALPHA,
            linewidth=0,
            label=_core.interval_label(shown.level),
            gid="cleverly-repeat-interval",
        )
        span.set_zorder(1.0)
    _core.reference_line(
        axis, shown.psi, orientation="vertical", label=f"Reported estimate ({combine} of repeats)"
    )
    axis.set_yticks(rows)
    axis.set_yticklabels([f"Repeat {int(row)}" for row in rows])
    axis.set_ylabel("Cross-fitting repeat")
    axis.set_ylim(len(values) + 0.7, 0.3)
    _core.style_axes(axis, grid="x")
    scale = str(estimate.scale)
    outcome = getattr(result.data, "outcome_name", None)
    axis.set_xlabel(_axis_label(scale, outcome_label, outcome, units))
    if scale == "ratio" and np.all(values > 0):
        _core.log_axis(axis, "x")
    spread = float(result.repeat_spread()[label])
    error = float(estimate.std_error if estimate.supplies_inference else estimate.plugin_std_error)
    both = ", both on the log scale" if scale == "ratio" else ""
    _finish(
        fig,
        [axis],
        title or "Estimate across repeated splits",
        f"{_core.estimand_words(label)}: {len(values)} cross-fitting repeats, SD "
        f"{spread:.3g} against a {_spread_word(estimate)} of {error:.3g}{both}",
        [
            "Descriptive: each point is the estimate from one fold split; no interval is "
            "drawn for a single repeat.",
            *shown.notes,
        ],
        owned=ax is None,
    )
    return fig


# --------------------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class _Replicates:
    """What the bootstrap plot draws, read from the fit's summary or the bare draws."""

    draws: np.ndarray
    low: float
    high: float
    level: float
    spread: float
    spread_word: str
    limits_word: str
    notes: tuple[str, ...]


def _from_summary(name: str, estimate: Any) -> _Replicates:
    """The fit's own bootstrap summary, under the names its printed summary uses."""
    from .._inference_status import MINIMUM_INTERVAL_CLUSTERS
    from .._summary._estimates import plain_support_reason
    from .._summary.vocabulary import BOOTSTRAP_DIAGNOSTIC
    from ..inference.influence import (
        BOOTSTRAP_INTERVAL_LABELS,
        bootstrap_diagnostic_reason,
        bootstrap_is_diagnostic,
    )

    summary = estimate.bootstrap
    kind = BOOTSTRAP_INTERVAL_LABELS[summary.interval]
    level = 1.0 - float(estimate.alpha)
    low, high = (float(value) for value in summary.ci)
    found: list[str] = []
    if bootstrap_is_diagnostic(estimate):
        reason = bootstrap_diagnostic_reason(estimate)
        floor = summary.cluster_floor or MINIMUM_INTERVAL_CLUSTERS
        found.append(BOOTSTRAP_DIAGNOSTIC[reason].format(names=name, clusters=floor))
        spread_word, limits_word = "bootstrap SD", f"{kind} range (diagnostic)"
    else:
        spread_word, limits_word = "bootstrap SE", f"{_core.level_text(level)} {kind} interval"
        if summary.support.label == "experimental":
            found.append(
                f"{name}: the bootstrap interval is experimental. "
                f"{plain_support_reason(summary.support.reason)}"
            )
    shown = _core.estimate_interval(estimate, name)
    if shown.experimental:
        found.append(shown.notes[0])
    return _Replicates(
        np.asarray(summary.draws, dtype=float),
        low,
        high,
        level,
        float(summary.std_error),
        spread_word,
        limits_word,
        tuple(found),
    )


def _scale_maps(estimate: Any) -> tuple[Callable[[Any], Any], Callable[[Any], Any], str | None]:
    """The map to the scale the Wald interval is built on, its inverse, and its words."""
    transform = getattr(estimate, "transform", None) if estimate is not None else None
    if transform is not None:
        return transform.forward, transform.inverse, f"the {transform.name} scale"
    if estimate is not None and str(estimate.scale) == "ratio":
        return np.log, np.exp, "the log scale"
    return (lambda value: value), (lambda value: value), None


def _inverse_ticks(
    axis: Axes, inverse: Callable[[Any], Any], forward: Callable[[Any], Any]
) -> None:
    """Ticks at round values of the reported scale on an axis drawn on another scale."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    low, high = axis.get_xlim()
    reported = np.sort(np.asarray(inverse(np.asarray([low, high])), dtype=float))
    ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(*reported)
    ticks = ticks[(ticks >= reported[0]) & (ticks <= reported[1])]
    with np.errstate(divide="ignore", invalid="ignore"):
        placed = np.asarray(forward(ticks), dtype=float)
    keep = np.isfinite(placed)
    axis.xaxis.set_major_locator(FixedLocator(placed[keep].tolist()))
    labels = dict(zip(np.round(placed[keep], 12), ticks[keep], strict=True))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{labels.get(round(value, 12), inverse(value)):g}")
    )


@_core.labelled
def bootstrap(
    source: Any,
    *,
    name: str | None = None,
    level: float = 0.95,
    units: str | None = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the bootstrap replicates with a normal curve for comparison.

    Use it to see whether the bootstrap distribution is symmetric and bell-shaped on the
    scale the Wald interval is built on: the log scale for a ratio, the transform's scale
    for a transformed estimate. When it is skewed or heavy-tailed there, the Wald interval
    can mislead. The histogram shows the replicates, the curve is a normal density with
    their mean and standard deviation, and the vertical lines mark the limits. Nothing is
    resampled here.

    From a fit the limits, the spread and their names are the fit's own, as its summary
    prints them: the percentile or BCa interval, or a range marked diagnostic when the fit
    withholds the interval, declares its bootstrap a diagnostic or has too few clusters,
    and the experimental label when no validation study covers the interval. A bare
    ``BootstrapResult`` draws the percentile limits of its replicates.

    Parameters
    ----------
    source : BootstrapResult or TMLEResult
        The replicates, or a fit that holds them. A fit also marks its reported estimate
        and takes its confidence level.
    name : str or None, default=None
        The estimand to draw; the first one with replicates when omitted.
    level : float, default=0.95
        Coverage of the percentile limits, for a bare ``BootstrapResult``.
    units : str or None, default=None
        Units of the outcome, for the axis label.
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
        Axes to draw on.

    Returns
    -------
    Figure
        One panel.

    Raises
    ------
    ValueError
        When a fit holds no bootstrap, the estimand has no replicates, fewer than two
        replicates are finite, or every replicate is equal.

    See Also
    --------
    cleverly.inference.BootstrapResult.interval : The percentile interval as numbers.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say the histogram is
    descriptive, how rows were resampled, and whether the bootstrap interval is
    experimental.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=120, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     cross_fit=False,
    ...     n_bootstrap=40,
    ...     random_state=0,
    ... )
    >>> fig = result.bootstrap.plot("ate")
    >>> fig.texts[0].get_text()
    'Bootstrap replicates'
    >>> plt.close(fig)
    """
    result = source if hasattr(source, "estimates") else None
    if result is not None:
        if getattr(result, "bootstrap", None) is None:
            raise ValueError("this fit holds no bootstrap replicates; fit with n_bootstrap > 0")
        replicates = result.bootstrap
    elif hasattr(source, "draws"):
        replicates = source
    else:
        raise TypeError(
            f"bootstrap() takes a fitted result or its BootstrapResult; got {type(source).__name__}"
        )
    held = list(replicates.draws)
    if name is None:
        ordered = [alias for alias in (result.estimates if result else ()) if alias in held]
        name = (ordered or held)[0]
    if name not in replicates.draws:
        raise CapabilityError(
            f"estimand {name!r} has no replicates; they hold {held}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    estimate = result.estimates.get(name) if result is not None else None
    if estimate is not None and estimate.bootstrap is not None:
        read = _from_summary(name, estimate)
    else:
        if result is not None:
            level = 1.0 - float(result.config.alpha_sig)
        bare = replicates.interval(name, 1.0 - level)
        low, high = (float(value) for value in bare.ci)
        read = _Replicates(
            np.asarray(bare.draws, dtype=float),
            low,
            high,
            level,
            float(bare.std_error),
            "SD of the replicates",
            f"{_core.level_text(level)} percentile limits",
            (),
        )
    forward, inverse, scale_words = _scale_maps(estimate)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.asarray(forward(read.draws), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(f"fewer than two finite replicates of {name!r}; there is nothing to plot")
    mean, spread = float(np.mean(values)), float(np.std(values, ddof=1))
    if not spread > 1e-12 * max(1.0, abs(mean)):
        raise ValueError(
            f"every replicate of {name!r} is equal, so there is no distribution to compare "
            "with a normal curve"
        )
    fig, axis = _core.figure_for(ax)
    axis.grid(False)  # a density is read by shape, and a grid line would strike the labels
    bins = int(min(max(10, round(math.sqrt(values.size))), 40))
    patches: Any
    _, _, patches = axis.hist(
        values,
        bins=bins,
        density=True,
        color=_core.PALETTE[0],
        edgecolor=_core.FACE,
        linewidth=0.6,
        alpha=0.9,
        zorder=2,
    )
    for patch in patches:
        patch.set_gid("cleverly-bootstrap-histogram")
    grid = np.linspace(
        min(float(values.min()), mean - 3.5 * spread),
        max(float(values.max()), mean + 3.5 * spread),
        240,
    )
    density = np.exp(-0.5 * ((grid - mean) / spread) ** 2) / (spread * math.sqrt(2.0 * math.pi))
    (curve,) = axis.plot(
        grid,
        density,
        color=_core.PALETTE[1],
        linewidth=2.0,
        zorder=3,
        gid="cleverly-bootstrap-normal",
    )
    curve.set_label("_nolegend_")
    top = max(float(np.max(density)), max(patch.get_height() for patch in patches))
    axis.set_ylim(0.0, top * 1.45)
    # Vertical lines stop below the two series labels at the top of the panel.
    reach = 1.12 / 1.45
    for limit in (read.low, read.high):
        line = _core.reference_line(
            axis,
            float(forward(limit)),
            orientation="vertical",
            label=read.limits_word,
            kind="marker",
        )
        line.set_ydata([0.0, reach])
    if estimate is not None:
        line = _core.reference_line(
            axis, float(forward(estimate.psi)), orientation="vertical", label="Reported estimate"
        )
        line.set_ydata([0.0, reach])
    _core.corner_label(axis, 0.97, "Bootstrap replicates", _core.PALETTE[0])
    _core.corner_label(axis, 0.86, "Normal approximation (same mean and SD)", _core.PALETTE[1])
    outcome = getattr(result.data, "outcome_name", None) if result is not None else None
    if estimate is None:
        axis.set_xlabel("Estimate in each bootstrap replicate")
    else:
        label = _axis_label(str(estimate.scale), None, outcome, units)
        if scale_words is not None and "log scale" not in label:
            label = f"{label} (on {scale_words})"
        axis.set_xlabel(label)
    if scale_words is not None:
        _inverse_ticks(axis, inverse, forward)
    axis.set_ylabel("Density of replicates")
    found = [
        "Descriptive: the histogram shows the bootstrap replicates, and the curve is a normal "
        "density with their mean and standard deviation, drawn for comparison only.",
        *read.notes,
        "Whole clusters resampled."
        if str(replicates.resampling) == "cluster"
        else "Rows resampled independently.",
    ]
    if scale_words is not None:
        found.append(
            f"Drawn on {scale_words}, where the reported Wald interval is built; the "
            "ticks read on the reported scale."
        )
    if replicates.n_failed:
        found.append(
            f"{replicates.n_failed} of {replicates.n_requested} replicates failed and are left out."
        )
    words = _core.estimand_words(name) if estimate is not None else name
    reported = " on the reported scale" if scale_words is not None else ""
    _finish(
        fig,
        [axis],
        title or "Bootstrap replicates",
        f"{words}: {values.size} replicates, {read.spread_word} {read.spread:.3g}{reported}; "
        f"{read.limits_word} {read.low:.3g} to {read.high:.3g}",
        found,
        owned=ax is None,
    )
    return fig


# --------------------------------------------------------------------------- influence


def _concentration(influence_curve: Any, cluster: Any = None) -> tuple[np.ndarray, np.ndarray, str]:
    """The cumulative share of the centred sum of squares carried by the top units.

    Units are the rows, or the clusters when ``cluster`` is given, in which case the
    influence values are summed within each cluster first (the unit that carries the
    variance on a clustered fit). The values are centred at their mean, as
    :func:`~cleverly.inference.influence_variance` centres them, so the shares partition
    the numerator of the reported variance.

    Returns the fraction of units taken (``0, 1/m, ..., 1``, largest squared value first),
    the share of the sum of squares those units carry, and ``"rows"`` or ``"clusters"``.
    """
    from ..inference.bootstrap import cluster_sums

    curve = np.asarray(influence_curve, dtype=float).reshape(-1)
    if not np.all(np.isfinite(curve)):
        raise ValueError("the influence values contain non-finite entries")
    if cluster is not None and np.asarray(cluster).reshape(-1).shape != curve.shape:
        raise ValueError("cluster must hold one code per influence value")
    values = cluster_sums(curve, None if cluster is None else np.asarray(cluster))
    if values.size < 2:
        raise ValueError("need at least two units to show how the influence is spread")
    squares = np.sort((values - values.mean()) ** 2)[::-1]
    total = float(squares.sum())
    if total <= 0.0:
        raise ValueError("the influence values are all equal, so there is no spread to show")
    count = squares.size
    fraction = np.concatenate([[0.0], np.arange(1, count + 1) / count])
    share = np.concatenate([[0.0], np.cumsum(squares) / total])
    return fraction, share, "rows" if cluster is None else "clusters"


def _top_words(top: int, count: int, unit: str) -> tuple[str, str]:
    """The headline and the point label of the top units, naming the share actually taken."""
    plural = f"{unit}s"
    if count % 100 == 0:
        return f"the top 1% of {plural} ({top} of {count}) carry", "Top 1%"
    taken = _percent(top / count)
    if top == 1:
        return f"the largest {unit} ({taken} of {count}) carries", f"Largest {unit}"
    return f"the top {top} of {count} {plural} ({taken}) carry", f"Top {top} {plural}"


@_core.labelled
def influence(
    source: Any,
    *,
    name: str | None = None,
    cluster: Any = None,
    title: str | Literal[False] | None = None,
    subtitle: str | Literal[False] | None = None,
    xlabel: str | Sequence[str] | None = None,
    ylabel: str | Sequence[str] | None = None,
    legend: str | Literal[False] = "top",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot how concentrated the uncertainty is in a few rows or clusters.

    Rank the units by their squared centred influence value, largest first, and plot the
    share of the sum of squares that the top ``k`` percent carry against ``k``. The sum is
    the numerator of the influence-curve variance. A curve on the dashed diagonal means
    every unit contributes equally; a curve that rises steeply means a few units carry the
    standard error, which is how a Wald interval misleads even without a bootstrap. On a
    clustered fit the influence values are summed within clusters and the unit is the
    cluster. The subtitle names the top 1% of units, or the smallest whole number of units
    above it when the count is not a multiple of 100.

    The curve describes the stored influence values. On a fit with several cross-fitting
    repeats the stored curve is the median draw's (the mean of the two central draws for
    an even count), and on a fit evaluated within folds, or on an estimate that adds the
    boundary-node supplement, the reported variance is not computed from this curve alone;
    :func:`~cleverly.plotting.notes` says so.

    Parameters
    ----------
    source : ParameterEstimate or TMLEResult
        An estimate, or a fit and the ``name`` of one of its estimates.
    name : str or None, default=None
        The estimate to draw from a fit; the first one when omitted.
    cluster : array-like or None, default=None
        Cluster codes, one per row. A fit supplies its own; a bare estimate from a
        clustered fit needs them.
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
        Axes to draw on.

    Returns
    -------
    Figure
        One panel.

    Raises
    ------
    ValueError
        When the estimate's interval is withheld (the reason is given), when a clustered
        estimate comes without cluster codes, or when the values cannot be ranked.

    See Also
    --------
    cleverly.assessment.DiagnosticsFacade.influence : The Kish effective count, skewness and
        kurtosis of the same curve.

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes say the curve is descriptive,
    whether influence values are summed within clusters, and when the curve is not the
    reported variance.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import influence
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
    >>> fig = influence(result)
    >>> fig.texts[0].get_text()
    'Concentration of the uncertainty'
    >>> plt.close(fig)
    """
    result = source if hasattr(source, "estimates") else None
    if result is not None:
        label, estimate = _estimate_of(result, name)
        if cluster is None:
            cluster = getattr(getattr(result, "data", None), "cluster", None)
    elif hasattr(source, "influence_curve"):
        estimate, label = source, str(source.name)
    else:
        raise TypeError(
            f"influence() takes a ParameterEstimate or a fitted result; got {type(source).__name__}"
        )
    if not estimate.supplies_inference:
        from .._inference_status import status_record

        raise ValueError(
            f"{label}: this estimate's interval is withheld, so its influence curve does not "
            "carry a reported uncertainty. "
            f"{status_record(str(estimate.inference)).plain_reason}"
        )
    if cluster is None and int(estimate.n_clusters) < int(estimate.n):
        raise ValueError(
            f"{label} comes from a clustered fit ({estimate.n_clusters} clusters in "
            f"{estimate.n} rows); pass the fitted result, or cluster= with one code per row"
        )
    fraction, share, units = _concentration(estimate.influence_curve, cluster)
    count = fraction.size - 1
    top = max(1, math.ceil(0.01 * count - 1e-12))
    top_share = float(share[top])
    unit = "cluster" if units == "clusters" else "row"
    headline, point_label = _top_words(top, count, unit)
    fig, axis = _core.figure_for(ax, figsize=(5.2, 5.6))
    _diagonal(axis, 0.0, 1.0, label="Even spread")
    (curve,) = axis.plot(
        fraction,
        share,
        color=_core.PALETTE[0],
        linewidth=2.0,
        zorder=3,
        gid="cleverly-influence-curve",
    )
    curve.set_label("_nolegend_")
    axis.scatter(
        [fraction[top]],
        [top_share],
        s=46,
        color=_core.PALETTE[0],
        edgecolor=_core.FACE,
        linewidth=0.8,
        zorder=4,
        gid="cleverly-influence-top",
    )
    axis.annotate(
        f"{point_label}: {_percent(top_share)}",
        xy=(fraction[top], top_share),
        xytext=(12.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=_core.TICK_SIZE,
        color=_core.INK,
        bbox={"boxstyle": "square,pad=0.15", "facecolor": _core.FACE, "edgecolor": "none"},
        gid="cleverly-influence-top-label",
    )
    from matplotlib.ticker import PercentFormatter

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    _square(axis)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(True, axis="both", color=_core.GRID, linewidth=0.8)
    axis.set_xlabel(f"Top share of {units}, largest squared influence first")
    axis.set_ylabel("Share of the sum of squared influence values")
    subtitle = (
        f"{_core.estimand_words(label)}: {headline} {_percent(top_share)} of the squared influence"
    )
    from ..inference.sparse import influence_row

    kish = influence_row(
        label, np.asarray(estimate.influence_curve, dtype=float), cluster=cluster
    ).kish
    found = [
        "Descriptive: a measure of how the stored influence values are spread; "
        "no interval is drawn. Values are centred at their mean, as the variance centres them.",
        f"Kish effective count: {kish:.0f} of {count} {units}.",
    ]
    if units == "clusters":
        found.append("Share of clusters: influence values are summed within each cluster first.")
    if result is not None:
        draws = len(getattr(result, "repeats", ()))
        if draws > 1:
            learned = getattr(result.config, "parameter_axis", None) == "learned_rule"
            whose = (
                "the mean of the repeats' curves"
                if learned
                else "the median draw's curve"
                + (" (the mean of the two central draws)" if draws % 2 == 0 else "")
            )
            found.append(
                f"This fit combines {draws} cross-fitting repeats: the curve is {whose}, and "
                "the reported variance also counts the spread between repeats, so this curve "
                "is not the reported variance."
            )
        if getattr(getattr(result, "config", None), "cv_evaluation", False):
            found.append(
                "This fit evaluates the estimate within folds, so this curve is not the "
                "reported variance."
            )
    if getattr(estimate, "node", None) is not None:
        found.append(
            "This estimate's recursion read a node with no event (or only events), whose term "
            "the influence curve leaves out; the reported variance adds a bound on that term, "
            "so this curve is not the reported variance."
        )
    if str(estimate.scale) == "ratio":
        found.append("Influence values are on the log scale.")
    found.extend(_core.estimate_interval(estimate, label).notes)
    _finish(
        fig,
        [axis],
        title or "Concentration of the uncertainty",
        subtitle,
        found,
        owned=ax is None,
    )
    return fig


__all__ = ["bootstrap", "calibration", "influence", "learners", "repeats"]
