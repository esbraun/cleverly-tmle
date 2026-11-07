"""V8, V15, V16, V17 and V19: diagnostics that read only what a fit already holds.

* :func:`calibration`: do the predicted effects (a fitted effect curve) or probabilities
  (the cross-fitted treatment and outcome models) match what was measured, bin by bin?
* :func:`learners`: which candidate learners carry a Super Learner, and how sharp was a
  collaborative fit's choice of treatment model?
* :func:`repeats`: does the estimate depend on how the cross-fitting split was drawn?
* :func:`bootstrap`: is the bootstrap distribution close to normal?
* :func:`influence`: how few units carry the sum of squared influence values?

Every quantity is descriptive and none carries an interval the object does not hold; each
note says so. Nothing is refitted and nothing is drawn at random.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

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

#: What the nuisance model names read as on a figure.
_MODEL_WORDS = {"propensity": "Treatment model", "outcome": "Outcome model"}


def _loss_text(loss: str) -> str:
    return _LOSS_WORDS.get(str(loss), str(loss).replace("_", " "))


def _percent(share: float) -> str:
    """A share as a percentage with one decimal below 10%, none above."""
    value = 100.0 * share
    return f"{value:.1f}%" if value < 10.0 else f"{value:.0f}%"


def _notes(ax: Axes, lines: Sequence[str], *, share: float = 0.72) -> None:
    """Note lines wrapped to ``share`` of the figure width, so none runs off the right edge.

    The core wraps to the whole figure width, but the notes start at the axes' left edge,
    which sits to the right of the figure edge by the width of the y labels.
    """
    figure: Any = ax.figure
    while not hasattr(figure, "get_figwidth"):
        figure = figure.figure
    characters = max(
        30, int(float(figure.get_figwidth()) * share * 72.0 / (_core.NOTE_SIZE * 0.52))
    )
    _core.notes(
        ax,
        ["\n".join(textwrap.fill(part, characters) for part in line.split("\n")) for line in lines],
    )


def _figure_layout(fig: Figure, top: Axes, lines: Sequence[str]) -> None:
    """Title, subtitle and notes of a figure with several columns of panels.

    Text anchored to one panel counts toward that panel's width in the layout, so a title
    or note wider than the panel squeezes it and pushes the next column away. Here the
    title, the subtitle and the notes sit outside the layout, and the layout area is
    shrunk to leave room for them above and below. Call after :func:`_header`. The notes
    belong to ``top`` so :func:`~cleverly.plotting._core.note_text` finds them.
    """
    width, height = (float(value) for value in fig.get_size_inches())
    axes_any: Any = top
    left_title = axes_any._left_title  # the title _core.header wrote
    left_title.set_in_layout(False)
    for text in top.texts:
        if text.get_gid() == "cleverly-subtitle":
            text.set_in_layout(False)
    # The layout still reserves the title's height; only its width is left out.
    distinct = list(dict.fromkeys(line for line in lines if line))
    bottom_points = 0.0
    if distinct:
        characters = max(30, int(width * 0.94 * 72.0 / (_core.NOTE_SIZE * 0.52)))
        words = "\n".join(textwrap.fill(line, characters) for line in distinct)
        bottom_points = (words.count("\n") + 1) * _core.NOTE_SIZE * 1.4 + 10.0
        note = top.text(
            0.012,
            0.006,
            words,
            transform=fig.transFigure,
            ha="left",
            va="bottom",
            fontsize=_core.NOTE_SIZE,
            color=_core.SECONDARY_INK,
            gid="cleverly-notes",
        )
        note.set_in_layout(False)
    low = bottom_points / 72.0 / height
    engine: Any = fig.get_layout_engine()
    engine.set(rect=(0.0, low, 1.0, 1.0 - low))


def _header(ax: Axes, title: str, subtitle: str | None = None) -> None:
    """The core header with the subtitle wrapped narrower than the figure."""
    if subtitle:
        figure: Any = ax.figure
        while not hasattr(figure, "get_figwidth"):
            figure = figure.figure
        width = max(
            30, int(float(figure.get_figwidth()) * 0.8 * 72.0 / (_core.SUBTITLE_SIZE * 0.52))
        )
        subtitle = textwrap.fill(subtitle, width)
    _core.header(ax, title, subtitle)


def _square(ax: Axes) -> None:
    """Equal axis scales, so the 45-degree line is at 45 degrees, anchored at the left."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("NW")


def _square_limits(values: Any) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.06 * (high - low) if high > low else 0.05
    return low - pad, high + pad


def _diagonal(ax: Axes, low: float, high: float) -> Any:
    """The 45-degree line of perfect calibration, in the reference style."""
    (line,) = ax.plot(
        [low, high],
        [low, high],
        color=_core.REFERENCE,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=1.5,
        label="Perfect calibration",
        gid="cleverly-reference",
    )
    return line


# ------------------------------------------------------------------------- calibration


def _curve_panel(ax: Axes, calibration: Any) -> list[str]:
    bins = calibration.bins
    x = np.asarray(bins["mean_prediction"], dtype=float)
    y = np.asarray(bins["pseudo_outcome_contrast"], dtype=float)
    low, high = _square_limits(np.concatenate([x, y]))
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    _diagonal(ax, low, high)
    (line,) = ax.plot(
        x,
        y,
        color=_core.PALETTE[0],
        marker="o",
        markersize=7,
        linewidth=1.2,
        markeredgecolor=_core.FACE,
        markeredgewidth=0.8,
        zorder=3,
        gid="cleverly-calibration-points",
    )
    ratio = str(calibration.scale) != "difference"
    _square(ax)
    ax.grid(True, axis="both", color=_core.GRID, linewidth=0.8)
    ax.set_xlabel(
        "Mean predicted log ratio in the bin" if ratio else "Mean predicted effect in the bin"
    )
    ax.set_ylabel(
        "Measured log ratio (doubly robust contrast)"
        if ratio
        else "Measured effect (doubly robust contrast)"
    )
    line.set_label("_nolegend_")
    return [
        "Descriptive: bins of out-of-fold refits "
        f"({int(calibration.n_folds)} folds); no interval is drawn.",
        "The slope and intercept come from projecting the doubly robust pseudo-outcome on "
        "(1, prediction); they are not the slope of the drawn bins.",
    ]


def _model_words(name: str) -> str:
    return _MODEL_WORDS.get(str(name), str(name))


def _probability_models(source: Any) -> tuple[list[Any], str | None]:
    """The reports to draw and the evaluation basis, when the source states one."""
    if hasattr(source, "models"):
        models = list(source.models)
        basis = getattr(source, "evaluation", None)
    elif hasattr(source, "diagnostics") and hasattr(source.diagnostics, "nuisance_models"):
        found = source.diagnostics.nuisance_models()
        if not hasattr(found, "models"):
            raise ValueError(
                "calibration() draws the point-treatment nuisance reports; this result "
                "reports its nuisance models another way"
            )
        models, basis = list(found.models), getattr(found, "evaluation", None)
    else:
        models, basis = [source], None
    drawn = [
        model
        for model in models
        if getattr(model, "kind", None) == "probability" and len(model.calibration.get("n", ()))
    ]
    if not drawn:
        raise ValueError(
            "no probability model with a calibration table to draw: a conditional-mean "
            "(continuous outcome) model has no probability to calibrate"
        )
    return drawn, basis


def _slope_text(model: Any) -> str:
    slope = float(model.metrics.get("calibration_slope", float("nan")))
    if math.isfinite(slope):
        return f"{slope:.2f}"
    omission = getattr(model, "calibration_omission", None)
    return f"not available ({omission})" if omission else "not available"


def _probability_panel(
    ax: Axes, model: Any, color: str, low: float, high: float, *, square: bool
) -> None:
    table = model.calibration
    x = np.asarray(table["mean_predicted"], dtype=float)
    y = np.asarray(table["mean_observed"], dtype=float)
    rows = np.asarray(table["n"], dtype=float)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    _diagonal(ax, low, high)
    ax.scatter(
        x,
        y,
        s=30.0 + 170.0 * rows / float(rows.max()),
        color=color,
        edgecolor=_core.FACE,
        linewidth=0.8,
        zorder=3,
        gid="cleverly-calibration-points",
    )
    if square:
        _square(ax)
    ax.grid(True, axis="both", color=_core.GRID, linewidth=0.8)
    ax.set_xlabel(f"{_model_words(model.name)}: mean predicted probability")
    ax.set_ylabel("Observed rate")


def calibration(
    source: Any,
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot whether predictions match what was measured, bin by bin.

    Use it to see where along the range of predictions a model over- or understates, which
    a single slope hides. Two kinds of source are drawn.

    A ``CurveCalibration`` (from ``curve.diagnostics.calibration()``) plots
    each bin's measured effect, a doubly robust contrast, against the bin's mean predicted
    effect. The linear calibration slope and intercept are in the subtitle; they come from
    projecting the doubly robust pseudo-outcome on the prediction, so they are not the
    slope of the drawn bins.

    A :class:`~cleverly.validation.NuisanceModelReport`, a fit's
    ``result.diagnostics.nuisance_models()`` or the result itself plots the weighted mean
    observed rate against the mean predicted probability of each cross-fitted probability
    model (the treatment model and a binary outcome model), with point area showing the
    rows in the bin. The logistic calibration slope is in the subtitle.

    Bins are descriptive: no interval is drawn, and the note says so.

    Parameters
    ----------
    source : CurveCalibration or NuisanceModelReport or NuisanceDiagnostics or TMLEResult
        What to plot.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on. Only for a single panel.

    Returns
    -------
    Figure
        One panel per probability model, or one panel for a curve.

    Raises
    ------
    ValueError
        When the source holds no probability model with a calibration table, or ``ax`` is
        given for more than one panel.

    See Also
    --------
    cleverly.plotting.learners : Which learners carry the models.

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
    >>> fig.axes[0].get_title(loc="left")
    'Are the predicted probabilities calibrated?'
    >>> plt.close(fig)
    """
    if hasattr(source, "bins") and hasattr(source, "slope") and hasattr(source, "intercept"):
        fig, axis = _core.figure_for(ax, figsize=(5.8, 5.4))
        found = _curve_panel(axis, source)
        _core.legend(axis)
        _header(
            axis,
            title or "Do the predicted effects match the measured ones?",
            "Linear calibration slope "
            f"{float(source.slope):.2f} and intercept {float(source.intercept):.2f} "
            "(1 and 0 for a calibrated curve)",
        )
        _notes(axis, found)
        return fig
    models, basis = _probability_models(source)
    count = len(models)
    ncols = min(count, 3)
    nrows = math.ceil(count / ncols)
    if ax is not None:
        if count > 1:
            raise ValueError(
                "ax= takes one panel, but this source has several probability models; "
                "pass one NuisanceModelReport"
            )
        fig, axes = ax.figure, [ax]  # type: ignore[assignment]
        _core.style_axes(ax)
    else:
        size = (5.8, 5.4) if count == 1 else (3.3 * ncols + 0.9, 3.3 * nrows + 2.0)
        fig, axes = _core.new_figure(nrows, ncols=ncols, figsize=size)
        for extra in axes[count:]:
            fig.delaxes(extra)
        axes = axes[:count]
    lows, highs = [], []
    for model in models:
        values = np.concatenate(
            [
                np.asarray(model.calibration["mean_predicted"], dtype=float),
                np.asarray(model.calibration["mean_observed"], dtype=float),
            ]
        )
        low, high = _square_limits(values)
        lows.append(low)
        highs.append(high)
    low, high = min(lows), max(highs)
    for index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        color = _core.PALETTE[index if count <= _core.MAX_SERIES else 0]
        _probability_panel(axis, model, color, low, high, square=count == 1)
        _core.corner_label(axis, 0.97, _model_words(model.name), color)
    _core.legend(axes[0], sources=axes)
    slopes = "; ".join(f"{_model_words(model.name)} {_slope_text(model)}" for model in models)
    _header(
        axes[0],
        title or "Are the predicted probabilities calibrated?",
        f"Logistic calibration slope (1 = calibrated): {slopes}",
    )
    where = (
        "bins of in-sample predictions (the fit used one fold)"
        if basis == "in_sample"
        else "bins of cross-fitted (out-of-fold) predictions"
    )
    found = [
        f"Descriptive: {where}; no interval is drawn. Point area shows the rows in the "
        "bin; the observed rate is weighted by the data's weights."
    ]
    if count > 1:
        _figure_layout(fig, axes[0], found)
    else:
        _notes(axes[0], found)
    return fig


# --------------------------------------------------------------------------- learners


def _super_learner(diagnostics: Any, ax: Axes | None) -> Figure:
    if ax is not None:
        raise ValueError("learners() draws two panels for a Super Learner; ax= is not accepted")
    names = [str(name) for name in diagnostics.names]
    risk = np.asarray(diagnostics.cv_risk, dtype=float)
    weights = np.asarray(diagnostics.weights, dtype=float)
    order = np.argsort(risk, kind="stable")
    count = len(names)
    fig, axes = _core.new_figure(1, ncols=2, figsize=(7.0, 2.4 + 0.42 * count), sharex=False)
    left, right = axes
    right.sharey(left)
    positions = np.arange(count)
    for axis, values, gid in (
        (left, risk[order], "cleverly-learner-risk"),
        (right, weights[order], "cleverly-learner-weight"),
    ):
        bars = axis.barh(
            positions,
            values,
            height=0.62,
            color=_core.PALETTE[0],
            edgecolor=_core.FACE,
            linewidth=0.6,
        )
        for bar in bars:
            bar.set_gid(gid)
        top = float(np.nanmax(values)) if len(values) else 1.0
        axis.set_xlim(0.0, top * 1.22 if top > 0 else 1.0)
        for position, value in zip(positions, values, strict=True):
            axis.annotate(
                f"{value:.2f}" if gid.endswith("weight") else f"{value:.3g}",
                xy=(value, position),
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
    left.set_yticklabels([names[index] for index in order])
    left.set_ylim(count - 0.4, -0.6)
    left.set_ylabel("Candidate learner")
    right.tick_params(labelleft=False)
    right.set_xlim(0.0, 1.18)
    right.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    loss = _loss_text(diagnostics.loss)
    left.set_xlabel(f"Cross-validated risk ({loss})")
    right.set_xlabel("Ensemble weight")
    _header(
        left,
        "Which learners carry the fit?",
        f"Lower risk is better. Lowest cross-validated risk: {diagnostics.best}. "
        f"Largest weight: {names[int(np.argmax(weights))]}",
    )
    found = [
        "Descriptive: cross-validated risk of each candidate. The ensemble's own risk is not "
        "drawn, because it is scored on the predictions its weights were fitted to and so "
        "flatters it."
    ]
    if diagnostics.failed:
        found.append("Dropped because fitting failed: " + ", ".join(map(str, diagnostics.failed)))
    _figure_layout(fig, left, found)
    return fig


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
        s=120,
        facecolor=_core.FACE,
        edgecolor=_core.PALETTE[0],
        linewidth=2.0,
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
    axis.set_xlabel("Candidate along the search path (0 = first)")
    axis.set_ylabel(f"Cross-validated risk ({_loss_text(selection.loss)})")
    labels = getattr(selection, "labels", ())
    covariates = len(selection.path[chosen]) if chosen < len(selection.path) else None
    pieces = [f"The search chose candidate {chosen}"]
    if covariates is not None:
        pieces.append(
            f"with {covariates} covariate{'s' if covariates != 1 else ''} in the treatment model"
        )
    _core.legend(axis)
    _header(
        axis,
        "How sharp was the choice of treatment model?",
        " ".join(pieces) + ". Lower is better",
    )
    found = ["Descriptive: cross-validated risk of each candidate; no interval is drawn."]
    if getattr(selection, "penalized", False):
        found.append("The criterion includes the selection's variance and bias penalty.")
    if chosen < len(labels):
        found.append(f"Chosen candidate: {labels[chosen]}")
    final = getattr(selection, "final", None)
    if final is not None and int(final) != chosen:
        found.append(
            f"The final step then reported candidate {int(final)}, not the one the "
            "cross-validated risk chose."
        )
    _notes(axis, found)
    return fig


def learners(source: Any, *, ax: Axes | None = None) -> Figure:
    """Plot which learners carry a fit.

    A :class:`~cleverly.learners.SuperLearnerDiagnostics` draws two panels with the
    candidates in the same order: each candidate's cross-validated risk (lower is better)
    and its weight in the ensemble. Use it to see whether one learner dominates and which
    ones fail. The ensemble's own risk is not drawn, because the stored value is scored on
    the predictions its weights were fitted to, so it flatters the ensemble.

    A :class:`~cleverly.estimators.CTMLESelection` (or a collaborative result that holds
    one) draws the cross-validated risk along the candidate sequence with the chosen
    candidate marked, which shows whether the choice was sharp.

    Parameters
    ----------
    source : SuperLearnerDiagnostics or CTMLESelection or TMLEResult
        What to plot. A fitted ``SuperLearner`` holds the first as ``diagnostics_``.
    ax : Axes or None, default=None
        Axes to draw on. Only for a collaborative selection, which is one panel.

    Returns
    -------
    Figure
        Two panels for a Super Learner, one for a collaborative selection.

    Raises
    ------
    ValueError
        When the source holds nothing to draw, or ``ax`` is given for the two-panel plot.

    See Also
    --------
    cleverly.plotting.calibration : Whether the models' probabilities are calibrated.

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
    >>> fig.axes[0].get_title(loc="left")
    'Which learners carry the fit?'
    >>> plt.close(fig)
    """
    if hasattr(source, "ctmle_selection"):
        selection = source.ctmle_selection
        if selection is None or not hasattr(selection, "cv_risk"):
            raise ValueError(
                "this result holds no candidate sequence with cross-validated risks; pass a "
                "SuperLearnerDiagnostics (a fitted SuperLearner's diagnostics_) instead"
            )
        source = selection
    if hasattr(source, "selected") and hasattr(source, "cv_risk") and hasattr(source, "path"):
        return _ctmle(source, ax)
    if hasattr(source, "names") and hasattr(source, "cv_risk") and hasattr(source, "weights"):
        return _super_learner(source, ax)
    raise TypeError(
        "learners() takes SuperLearnerDiagnostics, a CTMLESelection or a collaborative "
        f"result; got {type(source).__name__}"
    )


# ---------------------------------------------------------------------------- repeats


def _estimate_of(result: Any, name: str | None) -> tuple[str, Any]:
    estimates = result.estimates
    if name is None:
        name = next(iter(estimates))
    if name not in estimates:
        raise ValueError(f"estimand {name!r} is not in this result; it holds {list(estimates)}")
    return str(name), estimates[name]


def repeats(
    result: Any,
    *,
    name: str | None = None,
    units: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the estimate from each cross-fitting repeat against the reported estimate.

    A fit run with ``repeats=R`` draws ``R`` independent fold splits and reports their
    combined estimate. Use the plot to see whether the answer depends on how the data were
    split: each repeat is a point, the reported estimate is a line and its confidence
    interval is a shaded span. The repeats are descriptive points; none carries an
    interval of its own.

    Parameters
    ----------
    result : TMLEResult
        A fit with at least two cross-fitting repeats.
    name : str or None, default=None
        The estimand to draw; the first one when omitted.
    units : str or None, default=None
        Units of the outcome, for the axis label.
    title : str or None, default=None
        The title.
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

    See Also
    --------
    cleverly.estimators.TMLEResult.repeat_spread : The spread of the repeats as numbers.

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
    >>> fig.axes[0].get_title(loc="left")
    'Does the estimate depend on how the data were split?'
    >>> plt.close(fig)
    """
    draws = tuple(result.repeats)
    if len(draws) < 2:
        raise ValueError(
            "this fit has one cross-fitting repeat, so there is no spread to show; fit with "
            "repeats=3 or more"
        )
    label, estimate = _estimate_of(result, name)
    if any(label not in draw.psi for draw in draws):
        raise ValueError(f"the repeats do not hold a per-repeat estimate of {label!r}")
    values = np.asarray([float(draw.psi[label]) for draw in draws])
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
    axis.set_xlabel(_axis_label(scale, getattr(result.data, "outcome_name", None), units))
    if scale == "ratio" and np.all(values > 0):
        _core.log_axis(axis, "x")
    spread = float(result.repeat_spread()[label])
    spread_text = f"{spread:.3g}" + (" on the log scale" if scale == "ratio" else "")
    error = float(estimate.std_error if estimate.supplies_inference else estimate.plugin_std_error)
    _core.legend(axis)
    _header(
        axis,
        title or "Does the estimate depend on how the data were split?",
        f"{label}: {len(values)} cross-fitting repeats. The spread across repeats (SD) is "
        f"{spread_text}, against a standard error of {error:.3g}",
    )
    _notes(
        axis,
        [
            "Descriptive: each point is the estimate from one fold split; no interval is "
            "drawn for a single repeat.",
            *shown.notes,
        ],
    )
    return fig


# --------------------------------------------------------------------------- bootstrap


def bootstrap(
    source: Any,
    name: str | None = None,
    *,
    level: float = 0.95,
    units: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the bootstrap replicates with a normal curve for comparison.

    Use it to see whether the bootstrap distribution is symmetric and bell-shaped. When it
    is skewed or heavy-tailed, a Wald interval (the estimate plus or minus 1.96 standard
    errors) can mislead. The histogram shows the replicates, the curve is a normal density
    with their mean and standard deviation, and the vertical lines mark the percentile
    limits. Nothing is resampled here.

    Parameters
    ----------
    source : BootstrapResult or TMLEResult
        The replicates, or a fit that holds them. A fit also marks its reported estimate
        and takes its confidence level.
    name : str or None, default=None
        The estimand to draw. Needed when the replicates hold several.
    level : float, default=0.95
        Coverage of the percentile limits, for a bare ``BootstrapResult``.
    units : str or None, default=None
        Units of the outcome, for the axis label.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    Figure
        One panel.

    Raises
    ------
    ValueError
        When a fit holds no bootstrap, several estimands are held and none is named, or
        fewer than two replicates are finite.

    See Also
    --------
    cleverly.inference.BootstrapResult.interval : The percentile interval as numbers.

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
    >>> fig.axes[0].get_title(loc="left")
    'Is the bootstrap distribution close to normal?'
    >>> plt.close(fig)
    """
    estimate = None
    outcome = None
    if hasattr(source, "estimates"):
        if getattr(source, "bootstrap", None) is None:
            raise ValueError("this fit holds no bootstrap replicates; fit with n_bootstrap > 0")
        result = source
        source = result.bootstrap
        level = 1.0 - float(result.config.alpha_sig)
        outcome = getattr(result.data, "outcome_name", None)
    else:
        result = None
    held = list(source.draws)
    if name is None:
        if len(held) != 1:
            raise ValueError(f"name the estimand to draw; the replicates hold {held}")
        name = held[0]
    if name not in source.draws:
        raise ValueError(f"estimand {name!r} has no replicates; they hold {held}")
    if result is not None and name in result.estimates:
        estimate = result.estimates[name]
    summary = source.interval(name, 1.0 - level)
    draws = np.asarray(summary.draws, dtype=float)
    if draws.size < 2:
        raise ValueError(f"fewer than two finite replicates of {name!r}; there is nothing to plot")
    mean, spread = float(np.mean(draws)), float(np.std(draws, ddof=1))
    fig, axis = _core.figure_for(ax)
    bins = int(min(max(10, round(math.sqrt(draws.size))), 40))
    patches: Any
    _, _, patches = axis.hist(
        draws,
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
        min(float(draws.min()), mean - 3.5 * spread),
        max(float(draws.max()), mean + 3.5 * spread),
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
    low, high = summary.ci
    for limit in (low, high):
        _core.reference_line(
            axis,
            float(limit),
            orientation="vertical",
            label=f"{_core.level_text(level)} percentile limits",
            kind="marker",
        )
    if estimate is not None:
        _core.reference_line(
            axis, float(estimate.psi), orientation="vertical", label="Reported estimate"
        )
    top = max(float(np.max(density)), max(patch.get_height() for patch in patches))
    axis.set_ylim(0.0, top * 1.3)
    _core.corner_label(axis, 0.97, "Bootstrap replicates", _core.PALETTE[0])
    _core.corner_label(axis, 0.88, "Normal approximation (same mean and SD)", _core.PALETTE[1])
    scale = str(estimate.scale) if estimate is not None else "level"
    if estimate is None:
        axis.set_xlabel("Estimate in each bootstrap replicate")
    else:
        axis.set_xlabel(_axis_label(scale, outcome, units).replace(" (log scale)", ""))
    axis.set_ylabel("Density of replicates")
    _core.legend(axis)
    _header(
        axis,
        title or "Is the bootstrap distribution close to normal?",
        f"{name}: {draws.size} replicates, bootstrap standard error {spread:.3g}; "
        f"{_core.level_text(level)} percentile limits {low:.3g} to {high:.3g}",
    )
    found = [
        "Descriptive: the histogram shows the bootstrap replicates, and the curve is a normal "
        "density with their mean and standard deviation, drawn for comparison only.",
        f"Resampling by {source.resampling}.",
    ]
    if source.n_failed:
        found.append(
            f"{source.n_failed} of {source.n_requested} replicates failed and are left out."
        )
    _notes(axis, found)
    return fig


# --------------------------------------------------------------------------- influence


def concentration(influence_curve: Any, cluster: Any = None) -> tuple[np.ndarray, np.ndarray, str]:
    """The cumulative share of squared influence carried by the top units.

    Units are the rows, or the clusters when ``cluster`` is given, in which case the
    influence values are summed within each cluster first (the unit that carries the
    variance on a clustered fit).

    Parameters
    ----------
    influence_curve : array-like
        The per-row influence values.
    cluster : array-like or None, default=None
        Cluster codes, one per row.

    Returns
    -------
    fraction : ndarray
        The fraction of units taken, ``0, 1/m, ..., 1``, largest squared value first.
    share : ndarray
        The share of the sum of squared values those units carry, from 0 to 1.
    units : str
        ``"rows"`` or ``"clusters"``.

    Raises
    ------
    ValueError
        When fewer than two units, a non-finite value, or all values zero.
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
    squares = np.sort(values**2)[::-1]
    total = float(squares.sum())
    if total <= 0.0:
        raise ValueError("the influence values are all zero, so there is no spread to show")
    count = squares.size
    fraction = np.concatenate([[0.0], np.arange(1, count + 1) / count])
    share = np.concatenate([[0.0], np.cumsum(squares) / total])
    return fraction, share, "rows" if cluster is None else "clusters"


def influence(
    source: Any,
    *,
    name: str | None = None,
    cluster: Any = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot how concentrated the uncertainty is in a few rows or clusters.

    Rank the units by their squared influence value, largest first, and plot the share of
    the sum of squared values that the top ``k`` percent carry against ``k``. A curve on
    the dashed diagonal means every unit contributes equally; a curve that rises steeply
    means a few units carry the standard error, which is how a Wald interval misleads even
    without a bootstrap. On a clustered fit the influence values are summed within
    clusters and the unit is the cluster.

    The curve describes the stored influence values. On a fit with several cross-fitting
    repeats, or one evaluated within folds, the reported variance is not computed from this
    curve, and the note says so.

    Parameters
    ----------
    source : ParameterEstimate or TMLEResult
        An estimate, or a fit and the ``name`` of one of its estimates.
    name : str or None, default=None
        The estimate to draw from a fit; the first one when omitted.
    cluster : array-like or None, default=None
        Cluster codes, one per row. A fit supplies its own; a bare estimate from a
        clustered fit needs them.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    Figure
        One panel.

    Raises
    ------
    ValueError
        When the estimate's inference is not from the influence curve (the reason is
        given), when a clustered estimate comes without cluster codes, or when the values
        cannot be ranked.

    See Also
    --------
    cleverly.assessment.DiagnosticsFacade.influence : The same concentration as numbers.

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
    >>> fig.axes[0].get_title(loc="left")
    'How concentrated is the uncertainty?'
    >>> plt.close(fig)
    """
    result = source if hasattr(source, "estimates") else None
    if result is not None:
        label, estimate = _estimate_of(result, name)
        if cluster is None:
            cluster = getattr(result.data, "cluster", None)
    elif hasattr(source, "influence_curve"):
        estimate, label = source, str(source.name)
    else:
        raise TypeError(
            f"influence() takes a ParameterEstimate or a fitted result; got {type(source).__name__}"
        )
    if not estimate.supplies_inference:
        from .._inference_status import status_record

        raise ValueError(
            f"{label}: the interval is not computed from the influence curve, so there is "
            f"nothing to show. {status_record(str(estimate.inference)).plain_reason}"
        )
    if cluster is None and int(estimate.n_clusters) < int(estimate.n):
        raise ValueError(
            f"{label} comes from a clustered fit ({estimate.n_clusters} clusters in "
            f"{estimate.n} rows); pass the fitted result, or cluster= with one code per row"
        )
    fraction, share, units = concentration(estimate.influence_curve, cluster)
    count = fraction.size - 1
    top = max(1, math.ceil(0.01 * count - 1e-12))
    top_share = float(share[top])
    fig, axis = _core.figure_for(ax, figsize=(5.8, 5.4))
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=_core.REFERENCE,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        zorder=1.5,
        label="Even spread",
        gid="cleverly-reference",
    )
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
        f"Top 1%: {_percent(top_share)}",
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
    plural = "clusters" if units == "clusters" else "rows"
    axis.set_xlabel(f"Top share of {plural}, largest squared influence first")
    axis.set_ylabel("Share of the sum of squared influence values")
    _core.legend(axis)
    _header(
        axis,
        title or "How concentrated is the uncertainty?",
        f"{label}: the top 1% of {plural} ({top} of {count}) carry {_percent(top_share)} "
        "of the sum of squared influence values",
    )
    found = [
        "Descriptive: a measure of how the stored influence values are spread; "
        "no interval is drawn."
    ]
    if units == "clusters":
        found.append("Share of clusters: influence values are summed within each cluster first.")
    if result is not None:
        draws = len(getattr(result, "repeats", ()))
        if draws > 1:
            found.append(
                f"This fit combines {draws} cross-fitting repeats, so this curve is not the "
                "reported variance."
            )
        if getattr(getattr(result, "config", None), "cv_evaluation", False):
            found.append(
                "This fit evaluates the estimate within folds, so this curve is not the "
                "reported variance."
            )
    if str(estimate.scale) == "ratio":
        found.append("Influence values are on the log scale.")
    _notes(axis, found)
    return fig


__all__ = ["bootstrap", "calibration", "concentration", "influence", "learners", "repeats"]
