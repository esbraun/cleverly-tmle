"""V13 and V14: how strong would hidden confounding need to be, and how far does it move us.

Two analyses answer the question and neither is an interval:

- :class:`~cleverly.sensitivity.SensitivityContour` holds the bias-adjusted bound of the
  estimate over a grid of two confounder strengths (partial R-squared with treatment and with
  outcome). Each cell is the bound itself, with no sampling error, so the map says nothing
  about the confidence limit, which is not on the grid.
- :class:`~cleverly.sensitivity.SimulatedConfoundingResult` holds the refitted estimate under
  a simulated common cause of declared strengths, in the units of its own strength axes. It is
  a qualitative stress surface, not a bound.

Both draw cells the object holds and nothing between them that the object does not.
"""

from __future__ import annotations

import itertools
import math
import textwrap
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Panels a surface draws before it asks the caller to fix an axis with ``at=``.
MAX_PANELS = 6
#: Cells per side up to which a heat map writes the estimate in each cell.
_ANNOTATE_UP_TO = 7

COLORBAR_LABEL = "Bias-adjusted {bound} bound, no sampling error"
QUALITATIVE_NOTE = "Qualitative: not a bound or an interval."


def _warnings(obj: Any) -> list[str]:
    """The warnings the object's printed summary carries, so the figure says the same."""
    summary = getattr(obj, "summary", None)
    if summary is None:
        return []
    return [note.text for note in summary().notes if note.kind == "warning"]


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))  # type: ignore[return-value]


def _mix(color: str, other: str, share: float) -> tuple[float, float, float]:
    """``color`` moved ``share`` of the way toward ``other``."""
    a, b = _hex_to_rgb(color), _hex_to_rgb(other)
    return (
        a[0] + share * (b[0] - a[0]),
        a[1] + share * (b[1] - a[1]),
        a[2] + share * (b[2] - a[2]),
    )


def _colormap(diverging: bool, *, dark_low: bool) -> Any:
    """Light to dark blue, or orange to grey to blue when the cells straddle the null.

    Both are built from the palette's own slots: blue is slot 1, orange slot 2. The sequential
    map is a single hue, so lightness alone carries the order, and it is never a rainbow.
    """
    from matplotlib.colors import LinearSegmentedColormap

    blue, orange = _core.PALETTE[0], _core.PALETTE[1]
    if diverging:
        return LinearSegmentedColormap.from_list(
            "cleverly-diverging", [orange, _core.GRID, blue], N=256
        )
    light = _mix(blue, _core.FACE, 0.88)
    deep = _mix(blue, _core.INK, 0.4)
    shades: list[Any] = [light, blue, deep]
    cmap = LinearSegmentedColormap.from_list("cleverly-sequential", shades, N=256)
    return cmap.reversed() if dark_low else cmap


def _norm(vmin: float, vmax: float, null: float | None) -> tuple[Any, bool]:
    """The colour norm, and whether it is diverging about the null."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    if null is not None and vmin < null < vmax:
        return TwoSlopeNorm(vcenter=null, vmin=vmin, vmax=vmax), True
    return Normalize(vmin=vmin, vmax=vmax), False


def _tag(contour_set: Any, gid: str) -> None:
    """Name a contour set so tests and callers can find it, across matplotlib versions."""
    setter = getattr(contour_set, "set_gid", None)
    if setter is not None:
        setter(gid)
        return
    for artist in getattr(contour_set, "collections", ()):  # pragma: no cover - mpl < 3.8
        artist.set_gid(gid)


def _text_ink(cmap: Any, norm: Any, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return _core.INK if luminance > 0.45 else _core.FACE


def _number(value: float) -> str:
    return f"{value:.3g}"


# ----------------------------------------------------------------------- V13


def _grid(data: Any) -> tuple[list[float], list[float], np.ndarray[Any, Any]]:
    """The contour's cells as ``cf_d`` values, ``cf_y`` values and ``Z[cf_y, cf_d]``."""
    cf_d = [float(value) for value in data.column("cf_d")]
    cf_y = [float(value) for value in data.column("cf_y")]
    value = [float(item) for item in data.column("value")]
    d_values, y_values = sorted(set(cf_d)), sorted(set(cf_y))
    if len(d_values) * len(y_values) != len(value):
        raise ValueError(
            "the contour's cells do not form a full cf_d by cf_y grid, so they cannot be mapped"
        )
    grid = np.full((len(y_values), len(d_values)), np.nan)
    d_index = {item: index for index, item in enumerate(d_values)}
    y_index = {item: index for index, item in enumerate(y_values)}
    for d, y, v in zip(cf_d, cf_y, value, strict=True):
        grid[y_index[y], d_index[d]] = v
    if not np.all(np.isfinite(grid)):
        raise ValueError("the contour holds a cell with no finite value, so it cannot be mapped")
    return d_values, y_values, grid


def _levels(vmin: float, vmax: float, null: float) -> np.ndarray[Any, Any]:
    """Round contour levels that include the null whenever the cells reach it."""
    from matplotlib.ticker import MaxNLocator

    shifted = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin - null, vmax - null)
    return np.asarray(shifted, dtype=float) + null


def _check_robustness(data: Any, robustness: Any, null: float | None) -> tuple[float | None, float]:
    """The robustness value to draw (``None`` when it does not exist) and the null.

    Raises
    ------
    ValueError
        When the robustness value belongs to another estimand, another alignment, another
        estimate or another end of the range than the contour.
    """
    if robustness is None:
        return None, 0.0 if null is None else float(null)
    declared = float(robustness.null_hypothesis)
    if null is not None and not math.isclose(float(null), declared, abs_tol=1e-12):
        raise ValueError(
            f"null={null:g} differs from the robustness value's null_hypothesis={declared:g}; "
            "the bold contour and the robustness value must read the same null"
        )
    if robustness.estimand != data.estimand:
        raise ValueError(
            f"the robustness value is for {robustness.estimand!r} but the contour bounds "
            f"{data.estimand!r}"
        )
    if not math.isclose(robustness.rho, data.rho, abs_tol=1e-12):
        raise ValueError(
            f"the robustness value assumes rho = {robustness.rho:g} but the contour assumes "
            f"rho = {data.rho:g}"
        )
    if not math.isclose(robustness.psi, data.psi, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("the robustness value and the contour describe different estimates")
    needed = "lower" if data.psi > declared else "upper"
    if data.psi != declared and needed != data.bound:
        raise ValueError(
            f"the robustness value is where the {needed} bound reaches the null, but this "
            f"contour maps the {data.bound} bound; build it with contour_data(..., "
            f"bound={needed!r})"
        )
    rv = robustness.get("rv")
    return (None if rv is None else float(rv)), declared


def _narrow(text: str, ax: Axes, size: float, share: float) -> str:
    """``text`` wrapped to ``share`` of the width :func:`_core.wrap` allows.

    A colour bar sits beside the axes, and the layout places it past the widest text the
    axes carries. A title, subtitle or note as wide as the figure would leave a gap.
    """
    figure: Any = ax.figure
    while not hasattr(figure, "get_figwidth"):
        figure = figure.figure
    characters = max(24, int(figure.get_figwidth() * 0.92 * 72.0 / (size * 0.52) * share))
    return "\n".join(textwrap.fill(part, characters) for part in text.split("\n"))


def _notes(ax: Axes, lines: Sequence[str], share: float) -> None:
    _core.notes(ax, [_narrow(line, ax, _core.NOTE_SIZE, share) for line in lines])


def _header(ax: Axes, title: str, subtitle: str, share: float) -> None:
    _core.header(
        ax,
        _narrow(title, ax, _core.TITLE_SIZE * 1.1, min(share + 0.2, 1.0)),
        _narrow(subtitle, ax, _core.SUBTITLE_SIZE, share),
    )


def _label_points(lines: Any, d_max: float, y_max: float) -> list[tuple[float, float]]:
    """One vertex per level, nearest the middle of the map and clear of its edges."""
    target = np.array([0.6 * d_max, 0.6 * y_max])
    chosen = []
    for level_segments in lines.allsegs:
        segments = [seg for seg in level_segments if len(seg)]
        if not segments:
            continue
        points = np.vstack(segments)
        clear = (
            (points[:, 0] < 0.96 * d_max)
            & (points[:, 1] < 0.96 * y_max)
            & (points[:, 0] > 0.04 * d_max)
            & (points[:, 1] > 0.04 * y_max)
        )
        inside = points[clear]
        if not len(inside):
            continue
        best = inside[int(np.argmin(np.hypot(*(inside - target).T)))]
        chosen.append((float(best[0]), float(best[1])))
    return chosen


def _null_label(bold: Any, rv: float | None) -> tuple[float, float] | None:
    """A vertex of the bold contour clear of the robustness value, for its label."""
    segments = [seg for seg in getattr(bold, "allsegs", [[]])[0] if len(seg)]
    if not segments:
        return None
    points = np.vstack(segments)
    if rv is None:
        index = len(points) // 2
    else:
        # the contour passes through the robustness value; a quarter of the way along it
        # is clear of that marker and of the edges
        quarter = [len(points) // 4, (3 * len(points)) // 4]
        index = max(quarter, key=lambda i: float(np.hypot(*(points[i] - np.array([rv, rv])))))
    return float(points[index, 0]), float(points[index, 1])


def contour(
    data: Any,
    *,
    robustness: Any = None,
    benchmarks: Sequence[Any] | Any = (),
    multiples: Sequence[float] = (1.0, 2.0),
    null: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the bias-adjusted bound over the strengths of a hidden confounder.

    Parameters
    ----------
    data : SensitivityContour
        From :func:`~cleverly.sensitivity.contour_data`.
    robustness : RobustnessValue or None, default=None
        From :func:`~cleverly.sensitivity.robustness_value` for the same estimand and
        alignment. Its point value is marked on the diagonal ``cf_d = cf_y``. The confidence
        value is not drawn: the limit is not on this grid and the plot computes none.
    benchmarks : BenchmarkResult or sequence of BenchmarkResult, default=()
        From :func:`~cleverly.sensitivity.benchmark`. Each is marked at its multiples.
    multiples : sequence of float, default=(1.0, 2.0)
        How many times each benchmark's two gains to mark. A mark outside the grid is
        named in the note instead.
    null : float or None, default=None
        The value the bold contour marks. The robustness value's own null when ``robustness``
        is given, else ``0``.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on; a new figure when omitted.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    ValueError
        When the cells do not form a full grid or are all equal, when the robustness value or
        a benchmark does not belong to this contour, or when ``null`` and the robustness
        value disagree.
    """
    d_values, y_values, grid = _grid(data)
    rv, null_value = _check_robustness(data, robustness, null)
    marks = list(benchmarks) if isinstance(benchmarks, Sequence) else [benchmarks]
    for record in marks:
        if record.estimand != data.estimand:
            raise ValueError(
                f"a benchmark is for {record.estimand!r} but the contour bounds {data.estimand!r}"
            )
    vmin, vmax = float(np.min(grid)), float(np.max(grid))
    if not vmax > vmin:
        raise ValueError(
            "every cell of the contour has the same value, because the estimate has no bias "
            "scale to move it; there is no map to draw"
        )

    fig, axis = _core.figure_for(ax, figsize=(7.0, 5.2))
    _core.style_axes(axis, grid="none")
    lower = data.bound == "lower"
    norm, diverging = _norm(vmin, vmax, null_value)
    cmap = _colormap(diverging, dark_low=lower)
    levels = _levels(vmin, vmax, null_value)
    xs, ys = np.meshgrid(d_values, y_values)
    filled = axis.contourf(xs, ys, grid, levels=levels, cmap=cmap, norm=norm, extend="neither")
    _tag(filled, "cleverly-contour-fill")
    thin = [level for level in levels if vmin < level < vmax and level != null_value]
    lines = axis.contour(xs, ys, grid, levels=thin, colors=_core.FACE, linewidths=0.6, alpha=0.9)
    _tag(lines, "cleverly-contour-lines")
    label_at = _label_points(lines, d_values[-1], y_values[-1])
    if label_at:
        try:
            for text in axis.clabel(
                lines,
                fmt=lambda level: f"{level:.3g}",
                fontsize=_core.TICK_SIZE - 1,
                manual=label_at,
            ):
                text.set_color(_core.INK)
        except Exception:  # pragma: no cover - labelling is cosmetic
            pass
    reaches = vmin <= null_value <= vmax
    if reaches:
        bold = axis.contour(xs, ys, grid, levels=[null_value], colors=_core.INK, linewidths=2.6)
        _tag(bold, "cleverly-null-contour")
        where = _null_label(bold, rv)
        if where is not None:
            axis.annotate(
                "No effect", xy=where, xytext=(-7.0, -7.0), textcoords="offset points",
                ha="right", va="top", fontsize=_core.TICK_SIZE, fontweight="bold",
                color=_core.INK, gid="cleverly-null-label", zorder=6,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": _core.FACE,
                      "edgecolor": "none", "alpha": 0.85},
            )  # fmt: skip

    bar = fig.colorbar(filled, ax=axis)
    bar.set_label(COLORBAR_LABEL.format(bound=data.bound), fontsize=_core.LABEL_SIZE)
    bar.ax.tick_params(labelsize=_core.TICK_SIZE, colors=_core.SECONDARY_INK)
    bar.outline.set_visible(False)
    bar.ax.set_gid("cleverly-colorbar")

    d_max, y_max = d_values[-1], y_values[-1]
    found: list[str] = []
    if rv is not None:
        if rv <= min(d_max, y_max):
            diagonal = min(d_max, y_max)
            axis.plot(
                [0.0, diagonal], [0.0, diagonal],
                color=_core.SECONDARY_INK, linewidth=0.8, linestyle=(0, (1, 2)),
                gid="cleverly-diagonal", zorder=3,
            )  # fmt: skip
            axis.plot(
                [rv], [rv], marker="D", markersize=7, color=_core.INK,
                markeredgecolor=_core.FACE, markeredgewidth=1.0, linestyle="none",
                gid="cleverly-robustness-value", zorder=5,
            )  # fmt: skip
            axis.annotate(
                f"Robustness value {_number(rv)}",
                xy=(rv, rv), xytext=(-8.0, -6.0), textcoords="offset points",
                ha="right", va="top", fontsize=_core.TICK_SIZE, color=_core.INK,
                bbox={"boxstyle": "square,pad=0.2", "facecolor": _core.FACE,
                      "edgecolor": "none", "alpha": 0.85},
                gid="cleverly-robustness-label", zorder=6,
            )  # fmt: skip
        else:
            found.append(
                f"The robustness value, {_number(rv)}, lies beyond this grid, which reaches "
                f"{_number(d_max)} for cf_d and {_number(y_max)} for cf_y."
            )
    elif robustness is not None:
        found.append("No equal strength below one brings the estimate to the null.")

    off: list[str] = []
    for record in marks:
        label = ", ".join(record.covariates)
        for multiple in multiples:
            at_d, at_y = float(multiple) * record.cf_d, float(multiple) * record.cf_y
            mark = f"{label} \u00d7{multiple:g}" if multiple != 1 else label
            if at_d > d_max or at_y > y_max:
                off.append(f"{mark} (cf_d {_number(at_d)}, cf_y {_number(at_y)})")
                continue
            axis.plot(
                [at_d], [at_y], marker="^", markersize=7, color=_core.INK,
                markeredgecolor=_core.FACE, markeredgewidth=1.0, linestyle="none",
                gid="cleverly-benchmark", zorder=5,
            )  # fmt: skip
            axis.annotate(
                mark, xy=(at_d, at_y), xytext=(6.0, 5.0), textcoords="offset points",
                ha="left", va="bottom", fontsize=_core.TICK_SIZE, color=_core.INK,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": _core.FACE,
                      "edgecolor": "none", "alpha": 0.85},
                gid="cleverly-benchmark-label", zorder=6,
            )  # fmt: skip
    if off:
        found.append("Benchmarks beyond the grid: " + "; ".join(off) + ".")
    if not reaches:
        found.append(
            f"No cell on this grid reaches {null_value:g}: the estimate holds up to the "
            "strongest confounding shown."
        )

    axis.set_xlim(d_values[0], d_max)
    axis.set_ylim(y_values[0], y_max)
    axis.set_xlabel("Confounder strength in treatment, cf_d (partial R²)")
    axis.set_ylabel("Confounder strength in outcome, cf_y (partial R²)")
    subtitle = (
        f"{data.bound.capitalize()} bound of {data.estimand} (estimate "
        f"{_core.format_values([data.psi], [abs(vmax - vmin) / 20.0])[0]}) "
        f"at alignment \u03c1 = {data.rho:g}"
    )
    if rv is not None and rv <= min(d_max, y_max):
        subtitle += f"; point robustness value {_number(rv)}"
    _header(axis, title or "How strong would hidden confounding need to be?", subtitle, 0.66)
    _notes(
        axis,
        [
            f"Each cell is the bias-adjusted {data.bound} bound with no sampling error: no "
            "confidence limit is drawn, and the limit is not on this grid. Shading between "
            "the cells is contouring of the grid.",
            *found,
            *_warnings(data),
        ],
        0.72,
    )
    return fig


# ----------------------------------------------------------------------- V14


def _null_of(result: Any) -> float | None:
    """The no-effect value of the surface's estimand, or ``None`` for a level."""
    if result.movement_scale == "log_ratio":
        return 1.0
    stem = str(result.estimand).split("[", 1)[0].split("_", 1)[0]
    if stem in ("ate", "att", "atc", "par", "paf") or str(result.estimand).startswith("contrast"):
        return 0.0
    return None


def _treatment_labels(result: Any) -> list[str]:
    """One axis label per treatment dimension, each in the units the grid declares."""
    family = result.treatment_family
    names = result.treatment_components
    if names:
        return [f"Strength on {name} (its own units: flip probability or signed coefficient)"
                for name in names]  # fmt: skip
    if family == "binary":
        return ["Probability that treatment is flipped"]
    if family == "categorical":
        return ["Probability that the two arms are exchanged"]
    return ["Treatment shift per unit of the hidden cause (signed coefficient)"]


def _outcome_label(result: Any) -> str:
    if result.bounded_outcome_law is not None:
        return f"Outcome strength (on the {result.bounded_outcome_law} scale of a bounded outcome)"
    if result.outcome_family == "gaussian":
        return "Outcome shift per unit of the hidden cause (signed coefficient)"
    return "Probability that the outcome is flipped"


def _component(strength: Any, index: int) -> float:
    return float(strength[index] if isinstance(strength, tuple) else strength)


def _key(value: float) -> float:
    return round(float(value), 12)


def _edges(values: Sequence[float]) -> np.ndarray[Any, Any]:
    """Cell edges at the midpoints between declared values, so each value sits in its cell."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        half = 0.5 * abs(v[0]) if v[0] != 0.0 else 0.5
        return np.array([v[0] - half, v[0] + half])
    mid = (v[1:] + v[:-1]) / 2.0
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def _selected(at: Mapping[str, Any] | None, name: str, declared: list[float]) -> list[float]:
    if at is None or name not in at:
        return declared
    wanted = at[name]
    wanted_list = [wanted] if isinstance(wanted, int | float) else list(wanted)
    chosen = []
    for item in wanted_list:
        match = [value for value in declared if math.isclose(value, float(item), abs_tol=1e-12)]
        if not match:
            raise ValueError(
                f"at[{name!r}] = {item!r} is not a declared value; the grid declares {declared}"
            )
        chosen.append(match[0])
    return chosen


def _calibration_marks(
    result: Any,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Calibrations whose scale equals an axis's, per axis, and the rest as words."""
    x_marks: list[tuple[str, float]] = []
    y_marks: list[tuple[str, float]] = []
    elsewhere: list[str] = []
    for row in result.calibrations:
        strength = float(row.strength)
        unit = "share of predictions changed" if row.family == "binomial" else "signed coefficient"
        where = f"{row.covariate} ({row.role}, {unit}, "
        where += f"{_number(strength)})" if math.isfinite(strength) else "no finite value)"
        if not math.isfinite(strength):
            elsewhere.append(where)
        elif (
            row.role == "treatment"
            and row.family == "binomial"
            and result.treatment_family == "binary"
            and not result.treatment_components
        ):
            x_marks.append((row.covariate, strength))
        elif (
            row.role == "outcome"
            and row.family == "binomial"
            and result.outcome_family == "binomial"
            and result.bounded_outcome_law is None
        ):
            y_marks.append((row.covariate, strength))
        else:
            elsewhere.append(where)
    return x_marks, y_marks, elsewhere


def surface(
    result: Any,
    *,
    at: Mapping[str, Any] | None = None,
    null: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the refitted estimate over the declared strengths of a simulated common cause.

    Parameters
    ----------
    result : SimulatedConfoundingResult
        From :func:`~cleverly.sensitivity.simulated_confounding`.
    at : mapping of str to float or sequence of float or None, default=None
        For a grid with more than two strength axes (several treatment components, or an
        intermediate): which declared values of the extra axes to draw, one panel per
        combination. The first treatment axis and the outcome axis are always the plot's own
        axes. All declared values are drawn when omitted.
    null : float or None, default=None
        The no-effect value, outlined where the estimate crosses it. Read from the estimand
        when omitted; a level (a counterfactual mean) has none.
    title : str or None, default=None
        The title.
    ax : Axes or None, default=None
        Axes to draw on. Only a one-panel surface can use it.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    ValueError
        When the grid has more panels than :data:`MAX_PANELS` (pass ``at=``), when ``at``
        names an axis or value the grid does not declare, when ``ax=`` is given for a
        multi-panel surface, or when no cell produced an estimate.
    """
    cells = list(result.cells)
    components = tuple(result.treatment_components)
    treat_names = list(components) if components else ["treatment"]
    treat_labels = _treatment_labels(result)
    has_intermediate = result.intermediate_name is not None
    inter_values = sorted(
        {float(c.intermediate_strength) for c in cells if c.intermediate_strength is not None}
    )

    def treat(cell: Any, index: int) -> float:
        return _component(cell.treatment_strength, index)

    dims: dict[str, list[float]] = {
        name: sorted({_key(treat(cell, index)) for cell in cells})
        for index, name in enumerate(treat_names)
    }
    y_values = sorted({_key(cell.outcome_strength) for cell in cells})
    extra_names = treat_names[1:]
    if has_intermediate and len(inter_values) > 1:
        extra_names = [*extra_names, str(result.intermediate_name)]
        dims[str(result.intermediate_name)] = [_key(value) for value in inter_values]
    if at is not None:
        unknown = [name for name in at if name not in extra_names]
        if unknown:
            raise ValueError(
                f"at names {unknown}, which are not extra axes of this grid; the first treatment "
                f"axis ({treat_names[0]!r}) and the outcome are the plot's own axes, and the "
                f"extra axes are {extra_names}"
            )
    chosen = {name: _selected(at, name, dims[name]) for name in extra_names}
    combos = list(itertools.product(*[chosen[name] for name in extra_names]))
    if len(combos) > MAX_PANELS:
        raise ValueError(
            f"this grid has {len(combos)} combinations of {extra_names}, more than the "
            f"{MAX_PANELS} panels one figure draws; pass at= to fix some of them, for "
            f"example at={{{extra_names[0]!r}: 0.0}}"
        )
    if ax is not None and len(combos) > 1:
        raise ValueError("ax= takes a one-panel surface; this grid needs several panels")
    x_values = dims[treat_names[0]]

    def extras(cell: Any) -> tuple[float, ...]:
        found = [_key(treat(cell, treat_names.index(name))) for name in treat_names[1:]]
        if has_intermediate and len(inter_values) > 1:
            found.append(_key(cell.intermediate_strength))
        return tuple(found)

    lookup = {(_key(treat(cell, 0)), _key(cell.outcome_strength), extras(cell)): cell
              for cell in cells}  # fmt: skip
    drawn = [lookup[(x, y, combo)] for combo in combos for x in x_values for y in y_values
             if (x, y, combo) in lookup]  # fmt: skip
    estimates = [float(c.estimate) for c in drawn if c.failure is None and c.estimate is not None]
    if not estimates:
        raise ValueError("no cell of this surface produced an estimate, so there is nothing to map")
    null_value = _null_of(result) if null is None else float(null)
    vmin, vmax = min(estimates), max(estimates)
    if vmax == vmin:
        vmax = vmin + max(abs(vmin) * 1e-6, 1e-9)
    norm, diverging = _norm(vmin, vmax, null_value)
    cmap = _colormap(diverging, dark_low=False)

    panels = len(combos)
    ncols = min(3, panels)
    nrows = math.ceil(panels / ncols)
    if panels == 1:
        fig, axis = _core.figure_for(ax, figsize=(7.0, 4.9))
        axes_list = [axis]
        top = foot = axis
    else:
        fig = _core.pyplot().figure(
            figsize=(3.0 * ncols + 1.4, 2.7 * nrows + 3.4), layout="constrained"
        )
        fig.set_facecolor(_core.FACE)
        spec = fig.add_gridspec(nrows + 2, ncols, height_ratios=[0.02, *([1.0] * nrows), 0.02])
        top = fig.add_subplot(spec[0, :])
        top.axis("off")
        foot = fig.add_subplot(spec[-1, :])
        foot.axis("off")
        axes_list = [
            fig.add_subplot(spec[1 + index // ncols, index % ncols]) for index in range(panels)
        ]

    mesh = None
    failed_total = 0
    outlined = 0
    small = len(x_values) <= _ANNOTATE_UP_TO and len(y_values) <= _ANNOTATE_UP_TO
    annotate = small and (ncols < 3 or len(x_values) * len(y_values) <= 12)
    digits = _core.format_values(estimates, [(vmax - vmin) / 12.0] * len(estimates))
    shown = dict(zip(map(float, estimates), digits, strict=False))
    x_edges, y_edges = _edges(x_values), _edges(y_values)
    for axis_i, combo in zip(axes_list, combos, strict=True):
        _core.style_axes(axis_i, grid="none")
        values = np.full((len(y_values), len(x_values)), np.nan)
        failed: list[tuple[int, int]] = []
        for iy, y in enumerate(y_values):
            for ix, x in enumerate(x_values):
                cell = lookup.get((x, y, combo))
                if cell is None:
                    continue
                if cell.failure is not None or cell.estimate is None:
                    failed.append((ix, iy))
                else:
                    values[iy, ix] = float(cell.estimate)
        failed_total += len(failed)
        mesh = axis_i.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(values), cmap=cmap, norm=norm,
            shading="flat", edgecolor=_core.FACE, linewidth=0.8, gid="cleverly-heatmap",
        )  # fmt: skip
        for ix, iy in failed:
            from matplotlib.patches import Rectangle

            axis_i.add_patch(
                Rectangle(
                    (x_edges[ix], y_edges[iy]), x_edges[ix + 1] - x_edges[ix],
                    y_edges[iy + 1] - y_edges[iy], facecolor=_core.FACE,
                    edgecolor=_core.REFERENCE, hatch="////", linewidth=0.0,
                    gid="cleverly-failed-cell",
                )
            )  # fmt: skip
        if null_value is not None:
            outlined += _outline_null(axis_i, values, x_edges, y_edges, null_value)
        if annotate:
            for iy in range(len(y_values)):
                for ix in range(len(x_values)):
                    if np.isfinite(values[iy, ix]):
                        axis_i.text(
                            (x_edges[ix] + x_edges[ix + 1]) / 2.0,
                            (y_edges[iy] + y_edges[iy + 1]) / 2.0,
                            shown.get(float(values[iy, ix]), _number(values[iy, ix])),
                            ha="center", va="center", fontsize=_core.TICK_SIZE - 0.5,
                            color=_text_ink(cmap, norm, values[iy, ix]),
                            gid="cleverly-cell-value",
                        )  # fmt: skip
        axis_i.set_xlim(x_edges[0], x_edges[-1])
        axis_i.set_ylim(y_edges[0], y_edges[-1])
        axis_i.set_xticks(x_values)
        axis_i.set_xticklabels([f"{value:g}" for value in x_values])
        axis_i.set_yticks(y_values)
        axis_i.set_yticklabels([f"{value:g}" for value in y_values])
        axis_i.tick_params(length=0)
        for spine in axis_i.spines.values():
            spine.set_visible(False)
        if panels > 1:
            pairs = zip(extra_names, combo, strict=True)
            axis_i.set_title(
                ", ".join(f"{name} = {value:g}" for name, value in pairs),
                fontsize=_core.TICK_SIZE + 0.5,
                color=_core.INK,
                pad=4.0,
            )

    x_marks, y_marks, elsewhere = _calibration_marks(result)
    placed, outside = _mark_calibrations(
        axes_list[-1], sorted(x_marks, key=lambda m: m[1]), sorted(y_marks, key=lambda m: m[1]),
        x_values, y_values,
    )  # fmt: skip
    short_x = f"Strength on {treat_names[0]}"
    short_y = "Outcome strength"
    for index, axis_i in enumerate(axes_list):
        axis_i.set_xlabel(
            textwrap.fill(treat_labels[0] if panels == 1 else short_x, 44 if panels == 1 else 26)
            if index + ncols >= panels
            else ""
        )
        axis_i.set_ylabel(
            textwrap.fill(_outcome_label(result) if panels == 1 else short_y, 30)
            if index % ncols == 0
            else ""
        )

    assert mesh is not None
    bar = fig.colorbar(mesh, ax=axes_list)
    bar.set_label(f"Estimate of {result.estimand} (qualitative)", fontsize=_core.LABEL_SIZE)
    bar.ax.tick_params(labelsize=_core.TICK_SIZE, colors=_core.SECONDARY_INK)
    bar.outline.set_visible(False)
    bar.ax.set_gid("cleverly-colorbar")

    original = _core.format_values([float(result.original_estimate)], [(vmax - vmin) / 12.0])[0]
    subtitle = (
        f"Estimate of {result.estimand} refitted under a simulated common cause of treatment "
        f"and outcome (unperturbed estimate {original})"
    )
    _header(
        top,
        title or "How far does the estimate move under a simulated common cause?",
        subtitle,
        0.66 if panels == 1 else 0.85,
    )

    found = [
        QUALITATIVE_NOTE
        + " It shows how far the refitted estimate moves when a simulated common cause of the "
        "declared strengths is added; the strengths are in the axes' own units, and no "
        "strength is a statement about the confounding in these data."
    ]
    if null_value is not None:
        found.append(
            f"Heavy outline: where the estimate crosses {null_value:g}."
            if outlined
            else f"No drawn cell crosses {null_value:g}, so there is no outline."
        )
    if failed_total:
        found.append(
            f"Hatched cells ({failed_total} of {len(drawn)} drawn) failed to refit and have no "
            "estimate."
        )
    if panels > 1:
        found.append(
            f"One panel per combination of {', '.join(extra_names)}. Axes are in the strengths' "
            f"own units: {treat_labels[0][0].lower() + treat_labels[0][1:]}; "
            f"{_outcome_label(result)[0].lower() + _outcome_label(result)[1:]}."
        )
    if placed:
        found.append(
            "Triangles on the axes mark observed covariates on the same scale: "
            + "; ".join(f"{name} {_number(value)}" for name, value in placed)
            + "."
        )
    if outside:
        found.append(
            "Observed covariates beyond the axes: "
            + "; ".join(f"{name} {_number(value)}" for name, value in outside)
            + "."
        )
    if elsewhere:
        found.append(
            "Observed-covariate calibrations on another scale than an axis, so not marked: "
            + "; ".join(elsewhere)
            + "."
        )
    _notes(foot, [*found, *_warnings(result)], 0.72 if panels == 1 else 0.8)
    return fig


def _outline_null(
    axis: Axes,
    values: np.ndarray[Any, Any],
    x_edges: np.ndarray[Any, Any],
    y_edges: np.ndarray[Any, Any],
    null: float,
) -> int:
    """Heavy segments along the cell edges that separate estimates on either side of the null.

    Returns the number of segments drawn.
    """
    from matplotlib.collections import LineCollection

    segments = []
    ny, nx = values.shape
    for iy in range(ny):
        for ix in range(nx):
            here = values[iy, ix]
            if not np.isfinite(here):
                continue
            right = values[iy, ix + 1] if ix + 1 < nx else np.nan
            above = values[iy + 1, ix] if iy + 1 < ny else np.nan
            if np.isfinite(right) and (here >= null) != (right >= null):
                segments.append(
                    [(x_edges[ix + 1], y_edges[iy]), (x_edges[ix + 1], y_edges[iy + 1])]
                )
            if np.isfinite(above) and (here >= null) != (above >= null):
                segments.append(
                    [(x_edges[ix], y_edges[iy + 1]), (x_edges[ix + 1], y_edges[iy + 1])]
                )
    if segments:
        collection = LineCollection(
            segments, colors=_core.INK, linewidths=2.8, capstyle="round", zorder=4,
            gid="cleverly-null-crossing",
        )  # fmt: skip
        axis.add_collection(collection)
    return len(segments)


def _mark_calibrations(
    axis: Axes,
    x_marks: list[tuple[str, float]],
    y_marks: list[tuple[str, float]],
    x_values: list[float],
    y_values: list[float],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Triangles on an axis at observed-covariate strengths that lie within it."""
    placed: list[tuple[str, float]] = []
    outside: list[tuple[str, float]] = []
    for name, value in x_marks:
        if x_values[0] <= value <= x_values[-1]:
            axis.plot(
                [value], [0.0], marker="^", markersize=7, color=_core.INK,
                markeredgecolor=_core.FACE, transform=axis.get_xaxis_transform(),
                clip_on=False, linestyle="none", gid="cleverly-calibration", zorder=6,
            )  # fmt: skip
            placed.append((f"{name} (treatment)", value))
        else:
            outside.append((f"{name} (treatment)", value))
    for name, value in y_marks:
        if y_values[0] <= value <= y_values[-1]:
            axis.plot(
                [0.0], [value], marker=">", markersize=7, color=_core.INK,
                markeredgecolor=_core.FACE, transform=axis.get_yaxis_transform(),
                clip_on=False, linestyle="none", gid="cleverly-calibration", zorder=6,
            )  # fmt: skip
            placed.append((f"{name} (outcome)", value))
        else:
            outside.append((f"{name} (outcome)", value))
    return placed, outside


# ----------------------------------------------------------------------- dispatch


def confounding(obj: Any, **kwargs: Any) -> Figure:
    """Plot how strong hidden confounding would need to be, or how far a simulated one moves us.

    Use it to read a sensitivity analysis to confounding that was not measured. For a
    :class:`~cleverly.sensitivity.SensitivityContour` it maps the bias-adjusted bound of the
    estimate over two confounder strengths, with the contour where the bound reaches no effect
    in bold, the point robustness value on the diagonal and benchmark covariates marked at
    their multiples. For a :class:`~cleverly.sensitivity.SimulatedConfoundingResult` it maps
    the refitted estimate over the declared strengths of a simulated common cause, in the
    units of its own axes, and says it is qualitative.

    Parameters
    ----------
    obj : SensitivityContour or SimulatedConfoundingResult
        The analysis to draw.
    **kwargs
        For a contour: ``robustness=``, ``benchmarks=``, ``multiples=``, ``null=``, ``title=``
        and ``ax=``. For a surface: ``at=``, ``null=``, ``title=`` and ``ax=``.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    TypeError
        When ``obj`` is neither analysis.

    See Also
    --------
    cleverly.plotting.sensitivity : Estimates across a declared sensitivity parameter.
    cleverly.sensitivity.contour_data : Build the grid of bias-adjusted bounds.
    cleverly.sensitivity.simulated_confounding : Build the simulated-confounding surface.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.plotting import confounding
    >>> from cleverly.sensitivity import contour_data, robustness_value
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
    >>> grid = contour_data(result, "ate", grid_size=8)
    >>> fig = confounding(grid, robustness=robustness_value(result, "ate"))
    >>> fig.axes[0].get_title(loc="left")
    'How strong would hidden confounding need to be?'
    >>> plt.close(fig)
    """
    from ..sensitivity import SensitivityContour, SimulatedConfoundingResult

    if isinstance(obj, SensitivityContour):
        return contour(obj, **kwargs)
    if isinstance(obj, SimulatedConfoundingResult):
        return surface(obj, **kwargs)
    raise TypeError(
        "confounding() takes a SensitivityContour or SimulatedConfoundingResult; "
        f"got {type(obj).__name__}"
    )


__all__ = ["COLORBAR_LABEL", "MAX_PANELS", "confounding", "contour", "surface"]
