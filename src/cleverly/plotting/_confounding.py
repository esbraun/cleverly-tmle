"""V13 and V14: how strong would hidden confounding need to be, and how far does it move us.

Two analyses answer the question and neither is an interval:

- :class:`~cleverly.sensitivity.SensitivityContour` holds the bias-adjusted bound of the
  estimate over a grid of two confounder strengths, in the parametrisation of Chernozhukov,
  Cinelli, Newey, Sharma and Syrgkanis (Long story short, Theorem 2): ``cf_y``, the share of
  the outcome variation left after adjustment that left-out variables would explain, and
  ``cf_d``, the share of the average squared weight that only they would add. Each cell is
  the bound itself, with no sampling error, so the map says nothing about the confidence
  limit, which is not on the grid.
- :class:`~cleverly.sensitivity.SimulatedConfoundingResult` holds the refitted estimate under
  a simulated common cause of declared strengths, in the units of its own perturbation laws.
  It is a qualitative stress surface, not a bound.

Both draw cells the object holds and nothing between them that the object does not. The
null each outlines comes from the scale the object records, never from the estimand's name.
"""

from __future__ import annotations

import itertools
import math
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._refusal import RefusalReason
from ..exceptions import CapabilityError
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

#: The ``gid`` each kind of artist of these two plots carries.
GID_CONTOUR_FILL = "cleverly-contour-fill"
GID_CONTOUR_LINES = "cleverly-contour-lines"
GID_NULL_CONTOUR = "cleverly-null-contour"
GID_NULL_LABEL = "cleverly-null-label"
GID_DIAGONAL = "cleverly-diagonal"
GID_ROBUSTNESS = "cleverly-robustness-value"
GID_ROBUSTNESS_LABEL = "cleverly-robustness-label"
GID_BENCHMARK = "cleverly-benchmark"
GID_BENCHMARK_LABEL = "cleverly-benchmark-label"
GID_COLORBAR = "cleverly-colorbar"
GID_HEATMAP = "cleverly-heatmap"
GID_NULL_CROSSING = "cleverly-null-crossing"
GID_FAILED_CELL = "cleverly-failed-cell"
GID_CELL_VALUE = "cleverly-cell-value"
GID_CALIBRATION = "cleverly-calibration"
GID_CALIBRATION_LABEL = "cleverly-calibration-label"

#: The two strengths in the package's own words (``_summaries._definitions_bounds``).
STRENGTHS_NOTE = (
    "cf_y: the share of the outcome variation left after adjustment that left-out variables "
    "would explain. cf_d: the share of the average squared weight that only they would add, "
    "not a partial R\u00b2 with the treatment (Chernozhukov et al., Long story short)."
)
#: Where a simulated cause acts, in the words of the perturbation laws.
_TAIL = "those highest on the simulated cause"


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


def _colour_scale(
    vmin: float, vmax: float, null: float | None, *, ratio: bool, dark_low: bool
) -> tuple[Any, Any, bool]:
    """The colour map, the norm and whether the scale diverges about the null.

    A diverging scale is symmetric about the null: equal distances on either side (in logs,
    for a ratio) get equal saturation, and the map is cut to the range the cells reach, so a
    small excursion below the null stays pale. A ratio is coloured on a log scale.
    """
    from matplotlib.colors import ListedColormap, LogNorm, Normalize

    norm = LogNorm(vmin=vmin, vmax=vmax) if ratio else Normalize(vmin=vmin, vmax=vmax)
    if null is None or not vmin < null < vmax:
        return _colormap(False, dark_low=dark_low), norm, False
    move: Callable[[float], float] = math.log if ratio else float
    below, above = move(null) - move(vmin), move(vmax) - move(null)
    half = max(below, above)
    low, high = 0.5 - below / (2.0 * half), 0.5 + above / (2.0 * half)
    full = _colormap(True, dark_low=False)
    cut = ListedColormap(full(np.linspace(low, high, 256)), name="cleverly-diverging")
    return cut, norm, True


def _bar_ticks(bar: Any, vmin: float, vmax: float, null: float | None, ratio: bool) -> None:
    """Round colour-bar ticks over the cells' range, with the null among them."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

    ticks = [
        float(t) for t in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    ]
    ticks = [t for t in ticks if vmin - 1e-12 <= t <= vmax + 1e-12 and (t > 0 or not ratio)]
    if null is not None and vmin <= null <= vmax:
        spacing = (vmax - vmin) / 12.0
        ticks = [t for t in ticks if abs(t - null) > spacing] + [null]
    bar.set_ticks(sorted(ticks))
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    bar.ax.yaxis.set_minor_locator(NullLocator())


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


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _offset(fx: float, fy: float, *, right: bool, up: bool) -> tuple[tuple[float, float], str, str]:
    """A label offset that keeps the label inside the axes.

    ``fx`` and ``fy`` are the point's position in axes fractions. The label goes to the
    preferred side unless the point is close to that edge, where it flips.
    """
    if right and fx > 0.75:
        right = False
    elif not right and fx < 0.3:
        right = True
    if up and fy > 0.85:
        up = False
    elif not up and fy < 0.15:
        up = True
    return (
        (7.0 if right else -7.0, 5.0 if up else -6.0),
        "left" if right else "right",
        "bottom" if up else "top",
    )


_LABEL_BOX = {"boxstyle": "square,pad=0.15", "facecolor": _core.FACE, "edgecolor": "none",
              "alpha": 0.85}  # fmt: skip


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


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def _contour_null(data: Any, robustness: Any, null: float | None) -> tuple[float, bool]:
    """The value the bold contour marks, and whether it is the scale's no-effect value.

    Raises
    ------
    ValueError
        When ``null`` and the robustness value disagree, or when a level has no null and
        none is given.
    """
    natural = _core.null_for(str(data.scale))
    if robustness is not None:
        value = float(robustness.null_hypothesis)
        if null is not None and not math.isclose(float(null), value, abs_tol=1e-12):
            raise ValueError(
                f"null={null:g} differs from the robustness value's null_hypothesis={value:g}; "
                "the bold contour and the robustness value must read the same null"
            )
    elif null is not None:
        value = float(null)
    elif natural is not None:
        value = natural
    else:
        raise CapabilityError(
            f"{data.estimand!r} is a level (a mean outcome), which has no no-effect value; "
            "pass null= the value the bold contour should mark, or a robustness value, whose "
            "null_hypothesis it then reads",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return value, natural is not None and math.isclose(value, natural, abs_tol=1e-12)


def _check_side(data: Any, null: float) -> None:
    """Refuse a map of the end of the range that moves away from the null.

    The bound at strength zero is the estimate, and the lower bound only falls while the
    upper bound only rises. So only the end on the null's side can reach it: the lower end
    of an estimate above the null and the upper end of one below.
    """
    if _close(data.psi, null):
        return
    needed = "lower" if data.psi > null else "upper"
    if data.bound != needed:
        side = "above" if data.psi > null else "below"
        raise CapabilityError(
            f"this contour maps the {data.bound} bound, which moves away from {null:g} as "
            f"confounding grows, because the estimate {data.psi:.4g} lies {side} it; only the "
            f"{needed} bound can reach {null:g}. Build it with contour_data(..., "
            f"bound={needed!r})",
            reason=RefusalReason.INVALID_DECLARATION,
        )


def _check_robustness(data: Any, robustness: Any) -> float | None:
    """The robustness value to draw, or ``None`` when no equal strength reaches the null.

    Raises
    ------
    ValueError
        When the robustness value belongs to another estimand, alignment, estimate, estimator
        of ``nu^2`` or bias scale than the contour.
    """
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
    if not _close(robustness.psi, data.psi):
        raise ValueError("the robustness value and the contour describe different estimates")
    estimator = robustness.get("nu2_estimator", "doubly_robust")
    if estimator != data.nu2_estimator:
        raise ValueError(
            f"the robustness value uses nu2_estimator={estimator!r} but the contour uses "
            f"{data.nu2_estimator!r}; build both with the same estimator"
        )
    scale = robustness.get("max_bias")
    if scale is not None and _core.finite(data.max_bias) and not _close(scale, data.max_bias):
        raise ValueError(
            f"the robustness value's bias scale ({float(scale):.4g}) differs from the "
            f"contour's ({data.max_bias:.4g}), so they come from different settings, such as "
            "assume_consistent_mechanism or mechanism_learner"
        )
    rv = robustness.get("rv")
    return None if rv is None else float(rv)


def _check_benchmark(data: Any, record: Any) -> None:
    """Refuse a benchmark of another estimand, estimate or second moment than the contour."""
    if record.estimand != data.estimand:
        raise ValueError(
            f"a benchmark is for {record.estimand!r} but the contour bounds {data.estimand!r}"
        )
    if not _close(record.psi_long, data.psi):
        raise ValueError(
            f"the benchmark of {', '.join(record.covariates)} describes another estimate "
            f"({record.psi_long:.4g}) than the contour ({data.psi:.4g})"
        )
    if _core.finite(data.nu2) and not math.isclose(record.nu2_long, data.nu2, rel_tol=1e-7):
        raise CapabilityError(
            f"the benchmark of {', '.join(record.covariates)} reads a Riesz second moment of "
            f"{record.nu2_long:.4g} and the contour {data.nu2:.4g}; build both with the same "
            "nu2_estimator, and note that benchmark() has no consistent-mechanism form",
            reason=RefusalReason.INVALID_DECLARATION,
        )


def _narrow(text: str, ax: Axes, size: float, share: float) -> str:
    """``text`` wrapped to ``share`` of the figure's width.

    A colour bar sits beside the axes, and the layout places it past the widest text the
    axes carries. A subtitle as wide as the figure would leave a gap.
    """
    figure: Any = ax.figure
    while not hasattr(figure, "get_figwidth"):
        figure = figure.figure
    characters = max(24, int(figure.get_figwidth() * 0.92 * 72.0 / (size * 0.52) * share))
    return "\n".join(
        textwrap.fill(part, characters, break_on_hyphens=False) for part in text.split("\n")
    )


def _header(ax: Axes, title: str, subtitle: str, share: float) -> None:
    _core.header(ax, title, _narrow(subtitle, ax, _core.SUBTITLE_SIZE, share))


#: Half the width and height, in fractions of the map, of the zone around a marked point
#: (a benchmark or the robustness value, and its name beside it) that a contour level's label
#: keeps out of.
_MARK_ZONE = (0.22, 0.1)


def _label_points(
    lines: Any, d_max: float, y_max: float, avoid: Sequence[tuple[float, float]] = ()
) -> list[tuple[float, float]]:
    """One vertex per level, nearest the middle of the map and clear of its edges.

    A vertex inside the zone of a marked point in ``avoid`` is passed over, so a level's
    label does not sit on a benchmark or its name. A level whose every vertex is in a zone
    takes the vertex farthest from the marks.
    """
    target = np.array([0.6 * d_max, 0.6 * y_max])
    scale = np.array([d_max, y_max])
    marks = np.asarray(avoid, dtype=float).reshape(-1, 2) / scale
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
        if len(marks):
            gap = np.abs(inside[:, None, :] / scale - marks[None, :, :])
            covered = ((gap[..., 0] < _MARK_ZONE[0]) & (gap[..., 1] < _MARK_ZONE[1])).any(axis=1)
            if covered.all():
                nearest = np.hypot(*(gap / np.maximum(_MARK_ZONE, 1e-9)).transpose(2, 0, 1)).min(
                    axis=1
                )
                best = inside[int(np.argmax(nearest))]
                chosen.append((float(best[0]), float(best[1])))
                continue
            inside = inside[~covered]
        best = inside[int(np.argmin(np.hypot(*(inside - target).T)))]
        chosen.append((float(best[0]), float(best[1])))
    return chosen


def _null_label(
    bold: Any,
    rv: float | None,
    d_max: float,
    y_max: float,
    benchmarks: Sequence[tuple[float, float]] = (),
) -> tuple[float, float]:
    """A vertex of the bold contour clear of the edges, the robustness value and benchmarks.

    The label sits along the line a fifth of the map or more from the robustness value and
    outside the zone of each benchmark and its name. With no such vertex it takes the one
    farthest from the marks.
    """
    segments = [seg for seg in getattr(bold, "allsegs", [[]])[0] if len(seg)]
    points = np.vstack(segments)
    scale = np.array([d_max, y_max])
    scaled = points / scale
    inner = (scaled > 0.08).all(axis=1) & (scaled < 0.92).all(axis=1)
    candidates = points[inner] if inner.any() else points
    if rv is None and not benchmarks:
        return tuple(float(v) for v in candidates[len(candidates) // 2])  # type: ignore[return-value]
    clear = np.ones(len(candidates), dtype=bool)
    # How far each candidate is from the nearest mark, in units of that mark's clearance.
    room = np.full(len(candidates), np.inf)
    if rv is not None:
        away = np.hypot(*((candidates - np.array([rv, rv])) / scale).T)
        clear &= away > 0.2
        room = np.minimum(room, away / 0.2)
    if benchmarks:
        marks = np.asarray(benchmarks, dtype=float).reshape(-1, 2) / scale
        gap = np.abs(candidates[:, None, :] / scale - marks[None, :, :])
        clear &= ~((gap[..., 0] < _MARK_ZONE[0]) & (gap[..., 1] < _MARK_ZONE[1])).any(axis=1)
        room = np.minimum(
            room, np.hypot(*(gap / np.maximum(_MARK_ZONE, 1e-9)).transpose(2, 0, 1)).min(axis=1)
        )
    chosen = candidates[clear]
    pick = chosen[len(chosen) // 2] if len(chosen) else candidates[int(np.argmax(room))]
    return float(pick[0]), float(pick[1])


def _benchmark_marks(
    data: Any, marks: Sequence[Any], multiples: Sequence[float], d_max: float, y_max: float
) -> tuple[list[tuple[str, float, float]], list[str], list[str]]:
    """The benchmark points to draw, those beyond the grid, and those not drawable."""
    drawn: list[tuple[str, float, float]] = []
    off: list[str] = []
    invalid: list[str] = []
    for record in marks:
        label = ", ".join(record.covariates)
        clipped = record.cf_y >= 1.0 or record.cf_d >= 1.0
        for multiple in multiples:
            k = float(multiple)
            at_d, at_y = k * record.cf_d, k * record.cf_y
            mark = f"{label} \u00d7{k:g}" if k != 1.0 else label
            if (k != 1.0 and clipped) or at_y > 1.0 or at_d >= 1.0 or k <= 0.0:
                invalid.append(mark)
            elif at_d > d_max or at_y > y_max:
                off.append(f"{mark} (cf_d {_number(at_d)}, cf_y {_number(at_y)})")
            else:
                drawn.append((mark, at_d, at_y))
    return drawn, off, invalid


def contour(
    data: Any,
    *,
    robustness: Any = None,
    benchmarks: Sequence[Any] | Any = (),
    multiples: Sequence[float] = (1.0,),
    null: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the bias-adjusted bound over the strengths of a hidden confounder.

    Parameters
    ----------
    data : SensitivityContour
        From :func:`~cleverly.sensitivity.contour_data`, of the end of the range that can
        reach the null: the lower end of an estimate above it, the upper end of one below.
    robustness : RobustnessValue or None, default=None
        From :func:`~cleverly.sensitivity.robustness_value` for the same estimand, alignment
        and ``nu2_estimator``. Its point value is marked on the diagonal ``cf_d = cf_y``. The
        confidence value is not drawn: the limit is not on this grid and the plot computes
        none.
    benchmarks : BenchmarkResult or sequence of BenchmarkResult, default=()
        From :func:`~cleverly.sensitivity.benchmark` on the same fit. Each is marked at its
        two gains, read at the map's alignment.
    multiples : sequence of float, default=(1.0,)
        Multiples of both gains to mark. A multiple other than one is this plot's scaling
        heuristic, not a bound. A multiple that takes a gain out of the range of a share, or
        scales a gain clipped at one, is not drawn, and :func:`~cleverly.plotting.notes`
        lists it.
    null : float or None, default=None
        The value the bold contour marks. The robustness value's own null when ``robustness``
        is given, else the no-effect value of the estimate's scale. A level (a mean outcome)
        has none, so it needs ``null`` or ``robustness``.
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
        When the cells do not form a full grid or are all equal, when the contour holds the
        end of the range that moves away from the null, when the robustness value or a
        benchmark does not belong to this contour, when ``null`` and the robustness value
        disagree, or when a level has no null to mark.
    """
    d_values, y_values, grid = _grid(data)
    null_value, no_effect = _contour_null(data, robustness, null)
    _check_side(data, null_value)
    rv = None if robustness is None else _check_robustness(data, robustness)
    marks = list(benchmarks) if isinstance(benchmarks, Sequence) else [benchmarks]
    for record in marks:
        _check_benchmark(data, record)
    vmin, vmax = float(np.min(grid)), float(np.max(grid))
    if not vmax > vmin:
        raise CapabilityError(
            "every cell of the contour has the same value, because the estimate has no bias "
            "scale to move it; there is no map to draw",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    fig, axis = _core.figure_for(ax, figsize=(7.0, 4.7))
    _core.style_axes(axis, grid="none")
    lower = data.bound == "lower"
    cmap, norm, _ = _colour_scale(vmin, vmax, null_value, ratio=False, dark_low=lower)
    levels = _levels(vmin, vmax, null_value)
    xs, ys = np.meshgrid(d_values, y_values)
    filled = axis.contourf(xs, ys, grid, levels=levels, cmap=cmap, norm=norm, extend="neither")
    _tag(filled, GID_CONTOUR_FILL)
    thin = [level for level in levels if vmin < level < vmax and level != null_value]
    lines = axis.contour(xs, ys, grid, levels=thin, colors=_core.FACE, linewidths=0.6, alpha=0.9)
    _tag(lines, GID_CONTOUR_LINES)
    d_max, y_max = d_values[-1], y_values[-1]
    drawn, off, invalid = _benchmark_marks(data, marks, multiples, d_max, y_max)
    avoid = [(at_d, at_y) for _, at_d, at_y in drawn]
    if rv is not None and rv <= min(d_max, y_max):
        avoid.append((rv, rv))
    label_at = _label_points(lines, d_max, y_max, avoid)
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
    line_name = "No effect" if no_effect else f"{data.bound.capitalize()} bound = {null_value:g}"
    if reaches:
        bold = axis.contour(xs, ys, grid, levels=[null_value], colors=_core.INK, linewidths=2.6)
        _tag(bold, GID_NULL_CONTOUR)
        if any(len(seg) for seg in bold.allsegs[0]):
            where = _null_label(bold, rv, d_max, y_max, [(d, y) for _, d, y in drawn])
            for text in axis.clabel(
                bold,
                fmt=lambda _level: line_name,
                fontsize=_core.TICK_SIZE,
                manual=[where],
                inline=True,
                inline_spacing=4,
            ):
                text.set_color(_core.INK)
                text.set_fontweight("bold")
                text.set_gid(GID_NULL_LABEL)

    bar = fig.colorbar(filled, ax=axis)
    bar.set_label(COLORBAR_LABEL.format(bound=data.bound), fontsize=_core.LABEL_SIZE)
    bar.ax.tick_params(labelsize=_core.TICK_SIZE, colors=_core.SECONDARY_INK)
    bar.outline.set_visible(False)
    bar.ax.set_gid(GID_COLORBAR)

    found: list[str] = []
    placed: list[tuple[float, float]] = []

    def fraction(x: float, y: float) -> tuple[float, float]:
        return (x - d_values[0]) / (d_max - d_values[0]), (y - y_values[0]) / (y_max - y_values[0])

    if rv is not None:
        if rv <= min(d_max, y_max):
            diagonal = min(d_max, y_max)
            axis.plot(
                [0.0, diagonal], [0.0, diagonal],
                color=_core.SECONDARY_INK, linewidth=0.8, linestyle=(0, (1, 2)),
                gid=GID_DIAGONAL, zorder=3,
            )  # fmt: skip
            axis.plot(
                [rv], [rv], marker="D", markersize=7, color=_core.INK,
                markeredgecolor=_core.FACE, markeredgewidth=1.0, linestyle="none",
                gid=GID_ROBUSTNESS, zorder=5,
            )  # fmt: skip
            fx, fy = fraction(rv, rv)
            offset, ha, va = _offset(fx, fy, right=False, up=False)
            axis.annotate(
                f"Robustness value {_number(rv)}",
                xy=(rv, rv), xytext=offset, textcoords="offset points",
                ha=ha, va=va, fontsize=_core.TICK_SIZE, color=_core.INK,
                bbox=_LABEL_BOX, gid=GID_ROBUSTNESS_LABEL, zorder=6,
            )  # fmt: skip
            placed.append((fx, fy))
        else:
            found.append(
                f"The robustness value, {_number(rv)}, lies beyond this grid, which reaches "
                f"{_number(d_max)} for cf_d and {_number(y_max)} for cf_y."
            )
    elif robustness is not None:
        found.append(
            f"No equal strength below one brings the {data.bound} bound to {null_value:g}."
        )

    for mark, at_d, at_y in drawn:
        # A benchmark with no gain on one side sits on the map's edge; clip_on=False keeps
        # its triangle whole there.
        axis.plot(
            [at_d], [at_y], marker="^", markersize=7, color=_core.INK,
            markeredgecolor=_core.FACE, markeredgewidth=1.0, linestyle="none",
            gid=GID_BENCHMARK, zorder=5, clip_on=False,
        )  # fmt: skip
        fx, fy = fraction(at_d, at_y)
        crowded = any(abs(fx - px) < 0.12 and abs(fy - py) < 0.08 for px, py in placed)
        offset, ha, va = _offset(fx, fy, right=True, up=not crowded)
        axis.annotate(
            mark, xy=(at_d, at_y), xytext=offset, textcoords="offset points",
            ha=ha, va=va, fontsize=_core.TICK_SIZE, color=_core.INK,
            bbox=_LABEL_BOX, gid=GID_BENCHMARK_LABEL, zorder=6,
        )  # fmt: skip
        placed.append((fx, fy))
    if marks:
        implied = "; ".join(f"{', '.join(r.covariates)} {r.rho:.2g}" for r in marks)
        found.append(
            "Triangles: left-out variables as strong as the named covariates, at the gains in "
            "fit from adjusting for them (Chernozhukov et al., Remark 8), read at this map's "
            f"\u03c1 = {data.rho:g}, not at each benchmark's own implied \u03c1 ({implied})."
        )
    if any(float(k) != 1.0 for k in multiples) and marks:
        found.append(
            "\u00d7k marks k times both gains, a scaling heuristic of this plot, not a bound."
        )
    if off:
        found.append("Benchmarks beyond the grid: " + "; ".join(off) + ".")
    if invalid:
        found.append(
            "Not marked, because the multiple takes a gain out of the range of a share "
            "(cf_y up to 1, cf_d below 1) or scales a gain clipped at 1: "
            + "; ".join(invalid)
            + "."
        )
    if not reaches:
        side = "above" if lower else "below"
        found.append(
            f"No cell on this grid reaches {null_value:g}: the bias-adjusted {data.bound} "
            f"bound stays {side} {null_value:g} up to the strongest confounding shown, as a "
            "point bound without sampling error."
        )

    axis.set_xlim(d_values[0], d_max)
    axis.set_ylim(y_values[0], y_max)
    axis.set_xlabel("Treatment-side strength, cf_d")
    axis.set_ylabel("Outcome-side strength, cf_y")
    words = _lower_first(_core.estimand_words(str(data.estimand)))
    estimate = _core.format_values([data.psi], [abs(vmax - vmin) / 20.0])[0]
    subtitle = (
        f"{data.bound.capitalize()} bound of the {words} "
        f"(\u03c1\u00a0=\u00a0{data.rho:g}); estimate {estimate}"
    )
    if rv is not None and rv <= min(d_max, y_max):
        subtitle += f"; point robustness value {_number(rv)}"
    _header(axis, title or "Sensitivity to hidden confounding", subtitle, 0.68)
    lines_out = [
        f"Each cell is the bias-adjusted {data.bound} bound with no sampling error: no "
        "confidence limit is drawn, and the limit is not on this grid. Shading between "
        "the cells is contouring of the grid.",
        STRENGTHS_NOTE,
    ]
    if data.n_repeats > 1:
        lines_out.append(f"Each cell is the median over the fit's {data.n_repeats} draws.")
    if data.assumption is not None:
        from .._summary.vocabulary import NOTE_TEXTS as TEXTS

        lines_out.append(TEXTS["consistent_mechanism"])
    _core.add_notes(axis, [*lines_out, *found, *_warnings(data)])
    return fig


# ----------------------------------------------------------------------- V14


@dataclass(frozen=True)
class _Dim:
    """One strength axis of a surface: its name, declared values and words."""

    name: str
    values: list[float]
    read: Callable[[Any], float]
    label: str
    short: str
    #: ``"binary"`` (a flip share), ``"categorical"``, ``"dose"``, ``"intermediate"``.
    kind: str
    component: str | None = None


def _key(value: float) -> float:
    return round(float(value), 12)


def _component(strength: Any, index: int) -> float:
    return float(strength[index] if isinstance(strength, tuple) else strength)


def _treatment_dims(result: Any) -> list[_Dim]:
    """One dimension per treatment component, in the words of its perturbation law."""
    cells = result.cells
    components = tuple(result.treatment_components)

    def values(index: int) -> list[float]:
        return sorted({_key(_component(cell.treatment_strength, index)) for cell in cells})

    def reader(index: int) -> Callable[[Any], float]:
        return lambda cell: _key(_component(cell.treatment_strength, index))

    if components:
        kinds = tuple(result.component_kinds) or ("binary",) * len(components)
        dims = []
        for index, (name, kind) in enumerate(zip(components, kinds, strict=True)):
            if kind == "dose":
                label = f"Shift of {name} per SD of the simulated cause"
                short = f"{name} shift per SD"
            else:
                label = f"Share of units whose {name} is flipped ({_TAIL})"
                short = f"Share of {name} flipped"
            dims.append(_Dim(name, values(index), reader(index), label, short, kind, name))
        return dims
    family = result.treatment_family
    if family == "binary":
        label, short, kind = (
            f"Share of units whose treatment is flipped ({_TAIL})",
            "Share of treatment flipped",
            "binary",
        )
    elif family == "categorical":
        pair = result.arm_pair or ("a", "b")
        label = f"Share of units in arms {pair[0]} and {pair[1]} whose arms are swapped ({_TAIL})"
        short, kind = f"Share of arms {pair[0]}, {pair[1]} swapped", "categorical"
    else:
        label, short, kind = "Dose shift per SD of the simulated cause", "Dose shift per SD", "dose"
    return [_Dim("treatment", values(0), reader(0), label, short, kind)]


def _intermediate_dim(result: Any) -> _Dim:
    name = str(result.intermediate_name)
    declared = sorted(
        {_key(c.intermediate_strength) for c in result.cells if c.intermediate_strength is not None}
    )
    return _Dim(
        name,
        declared or [0.0],
        lambda cell: _key(cell.intermediate_strength or 0.0),
        f"Share of units whose {name} is flipped ({_TAIL})",
        f"Share of {name} flipped",
        "intermediate",
    )


def _outcome_label(result: Any) -> str:
    if result.bounded_outcome_law == "logit":
        return "Outcome strength on the log-odds scale of its bounds (not SD units)"
    if result.bounded_outcome_law == "normal_score":
        return "Outcome strength on a normal-score scale (not SD units)"
    if result.outcome_family == "gaussian":
        return "Outcome shift per SD of the simulated cause, sign reversed"
    return f"Share of units whose outcome is flipped ({_TAIL})"


def _outcome_short(result: Any) -> str:
    if result.bounded_outcome_law is not None:
        return "Outcome strength (not SD units)"
    if result.outcome_family == "gaussian":
        return "Outcome shift per SD, sign reversed"
    return "Share of outcome flipped"


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
    result: Any, x_dim: _Dim
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Calibrations on the scale of the x or y axis, at their axis positions, and the rest.

    A binomial row is a share of predictions changed, which is the scale of a flip share. A
    numeric Gaussian row is a signed coefficient per SD of the covariate, the scale of a
    dose shift; the Gaussian outcome law subtracts its strength, so an outcome row sits at
    minus its coefficient. An unsigned block strength has no place on a signed axis.
    """
    from ..sensitivity.simulated_confounding import _NUMERIC_GAUSSIAN_METHOD

    x_marks: list[tuple[str, float]] = []
    y_marks: list[tuple[str, float]] = []
    elsewhere: list[str] = []
    numeric_y = result.outcome_family == "gaussian" and result.bounded_outcome_law is None
    binary_y = result.outcome_family == "binomial" and result.bounded_outcome_law is None
    for row in result.calibrations:
        strength = float(row.strength)
        numeric = row.family == "gaussian" and row.method == _NUMERIC_GAUSSIAN_METHOD
        unit = (
            "share of predictions changed"
            if row.family == "binomial"
            else "signed coefficient"
            if numeric
            else "unsigned block strength"
        )
        where = f"{row.covariate} ({row.role}"
        where += f" {row.component}" if row.component else ""
        where += f", {unit}, {_number(strength)}" if math.isfinite(strength) else ", no value"
        if not math.isfinite(strength):
            elsewhere.append(where + ")")
        elif row.role == "treatment":
            on_x = (row.component or None) == x_dim.component and (
                (x_dim.kind == "binary" and row.family == "binomial")
                or (x_dim.kind == "dose" and numeric)
            )
            if on_x:
                x_marks.append((row.covariate, strength))
            elif x_dim.kind == "intermediate":
                elsewhere.append(where + ": the treatment is held)")
            else:
                elsewhere.append(where + ": no drawn axis on its scale)")
        elif binary_y and row.family == "binomial":
            y_marks.append((row.covariate, strength))
        elif numeric_y and numeric:
            y_marks.append((row.covariate, -strength))
        else:
            elsewhere.append(where + ": no drawn axis on its scale)")
    return x_marks, y_marks, elsewhere


def _position(value: float, declared: list[float]) -> float | None:
    """The index position of ``value`` between declared values, or ``None`` beyond them."""
    if not declared[0] - 1e-12 <= value <= declared[-1] + 1e-12:
        return None
    if len(declared) == 1:
        return 0.0
    return float(np.interp(value, declared, np.arange(len(declared), dtype=float)))


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
        intermediate beside a perturbed treatment): which declared values of the extra axes to
        draw, one panel per combination. The first strength axis and the outcome axis are
        always the plot's own axes. All declared values are drawn when omitted.
    null : float or None, default=None
        The value outlined where the estimate crosses it. The no-effect value of the
        estimate's scale when omitted; a level (a counterfactual mean) has none.
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
    dims = _treatment_dims(result)
    if result.intermediate_name is not None:
        # a controlled direct effect holds the treatment: the intermediate is the moved axis
        dims = [_intermediate_dim(result), *(dim for dim in dims if len(dim.values) > 1)]
    x_dim, extra = dims[0], dims[1:]
    extra_names = [dim.name for dim in extra]
    y_values = sorted({_key(cell.outcome_strength) for cell in cells})
    if at is not None:
        unknown = [name for name in at if name not in extra_names]
        if unknown:
            raise ValueError(
                f"at names {unknown}, which are not extra axes of this grid; the first strength "
                f"axis ({x_dim.name!r}) and the outcome are the plot's own axes, and the "
                f"extra axes are {extra_names}"
            )
    chosen = [_selected(at, dim.name, dim.values) for dim in extra]
    combos = list(itertools.product(*chosen))
    if len(combos) > MAX_PANELS:
        raise ValueError(
            f"this grid has {len(combos)} combinations of {extra_names}, more than the "
            f"{MAX_PANELS} panels one figure draws; pass at= to fix some of them, for "
            f"example at={{{extra_names[0]!r}: 0.0}}"
        )
    if ax is not None and len(combos) > 1:
        raise ValueError("ax= takes a one-panel surface; this grid needs several panels")
    x_values = x_dim.values

    lookup = {
        (
            x_dim.read(cell),
            _key(cell.outcome_strength),
            tuple(dim.read(cell) for dim in extra),
        ): cell
        for cell in cells
    }
    drawn = [lookup[(x, y, combo)] for combo in combos for x in x_values for y in y_values
             if (x, y, combo) in lookup]  # fmt: skip
    estimates = [float(c.estimate) for c in drawn if c.failure is None and c.estimate is not None]
    if not estimates:
        raise ValueError("no cell of this surface produced an estimate, so there is nothing to map")
    null_value = _core.null_for(str(result.scale)) if null is None else float(null)
    ratio = str(result.scale) == "ratio" and min(estimates) > 0.0
    vmin, vmax = min(estimates), max(estimates)
    if vmax == vmin:
        vmax = vmin + max(abs(vmin) * 1e-6, 1e-9)
    cmap, norm, _ = _colour_scale(vmin, vmax, null_value, ratio=ratio, dark_low=False)

    panels = len(combos)
    ncols = min(3, panels)
    nrows = math.ceil(panels / ncols)
    if panels == 1:
        fig, axis = _core.figure_for(ax, figsize=(7.0, 4.4))
        axes_list = [axis]
        top = foot = axis
    else:
        fig = _core.pyplot().figure(
            figsize=(3.0 * ncols + 1.4, 2.7 * nrows + 1.8), layout="constrained"
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

    from matplotlib.patches import Rectangle

    mesh = None
    failed_total = 0
    outlined = 0
    small = len(x_values) <= _ANNOTATE_UP_TO and len(y_values) <= _ANNOTATE_UP_TO
    annotate = small and (ncols < 3 or len(x_values) * len(y_values) <= 12)
    digits = _core.format_values(estimates, [(vmax - vmin) / 12.0] * len(estimates))
    shown = dict(zip(map(float, estimates), digits, strict=False))
    # equal cells, one per declared strength: declared values are a design, not a continuum
    x_edges = np.arange(len(x_values) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(y_values) + 1, dtype=float) - 0.5
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
            shading="flat", edgecolor=_core.FACE, linewidth=0.8, gid=GID_HEATMAP,
        )  # fmt: skip
        for ix, iy in failed:
            axis_i.add_patch(
                Rectangle(
                    (x_edges[ix], y_edges[iy]), 1.0, 1.0, facecolor=_core.FACE,
                    edgecolor=_core.REFERENCE, hatch="////", linewidth=0.0,
                    gid=GID_FAILED_CELL,
                )
            )  # fmt: skip
        if null_value is not None:
            outlined += _outline_null(axis_i, values, x_edges, y_edges, null_value)
        if annotate:
            for iy in range(len(y_values)):
                for ix in range(len(x_values)):
                    if np.isfinite(values[iy, ix]):
                        axis_i.text(
                            float(ix), float(iy),
                            shown.get(float(values[iy, ix]), _number(values[iy, ix])),
                            ha="center", va="center", fontsize=_core.TICK_SIZE - 0.5,
                            color=_text_ink(cmap, norm, values[iy, ix]),
                            gid=GID_CELL_VALUE,
                        )  # fmt: skip
        axis_i.set_xlim(x_edges[0], x_edges[-1])
        axis_i.set_ylim(y_edges[0], y_edges[-1])
        axis_i.set_xticks(range(len(x_values)))
        axis_i.set_xticklabels([f"{value:g}" for value in x_values])
        axis_i.set_yticks(range(len(y_values)))
        axis_i.set_yticklabels([f"{value:g}" for value in y_values])
        axis_i.tick_params(length=0)
        for spine in axis_i.spines.values():
            spine.set_visible(False)
        if panels > 1:
            pairs = zip(extra, combo, strict=True)
            axis_i.set_title(
                ", ".join(f"{dim.short} = {value:g}" for dim, value in pairs),
                fontsize=_core.TICK_SIZE + 0.5,
                color=_core.INK,
                pad=4.0,
            )

    x_marks, y_marks, elsewhere = _calibration_marks(result, x_dim)
    placed: list[tuple[str, float]] = []
    outside: list[tuple[str, float]] = []
    for index, axis_i in enumerate(axes_list):
        bottom_row = index + ncols >= panels
        left_column = index % ncols == 0
        got = _mark_calibrations(
            axis_i, sorted(x_marks, key=lambda m: m[1]), sorted(y_marks, key=lambda m: m[1]),
            x_values, y_values, name_x=bottom_row, name_y=left_column,
            reversed_y=result.outcome_family == "gaussian",
        )  # fmt: skip
        if index == 0:
            placed, outside = got
    for index, axis_i in enumerate(axes_list):
        x_words = x_dim.label if panels == 1 else x_dim.short
        y_words = _outcome_label(result) if panels == 1 else _outcome_short(result)
        axis_i.set_xlabel(
            textwrap.fill(x_words, 52 if panels == 1 else 26) if index + ncols >= panels else ""
        )
        axis_i.set_ylabel(textwrap.fill(y_words, 34) if index % ncols == 0 else "")

    assert mesh is not None
    bar = fig.colorbar(mesh, ax=axes_list)
    words = _core.estimand_words(str(result.estimand))
    bar.set_label(f"Refitted {_lower_first(words)}, qualitative", fontsize=_core.LABEL_SIZE)
    bar.ax.tick_params(labelsize=_core.TICK_SIZE, colors=_core.SECONDARY_INK)
    bar.outline.set_visible(False)
    bar.ax.set_gid(GID_COLORBAR)
    _bar_ticks(bar, vmin, vmax, null_value, ratio)

    from ..sensitivity._summaries import _surface_laws

    moved, laws = _surface_laws(result)
    original = _core.format_values([float(result.original_estimate)], [(vmax - vmin) / 12.0])[0]
    # the axes name what the simulated cause moves, so the subtitle gives the numbers
    subtitle = f"{words}; unperturbed estimate {original}"
    if result.intermediate_name is not None and result.intermediate_value is not None:
        subtitle += f"; the target sets {result.intermediate_name} to {result.intermediate_value:g}"
    _header(
        top,
        title or "Estimate under a simulated common cause",
        subtitle,
        0.7 if panels == 1 else 0.85,
    )

    found = [
        QUALITATIVE_NOTE + " No strength is a statement about the confounding in these data.",
        f"The estimate is refitted under a simulated common cause of {moved} and the outcome.",
        laws,
    ]
    spacing = [np.diff(values) for values in (x_values, y_values) if len(values) > 2]
    if any(not np.allclose(gaps, gaps[0]) for gaps in spacing):
        found.append(
            "Each row and column is one declared strength, drawn as an equal cell, so the "
            "axes are not to scale."
        )
    if result.n_repeats > 1:
        found.append(
            f"Each cell's estimate is the median over the fit's {result.n_repeats} "
            "cross-fitting draws."
        )
    if null_value is not None:
        found.append(
            f"Heavy outline: where the estimate crosses {null_value:g}."
            if outlined
            else f"No drawn cell crosses {null_value:g}, so there is no outline."
        )
    if failed_total:
        found.append("Hatched: a cell that failed to refit.")
    if panels > 1:
        found.append(f"One panel per declared value of {' and '.join(dim.short for dim in extra)}.")
    if placed:
        found.append(
            "Triangles mark observed covariates on the axis scale: "
            + "; ".join(f"{name} at {_number(value)}" for name, value in placed)
            + "."
        )
    if outside:
        found.append(
            "Beyond the axes: "
            + "; ".join(f"{name} at {_number(value)}" for name, value in outside)
            + "."
        )
    if elsewhere:
        found.append("Observed-covariate calibrations not marked: " + "; ".join(elsewhere) + ".")
    _core.add_notes(foot, [*found, *_warnings(result)])
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
            gid=GID_NULL_CROSSING,
        )  # fmt: skip
        axis.add_collection(collection)
    return len(segments)


def _label_sides(wheres: list[float | None]) -> list[int]:
    """Push the labels of marks closer than one cell apart to opposite sides so they do not collide.

    Returns ``-1`` (before the mark), ``1`` (after it) or ``0`` (centred) for each mark.
    """
    sides = [0] * len(wheres)
    drawn = sorted((w, i) for i, w in enumerate(wheres) if w is not None)
    for (left, i), (right, j) in itertools.pairwise(drawn):
        if right - left < 1.0:
            sides[i], sides[j] = -1, 1
    return sides


def _mark_calibrations(
    axis: Axes,
    x_marks: list[tuple[str, float]],
    y_marks: list[tuple[str, float]],
    x_values: list[float],
    y_values: list[float],
    *,
    name_x: bool,
    name_y: bool,
    reversed_y: bool,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Named triangles on the axes at observed-covariate strengths that lie within them."""
    placed: list[tuple[str, float]] = []
    outside: list[tuple[str, float]] = []
    reversed_note = " (sign reversed)" if reversed_y else ""
    x_sides = _label_sides([_position(value, x_values) for _, value in x_marks])
    for (name, value), side in zip(x_marks, x_sides, strict=True):
        where = _position(value, x_values)
        if where is None:
            outside.append((f"{name}, treatment", value))
            continue
        axis.plot(
            [where], [0.0], marker="^", markersize=7, color=_core.INK,
            markeredgecolor=_core.FACE, transform=axis.get_xaxis_transform(),
            clip_on=False, linestyle="none", gid=GID_CALIBRATION, zorder=6,
        )  # fmt: skip
        if name_x:
            axis.annotate(
                name, xy=(where, 0.0), xycoords=axis.get_xaxis_transform(),
                xytext=(side * 2.0, 6.0), textcoords="offset points",
                ha={-1: "right", 0: "center", 1: "left"}[side], va="bottom",
                fontsize=_core.TICK_SIZE - 1, color=_core.INK, bbox=_LABEL_BOX,
                gid=GID_CALIBRATION_LABEL, zorder=7,
            )  # fmt: skip
        placed.append((f"{name}, treatment", value))
    y_sides = _label_sides([_position(value, y_values) for _, value in y_marks])
    for (name, value), side in zip(y_marks, y_sides, strict=True):
        where = _position(value, y_values)
        if where is None:
            outside.append((f"{name}, outcome{reversed_note}", value))
            continue
        axis.plot(
            [0.0], [where], marker=">", markersize=7, color=_core.INK,
            markeredgecolor=_core.FACE, transform=axis.get_yaxis_transform(),
            clip_on=False, linestyle="none", gid=GID_CALIBRATION, zorder=6,
        )  # fmt: skip
        if name_y:
            axis.annotate(
                name, xy=(0.0, where),
                xycoords=axis.get_yaxis_transform(), xytext=(7.0, side * 2.0),
                textcoords="offset points", ha="left",
                va={-1: "top", 0: "center", 1: "bottom"}[side],
                fontsize=_core.TICK_SIZE - 1, color=_core.INK, bbox=_LABEL_BOX,
                gid=GID_CALIBRATION_LABEL, zorder=7,
            )  # fmt: skip
        placed.append((f"{name}, outcome{reversed_note}", value))
    return placed, outside


# ----------------------------------------------------------------------- dispatch


@_core.labelled
def confounding(obj: Any, **kwargs: Any) -> Figure:
    """Plot how strong hidden confounding would need to be, or how far a simulated one moves us.

    Use it to read a sensitivity analysis to confounding that was not measured. For a
    :class:`~cleverly.sensitivity.SensitivityContour` it maps the bias-adjusted bound of the
    estimate over the two strengths of left-out variables, with the contour where the bound
    reaches the null in bold, the point robustness value on the diagonal and benchmark
    covariates marked at their gains. For a
    :class:`~cleverly.sensitivity.SimulatedConfoundingResult` it maps the refitted estimate
    over the declared strengths of a simulated common cause, in the units of its own
    perturbation laws, and says it is qualitative.

    Parameters
    ----------
    obj : SensitivityContour or SimulatedConfoundingResult
        The analysis to draw.
    **kwargs
        For a contour: ``robustness=``, ``benchmarks=``, ``multiples=``, ``null=`` and
        ``ax=``. For a surface: ``at=``, ``null=`` and ``ax=``. Both take the formatting
        keywords ``title=``, ``subtitle=``, ``xlabel=``, ``ylabel=``, ``legend=`` and
        ``figsize=``, as :func:`~cleverly.plotting.forest` describes them. ``xlabel=`` and
        ``ylabel=`` replace the words on the two strength axes, such as ``"Weight side"``
        and ``"Score side"`` for a page that names the two strengths that way.

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

    Notes
    -----
    The figure carries no note text. It does not mark an experimental fit or band, one whose
    interval has no checked coverage because no registered study covers it or a stated
    condition applies, such as too few clusters: read :func:`~cleverly.plotting.notes` on
    the figure before you publish it. For this plot the notes define the two strengths, list
    benchmarks or calibrations that are not marked and why, and say that a
    simulated-confounding map is qualitative: not a bound or an interval.

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
    'Sensitivity to hidden confounding'
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
