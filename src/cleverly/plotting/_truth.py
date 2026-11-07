"""The ``truth=`` argument: known true values of simulated data, drawn beside the estimates.

A true value exists only when the data were simulated. It is drawn as an ink cross (a
dash-dot line for a curve over a continuous modifier), never in a series colour and never in
the dashed style of the no-effect line, and the key names it. Every value must match a drawn
estimate or a drawn x level; anything else is refused, so a mistyped name never draws
nothing in silence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def refuse(message: str, alternative: str) -> Exception:
    """The refusal of a ``truth=`` that does not match the figure."""
    from .._refusal import RefusalReason
    from ..exceptions import CapabilityError

    return CapabilityError(
        message, reason=RefusalReason.INVALID_DECLARATION, alternative=alternative
    )


def number(value: Any, what: str) -> float:
    """``value`` as a finite float, or a refusal that names ``what``.

    A boolean is refused: ``True`` would draw as 1 without saying so.
    """
    if isinstance(value, (bool, np.bool_)):
        raise refuse(
            f"truth= gives {value!r} for {what}, which is a boolean and not a number",
            "pass a number for each true value",
        )
    try:
        found = float(value)
    except (TypeError, ValueError):
        found = math.nan
    if not math.isfinite(found):
        raise refuse(
            f"truth= gives {value!r} for {what}, which is not a finite number",
            "pass a finite number for each true value",
        )
    return found


@dataclass(frozen=True)
class Spot:
    """One drawn estimate a true value can sit beside.

    Parameters
    ----------
    ax : Axes
        The axes it is drawn on.
    name : str or None
        The estimate's name in the fit, or ``None`` for a value that is not a named estimate.
    x : float
        Its x position.
    transform : Any
        The transform it is drawn with, so the mark takes the same sideways shift.
    series : str
        The series it belongs to.
    """

    ax: Axes
    name: str | None
    x: float
    transform: Any
    series: str


def _named(truth: Any) -> bool:
    return isinstance(truth, Mapping) and bool(truth) and all(isinstance(k, str) for k in truth)


def place(
    truth: Any, spots: Sequence[Spot], *, known: Sequence[str] = ()
) -> list[tuple[Spot, float]]:
    """Each true value with the drawn estimate it belongs to.

    ``truth`` is a mapping of estimate name to value, or the values over the x levels of a
    figure of one series: a sequence in the order of the levels, or a mapping of level to
    value. A name the fit reports but the figure does not draw (``known``) is passed over.
    """
    if isinstance(truth, Mapping) and not truth:
        raise refuse(
            "truth= is an empty mapping, so it gives no true value to draw",
            "pass the true values, or leave truth= out",
        )
    if _named(truth):
        drawn = {spot.name: spot for spot in spots if spot.name is not None}
        if not drawn:
            raise refuse(
                "truth= names estimates, and the points of this figure are not named estimates",
                "pass the true values over the x levels, in their order",
            )
        unknown = [name for name in truth if name not in drawn and name not in known]
        if unknown:
            raise refuse(
                f"truth= names {unknown}, which this fit does not report; it draws {list(drawn)}",
                "key truth= by the names the fit reports",
            )
        found = [
            (drawn[name], number(value, name)) for name, value in truth.items() if name in drawn
        ]
        if not found:
            raise refuse(
                "truth= names no estimate this figure draws",
                f"key truth= by the drawn estimates {list(drawn)}",
            )
        return found
    if isinstance(truth, (str, bytes)) or not isinstance(truth, (Mapping, Sequence, np.ndarray)):
        raise refuse(
            f"truth= must be a mapping or a sequence of numbers; got {type(truth).__name__}",
            "pass a mapping of estimate name to true value, or one value per x level",
        )
    series = list(dict.fromkeys(spot.series for spot in spots))
    if len(series) != 1:
        raise refuse(
            f"truth= over the x levels fits a figure of one series, and this one draws "
            f"{len(series)}: {series}",
            "pass a mapping of estimate name to true value, or draw one series",
        )
    ordered = sorted(spots, key=lambda spot: spot.x)
    levels = [spot.x for spot in ordered]
    if isinstance(truth, Mapping):
        found = []
        for key, value in truth.items():
            at = number(key, "an x level")
            match = [spot for spot in ordered if math.isclose(spot.x, at, abs_tol=1e-12)]
            if not match:
                raise refuse(
                    f"truth= gives a value at {key!r}, which is not an x level of this figure; "
                    f"it draws {[f'{x:g}' for x in levels]}",
                    "key truth= by the drawn x levels",
                )
            found.append((match[0], number(value, f"x = {key!r}")))
        return found
    values = list(np.ravel(np.asarray(truth, dtype=object)))
    if np.ndim(np.asarray(truth, dtype=object)) != 1 or len(values) != len(ordered):
        raise refuse(
            f"truth= has {len(values)} values, and this figure draws {len(ordered)} x levels "
            f"({', '.join(f'{x:g}' for x in levels)})",
            "pass one true value per x level, in increasing order of the level",
        )
    return [
        (spot, number(value, f"x = {spot.x:g}"))
        for spot, value in zip(ordered, values, strict=True)
    ]


def include(ax: Axes, values: Sequence[float], *, axis: str = "y") -> None:
    """Widen the axis limits so the true values are inside them.

    Marks drawn through a shifted transform do not feed the autoscale, so the limits are
    widened from the numbers drawn.
    """
    if not values:
        return
    get = ax.get_ylim if axis == "y" else ax.get_xlim
    put = ax.set_ylim if axis == "y" else ax.set_xlim
    scale = ax.get_yscale() if axis == "y" else ax.get_xscale()
    low, high = get()
    least, most = min(values), max(values)
    if least >= low and most <= high:
        return
    if scale == "log":
        put(min(low, least / 1.1), max(high, most * 1.1))
        return
    margin = 0.05 * ((max(high, most) - min(low, least)) or 1.0)
    put(min(low, least - margin), max(high, most + margin))


def check_log(ax: Axes, values: Sequence[float], *, axis: str = "y") -> None:
    """Refuse a true value that a log axis cannot show."""
    scale = ax.get_yscale() if axis == "y" else ax.get_xscale()
    if scale == "log" and any(value <= 0 for value in values):
        raise refuse(
            "truth= gives a value at or below zero for a ratio, which the log axis cannot show",
            "give each true ratio as a positive number",
        )


def draw(truth: Any, spots: Sequence[Spot], *, known: Sequence[str] = ()) -> bool:
    """Draw ``truth`` beside the matching spots; return whether anything was drawn."""
    if truth is None:
        return False
    found = place(truth, spots, known=known)
    by_axes: dict[Any, list[tuple[Spot, float]]] = {}
    for spot, value in found:
        by_axes.setdefault(spot.ax, []).append((spot, value))
    for ax, pairs in by_axes.items():
        check_log(ax, [value for _, value in pairs])
        for spot, value in pairs:
            _core.truth_mark(ax, spot.x, value, transform=spot.transform)
        include(ax, [value for _, value in pairs])
    return bool(found)


__all__ = ["Spot", "check_log", "draw", "include", "number", "place", "refuse"]
