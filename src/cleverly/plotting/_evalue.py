"""The E-value figure: the confounder strengths that could explain an estimate away.

An :class:`~cleverly.sensitivity.EValue` holds a risk ratio, the confidence limit nearest the
null and the E-value of each. Ding and VanderWeele (2016, Result 1) bound the ratio of the
observed to the true risk ratio by the bounding factor

.. math:: B(x, y) = \\frac{x y}{x + y - 1}

of a confounder's association ``x`` with the treatment and ``y`` with the outcome, both on the
risk-ratio scale. Confounding can explain a ratio ``t`` above 1 away only where
``B(x, y) >= t``. The figure draws the level set ``B(x, y) = t``, the curve
``y = t (x - 1) / (x - t)``, for the estimate and for the limit. It meets the diagonal at the
E-value ``t + sqrt(t (t - 1))``. A ratio below 1 is folded to its reciprocal, as the E-value is.

On a controlled direct effect the bias is a product of one bounding factor per arm
(``bias_factor="per_arm_product"``). With both arms at the same pair the product is
``B(x, y) ** 2``, so the curve is the level set at the square root of the ratio: the
two-dimensional slice whose diagonal is the reported per-arm E-value.

The curves are a deterministic function of the stored numbers: no interval or sampling error is
drawn beyond the stored confidence limit, whose reading the E-value already records.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .._summary._render import number_text
from . import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The ``gid`` each kind of artist of this plot carries.
GID_EVALUE_ESTIMATE = "cleverly-evalue-estimate"
GID_EVALUE_LIMIT = "cleverly-evalue-limit"
GID_EVALUE_MARK = "cleverly-evalue-mark"
GID_EVALUE_LABEL = "cleverly-evalue-label"
GID_DIAGONAL = "cleverly-diagonal"

TITLE = "Confounding that could explain the estimate away"

#: Vertices per curve.
_POINTS = 400
#: Relative tolerance of the check that each stored E-value is the E-value of its ratio.
_RTOL = 1e-9

CURVE_NOTE = (
    "Each curve holds the pairs of confounder associations, on the risk-ratio scale, whose "
    "bounding factor x·y / (x + y - 1) equals the risk ratio it is named for (Ding and "
    "VanderWeele, 2016). A confounder inside a curve cannot explain that number away; one on "
    "or beyond it may. The dot, where the two associations are equal, is the E-value."
)
PER_ARM_CURVE_NOTE = (
    "Each curve holds the pairs of associations that, taken by both arms alike, make the "
    "product of the two arms' bounding factors equal the risk ratio it is named for: the factor "
    "of one arm equals its square root. Each arm's treatment association is the cell ratio "
    "sup_u P(u | A = a, Z = z, W) / P(u | W) (its reciprocal for the comparison arm), which "
    "mixes the confounder's association with the treatment and with the intermediate. A "
    "confounder inside a curve cannot explain that number away; one on or beyond it may. Arms "
    "with unequal associations can trade off, which the curve does not show. The dot, where "
    "all four associations are equal, is the E-value."
)
BELOW_NULL_NOTE = (
    "The risk ratio is below 1, so the curves use its reciprocal, as the E-value does."
)
LIMIT_AT_NULL_NOTE = (
    "The {level} interval already reaches the null, so no confounding is needed to move its "
    "limit there and no limit curve is drawn."
)


def bounding_factor(x: Any, y: Any) -> Any:
    """Ding and VanderWeele's joint bounding factor of two risk-ratio associations.

    Parameters
    ----------
    x : float or ndarray
        The confounder's association with the treatment.
    y : float or ndarray
        Its association with the outcome.

    Returns
    -------
    float or ndarray
        ``x * y / (x + y - 1)``.
    """
    return x * y / (x + y - 1.0)


def partner(threshold: float, x: Any) -> Any:
    """The outcome association that, with treatment association ``x``, reaches ``threshold``.

    Parameters
    ----------
    threshold : float
        A risk ratio above 1.
    x : float or ndarray
        Treatment associations above ``threshold``.

    Returns
    -------
    float or ndarray
        ``threshold * (x - 1) / (x - threshold)``, the ``y`` with
        ``bounding_factor(x, y) == threshold``.
    """
    return threshold * (x - 1.0) / (x - threshold)


def _nearest_limit(data: Any) -> float:
    """The confidence limit nearest the null, by the side rule of the E-value."""
    low, high = data.risk_ratio_ci
    return float(low if data.risk_ratio >= 1.0 else high)


def _thresholds(data: Any) -> tuple[float, float, bool]:
    """The ratios the two curves reach, folded above 1, with the per-arm square root.

    Returns the estimate's threshold, the limit's threshold and whether the ratio lies below
    the null. The limit's threshold can be at or below 1, or ``nan``; the caller decides.
    """
    ratio = float(data.risk_ratio)
    limit = _nearest_limit(data)
    below = ratio < 1.0
    if below:
        ratio, limit = 1.0 / ratio, (1.0 / limit if limit > 0 else math.inf)
    if data.bias_factor == "per_arm_product":
        ratio, limit = math.sqrt(ratio), (math.sqrt(limit) if limit >= 0 else math.nan)
    return ratio, limit, below


def _evalue_of(threshold: float) -> float:
    """The diagonal point of the level set at ``threshold``; ``nan`` below 1.

    Unlike :func:`~cleverly.sensitivity.evalue_from_rr`, this does not fold a ratio below 1,
    so a threshold that missed its fold fails the check instead of matching by symmetry.
    """
    if not (math.isfinite(threshold) and threshold >= 1.0):
        return math.nan
    return float(threshold + math.sqrt(threshold * (threshold - 1.0)))


def _check_consistent(data: Any, threshold: float, limit_threshold: float) -> None:
    """Refuse an object whose E-values are not those of its risk ratio and limit."""
    per_arm = " after the per-arm square root" if data.bias_factor == "per_arm_product" else ""
    expected_limit = 1.0 if limit_threshold <= 1.0 else _evalue_of(limit_threshold)
    for which, stored, ratio, expected in (
        ("point", float(data.point), threshold, _evalue_of(threshold)),
        ("limit", float(data.limit), limit_threshold, expected_limit),
    ):
        if not math.isclose(stored, expected, rel_tol=_RTOL) or not math.isfinite(stored):
            raise ValueError(
                f"this E-value's {which} value, {stored:.4g}, is not the E-value of its risk "
                f"ratio {ratio:.4g}{per_arm}; it was changed after evalue() built it, so the "
                "curve would not pass through it"
            )


def _default_strength(point: float) -> float:
    return float(max(3, math.ceil(2.0 * point)))


def _notes(data: Any, below: bool, limit_drawn: bool) -> list[str]:
    lines = [PER_ARM_CURVE_NOTE if data.bias_factor == "per_arm_product" else CURVE_NOTE]
    if below:
        lines.append(BELOW_NULL_NOTE)
    lines.extend(note.text for note in data.summary().notes)
    if data.limit_descriptive and data.limit_reading is not None:
        lines.append(str(data.limit_reading.sentence))
    if not limit_drawn:
        lines.append(LIMIT_AT_NULL_NOTE.format(level=_core.level_text(float(data.level))))
    return lines


def _curve(axis: Axes, threshold: float, top: float, color: str, label: str, gid: str) -> None:
    start = threshold * (top - 1.0) / (top - threshold)
    xs = np.linspace(start, top, _POINTS)
    axis.plot(xs, partner(threshold, xs), color=color, linewidth=2.0, label=label, gid=gid)


def _mark(axis: Axes, value: float, color: str, *, outside: bool) -> None:
    """The E-value's dot and its name: beyond the curve for the estimate, inside it for the limit.

    The estimate's curve lies beyond the limit's, so the two names sit on opposite sides of
    their dots and stay apart when the two E-values are close.
    """
    offset, ha, va = ((6.0, 4.0), "left", "bottom") if outside else ((-6.0, -4.0), "right", "top")
    axis.plot(
        [value], [value], marker="o", markersize=7, color=color,
        markeredgecolor=_core.FACE, markeredgewidth=1.0, linestyle="none",
        gid=GID_EVALUE_MARK, zorder=5,
    )  # fmt: skip
    axis.annotate(
        f"E-value {number_text(value, 2)}", xy=(value, value), xytext=offset,
        textcoords="offset points", ha=ha, va=va, fontsize=_core.TICK_SIZE,
        color=_core.INK, gid=GID_EVALUE_LABEL, zorder=6,
    )  # fmt: skip


def evalue_curves(
    data: Any, *, max_strength: float | None = None, ax: Axes | None = None
) -> Figure:
    """Plot the confounder strengths that could explain an E-value's estimate away.

    Parameters
    ----------
    data : EValue
        From :func:`~cleverly.sensitivity.evalue`.
    max_strength : float or None, default=None
        The largest association both axes show, on the risk-ratio scale; ``max(3, ceil(2 *
        point))`` when omitted.
    ax : Axes or None, default=None
        Axes to draw on; a new figure when omitted.

    Returns
    -------
    Figure
        The figure drawn on.

    Raises
    ------
    CapabilityError
        When the estimate's E-value is not above 1, so there is no curve to draw.
    ValueError
        When the stored E-values are not those of the stored risk ratio and limit, or
        ``max_strength`` is not a finite number above the estimate's E-value.
    """
    from .._refusal import RefusalReason
    from ..exceptions import CapabilityError

    point = float(data.point)
    if not (math.isfinite(point) and point > 1.0):
        raise CapabilityError(
            f"the E-value of this estimate is {point:g}, not above 1: its risk ratio sits at "
            "the null, so no confounding is needed to explain it away, or it is not a number; "
            "there is no curve to draw",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    threshold, limit_threshold, below = _thresholds(data)
    _check_consistent(data, threshold, limit_threshold)
    top = _default_strength(point) if max_strength is None else float(max_strength)
    if not (math.isfinite(top) and top > point):
        raise ValueError(
            f"max_strength={top:g} must be a finite number above the E-value of the estimate, "
            f"{point:.3g}, so that its equal-strength point falls on the figure"
        )
    limit_drawn = float(data.limit) > 1.0
    level = _core.level_text(float(data.level))

    fig, axis = _core.figure_for(ax, figsize=(6.4, 5.6))
    _core.style_axes(axis, grid="none")
    _curve(
        axis, threshold, top, _core.PALETTE[0], "Could explain the estimate away",
        GID_EVALUE_ESTIMATE,
    )  # fmt: skip
    if limit_drawn:
        _curve(
            axis, limit_threshold, top, _core.PALETTE[1],
            f"Could move the {level} limit to the null", GID_EVALUE_LIMIT,
        )  # fmt: skip
    axis.plot(
        [1.0, top], [1.0, top], color=_core.REFERENCE, linewidth=1.0, linestyle=(0, (1, 2)),
        label="Equal strengths", gid=GID_DIAGONAL, zorder=1,
    )  # fmt: skip
    _mark(axis, point, _core.PALETTE[0], outside=True)
    if limit_drawn:
        _mark(axis, float(data.limit), _core.PALETTE[1], outside=False)
    axis.set_xlim(1.0, top)
    axis.set_ylim(1.0, top)
    axis.set_aspect("equal", adjustable="box")
    if data.bias_factor == "per_arm_product":
        axis.set_xlabel("Each arm's treatment association (risk ratio)")
        axis.set_ylabel("Each arm's outcome association (risk ratio)")
    else:
        axis.set_xlabel("Confounder's association with treatment (risk ratio)")
        axis.set_ylabel("Confounder's association with outcome (risk ratio)")
    scale = str(data.scale) + (", converted" if data.approximate else "")
    subtitle = (
        f"E-values {number_text(point, 2)} for the estimate and "
        f"{number_text(float(data.limit), 2)} for the {level} limit; {data.estimand}, {scale}"
    )
    _core.header(axis, TITLE, subtitle)
    _core.add_notes(axis, _notes(data, below, limit_drawn))
    return fig


__all__ = ["evalue_curves"]
