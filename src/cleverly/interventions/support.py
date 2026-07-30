r"""Does the data support the regime being asked about?

Positivity for a static arm is a statement about one propensity column: is
:math:`g_a(W)` bounded away from zero.  For a regime it is a statement about the
*ratio*

.. math::

    \frac{g^\star(a \mid W)}{g(a \mid W)} ,

and the two can differ sharply.  A rule that sends every unit over 65 to treatment is
perfectly well supported when the elderly are often treated and catastrophically
unsupported when they are not -- even though the marginal propensity, and every
arm-level overlap diagnostic, may look identical in the two cases.  The quantity that
matters is the propensity *at the arm the regime actually assigns*, and it is not
reported by anything that averages over arms.

This runs **before** estimation.  A regime with no support does not produce a wide
confidence interval; it produces a confident one around a number extrapolated from the
handful of rows that happened to receive the assigned arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._typing import FloatArray
from .base import RegimeSet

__all__ = ["RegimeSupport", "SupportReport", "check_support"]

#: Propensity thresholds the report counts rows below.  The same ladder
#: :mod:`cleverly.sensitivity.positivity` uses, so the two tables read together.
_THRESHOLDS = (0.01, 0.025, 0.05)

_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


@dataclass(frozen=True)
class RegimeSupport:
    """Overlap for one regime.

    Attributes
    ----------
    min_support_propensity:
        The smallest :math:`g_a(W_i)` over the rows and arms the regime puts mass on.
        For a deterministic rule that is :math:`\\min_i g_{d(W_i)}(W_i)`, the number a
        reader of a dynamic regime wants first.
    ratio_quantiles:
        Quantiles of the density ratio :math:`g^\\star(A \\mid W)/g(A \\mid W)` at the
        *observed* treatment -- the clever covariate's magnitude, before the missingness
        and intermediate mechanisms enter it.
    max_ratio:
        The largest such ratio: how much one row can move the estimate.
    effective_sample_size:
        Kish ESS of those ratios, and ``ess_ratio`` its share of ``n``.  A regime far
        from the observed mechanism throws away information even when nothing is
        formally violated, and this is the size of that loss.
    tail_mass:
        Fraction of rows whose assigned-arm propensity falls below each threshold.
    unsupported:
        Rows where the regime assigns positive probability to an arm with an estimated
        propensity of exactly zero -- a structural violation rather than a practical one.
        The parameter is not identified for those rows at all.
    """

    name: str
    min_support_propensity: float
    ratio_quantiles: dict[float, float]
    max_ratio: float
    effective_sample_size: float
    ess_ratio: float
    tail_mass: dict[float, float]
    unsupported: int


@dataclass(frozen=True)
class SupportReport:
    """:class:`RegimeSupport` for every regime in a fit, plus the worst case."""

    regimes: dict[str, RegimeSupport] = field(default_factory=dict)
    n: int = 0

    @property
    def worst(self) -> RegimeSupport | None:
        """The regime with the smallest assigned-arm propensity."""
        if not self.regimes:
            return None
        return min(self.regimes.values(), key=lambda item: item.min_support_propensity)

    def to_frame(self, data: Any = None) -> Any:
        """One row per regime, in the backend the data came from."""
        from ..utils.frames import frame_from_dict

        payload = {
            "regime": list(self.regimes),
            "min_propensity": [item.min_support_propensity for item in self.regimes.values()],
            "max_ratio": [item.max_ratio for item in self.regimes.values()],
            "effective_n": [item.effective_sample_size for item in self.regimes.values()],
            "unsupported": [item.unsupported for item in self.regimes.values()],
        }
        if data is not None and hasattr(data, "frame_like"):
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def summary(self) -> str:
        """A short human-readable table."""
        if not self.regimes:
            return "no regimes"
        lines = [f"regime support (n = {self.n})", ""]
        header = (
            f"{'regime':<24}{'min g':>10}{'max ratio':>12}{'effective n':>14}{'unsupported':>13}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for name, item in self.regimes.items():
            lines.append(
                f"{name:<24}{item.min_support_propensity:>10.4g}{item.max_ratio:>12.4g}"
                f"{item.effective_sample_size:>14.1f}{item.unsupported:>13d}"
            )
        return "\n".join(lines)


def check_support(
    regimes: RegimeSet,
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    thresholds: tuple[float, ...] = _THRESHOLDS,
) -> SupportReport:
    """Overlap diagnostics for each regime, from the untruncated mechanism.

    ``propensity`` is the ``(n, K)`` mechanism as estimated, *before* truncation: the
    question this answers is what the data supports, and a bound chosen to control
    variance would answer it by construction.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g = np.asarray(propensity, dtype=float)
    n = int(g.shape[0])
    arm_codes = np.arange(g.shape[1], dtype=float)
    observed_column = (a.reshape(-1, 1) == arm_codes.reshape(1, -1)).astype(float)

    out: dict[str, RegimeSupport] = {}
    for code in regimes.codes:
        star = regimes.column(code)
        mass = star > 0.0
        supported = g[mass] if np.any(mass) else np.asarray([1.0])
        # The ratio at the observed treatment: g*(A | W) / g(A | W), which is what the
        # clever covariate is, and so what a single row's leverage is measured by.
        numerator = np.sum(star * observed_column, axis=1)
        denominator = np.sum(g * observed_column, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), np.inf
            )
        finite = ratio[np.isfinite(ratio)]
        total = float(finite.sum())
        squared = float(np.square(finite).sum())
        ess = (total * total / squared) if squared > 0.0 else 0.0
        assigned = np.min(np.where(mass, g, np.inf), axis=1)
        out[regimes.label(code)] = RegimeSupport(
            name=regimes.label(code),
            min_support_propensity=float(np.min(supported)) if supported.size else float("nan"),
            ratio_quantiles={q: float(np.quantile(finite, q)) for q in _QUANTILES if finite.size},
            max_ratio=float(np.max(finite)) if finite.size else float("inf"),
            effective_sample_size=ess,
            ess_ratio=ess / n if n else 0.0,
            tail_mass={
                float(t): float(np.mean(assigned < t)) if assigned.size else 0.0 for t in thresholds
            },
            unsupported=int(np.sum(np.any(mass & (g <= 0.0), axis=1))),
        )
    return SupportReport(out, n)
