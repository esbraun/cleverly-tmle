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
from typing import Any, TypedDict

import numpy as np

from .._typing import FloatArray
from ..data.weighting import effective_sample_size, top_weight_share
from ..utils.frames import emit_frame
from ..utils.text import format_table
from .base import RegimeSet

__all__ = ["RegimeSupport", "SupportReport", "check_support"]

#: Propensity thresholds the report counts rows below.  The same ladder
#: :mod:`cleverly.sensitivity.positivity` uses, so the two tables read together.
_THRESHOLDS = (0.01, 0.025, 0.05)

_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


class _InterventionLoadRow(TypedDict):
    """One intervention equation's fitted absolute score-weight concentration."""

    equation: str
    reported_repeat: int
    n_repeats: int
    n_total: float
    n_targeted: float
    effective: float
    targeted_ratio: float
    total_ratio: float
    top_1pct: float
    top_5pct: float
    max_load: float
    zero_load: float


def _intervention_loads(
    labels: tuple[str, ...],
    absolute_score_weights: FloatArray | None,
    equations: tuple[str, ...],
    n_total: int,
    n_repeats: int = 1,
) -> tuple[dict[str, _InterventionLoadRow], str | None]:
    """Match fitted score columns to intervention labels without rebuilding them."""
    if absolute_score_weights is None:
        return {}, "the fitted artifact has no exact absolute score weights"
    loads = np.asarray(absolute_score_weights, dtype=float)
    if (
        loads.ndim != 2
        or loads.shape[1] != len(labels)
        or loads.shape[1] != len(equations)
        or loads.shape[1] == 0
    ):
        return {}, "the fitted absolute score weights do not match the intervention equations"
    if np.any(~np.isfinite(loads)) or np.any(loads < 0.0):
        return {}, "the fitted absolute score weights are not finite and nonnegative"
    n_targeted = int(loads.shape[0])
    if not 0 < n_targeted <= n_total:
        return {}, "the fitted score mask size is outside the fitted data"

    rows: dict[str, _InterventionLoadRow] = {}
    for index, label in enumerate(labels):
        load = loads[:, index]
        effective = effective_sample_size(load, on_degenerate=0.0)
        rows[label] = {
            "equation": equations[index],
            "reported_repeat": 1,
            "n_repeats": n_repeats,
            "n_total": float(n_total),
            "n_targeted": float(n_targeted),
            "effective": effective,
            "targeted_ratio": effective / float(n_targeted),
            "total_ratio": effective / float(n_total) if n_total else 0.0,
            "top_1pct": top_weight_share(load, 0.01),
            "top_5pct": top_weight_share(load, 0.05),
            "max_load": float(np.max(load)),
            "zero_load": float(np.count_nonzero(load == 0.0)),
        }
    return rows, None


@dataclass(frozen=True)
class RegimeSupport:
    """Overlap for one regime.

    Parameters
    ----------
    name : str
        Report label of the regime.
    min_support_propensity : float
        The smallest :math:`g_a(W_i)` over the rows and arms the regime puts mass on.
        For a deterministic rule that is :math:`\\min_i g_{d(W_i)}(W_i)`, the number a
        reader of a dynamic regime wants first.
    ratio_quantiles : dict of float to float
        Quantiles of the density ratio :math:`g^\\star(A \\mid W)/g(A \\mid W)` at the
        *observed* treatment -- the clever covariate's magnitude, before the missingness
        and intermediate mechanisms enter it.
    max_ratio : float
        The largest such ratio: how much one row can move the estimate.
    effective_sample_size : float
        Kish ESS of those ratios, and ``ess_ratio`` its share of ``n``.  A regime far
        from the observed mechanism throws away information even when nothing is
        formally violated, and this is the size of that loss.
    ess_ratio : float
        The effective sample size as a share of ``n``.
    tail_mass : dict of float to float
        Fraction of rows whose assigned-arm propensity falls below each threshold.
    unsupported : int
        Rows where the regime assigns positive probability to an arm with an estimated
        propensity of exactly zero -- a structural violation rather than a practical one.
        The parameter is not identified for those rows at all.
    score_load : dict of str to float or str or None
        Concentration of the fitted ``abs(w_i * H_ij)`` values for this regime's score
        equation. ``None`` means the fitted artifact cannot supply the exact values.
    score_load_omission : str or None
        Machine-readable reason why :attr:`score_load` is unavailable.
    """

    name: str
    min_support_propensity: float
    ratio_quantiles: dict[float, float]
    max_ratio: float
    effective_sample_size: float
    ess_ratio: float
    tail_mass: dict[float, float]
    unsupported: int
    score_load: _InterventionLoadRow | None = None
    score_load_omission: str | None = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a report whose pickle can predate the score-load fields.

        Parameters
        ----------
        state : dict of str to Any
            Instance values carried by the pickle.
        """
        self.__dict__.update(state)
        if "score_load" not in state:
            object.__setattr__(self, "score_load", None)
        if "score_load_omission" not in state:
            object.__setattr__(
                self,
                "score_load_omission",
                "the report predates fitted score-load diagnostics",
            )


@dataclass(frozen=True)
class SupportReport:
    """:class:`RegimeSupport` for every regime in a fit, plus the worst case.

    Parameters
    ----------
    regimes : dict of str to RegimeSupport
        One record per regime, keyed by its report label.
    n : int
        Number of observations the records were computed over.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns when ``data`` is omitted.
    """

    regimes: dict[str, RegimeSupport] = field(default_factory=dict)
    n: int = 0
    #: Name of the dataframe backend the fit's data arrived in, so that
    #: :meth:`to_frame` honours "results come back in the backend you passed in"
    #: without a caller having to thread the container back in by hand.
    backend: str | None = None

    @property
    def worst(self) -> RegimeSupport | None:
        """The regime with the smallest assigned-arm propensity."""
        if not self.regimes:
            return None
        return min(self.regimes.values(), key=lambda item: item.min_support_propensity)

    def to_frame(self, data: Any = None) -> Any:
        """One row per regime, in the backend the data came from.

        Parameters
        ----------
        data : Any
            A dataframe or fitted container whose backend to match. ``None`` uses
            :attr:`backend`.

        Returns
        -------
        dataframe
            One row per regime.
        """
        payload = {
            "regime": list(self.regimes),
            "min_propensity": [item.min_support_propensity for item in self.regimes.values()],
            "max_ratio": [item.max_ratio for item in self.regimes.values()],
            "effective_n": [item.effective_sample_size for item in self.regimes.values()],
            "unsupported": [item.unsupported for item in self.regimes.values()],
            "score_equation": [
                None if item.score_load is None else item.score_load["equation"]
                for item in self.regimes.values()
            ],
            "score_effective_n": [
                float("nan") if item.score_load is None else item.score_load["effective"]
                for item in self.regimes.values()
            ],
            "score_load_ratio": [
                float("nan") if item.score_load is None else item.score_load["targeted_ratio"]
                for item in self.regimes.values()
            ],
            "score_top_5pct": [
                float("nan") if item.score_load is None else item.score_load["top_5pct"]
                for item in self.regimes.values()
            ],
            "score_load_omission": [item.score_load_omission for item in self.regimes.values()],
        }
        return emit_frame(payload, data, backend=self.backend)

    def summary(self) -> str:
        """A short human-readable table.

        Returns
        -------
        str
            A printable table, one line per row of the report.
        """
        if not self.regimes:
            return "no regimes"
        rows = []
        for name, item in self.regimes.items():
            score = (
                "unavailable"
                if item.score_load is None
                else (
                    f"{item.score_load['effective']:.1f}/{item.score_load['n_targeted']:.0f} "
                    f"(draw 01/{item.score_load['n_repeats']:02d})"
                )
            )
            rows.append(
                [
                    name,
                    f"{item.min_support_propensity:.4g}",
                    f"{item.max_ratio:.4g}",
                    f"{item.effective_sample_size:.1f}",
                    score,
                    str(item.unsupported),
                ]
            )
        return "\n".join(
            [
                f"regime support (n = {self.n})",
                "",
                format_table(
                    [
                        "regime",
                        "min g",
                        "max ratio",
                        "ratio effective n",
                        "score load",
                        "unsupported",
                    ],
                    rows,
                ),
                "",
                "score load is Kish-equivalent mask rows from abs(w_i * H_ij), not estimator ESS.",
            ]
        )


def check_support(
    regimes: RegimeSet,
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    thresholds: tuple[float, ...] = _THRESHOLDS,
    backend: str | None = None,
    absolute_score_weights: FloatArray | None = None,
    equations: tuple[str, ...] = (),
    n_repeats: int = 1,
) -> SupportReport:
    """Overlap diagnostics for each regime, from the untruncated mechanism.

    ``propensity`` is the ``(n, K)`` mechanism as estimated, *before* truncation: the
    question this answers is what the data supports, and a bound chosen to control
    variance would answer it by construction.

    ``backend`` names the dataframe library the fit's data arrived in, so that
    :meth:`SupportReport.to_frame` returns it -- which is what this function's caller
    already promised in prose and did not deliver.  This takes arrays rather than a
    container, so it has to be told.

    Parameters
    ----------
    regimes : RegimeSet
        The evaluated regimes to report on.
    treatment : ndarray
        ``(n,)`` observed treatment, in arm codes.
    propensity : ndarray
        ``(n, K)`` mechanism as estimated, before truncation.
    thresholds : tuple of float
        Propensity levels the tail mass is reported at.
    backend : str or None
        Dataframe backend the fit's data arrived in, for :meth:`SupportReport.to_frame`.
    absolute_score_weights : ndarray or None
        Fitted ``abs(w_i * H_ij)`` columns, in regime order. ``None`` records an
        omission instead of rebuilding a possibly different equation.
    equations : tuple of str
        Fitted score-equation names, in regime order.
    n_repeats : int
        Number of stored cross-fitting draws. The retained weights describe draw 1.

    Returns
    -------
    SupportReport
        One record per regime, plus the worst case across them.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g = np.asarray(propensity, dtype=float)
    n = int(g.shape[0])
    arm_codes = np.arange(g.shape[1], dtype=float)
    observed_column = (a.reshape(-1, 1) == arm_codes.reshape(1, -1)).astype(float)

    out: dict[str, RegimeSupport] = {}
    labels = tuple(regimes.label(code) for code in regimes.codes)
    score_loads, load_omission = _intervention_loads(
        labels, absolute_score_weights, equations, n, n_repeats
    )
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
        ess = effective_sample_size(finite, on_degenerate=0.0)
        assigned = np.min(np.where(mass, g, np.inf), axis=1)
        label = regimes.label(code)
        out[label] = RegimeSupport(
            name=label,
            min_support_propensity=float(np.min(supported)) if supported.size else float("nan"),
            ratio_quantiles={q: float(np.quantile(finite, q)) for q in _QUANTILES if finite.size},
            max_ratio=float(np.max(finite)) if finite.size else float("inf"),
            effective_sample_size=ess,
            ess_ratio=ess / n if n else 0.0,
            tail_mass={
                float(t): float(np.mean(assigned < t)) if assigned.size else 0.0 for t in thresholds
            },
            unsupported=int(np.sum(np.any(mass & (g <= 0.0), axis=1))),
            score_load=score_loads.get(label),
            score_load_omission=load_omission,
        )
    return SupportReport(out, n, backend=backend)
