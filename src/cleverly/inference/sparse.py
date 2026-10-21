r"""Influence-curve measures of a sparse estimand, and the warning they raise.

A sequential estimate of a regimen that few units follow reads its influence curve from those
few rows. The Wald interval then under-covers for two reasons that the measures below
quantify. The variance estimate is noisy, with relative standard deviation
:math:`\sqrt{(\hat\kappa - 1)/n}` (:math:`\hat\kappa` the raw kurtosis of the curve). The
sampling distribution is skewed, and the second-order Edgeworth term of the two-sided
studentized statistic predicts the coverage loss.

==========================  ===================================================================
measure                     definition, on the centred curve :math:`D_1, \dots, D_n`
==========================  ===================================================================
``kish``                    :math:`(\sum_i D_i^2)^2 / \sum_i D_i^4 = n / \hat\kappa`, the Kish
                            effective sample size of the squared curve
``skewness``                :math:`\hat\gamma = m_3 / m_2^{3/2}`, with :math:`m_k` the
                            :math:`k`-th central moment
``kurtosis``                :math:`\hat\kappa = m_4 / m_2^2`, raw (3 for a normal curve)
``variance_rel_sd``         :math:`\sqrt{(\hat\kappa - 1)/n}`, the relative standard deviation
                            of the variance estimate
``edgeworth_loss``          :math:`-\phi(z) q_2(z) / n` at :math:`z = \Phi^{-1}(1 - \alpha/2)`,
                            with :math:`q_2(z) = \kappa_e (z^3 - 3z)/6 - \hat\gamma^2 (z^5 +
                            2z^3 - 3z)/9 - (z^3 + 3z)/2` and :math:`\kappa_e = \hat\kappa - 3`
                            (Hall 1992, through Rothe, IZA DP 8758, Proposition 1). A positive
                            value is a predicted shortfall of the Wald coverage and a negative
                            value a predicted excess. A flag reads ``variance_rel_sd`` only
``bca_z0``                  :math:`\Phi^{-1}(\#\{\psi^*_b < \hat\psi\} / B)` of the attached
                            bootstrap draws (a tie counts for half), ``nan`` without a bootstrap
``bca_acceleration``        :math:`\sum_i D_i^3 / (6 (\sum_i D_i^2)^{3/2})`
                            (:func:`~cleverly.inference.bootstrap.bca_acceleration`), ``nan``
                            without a bootstrap
==========================  ===================================================================

:data:`SPARSE_INFLUENCE_THRESHOLD` is the value of ``variance_rel_sd`` above which a fit
without a bootstrap raises :class:`~cleverly.exceptions.SparseInfluenceWarning`. The value
0.125 is close to a Kish count of 64. It was chosen after the X39 truth pilots were read, so
it is a disclosed post-reading choice and not a pre-registered one.

With clusters the unit is the cluster. Every measure reads the cluster sums of the curve, the
same unit as the cluster variance and the cluster bootstrap.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from .._typing import FloatArray, IntArray
from ..exceptions import SparseInfluenceWarning
from ..utils.frames import emit_frame
from .bootstrap import bca_acceleration, bca_bias_correction, cluster_sums
from .delta import reference_quantile

__all__ = [
    "SPARSE_INFLUENCE_THRESHOLD",
    "InfluenceReport",
    "InfluenceRow",
    "influence_report",
    "influence_row",
    "warn_sparse_influence",
]

#: The relative standard deviation of the variance estimate above which a Wald fit warns.
SPARSE_INFLUENCE_THRESHOLD = 0.125


@dataclass(frozen=True)
class InfluenceRow:
    """The influence-curve measures of one estimate.

    Parameters
    ----------
    name : str
        The reported parameter name.
    n : int
        The units of the curve: rows, or clusters for a clustered fit.
    kish : float
        The Kish effective sample size :math:`n / \\hat\\kappa`.
    skewness : float
        The skewness :math:`\\hat\\gamma` of the curve.
    kurtosis : float
        The raw kurtosis :math:`\\hat\\kappa` of the curve.
    variance_rel_sd : float
        The relative standard deviation of the variance estimate.
    edgeworth_loss : float
        The Edgeworth-predicted shortfall of the two-sided Wald coverage.
    bca_z0 : float
        The BCa bias correction of the attached bootstrap, ``nan`` without one.
    bca_acceleration : float
        The BCa acceleration, ``nan`` without a bootstrap.

    Attributes
    ----------
    sparse : bool
    """

    name: str
    n: int
    kish: float
    skewness: float
    kurtosis: float
    variance_rel_sd: float
    edgeworth_loss: float
    bca_z0: float = math.nan
    bca_acceleration: float = math.nan

    @property
    def sparse(self) -> bool:
        """Whether ``variance_rel_sd`` is above :data:`SPARSE_INFLUENCE_THRESHOLD`."""
        return self.variance_rel_sd > SPARSE_INFLUENCE_THRESHOLD


@dataclass(frozen=True)
class InfluenceReport:
    """The influence-curve measures of every estimate of a fit.

    Parameters
    ----------
    rows : tuple of InfluenceRow
        One row per estimate, in report order.
    alpha : float
        The two-sided level the Edgeworth loss is read at.
    backend : str or None, default=None
        Dataframe backend :meth:`to_frame` emits when it receives no data.

    Attributes
    ----------
    sparse : tuple of str
    """

    rows: tuple[InfluenceRow, ...]
    alpha: float
    backend: str | None = None

    @property
    def sparse(self) -> tuple[str, ...]:
        """The names whose variance estimate is above the warning threshold."""
        return tuple(row.name for row in self.rows if row.sparse)

    def __getitem__(self, name: str) -> InfluenceRow:
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(f"{name!r} is not in this report; it holds {[r.name for r in self.rows]}")

    def to_frame(self, data: Any = None) -> Any:
        """One row per estimate, in the input dataframe backend.

        Parameters
        ----------
        data : object or None, default=None
            A dataframe or fitted container whose backend to match. ``None`` uses the
            backend the fit recorded.

        Returns
        -------
        DataFrame
            One row per estimate and one column per measure, ``name`` first.
        """
        payload: dict[str, list[Any]] = {
            column: [getattr(row, column) for row in self.rows]
            for column in (
                "name",
                "n",
                "kish",
                "skewness",
                "kurtosis",
                "variance_rel_sd",
                "edgeworth_loss",
                "bca_z0",
                "bca_acceleration",
            )
        }
        return emit_frame(payload, data, backend=self.backend)


def edgeworth_loss(skewness: float, kurtosis: float, n: int, alpha: float) -> float:
    r"""The Edgeworth-predicted shortfall of a two-sided Wald interval's coverage.

    Parameters
    ----------
    skewness : float
        The skewness :math:`\hat\gamma` of the influence curve.
    kurtosis : float
        The raw kurtosis :math:`\hat\kappa` of the influence curve.
    n : int
        The sample size.
    alpha : float
        The two-sided level.

    Returns
    -------
    float
        :math:`-\phi(z) q_2(z) / n`, positive when the interval is predicted to under-cover.
    """
    z = reference_quantile(alpha)
    excess = kurtosis - 3.0
    q2 = (
        excess * (z**3 - 3.0 * z) / 6.0
        - skewness**2 * (z**5 + 2.0 * z**3 - 3.0 * z) / 9.0
        - (z**3 + 3.0 * z) / 2.0
    )
    return float(-norm.pdf(z) * q2 / n)


def influence_row(
    name: str,
    curve: FloatArray,
    *,
    alpha: float = 0.05,
    estimate: float | None = None,
    draws: FloatArray | None = None,
    cluster: IntArray | None = None,
) -> InfluenceRow:
    """The influence-curve measures of one estimate.

    Parameters
    ----------
    name : str
        The reported parameter name.
    curve : ndarray
        ``(n,)`` influence curve. A positive multiple gives the same row.
    alpha : float, default=0.05
        The two-sided level of the Edgeworth loss.
    estimate : float or None, default=None
        The full-sample estimate, which the BCa bias correction reads.
    draws : ndarray or None, default=None
        The bootstrap replicate estimates. With ``estimate``, the row reports the BCa
        bias correction and acceleration.
    cluster : ndarray or None, default=None
        ``(n,)`` cluster codes. Every measure then reads the cluster sums of the curve, so
        ``n`` is the number of clusters.

    Returns
    -------
    InfluenceRow
        The measures. A curve with no spread reports ``nan`` moments and an infinite Kish
        count.

    Examples
    --------
    >>> import numpy as np
    >>> from cleverly.inference.sparse import influence_row
    >>> row = influence_row("ey", np.array([1.0, -1.0, 1.0, -1.0]))
    >>> row.kish, row.skewness, row.kurtosis
    (4.0, 0.0, 1.0)
    """
    values = cluster_sums(curve, cluster)
    n = int(values.size)
    centred = values - values.mean()
    second = float(np.mean(centred**2))
    nan = math.nan
    if second <= 0.0:
        return InfluenceRow(name, n, math.inf, nan, nan, nan, nan)
    skewness = float(np.mean(centred**3)) / second**1.5
    kurtosis = float(np.mean(centred**4)) / second**2
    z0 = nan
    acceleration = nan
    if draws is not None and estimate is not None:
        finite = np.asarray(draws, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            z0 = bca_bias_correction(finite, estimate)
        acceleration = bca_acceleration(values)
    return InfluenceRow(
        name=name,
        n=n,
        kish=n / kurtosis,
        skewness=skewness,
        kurtosis=kurtosis,
        variance_rel_sd=math.sqrt(max(kurtosis - 1.0, 0.0) / n),
        edgeworth_loss=edgeworth_loss(skewness, kurtosis, n, alpha),
        bca_z0=z0,
        bca_acceleration=acceleration,
    )


def influence_report(
    estimates: Mapping[str, Any],
    *,
    alpha: float = 0.05,
    backend: str | None = None,
    cluster: IntArray | None = None,
) -> InfluenceReport:
    """The influence-curve measures of every estimate that carries a curve.

    Parameters
    ----------
    estimates : mapping of str to ParameterEstimate
        The reported estimates. An estimate with an attached bootstrap also reports the BCa
        bias correction and acceleration.
    alpha : float, default=0.05
        The two-sided level of the Edgeworth loss.
    backend : str or None, default=None
        Dataframe backend the report's :meth:`InfluenceReport.to_frame` emits.
    cluster : ndarray or None, default=None
        ``(n,)`` cluster codes of a clustered fit. Each row then reports the cluster-level
        measures.

    Returns
    -------
    InfluenceReport
        One row per estimate with a curve, in report order.
    """
    rows = []
    for name, estimate in estimates.items():
        curve = getattr(estimate, "influence_curve", None)
        if curve is None:
            continue
        bootstrap = getattr(estimate, "bootstrap", None)
        rows.append(
            influence_row(
                name,
                curve,
                alpha=alpha,
                estimate=None if bootstrap is None else float(estimate.psi),
                draws=None if bootstrap is None else bootstrap.draws,
                cluster=cluster,
            )
        )
    return InfluenceReport(tuple(rows), alpha, backend)


#: The advice of a fit whose full-refit bootstrap a study can license.
_IN_SAMPLE_ADVICE = (
    "Refit with n_bootstrap=199, bootstrap_interval='bca' for an interval that reads the "
    "skew, and see result.diagnostics.influence(). The BCa interval is experimental until a "
    "registered study measures it."
)
#: The advice of an in-sample fit whose percentile bootstrap already ran.
_PERCENTILE_ADVICE = (
    "The percentile bootstrap interval keeps the skew shortfall. Refit with "
    "bootstrap_interval='bca' for an interval that reads the skew, and see "
    "result.diagnostics.influence(). The BCa interval is experimental until a registered "
    "study measures it."
)
#: The advice of a cross-fitted fit: its bootstrap is a diagnostic only (roadmap X20).
_CROSS_FITTED_ADVICE = (
    "No validated interval exists for a cross-fitted fit in this regime: its full-refit "
    "bootstrap over-covers and is a diagnostic only (roadmap X20). Read the Wald interval "
    "with care, and see result.diagnostics.influence()."
)
#: The advice of an in-sample fit whose bootstrap no registered study measures.
_UNMEASURED_ADVICE = (
    "No registered study measures a bootstrap interval for this kind of fit, so its "
    "bootstrap is a diagnostic only. Read the Wald interval with care, and see "
    "result.diagnostics.influence()."
)


def warn_sparse_influence(
    report: InfluenceReport,
    *,
    cross_fitted: bool,
    bootstrap_measured: bool = True,
    percentile_bootstrap: bool = False,
    stacklevel: int | None = None,
) -> None:
    """Warn once for the estimates whose variance estimate is above the threshold.

    The flag reads the relative SD of the variance estimate. The Edgeworth loss in the text
    is information: a flagged row can show a negative loss, which predicts over-coverage
    from the skew and is no reason to clear the flag.

    Parameters
    ----------
    report : InfluenceReport
        The measures of the estimates that supply a Wald interval.
    cross_fitted : bool
        Whether the fit is cross-fitted. Its bootstrap is a diagnostic only, so the text
        does not advise it.
    bootstrap_measured : bool, default=True
        ``False`` for a hazard-model or isotonic-projected fit, whose bootstrap no study
        measures. The text then does not advise the bootstrap. Any other in-sample fit gets
        the BCa advice, which says the interval is experimental.
    percentile_bootstrap : bool, default=False
        Whether the fit already ran a percentile bootstrap. The advice then names the BCa
        interval and not a new bootstrap.
    stacklevel : int or None, default=None
        Passed to :func:`warnings.warn`. ``None`` points at the first caller outside the
        ``cleverly`` package, whichever route reached the fit.

    Warns
    -----
    SparseInfluenceWarning
        When any row's ``variance_rel_sd`` is above :data:`SPARSE_INFLUENCE_THRESHOLD`.
    """
    flagged = [row for row in report.rows if row.sparse]
    if not flagged:
        return
    listed = "; ".join(
        f"{row.name} (Kish {row.kish:.0f}, relative SD {row.variance_rel_sd:.3f}, "
        f"Edgeworth coverage loss {row.edgeworth_loss:.3f})"
        for row in flagged[:5]
    )
    more = f"; and {len(flagged) - 5} more" if len(flagged) > 5 else ""
    advice = (
        _CROSS_FITTED_ADVICE
        if cross_fitted
        else _UNMEASURED_ADVICE
        if not bootstrap_measured
        else _PERCENTILE_ADVICE
        if percentile_bootstrap
        else _IN_SAMPLE_ADVICE
    )
    warnings.warn(
        f"{len(flagged)} estimate(s) read their influence curve from few rows, so the "
        f"variance estimate's relative SD is above {SPARSE_INFLUENCE_THRESHOLD}: "
        f"{listed}{more}. The Wald interval may under-cover. {advice}",
        SparseInfluenceWarning,
        stacklevel=_caller_outside_package() if stacklevel is None else stacklevel,
    )


def _caller_outside_package() -> int:
    """The ``warnings.warn`` stacklevel of the first frame outside the ``cleverly`` package."""
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    frame: Any = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package):
        frame = frame.f_back
        level += 1
    return level
