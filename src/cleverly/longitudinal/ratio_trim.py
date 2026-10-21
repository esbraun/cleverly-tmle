r"""Quantile trimming of the per-node density ratios of a longitudinal fit.

``LTMLE(ratio_trim=q)`` applies the trimming rule of ``lmtp`` 1.5.4 (``lmtp_control(.trim=)``).
The rule reads the ``(n, T)`` matrix of per-node ratios

.. math::

    r_t(i) = \frac{\pi_t(A_t \mid H_t)\, 1\{C_t = 1\}}{g_t(A_t \mid H_t)\, c_t(H_t, A_t)},

evaluated at the observed arm and the observed history, from the unbounded mechanism.  An
entry is zero by node ``t`` alone: the unit was not at risk at ``t``, it was censored at
``t``, or its treatment at ``t`` has zero intervention density (``lmtp``'s ``followed`` reads
``A_t`` only, ``density_ratios.R:55-61``).  So a unit that left a static plan at ``s < t``
and follows it at ``t`` has a positive entry at ``t``.

One quantile ``Q_q`` of all ``n x T`` entries, zeros included, is computed with R's default
type 7 (``utils.R:41-42``, ``density_ratios.R:21``).  Every entry above it is replaced by
``Q_q``, and the cumulative trimmed ratio is the running product
:math:`\tilde R_t = \prod_{s \le t} \min(r_s, Q_q)` (``eif.R:8-15``).  ``lmtp`` therefore trims
the per-node ratios and not the cumulative one, and a cross-fitted fit trims the pooled
out-of-fold matrix once, not per fold.

The trim composes with ``g_bounds`` through the mechanism it implies.  The shipped clever
covariate is ``num_cum / bound(g_cum, lower, upper)``, with ``num_cum`` the running product
of :math:`\pi_t` and ``g_cum`` the running product of the mechanism factors.  The trimmed
ratio implies the cumulative mechanism :math:`\tilde g_{\mathrm{cum}} = num_{\mathrm{cum}} /
\tilde R_t`, and the clever covariate becomes ``num_cum / bound(g~_cum, lower, upper)``.
On a row whose entries through ``t`` were never trimmed, ``g~_cum`` is the shipped
``g_cum`` itself, so the clever covariate is the shipped one for every pair of bounds,
``upper < 1`` included.  A row with ``R~_t = 0`` has left the plan or the study by ``t``,
and the masks give it a zero clever covariate.

A trim that binds keeps a remainder term of the size of the outcome-regression error at every
sample size, so a fit whose trim binds reports the ``"trimmed_ratio_plugin"`` status
(:mod:`cleverly._inference_status`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import DataError
from .data import LongitudinalData
from .regimen import Plan

if TYPE_CHECKING:
    from .sequential import Mechanism

__all__ = ["RatioTrim", "check_ratio_trim", "node_ratio_entries", "trim_ratios"]


def check_ratio_trim(ratio_trim: float | None) -> None:
    """Refuse a ``ratio_trim`` that is not a quantile strictly between 0 and 1.

    Parameters
    ----------
    ratio_trim : float or None
        The declared quantile, or ``None`` for no trim.

    Raises
    ------
    ValueError
        When ``ratio_trim`` is not ``None`` and not a real number in ``(0, 1)``.
    """
    if ratio_trim is None:
        return
    valid = (
        isinstance(ratio_trim, (int, float, np.floating))
        and not isinstance(ratio_trim, bool)
        and 0.0 < float(ratio_trim) < 1.0
    )
    if not valid:
        raise ValueError(
            f"ratio_trim must be a quantile strictly between 0 and 1; got {ratio_trim!r}. "
            "ratio_trim=None trims nothing"
        )


@dataclass(frozen=True)
class RatioTrim:
    """One plan's trimmed per-node density ratios.

    Parameters
    ----------
    quantile : float
        The declared quantile ``q`` of ``ratio_trim=``.
    threshold : float
        ``Q_q``, the type-7 quantile of every entry of :attr:`entries`.
    entries : FloatArray
        ``(n, T)`` untrimmed per-node ratios at the observed arm and history.
    cumulative : FloatArray
        ``(n, T)`` running product of the trimmed entries, :math:`\\tilde R_t`.
    trimmed : BoolArray
        ``(n, T)`` indicator that an entry lay above :attr:`threshold`.

    Attributes
    ----------
    trimmed_share : float
    """

    quantile: float
    threshold: float
    entries: FloatArray
    cumulative: FloatArray
    trimmed: BoolArray

    @property
    def trimmed_share(self) -> float:
        """The share of the positive entries that the trim moved.

        Returns
        -------
        float
            Trimmed entries over positive entries; zero when no entry is positive.
        """
        positive = int(np.count_nonzero(self.entries > 0.0))
        return float(np.count_nonzero(self.trimmed)) / positive if positive else 0.0

    def implied_cumulative(self, numerator: FloatArray, raw: FloatArray) -> FloatArray:
        """``(n, T)`` cumulative mechanism the trimmed ratio implies, before ``g_bounds``.

        Parameters
        ----------
        numerator : FloatArray
            ``(n, T)`` running product of the intervention density at the observed arms.
        raw : FloatArray
            ``(n, T)`` running product of the unbounded mechanism factors at the plan.

        Returns
        -------
        FloatArray
            ``numerator / cumulative`` where an entry through ``t`` was trimmed and the
            trimmed ratio is positive, and ``raw`` elsewhere.
        """
        touched = np.logical_or.accumulate(self.trimmed, axis=1)
        positive = self.cumulative > 0.0
        safe = np.where(positive, self.cumulative, 1.0)
        return np.where(touched & positive, numerator / safe, raw)


def node_ratio_entries(data: LongitudinalData, plan: Plan, mechanism: Mechanism) -> FloatArray:
    """``(n, T)`` per-node ratio of one plan, as ``lmtp`` builds it before the trim.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plan : Plan
        The resolved plan, with its policy numerators attached.
    mechanism : Mechanism
        The fitted, unbounded mechanism.

    Returns
    -------
    FloatArray
        The intervention density at the observed arm times the retention indicator, over the
        treatment and retention probabilities at the observed history.  Zero where the unit
        is not at risk at the node, where it was censored there, or where the density or the
        denominator is zero.
    """
    density = plan.intervention_density(data)
    columns = []
    rows = np.arange(data.n)
    for time in range(1, data.n_times + 1):
        observed = mechanism.treatment_observed[time - 1]
        if data.node_kind(time) == "mixed":
            # The slot of a mixed node holds 0.0 on every present row, so it is no cell code.
            # The mechanism's factor is the observed cell's probability on the density route
            # and one on the classifier route, both at the observed history.
            treatment = np.asarray(mechanism.treatment[time - 1][plan.label], dtype=float)
        elif observed.shape[1]:
            codes = np.nan_to_num(data.treatment[:, time - 1], nan=0.0).astype(np.int64)
            treatment = observed[rows, codes]
        else:
            # A held node, or a continuous node, whose density ratio is the numerator.
            treatment = np.ones(data.n)
        if data.censoring_names:
            retention = mechanism.censoring_observed[time - 1]
            stayed = data.uncensored_through(time)
        else:
            retention = np.ones(data.n)
            stayed = np.ones(data.n, dtype=bool)
        at_risk = data.event_free_through(time - 1) & stayed
        denominator = treatment * retention
        positive = at_risk & (denominator > 0.0)
        safe = np.where(positive, denominator, 1.0)
        columns.append(np.where(positive, density[:, time - 1] / safe, 0.0))
    return np.column_stack(columns)


def trim_ratios(
    data: LongitudinalData, plans: Sequence[Plan], mechanism: Mechanism, quantile: float
) -> dict[str, RatioTrim]:
    """Trim every plan's per-node ratios at one declared quantile.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans, with their policy numerators attached.
    mechanism : Mechanism
        The fitted, unbounded mechanism.
    quantile : float
        The declared ``ratio_trim=``.

    Returns
    -------
    dict of str to RatioTrim
        One record per plan label.

    Raises
    ------
    DataError
        When the quantile of a plan's entries is zero, which would set every ratio to zero.
    """
    trims: dict[str, RatioTrim] = {}
    for plan in plans:
        entries = node_ratio_entries(data, plan, mechanism)
        threshold = float(np.quantile(entries.ravel(), quantile, method="linear"))
        if not threshold > 0.0:
            share = float(np.mean(entries == 0.0))
            raise DataError(
                f"ratio_trim={quantile!r} trims the per-node density ratios of regimen "
                f"{plan.label!r} at 0: {share:.3g} of its {entries.size} entries are zero, "
                "because the units left the plan, were censored or had the event. Every "
                "ratio would then be zero. Raise ratio_trim above that share, or omit it"
            )
        trimmed = entries > threshold
        kept = np.minimum(entries, threshold)
        trims[plan.label] = RatioTrim(
            quantile=float(quantile),
            threshold=threshold,
            entries=entries,
            cumulative=np.cumprod(kept, axis=1, dtype=np.float64),
            trimmed=trimmed,
        )
    return trims
