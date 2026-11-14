r"""The switched variance: the influence-curve variance raised to a predictable variation where it is sparse.

``ContinuousSurvivalTMLE(variance="switched")`` reads a second variance beside the
influence-curve one: the predictable variation :math:`\hat\sigma^2_{pv}` of the martingale part
of the efficient influence function (:func:`cleverly.survival._targeting.predictable_covariance`).
The influence-curve variance :math:`\hat\sigma^2_{IC}` squares the realized martingale increments;
the predictable variation integrates their conditional variance under the fitted law, so it does
not depend on whether the sample happens to hold the few events that carry a sparse curve.

The fit computes one matrix :math:`\hat\Sigma_{pv}` over its targeted levels
(:class:`PredictableCovariance`).  A reported estimate with gradient :math:`g` over those levels
(a level, a difference, the gradient of a log ratio or of a smooth function of them) carries

.. math::

    \hat\sigma^2 =
    \begin{cases}
        \max(\hat\sigma^2_{IC}, g^\top \hat\Sigma_{pv} g)
            & \text{if } r > \texttt{SPARSE\_INFLUENCE\_THRESHOLD}, \\
        \hat\sigma^2_{IC} & \text{otherwise,}
    \end{cases}

where :math:`r = \sqrt{(\hat\kappa - 1)/n}` is the relative standard deviation of the estimate's
own influence-curve variance (:func:`cleverly.inference.sparse.influence_row`) and the threshold
is :data:`~cleverly.inference.sparse.SPARSE_INFLUENCE_THRESHOLD`, 0.125.  So
:math:`\hat\sigma \ge \hat\sigma_{IC}` in every sample, the interval at the same centre contains
the influence-curve interval, and since :math:`r` falls like :math:`1/\sqrt n` the switched
variance equals the influence-curve variance with probability tending to one.

The record is the explicit-variance counterpart of the policy-weighted record of the longitudinal
estimators: an estimate stores both variances and reports which one it uses as
:attr:`~cleverly.inference.ParameterEstimate.variance_source`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

from .._typing import FloatArray

__all__ = [
    "SWITCHED_FLOOR",
    "PredictableCovariance",
    "SwitchedVariance",
    "VarianceSource",
    "switched_record",
]

#: Which variance an estimate under ``variance="switched"`` reports.
VarianceSource = Literal["influence_curve", "predictable_variation"]

#: A predictable variation at or below this multiple of the influence-curve variance is treated
#: as degenerate, and the estimate keeps the influence-curve variance.
SWITCHED_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True, eq=False)
class PredictableCovariance:
    r"""The predictable-variation covariance of one fit's targeted levels.

    Parameters
    ----------
    targets : tuple of str
        The level names, in the order of the matrix's rows: every arm's risk of every cause at
        every target time, then its restricted means time lost, in the walk's target order.
    matrix : ndarray
        ``(P, P)`` :math:`\hat\Sigma_{pv}` on the variance scale of the reported levels (time
        units for a restricted mean): the predictable-variation second moments divided by
        :math:`n - 1`, the divisor the influence-curve variance uses.
    gradients : mapping of str to ndarray
        Each stored estimate's ``(P,)`` gradient over :attr:`targets` on its own inference
        scale (the log for a ratio), so :meth:`quadratic` of it is that estimate's predictable
        variance.
    """

    targets: tuple[str, ...]
    matrix: FloatArray
    gradients: Mapping[str, FloatArray] = field(default_factory=dict)

    def quadratic(self, gradient: FloatArray) -> float:
        r"""The predictable variance of a functional, :math:`g^\top \hat\Sigma_{pv} g`.

        Parameters
        ----------
        gradient : ndarray
            ``(P,)`` :math:`g`, in the order of :attr:`targets`.

        Returns
        -------
        float
            The quadratic form, ``nan`` when an entry it reads is not finite.
        """
        weights = np.asarray(gradient, dtype=float).reshape(-1)
        if weights.size != len(self.targets):
            raise ValueError(
                f"gradient has {weights.size} entries and the covariance has "
                f"{len(self.targets)} targets"
            )
        used = weights != 0.0
        block = self.matrix[np.ix_(used, used)]
        if not np.all(np.isfinite(block)):
            return float("nan")
        return float(weights[used] @ block @ weights[used])

    def to_dict(self) -> dict[str, object]:
        """The targets and the matrix as nested lists, for a JSON record.

        Returns
        -------
        dict of str to object
            ``targets`` and ``matrix``.
        """
        return {"targets": list(self.targets), "matrix": self.matrix.tolist()}


@dataclass(frozen=True)
class SwitchedVariance:
    r"""Both variances of one estimate under ``variance="switched"``, and the one it reports.

    Parameters
    ----------
    influence_variance : float
        :math:`\hat\sigma^2_{IC}`, the variance read off the estimate's influence curve.
    predictable_variance : float
        :math:`g^\top \hat\Sigma_{pv} g`, its predictable variation.  Not used when it is not
        finite or not above :data:`SWITCHED_FLOOR` times the influence-curve variance.
    relative_sd : float
        :math:`\sqrt{(\hat\kappa - 1)/n}` of the estimate's influence curve, the relative
        standard deviation of its influence-curve variance.
    threshold : float
        The relative standard deviation above which the switch reads the predictable variation.

    Attributes
    ----------
    switched : bool
    usable : bool
    source : str
    variance : float
    """

    influence_variance: float
    predictable_variance: float
    relative_sd: float
    threshold: float

    @property
    def switched(self) -> bool:
        """Whether the influence curve is sparse: its relative SD exceeds the threshold."""
        return bool(np.isfinite(self.relative_sd) and self.relative_sd > self.threshold)

    @property
    def usable(self) -> bool:
        """Whether the predictable variation is finite and above the floor."""
        floor = SWITCHED_FLOOR * max(self.influence_variance, 0.0)
        return bool(np.isfinite(self.predictable_variance) and self.predictable_variance > floor)

    @property
    def source(self) -> VarianceSource:
        """``"predictable_variation"`` when the switch is on and that variance is the larger."""
        if (
            self.switched
            and self.usable
            and self.predictable_variance > self.influence_variance
        ):
            return "predictable_variation"
        return "influence_curve"

    @property
    def variance(self) -> float:
        """The reported variance."""
        if self.source == "predictable_variation":
            return float(self.predictable_variance)
        return float(self.influence_variance)


def switched_record(
    influence_curve: FloatArray,
    influence_variance: float,
    predictable_variance: float,
    *,
    threshold: float | None = None,
) -> SwitchedVariance:
    """The switched record of one estimate from its curve and its two variances.

    Parameters
    ----------
    influence_curve : ndarray
        ``(n,)`` the estimate's influence curve on its inference scale.
    influence_variance : float
        The variance read off that curve.
    predictable_variance : float
        Its predictable variation on the same scale.
    threshold : float or None
        The switch's threshold.  ``None`` reads
        :data:`~cleverly.inference.sparse.SPARSE_INFLUENCE_THRESHOLD`.

    Returns
    -------
    SwitchedVariance
        The record.
    """
    # Imported here: the sparse module imports the influence module, which imports this one.
    from .sparse import SPARSE_INFLUENCE_THRESHOLD, influence_row

    curve = np.asarray(influence_curve, dtype=float).reshape(-1)
    row = influence_row("switched", curve)
    return SwitchedVariance(
        influence_variance=float(influence_variance),
        predictable_variance=float(predictable_variance),
        relative_sd=float(row.variance_rel_sd),
        threshold=SPARSE_INFLUENCE_THRESHOLD if threshold is None else float(threshold),
    )
