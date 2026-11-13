r"""The records of the policy-weighted variance, and the rule that combines it with the curve's.

``LTMLE(variance="policy_weighted")`` and ``SDR(variance="policy_weighted")`` read a second
variance beside the influence-curve one: the change-of-measure sequential plug-in of
:mod:`cleverly.longitudinal.policy_variance`. It integrates every node's squared residual
over the intervened law through the ratio at each row's policy dose, so it does not depend
on whether the sample happens to hold the rare rows with very large cumulative ratios.

The fit computes one matrix :math:`\hat\Sigma_{pw}` over its levels (:class:`PolicyCovariance`).
A reported estimate with gradient :math:`\lambda` over those levels, a level, a contrast or the
gradient of a smooth function of them, then carries the variance

.. math::

    \hat\sigma^2 = \max(\hat\sigma^2_{IC}, \lambda^\top \hat\Sigma_{pw} \lambda),

with :math:`\hat\sigma^2_{IC}` the variance of its influence curve. A policy-weighted value that
is not finite, or not above :data:`POLICY_FLOOR` times :math:`\hat\sigma^2_{IC}`, is not used,
and the estimate keeps :math:`\hat\sigma^2_{IC}` (:attr:`PolicyVariance.source` then reads
``"influence_curve"``). So :math:`\hat\sigma \ge \hat\sigma_{IC}` in every sample, and the
interval contains the default Wald interval: its coverage is at least the default's at every
``n``, with no condition on the variance regressions.

The interval keeps the scale and the reference the default interval uses. Its half-width on
that scale is the larger of the default half-width and :math:`z\,\hat\sigma_{pw}`, with
:math:`z` the normal quantile (:meth:`PolicyVariance.interval`). On a normal reference this is
the Wald interval at :math:`\hat\sigma`, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from .._typing import FloatArray
from .delta import wald_ci, wald_statistic

__all__ = [
    "POLICY_FLOOR",
    "PolicyCovariance",
    "PolicyVariance",
    "VarianceSource",
]

#: Which variance an estimate under ``variance="policy_weighted"`` reports.
VarianceSource = Literal["influence_curve", "policy_weighted"]

#: A policy-weighted variance at or below this multiple of the influence-curve variance is
#: treated as degenerate, and the estimate keeps the influence-curve variance.
POLICY_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True, eq=False)
class PolicyCovariance:
    r"""The policy-weighted covariance of one fit's levels.

    One object is shared by every estimate of the fit, so a contrast, a ratio or a transform
    of those estimates reads the same matrix.

    Parameters
    ----------
    names : tuple of str
        The level names, in the order of the matrix's rows.
    matrix : ndarray
        ``(k, k)`` :math:`\hat\Sigma_{pw}`, the variances and covariances of the level
        estimates on the outcome's own scale: second moments divided by ``n``. An entry may
        be non-finite when its regressions failed.
    """

    names: tuple[str, ...]
    matrix: FloatArray

    def quadratic(self, gradient: FloatArray) -> float:
        r"""The policy-weighted variance of a functional, :math:`\lambda^\top \hat\Sigma \lambda`.

        Parameters
        ----------
        gradient : ndarray
            ``(k,)`` :math:`\lambda`, in the order of :attr:`names`.

        Returns
        -------
        float
            The quadratic form, ``nan`` when an entry it reads is not finite or a variance
            on the diagonal it reads is not positive: such a block is no covariance, and the
            form would mix a failed variance into a contrast.
        """
        weights = np.asarray(gradient, dtype=float).reshape(-1)
        used = weights != 0.0
        block = self.matrix[np.ix_(used, used)]
        if not np.all(np.isfinite(block)) or not np.all(np.diag(block) > 0.0):
            return float("nan")
        return float(weights[used] @ block @ weights[used])

    def same_as(self, other: PolicyCovariance) -> bool:
        """Whether ``other`` is this fit's covariance: the same object, or equal content.

        Parameters
        ----------
        other : PolicyCovariance
            The covariance of another estimate.

        Returns
        -------
        bool
            ``True`` when the names and the matrix agree.
        """
        return other is self or (
            other.names == self.names
            and bool(np.array_equal(other.matrix, self.matrix, equal_nan=True))
        )


@dataclass(frozen=True)
class PolicyVariance:
    r"""Both variances of one estimate under ``variance="policy_weighted"``, and the one it reports.

    Parameters
    ----------
    covariance : PolicyCovariance
        The fit's shared covariance of its levels.
    gradient : ndarray
        ``(k,)`` :math:`\lambda`, the estimate's gradient on its inference scale with
        respect to the levels of :attr:`covariance`.
    influence_variance : float
        :math:`\hat\sigma^2_{IC}`, the variance the default reads off the influence curve.
    policy_variance : float
        :math:`\lambda^\top \hat\Sigma_{pw} \lambda`. It can be non-finite or non-positive, and
        is then not used.

    Attributes
    ----------
    usable : bool
    source : str
    variance : float
    """

    covariance: PolicyCovariance
    gradient: FloatArray
    influence_variance: float
    policy_variance: float

    @property
    def usable(self) -> bool:
        """Whether the policy-weighted variance is finite and above the floor."""
        floor = POLICY_FLOOR * max(self.influence_variance, 0.0)
        return bool(np.isfinite(self.policy_variance) and self.policy_variance > floor)

    @property
    def source(self) -> VarianceSource:
        """``"policy_weighted"`` when that variance is used and larger, else the curve's.

        It names the larger variance, not the wider half-width. With a Student t reference
        the default half-width ``t * SE_IC`` can exceed ``z * SE_pw`` while ``SE_pw`` is the
        larger, and :meth:`interval` is then the default interval although this reads
        ``"policy_weighted"``.
        """
        if self.usable and self.policy_variance > self.influence_variance:
            return "policy_weighted"
        return "influence_curve"

    @property
    def variance(self) -> float:
        """The reported variance: the larger of the two, or the curve's when the other is unused."""
        if self.source == "policy_weighted":
            return float(self.policy_variance)
        return float(self.influence_variance)

    def interval(
        self, center: float, alpha: float, reference_df: int | None
    ) -> tuple[float, float]:
        """The interval on the inference scale: the default one widened to ``z`` times the pw SE.

        Parameters
        ----------
        center : float
            The estimate on its inference scale.
        alpha : float
            The two-sided level.
        reference_df : int or None
            The degrees of freedom of the default interval's Student t reference, or
            ``None`` for the normal reference. The policy-weighted half-width always reads
            the normal reference.

        Returns
        -------
        tuple of float
            The union of the default interval and the normal interval at the policy-weighted
            standard error.
        """
        default = wald_ci(center, _root(self.influence_variance), alpha, df=reference_df)
        if not self.usable:
            return default
        policy = wald_ci(center, _root(self.policy_variance), alpha)
        return (min(default[0], policy[0]), max(default[1], policy[1]))

    def test(self, value: float, null: float, reference_df: int | None) -> tuple[float, float]:
        """The Wald statistic and p-value that reject when :meth:`interval` excludes ``null``.

        Parameters
        ----------
        value : float
            The estimate on its inference scale.
        null : float
            The null on the same scale.
        reference_df : int or None
            The default reference's degrees of freedom, as in :meth:`interval`.

        Returns
        -------
        tuple of float
            The statistic and the p-value of whichever of the two tests has the larger
            p-value, or the smaller statistic when the p-values tie.
        """
        default = wald_statistic(value, null, _root(self.influence_variance), df=reference_df)
        if not self.usable:
            return default
        policy = wald_statistic(value, null, _root(self.policy_variance))
        if not np.isfinite(default[1]):
            return policy
        if policy[1] == default[1]:
            # Both p-values can underflow to zero far from the null: the smaller statistic
            # is then the less extreme test.
            return policy if abs(policy[0]) < abs(default[0]) else default
        return policy if policy[1] > default[1] else default


def _root(variance: float) -> float:
    """The standard error of a variance, ``nan`` when it is negative or not finite."""
    if not np.isfinite(variance) or variance < 0.0:
        return float("nan")
    return float(np.sqrt(variance))
