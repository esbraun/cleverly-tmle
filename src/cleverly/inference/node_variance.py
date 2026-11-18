r"""The node-variance supplement of a longitudinal estimate whose recursion read a boundary node.

An ``LTMLE`` node regression whose target holds one value, 0 or 1, on every fitted row is
that value, so the node's term :math:`C_{t,i}(Z_{t,i} - \bar Q^*_{t,i})` of the influence curve
is identically zero and the curve's variance leaves the node out
(:func:`~cleverly.longitudinal.sequential.constant_nodes`). This module bounds the omitted
variance.

**The bound.** With no event among :math:`m` independent units whose node hazard is
:math:`h`, :math:`P(\text{no event}) = (1-h)^m`. The value

.. math::

    \hat h_U = 1 - (\alpha/2)^{1/m}

solves :math:`(1-h)^m = \alpha/2`, the exact one-sided :math:`1-\alpha/2` upper limit of a
binomial proportion with zero events (:func:`hazard_bound`). Clustered, a cluster holds no
event with probability at most :math:`1-h` whatever the dependence inside it, so the limit over
:math:`m` clusters holds without a dependence model. The case "all hold 1" is the mirror image.
A row's residual variance is then at most :math:`s_t^2 = \hat h_U(1-\hat h_U)`.

**The supplement.** For each flagged node :math:`t` of each level fit :math:`r` and each
independent unit :math:`u` (a row, or a cluster) the loading is
:math:`\ell_{r,t,u} = s_t \sum_{i\in R_{r,t}\cap u}|C_{r,t,i}|`. An estimate with gradient
:math:`\lambda` over the levels adds

.. math::

    \hat\sigma^2_{node}(\lambda) = n^{-2}\sum_u\Big(\sum_t\sum_r|\lambda_r|\,\ell_{r,t,u}\Big)^2

when clustered, and :math:`n^{-2}\sum_u\sum_t(\sum_r|\lambda_r|\ell_{r,t,u})^2` on independent
rows, where a row's residuals at different nodes are martingale increments and so
uncorrelated (:meth:`NodeLoadings.bound`). Within a unit the bound is Minkowski's inequality
over every residual the unit holds, whatever their correlation, so the term bounds the omitted
variance whenever every row's residual variance is at most :math:`s_t^2`. The reported variance
is :math:`\hat\sigma^2_{IC} + \hat\sigma^2_{node}(\lambda)`, and the interval keeps the curve's
scale, slope and reference (:class:`NodeVariance`).

**What it claims.** At a law whose node hazards lie in :math:`(0, 1)` and whose node risk sets
grow linearly in :math:`n`, a flagged node has probability at most :math:`(1-h_{min})^{cn}`, and
the supplemented interval differs from the ordinary one only on that event, so it inherits the
shipped coverage. At a true hazard of exactly 0 or 1 the true node term and the upstream terms it
feeds are zero, and the supplement only widens a correct interval. In finite samples at an
interior law where the event is common nothing here guarantees coverage; the registered study
``boundary-node-ltmle`` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple

import numpy as np

from .._typing import FloatArray

__all__ = [
    "BOUNDARY_ALPHA_SPLIT",
    "BoundaryNode",
    "NodeLoadings",
    "NodeVariance",
    "hazard_bound",
]

#: The share of the two-sided level the one-sided bound spends: :math:`\alpha/2`.
BOUNDARY_ALPHA_SPLIT: Final[float] = 0.5


def hazard_bound(units: int, alpha: float) -> float:
    r"""The exact one-sided upper limit of a hazard with no event among ``units`` units.

    :math:`\hat h_U = 1 - (\alpha/2)^{1/m}`, which solves :math:`(1-h)^m = \alpha/2`.

    Parameters
    ----------
    units : int
        :math:`m`, the independent units with a positive-weight row at the node: rows when
        unclustered, clusters when clustered.
    alpha : float
        The two-sided level of the fit's intervals, in ``(0, 1)``.

    Returns
    -------
    float
        :math:`\hat h_U`, in ``(0, 1)``.

    Raises
    ------
    ValueError
        When ``units`` is below 1 or ``alpha`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from cleverly.inference.node_variance import hazard_bound
    >>> round(hazard_bound(1, 0.05), 6)
    0.975
    >>> bound = hazard_bound(200, 0.05)
    >>> round((1.0 - bound) ** 200, 6)
    0.025
    """
    if int(units) != units or units < 1:
        raise ValueError(f"a hazard bound needs at least one unit at the node; got {units!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    return float(1.0 - (BOUNDARY_ALPHA_SPLIT * alpha) ** (1.0 / int(units)))


class BoundaryNode(NamedTuple):
    """One flagged node of one level fit, as the supplement reads it."""

    #: The name of the level estimate whose fit read the node.
    level: str
    #: The one-based node.
    node: int
    #: The constant value the node's target held on its fitted rows, 0 or 1.
    value: float
    #: :math:`m_t`, the independent units with a positive-weight fitted row.
    units: int
    #: :math:`\hat h_{U,t}`, the hazard bound at ``units``.
    bound: float


@dataclass(frozen=True, eq=False)
class NodeLoadings:
    r"""The per-unit loadings of every flagged node of one fit, shared by its estimates.

    Parameters
    ----------
    levels : tuple of str
        The names of the level estimates with at least one flagged node, in the order of the
        second axis of :attr:`loadings`.
    loadings : ndarray
        ``(U, L, T)`` :math:`\ell_{r,t,u}` on the outcome's scale: units by levels by the
        flagged node times, zero where a level read no flagged node at that time.
    times : tuple of int
        The one-based node times of the third axis.
    clustered : bool
        Whether the units are clusters, so that the bound sums every node of a unit inside
        the square.
    n : int
        The rows behind the fit, the :math:`n` of the curve variance.
    nodes : tuple of BoundaryNode
        Each flagged node of each level.
    """

    levels: tuple[str, ...]
    loadings: FloatArray
    times: tuple[int, ...]
    clustered: bool
    n: int
    nodes: tuple[BoundaryNode, ...]

    def bound(self, gradient: FloatArray) -> float:
        r"""The node variance :math:`\hat\sigma^2_{node}(\lambda)` of a functional of the levels.

        Parameters
        ----------
        gradient : ndarray
            ``(L,)`` :math:`\lambda` on the estimate's inference scale, in the order of
            :attr:`levels`.

        Returns
        -------
        float
            The per-unit sum over nodes and levels, squared and summed over units, over
            :math:`n^2` when clustered; the per-unit-and-node sum over levels when not.
        """
        weights = np.abs(np.asarray(gradient, dtype=float).reshape(-1))
        if weights.size != len(self.levels):
            raise ValueError(
                f"the gradient has {weights.size} entries and the loadings have "
                f"{len(self.levels)} levels"
            )
        # (U, T): each unit's loading at each node, summed over levels with |lambda_r|.
        per_node = np.einsum("ult,l->ut", self.loadings, weights)
        if self.clustered:
            total = float(np.sum(np.sum(per_node, axis=1) ** 2))
        else:
            total = float(np.sum(per_node**2))
        return total / float(self.n) ** 2

    def same_as(self, other: NodeLoadings) -> bool:
        """Whether ``other`` is this fit's loadings: the same object, or equal content.

        Parameters
        ----------
        other : NodeLoadings
            The loadings of another estimate.

        Returns
        -------
        bool
            ``True`` when the levels, times, unit kind and loadings agree.
        """
        return other is self or (
            other.levels == self.levels
            and other.times == self.times
            and other.clustered == self.clustered
            and other.n == self.n
            and bool(np.array_equal(other.loadings, self.loadings))
        )


@dataclass(frozen=True)
class NodeVariance:
    r"""Both variance parts of an estimate whose recursion read a boundary node.

    The reported variance is :attr:`variance`, the curve's plus the node supplement, and the
    estimate's interval and test read it on the curve's own scale and reference.

    Parameters
    ----------
    loadings : NodeLoadings
        The fit's shared loadings.
    gradient : ndarray
        ``(L,)`` :math:`\lambda`, the estimate's gradient on its inference scale with respect
        to the levels of :attr:`loadings`.
    influence_variance : float
        :math:`\hat\sigma^2_{IC}`, the variance of the estimate's influence curve.

    Attributes
    ----------
    node_variance : float
    variance : float
    """

    loadings: NodeLoadings
    gradient: FloatArray
    influence_variance: float

    @property
    def node_variance(self) -> float:
        """The supplement :math:`\\hat\\sigma^2_{node}(\\lambda)`, non-negative."""
        return self.loadings.bound(self.gradient)

    @property
    def variance(self) -> float:
        """The reported variance, :math:`\\hat\\sigma^2_{IC} + \\hat\\sigma^2_{node}`."""
        return float(self.influence_variance + self.node_variance)
