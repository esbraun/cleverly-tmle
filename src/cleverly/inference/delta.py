r"""Delta-method transformations of estimands.

The risk ratio and odds ratio are smooth functions of the two counterfactual
means, so their influence curves follow from those of ``EY1`` and ``EY0`` by the
chain rule.  Both are handled on the log scale, because a ratio is bounded below
by zero and its sampling distribution is badly skewed in small samples: a
symmetric interval on the log scale, exponentiated, respects the boundary and has
much better coverage.  This is also what R's ``tmle`` reports (``log.psi`` and
``var.log.psi``).

For a risk ratio :math:`\psi = \psi_1 / \psi_0`,

.. math::

    \mathrm{IC}_{\log \psi} = \frac{\mathrm{IC}_{\psi_1}}{\psi_1}
                            - \frac{\mathrm{IC}_{\psi_0}}{\psi_0},

and for an odds ratio :math:`\psi = \frac{\psi_1/(1 - \psi_1)}{\psi_0/(1 - \psi_0)}`,

.. math::

    \mathrm{IC}_{\log \psi} = \frac{\mathrm{IC}_{\psi_1}}{\psi_1 (1 - \psi_1)}
                            - \frac{\mathrm{IC}_{\psi_0}}{\psi_0 (1 - \psi_0)}.

:func:`delta_method` generalises this to any differentiable function of any number
of estimands, which is how a user asks for something the library does not ship --
a ratio of ATTs, a percentage change, a contrast across subgroups.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.special import logit as _logit

from .._typing import FloatArray

__all__ = [
    "Transform",
    "delta_method",
    "log_odds_ratio_influence",
    "log_ratio_influence",
    "normal_ci",
    "two_sided_pvalue",
    "wald_statistic",
]


def _log(value: float) -> float:
    return float(np.log(value))


def _exp(value: float) -> float:
    return float(np.exp(value))


def _log_derivative(value: float) -> float:
    return 1.0 / value


def _logit_scalar(value: float) -> float:
    return float(_logit(value))


def _expit_scalar(value: float) -> float:
    return float(expit(value))


def _logit_derivative(value: float) -> float:
    return 1.0 / (value * (1.0 - value))


@dataclass(frozen=True)
class Transform:
    r"""A monotone map that a contrast's interval and test are computed on.

    The interval is :math:`f^{-1}(f(\hat h) \pm z \cdot se)`, with
    :math:`se` the standard error of :math:`f(\hat h)` by the chain rule. This is the
    ``contrast = list(f, f_inv, h, fh_grad)`` form of R ``drtmle`` 1.1.2 ``ci()``
    (``R/confint.R``, lines 146 to 167). The limits are sorted after the inverse map, so
    a decreasing ``f`` gives an ordered interval.

    A transform built from lambdas does not pickle. :meth:`log` and :meth:`logit` use
    module-level functions and pickle. A fitted result never stores a transformed
    estimate, so ``result.save()`` does not depend on this.

    Parameters
    ----------
    name : str
        A label for the transform, which ``to_dict`` reports.
    forward : callable
        The map :math:`f`, from the scale of the contrast to the inference scale.
    inverse : callable
        The inverse map :math:`f^{-1}`.
    derivative : callable or None, default=None
        The derivative :math:`f'`. ``None`` takes a central difference.

    See Also
    --------
    cleverly.estimators.TMLEResult.contrast : Takes ``transform=``.

    Examples
    --------
    >>> from cleverly.inference import Transform
    >>> log = Transform.log()
    >>> log.name, round(log.forward(1.0), 6), round(log.inverse(0.0), 6)
    ('log', 0.0, 1.0)
    """

    name: str
    forward: Callable[[float], float]
    inverse: Callable[[float], float]
    derivative: Callable[[float], float] | None = None

    @classmethod
    def log(cls) -> Transform:
        """The natural log, with ``exp`` as its inverse.

        Returns
        -------
        Transform
            The log transform, for a positive contrast such as a ratio.
        """
        return cls("log", _log, _exp, _log_derivative)

    @classmethod
    def logit(cls) -> Transform:
        """The logit, with the logistic function as its inverse.

        Returns
        -------
        Transform
            The logit transform, for a contrast inside ``(0, 1)`` such as a probability.
        """
        return cls("logit", _logit_scalar, _expit_scalar, _logit_derivative)

    def slope(self, value: float, *, step: float = 1e-6) -> float:
        """The derivative of :attr:`forward` at ``value``.

        Parameters
        ----------
        value : float
            The point on the scale of the contrast.
        step : float, default=1e-6
            Relative step of the central difference when :attr:`derivative` is ``None``.

        Returns
        -------
        float
            :math:`f'(value)`.
        """
        if self.derivative is not None:
            return float(self.derivative(value))
        h = step * max(1.0, abs(value))
        return (float(self.forward(value + h)) - float(self.forward(value - h))) / (2.0 * h)


def normal_ci(estimate: float, std_error: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wald confidence interval at level ``1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if not np.isfinite(std_error) or std_error < 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return (estimate - z * std_error, estimate + z * std_error)


def wald_statistic(value: float, null: float, std_error: float) -> tuple[float, float]:
    """The Wald statistic and its two-sided normal p-value.

    Parameters
    ----------
    value : float
        The estimate on its inference scale.
    null : float
        The null value on the same scale.
    std_error : float
        The standard error on that scale.

    Returns
    -------
    statistic : float
        ``(value - null) / std_error``, ``nan`` when the standard error is not positive.
    pvalue : float
        ``2 * Phi(-|statistic|)``.
    """
    if not np.isfinite(std_error) or std_error <= 0:
        return float("nan"), float("nan")
    z = (value - null) / std_error
    return float(z), float(2.0 * stats.norm.sf(abs(z)))


def two_sided_pvalue(estimate: float, std_error: float) -> float:
    """Two-sided p-value for ``H0: estimate = 0``."""
    return wald_statistic(estimate, 0.0, std_error)[1]


def log_ratio_influence(
    psi_one: float,
    ic_one: FloatArray,
    psi_zero: float,
    ic_zero: FloatArray,
) -> tuple[float, FloatArray]:
    """Log risk ratio and its influence curve."""
    if psi_one <= 0 or psi_zero <= 0:
        raise ValueError(
            "the risk ratio needs both counterfactual means strictly positive; got "
            f"EY1={psi_one:.6g}, EY0={psi_zero:.6g}"
        )
    log_psi = float(np.log(psi_one) - np.log(psi_zero))
    ic = np.asarray(ic_one, dtype=float) / psi_one - np.asarray(ic_zero, dtype=float) / psi_zero
    return log_psi, ic


def log_odds_ratio_influence(
    psi_one: float,
    ic_one: FloatArray,
    psi_zero: float,
    ic_zero: FloatArray,
) -> tuple[float, FloatArray]:
    """Log odds ratio and its influence curve."""
    for label, value in (("EY1", psi_one), ("EY0", psi_zero)):
        if not 0.0 < value < 1.0:
            raise ValueError(
                "the odds ratio needs both counterfactual means strictly inside (0, 1); "
                f"got {label}={value:.6g}"
            )
    log_psi = float(np.log(psi_one / (1.0 - psi_one)) - np.log(psi_zero / (1.0 - psi_zero)))
    ic = np.asarray(ic_one, dtype=float) / (psi_one * (1.0 - psi_one)) - np.asarray(
        ic_zero, dtype=float
    ) / (psi_zero * (1.0 - psi_zero))
    return log_psi, ic


def delta_method(
    function: Callable[[FloatArray], float],
    estimates: Sequence[float],
    influence_curves: Sequence[FloatArray],
    *,
    gradient: Callable[[FloatArray], FloatArray] | None = None,
    step: float = 1e-6,
) -> tuple[float, FloatArray]:
    r"""Influence curve of ``function`` applied to several estimands.

    Returns ``(value, influence_curve)`` where the influence curve is
    :math:`\nabla f(\hat\psi)^\top \mathrm{IC}`.  With ``gradient=None`` the
    gradient is obtained by central differences, which is accurate enough here
    because the functions of interest are smooth and low-dimensional.

    Parameters
    ----------
    function : callable
        Maps the vector of estimates to the scalar of interest.
    estimates : sequence of float
        The point estimates, in the order ``function`` reads.
    influence_curves : sequence of ndarray
        One ``(n,)`` influence curve per estimate, in the same order.
    gradient : callable or None
        Gradient of ``function``. ``None`` takes central differences.
    step : float
        Relative step size for those differences.

    Returns
    -------
    value : float
        ``function`` at the estimates.
    influence_curve : ndarray
        ``(n,)`` influence curve of that value, carrying the correlation between
        the inputs.

    Examples
    --------
    A ratio of two estimands, with the correlation between their influence curves
    carried through:

    >>> import numpy as np
    >>> from cleverly.inference import delta_method
    >>> rng = np.random.default_rng(0)
    >>> ic_a = rng.normal(size=4)
    >>> ic_b = rng.normal(size=4)
    >>> value, ic = delta_method(lambda p: p[0] / p[1], [2.0, 4.0], [ic_a, ic_b])
    >>> round(value, 6)
    0.5
    >>> ic.shape
    (4,)
    """
    psi = np.asarray(estimates, dtype=float)
    curves = np.column_stack([np.asarray(ic, dtype=float).reshape(-1) for ic in influence_curves])
    if curves.shape[1] != psi.shape[0]:
        raise ValueError(f"got {psi.shape[0]} estimate(s) but {curves.shape[1]} influence curve(s)")

    value = float(function(psi))
    if gradient is not None:
        grad = np.asarray(gradient(psi), dtype=float).reshape(-1)
    else:
        grad = np.empty_like(psi)
        for j in range(psi.size):
            h = step * max(1.0, abs(float(psi[j])))
            forward, backward = psi.copy(), psi.copy()
            forward[j] += h
            backward[j] -= h
            grad[j] = (float(function(forward)) - float(function(backward))) / (2.0 * h)
    if grad.shape[0] != psi.shape[0]:
        raise ValueError(f"gradient has length {grad.shape[0]}, expected {psi.shape[0]}")
    return value, np.asarray(curves @ grad, dtype=float)
