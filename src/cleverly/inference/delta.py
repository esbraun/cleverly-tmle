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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.special import logit as _logit

from .._typing import FloatArray

__all__ = [
    "Transform",
    "cornish_fisher_factor",
    "delta_method",
    "implied_correlation",
    "log_odds_ratio_influence",
    "log_ratio_influence",
    "mover_ci",
    "one_sided_quantile",
    "reference_quantile",
    "skew_widened_ci",
    "two_sided_pvalue",
    "wald_ci",
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


def _range_logit(value: float, low: float, high: float) -> float:
    return float(_logit((value - low) / (high - low)))


def _range_expit(value: float, low: float, high: float) -> float:
    return float(low + (high - low) * expit(value))


def _range_logit_derivative(value: float, low: float, high: float) -> float:
    return (high - low) / ((value - low) * (high - value))


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

    @classmethod
    def range_logit(cls, low: float, high: float) -> Transform:
        """The logit of a value's position in the open interval ``(low, high)``.

        The map is :math:`f(x) = \\operatorname{logit}((x - a) / (b - a))` with
        :math:`(a, b)` = ``(low, high)``. On ``(0, 1)`` it is the logit. On ``(-1, 1)`` it is
        :math:`2 \\operatorname{atanh}(x)`, so a symmetric interval on its scale is the
        ``atanh`` interval of a difference of two probabilities.

        Parameters
        ----------
        low : float
            The lower end of the range.
        high : float
            The upper end of the range, above ``low``.

        Returns
        -------
        Transform
            The transform, named ``"logit"`` on ``(0, 1)`` and ``"range_logit"`` otherwise.

        Raises
        ------
        ValueError
            If ``high`` is not above ``low``, or either end is not finite.
        """
        if not (np.isfinite(low) and np.isfinite(high) and high > low):
            raise ValueError(f"range_logit needs finite low < high; got ({low}, {high})")
        if (low, high) == (0.0, 1.0):
            return cls.logit()
        return cls(
            "range_logit",
            partial(_range_logit, low=float(low), high=float(high)),
            partial(_range_expit, low=float(low), high=float(high)),
            partial(_range_logit_derivative, low=float(low), high=float(high)),
        )

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


def _normal_reference(df: float | None) -> bool:
    """Whether ``df`` names the normal reference, after refusing an unusable one.

    ``None`` or infinite degrees of freedom name the normal.  Any other value must be a number
    of at least 1, so NaN and ``df < 1`` are refused here, for every interval, limit and test
    alike.
    """
    if df is None or df == math.inf:
        return True
    if not df >= 1:
        raise ValueError(f"a Student t reference needs at least 1 degree of freedom; got {df}")
    return False


def reference_quantile(alpha: float, df: float | None = None) -> float:
    """The two-sided critical value of a Wald interval at level ``1 - alpha``.

    The one place an interval or a sensitivity limit reads a reference quantile.

    Parameters
    ----------
    alpha : float
        The two-sided significance level, in ``(0, 1)``.
    df : float or None, default=None
        Degrees of freedom of a Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference.

    Returns
    -------
    float
        The ``1 - alpha / 2`` quantile of the normal, or of t with ``df`` degrees of
        freedom.

    Examples
    --------
    >>> from cleverly.inference import reference_quantile
    >>> round(reference_quantile(0.05), 4), round(reference_quantile(0.05, df=10), 4)
    (1.96, 2.2281)
    >>> round(reference_quantile(0.05, df=17.5), 4)
    2.1052
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if _normal_reference(df):
        return float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def reference_quantiles(alphas: Any, df: float | None = None) -> Any:
    """The two-sided critical values at many levels at once, in the tail form.

    :func:`reference_quantile` for each entry of ``alphas``, computed as the upper
    ``alpha / 2`` tail so that levels down to ``1e-300`` keep their precision. An asymmetric
    interval reads it on its inversion grid, and its test reads it at the p-value to report
    the statistic.

    Parameters
    ----------
    alphas : array-like
        Two-sided significance levels in ``(0, 1]``.
    df : float or None, default=None
        Degrees of freedom of a Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference.

    Returns
    -------
    numpy.ndarray
        The ``1 - alpha / 2`` quantiles, 0 at ``alpha = 1``.
    """
    values = np.asarray(alphas, dtype=float) / 2.0
    if _normal_reference(df):
        return stats.norm.isf(values)
    return stats.t.isf(values, df)


def one_sided_quantile(level: float, df: float | None = None) -> float:
    """The quantile at ``level`` of the reference of a one-sided limit.

    Parameters
    ----------
    level : float
        The one-sided coverage level, in ``(0, 1)``.
    df : float or None, default=None
        Degrees of freedom of a Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference.

    Returns
    -------
    float
        The normal quantile, or the t quantile with ``df`` degrees of freedom.
    """
    if _normal_reference(df):
        return float(stats.norm.ppf(level))
    return float(stats.t.ppf(level, df))


def normal_quantile(level: float) -> float:
    """The standard normal quantile at ``level``, for a bootstrap level map or a diagnostic.

    A Wald interval takes :func:`reference_quantile` or :func:`one_sided_quantile`, which honour
    a Student t reference. This helper is for the reads that are not a Wald reference: the
    BCa bias correction and level map, and the Edgeworth term of the influence report.

    Parameters
    ----------
    level : float
        Probability in ``[0, 1]``.

    Returns
    -------
    float
        :math:`\\Phi^{-1}(\\text{level})`, infinite at 0 and 1.
    """
    return float(stats.norm.ppf(level))


def normal_cdf(value: float) -> float:
    """The standard normal distribution function at ``value``.

    Parameters
    ----------
    value : float
        The point to evaluate.

    Returns
    -------
    float
        :math:`\\Phi(\\text{value})`.
    """
    return float(stats.norm.cdf(value))


def wald_ci(
    estimate: float,
    std_error: float,
    alpha: float = 0.05,
    *,
    df: float | None = None,
    margin: float = 0.0,
) -> tuple[float, float]:
    """Wald confidence interval at level ``1 - alpha``, optionally widened by a margin.

    The half-width is ``critical * std_error + margin``. A fit under a declared stopping
    rule passes the absolute score it leaves as ``margin``
    (:attr:`~cleverly.inference.ParameterEstimate.rule_widened`).

    Parameters
    ----------
    estimate : float
        The estimate on its inference scale.
    std_error : float
        The standard error on that scale.
    alpha : float, default=0.05
        The two-sided significance level.
    df : float or None, default=None
        Degrees of freedom of a Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference.
    margin : float, default=0.0
        A non-negative amount added to the half-width, on the same scale.

    Returns
    -------
    tuple of float
        The lower and upper limits, ``nan`` when the standard error is not finite or is
        negative.

    Examples
    --------
    >>> from cleverly.inference import wald_ci
    >>> [round(limit, 4) for limit in wald_ci(1.0, 0.5, df=10)]
    [-0.1141, 2.1141]
    """
    critical = reference_quantile(alpha, df)
    if not np.isfinite(std_error) or std_error < 0:
        return (float("nan"), float("nan"))
    half = critical * std_error + margin
    return (estimate - half, estimate + half)


def cornish_fisher_factor(alpha: float) -> float:
    """The factor ``K = 2 z^2 + 1`` of :func:`skew_widened_ci` at level ``1 - alpha``.

    ``z`` is the ``1 - alpha / 2`` normal quantile. ``K`` is the polynomial of the one-term
    Edgeworth expansion of a studentized mean, ``(2 x^2 + 1) / 6`` times the skewness over the
    square root of the sample size, evaluated at ``z``.

    Parameters
    ----------
    alpha : float
        The two-sided significance level, in ``(0, 1)``.

    Returns
    -------
    float
        ``2 z^2 + 1``; 8.683 at ``alpha = 0.05``.
    """
    z = reference_quantile(alpha)
    return 2.0 * z * z + 1.0


def skew_widened_ci(
    estimate: float,
    std_error: float,
    alpha: float = 0.05,
    *,
    df: float | None,
    acceleration: float,
    margin: float = 0.0,
) -> tuple[float, float]:
    r"""The skew-widened Wald interval: the hull of a t interval and its skew correction.

    With ``q`` the critical value of :func:`reference_quantile` at ``df``, ``a`` the
    acceleration of the estimate's influence curve
    (:func:`~cleverly.inference.curve_acceleration`) and ``K = 2 z^2 + 1``
    (:func:`cornish_fisher_factor`), the limits are

    .. math::

        \hat\psi - s \max(q, q - a K) - m, \qquad \hat\psi + s \max(q, q + a K) + m,

    with ``s`` the standard error and ``m`` the ``margin``. For ``a > 0`` the upper limit moves
    out by ``a K s`` and the lower one stays at the t limit, and the reverse for ``a < 0``. The
    interval is the convex hull of the t interval and the Cornish-Fisher-corrected interval
    ``[psi - s (q - a K), psi + s (q + a K)]``, so it contains the t interval on every sample
    and never narrows a side. The correction removes the curve-skewness term of the
    studentized statistic's one-term Edgeworth expansion and leaves its median-bias term, so
    no per-tail rate is claimed for it.

    Parameters
    ----------
    estimate : float
        The estimate.
    std_error : float
        Its standard error.
    alpha : float, default=0.05
        The two-sided significance level.
    df : float or None
        Degrees of freedom of the Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference. A caller that caps the degrees of freedom
        passes the capped value.
    acceleration : float
        The acceleration ``a`` of the estimate's influence curve.
    margin : float, default=0.0
        A non-negative amount added to each side's half-width.

    Returns
    -------
    tuple of float
        The lower and upper limits, ``nan`` when the standard error is not finite or is
        negative, or the acceleration is not finite.

    See Also
    --------
    wald_ci : The symmetric interval this one contains.

    Examples
    --------
    >>> from cleverly.inference import skew_widened_ci, wald_ci
    >>> low, high = skew_widened_ci(1.0, 0.5, df=12, acceleration=0.02)
    >>> round(low, 4), round(high, 4)
    (-0.0894, 2.1762)
    >>> round(wald_ci(1.0, 0.5, df=12)[0], 4)
    -0.0894
    """
    critical = reference_quantile(alpha, df)
    if not (np.isfinite(std_error) and std_error >= 0 and np.isfinite(acceleration)):
        return (float("nan"), float("nan"))
    shift = float(acceleration) * cornish_fisher_factor(alpha)
    lower = std_error * max(critical, critical - shift) + margin
    upper = std_error * max(critical, critical + shift) + margin
    return (float(estimate - lower), float(estimate + upper))


def implied_correlation(
    difference_variance: float, minuend_variance: float, subtrahend_variance: float
) -> float:
    """The correlation of two estimates implied by their variances and their difference's.

    ``c = (v_1 + v_0 - v_d) / 2`` and ``r = c / sqrt(v_1 v_0)``, clipped to ``[-1, 1]``. It is
    the correlation of the two estimates under the covariance rule of their variances when the
    difference's influence curve is the difference of theirs, as for the treatment-arm means
    and the difference of one fit. Read this way, :func:`mover_ci` with symmetric arm intervals
    returns the difference's own Wald interval exactly.

    Parameters
    ----------
    difference_variance : float
        The variance of the difference ``psi_1 - psi_0``.
    minuend_variance : float
        The variance of ``psi_1``.
    subtrahend_variance : float
        The variance of ``psi_0``.

    Returns
    -------
    float
        The correlation, ``nan`` when either arm variance is not positive and finite or the
        difference's variance is not finite.

    Examples
    --------
    >>> from cleverly.inference import implied_correlation
    >>> implied_correlation(1.0, 1.0, 1.0)
    0.5
    """
    values = (difference_variance, minuend_variance, subtrahend_variance)
    if not all(np.isfinite(value) for value in values):
        return float("nan")
    if not (minuend_variance > 0.0 and subtrahend_variance > 0.0):
        return float("nan")
    covariance = (minuend_variance + subtrahend_variance - difference_variance) / 2.0
    return float(np.clip(covariance / np.sqrt(minuend_variance * subtrahend_variance), -1.0, 1.0))


def mover_ci(
    minuend: float,
    minuend_interval: tuple[float, float],
    subtrahend: float,
    subtrahend_interval: tuple[float, float],
    *,
    correlation: float,
) -> tuple[float, float]:
    r"""The correlated MOVER interval of a difference from the intervals of its two terms.

    With ``d = psi_1 - psi_0``, arm intervals ``(l_j, u_j)`` around ``psi_j`` and the
    correlation ``r`` of the two estimates, the limits are

    .. math::

        L = d - \sqrt{(\psi_1 - l_1)^2 + (u_0 - \psi_0)^2 - 2 r (\psi_1 - l_1)(u_0 - \psi_0)},

        U = d + \sqrt{(u_1 - \psi_1)^2 + (\psi_0 - l_0)^2 - 2 r (u_1 - \psi_1)(\psi_0 - l_0)}.

    Each limit recovers the variance that the arm half-widths on its side imply. With ``r = 0``
    it is the MOVER interval for independent estimates (Tang 2021, display 1). When both arm
    intervals are symmetric, ``l_j = psi_j - q s_j``, it equals the Wald interval
    ``d +/- q sqrt(s_1^2 + s_0^2 - 2 r s_1 s_0)`` for every ``r``.

    Parameters
    ----------
    minuend : float
        The estimate ``psi_1``.
    minuend_interval : tuple of float
        Its interval ``(l_1, u_1)``, which must contain it.
    subtrahend : float
        The estimate ``psi_0``.
    subtrahend_interval : tuple of float
        Its interval ``(l_0, u_0)``, which must contain it.
    correlation : float
        The correlation ``r`` of the two estimates, in ``[-1, 1]``
        (:func:`implied_correlation`).

    Returns
    -------
    tuple of float
        The lower and upper limits, ``nan`` when any input is not finite.

    Raises
    ------
    ValueError
        When ``correlation`` lies outside ``[-1, 1]`` or an interval does not contain its
        estimate.

    See Also
    --------
    implied_correlation : The correlation of two arm estimates of one fit.

    Examples
    --------
    >>> from cleverly.inference import mover_ci
    >>> low, high = mover_ci(0.6, (0.5, 0.68), 0.2, (0.15, 0.27), correlation=0.0)
    >>> round(low, 4), round(high, 4)
    (0.2779, 0.4943)
    """
    values = (minuend, *minuend_interval, subtrahend, *subtrahend_interval, correlation)
    if not all(np.isfinite(value) for value in values):
        return (float("nan"), float("nan"))
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"the correlation must lie in [-1, 1]; got {correlation}")
    low_1, high_1 = minuend_interval
    low_0, high_0 = subtrahend_interval
    if not (low_1 <= minuend <= high_1 and low_0 <= subtrahend <= high_0):
        raise ValueError("each arm interval must contain its estimate")
    difference = minuend - subtrahend
    down_1, up_1 = minuend - low_1, high_1 - minuend
    down_0, up_0 = subtrahend - low_0, high_0 - subtrahend
    lower = down_1**2 + up_0**2 - 2.0 * correlation * down_1 * up_0
    upper = up_1**2 + down_0**2 - 2.0 * correlation * up_1 * down_0
    return (
        float(difference - np.sqrt(max(lower, 0.0))),
        float(difference + np.sqrt(max(upper, 0.0))),
    )


def wald_statistic(
    value: float,
    null: float,
    std_error: float,
    *,
    df: float | None = None,
    margin: float = 0.0,
) -> tuple[float, float]:
    """The Wald statistic and its two-sided p-value.

    With a positive ``margin`` the p-value is that of the test which inverts
    :func:`wald_ci` at the same margin: the distance to the null is first reduced by
    ``margin``, so the p-value is at most ``alpha`` exactly when the null lies outside the
    widened interval. The statistic itself stays ``(value - null) / std_error``.

    Parameters
    ----------
    value : float
        The estimate on its inference scale.
    null : float
        The null value on the same scale.
    std_error : float
        The standard error on that scale.
    df : float or None, default=None
        Degrees of freedom of a Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference.
    margin : float, default=0.0
        The non-negative half-width margin of the interval the test inverts.

    Returns
    -------
    statistic : float
        ``(value - null) / std_error``, ``nan`` when the standard error is not positive.
    pvalue : float
        ``2 * Phi(-d)``, or ``2 * F_t(-d; df)``, with
        ``d = max(0, |statistic| - margin / std_error)``; ``nan`` when the statistic is ``nan``.
    """
    normal = _normal_reference(df)
    if not np.isfinite(std_error) or std_error <= 0:
        return float("nan"), float("nan")
    z = (value - null) / std_error
    distance = float(np.maximum(0.0, abs(z) - margin / std_error))
    tail = stats.norm.sf(distance) if normal else stats.t.sf(distance, df)
    return float(z), float(2.0 * tail)


def two_sided_pvalue(estimate: float, std_error: float, *, df: float | None = None) -> float:
    """Two-sided p-value for ``H0: estimate = 0``.

    Parameters
    ----------
    estimate : float
        The estimate on its inference scale.
    std_error : float
        The standard error on that scale.
    df : float or None, default=None
        Degrees of freedom of a Student t reference, which may be fractional. ``None`` or
        ``math.inf`` takes the normal reference.

    Returns
    -------
    float
        The p-value of :func:`wald_statistic` at a null of zero.
    """
    return wald_statistic(estimate, 0.0, std_error, df=df)[1]


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
