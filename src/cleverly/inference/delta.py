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

import numpy as np
from scipy import stats

from .._typing import FloatArray

__all__ = [
    "delta_method",
    "log_odds_ratio_influence",
    "log_ratio_influence",
    "normal_ci",
    "two_sided_pvalue",
]


def normal_ci(estimate: float, std_error: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wald confidence interval at level ``1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if not np.isfinite(std_error) or std_error < 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return (estimate - z * std_error, estimate + z * std_error)


def two_sided_pvalue(estimate: float, std_error: float) -> float:
    """Two-sided p-value for ``H0: estimate = 0``."""
    if not np.isfinite(std_error) or std_error <= 0:
        return float("nan")
    z = estimate / std_error
    return float(2.0 * stats.norm.sf(abs(z)))


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

    >>> # A ratio of two estimands, with correct correlation handling:
    >>> value, ic = delta_method(lambda p: p[0] / p[1], [2.0, 4.0], [ic_a, ic_b])
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
