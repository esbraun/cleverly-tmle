r"""The C-TMLE0 update: one logistic fit of the clever covariates and their derivatives.

C-TMLE0 (Ju, Wyss, Franklin, Schneeweiss, Häggström and van der Laan, C-LASSO, Section 4.2,
equation (6), and Appendix B, Algorithm 2; R ``ctmle`` 0.1.2, ``ctmletype = 3``) is an
ordinary TMLE whose fluctuation adds, for each arm, the derivative of that arm's clever
covariate along a one-dimensional path of treatment mechanisms.  The derivative is the
finite difference between the fitted mechanism ``g`` and a partner mechanism ``g+``.
Nothing is selected collaboratively.

:func:`derivative_update` is a pure function of arrays.  It takes the initial outcome
regression, the ``mean`` submodel at the bounded mechanism, and the bounded mechanism and
partner, and it fits the ``2K`` coefficients in one call of the shipped solver.  The R
witness fixture feeds it R's own arrays.

The ``K`` columns ``h_a`` are among the ``2K``, so a converged solve sets every ``h_a``
score to zero and the plug-in solves :math:`P_n D^*(\bar Q^*, g_n) = 0`.  The reported
curve is :math:`D^*(\bar Q^*, g_n)` with the bounded ``g`` alone.  The derivative columns
move :math:`\bar Q^*` and never enter the curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray
from ..exceptions import CapabilityError
from ..fluctuation.iterative import Fluctuation, InitialFit
from ..fluctuation.submodel import Submodel, derivative_columns
from .targeting import TargetingSpec, solve_submodel

__all__ = [
    "DERIVATIVE_ONE_STEP_REFUSAL",
    "DERIVATIVE_TARGET_WEIGHTS_REFUSAL",
    "DerivativeUpdate",
    "derivative_ratio",
    "derivative_update",
]

#: The refusal of ``targeting="one_step"`` under ``strategy="derivative"``.
DERIVATIVE_ONE_STEP_REFUSAL = (
    "strategy='derivative' runs R's single logistic fit of the 2K clever covariates, the "
    "mechanism's and their derivatives. targeting='one_step' walks the universal "
    "least-favorable path of one target instead, which ctmle type 3 does not define. Use "
    "targeting='iterative'."
)

#: The refusal of ``target_weights=True`` under ``strategy="derivative"``.  Each treated row
#: loads its arm's clever covariate and that covariate's derivative, and the weighted form
#: moves a row's covariate into one scalar weight, which cannot keep two scores apart.
DERIVATIVE_TARGET_WEIGHTS_REFUSAL = (
    "strategy='derivative' fits each arm's clever covariate and its derivative together, so "
    "every row loads two columns. target_weights=True moves a row's covariate into one "
    "observation weight, which keeps only one of the two scores. Use target_weights=False."
)

#: The size below which a derivative column counts as zero, relative to one.
_ZERO_COLUMN = 1e-12


@dataclass(frozen=True)
class DerivativeUpdate:
    """The result of the C-TMLE0 update.

    Parameters
    ----------
    targeted : InitialFit
        The targeted outcome regression, at the observed treatment and at each arm.
    fluctuation : Fluctuation
        The ``2K``-column solve on the normalized columns.
    epsilon : ndarray
        The ``2K`` coefficients on R's scale: the ``h_a`` coefficients, then the
        derivative coefficients multiplied back by their normalization.
    names : tuple of str
        ``h0``, ``h1``, ..., then ``dh0``, ``dh1``, ..., in the order of ``epsilon``.
    scale : ndarray
        The normalization factor of each arm's derivative column.
    score_before : ndarray
        Per arm, the normalized derivative score :math:`P_n dh_a (Y - \\bar Q)` at the
        initial fit.
    score_after : ndarray
        The same score at the targeted fit.
    """

    targeted: InitialFit
    fluctuation: Fluctuation
    epsilon: FloatArray
    names: tuple[str, ...]
    scale: FloatArray
    score_before: FloatArray
    score_after: FloatArray


def derivative_ratio(mechanism: FloatArray, partner: FloatArray) -> FloatArray:
    """``(g - g+) / g`` per row and arm, from the bounded mechanism and the bounded partner.

    Parameters
    ----------
    mechanism : ndarray
        ``(n, K)``, the bounded mechanism ``g``.
    partner : ndarray
        ``(n, K)``, the partner ``g+`` bounded at the same ``g_bounds``.

    Returns
    -------
    ndarray
        ``(n, K)``, the factor that turns ``h_a`` into its derivative column.
    """
    g = np.asarray(mechanism, dtype=float)
    plus = np.asarray(partner, dtype=float)
    if g.shape != plus.shape:
        raise ValueError(
            f"the mechanism has shape {g.shape} and the partner has shape {plus.shape}"
        )
    return (g - plus) / g


def derivative_update(
    scaled: FloatArray,
    initial: InitialFit,
    base: Submodel,
    mechanism: FloatArray,
    partner: FloatArray,
    treatment: FloatArray,
    observed: BoolArray,
    weights: FloatArray,
    spec: TargetingSpec,
) -> DerivativeUpdate:
    """Fit the C-TMLE0 fluctuation of the clever covariates and their derivatives.

    Parameters
    ----------
    scaled : ndarray
        The outcome on the ``[0, 1]`` scale.
    initial : InitialFit
        The initial outcome regression.
    base : Submodel
        The ``mean`` submodel at the bounded ``mechanism``, one column per arm.
    mechanism : ndarray
        ``(n, K)``, the bounded mechanism ``g``, columns in ``base.levels`` order.
    partner : ndarray
        ``(n, K)``, the partner ``g+``, bounded at the same ``g_bounds``.
    treatment : ndarray
        The arm codes.
    observed : ndarray
        The rows with an observed outcome.
    weights : ndarray
        The observation weights, normalized to mean one.
    spec : TargetingSpec
        The targeting settings of the fit.

    Returns
    -------
    DerivativeUpdate
        The targeted fit, the solve, and the coefficients on R's scale.

    Raises
    ------
    CapabilityError
        Under ``targeting="one_step"`` or ``target_weights=True``.
    ValueError
        When an arm's derivative column is zero or repeats its clever covariate on every
        row with that treatment and an observed outcome, or when the solve does not
        converge.
    """
    if spec.targeting == "one_step":
        raise CapabilityError(DERIVATIVE_ONE_STEP_REFUSAL)
    if spec.target_weights:
        raise CapabilityError(DERIVATIVE_TARGET_WEIGHTS_REFUSAL)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    mask = np.asarray(observed, dtype=bool)
    ratio = derivative_ratio(mechanism, partner)
    for j, level in enumerate(base.levels):
        rows = (a == level) & mask
        values = ratio[rows, j]
        largest = float(np.max(np.abs(values))) if values.size else 0.0
        if largest <= _ZERO_COLUMN:
            raise ValueError(
                "The partner mechanism equals the mechanism after g_bounds on every row "
                f"with treatment {level:g} and an observed outcome, so that arm's "
                "derivative column is zero and its derivative equation is empty. Pass a "
                "partner fitted at another penalty or by another learner."
            )
        if float(np.max(np.abs(values - values.mean()))) <= _ZERO_COLUMN * largest:
            raise ValueError(
                "The partner differs from the mechanism by a constant factor on every row "
                f"with treatment {level:g} and an observed outcome, so that arm's "
                "derivative column repeats its clever covariate and its derivative "
                "equation adds nothing. Pass a partner fitted at another penalty or by "
                "another learner."
            )
    submodel, scale = derivative_columns(base, ratio, a, mask)
    fluctuation = solve_submodel(scaled, initial, submodel, weights, mask, spec, warn=False)
    k = len(base.levels)
    if not fluctuation.converged:
        relative = np.abs(fluctuation.score[k:]) / np.maximum(
            np.asarray(fluctuation.score_scale)[k:], 1e-300
        )
        worst = int(np.argmax(relative))
        raise ValueError(
            "The derivative fluctuation of strategy='derivative' did not converge in "
            f"{spec.max_iter} iterations. The derivative score of arm "
            f"{base.levels[worst]:g} is {relative[worst]:.3g} on the normalized scale "
            f"against a tolerance of {spec.tol:.3g}. The fit is then not C-TMLE0. Raise "
            "max_iter, widen g_bounds, or pass a partner further from the mechanism."
        )
    epsilon = np.asarray(fluctuation.epsilon, dtype=float).copy()
    epsilon[k:] = epsilon[k:] / scale
    before = (
        np.zeros(k)
        if fluctuation.score_initial is None
        else np.asarray(fluctuation.score_initial)[k:]
    )
    return DerivativeUpdate(
        targeted=fluctuation.targeted,
        fluctuation=fluctuation,
        epsilon=epsilon,
        names=submodel.names,
        scale=scale,
        score_before=before,
        score_after=np.asarray(fluctuation.score)[k:],
    )
