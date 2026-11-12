r"""One pooled logistic fluctuation of an ATE fit's outcome regression along declared columns.

The targeting curves (:mod:`cleverly.targeting_curves`) and the assessment of external scores
(:func:`cleverly.effect_curves.score_assessment`) report substitution estimates of functionals
:math:`E[c_t(W)\bar Q_t + c_r(W)\bar Q_r]` with known numerators :math:`(c_t, c_r)`.  Both
fluctuate the cross-fitted :math:`\bar Q^0` of one ATE fit once, over every row, along the clever
covariate

.. math::

    H(A, W) = \sum_{a \in \{t, r\}} \frac{\mathbb 1\{A = a\}\, B_a(W)}{g(a \mid W)\,\Pi(a, W)},

where :math:`(B_t, B_r)` is an orthonormal basis of the span of the numerator columns.  The score
equations then hold for every column of the span, so a functional whose numerators lie in it is
targeted by the one solve.  The mechanism is the fit's truncated :math:`g`, and the missingness
mechanism the fit's bounded :math:`\Pi`, the ones its AIPW scores read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from ._typing import FloatArray
from .exceptions import DataError

__all__ = [
    "TARGETING_SPAN_RTOL",
    "PooledFit",
    "arm_span_basis",
    "clever_propensity",
    "solve_pooled",
    "span_basis",
]

#: The relative singular-value tolerance of the span basis the fluctuation runs along.
TARGETING_SPAN_RTOL: Final = 1e-10

#: A fluctuation whose Newton Hessian has a larger condition number is barely identified.
_SINGULAR_HESSIAN: Final = 1e12


def span_basis(columns: FloatArray, rank: int | None = None) -> tuple[FloatArray, float]:
    """An orthonormal basis of the column span, scaled to ``B'B = n I``, and the residual.

    Parameters
    ----------
    columns : ndarray
        ``(n, p)`` columns.
    rank : int or None
        Keep at most this many leading directions.

    Returns
    -------
    tuple
        The ``(n, k)`` basis, and the largest relative norm of a column's component outside
        it.
    """
    n = columns.shape[0]
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    keep = singular > TARGETING_SPAN_RTOL * singular[0]
    if rank is not None:
        keep &= np.arange(singular.size) < rank
    basis = u[:, keep] * math.sqrt(n)
    residual = columns - basis @ (basis.T @ columns) / n
    norms = np.linalg.norm(columns, axis=0)
    relative = np.linalg.norm(residual, axis=0) / np.where(norms > 0.0, norms, 1.0)
    return basis, float(np.max(relative)) if relative.size else 0.0


def arm_span_basis(
    numerators_t: FloatArray, numerators_r: FloatArray, rank: int | None = None
) -> tuple[FloatArray, FloatArray, float]:
    """A basis of the span of arm-specific numerator pairs, split back into its two arms.

    A column :math:`j` is the pair :math:`(c_{t,j}, c_{r,j})`.  The span is taken of the stacked
    ``(2n, p)`` matrix :math:`[c_t; c_r]` with :func:`span_basis`, so two pairs are collinear
    only when both arms are.

    Parameters
    ----------
    numerators_t : ndarray
        ``(n, p)`` numerators at arm ``t``.
    numerators_r : ndarray
        ``(n, p)`` numerators at arm ``r``.
    rank : int or None
        Keep at most this many leading directions.

    Returns
    -------
    tuple
        The basis at arm ``t``, the basis at arm ``r`` (each ``(n, k)``), and the largest
        relative norm of a column's component outside the span.
    """
    n = numerators_t.shape[0]
    basis, out_of_span = span_basis(np.vstack([numerators_t, numerators_r]), rank)
    return basis[:n], basis[n:], out_of_span


def clever_propensity(nuisance: Any, g_bounds: tuple[float, float]) -> FloatArray:
    """The truncated mechanism the clever covariate divides by: the scores' own ``g``.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The ATE fit's nuisances.
    g_bounds : tuple of float
        The fit's truncation of the treatment mechanism.

    Returns
    -------
    ndarray
        ``(n, K)`` truncated mechanism.
    """
    return np.asarray(nuisance.propensity.bounded(tuple(g_bounds)), dtype=float)


def _arm_matrix(values: Any, n: int, k: int) -> FloatArray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return np.repeat(array[:, None], k, axis=1)
    return array


@dataclass(frozen=True)
class PooledFit:
    r"""The targeted arm regressions of one pooled fluctuation, and the initial ones.

    Every array is on the outcome's own scale, one value per row.

    Parameters
    ----------
    q_t : ndarray
        :math:`\bar Q^*_t`.
    q_r : ndarray
        :math:`\bar Q^*_r`.
    weight_t : ndarray
        :math:`\Delta\,\mathbb 1\{A = t\} / (g(t \mid W)\,\Pi(t, W))`, from the truncated
        mechanism.
    weight_r : ndarray
        The same at arm ``r``.
    y : ndarray
        The outcome, 0 where it is missing.
    initial_t : ndarray
        :math:`\bar Q^0_t`.
    initial_r : ndarray
        :math:`\bar Q^0_r`.
    d_t : ndarray
        The initial AIPW pseudo-outcome at arm ``t``.
    d_r : ndarray
        The initial AIPW pseudo-outcome at arm ``r``.
    score_norm : float
        The norm of the solved score equations.
    hessian_condition : float
        The condition number of the Newton Hessian, NaN when the solver does not report it.
    """

    q_t: FloatArray
    q_r: FloatArray
    weight_t: FloatArray
    weight_r: FloatArray
    y: FloatArray
    initial_t: FloatArray
    initial_r: FloatArray
    d_t: FloatArray
    d_r: FloatArray
    score_norm: float
    hessian_condition: float

    @property
    def targeted_t(self) -> FloatArray:
        r""":math:`D^*_t = \bar Q^*_t + w_t (Y - \bar Q^*_t)`, the targeted AIPW score."""
        return self.q_t + self.weight_t * (self.y - self.q_t)

    @property
    def targeted_r(self) -> FloatArray:
        r""":math:`D^*_r`, the AIPW score of arm ``r`` at the targeted fit."""
        return self.q_r + self.weight_r * (self.y - self.q_r)


def solve_pooled(
    fit: Any,
    data: Any,
    numerators_t: FloatArray,
    numerators_r: FloatArray,
    pair: tuple[float, float],
    *,
    what: str,
) -> PooledFit:
    """Fluctuate the fit's outcome regression once along the given clever-covariate numerators.

    Parameters
    ----------
    fit : TMLEResult
        The single ATE fit on these rows; its nuisances, bounds and targeting settings are read.
    data : CausalData
        The rows, as the fit prepared them.
    numerators_t : ndarray
        ``(n, k)`` numerators at arm ``t``: a basis, so the columns are linearly independent.
    numerators_r : ndarray
        ``(n, k)`` numerators at arm ``r``.
    pair : tuple of float
        The arm codes ``(t, r)``.
    what : str
        The result being fitted, as the error messages name it.

    Returns
    -------
    PooledFit
        The targeted and the initial arm regressions.

    Raises
    ------
    DataError
        When the fluctuation does not converge or its Hessian is near-singular.
    """
    from .effect_curves import _pseudo_outcomes
    from .estimators.targeting import solve_submodel
    from .fluctuation.submodel import effect_modification_submodel

    nuisance = fit.nuisance
    spec = fit.config.targeting_spec
    g_bounds = (float(fit.config.g_bounds[0]), float(fit.config.g_bounds[1]))
    lower = float(fit.config.missingness_bound)
    mass = np.asarray(data.weights, dtype=float)
    n = int(mass.size)
    arms = tuple(nuisance.arms)
    t, r = pair
    numerators = np.zeros((n, len(arms), numerators_t.shape[1]))
    numerators[:, arms.index(t), :] = numerators_t
    numerators[:, arms.index(r), :] = numerators_r
    propensity = clever_propensity(nuisance, g_bounds)
    missingness = _arm_matrix(nuisance.bounded_missingness(lower), n, len(arms))
    submodel = effect_modification_submodel(
        np.asarray(data.treatment, dtype=float),
        propensity,
        arms=arms,
        missingness=missingness,
        msm=numerators,
    )
    scaler = nuisance.scaler
    fluctuation = solve_submodel(
        scaler.scale(np.asarray(data.outcome, dtype=float)),
        nuisance.outcome,
        submodel,
        mass,
        np.asarray(data.observed, dtype=bool),
        spec,
        warn=False,
    )
    if not fluctuation.converged or fluctuation.failure is not None:
        code = fluctuation.failure or "max_iter_reached"
        raise DataError(
            f"the targeting step of {what} did not converge ({code}): "
            f"{fluctuation.describe_failure()}. Widen g_bounds, or check the overlap of the "
            "treatment arms in the evaluation rows."
        )
    condition = float(getattr(fluctuation, "hessian_condition", float("nan")))
    if math.isfinite(condition) and condition > _SINGULAR_HESSIAN:
        raise DataError(
            f"the targeting step of {what} did not converge (singular_hessian): "
            "the Newton Hessian was singular or near-singular, so the fluctuation is barely "
            "identified; usually two clever-covariate columns that are nearly collinear."
        )
    targeted = fluctuation.targeted
    q_t = scaler.unscale_levels(np.asarray(targeted.arms[t], dtype=float))
    q_r = scaler.unscale_levels(np.asarray(targeted.arms[r], dtype=float))
    initial_t, initial_r, d_t, d_r = _pseudo_outcomes(
        nuisance, data, pair, g_bounds=g_bounds, missingness_bound=lower
    )
    # The targeted scores read the scores' own truncated mechanism (not the clever
    # covariate's helper), so a covariate built from another mechanism breaks the identity.
    scores_g = np.asarray(nuisance.propensity.bounded(g_bounds), dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    treatment = np.asarray(data.treatment, dtype=float)
    y = np.where(observed, np.asarray(data.outcome, dtype=float), 0.0)
    pi_t: Any = 1.0 if missingness is None else missingness[:, arms.index(t)]
    pi_r: Any = 1.0 if missingness is None else missingness[:, arms.index(r)]
    weight_t = observed * (treatment == t) / (scores_g[:, arms.index(t)] * pi_t)
    weight_r = observed * (treatment == r) / (scores_g[:, arms.index(r)] * pi_r)
    return PooledFit(
        q_t=q_t,
        q_r=q_r,
        weight_t=weight_t,
        weight_r=weight_r,
        y=y,
        initial_t=initial_t,
        initial_r=initial_r,
        d_t=d_t,
        d_r=d_r,
        score_norm=float(fluctuation.score_norm),
        hessian_condition=condition,
    )
