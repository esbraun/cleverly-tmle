r"""The population standard deviation of an outcome that is missing at random.

The standardized E-value divides a mean difference by the outcome's standard deviation.
Under missing at random given :math:`(A, W)`, the respondents' standard deviation estimates
another quantity, so this module writes the population one as two natural-course means:

.. math::

    \hat E[Y^k] = E_n\bigl[\Delta Y^k + (1 - \Delta)\, \bar Q_k(A, W)\bigr],
    \qquad \bar Q_k(A, W) = E[Y^k \mid A, W, \Delta = 1],

for :math:`k = 1, 2`, and :math:`\widehat{\mathrm{sd}} = \{\hat E[Y^2] - \hat E[Y]^2\}^{1/2}`.
:math:`\bar Q_1` is the fit's own initial outcome regression.  :math:`\bar Q_2` is one more
regression of :math:`Y^2` on the same design, fitted with the fit's outcome learner on the
fit's folds.  Both moments are untargeted plug-ins.  The E-value conversion reads the
standard deviation as a constant and ignores its sampling error, as the complete-outcome
standardized branch does, so a targeted moment would add no claim.

The variance carries the reliability-weight correction of the complete-outcome branch,
:math:`\sum w / (\sum w - \sum w^2 / \sum w)`, which is :math:`n / (n - 1)` when every weight is
one.  So with every outcome observed the two branches give the same standard deviation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._refusal import RefusalReason
from .._typing import BoolArray
from ..exceptions import CapabilityError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators._nuisance import RepeatFit
    from ..estimators.base import TMLEResult

__all__ = ["missing_at_random_sd"]


def _second_moment_regression(result: TMLEResult, repeat: RepeatFit) -> np.ndarray:
    """``E[Y^2 | A, W, Delta = 1]`` at the observed design, out of fold, on the outcome scale.

    The target is scaled to ``[0, 1]`` by the respondents' range of ``Y^2``, because the
    fit's learner may clip its predictions to that interval, and the predictions are mapped
    back afterwards.
    """
    from ..estimators._nuisance import cross_fit_predictions

    data = result.data
    observed = np.asarray(data.observed, dtype=bool)
    squared = np.where(observed, np.asarray(data.outcome, dtype=float) ** 2, 0.0)
    low = float(np.min(squared[observed]))
    high = float(np.max(squared[observed]))
    span = high - low
    if span <= 0.0:
        return np.full(data.n, low)
    design = data.treatment_design(include_intermediate=data.has_intermediate)
    learner = result.estimator._resolve_learner(
        result.estimator.outcome_learner, task="regression", seed=repeat.seed
    )
    predictions, _ = cross_fit_predictions(
        learner,
        design,
        (squared - low) / span,
        np.asarray(data.weights, dtype=float),
        repeat.folds,
        task="regression",
        predict_designs={"observed": design},
        fit_mask=observed,
        groups=data.cluster,
        clip=(0.0, 1.0),
    )
    return low + span * np.asarray(predictions["observed"], dtype=float)


def _draw_sd(result: TMLEResult, repeat: RepeatFit, rows: BoolArray | None) -> float:
    """The plug-in standard deviation under one cross-fitting draw."""
    data = result.data
    observed = np.asarray(data.observed, dtype=bool)
    outcome = np.where(observed, np.asarray(data.outcome, dtype=float), 0.0)
    scaler = repeat.nuisance.scaler
    first = np.asarray(repeat.nuisance.outcome.observed, dtype=float)
    first = first if scaler.is_identity else scaler.unscale_levels(first)
    second = _second_moment_regression(result, repeat)
    weights = np.asarray(data.weights, dtype=float)
    if rows is not None:
        weights = weights * rows
    mean = float(np.average(np.where(observed, outcome, first), weights=weights))
    square = float(np.average(np.where(observed, outcome**2, second), weights=weights))
    total = float(np.sum(weights))
    correction = total - float(np.sum(weights**2)) / total
    if square <= mean**2 or correction <= 0.0:
        return 0.0
    return float(np.sqrt((square - mean**2) * total / correction))


def missing_at_random_sd(result: TMLEResult, rows: BoolArray | None = None) -> float:
    """Return the population outcome standard deviation under missing at random.

    Parameters
    ----------
    result : TMLEResult
        A fitted point result with a response mechanism.
    rows : ndarray of bool or None
        The rows of a baseline stratum, for a stratum-conditional contrast. ``None`` reads
        every row.

    Returns
    -------
    float
        The median over the fit's cross-fitting draws of the plug-in standard deviation.

    Raises
    ------
    CapabilityError
        If the plug-in variance is not positive, which names no scale to standardize by.
    """
    values = [_draw_sd(result, repeat, rows) for repeat in result.repeats]
    sd = float(np.median(values))
    if not np.isfinite(sd) or sd <= 0.0:
        raise CapabilityError(
            "cannot standardise the effect: the missing-at-random plug-in of the outcome "
            f"variance is not positive (standard deviation {sd:.4g})",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return sd
