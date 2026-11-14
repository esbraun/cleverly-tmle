"""The quasibinomial learner every study with a fractional target fits.

A fractional outcome is one that lies in ``[0, 1]`` without being a class label: a
longitudinal pseudo-outcome carried back from a later node, or a proportion drawn from a
beta law.  Scikit-learn's logistic classifier refuses such a target, so the studies carry
the score solver instead of silently comparing a different regression family.

The class moved here from :mod:`tests.studies.canonical_ltmle`, unchanged, because the
bounded point-treatment laws in :mod:`tests.studies.bounded_cv_laws` need the same learner
and the longitudinal study is not where a point-treatment study should reach for it.  The
move is result-neutral: the bytes of :class:`QuasiBinomialGLM` are identical, so every
study that already fitted it fits the same numbers.  ``tests/canonical/provenance-revisions.md``
records that judgement for each study whose manifest predates the move.

A fit whose coefficients do not converge in ``max_iter`` steps used to raise.  It now returns
its last iterate with :class:`QuasiBinomialSeparationWarning` when the deviance has converged
by R's ``glm.control`` rule, as R's ``glm`` returns its fit with a warning.  That is the
quasi-complete separation case: a covariate cell with no event, whose coefficient diverges
while the fitted values and the deviance settle.  Every fit that converged before takes the
same iterates and stops at the same step, because only the branch that raised changed.  A fit
whose deviance has not settled still raises.

A fit whose binary target is completely separated also keeps its last iterate, with the same
warning.  There the maximum-likelihood estimate does not exist: the coefficients grow without
bound, the fitted values tend to 0 and 1 on the separated rows, and the deviance tends to zero
geometrically, so its relative change never meets R's rule.  The learner detects the case
directly, on the target and the fitted values rather than on the deviance: the target is binary
on the rows of positive weight, and every such row's fitted value is within
:data:`SEPARATION_TOLERANCE` of its observed class.  A fractional target never meets that test.
The last iterate approximates the limiting predictions, and a fit that converged before takes
the same iterates and stops at the same step.  The test reads the last iterate only, so a
separated design whose rows near the boundary are still farther than the tolerance from their
class after ``max_iter`` steps still raises.

Neither change moves a registered study that ran before it, for two structural reasons.  No
committed failure record carries the text the learner raised, so no committed run reached the
branch that raised.  And no registered study puts this learner inside a ``SuperLearner``
library or a bootstrap refit, the two places where a raise would have been swallowed rather
than failing the run.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Final

import numpy as np
from scipy.special import expit, xlogy
from sklearn.base import BaseEstimator

#: R's ``glm.control(epsilon=)``: the relative deviance change at which ``glm`` stops.
DEVIANCE_EPSILON = 1e-8

#: How close every positive-weight fitted value must be to its observed class, 0 or 1, for an
#: unconverged fit of a binary target to read as complete separation.
SEPARATION_TOLERANCE: Final = 1e-6


class QuasiBinomialSeparationWarning(RuntimeWarning):
    """The coefficients diverged under separation, so the last iterate is kept."""


def deviance(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    """The weighted binomial deviance of a fractional target, with ``0 log 0 = 0``."""
    terms = xlogy(target, target) - xlogy(target, fitted)
    terms += xlogy(1.0 - target, 1.0 - target) - xlogy(1.0 - target, 1.0 - fitted)
    return float(2.0 * np.sum(weights * terms))


def completely_separated(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> bool:
    """Whether a binary target is fitted to within :data:`SEPARATION_TOLERANCE` on every row.

    Only the rows of positive weight count.  A target with a fractional value on such a row is
    never separated, whatever the fitted values.
    """
    informative = weights > 0
    observed = target[informative]
    if not np.all((observed == 0.0) | (observed == 1.0)):
        return False
    return bool(np.all(np.abs(fitted[informative] - observed) <= SEPARATION_TOLERANCE))


class QuasiBinomialGLM(BaseEstimator):
    """Small scikit-compatible unpenalized quasibinomial IRLS learner.

    R ``ltmle`` uses ``glm(..., family=quasibinomial())`` for both the binary final
    outcome and fractional earlier pseudo-outcomes.  Scikit-learn's logistic classifier
    refuses fractional targets, so the canonical study carries the corresponding score
    solver rather than silently comparing different regression families.

    Under complete separation of a binary target the maximum-likelihood estimate does not
    exist, and the predictions tend to 0 and 1 on the separated rows.  The fit then keeps the
    last iterate, which approximates those limits, with
    :class:`QuasiBinomialSeparationWarning` (:func:`completely_separated`).
    """

    #: Unpenalized, and started from the weighted mean, so a row of integer weight ``c``
    #: fits as ``c`` copies of the row (:mod:`cleverly.longitudinal.frequency`).
    frequency_exact = True

    def __init__(self, *, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuasiBinomialGLM:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        design = np.column_stack([np.ones(len(matrix)), matrix])
        mean = float(np.average(target, weights=weights))
        coefficient = np.zeros(design.shape[1], dtype=float)
        coefficient[0] = math.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))
        previous = coefficient
        for _ in range(self.max_iter):
            previous = coefficient
            fitted = expit(design @ coefficient)
            variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
            working = design @ coefficient + (target - fitted) / variance
            root_weight = np.sqrt(weights * variance)
            updated = np.linalg.lstsq(
                design * root_weight[:, None], working * root_weight, rcond=None
            )[0]
            if np.max(np.abs(updated - coefficient)) <= self.tol:
                coefficient = updated
                break
            coefficient = updated
        else:
            last = expit(design @ coefficient)
            if completely_separated(target, last, weights):
                warnings.warn(
                    f"quasibinomial IRLS coefficients did not converge in {self.max_iter} "
                    "steps: complete separation, last iterate kept",
                    QuasiBinomialSeparationWarning,
                    stacklevel=2,
                )
            else:
                before = deviance(target, expit(design @ previous), weights)
                after = deviance(target, last, weights)
                if abs(after - before) / (abs(after) + 0.1) >= DEVIANCE_EPSILON:
                    raise RuntimeError("quasibinomial IRLS did not converge")
                warnings.warn(
                    f"quasibinomial IRLS coefficients did not converge in {self.max_iter} "
                    "steps, but the deviance did: quasi-complete separation, last iterate kept",
                    QuasiBinomialSeparationWarning,
                    stacklevel=2,
                )
        self.coef_ = coefficient[1:][None, :]
        self.intercept_ = coefficient[:1]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return expit(self.intercept_[0] + matrix @ self.coef_[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = self.predict(X)
        return np.column_stack([1.0 - probability, probability])
