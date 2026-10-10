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

No registered study that ran before the change moves, for two structural reasons.  No
committed ``properties.csv`` has a failed replication, and no committed summary records a
failure, so no committed run reached the branch that raised.  And no registered study puts this
learner inside a ``SuperLearner`` library or a bootstrap refit, the two places where a raise
would have been swallowed rather than failing the run.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.special import expit, xlogy
from sklearn.base import BaseEstimator

#: R's ``glm.control(epsilon=)``: the relative deviance change at which ``glm`` stops.
DEVIANCE_EPSILON = 1e-8


class QuasiBinomialSeparationWarning(RuntimeWarning):
    """The coefficients diverged while the deviance converged, so the last iterate is kept."""


def deviance(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    """The weighted binomial deviance of a fractional target, with ``0 log 0 = 0``."""
    terms = xlogy(target, target) - xlogy(target, fitted)
    terms += xlogy(1.0 - target, 1.0 - target) - xlogy(1.0 - target, 1.0 - fitted)
    return float(2.0 * np.sum(weights * terms))


class QuasiBinomialGLM(BaseEstimator):
    """Small scikit-compatible unpenalized quasibinomial IRLS learner.

    R ``ltmle`` uses ``glm(..., family=quasibinomial())`` for both the binary final
    outcome and fractional earlier pseudo-outcomes.  Scikit-learn's logistic classifier
    refuses fractional targets, so the canonical study carries the corresponding score
    solver rather than silently comparing different regression families.
    """

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
            before = deviance(target, expit(design @ previous), weights)
            after = deviance(target, expit(design @ coefficient), weights)
            if abs(after - before) / (abs(after) + 0.1) >= DEVIANCE_EPSILON:
                raise RuntimeError("quasibinomial IRLS did not converge")
            warnings.warn(
                f"quasibinomial IRLS coefficients did not converge in {self.max_iter} steps, "
                "but the deviance did: quasi-complete separation, last iterate kept",
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
