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
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator


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
        for _ in range(self.max_iter):
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
            raise RuntimeError("quasibinomial IRLS did not converge")
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
