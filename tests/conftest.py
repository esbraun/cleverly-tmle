"""Shared fixtures and helpers.

The fast tier keeps runtime down by using parametric nuisance learners
(``library="glm"``) wherever the test is about the estimator's machinery rather than
about the Super Learner.  Tests that specifically exercise flexible learning say so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly import TMLE
from cleverly.datasets import (
    binary_outcome_dgp,
    linear_dgp,
    make_linear_ate,
    make_nonlinear_ate,
    nonlinear_dgp,
)

#: Estimator settings for the fast tier: parametric nuisances, few folds, seeded.
FAST_KWARGS: dict[str, Any] = {
    "outcome_learner": "glm",
    "treatment_learner": "glm",
    "n_folds": 5,
    "learner_folds": 3,
    "random_state": 0,
    "simultaneous": False,
}


def fast_tmle(**overrides: Any) -> TMLE:
    """A quick, reproducible estimator for tests."""
    return TMLE(**{**FAST_KWARGS, **overrides})


class OracleTreatment(BaseEstimator):
    """A treatment model that returns the data-generating propensity exactly.

    Used to isolate the estimator from nuisance-estimation error: with the truth
    plugged in, any remaining discrepancy is the estimator's own.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleTreatment:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        p = np.clip(np.asarray(self.dgp.propensity(np.asarray(X, dtype=float))), 1e-9, 1 - 1e-9)
        return np.column_stack([1.0 - p, p])


class OracleOutcome(BaseEstimator):
    """An outcome model returning the true conditional mean given ``[A, W]``.

    Only valid for a binary outcome, where the estimator does not rescale ``Y`` and the
    true conditional mean is directly on the ``[0, 1]`` scale the fluctuation uses.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.outcome_mean(w, 1.0, None), dtype=float)
        zero = np.asarray(self.dgp.outcome_mean(w, 0.0, None), dtype=float)
        return np.clip(np.where(a == 1.0, one, zero), 1e-9, 1.0 - 1e-9)

    def predict_proba(self, X: Any) -> Any:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> Any:
        return self._mean(X)


def aipw_ate(
    y: Any, a: Any, propensity: Any, q_one: Any, q_zero: Any, weights: Any = None
) -> float:
    """The augmented IPW (one-step) ATE, computed independently of the estimator.

    A second implementation of the same estimating equation, written out longhand:
    with the same nuisance inputs, TMLE and AIPW solve the identical efficient score
    equation and must agree up to the second-order difference between a substitution
    estimator and a one-step correction.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    g = np.asarray(propensity, dtype=float)
    q1 = np.asarray(q_one, dtype=float)
    q0 = np.asarray(q_zero, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    contribution = q1 - q0 + a / g * (y - q1) - (1.0 - a) / (1.0 - g) * (y - q0)
    return float(np.average(contribution, weights=w))


@pytest.fixture
def linear_frame() -> tuple[Any, dict[str, float]]:
    """A moderate linear-DGP sample; both nuisance models are correctly specified."""
    return make_linear_ate(n=800, seed=101)


@pytest.fixture
def nonlinear_frame() -> tuple[Any, dict[str, float]]:
    """A nonlinear sample where a GLM is misspecified for both nuisances."""
    return make_nonlinear_ate(n=800, seed=102)


@pytest.fixture
def linear_process() -> Any:
    return linear_dgp()


@pytest.fixture
def nonlinear_process() -> Any:
    return nonlinear_dgp()


@pytest.fixture
def binary_process() -> Any:
    return binary_outcome_dgp()
