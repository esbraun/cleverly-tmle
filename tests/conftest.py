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


class OracleOutcomeContinuous(BaseEstimator):
    """The true conditional mean for a *continuous* outcome, on the scaled scale.

    The estimator maps ``Y`` onto ``[0, 1]`` before fitting ``Qbar``, so an oracle for a
    continuous outcome cannot simply return the structural mean -- it has to apply the same
    affine map, and it does not know the map in advance because the scaler is derived from
    the observed outcome range.  Recovering it by regressing the scaled outcome the
    estimator hands over on the raw structural mean is exact: both are affine images of the
    same quantity, so the fit is a line through the points rather than an approximation.

    :class:`OracleOutcome` is the binary counterpart, where the scaler is the identity and
    none of this is needed.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleOutcomeContinuous:
        design = np.asarray(X, dtype=float)
        raw = self._raw_mean(design)
        keep = np.isfinite(y)
        slope, intercept = np.polyfit(raw[keep], np.asarray(y)[keep], 1)
        self._slope, self._intercept = float(slope), float(intercept)
        return self

    def _raw_mean(self, design: Any) -> Any:
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.outcome_mean(w, 1.0, None), dtype=float)
        zero = np.asarray(self.dgp.outcome_mean(w, 0.0, None), dtype=float)
        return np.where(a == 1.0, one, zero)

    def predict(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        return np.clip(self._intercept + self._slope * self._raw_mean(design), 1e-9, 1 - 1e-9)


class OracleMissingness(BaseEstimator):
    """A missingness model returning the true ``P(Delta = 1 | A, W)``.

    Its design matrix is ``[A, W]``, not ``W`` -- the mechanism is allowed to depend on
    treatment, and the estimator predicts it at both arms -- so this follows
    :class:`OracleOutcome`'s convention of reading the arm out of the first column, not
    :class:`OracleTreatment`'s.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleMissingness:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.missingness(w, 1.0), dtype=float)
        zero = np.asarray(self.dgp.missingness(w, 0.0), dtype=float)
        p = np.clip(np.where(a == 1.0, one, zero), 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p, p])


class OracleIntermediate(BaseEstimator):
    """An intermediate model returning the true ``P(Z = 1 | A, W)``.

    Fitted on :meth:`~cleverly.data.causal_data.CausalData.treatment_design` -- ``[A, W]``
    -- and predicted at both arms, so it follows :class:`OracleMissingness`'s convention of
    reading the arm out of the first column rather than :class:`OracleTreatment`'s.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleIntermediate:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        one = np.asarray(self.dgp.intermediate_mean(w, 1.0), dtype=float)
        zero = np.asarray(self.dgp.intermediate_mean(w, 0.0), dtype=float)
        p = np.clip(np.where(a == 1.0, one, zero), 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - p, p])


class OracleDirectOutcome(BaseEstimator):
    """An outcome model returning the true ``E[Y | A, Z, W]`` for a direct-effect fit.

    A controlled-direct-effect fit trains the outcome model on ``[A, W, Z]`` and predicts
    it at ``[a, W, z]`` for a *fixed* level ``z``, so the design carries the intermediate
    in its last column -- which is why :class:`OracleOutcome`, which reads everything after
    the arm as covariates, cannot be reused here.  Reading ``z`` per row rather than from a
    stored level is deliberate: the same object serves the observed design and both
    counterfactual ones.

    Only valid for a binary outcome, for the reason :class:`OracleOutcome` gives.
    """

    def __init__(self, dgp: Any) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleDirectOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w, z = design[:, 0], design[:, 1:-1], design[:, -1]
        values = np.empty(design.shape[0], dtype=float)
        for arm in (0.0, 1.0):
            for level in (0.0, 1.0):
                rows = (a == arm) & (z == level)
                if not rows.any():
                    continue
                values[rows] = np.asarray(self.dgp.outcome_mean(w[rows], arm, level), dtype=float)
        return np.clip(values, 1e-9, 1.0 - 1e-9)

    def predict_proba(self, X: Any) -> Any:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> Any:
        return self._mean(X)


def aipw_ate(
    y: Any,
    a: Any,
    propensity: Any,
    q_one: Any,
    q_zero: Any,
    weights: Any = None,
    *,
    delta: Any = None,
    missingness: Any = None,
) -> float:
    """The augmented IPW (one-step) ATE, computed independently of the estimator.

    A second implementation of the same estimating equation, written out longhand:
    with the same nuisance inputs, TMLE and AIPW solve the identical efficient score
    equation and must agree up to the second-order difference between a substitution
    estimator and a one-step correction.

    ``delta`` and ``missingness`` extend that cross-check to missing outcomes.  Pass the
    observed-outcome indicator and an ``(n, 2)`` array of ``P(Delta = 1 | A = a, W)``:
    the indicator multiplies the residual term and the arm's observation probability
    joins the propensity in its denominator.  ``y`` is then read only where ``delta`` is
    one, so it may be anything (zero, ``nan``) elsewhere.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    g = np.asarray(propensity, dtype=float)
    q1 = np.asarray(q_one, dtype=float)
    q0 = np.asarray(q_zero, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if delta is None:
        d, pi0, pi1 = np.ones_like(y), np.ones_like(y), np.ones_like(y)
    else:
        d = np.asarray(delta, dtype=float).reshape(-1)
        pi = np.ones((y.shape[0], 2)) if missingness is None else np.asarray(missingness, float)
        pi0, pi1 = pi[:, 0], pi[:, 1]
    # Read Y only where it exists, so an unobserved NaN cannot propagate through the
    # multiply-by-zero that the Delta factor is.
    residual = np.where(d == 1.0, y, 0.0)
    contribution = (
        q1
        - q0
        + a * d / (g * pi1) * (residual - q1)
        - (1.0 - a) * d / ((1.0 - g) * pi0) * (residual - q0)
    )
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
