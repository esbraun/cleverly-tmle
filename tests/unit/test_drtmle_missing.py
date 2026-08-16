"""The theorem-backed randomized missing-outcome DR-TMLE surface.

The canonical R implementation writes the mechanism correction with
``I(A=a, Delta=1)`` while its ordinary complete-data reduction is easy to read as
``I(A=a)``.  The nonzero array witness below is the acceptance evidence for that mask;
the end-to-end tests cover the public eligibility and known-randomization contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from cleverly import DRTMLE, load
from cleverly.estimators.reduced import ReducedSet
from cleverly.fluctuation.iterative import InitialFit
from cleverly.inference.influence import reduced_correction_parts


def _trial(n: int = 320, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    pi = 1.0 / (1.0 + np.exp(-(-0.1 + 0.4 * a + 0.3 * w1)))
    observed = rng.binomial(1, pi, size=n).astype(float)
    y = 0.8 + 1.1 * a + 0.4 * w1 - 0.2 * w2 + rng.normal(scale=0.6, size=n)
    y[observed == 0.0] = np.nan
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Delta": observed, "Y": y})


def _estimator(**settings: object) -> DRTMLE:
    return DRTMLE(
        randomized=True,
        cross_fit=False,
        outcome_learner="glm",
        treatment_learner="glm",
        missingness_learner="glm",
        reduced_outcome_learner="glm",
        reduced_treatment_learner="glm",
        estimands=("ate", "ey1", "ey0"),
        simultaneous=False,
        random_state=0,
        **settings,
    )


@pytest.fixture(scope="module")
def randomized_fit():
    return _estimator().fit(
        _trial(),
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        delta="Delta",
    ).single()


def test_randomized_missing_outcomes_solve_the_reported_equations(randomized_fit) -> None:
    assert randomized_fit.validation.score_check().passed
    assert randomized_fit.validation.correction_check().passed
    joint = randomized_fit.nuisance.reduction_mechanism
    assert joint is not None
    np.testing.assert_allclose(
        joint.values,
        randomized_fit.nuisance.propensity.values * randomized_fit.nuisance.missingness,
        rtol=0,
        atol=0,
    )


def test_the_joint_mechanism_survives_serialization(randomized_fit, tmp_path) -> None:
    path = tmp_path / "drtmle-missing.cleverly"
    randomized_fit.save(path)
    restored = load(path)
    assert restored.nuisance.reduction_mechanism is not None
    np.testing.assert_array_equal(
        restored.nuisance.reduction_mechanism.values,
        randomized_fit.nuisance.reduction_mechanism.values,
    )
    assert restored.validation.score_check() == randomized_fit.validation.score_check()


def test_the_mechanism_correction_uses_the_observation_mask() -> None:
    observed = np.array([True, False, True, False])
    treatment = np.array([0.0, 0.0, 1.0, 1.0])
    joint = np.array([[0.30, 0.40], [0.25, 0.35], [0.30, 0.40], [0.25, 0.35]])
    reduced = ReducedSet(
        qr=np.array([[0.2, -0.1], [0.3, -0.2], [0.4, -0.3], [0.5, -0.4]]),
        gr1=np.full((4, 2), 0.5),
        gr2=np.full((4, 2), 0.1),
        arms=(0.0, 1.0),
        g_bounds=(1e-6, 1 - 1e-6),
    )
    targeted = InitialFit(np.array([0.2, 0.3, 0.4, 0.5]), {0.0: np.zeros(4), 1.0: np.ones(4)})
    parts = reduced_correction_parts(
        np.array([0.1, 0.2, 0.3, 0.4]),
        targeted,
        treatment,
        reduced,
        joint,
        bounds=(1e-6, 1 - 1e-6),
        observed=observed,
        guard=("Q", "g"),
    )
    for column, arm in enumerate(reduced.arms):
        indicator = ((treatment == arm) & observed).astype(float)
        expected = reduced.qr[:, column] / joint[:, column] * (
            indicator - joint[:, column]
        )
        np.testing.assert_allclose(parts.d_g[arm], expected, rtol=0, atol=1e-15)
        wrong = reduced.qr[:, column] / joint[:, column] * (
            (treatment == arm).astype(float) - joint[:, column]
        )
        assert np.max(np.abs(parts.d_g[arm] - wrong)) > 0.1


def test_joint_mechanism_correction_is_diaz_theorem_decomposition() -> None:
    """D_M + D_A collapses to e/g {I(A=a, Delta=1) - g}."""
    treatment = np.array([0.0, 1.0, 1.0, 0.0])
    observed = np.array([1.0, 0.0, 1.0, 0.0])
    arm = 1.0
    indicator = (treatment == arm).astype(float)
    g_a = np.array([0.42, 0.57, 0.63, 0.48])
    g_delta = np.array([0.76, 0.69, 0.81, 0.72])
    e = np.array([0.17, -0.23, 0.31, -0.14])
    joint = g_a * g_delta

    collapsed = e / joint * (indicator * observed - joint)
    d_a = e / g_a * (indicator - g_a)
    d_m = indicator * e / joint * (observed - g_delta)

    np.testing.assert_allclose(collapsed, d_a + d_m, rtol=0, atol=2e-16)
    assert np.max(np.abs(collapsed)) > 0.1


class _FailIfFit(BaseEstimator):
    def fit(self, X, y, sample_weight=None):  # pragma: no cover - failure is the assertion
        raise AssertionError("known randomization probabilities must bypass treatment fitting")


def test_known_probabilities_bypass_the_treatment_learner() -> None:
    frame = _trial(n=260, seed=17)
    estimator = DRTMLE(
        randomized=False,
        cross_fit=False,
        outcome_learner="glm",
        treatment_learner=_FailIfFit(),
        missingness_learner="glm",
        reduced_outcome_learner="glm",
        reduced_treatment_learner="glm",
        estimands=("ate",),
        simultaneous=False,
    )
    result = estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        delta="Delta",
        treatment_probabilities=np.full(len(frame), 0.5),
    ).single()
    np.testing.assert_array_equal(result.nuisance.propensity.values, np.full((len(frame), 2), 0.5))
    assert result.validation.score_check().passed


def test_observational_missing_outcomes_are_refused() -> None:
    with pytest.raises(NotImplementedError, match="randomized trial"):
        DRTMLE(cross_fit=False, estimands=("ate",)).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


def test_cross_fitted_missing_outcomes_are_refused() -> None:
    with pytest.raises(NotImplementedError, match="cross-validated extension"):
        DRTMLE(randomized=True, estimands=("ate",)).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (np.full(99, 0.5), "99 rows"),
        (np.full((100, 3), 1 / 3), r"must be \(n,\)"),
        (np.zeros(100), "strictly between"),
    ],
)
def test_invalid_known_probability_shapes_are_refused(probabilities, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _estimator().fit(
            _trial(100),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=probabilities,
        )
