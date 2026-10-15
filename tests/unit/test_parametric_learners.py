"""The parametric outcome and mechanism learners of the per-arm OAT stack."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from cleverly.exceptions import DataError
from cleverly.learners import (
    ParametricFitDiagnostics,
    ParametricMechanism,
    ParametricOutcome,
)
from tests.studies import oat_per_arm_laws as laws


def _logistic_sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    design = np.column_stack([rng.integers(0, 2, n), rng.normal(size=(n, 2))])
    eta = -0.2 + 0.5 * design[:, 0] + design[:, 1] - 0.7 * design[:, 0] * design[:, 2]
    return design, rng.binomial(1, expit(eta))


def test_outcome_feature_maps() -> None:
    design = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]])
    main = ParametricOutcome("main_terms").design_matrix(design)
    full = ParametricOutcome("arm_interactions").design_matrix(design)
    reduced = ParametricOutcome("arm_interactions_no_intercept").design_matrix(design)
    np.testing.assert_array_equal(main, [[1, 1, 2, 3], [1, 0, 4, 5]])
    np.testing.assert_array_equal(full, [[1, 1, 2, 3, 2, 3], [1, 0, 4, 5, 0, 0]])
    np.testing.assert_array_equal(reduced, [[1, 2, 3, 2, 3], [0, 4, 5, 0, 0]])
    np.testing.assert_array_equal(full[:, 1:], laws.arm_interactions(design, indicators=1))


def test_outcome_features_outside_the_vocabulary_are_refused() -> None:
    design, y = _logistic_sample(50, 0)
    with pytest.raises(ValueError, match="features must be one of"):
        ParametricOutcome("splines").fit(design, y)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="treatment column"):
        ParametricOutcome(arm_columns=3).fit(design, y)


def test_outcome_fit_matches_unpenalized_sklearn() -> None:
    design, y = _logistic_sample(3000, 1)
    model = ParametricOutcome().fit(design, y)
    reference = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12).fit(
        laws.arm_interactions(design, indicators=1), y
    )
    np.testing.assert_allclose(model.coef_[0], reference.intercept_[0], atol=1e-8)
    np.testing.assert_allclose(model.coef_[1:], reference.coef_[0], atol=1e-8)
    np.testing.assert_allclose(
        model.predict_proba(design),
        reference.predict_proba(laws.arm_interactions(design, indicators=1)),
        atol=1e-10,
    )
    np.testing.assert_array_equal(model.classes_, [0, 1])
    assert model.diagnostics_.converged
    assert model.diagnostics_.score_norm < 1e-12
    np.testing.assert_array_equal(model.diagnostics_.coef, model.coef_)


def test_fractional_target_solves_the_quasi_binomial_score() -> None:
    design, _ = _logistic_sample(2000, 2)
    rng = np.random.default_rng(3)
    y = np.asarray(rng.beta(2.0, 3.0, size=design.shape[0]))
    model = ParametricOutcome().fit(design, y)
    z = model.design_matrix(design)
    score = z.T @ (y - model.predict(design)) / y.size
    assert np.max(np.abs(score)) < 1e-12


def test_integer_weights_equal_row_duplication() -> None:
    design, y = _logistic_sample(400, 4)
    counts = np.random.default_rng(5).integers(0, 3, size=y.size)
    weighted = ParametricOutcome().fit(design, y, sample_weight=counts.astype(float))
    duplicated = ParametricOutcome().fit(np.repeat(design, counts, axis=0), np.repeat(y, counts))
    np.testing.assert_allclose(weighted.coef_, duplicated.coef_, atol=1e-9)


def test_column_scale_does_not_change_the_fit() -> None:
    design, y = _logistic_sample(800, 6)
    scaled = design.copy()
    scaled[:, 1] *= 1e4
    base = ParametricOutcome("main_terms").fit(design, y)
    wide = ParametricOutcome("main_terms").fit(scaled, y)
    np.testing.assert_allclose(wide.predict(scaled), base.predict(design), atol=1e-10)
    assert wide.diagnostics_.reciprocal_condition > 1e-3


def test_separation_is_refused() -> None:
    x = np.column_stack([np.zeros(40), np.linspace(-1.0, 1.0, 40)])
    y = (x[:, 1] > 0).astype(int)
    with pytest.raises(DataError, match="separation"):
        ParametricOutcome("main_terms").fit(x, y)


def test_collinear_features_are_refused() -> None:
    design, y = _logistic_sample(200, 7)
    design[:, 2] = 2.0 * design[:, 1]
    with pytest.raises(DataError, match="singular"):
        ParametricOutcome("main_terms").fit(design, y)


def test_mechanism_basis_and_derivative() -> None:
    q = np.array([1e-8, 0.2, 0.5, 0.9, 1.0 - 1e-8])
    mechanism = ParametricMechanism(degree=3, clip=1e-6)
    level = logit(np.clip(q, 1e-6, 1.0 - 1e-6))
    np.testing.assert_allclose(
        mechanism.design_matrix(q[:, None]),
        np.column_stack([np.ones(5), level, level**2, level**3]),
    )
    derivative = mechanism.derivative(q[:, None])
    np.testing.assert_array_equal(derivative[[0, 4]], 0.0)
    inside = np.array([1, 2, 3])
    step = 1e-6
    upper = mechanism.design_matrix(expit(level[inside] + step))
    lower = mechanism.design_matrix(expit(level[inside] - step))
    np.testing.assert_allclose(
        derivative[inside], (upper - lower) / (2 * step), rtol=1e-7, atol=1e-9
    )


def test_mechanism_matches_the_study_pipeline_basis() -> None:
    q = np.random.default_rng(8).uniform(size=50)
    np.testing.assert_allclose(
        ParametricMechanism().design_matrix(q)[:, 1:], laws.logit_basis(q), rtol=0, atol=0
    )


def test_mechanism_reads_one_column() -> None:
    with pytest.raises(ValueError, match="one column"):
        ParametricMechanism().fit(np.ones((10, 2)), np.arange(10) % 2)
    with pytest.raises(ValueError, match="degree"):
        ParametricMechanism(degree=0).design_matrix(np.full(3, 0.5))


def test_learners_clone_and_pickle_by_value() -> None:
    design, y = _logistic_sample(300, 9)
    model = ParametricOutcome("main_terms").fit(design, y)
    copy = clone(model)
    assert copy.get_params() == {"features": "main_terms", "arm_columns": 1}
    assert not hasattr(copy, "coef_")
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(design), model.predict(design))
    assert restored.diagnostics_ == model.diagnostics_
    assert clone(ParametricMechanism(degree=2)).get_params() == {"degree": 2, "clip": 1e-6}


def _per_arm(frame: object, outcome: object, treatment: object) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(
                strategy="oat",
                outcome_learner=outcome,
                treatment_learner=treatment,
                cross_fit=False,
                simultaneous=False,
                estimands=("ey0", "ey1", "ate"),
                g_bounds=laws.G_BOUNDS,
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )


@pytest.mark.parametrize("seed", [11, 12])
def test_point_parity_with_the_study_pipelines(seed: int) -> None:
    """W7: the parametric learners reproduce the study pipelines' estimates."""
    frame, _ = laws.BINARY_ACTIVE.sample(2000, seed)
    pipeline = _per_arm(frame, laws.interaction_outcome(2), laws.cubic_logit_learner())
    parametric = _per_arm(frame, ParametricOutcome(), ParametricMechanism())
    for name in ("ey0", "ey1", "ate"):
        assert abs(parametric[name].psi - pipeline[name].psi) < 1e-8  # type: ignore[index]
    diagnostics = parametric.nuisance.diagnostics  # type: ignore[attr-defined]
    (outcome,) = diagnostics["outcome"]
    assert isinstance(outcome, ParametricFitDiagnostics)
    assert len(outcome.coef) == 8
    arms = diagnostics["propensity"]
    assert [len(entries) for entries in arms] == [1, 1]
    assert all(isinstance(entry[0], ParametricFitDiagnostics) for entry in arms)


def test_a_mutated_basis_breaks_point_parity() -> None:
    """W7 control: a mechanism basis unlike the pipeline's moves the estimate."""
    frame, _ = laws.BINARY_ACTIVE.sample(2000, 11)
    pipeline = _per_arm(frame, laws.interaction_outcome(2), laws.cubic_logit_learner())
    mutated = _per_arm(frame, ParametricOutcome(), ParametricMechanism(degree=2))
    assert abs(mutated["ate"].psi - pipeline["ate"].psi) > 1e-6  # type: ignore[index]
