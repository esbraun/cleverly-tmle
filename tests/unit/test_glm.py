"""The parametric GLM learner: its fit, its aliasing and its estimating-equation pieces."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import DataError
from cleverly.learners import GLM, fit_learner, predict_mean
from cleverly.learners._fitting import predict_probabilities


def _law(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    weights = rng.uniform(0.5, 2.0, n)
    probability = 1.0 / (1.0 + np.exp(-(0.3 + x @ np.array([0.5, -0.4, 0.2]))))
    return x, (rng.uniform(size=n) < probability).astype(int), weights


def _unpenalized(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    return LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000).fit(x, y, weights)


def test_binomial_equals_the_unpenalized_logistic_mle() -> None:
    x, y, weights = _law()
    model = GLM().fit(x, y, weights)
    reference = _unpenalized(x, y, weights)
    assert model.family_ == "binomial"
    expected = np.r_[reference.intercept_, reference.coef_[0]]
    np.testing.assert_allclose(model.coef_, expected, atol=1e-6)
    np.testing.assert_allclose(predict_mean(model, x, "classification"), model.predict(x))


def test_multinomial_equals_the_unpenalized_softmax_mle() -> None:
    x, _, weights = _law()
    labels = np.random.default_rng(1).integers(0, 3, x.shape[0])
    model = GLM().fit(x, labels, weights)
    reference = _unpenalized(x, labels, weights)
    assert model.family_ == "multinomial"
    np.testing.assert_allclose(model.predict_proba(x), reference.predict_proba(x), atol=1e-6)
    np.testing.assert_allclose(
        predict_probabilities(model, x, [0.0, 1.0, 2.0]), model.predict_proba(x)
    )


def test_fractional_target_is_quasi_binomial_and_solves_its_score() -> None:
    x, _, weights = _law()
    target = np.random.default_rng(2).uniform(0.1, 0.9, x.shape[0])
    model = GLM().fit(x, target, weights)
    assert model.family_ == "binomial"
    score = model.score_contributions(x, target, weights).sum(axis=0)
    assert np.max(np.abs(score)) < 1e-8


def test_gaussian_is_weighted_least_squares() -> None:
    x, _, weights = _law()
    target = x @ np.array([1.0, 2.0, 3.0]) + 5.0 + np.random.default_rng(3).normal(size=x.shape[0])
    model = GLM().fit(x, target, weights)
    assert model.family_ == "gaussian"
    root = np.sqrt(weights)
    expected, *_ = np.linalg.lstsq(model.design(x) * root[:, None], target * root, rcond=None)
    np.testing.assert_allclose(model.coef_, expected, rtol=1e-10)


@pytest.mark.parametrize("labels", [False, True])
def test_jacobian_and_information_match_finite_differences(labels: bool) -> None:
    x, y, weights = _law()
    target = np.random.default_rng(1).integers(0, 3, x.shape[0]) if labels else y
    model = GLM().fit(x, target, weights)
    coef = model.coef_.copy()
    jacobian = model.mean_jacobian(x)
    information = model.information(x, weights)
    step = 1e-6
    for k in range(model.n_parameters):
        model.coef_ = coef.copy()
        base_mean = model.predict_proba(x)
        base_score = model.score_contributions(x, target, weights).sum(axis=0)
        moved = coef.copy()
        moved.flat[k] += step
        model.coef_ = moved
        slope = (model.predict_proba(x) - base_mean) / step
        score_slope = (
            model.score_contributions(x, target, weights).sum(axis=0) - base_score
        ) / step
        model.coef_ = coef
        analytic = jacobian[:, :, k] if labels else jacobian[:, k]
        numeric = slope if labels else slope[:, 1]
        np.testing.assert_allclose(numeric, analytic, atol=1e-5)
        np.testing.assert_allclose(-score_slope, information[:, k], rtol=1e-4, atol=1e-4)


def test_aliased_columns_leave_the_design() -> None:
    x, y, _ = _law()
    design = np.column_stack([x, np.ones(x.shape[0]), 2.0 * x[:, 0]])
    model = GLM().fit(design, y)
    np.testing.assert_array_equal(model.active_, [True, True, True, True, False, False])
    np.testing.assert_allclose(model.predict(design), GLM().fit(x, y).predict(x), atol=1e-10)


def test_aliasing_reads_only_rows_with_positive_weight() -> None:
    x, y, _ = _law()
    weights = np.ones(x.shape[0])
    constant = np.where(np.arange(x.shape[0]) < 50, 0.0, 1.0)
    weights[:50] = 0.0
    model = GLM().fit(np.column_stack([x, constant]), y, weights)
    assert not model.active_[-1]


def test_interactions_build_every_product_of_distinct_columns() -> None:
    x = np.arange(12.0).reshape(4, 3)
    design = GLM(interactions=3).full_design(x)
    expected = np.column_stack(
        [
            np.ones(4),
            x,
            x[:, 0] * x[:, 1],
            x[:, 0] * x[:, 2],
            x[:, 1] * x[:, 2],
            x[:, 0] * x[:, 1] * x[:, 2],
        ]
    )
    np.testing.assert_array_equal(design, expected)


def test_separation_is_refused() -> None:
    x = np.array([[0.0], [0.0], [1.0], [1.0]])
    with pytest.raises(DataError, match="separation"):
        GLM().fit(x, np.array([0, 0, 1, 1]))


def test_one_label_is_refused() -> None:
    with pytest.raises(DataError, match="every fit row holds the label"):
        GLM().fit(np.ones((3, 1)), np.array([1, 1, 1]))


def test_fit_learner_routes_weights_and_clones() -> None:
    x, y, weights = _law()
    template = GLM()
    model = fit_learner(template, x, y, weights)
    assert not hasattr(template, "coef_")
    np.testing.assert_allclose(model.coef_, GLM().fit(x, y, weights).coef_)
