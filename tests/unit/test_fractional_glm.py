"""The quasibinomial study learner: separation keeps the last iterate, convergence is unchanged.

Study A of ``point-treatment-survival`` stopped on replication 189 of its ``competing``
scenario.  At one node an arm's ``W1 = 0`` cell had no event of one cause, so the
coefficient of ``W1`` diverged while the deviance settled.  R's ``glm`` returns such a fit with
a warning.  The learner raised.  It now keeps the last iterate with
``QuasiBinomialSeparationWarning`` when the deviance has converged by R's rule, and it still
raises when the deviance has not.

The learner is shared by many registered studies, so a converging fit must take the same
iterates as before.  ``_previous_fit`` is the learner as it was, and the bit-identity test
compares the two on converging fits.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from tests.studies.fractional_glm import QuasiBinomialGLM, QuasiBinomialSeparationWarning


def _previous_fit(X: Any, y: Any, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """The learner before the separation branch, verbatim, returning its coefficients."""
    matrix = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    weights = np.ones_like(target)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    mean = float(np.average(target, weights=weights))
    coefficient = np.zeros(design.shape[1], dtype=float)
    coefficient[0] = math.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))
    for _ in range(max_iter):
        fitted = expit(design @ coefficient)
        variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        working = design @ coefficient + (target - fitted) / variance
        root_weight = np.sqrt(weights * variance)
        updated = np.linalg.lstsq(design * root_weight[:, None], working * root_weight, rcond=None)[
            0
        ]
        if np.max(np.abs(updated - coefficient)) <= tol:
            coefficient = updated
            break
        coefficient = updated
    else:
        raise RuntimeError("quasibinomial IRLS did not converge")
    return coefficient


def _separated(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A binary ``W1`` whose ``W1 = 0`` cell has no event: quasi-complete separation."""
    rng = np.random.default_rng(seed)
    w1 = rng.integers(0, 2, n).astype(float)
    w2 = rng.integers(0, 4, n).astype(float)
    y = np.where(w1 == 1.0, rng.binomial(1, 0.05, n), 0).astype(float)
    return np.column_stack([w1, w2]), y


@pytest.mark.parametrize("seed", range(6))
def test_a_converging_fit_takes_the_same_iterates(seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 3))
    y = rng.binomial(1, expit(0.3 + X @ np.array([0.5, -0.4, 0.2]))).astype(float)
    if seed % 2:
        y = np.clip(y * 0.7 + rng.uniform(0, 0.3, size=y.size), 0.0, 1.0)  # fractional target
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = QuasiBinomialGLM().fit(X, y)
    expected = _previous_fit(X, y)
    assert np.array_equal(model.intercept_, expected[:1])
    assert np.array_equal(model.coef_[0], expected[1:])


def test_separation_keeps_the_last_iterate_with_a_warning() -> None:
    X, y = _separated()
    with pytest.raises(RuntimeError, match="did not converge"):
        _previous_fit(X, y)
    with pytest.warns(QuasiBinomialSeparationWarning, match="quasi-complete separation"):
        model = QuasiBinomialGLM().fit(X, y)
    empty = model.predict(X[X[:, 0] == 0.0])
    assert np.all(empty < 1e-8)
    events = model.predict(X[X[:, 0] == 1.0])
    assert abs(float(events.mean()) - float(y[X[:, 0] == 1.0].mean())) < 1e-6


def test_a_deviance_that_has_not_settled_still_raises() -> None:
    """Mutation control: a fallback that accepted every unconverged fit would pass this."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = rng.binomial(1, expit(1.0 + 2.0 * X[:, 0])).astype(float)
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM(max_iter=2).fit(X, y)


def test_the_failing_study_replication_now_fits() -> None:
    """Replication 189 of study A's ``competing`` scenario, the one that stopped the run."""
    from tests.studies import canonical_point_survival as study

    frame, _ = study.draw_scenario("competing", study.STUDY.n, 189)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = study.fit("competing", frame)
    assert any(issubclass(w.category, QuasiBinomialSeparationWarning) for w in caught)
    for name in study.ESTIMANDS["competing"]:
        assert np.isfinite(result[name].psi)
        assert np.isfinite(result[name].std_error)
