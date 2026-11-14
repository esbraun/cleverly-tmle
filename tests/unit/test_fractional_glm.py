"""The quasibinomial study learner: separation keeps the last iterate, convergence is unchanged.

Study A of ``point-treatment-survival`` stopped on replication 189 of its ``competing``
scenario.  At one node an arm's ``W1 = 0`` cell had no event of one cause, so the
coefficient of ``W1`` diverged while the deviance settled.  R's ``glm`` returns such a fit with
a warning.  The learner raised.  It now keeps the last iterate with
``QuasiBinomialSeparationWarning`` when the deviance has converged by R's rule, and it still
raises when the deviance has not.

Complete separation of a binary target is a second case.  The deviance tends to zero there
and never meets R's relative rule, so the learner raised on 8% of the cross-fitted LTMLE draws of
``few_cluster_crossfit_properties`` at 4 clusters.  It now detects the case on the target and
the fitted values (:func:`~tests.studies.fractional_glm.completely_separated`) and keeps the last
iterate with the same warning.  The draws that raised are must-not-raise fixtures below.

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

from tests.studies import fractional_glm
from tests.studies.fractional_glm import (
    SEPARATION_TOLERANCE,
    QuasiBinomialGLM,
    QuasiBinomialSeparationWarning,
    completely_separated,
    deviance,
)


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


@pytest.mark.parametrize("seed", range(20))
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


# ------------------------------------------------------------------ complete separation

#: The cross-fitted LTMLE draws of ``few_cluster_crossfit_properties`` (law ``equal40``, 4
#: clusters) on which the learner raised at complete separation before it detected the case:
#: the first 12 of the seeds ``9_204_000 + r``.
SEPARATED_DRAW_SEEDS = (
    9_204_012,
    9_204_031,
    9_204_085,
    9_204_094,
    9_204_097,
    9_204_108,
    9_204_121,
    9_204_137,
    9_204_144,
    9_204_152,
    9_204_157,
    9_204_163,
)


def _completely_separated_design(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``Y = 1{x > 0}``, with a quarter of the rows on each side at distance ``0.005``.

    The rows at the margin are many, so the deviance of the last iterate is a sum of many
    small terms: above ``1e-6`` at 320 rows while every residual is about ``2e-8``.  That gap is
    what lets the design tell direct detection from an absolute deviance floor.
    """
    quarter = n // 4
    side = np.concatenate([np.full(quarter, 0.005), np.linspace(0.005, 1.005, quarter)])
    x = np.concatenate([-side, side])[:, None]
    return x, (x[:, 0] > 0.0).astype(float)


def _fits_without_raising(seeds: tuple[int, ...]) -> list[int]:
    """The seeds whose study fit raised ``RuntimeError``."""
    from tests.studies import few_cluster_crossfit_properties as study

    raised = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in seeds:
            try:
                estimate, _ = study.fit_drawn(study.draw("equal40", 4, seed))
            except RuntimeError:
                raised.append(seed)
            else:
                assert np.isfinite(estimate.psi)
    return raised


@pytest.mark.parametrize("n", [40, 320])
def test_complete_separation_keeps_the_last_iterate(n: int) -> None:
    X, y = _completely_separated_design(n)
    with pytest.raises(RuntimeError, match="did not converge"):
        _previous_fit(X, y)
    with pytest.warns(QuasiBinomialSeparationWarning, match="complete separation"):
        model = QuasiBinomialGLM().fit(X, y)
    fitted = model.predict(X)
    assert np.all(np.abs(fitted - y) <= SEPARATION_TOLERANCE)
    # The rows at the margin are near their class, not at it: a nonzero witness that the fit
    # kept an iterate rather than returning the labels.
    assert float(np.max(np.abs(fitted - y))) > 0.0


def test_complete_separation_on_the_study_draws_that_raised() -> None:
    """Review fixtures: every draw that raised at complete separation now fits."""
    assert _fits_without_raising(SEPARATED_DRAW_SEEDS) == []


def test_a_zero_tolerance_restores_the_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: the study draws above are witnesses of the separation branch."""
    monkeypatch.setattr(fractional_glm, "SEPARATION_TOLERANCE", 0.0)
    assert _fits_without_raising(SEPARATED_DRAW_SEEDS) == list(SEPARATED_DRAW_SEEDS)


def test_a_rule_relative_to_the_null_deviance_misses_study_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: separation read as deviance at most ``1e-8`` of the null deviance."""

    def relative(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> bool:
        null = np.full_like(target, np.average(target, weights=weights))
        return deviance(target, fitted, weights) <= 1e-8 * deviance(target, null, weights)

    monkeypatch.setattr(fractional_glm, "completely_separated", relative)
    assert len(_fits_without_raising(SEPARATED_DRAW_SEEDS)) >= 1


def test_an_absolute_deviance_floor_misses_the_large_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: separation read as a deviance of at most ``1e-6``."""

    def floor(target: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> bool:
        return deviance(target, fitted, weights) <= 1e-6

    monkeypatch.setattr(fractional_glm, "completely_separated", floor)
    X, y = _completely_separated_design(320)
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM().fit(X, y)


def test_a_fractional_target_is_never_separated() -> None:
    """Values 0.001 and 0.999 split by a threshold: a finite fit, not the separation branch."""
    X, labels = _completely_separated_design(40)
    target = np.where(labels == 1.0, 0.999, 0.001)
    weights = np.ones_like(target)
    assert not completely_separated(target, target, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = QuasiBinomialGLM().fit(X, target)
    assert np.all(np.isfinite(model.coef_))


def test_a_zero_weight_row_does_not_count_toward_separation() -> None:
    target = np.array([0.0, 1.0, 0.5])
    fitted = np.array([1e-9, 1.0 - 1e-9, 0.9])
    assert completely_separated(target, fitted, np.array([1.0, 1.0, 0.0]))
    assert not completely_separated(target, fitted, np.ones(3))


def test_an_oscillating_fit_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: a deviance that neither settles nor separates still raises."""
    X, y = _completely_separated_design(40)
    steps = iter(np.array([[0.0, sign * 2.0] for sign in (1, -1) * 100]))

    def oscillate(*args: Any, **kwargs: Any) -> tuple[np.ndarray, None, None, None]:
        return next(steps), None, None, None

    monkeypatch.setattr(fractional_glm.np.linalg, "lstsq", oscillate)
    with pytest.raises(RuntimeError, match="did not converge"):
        QuasiBinomialGLM().fit(X, y)
