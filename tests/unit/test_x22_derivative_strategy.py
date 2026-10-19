"""The targeting settings beside ``CTMLE(strategy="derivative")`` (C-TMLE0, X30).

C-TMLE0 solves one update of the arm clever covariates and their derivative columns on every
row, then continues as an ordinary TMLE.  The update is a targeting solve, so the targeting settings
reach it.

* ``stopping`` reaches the derivative solve with its stop context.  Before it did, a declared
  rule raised the solver's backstop ``ValueError`` on every derivative fit.
* ``treatment_adjustment=`` restricts the lasso pair and the declared learners to ``Wg``, the
  columns the fit then claims in its summary.
* ``outcome_by_arm`` reaches the derivative columns through the shared initial fit.
* ``iterate`` and ``step_bound`` walk the derivative solve, and a walk that stops short of the
  derivative equation is refused rather than reported as C-TMLE0.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from tests.unit._source_mutation import mutate
from tests.unit.test_outcome_by_arm import binary_frame, hand_models, own_arm
from tests.unit.test_treatment_covariates_collaborative import COVARIATES, lasso_frame

ctmle_module = importlib.import_module("cleverly.estimators.ctmle")
derivative_module = importlib.import_module("cleverly.estimators._ctmle_derivative")

PAIR = (0.05, 0.02)
WG = ["c2", "c3"]


def derivative_fit(
    covariates: list[str] = COVARIATES, wg: list[str] | None = None, **settings: Any
) -> Any:
    """A C-TMLE0 fit whose outcome regression ignores the covariates, so only ``g`` can differ."""
    options: dict[str, Any] = {
        "strategy": "derivative",
        "outcome_learner": DummyRegressor(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ate",),
        "q_bounds": (-20.0, 20.0),
        "lambdas": PAIR,
    }
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**options)
            .fit(
                lasso_frame(),
                outcome="Y",
                treatment="A",
                covariates=covariates,
                treatment_covariates=wg,
            )
            .single()
        )


def psi(result: Any) -> float:
    return float(result.estimates["ate"].psi)


# ------------------------------------------------------------------------------ stopping


@pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
def test_a_declared_rule_fits_the_derivative_update(rule: str) -> None:
    """Before the stop context reached the update, any declared rule raised ``ValueError``."""
    result = derivative_fit(stopping=rule)
    assert next(iter(result.fluctuations.values())).stopping == rule
    assert psi(result) == pytest.approx(psi(derivative_fit()), abs=1e-9)


def test_a_declared_rule_judges_the_derivative_score_and_stops_the_bounded_walk() -> None:
    """A bounded walk stops at the rule, far from the ``1e-10`` bar of ``relative``.

    The witness is nonzero: the derivative score at the stop is at most the rule's threshold
    and well above the ``relative`` bar, and the fit is C-TMLE0 (the derivative solve converged
    under the declared rule).
    """
    n = len(lasso_frame())
    settings: dict[str, Any] = {"step_bound": 1.0, "max_iter": 100}
    ruled = derivative_fit(stopping="sample_size", **settings)
    after = np.abs(ruled.extra["ctmle"].derivative_score_after)
    span = ruled.nuisance.scaler.range
    assert np.all(after * span <= 1.0 / n)
    assert np.max(after) > 1e-8
    with pytest.raises(ValueError, match="did not converge in 100 iterations"):
        derivative_fit(**settings)


def test_mutation_a_derivative_solve_without_the_stop_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def witness() -> None:
        derivative_fit(stopping="sample_size")

    witness()
    mutate(
        monkeypatch,
        derivative_module,
        derivative_module,
        "derivative_update",
        [("warn=False, stop=stop", "warn=False")],
    )
    with pytest.raises(ValueError, match="StopContext"):
        witness()


def test_substituting_the_relative_rule_fails_the_bounded_walk_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    witness = test_a_declared_rule_judges_the_derivative_score_and_stops_the_bounded_walk
    witness()
    monkeypatch.setattr(derivative_module, "replace", replace, raising=False)
    mutate(
        monkeypatch,
        derivative_module,
        derivative_module,
        "derivative_update",
        [
            (
                "mask, spec, warn=False, stop=stop",
                'mask, replace(spec, stopping="relative"), warn=False, stop=stop',
            )
        ],
    )
    with pytest.raises(ValueError, match="did not converge in 100 iterations"):
        witness()


def test_the_retarget_rebuilds_the_derivative_update_under_a_rule() -> None:
    """A truncation sweep refits the update at each bound, so its call carries the context."""
    result = derivative_fit(stopping="scaled_var")
    curve = result.diagnostics.truncation_curve()
    assert len(curve) > 1


# ------------------------------------------------------------------- iterate and step_bound


def test_the_best_iterate_composes_with_the_derivative_update() -> None:
    last = derivative_fit()
    best = derivative_fit(iterate="best")
    assert psi(best) == pytest.approx(psi(last), abs=1e-9)


def test_a_step_bound_that_stops_short_of_the_derivative_equation_is_refused() -> None:
    with pytest.raises(ValueError, match="did not converge in 20 iterations"):
        derivative_fit(step_bound=0.5)


# --------------------------------------------------------------------------- outcome_by_arm


def test_the_derivative_update_starts_from_the_per_arm_outcome_regression() -> None:
    frame = binary_frame(n=600)
    options: dict[str, Any] = {
        "strategy": "derivative",
        "outcome_learner": LogisticRegression(C=1e6, max_iter=5000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 3,
        "cross_fit": True,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ate",),
        "lambdas": PAIR,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_arm = (
            CTMLE(outcome_by_arm=True, **options)
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
            .single()
        )
        pooled = (
            CTMLE(**options)
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
            .single()
        )
    data = per_arm.data
    reference = hand_models(
        data, per_arm.nuisance.folds, per_arm.nuisance.scaler.scale(data.outcome)
    )
    for arm in data.arm_codes:
        np.testing.assert_allclose(
            per_arm.nuisance.outcome.arms[arm], reference[arm], atol=1e-12, rtol=0
        )
    np.testing.assert_allclose(
        per_arm.nuisance.outcome.observed, own_arm(data, reference), atol=1e-12, rtol=0
    )
    # The control: the pooled regression differs, and the update starts from the one it is given.
    gap = max(
        float(np.max(np.abs(per_arm.nuisance.outcome.arms[a] - pooled.nuisance.outcome.arms[a])))
        for a in data.arm_codes
    )
    assert gap > 1e-3
    assert abs(psi(per_arm) - psi(pooled)) > 1e-6


# ---------------------------------------------------------------- treatment_adjustment (Wg)

PAIRS: list[dict[str, Any]] = [
    {"lambdas": PAIR},
    {"lambdas": None, "learner_folds": 3},
    {
        "lambdas": None,
        "treatment_candidates": (LogisticRegression(C=0.001), LogisticRegression(C=0.05)),
    },
]


@pytest.mark.parametrize("settings", PAIRS, ids=["pair", "cv_grid", "learners"])
def test_a_derivative_fit_on_wg_equals_the_fit_that_omits_the_other_covariates(
    settings: dict[str, Any],
) -> None:
    """The outcome regression ignores the covariates, so only the pool of ``g`` can differ."""
    restricted = derivative_fit(COVARIATES, WG, **settings)
    omitted = derivative_fit(WG, None, **settings)
    full = derivative_fit(COVARIATES, None, **settings)
    np.testing.assert_allclose(
        restricted.nuisance.propensity.values, omitted.nuisance.propensity.values, atol=1e-12
    )
    np.testing.assert_allclose(
        restricted.nuisance.treatment_partner.values,
        omitted.nuisance.treatment_partner.values,
        atol=1e-12,
    )
    assert psi(restricted) == pytest.approx(psi(omitted))
    assert restricted.nuisance.treatment_covariates == ("c2", "c3")
    # The control: c1 drives the treatment, so a mechanism that reads it moves g.
    assert not np.allclose(
        restricted.nuisance.propensity.values, full.nuisance.propensity.values, atol=1e-3
    )


@pytest.mark.parametrize("settings", PAIRS[:1] + PAIRS[2:], ids=["pair", "learners"])
def test_mutation_a_derivative_mechanism_on_the_full_set_fails_the_witness(
    settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def witness() -> None:
        restricted = derivative_fit(COVARIATES, WG, **settings)
        omitted = derivative_fit(WG, None, **settings)
        np.testing.assert_allclose(
            restricted.nuisance.propensity.values, omitted.nuisance.propensity.values, atol=1e-12
        )

    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        ctmle_module.CTMLE,
        "_derivative_nuisances",
        [("design = data.mechanism_covariates", "design = data.covariates")],
    )
    with pytest.raises((AssertionError, ValueError, IndexError)):
        witness()


def test_the_derivative_fit_on_wg_carries_the_restriction_text() -> None:
    summary = derivative_fit(COVARIATES, WG).summary()
    assert "model restrictions" in summary
    assert "c2, c3" in summary
