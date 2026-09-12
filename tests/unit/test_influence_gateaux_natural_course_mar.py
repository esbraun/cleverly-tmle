r"""Exact-law evidence for the missing-outcome natural-course mean.

The target is the mean of ``Qbar(A, W)`` over the observed joint law of ``(A, W)``.
It therefore has one response-score equation and no treatment-score equation.  The
finite law in :mod:`tests.discrete_law_mar` makes every conditional probability exact,
so these checks compare the reported curve with the definition-level Gateaux derivative
without sampling error or a second implementation of the estimator.

The mutation checks are deliberately away from the truth.  At the truth every
within-``(A, W)`` outcome residual has mean zero, so replacing ``1 / pi`` by almost
anything would leave an exact-law score at zero and make the test blind.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly.estimators import TMLE
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment


class _FailTreatment(BaseEstimator):
    """A treatment learner whose use is the failure this target must expose."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _FailTreatment:
        raise AssertionError("NaturalCourseMean must not fit a treatment mechanism")

    def predict_proba(self, X: Any) -> Any:
        raise AssertionError("NaturalCourseMean must not predict a treatment mechanism")


def _estimator(dgp: Any, *, treatment_learner: Any = None) -> TMLE:
    return TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=_FailTreatment() if treatment_learner is None else treatment_learner,
        missingness_learner=OracleMissingness(dgp),
        cross_fit=False,
        fluctuation="logistic",
        targeting="iterative",
        target_weights=False,
        estimands=("ey_obs",),
        simultaneous=False,
        random_state=0,
        max_iter=100,
        tol=1e-12,
    )


@pytest.fixture(scope="module")
def exact_fit():
    dgp = law.DiscreteLaw()
    return (
        _estimator(dgp)
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def _fluctuation(result: Any) -> Any:
    """The natural-course fit has exactly one scalar score group."""
    assert len(result.fluctuations) == 1
    return next(iter(result.fluctuations.values()))


def _longhand_eif() -> np.ndarray:
    out = []
    psi = law.TRUTH["ey_obs"]
    for w, a, kind in law.SUPPORT:
        plug_in = law.Q[w, a] - psi
        if kind == law.UNOBSERVED:
            out.append(plug_in)
        else:
            out.append((kind - law.Q[w, a]) / law.PI[w, a] + plug_in)
    return np.asarray(out)


class TestTheObservedDataFunctional:
    def test_the_point_estimate_is_the_exact_functional(self, exact_fit: Any) -> None:
        assert exact_fit.psi("ey_obs") == pytest.approx(law.TRUTH["ey_obs"], abs=1e-12)

    def test_the_reported_curve_is_the_gateaux_derivative(self, exact_fit: Any) -> None:
        reported = np.asarray(exact_fit["ey_obs"].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif("ey_obs"), atol=1e-12, rtol=0.0)

    def test_the_gateaux_derivative_matches_the_formula(self) -> None:
        np.testing.assert_allclose(law.eif("ey_obs"), _longhand_eif(), atol=1e-12, rtol=0.0)
        assert float(np.sum(law.PROBS.reshape(-1) * law.eif("ey_obs"))) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_unobserved_rows_have_only_the_plugin_term(self, exact_fit: Any) -> None:
        reported = np.asarray(exact_fit["ey_obs"].influence_curve)[law.first_row_of()]
        for point, (w, a, kind) in enumerate(law.SUPPORT):
            if kind == law.UNOBSERVED:
                assert reported[point] == pytest.approx(
                    law.Q[w, a] - law.TRUTH["ey_obs"], abs=1e-12
                )


class TestTheTargetingConstruction:
    def test_the_plugin_averages_over_all_rows(self, exact_fit: Any) -> None:
        targeted = np.asarray(_fluctuation(exact_fit).targeted.observed, dtype=float)
        observed = law.frame()["Delta"].to_numpy(dtype=bool)
        all_rows = float(np.mean(targeted))
        respondents = float(np.mean(targeted[observed]))
        assert exact_fit.psi("ey_obs") == pytest.approx(all_rows, abs=1e-12)
        assert abs(respondents - all_rows) > 1e-3
        assert respondents == pytest.approx(0.4874476987447698, abs=1e-12)

    def test_the_score_uses_the_respondent_mask_and_full_sample_denominator(
        self, exact_fit: Any
    ) -> None:
        frame = law.frame()
        observed = frame["Delta"].to_numpy(dtype=bool)
        y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
        a = frame["A"].to_numpy(dtype=int)
        w = frame["W"].to_numpy(dtype=int)
        targeted = np.asarray(_fluctuation(exact_fit).targeted.observed, dtype=float)
        pi = law.PI[w, a]
        score = np.mean(np.where(observed, (y - targeted) / pi, 0.0))
        unmasked = np.mean((y - targeted) / pi)
        assert score == pytest.approx(0.0, abs=1e-12)
        assert np.asarray(_fluctuation(exact_fit).score)[0] == pytest.approx(score, abs=1e-12)
        assert abs(unmasked) > 0.1

    def test_the_true_nuisances_need_no_targeting_step(self, exact_fit: Any) -> None:
        fluctuation = _fluctuation(exact_fit)
        assert np.asarray(fluctuation.epsilon)[0] == pytest.approx(0.0, abs=1e-12)
        assert np.asarray(fluctuation.score)[0] == pytest.approx(0.0, abs=1e-12)


def test_response_score_mutation_is_detected() -> None:
    """Freeze a nonzero targeting path, then replace only its response denominator."""
    dgp = law.DiscreteLaw()
    dgp.q = np.asarray(law.Q + np.array([[0.10, -0.15], [-0.15, 0.10], [0.05, 0.20]]), dtype=float)
    result = (
        _estimator(dgp)
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )
    frame = law.frame()
    observed = frame["Delta"].to_numpy(dtype=bool)
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    a = frame["A"].to_numpy(dtype=int)
    w = frame["W"].to_numpy(dtype=int)
    targeted = np.asarray(_fluctuation(result).targeted.observed, dtype=float)
    wrong_pi = law.PI + np.array([[0.30, -0.15], [-0.20, 0.25], [-0.35, 0.10]])
    true_score = np.mean(np.where(observed, (y - targeted) / law.PI[w, a], 0.0))
    mutated_score = np.mean(np.where(observed, (y - targeted) / wrong_pi[w, a], 0.0))
    assert abs(np.asarray(_fluctuation(result).epsilon)[0]) > 1e-3
    assert true_score == pytest.approx(0.0, abs=1e-12)
    assert mutated_score == pytest.approx(0.05996906193973897, abs=1e-12)


def test_pi_equal_one_reduces_exactly_to_the_empirical_mean_and_curve() -> None:
    frame = law.complete_frame()
    dgp = law.DiscreteLaw()
    dgp.pi = np.ones_like(law.PI)
    result = (
        _estimator(dgp, treatment_learner=OracleTreatment(dgp))
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )
    empirical = float(frame["Y"].mean())
    assert result.psi("ey_obs") == empirical
    np.testing.assert_array_equal(
        np.asarray(result["ey_obs"].influence_curve), frame["Y"].to_numpy(dtype=float) - empirical
    )
