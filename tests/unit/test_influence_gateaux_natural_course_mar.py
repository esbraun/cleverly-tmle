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

from cleverly.estimators import TMLE
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import natural_course_submodel
from cleverly.inference.influence import natural_course_mean
from tests import discrete_law_mar as law
from tests.conftest import (
    OracleMissingness,
    OracleOutcome,
    OracleOutcomeContinuous,
    OracleTreatment,
)
from tests.studies.missing_outcome_study_helpers import FailTreatment


def _estimator(dgp: Any, *, treatment_learner: Any = None) -> TMLE:
    return TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=FailTreatment() if treatment_learner is None else treatment_learner,
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


def test_a_fully_observed_frame_keeps_the_complete_data_boundary() -> None:
    """The complete-data branch is unchanged when no outcome is missing.

    ``complete_frame`` sets ``Delta = 1`` on every row, so ``has_missing_outcome`` is
    False and the fit takes the pre-existing empirical branch.  This pins that the new
    target did not move that boundary.  It deliberately does **not** evidence the
    natural-course construction: the mutation control below is what does that, and it
    fails this file rather than this test when the new code is wrong.
    """
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


def test_pi_equal_one_reduces_exactly_to_the_empirical_mean_and_curve() -> None:
    r"""At :math:`\pi \equiv 1` the natural-course code returns the empirical mean.

    The reduction has to be checked on the natural-course code itself.  Routing it
    through ``fit`` cannot do that: the path is selected by ``has_missing_outcome``,
    which is false exactly when every row is observed, so a frame with
    :math:`\pi \equiv 1` and no missing outcome takes the complete-data branch and the
    check passes with :func:`natural_course_mean` deleted entirely.  This calls the
    submodel and the influence function directly, which is the only way the limit is
    reachable, and asserts both halves of the RM7 acceptance sentence: the estimate is
    the empirical mean and the curve is :math:`Y - \psi`.
    """
    frame = law.complete_frame()
    y = frame["Y"].to_numpy(dtype=float)
    treatment = frame["A"].to_numpy(dtype=float)
    n = y.shape[0]
    observed = np.ones(n, dtype=bool)

    submodel = natural_course_submodel(
        treatment,
        np.empty(n, dtype=float),
        arms=(0.0, 1.0),
        missingness=np.ones((n, 2), dtype=float),
    )
    np.testing.assert_array_equal(submodel.observed[:, 0], np.ones(n))

    # Targeting at pi = 1 over a fully observed frame solves P_n(Y - m*) = 0, so the
    # targeted regression the limit is taken at is any fit with the empirical mean.
    # Using Y itself is that fit, and it keeps the assertion exact rather than to a
    # tolerance.
    targeted = InitialFit(observed=y, arms={0.0: y, 1.0: y})
    mean = natural_course_mean(y, targeted, submodel, np.ones(n), observed)

    empirical = float(y.mean())
    assert mean.psi == empirical
    np.testing.assert_array_equal(mean.influence_curve, y - empirical)


def test_the_pi_equal_one_reduction_fails_when_the_plug_in_term_is_dropped() -> None:
    """The control that makes the reduction above non-vacuous.

    Without it the reduction is satisfied by any function returning the empirical mean,
    including one that never reads the clever covariate. Here the plug-in term is
    removed by hand and both halves of the reduction must break.
    """
    frame = law.complete_frame()
    y = frame["Y"].to_numpy(dtype=float)
    n = y.shape[0]
    submodel = natural_course_submodel(
        frame["A"].to_numpy(dtype=float),
        np.empty(n, dtype=float),
        arms=(0.0, 1.0),
        missingness=np.ones((n, 2), dtype=float),
    )
    # A targeted regression that does *not* already average to the empirical mean, so a
    # construction that drops either term disagrees with the one that keeps both.
    shifted = np.clip(y * 0.5 + 0.2, 0.0, 1.0)
    targeted = InitialFit(observed=shifted, arms={0.0: shifted, 1.0: shifted})
    mean = natural_course_mean(y, targeted, submodel, np.ones(n), np.ones(n, dtype=bool))

    residual_only = float(np.mean(y - shifted))
    assert mean.psi != pytest.approx(residual_only, abs=1e-9)
    assert mean.psi == pytest.approx(float(shifted.mean()), abs=1e-12)
    # The curve keeps the response residual: dropping it would leave m* - psi alone.
    np.testing.assert_allclose(mean.influence_curve, (y - shifted) + shifted - mean.psi)
    assert not np.allclose(mean.influence_curve, shifted - mean.psi)


def test_a_bounded_continuous_outcome_recovers_the_exact_target_and_curve() -> None:
    r"""The bounded-continuous half of the supported surface, against the exact law.

    The registered continuous scenario declares ``q_bounds = (0, 1)``, which makes
    :class:`~cleverly.utils.bounds.OutcomeScaler` the identity and leaves the
    scale-and-unscale path unexercised against a known truth.  Here the binary law's
    outcome is mapped affinely onto a wider range and the declared bounds are the image
    of ``[0, 1]``, so the target is the image of :data:`law.TRUTH` and the curve is the
    image of :func:`law.eif` scaled by the range.  Both are exact rather than
    approximate: an affine map commutes with the mean and with the Gateaux derivative.
    """
    lower, upper = -1.0, 3.0
    span = upper - lower
    frame = law.frame()
    frame = frame.assign(Y=lower + span * frame["Y"])

    dgp = law.DiscreteLaw()
    result = (
        TMLE(
            outcome_learner=OracleOutcomeContinuous(dgp),
            treatment_learner=FailTreatment(),
            missingness_learner=OracleMissingness(dgp),
            estimands=("ey_obs",),
            cross_fit=False,
            q_bounds=(lower, upper),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )

    expected_psi = lower + span * law.TRUTH["ey_obs"]
    assert result.psi("ey_obs") == pytest.approx(expected_psi, abs=1e-9)

    # The curve must come back on the *original* outcome scale, which is the reported
    # standard error's scale. A missing range factor is a silent divide-by-four here.
    # ``frame`` lays out one contiguous block per support point, in SUPPORT order, so
    # repeating each point by its own count gives the per-row cell index.
    counts = [law.COUNTS[w, a, k] for w, a, k in law.SUPPORT]
    cells = np.repeat(np.arange(len(law.SUPPORT)), counts)
    expected_curve = span * law.eif("ey_obs")[cells]
    np.testing.assert_allclose(
        np.asarray(result["ey_obs"].influence_curve), expected_curve, atol=1e-8
    )
