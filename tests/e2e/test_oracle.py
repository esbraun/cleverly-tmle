r"""End-to-end behaviour with the true nuisance functions plugged in.

Handing the estimator the data-generating ``g`` and ``Qbar`` removes
nuisance-estimation error entirely, which isolates the estimator itself.  Three
things must then hold, and each would break for a different reason:

1. **the score equation is solved exactly** -- if it is not, the targeting step is
   broken;
2. **the estimate matches an independently written AIPW estimator** -- if it does not,
   the plug-in step or the clever covariate is wrong;
3. **the estimate lands within sampling error of the truth** -- if it does not, the
   estimand is misdefined.

A binary outcome is used throughout so the outcome scaler is the identity and the true
conditional mean can be supplied directly on the scale the fluctuation works on.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import binary_outcome_dgp, missing_outcome_binary_dgp
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment, aipw_ate


@pytest.fixture(scope="module")
def oracle_fit() -> tuple[object, dict[str, float], object]:
    dgp = binary_outcome_dgp()
    frame, truth = dgp.sample(4000, seed=5)
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        estimands="all",
        random_state=0,
        simultaneous=False,
    )
    result = estimator.fit(frame, outcome="Y", treatment="A")
    return result, truth, frame


class TestOracleNuisances:
    def test_the_score_equation_is_solved_exactly(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        check = result.validation.score_check()
        assert check.passed
        # Not merely "within tolerance" -- the fluctuation is a maximum-likelihood
        # solution, so the score is at floating-point zero.
        for row in check.rows:
            assert abs(row.score) < 1e-12

    def test_the_fluctuation_is_small_when_the_initial_fit_is_correct(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        # With the truth as the starting point, epsilon is pure sampling noise.
        for fluctuation in result.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) < 0.15

    def test_agrees_with_an_independent_aipw_implementation(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        dgp = binary_outcome_dgp()
        data = result.data
        w = data.covariates
        reference = aipw_ate(
            data.outcome,
            data.treatment,
            np.clip(dgp.propensity(w), 1e-9, 1 - 1e-9),
            dgp.outcome_mean(w, 1.0, None),
            dgp.outcome_mean(w, 0.0, None),
        )
        # Both solve the same efficient score equation with the same inputs, so they
        # differ only by the second-order gap between a substitution estimator and a
        # one-step correction.
        assert result.psi("ate") == pytest.approx(reference, abs=2e-3)

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc", "ey1", "ey0"])
    def test_recovers_the_truth_within_sampling_error(self, oracle_fit, estimand: str) -> None:
        result, truth, _ = oracle_fit
        estimate = result[estimand]
        deviation = abs(estimate.psi - truth[estimand])
        assert deviation < 4.0 * estimate.std_error, (
            f"{estimand}: {estimate.psi:.4f} vs truth {truth[estimand]:.4f}, "
            f"se {estimate.std_error:.4f}"
        )

    @pytest.mark.parametrize("estimand", ["rr", "or"])
    def test_recovers_the_ratio_estimands(self, oracle_fit, estimand: str) -> None:
        result, truth, _ = oracle_fit
        low, high = result[estimand].ci
        assert low <= truth[estimand] <= high

    def test_the_intervals_cover_the_truth(self, oracle_fit) -> None:
        result, truth, _ = oracle_fit
        covered = sum(
            result[name].ci[0] <= truth[name] <= result[name].ci[1]
            for name in ("ate", "att", "atc", "ey1", "ey0", "rr", "or")
        )
        # A single fit, so this is not a coverage study; but with oracle nuisances at
        # n=4000 a majority missing would signal a systematic problem.
        assert covered >= 6


class TestEstimandIdentities:
    def test_the_ate_is_exactly_the_difference_of_the_means(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        # EY1, EY0 and ATE come from the same two-column fluctuation, so this is exact.
        assert result.psi("ate") == pytest.approx(result.psi("ey1") - result.psi("ey0"), abs=1e-12)
        assert np.allclose(
            result["ate"].influence_curve,
            result["ey1"].influence_curve - result["ey0"].influence_curve,
        )

    def test_the_risk_ratio_is_the_ratio_of_the_means(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        assert result.psi("rr") == pytest.approx(result.psi("ey1") / result.psi("ey0"), abs=1e-12)

    def test_the_odds_ratio_is_the_ratio_of_the_odds(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        p1, p0 = result.psi("ey1"), result.psi("ey0")
        assert result.psi("or") == pytest.approx((p1 / (1 - p1)) / (p0 / (1 - p0)), abs=1e-12)

    def test_the_ate_decomposes_over_the_arms(self, oracle_fit) -> None:
        result, _, _ = oracle_fit
        share = result.data.treated_fraction
        combined = share * result.psi("att") + (1.0 - share) * result.psi("atc")
        # This identity is exact for the estimands but only approximate for these
        # estimates: the ATT and ATC get their own fluctuations, each solving its own
        # score equation, so the three targeted fits differ slightly.
        assert combined == pytest.approx(result.psi("ate"), abs=5.0 * result["ate"].std_error)


class TestOracleNuisancesWithMissingOutcomes:
    """The same three checks, with a third nuisance in the clever covariate.

    The AIPW comparison is the one that gains most from being extended here.  Under
    missingness the one-step estimator's residual term carries ``Delta / pi_a(W)`` on top
    of ``1{A=a} / g_a(W)``, and it is written out longhand in :func:`tests.conftest.aipw_ate`
    -- a second implementation of the estimating equation that shares no code with the
    fluctuation.  Agreement to the second-order gap between a substitution estimator and
    a one-step correction is then evidence about the assembled path, not about either
    implementation's internal consistency.
    """

    @pytest.fixture(scope="class")
    def oracle_fit(self) -> tuple[object, dict[str, float]]:
        dgp = missing_outcome_binary_dgp()
        frame, truth = dgp.sample(4000, seed=11)
        estimator = TMLE(
            outcome_learner=OracleOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            missingness_learner=OracleMissingness(dgp),
            cross_fit=False,
            estimands="all",
            random_state=0,
            simultaneous=False,
        )
        return estimator.fit(frame, outcome="Y", treatment="A", delta="Delta"), truth

    def test_a_material_share_of_outcomes_is_missing(self, oracle_fit) -> None:
        result, _ = oracle_fit
        assert 0.6 < float(result.data.observed.mean()) < 0.85

    def test_the_score_equation_is_solved_exactly(self, oracle_fit) -> None:
        result, _ = oracle_fit
        check = result.validation.score_check()
        assert check.passed
        for row in check.rows:
            assert abs(row.score) < 1e-12

    def test_agrees_with_an_independent_aipw_implementation(self, oracle_fit) -> None:
        result, _ = oracle_fit
        dgp = missing_outcome_binary_dgp()
        data = result.data
        w = data.covariates
        missingness = np.column_stack(
            [
                np.clip(dgp.missingness(w, 0.0), 1e-9, 1.0),
                np.clip(dgp.missingness(w, 1.0), 1e-9, 1.0),
            ]
        )
        reference = aipw_ate(
            data.outcome,
            data.treatment,
            np.clip(dgp.propensity(w), 1e-9, 1 - 1e-9),
            dgp.outcome_mean(w, 1.0, None),
            dgp.outcome_mean(w, 0.0, None),
            delta=data.observed.astype(float),
            missingness=missingness,
        )
        assert result.psi("ate") == pytest.approx(reference, abs=3e-3)

    def test_the_correction_is_what_carries_a_misspecified_outcome_model(self, oracle_fit) -> None:
        """What the ``1 / pi`` factor is actually for, priced on the reference estimator.

        With the *true* ``Qbar`` the augmentation term is mean-zero whether or not it is
        weighted, so dropping ``1 / pi`` costs nothing and comparing the two would prove
        nothing -- under missingness at random a correct outcome regression identifies
        the estimand on its own.  Replacing ``Qbar`` by a constant removes that half of
        double robustness, and then the weighting is the only thing left holding the
        estimate up: corrected, it still finds the truth; uncorrected, it converges to
        ``E[pi_a(W) (Qbar(a, W) - c)]`` and is visibly biased.
        """
        result, truth = oracle_fit
        dgp = missing_outcome_binary_dgp()
        data = result.data
        w = data.covariates
        propensity = np.clip(dgp.propensity(w), 1e-9, 1 - 1e-9)
        missingness = np.column_stack(
            [
                np.clip(dgp.missingness(w, 0.0), 1e-9, 1.0),
                np.clip(dgp.missingness(w, 1.0), 1e-9, 1.0),
            ]
        )
        flat = np.full(data.n, float(np.mean(data.outcome[data.observed])))
        delta = data.observed.astype(float)

        corrected = aipw_ate(
            data.outcome,
            data.treatment,
            propensity,
            flat,
            flat,
            delta=delta,
            missingness=missingness,
        )
        uncorrected = aipw_ate(data.outcome, data.treatment, propensity, flat, flat, delta=delta)
        assert corrected == pytest.approx(truth["ate"], abs=4.0 * result["ate"].std_error)
        assert abs(uncorrected - truth["ate"]) > 1e-2
        assert abs(uncorrected - truth["ate"]) > 3.0 * abs(corrected - truth["ate"])

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc", "ey1", "ey0"])
    def test_recovers_the_truth_within_sampling_error(self, oracle_fit, estimand: str) -> None:
        result, truth = oracle_fit
        estimate = result[estimand]
        deviation = abs(estimate.psi - truth[estimand])
        assert deviation < 4.0 * estimate.std_error, (
            f"{estimand}: {estimate.psi:.4f} vs truth {truth[estimand]:.4f}, "
            f"se {estimate.std_error:.4f}"
        )

    @pytest.mark.parametrize("estimand", ["rr", "or"])
    def test_the_ratio_estimands_have_a_truth_to_be_checked_against(
        self, oracle_fit, estimand: str
    ) -> None:
        # No other process combines a binary outcome with missing outcomes, so until now
        # rr and or under `delta=` had no population value to compare with at all.
        result, truth = oracle_fit
        low, high = result[estimand].ci
        assert low <= truth[estimand] <= high
