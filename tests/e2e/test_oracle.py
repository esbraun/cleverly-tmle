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

Most of the module uses a binary outcome, so the outcome scaler is the identity and the
true conditional mean can be supplied directly on the scale the fluctuation works on.
:class:`TestRandomisedTreatment` also runs the continuous case, where the estimator maps
``Y`` onto ``[0, 1]`` first and the oracle has to follow it there
(:class:`tests.conftest.OracleOutcomeContinuous`) -- otherwise the scaling path would only
ever be exercised alongside nuisance-estimation error.

Between them the classes here cover the oracle cases worth distinguishing: randomised
treatment with a binary and with a continuous outcome; confounded treatment with a known
logistic ``g``; known missingness at random; and effect heterogeneity that separates
``att``, ``ate`` and ``atc``.  The remaining two -- a nonlinear ``Qbar`` with known ``g``,
and a weighted-population target -- live in :mod:`tests.e2e.test_double_robustness` and
:mod:`tests.unit.test_weighted_estimand`, and the controlled intervention on ``Z`` in
:mod:`tests.unit.test_influence_gateaux_cde`, where each has machinery of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import (
    DGP,
    binary_outcome_dgp,
    heterogeneous_dgp,
    missing_outcome_binary_dgp,
)
from cleverly.utils.bounds import expit
from tests.conftest import (
    OracleMissingness,
    OracleOutcome,
    OracleOutcomeContinuous,
    OracleTreatment,
    aipw_ate,
)


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


def _randomised(*, binary: bool) -> DGP:
    """A DGP where treatment is assigned by a coin flip, independently of ``W``.

    Every other oracle process in the suite confounds.  A randomised arm is worth having
    on its own because it is the one setting with a reference that needs no nuisance model
    at all: when ``A`` is independent of ``W``, the unadjusted difference in means already
    estimates the ATE, so :class:`TestRandomisedTreatment` can check the estimator against
    something outside the targeted-learning machinery entirely.

    It is also the setting where the estimator has the least excuse to move: ``g`` is known
    and constant, the outcome model is correct, and so the fluctuation has nothing to
    correct.  An estimator that manufactured signal here would be doing so out of nothing.
    """

    def propensity(w: Any) -> Any:
        return np.full(w.shape[0], 0.5)

    if binary:

        def outcome_mean(w: Any, a: float, z: float | None) -> Any:
            del z
            return expit(-0.3 + 0.8 * a + 0.6 * w[:, 0] - 0.4 * w[:, 1])

        return DGP(
            name="randomised_binary",
            n_latent=2,
            covariate_names=("W1", "W2"),
            propensity=propensity,
            outcome_mean=outcome_mean,
            family="binomial",
        )

    def outcome_mean(w: Any, a: float, z: float | None) -> Any:
        del z
        # Heterogeneous in W1, so ATT and ATC are not equal by construction of the
        # outcome model -- they are equal because the *assignment* is independent of W.
        return 1.0 + 0.8 * w[:, 0] - 0.5 * w[:, 1] + (1.2 + 0.6 * w[:, 0]) * a

    return DGP(
        name="randomised_continuous",
        n_latent=2,
        covariate_names=("W1", "W2"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        noise_scale=1.0,
    )


class TestRandomisedTreatment:
    """The two oracle cells the suite was missing: a randomised arm, binary and continuous.

    A continuous outcome is the case :class:`tests.conftest.OracleOutcome` cannot serve,
    because the estimator rescales ``Y`` onto ``[0, 1]`` before fitting and the oracle has
    to follow it there -- which is what :class:`tests.conftest.OracleOutcomeContinuous`
    exists for.  Running both here means the scaling path is exercised with the truth
    plugged in, rather than only alongside nuisance-estimation error.
    """

    @pytest.fixture(scope="class", params=["binary", "continuous"])
    def fit(self, request):
        binary = request.param == "binary"
        dgp = _randomised(binary=binary)
        frame, truth = dgp.sample(4000, seed=7)
        outcome_learner = OracleOutcome(dgp) if binary else OracleOutcomeContinuous(dgp)
        estimator = TMLE(
            outcome_learner=outcome_learner,
            treatment_learner=OracleTreatment(dgp),
            cross_fit=False,
            estimands=("ate", "att", "atc", "ey1", "ey0"),
            random_state=0,
            simultaneous=False,
        )
        return estimator.fit(frame, outcome="Y", treatment="A"), truth, frame

    def test_randomisation_makes_the_three_contrasts_coincide(self, fit) -> None:
        # A is independent of W, so the treated and control populations have the same
        # covariate distribution and hence the same average effect -- even though the
        # effect itself varies with W in the continuous process.
        _, truth, _ = fit
        assert truth["att"] == pytest.approx(truth["ate"], abs=1e-3)
        assert truth["atc"] == pytest.approx(truth["ate"], abs=1e-3)

    def test_the_score_equation_is_solved_exactly(self, fit) -> None:
        result, _, _ = fit
        check = result.validation.score_check()
        assert check.passed
        for row in check.rows:
            assert abs(row.score) < 1e-12

    def test_targeting_has_almost_nothing_to_do(self, fit) -> None:
        # Both nuisances are exactly right, so epsilon is pure sampling noise. This is the
        # check that the machinery does not manufacture a correction out of nothing.
        result, _, _ = fit
        for fluctuation in result.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) < 0.1

    def test_it_agrees_with_the_unadjusted_difference_in_means(self, fit) -> None:
        """The reference that owes nothing to targeted learning.

        Under randomisation ``E[Y | A = 1] - E[Y | A = 0]`` is already the ATE, so this
        compares the estimator against arithmetic on the raw data.  It is the only anchor
        in the suite that shares no assumption, no nuisance model and no code with the
        thing it is checking.
        """
        result, _, frame = fit
        y = frame["Y"].to_numpy(dtype=float)
        a = frame["A"].to_numpy(dtype=float)
        naive = float(y[a == 1.0].mean() - y[a == 0.0].mean())
        assert result.psi("ate") == pytest.approx(naive, abs=3.0 * result["ate"].std_error)

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc", "ey1", "ey0"])
    def test_recovers_the_truth_within_sampling_error(self, fit, estimand: str) -> None:
        result, truth, _ = fit
        estimate = result[estimand]
        deviation = abs(estimate.psi - truth[estimand])
        assert deviation < 4.0 * estimate.std_error, (
            f"{estimand}: {estimate.psi:.4f} vs truth {truth[estimand]:.4f}, "
            f"se {estimate.std_error:.4f}"
        )


class TestHeterogeneousContrastsAreRecovered:
    """``att > ate > atc`` recovered from data, not merely true of the process.

    :mod:`tests.unit.test_datasets` establishes that
    :func:`~cleverly.datasets.heterogeneous_dgp` really does order its three contrasts,
    with a margin of 0.69 on either side.  That is a statement about the integration.  This
    is the statement about the estimator: handed the true nuisances, it has to find the
    same order from a finite sample, and each contrast has to land near its own truth
    rather than the three merely coming out in the right sequence.

    A constant-effect process cannot make this check -- there is no order to recover -- and
    :func:`~cleverly.datasets.nonlinear_dgp` separates the contrasts without fixing which
    way round, so neither would notice an estimator that conditioned on the wrong arm.
    """

    SEEDS = (3, 17, 41, 59, 83)

    @staticmethod
    def _fit(seed: int):
        dgp = heterogeneous_dgp()
        frame, truth = dgp.sample(4000, seed=seed)
        estimator = TMLE(
            outcome_learner=OracleOutcomeContinuous(dgp),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=False,
            estimands=("ate", "att", "atc"),
            random_state=0,
            simultaneous=False,
        )
        return estimator.fit(frame, outcome="Y", treatment="A"), truth

    @pytest.fixture(scope="class")
    def fits(self):
        return [self._fit(seed) for seed in self.SEEDS]

    def test_the_ordering_holds_on_every_seed(self, fits) -> None:
        # The population margin is 0.69 and the standard error is an order of magnitude
        # smaller, so the ordering is not a coin flip that happens to land right; asking
        # for five out of five is asking for something an estimator with the arms crossed
        # could not supply.
        for result, _ in fits:
            att, ate, atc = (result.psi(name) for name in ("att", "ate", "atc"))
            assert att > ate > atc, f"got att={att:.3f}, ate={ate:.3f}, atc={atc:.3f}"

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_each_contrast_lands_near_its_own_truth(self, fits, estimand: str) -> None:
        # The ordering alone would be satisfied by three numbers that are merely spread
        # out in the right direction, so pin each one to its own reference.
        for result, truth in fits:
            estimate = result[estimand]
            deviation = abs(estimate.psi - truth[estimand])
            assert deviation < 4.0 * estimate.std_error, (
                f"{estimand}: {estimate.psi:.4f} vs truth {truth[estimand]:.4f}, "
                f"se {estimate.std_error:.4f}"
            )

    def test_the_separation_is_recovered_not_just_the_order(self, fits) -> None:
        # Averaging over the seeds removes most of the sampling error, so the estimated
        # gaps can be compared against the population ones rather than only signed.
        gaps = np.array(
            [
                [result.psi("att") - result.psi("ate"), result.psi("ate") - result.psi("atc")]
                for result, _ in fits
            ]
        )
        truth = heterogeneous_dgp().truth()
        expected = [truth["att"] - truth["ate"], truth["ate"] - truth["atc"]]
        np.testing.assert_allclose(gaps.mean(axis=0), expected, atol=0.06)
