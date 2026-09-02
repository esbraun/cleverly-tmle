"""A three-armed fit end to end: does it recover the truth, and does the rest still work?

The fast suite buys exactness where it can. The deterministic structural claims -- that a
contrast equals the difference of the
means it contrasts, that a round trip changes nothing, that every cross-fitting scheme
produces the same parameters -- are checked here on a single fit and fail deterministically.
The consistency claim needs replication and belongs in the registered multi-arm study. A single
fit cannot distinguish an unbiased estimator from one that is off by half a standard error.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import sklearn.linear_model

from cleverly import load
from cleverly._typing import FloatArray
from cleverly.datasets import MultiArmDGP, make_multi_arm, multi_arm_dgp
from cleverly.estimators import TMLE

#: Every parameter a default three-armed fit reports, and its population value.
TRUTH = multi_arm_dgp().truth()


def _fit(n: int = 800, seed: int = 0, **overrides):
    frame, _ = make_multi_arm(n=n, seed=seed)
    estimator = TMLE(
        **{
            "outcome_learner": sklearn.linear_model.LinearRegression(),
            "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
            "n_folds": 5,
            "learner_folds": 3,
            "random_state": 0,
            "simultaneous": False,
            **overrides,
        }
    )
    return estimator.fit(frame, outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def fit():
    return _fit()


class TestWhatIsReported:
    def test_one_mean_per_arm_and_one_contrast_per_non_reference_arm(self, fit) -> None:
        assert set(fit.estimates) == {
            "ey[low]",
            "ey[medium]",
            "ey[high]",
            "ate[low vs high]",
            "ate[medium vs high]",
        }

    def test_the_reference_is_the_lowest_sorted_label(self, fit) -> None:
        # "high" sorts before "low" and "medium", so it is the default reference even
        # though it is the *last* level a reader would have written down. Worth pinning:
        # it is the surprise this API has, and `reference=` is how to avoid it.
        assert fit.config.reference_arm == 0.0
        assert fit.data.arm_label(fit.config.reference_arm) == "high"

    def test_every_estimate_lands_near_the_truth(self, fit) -> None:
        """A smoke test with a fixed seed, not a consistency claim.

        Three standard errors is wide enough not to be a coin flip and narrow enough to
        catch an arm wired to the wrong denominator; ``TestConsistency`` is where the
        actual bias claim is made.
        """
        for name, value in TRUTH.items():
            estimate = fit.estimates[name]
            assert abs(estimate.psi - value) < 3.0 * estimate.std_error, name


class TestTheJointInfluenceCurve:
    def test_a_contrast_is_the_difference_of_its_two_means(self, fit) -> None:
        """To floating-point rounding here, and *exactly* on an unscaled outcome.

        The gap is the outcome scaling and nothing else.  A contrast unscales as
        ``range * (ic_a - ic_ref)`` while the two means unscale as
        ``range * ic_a`` and ``range * ic_ref`` separately, and multiplying before
        subtracting is not the same last bit as subtracting before multiplying.  On a
        binary outcome the scaler is the identity and the two agree bit for bit --
        ``tests/unit/test_influence_gateaux_multi.py`` asserts that with
        ``assert_array_equal``, which is where the exact claim belongs.
        """
        reference = fit.estimates["ey[high]"].influence_curve
        for arm in ("low", "medium"):
            expected = fit.estimates[f"ey[{arm}]"].influence_curve - reference
            np.testing.assert_allclose(
                fit.estimates[f"ate[{arm} vs high]"].influence_curve, expected, rtol=1e-11, atol=0
            )

    def test_an_unreported_contrast_comes_from_the_delta_method(self, fit) -> None:
        """The pay-off of reporting K means with a joint covariance.

        ``medium`` against ``low`` is not among the reported parameters -- the reference
        is ``high`` -- and needs no refit to obtain.
        """
        derived = fit.contrast(lambda psi: psi[0] - psi[1], ["ey[medium]", "ey[low]"])
        expected = TRUTH["ate[medium vs high]"] - TRUTH["ate[low vs high]"]
        assert abs(derived.psi - expected) < 3.0 * derived.std_error

    def test_the_covariance_matrix_covers_every_arm(self, fit) -> None:
        names = ["ey[low]", "ey[medium]", "ey[high]"]
        covariance = fit.covariance(names)
        assert covariance.shape == (3, 3)
        np.testing.assert_allclose(covariance, covariance.T, atol=0, rtol=0)
        assert np.all(np.linalg.eigvalsh(covariance) > 0)


class TestTheRestOfTheStackStillWorks:
    def test_positivity_reports_every_arm(self, fit) -> None:
        report = fit.diagnostics.support()
        assert set(report.effective_sample_size) == {"low", "medium", "high"}
        assert set(report.propensity_quantiles) == {"g[low]", "g[medium]", "g[high]"}
        assert report.summary()

    def test_the_nuisance_diagnostics_report_every_arm(self, fit) -> None:
        names = {model.name for model in fit.diagnostics.nuisance_models().models}
        assert {"propensity[low]", "propensity[medium]", "propensity[high]"} <= names

    def test_the_score_check_passes(self, fit) -> None:
        assert fit.diagnostics.score_equations().passed

    def test_a_round_trip_changes_nothing(self, fit, tmp_path) -> None:
        path = tmp_path / "multi.joblib"
        fit.save(path)
        reloaded = load(path)
        assert set(reloaded.estimates) == set(fit.estimates)
        assert reloaded.config.reference_arm == fit.config.reference_arm
        assert reloaded.data.treatment_levels == fit.data.treatment_levels
        for name, estimate in fit.estimates.items():
            np.testing.assert_array_equal(
                reloaded.estimates[name].influence_curve, estimate.influence_curve
            )
        np.testing.assert_array_equal(
            reloaded.nuisance.propensity.values, fit.nuisance.propensity.values
        )

    def test_the_truncation_curve_sweeps_a_multi_arm_fit(self, fit) -> None:
        frame = fit.diagnostics.truncation_curve(bounds=[0.01, 0.05])
        assert len(frame) > 0

    def test_the_omitted_variable_bound_survives_the_round_trip(self, fit, tmp_path) -> None:
        """One bound per contrast, and the same one after a reload.

        The bound reads the fit's arms, its reference and its cached nuisances, all three
        of which cross the file format -- so a reloaded multi-arm fit is where a bound
        that had gone back to arms 1 and 0 would report a plausible number for a contrast
        nobody asked for. ``tests/unit/test_sensitivity_multi_arm.py`` checks the value
        against the closed form; this checks that persistence does not move it.
        """
        name = "ate[medium vs high]"
        path = tmp_path / "multi-sensitivity.joblib"
        fit.save(path)
        reloaded = load(path)
        assert reloaded.sensitivity.omitted_confounding(name).max_bias == pytest.approx(
            fit.sensitivity.omitted_confounding(name).max_bias, rel=1e-12
        )

    @pytest.mark.parametrize("scheme", ["pooled", "fold"])
    def test_both_targeting_schemes_report_the_same_parameters(self, scheme: str) -> None:
        result = _fit(targeting_scheme=scheme)
        assert set(result.estimates) == set(TRUTH)
        assert result.fluctuations["mean"].epsilon.shape == (3,)


class TestTheConditionalEffects:
    """``att`` / ``atc`` on a process where the two are genuinely different numbers.

    :func:`~cleverly.datasets.multi_arm_dgp` cannot serve here: its contrast is a constant,
    so the ATE, the ATT and the ATC coincide and every arrangement of the conditioning
    population would pass.  Adding an arm-covariate interaction to the *same* confounded
    mechanism separates them -- the units that select into an arm are the ones the
    interaction favours -- which is what gives the comparison teeth.

    The outcome regression is then misspecified by the indicator design, deliberately: the
    mechanism is a softmax linear in ``W`` and ``glm`` fits exactly that, so the estimate
    is consistent through the mechanism half of the double-robustness statement.  This is
    a smoke test at three standard errors either way, not a bias claim.
    """

    @staticmethod
    def _process() -> MultiArmDGP:
        step = np.array([0.0, 1.0, 2.4])

        def outcome_mean(w: FloatArray, arm: int) -> FloatArray:
            return 0.6 * step[arm] * (1.0 + 0.8 * w[:, 0]) + w[:, 0] - 0.5 * w[:, 1]

        return replace(multi_arm_dgp(), name="multi_arm_modified", outcome_mean=outcome_mean)

    @pytest.fixture(scope="class")
    def conditional_fit(self):  # type: ignore[no-untyped-def]
        process = self._process()
        frame, _ = process.sample(2000, seed=3)
        result = (
            TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                random_state=0,
                simultaneous=False,
                estimands=("ate", "att", "atc"),
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        return result, process.truth(conditional=True)

    def test_one_effect_per_non_reference_arm_and_group(self, conditional_fit) -> None:
        result, _ = conditional_fit
        assert set(result.estimates) == {
            "ate[low vs high]",
            "ate[medium vs high]",
            "att[low vs high]",
            "att[medium vs high]",
            "atc[low vs high]",
            "atc[medium vs high]",
        }

    def test_every_conditional_estimate_lands_near_its_own_truth(self, conditional_fit) -> None:
        result, truth = conditional_fit
        for name in result.estimates:
            estimate = result.estimates[name]
            assert abs(estimate.psi - truth[name]) < 3.0 * estimate.std_error, name

    def test_the_three_populations_are_distinguishable_on_this_process(
        self, conditional_fit
    ) -> None:
        """The premise of the test above, asserted rather than assumed.

        If the ATE, ATT and ATC coincided in the truth, agreeing with all three would say
        nothing about which population the estimator conditioned on.
        """
        _, truth = conditional_fit
        for arm in ("low", "medium"):
            values = [truth[f"{stem}[{arm} vs high]"] for stem in ("ate", "att", "atc")]
            assert max(values) - min(values) > 0.1

    def test_the_score_check_passes_for_every_group(self, conditional_fit) -> None:
        result, _ = conditional_fit
        assert result.diagnostics.score_equations().passed
        for group in ("mean", "att", "atc"):
            assert result.fluctuations[group].converged


class TestTheReferenceIsPartOfTheEstimand:
    def test_choosing_a_reference_changes_which_contrasts_are_reported(self) -> None:
        result = _fit(reference="low")
        assert set(result.estimates) == {
            "ey[low]",
            "ey[medium]",
            "ey[high]",
            "ate[medium vs low]",
            "ate[high vs low]",
        }
        assert result.config.reference_arm == 1.0  # "low" sorts second

    def test_the_means_do_not_depend_on_the_reference(self, fit) -> None:
        """Only the contrasts move: the targeted distribution is the same one."""
        default = fit
        chosen = _fit(reference="low")
        for arm in ("low", "medium", "high"):
            assert default.estimates[f"ey[{arm}]"].psi == pytest.approx(
                chosen.estimates[f"ey[{arm}]"].psi, abs=1e-12
            )
