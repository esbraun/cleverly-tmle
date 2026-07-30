"""A three-armed fit end to end: does it recover the truth, and does the rest still work?

Split by cost, following the project's rule that the fast tier buys exactness where it
can.  The deterministic structural claims -- that a contrast equals the difference of the
means it contrasts, that a round trip changes nothing, that every cross-fitting scheme
produces the same parameters -- are checked here on a single fit and fail deterministically.
The one claim that genuinely needs replication, *consistency*, is a ``slow`` study: a
single fit cannot distinguish an estimator that is unbiased from one that is off by half a
standard error, and pretending otherwise with one seed would be a coin flip.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE, load
from cleverly.datasets import make_multi_arm, multi_arm_dgp

#: Every parameter a default three-armed fit reports, and its population value.
TRUTH = multi_arm_dgp().truth()


def _fit(n: int = 800, seed: int = 0, **overrides):
    frame, _ = make_multi_arm(n=n, seed=seed)
    estimator = TMLE(
        **{
            "outcome_learner": "glm",
            "treatment_learner": "glm",
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
        report = fit.sensitivity.positivity()
        assert set(report.effective_sample_size) == {"low", "medium", "high"}
        assert set(report.propensity_quantiles) == {"g[low]", "g[medium]", "g[high]"}
        assert report.summary()

    def test_the_nuisance_diagnostics_report_every_arm(self, fit) -> None:
        names = {model.name for model in fit.validation.nuisance().models}
        assert {"propensity[low]", "propensity[medium]", "propensity[high]"} <= names

    def test_the_score_check_passes(self, fit) -> None:
        assert fit.validation.score_check().passed

    def test_a_round_trip_changes_nothing(self, fit, tmp_path) -> None:
        path = tmp_path / "multi.npz"
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
        frame = fit.sensitivity.truncation_curve(bounds=[0.01, 0.05])
        assert len(frame) > 0

    @pytest.mark.parametrize("scheme", ["pooled", "fold"])
    def test_both_targeting_schemes_report_the_same_parameters(self, scheme: str) -> None:
        result = _fit(targeting_scheme=scheme)
        assert set(result.estimates) == set(TRUTH)
        assert result.fluctuations["mean"].epsilon.shape == (3,)


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


@pytest.mark.slow
class TestConsistency:
    """The claim a single fit cannot make: the estimator is unbiased for every arm.

    Averaged over replications and compared against the Monte Carlo standard error of
    that average, which is the only comparison that distinguishes "unbiased" from "biased
    by less than one standard error".  The outcome model is correctly specified by the
    indicator design for this process (see :func:`~cleverly.datasets.multi_arm_dgp`), so
    what remains after averaging is sampling error and nothing else.
    """

    REPLICATIONS = 60

    @pytest.fixture(scope="class")
    def replicates(self) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {name: [] for name in TRUTH}
        for replicate in range(self.REPLICATIONS):
            result = _fit(n=600, seed=1000 + replicate)
            for name in TRUTH:
                values[name].append(result.estimates[name].psi)
        return {name: np.asarray(v, dtype=float) for name, v in values.items()}

    @pytest.mark.parametrize("name", sorted(TRUTH))
    def test_the_bias_is_within_monte_carlo_error(self, replicates, name: str) -> None:
        draws = replicates[name]
        bias = float(draws.mean()) - TRUTH[name]
        monte_carlo = float(draws.std(ddof=1)) / np.sqrt(draws.size)
        # Three Monte Carlo standard errors: wide enough that a consistent estimator
        # passes reliably, narrow enough that a wrong arm denominator does not.
        assert abs(bias) < 3.0 * monte_carlo, f"{name}: bias {bias:+.4f}, mcse {monte_carlo:.4f}"
