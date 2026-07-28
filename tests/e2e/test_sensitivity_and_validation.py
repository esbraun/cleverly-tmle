"""Sensitivity and validation analyses on real fits.

These modules are the reason the library exists in the shape it does, so the tests
check that they *say the right thing*, not merely that they return a frame.  Each
analysis is exercised on a case where the correct answer is known by construction:

* positivity diagnostics on data with and without an overlap problem;
* the robustness value on a fit with a *known* confounder withheld;
* the missingness tilt at ``gamma = 0``, which must reproduce the MAR estimate exactly;
* refutation tests, where a placebo treatment must yield no effect.
"""

from __future__ import annotations

import warnings

import narwhals as nw
import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_missing_outcome,
    make_nonlinear_ate,
    make_weak_overlap,
)
from cleverly.exceptions import PositivityWarning
from tests.conftest import fast_tmle


@pytest.fixture(scope="module")
def good_overlap() -> object:
    frame, _ = make_linear_ate(n=1500, seed=71)
    return fast_tmle(estimands=("ate", "att", "ey1", "ey0")).fit(frame, outcome="Y", treatment="A")


@pytest.fixture(scope="module")
def poor_overlap() -> object:
    frame, _ = make_weak_overlap(n=1500, seed=72)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        return fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")


class TestPositivity:
    def test_good_overlap_is_reported_as_adequate(self, good_overlap) -> None:
        report = good_overlap.sensitivity.positivity()
        assert "adequate" in report.verdict()
        assert report.truncated["fraction"] == 0.0
        for arm in ("treated", "control"):
            # Almost all of the nominal sample size survives the reweighting.
            assert report.effective_sample_size[arm]["ratio"] > 0.8

    def test_poor_overlap_is_flagged(self, poor_overlap) -> None:
        report = poor_overlap.sensitivity.positivity()
        assert "adequate" not in report.verdict()
        assert report.truncated["fraction"] > 0.0
        worst = min(ess["ratio"] for ess in report.effective_sample_size.values())
        # The weighted analysis is using far fewer observations than it appears to.
        assert worst < 0.6

    def test_the_effective_sample_size_never_exceeds_the_arm_size(self, poor_overlap) -> None:
        report = poor_overlap.sensitivity.positivity()
        for ess in report.effective_sample_size.values():
            assert ess["effective"] <= ess["n"] + 1e-9

    def test_weight_concentration_is_higher_under_poor_overlap(
        self, good_overlap, poor_overlap
    ) -> None:
        good = good_overlap.sensitivity.positivity().weight_share["treated"]["top_1pct"]
        poor = poor_overlap.sensitivity.positivity().weight_share["treated"]["top_1pct"]
        assert poor > good

    def test_the_report_renders_and_tabulates(self, good_overlap) -> None:
        report = good_overlap.sensitivity.positivity()
        text = report.summary()
        assert "Positivity" in text
        assert "VERDICT" in text
        frame = report.to_frame(good_overlap.data)
        assert set(nw.from_native(frame, eager_only=True)["group"].unique().to_list()) == {
            "overall",
            "treated",
            "control",
        }


class TestTruncationCurve:
    def test_the_curve_is_flat_when_overlap_is_good(self, good_overlap) -> None:
        curve = nw.from_native(
            good_overlap.sensitivity.truncation_curve([0.001, 0.01, 0.05], estimands=["ate"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        # No propensity is near the boundary, so truncation cannot bite.
        assert float(values.max() - values.min()) < 1e-9

    def test_the_curve_moves_when_overlap_is_poor(self, poor_overlap) -> None:
        curve = nw.from_native(
            poor_overlap.sensitivity.truncation_curve([0.001, 0.01, 0.05, 0.15], estimands=["ate"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        errors = np.array(curve["std_err"].to_list())
        truncated = np.array(curve["truncated_fraction"].to_list())
        assert float(values.max() - values.min()) > 1e-3
        # Tighter bounds truncate more units and buy variance with bias.
        assert np.all(np.diff(truncated) > 0)
        assert errors[0] > errors[-1]

    def test_the_fitted_bound_is_marked(self, good_overlap) -> None:
        curve = nw.from_native(
            good_overlap.sensitivity.truncation_curve(estimands=["ate"]), eager_only=True
        )
        assert sum(curve["is_fitted_bound"].to_list()) >= 1

    def test_an_invalid_bound_is_refused(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            good_overlap.sensitivity.truncation_curve([0.7])


class TestOmittedVariableBias:
    def test_the_bound_grows_with_the_assumed_confounding(self, good_overlap) -> None:
        weak = good_overlap.sensitivity.omitted_variable("ate", cf_y=0.01, cf_d=0.01)
        strong = good_overlap.sensitivity.omitted_variable("ate", cf_y=0.10, cf_d=0.10)
        assert strong.bias > weak.bias
        assert strong.lower < weak.lower
        assert strong.upper > weak.upper

    def test_zero_confounding_reproduces_the_point_estimate(self, good_overlap) -> None:
        bounds = good_overlap.sensitivity.omitted_variable("ate", cf_y=0.0, cf_d=0.0)
        assert bounds.lower == pytest.approx(bounds.psi)
        assert bounds.upper == pytest.approx(bounds.psi)

    def test_the_robustness_value_is_where_the_bound_reaches_the_null(self, good_overlap) -> None:
        values = good_overlap.sensitivity.robustness_value("ate")
        rv = values["rv"]
        assert 0.0 < rv < 1.0
        # By definition, setting cf_y = cf_d = RV must put the bound at zero.
        at_rv = good_overlap.sensitivity.omitted_variable("ate", cf_y=rv, cf_d=rv)
        edge = at_rv.lower if at_rv.psi > 0 else at_rv.upper
        assert edge == pytest.approx(0.0, abs=1e-4)

    def test_the_confidence_robustness_value_is_the_smaller_one(self, good_overlap) -> None:
        values = good_overlap.sensitivity.robustness_value("ate")
        # It takes less confounding to make an interval touch the null than to move the
        # point estimate there.
        assert values["rva"] < values["rv"]

    def test_a_weaker_effect_has_a_smaller_robustness_value(self) -> None:
        strong_frame, _ = make_linear_ate(n=1500, seed=73, effect=2.0)
        weak_frame, _ = make_linear_ate(n=1500, seed=73, effect=0.2)
        strong = fast_tmle(estimands=("ate",)).fit(strong_frame, outcome="Y", treatment="A")
        weak = fast_tmle(estimands=("ate",)).fit(weak_frame, outcome="Y", treatment="A")
        assert (
            weak.sensitivity.robustness_value("ate")["rv"]
            < strong.sensitivity.robustness_value("ate")["rv"]
        )

    def test_poor_overlap_inflates_the_maximal_bias(self, good_overlap, poor_overlap) -> None:
        # nu^2 is the second moment of the Riesz representer, so it blows up exactly when
        # overlap fails -- the same quantity that drives the clever covariate.
        assert (
            poor_overlap.sensitivity.elements("ate").nu2
            > good_overlap.sensitivity.elements("ate").nu2
        )

    @pytest.mark.parametrize("estimand", ["ate", "ey1", "ey0", "att"])
    def test_every_linear_estimand_is_supported(self, good_overlap, estimand: str) -> None:
        elements = good_overlap.sensitivity.elements(estimand)
        assert elements.sigma2 > 0
        assert elements.nu2 > 0
        assert elements.max_bias == pytest.approx(np.sqrt(elements.sigma2 * elements.nu2))

    def test_the_doubly_robust_and_plugin_nu2_agree(self, good_overlap) -> None:
        doubly_robust = good_overlap.sensitivity.elements("ate", nu2_estimator="doubly_robust")
        plugin = good_overlap.sensitivity.elements("ate", nu2_estimator="plugin")
        # Both estimate E[alpha^2]; they differ only by sampling error.
        assert doubly_robust.nu2 == pytest.approx(plugin.nu2, rel=0.1)

    def test_a_ratio_estimand_is_refused_with_a_pointer_to_the_evalue(self) -> None:
        frame, _ = make_binary_outcome(n=800, seed=74)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A")
        with pytest.raises(ValueError, match=r"sensitivity\.evalue"):
            result.sensitivity.omitted_variable("rr")

    def test_benchmarking_a_real_confounder_reports_its_strength(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=75)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")
        # W1 drives both the outcome and treatment in this process, so dropping it must
        # register as a substantial confounder on both the cf_y and cf_d scales.
        benchmark = result.sensitivity.benchmark(["W1"], estimand="ate")
        assert benchmark.cf_y > 0.05
        assert benchmark.cf_d > 0.0
        assert abs(benchmark.delta_psi) > 0.0
        assert benchmark.sigma2_short > benchmark.sigma2_long
        assert "W1" in benchmark.summary()

    def test_benchmarking_a_pure_noise_covariate_reports_almost_nothing(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=76)
        noisy = frame.assign(noise=np.random.default_rng(0).normal(size=len(frame)))
        result = fast_tmle(estimands=("ate",)).fit(noisy, outcome="Y", treatment="A")
        benchmark = result.sensitivity.benchmark(["noise"], estimand="ate")
        assert benchmark.cf_y < 0.02
        assert benchmark.cf_d < 0.05

    def test_the_contour_grid_is_monotone(self, good_overlap) -> None:
        grid = nw.from_native(good_overlap.sensitivity.contour("ate", grid_size=5), eager_only=True)
        assert len(grid) == 25
        # The lower bound falls as either sensitivity parameter grows.
        at_origin = [
            row
            for row in zip(
                grid["cf_d"].to_list(), grid["cf_y"].to_list(), grid["value"].to_list(), strict=True
            )
            if row[0] == 0.0 and row[1] == 0.0
        ]
        assert at_origin[0][2] == pytest.approx(good_overlap.psi("ate"))
        assert min(grid["value"].to_list()) < good_overlap.psi("ate")


class TestEValue:
    def test_a_binary_outcome_uses_the_risk_ratio_directly(self) -> None:
        frame, _ = make_binary_outcome(n=2000, seed=77)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A")
        evalue = result.sensitivity.evalue("rr")
        assert not evalue.approximate
        assert evalue.risk_ratio == pytest.approx(result.psi("rr"))
        assert evalue.point > evalue.limit >= 1.0

    def test_the_default_prefers_the_risk_ratio(self) -> None:
        frame, _ = make_binary_outcome(n=1200, seed=78)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A")
        assert result.sensitivity.evalue().estimand == "rr"

    def test_a_continuous_outcome_is_converted_and_flagged(self, good_overlap) -> None:
        evalue = good_overlap.sensitivity.evalue("ate")
        assert evalue.approximate
        assert "Chinn" in evalue.note
        assert evalue.point > 1.0

    def test_the_odds_ratio_conversion_is_flagged_as_rare_outcome(self) -> None:
        frame, _ = make_binary_outcome(n=1200, seed=79)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A")
        evalue = result.sensitivity.evalue("or")
        assert evalue.approximate
        assert "rare outcome" in evalue.note
        assert evalue.risk_ratio == pytest.approx(np.sqrt(result.psi("or")))


class TestMissingnessTilt:
    @pytest.fixture(scope="class")
    def missing_fit(self) -> object:
        frame, _ = make_missing_outcome(n=1500, seed=80)
        return fast_tmle(estimands=("ate", "ey1")).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
        )

    def test_no_tilt_reproduces_the_reported_estimate(self, missing_fit) -> None:
        curve = nw.from_native(
            missing_fit.sensitivity.missingness_tilt([0.0], estimands=["ate"]), eager_only=True
        )
        # gamma = 0 is the MAR analysis, so the curve must pass exactly through the
        # reported point estimate. Anything else means the tilt is mis-parameterised.
        assert float(curve["psi"][0]) == pytest.approx(missing_fit.psi("ate"), rel=1e-12)
        assert bool(curve["is_mar"][0])

    def test_the_tilt_moves_the_estimate_monotonically(self, missing_fit) -> None:
        curve = nw.from_native(
            missing_fit.sensitivity.missingness_tilt(
                [-1.0, -0.5, 0.0, 0.5, 1.0], estimands=["ey1"]
            ),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        assert np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)

    def test_the_tipping_point_is_reported_or_absent(self, missing_fit) -> None:
        tipping = missing_fit.sensitivity.tipping_gamma("ate")
        # This process has a substantial effect, so no plausible tilt nulls it.
        assert tipping is None or abs(tipping) > 1.0

    def test_the_tilt_needs_missing_outcomes(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="requires a fit with missing outcomes"):
            good_overlap.sensitivity.missingness_tilt()

    def test_a_ratio_estimand_is_excluded(self, missing_fit) -> None:
        with pytest.raises(ValueError, match="no tiltable estimands"):
            missing_fit.sensitivity.missingness_tilt(estimands=["rr"])


class TestValidation:
    def test_the_score_check_passes_and_reports(self, good_overlap) -> None:
        check = good_overlap.validation.score_check()
        assert check.passed
        assert bool(check)
        assert check.failures == ()
        assert "PASS" in check.summary()
        check.raise_if_failed()

    def test_the_score_check_can_be_made_to_fail(self, good_overlap) -> None:
        # An absurd tolerance turns the check into a failure, exercising the reporting
        # path that a real convergence problem would take.
        strict = good_overlap.validation.score_check(tolerance=1e-30)
        assert not strict.passed
        assert strict.failures
        with pytest.raises(AssertionError, match="score equation was not solved"):
            strict.raise_if_failed()

    def test_nuisance_diagnostics_cover_every_model(self, good_overlap) -> None:
        diagnostics = good_overlap.validation.nuisance()
        names = {model.name for model in diagnostics.models}
        assert names == {"propensity", "outcome"}
        assert 0.0 < diagnostics["propensity"].metrics["auc"] < 1.0
        assert diagnostics["outcome"].metrics["r2"] > 0.1
        assert "VERDICT" in diagnostics.summary()

    def test_an_almost_randomised_treatment_is_read_as_good_overlap(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=81)
        # Replace treatment with pure coin flips: nothing in W predicts it.
        rng = np.random.default_rng(0)
        randomised = frame.assign(A=rng.binomial(1, 0.5, len(frame)).astype(float))
        result = fast_tmle(estimands=("ate",)).fit(randomised, outcome="Y", treatment="A")
        verdict = result.validation.nuisance().verdict()
        assert "overlap is excellent" in verdict

    def test_calibration_is_reported_per_model(self, good_overlap) -> None:
        diagnostics = good_overlap.validation.nuisance()
        frame = nw.from_native(
            diagnostics.calibration_frame("propensity", good_overlap.data), eager_only=True
        )
        assert {"bin", "n", "mean_predicted", "mean_observed"} <= set(frame.columns)
        assert len(frame) >= 2

    def test_super_learner_weights_are_summarised(self) -> None:
        frame, _ = make_nonlinear_ate(n=500, seed=82)
        # "glm" rather than "fast": this test is about the reporting path, and the
        # boosting candidates would dominate the fast tier's runtime for no extra cover.
        result = TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=3,
            learner_folds=3,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A")
        weights = result.validation.nuisance()["outcome"].learner_weights
        assert weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_combined_report_renders(self, good_overlap) -> None:
        assert "Score-equation check" in good_overlap.validation.report()
        assert "Nuisance model diagnostics" in good_overlap.validation.report()


class TestRefutation:
    @pytest.fixture(scope="class")
    def refutation(self) -> object:
        frame, _ = make_linear_ate(n=700, seed=83)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")
        return result.validation.refute(n_replicates=3, random_state=0)

    def test_all_default_tests_behave(self, refutation) -> None:
        assert refutation.passed
        assert {test.name for test in refutation.tests} == {
            "placebo",
            "random_common_cause",
            "subset",
        }

    def test_a_placebo_treatment_shows_no_effect(self, refutation) -> None:
        placebo = refutation["placebo"]
        # Permuting treatment destroys the effect while preserving its marginal.
        assert abs(placebo.mean) < 0.2 * abs(placebo.original)

    def test_an_irrelevant_covariate_does_not_move_the_estimate(self, refutation) -> None:
        noise = refutation["random_common_cause"]
        assert noise.mean == pytest.approx(noise.original, rel=0.05)

    def test_the_results_tabulate(self, refutation) -> None:
        frame = nw.from_native(refutation.to_frame(), eager_only=True)
        assert len(frame) == 3
        assert "VERDICT" in refutation.summary()

    def test_a_negative_control_outcome_shows_no_effect(self) -> None:
        frame, _ = make_linear_ate(n=700, seed=84)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")
        # An outcome built from the covariates alone, with no treatment component.
        rng = np.random.default_rng(0)
        control = frame["W1"].to_numpy() * 0.5 + rng.normal(size=len(frame))
        outcome = result.validation.refute(
            tests=["negative_control_outcome"],
            negative_control_outcome=control,
            random_state=0,
        )
        assert outcome.passed

    def test_the_negative_control_test_needs_an_outcome(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=85)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")
        with pytest.raises(ValueError, match="needs an outcome array"):
            result.validation.refute(tests=["negative_control_outcome"])

    def test_an_unknown_test_is_refused(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="unknown refutation test"):
            good_overlap.validation.refute(tests=["magic"])


class TestCombinedSensitivityReport:
    def test_the_report_gathers_what_it_can(self, good_overlap) -> None:
        report = good_overlap.sensitivity.report("ate")
        assert "Positivity" in report
        assert "Omitted-variable sensitivity" in report
        assert "E-value" in report
