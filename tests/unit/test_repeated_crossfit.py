"""Repeated cross-fitting: averaging away the fold draw.

``repeats=R`` runs the whole construction over ``R`` independent splits and reports
``mean_r psi_r`` with influence curve ``mean_r IC_r``.  What is checked here is what can
be checked *exactly*, which is nearly all of it:

* ``repeats=1`` is bit-for-bit an ordinary fit -- the regression guard;
* the averaging rule itself, on hand-built estimates, including the log scale for ratios;
* the draws differ, and are reproducible given a seed;
* the reported variance is the variance of the reported curve, so the delta method and
  the score diagnostic still hold after averaging;
* the two refused combinations, and why.

The one claim that is *not* here is that repeats reduce the across-seed spread of ``psi``.
That is a statement about repeated sampling, so it belongs in the slow tier beside the
coverage studies, not in a fast test that would be a coin flip on a lucky seed.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_binary_outcome, make_linear_ate, make_missing_outcome
from cleverly.estimators.serialize import result_from_dict, result_to_dict
from cleverly.inference.influence import ParameterEstimate, average_estimates, make_estimate
from cleverly.provenance import fingerprint_array
from cleverly.sensitivity import missingness_tilt, positivity_report, truncation_curve
from cleverly.validation import score_check
from tests.conftest import FAST_KWARGS, fast_tmle

COLUMNS: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2", "W3"]}

#: Small and parametric: every assertion below is exact or structural, so the sample only
#: has to be large enough to fit five folds of a GLM.  Three repeats rather than two, so a
#: mean is distinguishable from a first-or-last.
N = 400
REPEATS = 3


@pytest.fixture(scope="module")
def frame() -> Any:
    return make_linear_ate(n=N, seed=11)[0]


@pytest.fixture(scope="module")
def once(frame: Any) -> Any:
    return fast_tmle().fit(frame, **COLUMNS).single()


@pytest.fixture(scope="module")
def repeated(frame: Any) -> Any:
    """One repeated fit, shared by every test that only needs to read one."""
    return fast_tmle(repeats=REPEATS).fit(frame, **COLUMNS).single()


class TestOneRepeatIsAnOrdinaryFit:
    """The regression guard.

    Repeated cross-fitting is a new path through ``_fit_single``, and the cheapest way to
    know it did not disturb the old one is to require that ``repeats=1`` produces the same
    bits -- not the same numbers to a tolerance, the same bits.
    """

    def test_the_point_estimate_is_identical(self, frame: Any, once: Any) -> None:
        explicit = fast_tmle(repeats=1).fit(frame, **COLUMNS).single()
        for name in once.estimates:
            assert explicit.psi(name) == once.psi(name)

    def test_the_influence_curve_is_identical(self, frame: Any, once: Any) -> None:
        explicit = fast_tmle(repeats=1).fit(frame, **COLUMNS).single()
        for name in once.estimates:
            np.testing.assert_array_equal(
                explicit[name].influence_curve, once[name].influence_curve
            )

    def test_the_split_is_identical(self, frame: Any, once: Any) -> None:
        # Not implied by the estimates agreeing -- it is what makes them agree, and it is
        # the thing a spawned per-repeat seed would silently break.
        explicit = fast_tmle(repeats=1).fit(frame, **COLUMNS).single()
        np.testing.assert_array_equal(
            explicit.nuisance.folds.assignment, once.nuisance.folds.assignment
        )
        assert explicit.provenance.fold_fingerprint == once.provenance.fold_fingerprint

    def test_an_ordinary_fit_holds_exactly_one_repeat(self, once: Any) -> None:
        assert once.n_repeats == 1
        assert once.repeats[0].nuisance is once.nuisance
        assert once.repeats[0].fluctuations is once.fluctuations


class TestTheAveragingRule:
    """``average_estimates`` on hand-built inputs, where the right answer is arithmetic."""

    @staticmethod
    def _estimate(name: str, psi: float, ic: Any, scale: str = "difference") -> ParameterEstimate:
        log_psi = float(np.log(psi)) if scale == "ratio" else None
        return make_estimate(
            name,
            psi,
            np.asarray(ic, dtype=float),
            n=len(ic),
            scale=scale,  # type: ignore[arg-type]
            log_psi=log_psi,
        )

    def test_one_repeat_returns_its_input_untouched(self) -> None:
        report = {"ate": self._estimate("ate", 1.0, [0.5, -0.5, 1.0, -1.0])}
        assert average_estimates([report]) is not None
        assert average_estimates([report])["ate"] is report["ate"]

    def test_the_point_estimate_is_the_mean(self) -> None:
        reports = [
            {"ate": self._estimate("ate", psi, np.full(6, psi - 2.0))} for psi in (1.0, 2.0, 6.0)
        ]
        assert average_estimates(reports)["ate"].psi == pytest.approx(3.0)

    def test_the_influence_curve_is_the_elementwise_mean(self) -> None:
        curves = [np.array([1.0, -1.0, 0.0]), np.array([3.0, 1.0, 2.0])]
        reports = [{"ate": self._estimate("ate", 1.0, curve)} for curve in curves]
        np.testing.assert_allclose(
            average_estimates(reports)["ate"].influence_curve, np.array([2.0, 0.0, 1.0])
        )

    def test_the_variance_comes_from_the_averaged_curve(self) -> None:
        # The whole coherence claim: not a pooled per-draw variance, but the variance of
        # the curve that is actually reported. Averaging independent-ish curves shrinks
        # them, and the reported variance has to follow.
        curves = [np.array([2.0, -2.0, 1.0, -1.0]), np.array([-2.0, 2.0, -1.0, 1.0])]
        reports = [{"ate": self._estimate("ate", 1.0, curve)} for curve in curves]
        averaged = average_estimates(reports)["ate"]
        expected = make_estimate("ate", 1.0, np.mean(curves, axis=0), n=4)
        assert averaged.variance == pytest.approx(expected.variance)
        assert averaged.variance == pytest.approx(0.0)

    def test_a_ratio_is_averaged_on_the_log_scale(self) -> None:
        reports = [
            {"rr": self._estimate("rr", psi, np.zeros(4), scale="ratio")} for psi in (2.0, 8.0)
        ]
        averaged = average_estimates(reports)["rr"]
        # The geometric mean, which is where the influence curve and the Wald interval
        # live -- not (2 + 8) / 2.
        assert averaged.psi == pytest.approx(4.0)
        assert averaged.log_psi is not None
        assert averaged.psi == pytest.approx(float(np.exp(averaged.log_psi)))

    def test_an_estimand_missing_from_a_draw_is_dropped_loudly(self) -> None:
        reports = [
            {
                "ate": self._estimate("ate", 1.0, np.zeros(4)),
                "att": self._estimate("att", 2.0, np.zeros(4)),
            },
            {"ate": self._estimate("ate", 3.0, np.zeros(4))},
        ]
        with pytest.warns(UserWarning, match="some cross-fitting repeats but not"):
            averaged = average_estimates(reports)
        assert "att" not in averaged
        assert averaged["ate"].psi == pytest.approx(2.0)

    def test_no_repeats_is_an_error_rather_than_an_empty_report(self) -> None:
        with pytest.raises(ValueError, match="at least one repeat"):
            average_estimates([])


class TestTheDrawsAreDistinctAndReproducible:
    def test_every_draw_gets_its_own_split(self, repeated: Any) -> None:
        assignments = {tuple(repeat.nuisance.folds.assignment) for repeat in repeated.repeats}
        assert len(assignments) == REPEATS

    def test_a_draw_carries_the_split_that_made_it(self, repeated: Any) -> None:
        for repeat in repeated.repeats:
            assert repeat.folds is repeat.nuisance.folds

    def test_the_same_seed_gives_the_same_draws(self, frame: Any, repeated: Any) -> None:
        again = fast_tmle(repeats=REPEATS).fit(frame, **COLUMNS).single()
        for first, second in zip(repeated.repeats, again.repeats, strict=True):
            np.testing.assert_array_equal(
                first.nuisance.folds.assignment, second.nuisance.folds.assignment
            )
        assert again.psi("ate") == repeated.psi("ate")

    def test_a_different_seed_gives_different_draws(self, frame: Any, repeated: Any) -> None:
        other = fast_tmle(repeats=REPEATS, random_state=99).fit(frame, **COLUMNS).single()
        assert not np.array_equal(
            other.repeats[0].nuisance.folds.assignment,
            repeated.repeats[0].nuisance.folds.assignment,
        )

    def test_the_provenance_fingerprint_covers_every_draw(self, repeated: Any) -> None:
        # A repeated fit is reproducible only if *all* R splits are, so a digest of one
        # of them would state a guarantee the fit does not make.
        assignments = [repeat.nuisance.folds.assignment for repeat in repeated.repeats]
        assert repeated.provenance.fold_fingerprint == fingerprint_array(*assignments)
        assert repeated.provenance.fold_fingerprint != fingerprint_array(assignments[0])


class TestTheAveragedReportStaysCoherent:
    """What averaging the curve buys: everything computed *from* the curve still holds."""

    def test_the_reported_estimate_is_the_mean_of_the_draws(
        self, frame: Any, repeated: Any
    ) -> None:
        estimator = fast_tmle(repeats=REPEATS)
        per_draw = [
            estimator.retarget(
                repeated.data,
                repeat.nuisance,
                estimands=("ate",),
                g_bounds=repeated.config.g_bounds,
                g_bounds_conditional=repeated.config.g_bounds_conditional,
            )[0]["ate"].psi
            for repeat in repeated.repeats
        ]
        assert repeated.psi("ate") == pytest.approx(float(np.mean(per_draw)))

    def test_the_delta_method_identity_survives_averaging(self, repeated: Any) -> None:
        # IC_ate == IC_ey1 - IC_ey0 holds per draw and is linear, so it must hold after
        # averaging. Exact, and it is what would break if the curves were pooled by some
        # other rule than the mean.
        np.testing.assert_allclose(
            repeated["ate"].influence_curve,
            repeated["ey1"].influence_curve - repeated["ey0"].influence_curve,
            atol=1e-12,
        )

    def test_the_variance_is_the_variance_of_the_reported_curve(self, repeated: Any) -> None:
        rebuilt = make_estimate(
            "ate", repeated.psi("ate"), repeated["ate"].influence_curve, n=repeated.n
        )
        assert repeated["ate"].variance == pytest.approx(rebuilt.variance)

    def test_the_score_equation_is_still_solved(self, repeated: Any) -> None:
        check = score_check(repeated)
        assert check.passed, check.summary()

    def test_the_score_check_examines_every_draw(self, repeated: Any, once: Any) -> None:
        # A draw whose targeting step failed contributes to the reported estimate exactly
        # as the others do, so checking one of R would let that failure through.
        names = [row.name for row in score_check(repeated).rows if row.kind == "fluctuation"]
        assert len(names) == REPEATS * len(repeated.fluctuations)
        assert "mean[draw 2]" in names
        # An ordinary fit's rows keep their plain group names, unsuffixed.
        plain = [row.name for row in score_check(once).rows if row.kind == "fluctuation"]
        assert plain == list(once.fluctuations)
        assert not any("[draw" in name for name in plain)


class TestWhatTheResultExposes:
    def test_nuisance_and_fluctuations_read_through_to_the_first_draw(self, repeated: Any) -> None:
        assert repeated.nuisance is repeated.repeats[0].nuisance
        assert repeated.fluctuations is repeated.repeats[0].fluctuations
        assert repeated.nuisances == tuple(r.nuisance for r in repeated.repeats)
        assert repeated.n_repeats == REPEATS

    def test_the_summary_says_the_fit_was_repeated(self, repeated: Any, once: Any) -> None:
        assert "averaged over 3 independent draws" in repeated.summary()
        assert "independent draws" not in once.summary()

    def test_the_declared_plan_records_the_count(self, repeated: Any) -> None:
        assert repeated.config.crossfit.repeats == REPEATS
        assert repeated.config.crossfit.repeated
        assert "averaged over 3 draws" in repeated.config.crossfit.describe()


class TestRefusedCombinations:
    def test_repeats_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="repeats must be at least 1"):
            fast_tmle(repeats=0)

    def test_repeats_needs_a_split_to_draw(self) -> None:
        with pytest.raises(ValueError, match="no split to draw"):
            fast_tmle(repeats=3, cross_fit=False)

    def test_repeats_and_cv_evaluation_name_the_fold_partition(self) -> None:
        with pytest.raises(ValueError, match="defined by a fold partition"):
            fast_tmle(repeats=3, targeting_scheme="fold", cv_evaluation=True)

    def test_fold_targeting_without_cv_evaluation_is_allowed(self, frame: Any) -> None:
        # Only the cross-validated *variance* is incompatible; fold-wise targeting is
        # just a different fluctuation and repeats fine.
        result = fast_tmle(repeats=2, targeting_scheme="fold").fit(frame, **COLUMNS).single()
        assert result.n_repeats == 2
        assert np.isfinite(result["ate"].std_error)


class TestTheSensitivityLayerFollowsTheDraws:
    def test_the_truncation_curve_averages_over_them(self, repeated: Any) -> None:
        # At the bound the fit used, the swept estimate must reproduce the reported one.
        # It does only if the sweep averages the same way the fit did.
        curve = truncation_curve(repeated, bounds=[repeated.config.g_bounds[0]], estimands=["ate"])
        assert float(curve["psi"][0]) == pytest.approx(repeated.psi("ate"), rel=1e-9)

    def test_a_swept_bound_moves_all_the_draws(self, repeated: Any) -> None:
        curve = truncation_curve(repeated, bounds=[0.001, 0.2], estimands=["ate"])
        assert len(curve["psi"]) == 2

    def test_the_positivity_report_says_which_draw_it_describes(
        self, repeated: Any, once: Any
    ) -> None:
        assert positivity_report(repeated).n_repeats == REPEATS
        assert "draw 1 of 3" in positivity_report(repeated).summary()
        assert "draw 1 of" not in positivity_report(once).summary()

    def test_the_nuisance_diagnostics_say_so_too(self, repeated: Any) -> None:
        report = repeated.validation.nuisance()
        assert report.n_repeats == REPEATS
        assert "draw 1 of 3" in report.summary()


class TestTheMnarTiltFollowsTheDraws:
    @pytest.fixture(scope="class")
    def missing_fit(self) -> Any:
        frame, _ = make_missing_outcome(n=N, seed=13)
        return (
            fast_tmle(repeats=2, estimands=["ate", "ey1", "ey0"])
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], delta="Delta")
            .single()
        )

    def test_the_tilt_at_zero_reproduces_the_reported_estimate(self, missing_fit: Any) -> None:
        # gamma = 0 is MAR, so the tilt is the identity and the curve must pass through
        # the fit's own psi. Reading one draw's targeted Qbar would put it elsewhere --
        # the curve would step at its own origin.
        tilt = missingness_tilt(missing_fit, gamma=[0.0], estimands=["ate"])
        assert float(tilt["psi"][0]) == pytest.approx(missing_fit.psi("ate"), rel=1e-9)

    def test_the_tilt_still_moves_the_estimate(self, missing_fit: Any) -> None:
        tilt = missingness_tilt(missing_fit, gamma=[0.0, 2.0], estimands=["ate"])
        assert float(tilt["psi"][0]) != float(tilt["psi"][1])


class TestTheOmittedVariableBoundFollowsTheDraws:
    @pytest.fixture(scope="class")
    def binary_fit(self) -> Any:
        frame, _ = make_binary_outcome(n=N, seed=17)
        return fast_tmle(repeats=2, estimands=["ate"]).fit(frame, **COLUMNS).single()

    def test_the_bound_is_the_mean_of_the_draws(self, binary_fit: Any) -> None:
        from cleverly.sensitivity.omitted_variable import _elements_for, sensitivity_elements

        averaged = sensitivity_elements(binary_fit, "ate")
        per_draw = [
            _elements_for(binary_fit, repeat, "ate", "auto").max_bias
            for repeat in binary_fit.repeats
        ]
        assert averaged.max_bias == pytest.approx(float(np.mean(per_draw)))

    def test_it_reads_each_draws_own_regression_and_mechanism(self, binary_fit: Any) -> None:
        # Crossing them would still produce a number; this pins that the pairing is the
        # one RepeatFit holds rather than an incidental zip.
        from cleverly.sensitivity.omitted_variable import _elements_for

        first = _elements_for(binary_fit, binary_fit.repeats[0], "ate", "auto")
        second = _elements_for(binary_fit, binary_fit.repeats[1], "ate", "auto")
        assert first.sigma2 != second.sigma2


class TestSerialization:
    def test_every_draw_survives_the_round_trip(self, repeated: Any) -> None:
        reloaded = result_from_dict(*result_to_dict(repeated))
        assert reloaded.n_repeats == REPEATS
        assert reloaded.config.crossfit.repeats == REPEATS
        for original, restored in zip(repeated.repeats, reloaded.repeats, strict=True):
            np.testing.assert_array_equal(
                original.nuisance.propensity.values, restored.nuisance.propensity.values
            )
            np.testing.assert_array_equal(
                original.nuisance.folds.assignment, restored.nuisance.folds.assignment
            )
            np.testing.assert_array_equal(
                original.fluctuations["mean"].epsilon, restored.fluctuations["mean"].epsilon
            )

    def test_the_recipe_carries_the_repeat_count(self, repeated: Any) -> None:
        # A reloaded result rebuilds its estimator from the recipe, and refit() then
        # re-runs the whole fit. A recipe that dropped repeats= would quietly refit a
        # single draw under the same name -- the failure is silent, so it is pinned here.
        from cleverly.estimators.recipe import TMLERecipe

        recipe = TMLERecipe.from_estimator(repeated.estimator)
        assert TMLERecipe.from_dict(recipe.to_dict()).build().repeats == REPEATS

    def test_a_reloaded_fit_reproduces_its_own_report(self, repeated: Any) -> None:
        reloaded = result_from_dict(*result_to_dict(repeated))
        assert reloaded.psi("ate") == repeated.psi("ate")
        curve = truncation_curve(reloaded, bounds=[reloaded.config.g_bounds[0]], estimands=["ate"])
        assert float(curve["psi"][0]) == pytest.approx(reloaded.psi("ate"), rel=1e-9)


class TestVariantsInheritRepeats:
    def test_ctmle_repeats_its_selection_per_draw(self, frame: Any) -> None:
        # CTMLE overrides _nuisances alone, and the repeat loop sits around that method,
        # so this works without estimators/ctmle.py knowing repeats exist.
        from cleverly import CTMLE

        kwargs = {**FAST_KWARGS, "repeats": 2, "estimands": ["ate"]}
        result = CTMLE(**kwargs).fit(frame, **COLUMNS).single()
        assert result.n_repeats == 2
        selections = {tuple(repeat.nuisance.treatment_covariates) for repeat in result.repeats}
        assert selections  # a selection was made in each draw
        assert np.isfinite(result["ate"].std_error)

    def test_the_bootstrap_repeats_the_draws(self, frame: Any) -> None:
        # A replicate must resample the estimator that was reported -- the average of R
        # draws -- rather than a single draw, whose fold noise the report does not carry.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                fast_tmle(repeats=2, n_bootstrap=4, estimands=["ate"])
                .fit(frame, **COLUMNS)
                .single()
            )
        assert result["ate"].bootstrap is not None
        assert np.isfinite(result["ate"].bootstrap.std_error)
