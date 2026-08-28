"""Repeated cross-fitting: robust aggregation over fold draws.

``repeats=R`` runs the whole construction over ``R`` independent splits and reports
``median_r psi_r`` with within-plus-between variance. What is checked here is what can
be checked *exactly*, which is nearly all of it:

* ``repeats=1`` is bit-for-bit an ordinary fit -- the regression guard;
* the median rule itself, on hand-built estimates, including the log scale for ratios;
* the draws differ at *every* stage of the split, and are reproducible given a seed;
* the reported variance includes within-draw uncertainty and between-draw displacement;
* under ``cv_evaluation`` the same rule reads each draw's cross-validated variance;
* the refused combinations, and why.

The one claim that is *not* here is that repeats reduce the across-seed spread of ``psi``.
That is a statement about repeated sampling, so it belongs in the slow tier beside the
coverage studies, not in a fast test that would be a coin flip on a lucky seed.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_binary_outcome, make_linear_ate, make_missing_outcome
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.inference.cluster import cross_validated_variance, influence_variance
from cleverly.inference.influence import ParameterEstimate, make_estimate, median_estimates
from cleverly.provenance import fingerprint_array
from cleverly.sensitivity import missingness_tilt, positivity_report, truncation_curve
from cleverly.targets import parameter_stem
from cleverly.validation import score_check
from tests.conftest import FAST_KWARGS, fast_tmle

COLUMNS: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2", "W3"]}

#: Small and parametric: every assertion below is exact or structural, so the sample only
#: has to be large enough to fit five folds of a GLM.  Three repeats rather than two, so a
#: median is distinguishable from a first-or-last.
N = 400
REPEATS = 3

#: The canonical construction includes ATT, whose fold-specific conditioning population
#: makes fold-wise evaluation differ from pooled. Ratio parameters are deliberately absent:
#: their fold-varying gradient needs a separate canonical targeting score.
CANONICAL: dict[str, Any] = {
    "cv_evaluation": True,
    "estimands": ["ate", "att"],
}


@pytest.fixture(scope="module")
def frame() -> Any:
    return make_linear_ate(n=N, seed=11)[0]


@pytest.fixture(scope="module")
def binary_frame() -> Any:
    return make_binary_outcome(n=N, seed=17)[0]


@pytest.fixture(scope="module")
def canonical(binary_frame: Any) -> Any:
    """One repeated fold-evaluated CV-TMLE, shared by everything that reads one."""
    return fast_tmle(repeats=REPEATS, **CANONICAL).fit(binary_frame, **COLUMNS).single()


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


class TestTheMedianRule:
    """``median_estimates`` on hand-built inputs, where the right answer is arithmetic."""

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
        assert median_estimates([report]) is not None
        assert median_estimates([report])["ate"] is report["ate"]

    def test_the_point_estimate_is_the_median(self) -> None:
        reports = [
            {"ate": self._estimate("ate", psi, np.full(6, psi - 2.0))} for psi in (1.0, 2.0, 6.0)
        ]
        assert median_estimates(reports)["ate"].psi == pytest.approx(2.0)

    def test_the_curve_comes_from_the_draw_at_the_median_point(self) -> None:
        curves = [np.full(4, value) for value in (1.0, 2.0, 3.0)]
        reports = [
            {"ate": self._estimate("ate", point, curve)}
            for point, curve in zip((1.0, 5.0, 2.0), curves, strict=True)
        ]
        np.testing.assert_array_equal(median_estimates(reports)["ate"].influence_curve, curves[2])

    def test_even_repeats_average_the_two_central_curves(self) -> None:
        curves = [np.full(4, value) for value in (1.0, 3.0)]
        reports = [
            {"ate": self._estimate("ate", point, curve)}
            for point, curve in zip((1.0, 3.0), curves, strict=True)
        ]
        np.testing.assert_array_equal(
            median_estimates(reports)["ate"].influence_curve, np.full(4, 2.0)
        )

    def test_the_variance_is_the_median_within_plus_between_quantity(self) -> None:
        reports = [
            {"ate": self._estimate("ate", point, curve)}
            for point, curve in zip(
                (1.0, 2.0, 6.0),
                (
                    np.array([2.0, -2.0, 1.0, -1.0]),
                    np.array([1.0, -1.0, 0.5, -0.5]),
                    np.array([3.0, -3.0, 1.5, -1.5]),
                ),
                strict=True,
            )
        ]
        combined = median_estimates(reports)["ate"]
        expected = np.median(
            [report["ate"].variance + (report["ate"].psi - 2.0) ** 2 for report in reports]
        )
        assert combined.variance == pytest.approx(float(expected))

    def test_a_ratio_uses_the_median_on_the_log_scale(self) -> None:
        reports = [
            {"rr": self._estimate("rr", psi, np.zeros(4), scale="ratio")}
            for psi in (2.0, 8.0, 32.0)
        ]
        combined = median_estimates(reports)["rr"]
        assert combined.psi == pytest.approx(8.0)
        assert combined.log_psi is not None
        assert combined.psi == pytest.approx(float(np.exp(combined.log_psi)))

    def test_an_estimand_missing_from_a_draw_is_dropped_loudly(self) -> None:
        reports = [
            {
                "ate": self._estimate("ate", 1.0, np.zeros(4)),
                "att": self._estimate("att", 2.0, np.zeros(4)),
            },
            {"ate": self._estimate("ate", 3.0, np.zeros(4))},
        ]
        with pytest.warns(UserWarning, match="some cross-fitting repeats but not"):
            combined = median_estimates(reports)
        assert "att" not in combined
        assert combined["ate"].psi == pytest.approx(2.0)

    def test_no_repeats_is_an_error_rather_than_an_empty_report(self) -> None:
        with pytest.raises(ValueError, match="at least one repeat"):
            median_estimates([])


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


class TestTheMedianReportStaysCoherent:
    """The median point, adjusted variance, and per-draw score checks agree."""

    def test_the_reported_estimate_is_the_median_of_the_draws(
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
        assert repeated.psi("ate") == pytest.approx(float(np.median(per_draw)))

    def test_the_variance_uses_every_draw(self, repeated: Any) -> None:
        reports = []
        for repeat in repeated.repeats:
            reports.append(
                repeated.estimator.retarget(
                    repeated.data,
                    repeat.nuisance,
                    estimands=("ate",),
                    g_bounds=repeated.config.g_bounds,
                    g_bounds_conditional=repeated.config.g_bounds_conditional,
                )[0]["ate"]
            )
        centre = float(np.median([report.psi for report in reports]))
        expected = float(
            np.median([report.variance + (report.psi - centre) ** 2 for report in reports])
        )
        assert repeated["ate"].variance == pytest.approx(expected)

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


def _per_draw_detail(result: Any) -> list[Any]:
    """Each draw's own ``CVTargeting``, re-solved from its cached nuisance fits.

    Cheap -- ``_retarget_detailed`` re-runs the targeting step only -- and it is the only
    way to recover the per-draw reports behind the median result, since the
    result keeps the fold-level detail of the first draw alone.
    """
    estimands = tuple(dict.fromkeys(parameter_stem(name) for name in result.estimates))
    details = []
    for repeat in result.repeats:
        _, _, detail = result.estimator._retarget_detailed(
            result.data,
            repeat.nuisance,
            estimands=estimands,
            g_bounds=result.config.g_bounds,
            g_bounds_conditional=result.config.g_bounds_conditional,
        )
        details.append(detail)
    return details


class TestRepeatedCanonicalCVTMLE:
    """``repeats=R`` with ``cv_evaluation=True``: median-combine fold-evaluated fits.

    The combination used to be refused, on the ground that the cross-validated variance is
    Each draw contributes its canonical point and cross-validated variance. The repeated
    report applies the same median within-plus-between rule as the stacked path.
    """

    def test_one_repeat_is_bit_for_bit_an_ordinary_canonical_fit(self, binary_frame: Any) -> None:
        # The regression guard, as for the pooled path: repeats=1 must not merely agree
        # with the old code path, it must be it.
        plain = fast_tmle(**CANONICAL).fit(binary_frame, **COLUMNS).single()
        explicit = fast_tmle(repeats=1, **CANONICAL).fit(binary_frame, **COLUMNS).single()
        for name in plain.estimates:
            assert explicit[name].psi == plain[name].psi
            assert explicit[name].variance == plain[name].variance
            np.testing.assert_array_equal(
                explicit[name].influence_curve, plain[name].influence_curve
            )

    def test_the_estimate_is_the_median_of_the_draws_canonical_estimates(
        self, canonical: Any
    ) -> None:
        details = _per_draw_detail(canonical)
        for name, estimate in canonical.estimates.items():
            parts = [detail.canonical[name] for detail in details]
            expected = (
                float(np.exp(np.median([part.log_psi for part in parts])))
                if estimate.scale == "ratio"
                else float(np.median([part.psi for part in parts]))
            )
            assert estimate.psi == pytest.approx(expected, rel=1e-12)

    def test_the_variance_is_the_median_within_plus_between_quantity(self, canonical: Any) -> None:
        details = _per_draw_detail(canonical)
        for name, estimate in canonical.estimates.items():
            points = np.asarray([detail.canonical[name].psi for detail in details])
            centre = float(np.median(points))
            expected = float(
                np.median(
                    [
                        detail.variance[name] + (point - centre) ** 2
                        for detail, point in zip(details, points, strict=True)
                    ]
                )
            )
            assert estimate.variance == pytest.approx(expected)

    def test_it_is_not_the_variance_of_the_median_draw_curve(self, canonical: Any) -> None:
        for name, estimate in canonical.estimates.items():
            from_curve = influence_variance(estimate.influence_curve)
            assert estimate.variance != pytest.approx(from_curve, rel=1e-6), name

    def test_the_canonical_report_is_the_one_the_fit_headlines(self, canonical: Any) -> None:
        detail = canonical.cv_targeting
        assert detail.repeats == REPEATS
        for name, estimate in canonical.estimates.items():
            assert detail.canonical[name].psi == estimate.psi
            assert detail.canonical[name].variance == estimate.variance

    def test_the_pooled_report_stitches_the_same_fold_weighted_update(
        self, binary_frame: Any, canonical: Any
    ) -> None:
        # ``detail.pooled`` changes only the evaluation of the fold-evaluated fit; it
        # deliberately does not rerun Levy's default row-weighted targeting loss. The
        # standalone default below is therefore close under balanced folds, but need not
        # be identical. Both pooled reports use the median within-plus-between rule.
        settings = {**CANONICAL, "cv_evaluation": False}
        levy = fast_tmle(repeats=REPEATS, **settings).fit(binary_frame, **COLUMNS).single()
        differs = []
        for name, estimate in levy.estimates.items():
            comparison = canonical.cv_targeting.pooled[name]
            assert np.isfinite(comparison.variance)
            assert abs(comparison.psi - estimate.psi) < 0.1 * estimate.std_error
            differs.append(comparison.psi != estimate.psi)
        assert any(differs)

    def test_the_two_reports_still_diverge_on_att(self, canonical: Any) -> None:
        # The pooled ATT conditions under the whole-sample arm share; canonical ATT
        # averages the fold-specific conditioning populations.
        detail = canonical.cv_targeting
        assert detail.canonical["ate"].psi == pytest.approx(detail.pooled["ate"].psi, rel=1e-9)
        assert detail.canonical["att"].psi != pytest.approx(detail.pooled["att"].psi, rel=1e-9)

    def test_the_fold_level_detail_describes_the_first_draw(self, canonical: Any) -> None:
        # Fold 3 of one draw is not fold 3 of another, so these are the only fields with
        # nothing to average along -- and a reader has to be able to tell which they are.
        first = _per_draw_detail(canonical)[0]
        assert canonical.cv_targeting.fold_epsilon == first.fold_epsilon
        assert canonical.cv_targeting.fold_estimates == first.fold_estimates
        assert "the fold columns describe the first" in canonical.cv_targeting.summary()

    def test_the_score_equation_is_still_solved(self, canonical: Any) -> None:
        check = score_check(canonical)
        assert check.passed, check.summary()

    def test_the_summary_names_the_estimator_and_the_variance_rule(self, canonical: Any) -> None:
        summary = canonical.summary()
        assert "fold-evaluated CV-TMLE" in summary
        assert "independent draws" in summary
        assert "split dispersion" in summary


class TestTheRejectedVarianceRule:
    """Why a combined curve gets no cross-validated variance of its own.

    Not merely that the partition would be an arbitrary pick among the ``R``: with equal
    fold sizes the partition makes no difference at all, so the number would be the pooled
    uncentred second moment wearing a cross-validated name.  On arrays, where it is
    arithmetic.
    """

    def test_equal_folds_make_the_partition_carry_no_information(self) -> None:
        curve = np.random.default_rng(0).normal(size=60)
        pooled = float(np.mean(curve**2)) / curve.size
        for seed in range(4):
            order = np.random.default_rng(seed).permutation(curve.size)
            partition = [order[start::4] for start in range(4)]
            assert {index.size for index in partition} == {15}
            assert cross_validated_variance(curve, partition) == pytest.approx(pooled, rel=1e-12)


class TestEveryStageOfTheSplitIsRedrawn:
    """A draw redraws the nested cross-validation too, not only the outer folds.

    Averaging over one stage of a randomised procedure while pinning the rest would leave
    every draw scoring its Super Learner candidates -- and, for C-TMLE, choosing where to
    stop -- against one fixed partition.
    """

    @staticmethod
    def _seeds_seen(where: str, name: str, monkeypatch: Any, key: str) -> list[Any]:
        # import_module rather than ``from cleverly.estimators import tmle``: the package
        # re-exports the *function* of that name, which shadows the module.
        module = import_module(where)
        seen: list[Any] = []
        original = getattr(module, name)

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get(key))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
        return seen

    def test_every_draw_resolves_its_learners_at_its_own_seed(
        self, frame: Any, monkeypatch: Any
    ) -> None:
        seen = self._seeds_seen(
            "cleverly.estimators.tmle", "resolve_learner", monkeypatch, "random_state"
        )
        estimator = fast_tmle(repeats=REPEATS, estimands=["ate"])
        result = estimator.fit(frame, **COLUMNS).single()
        assert set(seen) == set(estimator.crossfit_plan(result.data).seeds())
        assert len(set(seen)) == REPEATS

    def test_an_ordinary_fit_still_uses_the_estimators_own_random_state(
        self, frame: Any, monkeypatch: Any
    ) -> None:
        seen = self._seeds_seen(
            "cleverly.estimators.tmle", "resolve_learner", monkeypatch, "random_state"
        )
        fast_tmle(estimands=["ate"]).fit(frame, **COLUMNS)
        assert set(seen) == {FAST_KWARGS["random_state"]}

    def test_the_ctmle_selection_folds_follow_the_draw(self, frame: Any, monkeypatch: Any) -> None:
        from cleverly.estimators import CTMLE

        seen = self._seeds_seen(
            "cleverly.estimators.ctmle", "make_folds", monkeypatch, "random_state"
        )
        estimator = CTMLE(**{**FAST_KWARGS, "repeats": 2, "estimands": ["ate"]})
        result = estimator.fit(frame, **COLUMNS).single()
        assert set(seen) == set(estimator.crossfit_plan(result.data).seeds())


class TestTheSpreadAcrossDraws:
    """``repeat_spread`` -- how far the fold assignment moved the answer."""

    def test_it_is_the_standard_deviation_of_the_draws(self, repeated: Any) -> None:
        spread = repeated.repeat_spread()
        assert set(spread) == set(repeated.estimates)
        for name, value in spread.items():
            per_draw = [repeat.psi[name] for repeat in repeated.repeats]
            assert value == pytest.approx(float(np.std(per_draw, ddof=1)))
        # And the draws it is the spread of are the ones the report takes the median of.
        assert repeated.psi("ate") == pytest.approx(
            float(np.median([repeat.psi["ate"] for repeat in repeated.repeats]))
        )

    def test_one_draw_has_no_spread_to_report(self, once: Any) -> None:
        with pytest.raises(ValueError, match="moved between draws"):
            once.repeat_spread()

    def test_the_summary_shows_it_beside_the_standard_error(self, repeated: Any, once: Any) -> None:
        assert "split noise" in repeated.summary()
        assert "of std_err" in repeated.summary()
        assert "split noise" not in once.summary()

    def test_it_survives_the_round_trip(self, repeated: Any) -> None:
        reloaded = loads(dumps(repeated))
        assert reloaded.repeat_spread() == repeated.repeat_spread()


class TestWhatTheResultExposes:
    def test_nuisance_and_fluctuations_read_through_to_the_first_draw(self, repeated: Any) -> None:
        assert repeated.nuisance is repeated.repeats[0].nuisance
        assert repeated.fluctuations is repeated.repeats[0].fluctuations
        assert repeated.nuisances == tuple(r.nuisance for r in repeated.repeats)
        assert repeated.n_repeats == REPEATS

    def test_the_summary_says_the_fit_was_repeated(self, repeated: Any, once: Any) -> None:
        assert "median over 3 independent draws" in repeated.summary()
        assert "independent draws" not in once.summary()

    def test_the_declared_plan_records_the_count(self, repeated: Any) -> None:
        assert repeated.config.crossfit.repeats == REPEATS
        assert repeated.config.crossfit.repeated
        assert "median over 3 draws" in repeated.config.crossfit.describe()


class TestRefusedCombinations:
    def test_repeats_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="repeats must be at least 1"):
            fast_tmle(repeats=0)

    def test_repeats_needs_a_split_to_draw(self) -> None:
        with pytest.raises(ValueError, match="no split to draw"):
            fast_tmle(repeats=3, cross_fit=False)

    def test_fold_targeting_without_cv_evaluation_is_allowed(self, frame: Any) -> None:
        result = fast_tmle(repeats=2, targeting_scheme="fold").fit(frame, **COLUMNS).single()
        assert result.n_repeats == 2
        assert np.isfinite(result["ate"].std_error)

    def test_cv_evaluation_needs_cross_fitted_predictions(self) -> None:
        with pytest.raises(ValueError, match="cross_fit=True"):
            fast_tmle(repeats=3, cv_evaluation=True, cross_fit=False)

    def test_multiplier_bands_do_not_represent_the_median_estimator(self, frame: Any) -> None:
        with pytest.raises(ValueError, match="simultaneous=True is not defined"):
            fast_tmle(repeats=3, simultaneous=True, estimands=["ate"]).fit(frame, **COLUMNS)

    def test_post_fit_covariance_is_refused(self, repeated: Any) -> None:
        with pytest.raises(CapabilityError, match="median-combined repeats"):
            repeated.covariance()

    def test_post_fit_contrasts_are_refused(self, repeated: Any) -> None:
        with pytest.raises(CapabilityError, match="medians do not preserve"):
            repeated.contrast(lambda point: point[0] - point[1], ["ey1", "ey0"])


class TestTheSensitivityLayerFollowsTheDraws:
    def test_the_truncation_curve_combines_all_draws(self, repeated: Any) -> None:
        # At the bound the fit used, the swept estimate must reproduce the reported one.
        # It does only if the sweep combines the same way the fit did.
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
        report = repeated.diagnostics.nuisance_models()
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


class TestTheOmittedVariableBoundNeedsItsOwnMedianInfluenceFunction:
    @pytest.fixture(scope="class")
    def binary_fit(self) -> Any:
        frame, _ = make_binary_outcome(n=N, seed=17)
        return fast_tmle(repeats=2, estimands=["ate"]).fit(frame, **COLUMNS).single()

    def test_the_bound_is_refused_instead_of_combining_influence_terms(
        self, binary_fit: Any
    ) -> None:
        from cleverly.sensitivity.omitted_variable import sensitivity_elements

        with pytest.raises(CapabilityError, match="median-combined repeats"):
            sensitivity_elements(binary_fit, "ate")


class TestSerialization:
    def test_every_draw_survives_the_round_trip(self, repeated: Any) -> None:
        reloaded = loads(dumps(repeated))
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

    def test_the_live_estimator_carries_the_repeat_count(self, repeated: Any) -> None:
        reloaded = loads(dumps(repeated))
        assert reloaded.estimator.repeats == REPEATS

    def test_a_reloaded_fit_reproduces_its_own_report(self, repeated: Any) -> None:
        reloaded = loads(dumps(repeated))
        assert reloaded.psi("ate") == repeated.psi("ate")
        curve = truncation_curve(reloaded, bounds=[reloaded.config.g_bounds[0]], estimands=["ate"])
        assert float(curve["psi"][0]) == pytest.approx(reloaded.psi("ate"), rel=1e-9)


class TestVariantsInheritRepeats:
    def test_ctmle_repeats_its_selection_per_draw(self, frame: Any) -> None:
        # CTMLE overrides _nuisances alone, and the repeat loop sits around that method,
        # so this works without estimators/ctmle.py knowing repeats exist.
        from cleverly.estimators import CTMLE

        kwargs = {**FAST_KWARGS, "repeats": 2, "estimands": ["ate"]}
        result = CTMLE(**kwargs).fit(frame, **COLUMNS).single()
        assert result.n_repeats == 2
        selections = {tuple(repeat.nuisance.treatment_covariates) for repeat in result.repeats}
        assert selections  # a selection was made in each draw
        assert np.isfinite(result["ate"].std_error)

    def test_the_bootstrap_repeats_the_draws(self, frame: Any) -> None:
        # A replicate must resample the estimator that was reported -- the median of R
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
