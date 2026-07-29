"""The cross-validated targeting step, and the variance that goes with it.

CV-TMLE (Zheng & van der Laan, 2011) has three parts, and this package lets you take
the first without the other two.  The fluctuation is solved separately within each
validation fold, against nuisance predictions from models that never saw those rows
(``targeting_scheme="fold"``); the parameter is then evaluated fold by fold and paired
with the cross-validated variance only if you also ask for ``cv_evaluation=True``.
The difference between the two reports is not cosmetic and is pinned below.

Everything asserted here is an exact algebraic consequence of the construction, so
these tests fail deterministically rather than statistically.  The claim CV-TMLE
actually exists to make -- that it keeps nominal coverage where a pooled fit does not
-- needs replications and lives in the slow tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.estimators.tmle import _average_over_folds
from cleverly.fluctuation import restrict
from cleverly.fluctuation.submodel import mean_submodel
from cleverly.inference import cross_validated_variance, influence_variance
from cleverly.learners.crossfit import Folds, make_folds
from tests.conftest import FAST_KWARGS

#: Every estimand the binary-outcome fixture supports: the three that are linear in
#: the targeted predictions and the four that are not.
ALL_BINARY = ("ey1", "ey0", "ate", "rr", "or", "att", "atc")
LINEAR = ("ey1", "ey0", "ate")
NONLINEAR = ("rr", "or", "att", "atc")


@pytest.fixture(scope="module")
def cv_fit() -> object:
    frame, _ = make_linear_ate(n=600, seed=17)
    return TMLE(**{**FAST_KWARGS, "targeting_scheme": "fold", "estimands": ("ate", "ey1")}).fit(
        frame, outcome="Y", treatment="A"
    )


@pytest.fixture(scope="module")
def binary_frame() -> object:
    # n divisible by the 5 fast-tier folds, so the fold weights cancel exactly and any
    # gap between the two reports is the construction rather than unequal fold sizes.
    return make_binary_outcome(n=800, seed=3)[0]


@pytest.fixture(scope="module")
def pooled_report(binary_frame) -> object:
    settings = {**FAST_KWARGS, "targeting_scheme": "fold", "estimands": ALL_BINARY}
    return TMLE(**settings).fit(binary_frame, outcome="Y", treatment="A")


@pytest.fixture(scope="module")
def canonical_report(binary_frame) -> object:
    settings = {
        **FAST_KWARGS,
        "targeting_scheme": "fold",
        "cv_evaluation": True,
        "estimands": ALL_BINARY,
    }
    return TMLE(**settings).fit(binary_frame, outcome="Y", treatment="A")


class TestFoldWiseTargeting:
    def test_each_fold_solves_its_own_score_equation(self, cv_fit) -> None:
        # The defining property. A pooled fit only zeroes the score over the whole
        # sample; fold-wise targeting zeroes it inside every fold separately, and the
        # pooled score is then zero as a sum of zeros rather than by cancellation.
        for fluctuation in cv_fit.fluctuations.values():
            assert fluctuation.folds, "fold-wise targeting must record its folds"
            for record in fluctuation.folds:
                assert np.max(np.abs(record.score)) < 1e-9

    def test_the_folds_partition_the_sample(self, cv_fit) -> None:
        for fluctuation in cv_fit.fluctuations.values():
            covered = np.concatenate([record.index for record in fluctuation.folds])
            assert np.array_equal(np.sort(covered), np.arange(cv_fit.n))

    def test_the_reported_epsilon_is_the_mass_weighted_fold_average(self, cv_fit) -> None:
        for fluctuation in cv_fit.fluctuations.values():
            sizes = np.array([record.n for record in fluctuation.folds], dtype=float)
            stacked = np.vstack([record.epsilon for record in fluctuation.folds])
            expected = np.average(stacked, axis=0, weights=sizes)
            # Equal, unweighted folds here, so the mass weights are the fold sizes.
            assert fluctuation.epsilon == pytest.approx(expected, abs=1e-12)

    def test_a_pooled_fit_records_no_fold_detail(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=18)
        result = TMLE(**{**FAST_KWARGS, "estimands": ("ate",)}).fit(
            frame, outcome="Y", treatment="A"
        )
        assert result.cv_targeting is None
        assert all(not f.folds for f in result.fluctuations.values())

    def test_one_fold_reduces_to_a_pooled_fit(self) -> None:
        # With a single fold there is no validation split to target within, so the two
        # schemes must agree exactly rather than merely closely -- and the fallback has
        # to be announced rather than silently honoured, because a caller who asked for
        # a CV-TMLE and got a cross-fitted TMLE should hear about it.
        frame, _ = make_linear_ate(n=400, seed=19)
        settings = {**FAST_KWARGS, "estimands": ("ate",), "cross_fit": False}
        pooled = TMLE(**settings).fit(frame, outcome="Y", treatment="A")
        with pytest.warns(UserWarning, match="falling back to pooled targeting"):
            fold_wise = TMLE(**{**settings, "targeting_scheme": "fold"}).fit(
                frame, outcome="Y", treatment="A"
            )
        assert fold_wise.psi("ate") == pooled.psi("ate")
        assert fold_wise["ate"].std_error == pooled["ate"].std_error
        assert fold_wise.cv_targeting is None


class TestCrossValidatedVariance:
    def test_it_matches_a_longhand_loop(self) -> None:
        rng = np.random.default_rng(3)
        ic = rng.normal(size=60)
        folds = make_folds(60, 5, random_state=0)
        index = [test for _, test in folds]

        expected = np.mean([np.mean(ic[test] ** 2) for test in index]) / 60
        assert cross_validated_variance(ic, index) == pytest.approx(expected, rel=1e-12)

    def test_equal_folds_reduce_to_the_pooled_second_moment(self) -> None:
        # n divisible by the fold count, so the fold weights cancel exactly. This is
        # the algebraic reason the CV and pooled variances agree on a solved fit.
        rng = np.random.default_rng(4)
        ic = rng.normal(size=100)
        index = [np.arange(100)[k::5] for k in range(5)]
        assert cross_validated_variance(ic, index) == pytest.approx(
            float(np.mean(ic**2) / 100), rel=1e-12
        )

    def test_it_tracks_the_pooled_variance_on_a_real_fit(self, cv_fit) -> None:
        # Not equal by construction -- these are two variances of two estimators, and
        # the pooled one centres and uses ddof=1 -- but they estimate the same quantity,
        # so a gap of more than a percent would mean something is wrong with one of them.
        cv = cv_fit.cv_targeting
        for name, estimate in cv_fit.estimates.items():
            assert cv.variance[name] == pytest.approx(estimate.variance, rel=0.01)

    def test_it_rejects_folds_that_do_not_partition_the_sample(self) -> None:
        ic = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="partition"):
            cross_validated_variance(ic, [np.arange(5)])
        with pytest.raises(ValueError, match="partition"):
            cross_validated_variance(ic, [np.arange(6), np.arange(4, 10)])

    def test_singleton_clusters_agree_with_the_unclustered_form(self) -> None:
        rng = np.random.default_rng(5)
        ic = rng.normal(size=40)
        index = [np.arange(40)[k::4] for k in range(4)]
        cluster = np.arange(40)
        assert cross_validated_variance(ic, index, cluster) == pytest.approx(
            cross_validated_variance(ic, index), rel=1e-12
        )


class TestCanonicalEvaluation:
    """Fold-wise evaluation is a different estimator, not a different view of one.

    Stitching the fold-targeted predictions back together and evaluating once is not
    what Zheng & van der Laan analyse.  For an estimand linear in those predictions the
    distinction is invisible; for every other estimand it is real, and the whole point
    of these tests is that it stays visible in the code rather than being papered over.
    """

    @pytest.mark.parametrize("name", LINEAR)
    def test_the_two_reports_agree_on_linear_estimands(
        self, name: str, pooled_report, canonical_report
    ) -> None:
        # A weighted mean of the targeted predictions over the whole sample *is* the
        # average of the fold means when the folds are equal sized. Exact identity, so
        # assert it to machine precision rather than to a tolerance.
        assert canonical_report.psi(name) == pytest.approx(pooled_report.psi(name), abs=1e-12)

    @pytest.mark.parametrize("name", NONLINEAR)
    def test_the_two_reports_diverge_on_every_other_estimand(
        self, name: str, pooled_report, canonical_report
    ) -> None:
        # A ratio of means is not a mean of ratios, and the pooled ATT/ATC weight by the
        # whole sample's arm share rather than each fold's. The gap is small -- it is a
        # second-order term -- but it is not zero, and documentation that says these two
        # constructions are the same thing is wrong about exactly these four estimands.
        pooled, canonical = pooled_report.psi(name), canonical_report.psi(name)
        assert canonical != pooled
        assert abs(canonical - pooled) < 0.5 * pooled_report[name].std_error

    def test_the_canonical_estimate_is_the_fold_average(self, canonical_report) -> None:
        cv = canonical_report.cv_targeting
        assert len(set(cv.fold_sizes)) == 1, "equal folds, so 1/V weighting is unambiguous"
        for name in (*LINEAR, "att", "atc"):
            assert canonical_report.psi(name) == pytest.approx(
                float(np.mean(cv.fold_estimates[name])), rel=1e-12
            )
        for name in ("rr", "or"):
            # Ratios are averaged on the log scale -- where the influence curve and the
            # interval already live -- so that psi == exp(log_psi) continues to hold.
            expected = float(np.exp(np.mean(np.log(cv.fold_estimates[name]))))
            assert canonical_report.psi(name) == pytest.approx(expected, rel=1e-12)
            assert canonical_report[name].psi == pytest.approx(
                float(np.exp(canonical_report[name].log_psi)), rel=1e-12
            )

    def test_the_canonical_influence_curve_is_centred_inside_every_fold(
        self, canonical_report, pooled_report
    ) -> None:
        # This is the structural difference, and it is what licenses the *uncentred*
        # second moment in cross_validated_variance: a fold-specific curve is centred at
        # its own fold's estimate, so nothing is discarded by not centring again. The
        # pooled curve is only mean-zero over the whole sample, which is why feeding it
        # to the same formula was an approximation.
        folds = [record.index for record in canonical_report.fluctuations["mean"].folds]
        for name in canonical_report.estimates:
            canonical = canonical_report[name].influence_curve
            pooled = pooled_report[name].influence_curve
            assert max(abs(float(np.mean(canonical[i]))) for i in folds) < 1e-10
            assert max(abs(float(np.mean(pooled[i]))) for i in folds) > 1e-3

    def test_the_canonical_standard_error_is_the_cross_validated_one(
        self, canonical_report
    ) -> None:
        cv = canonical_report.cv_targeting
        for name, estimate in canonical_report.estimates.items():
            assert estimate.variance == cv.variance[name]

    def test_both_reports_are_kept_whichever_one_is_headline(
        self, pooled_report, canonical_report
    ) -> None:
        # Switching cv_evaluation must never make the other construction unavailable,
        # because comparing them is how you find out whether the choice mattered.
        for result in (pooled_report, canonical_report):
            cv = result.cv_targeting
            assert sorted(cv.pooled) == sorted(cv.canonical) == sorted(ALL_BINARY)
        headline = {"pooled": pooled_report, "canonical": canonical_report}
        for label, result in headline.items():
            for name, estimate in result.estimates.items():
                assert estimate.psi == getattr(result.cv_targeting, label)[name].psi

    def test_the_score_equation_still_holds(self, canonical_report) -> None:
        # A curve that is mean-zero in every fold is mean-zero pooled, so the diagnostic
        # that checks the targeting step has to keep passing.
        assert canonical_report.validation.score_check().passed

    def test_it_names_the_estimator_it_ran(self, pooled_report, canonical_report) -> None:
        assert pooled_report.config.estimator_name == "fold-targeted CV-TMLE"
        assert canonical_report.config.estimator_name == "canonical CV-TMLE"

    def test_canonical_evaluation_needs_something_to_evaluate(self) -> None:
        with pytest.raises(ValueError, match="targeting_scheme='fold'"):
            TMLE(cv_evaluation=True)

    def test_a_fold_that_cannot_produce_an_estimand_drops_it(self) -> None:
        # Constructed rather than fitted: a fold with no units in the conditioning arm
        # is rare enough under stratified folds that a DGP would be a fragile way to
        # reach it, and the behaviour under test is the aggregation, not the DGP.
        indices = [np.arange(20)[k::2] for k in range(2)]
        rng = np.random.default_rng(11)
        complete = [
            {
                "ate": _fake_estimate("ate", 0.3, rng.normal(size=10)),
                "att": _fake_estimate("att", 0.2, rng.normal(size=10)),
            }
            for _ in indices
        ]
        complete[1].pop("att")  # the second fold had nobody in the treated arm

        with pytest.warns(UserWarning, match="att"):
            out = _average_over_folds(
                complete, ("ate", "att"), indices, n=20, cluster=None, alpha=0.05
            )
        assert list(out) == ["ate"]


def _fake_estimate(name: str, psi: float, ic: np.ndarray) -> object:
    from cleverly.inference import ParameterEstimate

    return ParameterEstimate(
        name=name, psi=psi, influence_curve=ic, variance=1.0, n=ic.size, n_clusters=ic.size
    )


class TestCVTargetingReport:
    def test_fold_estimates_average_to_the_pooled_estimate(self, cv_fit) -> None:
        # Every estimand here is a weighted mean of the targeted predictions, so the
        # pooled plug-in *is* the fold average when the folds are equal sized.
        cv = cv_fit.cv_targeting
        assert len(set(cv.fold_sizes)) == 1, "this identity needs equal folds"
        for name, estimate in cv_fit.estimates.items():
            assert float(np.mean(cv.fold_estimates[name])) == pytest.approx(estimate.psi, rel=1e-10)

    def test_the_summary_names_the_folds_and_the_coefficients(self, cv_fit) -> None:
        text = cv_fit.cv_targeting.summary()
        assert "Cross-validated targeting over 5 folds" in text
        assert "fluctuation coefficients by fold" in text
        assert "ate" in text

    def test_to_frame_has_one_row_per_estimand(self, cv_fit) -> None:
        frame = cv_fit.cv_targeting.to_frame(cv_fit.data)
        assert len(frame) == len(cv_fit.estimates)
        assert "cv_std_err" in frame.columns


class TestRestrict:
    def test_it_subsets_by_index_and_by_mask_alike(self) -> None:
        rng = np.random.default_rng(6)
        treatment = rng.binomial(1, 0.5, 20).astype(float)
        submodel = mean_submodel(treatment, np.full(20, 0.4))
        index = np.array([1, 4, 9, 15])
        mask = np.zeros(20, dtype=bool)
        mask[index] = True

        by_index = restrict(submodel, index)
        by_mask = restrict(submodel, mask)
        assert np.array_equal(by_index.observed, by_mask.observed)
        assert np.array_equal(by_index.observed, submodel.observed[index])
        assert by_index.names == submodel.names
        assert by_index.group == submodel.group

    def test_it_matches_what_the_fold_targeting_needs(self) -> None:
        # The fold loop hands `restrict` an integer index straight from Folds.
        rng = np.random.default_rng(7)
        treatment = rng.binomial(1, 0.5, 30).astype(float)
        submodel = mean_submodel(treatment, rng.uniform(0.2, 0.8, 30))
        folds = Folds(np.repeat(np.arange(3), 10), 3)
        rebuilt = np.empty_like(submodel.observed)
        for _, test in folds:
            rebuilt[test] = restrict(submodel, test).observed
        assert np.array_equal(rebuilt, submodel.observed)


def test_influence_variance_is_unchanged_by_the_new_helper() -> None:
    # cross_validated_variance was added next to influence_variance; make sure the
    # original still means what it did.
    rng = np.random.default_rng(8)
    ic = rng.normal(size=50)
    assert influence_variance(ic) == pytest.approx(float(np.var(ic, ddof=1) / 50), rel=1e-12)
