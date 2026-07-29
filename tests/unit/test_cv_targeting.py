"""The cross-validated targeting step, and the variance that goes with it.

CV-TMLE (Zheng & van der Laan, 2011) differs from a pooled fit in one place: the
fluctuation is solved separately within each validation fold, against nuisance
predictions from models that never saw those rows.  Everything asserted here is an
exact algebraic consequence of that, so these tests fail deterministically rather
than statistically.  The claim CV-TMLE actually exists to make -- that it keeps
nominal coverage where a pooled fit does not -- needs replications and lives in the
slow tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import make_linear_ate
from cleverly.fluctuation import restrict
from cleverly.fluctuation.submodel import mean_submodel
from cleverly.inference import cross_validated_variance, influence_variance
from cleverly.learners.crossfit import Folds, make_folds
from tests.conftest import FAST_KWARGS


@pytest.fixture(scope="module")
def cv_fit() -> object:
    frame, _ = make_linear_ate(n=600, seed=17)
    return TMLE(**{**FAST_KWARGS, "targeting_scheme": "fold", "estimands": ("ate", "ey1")}).fit(
        frame, outcome="Y", treatment="A"
    )


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
        # schemes must agree exactly rather than merely closely.
        frame, _ = make_linear_ate(n=400, seed=19)
        settings = {**FAST_KWARGS, "estimands": ("ate",), "cross_fit": False}
        pooled = TMLE(**settings).fit(frame, outcome="Y", treatment="A")
        fold_wise = TMLE(**{**settings, "targeting_scheme": "fold"}).fit(
            frame, outcome="Y", treatment="A"
        )
        assert fold_wise.psi("ate") == pooled.psi("ate")
        assert fold_wise["ate"].std_error == pooled["ate"].std_error


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
        # Not equal by construction -- the pooled estimator centres and uses ddof=1 --
        # but a solved score equation leaves the influence curve mean-zero, so a gap
        # of more than a percent would mean something is wrong with one of them.
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
