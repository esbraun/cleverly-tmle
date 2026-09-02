"""The cross-validated targeting step, and the variance that goes with it.

Original CV-TMLE (Zheng & van der Laan, 2011) commonly fits one fluctuation by
minimising the average validation-fold loss, then evaluates the updated distribution
fold by fold. Levy's easy implementation stacks the out-of-fold predictions and performs
an otherwise ordinary TMLE; that is the package default and is corroborated by the pinned
``cvtmle=TRUE`` path in R ``tmle3``.
``targeting_scheme="fold"`` is a separate extension with one epsilon per fold.

Everything asserted here is an exact algebraic consequence of the construction, so
these tests fail deterministically rather than statistically.  The claim CV-TMLE
actually exists to make -- that it keeps nominal coverage where a pooled fit does not
-- needs replications and belongs in a registered study.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.estimators import TMLE
from cleverly.estimators.tmle import _average_over_folds
from cleverly.fluctuation import restrict, stitch
from cleverly.fluctuation.submodel import att_submodel, mean_submodel
from cleverly.inference import cross_validated_variance, influence_variance
from cleverly.learners.crossfit import Folds, make_folds
from tests.conftest import FAST_KWARGS

#: Every estimand the binary-outcome fixture supports: the three that are linear in
#: the targeted predictions and the four that are not.
ALL_BINARY = ("ey1", "ey0", "ate", "rr", "or", "att", "atc")
LINEAR = ("ey1", "ey0", "ate")
NONLINEAR = ("rr", "or", "att", "atc")
CANONICAL = (*LINEAR, "att", "atc")


@pytest.fixture(scope="module")
def cv_fit() -> object:
    frame, _ = make_linear_ate(n=600, seed=17)
    return (
        TMLE(**{**FAST_KWARGS, "targeting_scheme": "fold", "estimands": ("ate", "ey1")})
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@pytest.fixture(scope="module")
def binary_frame() -> object:
    # n divisible by the 5 fast-tier folds, so the fold weights cancel exactly and any
    # gap between the two reports is the construction rather than unequal fold sizes.
    return make_binary_outcome(n=800, seed=3)[0]


@pytest.fixture(scope="module")
def pooled_report(binary_frame) -> object:
    settings = {**FAST_KWARGS, "estimands": ALL_BINARY}
    return TMLE(**settings).fit(binary_frame, outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def canonical_report(binary_frame) -> object:
    settings = {
        **FAST_KWARGS,
        "cv_evaluation": True,
        "estimands": CANONICAL,
    }
    return TMLE(**settings).fit(binary_frame, outcome="Y", treatment="A").single()


class TestFoldWiseTargeting:
    def test_each_fold_solves_its_own_score_equation(self, cv_fit) -> None:
        # The defining property. A pooled fit only zeroes the score over the whole
        # sample; fold-wise targeting zeroes it inside every fold separately, and the
        # pooled score is then zero as a sum of zeros rather than by cancellation.
        for fluctuation in cv_fit.fluctuations.values():
            assert fluctuation.folds, "fold-wise targeting must record its folds"
            assert fluctuation.trace == ()
            assert fluctuation.n_iter == sum(record.n_iter for record in fluctuation.folds)
            for record in fluctuation.folds:
                assert np.max(np.abs(record.score)) < 1e-9
                assert record.trace
                assert record.score_scale is not None

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
        result = (
            TMLE(**{**FAST_KWARGS, "estimands": ("ate",)})
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert result.cv_targeting is None
        assert all(not f.folds for f in result.fluctuations.values())

    def test_one_fold_reduces_to_a_pooled_fit(self) -> None:
        # With a single fold there is no validation split to target within, so the two
        # schemes must agree exactly rather than merely closely -- and the fallback has
        # to be announced rather than silently honoured, because a caller who asked for
        # the fold-specific extension and got pooled targeting should hear about it.
        frame, _ = make_linear_ate(n=400, seed=19)
        settings = {**FAST_KWARGS, "estimands": ("ate",), "cross_fit": False}
        pooled = TMLE(**settings).fit(frame, outcome="Y", treatment="A").single()
        with pytest.warns(UserWarning, match="falling back to pooled targeting"):
            fold_wise = (
                TMLE(**{**settings, "targeting_scheme": "fold"})
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
        assert fold_wise.psi("ate") == pooled.psi("ate")
        assert fold_wise["ate"].std_error == pooled["ate"].std_error
        assert fold_wise.cv_targeting is None


class TestCanonicalTargeting:
    """Structural pins for the common validation update used by the source algorithm."""

    def test_it_fits_one_common_epsilon_not_one_per_fold(
        self, canonical_report, pooled_report
    ) -> None:
        differs_from_stacked = []
        for group, fluctuation in canonical_report.fluctuations.items():
            assert not fluctuation.folds
            assert canonical_report.cv_targeting.epsilon[group] == pytest.approx(
                fluctuation.epsilon
            )
            differs_from_stacked.append(
                not np.array_equal(fluctuation.epsilon, pooled_report.fluctuations[group].epsilon)
            )
        # This stratified partition is not exactly row-balanced. The original
        # construction gives every fold risk mass 1/V, whereas Levy stacks rows,
        # so their common coefficients are allowed (and here known) to differ.
        assert any(differs_from_stacked)

    def test_the_fold_specific_extension_is_not_the_canonical_update(
        self, canonical_report, cv_fit
    ) -> None:
        common = canonical_report.fluctuations["mean"].epsilon
        per_fold = [record.epsilon for record in cv_fit.fluctuations["mean"].folds]
        assert len(per_fold) == canonical_report.cv_targeting.n_folds
        assert any(not np.allclose(value, common) for value in per_fold)

    def test_the_common_loss_normalises_observation_weights_inside_each_fold(
        self, canonical_report
    ) -> None:
        weights = np.linspace(0.2, 2.0, canonical_report.n)
        weights /= weights.mean()
        data = replace(canonical_report.data, weights=weights)
        balanced = canonical_report.estimator._validation_weights(data, canonical_report.nuisance)
        masses = [float(np.sum(balanced[test])) for _, test in canonical_report.nuisance.folds]
        assert masses == pytest.approx([canonical_report.n / len(masses)] * len(masses), rel=1e-12)

    def test_levys_stacked_loss_keeps_the_empirical_weights(self, pooled_report) -> None:
        weights = np.linspace(0.2, 2.0, pooled_report.n)
        weights /= weights.mean()
        data = replace(pooled_report.data, weights=weights)
        stacked = pooled_report.estimator._validation_weights(data, pooled_report.nuisance)
        np.testing.assert_array_equal(stacked, weights)


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
        with pytest.raises(ValueError, match="must lie in"):
            cross_validated_variance(ic, [np.arange(-1, 4), np.arange(4, 9)])
        with pytest.raises(ValueError, match="must be integers"):
            cross_validated_variance(ic, [np.arange(5, dtype=float), np.arange(5, 10)])

    def test_singleton_clusters_agree_with_the_unclustered_form(self) -> None:
        rng = np.random.default_rng(5)
        ic = rng.normal(size=40)
        index = [np.arange(40)[k::4] for k in range(4)]
        cluster = np.arange(40)
        assert cross_validated_variance(ic, index, cluster) == pytest.approx(
            cross_validated_variance(ic, index), rel=1e-12
        )

    def test_unequal_folds_weight_each_fold_equally_not_each_row(self) -> None:
        r"""``1/V``, not ``n_v/n`` -- and on unequal folds the two are different numbers.

        Every other test of this function uses folds that divide the sample evenly, where
        the two weightings coincide and the choice is invisible.  Zheng & van der Laan
        average the folds, so a fold of two rows counts as much as a fold of six; the
        row-weighted alternative collapses to ``mean(IC**2) / n``, which is a different
        estimator and the one a reader would probably assume.  Pinning the difference is
        what stops the weighting being "optimised" into the other one later.
        """
        ic = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 3.0, -3.0, 2.5, -2.5])
        index = [np.arange(6), np.array([6, 7]), np.array([8, 9])]

        exact = float(sum(np.sum(ic[test] ** 2) / test.size**2 for test in index) / len(index) ** 2)
        old_scaling = float(np.mean([np.mean(ic[test] ** 2) for test in index]) / 10)
        assert cross_validated_variance(ic, index) == pytest.approx(exact, rel=1e-12)
        # Dividing the fold-averaged second moment by total n is correct only when
        # n_v=n/V.  This partition makes the old shortcut visibly wrong.
        assert exact > 1.6 * old_scaling

    def test_it_matches_a_longhand_cluster_aware_loop(self) -> None:
        """Real clusters, not the singleton degenerate case.

        With several observations per cluster the fold contribution uses squared
        *cluster sums* rather than squared rows. The singleton test above cannot see that
        distinction because both reduce to the same expression with one row per cluster.
        """
        rng = np.random.default_rng(23)
        cluster = np.repeat(np.arange(10), 4)
        # A shared per-cluster component, so the rows inside a cluster really are
        # dependent and the cluster-robust form has something to correct for.
        ic = np.repeat(rng.normal(size=10), 4) + 0.3 * rng.normal(size=40)
        index = [np.arange(40).reshape(10, 4)[k::5].reshape(-1) for k in range(5)]

        contributions = []
        for test in index:
            codes = cluster[test]
            sums = [ic[test][codes == code].sum() for code in np.unique(codes)]
            contributions.append(float(np.sum(np.square(sums))) / test.size**2)
        expected = float(np.sum(contributions)) / len(index) ** 2
        assert cross_validated_variance(ic, index, cluster) == pytest.approx(expected, rel=1e-12)

    def test_ignoring_real_clusters_understates_the_variance(self) -> None:
        # Teeth for the test above: if the clustered and unclustered forms happened to
        # agree on this input, matching the longhand loop would prove very little.
        rng = np.random.default_rng(24)
        cluster = np.repeat(np.arange(10), 4)
        ic = np.repeat(rng.normal(size=10), 4) + 0.3 * rng.normal(size=40)
        index = [np.arange(40).reshape(10, 4)[k::5].reshape(-1) for k in range(5)]
        clustered = cross_validated_variance(ic, index, cluster)
        naive = cross_validated_variance(ic, index)
        assert clustered > 2.0 * naive

    def test_it_refuses_a_cluster_split_across_folds(self) -> None:
        """A split cluster is detectable invalid input, not a variance to report."""
        rng = np.random.default_rng(25)
        cluster = np.repeat(np.arange(10), 4)
        ic = np.repeat(rng.normal(size=10), 4) + 0.3 * rng.normal(size=40)
        shredded = [np.arange(40)[k::5] for k in range(5)]
        with pytest.raises(ValueError, match="clusters must be assigned whole"):
            cross_validated_variance(ic, shredded, cluster)

    def test_unequal_fold_aggregation_scales_the_stored_curve(self) -> None:
        indices = [np.arange(6), np.array([6, 7]), np.array([8, 9])]
        raw = [np.arange(1, index.size + 1, dtype=float) for index in indices]
        per_fold = [
            {"ate": _fake_estimate("ate", psi, curve)}
            for psi, curve in zip((0.1, 0.4, 0.7), raw, strict=True)
        ]

        estimate = _average_over_folds(per_fold, ("ate",), indices, n=10, cluster=None, alpha=0.05)[
            "ate"
        ]

        assert estimate.psi == pytest.approx(0.4, rel=1e-12)
        expected_curve = np.empty(10)
        for index, curve in zip(indices, raw, strict=True):
            expected_curve[index] = 10 / (3 * index.size) * curve
        np.testing.assert_allclose(estimate.influence_curve, expected_curve, rtol=0, atol=0)
        stitched = np.concatenate(raw)
        assert estimate.variance == pytest.approx(
            cross_validated_variance(stitched, indices), rel=1e-12
        )


class TestCanonicalEvaluation:
    """Fold-wise evaluation is a different estimator, not a different view of one.

    Stitching the common-update validation predictions and evaluating once is not what
    Zheng & van der Laan analyse. For an estimand linear in those predictions the
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

    @pytest.mark.parametrize("name", ("att", "atc"))
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
        for name in CANONICAL:
            assert canonical_report.psi(name) == pytest.approx(
                float(np.mean(cv.fold_estimates[name])), rel=1e-12
            )

    def test_the_canonical_influence_curve_is_not_recentred_inside_folds(
        self, canonical_report, pooled_report
    ) -> None:
        # One common epsilon solves the *average* validation score, not every fold's
        # score separately.  The nonzero fold means are therefore part of the canonical
        # influence curve and must not be erased by recentering before its second moment.
        folds = [test for _, test in canonical_report.nuisance.folds]
        for name in canonical_report.estimates:
            canonical = canonical_report[name].influence_curve
            pooled = pooled_report[name].influence_curve
            fold_means = np.array([np.mean(canonical[i]) for i in folds])
            assert np.max(np.abs(fold_means)) > 1e-4
            assert float(np.mean(fold_means)) == pytest.approx(0.0, abs=1e-9)
            assert not np.array_equal(canonical, pooled)

    def test_the_canonical_standard_error_is_the_cross_validated_one(
        self, canonical_report
    ) -> None:
        cv = canonical_report.cv_targeting
        for name, estimate in canonical_report.estimates.items():
            assert estimate.variance == cv.variance[name]

    def test_both_reports_are_kept_whichever_one_is_headline(
        self, pooled_report, canonical_report
    ) -> None:
        cv = canonical_report.cv_targeting
        assert pooled_report.cv_targeting is None
        assert sorted(cv.pooled) == sorted(cv.canonical) == sorted(CANONICAL)
        assert cv.fold_evaluated is cv.canonical
        for name, estimate in canonical_report.estimates.items():
            assert estimate.psi == cv.canonical[name].psi
            assert cv.pooled[name].variance == pytest.approx(
                influence_variance(cv.pooled[name].influence_curve, canonical_report.data.cluster),
                rel=1e-12,
            )

    def test_the_score_equation_still_holds(self, canonical_report) -> None:
        # The common update solves the average validation score even though individual
        # folds retain nonzero score contributions.
        assert canonical_report.diagnostics.score_equations().passed

    def test_it_names_the_estimator_it_ran(self, pooled_report, canonical_report) -> None:
        assert pooled_report.config.estimator_name == "stacked CV-TMLE (Levy)"
        assert canonical_report.config.estimator_name == "fold-evaluated CV-TMLE"

    def test_canonical_evaluation_needs_something_to_evaluate(self) -> None:
        with pytest.raises(ValueError, match="cross_fit=True"):
            TMLE(cv_evaluation=True, cross_fit=False)

    @pytest.mark.parametrize("name", ("rr", "or"))
    def test_nonlinear_fold_aggregation_is_refused_until_its_score_is_targeted(
        self, binary_frame, name: str
    ) -> None:
        with pytest.raises(ValueError, match="nonlinear parameter"):
            TMLE(**{**FAST_KWARGS, "cv_evaluation": True, "estimands": (name,)}).fit(
                binary_frame, outcome="Y", treatment="A"
            )

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


class TestWhichVarianceIsTheInferentialOne:
    """Which number the interval is built from, asserted rather than left to the prose.

    The package reports two variances for a fold-evaluated fit and the README says which is
    headline; nothing checked that the code agreed.  The distinction matters because they
    are variances of two different estimators: the pooled one is the sample variance of the
    stitched influence curve, the canonical one is the fold-averaged uncentred second
    moment that Zheng & van der Laan pair with fold-wise evaluation.  Mixing them -- a
    canonical point estimate with a pooled standard error, or the reverse -- would be
    wrong in a way no coverage study at one sample size would reliably catch.
    """

    def test_the_pooled_report_uses_the_stitched_influence_curve(self, pooled_report) -> None:
        for name, estimate in pooled_report.estimates.items():
            expected = influence_variance(estimate.influence_curve, pooled_report.data.cluster)
            assert estimate.variance == pytest.approx(expected, rel=1e-12), name

    def test_the_canonical_report_uses_the_cross_validated_variance(self, canonical_report) -> None:
        folds = [test for _, test in canonical_report.nuisance.folds]
        for name in LINEAR:
            estimate = canonical_report[name]
            expected = cross_validated_variance(
                estimate.influence_curve, folds, canonical_report.data.cluster
            )
            assert estimate.variance == pytest.approx(expected, rel=1e-12), name

    def test_the_two_variances_are_not_interchangeable(
        self, pooled_report, canonical_report
    ) -> None:
        # They estimate the same quantity, so they are close; they are computed from
        # different curves by different formulas, so they are not equal. If they came out
        # identical, the two tests above would not be distinguishing anything.
        differs = 0
        for name in CANONICAL:
            pooled = pooled_report[name].variance
            canonical = canonical_report[name].variance
            assert canonical == pytest.approx(pooled, rel=0.05), name
            differs += canonical != pooled
        assert differs == len(CANONICAL)

    def test_a_pooled_fit_reports_no_cross_validated_variance_as_headline(
        self, pooled_report, canonical_report
    ) -> None:
        # Without fold-wise evaluation the ordinary stitched-curve variance remains the
        # headline. The fold-evaluated fit also keeps the ordinary variance of its own
        # common update for comparison; it is not a second fit under tmle3's row-weighted
        # loss, so the two comparison reports need not be numerically identical.
        assert pooled_report.cv_targeting is None
        cv = canonical_report.cv_targeting
        for name in CANONICAL:
            estimate = pooled_report[name]
            assert estimate.variance == pytest.approx(
                influence_variance(estimate.influence_curve, pooled_report.data.cluster),
                rel=1e-12,
            )
            assert cv.pooled[name].variance == pytest.approx(
                influence_variance(cv.pooled[name].influence_curve, canonical_report.data.cluster),
                rel=1e-12,
            )
            assert estimate.variance != cv.variance[name]


class TestRepeatedDraws:
    """The original fold-evaluated construction over several split draws.

    ``repeats=R`` median-combines ``R`` fold-evaluated CV-TMLEs. Each draw contributes
    its cross-validated variance to the within-plus-between aggregation. The arithmetic is
    pinned in ``tests/unit/test_repeated_crossfit.py``; what is checked here is that the
    *estimator* the setting names is still the one that ran.
    """

    @pytest.fixture(scope="class")
    def repeated(self, binary_frame) -> object:
        settings = {
            **FAST_KWARGS,
            "cv_evaluation": True,
            "repeats": 3,
            "estimands": CANONICAL,
        }
        return TMLE(**settings).fit(binary_frame, outcome="Y", treatment="A").single()

    def test_it_is_still_the_canonical_estimator(self, repeated) -> None:
        assert repeated.config.estimator_name == "fold-evaluated CV-TMLE"
        assert repeated.cv_targeting.repeats == 3

    def test_the_median_variance_is_not_rebuilt_from_one_curve(self, repeated) -> None:
        for estimate in repeated.estimates.values():
            assert estimate.variance != pytest.approx(
                influence_variance(estimate.influence_curve), rel=1e-6
            )

    def test_the_linear_estimands_still_agree_with_the_pooled_report(self, repeated) -> None:
        # Taking medians over draws cannot change *which* estimands the two evaluations agree
        # on: `ate`, `ey1` and `ey0` are linear in the targeted predictions in every draw,
        # so they stay equal after aggregation, and the rest stay apart.
        detail = repeated.cv_targeting
        for name in LINEAR:
            assert detail.canonical[name].psi == pytest.approx(detail.pooled[name].psi, rel=1e-9)
        for name in ("att", "atc"):
            assert detail.canonical[name].psi != pytest.approx(detail.pooled[name].psi, rel=1e-9)

    def test_the_cross_validated_variance_is_reported_for_every_estimand(self, repeated) -> None:
        assert set(repeated.cv_targeting.variance) == set(repeated.estimates)
        for name, estimate in repeated.estimates.items():
            assert repeated.cv_targeting.variance[name] == estimate.variance


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
        assert "fold-specific fluctuation coefficients" in text
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


class TestStitch:
    """``restrict``'s inverse, which the fold loop needs once a covariate is fold-specific.

    Every fold builds its own clever covariate under a linked working model, and the
    *pooled* score has to be taken against the covariate each row was actually fluctuated
    by. Where the covariate is the same on every fold -- which is every other group --
    this has to give back exactly what it was handed, or fold targeting would move fits
    that have nothing to do with a working model.
    """

    @staticmethod
    def _submodel(n: int = 30, seed: int = 11):
        rng = np.random.default_rng(seed)
        treatment = rng.binomial(1, 0.5, n).astype(float)
        return mean_submodel(treatment, rng.uniform(0.2, 0.8, n))

    def test_round_tripping_through_the_folds_changes_nothing(self) -> None:
        submodel = self._submodel()
        folds = Folds(np.repeat(np.arange(3), 10), 3)
        pieces = [(test, restrict(submodel, test)) for _, test in folds]
        rebuilt = stitch(pieces, 30)
        np.testing.assert_array_equal(rebuilt.observed, submodel.observed)
        for level, values in submodel.arms.items():
            np.testing.assert_array_equal(rebuilt.arms[level], values)
        assert rebuilt.group == submodel.group
        assert rebuilt.names == submodel.names
        assert rebuilt.arm_columns == submodel.arm_columns

    def test_each_fold_keeps_its_own_values(self) -> None:
        """The case that matters: the pieces disagree, and every row keeps its own."""
        first, second = self._submodel(seed=1), self._submodel(seed=2)
        left, right = np.arange(0, 15), np.arange(15, 30)
        rebuilt = stitch([(left, restrict(first, left)), (right, restrict(second, right))], 30)
        np.testing.assert_array_equal(rebuilt.observed[left], first.observed[left])
        np.testing.assert_array_equal(rebuilt.observed[right], second.observed[right])

    def test_pieces_that_do_not_cover_the_sample_are_refused(self) -> None:
        submodel = self._submodel()
        index = np.arange(0, 20)
        with pytest.raises(ValueError, match="20 of 30 rows"):
            stitch([(index, restrict(submodel, index))], 30)

    def test_pieces_describing_different_fluctuations_are_refused(self) -> None:
        rng = np.random.default_rng(3)
        treatment = rng.binomial(1, 0.5, 30).astype(float)
        propensity = rng.uniform(0.2, 0.8, 30)
        left, right = np.arange(0, 15), np.arange(15, 30)
        mean = restrict(mean_submodel(treatment, propensity), left)
        other = restrict(att_submodel(treatment, propensity, arm_fractions=0.5), right)
        with pytest.raises(ValueError, match="different fluctuations"):
            stitch([(left, mean), (right, other)], 30)


def test_influence_variance_is_unchanged_by_the_new_helper() -> None:
    # cross_validated_variance was added next to influence_variance; make sure the
    # original still means what it did.
    rng = np.random.default_rng(8)
    ic = rng.normal(size=50)
    assert influence_variance(ic) == pytest.approx(float(np.var(ic, ddof=1) / 50), rel=1e-12)
