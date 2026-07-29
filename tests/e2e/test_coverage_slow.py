r"""The statistical validation tier: does the estimator's *inference* work?

Everything in the fast tier checks that the estimator computes what it claims on a
given sample.  This tier checks the claim that actually matters to a user reading a
p-value: that the reported uncertainty is honest.

Three properties, each of which fails in a different, diagnosable way:

**Coverage.**  A nominal 95% interval must contain the truth about 95% of the time.
Under-coverage means every inference drawn from the estimator is overconfident, and it
is invisible on any single dataset.

**Root-n consistency.**  :math:`\sqrt{n} \times \mathrm{bias}` must stay bounded as
:math:`n` grows.  Bias that shrinks more slowly than :math:`n^{-1/2}` eventually
dominates the standard error, and coverage degrades as the sample gets *larger* -- the
opposite of the usual intuition, and the reason this is checked as a rate rather than a
level.

**Type I error.**  Under a null process the rejection rate must sit near the nominal
level.  This is the direct check that the whole machinery is calibrated.

The estimator *variants* are validated here too, for the same reason: what CV-TMLE
and C-TMLE are for is a statement about repeated sampling, and no single fit can
show it.  Those tests compare two configurations on the same processes rather than
asserting an absolute level, because the claims themselves are comparative -- C-TMLE
pays less variance than TMLE for an instrument; cross-fitting keeps the standard
error honest where in-sample fitting does not.

These runs take minutes rather than seconds, so they are marked ``slow`` and run
nightly rather than on every push.  The thresholds are set from the Monte Carlo standard
error of each quantity, so a pass is evidence rather than a formality.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import CTMLE, TMLE
from cleverly.datasets import (
    DGP,
    binary_outcome_dgp,
    instrument_dgp,
    linear_dgp,
    nonlinear_dgp,
    weak_overlap_dgp,
)
from cleverly.utils.bounds import expit
from cleverly.validation import CoverageStudy

pytestmark = pytest.mark.slow

#: Replications per study.  At 400 the Monte Carlo standard error of a coverage estimate
#: near 0.95 is about 1.1 percentage points, enough to resolve a real shortfall.
REPLICATES = 400


def _study(
    dgp: DGP,
    *,
    n: int,
    reps: int = REPLICATES,
    estimands: tuple[str, ...] = ("ate",),
    library: str = "glm",
    fit_kwargs: dict[str, object] | None = None,
    seed: int = 2024,
) -> object:
    return CoverageStudy(
        dgp=dgp,
        estimator=lambda: TMLE(
            outcome_learner=library,
            treatment_learner=library,
            n_folds=5,
            learner_folds=3,
            estimands=estimands,
            simultaneous=False,
            random_state=0,
        ),
        n=n,
        n_replicates=reps,
        estimands=estimands,
        fit_kwargs=fit_kwargs or {"outcome": "Y", "treatment": "A"},
        seed=seed,
        n_jobs=2,
    ).run()


class TestCoverage:
    @pytest.mark.parametrize("n", [500, 2000])
    def test_confidence_intervals_cover_at_the_nominal_rate(self, n: int) -> None:
        summary = _study(linear_dgp(), n=n)["ate"]
        # Both nuisance models are correctly specified here, so nothing but the
        # inference machinery can be responsible for a shortfall.
        assert 0.93 <= summary.coverage <= 0.97, summary
        assert abs(summary.coverage - 0.95) < 3.0 * summary.coverage_se

    def test_the_reported_standard_error_matches_the_actual_variability(self) -> None:
        summary = _study(linear_dgp(), n=1000)["ate"]
        # The ratio of the mean reported standard error to the observed spread of the
        # estimates. Below one means the reported uncertainty is optimistic, which is
        # how under-coverage almost always arises.
        assert 0.93 <= summary.se_ratio <= 1.07, summary

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc", "ey1", "ey0"])
    def test_every_estimand_covers(self, estimand: str) -> None:
        summary = _study(linear_dgp(), n=1000, estimands=("ate", "att", "atc", "ey1", "ey0"))[
            estimand
        ]
        assert 0.92 <= summary.coverage <= 0.98, summary

    def test_ratio_estimands_cover_on_the_log_scale(self) -> None:
        study = _study(
            binary_outcome_dgp(),
            n=2000,
            estimands=("rr", "or"),
        )
        for name in ("rr", "or"):
            assert 0.92 <= study[name].coverage <= 0.98, study[name]

    def test_flexible_learners_still_cover(self) -> None:
        # The case that motivates cross-fitting: machine-learning nuisance estimators,
        # where in-sample predictions would destroy the coverage guarantee. Fewer
        # replications than the parametric studies because each fit costs ~10x as much;
        # 120 still resolves the +/- 0.05 window asserted here.
        summary = _study(nonlinear_dgp(), n=1000, reps=120, library="fast")["ate"]
        assert 0.90 <= summary.coverage <= 0.99, summary


class TestConsistency:
    def test_root_n_bias_stays_bounded(self) -> None:
        sizes = (500, 2000, 8000)
        scaled = []
        for n in sizes:
            summary = _study(linear_dgp(), n=n, reps=200)["ate"]
            scaled.append(abs(summary.root_n_bias))
        # sqrt(n) * bias must not grow with n. Allowing a factor of 2.5 leaves room for
        # Monte Carlo noise while still catching a bias that vanishes too slowly.
        assert scaled[-1] < 2.5 * min(scaled), dict(zip(sizes, scaled, strict=True))

    def test_the_standard_error_halves_when_n_quadruples(self) -> None:
        small = _study(linear_dgp(), n=500, reps=200)["ate"]
        large = _study(linear_dgp(), n=2000, reps=200)["ate"]
        assert large.mean_std_error / small.mean_std_error == pytest.approx(0.5, abs=0.05)

    def test_the_estimate_converges_on_the_truth(self) -> None:
        small = _study(nonlinear_dgp(), n=400, reps=100, library="fast")["ate"]
        large = _study(nonlinear_dgp(), n=1600, reps=100, library="fast")["ate"]
        assert abs(large.bias) < abs(small.bias)


def _null_dgp() -> DGP:
    """A process with confounding but genuinely no treatment effect."""

    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.5 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del a, z  # the treatment does not enter: the sharp null holds
        return 1.0 + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2]

    return DGP(
        name="null_effect",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


class TestTypeIError:
    def test_the_rejection_rate_is_near_the_nominal_level(self) -> None:
        dgp = _null_dgp()
        assert dgp.truth()["ate"] == pytest.approx(0.0, abs=1e-9)
        summary = _study(dgp, n=1000)["ate"]
        # Under the null, "reject at 5%" should happen 5% of the time. The binomial
        # standard error at 400 replications is about 1.1 points.
        assert abs(summary.rejection_rate - 0.05) < 3.0 * np.sqrt(
            0.05 * 0.95 / summary.n_replicates
        ), summary
        assert 0.93 <= summary.coverage <= 0.97

    def test_the_null_estimate_is_unbiased(self) -> None:
        summary = _study(_null_dgp(), n=1000)["ate"]
        assert abs(summary.bias) < 3.0 * summary.bias_se


class TestCvTmle:
    """Cross-fitted TMLE and CV-TMLE, and what separates them empirically.

    **Cross-fitted nuisances** (``cross_fit=True``) remove the Donsker condition on the
    nuisance *estimators* -- a smoothness condition that aggressive machine learning
    cheerfully violates.  Fit out of fold, no model predicts a row it was trained on,
    the empirical-process term involving them vanishes, and the influence-curve variance
    stops being understated.  That is the effect these studies are able to resolve, and
    it is large.

    **Fold-wise targeting** (``targeting_scheme="fold"``) solves the fluctuation inside
    each validation fold instead of once over all of them, which is the construction
    Zheng & van der Laan (2011) analyse.  Pooled targeting on cross-fitted nuisances is
    a different estimator -- the cross-fitted TMLE of the debiased-ML literature -- whose
    own empirical-process term is controlled by a separate argument: conditional on the
    training-fold fits ``Qbar`` is fixed, and ``{Qbar(epsilon)}`` is then indexed by a
    fixed finite-dimensional coefficient over a compact set -- two entries here, one per
    arm -- so it is Donsker however complex ``Qbar`` is.  That argument covers the
    empirical-process term only; the product-rate remainder, positivity and ``L_2``
    convergence of the influence curve are still required, and are what these studies
    are really exercising.

    Their first-order limits coincide under those conditions, so the honest expectation
    is that they *agree* here, and agreement is what is asserted.  Note what that does
    and does not establish: it is evidence about this process at this sample size, not a
    general interchangeability result, and the two constructions' finite-sample
    behaviour and remainder arguments differ in general.
    """

    @staticmethod
    def _overfitting_study(reps: int = 150, **overrides: object) -> object:
        # A tree grown to purity is the textbook Donsker violation: it interpolates
        # its training rows, so in-sample residuals are far too small.
        from sklearn.tree import DecisionTreeRegressor

        settings = {
            "outcome_learner": DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            # A parametric propensity on purpose. An overfitting *treatment* learner
            # drives g to 0 and 1, and the truncation that follows swamps the effect
            # being measured with a positivity artefact.
            "treatment_learner": "glm",
            "n_folds": 5,
            "learner_folds": 3,
            "estimands": ("ate",),
            "simultaneous": False,
            "random_state": 0,
            **overrides,
        }
        return CoverageStudy(
            dgp=nonlinear_dgp(),
            estimator=lambda: TMLE(**settings),
            n=500,
            n_replicates=reps,
            estimands=("ate",),
            seed=11,
            n_jobs=2,
        ).run()["ate"]

    def test_cross_fitting_restores_calibrated_standard_errors(self) -> None:
        in_sample = self._overfitting_study(cross_fit=False)
        cross_fitted = self._overfitting_study(cross_fit=True)

        # se_ratio is the reported standard error divided by the actual spread of the
        # estimates, so it isolates the inference from the estimation: a value near 1
        # means the standard error is honest whatever the bias is doing.
        assert in_sample.se_ratio < 0.75, in_sample
        assert cross_fitted.se_ratio > 0.85, cross_fitted
        assert cross_fitted.coverage > in_sample.coverage + 0.15

    def test_fold_wise_and_pooled_targeting_agree(self) -> None:
        pooled = self._overfitting_study(cross_fit=True, targeting_scheme="pooled")
        fold_wise = self._overfitting_study(cross_fit=True, targeting_scheme="fold")

        # The two constructions share a first-order limit under the conditions in the
        # class docstring, all of which hold here, so equivalence is what theory predicts
        # and improvement is not. Asserting equivalence is the point: if this test ever
        # showed fold-wise targeting *winning* on coverage, one of those conditions would
        # have stopped holding and the pooled path would be the thing to go and look at.
        assert abs(fold_wise.coverage - pooled.coverage) < 0.05, (pooled, fold_wise)
        assert fold_wise.se_ratio > 0.85
        assert abs(fold_wise.rmse - pooled.rmse) < 0.1 * pooled.rmse

    def test_cross_fitting_protects_coverage_with_a_super_learner(self) -> None:
        """The same comparison with the shipped ``"fast"`` library rather than a tree.

        A boosted ensemble is less pathological than an interpolating tree, so the
        effect is smaller -- but it is the configuration a user actually reaches for,
        which is why it is worth measuring separately.
        """
        common = {
            "outcome_learner": "fast",
            "treatment_learner": "fast",
            "n_folds": 5,
            "learner_folds": 3,
            "estimands": ("ate",),
            "simultaneous": False,
            "random_state": 0,
        }
        results = {}
        for label, cross_fit in (("cross-fitted", True), ("in-sample", False)):
            results[label] = CoverageStudy(
                dgp=nonlinear_dgp(),
                estimator=lambda cf=cross_fit: TMLE(cross_fit=cf, **common),
                n=600,
                n_replicates=120,
                estimands=("ate",),
                seed=99,
                n_jobs=2,
            ).run()["ate"]

        assert results["cross-fitted"].coverage > results["in-sample"].coverage
        assert results["cross-fitted"].coverage >= 0.90


class TestCollaborativeTmle:
    """C-TMLE: does choosing ``g`` against the target parameter actually pay?

    The claim from van der Laan & Gruber (2010) is not that C-TMLE is less biased in
    general -- with a correct propensity model a plain TMLE is already consistent --
    but that it stops paying variance for covariates that buy no bias reduction.  So
    the quantities to watch are the standard error and the root mean squared error,
    and the comparison has to be against a plain TMLE on the very same samples.
    """

    @staticmethod
    def _pair(dgp: DGP, *, n: int = 1000, reps: int = 120, **overrides: object) -> dict:
        common = {
            "outcome_learner": "glm",
            "treatment_learner": "glm",
            "n_folds": 5,
            "learner_folds": 3,
            "estimands": ("ate",),
            "simultaneous": False,
            "random_state": 0,
        }
        out = {}
        for label, factory in (
            ("tmle", lambda: TMLE(**common)),
            ("ctmle", lambda: CTMLE(**{**common, **overrides})),
        ):
            out[label] = CoverageStudy(
                dgp=dgp,
                estimator=factory,
                n=n,
                n_replicates=reps,
                estimands=("ate",),
                seed=11,
                n_jobs=2,
            ).run()["ate"]
        return out

    def test_it_does_not_pay_variance_for_an_instrument(self) -> None:
        # W2 predicts treatment strongly and the outcome not at all. A plain TMLE puts
        # it in g because a propensity learner is scored on treatment prediction; the
        # cost is a 1/g that reaches further into the tails for no bias reduction.
        studies = self._pair(instrument_dgp())
        tmle, ctmle = studies["tmle"], studies["ctmle"]

        # Measured when written: se 0.064 vs 0.087, rmse 0.075 vs 0.097, coverage
        # 0.875 vs 0.908. The thresholds sit well inside those gaps.
        assert ctmle.mean_std_error < 0.9 * tmle.mean_std_error, studies
        assert ctmle.rmse < 0.9 * tmle.rmse, studies
        # A floor rather than a nominal-rate check. C-TMLE's interval is narrower, and
        # its standard error does not price in the selection, so a small shortfall
        # against 0.95 is expected here; a collapse would not be. The Monte Carlo
        # standard error at 120 replications is about 0.03.
        assert ctmle.coverage >= 0.83, ctmle

    def test_the_scalable_search_matches_the_greedy_one(self) -> None:
        # Ju et al.'s ordered search costs O(p) fits instead of O(p^2). If it gave up
        # much accuracy for that there would be no reason to offer it.
        studies = self._pair(instrument_dgp(), search="ordered")
        assert studies["ctmle"].rmse < 0.9 * studies["tmle"].rmse, studies

    def test_it_degrades_more_gracefully_under_weak_overlap(self) -> None:
        # Practical positivity violation: a few units carry enormous weight. Truncation
        # is the blunt fix and it trades bias for variance; a collaborative selection
        # can instead decline the covariates that caused the violation.
        studies = self._pair(weak_overlap_dgp())
        tmle, ctmle = studies["tmle"], studies["ctmle"]

        # Measured when written: rmse 0.116 vs 0.191, coverage 0.783 vs 0.708. Neither
        # covers at the nominal rate -- this process is brutal at n=1000 and that is
        # the point of it -- so the comparison is relative on purpose.
        assert ctmle.rmse < 0.8 * tmle.rmse, studies
        assert ctmle.coverage > tmle.coverage, studies

    def test_collaborative_double_robustness(self) -> None:
        """A correct outcome model lets ``g`` adjust for almost nothing, unbiasedly.

        On this process a GLM is correctly specified for ``Qbar``, so the confounding
        is already handled before ``g`` is asked for anything.  C-TMLE notices and
        selects a nearly empty propensity model -- and the estimate stays unbiased.
        That is collaborative double robustness: the two nuisance fits only have to be
        right *between* them.
        """
        summary = self._pair(instrument_dgp(), reps=200)["ctmle"]
        assert abs(summary.bias) < 3.0 * summary.bias_se, summary


class TestClusteredInference:
    def test_cluster_robust_intervals_cover_when_plain_ones_do_not(self) -> None:
        from cleverly.datasets import clustered_dgp

        dgp = clustered_dgp(cluster_size=10)
        columns = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
        ignoring = _study(dgp, n=1000, reps=200, fit_kwargs=columns)["ate"]
        clustered = _study(dgp, n=1000, reps=200, fit_kwargs={**columns, "id": "cluster"})["ate"]
        # The DGP shares an unobserved latent within clusters. Ignoring it understates
        # the variance and coverage collapses; accounting for it restores calibration.
        assert ignoring.coverage < 0.90
        assert clustered.coverage > ignoring.coverage + 0.03
        assert clustered.se_ratio > ignoring.se_ratio


class TestWeightedInference:
    """The survey use case, end to end: does the weighted interval cover?

    Everything the fast tier can check about weighting is exact -- the estimand, the
    influence curve, the conventions.  What it cannot check is the claim a user actually
    relies on: that with known, unequal selection probabilities the weighted interval
    covers the *population* parameter at the nominal rate.  That is a statement about
    repeated sampling from the population, so it needs repeated sampling from the
    population.

    :func:`~cleverly.datasets.make_biased_sample` supplies it, with selection depending
    on ``W1`` and the treatment effect varying in ``W1``, so the selected sample's ATE is
    a different number from the population's.  The unweighted study is run alongside as
    the control: it is not a broken estimator, it is a correct estimator of the other
    parameter, and seeing its coverage collapse is what shows the weighting is load
    bearing rather than decorative.
    """

    def _study_weighted(self, *, weights: str | None, reps: int = 200) -> object:
        from cleverly.datasets import make_biased_sample

        columns: dict[str, object] = {
            "outcome": "Y",
            "treatment": "A",
            "covariates": ["W1", "W2"],
        }
        if weights is not None:
            columns["weights"] = weights
        return _study(make_biased_sample, n=3000, reps=reps, fit_kwargs=columns)["ate"]

    def test_weighted_intervals_cover_the_population_estimand(self) -> None:
        summary = self._study_weighted(weights="sampling_weight")
        # 200 replications put the Monte Carlo standard error of the coverage at about
        # 1.5 percentage points, so a +/- 0.04 window is roughly 2.5 of them.
        assert 0.91 <= summary.coverage <= 0.99, summary
        assert abs(summary.bias) < 3.0 * summary.bias_se, summary
        # The outcome model is a GLM and the effect is heterogeneous, so Qbar is
        # misspecified while g is not. Double robustness keeps the estimate consistent,
        # and the influence-curve standard error is then conservative rather than
        # optimistic -- which is the safe direction, and worth asserting as such.
        assert summary.se_ratio > 0.95, summary

    def test_ignoring_the_weights_estimates_the_other_parameter(self) -> None:
        summary = self._study_weighted(weights=None, reps=100)
        # Not a defect of the unweighted fit: it is consistent for the ATE among the
        # selected, which is roughly 0.5 away from the population ATE here. Coverage of
        # the population value collapses, which is exactly why the estimand statement
        # in cleverly.data.weighting has to be part of the reported output.
        assert summary.coverage < 0.2, summary
        assert summary.bias > 0.3, summary
