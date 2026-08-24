r"""The statistical validation tier: does the estimator's *inference* work?

Everything in the fast tier checks that the estimator computes what it claims on a
given sample.  This tier checks the claim that actually matters to a user reading a
p-value: that the reported uncertainty is honest.

The three headline properties -- nominal coverage, root-n consistency, and type-I error
under a null -- are no longer asserted here.  They are registered cells of the method
evidence studies instead, declared in :mod:`tests.studies.canonical_properties` and
published with their margins, their Monte Carlo intervals, and their controls in
``docs/technical-reference/method-evidence/``.  That move is what let each of them gain a
control that makes the same instrument fail: a both-wrong nuisance pair beside double
robustness, a power cell beside type-I error, a deliberately in-sample fit beside
cross-fitting.  A bare threshold in a test function could not carry any of that.

What remains here is everything those cells do not reach, and it is the larger half:

**Estimand and scale coverage.**  Every reported estimand, ratio estimands on their log
scale, and flexible nuisance learners at a size the fast tier cannot afford.

**Comparative claims about the variants.**  What CV-TMLE and C-TMLE are *for* is a
statement about repeated sampling, and no single fit can show it.  These compare two
configurations on the same processes rather than asserting an absolute level, because the
claims themselves are comparative -- C-TMLE pays less variance than TMLE for an
instrument; cross-fitting keeps the standard error honest where in-sample fitting does
not; repeated folds shrink the spread across fold draws.

**Designs with no registered study row.**  Clustered and weighted inference, missing
outcomes, controlled direct effects, incremental interventions, weighted longitudinal fits,
and competing risks.  Each carries its own coverage and root-n checks, because none of them
is covered by a point-treatment property cell.

Two longitudinal families used to be in that list and are not any more.
``TestLongitudinalInference`` and ``TestASurvivalOutcomeUnderRepeatedSampling`` were removed
once ``canonical-ltmle`` and ``canonical-ltmle-survival`` registered rows that dominate them:
the same laws, four times the replications, both static and dynamic plans, both horizons, and
-- which is the part a threshold in a test function cannot carry -- a control beside every
positive cell.  Neither class had ever run at its declared settings, so the registered rows
are the first repeated-sampling evidence those two estimands have actually had.  Removing them
follows the move recorded above rather than extending it: a claim leaves this file when a
registered cell establishes it more strongly, and not before.

These runs take minutes rather than seconds, so they are marked ``slow``.  The thresholds
are set from the Monte Carlo standard error of each quantity, so a pass is evidence rather
than a formality.

What is *not* here, deliberately: the misspecification grid crossed with overlap.  Both
arms of it live in :mod:`tests.e2e.test_double_robustness` and run in the fast tier, where
they are cheap enough to be checked on every push -- and where the finding they carry (the
propensity half of double robustness does not survive a practical positivity violation,
while the outcome half does) was measured rather than assumed.  The flexible-learner bias
claim lives there too, in ``TestFlexibleLearners``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pytest
import sklearn.ensemble
import sklearn.linear_model

from cleverly.datasets import (
    DGP,
    binary_outcome_dgp,
    cde_dgp,
    instrument_dgp,
    linear_dgp,
    make_longitudinal_competing,
    make_longitudinal_weighted,
    make_multi_arm,
    missing_outcome_dgp,
    multi_arm_dgp,
    nonlinear_dgp,
    weak_overlap_dgp,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.interventions import Incremental
from cleverly.learners import SuperLearner
from cleverly.longitudinal import LTMLE
from cleverly.validation import CoverageStudy
from tests.parallel import STUDY_JOBS

pytestmark = pytest.mark.slow

#: Replications per study.  At 400 the Monte Carlo standard error of a coverage estimate
#: near 0.95 is about 1.1 percentage points, enough to resolve a real shortfall.
REPLICATES = 400


def _study(
    dgp: DGP | Callable[..., tuple[Any, dict[str, float]]],
    *,
    n: int,
    reps: int = REPLICATES,
    estimands: tuple[str, ...] = ("ate",),
    flexible: bool = False,
    fit_kwargs: dict[str, object] | None = None,
    intermediate_value: float | None = None,
    seed: int = 2024,
) -> object:
    return CoverageStudy(
        dgp=dgp,
        estimator=lambda: TMLE(
            outcome_learner=(
                SuperLearner() if flexible else sklearn.linear_model.LinearRegression()
            ),
            treatment_learner=(
                SuperLearner()
                if flexible
                else sklearn.linear_model.LogisticRegression(max_iter=1000)
            ),
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
        intermediate_value=intermediate_value,
        seed=seed,
        n_jobs=STUDY_JOBS,
    ).run()


class TestEstimandAndLearnerCoverage:
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
        summary = _study(nonlinear_dgp(), n=1000, reps=120, flexible=True)["ate"]
        assert 0.90 <= summary.coverage <= 0.99, summary


class TestRepeatedCrossFitting:
    """What ``repeats=`` is for, which no single fit can show.

    Two separate claims, and only the first is what the option is *sold* on:

    * the spread of ``psi`` across fold draws shrinks -- measured directly, on one fixed
      dataset, so the only thing varying is the split;
    * coverage is unharmed -- the averaged interval is not buying its stability by
      understating the uncertainty that remains.

    The first needs a fixed sample and many seeds; the second needs many samples.  They
    are different experiments and are run as such rather than folded into one study.
    """

    #: Fold seeds per configuration.  The quantity measured is a standard deviation, whose
    #: own relative Monte Carlo error is about ``1/sqrt(2(m-1))`` -- 16% at 20 draws, which
    #: comfortably resolves the gap asserted below and costs 20 * (1 + 5) = 120 fits.
    SEEDS = 20

    @staticmethod
    def _spread(frame: Any, repeats: int, seeds: int) -> float:
        """Standard deviation of ``psi`` across fold draws, on one fixed dataset."""
        estimates = []
        for seed in range(seeds):
            fit = TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                estimands=("ate",),
                simultaneous=False,
                repeats=repeats,
                random_state=1000 + seed,
            ).fit(frame, outcome="Y", treatment="A")
            estimates.append(fit.single().psi("ate"))
        return float(np.std(estimates, ddof=1))

    def test_repeats_shrink_the_spread_across_fold_draws(self) -> None:
        # The data are held fixed, so every difference between these fits comes from the
        # split. A moderate n where the fold draw actually matters: at n = 20000 the
        # nuisance fits barely move between splits and there would be nothing to average.
        frame, _ = nonlinear_dgp().sample(600, seed=7)
        single = self._spread(frame, repeats=1, seeds=self.SEEDS)
        averaged = self._spread(frame, repeats=5, seeds=self.SEEDS)
        # Measured here at sd 0.0132 -> 0.0065, a ratio of 0.49 against the 1/sqrt(5) =
        # 0.45 that independent draws would give -- so the fold noise behaves very nearly
        # as an independent component, which is the premise the option rests on. The
        # threshold is 0.8 rather than 0.5 because the quantity is a standard deviation at
        # 20 seeds, whose own relative error is about 16%.
        assert averaged < 0.8 * single, f"single={single:.5f} averaged={averaged:.5f}"

    @staticmethod
    def _repeat_study(repeats: int) -> Any:
        # Fewer replications than the parametric studies because the pair costs 4 fits
        # per replication; 150 resolves coverage to about +/- 1.8 points, enough for the
        # windows below, which are comparative rather than tight.
        return CoverageStudy(
            dgp=linear_dgp(),
            estimator=lambda: TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                estimands=("ate",),
                simultaneous=False,
                repeats=repeats,
                random_state=0,
            ),
            n=1000,
            n_replicates=150,
            estimands=("ate",),
            seed=2024,
            n_jobs=STUDY_JOBS,
        ).run()["ate"]

    def test_averaging_costs_no_coverage_and_buys_a_conservative_interval(self) -> None:
        """The price of ``repeats=``, measured against the same process without it.

        Two configurations rather than an absolute window, because the claim is
        comparative and because the *direction* here is the interesting part.  Measured:
        ``se_ratio`` rises from about 1.0 at one draw to about 1.12 at three.

        That is not a defect and it is worth stating.  ``se_ratio`` is the mean reported
        standard error over the observed spread of the estimates across replications.  The
        reported one is the influence-curve standard error, which targets the *sampling*
        variance at the efficiency bound and never claimed to include fold noise.  The
        observed spread does include it -- and averaging is precisely what takes it out.
        So the numerator holds still while the denominator shrinks, and the interval ends
        up slightly wider than it strictly needs to be.

        The failure mode this rules out is the opposite one: an interval that looks stable
        because the averaged curve came out too small, which would show up as ``se_ratio``
        falling below one and coverage falling with it.
        """
        single = self._repeat_study(1)
        averaged = self._repeat_study(3)

        assert 0.93 <= averaged.coverage <= 0.99, averaged
        # Not harmed relative to a single draw, allowing for Monte Carlo error on both.
        assert averaged.coverage >= single.coverage - 3.0 * single.coverage_se, (
            single,
            averaged,
        )
        # The estimates themselves are more stable across replications.
        assert averaged.monte_carlo_se <= single.monte_carlo_se
        # And the interval is conservative rather than optimistic -- the direction above.
        assert averaged.se_ratio >= single.se_ratio
        assert averaged.se_ratio >= 1.0, averaged


class TestCvTmle:
    """Levy's default stacked CV-TMLE and the per-fold epsilon extension.

    **Cross-fitted nuisances** (``cross_fit=True``) remove the Donsker condition on the
    nuisance *estimators* -- a smoothness condition that aggressive machine learning
    cheerfully violates.  Fit out of fold, no model predicts a row it was trained on,
    the empirical-process term involving them vanishes, and the influence-curve variance
    stops being understated.  That is the effect these studies are able to resolve, and
    it is large.

    **Stacked CV-TMLE** (the default ``targeting_scheme="pooled"``) fits one common
    epsilon over all out-of-fold rows and evaluates their stitched distribution, as
    Levy's easy implementation defines. The pinned ``tmle3`` snapshot corroborates that
    path. The original fold-evaluated construction instead averages fold empirical risks
    and plug-ins. **Fold-specific targeting** solves a separate fluctuation inside each
    validation fold and is a package extension. Conditional on the training-fold fits,
    ``Qbar_v`` is fixed and the common fluctuated family is finite-dimensional. The
    product-rate remainder, positivity and ``L_2`` convergence of the influence curve
    are still required, and are what these studies are really exercising.

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
            "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
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
            n_jobs=STUDY_JOBS,
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
        # showed the extension *winning* on coverage, one of those conditions would have
        # stopped holding and the common-update path would be the thing to inspect.
        assert abs(fold_wise.coverage - pooled.coverage) < 0.05, (pooled, fold_wise)
        assert fold_wise.se_ratio > 0.85
        assert abs(fold_wise.rmse - pooled.rmse) < 0.1 * pooled.rmse

    def test_cross_fitting_protects_coverage_with_a_super_learner(self) -> None:
        """The same comparison with the automatic estimator-object library rather than a tree.

        A boosted ensemble is less pathological than an interpolating tree, so the
        effect is smaller -- but it is the configuration a user actually reaches for,
        which is why it is worth measuring separately.
        """
        common = {
            "outcome_learner": sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
            "treatment_learner": sklearn.ensemble.HistGradientBoostingClassifier(random_state=0),
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
                n_jobs=STUDY_JOBS,
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

    Read what these comparisons are evidence *for* with some care.  Both processes used
    here have an outcome model a GLM fits well, so an empty propensity model is a
    legitimate mean-squared-error-minimising choice and C-TMLE frequently makes it --
    10 of 10 seeds for the greedy search on ``instrument_dgp`` at ``n = 700``.  A selector
    hard-wired to return the empty model therefore passes every assertion in this class.
    What they establish is that carrying an instrument in ``g`` costs variance and that
    C-TMLE declines to; they do not establish that the search discriminates among
    covariates, and dominance in a finite simulation would not establish it either -- a
    valid implementation is not guaranteed to win, and a consistent win can just as easily
    come from an over-eager penalty.

    The claim that the search *selects* what it needs is made where selecting nothing is
    wrong, in :class:`tests.e2e.test_ctmle.TestSelectionIsForcedWhenTheOutcomeModelCannotHelp`,
    which fails outright under a degenerate selector.  These studies are the complementary
    half: they price the variance C-TMLE saves, in a regime where the saving is real.
    """

    @staticmethod
    def _pair(dgp: DGP, *, n: int = 1000, reps: int = 120, **overrides: object) -> dict:
        common = {
            "outcome_learner": sklearn.linear_model.LinearRegression(),
            "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
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
                n_jobs=STUDY_JOBS,
            ).run()["ate"]
        return out

    def test_it_does_not_pay_variance_for_an_instrument(self) -> None:
        # W2 predicts treatment strongly and the outcome not at all. A plain TMLE puts
        # it in g because a propensity learner is scored on treatment prediction; the
        # cost is a 1/g that reaches further into the tails for no bias reduction.
        studies = self._pair(instrument_dgp())
        tmle, ctmle = studies["tmle"], studies["ctmle"]

        # Measured after nested selection cross-fitting (120 replicates): se 0.0641
        # vs 0.0872, rmse 0.0773 vs 0.0972, coverage 0.867 vs 0.908. The
        # thresholds sit well inside those gaps.
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
        studies = self._pair(instrument_dgp(), strategy="ordered")
        assert studies["ctmle"].rmse < 0.9 * studies["tmle"].rmse, studies

    def test_it_degrades_more_gracefully_under_weak_overlap(self) -> None:
        # Practical positivity violation: a few units carry enormous weight. Truncation
        # is the blunt fix and it trades bias for variance; a collaborative selection
        # can instead decline the covariates that caused the violation.
        studies = self._pair(weak_overlap_dgp())
        tmle, ctmle = studies["tmle"], studies["ctmle"]

        # Measured after nested selection cross-fitting (120 replicates): rmse 0.117
        # vs 0.191, coverage 0.767 vs 0.708. Neither covers at the nominal rate --
        # this process is brutal at n=1000 and that is the point of it -- so the
        # comparison is relative on purpose.
        assert ctmle.rmse < 0.8 * tmle.rmse, studies
        assert ctmle.coverage > tmle.coverage, studies

    def test_collaborative_double_robustness(self) -> None:
        """A correct outcome model lets ``g`` adjust for almost nothing, unbiasedly.

        On this process a GLM is correctly specified for ``Qbar``, so the confounding
        is already handled before ``g`` is asked for anything.  C-TMLE notices and
        selects a nearly empty propensity model -- usually an exactly empty one -- and the
        estimate stays unbiased.  That is collaborative double robustness: the two nuisance
        fits only have to be right *between* them.

        This is also the one assertion in the class that a correct implementation must pass
        unconditionally.  The root-mean-squared-error comparisons above are contingent on
        the process, the sample size and how much variance the discarded covariates were
        costing, so a valid implementation can lose them; consistency is not contingent.  A
        selection that bought its variance reduction with bias -- what an over-eager penalty
        would produce -- fails here and passes everything else.
        """
        summary = self._pair(instrument_dgp(), reps=200)["ctmle"]
        assert abs(summary.bias) < 3.0 * summary.bias_se, summary


def _off_diagonal_dgp() -> DGP:
    """Linear outcome, nonlinear treatment: ``glm`` is correct for one nuisance and not the other.

    The off-diagonal cell of
    :mod:`tests.e2e.test_double_robustness`'s grid, with the "correct" nuisance
    *estimated* rather than an oracle.  That distinction is the whole point: with an oracle
    the good nuisance is exactly right, ``R_2`` is exactly zero, and a plain TMLE's interval
    is already valid, so a study built that way would have nothing to show.
    """
    linear, hard = linear_dgp(), nonlinear_dgp()
    return DGP(
        name="linear outcome, nonlinear treatment",
        n_latent=4,
        covariate_names=("W1", "W2", "W3", "W4"),
        propensity=hard.propensity,
        outcome_mean=linear.outcome_mean,
    )


class TestDoublyRobustInference:
    """``DRTMLE``: an interval that stays valid when only one nuisance is consistent.

    ``TMLE`` is doubly robust for *consistency* and singly robust for *inference*: the
    remainder carries both nuisance errors, so one consistent nuisance keeps
    ``psi-hat`` consistent while the interval needs the strictly stronger
    ``sqrt(n) R_2 -> 0``.  Solving two further score equations against reduced-dimension
    regressions is supposed to close that gap.

    **This is a catastrophic-regression guard at fixed seeds, and nothing more.**  It asserts
    that estimates are finite, that no replicate was silently dropped, that the reported
    standard error is the right order of magnitude, and that coverage has not collapsed.  It
    deliberately does **not** require coverage near ``0.95``: coverage is a property of the
    learner, the DGP, the sample size and the dependency versions, not a deterministic
    software invariant, and a nominal floor here would be a version-pinned flake that fails
    for reasons no commit caused.  ``docs/drtmle.md`` is where the release claim lives, and it
    is *conditional* validity -- the interval is valid given adequate nuisance fits, which is
    a rate condition no simulation in a nightly budget establishes.

    The cells that showed what the variant buys -- ``q-drift`` and ``g-drift`` with injected
    nuisance sequences, where it recovered 74.6% and 78.7% of the plain estimator's coverage
    deficit -- belonged to the validation study's harness and were retired with it.  They are
    reachable from the ``drtmle-validation-archive-2026-08`` tag and are not reproducible
    here; do not read this class as standing in for them.

    **What it can show, and what it cannot.**  It prices what a nightly budget can
    reach: that the point estimate is still doubly robust and that the doubly-robust
    interval does not *cost* coverage where the plain one already has it.  It does **not**
    demonstrate the headline claim, and the reason is a measurement rather than a hedge.

    A pilot at ``n = 500`` over 24 replicates on the process below -- a correctly specified
    *parametric* outcome model against a misspecified propensity -- put both estimators at
    coverage 0.958, with biases of -0.013 and -0.008 against a Monte Carlo standard error of
    0.018.  The mirror cell (nonlinear outcome, linear treatment) put both at 1.000.  There
    was nothing to buy, and that is not a defect: a correctly specified parametric nuisance
    converges at ``n^(-1/2)``, so the product condition is nowhere near binding and the
    remainder is a small constant times ``n^(-1/2)`` rather than a first-order term.

    The regime this variant is *for* is an **adaptive** good nuisance converging more slowly
    than ``n^(-1/4)`` -- a Super Learner in enough dimensions -- at an ``n`` large enough for
    the coverage decay to show.  That is out of reach here rather than uninteresting: the
    pilot's two ``DRTMLE`` studies took 358s and 372s against the plain estimator's 5s and 3s,
    because the alternation fits three reduced families per arm on every round. Scaling
    that to flexible learners at ``n = 2000`` over 200 replicates is hours, not minutes.

    So: this class is a guard, not a demonstration, and it says so rather than letting a
    passing nightly run read as evidence for something it did not test.
    """

    #: One cell rather than the pair the pilot ran. The mirror adds ten minutes to the
    #: nightly tier for a claim this one already makes, and the finding above is the same
    #: in both.
    N = 500
    REPLICATES = 40

    @staticmethod
    def _pair(dgp: DGP, *, n: int, reps: int) -> dict:
        common = {
            "outcome_learner": sklearn.linear_model.LinearRegression(),
            "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
            "n_folds": 4,
            "learner_folds": 3,
            "estimands": ("ate",),
            "simultaneous": False,
            "random_state": 0,
        }
        out = {}
        for label, factory in (
            ("tmle", lambda: TMLE(**common)),
            ("drtmle", lambda: DRTMLE(**common)),
        ):
            # The whole :class:`~cleverly.validation.simulation.StudyResult`, not just the
            # ``ate`` summary: ``n_failed`` lives on it, and a silently dropped replicate is
            # exactly the failure this tier exists to catch.
            out[label] = CoverageStudy(
                dgp=dgp,
                estimator=factory,
                n=n,
                n_replicates=reps,
                estimands=("ate",),
                seed=11,
                n_jobs=STUDY_JOBS,
            ).run()
        return out

    @pytest.fixture(scope="class")
    def studies(self) -> dict:
        return self._pair(_off_diagonal_dgp(), n=self.N, reps=self.REPLICATES)

    def test_no_replicate_was_silently_dropped(self, studies: dict) -> None:
        """Fails closed: a study that quietly lost half its draws would pass everything else.

        Every assertion below is an average over replicates, so a harness that swallowed the
        hard ones would report better numbers on the survivors and read as a healthier
        estimator.  This is the first thing to check and the cheapest.
        """
        for label, study in studies.items():
            assert study.n_failed == 0, (label, study.n_failed)
            assert study.n_replicates == self.REPLICATES, label

    def test_every_estimate_and_standard_error_is_usable(self, studies: dict) -> None:
        """The floor beneath every other claim: finite numbers, and a positive interval width.

        A ``nan`` estimate or a zero standard error is a fit that failed without saying so.
        Asserted on the raw arrays rather than on their means, since one bad replicate in
        forty is invisible in an average and is exactly the regression worth catching.
        """
        summary = studies["drtmle"]["ate"]
        assert np.isfinite(summary.estimates).all(), summary.estimates
        assert np.isfinite(summary.std_errors).all(), summary.std_errors
        assert (summary.std_errors > 0.0).all(), summary.std_errors

    def test_the_point_estimate_is_still_doubly_robust(self, studies: dict) -> None:
        """The unconditional claim, and the one a broken implementation fails.

        Solving two more equations must not cost consistency: all three are solved at the
        same ``Qbar*``, and the extra terms are mean-zero by construction. A variant that
        bought its variance with bias fails here and could pass everything else.  This is a
        *deterministic* property of the construction rather than a coverage claim, which is
        why it keeps a Monte Carlo bound where the two below do not.
        """
        summary = studies["drtmle"]["ate"]
        assert abs(summary.bias) < 3.5 * summary.bias_se, studies

    def test_the_reported_standard_error_is_the_right_order_of_magnitude(
        self, studies: dict
    ) -> None:
        """A factor of two either way, which is a regression bound and not a calibration one.

        The band was ``+/- 0.2`` around 1.0, which is a *calibration* assertion: it fails on a
        dependency bump that moves the learner slightly, and such a failure names no commit.
        What is worth catching here is the order of magnitude -- a wrong sign in
        ``D = D* - D*_Q - D*_g`` doubles the correction rather than removing it, and a
        dropped term removes it entirely, both of which land far outside a factor of two.
        The calibration question belongs to ``docs/drtmle.md``'s nuisance conditions, which
        no simulation of this size settles.
        """
        ratio = studies["drtmle"]["ate"].se_ratio
        assert 0.5 <= ratio <= 2.0, studies

    def test_coverage_has_not_collapsed(self, studies: dict) -> None:
        """A catastrophic floor and a *relative* comparison, deliberately not a nominal one.

        ``0.6`` is far below anything a working estimator produces on this cell -- the pilot
        put both estimators at ``0.958`` -- and far above what a broken curve gives.  It is
        chosen to sit in the gap rather than to certify the interval: requiring ``0.95``, or
        even the ``0.88`` this once asserted, makes a nightly run a referendum on the
        learner's luck rather than on the package's code.

        The second assertion is the one with content, and it is relative: where the plain
        interval already covers there is nothing to improve, so what the doubly-robust one
        must not do is fall *away* from it.
        """
        drtmle, tmle = studies["drtmle"]["ate"], studies["tmle"]["ate"]
        assert drtmle.coverage > 0.6, studies
        assert drtmle.coverage >= tmle.coverage - 3.0 * drtmle.coverage_se, studies


class TestMultiArmCollaborativeCoverage:
    """Nightly regression guard for the multi-arm DRTMLE and OAT branches.

    This is deliberately the same modest claim as ``TestDoublyRobustInference``: finite
    estimates, no swallowed replicates, bias compatible with Monte Carlo error, an SE of
    the right order, and coverage that has not collapsed relative to ordinary TMLE.  It is
    not a nuisance-rate experiment and therefore does not establish the union-model
    theorem or generated-regressor asymptotics.
    """

    N = 500
    REPLICATES = 40
    ESTIMANDS = tuple(multi_arm_dgp().truth())

    @pytest.fixture(scope="class")
    def studies(self) -> dict[str, Any]:
        common = {
            "outcome_learner": sklearn.linear_model.LinearRegression(),
            "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
            "n_folds": 4,
            "learner_folds": 3,
            "estimands": ("ey", "ate"),
            "simultaneous": False,
            "random_state": 0,
        }
        factories = {
            "tmle": lambda: TMLE(**common),
            "drtmle": lambda: DRTMLE(**common),
            "oat": lambda: CTMLE(strategy="oat", **common),
            "selector": lambda: CTMLE(
                strategy="discrete",
                candidates=((), ("W1",), ("W1", "W2"), ("W1", "W2", "W3")),
                selection_folds=2,
                selection_inner_folds=2,
                ctmle_estimand="ate",
                **common,
            ),
        }
        return {
            label: CoverageStudy(
                dgp=make_multi_arm,
                estimator=factory,
                n=self.N,
                n_replicates=self.REPLICATES,
                estimands=self.ESTIMANDS,
                seed=31,
                n_jobs=STUDY_JOBS,
            ).run()
            for label, factory in factories.items()
        }

    def test_no_replicate_was_silently_dropped(self, studies: dict[str, Any]) -> None:
        for study in studies.values():
            assert study.n_replicates == self.REPLICATES
            assert study.n_failed == 0

    @pytest.mark.parametrize("variant", ["drtmle", "oat", "selector"])
    @pytest.mark.parametrize("estimand", ESTIMANDS)
    def test_bias_and_standard_errors_have_not_collapsed(
        self, studies: dict[str, Any], variant: str, estimand: str
    ) -> None:
        summary = studies[variant][estimand]
        assert abs(summary.bias) < 3.5 * summary.bias_se, (variant, summary)
        assert 0.5 <= summary.se_ratio <= 2.0, (variant, summary)

    @pytest.mark.parametrize("variant", ["drtmle", "oat", "selector"])
    @pytest.mark.parametrize("estimand", ESTIMANDS)
    def test_coverage_is_not_catastrophically_worse_than_tmle(
        self, studies: dict[str, Any], variant: str, estimand: str
    ) -> None:
        summary = studies[variant][estimand]
        plain = studies["tmle"][estimand]
        assert summary.coverage > 0.60, (variant, summary)
        assert summary.coverage >= plain.coverage - 3.0 * summary.coverage_se, (
            variant,
            plain,
            summary,
        )


def _make_multi_arm_ratios(
    n: int, *, seed: int | np.random.Generator | None = None
) -> tuple[Any, dict[str, float]]:
    frame, truth = make_multi_arm(n=n, seed=seed, family="binomial")
    reference = truth["ey[high]"]
    for label in ("low", "medium"):
        mean = truth[f"ey[{label}]"]
        truth[f"rr[{label} vs high]"] = mean / reference
        truth[f"or[{label} vs high]"] = (mean / (1.0 - mean)) / (reference / (1.0 - reference))
    return frame, truth


class TestMultiArmSelectorRatioCoverage:
    """Nightly guard for joint log-risk and log-odds selector targets."""

    N = 500
    REPLICATES = 40
    ESTIMANDS = (
        "rr[low vs high]",
        "rr[medium vs high]",
        "or[low vs high]",
        "or[medium vs high]",
    )

    @pytest.fixture(scope="class")
    def studies(self) -> dict[str, Any]:
        common = {
            "outcome_learner": sklearn.linear_model.LinearRegression(),
            "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
            "n_folds": 4,
            "learner_folds": 3,
            "estimands": ("rr", "or"),
            "simultaneous": False,
            "random_state": 0,
        }
        factories = {
            "tmle": lambda: TMLE(**common),
            "selector": lambda: CTMLE(
                strategy="discrete",
                candidates=((), ("W1",), ("W1", "W2"), ("W1", "W2", "W3")),
                selection_folds=2,
                selection_inner_folds=2,
                ctmle_estimand="rr",
                **common,
            ),
        }
        return {
            label: CoverageStudy(
                dgp=_make_multi_arm_ratios,
                estimator=factory,
                n=self.N,
                n_replicates=self.REPLICATES,
                estimands=self.ESTIMANDS,
                seed=8311,
                n_jobs=STUDY_JOBS,
            ).run()
            for label, factory in factories.items()
        }

    @pytest.mark.parametrize("estimand", ESTIMANDS)
    def test_ratio_inference_has_not_collapsed(
        self, studies: dict[str, Any], estimand: str
    ) -> None:
        summary = studies["selector"][estimand]
        plain = studies["tmle"][estimand]
        assert studies["selector"].n_failed == 0
        assert abs(summary.bias) < 3.5 * summary.bias_se, summary
        assert 0.5 <= summary.se_ratio <= 2.0, summary
        assert summary.coverage > 0.60, summary
        assert summary.coverage >= plain.coverage - 3.0 * summary.coverage_se, (plain, summary)


class TestClusteredInference:
    def test_cluster_robust_intervals_cover_when_plain_ones_do_not(self) -> None:
        from cleverly.datasets import clustered_dgp

        dgp = clustered_dgp(cluster_size=10)
        columns = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
        ignoring = _study(dgp, n=1000, reps=200, fit_kwargs=columns)["ate"]
        clustered = _study(dgp, n=1000, reps=200, fit_kwargs={**columns, "id": "cluster"})["ate"]
        # The DGP shares an unobserved latent within clusters, as an effect modifier rather
        # than a confounder, so the ATE stays identified from W1 and W2 while the influence
        # curves stay correlated. Ignoring that correlation understates the variance and
        # coverage collapses; accounting for it restores calibration.
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


#: ``fit_kwargs`` for the missing-outcome process.  ``CoverageStudy`` defaults to
#: ``{"outcome": "Y", "treatment": "A"}``, which would silently drop the indicator and
#: fail on the ``NaN`` outcomes, so the roles have to be spelled out.
MISSING_FIT_KWARGS: dict[str, object] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W1", "W2", "W3"],
    "delta": "Delta",
}

#: The same roles minus the indicator, for the complete-case control.
COMPLETE_CASE_FIT_KWARGS: dict[str, object] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W1", "W2", "W3"],
}


def _complete_cases(strength: float) -> Callable[..., tuple[Any, dict[str, float]]]:
    """A sampler that hands the estimator only the rows with a recorded outcome.

    ``CoverageStudy`` accepts any ``(n, seed) -> (frame, truth)`` callable, so the
    control needs no special support: draw from the same process, drop the unobserved
    rows, and keep the *population* truth as the reference.  That is the point --
    the complete-case analysis is being measured against the estimand it claims to
    estimate, not against the one it actually converges to.
    """
    dgp = missing_outcome_dgp(strength)

    def sample(n: int, seed: int | None = None) -> tuple[Any, dict[str, float]]:
        frame, truth = dgp.sample(n, seed=seed)
        return frame.dropna(subset=["Y"]), truth

    sample.__name__ = f"complete_case_x{strength:g}"
    return sample


class TestMissingOutcomes:
    r"""Does the interval cover when a third nuisance sits in the clever covariate?

    The exact modules -- ``tests/unit/test_influence_gateaux_mar.py`` and
    ``tests/unit/test_remainder_mar.py`` -- settle that the influence curve is the
    efficient one for the observed-data model and that its remainder is second-order.
    Neither says the *interval* built from it covers once the nuisances are estimated
    rather than handed over, which is the only thing a user reading a p-value cares
    about, and no study in this tier has ever run with ``delta=``.

    Note which process each test uses, because the two say different things.  At
    ``strength = 1`` the outcome mean is linear, a GLM is correctly specified for it, and
    missingness at random is enough for a complete-case analysis to be consistent -- so
    coverage there tests the inference machinery with the estimand already identified
    twice over.  At ``strength = 2`` the outcome mean is out of reach of the learner and
    the mechanism is sharp, so the estimate rests on ``1 / (g pi)`` alone; that is where
    modelling the missingness has to earn its place, and where ignoring it visibly fails.
    """

    @pytest.mark.parametrize("n", [500, 2000])
    def test_confidence_intervals_cover_at_the_nominal_rate(self, n: int) -> None:
        # Measured when written, at 400 replications: coverage 0.9625 at n=500 and
        # 0.9550 at n=2000, against a Monte Carlo standard error of about 0.010. For
        # comparison the long-standing no-missingness study on linear_dgp gives 0.9675
        # and 0.9325 on the same budget, so the delta path is not the noisier one.
        summary = _study(missing_outcome_dgp(), n=n, fit_kwargs=MISSING_FIT_KWARGS)["ate"]
        assert 0.93 <= summary.coverage <= 0.97, summary
        assert abs(summary.coverage - 0.95) < 3.0 * summary.coverage_se, summary

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc", "ey1", "ey0"])
    def test_every_estimand_covers(self, estimand: str) -> None:
        # Measured when written: 0.940, 0.950, 0.945, 0.950, 0.950 in the order below.
        summary = _study(
            missing_outcome_dgp(),
            n=1000,
            estimands=("ate", "att", "atc", "ey1", "ey0"),
            fit_kwargs=MISSING_FIT_KWARGS,
        )[estimand]
        assert 0.92 <= summary.coverage <= 0.98, summary

    def test_the_reported_standard_error_matches_the_actual_variability(self) -> None:
        # The place a missing-outcome fit would most plausibly go wrong: 1 / pi inflates
        # the influence curve, and an implementation that left it out of the variance
        # while keeping it in the point estimate would look fine on a single fit.
        # Measured when written: 1.002 at n=1000, and 0.98-1.04 across n=500..8000.
        summary = _study(missing_outcome_dgp(), n=1000, fit_kwargs=MISSING_FIT_KWARGS)["ate"]
        assert 0.93 <= summary.se_ratio <= 1.07, summary

    def test_root_n_bias_stays_bounded(self) -> None:
        sizes = (500, 2000, 8000)
        scaled, noise = [], []
        for n in sizes:
            summary = _study(missing_outcome_dgp(), n=n, reps=200, fit_kwargs=MISSING_FIT_KWARGS)[
                "ate"
            ]
            scaled.append(abs(summary.root_n_bias))
            noise.append(float(np.sqrt(n)) * summary.bias_se)
        # The bare ratio test the other studies in this module use needs a floor here,
        # because this process has no bias to detect. Measured at these settings the
        # scaled bias runs 0.125, 0.019, 0.292 against a Monte Carlo standard error of
        # 0.156, 0.172, 0.179 -- every one of them inside one standard error of zero. So
        # `min(scaled)` is a draw from noise, and `scaled[-1] < 2.5 * min(scaled)` would
        # have failed outright at 0.292 against 0.046. What the claim actually needs is
        # that sqrt(n) * bias does not *grow* by more than Monte Carlo error explains.
        assert scaled[-1] < max(2.5 * min(scaled), 3.0 * noise[-1]), dict(
            zip(sizes, scaled, strict=True)
        )
        for n, bias, mc in zip(sizes, scaled, noise, strict=True):
            assert bias < 3.5 * mc, f"n={n}: scaled bias {bias:.4f} vs Monte Carlo se {mc:.4f}"

    def test_it_covers_when_the_outcome_model_cannot_help(self) -> None:
        # strength=2: curvature and an A-by-W1 interaction a main-effects GLM cannot
        # reach, so the estimate rests on the mechanisms.  Coverage is allowed a wider
        # window than the correctly-specified case -- the remainder is no longer zero,
        # only second-order -- but it must still be near nominal rather than collapsing.
        # Measured at 400 replications: coverage 0.9450, bias -0.0117, se_ratio 1.005.
        summary = _study(missing_outcome_dgp(strength=2.0), n=2000, fit_kwargs=MISSING_FIT_KWARGS)[
            "ate"
        ]
        assert 0.90 <= summary.coverage <= 0.98, summary

    def test_ignoring_the_missingness_is_visibly_worse(self) -> None:
        """The failing control, without which the tests above prove nothing.

        A complete-case fit on the same process drops the rows with no outcome and never
        forms ``1 / pi``.  Because the outcome model is misspecified *and* the complete
        cases carry a shifted ``W1`` distribution, the linear approximation it fits is
        the wrong one to extrapolate over the full marginal, and the interval misses far
        more often than one time in twenty.

        Deliberately *not* run at ``strength = 1``: there the complete-case analysis is
        consistent, and a control that passed for that process would be measuring noise.

        Measured at 400 replications: modelling the missingness gives coverage 0.945 and
        a bias of -0.012; dropping the incomplete rows gives coverage 0.243 and a bias of
        +0.237, twenty times larger and in the opposite direction.
        """
        modelled = _study(missing_outcome_dgp(strength=2.0), n=2000, fit_kwargs=MISSING_FIT_KWARGS)[
            "ate"
        ]
        ignored = _study(_complete_cases(2.0), n=2000, fit_kwargs=COMPLETE_CASE_FIT_KWARGS)["ate"]
        assert ignored.coverage < 0.90, ignored
        assert ignored.coverage < modelled.coverage - 0.05, (modelled, ignored)
        assert abs(ignored.bias) > 2.0 * abs(modelled.bias), (modelled, ignored)


CDE_FIT_KWARGS: dict[str, object] = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W1", "W2", "W3"],
    "intermediate": "Z",
}


class TestControlledDirectEffectInference:
    """Coverage for the ``intermediate=`` estimand, one study per level.

    Until ``CoverageStudy`` learned to select a level this could not be run at all, so
    the controlled direct effect was the one estimand in the package with no empirical
    evidence that its intervals cover.  The fast tier still has none: as
    :mod:`cleverly.validation.score` notes, a wrong clever covariate used consistently
    solves its own score equation to machine precision, so nothing short of repeated
    sampling against a known truth can distinguish it.

    ``cde_dgp`` is chosen because its truth is exact rather than simulated -- the outcome
    mean is linear with a ``0.6 * a * z`` term, so the controlled direct effect is
    ``0.9 + 0.6 * z`` in closed form -- and because a main-effects ``glm`` outcome
    learner *cannot* represent that interaction.  The initial regression is therefore
    misspecified for this estimand by construction, while the propensity and the
    intermediate mechanism are logistic in ``W`` and correctly specified.  That makes
    these studies a direct test of the double-robustness claim the estimand actually
    relies on: consistency resting on the product ``g * q_z`` rather than on ``Qbar``.

    That regime also dictates what may be asserted.  With ``Qbar`` inconsistent the
    estimator is doubly robust but *not* efficient, and in that case -- as
    :mod:`cleverly.estimators.tmle` says in as many words -- the point estimate is still
    consistent while the influence-curve standard error generally is not.  So the
    coverage window here is wider than the correctly-specified studies above, and there
    is no ``se_ratio`` assertion: the theory does not promise one, and a test asserting
    it would be measuring luck.  What *is* asserted is the half the theory does give,
    that the bias vanishes.

    Thresholds below were set from a reduced-scale check at 100-150 replications rather
    than at the module's 400, since the slow tier does not run in development.  Treat the
    quoted numbers as indicative rather than as tight calibration.
    """

    @pytest.mark.parametrize(("z", "truth"), [(0.0, 0.9), (1.0, 1.5)])
    def test_confidence_intervals_cover_at_the_nominal_rate(self, z: float, truth: float) -> None:
        # Measured at 100 replications, n=2000: coverage 0.970 at z=0 and 0.920 at z=1,
        # against a Monte Carlo standard error of about 0.02 apiece.
        study = _study(cde_dgp(), n=2000, fit_kwargs=CDE_FIT_KWARGS, intermediate_value=z)["ate"]
        assert study.truth == pytest.approx(truth, abs=1e-6), study
        assert 0.90 <= study.coverage <= 0.99, study

    @pytest.mark.parametrize("z", [0.0, 1.0])
    def test_the_mechanisms_carry_the_estimate_when_the_outcome_model_cannot(
        self, z: float
    ) -> None:
        """The double-robustness claim, stated for the three-way product.

        A main-effects ``glm`` cannot represent the ``0.6 * a * z`` term, so the initial
        regression is wrong at both levels and everything that keeps the estimate on
        target comes from ``g`` and ``q_z``.  An implementation that dropped ``1 / q_z``
        from the clever covariate would still solve its own score equation and still pass
        every single-fit check in the suite; it would show up here, and only here, as a
        bias that does not vanish.

        Measured at 100 replications, n=2000: bias +0.003 at z=0 and +0.005 at z=1.
        """
        study = _study(cde_dgp(), n=2000, fit_kwargs=CDE_FIT_KWARGS, intermediate_value=z)["ate"]
        assert abs(study.bias) < max(3.0 * study.bias_se, 0.02), study

    def test_the_two_levels_are_different_parameters(self) -> None:
        # The failing control for the whole estimand: a study that never passed the level
        # through would compare both fits against the same truth, and this assertion --
        # that the two studies are centred 0.6 apart -- is what that mistake breaks.
        low, high = (
            _study(cde_dgp(), n=2000, reps=200, fit_kwargs=CDE_FIT_KWARGS, intermediate_value=z)[
                "ate"
            ]
            for z in (0.0, 1.0)
        )
        assert high.truth - low.truth == pytest.approx(0.6, abs=1e-9)
        gap = (high.truth + high.bias) - (low.truth + low.bias)
        assert gap == pytest.approx(0.6, abs=0.1), (low, high)


class TestAnIncrementalIntervention:
    """Coverage for the one estimand here that is not doubly robust.

    Two things need a repeated-sampling check that no single fit can give.  That the
    interval covers at all -- the estimator solves *two* score equations and the second
    one moves the mechanism, so an alternation that stopped a round early would show up
    as under-coverage and nowhere else.  And that it keeps covering under weak overlap,
    which is the claim the axis exists for: the clever covariate is bounded by
    ``max(delta, 1/delta)`` there, so nothing about the interval should degrade.

    The mechanism is estimated with a ``glm`` throughout and both processes have a
    propensity a ``glm`` can represent -- which is not a shortcut but a requirement.
    ``ey_ipsi`` has no doubly-robust fallback, so a misspecified mechanism would bias it
    and this study would be measuring that rather than coverage.
    """

    DELTAS = (1.0, 2.0)
    ESTIMANDS = ("ey_ipsi", "ate_ipsi")
    REPORTED = "ate_ipsi[odds x2 vs natural course]"

    @staticmethod
    def _process(dgp: DGP) -> Callable[..., tuple[Any, dict[str, float]]]:
        """Adapt a ``DGP`` to the ``(n, seed) -> (frame, truth)`` convention.

        The truth comes from :meth:`DGP.incremental_truth` rather than :meth:`DGP.truth`,
        because the tilts are not among the arm-indexed estimands that method reports.
        """

        def draw(n: int, seed: int) -> tuple[Any, dict[str, float]]:
            frame, _ = dgp.sample(n=n, seed=seed)
            return frame, dgp.incremental_truth(TestAnIncrementalIntervention.DELTAS)

        return draw

    def _run(self, dgp: DGP, *, n: int, reps: int) -> object:
        return CoverageStudy(
            dgp=self._process(dgp),
            estimator=lambda: TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                incremental=[Incremental(delta) for delta in self.DELTAS],
                estimands=self.ESTIMANDS,
                simultaneous=False,
                random_state=0,
            ),
            n=n,
            n_replicates=reps,
            fit_kwargs={"outcome": "Y", "treatment": "A"},
            seed=2024,
            n_jobs=STUDY_JOBS,
        ).run()

    def test_the_interval_covers_under_ordinary_overlap(self) -> None:
        study = self._run(linear_dgp(), n=1000, reps=REPLICATES)[self.REPORTED]
        assert study.coverage > 0.90, study

    def test_it_still_covers_where_the_arm_estimands_are_in_trouble(self) -> None:
        """The claim, under repeated sampling: no positivity assumption, no degradation."""
        study = self._run(weak_overlap_dgp(), n=1000, reps=REPLICATES)[self.REPORTED]
        assert study.coverage > 0.90, study
        assert abs(study.bias) < max(3.0 * study.bias_se, 0.02), study

    def test_the_natural_course_is_unbiased_by_construction(self) -> None:
        """``delta = 1`` is ``E[Y]``, so a bias here would be a bug and not a rate."""
        study = self._run(linear_dgp(), n=1000, reps=200)["ey_ipsi[natural course]"]
        assert abs(study.bias) < max(3.0 * study.bias_se, 0.01), study


class TestAWeightedLongitudinalFitUnderRepeatedSampling:
    """Does the weighted sequential fit cover, on a sample that was drawn on purpose?

    ``tests/unit/test_weighted_estimand_longitudinal.py`` proves the estimand and the
    influence curve exactly, on a law a sample realises and handed a saturated learner.
    What it cannot ask is whether the interval built from that curve covers under repeated
    sampling when the nuisances are estimated -- which is what this asks, on the one design
    where the answer is checkable against a truth nobody had to re-derive.

    ``make_longitudinal_weighted`` keeps each unit with a known probability
    :math:`\\pi(W_1)` and hands it :math:`w = 1/\\pi`.  Tilting the sampling law by those
    weights reproduces the original law *exactly*, so the truth is
    ``make_longitudinal``'s unchanged and the comparison is honest in both directions: a
    fit that applied the weights is estimating this parameter, and one that ignored them is
    estimating the selected population's -- which selection on ``W1`` makes a different
    number, since ``W1`` moves both treatment decisions and the outcome.

    **This class has not run at 400 replicates**, and is recorded that way rather than
    implied to have passed. What *was* run, at ``n=2000``, is 60 replicates at one seed and 120
    at another.  Bias came back within one Monte Carlo standard error of zero on all three
    parameters (largest ``0.011``, se ``0.005``), coverage between ``0.93`` and ``0.97``,
    and ``se_ratio`` between ``0.90`` and ``1.18`` -- the spread of a ratio estimated from
    that many draws, which is why the band asserted below is the sibling class's and not a
    tighter one read off a single run.  To check it before the next nightly, dispatch that
    workflow with ``selection`` set to this class's node id.
    """

    COLUMNS: ClassVar[dict[str, Any]] = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
        "weights": "w",
    }

    ESTIMANDS: ClassVar[tuple[str, ...]] = (
        "ey_regimen[always]",
        "ey_regimen[never]",
        "ate_regimen[always vs never]",
    )

    def _run(self, *, n: int, reps: int = REPLICATES, weighted: bool = True) -> Any:
        columns = dict(self.COLUMNS)
        if not weighted:
            del columns["weights"]
        return CoverageStudy(
            dgp=make_longitudinal_weighted,
            estimator=lambda: LTMLE(
                {"always": 1, "never": 0},
                reference="never",
                outcome_learner=sklearn.linear_model.LinearRegression(),
                pseudo_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                simultaneous=False,
                random_state=0,
            ),
            n=n,
            n_replicates=reps,
            estimands=self.ESTIMANDS,
            fit_kwargs=columns,
            seed=2024,
            n_jobs=STUDY_JOBS,
        ).run()

    @pytest.fixture(scope="class")
    def study(self) -> Any:
        """One study, shared by the tests that read it -- ``n`` is before selection.

        About 60% of the units are retained, so this is a fit on roughly 1200 rows with a
        design effect near 1.4.
        """
        return self._run(n=2000)

    def test_the_intervals_cover_at_the_nominal_rate(self, study: Any) -> None:
        """The claim: ``(w / E[w]) D*(P_w)`` is the curve of the parameter being reported.

        A weighted fit has two ways to miss here and they are not the same failure.  If the
        *estimand* were wrong -- weights applied to the plug-in but not to the nuisances,
        say -- the intervals would sit off the truth and coverage would collapse.  If only
        the *curve* were wrong -- the centring left outside the weight, the normalisation
        forgotten -- they would be centred correctly and the wrong width.  Coverage catches
        both; the standard-error test below separates them.
        """
        for name in study.summaries:
            summary = study[name]
            assert summary.coverage > 0.90, summary
            assert abs(summary.coverage - 0.95) < 3.0 * summary.coverage_se + 0.02, summary

    def test_the_reported_standard_error_is_honest(self, study: Any) -> None:
        """The influence-curve variance against the actual spread, under the tilt.

        The Hajek centring ``w (f - psi)`` is what linearises the ratio the estimator is,
        and dropping it inflates the variance rather than the estimate -- so it shows up
        here and nowhere in the point estimates.
        """
        for name in study.summaries:
            assert 0.85 < study[name].se_ratio < 1.15, study[name]

    def test_ignoring_the_weights_misses_the_truth(self) -> None:
        """The negative control, and the reason the design is a biased sample.

        The same replications with the weight column left out estimate the *selected*
        population's parameter, so the coverage above is evidence that the weighting is
        doing the work rather than that this process is forgiving.

        **On a level, not on the contrast**, and that is the point worth recording.
        Selection on ``W1`` shifts both counterfactual means in the same direction, so most
        of the bias cancels in ``ate_regimen[always vs never]``: a 60-replicate run at
        ``n=2000`` put the unweighted bias at ``-0.050`` and ``-0.060`` on the two means --
        fourteen Monte Carlo standard errors each, with coverage collapsing to ``0.43`` and
        ``0.53`` -- while the contrast came back at ``+0.011`` with coverage ``0.92``.  A
        negative control taken on the contrast would therefore be nearly silent, and would
        read as though the weighting barely mattered.
        """
        ignored = self._run(n=2000, reps=100, weighted=False)
        for name in ("ey_regimen[always]", "ey_regimen[never]"):
            summary = ignored[name]
            assert abs(summary.bias) > 5.0 * summary.bias_se, summary
            assert summary.coverage < 0.8, summary


class TestCompetingRisksUnderRepeatedSampling:
    """The statistical tier for the cumulative incidence curves.

    ``tests/discrete_law_competing.py`` proves the influence curve *is* the efficient one,
    on a law the sample realises exactly.  That is a statement at one distribution with
    exact nuisances; this asks the different question the exact law cannot -- whether the
    interval built from that curve covers under repeated sampling, with estimated
    nuisances and a misspecified outcome regression.

    Both causes are checked at both horizons.  They are not interchangeable in either
    direction: the survival factor the recursion multiplies by is all-cause, so an error
    in it shows only from ``t = 2`` on, and the two causes have different shares and
    different contrast signs, so a fit that answered for the wrong one would look
    plausible at one and not the other.

    **This class has not run at these settings**, on the same footing as its survival
    sibling above and recorded the same way rather than implied to have passed. What *was* run is
    a reduced check at ``n=2500`` over 6 replicates in ``tests/e2e/test_ltmle.py``, which
    found the bias within Monte Carlo error at every cause and horizon -- that says the
    estimator is pointed at the right parameter and says nothing about coverage, since 6
    replicates cannot resolve a rate.  To check it before the next nightly, dispatch the
    workflow with ``selection`` set to this class's node id.
    """

    COLUMNS: ClassVar[dict[str, Any]] = {
        "outcome": {"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }

    ESTIMANDS: ClassVar[tuple[str, ...]] = (
        "cif_regimen[always, relapse @ t=1]",
        "cif_regimen[always, relapse @ t=2]",
        "cif_regimen[always, death @ t=1]",
        "cif_regimen[always, death @ t=2]",
        "cif_regimen[never, relapse @ t=2]",
        "cif_regimen[never, death @ t=2]",
        "ate_regimen[always vs never, relapse @ t=2]",
        "ate_regimen[always vs never, death @ t=2]",
    )

    def _run(self, *, n: int, reps: int = REPLICATES) -> Any:
        return CoverageStudy(
            dgp=make_longitudinal_competing,
            estimator=lambda: LTMLE(
                {"always": 1, "never": 0},
                reference="never",
                outcome_learner=sklearn.linear_model.LinearRegression(),
                pseudo_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                simultaneous=False,
                random_state=0,
            ),
            n=n,
            n_replicates=reps,
            estimands=self.ESTIMANDS,
            fit_kwargs=self.COLUMNS,
            seed=2026,
            n_jobs=STUDY_JOBS,
        ).run()

    @pytest.fixture(scope="class")
    def study(self) -> Any:
        """One ``n=2000`` study, shared by the tests that read it."""
        return self._run(n=2000)

    def test_the_intervals_cover_at_the_nominal_rate(self, study: Any) -> None:
        for name in study.summaries:
            summary = study[name]
            assert summary.coverage > 0.90, summary
            assert abs(summary.coverage - 0.95) < 3.0 * summary.coverage_se + 0.02, summary

    def test_the_reported_standard_error_is_honest(self, study: Any) -> None:
        """The influence-curve variance against the actual spread of the estimates."""
        for name in study.summaries:
            summary = study[name]
            assert summary.se_ratio == pytest.approx(1.0, abs=0.15), summary
