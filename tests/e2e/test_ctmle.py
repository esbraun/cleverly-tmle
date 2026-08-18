"""C-TMLE end to end: a real fit, and everything downstream of one.

The design commitment behind :class:`~cleverly.CTMLE` is that collaborative selection
returns the complete selected ``(g_k, Qbar*_k)`` pair and then uses the ordinary pooled
targeting, influence-curve, sensitivity and validation layers.  This file checks that
commitment by taking a C-TMLE result through the whole public surface.

The statistical payoff -- smaller variance when an instrument is present -- needs
replications and lives in ``test_coverage_slow.py``.

:class:`TestSelectionIsForcedWhenTheOutcomeModelCannotHelp` is the exception, and the one
class here that is about the *selection* rather than about the plumbing around it.  It
exists because every other C-TMLE claim in the suite happens to be satisfiable by a
selector that always returns the empty propensity model; it takes that escape route away.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import sklearn.linear_model

from cleverly import SuperLearner
from cleverly.datasets import instrument_dgp, make_instrument, make_missing_outcome
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.targeting import build_submodel
from cleverly.inference.influence import counterfactual_means
from tests.conftest import FAST_KWARGS

TMLE_SETTINGS = {**FAST_KWARGS, "estimands": ("ate", "ey1", "ey0")}

#: Three selection folds rather than the default five keeps this file in the fast tier;
#: nothing asserted here turns on the fold count.
SETTINGS = {**TMLE_SETTINGS, "selection_folds": 3}


@pytest.fixture(scope="module")
def frame_and_truth() -> tuple[object, dict[str, float]]:
    return make_instrument(n=900, seed=5)


@pytest.fixture(scope="module")
def fit(frame_and_truth) -> object:
    frame, _ = frame_and_truth
    return CTMLE(**SETTINGS).fit(frame, outcome="Y", treatment="A").single()


class TestTheFit:
    def test_it_recovers_the_truth(self, fit, frame_and_truth) -> None:
        _, truth = frame_and_truth
        estimate = fit["ate"]
        assert abs(estimate.psi - truth["ate"]) < 3.0 * estimate.std_error

    def test_it_solves_the_score_equation(self, fit) -> None:
        check = fit.validation.score_check()
        assert check.passed, check.summary()

    def test_the_ate_influence_curve_is_the_difference_of_the_means(self, fit) -> None:
        # Exact identity, and it holds only if the selected propensity reached every
        # estimand through the same fluctuation.
        assert np.allclose(
            fit["ate"].influence_curve,
            fit["ey1"].influence_curve - fit["ey0"].influence_curve,
            atol=1e-12,
        )

    def test_the_selection_is_reported(self, fit) -> None:
        selection = fit.extra["ctmle"]
        assert selection.strategy == "greedy"
        assert selection.estimand == "ate"
        assert 0 <= selection.selected < len(selection.path)
        assert set(selection.selected_covariates) <= set(fit.data.covariate_names)

    def test_it_buys_a_smaller_standard_error_than_a_plain_fit(self, fit, frame_and_truth) -> None:
        # One sample, so this is a statement about this fit rather than about the
        # estimator -- but the mechanism is deterministic given the data: a narrower
        # propensity model means a smaller 1/g and a smaller influence curve.
        frame, _ = frame_and_truth
        plain = TMLE(**TMLE_SETTINGS).fit(frame, outcome="Y", treatment="A").single()
        assert fit["ate"].std_error < plain["ate"].std_error


class TestDownstreamMachineryStillWorks:
    def test_sensitivity_analyses_run_against_the_selected_model(self, fit) -> None:
        report = fit.sensitivity.positivity()
        assert report is not None
        curve = fit.sensitivity.truncation_curve()
        assert len(curve) > 0

    def test_the_nuisance_diagnostics_describe_the_selected_model(self, fit) -> None:
        diagnostics = fit.validation.nuisance()
        assert "propensity" in {model.name for model in diagnostics.models}
        assert diagnostics.summary()
        # Computed from the selected mechanism itself, not from a shared g(W) fit.
        report = diagnostics["propensity"]
        assert report.metrics["mean_predicted"] == pytest.approx(
            float(np.mean(fit.nuisance.propensity.arm(1.0)))
        )

    def test_and_report_no_learner_table_for_a_selected_mechanism(self, fit) -> None:
        """Empty on purpose, and the one thing an accepted regression would look like.

        A selector's ``g`` comes off the candidate path -- often the intercept-only
        candidate, which has no learner behind it at all -- so there is no super-learner
        weighting to report.  Before the shared pass went outcome-first this key held the
        ordinary ``g(W)`` table, describing a model the estimate never used.  ``oat`` does
        have one shared fit, and the assertion below keeps the two paths distinguishable.
        """
        report = fit.validation.nuisance()["propensity"]
        assert report.learner_weights == {}
        assert report.learner_risks == {}

    def test_but_oat_reports_the_table_from_its_one_shared_fit(self, frame_and_truth) -> None:
        frame, _ = frame_and_truth
        oat = (
            CTMLE(
                **{
                    **TMLE_SETTINGS,
                    "treatment_learner": SuperLearner(
                        [sklearn.linear_model.LogisticRegression(max_iter=1000)],
                        n_folds=3,
                    ),
                },
                strategy="oat",
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        report = oat.validation.nuisance()["propensity"]
        assert report.learner_weights and report.learner_risks

    def test_refutation_runs(self, fit) -> None:
        # A placebo refit goes back through CTMLE._nuisances, so the selection is
        # redone on the permuted data rather than reused -- which is the point.
        refutation = fit.validation.refute(tests=("placebo",), n_replicates=2, random_state=0)
        assert "placebo" in {test.name for test in refutation.tests}

    def test_the_summary_prints(self, fit) -> None:
        assert "Targeted maximum likelihood estimation" in fit.summary()

    def test_to_frame_returns_the_callers_backend(self, fit) -> None:
        frame = fit.to_frame()
        assert len(frame) == len(fit.estimates)

    @pytest.mark.parametrize("targeting", ["iterative", "one_step"])
    def test_every_truncation_retarget_solves_the_score(self, targeting: str) -> None:
        from sklearn.dummy import DummyRegressor

        frame, _ = instrument_dgp().sample(400, seed=22)
        result = (
            CTMLE(
                **{
                    **SETTINGS,
                    "outcome_learner": DummyRegressor(strategy="mean"),
                    "n_folds": 3,
                    "selection_folds": 2,
                    "targeting": targeting,
                }
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        for lower in (0.05, 0.2, 0.4):
            estimates, fluctuations = result.estimator.retarget(
                result.data,
                result.nuisance,
                estimands=("ate",),
                g_bounds=(lower, 1.0 - lower),
                g_bounds_conditional=(lower, 1.0 - lower),
            )
            assert fluctuations["mean"].score_norm < 1e-8
            assert abs(float(np.mean(estimates["ate"].influence_curve))) < 1e-8


class TestBackendParity:
    def test_pandas_and_polars_agree_bit_for_bit(self) -> None:
        pandas_frame, _ = make_instrument(n=500, seed=6, backend="pandas")
        polars_frame, _ = make_instrument(n=500, seed=6, backend="polars")
        columns = {"outcome": "Y", "treatment": "A"}
        from_pandas = CTMLE(**SETTINGS).fit(pandas_frame, **columns).single()
        from_polars = CTMLE(**SETTINGS).fit(polars_frame, **columns).single()

        assert from_pandas.psi("ate") == from_polars.psi("ate")
        assert (
            from_pandas.extra["ctmle"].selected_covariates
            == from_polars.extra["ctmle"].selected_covariates
        )


#: The option each test in :class:`TestCombinedWithOtherOptions` combines the selection
#: with.  Fitted once in the ``variants`` fixture.
COMBINATIONS: dict[str, dict[str, object]] = {
    "one_step": {"targeting": "one_step"},
    "weighted_form": {"target_weights": True},
    "linear": {"fluctuation": "linear"},
    "ordered": {"strategy": "ordered"},
}


class TestCombinedWithOtherOptions:
    @pytest.fixture(scope="class")
    def variants(self, frame_and_truth) -> dict[str, object]:
        frame, _ = frame_and_truth
        return {
            name: CTMLE(**{**SETTINGS, **overrides}).fit(frame, outcome="Y", treatment="A").single()
            for name, overrides in COMBINATIONS.items()
        }

    @pytest.mark.parametrize("variant", list(COMBINATIONS))
    def test_the_selection_composes_with_the_targeting_options(
        self, variants, variant: str
    ) -> None:
        result = variants[variant]
        assert result.validation.score_check().passed
        assert "ctmle" in result.extra

    def test_fold_targeted_composition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="published pooled collaborative estimator"):
            CTMLE(**{**SETTINGS, "targeting_scheme": "fold"})

    def test_missingness_is_refit_inside_selection_folds(self) -> None:
        frame, _ = make_missing_outcome(n=300, seed=17)
        result = (
            CTMLE(
                **{
                    **SETTINGS,
                    "missingness_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
                    "selection_folds": 2,
                    "n_folds": 3,
                }
            )
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        assert result.nuisance.missingness is not None
        assert result.validation.score_check().passed

    def test_the_bootstrap_repeats_the_selection(self, frame_and_truth) -> None:
        # The influence-curve standard error treats the selected propensity model as
        # given, so it cannot see the variability the selection itself contributes.
        # The bootstrap can, because each replicate re-runs the search -- which is why
        # _bootstrap_point_estimates goes through the selection hook.
        frame, _ = make_instrument(n=400, seed=7)
        result = (
            CTMLE(**{**SETTINGS, "n_bootstrap": 4}).fit(frame, outcome="Y", treatment="A").single()
        )
        assert result.bootstrap is not None
        assert result["ate"].bootstrap is not None
        assert result["ate"].bootstrap.std_error > 0.0


class TestSelectionIsForcedWhenTheOutcomeModelCannotHelp:
    """The test that a do-nothing selector fails, and the reason the others do not.

    Every C-TMLE claim elsewhere in the suite is made on a process whose outcome model is
    *correctly specified* -- :func:`~cleverly.datasets.instrument_dgp` has outcome mean
    ``1 + a + 1.5 W1 + 0.8 W3``, which a GLM fits exactly.  Under collaborative double
    robustness the confounding is then already handled before ``g`` is asked for anything,
    so an **empty** propensity model is genuinely the mean-squared-error-minimising choice,
    and C-TMLE duly selects one: the ordered search selects nothing on all five fixed
    ``n = 700`` seeds in the unit evidence tier.

    That is correct behaviour, and it is also why the variance and RMSE comparisons against
    plain TMLE prove less than they appear to.  A hypothetical selector hard-wired to
    return the empty model would pass all of them, because on such a process adjusting for
    nothing really is unbiased and minimum-variance.  Those comparisons establish that a
    propensity model containing an instrument costs variance -- a fact about plain TMLE --
    not that the collaborative search discriminates between covariates.

    This class removes the escape route.  The outcome learner is reduced to a constant, so
    every bit of confounding adjustment has to come through ``g``, and the empty model goes
    from optimal to badly biased.  A working search must now *include* the confounder, and
    the measured gap is not subtle: mean absolute error 0.017 for the collaborative fit
    against 0.696 for a selector restricted to the empty candidate, a factor of forty-one.
    """

    SEEDS = (0, 1, 2)
    N = 1500

    #: Settings whose only unusual feature is an outcome model that cannot fit anything.
    FORCED: ClassVar[dict[str, object]] = {
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "n_folds": 5,
        "learner_folds": 3,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
        "selection_folds": 3,
    }

    @pytest.fixture(scope="class")
    def fits(self) -> list[tuple[object, object, float]]:
        from sklearn.dummy import DummyRegressor

        dgp = instrument_dgp()
        truth = float(dgp.truth()["ate"])
        settings = {**self.FORCED, "outcome_learner": DummyRegressor(strategy="mean")}
        out = []
        for seed in self.SEEDS:
            frame, _ = dgp.sample(self.N, seed=seed)
            collaborative = CTMLE(**settings).fit(frame, outcome="Y", treatment="A").single()
            nothing = (
                CTMLE(**{**settings, "strategy": "discrete", "candidates": [()]})
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
            out.append((collaborative, nothing, truth))
        return out

    def test_the_outcome_model_really_is_useless(self, fits) -> None:
        """The premise, and worth asserting precisely rather than trusting the learner.

        If the outcome regression retained any predictive power the empty propensity model
        would stay defensible and this class would prove nothing.  Two things make it
        useless here: it ignores the treatment, so the two counterfactual predictions are
        *identical*; and it ignores the covariates, so the only variation left is the
        fold-to-fold difference between five training means -- about 0.002 on the ``[0, 1]``
        scale, against an outcome that spans it.
        """
        for collaborative, _, _ in fits:
            initial = collaborative.nuisance.outcome
            np.testing.assert_allclose(initial.arms[1.0], initial.arms[0.0], atol=1e-12, rtol=0)
            spread = float(np.std(initial.arms[1.0]))
            assert spread < 0.01, spread
            # And negligible beside the outcome's own spread, which is what "carries no
            # information about this row" means.
            scaled = collaborative.nuisance.scaler.scale(collaborative.data.outcome)
            assert spread < 0.05 * float(np.std(scaled))

    def test_it_selects_the_confounder(self, fits) -> None:
        # The claim no degenerate selector can satisfy: W1 drives both treatment and
        # outcome, so with a constant Qbar it is the covariate g cannot do without.
        for collaborative, _, _ in fits:
            selected = collaborative.extra["ctmle"].selected_covariates
            assert "W1" in selected, selected

    def test_it_never_selects_nothing(self, fits) -> None:
        for collaborative, _, _ in fits:
            assert collaborative.extra["ctmle"].selected_covariates != ()

    def test_a_multistep_selection_reports_the_selected_targeted_state(self, fits) -> None:
        collaborative = fits[0][0]
        selection = collaborative.extra["ctmle"]
        assert selection.n_steps[selection.selected] > 1
        targeted = collaborative.nuisance.targeting_outcome
        assert targeted is not None
        submodel = build_submodel(
            collaborative.data,
            collaborative.nuisance,
            "mean",
            bounds=collaborative.config.g_bounds,
            nuisance_bound=collaborative.config.missingness_bound,
            intermediate_value=None,
        )
        scaled = collaborative.nuisance.scaler.scale(collaborative.data.outcome)
        means = counterfactual_means(
            scaled,
            targeted,
            submodel,
            collaborative.data.weights,
            collaborative.data.observed,
        )
        expected = collaborative.nuisance.scaler.unscale_difference(means[1.0].psi - means[0.0].psi)
        assert collaborative.psi("ate") == pytest.approx(expected, abs=1e-10)

    def test_it_still_leaves_the_instrument_out(self, fits) -> None:
        # And this is now a real exclusion rather than a consequence of selecting nothing:
        # the selected sets here are non-empty by the test above.
        included = sum("W2" in c.extra["ctmle"].selected_covariates for c, _, _ in fits)
        assert included <= 1, "the instrument should be the covariate of last resort"

    def test_a_do_nothing_selector_would_be_badly_biased(self, fits) -> None:
        # The comparison that gives the three assertions above their force. Restricted to
        # the empty candidate the estimator is essentially unadjusted, and on a confounded
        # process that is a first-order error rather than a variance penalty.
        collaborative = np.array([abs(c.psi("ate") - t) for c, _, t in fits])
        nothing = np.array([abs(n.psi("ate") - t) for _, n, t in fits])
        assert nothing.mean() > 0.4, nothing
        assert collaborative.mean() < 0.15, collaborative
        assert nothing.mean() > 5.0 * collaborative.mean()

    def test_the_collaborative_fit_covers_the_truth_where_the_empty_one_cannot(self, fits) -> None:
        # Not a coverage study -- three fits -- but the failure here is systematic, not a
        # matter of luck: the empty model's interval is narrow and centred in the wrong
        # place, which is the signature of a bias a standard error cannot see.
        for collaborative, nothing, truth in fits:
            low, high = collaborative["ate"].ci
            assert low <= truth <= high, (low, high, truth)
            low, high = nothing["ate"].ci
            assert not (low <= truth <= high), (low, high, truth)
