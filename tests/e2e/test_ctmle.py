"""C-TMLE end to end: a real fit, and everything downstream of one.

The design commitment behind :class:`~cleverly.CTMLE` is that collaborative selection
changes *which propensity model is used* and nothing else -- the targeting step, the
influence curves, the sensitivity analyses and the validation suite are the same code
a plain fit runs.  This file is where that commitment is checked, by taking a C-TMLE
result and putting it through the whole public surface.

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

from cleverly import CTMLE, TMLE
from cleverly.datasets import instrument_dgp, make_instrument
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
    return CTMLE(**SETTINGS).fit(frame, outcome="Y", treatment="A")


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
        assert selection.search == "greedy"
        assert selection.estimand == "ate"
        assert 0 <= selection.selected < len(selection.path)
        assert set(selection.selected_covariates) <= set(fit.data.covariate_names)

    def test_it_buys_a_smaller_standard_error_than_a_plain_fit(self, fit, frame_and_truth) -> None:
        # One sample, so this is a statement about this fit rather than about the
        # estimator -- but the mechanism is deterministic given the data: a narrower
        # propensity model means a smaller 1/g and a smaller influence curve.
        frame, _ = frame_and_truth
        plain = TMLE(**TMLE_SETTINGS).fit(frame, outcome="Y", treatment="A")
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


class TestBackendParity:
    def test_pandas_and_polars_agree_bit_for_bit(self) -> None:
        pandas_frame, _ = make_instrument(n=500, seed=6, backend="pandas")
        polars_frame, _ = make_instrument(n=500, seed=6, backend="polars")
        columns = {"outcome": "Y", "treatment": "A"}
        from_pandas = CTMLE(**SETTINGS).fit(pandas_frame, **columns)
        from_polars = CTMLE(**SETTINGS).fit(polars_frame, **columns)

        assert from_pandas.psi("ate") == from_polars.psi("ate")
        assert (
            from_pandas.extra["ctmle"].selected_covariates
            == from_polars.extra["ctmle"].selected_covariates
        )


class TestCombinedWithOtherOptions:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"targeting": "one_step"},
            {"targeting_scheme": "fold"},
            {"target_weights": True},
            {"fluctuation": "linear"},
            {"search": "ordered"},
        ],
        ids=["one_step", "cv_tmle", "weighted_form", "linear", "ordered"],
    )
    def test_the_selection_composes_with_the_targeting_options(
        self, frame_and_truth, overrides: dict[str, object]
    ) -> None:
        frame, _ = frame_and_truth
        result = CTMLE(**{**SETTINGS, **overrides}).fit(frame, outcome="Y", treatment="A")
        assert result.validation.score_check().passed
        assert "ctmle" in result.extra

    def test_a_cross_validated_ctmle_reports_both_diagnostics(self, frame_and_truth) -> None:
        frame, _ = frame_and_truth
        result = CTMLE(**{**SETTINGS, "targeting_scheme": "fold"}).fit(
            frame, outcome="Y", treatment="A"
        )
        assert result.extra["ctmle"] is not None
        assert result.cv_targeting is not None

    def test_the_bootstrap_repeats_the_selection(self, frame_and_truth) -> None:
        # The influence-curve standard error treats the selected propensity model as
        # given, so it cannot see the variability the selection itself contributes.
        # The bootstrap can, because each replicate re-runs the search -- which is why
        # _bootstrap_point_estimates goes through the selection hook.
        frame, _ = make_instrument(n=400, seed=7)
        result = CTMLE(**{**SETTINGS, "n_bootstrap": 4}).fit(frame, outcome="Y", treatment="A")
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
    and C-TMLE duly selects one: measured over ten seeds at ``n = 700``, the greedy search
    selects nothing 10 times out of 10 and the ordered search 7 times out of 10.

    That is correct behaviour, and it is also why the variance and RMSE comparisons against
    plain TMLE prove less than they appear to.  A hypothetical selector hard-wired to
    return the empty model would pass all of them, because on such a process adjusting for
    nothing really is unbiased and minimum-variance.  Those comparisons establish that a
    propensity model containing an instrument costs variance -- a fact about plain TMLE --
    not that the collaborative search discriminates between covariates.

    This class removes the escape route.  The outcome learner is reduced to a constant, so
    every bit of confounding adjustment has to come through ``g``, and the empty model goes
    from optimal to badly biased.  A working search must now *include* the confounder, and
    the measured gap is not subtle: bias 0.037 for the collaborative fit against 0.810 for
    a selector restricted to the empty candidate, a factor of twenty-two.
    """

    SEEDS = (0, 1, 2)
    N = 1500

    #: Settings whose only unusual feature is an outcome model that cannot fit anything.
    FORCED: ClassVar[dict[str, object]] = {
        "treatment_learner": "glm",
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
            collaborative = CTMLE(**settings).fit(frame, outcome="Y", treatment="A")
            nothing = CTMLE(**{**settings, "search": "discrete", "candidates": [()]}).fit(
                frame, outcome="Y", treatment="A"
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
            np.testing.assert_allclose(initial.at_one, initial.at_zero, atol=1e-12, rtol=0)
            spread = float(np.std(initial.at_one))
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
