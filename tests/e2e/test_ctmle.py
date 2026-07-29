"""C-TMLE end to end: a real fit, and everything downstream of one.

The design commitment behind :class:`~cleverly.CTMLE` is that collaborative selection
changes *which propensity model is used* and nothing else -- the targeting step, the
influence curves, the sensitivity analyses and the validation suite are the same code
a plain fit runs.  This file is where that commitment is checked, by taking a C-TMLE
result and putting it through the whole public surface.

The statistical payoff -- smaller variance when an instrument is present -- needs
replications and lives in ``test_coverage_slow.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import CTMLE, TMLE
from cleverly.datasets import make_instrument
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
