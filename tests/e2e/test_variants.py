"""The estimator's configuration options, and that they agree with each other.

Every variant here targets the same estimand by a different route.  On the same data
they must land in the same place -- if ``targeting="one_step"`` and
``targeting="iterative"`` disagreed by more than numerical tolerance, at least one of
them would be wrong.  Mutual agreement across independent implementations is the
strongest evidence available without an external reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.datasets import make_linear_ate, make_nonlinear_ate, make_weak_overlap
from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from tests.conftest import fast_tmle


@pytest.fixture(scope="module")
def frame_and_truth() -> tuple[object, dict[str, float]]:
    return make_nonlinear_ate(n=1500, seed=31)


def _fit(frame: object, **overrides: object) -> object:
    return (
        fast_tmle(estimands=("ate", "att"), **overrides)
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


#: The configurations every test in :class:`TestVariantsAgree` draws from.  Each is fitted
#: once, in the ``variants`` fixture, and the comparisons then read two entries out of it
#: rather than refitting -- the same six fits used to be repeated nine more times.
VARIANTS: dict[str, dict[str, object]] = {
    "iterative": {},
    "one_step": {"targeting": "one_step"},
    "cv_tmle": {"targeting_scheme": "fold"},
    "weighted_form": {"target_weights": True},
    "linear": {"fluctuation": "linear"},
    "no_crossfit": {"cross_fit": False},
}


class TestVariantsAgree:
    @pytest.fixture(scope="class")
    def variants(self, frame_and_truth) -> dict[str, object]:
        frame, _ = frame_and_truth
        return {name: _fit(frame, **overrides) for name, overrides in VARIANTS.items()}

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_every_variant_solves_the_score_equation(self, variants, variant: str) -> None:
        assert variants[variant].validation.score_check().passed

    def test_targeting_methods_agree(self, variants) -> None:
        iterative = variants["iterative"]
        one_step = variants["one_step"]
        # The universal least-favorable submodel walks to the same root the Newton
        # solver jumps to.
        assert one_step.psi("ate") == pytest.approx(iterative.psi("ate"), abs=2e-3)
        assert one_step.psi("att") == pytest.approx(iterative.psi("att"), abs=2e-3)

    def test_the_weighted_form_agrees_with_the_covariate_form(self, variants) -> None:
        plain = variants["iterative"]
        weighted = variants["weighted_form"]
        # Different submodels solving the same estimating equation, so the estimates
        # differ only at second order.
        assert weighted.psi("ate") == pytest.approx(
            plain.psi("ate"), abs=3.0 * plain["ate"].std_error
        )

    def test_cv_tmle_agrees_with_pooled_targeting(self, variants) -> None:
        pooled = variants["iterative"]
        fold_wise = variants["cv_tmle"]
        assert fold_wise.psi("ate") == pytest.approx(
            pooled.psi("ate"), abs=3.0 * pooled["ate"].std_error
        )

    def test_fold_wise_targeting_still_solves_the_pooled_score(self, variants) -> None:
        # Each fold's score is zero on its own rows, so the sum over folds -- the
        # full-sample score -- is zero too.
        for fluctuation in variants["cv_tmle"].fluctuations.values():
            assert fluctuation.score_norm < 1e-10

    def test_the_linear_fluctuation_agrees_with_the_logistic_one(self, variants) -> None:
        logistic = variants["iterative"]
        linear = variants["linear"]
        assert linear.psi("ate") == pytest.approx(
            logistic.psi("ate"), abs=3.0 * logistic["ate"].std_error
        )


class TestReproducibility:
    def test_the_same_seed_gives_the_same_answer(self, frame_and_truth) -> None:
        frame, _ = frame_and_truth
        first = _fit(frame)
        second = _fit(frame)
        assert first.psi("ate") == second.psi("ate")
        assert first["ate"].std_error == second["ate"].std_error

    def test_different_seeds_move_the_answer_only_a_little(self, frame_and_truth) -> None:
        frame, _ = frame_and_truth
        first = _fit(frame, random_state=1)
        second = _fit(frame, random_state=2)
        # Fold assignment changes, so the estimate moves -- but far less than a standard
        # error, or the cross-fitting would be adding meaningful noise of its own.
        assert first.psi("ate") != second.psi("ate")
        assert abs(first.psi("ate") - second.psi("ate")) < first["ate"].std_error

    def test_an_estimator_can_be_reused(self, frame_and_truth) -> None:
        frame, _ = frame_and_truth
        estimator = fast_tmle(estimands=("ate",))
        first = estimator.fit(frame, outcome="Y", treatment="A").single()
        second = estimator.fit(frame, outcome="Y", treatment="A").single()
        assert first.psi("ate") == second.psi("ate")


class TestBoundsAndScaling:
    def test_tighter_truncation_reduces_the_standard_error(self) -> None:
        frame, _ = make_weak_overlap(n=1200, seed=32)
        with pytest.warns(PositivityWarning):
            loose = (
                fast_tmle(estimands=("ate",), g_bounds=0.001)
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
        with pytest.warns(PositivityWarning):
            # A bound of 0.1 truncates most of this sample, which is exactly the
            # situation the warning exists to flag.
            tight = (
                fast_tmle(estimands=("ate",), g_bounds=0.1)
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
        # Truncation trades bias for variance; the variance side must be visible.
        assert tight["ate"].std_error < loose["ate"].std_error

    def test_explicit_outcome_bounds_are_recorded_and_respected(self) -> None:
        frame, truth = make_linear_ate(n=800, seed=33)
        y = frame["Y"].to_numpy()
        bounds = (float(y.min()) - 1.0, float(y.max()) + 1.0)
        result = (
            fast_tmle(estimands=("ate",), q_bounds=bounds)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert result.config.q_bounds == bounds
        low, high = result["ate"].ci
        assert low <= truth["ate"] <= high

    def test_outcome_bounds_that_exclude_the_data_are_refused(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=34)
        with pytest.raises(ValueError, match="outside q_bounds"):
            fast_tmle(q_bounds=(0.0, 0.1)).fit(frame, outcome="Y", treatment="A").single()

    def test_counterfactual_means_stay_inside_the_outcome_bounds(self) -> None:
        frame, _ = make_linear_ate(n=800, seed=35)
        result = fast_tmle(estimands=("ey1", "ey0")).fit(frame, outcome="Y", treatment="A").single()
        lower, upper = result.config.q_bounds
        # The logistic fluctuation is bounded by construction, so this cannot fail
        # unless the unscaling is wrong.
        for name in ("ey1", "ey0"):
            assert lower <= result.psi(name) <= upper

    def test_auto_bounds_differ_between_the_ate_and_the_att(self) -> None:
        frame, _ = make_linear_ate(n=1000, seed=36)
        result = fast_tmle(estimands=("ate", "att")).fit(frame, outcome="Y", treatment="A").single()
        assert result.config.g_bounds != result.config.g_bounds_conditional
        assert result.config.g_bounds_conditional[0] == pytest.approx(0.025)


class TestWeightsAndClusters:
    def test_observation_weights_change_the_estimate(self) -> None:
        frame, _ = make_nonlinear_ate(n=1200, seed=37)
        rng = np.random.default_rng(0)
        weighted_frame = frame.assign(w=rng.uniform(0.3, 2.0, len(frame)))
        plain = (
            fast_tmle(estimands=("ate",))
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3", "W4"])
            .single()
        )
        weighted = (
            fast_tmle(estimands=("ate",))
            .fit(
                weighted_frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3", "W4"],
                weights="w",
            )
            .single()
        )
        assert weighted.psi("ate") != plain.psi("ate")
        assert weighted.validation.score_check().passed

    def test_uniform_weights_reproduce_the_unweighted_fit(self) -> None:
        frame, _ = make_linear_ate(n=800, seed=38)
        plain = (
            fast_tmle(estimands=("ate",))
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3", "W4"])
            .single()
        )
        weighted = (
            fast_tmle(estimands=("ate",))
            .fit(
                frame.assign(w=2.5),
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3", "W4"],
                weights="w",
            )
            .single()
        )
        # Weights are normalised to mean one, so a constant weight is a no-op.
        assert weighted.psi("ate") == pytest.approx(plain.psi("ate"), rel=1e-9)

    def test_sampling_weights_move_the_estimate_to_the_population(self) -> None:
        """The survey case, on one sample: which parameter does each fit land on?

        Selection depends on ``W1`` and the treatment effect varies in ``W1``, so the
        population ATE and the ATE among the selected are different numbers -- about 1.0
        and 1.5.  With ``w = 1 / P(S = 1 | W1)`` the tilted law is the population law, so
        the weighted fit targets the first and the unweighted fit the second.  This is a
        bias-direction check on a single fit, not a coverage claim; the coverage claim is
        in ``tests/e2e/test_coverage_slow.py``, where it can be averaged over replications.
        """
        from cleverly.datasets import make_biased_sample

        frame, truth = make_biased_sample(6000, seed=40)
        columns = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
        weighted = (
            fast_tmle(estimands=("ate",)).fit(frame, weights="sampling_weight", **columns).single()
        )
        unweighted = fast_tmle(estimands=("ate",)).fit(frame, **columns).single()

        assert truth["ate_selected"] - truth["ate"] > 0.3
        assert weighted.psi("ate") == pytest.approx(
            truth["ate"], abs=3.0 * weighted["ate"].std_error
        )
        assert unweighted.psi("ate") == pytest.approx(
            truth["ate_selected"], abs=3.0 * unweighted["ate"].std_error
        )
        assert weighted.data.weight_report().effective_n < weighted.n

    def test_clustering_inflates_the_standard_error(self) -> None:
        from cleverly.datasets import make_clustered

        frame, _ = make_clustered(n=1500, seed=39, cluster_size=15)
        columns = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
        ignoring = fast_tmle(estimands=("ate",)).fit(frame, **columns).single()
        clustered = fast_tmle(estimands=("ate",)).fit(frame, id="cluster", **columns).single()
        # The DGP shares an unobserved latent within clusters, so ignoring the structure
        # understates the uncertainty.
        assert clustered["ate"].std_error > 1.2 * ignoring["ate"].std_error
        # The point estimates are close but not identical: passing id= also keeps
        # clusters intact when building the cross-fitting folds, which changes the
        # out-of-fold nuisance predictions slightly.
        #
        # How *much* they differ is a property of the fold split, and the fold split is a
        # property of the installed scikit-learn rather than of this package -- so the
        # number here has to leave room for a resolver that picks a different one. It did:
        # the gap is 0.09 standard errors on Python 3.11 and 0.29 on 3.10, deterministically
        # and identically across runs, which the old `0.2` sat exactly between. That `0.2`
        # was never derived; it was whatever one interpreter happened to produce, and it
        # spent the outage failing on a quarter of the matrix with nothing to say so.
        # `0.5` still fails on the thing this line is for -- clustering moving the estimate
        # rather than the interval -- which would be a shift of several standard errors.
        assert clustered.psi("ate") == pytest.approx(
            ignoring.psi("ate"), abs=0.5 * ignoring["ate"].std_error
        )
        assert clustered.data.n_clusters == 100


class TestBinaryOutcome:
    def test_ratio_estimands_are_reported_for_a_binary_outcome(self) -> None:
        from cleverly.datasets import make_binary_outcome

        frame, truth = make_binary_outcome(n=2500, seed=40)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A").single()
        assert {"rr", "or"} <= set(result.estimates)
        for name in ("rr", "or"):
            low, high = result[name].ci
            assert low > 0
            assert low <= truth[name] <= high

    def test_ratios_are_refused_for_a_continuous_outcome(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=41)
        with pytest.raises(ValueError, match="require a binary outcome"):
            fast_tmle(estimands=("rr",)).fit(frame, outcome="Y", treatment="A").single()

    def test_probabilities_stay_in_range(self) -> None:
        from cleverly.datasets import make_binary_outcome

        frame, _ = make_binary_outcome(n=1200, seed=42)
        result = fast_tmle(estimands=("ey1", "ey0")).fit(frame, outcome="Y", treatment="A").single()
        for name in ("ey1", "ey0"):
            assert 0.0 <= result.psi(name) <= 1.0


class TestBootstrapAndBands:
    def test_the_bootstrap_standard_error_is_comparable_to_the_influence_curve_one(self) -> None:
        frame, _ = make_linear_ate(n=500, seed=43)
        # The bootstrap refits the model per replicate, so this is the most expensive
        # test in the fast tier; 40 replicates is the minimum that makes the
        # standard-error comparison below meaningful.
        result = (
            fast_tmle(estimands=("ate",), n_bootstrap=40)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        bootstrap = result["ate"].bootstrap
        assert bootstrap is not None
        assert bootstrap.n_replicates >= 35
        # Two routes to the same variance; agreement within a factor of two is the
        # honest expectation at this sample size and replicate count.
        ratio = bootstrap.std_error / result["ate"].std_error
        assert 0.5 < ratio < 2.0
        assert bootstrap.ci[0] < result.psi("ate") < bootstrap.ci[1]

    def test_simultaneous_bands_contain_the_pointwise_intervals(self) -> None:
        frame, _ = make_linear_ate(n=800, seed=44)
        result = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=4,
                learner_folds=3,
                random_state=0,
                estimands=("ate", "att", "atc", "ey1", "ey0"),
                simultaneous=True,
                n_multiplier=400,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert result.simultaneous is not None
        assert result.simultaneous.critical_value > result.simultaneous.pointwise_critical_value
        for name, estimate in result.estimates.items():
            band = result.simultaneous.bands[name]
            point = estimate.ci
            assert band[0] <= point[0]
            assert band[1] >= point[1]

    def test_a_cluster_bootstrap_resamples_clusters(self) -> None:
        from cleverly.datasets import make_clustered

        frame, _ = make_clustered(n=900, seed=45, cluster_size=15)
        result = (
            fast_tmle(estimands=("ate",), n_bootstrap=20)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                id="cluster",
            )
            .single()
        )
        assert result.bootstrap is not None
        assert result.bootstrap.resampling == "cluster"


class TestConfigurationErrors:
    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"fluctuation": "cubic"}, "fluctuation must be"),
            ({"targeting": "magic"}, "targeting must be"),
            ({"targeting_scheme": "wrong"}, "targeting_scheme must be"),
            ({"targeting": "one_step", "fluctuation": "linear"}, "cannot be combined"),
            ({"alpha_sig": 1.5}, "alpha_sig must lie"),
            ({"nuisance_bound": 0.9}, "nuisance_bound must lie"),
            ({"n_bootstrap": 1}, "n_bootstrap must be"),
        ],
    )
    def test_bad_settings_are_refused_at_construction(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TMLE(**kwargs)

    def test_missing_column_names_are_refused(self) -> None:
        frame, _ = make_linear_ate(n=200, seed=46)
        with pytest.raises(ValueError, match="outcome= and treatment= are required"):
            fast_tmle().fit(frame).single()

    def test_a_numpy_array_is_refused_with_guidance(self) -> None:
        with pytest.raises(TypeError, match=r"cleverly\.tmle"):
            fast_tmle().fit(np.zeros((50, 4)), outcome="Y", treatment="A").single()

    def test_column_names_cannot_be_mixed_with_causal_data(self) -> None:
        from cleverly.data import CausalData

        frame, _ = make_linear_ate(n=200, seed=47)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        with pytest.raises(ValueError, match="cannot be combined"):
            fast_tmle().fit(data, outcome="Y").single()

    def test_an_unknown_estimand_is_refused(self) -> None:
        frame, _ = make_linear_ate(n=200, seed=48)
        with pytest.raises(ValueError, match="unknown estimand"):
            fast_tmle(estimands=("nope",)).fit(frame, outcome="Y", treatment="A").single()

    def test_q_bounds_are_refused_for_a_binary_outcome(self) -> None:
        from cleverly.datasets import make_binary_outcome

        frame, _ = make_binary_outcome(n=300, seed=49)
        with pytest.raises(ValueError, match="q_bounds does not apply"):
            fast_tmle(q_bounds=(0.0, 1.0)).fit(frame, outcome="Y", treatment="A").single()


class TestArrayEntryPoint:
    def test_the_r_style_function_recovers_the_truth(self) -> None:
        from cleverly.estimators import tmle

        frame, truth = make_linear_ate(n=1000, seed=50)
        result = tmle(
            frame["Y"].to_numpy(),
            frame["A"].to_numpy(),
            frame[["W1", "W2", "W3", "W4"]].to_numpy(),
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=4,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
        ).single()
        low, high = result["ate"].ci
        assert low <= truth["ate"] <= high

    def test_r_style_keyword_names_are_accepted(self) -> None:
        from cleverly.datasets import make_missing_outcome
        from cleverly.estimators import tmle

        frame, truth = make_missing_outcome(n=1000, seed=51)
        result = tmle(
            frame["Y"].to_numpy(),
            frame["A"].to_numpy(),
            frame[["W1", "W2", "W3"]].to_numpy(),
            Delta=frame["Delta"].to_numpy(),
            obsWeights=np.ones(len(frame)),
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=4,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
        ).single()
        low, high = result["ate"].ci
        assert low <= truth["ate"] <= high


class TestScreening:
    def test_screening_reports_the_retained_covariates(self) -> None:
        frame, truth = make_linear_ate(n=1200, seed=52)
        noisy = frame.assign(
            **{f"noise{i}": np.random.default_rng(i).normal(size=len(frame)) for i in range(5)}
        )
        result = (
            fast_tmle(estimands=("ate",), screen_treatment=True, min_retain=2)
            .fit(noisy, outcome="Y", treatment="A")
            .single()
        )
        retained = result.nuisance.treatment_covariates
        assert 0 < len(retained) < noisy.shape[1] - 2
        low, high = result["ate"].ci
        assert low <= truth["ate"] <= high
