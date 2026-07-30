"""The synthetic processes and their reference values.

Anything that validates the estimator is only as good as the truth it compares
against, so the truth itself is tested: the quasi-Monte Carlo integration must be
deterministic, must agree with a large plain Monte Carlo draw, and must reproduce the
estimands that are known in closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleverly.datasets import (
    GENERATORS,
    available,
    binary_outcome_dgp,
    cde_dgp,
    clustered_dgp,
    heterogeneous_dgp,
    instrument_dgp,
    linear_dgp,
    make_binary_outcome,
    make_cde,
    make_clustered,
    make_instrument,
    make_linear_ate,
    make_missing_outcome,
    make_missing_outcome_binary,
    make_nonlinear_ate,
    make_weak_overlap,
    missing_outcome_dgp,
    nonlinear_dgp,
    weak_overlap_dgp,
)


class TestTruth:
    def test_a_constant_effect_makes_every_contrast_equal(self) -> None:
        truth = linear_dgp(effect=1.5).truth()
        # With a homogeneous effect the ATE, ATT and ATC coincide exactly, so this is a
        # closed-form check on the integration, not an approximation.
        assert truth["ate"] == pytest.approx(1.5, abs=1e-4)
        assert truth["att"] == pytest.approx(1.5, abs=1e-4)
        assert truth["atc"] == pytest.approx(1.5, abs=1e-4)

    def test_counterfactual_means_are_known_in_closed_form(self) -> None:
        # E[Y(a)] = 2 + 1.5a + E[linear in W] = 2 + 1.5a, since E[W] = 0.
        truth = linear_dgp(effect=1.5).truth()
        assert truth["ey0"] == pytest.approx(2.0, abs=1e-3)
        assert truth["ey1"] == pytest.approx(3.5, abs=1e-3)
        assert truth["ate"] == pytest.approx(truth["ey1"] - truth["ey0"], abs=1e-9)

    def test_heterogeneous_effects_separate_the_contrasts(self) -> None:
        truth = nonlinear_dgp().truth()
        # Treatment depends on W and the effect depends on W, so the treated and control
        # subpopulations must have different average effects.
        assert truth["att"] != pytest.approx(truth["atc"], abs=1e-3)
        assert min(truth["att"], truth["atc"]) < truth["ate"] < max(truth["att"], truth["atc"])

    def test_the_heterogeneous_process_orders_the_contrasts_in_a_known_direction(self) -> None:
        """``att > ate > atc``, in that order, by a margin no sampling error can cross.

        The check above says the three contrasts differ; it does not say which is which,
        so it passes for an estimator that conditions on the wrong arm.  Here the
        conditional effect and the propensity both increase in ``W1``, so the treated are
        drawn from the covariate values where the effect is larger and the ordering is
        fixed in advance by :math:`\\operatorname{Cov}(\\tau, g) > 0`.
        """
        truth = heterogeneous_dgp().truth()
        assert truth["att"] > truth["ate"] > truth["atc"]
        assert min(truth["att"] - truth["ate"], truth["ate"] - truth["atc"]) > 0.5
        # E[tau(W)] = 1 + slope * E[W1] = 1, in closed form.
        assert truth["ate"] == pytest.approx(1.0, abs=1e-3)

    def test_the_heterogeneous_ordering_is_produced_by_the_covariance(self) -> None:
        # Not a coincidence of the coefficients: recompute ATT and ATC from the defining
        # ratios by plain Monte Carlo and confirm they bracket the ATE the same way.
        dgp = heterogeneous_dgp()
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(400_000, dgp.n_latent))
        g = dgp.propensity(latent)
        tau = dgp.outcome_mean(latent, 1.0, None) - dgp.outcome_mean(latent, 0.0, None)
        assert np.cov(tau, g)[0, 1] > 0.1
        truth = dgp.truth()
        assert truth["att"] == pytest.approx(float(np.average(tau, weights=g)), abs=0.01)
        assert truth["atc"] == pytest.approx(float(np.average(tau, weights=1.0 - g)), abs=0.01)

    def test_quasi_monte_carlo_agrees_with_plain_monte_carlo(self) -> None:
        dgp = nonlinear_dgp()
        truth = dgp.truth()
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(400_000, dgp.n_latent))
        reference = dgp.sample_truth(latent)
        for key in ("ate", "att", "atc", "ey1", "ey0"):
            assert truth[key] == pytest.approx(reference[key], abs=0.01)

    def test_truth_is_deterministic(self) -> None:
        assert nonlinear_dgp().truth() == nonlinear_dgp().truth()

    def test_binary_outcome_truth_includes_marginal_ratios(self) -> None:
        truth = binary_outcome_dgp().truth()
        assert 0.0 < truth["ey0"] < truth["ey1"] < 1.0
        assert truth["rr"] == pytest.approx(truth["ey1"] / truth["ey0"])
        odds_one = truth["ey1"] / (1 - truth["ey1"])
        odds_zero = truth["ey0"] / (1 - truth["ey0"])
        assert truth["or"] == pytest.approx(odds_one / odds_zero)
        # The marginal odds ratio is attenuated relative to the conditional one (0.9 on
        # the log-odds scale in this process) -- a distinct estimand, not an error.
        assert truth["or"] < np.exp(0.9)

    def test_controlled_direct_effects_differ_across_the_intermediate(self) -> None:
        dgp = cde_dgp()
        # The process has an A-by-Z interaction of 0.6, so the CDE at z=1 exceeds the
        # CDE at z=0 by exactly that amount.
        at_zero = dgp.truth(0.0)["ate"]
        at_one = dgp.truth(1.0)["ate"]
        assert at_zero == pytest.approx(0.9, abs=1e-6)
        assert at_one == pytest.approx(1.5, abs=1e-6)
        assert at_one - at_zero == pytest.approx(0.6, abs=1e-9)

    def test_weak_overlap_pushes_propensities_into_the_tails(self) -> None:
        frame, _ = make_weak_overlap(n=4000, seed=0, strength=3.0)
        from cleverly.utils.bounds import expit

        w = frame[["W1", "W2"]].to_numpy()
        g = expit(3.0 * w[:, 0] + 2.1 * w[:, 1])
        assert np.mean((g < 0.01) | (g > 0.99)) > 0.2

    def test_the_instrument_process_separates_the_three_covariate_roles(self) -> None:
        # The process is only useful for demonstrating instrument inflation if the
        # roles really are disjoint, so check the structural functions directly rather
        # than trusting the docstring.
        dgp = instrument_dgp()
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(2000, 3))

        def vary(column: int, fn) -> bool:
            moved = latent.copy()
            moved[:, column] += 1.0
            return not np.allclose(fn(latent), fn(moved))

        outcome = lambda w: dgp.outcome_mean(w, 1.0, None)  # noqa: E731
        # W1 confounds: it moves both. W2 is an instrument: treatment only. W3 is a
        # pure outcome predictor.
        assert vary(0, dgp.propensity) and vary(0, outcome)
        assert vary(1, dgp.propensity) and not vary(1, outcome)
        assert not vary(2, dgp.propensity) and vary(2, outcome)

    def test_the_instrument_effect_is_constant(self) -> None:
        truth = instrument_dgp().truth()
        assert truth["ate"] == pytest.approx(1.0, abs=1e-6)
        assert truth["att"] == pytest.approx(truth["ate"], abs=1e-6)
        assert truth["atc"] == pytest.approx(truth["ate"], abs=1e-6)

    def test_a_stronger_instrument_squeezes_the_propensity_harder(self) -> None:
        # The instrument is what causes the positivity strain, so the knob has to bite.
        rng = np.random.default_rng(1)
        latent = rng.normal(size=(20_000, 3))
        mild = instrument_dgp(instrument_strength=0.5).propensity(latent)
        severe = instrument_dgp(instrument_strength=3.0).propensity(latent)
        assert np.mean(mild < 0.05) < np.mean(severe < 0.05)

    def test_the_instrument_frame_carries_all_three_covariates(self) -> None:
        frame, truth = make_instrument(n=300, seed=0)
        assert {"Y", "A", "W1", "W2", "W3"} <= set(frame.columns)
        assert truth["ate"] == pytest.approx(1.0, abs=1e-6)

    def test_a_stronger_overlap_violation_is_worse(self) -> None:
        mild = weak_overlap_dgp(strength=1.0)
        severe = weak_overlap_dgp(strength=4.0)
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(20_000, 3))
        assert np.mean(mild.propensity(latent) < 0.05) < np.mean(severe.propensity(latent) < 0.05)


class TestSampling:
    @pytest.mark.parametrize("name", sorted(GENERATORS))
    def test_every_generator_produces_a_usable_frame(self, name: str) -> None:
        frame, truth = GENERATORS[name](n=300, seed=0)
        assert len(frame) == 300
        assert {"Y", "A"} <= set(frame.columns)
        assert "ate" in truth
        assert "sample_ate" in truth

    @pytest.mark.parametrize("name", sorted(GENERATORS))
    def test_sampling_is_reproducible(self, name: str) -> None:
        first, _ = GENERATORS[name](n=200, seed=42)
        second, _ = GENERATORS[name](n=200, seed=42)
        assert first.equals(second)

    @pytest.mark.parametrize("name", sorted(GENERATORS))
    def test_different_seeds_give_different_data(self, name: str) -> None:
        first, _ = GENERATORS[name](n=200, seed=1)
        second, _ = GENERATORS[name](n=200, seed=2)
        assert not first.equals(second)

    @pytest.mark.parametrize(
        "backend,expected", [("pandas", pd.DataFrame), ("polars", pl.DataFrame)]
    )
    def test_the_requested_backend_is_produced(self, backend: str, expected: type) -> None:
        frame, _ = make_linear_ate(n=100, seed=0, backend=backend)
        assert isinstance(frame, expected)

    def test_the_sample_estimand_is_close_to_the_population_one(self) -> None:
        _, truth = make_nonlinear_ate(n=20_000, seed=0)
        assert truth["sample_ate"] == pytest.approx(truth["ate"], abs=0.05)

    def test_missing_outcomes_carry_a_delta_column(self) -> None:
        frame, _ = make_missing_outcome(n=500, seed=0)
        assert "Delta" in frame.columns
        observed = frame["Delta"].to_numpy()
        assert 0.0 < observed.mean() < 1.0
        # Y is NaN exactly where Delta is 0.
        assert np.array_equal(np.isnan(frame["Y"].to_numpy()), observed == 0.0)

    def test_strength_one_reproduces_the_original_process(self) -> None:
        # The `strength` parameter was added later; existing truths, seeds and any
        # threshold tuned against this process must be unaffected at the default.
        _, truth = make_missing_outcome(n=800, seed=17)
        assert truth["ate"] == pytest.approx(1.2, abs=1e-4)
        assert truth["att"] == pytest.approx(1.2, abs=1e-4)
        from cleverly.utils.bounds import expit

        dgp = missing_outcome_dgp()
        assert dgp.name == "missing_outcome"
        w = np.random.default_rng(0).normal(size=(200, 3))
        np.testing.assert_allclose(
            dgp.outcome_mean(w, 1.0, None),
            1.0 + 1.2 + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2],
        )
        np.testing.assert_allclose(
            dgp.missingness(w, 1.0), expit(1.8 - 0.8 * w[:, 0] + 0.3 * w[:, 2])
        )

    def test_raising_the_strength_sharpens_the_mechanism_and_bends_the_outcome(self) -> None:
        w = np.random.default_rng(1).normal(size=(20_000, 3))
        mild, sharp = missing_outcome_dgp(), missing_outcome_dgp(strength=2.0)
        # The mechanism reaches further towards zero, which is what puts the estimate on
        # the inverse-probability half of double robustness.
        assert np.quantile(sharp.missingness(w, 0.0), 0.01) < 0.5 * np.quantile(
            mild.missingness(w, 0.0), 0.01
        )
        # And the outcome mean is no longer a linear function of (A, W), so a GLM cannot
        # be correctly specified for it -- the point of the harder process.
        design = np.column_stack([np.ones(len(w)), w])
        for arm in (0.0, 1.0):
            target = sharp.outcome_mean(w, arm, None)
            residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
            assert float(np.std(residual)) > 0.1

    def test_a_binary_outcome_can_also_be_missing(self) -> None:
        frame, truth = make_missing_outcome_binary(n=1000, seed=18)
        assert "Delta" in frame.columns
        observed = frame["Delta"].to_numpy()
        assert 0.0 < observed.mean() < 1.0
        assert np.array_equal(np.isnan(frame["Y"].to_numpy()), observed == 0.0)
        assert set(np.unique(frame["Y"].to_numpy()[observed == 1.0])) <= {0.0, 1.0}
        # The combination is what gives the ratio estimands a truth under `delta=`.
        assert truth["rr"] == pytest.approx(truth["ey1"] / truth["ey0"], abs=1e-9)
        assert truth["or"] == pytest.approx(
            (truth["ey1"] / (1 - truth["ey1"])) / (truth["ey0"] / (1 - truth["ey0"])), abs=1e-9
        )

    def test_the_intermediate_responds_to_treatment(self) -> None:
        frame, _ = make_cde(n=4000, seed=0)
        assert "Z" in frame.columns
        a = frame["A"].to_numpy()
        z = frame["Z"].to_numpy()
        # The process has a +1.1 coefficient on A, so Z must be commoner when treated.
        assert z[a == 1.0].mean() > z[a == 0.0].mean() + 0.1

    def test_clusters_share_a_latent_variable(self) -> None:
        frame, _ = make_clustered(n=1000, seed=0, cluster_size=10)
        assert "cluster" in frame.columns
        cluster = frame["cluster"].to_numpy()
        assert len(np.unique(cluster)) == 100
        # The shared effect is unobserved, so within-cluster outcomes correlate.
        y = frame["Y"].to_numpy()
        means = np.array([y[cluster == k].mean() for k in np.unique(cluster)])
        assert float(np.var(means)) > float(np.var(y)) / 10.0

    def test_binary_outcomes_are_zero_one(self) -> None:
        frame, _ = make_binary_outcome(n=500, seed=0)
        assert set(np.unique(frame["Y"].to_numpy())) <= {0.0, 1.0}

    def test_a_tiny_cluster_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cluster_size must be at least 2"):
            clustered_dgp(cluster_size=1).sample(100, seed=0)

    def test_the_registry_lists_every_generator(self) -> None:
        assert set(available()) == set(GENERATORS)
        assert len(available()) >= 7
