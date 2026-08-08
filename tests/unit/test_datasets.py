"""The synthetic processes and their reference values.

Anything that validates the estimator is only as good as the truth it compares
against, so the truth itself is tested: the quasi-Monte Carlo integration must be
deterministic, must agree with a large plain Monte Carlo draw, and must reproduce the
estimands that are known in closed form.
"""

from __future__ import annotations

import dataclasses

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
    make_multi_arm,
    make_nonlinear_ate,
    make_weak_overlap,
    missing_outcome_dgp,
    multi_arm_dgp,
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

    def test_expectation_is_the_same_rule_the_truth_is_integrated_with(self) -> None:
        """Bit for bit, which is the whole reason the method exists rather than an approximation.

        A caller integrating a remainder term or a drift coefficient compares it against a
        ``truth()`` value, so a second quadrature -- a large random draw, or Sobol points at
        another seed -- would put a Monte Carlo error of its own between the two and leave a
        disagreement unattributable.  ``approx`` would pass against exactly that, so this
        asserts equality.
        """
        dgp = nonlinear_dgp()

        assert dgp.expectation(lambda w: dgp.outcome_mean(w, 1.0, None)) == dgp.truth()["ey1"]

    def test_expectation_refuses_an_integrand_of_the_wrong_length(self) -> None:
        """A scalar or a subsample would average to something plausible and wrong."""
        with pytest.raises(ValueError, match="one value per integration point"):
            nonlinear_dgp().expectation(lambda w: np.zeros(3))

    def test_the_grids_are_prefixes_of_one_another(self) -> None:
        """The property a convergence ladder rests on, and it is exact or it is nothing.

        Three grids that are not nested are three rules, and the movement between two of
        them is then reshuffling rather than refinement.  A caller reading a whole ladder off
        one companion by slicing prefixes gets the same integral at a coarser grid only
        because of this.
        """
        dgp = nonlinear_dgp()
        coarse, fine, whole = dgp.quadrature(2**10), dgp.quadrature(2**13), dgp.quadrature()

        assert np.array_equal(coarse, fine[: 2**10])
        assert np.array_equal(coarse, whole[: 2**10])

    def test_the_default_grid_is_the_rule_the_truth_is_integrated_with(self) -> None:
        """Bit for bit, and it is the *composition* being pinned rather than the point set.

        A remainder diagnostic integrates :math:`\\psi_0` on the companion's
        own grid so that it cancels against :math:`P_0\\hat D`'s plug-in half, and the whole
        cancellation rests on ``quadrature`` returning the points ``truth`` uses.  Those are
        two methods that could drift apart -- the reason there is one call rather than two
        literals -- so this asserts the identity rather than a tolerance.
        """
        dgp = nonlinear_dgp()

        levels = dgp.outcome_mean(dgp.quadrature(), 1.0, None)
        assert dgp.truth()["ey1"] == float(np.mean(levels))

    def test_a_grid_that_is_not_a_power_of_two_is_refused(self) -> None:
        """Nesting is only meaningful between powers of two, and so is Sobol's balance."""
        with pytest.raises(ValueError, match="positive power of two"):
            nonlinear_dgp().quadrature(1000)

    def test_the_default_scramble_is_bit_for_bit_what_it_was(self) -> None:
        """The regression pin on E1b's only ``src/`` change, and it protects ``truth``.

        ``scramble=`` is additive or it is a silent change to every reference value in the
        package -- every ``truth()``, every committed drift coefficient, every fixture
        tolerance sized against ``1e-5``.  A default argument makes that true by
        construction, so what this pins is that the default *is* the constant the reference
        values were taken at rather than merely some constant.
        """
        dgp = nonlinear_dgp()

        assert np.array_equal(dgp.quadrature(2**10), dgp.quadrature(2**10, scramble=20240101))

    def test_two_scrambles_are_independent_randomisations_rather_than_two_grids(self) -> None:
        """Different points, same integral -- which is the whole of why a spread is an error.

        A second scramble is not a finer or a coarser rule and the difference between two of
        them is not a discretisation.  Both are unbiased for the same integral at the same
        point count, so they agree to the rule's own accuracy while sharing no point: that
        pairing is what makes an across-scramble spread a standard error rather than a
        comparison of two things that might be converging to different answers.
        """
        dgp = nonlinear_dgp()
        one, two = dgp.quadrature(2**13), dgp.quadrature(2**13, scramble=7)

        assert not np.array_equal(one, two)
        levels = [float(np.mean(dgp.outcome_mean(grid, 1.0, None))) for grid in (one, two)]
        assert levels[0] == pytest.approx(levels[1], abs=5e-3)

    def test_the_grids_nest_within_a_scramble_and_not_across_scrambles(self) -> None:
        """The ladder still works per scramble, which is what lets one fit carry both.

        The negative half is the point: a prefix of *another* scramble's grid is not a
        coarser version of this one, so a caller that slices a ladder out of a stack of
        scrambles has to slice within a block, and this is the property that makes a
        within-block window well defined.
        """
        dgp = nonlinear_dgp()

        assert np.array_equal(
            dgp.quadrature(2**10, scramble=7), dgp.quadrature(2**13, scramble=7)[: 2**10]
        )
        assert not np.array_equal(dgp.quadrature(2**10), dgp.quadrature(2**13, scramble=7)[: 2**10])

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


class TestTheMultiArmBinomialLaw:
    r"""``family="binomial"`` on the three-armed process, and the clip it replaced.

    ``MultiArmDGP.sample`` used to ``np.clip`` the outcome mean into ``[0, 1]`` before
    drawing, while :meth:`MultiArmDGP.truth` went on integrating the *unclipped* mean.  On
    ``multi_arm_dgp()`` -- whose arm means are ``0``, ``0.6`` and ``1.44`` -- that reported
    ``ey[high] = 1.44`` as the counterfactual mean of a binary outcome, and drew the high
    arm as a constant one.  Nothing shipped reached it, which is exactly why it survived:
    a footgun with no caller has no failing test either.

    **Verified by mutation**: restoring the clip (``rng.binomial(1, np.clip(mean, 0, 1))``
    and deleting the guard call) turns :meth:`test_the_gaussian_law_declared_binomial_is_
    refused` red and leaves every other test in this file green -- including the ones about
    the gaussian law, which is the point.

    The binary law itself exists because ``rr`` and ``or``, and the E-value built on them,
    need a binomial outcome and there was no multi-arm law to ask them of.
    """

    def test_the_arm_means_are_probabilities(self) -> None:
        truth = multi_arm_dgp(family="binomial").truth()
        means = [truth[f"ey[{label}]"] for label in ("low", "medium", "high")]
        assert all(0.0 < mean < 1.0 for mean in means), means

    def test_the_middle_arm_is_still_not_halfway(self) -> None:
        """The property the gaussian law exists for, carried onto the logit scale.

        An estimator that treated the treatment as one numeric column has to be *biased*
        here rather than merely inefficient, and that needs the middle arm off centre.  The
        expit is close to linear over this range, so a step vector chosen for the gaussian
        law would have come out near halfway on the mean scale -- this asserts it did not.
        """
        truth = multi_arm_dgp(family="binomial").truth()
        low, middle, high = (truth[f"ey[{label}]"] for label in ("low", "medium", "high"))
        position = (middle - low) / (high - low)
        assert 0.2 < position < 0.4, position

    def test_the_draw_is_binary_and_the_gaussian_law_is_untouched(self) -> None:
        frame, truth = make_multi_arm(400, seed=0, family="binomial")
        assert set(np.unique(np.asarray(frame["Y"]))) == {0.0, 1.0}
        # The default is unchanged to the last digit: this added a family, not a law.
        _, gaussian = make_multi_arm(400, seed=0)
        assert gaussian["ey[high]"] == pytest.approx(1.44, abs=1e-6)
        assert truth["ey[high]"] != gaussian["ey[high]"]

    def test_the_gaussian_law_declared_binomial_is_refused(self) -> None:
        """The guard, at the configuration that used to lie.

        Refused rather than rescaled: a law on the gaussian scale is a *different law*,
        and squashing it would draw from one and report the other.
        """
        law = dataclasses.replace(multi_arm_dgp(), family="binomial")
        with pytest.raises(ValueError, match="needs an outcome_mean that is a probability"):
            law.sample(50, seed=0)

    def test_the_refusal_names_the_arm_and_the_range(self) -> None:
        law = dataclasses.replace(multi_arm_dgp(), family="binomial")
        with pytest.raises(ValueError) as raised:
            law.sample(50, seed=0)
        message = str(raised.value)
        assert "'low'" in message, message
        assert "multi_arm_dgp(family='binomial')" in message, message

    def test_the_verdict_does_not_depend_on_the_draw(self) -> None:
        """Checked on the quadrature points, not the sample.

        A law admissible at one ``n`` or seed and refused at the next would be worse than
        either verdict, because it would turn a property of the law into a flake.
        """
        law = dataclasses.replace(multi_arm_dgp(), family="binomial")
        for n, seed in ((10, 0), (10, 7), (5_000, 1)):
            with pytest.raises(ValueError):
                law.sample(n, seed=seed)

    def test_an_unknown_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="family must be"):
            multi_arm_dgp(family="poisson")
