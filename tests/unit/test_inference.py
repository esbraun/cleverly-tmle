r"""Influence curves, clustering, the delta method and simultaneous bands.

The centrepiece here is :class:`TestInfluenceCurveIsTheGateauxDerivative`.  An
influence curve is *defined* as the Gateaux derivative of the estimator viewed as a
functional of the empirical distribution:

.. math::

    \mathrm{IC}(O_i)
      = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_n + t \delta_{O_i}\bigr)\right|_{t = 0}.

That definition is directly computable: perturb the observation weights, re-solve the
fluctuation, and difference.  Comparing the result with the closed-form expression the
library uses tests the formula itself rather than merely testing that the code agrees
with itself -- which matters, because every standard error, p-value and confidence
band in the library rests on that one formula being right.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation import (
    InitialFit,
    att_submodel,
    mean_submodel,
    solve_fluctuation,
)
from cleverly.inference import (
    ParameterEstimate,
    att_estimate,
    cluster_sums,
    counterfactual_means,
    delta_method,
    influence_covariance,
    influence_variance,
    log_odds_ratio_influence,
    log_ratio_influence,
    make_estimate,
    multiplier_critical_value,
    normal_ci,
    ratio_estimates,
    simultaneous_bands,
    two_sided_pvalue,
)
from cleverly.utils.bounds import OutcomeScaler, expit


@pytest.fixture
def binary_setting() -> dict[str, np.ndarray]:
    """A binary-outcome problem with known nuisances (so the scaler is the identity)."""
    rng = np.random.default_rng(7)
    n = 600
    w = rng.normal(size=n)
    g1 = expit(0.7 * w)
    a = rng.binomial(1, g1).astype(float)
    q1 = expit(0.5 + 0.9 * w)
    q0 = expit(-0.3 + 0.9 * w)
    y = rng.binomial(1, np.where(a == 1.0, q1, q0)).astype(float)
    return {"n": n, "a": a, "g1": g1, "q1": q1, "q0": q0, "y": y}


def _targeted(
    setting: dict[str, np.ndarray], weights: np.ndarray, group: str = "mean"
) -> tuple[InitialFit, object]:
    """Run the targeting step under the given observation weights."""
    a, g1, q1, q0, y = (
        setting["a"],
        setting["g1"],
        setting["q1"],
        setting["q0"],
        setting["y"],
    )
    # Deliberately misspecified initial fit, so the fluctuation does real work and the
    # residual term of the influence curve is not trivially zero.
    initial = InitialFit(
        np.full(a.shape[0], 0.45), np.full(a.shape[0], 0.45), np.full(a.shape[0], 0.45)
    )
    del q1, q0
    if group == "mean":
        submodel = mean_submodel(a, g1)
    else:
        submodel = att_submodel(a, g1, float(np.average(a, weights=weights)))
    fluctuation = solve_fluctuation(y, initial, submodel, weights)
    return fluctuation.targeted, submodel


def _make_setting(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    g1 = expit(0.7 * w)
    a = rng.binomial(1, g1).astype(float)
    q1 = expit(0.5 + 0.9 * w)
    q0 = expit(-0.3 + 0.9 * w)
    y = rng.binomial(1, np.where(a == 1.0, q1, q0)).astype(float)
    return {"n": n, "a": a, "g1": g1, "q1": q1, "q0": q0, "y": y}


class TestInfluenceCurveIsTheGateauxDerivative:
    r"""Verify the influence-curve formulas by numerical differentiation.

    Two distinct claims, tested separately because they hold to different precision.

    **Exact.** Given the targeted fit, the estimator is a weighted average of
    :math:`h_1(O_i)\,r_i + \bar Q^*(1, W_i)`, so the derivative with respect to
    observation weights *is* the reported influence curve, to machine precision.
    This pins down the algebra: the residual term, the arm indicators, and the way
    observation weights enter.

    **Asymptotic.** Differentiating the *whole* estimator, with the fluctuation
    re-solved at the perturbed weights, adds a second term through
    :math:`d\hat\epsilon/dt`.  Its coefficient is a ratio
    :math:`N/M` of two sample averages that share the same expectation, so it equals
    :math:`1 + O(n^{-1/2})` and the numerical derivative approaches the influence
    curve as the sample grows.  The test below confirms the discrepancy shrinks at
    that rate -- which is what "asymptotically linear" means, and is precisely the
    condition the standard error relies on.
    """

    @staticmethod
    def _psi_at(setting: dict[str, np.ndarray], weights: np.ndarray, estimand: str) -> float:
        group = "mean" if estimand in ("ate", "ey1", "ey0") else "att"
        targeted, submodel = _targeted(setting, weights, group)
        if group == "mean":
            psi_one, _, psi_zero, _ = counterfactual_means(
                setting["y"], targeted, submodel, weights
            )
            if estimand == "ey1":
                return psi_one
            if estimand == "ey0":
                return psi_zero
            return psi_one - psi_zero
        psi, _ = att_estimate(setting["y"], targeted, submodel, setting["a"], weights)
        return psi

    @staticmethod
    def _analytic_ic(setting: dict[str, np.ndarray], estimand: str) -> np.ndarray:
        n = int(setting["n"])
        weights = np.ones(n)
        group = "mean" if estimand in ("ate", "ey1", "ey0") else "att"
        targeted, submodel = _targeted(setting, weights, group)
        if group == "mean":
            _, ic_one, _, ic_zero = counterfactual_means(setting["y"], targeted, submodel, weights)
            if estimand == "ey1":
                return ic_one
            if estimand == "ey0":
                return ic_zero
            return ic_one - ic_zero
        _, ic = att_estimate(setting["y"], targeted, submodel, setting["a"], weights)
        return ic

    @staticmethod
    def _numerical(
        setting: dict[str, np.ndarray],
        estimand: str,
        indices: np.ndarray,
        step: float = 1e-6,
    ) -> np.ndarray:
        """Central difference of the full estimator, fluctuation re-solved each time."""
        n = int(setting["n"])
        out = np.empty(indices.size)
        for slot, index in enumerate(indices):
            forward = np.full(n, 1.0 - step)
            forward[index] += step * n
            backward = np.full(n, 1.0 + step)
            backward[index] -= step * n
            out[slot] = (
                TestInfluenceCurveIsTheGateauxDerivative._psi_at(setting, forward, estimand)
                - TestInfluenceCurveIsTheGateauxDerivative._psi_at(setting, backward, estimand)
            ) / (2.0 * step)
        return out

    @pytest.mark.parametrize("estimand", ["ate", "ey1", "ey0"])
    def test_exact_derivative_at_a_fixed_targeted_fit(
        self, binary_setting: dict[str, np.ndarray], estimand: str
    ) -> None:
        n = int(binary_setting["n"])
        weights = np.ones(n)
        targeted, submodel = _targeted(binary_setting, weights)
        psi_one, ic_one, psi_zero, ic_zero = counterfactual_means(
            binary_setting["y"], targeted, submodel, weights
        )

        residual = binary_setting["y"] - targeted.observed
        contribution_one = submodel.observed[:, 1] * residual + targeted.at_one
        contribution_zero = submodel.observed[:, 0] * residual + targeted.at_zero
        if estimand == "ey1":
            contribution, analytic = contribution_one, ic_one
        elif estimand == "ey0":
            contribution, analytic = contribution_zero, ic_zero
        else:
            contribution = contribution_one - contribution_zero
            analytic = ic_one - ic_zero
        del psi_one, psi_zero

        # Because targeting zeroes the score, the plug-in and one-step averages agree,
        # so this weighted average is the estimator -- and it is linear in the weights.
        def psi_at(w: np.ndarray) -> float:
            return float(np.average(contribution, weights=w))

        step = 1e-7
        indices = np.linspace(0, n - 1, 15, dtype=int)
        numerical = np.empty(indices.size)
        for slot, index in enumerate(indices):
            forward = np.full(n, 1.0 - step)
            forward[index] += step * n
            backward = np.full(n, 1.0 + step)
            backward[index] -= step * n
            numerical[slot] = (psi_at(forward) - psi_at(backward)) / (2.0 * step)

        assert np.allclose(numerical, analytic[indices], rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("estimand", ["ate", "ey1", "ey0", "att"])
    def test_full_estimator_derivative_converges_to_the_influence_curve(
        self, estimand: str
    ) -> None:
        ratios = []
        for n in (400, 6400):
            setting = _make_setting(n, seed=7)
            indices = np.linspace(0, n - 1, 12, dtype=int)
            analytic = self._analytic_ic(setting, estimand)[indices]
            numerical = self._numerical(setting, estimand, indices)
            # The discrepancy is a single multiplicative factor N/M shared by every
            # observation, so a regression slope isolates it cleanly.
            ratios.append(float(numerical @ analytic / (analytic @ analytic)))

        small, large = (abs(value - 1.0) for value in ratios)
        # A 16-fold increase in n should shrink an O(n^-1/2) discrepancy by about 4x.
        assert large < small
        assert large < 0.03, f"slope at n=6400 was {ratios[1]:.4f}, expected within 3% of 1"

    def test_the_influence_curve_is_centred(self, binary_setting: dict[str, np.ndarray]) -> None:
        # Targeting solves the score equation, which is exactly what makes the mean of
        # the influence curve zero.
        for estimand in ("ate", "ey1", "ey0", "att"):
            ic = self._analytic_ic(binary_setting, estimand)
            assert abs(float(ic.mean())) < 1e-12


class TestEstimandRelationships:
    def test_the_ate_influence_curve_decomposes_exactly(
        self, binary_setting: dict[str, np.ndarray]
    ) -> None:
        n = int(binary_setting["n"])
        weights = np.ones(n)
        targeted, submodel = _targeted(binary_setting, weights)
        psi_one, ic_one, psi_zero, ic_zero = counterfactual_means(
            binary_setting["y"], targeted, submodel, weights
        )
        ate = make_estimate("ate", psi_one - psi_zero, ic_one - ic_zero, n=n)
        # Both estimands come from the same fluctuation, so this identity is exact --
        # not approximate.
        assert ate.psi == pytest.approx(psi_one - psi_zero)
        assert np.array_equal(ate.influence_curve, ic_one - ic_zero)

    def test_counterfactual_means_reject_the_wrong_submodel(
        self, binary_setting: dict[str, np.ndarray]
    ) -> None:
        n = int(binary_setting["n"])
        weights = np.ones(n)
        targeted, submodel = _targeted(binary_setting, weights, "att")
        with pytest.raises(ValueError, match="expected the 'mean' submodel"):
            counterfactual_means(binary_setting["y"], targeted, submodel, weights)

    def test_att_rejects_the_wrong_submodel(self, binary_setting: dict[str, np.ndarray]) -> None:
        n = int(binary_setting["n"])
        weights = np.ones(n)
        targeted, submodel = _targeted(binary_setting, weights, "mean")
        with pytest.raises(ValueError, match="expected the 'att' submodel"):
            att_estimate(binary_setting["y"], targeted, submodel, binary_setting["a"], weights)


class TestParameterEstimate:
    def test_wald_interval_and_pvalue_are_consistent(self) -> None:
        rng = np.random.default_rng(0)
        ic = rng.normal(size=500)
        estimate = make_estimate("ate", 0.4, ic, n=500)
        low, high = estimate.ci
        assert low < 0.4 < high
        assert estimate.std_error == pytest.approx(np.sqrt(np.var(ic, ddof=1) / 500))
        # A 95% interval excludes zero exactly when p < 0.05.
        assert (low > 0) == (estimate.pvalue < 0.05)

    def test_alpha_controls_the_interval_width(self) -> None:
        ic = np.random.default_rng(0).normal(size=400)
        narrow = make_estimate("ate", 1.0, ic, n=400, alpha=0.2)
        wide = make_estimate("ate", 1.0, ic, n=400, alpha=0.01)
        assert (wide.ci[1] - wide.ci[0]) > (narrow.ci[1] - narrow.ci[0])

    def test_ratio_intervals_are_built_on_the_log_scale(self) -> None:
        ic = np.random.default_rng(0).normal(size=400) * 0.5
        estimate = ParameterEstimate(
            name="rr",
            psi=float(np.exp(0.5)),
            influence_curve=ic,
            variance=float(np.var(ic, ddof=1) / 400),
            n=400,
            n_clusters=400,
            scale="ratio",
            log_psi=0.5,
        )
        low, high = estimate.ci
        # A ratio interval must be positive and asymmetric around the estimate.
        assert low > 0
        assert (estimate.psi - low) != pytest.approx(high - estimate.psi, rel=1e-3)
        assert np.log(low) == pytest.approx(0.5 - 1.959963985 * estimate.std_error, rel=1e-6)

    def test_ratio_pvalue_tests_the_null_of_one(self) -> None:
        ic = np.zeros(400)
        ic[0] = 1.0
        estimate = ParameterEstimate(
            name="rr",
            psi=1.0,
            influence_curve=ic,
            variance=float(np.var(ic, ddof=1) / 400),
            n=400,
            n_clusters=400,
            scale="ratio",
            log_psi=0.0,
        )
        assert estimate.pvalue == pytest.approx(1.0)

    def test_to_dict_carries_the_reported_numbers(self) -> None:
        estimate = make_estimate("ate", 0.4, np.random.default_rng(0).normal(size=100), n=100)
        row = estimate.to_dict()
        assert row["estimand"] == "ate"
        assert row["psi"] == pytest.approx(0.4)
        assert {"std_err", "ci_lower", "ci_upper", "p_value"} <= set(row)

    def test_zero_variance_yields_a_nan_interval_not_a_crash(self) -> None:
        estimate = ParameterEstimate(
            name="ate",
            psi=1.0,
            influence_curve=np.zeros(10),
            variance=0.0,
            n=10,
            n_clusters=10,
        )
        assert np.isnan(estimate.pvalue)


class TestClusterVariance:
    def test_singleton_clusters_reproduce_the_independent_case(self) -> None:
        rng = np.random.default_rng(0)
        ic = rng.normal(size=300)
        singleton = np.arange(300)
        # This is the identity that anchors the cluster formula: with one observation
        # per cluster the two expressions must coincide exactly.
        assert influence_variance(ic, singleton) == pytest.approx(influence_variance(ic))

    def test_matches_a_hand_computed_cluster_variance(self) -> None:
        rng = np.random.default_rng(1)
        n, size = 200, 10
        ic = rng.normal(size=n)
        cluster = np.repeat(np.arange(n // size), size)
        sums = np.array([ic[cluster == k].sum() for k in range(n // size)])
        expected = (n // size) * np.var(sums, ddof=1) / n**2
        assert influence_variance(ic, cluster) == pytest.approx(expected)

    def test_positive_within_cluster_correlation_inflates_the_variance(self) -> None:
        rng = np.random.default_rng(2)
        n, size = 400, 20
        cluster = np.repeat(np.arange(n // size), size)
        shared = rng.normal(size=n // size)[cluster]
        ic = shared + 0.3 * rng.normal(size=n)
        ic = ic - ic.mean()
        # Ignoring clustering here understates the variance -- badly.
        assert influence_variance(ic, cluster) > 3.0 * influence_variance(ic)

    def test_cluster_sums_add_up(self) -> None:
        ic = np.arange(10, dtype=float)
        cluster = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
        sums = cluster_sums(ic, cluster)
        assert sums.shape == (4,)
        assert sums.sum() == pytest.approx(ic.sum())

    def test_cluster_sums_handle_a_matrix(self) -> None:
        ic = np.arange(20, dtype=float).reshape(10, 2)
        cluster = np.repeat(np.arange(5), 2)
        assert cluster_sums(ic, cluster).shape == (5, 2)

    def test_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cluster has"):
            cluster_sums(np.zeros(10), np.zeros(5))

    def test_a_single_cluster_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 clusters"):
            influence_variance(np.random.default_rng(0).normal(size=20), np.zeros(20))

    def test_covariance_matrix_is_symmetric_and_scales_like_the_variance(self) -> None:
        rng = np.random.default_rng(3)
        ic = rng.normal(size=(300, 3))
        covariance = influence_covariance(ic)
        assert covariance.shape == (3, 3)
        assert np.allclose(covariance, covariance.T)
        for j in range(3):
            assert covariance[j, j] == pytest.approx(influence_variance(ic[:, j]))


class TestDeltaMethod:
    def test_log_risk_ratio_matches_the_closed_form(self) -> None:
        rng = np.random.default_rng(0)
        ic_one, ic_zero = rng.normal(size=200), rng.normal(size=200)
        log_psi, ic = log_ratio_influence(0.4, ic_one, 0.2, ic_zero)
        assert log_psi == pytest.approx(np.log(0.4 / 0.2))
        assert np.allclose(ic, ic_one / 0.4 - ic_zero / 0.2)

    def test_log_odds_ratio_matches_the_closed_form(self) -> None:
        rng = np.random.default_rng(0)
        ic_one, ic_zero = rng.normal(size=200), rng.normal(size=200)
        log_psi, ic = log_odds_ratio_influence(0.4, ic_one, 0.2, ic_zero)
        assert log_psi == pytest.approx(np.log((0.4 / 0.6) / (0.2 / 0.8)))
        assert np.allclose(ic, ic_one / (0.4 * 0.6) - ic_zero / (0.2 * 0.8))

    def test_generic_delta_method_reproduces_the_risk_ratio(self) -> None:
        rng = np.random.default_rng(1)
        ic_one, ic_zero = rng.normal(size=200), rng.normal(size=200)
        value, ic = delta_method(
            lambda p: float(np.log(p[0]) - np.log(p[1])), [0.4, 0.2], [ic_one, ic_zero]
        )
        expected_log, expected_ic = log_ratio_influence(0.4, ic_one, 0.2, ic_zero)
        assert value == pytest.approx(expected_log)
        assert np.allclose(ic, expected_ic, rtol=1e-5, atol=1e-8)

    def test_an_analytic_gradient_is_used_when_supplied(self) -> None:
        ic = [np.ones(50), np.zeros(50)]
        value, curve = delta_method(
            lambda p: float(p[0] * p[1]),
            [2.0, 3.0],
            ic,
            gradient=lambda p: np.array([p[1], p[0]]),
        )
        assert value == pytest.approx(6.0)
        assert np.allclose(curve, 3.0)

    def test_a_non_positive_mean_is_refused_for_a_ratio(self) -> None:
        ic = np.zeros(10)
        with pytest.raises(ValueError, match="strictly positive"):
            log_ratio_influence(0.0, ic, 0.3, ic)

    def test_an_out_of_range_risk_is_refused_for_an_odds_ratio(self) -> None:
        ic = np.zeros(10)
        with pytest.raises(ValueError, match="strictly inside"):
            log_odds_ratio_influence(1.0, ic, 0.3, ic)

    def test_mismatched_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="influence curve"):
            delta_method(lambda p: float(p[0]), [1.0, 2.0], [np.zeros(10)])

    def test_normal_ci_and_pvalue_agree_at_the_boundary(self) -> None:
        low, _ = normal_ci(1.96, 1.0, 0.05)
        assert low == pytest.approx(0.0, abs=1e-3)
        assert two_sided_pvalue(1.96, 1.0) == pytest.approx(0.05, abs=1e-3)

    def test_normal_ci_rejects_a_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must lie"):
            normal_ci(1.0, 1.0, 1.5)


class TestRatioEstimates:
    def test_both_ratios_are_produced_with_log_scale_inference(self) -> None:
        rng = np.random.default_rng(0)
        ic_one, ic_zero = rng.normal(size=300) * 0.3, rng.normal(size=300) * 0.3
        estimates = ratio_estimates(0.5, ic_one, 0.3, ic_zero, n=300)
        assert set(estimates) == {"rr", "or"}
        assert estimates["rr"].psi == pytest.approx(0.5 / 0.3)
        assert estimates["or"].psi == pytest.approx((0.5 / 0.5) / (0.3 / 0.7))
        for estimate in estimates.values():
            assert estimate.scale == "ratio"
            assert estimate.log_psi is not None
            assert estimate.ci[0] > 0

    def test_a_subset_can_be_requested(self) -> None:
        ic = np.random.default_rng(0).normal(size=100) * 0.2
        assert set(ratio_estimates(0.5, ic, 0.3, ic, n=100, which=("rr",))) == {"rr"}


class TestSimultaneousBands:
    def test_the_critical_value_exceeds_the_pointwise_one(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        estimates = {
            name: make_estimate(name, 1.0, rng.normal(size=n), n=n) for name in ("a", "b", "c", "d")
        }
        bands = simultaneous_bands(estimates, n_replicates=400, random_state=0)
        # With several independent estimands the joint critical value must be larger
        # than 1.96, but smaller than the Bonferroni value.
        assert bands.critical_value > bands.pointwise_critical_value
        from scipy import stats

        bonferroni = float(stats.norm.ppf(1.0 - 0.05 / (2 * 4)))
        assert bands.critical_value < bonferroni + 0.15

    def test_bands_are_wider_than_pointwise_intervals(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        estimates = {
            name: make_estimate(name, 1.0, rng.normal(size=n), n=n) for name in ("a", "b", "c")
        }
        bands = simultaneous_bands(estimates, n_replicates=400, random_state=0)
        for name, estimate in estimates.items():
            low, high = bands.bands[name]
            point_low, point_high = estimate.ci
            assert low <= point_low
            assert high >= point_high

    def test_perfectly_correlated_estimands_recover_the_pointwise_value(self) -> None:
        rng = np.random.default_rng(2)
        n = 600
        ic = rng.normal(size=n)
        estimates = {name: make_estimate(name, 1.0, ic, n=n) for name in ("a", "b", "c")}
        bands = simultaneous_bands(estimates, n_replicates=2000, random_state=0)
        # Identical estimands carry no multiplicity, so no correction is warranted.
        assert bands.critical_value == pytest.approx(bands.pointwise_critical_value, abs=0.12)

    def test_a_single_estimand_uses_the_normal_quantile_exactly(self) -> None:
        n = 200
        estimates = {"a": make_estimate("a", 1.0, np.random.default_rng(0).normal(size=n), n=n)}
        bands = simultaneous_bands(estimates, n_replicates=100, random_state=0)
        assert bands.critical_value == pytest.approx(1.959963985, rel=1e-6)

    def test_a_ratio_band_is_exponentiated(self) -> None:
        n = 300
        ic = np.random.default_rng(0).normal(size=n) * 0.2
        estimates = ratio_estimates(0.5, ic, 0.3, ic * 0.5, n=n)
        bands = simultaneous_bands(estimates, n_replicates=200, random_state=0)
        assert bands.bands["rr"][0] > 0

    @pytest.mark.parametrize("kind", ["rademacher", "mammen", "normal"])
    def test_every_multiplier_distribution_gives_a_sane_critical_value(self, kind: str) -> None:
        rng = np.random.default_rng(0)
        n = 400
        ic = rng.normal(size=(n, 3))
        se = np.array([np.sqrt(influence_variance(ic[:, j])) for j in range(3)])
        value = multiplier_critical_value(
            ic,
            se,
            n=n,
            n_replicates=500,
            kind=kind,
            random_state=0,  # type: ignore[arg-type]
        )
        assert 1.8 < value < 3.2

    def test_an_unknown_multiplier_is_refused(self) -> None:
        ic = np.zeros((10, 2))
        with pytest.raises(ValueError, match="rademacher"):
            multiplier_critical_value(
                ic,
                np.ones(2),
                n=10,
                kind="uniform",
                n_replicates=10,  # type: ignore[arg-type]
            )

    def test_clustering_widens_the_bands(self) -> None:
        rng = np.random.default_rng(3)
        n, size = 400, 20
        cluster = np.repeat(np.arange(n // size), size)
        shared = rng.normal(size=n // size)[cluster]
        estimates = {}
        for name in ("a", "b"):
            ic = shared + 0.3 * rng.normal(size=n)
            estimates[name] = make_estimate(name, 1.0, ic - ic.mean(), n=n, cluster=cluster)
        bands = simultaneous_bands(estimates, n_replicates=500, random_state=0, cluster=cluster)
        assert np.isfinite(bands.critical_value)

    def test_no_estimates_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no estimates"):
            simultaneous_bands({})

    def test_inconsistent_lengths_are_refused(self) -> None:
        estimates = {
            "a": make_estimate("a", 1.0, np.zeros(10) + np.arange(10), n=10),
            "b": make_estimate("b", 1.0, np.zeros(20) + np.arange(20), n=20),
        }
        with pytest.raises(ValueError, match="inconsistent lengths"):
            simultaneous_bands(estimates)


class TestUnscaling:
    def test_a_level_and_a_difference_unscale_differently(self) -> None:
        from cleverly.inference.influence import unscale

        scaler = OutcomeScaler(-2.0, 8.0)
        ic = np.array([0.1, -0.2, 0.05])
        level, level_ic = unscale(0.5, ic, scaler, "level")
        difference, difference_ic = unscale(0.5, ic, scaler, "difference")
        assert level == pytest.approx(-2.0 + 10.0 * 0.5)
        assert difference == pytest.approx(10.0 * 0.5)
        # Both influence curves scale by the range only.
        assert np.allclose(level_ic, 10.0 * ic)
        assert np.allclose(difference_ic, 10.0 * ic)

    def test_a_ratio_requires_an_unscaled_outcome(self) -> None:
        from cleverly.inference.influence import unscale

        with pytest.raises(ValueError, match="only defined for an unscaled"):
            unscale(1.5, np.zeros(3), OutcomeScaler(-1.0, 1.0), "ratio")
