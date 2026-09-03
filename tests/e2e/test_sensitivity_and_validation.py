"""Sensitivity and validation analyses on real fits.

These modules are the reason the library exists in the shape it does, so the tests
check that they *say the right thing*, not merely that they return a frame.  Each
analysis is exercised on a case where the correct answer is known by construction:

* positivity diagnostics on data with and without an overlap problem;
* the robustness value on a fit with a *known* confounder withheld;
* the missingness tilt at ``gamma = 0``, which must reproduce the MAR estimate exactly;
* refutation tests, where a placebo treatment must yield no effect.

Every fixture above carries unit observation weights, so none of those cases can see a
diagnostic or a tilt that drops the weights.  ``weighted_missing_fit`` is the one that
can: :class:`TestPositivityUnderObservationWeights` and
:class:`TestTheTiltUnderObservationWeights` run on it, and each writes its expectation out
from the displayed formula rather than from the helper it is checking.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import replace

import narwhals as nw
import numpy as np
import pytest
import sklearn.linear_model

from cleverly import AssessmentStatus, ParameterKey, SuperLearner
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_missing_outcome,
    make_nonlinear_ate,
    make_weak_overlap,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, PositivityWarning
from cleverly.sensitivity.omitted_variable import benchmark
from cleverly.validation.refute import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    RelativeGaussianNoise,
    refute,
)
from tests.conftest import fast_tmle


@pytest.fixture(scope="module")
def good_overlap() -> object:
    frame, _ = make_linear_ate(n=1500, seed=71)
    return (
        fast_tmle(estimands=("ate", "att", "ey1", "ey0"))
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@pytest.fixture(scope="module")
def poor_overlap() -> object:
    frame, _ = make_weak_overlap(n=1500, seed=72)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        return fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()


def _weight_by_covariate_rank(column: np.ndarray) -> np.ndarray:
    """``np.linspace(0.5, 1.5, n)``, handed out in the order ``column`` ranks the rows.

    ``linspace`` gives the profile a fixed nonconstant shape, and the covariate order
    gives it something to say.  Handed out by row index it would be independent of
    everything the fit uses, so the weighted and the unweighted answer would differ only
    by sampling noise and the witnesses below could state no margin at all.
    """
    values = np.asarray(column, dtype=float)
    return np.linspace(0.5, 1.5, values.size)[np.argsort(np.argsort(values))]


def _kish(weights: np.ndarray) -> float:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``, written out.

    A sibling of the helper of the same name in
    :mod:`tests.unit.test_sensitivity_multi_arm`.  Six lines duplicated rather than
    shared, because the only home both modules could import from is ``tests/conftest.py``
    and this change does not own it.
    """
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def _top_share(weights: np.ndarray, fraction: float) -> float:
    """Share of the total weight held by the largest ``fraction`` of the rows."""
    w = np.asarray(weights, dtype=float)
    count = max(1, int(np.ceil(fraction * w.size)))
    return float(np.sort(w)[-count:].sum() / w.sum())


@pytest.fixture(scope="module")
def weighted_missing_fit() -> object:
    """One weighted fit, carrying missing outcomes, for the two weighted witnesses.

    Missingness costs the positivity witness nothing and the tilt cannot run without it,
    so the two share a fit rather than paying for one each.
    """
    frame, _ = make_missing_outcome(n=800, seed=11)
    weighted = frame.assign(obs_weight=_weight_by_covariate_rank(frame["W1"].to_numpy()))
    return (
        fast_tmle(estimands=("ate", "att", "atc", "ey1", "ey0"))
        .fit(
            weighted,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
            weights="obs_weight",
        )
        .single()
    )


class TestPositivity:
    def test_good_overlap_is_reported_as_adequate(self, good_overlap) -> None:
        report = good_overlap.diagnostics.support()
        assert "adequate" in report.verdict()
        assert report.truncated["fraction"] == 0.0
        for arm in ("treated", "control"):
            # Almost all of the nominal sample size survives the reweighting.
            assert report.effective_sample_size[arm]["ratio"] > 0.8

    def test_poor_overlap_is_flagged(self, poor_overlap) -> None:
        report = poor_overlap.diagnostics.support()
        assert "adequate" not in report.verdict()
        assert report.truncated["fraction"] > 0.0
        worst = min(ess["ratio"] for ess in report.effective_sample_size.values())
        # The weighted analysis is using far fewer observations than it appears to.
        assert worst < 0.6

    def test_the_effective_sample_size_never_exceeds_the_arm_size(self, poor_overlap) -> None:
        report = poor_overlap.diagnostics.support()
        for ess in report.effective_sample_size.values():
            assert ess["effective"] <= ess["n"] + 1e-9

    def test_weight_concentration_is_higher_under_poor_overlap(
        self, good_overlap, poor_overlap
    ) -> None:
        good = good_overlap.diagnostics.support().weight_share["treated"]["top_1pct"]
        poor = poor_overlap.diagnostics.support().weight_share["treated"]["top_1pct"]
        assert poor > good

    def test_the_report_renders_and_tabulates(self, good_overlap) -> None:
        report = good_overlap.diagnostics.support()
        text = report.summary()
        assert "Positivity" in text
        assert "VERDICT" in text
        frame = report.to_frame(good_overlap.data)
        assert set(nw.from_native(frame, eager_only=True)["group"].unique().to_list()) == {
            "overall",
            "treated",
            "control",
        }


class TestPositivityUnderObservationWeights:
    """The leverage a weighted point-treatment fit reports is the *product* of the two.

    The sibling of
    ``tests/e2e/test_ltmle.py::TestObservationWeights::test_the_diagnostics_fold_the_weights_into_the_leverage``,
    which makes this statement for the longitudinal estimator and cites
    ``result.diagnostics.support()`` as having made the same choice.  This is the check
    that holds the point-treatment side to it.  Both the module docstring of
    :mod:`cleverly.sensitivity.positivity` and
    ``docs/technical-reference/validation-methods.md`` state the claim in prose.

    Every check in :class:`TestPositivity` above passes unchanged on an implementation
    that ignores the observation weights, because none of its fits carries any.
    """

    def test_the_diagnostics_fold_the_weights_into_the_leverage(self, weighted_missing_fit) -> None:
        report = weighted_missing_fit.diagnostics.support()
        data = weighted_missing_fit.data
        lower, upper = weighted_missing_fit.config.g_bounds
        bounded = np.clip(weighted_missing_fit.nuisance.propensity.arm(1.0), lower, upper)
        treated = np.asarray(data.treatment == 1.0)

        for arm, mask, covariate in (
            ("treated", treated, 1.0 / bounded),
            ("control", ~treated, 1.0 / (1.0 - bounded)),
        ):
            clever = covariate[mask]
            leverage = clever * data.weights[mask]
            nominal = float(mask.sum())

            ess = report.effective_sample_size[arm]
            assert ess["effective"] == pytest.approx(_kish(leverage), abs=0)
            assert ess["ratio"] == pytest.approx(_kish(leverage) / nominal, abs=0)
            share = report.weight_share[arm]
            assert share["top_1pct"] == pytest.approx(_top_share(leverage, 0.01), abs=0)
            assert share["top_5pct"] == pytest.approx(_top_share(leverage, 0.05), abs=0)

            # The observation weighting materially changes the leverage rather than
            # merely carrying an unused array alongside it.
            assert not np.allclose(leverage, clever)
            # And it changes what is *reported*.  The smaller of the two arms moves by 15
            # units of effective sample size and by 0.017 of the top-5% share; the other
            # moves by 65 and by 0.037.
            assert abs(_kish(leverage) - _kish(clever)) > 10.0
            assert abs(_top_share(leverage, 0.05) - _top_share(clever, 0.05)) > 0.01


class TestTruncationCurve:
    def test_the_curve_is_flat_when_overlap_is_good(self, good_overlap) -> None:
        curve = nw.from_native(
            good_overlap.diagnostics.truncation_curve([0.001, 0.01, 0.05], estimands=["ate"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        # No propensity is near the boundary, so truncation cannot bite.
        assert float(values.max() - values.min()) < 1e-9

    def test_the_curve_moves_when_overlap_is_poor(self, poor_overlap) -> None:
        curve = nw.from_native(
            poor_overlap.diagnostics.truncation_curve([0.001, 0.01, 0.05, 0.15], estimands=["ate"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        errors = np.array(curve["std_err"].to_list())
        truncated = np.array(curve["truncated_fraction"].to_list())
        assert float(values.max() - values.min()) > 1e-3
        # Tighter bounds truncate more units and buy variance with bias.
        assert np.all(np.diff(truncated) > 0)
        assert errors[0] > errors[-1]

    def test_the_fitted_bound_is_marked(self, good_overlap) -> None:
        curve = nw.from_native(
            good_overlap.diagnostics.truncation_curve(estimands=["ate"]), eager_only=True
        )
        assert sum(curve["is_fitted_bound"].to_list()) >= 1

    def test_an_invalid_bound_is_refused(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            good_overlap.diagnostics.truncation_curve([0.7])


class TestOmittedVariableBias:
    def test_the_bound_grows_with_the_assumed_confounding(self, good_overlap) -> None:
        weak = good_overlap.sensitivity.omitted_confounding("ate", cf_y=0.01, cf_d=0.01)
        strong = good_overlap.sensitivity.omitted_confounding("ate", cf_y=0.10, cf_d=0.10)
        assert strong.bias > weak.bias
        assert strong.lower < weak.lower
        assert strong.upper > weak.upper

    def test_zero_confounding_reproduces_the_point_estimate(self, good_overlap) -> None:
        bounds = good_overlap.sensitivity.omitted_confounding("ate", cf_y=0.0, cf_d=0.0)
        assert bounds.lower == pytest.approx(bounds.psi)
        assert bounds.upper == pytest.approx(bounds.psi)

    def test_the_robustness_value_is_where_the_bound_reaches_the_null(self, good_overlap) -> None:
        values = good_overlap.sensitivity.robustness_value("ate")
        rv = values["rv"]
        assert 0.0 < rv < 1.0
        # By definition, setting cf_y = cf_d = RV must put the bound at zero.
        at_rv = good_overlap.sensitivity.omitted_confounding("ate", cf_y=rv, cf_d=rv)
        edge = at_rv.lower if at_rv.psi > 0 else at_rv.upper
        assert edge == pytest.approx(0.0, abs=1e-4)

    def test_the_confidence_robustness_value_is_the_smaller_one(self, good_overlap) -> None:
        values = good_overlap.sensitivity.robustness_value("ate")
        # It takes less confounding to make an interval touch the null than to move the
        # point estimate there.
        assert values["rva"] < values["rv"]

    def test_a_weaker_effect_has_a_smaller_robustness_value(self) -> None:
        strong_frame, _ = make_linear_ate(n=1500, seed=73, effect=2.0)
        weak_frame, _ = make_linear_ate(n=1500, seed=73, effect=0.2)
        strong = (
            fast_tmle(estimands=("ate",)).fit(strong_frame, outcome="Y", treatment="A").single()
        )
        weak = fast_tmle(estimands=("ate",)).fit(weak_frame, outcome="Y", treatment="A").single()
        assert (
            weak.sensitivity.robustness_value("ate")["rv"]
            < strong.sensitivity.robustness_value("ate")["rv"]
        )

    def test_poor_overlap_inflates_the_maximal_bias(self, good_overlap, poor_overlap) -> None:
        # nu^2 is the second moment of the Riesz representer, so it blows up exactly when
        # overlap fails -- the same quantity that drives the clever covariate.
        assert (
            poor_overlap.sensitivity.elements("ate").nu2
            > good_overlap.sensitivity.elements("ate").nu2
        )

    @pytest.mark.parametrize("estimand", ["ate", "ey1", "ey0", "att"])
    def test_every_linear_estimand_is_supported(self, good_overlap, estimand: str) -> None:
        elements = good_overlap.sensitivity.elements(estimand)
        assert elements.sigma2 > 0
        assert elements.nu2 > 0
        assert elements.max_bias == pytest.approx(np.sqrt(elements.sigma2 * elements.nu2))

    def test_the_doubly_robust_and_plugin_nu2_agree(self, good_overlap) -> None:
        doubly_robust = good_overlap.sensitivity.elements("ate", nu2_estimator="doubly_robust")
        plugin = good_overlap.sensitivity.elements("ate", nu2_estimator="plugin")
        # Both estimate E[alpha^2]; they differ only by sampling error.
        assert doubly_robust.nu2 == pytest.approx(plugin.nu2, rel=0.1)

    def test_a_ratio_estimand_is_refused_with_a_pointer_to_the_evalue(self) -> None:
        frame, _ = make_binary_outcome(n=800, seed=74)
        result = fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match=r"sensitivity\.evalue"):
            result.sensitivity.omitted_confounding("rr")

    def test_benchmarking_a_real_confounder_reports_its_strength(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=75)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        # W1 drives both the outcome and treatment in this process, so dropping it must
        # register as a substantial confounder on both the cf_y and cf_d scales.
        benchmark = result.sensitivity.benchmark(["W1"], estimand="ate")
        assert benchmark.cf_y > 0.05
        assert benchmark.cf_d > 0.0
        assert abs(benchmark.delta_psi) > 0.0
        assert benchmark.sigma2_short > benchmark.sigma2_long
        assert "W1" in benchmark.summary()

    def test_benchmarking_a_pure_noise_covariate_reports_almost_nothing(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=76)
        noisy = frame.assign(noise=np.random.default_rng(0).normal(size=len(frame)))
        result = fast_tmle(estimands=("ate",)).fit(noisy, outcome="Y", treatment="A").single()
        benchmark = result.sensitivity.benchmark(["noise"], estimand="ate")
        assert benchmark.cf_y < 0.02
        assert benchmark.cf_d < 0.05

    def test_a_benchmark_repeats_from_the_seed_it_reports(self) -> None:
        """A benchmark refits, so it carries the reproducibility question a refutation does.

        The result is cached on the fit and survives ``save``, so a benchmark that named no
        seed would persist a number nobody could recompute.  These call the free function,
        because ``sensitivity.benchmark`` memoises and two facade calls return one object.
        """
        frame, _ = make_linear_ate(n=500, seed=88)
        unseeded = (
            fast_tmle(estimands=("ate",), random_state=None)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert unseeded.estimator.random_state is None
        first = benchmark(unseeded, ["W1"], estimand="ate")
        assert isinstance(first.random_state, int)
        replay = benchmark(unseeded, ["W1"], estimand="ate", random_state=first.random_state)
        assert replay.psi_short == first.psi_short
        assert replay.cf_y == first.cf_y
        # The seed applies to a copy, exactly as it does for a refutation.
        assert unseeded.estimator.random_state is None

    def test_a_benchmark_of_a_seeded_fit_reports_that_fits_seed(self) -> None:
        frame, _ = make_linear_ate(n=500, seed=88)
        seeded = (
            fast_tmle(estimands=("ate",), random_state=11)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert benchmark(seeded, ["W1"], estimand="ate").random_state == 11

    def test_a_zero_seed_still_overrides_the_fit_for_a_benchmark(self) -> None:
        """``random_state=0`` is falsy, so the resolution has to read ``is None``."""
        frame, _ = make_linear_ate(n=500, seed=88)
        seeded = (
            fast_tmle(estimands=("ate",), random_state=11)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert benchmark(seeded, ["W1"], estimand="ate", random_state=0).random_state == 0

    def test_the_contour_grid_is_monotone(self, good_overlap) -> None:
        grid = nw.from_native(good_overlap.sensitivity.contour("ate", grid_size=5), eager_only=True)
        assert len(grid) == 25
        # The lower bound falls as either sensitivity parameter grows.
        at_origin = [
            row
            for row in zip(
                grid["cf_d"].to_list(), grid["cf_y"].to_list(), grid["value"].to_list(), strict=True
            )
            if row[0] == 0.0 and row[1] == 0.0
        ]
        assert at_origin[0][2] == pytest.approx(good_overlap.psi("ate"))
        assert min(grid["value"].to_list()) < good_overlap.psi("ate")


class TestEValue:
    @staticmethod
    def _with_keys(result):  # type: ignore[no-untyped-def]
        return replace(
            result,
            parameter_keys={
                "ate": ParameterKey("ate", "ate", value=1, reference=0),
                "rr": ParameterKey("rr", "rr", value=1, reference=0),
                "or": ParameterKey("or", "or", value=1, reference=0),
                "ey1": ParameterKey("ey1", "mean", value=1),
                "ey0": ParameterKey("ey0", "mean", value=0),
            },
        )

    @pytest.fixture(scope="class")
    def binary_fit(self) -> object:
        """One binary-outcome fit for the three tests that only need a ratio to read.

        Nothing asserted below turns on the sample or the seed -- each test reads a
        different field off the same kind of result -- so three fits would be three
        copies of one.
        """
        frame, _ = make_binary_outcome(n=2000, seed=77)
        return fast_tmle(estimands="all").fit(frame, outcome="Y", treatment="A").single()

    def test_a_binary_outcome_uses_the_risk_ratio_directly(self, binary_fit) -> None:
        binary_fit = self._with_keys(binary_fit)
        evalue = binary_fit.sensitivity.evalue("rr")
        assert not evalue.approximate
        assert evalue.risk_ratio == pytest.approx(binary_fit.psi("rr"))
        assert evalue.point > evalue.limit >= 1.0

    def test_the_default_prefers_the_risk_ratio(self, binary_fit) -> None:
        binary_fit = self._with_keys(binary_fit)
        assert binary_fit.sensitivity.evalue().estimand == "rr"

    def test_a_continuous_outcome_is_converted_and_flagged(self, good_overlap) -> None:
        good_overlap = self._with_keys(good_overlap)
        evalue = good_overlap.sensitivity.evalue("ate")
        assert evalue.approximate
        assert "Chinn" in evalue.note
        assert evalue.point > 1.0

    def test_the_odds_ratio_conversion_is_flagged_for_common_outcomes(self, binary_fit) -> None:
        binary_fit = self._with_keys(binary_fit)
        evalue = binary_fit.sensitivity.evalue("or")
        assert evalue.approximate
        assert "common outcomes" in evalue.note
        assert "understates" not in evalue.note
        assert evalue.risk_ratio == pytest.approx(np.sqrt(binary_fit.psi("or")))


class TestMissingnessTilt:
    @pytest.fixture(scope="class")
    def missing_fit(self) -> object:
        frame, _ = make_missing_outcome(n=1500, seed=80)
        return (
            fast_tmle(estimands=("ate", "ey1"))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            )
            .single()
        )

    def test_no_tilt_reproduces_the_reported_estimate(self, missing_fit) -> None:
        curve = nw.from_native(
            missing_fit.sensitivity.missingness([0.0], estimands=["ate"]), eager_only=True
        )
        # gamma = 0 is the MAR analysis, so the curve must pass exactly through the
        # reported point estimate. Anything else means the tilt is mis-parameterised.
        assert float(curve["psi"][0]) == pytest.approx(missing_fit.psi("ate"), rel=1e-12)
        assert bool(curve["is_mar"][0])

    def test_the_tilt_moves_the_estimate_monotonically(self, missing_fit) -> None:
        curve = nw.from_native(
            missing_fit.sensitivity.missingness([-1.0, -0.5, 0.0, 0.5, 1.0], estimands=["ey1"]),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        # The direction, not just that there is one.  `all(diff > 0) or all(diff < 0)`
        # accepted either, so a flipped gamma passed it: exactly the kind of error a
        # sign-blind check cannot expose.
        #
        # Which direction is read off the derivation rather than off a run.  The tilt is
        # Q_miss = expit(logit(Q*) + gamma), mixed in at weight (1 - pi_a), and the module
        # docstring states what positive gamma means: the unobserved outcomes were
        # systematically *higher*.  So E[Y^1] increases with gamma, strictly, wherever
        # any row is unobserved -- which the `missing_fit` fixture guarantees.
        assert np.all(np.diff(values) > 0), values

    def test_the_tipping_point_is_reported_or_absent(self, missing_fit) -> None:
        tipping = missing_fit.sensitivity.tipping_gamma("ate")
        # This process has a substantial effect, so no plausible tilt nulls it.
        assert tipping is None or abs(tipping) > 1.0

    def test_the_mnar_analyses_wait_to_be_asked_for(self, missing_fit) -> None:
        """A fit that *can* run the tilt still does not run it by default.

        ``tipping_gamma`` searches for a root by retargeting the whole tilt at every
        probe, so a bare combined report must not pay for it -- and must say which flag
        would.
        """
        default = missing_fit.sensitivity.run_all()
        for operation in ("missingness", "tipping_gamma"):
            assert default[operation].status is AssessmentStatus.UNAVAILABLE
            assert "pass include_retargets=True" in default[operation].detail

        asked = missing_fit.sensitivity.run_all(include_retargets=True)
        assert asked["missingness"].status is AssessmentStatus.COMPLETED
        assert asked["tipping_gamma"].status is AssessmentStatus.COMPLETED

    def test_the_tilt_needs_missing_outcomes(self, good_overlap) -> None:
        with pytest.raises(CapabilityError, match="not_applicable"):
            good_overlap.sensitivity.missingness()

    def test_a_ratio_estimand_is_excluded(self, missing_fit) -> None:
        with pytest.raises(ValueError, match="no tiltable estimands"):
            missing_fit.sensitivity.missingness(estimands=["rr"])


class TestTheTiltUnderObservationWeights:
    r"""The tilted curve averages over the population the weights describe.

    A weighted fit reports a weighted estimand, so the MNAR curve through it has to be
    weighted at every ``gamma`` and not only at the origin.  The tilt takes three separate
    averages -- one for a mean, one for an unconditional contrast, one for a conditional
    effect whose weight is the arm indicator times the observation weight -- and none of
    them is reachable from the ``gamma = 0`` identity alone, because at ``gamma = 0`` the
    mixture collapses to the fit's own targeted regression and the ``ate`` moves by 0.0004
    where the weighted and unweighted averages of the *tilted* regression differ by 0.065.

    So the check is the mixing formula from the module docstring of
    :mod:`cleverly.sensitivity.missingness`, written out at a nonzero ``gamma``:

    .. math::

        \bar Q^{\text{full}}_\gamma(a, W)
          = \pi_a(W) \bar Q^*(a, W)
          + (1 - \pi_a(W)) \operatorname{expit}(\operatorname{logit} \bar Q^*(a, W) + \gamma).

    ``tipping_gamma`` needs no witness of its own: it reaches the same three averages
    through :func:`~cleverly.sensitivity.missingness.missingness_tilt`, one probe at a
    time, and has no arithmetic of its own past the root search.
    """

    #: Far enough from the origin that the mixture is doing work, and inside the range an
    #: analyst reads: the module's own default grid runs to 2.
    GAMMA = 1.25

    def _longhand(self, result, name: str, gamma: float, *, weighted: bool) -> float:
        """One tilted estimand, from the displayed formula and the fit's own nuisances."""
        from cleverly.utils.bounds import expit, logit

        data = result.data
        weights = data.weights if weighted else np.ones(data.n)
        group = "mean" if name in ("ey1", "ate") else name
        draws = []
        for repeat in result.repeats:
            scaler = repeat.nuisance.scaler
            arms = repeat.nuisance.arms
            observed = repeat.nuisance.bounded_missingness(result.config.missingness_bound)
            targeted = repeat.fluctuations[group].targeted

            def full(arm: float, targeted=targeted, observed=observed, arms=arms):
                q = np.asarray(targeted.arms[arm], dtype=float)
                pi = observed[:, arms.index(arm)]
                return pi * q + (1.0 - pi) * expit(logit(q) + gamma)

            if name == "ey1":
                scaled = float(np.average(full(1.0), weights=weights))
                draws.append(scaler.unscale_level(scaled) if not scaler.is_identity else scaled)
                continue
            contrast = full(1.0) - full(0.0)
            if name == "ate":
                over = weights
            else:
                conditioning = 1.0 if name == "att" else 0.0
                over = weights * np.asarray(data.treatment == conditioning, dtype=float)
            scaled = float(np.average(contrast, weights=over))
            draws.append(scaler.unscale_difference(scaled) if not scaler.is_identity else scaled)
        return float(np.median(draws))

    @pytest.mark.parametrize("name", ["ey1", "ate", "att", "atc"])
    def test_the_tilted_estimate_averages_over_the_weighted_population(
        self, weighted_missing_fit, name: str
    ) -> None:
        curve = nw.from_native(
            weighted_missing_fit.sensitivity.missingness([self.GAMMA], estimands=[name]),
            eager_only=True,
        )
        expected = self._longhand(weighted_missing_fit, name, self.GAMMA, weighted=True)
        assert float(curve["psi"][0]) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("name", ["ey1", "ate", "att", "atc"])
    def test_the_unweighted_average_is_a_different_number(
        self, weighted_missing_fit, name: str
    ) -> None:
        """The control that makes the comparison above worth making.

        Both sides come from this module's own arithmetic.  The measured gaps are 0.348
        for ``ey1``, 0.065 for ``ate``, 0.058 for ``att`` and 0.066 for ``atc``, against a
        ``rel=1e-12`` tolerance above.
        """
        weighted = self._longhand(weighted_missing_fit, name, self.GAMMA, weighted=True)
        unweighted = self._longhand(weighted_missing_fit, name, self.GAMMA, weighted=False)
        assert abs(weighted - unweighted) > 0.02

    def test_the_curve_still_passes_through_the_weighted_estimate(
        self, weighted_missing_fit
    ) -> None:
        """``gamma = 0`` is the MAR analysis of the *weighted* fit, not of the sample."""
        for name in ("ey1", "ate", "att", "atc"):
            curve = nw.from_native(
                weighted_missing_fit.sensitivity.missingness([0.0], estimands=[name]),
                eager_only=True,
            )
            assert float(curve["psi"][0]) == pytest.approx(
                weighted_missing_fit.psi(name), rel=1e-12
            )


class TestValidation:
    def test_the_score_check_passes_and_reports(self, good_overlap) -> None:
        check = good_overlap.diagnostics.score_equations()
        assert check.passed
        assert bool(check)
        assert check.failures == ()
        assert "PASS" in check.summary()
        check.raise_if_failed()

    def test_the_score_check_can_be_made_to_fail(self, good_overlap) -> None:
        # An absurd tolerance turns the check into a failure, exercising the reporting
        # path that a real convergence problem would take.
        strict = good_overlap.diagnostics.score_equations(tolerance=1e-30)
        assert not strict.passed
        assert strict.failures
        with pytest.raises(AssertionError, match="score equation was not solved"):
            strict.raise_if_failed()

    def test_a_passing_fit_prints_no_verdict(self, good_overlap) -> None:
        """Silent on the common path, which is what keeps every transcript untouched."""
        assert good_overlap.score_verdict.passed
        assert "score check" not in good_overlap.summary()

    def test_a_failing_score_check_is_visible_in_the_summary(self, good_overlap) -> None:
        """An unlicensed interval must not be formatted like any other.

        The fit is grafted rather than found: `weak_overlap_dgp` fails this check 23 times
        in 24 but costs a sweep to reach, and what is under test is the reporting rather
        than the cause.  A score the targeting could not have left is exactly the state
        the validation contract describes arriving in practice.
        """
        fluctuation = good_overlap.repeats[0].fluctuations["mean"]
        broken = dataclasses.replace(
            good_overlap,
            repeats=(
                dataclasses.replace(
                    good_overlap.repeats[0],
                    fluctuations={
                        "mean": dataclasses.replace(
                            fluctuation, score=np.full_like(fluctuation.score, 0.5)
                        )
                    },
                ),
            ),
        )

        assert not broken.score_verdict.passed
        summary = broken.summary()
        assert "score check: FAIL" in summary
        assert "mean" in summary.split("score check: FAIL")[1]
        assert "do not describe this estimate" in summary
        # The interval is still printed -- the line says it is not licensed, it does not
        # withhold it. Predeclaring which regimes are refused outright needs the
        # targeting-and-exit study's evidence; see
        # docs/technical-reference/dr-tmle/validation-programme.md.
        assert "95% CI" in summary

    def test_nuisance_diagnostics_cover_every_model(self, good_overlap) -> None:
        diagnostics = good_overlap.diagnostics.nuisance_models()
        names = {model.name for model in diagnostics.models}
        assert names == {"propensity", "outcome"}
        assert 0.0 < diagnostics["propensity"].metrics["auc"] < 1.0
        assert diagnostics["outcome"].metrics["r2"] > 0.1
        assert "VERDICT" in diagnostics.summary()

    def test_an_almost_randomised_treatment_is_read_as_good_overlap(self) -> None:
        frame, _ = make_linear_ate(n=1500, seed=81)
        # Replace treatment with pure coin flips: nothing in W predicts it.
        rng = np.random.default_rng(0)
        randomised = frame.assign(A=rng.binomial(1, 0.5, len(frame)).astype(float))
        result = fast_tmle(estimands=("ate",)).fit(randomised, outcome="Y", treatment="A").single()
        verdict = result.diagnostics.nuisance_models().verdict()
        assert "overlap is excellent" in verdict

    def test_calibration_is_reported_per_model(self, good_overlap) -> None:
        diagnostics = good_overlap.diagnostics.nuisance_models()
        frame = nw.from_native(
            diagnostics.calibration_frame("propensity", good_overlap.data), eager_only=True
        )
        assert {"bin", "n", "mean_predicted", "mean_observed"} <= set(frame.columns)
        assert len(frame) >= 2

    def test_super_learner_weights_are_summarised(self) -> None:
        frame, _ = make_nonlinear_ate(n=500, seed=82)
        # A one-model SuperLearner isolates the reporting path without paying for flexible
        # candidates that add no coverage here.
        result = (
            TMLE(
                outcome_learner=SuperLearner(
                    [sklearn.linear_model.LinearRegression()],
                    n_folds=3,
                ),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                learner_folds=3,
                estimands=("ate",),
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        weights = result.diagnostics.nuisance_models()["outcome"].learner_weights
        assert weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_combined_report_renders(self, good_overlap) -> None:
        assert "score_equations" in good_overlap.diagnostics.run_all().summary()
        assert "nuisance_models" in good_overlap.diagnostics.run_all().summary()


class TestRefutation:
    @pytest.fixture(scope="class")
    def refutation(self) -> object:
        frame, _ = make_linear_ate(n=700, seed=83)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        return result.diagnostics.refute(n_replicates=3, random_state=0)

    def test_all_default_tests_behave(self, refutation) -> None:
        assert refutation.passed
        assert {test.name for test in refutation.tests} == {
            "placebo",
            "random_common_cause",
            "subset",
        }

    def test_a_placebo_treatment_shows_no_effect(self, refutation) -> None:
        placebo = refutation["placebo"]
        # Permuting treatment destroys the effect while preserving its marginal.
        assert abs(placebo.mean) < 0.2 * abs(placebo.original)

    def test_an_irrelevant_covariate_does_not_move_the_estimate(self, refutation) -> None:
        noise = refutation["random_common_cause"]
        assert noise.mean == pytest.approx(noise.original, rel=0.05)

    def test_the_results_tabulate(self, refutation) -> None:
        frame = nw.from_native(refutation.to_frame(), eager_only=True)
        assert len(frame) == 3
        assert "VERDICT" in refutation.summary()

    def test_a_negative_control_outcome_shows_no_effect(self) -> None:
        frame, _ = make_linear_ate(n=700, seed=84)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        # An outcome built from the covariates alone, with no treatment component.
        rng = np.random.default_rng(0)
        control = frame["W1"].to_numpy() * 0.5 + rng.normal(size=len(frame))
        outcome = result.diagnostics.refute(
            tests=["negative_control_outcome"],
            negative_control_outcome=control,
            random_state=0,
        )
        assert outcome.passed

    def test_the_negative_control_test_needs_an_outcome(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=85)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match="needs an outcome array"):
            result.diagnostics.refute(tests=["negative_control_outcome"])

    def test_an_unknown_test_is_refused(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="unknown refutation test"):
            good_overlap.diagnostics.refute(tests=["magic"])

    def test_nonzero_measurement_error_changes_real_refits(self) -> None:
        frame, _ = make_linear_ate(n=120, seed=91)
        result = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        rule = EmpiricalInclusionRule(alpha=0.5, minimum_draws=4)

        def run(noise: float):
            return result.diagnostics.refute(
                tests=("bootstrap_measurement_error",),
                n_replicates=4,
                bootstrap_measurement_error=BootstrapMeasurementError(
                    ("W1",), numeric_noise=RelativeGaussianNoise(noise)
                ),
                measurement_error_rule=rule,
                random_state=17,
            )["bootstrap_measurement_error"]

        active = run(0.5)
        zero = run(0.0)
        child_seeds = tuple(
            int(sequence.generate_state(1)[0]) for sequence in np.random.SeedSequence(17).spawn(4)
        )
        assert active.child_seeds == zero.child_seeds == child_seeds
        assert active.failures == zero.failures == ()
        assert active.family == zero.family == "gaussian"
        assert len(active.records) == len(zero.records) == 4
        active_estimates = np.asarray([record.estimate for record in active.records])
        zero_estimates = np.asarray([record.estimate for record in zero.records])
        assert np.max(np.abs(active_estimates - zero_estimates)) > 1e-3


class TestARefutationInheritsTheFitsSeed:
    """``refute()`` used to draw from OS entropy whenever the caller named no seed.

    It was the only public stochastic operation in the package that did.  Every one of
    these tests calls the module function rather than ``result.diagnostics.refute``,
    because the facade memoises through ``_cached``: two facade calls with the same
    keywords return the *same object*, so an equality assertion written through the facade
    compares one object with itself and holds whatever the function does.
    """

    @staticmethod
    def _placebo(result: object, **kwargs: object) -> tuple[float, ...]:
        report = refute(result, n_replicates=1, tests=["placebo"], **kwargs)  # type: ignore[arg-type]
        return report["placebo"].values

    @pytest.fixture(scope="class")
    def seeded(self) -> object:
        frame, _ = make_linear_ate(n=500, seed=86)
        return fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()

    def test_two_refutations_of_one_seeded_fit_agree(self, seeded) -> None:
        assert self._placebo(seeded) == self._placebo(seeded)

    def test_the_inherited_seed_is_the_fit_own(self, seeded) -> None:
        """Not merely repeatable: repeatable *from the seed the fit records*."""
        assert seeded.estimator.random_state == 0
        assert self._placebo(seeded) == self._placebo(
            seeded, random_state=seeded.estimator.random_state
        )

    def test_an_explicit_seed_overrides_the_fit(self, seeded) -> None:
        assert self._placebo(seeded, random_state=3) != self._placebo(seeded)

    def test_a_zero_seed_still_overrides_the_fit(self) -> None:
        """``random_state=0`` is falsy, so the fallback has to read ``is None``.

        The other tests in this class hold under a truthiness test as well, because the
        fit they use carries seed 0 itself.  This one fails under it.
        """
        frame, _ = make_linear_ate(n=500, seed=86)
        fit = (
            fast_tmle(estimands=("ate",), random_state=21)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert self._placebo(fit, random_state=0) != self._placebo(fit)

    def test_an_unseeded_fit_still_draws_from_entropy(self) -> None:
        """The guarantee is conditional on the fit carrying a seed, and says so."""
        unseeded = self._unseeded_fit()
        assert unseeded.estimator.random_state is None
        assert self._placebo(unseeded) != self._placebo(unseeded)

    @staticmethod
    def _unseeded_fit() -> object:
        frame, _ = make_linear_ate(n=500, seed=87)
        return (
            fast_tmle(estimands=("ate",), random_state=None)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_the_report_records_the_seed_it_resolved(self, seeded) -> None:
        """Every branch of the resolution, so the field cannot drift from the generator."""
        assert refute(seeded, n_replicates=1, tests=["placebo"]).random_state == 0
        assert refute(seeded, n_replicates=1, tests=["placebo"], random_state=3).random_state == 3

    def test_replaying_a_recorded_seed_repeats_a_seeded_fits_report(self, seeded) -> None:
        """What the recorded seed is for: the report says how to obtain itself again."""
        report = refute(seeded, n_replicates=1, tests=["placebo"])
        replay = refute(seeded, n_replicates=1, tests=["placebo"], random_state=report.random_state)
        assert replay["placebo"].values == report["placebo"].values

    def test_replaying_a_recorded_seed_repeats_an_unseeded_fits_report(self) -> None:
        """The case the recorded seed exists for, and the one that needed the refit seeded.

        A perturbation is half of a refutation.  ``refit`` re-learns the nuisances, and an
        estimator carrying no ``random_state`` redraws its folds every time, so seeding the
        draws alone left this report unrepeatable.  The report is cached on the result and
        survives ``save``, so an unrepeatable one is a number nobody can check.
        """
        unseeded = self._unseeded_fit()
        report = refute(unseeded, n_replicates=1, tests=["placebo"])
        assert isinstance(report.random_state, int)
        replay = refute(
            unseeded, n_replicates=1, tests=["placebo"], random_state=report.random_state
        )
        assert replay["placebo"].values == report["placebo"].values

    def test_a_seeded_refit_leaves_the_estimator_alone(self) -> None:
        """The seed applies to a copy.  A refutation may not silently seed the fit itself."""
        unseeded = self._unseeded_fit()
        refute(unseeded, n_replicates=1, tests=["placebo"])
        assert unseeded.estimator.random_state is None

    def test_a_seeded_refit_matches_a_fit_that_carried_that_seed(self) -> None:
        """``refit(random_state=s)`` is the fit the estimator would have run at ``s``."""
        frame, _ = make_linear_ate(n=500, seed=87)
        unseeded = self._unseeded_fit()
        genuine = (
            fast_tmle(estimands=("ate",), random_state=4242)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        borrowed = unseeded.estimator.refit(
            unseeded.data,
            intermediate_value=unseeded.intermediate_value,
            random_state=4242,
        )
        assert (
            borrowed["ate"].psi
            == genuine.estimator.refit(genuine.data, intermediate_value=genuine.intermediate_value)[
                "ate"
            ].psi
        )


class TestTheDefaultEstimandOfTheOmittedVariableBound:
    """These analyses default to ``estimand="ate"``, which not every fit reports."""

    def test_a_sole_reported_parameter_is_supplied_without_being_named(self) -> None:
        frame, _ = make_linear_ate(n=800, seed=73)
        fit = fast_tmle(estimands=("ey1",)).fit(frame, outcome="Y", treatment="A").single()
        assert "ate" not in fit.estimates
        assert fit.sensitivity.robustness_value() == fit.sensitivity.robustness_value("ey1")

    def test_a_choice_between_parameters_is_the_callers_to_make(self) -> None:
        """``ey1`` and ``ey0`` are both linear and both reported, and they are different
        questions.

        Filling the gap by position would answer about the treated arm's counterfactual
        mean for a caller who asked nothing about arms, and the returned bound names no
        estimand for them to notice with.
        """
        frame, _ = make_linear_ate(n=800, seed=73)
        fit = fast_tmle(estimands=("ey1", "ey0")).fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match="was not requested in this fit"):
            fit.sensitivity.robustness_value()
        assert fit.sensitivity.robustness_value("ey0")["rv"] > 0.0


class TestCombinedSensitivityReport:
    def test_the_report_gathers_what_it_can(self, good_overlap) -> None:
        report = good_overlap.sensitivity.run_all().summary()
        assert "omitted_confounding" in report
        assert "robustness_value" in report
        assert "evalue" in report


class TestTheMechanismDenominatorsAreDiagnosed:
    r"""``P(Delta = 1 | A, W)`` divides the clever covariate; it needs the same scrutiny as ``g``.

    Nothing used to report it.  ``positivity()`` described only the propensity,
    ``truncation_curve()`` swept only ``g_bounds``, the positivity warning inspected only
    ``g``, and ``summary()`` printed only ``g_bounds`` and ``q_bounds`` -- so a fit could
    be resting on a handful of rows that were very unlikely to have been observed at all,
    with immaculate propensity overlap and nothing anywhere saying so.
    """

    @pytest.fixture(scope="class")
    def strained(self) -> object:
        # strength=2 sharpens the mechanism on W1: the first percentile of pi is ~0.13.
        frame, _ = make_missing_outcome(n=2000, seed=91, strength=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            return (
                fast_tmle(estimands=("ate",))
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2", "W3"],
                    delta="Delta",
                )
                .single()
            )

    def test_the_report_carries_the_mechanism(self, strained) -> None:
        report = strained.diagnostics.support()
        assert "P(Delta=1|A,W)" in report.mechanisms
        stats = report.mechanisms["P(Delta=1|A,W)"]
        assert 0.0 < stats["min"] < stats["q01"] < stats["q05"] < stats["median"] < 1.0
        assert "P(Delta=1|A,W)" in report.summary()

    def test_the_mechanism_explains_leverage_the_propensity_does_not(self, strained) -> None:
        """The case the diagnostic exists for, asserted as a whole.

        On this fit the propensity overlap is immaculate -- nothing truncated, effective
        sample size near 90% of nominal in both arms -- and yet the largest clever
        covariate is in the hundreds.  Every bit of that comes from ``pi`` reaching
        0.04, an order of magnitude below the smallest propensity.  Before this the
        report had nothing to say about it: a reader saw a three-figure covariate next
        to a clean bill of health and no way to connect them.
        """
        # Measured across seeds 91-95 at this n and strength: pi bottoms out at
        # 0.019-0.039 against a smallest propensity of 0.105-0.165, the largest clever
        # covariate runs 53-195, the propensity ESS stays above 0.88 and nothing is
        # truncated. The windows below are set to hold across that whole range rather
        # than to the one seed the fixture happens to use.
        report = strained.diagnostics.support()
        mechanism = report.mechanisms["P(Delta=1|A,W)"]
        assert report.truncated["fraction"] == 0.0
        assert min(ess["ratio"] for ess in report.effective_sample_size.values()) > 0.88
        assert report.clever_covariate_max["mean"] > 40.0
        # The mechanism is where the leverage lives, and its ESS says so on the same
        # scale the propensity's is reported on.
        assert mechanism["min"] < 0.5 * float(np.min(strained.nuisance.propensity.values))
        assert mechanism["ess_ratio"] < 0.90

    def test_clipping_the_mechanism_reaches_the_verdict(self) -> None:
        """The verdict's truncation branch, forced deterministically.

        Driving it through the data instead -- a process sharp enough for the mechanism's
        effective sample size to fall past 0.6 -- lands at 0.58-0.65 depending on the
        seed, because the statistic is governed by the extreme tail of a normal
        covariate. That is a coin flip dressed as a test, so the bound is raised until it
        bites instead, which is deterministic and exercises the same verdict.
        """
        frame, _ = make_missing_outcome(n=1500, seed=94, strength=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            result = (
                fast_tmle(estimands=("ate",), nuisance_bound=0.35)
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2", "W3"],
                    delta="Delta",
                )
                .single()
            )
        verdict = result.diagnostics.support().verdict()
        assert "P(Delta=1|A,W) strains the estimate" in verdict
        assert "truncation_curve(mechanism=True)" in verdict

    def test_a_low_mechanism_ess_reaches_the_verdict(self, strained) -> None:
        # The other branch, checked on the rule rather than through a process: what a
        # data-driven version would be measuring is the tail of a normal, not the rule.
        report = strained.diagnostics.support()
        assert "adequate" in report.verdict()
        degenerate = dataclasses.replace(
            report,
            mechanisms={
                "P(Delta=1|A,W)": {**report.mechanisms["P(Delta=1|A,W)"], "ess_ratio": 0.4}
            },
        )
        assert "P(Delta=1|A,W) strains the estimate" in degenerate.verdict()

    def test_a_fit_without_missingness_reports_no_mechanism(self, good_overlap) -> None:
        report = good_overlap.diagnostics.support()
        assert report.mechanisms == {}
        assert "P(Delta=1|A,W)" not in report.summary()

    def test_the_bound_appears_in_the_fit_summary(self, strained, good_overlap) -> None:
        # Traceability: a reported number must be traceable to every bound that shaped
        # it, not just the one with a familiar name.
        assert "P(Delta=1|A,W) truncated to" in strained.summary()
        assert "truncated to [0.01, 1]" not in good_overlap.summary()

    def test_the_curve_sweeps_the_mechanism_bound(self, strained) -> None:
        curve = nw.from_native(
            strained.diagnostics.truncation_curve(
                [0.01, 0.1, 0.25], estimands=["ate"], mechanism=True
            ),
            eager_only=True,
        )
        truncated = np.array(curve["truncated_fraction"].to_list())
        values = np.array(curve["psi"].to_list())
        # A tighter bound on pi clips more rows and moves the estimate, exactly as a
        # tighter bound on g does -- which is the whole reason it deserves a curve.
        assert np.all(np.diff(truncated) > 0)
        assert float(values.max() - values.min()) > 1e-3

    def test_the_mechanism_curve_is_flat_when_the_bound_never_binds(self, strained) -> None:
        # Below the smallest fitted pi nothing is clipped, so the estimate cannot move.
        smallest = float(np.min(strained.nuisance.missingness))
        grid = [smallest / 8.0, smallest / 4.0, smallest / 2.0]
        curve = nw.from_native(
            strained.diagnostics.truncation_curve(grid, estimands=["ate"], mechanism=True),
            eager_only=True,
        )
        values = np.array(curve["psi"].to_list())
        assert float(values.max() - values.min()) < 1e-9

    def test_sweeping_the_mechanism_needs_a_mechanism(self, good_overlap) -> None:
        with pytest.raises(ValueError, match="needs a fit with missing outcomes"):
            good_overlap.diagnostics.truncation_curve(mechanism=True)

    def test_a_degenerate_mechanism_warns(self) -> None:
        """The warning half: a fit that leans on the bound has to say so at fit time."""
        frame, _ = make_missing_outcome(n=1500, seed=92, strength=2.0)
        with pytest.warns(PositivityWarning, match=r"P\(Delta = 1 \| A, W\)"):
            fast_tmle(estimands=("ate",), nuisance_bound=0.35).fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            ).single()

    def test_an_untroubled_mechanism_does_not_warn(self) -> None:
        frame, _ = make_missing_outcome(n=1500, seed=93)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PositivityWarning)
            fast_tmle(estimands=("ate",)).fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            ).single()
