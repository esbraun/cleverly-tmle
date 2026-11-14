"""Under a declared stopping rule the interval adds the remaining score to its half-width.

Fix-8 plan section 2.1 (X76 part (a)).  A walk under ``stopping="scaled_var"`` or
``"sample_size"`` stops before the score equation is solved, so the estimate's error is
``se (Z - B)`` with ``B = r_n / se`` and ``r_n`` the mean of the influence curve.  Nothing
bounds the plain interval's coverage under an arbitrary dependence of ``B`` on ``Z``; the
interval ``psi +/- (z se + |r_n|)`` covers whenever ``|Z| <= z``.  The tests below check:

* the arithmetic of the widened interval and of the test that inverts it, on the inference
  scale, for an arm mean, a difference, a ratio, a t reference, a doubly-robust fit and a
  longitudinal parameter;
* a nonzero witness: a one-step ``scaled_var`` walk leaves a residual above ``1e-3`` SE, and
  the interval is wider than the plain Wald interval by exactly ``2 |r_n|``;
* that ``plugin_interval`` is the widened interval too, so a study row reads what ``ci`` reads;
* that a fit under the default rule is not widened, and that ``ci == plugin_interval``
  exactly there;
* that every derived estimate inherits the widening, with its own curve's mean;
* the ``sample_size`` threshold in the outcome's units, where the residual exceeds an SE;
* that a simultaneous band widens each component by its own residual;
* that the widening adds to a fractional Student t half-width, and that `ci` does not yet
  read the small-sample reference of `small-sample-wald`;
* deliberate mutations of the source that each of these must catch.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import Transform, simultaneous_bands
from cleverly.inference.delta import wald_ci, wald_statistic
from cleverly.inference.influence import ParameterEstimate
from tests.conftest import linear_drtmle
from tests.studies import _option_stopping_laws as laws
from tests.studies import canonical_tmle
from tests.unit import test_stopping_rule_engines as engines
from tests.unit._source_mutation import mutate

influence_module = importlib.import_module("cleverly.inference.influence")
results_module = importlib.import_module("cleverly.inference.results")
tmle_module = importlib.import_module("cleverly.estimators.tmle")

Z = float(stats.norm.ppf(0.975))
#: A tolerance for identities computed two ways in floating point.
EXACT = 1e-12


# ------------------------------------------------------------------------------ the fits


@functools.cache
def l1_sample() -> pd.DataFrame:
    """A fixed sample of the ``canonical-tmle`` binary law at the studies' size."""
    return laws.draw_l1("binary", laws.N, 20261112)[0]


def one_step_binary(rule: str = "scaled_var") -> Any:
    """A one-step walk on :func:`l1_sample` under ``rule``, with every binary estimand."""
    return laws.fit_point(
        "binary",
        l1_sample(),
        stopping=rule,
        targeting="one_step",
        estimands=("ey1", "ey0", "ate", "rr", "or"),
    )


@functools.cache
def ruled() -> Any:
    return one_step_binary()


@functools.cache
def relative() -> Any:
    return laws.fit_point("binary", l1_sample(), stopping="relative", targeting="one_step")


@functools.cache
def ruled_drtmle(rule: str = "scaled_var") -> Any:
    estimator = DRTMLE(
        **linear_drtmle(
            stopping=rule,
            cross_fit=False,
            estimands=("ate", "ey1", "ey0"),
            targeting="one_step",
            step_size=0.01,
        )
    )
    return estimator.fit(engines._lin(), outcome="Y", treatment="A").single()


@functools.cache
def hazard(rule: str) -> Any:
    return laws.fit_l4(laws.draw_l4(laws.N, 20261113)[0], stopping=rule)


@functools.cache
def end_of_study(rule: str) -> Any:
    return laws.fit_l3(laws.draw_l3(laws.N, 20261114)[0], stopping=rule)


def logit(value: float) -> float:
    """SciPy's logit, so the tests do not read the package's map but round as it does."""
    return float(special.logit(value))


def expit(value: float) -> float:
    return float(special.expit(value))


def interval_slope(estimate: ParameterEstimate) -> float:
    """The slope from the reported scale to the interval's scale at ``psi``.

    ``1 / (psi (1 - psi))`` for a bounded binary mean, whose ``ci`` is formed on the logit
    scale, and 1 for every other estimate, whose inference scale already carries the curve.
    """
    if estimate.interval_scale == "range_logit":
        return 1.0 / (estimate.psi * (1.0 - estimate.psi))
    return 1.0


def half_widths(estimate: ParameterEstimate, interval: tuple[float, float]) -> tuple[float, float]:
    """The distances from the estimate to each end of ``interval`` on the interval's scale."""
    low, high = interval
    if estimate.interval_scale == "range_logit":
        center = logit(estimate.psi)
        return center - logit(low), logit(high) - center
    if estimate.scale == "ratio":
        low, high = np.log(low), np.log(high)
    elif estimate.transform is not None:
        low, high = sorted(
            (float(estimate.transform.forward(low)), float(estimate.transform.forward(high)))
        )
    center = estimate.inference_value
    return center - low, high - center


def critical(estimate: ParameterEstimate) -> float:
    if estimate.reference_df is None:
        return Z
    return float(stats.t.ppf(0.975, estimate.reference_df))


def plain(estimate: ParameterEstimate) -> tuple[float, float]:
    """The plain Wald interval of ``estimate``: its interval with the widening switched off."""
    return replace(estimate, rule_widened=False).ci


def assert_widened(estimate: ParameterEstimate) -> None:
    """``ci`` is ``inference_value +/- (critical * se + |score|)`` on the inference scale.

    On a bounded binary mean the interval is formed on the logit scale, and both terms are
    carried there by the slope at ``psi``: ``(z se + |score|) / (psi (1 - psi))``.
    ``plugin_interval`` is the same interval, so a study row that reads it reads ``ci``.
    """
    assert estimate.plugin_interval == estimate.ci
    expected = interval_slope(estimate) * (
        critical(estimate) * estimate.std_error + abs(float(np.mean(estimate.influence_curve)))
    )
    below, above = half_widths(estimate, estimate.ci)
    assert below == pytest.approx(expected, rel=0.0, abs=EXACT)
    assert above == pytest.approx(expected, rel=0.0, abs=EXACT)


# ------------------------------------------------------------------------------ arithmetic


class TestTheWidenedInterval:
    @pytest.mark.parametrize("name", ["ey1", "ate", "rr", "or"])
    def test_a_one_step_walk_widens_by_its_residual(self, name: str) -> None:
        estimate = ruled()[name]
        assert estimate.rule_widened
        assert_widened(estimate)

    def test_a_t_reference_widens_its_t_half_width(self) -> None:
        estimate = replace(ruled()["ate"], reference_df=8)
        assert_widened(estimate)
        plain_below, _ = half_widths(estimate, plain(estimate))
        assert plain_below == pytest.approx(
            float(stats.t.ppf(0.975, 8)) * estimate.std_error, abs=EXACT
        )

    @pytest.mark.parametrize("name", ["ate", "ey1"])
    def test_a_doubly_robust_fit_widens(self, name: str) -> None:
        estimate = ruled_drtmle()[name]
        assert estimate.rule_widened
        assert abs(estimate.score) / estimate.std_error > 1e-4
        assert_widened(estimate)

    @pytest.mark.parametrize("name", [laws.L4_CONTRAST, laws.L4_RISK])
    def test_a_hazard_parameter_widens(self, name: str) -> None:
        estimate = hazard("scaled_var")[name]
        assert estimate.rule_widened
        assert abs(estimate.score) / estimate.std_error > 1e-4
        assert_widened(estimate)

    def test_an_end_of_study_parameter_carries_the_flag(self) -> None:
        estimate = end_of_study("scaled_var")[laws.L3_CONTRAST]
        assert estimate.rule_widened
        assert_widened(estimate)

    @pytest.mark.parametrize("name", ["ey1", "ate", "rr"])
    def test_the_test_rejects_exactly_when_the_null_is_outside_the_interval(
        self, name: str
    ) -> None:
        estimate = ruled()[name]
        low, high = estimate.ci
        nulls = [low - 1e-3, low + 1e-9, estimate.psi, high - 1e-9, high + 1e-3]
        if estimate.scale == "ratio":
            nulls = [max(value, 1e-6) for value in nulls]
        for null in nulls:
            outside = not low <= null <= high
            assert (estimate.wald_test(null=null).pvalue <= 0.05) == outside, null

    def test_the_p_value_inverts_the_widened_interval(self) -> None:
        estimate = ruled()["ate"]
        low, _ = estimate.ci
        assert estimate.wald_test(null=low).pvalue == pytest.approx(0.05, abs=1e-9)
        distance = abs(estimate.psi) / estimate.std_error - abs(estimate.score) / estimate.std_error
        expected = 2.0 * stats.norm.sf(max(0.0, distance))
        assert estimate.pvalue == pytest.approx(expected, rel=1e-12)
        # The statistic stays the plain Wald statistic.
        assert estimate.wald_test().statistic == pytest.approx(
            estimate.psi / estimate.std_error, rel=1e-12
        )


class TestTheSmallSampleReference:
    """The widening composes with the small-sample intervals of ``small-sample-wald``.

    Fix-3 PR 2 forms the interval of an unclustered bounded binary mean on the logit scale
    with the normal reference. The margin is carried there by the slope that carries the
    standard error, so the half-width on the logit scale is ``(z se + |r_n|) f'(psi)`` with
    ``f'(psi) = 1 / (psi (1 - psi))``. Every other estimate keeps ``crit * se + |r_n|`` on its
    inference scale. A continuous outcome keeps the normal reference until PR 3, although
    :attr:`ParameterEstimate.effective_df` is computed; the fractional t half-width it will take
    is checked at the helper.
    """

    def test_a_bounded_mean_adds_the_margin_on_the_logit_scale(self) -> None:
        estimate = ruled()["ey1"]
        assert estimate.interval_scale == "range_logit"
        assert estimate.rule_widened
        slope = interval_slope(estimate)
        margin = abs(float(np.mean(estimate.influence_curve)))
        assert margin / estimate.std_error > 1e-3
        center = logit(estimate.psi)
        low, high = wald_ci(
            center, estimate.std_error * slope, estimate.alpha, margin=margin * slope
        )
        assert estimate.ci == pytest.approx((expit(low), expit(high)), rel=0.0, abs=EXACT)
        below, above = half_widths(estimate, estimate.ci)
        expected = (Z * estimate.std_error + margin) * slope
        assert below == pytest.approx(expected, rel=0.0, abs=EXACT)
        assert above == pytest.approx(expected, rel=0.0, abs=EXACT)
        # Nonzero witness: the slope is at least 4, so a margin added unscaled, or added on the
        # level scale after the map, moves an end by more than the tolerance.
        assert margin * (slope - 1.0) > 1e3 * EXACT
        unscaled = wald_ci(center, estimate.std_error * slope, estimate.alpha, margin=margin)
        assert abs(expit(unscaled[1]) - estimate.ci[1]) > 1e3 * EXACT
        _, plain_high = plain(estimate)
        assert abs((plain_high + margin) - estimate.ci[1]) > 1e3 * EXACT
        # The test inverts this interval: the p-value reaches alpha at each end.
        for end in estimate.ci:
            assert estimate.wald_test(null=end).pvalue == pytest.approx(estimate.alpha, abs=1e-9)

    def test_the_mutation_that_adds_the_margin_unscaled_on_the_logit_scale_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        estimate = ruled()["ey1"]
        mutate(
            monkeypatch,
            influence_module,
            influence_module.ParameterEstimate,
            "_wald_interval",
            [("margin=margin * slope,", "margin=margin,")],
        )
        below, _ = half_widths(estimate, estimate.ci)
        expected = (Z * estimate.std_error + abs(estimate.score)) * interval_slope(estimate)
        assert abs(below - expected) > 1e3 * EXACT

    def test_a_clustered_bounded_mean_keeps_the_level_scale_margin(self) -> None:
        estimate = ruled()["ey1"]
        clustered = replace(estimate, n_clusters=estimate.n // 2)
        assert clustered.interval_scale == "identity"
        assert_widened(clustered)

    @pytest.mark.parametrize("df", [5.0, 17.5])
    def test_a_fractional_t_reference_adds_the_margin_to_its_half_width(self, df: float) -> None:
        estimate = ruled()["ate"]
        margin = estimate.residual_margin
        assert margin > 0.0
        center, se = estimate.inference_value, estimate.std_error
        low, high = wald_ci(center, se, estimate.alpha, df=df, margin=margin)
        expected = float(stats.t.ppf(1.0 - estimate.alpha / 2.0, df)) * se + margin
        assert center - low == pytest.approx(expected, rel=0.0, abs=EXACT)
        assert high - center == pytest.approx(expected, rel=0.0, abs=EXACT)
        # The test that inverts it reaches alpha exactly at the widened interval's end.
        _, pvalue = wald_statistic(high, center, se, df=df, margin=margin)
        assert pvalue == pytest.approx(estimate.alpha, abs=1e-12)

    def test_an_infinite_df_is_the_normal_widened_interval(self) -> None:
        estimate = ruled()["ate"]
        args = (estimate.inference_value, estimate.std_error, estimate.alpha)
        margin = estimate.residual_margin
        assert wald_ci(*args, df=float("inf"), margin=margin) == wald_ci(*args, margin=margin)

    def test_a_gaussian_widened_estimate_keeps_the_normal_half_width(self) -> None:
        estimate = replace(ruled()["ate"], outcome_family="gaussian")
        assert estimate.rule_widened
        assert estimate.effective_df is not None
        assert estimate.reference_df is None
        assert_widened(estimate)


class TestAnUndefinedStatistic:
    """A ``nan`` value or null gives a ``nan`` p-value, with or without a margin."""

    @pytest.mark.parametrize("margin", [0.0, 0.3])
    @pytest.mark.parametrize(("value", "null"), [(float("nan"), 0.0), (1.0, float("nan"))])
    def test_the_p_value_is_nan(self, value: float, null: float, margin: float) -> None:
        statistic, pvalue = wald_statistic(value, null, 1.0, margin=margin)
        assert np.isnan(statistic)
        assert np.isnan(pvalue)

    def test_a_wald_test_against_a_nan_null_is_undefined(self) -> None:
        for estimate in (ruled()["ate"], relative()["ate"]):
            assert np.isnan(estimate.wald_test(null=float("nan")).pvalue)

    def test_the_mutation_that_clips_with_the_builtin_max_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delta = importlib.import_module("cleverly.inference.delta")
        mutate(
            monkeypatch,
            delta,
            delta,
            "wald_statistic",
            [
                (
                    "distance = float(np.maximum(0.0, abs(z) - margin / std_error))",
                    "distance = max(0.0, abs(z) - margin / std_error)",
                )
            ],
        )
        assert not np.isnan(delta.wald_statistic(float("nan"), 0.0, 1.0)[1])


class TestTheWitness:
    """The residual is not at roundoff, so the identities above see the widening."""

    def test_the_walk_leaves_a_residual_and_the_interval_is_wider_by_twice_it(self) -> None:
        # ey1 is a bounded binary mean, so the margin is added on the logit scale.
        estimate = ruled()["ey1"]
        assert estimate.interval_scale == "range_logit"
        residual = abs(float(np.mean(estimate.influence_curve)))
        assert residual / estimate.std_error > 1e-3
        widened = half_widths(estimate, estimate.ci)
        unwidened = half_widths(estimate, plain(estimate))
        expected = residual * interval_slope(estimate)
        for side in (0, 1):
            assert widened[side] - unwidened[side] == pytest.approx(expected, rel=1e-9)
        assert estimate.residual_margin == pytest.approx(residual, rel=1e-12)

    def test_the_mutation_that_drops_the_widening_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        estimate = ruled()["ey1"]
        monkeypatch.setattr(ParameterEstimate, "_residual_margin", lambda self, widen: 0.0)
        widened = half_widths(estimate, estimate.ci)
        unwidened = half_widths(estimate, plain(estimate))
        expected = abs(estimate.score) * interval_slope(estimate)
        assert widened[0] - unwidened[0] != pytest.approx(expected, rel=1e-9)

    def test_the_mutation_that_widens_on_the_reported_scale_fails_the_ratio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        estimate = ruled()["rr"]
        assert abs(estimate.score) / estimate.std_error > 1e-4
        mutate(
            monkeypatch,
            influence_module,
            influence_module.ParameterEstimate,
            "_wald_interval",
            [
                (
                    "low, high = wald_ci(center, self._plugin_std_error(), self.alpha, df=df, "
                    "margin=margin)\n            return (float(np.exp(low)), float(np.exp(high)))",
                    "low, high = wald_ci(center, self._plugin_std_error(), self.alpha, df=df)\n"
                    "            return (float(np.exp(low)) - margin, float(np.exp(high)) + margin)",
                )
            ],
        )
        below, above = half_widths(estimate, estimate.ci)
        expected = Z * estimate.std_error + abs(estimate.score)
        assert abs(below - expected) > 1e-9 or abs(above - expected) > 1e-9


def plain_wald(estimate: ParameterEstimate) -> tuple[float, float]:
    """The plain Wald interval, built from :func:`wald_ci` without the estimate's own code.

    Mapped to the reported scale as ``ParameterEstimate`` maps it, so a default-rule fit's
    ``ci`` equals it bit for bit.
    """
    if estimate.interval_scale == "range_logit":
        low, high = wald_ci(
            logit(estimate.psi), estimate.std_error * interval_slope(estimate), estimate.alpha
        )
        return (expit(low), expit(high))
    low, high = wald_ci(
        estimate.inference_value,
        estimate.std_error,
        estimate.alpha,
        df=estimate.reference_df,
    )
    if estimate.transform is not None:
        mapped = sorted(
            (float(estimate.transform.inverse(low)), float(estimate.transform.inverse(high)))
        )
        return (mapped[0], mapped[1])
    if estimate.scale == "ratio":
        return (float(np.exp(low)), float(np.exp(high)))
    return (low, high)


@functools.cache
def relative_drtmle() -> Any:
    estimator = DRTMLE(
        **linear_drtmle(cross_fit=False, estimands=("ate", "ey1", "ey0"), targeting="one_step")
    )
    return estimator.fit(engines._lin(), outcome="Y", treatment="A").single()


def default_rule_fits() -> dict[str, Any]:
    return {
        "tmle": relative(),
        "drtmle": relative_drtmle(),
        "hazard": hazard("relative"),
        "end_of_study": end_of_study("relative"),
    }


def assert_plain_estimate(estimate: ParameterEstimate) -> None:
    """No flag, no margin, and ``ci`` the plain interval exactly."""
    assert not estimate.rule_widened, estimate.name
    assert estimate.residual_margin == 0.0, estimate.name
    assert estimate.ci == plain_wald(estimate), estimate.name


def assert_plain(result: Any) -> None:
    """Every estimate of a default-rule fit is plain (:func:`assert_plain_estimate`)."""
    for estimate in result.estimates.values():
        assert_plain_estimate(estimate)


class TestTheDefaultRule:
    @pytest.mark.parametrize("fit", ["tmle", "drtmle", "hazard", "end_of_study"])
    def test_a_default_rule_fit_reports_the_plain_interval_exactly(self, fit: str) -> None:
        assert_plain(default_rule_fits()[fit])

    def test_the_default_rule_residual_is_not_zero(self) -> None:
        # The witness that makes the control able to fail: a margin of |score| would move
        # the interval, because the relative walk stops at tol, not at an exact zero.
        scores = [
            abs(estimate.score)
            for result in default_rule_fits().values()
            for estimate in result.estimates.values()
        ]
        assert max(scores) > 0.0

    def test_the_mutation_that_widens_every_interval_fails_the_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ParameterEstimate, "_residual_margin", lambda self, widen: abs(self.score)
        )
        for result in default_rule_fits().values():
            with pytest.raises(AssertionError):
                assert_plain(result)

    def test_the_mutation_that_flags_every_fit_fails_the_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            tmle_module,
            tmle_module.TMLE,
            "_estimates_for",
            [('rule_widened=self.stopping != "relative"', "rule_widened=True")],
        )
        result = laws.fit_point("binary", l1_sample(), stopping="relative", targeting="one_step")
        with pytest.raises(AssertionError):
            assert_plain(result)


# ------------------------------------------------------------------------------ derived


class TestDerivedEstimates:
    @staticmethod
    def contrasts() -> dict[str, ParameterEstimate]:
        result = ruled()
        return {
            "difference": result.contrast(lambda p: p[0] - p[1], ["ey1", "ey0"]),
            "log": result.contrast(
                lambda p: p[0] - p[1], ["ey1", "ey0"], transform=Transform.log()
            ),
            "ratio": result.contrast(lambda p: p[0] / p[1], ["ey1", "ey0"], scale="ratio"),
        }

    @pytest.mark.parametrize("kind", ["difference", "log", "ratio"])
    def test_a_contrast_widens_by_its_own_curve_mean(self, kind: str) -> None:
        estimate = self.contrasts()[kind]
        assert estimate.rule_widened
        assert abs(estimate.score) / estimate.std_error > 1e-4
        assert_widened(estimate)

    def test_the_difference_curve_mean_is_the_difference_of_the_arm_residuals(self) -> None:
        result = ruled()
        estimate = self.contrasts()["difference"]
        assert estimate.score == pytest.approx(result["ey1"].score - result["ey0"].score, rel=1e-8)

    @pytest.mark.parametrize("kind", ["rr", "or"])
    def test_a_ratio_of_two_levels_widens(self, kind: str) -> None:
        from cleverly.inference.results import ratio_contrast

        estimate = ratio_contrast(
            ruled().estimates, "ey1", "ey0", kind=kind, n=laws.N, name=f"derived_{kind}"
        )
        assert estimate.rule_widened
        assert_widened(estimate)

    def test_a_stratified_fit_widens_its_stratum_and_marginal_estimates(self) -> None:
        frame, columns, settings, _ = engines.KINDS["strata"]()
        result = engines._fit(
            "scaled_var", frame, columns=columns, **settings, targeting="one_step", step_size=0.01
        )
        stratum = [name for name in result.estimates if "[" in name]
        assert stratum
        for name, estimate in result.estimates.items():
            assert estimate.rule_widened, name
            assert_widened(estimate)
        assert max(abs(e.score) / e.std_error for e in result.estimates.values()) > 1e-4

    def test_the_mutation_that_drops_the_contrast_inheritance_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            results_module,
            results_module,
            "smooth_contrast",
            [
                (
                    "rule_widened=any(estimates[key].rule_widened for key in chosen)",
                    "rule_widened=False",
                )
            ],
        )
        estimate = results_module.smooth_contrast(
            ruled().estimates, lambda p: p[0] - p[1], ["ey1", "ey0"], n=laws.N
        )
        assert not estimate.rule_widened
        below, _ = half_widths(estimate, estimate.ci)
        assert below != pytest.approx(
            Z * estimate.std_error + abs(estimate.score), rel=0.0, abs=EXACT
        )


class TestInheritanceSites:
    """Each builder that assembles an estimate from others inherits the flag.

    One fit per site, on the cheapest path that reaches it, with a mutation that forces the
    site's flag to ``False`` as the control.
    """

    SITE = "rule_widened=any(part.rule_widened for part in parts)"

    @staticmethod
    def cross_fitted(**settings: Any) -> Any:
        frame = laws.draw_l1("binary", 1_000, 7)[0]
        covariates = ["W1", "W2", "W3"]
        if "strata" in settings:
            frame = frame.assign(V=(frame["W1"] > 0).astype(int))
            covariates = [*covariates, "V"]
        strata = settings.pop("strata", None)
        estimator = TMLE(
            estimands=("ey1", "ey0", "ate"),
            simultaneous=False,
            g_bounds=canonical_tmle.G_BOUNDS,
            random_state=0,
            stopping="scaled_var",
            targeting="one_step",
            max_iter=laws.ONE_STEP_MAX_ITER,
            tol=laws.TOL,
            cross_fit=True,
            n_folds=2,
            **laws.point_learners("binary"),
            **settings,
        )
        columns: dict[str, Any] = {"covariates": covariates}
        if strata is not None:
            columns["strata"] = strata
        return estimator.fit(frame, outcome="Y", treatment="A", **columns).single()

    @staticmethod
    def assert_inherited(estimates: dict[str, ParameterEstimate]) -> None:
        for name, estimate in estimates.items():
            assert estimate.rule_widened, name
            assert_widened(estimate)
        assert max(abs(e.score) / e.std_error for e in estimates.values()) > 1e-4

    @staticmethod
    def repeated_contrast() -> ParameterEstimate:
        # A point fit refuses a contrast of median-combined repeats; a longitudinal one
        # computes it in every draw and combines the draws by the median rule.
        from cleverly.longitudinal import LTMLE
        from tests.studies import canonical_point_survival_curve as curve

        estimator = LTMLE(
            curve.REGIMENS,
            reference=curve.REFERENCE,
            n_folds=2,
            repeats=2,
            simultaneous=False,
            survival_model="hazard",
            targeting="one_step",
            step_size=laws.L4_STEP_SIZE,
            tol=laws.TOL,
            max_iter=laws.ONE_STEP_MAX_ITER,
            random_state=0,
            stopping="scaled_var",
            **laws.l4_learners("primary"),
        )
        result = estimator.fit(curve.LAW.container(laws.draw_l4(800, 3)[0]))
        assert result.n_repeats == 2
        return result.contrast(lambda p: p[0] - p[1], [laws.L4_RISK, "risk_regimen[arm0 @ t=5]"])

    def test_the_median_over_repeats_inherits(self) -> None:
        self.assert_inherited({"contrast": self.repeated_contrast()})

    def test_the_mutation_that_drops_the_median_inheritance_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            influence_module,
            results_module,
            "median_estimates",
            [(self.SITE, "rule_widened=False")],
        )
        with pytest.raises(AssertionError):
            self.assert_inherited({"contrast": self.repeated_contrast()})

    def test_the_fold_average_inherits(self) -> None:
        self.assert_inherited(self.cross_fitted(cv_evaluation=True).estimates)

    def test_the_mutation_that_drops_the_fold_average_inheritance_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            tmle_module,
            tmle_module,
            "_average_over_folds",
            [(self.SITE, "rule_widened=False")],
        )
        with pytest.raises(AssertionError):
            self.assert_inherited(self.cross_fitted(cv_evaluation=True).estimates)

    def stratified(self) -> dict[str, ParameterEstimate]:
        result = self.cross_fitted(cv_evaluation=True, strata=("V",))
        marginal = {name: e for name, e in result.estimates.items() if "[" not in name}
        assert set(marginal) == {"ey1", "ey0", "ate"}
        return marginal

    def test_the_stratum_mixture_inherits(self) -> None:
        self.assert_inherited(self.stratified())

    def test_the_mutation_that_drops_the_stratum_mixture_inheritance_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strata_module = importlib.import_module("cleverly.estimators._strata")
        mutate(
            monkeypatch,
            strata_module,
            tmle_module,
            "stratum_mixture",
            [(self.SITE, "rule_widened=False")],
        )
        with pytest.raises(AssertionError):
            self.assert_inherited(self.stratified())

    @staticmethod
    def msm_coefficients(rule: str) -> dict[str, ParameterEstimate]:
        from tests.unit import test_stopping_rule_longitudinal as longitudinal

        result = longitudinal.working_model(rule)
        return {n: e for n, e in result.estimates.items() if n.startswith("msm_regimen[")}

    def test_the_working_model_coefficients_inherit(self) -> None:
        # The node walks of this fit solve their score far inside the rule, so the residual
        # is at roundoff and the flag is what the control reads.
        for rule, flagged in (("scaled_var", True), ("relative", False)):
            coefficients = self.msm_coefficients(rule)
            assert coefficients
            for name, estimate in coefficients.items():
                assert estimate.rule_widened is flagged, (rule, name)
                assert_widened(estimate) if flagged else assert_plain_estimate(estimate)

    def test_the_mutation_that_drops_the_working_model_inheritance_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        longitudinal_module = importlib.import_module("cleverly.longitudinal.estimator")
        mutate(
            monkeypatch,
            longitudinal_module,
            longitudinal_module,
            "_msm_estimates",
            [
                (
                    "rule_widened=any(_rule_widened(fit) for fit in msm_fit.fits)",
                    "rule_widened=False",
                )
            ],
        )
        coefficients = self.msm_coefficients("scaled_var")
        assert not any(estimate.rule_widened for estimate in coefficients.values())


# ------------------------------------------------------------------------------ units


class TestSampleSizeUnits:
    """``sample_size`` stops at ``1 / n_u`` in the outcome's units, which can be many SEs."""

    @staticmethod
    @functools.cache
    def small_units() -> Any:
        # LN's misspecified outcome regression leaves an initial score of about 3 SE; with the
        # outcome in units of 1e-4 the threshold 1/n is about 47 SE, so the walk stops at once.
        frame, _ = laws.draw_ln(laws.N, 20261115)
        frame = frame.assign(Y=frame["Y"] * 1e-4)
        return laws.fit_point(
            "LN", frame, stopping="sample_size", targeting="one_step", estimands=("ey1",)
        )

    def test_the_threshold_is_one_over_n_in_the_outcome_units(self) -> None:
        result = self.small_units()
        fluctuation = result.fluctuations["mean"]
        np.testing.assert_allclose(fluctuation.thresholds, 1.0 / result.data.n, rtol=1e-12)

    def test_the_residual_exceeds_a_standard_error_and_the_interval_absorbs_it(self) -> None:
        estimate = self.small_units()["ey1"]
        assert abs(estimate.score) / estimate.std_error > 1.0
        assert_widened(estimate)


# ------------------------------------------------------------------------------ bands


class TestBands:
    def test_each_band_component_widens_by_its_residual(self) -> None:
        estimates = {name: ruled()[name] for name in ("ey1", "ey0", "ate")}
        bands = simultaneous_bands(estimates, random_state=0, band_scale="identity")
        for name, estimate in estimates.items():
            low, high = bands.bands[name]
            expected = bands.critical_value * estimate.std_error + abs(estimate.score)
            assert estimate.psi - low == pytest.approx(expected, abs=EXACT)
            assert high - estimate.psi == pytest.approx(expected, abs=EXACT)

    def test_a_logit_band_widens_by_the_residual_times_the_slope(self) -> None:
        estimate = ruled()["ey1"]
        band = simultaneous_bands({"ey1": estimate, "ey0": ruled()["ey0"]}, random_state=0)
        low, high = band.bands["ey1"]
        logit = lambda p: np.log(p / (1.0 - p))  # noqa: E731
        center = logit(estimate.psi)
        slope = 1.0 / (estimate.psi * (1.0 - estimate.psi))
        expected = (band.critical_value * estimate.std_error + abs(estimate.score)) * slope
        assert center - logit(low) == pytest.approx(expected, abs=1e-10)
        assert logit(high) - center == pytest.approx(expected, abs=1e-10)

    def test_a_widened_band_is_wider_than_the_unwidened_band(self) -> None:
        estimates = {name: ruled()[name] for name in ("ey1", "ey0")}
        plain = {name: replace(e, rule_widened=False) for name, e in estimates.items()}
        widened = simultaneous_bands(estimates, random_state=0, band_scale="identity")
        unwidened = simultaneous_bands(plain, random_state=0, band_scale="identity")
        for name in estimates:
            assert widened.bands[name][1] - widened.bands[name][0] > (
                unwidened.bands[name][1] - unwidened.bands[name][0]
            )


# ------------------------------------------------------------------------------ consumers

effect_curves_module = importlib.import_module("cleverly.effect_curves")
msm_module = importlib.import_module("cleverly.msm")
omitted_module = importlib.import_module("cleverly.sensitivity.omitted_variable")
score_module = importlib.import_module("cleverly.fluctuation._score")

#: The grid an effect curve is read on.
CURVE_GRID = pd.DataFrame({"W1": np.linspace(-2.0, 2.0, 9)})


@functools.cache
def ruled_curve() -> Any:
    """A one-step ``scaled_var`` projection of the CATE on ``W1``, on 1,000 rows of L1."""
    from cleverly import CausalStudy, ConditionalContrast, PointTreatment
    from cleverly.bases import Linear, Modifiers

    frame = laws.draw_l1("binary", 1000, 20261112)[0]
    covariates = tuple(column for column in frame.columns if column not in ("A", "Y"))
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=covariates)
    )
    estimand = ConditionalContrast(
        Modifiers(["W1"], domains={"W1": (-5.0, 5.0)}), model=Linear(["W1"])
    )
    return study.identify(estimand).estimate(
        **laws.point_learners("binary"),
        cross_fit=False,
        random_state=0,
        targeting="one_step",
        stopping="scaled_var",
    )


@functools.cache
def ruled_two_term_curve() -> Any:
    """The fit of :func:`ruled_curve` projected on ``(W1, W2)``: two tested terms (review R1)."""
    from cleverly import CausalStudy, ConditionalContrast, PointTreatment
    from cleverly.bases import Linear, Modifiers

    frame = laws.draw_l1("binary", 1000, 20261112)[0]
    covariates = tuple(column for column in frame.columns if column not in ("A", "Y"))
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=covariates)
    )
    estimand = ConditionalContrast(
        Modifiers(["W1", "W2"], domains={"W1": (-5.0, 5.0), "W2": (-5.0, 5.0)}),
        model=Linear(["W1", "W2"]),
    )
    return study.identify(estimand).estimate(
        **laws.point_learners("binary"),
        cross_fit=False,
        random_state=0,
        targeting="one_step",
        stopping="scaled_var",
    )


@functools.cache
def ruled_msm() -> Any:
    """A one-step ``scaled_var`` logit working model over three dose levels."""
    from cleverly import MSMProjection
    from cleverly.msm import MSM
    from tests.unit import test_plotting_curves as curves

    study = curves._binary_study()
    return study.identify(MSMProjection(MSM.linear(link="logit"))).estimate(
        **curves._BINARY, targeting="one_step", stopping="scaled_var"
    )


def curve_scores(fit: Any) -> np.ndarray:
    """The coefficients' scores, intercept first."""
    return np.array([estimate.score for estimate in fit.estimates.values()])


def curve_margin(fit: Any, w1: np.ndarray) -> np.ndarray:
    """``|phi(w)' s|`` for the linear basis ``(1, w)``, computed from the estimates."""
    scores = curve_scores(fit)
    return np.abs(scores[0] + scores[1] * w1)


def assert_prediction_widened(fit: Any) -> None:
    table = fit.effect_curve().predict(CURVE_GRID, interval=True)
    estimate = np.asarray(table["estimate"], dtype=float)
    se = np.asarray(table["std_err"], dtype=float)
    expected = Z * se + curve_margin(fit, CURVE_GRID["W1"].to_numpy())
    np.testing.assert_allclose(estimate - np.asarray(table["ci_lower"]), expected, atol=EXACT)
    np.testing.assert_allclose(np.asarray(table["ci_upper"]) - estimate, expected, atol=EXACT)


def assert_band_widened(fit: Any) -> None:
    band = fit.effect_curve().model.band(CURVE_GRID, random_state=0)
    se = fit.effect_curve().model.std_error(CURVE_GRID)
    expected = band.critical_value * se + curve_margin(fit, CURVE_GRID["W1"].to_numpy())
    np.testing.assert_allclose(band.estimate - band.lower, expected, atol=EXACT)
    np.testing.assert_allclose(band.upper - band.estimate, expected, atol=EXACT)


def assert_heterogeneity_read_at_the_residual(fit: Any) -> None:
    """One tested term: the test reads ``(|psi| - |s|)^2 / se^2`` on one df."""
    test = fit.heterogeneity_test()
    slope = list(fit.estimates.values())[1]
    margin = abs(slope.score) * np.sqrt(test.statistic) / abs(slope.psi)
    assert test.residual_margin == pytest.approx(margin, rel=1e-10)
    read = max(0.0, np.sqrt(test.statistic) - margin) ** 2
    assert test.p_value == pytest.approx(float(stats.chi2.sf(read, 1)), rel=1e-10)
    assert test.p_value > float(stats.chi2.sf(test.statistic, 1))


def assert_two_term_margin(fit: Any) -> None:
    """Two tested terms: the margin is ``sqrt(s' Sigma^-1 s)`` in the joint covariance.

    A margin that ignores the correlation of the coefficients, ``sqrt(sum s_j^2 / Sigma_jj)``,
    differs from it, so this check fails a diagonal-only margin.
    """
    from cleverly.inference.results import estimate_covariance

    test = fit.heterogeneity_test()
    names = list(fit.estimates)[1:]
    covariance = estimate_covariance(fit.estimates, names)
    scores = np.array([fit.estimates[name].score for name in names])
    margin = float(np.sqrt(scores @ np.linalg.solve(covariance, scores)))
    assert test.df == 2
    assert test.residual_margin == pytest.approx(margin, rel=1e-10)
    read = max(0.0, np.sqrt(test.statistic) - margin) ** 2
    assert test.p_value == pytest.approx(float(stats.chi2.sf(read, 2)), rel=1e-10)


def assert_fitted_means_widened(fit: Any) -> None:
    from cleverly.msm import working_model_means

    means = working_model_means(fit)
    levels = np.array([0.0, 1.0, 2.0])
    scores = curve_scores(fit)
    margin = np.abs(scores[0] + scores[1] * levels)
    logit = stats.logistic.ppf
    linear = logit(means.estimate)
    half = Z * means.link_std_error + margin
    np.testing.assert_allclose(logit(means.lower), linear - half, atol=1e-9)
    np.testing.assert_allclose(logit(means.upper), linear + half, atol=1e-9)


def ovb_pair(fit: Any) -> tuple[Any, Any]:
    """The bounds of ``fit``'s ATE, and the same bounds with the margin switched off."""
    from cleverly.sensitivity.omitted_variable import omitted_variable_bounds

    widened = omitted_variable_bounds(fit, "ate", cf_y=0.02, cf_d=0.02)
    draw = omitted_module._Draw
    with pytest.MonkeyPatch.context() as patch:
        # Each draw built without its flag: the same bounds from an unwidened estimate.
        patch.setattr(
            omitted_module, "_Draw", lambda psi, curve, elements, _: draw(psi, curve, elements)
        )
        plain_bounds = omitted_variable_bounds(fit, "ate", cf_y=0.02, cf_d=0.02)
    return widened, plain_bounds


def assert_limits_widened(fit: Any) -> None:
    widened, plain_bounds = ovb_pair(fit)
    margin = abs(fit["ate"].score)
    assert widened.lower == plain_bounds.lower
    assert widened.ci_lower == pytest.approx(plain_bounds.ci_lower - margin, rel=0.0, abs=EXACT)
    assert widened.ci_upper == pytest.approx(plain_bounds.ci_upper + margin, rel=0.0, abs=EXACT)


class TestWidenedConsumers:
    """Every interval read from a widened fit adds the residual its own quantity leaves.

    Each consumer is checked on a fit whose residual is far from zero, and a mutation that
    drops the consumer's margin must fail the check.
    """

    def test_the_witnesses_carry_a_residual(self) -> None:
        for fit in (ruled_curve(), ruled_msm()):
            estimates = list(fit.estimates.values())
            assert all(estimate.rule_widened for estimate in estimates)
            assert max(abs(e.score) / e.std_error for e in estimates) > 1e-2
        assert abs(ruled()["ate"].score) / ruled()["ate"].std_error > 1e-2

    def test_an_effect_curve_interval_adds_the_curve_residual(self) -> None:
        assert_prediction_widened(ruled_curve())

    def test_an_effect_curve_band_adds_the_curve_residual(self) -> None:
        assert_band_widened(ruled_curve())

    def test_the_heterogeneity_test_reads_the_statistic_less_the_residual(self) -> None:
        assert_heterogeneity_read_at_the_residual(ruled_curve())

    def test_two_tested_terms_read_the_margin_in_the_joint_covariance(self) -> None:
        fit = ruled_two_term_curve()
        from cleverly.inference.results import estimate_covariance

        names = list(fit.estimates)[1:]
        covariance = estimate_covariance(fit.estimates, names)
        scores = np.array([fit.estimates[name].score for name in names])
        joint = float(np.sqrt(scores @ np.linalg.solve(covariance, scores)))
        diagonal = float(np.sqrt(np.sum(scores**2 / np.diag(covariance))))
        # The witness: the two forms differ well beyond the check's tolerance.
        assert abs(joint - diagonal) > 1e-2 * joint
        assert_two_term_margin(fit)

    def test_a_working_model_fitted_mean_adds_the_linear_predictor_residual(self) -> None:
        assert_fitted_means_widened(ruled_msm())

    def test_the_omitted_variable_limits_add_the_residual(self) -> None:
        assert_limits_widened(ruled())

    def test_the_mutation_that_drops_the_curve_margin_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            effect_curves_module,
            effect_curves_module.ProjectionCurve,
            "residual_margin",
            [("if not self.rule_widened:", "if True:")],
        )
        with pytest.raises(AssertionError):
            assert_prediction_widened(ruled_curve())
        with pytest.raises(AssertionError):
            assert_band_widened(ruled_curve())

    def test_the_mutation_that_drops_the_heterogeneity_margin_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            effect_curves_module,
            effect_curves_module,
            "heterogeneity_test",
            [("np.sqrt(statistic) - margin", "np.sqrt(statistic)")],
        )
        with pytest.raises(AssertionError):
            assert_heterogeneity_read_at_the_residual(ruled_curve())

    def test_the_mutation_that_reads_a_diagonal_margin_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review R1, mutation M-a: s' Sigma^-1 s read as sum s_j^2 / Sigma_jj.
        mutate(
            monkeypatch,
            effect_curves_module,
            effect_curves_module,
            "heterogeneity_test",
            [
                (
                    "scores @ np.linalg.solve(covariance, scores)",
                    "np.sum(scores**2 / np.diag(covariance))",
                )
            ],
        )
        with pytest.raises(AssertionError):
            assert_two_term_margin(ruled_two_term_curve())

    def test_the_mutation_that_drops_the_fitted_means_margin_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            msm_module,
            msm_module,
            "working_model_means",
            [("if any(estimate.rule_widened for estimate in estimates):", "if False:")],
        )
        with pytest.raises(AssertionError):
            assert_fitted_means_widened(ruled_msm())

    def test_the_mutation_that_drops_the_limit_margin_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            omitted_module,
            omitted_module,
            "_limit",
            [("quantile * error + max(margins)", "quantile * error")],
        )
        with pytest.raises(AssertionError):
            assert_limits_widened(ruled())


class TestTheBootstrapUnderARule:
    """A declared rule refuses the full-refit bootstrap: no derivation bounds its law."""

    @pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
    def test_a_point_fit_refuses(self, rule: str) -> None:
        from cleverly.exceptions import CapabilityError

        with pytest.raises(CapabilityError, match="residual score in every bootstrap refit"):
            TMLE(stopping=rule, n_bootstrap=20)

    def test_a_longitudinal_fit_refuses(self) -> None:
        from cleverly.exceptions import CapabilityError
        from cleverly.longitudinal import LTMLE
        from tests.unit import test_stopping_rule_longitudinal as longitudinal

        with pytest.raises(CapabilityError, match="residual score in every bootstrap refit"):
            LTMLE(longitudinal.PLANS, stopping="scaled_var", n_bootstrap=20)

    def test_the_default_rule_keeps_the_bootstrap(self) -> None:
        assert TMLE(n_bootstrap=20).n_bootstrap == 20

    def test_the_mutation_that_drops_the_refusal_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            score_module,
            tmle_module,
            "check_rule_bootstrap",
            [("and n_bootstrap:", "and False:")],
        )
        with pytest.raises(pytest.fail.Exception):
            self.test_a_point_fit_refuses("scaled_var")


class TestTheRecord:
    def test_a_widened_estimate_records_its_margin(self) -> None:
        estimate = ruled()["ate"]
        row = estimate.to_dict()
        assert row["rule_widened"] is True
        assert row["residual_margin"] == abs(estimate.score) > 0.0

    def test_a_default_rule_estimate_records_nothing(self) -> None:
        row = relative()["ate"].to_dict()
        assert "rule_widened" not in row and "residual_margin" not in row
