"""The ``variance="policy_weighted"`` option of ``LTMLE`` and ``SDR``.

The option adds the change-of-measure variance of :mod:`cleverly.longitudinal.policy_variance`
beside the influence-curve one, and reports the larger (:mod:`cleverly.inference.policy`).
The fits below are of the five-node pricing law (``tests/studies/mtp_pricing_common.py``) at
``tau = 2``, with its exact ratio on the classifier route.  There the policy-weighted variance
wins for the moved plans and the natural plan's diagonal is its influence-curve variance, so
both sources are exercised.

Witnesses, by number: 3b the residual at the observed dose, 4 the fallback, 5 the
finite-sample value on law B (``p4``'s variance and its covariance with ``p1``, with the control
of reading the source plan's ratio), 7 the refusals before any learner, 8 the support kind,
11 nesting, 12 ``contrast()`` and ``ratio()``, 13 ``transform=``, 14 repeats, 15 the
downstream refusals, 16 the diagnostics and the warning text, plus the save and load round
trip, the composed interval of a Student-t default, the natural level on a rescaled outcome,
the density route's ratios at the policy doses, and the exact covariance of two drawn plans at
a categorical node.  Witness 6 reproduces the measured probe
on committed draws (``tests/diagnostics/fix5_heavy_ratios/probe_variance.csv``).
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn
from scipy.stats import norm, t
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression

from cleverly import load
from cleverly.datasets import make_longitudinal
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, SparseInfluenceWarning
from cleverly.inference import simultaneous_bands
from cleverly.inference.delta import Transform
from cleverly.inference.influence import make_estimate, median_estimates
from cleverly.inference.multiplier import POLICY_VARIANCE_BAND_REFUSAL
from cleverly.inference.policy import PolicyCovariance, PolicyVariance
from cleverly.inference.results import (
    POLICY_COVARIANCE_REFUSAL,
    smooth_contrast,
    transform_estimate,
)
from cleverly.inference.sparse import warn_sparse_influence
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, SDR
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import policy_variance as policy_module
from cleverly.longitudinal.estimator import POLICY_VARIANCE_REFUSALS
from cleverly.msm import MSM
from tests.studies import canonical_mtp_boundaries as boundaries
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_common as law_b
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_tail_size_common as tail
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.unit._natural_course_support import Counting, CountingLinear, CountingLogistic

N = 2000
TAU = 2
NATURAL = "ey_regimen[natural_prices]"
UP = "ey_regimen[price_up]"
DOWN = "ey_regimen[price_down]"
PROBE = (
    Path(__file__).resolve().parents[1] / "diagnostics" / "fix5_heavy_ratios" / "probe_variance.csv"
)


def _settings(**overrides: Any) -> dict[str, Any]:
    """The pricing study's in-sample settings on its exact ratio, without a band."""
    return {
        "reference": "natural_prices",
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": common.OracleRatio(),
        "n_folds": 1,
        "g_bounds": pricing.G_BOUNDS,
        "ratio": "classifier",
        "simultaneous": False,
        "random_state": 0,
        **overrides,
    }


def _fit(
    estimator: type[LTMLE] | type[SDR] = LTMLE,
    *,
    tau: int = TAU,
    n: int = N,
    seed: int = 11,
    **overrides: Any,
) -> Any:
    """One fit of the pricing law at ``tau`` with the three plans."""
    frame = common.sample(n, seed, tau)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(common.regimens(pricing.LABELS, tau), **_settings(**overrides)).fit(
            frame, outcome="Y", **pricing.roles(tau)
        )


@cache
def default_fit() -> Any:
    """The fit with the default variance."""
    return _fit()


@cache
def option_fit() -> Any:
    """The same fit with ``variance="policy_weighted"``."""
    return _fit(variance="policy_weighted")


@cache
def option_sdr() -> Any:
    """The SDR fit of the same draw under the option."""
    return _fit(SDR, variance="policy_weighted")


def _gradient(covariance: PolicyCovariance, weights: dict[str, float]) -> np.ndarray:
    """The gradient over the fit's levels with ``weights`` on the named levels."""
    return np.array([weights.get(name, 0.0) for name in covariance.names])


# ---------------------------------------------------------------------------- witness 11


@pytest.mark.parametrize("estimator", ["ltmle", "sdr"])
def test_each_estimate_reports_the_larger_variance(estimator: str) -> None:
    result = option_fit() if estimator == "ltmle" else option_sdr()
    default = default_fit() if estimator == "ltmle" else _cached_default_sdr()
    for name, estimate in result.estimates.items():
        record = estimate.policy
        assert record is not None
        assert estimate.covariance_rule == "policy_weighted"
        assert record.influence_variance == pytest.approx(default[name].variance, rel=1e-12)
        assert estimate.psi == pytest.approx(default[name].psi, rel=1e-12)
        assert record.usable
        assert estimate.variance == max(record.influence_variance, record.policy_variance)
        lower, upper = estimate.ci
        assert lower <= default[name].ci[0]
        assert upper >= default[name].ci[1]


@cache
def _cached_default_sdr() -> Any:
    return _fit(SDR)


def test_both_sources_win_somewhere() -> None:
    """The nonzero witness: the moved plans read the policy-weighted variance."""
    result = option_fit()
    sources = {name: estimate.variance_source for name, estimate in result.estimates.items()}
    assert sources[NATURAL] == "influence_curve"
    assert sources[UP] == "policy_weighted"
    assert sources[DOWN] == "policy_weighted"
    # The natural plan moves no dose, so its diagonal is its influence-curve variance.
    natural = result[NATURAL].policy
    assert natural.policy_variance == pytest.approx(natural.influence_variance, rel=1e-12)
    assert result[DOWN].policy.policy_variance > 1.05 * result[DOWN].policy.influence_variance


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_the_natural_plan_reads_its_curve_on_a_rescaled_outcome(estimator: type) -> None:
    """On an outcome of range 0.37 the natural level's two variances are one number.

    Its diagonal is written from the same curve as its influence-curve variance, so the
    natural level never reads ``"policy_weighted"`` from a last-bit difference.  Rescaling the
    targeting-scale entry by ``range**2`` instead differed by 3.4e-21 here.
    """
    frame = common.sample(1_500, 11, TAU)
    frame = frame.assign(Y=0.37 * frame["Y"] + 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator(
            common.regimens(pricing.LABELS, TAU),
            **_settings(variance="policy_weighted", q_bounds=(0.2, 0.37 + 0.2)),
        ).fit(frame, outcome="Y", **pricing.roles(TAU))
    natural = result[NATURAL]
    assert natural.policy.policy_variance == natural.policy.influence_variance
    assert natural.variance_source == "influence_curve"
    assert result.diagnostics.influence()[NATURAL].variance_source == "influence_curve"


def test_the_test_interval_and_frame_read_the_stored_variance() -> None:
    result = option_fit()
    frame = result.to_frame().set_index("estimand")
    z = norm.ppf(0.975)
    scales = set()
    for name, estimate in result.estimates.items():
        assert estimate.std_error == pytest.approx(np.sqrt(estimate.variance), rel=1e-14)
        lower, upper = estimate.ci
        scales.add(estimate.interval_scale)
        if estimate.interval_scale == "range_logit":
            # A bounded mean: the Wald interval at the larger SE on the logit scale.
            logit = Transform.range_logit(0.0, 1.0)
            slope = logit.slope(estimate.psi)
            center = float(logit.forward(estimate.psi))
            half = z * estimate.std_error * slope
            assert lower == pytest.approx(float(logit.inverse(center - half)), rel=1e-12)
            assert upper == pytest.approx(float(logit.inverse(center + half)), rel=1e-12)
            for null in (0.3, estimate.psi + 0.02):
                test = estimate.wald_test(null)
                expected = (center - float(logit.forward(null))) / (estimate.std_error * slope)
                assert test.statistic == pytest.approx(expected)
            continue
        assert estimate.interval_scale == "identity"
        assert lower == pytest.approx(estimate.psi - z * estimate.std_error, rel=1e-12)
        assert upper == pytest.approx(estimate.psi + z * estimate.std_error, rel=1e-12)
        for null in (0.0, estimate.psi + 0.02):
            test = estimate.wald_test(null)
            assert test.statistic == pytest.approx((estimate.psi - null) / estimate.std_error)
        assert frame.loc[name, "std_err"] == pytest.approx(estimate.std_error, rel=1e-14)
        payload = estimate.to_dict()
        assert payload["variance_source"] == estimate.variance_source
        assert payload["influence_curve_std_err"] == pytest.approx(
            np.sqrt(estimate.policy.influence_variance)
        )
        assert payload["policy_weighted_std_err"] == pytest.approx(
            np.sqrt(estimate.policy.policy_variance)
        )
    # Law P's outcome lies in [0, 1]: the levels read the logit scale, the contrasts not.
    assert scales == {"range_logit", "identity"}


def test_the_default_fit_carries_no_policy_record() -> None:
    estimate = default_fit()[DOWN]
    assert estimate.policy is None
    assert estimate.variance_source == "influence_curve"
    assert "variance_source" not in estimate.to_dict()


# ---------------------------------------------------------------------------- witness 12


def test_a_contrast_reads_the_larger_of_the_two_derived_variances() -> None:
    result, default = option_fit(), default_fit()
    names = [UP, DOWN]
    derived = result.contrast(lambda x: x[0] - x[1], names)
    reference = default.contrast(lambda x: x[0] - x[1], names)
    covariance = result[UP].policy.covariance
    gradient = _gradient(covariance, {UP: 1.0, DOWN: -1.0})
    # contrast() differentiates numerically when no gradient is given.
    assert derived.policy.gradient == pytest.approx(gradient, rel=1e-6, abs=1e-8)
    expected = float(gradient @ covariance.matrix @ gradient)
    assert derived.policy.policy_variance == pytest.approx(expected, rel=1e-6)
    assert derived.policy.influence_variance == pytest.approx(reference.variance, rel=1e-12)
    assert derived.variance == max(derived.policy.policy_variance, reference.variance)


def test_a_nonlinear_contrast_reads_the_analytic_gradient() -> None:
    result, default = option_fit(), default_fit()
    names = [DOWN, NATURAL]
    derived = result.contrast(lambda x: x[0] / x[1], names, scale="level")
    reference = default.contrast(lambda x: x[0] / x[1], names, scale="level")
    down, natural = result[DOWN].psi, result[NATURAL].psi
    covariance = result[DOWN].policy.covariance
    gradient = _gradient(covariance, {DOWN: 1.0 / natural, NATURAL: -down / natural**2})
    assert derived.policy.gradient == pytest.approx(gradient, rel=1e-6)
    expected = float(gradient @ covariance.matrix @ gradient)
    assert derived.policy.policy_variance == pytest.approx(expected, rel=1e-6)
    assert derived.variance == pytest.approx(max(expected, reference.variance), rel=1e-6)


def test_a_ratio_reads_the_log_scale_gradient() -> None:
    result, default = option_fit(), default_fit()
    derived = result.ratio(DOWN, NATURAL)
    reference = default.ratio(DOWN, NATURAL)
    covariance = result[DOWN].policy.covariance
    gradient = _gradient(
        covariance, {DOWN: 1.0 / result[DOWN].psi, NATURAL: -1.0 / result[NATURAL].psi}
    )
    assert derived.policy.gradient == pytest.approx(gradient, rel=1e-12)
    expected = float(gradient @ covariance.matrix @ gradient)
    assert derived.policy.policy_variance == pytest.approx(expected, rel=1e-12)
    assert derived.policy.influence_variance == pytest.approx(reference.variance, rel=1e-12)
    assert derived.variance == max(expected, reference.variance)
    lower, upper = derived.ci
    assert lower <= reference.ci[0]
    assert upper >= reference.ci[1]


def test_a_selection_that_mixes_the_rules_is_refused() -> None:
    mixed = {UP: option_fit()[UP], DOWN: default_fit()[DOWN]}
    with pytest.raises(ValueError, match="different covariance rules"):
        smooth_contrast(mixed, lambda x: x[0] - x[1], [UP, DOWN], n=N)


def test_a_selection_from_two_option_fits_is_refused() -> None:
    mixed = {UP: option_fit()[UP], DOWN: option_sdr()[DOWN]}
    with pytest.raises(CapabilityError, match="different policy-weighted covariances"):
        smooth_contrast(mixed, lambda x: x[0] - x[1], [UP, DOWN], n=N)


# ---------------------------------------------------------------------------- witness 13


@pytest.mark.parametrize("name", [NATURAL, UP, DOWN])
def test_a_transform_scales_both_variances(name: str) -> None:
    estimate = option_fit()[name]
    mapped = transform_estimate(estimate, Transform.logit())
    slope = 1.0 / (estimate.psi * (1.0 - estimate.psi))
    assert mapped.policy.influence_variance == pytest.approx(
        slope**2 * estimate.policy.influence_variance, rel=1e-12
    )
    assert mapped.policy.policy_variance == pytest.approx(
        slope**2 * estimate.policy.policy_variance, rel=1e-12
    )
    assert mapped.variance == pytest.approx(slope**2 * estimate.variance, rel=1e-12)
    assert mapped.variance_source == estimate.variance_source


def test_a_transformed_contrast_equals_the_transform_of_the_contrast() -> None:
    result = option_fit()
    names = [DOWN, NATURAL]
    direct = result.contrast(lambda x: x[0] / x[1], names, scale="level", transform=Transform.log())
    later = transform_estimate(
        result.contrast(lambda x: x[0] / x[1], names, scale="level"), Transform.log()
    )
    assert direct.variance == pytest.approx(later.variance, rel=1e-12)
    assert direct.variance_source == later.variance_source


# ---------------------------------------------------------------------------- witness 14


def test_the_median_rule_refuses_the_option() -> None:
    """``repeats`` is refused at construction (witness 7), and the median rule refuses too."""
    with pytest.raises(ValueError, match="a median over draws"):
        median_estimates([option_fit().estimates, option_fit().estimates])


# ---------------------------------------------------------------------------- witness 15


def test_the_covariance_matrix_is_refused() -> None:
    with pytest.raises(CapabilityError) as raised:
        option_fit().covariance()
    assert POLICY_COVARIANCE_REFUSAL in str(raised.value)


def test_a_band_over_option_estimates_is_refused() -> None:
    with pytest.raises(CapabilityError) as raised:
        simultaneous_bands(option_fit().estimates)
    assert raised.value.refusal == POLICY_VARIANCE_BAND_REFUSAL


# ---------------------------------------------------------------------------- witness 16


def test_the_influence_report_names_each_source() -> None:
    result = option_fit()
    report = result.diagnostics.influence()
    frame = report.to_frame().set_index("name")
    for name, estimate in result.estimates.items():
        assert frame.loc[name, "variance_source"] == estimate.variance_source
    sources = [estimate.variance_source for estimate in result.estimates.values()]
    shares = report.variance_sources
    assert shares["policy_weighted"] == pytest.approx(sources.count("policy_weighted") / 5)
    assert shares["influence_curve"] == pytest.approx(sources.count("influence_curve") / 5)


@pytest.mark.parametrize("policy", [False, True])
def test_the_warning_text_follows_the_option(policy: bool) -> None:
    report = option_fit().diagnostics.influence()
    rows = tuple(dataclasses.replace(row, variance_rel_sd=0.5) for row in report.rows)
    flagged = dataclasses.replace(report, rows=rows)
    with pytest.warns(SparseInfluenceWarning) as caught:
        warn_sparse_influence(flagged, cross_fitted=False, policy_variance=policy)
    text = str(caught[0].message)
    assert ("larger of the influence-curve and policy-weighted variances" in text) is policy
    assert ("BCa" in text) is not policy


def test_an_option_fit_with_few_driving_rows_warns_with_the_note() -> None:
    frame = common.sample(1500, 11, 5)
    estimator = LTMLE(common.regimens(pricing.LABELS, 5), **_settings(variance="policy_weighted"))
    with pytest.warns(SparseInfluenceWarning, match="policy-weighted variances"):
        estimator.fit(frame, outcome="Y", **pricing.roles(5))


# ----------------------------------------------------------------------------- witness 4


@pytest.mark.parametrize("value", [-1e6, np.nan])
def test_a_failed_variance_regression_keeps_the_default_interval(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(design: Any, target: Any, evaluate_at: Any, loss: Any, *, seed: Any = None) -> Any:
        return [np.full(np.asarray(points).shape[0], value) for points in evaluate_at]

    monkeypatch.setattr(policy_module, "_fit_variance_regression", failed)
    result, default = _fit(variance="policy_weighted"), default_fit()
    for name, estimate in result.estimates.items():
        assert estimate.variance_source == "influence_curve"
        assert estimate.variance == default[name].variance
        assert estimate.ci == default[name].ci
        assert estimate.wald_test(0.5).pvalue == default[name].wald_test(0.5).pvalue
    assert not result[DOWN].policy.usable


def test_a_contrast_whose_quadratic_form_is_negative_keeps_its_curve_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing large covariances leaves the diagonals, so the levels keep the max."""
    shipped = policy_module._fit_variance_regression

    def inflated(design: Any, target: Any, evaluate_at: Any, loss: Any, *, seed: Any = None) -> Any:
        if loss == "squared_error":
            return [np.full(np.asarray(points).shape[0], 1e3) for points in evaluate_at]
        return shipped(design, target, evaluate_at, loss, seed=seed)

    monkeypatch.setattr(policy_module, "_fit_variance_regression", inflated)
    result = _fit(variance="policy_weighted")
    assert result[UP].variance_source == "policy_weighted"
    assert result[DOWN].variance_source == "policy_weighted"
    assert result[UP].variance == option_fit()[UP].variance
    derived = result.contrast(lambda x: x[0] - x[1], [UP, DOWN])
    assert derived.policy.policy_variance < 0.0
    assert derived.variance_source == "influence_curve"
    assert derived.variance == derived.policy.influence_variance


# ------------------------------------------------------------- the composed interval (rev)


@pytest.mark.parametrize(
    ("influence", "policy", "df", "wins"),
    [
        (1.0, 1.1, 5, "default"),  # t_5 * 1 > z * sqrt(1.1)
        (1.0, 4.0, 5, "policy"),
        (1.0, 1.1, None, "policy"),
        (1.0, 0.9, None, "default"),
        (1.0, -1.0, 5, "default"),
        (1.0, np.nan, None, "default"),
    ],
)
def test_the_half_width_is_the_larger_of_the_default_and_z_times_the_policy_error(
    influence: float, policy: float, df: int | None, wins: str
) -> None:
    covariance = PolicyCovariance(("a",), np.array([[policy]]))
    record = PolicyVariance(covariance, np.array([1.0]), influence, policy)
    center = 0.3
    lower, upper = record.interval(center, 0.05, df)
    default = (t.ppf(0.975, df) if df is not None else norm.ppf(0.975)) * np.sqrt(influence)
    expected = default if wins == "default" else norm.ppf(0.975) * np.sqrt(policy)
    assert upper - center == pytest.approx(expected, rel=1e-12)
    assert center - lower == pytest.approx(expected, rel=1e-12)
    for null in np.linspace(center - 2 * expected, center + 2 * expected, 41):
        if abs(abs(null - center) - expected) < 1e-9:
            continue
        _, p_value = record.test(center, float(null), df)
        assert (p_value < 0.05) == (null < lower or null > upper)


def _composed(
    multiple: float,
    *,
    bounds: tuple[float, float] | None,
    scale: str,
    reference_df: int | None = None,
    family: str | None = None,
) -> tuple[Any, Any]:
    """An estimate with and without the option, its pw variance ``multiple`` times the curve's."""
    rng = np.random.default_rng(5)
    curve = rng.normal(size=400)
    curve -= curve.mean()
    common_args: dict[str, Any] = {
        "n": curve.size,
        "rule_widened": False,
        "scale": scale,
        "reference_df": reference_df,
        "bounds": bounds,
        "outcome_family": family,
    }
    psi = 0.8 if scale == "level" else 0.1
    default = make_estimate("q", psi, curve, **common_args)
    pw = multiple * float(np.var(curve, ddof=1)) / curve.size
    option = make_estimate(
        "q",
        psi,
        curve,
        covariance_rule="policy_weighted",
        policy=PolicyCovariance(("q",), np.array([[pw]])),
        gradient=np.array([1.0]),
        **common_args,
    )
    return default, option


@pytest.mark.parametrize("multiple", [0.5, 3.0])
def test_a_bounded_mean_widens_its_logit_half_width_to_z_times_the_policy_error(
    multiple: float,
) -> None:
    """Rev 3.2 with Fix-3 PR 2: the option keeps the default's range-logit scale."""
    default, option = _composed(multiple, bounds=(0.0, 1.0), scale="level")
    assert default.interval_scale == option.interval_scale == "range_logit"
    logit = Transform.range_logit(0.0, 1.0)
    slope = logit.slope(option.psi)
    center = float(logit.forward(option.psi))
    z = norm.ppf(0.975)
    se_ic = np.sqrt(option.policy.influence_variance)
    se_pw = np.sqrt(option.policy.policy_variance)
    half = max(z * se_ic * slope, z * se_pw * slope)
    lower, upper = option.ci
    assert float(logit.forward(upper)) - center == pytest.approx(half, rel=1e-12)
    assert center - float(logit.forward(lower)) == pytest.approx(half, rel=1e-12)
    assert lower <= default.ci[0] and upper >= default.ci[1]
    if multiple < 1.0:
        assert option.ci == default.ci
    for null in np.linspace(lower - 0.05, upper + 0.05, 41):
        if min(abs(null - lower), abs(null - upper)) < 1e-9:
            continue
        assert (option.wald_test(float(null)).pvalue < 0.05) == (null < lower or null > upper)


@pytest.mark.parametrize("multiple", [1.1, 4.0])
def test_a_t_reference_applies_only_to_the_influence_curve_half_width(multiple: float) -> None:
    """Rev 3.2: ``max(t_df * SE_IC, z * SE_pw)``; the curve's df never reaches ``SE_pw``."""
    default, option = _composed(
        multiple, bounds=None, scale="difference", reference_df=5, family="gaussian"
    )
    assert option.interval_scale == "identity"
    se_ic = np.sqrt(option.policy.influence_variance)
    se_pw = np.sqrt(option.policy.policy_variance)
    half = max(t.ppf(0.975, 5) * se_ic, norm.ppf(0.975) * se_pw)
    lower, upper = option.ci
    assert upper - option.psi == pytest.approx(half, rel=1e-12)
    assert option.psi - lower == pytest.approx(half, rel=1e-12)
    # 1.1 times the variance is below t_5 / z squared: the default interval wins.
    assert (option.ci == default.ci) == (multiple < (t.ppf(0.975, 5) / norm.ppf(0.975)) ** 2)


def test_the_policy_quadratic_reads_only_the_used_entries() -> None:
    covariance = PolicyCovariance(("a", "b"), np.array([[1.0, np.nan], [np.nan, 2.0]]))
    assert covariance.quadratic(np.array([0.0, 3.0])) == pytest.approx(18.0)
    assert np.isnan(covariance.quadratic(np.array([1.0, 1.0])))


def test_a_failed_diagonal_fails_every_form_that_reads_it() -> None:
    """A negative cross term would make ``a - b`` large; the failed ``b`` withholds it."""
    covariance = PolicyCovariance(("a", "b"), np.array([[1.0, -5.0], [-5.0, -2.0]]))
    assert covariance.quadratic(np.array([2.0, 0.0])) == pytest.approx(4.0)
    assert np.isnan(covariance.quadratic(np.array([1.0, -1.0])))
    assert np.isnan(covariance.quadratic(np.array([0.0, 1.0])))


# ----------------------------------------------------------------------------- witness 8


def test_the_option_reads_its_own_kind_and_is_experimental() -> None:
    result = option_fit()
    for estimate in result.estimates.values():
        assert estimate.support.label == "experimental"
        assert "licenses no Wald interval, policy-weighted variance" in estimate.support.reason
    # The same configuration with the default variance is licensed: the kind is what moves.
    assert default_fit()[DOWN].support.label == "supported"


# ----------------------------------------------------------------------------- witness 7


def _intercept(label: Any, horizon: int, frame: Any) -> np.ndarray:
    """A working model with an intercept only."""
    del label, horizon
    return np.ones((len(frame), 1))


SETTINGS_REFUSALS: dict[str, dict[str, Any]] = {
    "msm": {
        "msm": MSM(design=_intercept, terms=("(intercept)",), design_kind="known"),
        "reference": None,
    },
    "n_folds": {"n_folds": 2},
    "repeats": {"repeats": 3, "n_folds": 2},
    "simultaneous": {"simultaneous": True},
    "n_bootstrap": {"n_bootstrap": 20},
    "ratio_trim": {"ratio_trim": 0.99},
    "survival": {"survival_model": "hazard", "pseudo_learner": None},
}

DATA_REFUSALS: dict[str, dict[str, Any]] = {
    "learned": {"learned": True},
    "separable": {"separable": True},
    "tuned": {"tuned": True},
    "cluster": {"cluster": np.zeros(4)},
    "survival": {"is_survival": True},
    "censoring": {"censoring_names": ("C1",)},
    "weights": {"is_weighted": True},
    "held": {"treatment_held": True},
    "mixed": {"mixed": True},
}


def test_every_refusal_is_reached() -> None:
    assert set(SETTINGS_REFUSALS) | set(DATA_REFUSALS) == set(POLICY_VARIANCE_REFUSALS)


@pytest.mark.parametrize(
    ("estimator", "key"),
    [
        (estimator, key)
        for estimator in (LTMLE, SDR)
        for key in sorted(SETTINGS_REFUSALS)
        # SDR takes no survival_model=; its survival refusal is the panel's (DATA_REFUSALS).
        if not (estimator is SDR and key == "survival")
    ],
)
def test_a_refused_setting_is_refused_at_construction(estimator: type, key: str) -> None:
    settings = {**_settings(variance="policy_weighted"), **SETTINGS_REFUSALS[key]}
    with pytest.raises(CapabilityError) as raised:
        estimator(common.regimens(pricing.LABELS, TAU), **settings)
    assert raised.value.refusal == POLICY_VARIANCE_REFUSALS[key]


@pytest.mark.parametrize("key", sorted(SETTINGS_REFUSALS))
def test_the_same_setting_with_the_default_variance_is_not_refused_by_the_option(
    key: str,
) -> None:
    settings = {**_settings(), **SETTINGS_REFUSALS[key]}
    try:
        LTMLE(common.regimens(pricing.LABELS, TAU), **settings)
    except CapabilityError as error:  # pragma: no cover - a refusal of another rule
        assert error.refusal not in POLICY_VARIANCE_REFUSALS.values()
    except ValueError:
        pass


def test_an_unknown_variance_is_refused() -> None:
    with pytest.raises(ValueError, match="variance must be"):
        LTMLE(common.regimens(pricing.LABELS, TAU), **_settings(variance="sandwich"))


@pytest.mark.parametrize("key", sorted(DATA_REFUSALS))
def test_each_design_refusal(key: str) -> None:
    """Each design check of ``_refuse_policy_variance``, on a stand-in for the panel."""
    flags = DATA_REFUSALS[key]
    data = SimpleNamespace(
        cluster=flags.get("cluster"),
        is_survival=flags.get("is_survival", False),
        is_repeated=False,
        censoring_names=flags.get("censoring_names", ()),
        is_weighted=flags.get("is_weighted", False),
        treatment_held=flags.get("treatment_held", False),
        n_times=2,
        node_kind=lambda time: "mixed" if flags.get("mixed") else "binary",
    )
    estimator = LTMLE(common.regimens(pricing.LABELS, TAU), **_settings(variance="policy_weighted"))
    with pytest.raises(CapabilityError) as raised:
        estimator._refuse_policy_variance(
            data,
            learned=flags.get("learned", False),
            separable=flags.get("separable", False),
            tuned=flags.get("tuned", False),
        )
    assert raised.value.refusal == POLICY_VARIANCE_REFUSALS[key]
    default = LTMLE(common.regimens(pricing.LABELS, TAU), **_settings())
    default._refuse_policy_variance(
        data,
        learned=flags.get("learned", False),
        separable=flags.get("separable", False),
        tuned=flags.get("tuned", False),
    )


def _panel(key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A two-node binary panel that carries the refused feature."""
    roles: dict[str, Any] = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
    }
    if key == "censoring":
        frame, _ = make_longitudinal(300, seed=3, censoring=True)
        roles["censoring"] = ["C1", "C2"]
    elif key == "cluster":
        frame, _ = make_longitudinal(300, seed=3, censoring=False, cluster_size=3)
        roles["id"] = "id"
    else:
        frame, _ = make_longitudinal(300, seed=3, censoring=False)
        frame = frame.assign(w=np.linspace(0.5, 1.5, len(frame)))
        roles["weights"] = "w"
    return frame, roles


@pytest.mark.parametrize("key", ["censoring", "cluster", "weights"])
def test_a_refused_panel_is_refused_before_any_learner(key: str) -> None:
    frame, roles = _panel(key)
    estimator = LTMLE(
        {"always": 1, "never": 0},
        reference="never",
        outcome_learner=CountingLinear(),
        pseudo_learner=CountingLinear(),
        treatment_learner=CountingLogistic(),
        censoring_learner=CountingLogistic(),
        n_folds=1,
        simultaneous=False,
        variance="policy_weighted",
    )
    Counting.calls = 0
    with pytest.raises(CapabilityError) as raised:
        estimator.fit(frame, **roles)
    assert raised.value.refusal == POLICY_VARIANCE_REFUSALS[key]
    assert Counting.calls == 0


# ----------------------------------------------------------------- a categorical static fit


def test_two_static_binary_regimens_read_the_categorical_sum() -> None:
    """At a binary node a static plan's ratio is ``1 / g(a)`` at its own dose.

    The diagonal is then ``E[b^2] + E[S_1(a, H_1) / g_1(a)]``, the exact sum over the arms.
    """
    frame, _ = make_longitudinal(1500, seed=4, censoring=False)
    settings: dict[str, Any] = {
        "reference": "never",
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": CountingLogistic(),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 0,
    }
    roles = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
    }
    regimens = {"always": 1, "never": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = LTMLE(regimens, **settings).fit(frame, **roles)
        option = LTMLE(regimens, variance="policy_weighted", **settings).fit(frame, **roles)
    matrix = option["ey_regimen[always]"].policy.covariance.matrix
    # The plans differ at the first node, so no row reaches a dose both draw and the pair's
    # carried term is zero; before that case was handled the covariance read nan.
    assert np.all(np.isfinite(matrix))
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) > 0.0)
    for name, estimate in option.estimates.items():
        assert estimate.policy.influence_variance == pytest.approx(
            default[name].variance, rel=1e-12
        )
        assert estimate.variance >= default[name].variance
        # Both read the same variance: close, but not the same number, since neither static
        # plan's diagonal is its curve's.
        ratio = estimate.policy.policy_variance / estimate.policy.influence_variance
        assert 0.7 < ratio < 1.3
        assert ratio != pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------------- save and load (rev 3.1)


def test_the_option_survives_save_and_load(tmp_path: Path) -> None:
    result = option_fit()
    expected = result.contrast(lambda x: x[0] - x[1], [UP, DOWN])
    for restored in (load(result.save(tmp_path / "option.joblib")), loads(dumps(result))):
        for name, estimate in result.estimates.items():
            assert restored[name].variance_source == estimate.variance_source
            assert restored[name].variance == estimate.variance
        again = restored.contrast(lambda x: x[0] - x[1], [UP, DOWN])
        assert again.variance == expected.variance
        assert again.variance_source == expected.variance_source
        assert again.policy.policy_variance == expected.policy.policy_variance
        with pytest.raises(CapabilityError):
            restored.covariance()


# ----------------------------------------------------------------------------- witness 6


def _probe_draws() -> pd.DataFrame:
    return pd.read_csv(PROBE)


#: The scikit-learn release under which the shipped variance reproduces the probe to 1e-10.
#: The variance regressions are ``HistGradientBoostingRegressor`` fits, whose binning and
#: split ties can move between releases, so another release is held to 1e-3: far inside the
#: 1.4% to 8% that a defect of the residual or the ratio moves the variance (witness 3b).
PROBE_SKLEARN = "1.9.1"


def _probe_tolerance() -> tuple[float, float]:
    """The relative tolerances of ``psi`` and of the variance entries under this release."""
    return (1e-10, 1e-10) if sklearn.__version__ == PROBE_SKLEARN else (1e-8, 1e-3)


@pytest.mark.parametrize(("estimator", "rep"), [("ltmle", 0), ("sdr", 0)])
def test_the_shipped_variance_reproduces_the_probe(estimator: str, rep: int) -> None:
    """The fast-tier slice of witness 6; ``run.py`` checks all 48 committed draws."""
    reference = _probe_draws().set_index(["est", "rep"]).loc[(estimator, rep)]
    labels = ("natural_prices", "price_down")
    frame, _ = pricing.draw_scenario(f"pricing_{estimator}_tau5", int(reference["n"]), rep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if estimator == "ltmle":
            model: Any = LTMLE(
                common.regimens(labels, 5),
                max_iter=100,
                tol=1e-10,
                alpha=pricing.PRIMARY_ALPHA,
                **_settings(variance="policy_weighted"),
            )
        else:
            model = SDR(common.regimens(labels, 5), **_settings(variance="policy_weighted"))
        result = model.fit(frame, outcome="Y", **pricing.roles(5))
    estimate = result[DOWN]
    covariance = estimate.policy.covariance
    matrix = covariance.matrix * int(reference["n"])
    down, natural = covariance.names.index(DOWN), covariance.names.index(NATURAL)
    exact, regression = _probe_tolerance()
    assert estimate.psi == pytest.approx(reference["psi"], rel=exact)
    assert matrix[down, down] == pytest.approx(reference["variance"], rel=regression)
    assert matrix[down, natural] == pytest.approx(reference["covariance"], rel=regression)


# ----------------------------------------------------------------------------- witness 5

#: The Monte Carlo SEs of ``n * Sigma[p4, p4]`` and ``n * Sigma[p1, p4]`` at 200,000 law-B
#: rows: the SD over seeds 2 to 11, disjoint from the tested seed 1 (0.0703 and 0.0057; means
#: 3.157 and 0.580 against the exact 3.1357 and 0.5818).  The influence-curve variance of
#: ``p4`` read 2.69 to 5.27 on the same draws.
LAW_B_MC_SE = {"p4": 0.07, "p1p4": 0.0057}
#: The rows and the SE of ``n * Sigma[p1, p4]`` of the mutation control: the SD over seeds 2
#: to 13 at 20,000 rows (0.0115).  The mutated entry read 0.388 to 0.394 on the same seeds.
CROSS_CONTROL_N = 20_000
CROSS_CONTROL_SE = 0.0115


def _entries(result: Any, n: int) -> dict[str, float]:
    """``n Sigma[p4, p4]`` and ``n Sigma[p1, p4]`` of a law-B fit over natural, p1 and p4."""
    covariance = result["ey_regimen[p4]"].policy.covariance
    p1, p4 = covariance.names.index("ey_regimen[p1]"), covariance.names.index("ey_regimen[p4]")
    return {"p4": n * covariance.matrix[p4, p4], "p1p4": n * covariance.matrix[p1, p4]}


def test_the_finite_sample_variance_meets_the_exact_law() -> None:
    """Witness 5: one 200,000-row law-B draw lands within 3 MC SE of the quadrature values.

    ``p4``'s variance and its covariance with ``p1``, a pair that both move the dose.  The
    tolerance is 6.7% of ``p4``'s variance and 2.9% of the covariance, so this detects a gross
    error only.  The exact witnesses of the change of measure are in
    ``tests/unit/test_mtp_tail_size.py``, and the density route is checked below.  The order-24
    quadrature meets the order-40 one to 1e-12.
    """
    n = 200_000
    entries = _entries(_law_b_fit(law_b.sample_b(n, 1), ("natural", "p1", "p4")), n)
    exact = {
        "p4": tail.diagonal_variance("p4", order=24),
        "p1p4": tail.diagonal_variance("p4", "p1", order=24),
    }
    for key, value in entries.items():
        assert abs(value - exact[key]) < 3 * LAW_B_MC_SE[key], key


def test_the_covariance_of_two_moved_plans_reads_the_other_plans_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control of witness 5's covariance: read the source plan's own ratio.

    For the pair ``(p1, p4)`` the change of measure pushes forward through ``p1`` and reads
    ``p4``'s ratio at ``p1``'s policy doses.  The defect reads ``p1``'s ratio there instead,
    whose exact value is 0.392 against 0.582: on a 20,000-row draw it lands more than 10 MC SE
    from the exact covariance.
    """
    measure = policy_module._measure

    def source_ratio(data: Any, mechanism: Any, left: Any, right: Any, time: int) -> Any:
        ratios = dict(mechanism.shifted_ratios)
        ratios[left.label] = mechanism.shifted_ratios[right.label]
        ratios[right.label] = mechanism.shifted_ratios[left.label]
        return measure(
            data, dataclasses.replace(mechanism, shifted_ratios=ratios), left, right, time
        )

    monkeypatch.setattr(policy_module, "_measure", source_ratio)
    frame = law_b.sample_b(CROSS_CONTROL_N, 1)
    mutated = _entries(_law_b_fit(frame, ("natural", "p1", "p4")), CROSS_CONTROL_N)["p1p4"]
    exact = tail.diagonal_variance("p4", "p1", order=24)
    assert abs(mutated - exact) > 10 * CROSS_CONTROL_SE


# ----------------------------------------------------------------------------- witness 3b


def _law_b_fit(frame: pd.DataFrame, labels: tuple[str, ...]) -> Any:
    """The boundaries law's correct regressions and exact ratio under the option."""
    outcome, pseudo, _ = boundaries.learners(frame, "correct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            law_b.regimens(labels),
            reference="natural",
            outcome_learner=outcome,
            pseudo_learner=pseudo,
            treatment_learner=law_b.OracleRatio(),
            n_folds=1,
            g_bounds=boundaries.G_BOUNDS,
            ratio="classifier",
            simultaneous=False,
            random_state=0,
            max_iter=100,
            tol=1e-10,
            variance="policy_weighted",
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            continuous_treatment={"A1": law_b.SUPPORT, "A2": law_b.SUPPORT},
        )


def _diagonal(result: Any) -> dict[str, float]:
    covariance = next(iter(result.estimates.values())).policy.covariance
    n = len(next(iter(result.estimates.values())).influence_curve)
    return {name: n * covariance.matrix[i, i] for i, name in enumerate(covariance.names)}


def test_a_residual_at_the_policy_dose_moves_the_paired_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Witness 3b: on one 5,000-row law-B sample the mutation moves each moved plan's
    variance by more than 0.5%, upward, the sign of the exact-law move of witness 3a."""
    frame = law_b.sample_b(5_000, 1)
    labels = ("natural", "p1", "p4")
    shipped = _diagonal(_law_b_fit(frame, labels))

    def at_policy_dose(fit: Any) -> tuple[np.ndarray, ...]:
        steps = sorted(fit.steps, key=lambda step: step.time)
        return tuple(
            np.where(
                step.trained_on,
                np.asarray(step.pseudo_outcome)
                - np.asarray(step.targeted if step.marginal is None else step.marginal),
                0.0,
            )
            for step in steps
        )

    monkeypatch.setattr(policy_module, "_residuals", at_policy_dose)
    mutated = _diagonal(_law_b_fit(frame, labels))
    assert mutated["ey_regimen[natural]"] == shipped["ey_regimen[natural]"]
    for name in ("ey_regimen[p1]", "ey_regimen[p4]"):
        assert mutated[name] / shipped[name] - 1.0 > 0.005


# ------------------------------------------------------------- the density route (rev 3.3)

#: The median relative error the oracle binned density's ratio may have against the exact
#: ratio at the policy dose: the binning error.  On law B at 3,000 rows it measured 0.005.
DENSITY_ROUTE_TOLERANCE = 0.02


def test_the_density_route_reads_each_ratio_at_the_policy_dose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ratios the variance reads on the density route are the exact ratios at the doses.

    ``shifted_ratios[plan][t][(source, branch)]`` must be ``plan``'s ratio at ``source``'s
    policy dose of that branch.  The gated ``p4`` cells read this route with the oracle hazard,
    so each entry is checked against ``law.node_ratio`` within the binning error.  The control
    is the defect of reading the ratio at the observed dose: it misses by far more.
    """
    from tests.studies.oracle_density_bins import oracle_bins

    captured: dict[str, Any] = {}
    shipped = estimator_module.policy_weighted_variance

    def spy(data: Any, fits: Any, plans: Any, mechanism: Any, *, seed: Any = None) -> Any:
        captured["mechanism"], captured["plans"] = mechanism, plans
        return shipped(data, fits, plans, mechanism, seed=seed)

    monkeypatch.setattr(estimator_module, "policy_weighted_variance", spy)
    n = 2_000
    frame = law_b.sample_b(n, 5)
    outcome, pseudo, hazard = boundaries.learners(frame, "correct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LTMLE(
            law_b.regimens(("natural", "p1", "p4")),
            reference="natural",
            outcome_learner=outcome,
            pseudo_learner=pseudo,
            treatment_learner=hazard,
            n_folds=1,
            g_bounds=boundaries.G_BOUNDS,
            density_bins=oracle_bins(n),
            ratio="density",
            simultaneous=False,
            random_state=0,
            max_iter=100,
            tol=1e-10,
            variance="policy_weighted",
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            continuous_treatment={"A1": law_b.SUPPORT, "A2": law_b.SUPPORT},
        )
    mechanism, plans = captured["mechanism"], captured["plans"]
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    a1, l2, a2 = frame["A1"].to_numpy(), frame["L2"].to_numpy(), frame["A2"].to_numpy()
    u = law_b.upper(w2)
    nodes = {1: (a1, law_b.mean1(w1, w2)), 2: (a2, law_b.mean2(w1, w2, a1, l2))}
    checked = 0
    for node, (dose, mean) in nodes.items():
        expected = {
            (source, branch)
            for source in ("natural", "p1", "p4")
            for branch in range(len(plans[source].carry_targets(node)))
        }
        for label in ("p1", "p4"):
            assert set(mechanism.shifted_ratios[label][node]) == expected
            for (source, branch), value in mechanism.shifted_ratios[label][node].items():
                target = plans[source].carry_targets(node)[branch]
                exact = law_b.node_ratio(label, node, target, mean, u)
                at_dose = law_b.node_ratio(label, node, dose, mean, u)
                weight = np.maximum(exact, 1e-9)
                error = float(np.median(np.abs(value - exact) / weight))
                assert error < DENSITY_ROUTE_TOLERANCE, (label, node, source, branch, error)
                if source != "natural":
                    # The control: the ratio at the observed dose is far from the target's.
                    assert float(np.median(np.abs(at_dose - exact) / weight)) > 0.1
                checked += 1
        # The natural plan moves nothing: its ratio is one at every plan's dose in the
        # support and zero outside it.  The binned support's edge can disagree with the exact
        # one at a dose on the boundary, which is a row or two.
        for (source, branch), value in mechanism.shifted_ratios["natural"][node].items():
            target = plans[source].carry_targets(node)[branch]
            exact = law_b.node_ratio("natural", node, target, mean, u)
            assert set(np.unique(value)) <= {0.0, 1.0}
            assert np.mean(value != exact) < 0.005, (node, source, branch)
    assert checked >= 2 * 2 * 3


# ------------------------------------------- two drawn plans at a categorical node (exact)
# A two-node binary law with every conditional law known, and two stochastic plans that draw
# the arm with probability 0.2 and 0.8 at both nodes.  The efficient influence function of
# each plan's mean is enumerated over the 32 cells, so E[D^L D^R] is exact.  The fit reads
# the cells by their training means, so its regressions are saturated.


def _cat_g1(w: Any) -> Any:
    return 0.3 + 0.4 * w


def _cat_q(w: Any, a1: Any) -> Any:
    return 0.2 + 0.3 * w + 0.3 * a1


def _cat_g2(w: Any, a1: Any, l2: Any) -> Any:
    return 0.25 + 0.2 * a1 + 0.3 * l2


def _cat_m(w: Any, a1: Any, l2: Any, a2: Any) -> Any:
    return 0.1 + 0.2 * w + 0.2 * a1 + 0.15 * l2 + 0.25 * a2


#: Each plan's probability of drawing arm 1, at both nodes.
CAT_PLANS = {"low": 0.2, "high": 0.8}
#: The rows of the tested draw, its seed, and the Monte Carlo SE of ``n Sigma[low, high]``
#: there: the SD over seeds 2 to 11, disjoint from the tested seed (0.00193; mean 0.1481
#: against the exact 0.14928).
CAT_N = 20_000
CAT_SEED = 1
CAT_MC_SE = 0.0019


def _bernoulli(p: float, x: int) -> float:
    return p if x else 1.0 - p


def _cat_curve(label: str) -> Any:
    """The efficient influence function of ``E[Y^label]`` as a function of one cell."""
    s = CAT_PLANS[label]

    def carried2(w: int, a1: int, l2: int) -> float:
        return s * _cat_m(w, a1, l2, 1) + (1 - s) * _cat_m(w, a1, l2, 0)

    def q1(w: int, a1: int) -> float:
        return _cat_q(w, a1) * carried2(w, a1, 1) + (1 - _cat_q(w, a1)) * carried2(w, a1, 0)

    def carried1(w: int) -> float:
        return s * q1(w, 1) + (1 - s) * q1(w, 0)

    psi = 0.5 * carried1(0) + 0.5 * carried1(1)

    def curve(w: int, a1: int, l2: int, a2: int, y: int) -> float:
        r1 = _bernoulli(s, a1) / _bernoulli(_cat_g1(w), a1)
        r2 = _bernoulli(s, a2) / _bernoulli(_cat_g2(w, a1, l2), a2)
        return (
            carried1(w)
            - psi
            + r1 * (carried2(w, a1, l2) - q1(w, a1))
            + r1 * r2 * (y - _cat_m(w, a1, l2, a2))
        )

    return curve


def _cat_exact(left: str, right: str) -> float:
    """:math:`E[D^L D^R]` by enumeration of the 32 cells."""
    first, second = _cat_curve(left), _cat_curve(right)
    total = 0.0
    for w, a1, l2, a2, y in itertools.product((0, 1), repeat=5):
        mass = (
            0.5
            * _bernoulli(_cat_g1(w), a1)
            * _bernoulli(_cat_q(w, a1), l2)
            * _bernoulli(_cat_g2(w, a1, l2), a2)
            * _bernoulli(_cat_m(w, a1, l2, a2), y)
        )
        total += mass * first(w, a1, l2, a2, y) * second(w, a1, l2, a2, y)
    return total


def _cat_sample(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.5, n)
    a1 = rng.binomial(1, _cat_g1(w))
    l2 = rng.binomial(1, _cat_q(w, a1))
    a2 = rng.binomial(1, _cat_g2(w, a1, l2))
    y = rng.binomial(1, _cat_m(w, a1, l2, a2))
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y}).astype(float)


def _cell_table(design: Any, target: Any) -> dict[tuple[float, ...], float]:
    keys, inverse = np.unique(np.asarray(design, dtype=float), axis=0, return_inverse=True)
    values = np.asarray(target, dtype=float)
    inverse = inverse.reshape(-1)
    return {tuple(key): float(values[inverse == i].mean()) for i, key in enumerate(keys)}


class _CellMean(RegressorMixin, BaseEstimator):
    """The training mean of the target in each design cell."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _CellMean:
        self.table_ = _cell_table(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.array([self.table_[tuple(row)] for row in np.asarray(X, dtype=float)])


class _CellMeanClassifier(ClassifierMixin, BaseEstimator):
    """The training share of arm 1 in each design cell."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _CellMeanClassifier:
        self.classes_ = np.array([0.0, 1.0])
        self.table_ = _cell_table(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        one = np.array([self.table_[tuple(row)] for row in np.asarray(X, dtype=float)])
        return np.column_stack([1.0 - one, one])


def _cat_cross(frame: pd.DataFrame) -> float:
    """``n Sigma[low, high]`` of the option fit of ``frame``."""
    regimens = {
        label: Stochastic(
            lambda h, s=s: np.tile([1.0 - s, s], (len(h), 1)), label, density_kind="known"
        )
        for label, s in CAT_PLANS.items()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            regimens,
            reference="low",
            outcome_learner=_CellMean(),
            pseudo_learner=_CellMean(),
            treatment_learner=_CellMeanClassifier(),
            n_folds=1,
            simultaneous=False,
            random_state=0,
            variance="policy_weighted",
        ).fit(frame, outcome="Y", treatment=["A1", "A2"], baseline=["W"], time_varying=[[], ["L2"]])
    covariance = result["ey_regimen[low]"].policy.covariance
    low, high = (
        covariance.names.index("ey_regimen[low]"),
        covariance.names.index("ey_regimen[high]"),
    )
    return len(frame) * float(covariance.matrix[low, high])


def test_two_drawn_plans_meet_the_exact_categorical_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At a categorical node two drawn plans weigh each level by ``g^L g^R / g``.

    The fit's ``n Sigma[low, high]`` lands within 3 MC SE of the enumerated
    :math:`E[D^{low} D^{high}]`.  The control reads the source plan's draw law twice,
    ``(g^L)^2 / g``: on the same draw it moves the entry by more than 100 MC SE.
    """
    exact = _cat_exact("low", "high")
    frame = _cat_sample(CAT_N, CAT_SEED)
    shipped = _cat_cross(frame)
    assert abs(shipped - exact) < 3 * CAT_MC_SE
    measure = policy_module._measure

    def source_twice(data: Any, mechanism: Any, left: Any, right: Any, time: int) -> Any:
        return measure(data, mechanism, left, left, time)

    monkeypatch.setattr(policy_module, "_measure", source_twice)
    assert abs(_cat_cross(frame) - exact) > 100 * CAT_MC_SE
