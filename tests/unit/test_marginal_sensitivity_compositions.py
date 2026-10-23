"""X82 part H2: the compositions of the marginal sensitivity model bounds.

Each test pins one composition to a hand-written estimating function on the fit's own
nuisances. Each also carries a nonzero witness: the composition's own term (a weight, a
stratum share, a response probability, a conditioning share, a refitted mechanism) must
move the number, so a version that dropped the term would fail.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators._nuisance import tilt_missingness
from cleverly.exceptions import CapabilityError
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity.marginal import (
    _binary_side,
    marginal_bound_refusal,
    marginal_sensitivity_bounds,
)
from cleverly.validation import _interpretation as interpretation
from tests.unit.test_marginal_sensitivity import (
    binary_fit,
    binary_frame,
    covering_row,
    gaussian_fit,
    gaussian_frame,
)

pytestmark = pytest.mark.xdist_group("marginal_sensitivity")


def parts(fit: Any) -> dict[str, Any]:
    """The first draw's bounded mechanism, unscaled outcome regressions and data arrays."""
    nuisance = fit.repeats[0].nuisance
    g = nuisance.propensity.bounded(fit.config.g_bounds)
    arms = [float(arm) for arm in nuisance.propensity.arms]
    return {
        "g": {arm: g[:, i] for i, arm in enumerate(arms)},
        "mu": {
            arm: nuisance.scaler.unscale_levels(np.asarray(nuisance.outcome.arms[arm]))
            for arm in arms
        },
        "y": np.asarray(fit.data.outcome, dtype=float),
        "a": np.asarray(fit.data.treatment, dtype=float),
    }


def aipw_summand(p: dict[str, Any], arm: float) -> np.ndarray:
    treated = (p["a"] == arm).astype(float)
    mu = p["mu"][arm]
    return mu + treated / p["g"][arm] * (p["y"] - mu)


def mean_and_se(summand: np.ndarray, mass: np.ndarray) -> tuple[float, float]:
    share = float(np.mean(mass))
    value = float(np.mean(mass * summand)) / share
    return value, float(np.sqrt(influence_variance(mass * (summand - value) / share)))


# ---------------------------------------------------------------------------------------
# Multi-arm treatment


@pytest.fixture(scope="module")
def three_arm() -> Any:
    frame = binary_frame()
    rng = np.random.default_rng(2)
    w = frame["W1"].to_numpy()
    probabilities = np.exp(np.column_stack([np.zeros_like(w), 0.6 * w, -0.6 * w]))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    arm = np.array([rng.choice(3, p=row) for row in probabilities])
    return binary_fit(frame.assign(A=arm))


def test_each_arm_reads_its_indicator_odds(three_arm: Any) -> None:
    p = parts(three_arm)
    contrasts = [name for name in three_arm.estimates if name.startswith("ate[")]
    assert contrasts
    name = contrasts[-1]
    target_arm = max(p["g"])
    bounds = marginal_sensitivity_bounds(three_arm, 2.0, estimand=name)
    up = _binary_side(
        p["y"],
        (p["a"] == target_arm).astype(float),
        p["g"][target_arm],
        p["mu"][target_arm],
        2.0,
        True,
    )
    low = _binary_side(p["y"], (p["a"] == 0.0).astype(float), p["g"][0.0], p["mu"][0.0], 2.0, False)
    assert bounds.rows[0].upper == pytest.approx(up.value - low.value, rel=1e-12)
    assert bounds.null == 0.0
    # Witness: arm 2's bound reads arm 2's own column, not arm 1's.
    wrong = _binary_side(
        p["y"],
        (p["a"] == target_arm).astype(float),
        p["g"][1.0],
        p["mu"][target_arm],
        2.0,
        True,
    )
    assert abs(wrong.value - up.value) > 1e-3


def test_a_multi_arm_default_is_the_first_reported_contrast(three_arm: Any) -> None:
    bounds = marginal_sensitivity_bounds(three_arm, 1.0)
    first = next(name for name in three_arm.estimates if name.startswith("ate["))
    assert bounds.estimand == first


def test_a_multi_arm_effect_on_the_treated_is_refused(three_arm: Any) -> None:
    with pytest.raises(CapabilityError) as raised:
        marginal_sensitivity_bounds(three_arm, 2.0, estimand="att[1.0 vs 0.0]")
    assert raised.value.refusal.roadmap == "X87"


# ---------------------------------------------------------------------------------------
# Sampling weights


class Recording(LinearRegression):
    weighted: ClassVar[list[bool]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        Recording.weighted.append(sample_weight is not None)
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.fixture(scope="module")
def weighted() -> Any:
    frame = gaussian_frame()
    # Weights that move with W1 move the target population, so they move the mean.
    return gaussian_fit(frame.assign(wt=np.exp(frame["W1"])), weights="wt")


def test_weights_enter_every_mean(weighted: Any) -> None:
    p = parts(weighted)
    w = np.asarray(weighted.data.weights, dtype=float)
    summand = aipw_summand(p, 1.0) - aipw_summand(p, 0.0)
    value, se = mean_and_se(summand, w)
    row = marginal_sensitivity_bounds(weighted, 1.0).rows[0]
    assert row.lower == pytest.approx(value, rel=1e-10)
    assert row.lower_std_error == pytest.approx(se, rel=1e-10)
    assert abs(value - float(np.mean(summand))) > 1e-2


def test_weights_reach_every_regression(weighted: Any) -> None:
    Recording.weighted.clear()
    marginal_sensitivity_bounds(
        weighted, 1.5, quantile_learner=lambda tau: Recording(), pseudo_outcome_learner=Recording()
    )
    assert Recording.weighted and all(Recording.weighted)


# ---------------------------------------------------------------------------------------
# Baseline strata


@pytest.fixture(scope="module")
def stratified() -> Any:
    rng = np.random.default_rng(7)
    n = 600
    w = rng.normal(size=n)
    v = rng.integers(0, 2, size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    # The effect is larger in stratum V = 1, so the conditional parameter differs.
    y = rng.binomial(1, 1 / (1 + np.exp(-(w + a * (0.2 + 1.5 * v) - 0.5))))
    frame = pd.DataFrame({"W1": w, "V": v, "A": a, "Y": y})
    return (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
            n_folds=5,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "V"], strata=["V"])
        .single()
    )


def test_a_stratum_parameter_reads_its_rows_over_its_share(stratified: Any) -> None:
    p = parts(stratified)
    mask = (np.asarray(stratified.data.strata) == 1).astype(float)
    summand = aipw_summand(p, 1.0) - aipw_summand(p, 0.0)
    value, se = mean_and_se(summand, mask)
    name = next(name for name in stratified.estimates if name.startswith("ate[") and "1" in name)
    row = marginal_sensitivity_bounds(stratified, 1.0, estimand=name).rows[0]
    assert row.lower == pytest.approx(value, rel=1e-10)
    assert row.lower_std_error == pytest.approx(se, rel=1e-10)
    marginal = marginal_sensitivity_bounds(stratified, 1.0).rows[0]
    assert abs(marginal.lower - value) > 1e-2


# ---------------------------------------------------------------------------------------
# Missing outcomes


def missing_frame(frame: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Response depends on the arm and on W1, so 1 / P(Delta = 1 | A, W) varies by row.
    response = 1 / (1 + np.exp(-(1.0 + 1.2 * frame["W1"] - 0.8 * frame["A"])))
    delta = (rng.random(len(frame)) < response).astype(float)
    return frame.assign(D=delta, Y=np.where(delta > 0, frame["Y"], np.nan))


def test_a_missing_outcome_reweights_by_the_response() -> None:
    fit = (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            random_state=0,
            n_folds=5,
        )
        .fit(
            missing_frame(binary_frame()),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="D",
        )
        .single()
    )
    p = parts(fit)
    nuisance = fit.repeats[0].nuisance
    response = tilt_missingness(nuisance, fit.config.missingness_bound)
    observed = np.asarray(fit.data.observed, dtype=bool)
    y = np.where(observed, p["y"], 0.0)
    treated = (p["a"] == 1.0).astype(float)
    column = list(nuisance.arms).index(1.0)
    weight = np.where(observed, 1.0 / response[:, column], 0.0)
    mu = p["mu"][1.0]
    summand = mu + treated * weight / p["g"][1.0] * (y - mu)
    row = marginal_sensitivity_bounds(fit, 1.0, estimand="ey1").rows[0]
    assert row.lower == pytest.approx(float(np.mean(summand)), rel=1e-10)
    ignored = mu + treated * observed / p["g"][1.0] * (y - mu)
    assert abs(float(np.mean(ignored)) - row.lower) > 1e-3
    wide = marginal_sensitivity_bounds(fit, 2.0).rows[0]
    assert np.isfinite([wide.lower, wide.upper]).all() and wide.lower < wide.upper


def test_a_continuous_missing_outcome_trains_on_observed_rows() -> None:
    frame = missing_frame(gaussian_frame())
    fit = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            q_bounds=(-12.0, 12.0),
            random_state=0,
            n_folds=5,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], delta="D")
        .single()
    )
    bounds = marginal_sensitivity_bounds(
        fit,
        (1.0, 1.5),
        quantile_learner=lambda tau: LinearRegression(),
        pseudo_outcome_learner=LinearRegression(),
    )
    assert all(np.isfinite([row.lower, row.upper]).all() for row in bounds.rows)
    assert bounds.rows[1].lower < bounds.rows[0].lower


# ---------------------------------------------------------------------------------------
# Effects on the treated and on the controls


@pytest.fixture(scope="module")
def binary() -> Any:
    return binary_fit()


def test_the_effect_on_the_treated_conditions_on_the_treated_share(binary: Any) -> None:
    p = parts(binary)
    a, y = p["a"], p["y"]
    mu0, e1, e0 = p["mu"][0.0], p["g"][1.0], p["g"][0.0]
    counterfactual = a * mu0 + (1 - a) * e1 / e0 * (y - mu0)
    numerator = a * y - counterfactual
    share = float(np.mean(a))
    value = float(np.mean(numerator)) / share
    se = float(np.sqrt(influence_variance((numerator - value * a) / share)))
    row = marginal_sensitivity_bounds(binary, 1.0, estimand="att").rows[0]
    assert row.lower == pytest.approx(value, rel=1e-10)
    assert row.upper == pytest.approx(value, rel=1e-10)
    assert row.lower_std_error == pytest.approx(se, rel=1e-10)
    ate = marginal_sensitivity_bounds(binary, 1.0).rows[0]
    assert abs(ate.lower - value) > 1e-3
    wide = marginal_sensitivity_bounds(binary, 2.0, estimand="att").rows[0]
    assert wide.lower < value < wide.upper


def test_the_effect_on_the_controls_conditions_on_the_control_share(binary: Any) -> None:
    p = parts(binary)
    a, y = p["a"], p["y"]
    mu1, e1, e0 = p["mu"][1.0], p["g"][1.0], p["g"][0.0]
    counterfactual = (1 - a) * mu1 + a * e0 / e1 * (y - mu1)
    numerator = counterfactual - (1 - a) * y
    share = float(np.mean(1 - a))
    row = marginal_sensitivity_bounds(binary, 1.0, estimand="atc").rows[0]
    assert row.lower == pytest.approx(float(np.mean(numerator)) / share, rel=1e-10)


# ---------------------------------------------------------------------------------------
# Ratios


@pytest.mark.parametrize("stem", ["rr", "or"])
def test_a_ratio_divides_the_arm_bounds_on_the_log_scale(binary: Any, stem: str) -> None:
    p = parts(binary)
    up1 = _binary_side(p["y"], p["a"], p["g"][1.0], p["mu"][1.0], 2.0, True)
    low0 = _binary_side(p["y"], 1 - p["a"], p["g"][0.0], p["mu"][0.0], 2.0, False)

    def log(value: float) -> float:
        return float(np.log(value / (1 - value))) if stem == "or" else float(np.log(value))

    def slope(value: float) -> float:
        return value * (1 - value) if stem == "or" else value

    bounds = marginal_sensitivity_bounds(binary, (1.0, 2.0), estimand=stem)
    row = bounds.rows[1]
    assert np.log(row.upper) == pytest.approx(log(up1.value) - log(low0.value), rel=1e-10)
    curve = up1.curve / slope(up1.value) - low0.curve / slope(low0.value)
    assert row.upper_std_error == pytest.approx(np.sqrt(influence_variance(curve)), rel=1e-10)
    assert row.upper_limit == pytest.approx(
        row.upper * np.exp(1.959963984540054 * row.upper_std_error), rel=1e-10
    )
    assert bounds.null == 1.0 and bounds.scale == "log"
    assert bounds.rows[0].lower == pytest.approx(bounds.rows[0].upper, rel=1e-12)


def test_a_ratio_needs_its_outcome_scale() -> None:
    fit = gaussian_fit()
    with pytest.raises(CapabilityError, match="binary outcome"):
        marginal_sensitivity_bounds(fit, 2.0, estimand="or")
    with pytest.raises(CapabilityError, match="nonnegative"):
        marginal_sensitivity_bounds(fit, 2.0, estimand="rr")


# ---------------------------------------------------------------------------------------
# Collaborative fits


def test_a_selecting_collaborative_fit_reads_a_refitted_mechanism() -> None:
    from cleverly.sensitivity.omitted_variable import _full_mechanism

    fit = (
        CTMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
            estimands=("ate", "ey1", "ey0"),
        )
        .fit(binary_frame(), outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )
    assert marginal_bound_refusal(fit) is None
    p = parts(fit)
    refit = _full_mechanism(fit, fit.repeats[0]).bounded(fit.config.g_bounds)
    expected = p["mu"][1.0] + p["a"] / refit[:, 1] * (p["y"] - p["mu"][1.0])
    row = marginal_sensitivity_bounds(fit, 1.0, estimand="ey1").rows[0]
    assert row.lower == pytest.approx(float(np.mean(expected)), rel=1e-10)
    stored = aipw_summand(p, 1.0)
    assert abs(float(np.mean(stored)) - row.lower) > 1e-6


def test_a_per_arm_outcome_adaptive_fit_is_refused() -> None:
    from tests.unit.test_outcome_adaptive_per_arm_extensions import (
        BINARY_ACTIVE,
        extended_frame,
        per_arm_fit,
    )

    fit = per_arm_fit(extended_frame(BINARY_ACTIVE, 500, 93))
    refusal = marginal_bound_refusal(fit)
    assert refusal is not None
    assert refusal.reason.value == "implementation_gap" and refusal.roadmap == "X82"


# ---------------------------------------------------------------------------------------
# The limit reading of a composition


def test_a_composition_is_not_covered_by_the_plain_rule(
    monkeypatch: pytest.MonkeyPatch, binary: Any
) -> None:
    monkeypatch.setattr(
        interpretation,
        "VERDICT_EVIDENCE",
        (*interpretation.VERDICT_EVIDENCE, covering_row("binary")),
    )
    assert not marginal_sensitivity_bounds(binary, 1.5).limit_descriptive
    for estimand in ("att", "rr"):
        assert marginal_sensitivity_bounds(binary, 1.5, estimand=estimand).limit_descriptive
    monkeypatch.setattr(
        interpretation,
        "VERDICT_EVIDENCE",
        (*interpretation.VERDICT_EVIDENCE, covering_row("binary+conditional")),
    )
    assert not marginal_sensitivity_bounds(binary, 1.5, estimand="att").limit_descriptive


@pytest.mark.parametrize(("value", "odds"), [(0.0, False), (-0.1, False), (1.0, True)])
def test_a_ratio_refuses_an_arm_bound_without_a_logarithm(value: float, odds: bool) -> None:
    from cleverly.sensitivity.marginal import _End, _log_end

    with pytest.raises(CapabilityError, match="no log"):
        _log_end(_End(value, np.zeros(3)), odds)
    assert np.isfinite(_log_end(_End(0.5, np.ones(3)), odds).value)
