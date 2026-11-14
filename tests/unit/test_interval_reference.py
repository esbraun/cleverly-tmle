"""``interval_reference``: the shipped interval rule, read at a sensitivity bound's end point.

A sensitivity bound forms its limits about its own end point with its own standard error. Its
limits must follow the rule the estimate's ``ci`` follows (the t reference, the range-logit
scale of a binary mean, correlated MOVER of a bounded difference and the skew-widened t of a
continuous outcome), so at no confounding the omitted-variable limits are the shipped interval.
Each asymmetric rule is invisible to a check at a symmetric truth, so every witness here has a
nonzero asymmetry and a mutation control that a symmetric ``psi +/- z * se`` reader fails
(CLAUDE.md, Scientific changes).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.inference import (
    IntervalReference,
    Transform,
    interval_reference,
    make_estimate,
    mover_ci,
    skew_widened_ci,
    wald_ci,
    with_components,
)
from cleverly.sensitivity import _limits as limits_module
from cleverly.sensitivity import marginal_sensitivity_bounds, omitted_variable_bounds

pytestmark = pytest.mark.xdist_group("interval_reference")

EXACT = 1e-12
Z = float(stats.norm.ppf(0.975))
SKEWED = np.array([3.0, -1.0, -1.0, -1.0, 2.0, -2.0, 0.5, -0.5])


def expect_failure(test: Callable[[], None]) -> None:
    """A mutation control: the witness must reject the mutated reader."""
    with pytest.raises(AssertionError):
        test()


def symmetric_reader(estimate: Any, **options: Any) -> IntervalReference:
    """The rebuilt ``psi +/- z * se`` normal interval that the helper replaces."""
    level = options.get("level") or 1.0 - estimate.alpha
    sided = options.get("sided", "two")
    alpha = 1.0 - level if sided == "two" else 2.0 * (1.0 - level)
    psi = float(estimate.psi if options.get("psi") is None else options["psi"])
    se = float(
        estimate.plugin_std_error if options.get("std_error") is None else options["std_error"]
    )
    margin = float(options.get("margin", 0.0))
    low, high = wald_ci(psi, se, alpha, margin=margin)
    return IntervalReference(psi, se, low, high, level, sided, "identity", None, None, margin, True)


def _skew_t() -> Any:
    return make_estimate("ey1", 1.0, SKEWED, n=8, rule_widened=False, outcome_family="gaussian")


def _logit_mean() -> Any:
    curve = np.array([0.6, -0.15, -0.15, -0.15, 0.6, -0.15, -0.15, -0.45])
    return replace(
        make_estimate("ey1", 0.12, curve, n=8, scale="level", rule_widened=False),
        bounds=(0.0, 1.0),
    )


def _mover() -> Any:
    one = replace(
        make_estimate(
            "ey1", 0.6, np.array([0.4, -0.6, 0.4, -0.2]), n=4, scale="level", rule_widened=False
        ),
        bounds=(0.0, 1.0),
    )
    zero = replace(
        make_estimate(
            "ey0", 0.1, np.array([-0.1, 0.3, -0.1, -0.1]), n=4, scale="level", rule_widened=False
        ),
        bounds=(0.0, 1.0),
    )
    ate = replace(
        make_estimate(
            "ate", 0.5, one.influence_curve - zero.influence_curve, n=4, rule_widened=False
        ),
        bounds=(-1.0, 1.0),
    )
    return with_components(ate, one, zero)


RULES = {"skew_t": _skew_t, "range_logit": _logit_mean, "mover": _mover}


def _asymmetry(read: IntervalReference) -> float:
    return abs(read.upper_half_width - read.lower_half_width)


# ------------------------------------------------------------------ the estimate's own rule


@pytest.mark.parametrize("rule", sorted(RULES))
def test_the_own_reading_is_the_shipped_interval(rule: str) -> None:
    estimate = RULES[rule]()
    assert estimate.interval_scale == rule
    read = interval_reference(estimate)
    assert (read.lower, read.upper) == pytest.approx(estimate.ci, rel=EXACT)
    assert read.interval_scale == rule
    # Nonzero witness: each rule is asymmetric about its point here.
    assert _asymmetry(read) > 1e-3 * read.std_error


@pytest.mark.parametrize("rule", sorted(RULES))
def test_a_symmetric_reader_fails_the_shipped_interval(rule: str) -> None:
    estimate = RULES[rule]()

    def witness() -> None:
        read = symmetric_reader(estimate)
        assert (read.lower, read.upper) == pytest.approx(estimate.ci, rel=1e-6)

    expect_failure(witness)


def test_the_t_reference_ratio_and_transform_are_read() -> None:
    curve = np.array([0.5, -0.2, 0.1, -0.4, 0.3, -0.3])
    clustered = make_estimate(
        "ate", 0.3, curve, n=6, cluster=np.array([0, 0, 1, 1, 2, 2]), rule_widened=False
    )
    t_ref = replace(clustered, reference_df=4)
    read = interval_reference(t_ref)
    assert (read.lower, read.upper) == pytest.approx(t_ref.ci, rel=EXACT)
    assert (read.reference, read.df) == ("t", 4.0)
    assert read.upper_half_width == pytest.approx(
        stats.t.ppf(0.975, 4) * t_ref.std_error, rel=EXACT
    )
    ratio = make_estimate(
        "rr", 1.5, curve, n=6, scale="ratio", log_psi=math.log(1.5), rule_widened=False
    )
    assert (interval_reference(ratio).lower, interval_reference(ratio).upper) == pytest.approx(
        ratio.ci, rel=EXACT
    )
    moved = interval_reference(ratio, psi=2.0, std_error=0.1)
    assert (moved.lower, moved.upper) == pytest.approx(
        (2.0 * math.exp(-Z * 0.1), 2.0 * math.exp(Z * 0.1)), rel=1e-9
    )
    logit = replace(
        make_estimate("p", 0.3, curve, n=6, scale="level", rule_widened=False),
        transform=Transform.logit(),
    )
    assert (interval_reference(logit).lower, interval_reference(logit).upper) == pytest.approx(
        logit.ci, rel=EXACT
    )


def test_the_reference_names_its_degrees_of_freedom() -> None:
    skew = interval_reference(_skew_t())
    assert skew.reference == "t"
    assert skew.df == pytest.approx(_skew_t().effective_df)
    assert skew.acceleration == pytest.approx(_skew_t().acceleration)
    assert skew.acceleration != 0.0
    for rule in ("range_logit", "mover"):
        read = interval_reference(RULES[rule]())
        assert (read.reference, read.df, read.acceleration) == ("normal", None, None)


# ------------------------------------------------------------------ an end point and a level


def test_a_one_sided_level_is_the_two_sided_interval_at_twice_its_tail() -> None:
    for build in RULES.values():
        estimate = build()
        one = interval_reference(estimate, level=0.95, sided="one")
        two = interval_reference(estimate, level=0.90)
        assert (one.lower, one.upper) == pytest.approx((two.lower, two.upper), rel=EXACT)
        assert one.sided == "one"


def test_a_moved_skew_t_end_point_reads_the_skew_widened_t() -> None:
    estimate = _skew_t()
    df, acceleration = estimate.effective_df, estimate.acceleration
    assert df is not None and acceleration is not None
    read = interval_reference(estimate, psi=2.5, std_error=0.7)
    expected = skew_widened_ci(2.5, 0.7, 0.05, df=df, acceleration=acceleration)
    assert (read.lower, read.upper) == pytest.approx(expected, rel=EXACT)
    assert _asymmetry(read) > 1e-3


def test_an_end_point_curve_sets_the_skew_degrees_and_acceleration() -> None:
    estimate = _skew_t()
    mirrored = -SKEWED
    read = interval_reference(estimate, psi=1.0, std_error=0.5, influence_curve=mirrored)
    # The mirrored curve has the opposite acceleration, so the widened side flips.
    assert read.acceleration == pytest.approx(-float(estimate.acceleration), rel=EXACT)
    own = interval_reference(estimate, psi=1.0, std_error=0.5)
    assert read.lower_half_width == pytest.approx(own.upper_half_width, rel=EXACT)
    assert read.upper_half_width == pytest.approx(own.lower_half_width, rel=EXACT)


def test_a_moved_logit_mean_reads_the_logit_scale() -> None:
    estimate = _logit_mean()
    read = interval_reference(estimate, psi=0.3, std_error=0.05)
    slope = 1.0 / (0.3 * 0.7)
    centre = math.log(0.3 / 0.7)
    expit = 1.0 / (1.0 + np.exp(-np.array([centre - Z * 0.05 * slope, centre + Z * 0.05 * slope])))
    assert (read.lower, read.upper) == pytest.approx(tuple(expit), rel=1e-9)
    assert read.lower_half_width < read.upper_half_width  # below one half the upper side is longer


def test_a_logit_end_point_off_the_range_has_no_limits() -> None:
    estimate = _logit_mean()
    for psi in (0.0, 1.0, -0.1):
        read = interval_reference(estimate, psi=psi, std_error=0.05)
        assert math.isnan(read.lower) and math.isnan(read.upper)


def test_a_moved_mover_end_point_keeps_each_sides_multiple() -> None:
    estimate = _mover()
    base = interval_reference(estimate)
    read = interval_reference(estimate, psi=0.4, std_error=2.0 * estimate.std_error)
    assert read.interval_scale == "mover"
    assert read.lower_half_width == pytest.approx(2.0 * base.lower_half_width, rel=1e-9)
    assert read.upper_half_width == pytest.approx(2.0 * base.upper_half_width, rel=1e-9)
    assert _asymmetry(read) > 1e-3


def test_end_point_components_read_mover_itself() -> None:
    estimate = _mover()
    one, zero = estimate.components
    shifted = replace(one, psi=0.55)
    moved = replace(estimate, psi=0.45, components=None)
    read = interval_reference(estimate, psi=0.45, components=(shifted, zero))
    expected = with_components(moved, shifted, zero).ci
    assert (read.lower, read.upper) == pytest.approx(expected, rel=EXACT)
    arm_limits = [interval_reference(arm) for arm in (shifted, zero)]
    r = (estimate.variance - one.variance - zero.variance) / (
        -2.0 * math.sqrt(one.variance * zero.variance)
    )
    assert expected == pytest.approx(
        mover_ci(
            0.55,
            (arm_limits[0].lower, arm_limits[0].upper),
            0.1,
            (arm_limits[1].lower, arm_limits[1].upper),
            correlation=r,
        ),
        rel=1e-9,
    )
    with pytest.raises(ValueError, match="components"):
        interval_reference(estimate, psi=0.2, components=(shifted, zero))


def test_the_margin_adds_to_each_side() -> None:
    estimate = _skew_t()
    plain = interval_reference(estimate, psi=2.0, std_error=0.5)
    widened = interval_reference(estimate, psi=2.0, std_error=0.5, margin=0.1)
    assert widened.lower == pytest.approx(plain.lower - 0.1, rel=EXACT)
    assert widened.upper == pytest.approx(plain.upper + 0.1, rel=EXACT)


@pytest.mark.parametrize("status", ["constant_arm_cell_plugin", "out_of_range_plugin"])
def test_a_withheld_interval_gives_no_limits(status: str) -> None:
    withheld = replace(_logit_mean(), inference=status)
    read = interval_reference(withheld)
    assert math.isnan(read.lower) and math.isnan(read.upper)
    assert read.supplies_inference is False
    moved = interval_reference(withheld, psi=0.3, std_error=0.05)
    assert math.isnan(moved.lower) and math.isnan(moved.upper)


def test_invalid_requests_are_refused() -> None:
    estimate = _skew_t()
    with pytest.raises(ValueError, match="level"):
        interval_reference(estimate, level=1.0)
    with pytest.raises(ValueError, match="one-sided"):
        interval_reference(estimate, level=0.4, sided="one")
    with pytest.raises(ValueError, match="declared level"):
        interval_reference(estimate, sided="one")
    with pytest.raises(ValueError, match="std_error"):
        interval_reference(estimate, influence_curve=SKEWED)
    with pytest.raises(ValueError, match="margin"):
        interval_reference(estimate, margin=-1.0)


# ------------------------------------------------------------------ the sensitivity bounds


def _frame(binary: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    w = rng.normal(size=300)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-(w + a - 0.5))))
    else:
        y = w + a + rng.exponential(size=300)
    return pd.DataFrame({"W": w, "A": a, "Y": y})


@pytest.fixture(scope="module")
def binary_fit() -> Any:
    return (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        .fit(_frame(True), outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


@pytest.fixture(scope="module")
def continuous_fit() -> Any:
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        .fit(_frame(False), outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


CASES = [
    ("binary_fit", "ey1", "range_logit"),
    ("binary_fit", "ate", "mover"),
    ("continuous_fit", "ate", "skew_t"),
]


@pytest.mark.parametrize(("fixture", "estimand", "rule"), CASES)
def test_omitted_variable_limits_at_no_confounding_are_the_shipped_interval(
    fixture: str, estimand: str, rule: str, request: pytest.FixtureRequest
) -> None:
    fit = request.getfixturevalue(fixture)
    estimate = fit[estimand]
    assert estimate.interval_scale == rule
    level = 1.0 - estimate.alpha / 2.0
    bounds = omitted_variable_bounds(fit, estimand, cf_y=0.0, cf_d=0.0, level=level)
    assert (bounds.ci_lower, bounds.ci_upper) == pytest.approx(estimate.ci, rel=1e-9)
    assert abs((estimate.ci[1] - estimate.psi) - (estimate.psi - estimate.ci[0])) > 1e-4


@pytest.mark.parametrize(("fixture", "estimand", "rule"), CASES)
def test_a_symmetric_omitted_variable_limit_fails_the_witness(
    fixture: str,
    estimand: str,
    rule: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = request.getfixturevalue(fixture)
    monkeypatch.setattr(limits_module, "interval_reference", symmetric_reader)
    expect_failure(
        lambda: test_omitted_variable_limits_at_no_confounding_are_the_shipped_interval(
            fixture, estimand, rule, request
        )
    )
    assert fit is not None


@pytest.mark.parametrize(("fixture", "estimand", "rule"), CASES)
def test_marginal_limits_at_gamma_one_read_the_shipped_rule(
    fixture: str, estimand: str, rule: str, request: pytest.FixtureRequest
) -> None:
    """At gamma 1 the set interval is the estimate's own rule about the bound's ends.

    The bounds re-estimate the parameter on their own tail regressions, so at gamma 1 the end
    is not the TMLE point and the interval is not ``ci`` itself. It is the rule ``ci`` follows,
    formed about that end, and so asymmetric on all three rules.
    """
    fit = request.getfixturevalue(fixture)
    estimate = fit[estimand]
    (row,) = marginal_sensitivity_bounds(fit, gamma=(1.0,), estimand=estimand).rows
    assert row.lower == pytest.approx(row.upper, rel=1e-9)
    if rule != "skew_t":
        expected = interval_reference(estimate, psi=row.lower, std_error=row.lower_std_error)
        assert (row.lower_limit, row.upper_limit) == pytest.approx(
            (expected.lower, expected.upper), rel=1e-9
        )
    below, above = row.lower - row.lower_limit, row.upper_limit - row.upper
    assert abs(above - below) > 1e-4
    # Each side spans about the multiple of its standard error that the shipped interval does;
    # the end's own curve and spread differ a little from the estimate's.
    low, high = estimate.ci
    assert below / row.lower_std_error == pytest.approx(
        (estimate.psi - low) / estimate.std_error, rel=0.05
    )
    assert above / row.upper_std_error == pytest.approx(
        (high - estimate.psi) / estimate.std_error, rel=0.05
    )


@pytest.mark.parametrize(("fixture", "estimand", "rule"), CASES)
def test_a_symmetric_marginal_limit_fails_the_witness(
    fixture: str,
    estimand: str,
    rule: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(limits_module, "interval_reference", symmetric_reader)
    expect_failure(
        lambda: test_marginal_limits_at_gamma_one_read_the_shipped_rule(
            fixture, estimand, rule, request
        )
    )


def test_omitted_variable_limits_move_continuously_off_no_confounding(binary_fit: Any) -> None:
    """The MOVER multiples carry over to a bound with a bias term, so the limit has no jump."""
    estimate = binary_fit["ate"]
    level = 1.0 - estimate.alpha / 2.0
    lows = [
        omitted_variable_bounds(binary_fit, "ate", cf_y=c, cf_d=c, level=level).ci_lower
        for c in (0.0, 1e-8)
    ]
    assert lows[1] == pytest.approx(lows[0], abs=1e-5)
