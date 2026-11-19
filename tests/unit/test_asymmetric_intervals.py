"""The PR 3 constructions of Fix-3: the curve acceleration, the skew-widened t and correlated MOVER.

The witnesses of plan rev 6, section 9.9, that the declaration needs: ``curve_acceleration``
against a value written out, its symmetry, scale, sign and mask (T26); the skew-widened
interval's sign, shift and widening, each with a deliberate mutation that the written-out
limits reject (T27); the symmetric limit and the cap (T28); MOVER's Wald identity at every
correlation, a fixture with ``r = 0.4`` and asymmetric arms written out, and mutations that drop
the correlation term or swap each arm's sides (T29).  ``ParameterEstimate.acceleration`` is
checked on its rows and its ``None`` cases.  Each component that vanishes at a symmetric or
uncorrelated truth (the acceleration term, the correlation term) has a nonzero witness and a
mutation control here (CLAUDE.md, Scientific changes).

Study 2 (``asymmetric-wald``) passed, so :attr:`~cleverly.inference.ParameterEstimate.ci` reads
these functions (plan section 7, item 8): the flip's own witnesses are in
``tests/unit/test_asymmetric_default.py`` (T30 to T33).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import stats

from cleverly.inference import (
    cornish_fisher_factor,
    curve_acceleration,
    implied_correlation,
    make_estimate,
    mover_ci,
    skew_widened_ci,
    wald_ci,
)
from cleverly.inference import delta as delta_module
from cleverly.inference import influence as influence_module
from tests.studies import canonical_stratified_tmle as stratified_tmle
from tests.studies import stratified_law as l1
from tests.unit._source_mutation import mutate

#: The relative tolerance of an identity that holds in exact arithmetic.
EXACT = 1e-12
Z = float(stats.norm.ppf(0.975))
K = 2.0 * Z * Z + 1.0


def expect_failure(test: Callable[[], None]) -> None:
    """A mutation control: the witness must reject the mutated code."""
    with pytest.raises(AssertionError):
        test()


# ------------------------------------------------------------------------ T26 acceleration


def test_the_acceleration_of_three_ones_is_written_out() -> None:
    # Centred terms (3, -1, -1, -1): sum u^3 = 24, sum u^2 = 12, a = 24 / (6 * 12^1.5).
    assert abs(24.0 / (6.0 * 12.0**1.5) - 0.096225) < 5e-7
    assert curve_acceleration(np.array([3.0, -1.0, -1.0, -1.0])) == pytest.approx(
        24.0 / (6.0 * 12.0**1.5), rel=EXACT
    )


def test_the_acceleration_is_the_sample_skewness_over_six_root_m() -> None:
    curve = np.random.default_rng(3).gamma(2.0, size=500)
    centred = curve - curve.mean()
    skewness = np.mean(centred**3) / np.mean(centred**2) ** 1.5
    assert curve_acceleration(curve) == pytest.approx(
        skewness / (6.0 * math.sqrt(curve.size)), rel=EXACT
    )
    assert curve_acceleration(curve) > 0.0


def test_a_symmetric_curve_has_zero_acceleration() -> None:
    assert curve_acceleration(np.array([2.0, -2.0, 0.5, -0.5, 7.0, -7.0])) == 0.0
    assert curve_acceleration(np.zeros(5)) == 0.0


def test_the_acceleration_is_scale_invariant_and_odd() -> None:
    curve = np.array([3.0, -1.0, -1.0, -1.0, 0.4])
    value = curve_acceleration(curve)
    assert curve_acceleration(7.5 * curve) == pytest.approx(value, rel=EXACT)
    assert curve_acceleration(-curve) == pytest.approx(-value, rel=EXACT)
    assert curve_acceleration(curve + 10.0) == pytest.approx(value, rel=EXACT)


def test_appended_zero_rows_change_nothing_under_the_mask() -> None:
    curve = np.array([3.5, -0.5, -0.5, -0.5])
    padded = np.concatenate([curve, np.zeros(96)])
    assert curve_acceleration(padded, rows=padded != 0.0) == pytest.approx(
        curve_acceleration(curve), rel=EXACT
    )
    # The witness that the mask matters: read unmasked, the zero rows move the centre.
    assert curve_acceleration(padded) != pytest.approx(curve_acceleration(curve), rel=1e-2)


def test_uncentred_acceleration_differs_on_a_curve_off_zero() -> None:
    curve = np.array([3.0, -1.0, -1.0, -1.0]) + 0.5
    assert curve_acceleration(curve, centered=False) != pytest.approx(
        curve_acceleration(curve), rel=1e-2
    )


def test_dropping_the_centring_fails_the_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    mutate(
        monkeypatch,
        influence_module,
        influence_module,
        "curve_acceleration",
        [("values = values - values.mean()", "values = values")],
    )
    monkeypatch.setattr(
        "tests.unit.test_asymmetric_intervals.curve_acceleration",
        influence_module.curve_acceleration,
    )
    expect_failure(test_the_acceleration_is_scale_invariant_and_odd)


# ------------------------------------------------------------- the estimate's acceleration


def _skewed_curve(n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(11)
    curve = np.zeros(n)
    carrying = rng.choice(n, size=120, replace=False)
    curve[carrying] = rng.gamma(1.5, size=120)
    return curve


def test_the_estimate_reads_its_carrying_rows() -> None:
    curve = _skewed_curve()
    estimate = make_estimate("ey1", 1.0, curve, n=curve.size, rule_widened=False)
    assert estimate.acceleration == pytest.approx(
        curve_acceleration(curve, rows=curve != 0.0), rel=EXACT
    )
    assert estimate.acceleration is not None and estimate.acceleration > 0.01
    assert estimate.acceleration != pytest.approx(curve_acceleration(curve), rel=1e-2)
    assert replace(estimate, influence_curve=-curve).acceleration == pytest.approx(
        -estimate.acceleration, rel=EXACT
    )


def test_dropping_the_estimate_mask_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_acceleration",
        [("rows=curve != 0.0", "rows=None")],
    )
    expect_failure(test_the_estimate_reads_its_carrying_rows)


def test_a_second_moment_estimate_reads_its_curve_uncentred() -> None:
    curve = _skewed_curve()
    estimate = make_estimate(
        "ey1", 1.0, curve, n=curve.size, rule_widened=False, covariance_rule="second_moment"
    )
    uncentred = curve_acceleration(curve, rows=curve != 0.0, centered=False)
    assert estimate.acceleration == pytest.approx(uncentred, rel=EXACT)
    # The witness: on these rows, off zero, the two readings differ.
    assert uncentred != pytest.approx(curve_acceleration(curve, rows=curve != 0.0), rel=1e-2)


def test_centring_every_rule_fails_the_second_moment_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_acceleration",
        [('centered=self.covariance_rule != "second_moment"', "centered=True")],
    )
    expect_failure(test_a_second_moment_estimate_reads_its_curve_uncentred)


def test_a_clustered_or_zero_variance_estimate_has_no_acceleration() -> None:
    curve = _skewed_curve()
    clustered = make_estimate(
        "ey1", 1.0, curve, n=curve.size, rule_widened=False, cluster=np.arange(curve.size) // 4
    )
    assert clustered.acceleration is None
    flat = make_estimate("ey1", 1.0, np.zeros(10), n=10, rule_widened=False)
    assert flat.acceleration is None


def test_the_shipped_interval_of_a_continuous_target_is_the_skew_widened_t() -> None:
    curve = _skewed_curve()
    estimate = make_estimate(
        "ey1", 1.0, curve, n=curve.size, rule_widened=False, outcome_family="gaussian"
    )
    assert estimate.acceleration is not None and estimate.effective_df is not None
    assert estimate.ci == skew_widened_ci(
        1.0,
        estimate.std_error,
        0.05,
        df=estimate.effective_df,
        acceleration=estimate.acceleration,
    )
    assert estimate.ci != wald_ci(1.0, estimate.std_error, 0.05)


# --------------------------------------------------------------- T27, T28 skew-widened t


def _t(df: float) -> float:
    return float(stats.t.ppf(0.975, df))


def test_the_factor_is_two_z_squared_plus_one() -> None:
    assert cornish_fisher_factor(0.05) == pytest.approx(8.683, abs=5e-4)
    z = float(stats.norm.ppf(0.95))
    assert cornish_fisher_factor(0.10) == pytest.approx(2 * z * z + 1, rel=EXACT)


def test_a_positive_acceleration_moves_only_the_upper_limit() -> None:
    psi, se, a, nu = 1.3, 0.4, 0.02, 12.0
    low, high = skew_widened_ci(psi, se, df=nu, acceleration=a)
    assert high == pytest.approx(psi + se * (_t(nu) + a * K), rel=EXACT)
    assert low == pytest.approx(psi - se * _t(nu), rel=EXACT)


def test_a_negative_acceleration_moves_only_the_lower_limit() -> None:
    psi, se, a, nu = 1.3, 0.4, -0.03, 12.0
    low, high = skew_widened_ci(psi, se, df=nu, acceleration=a)
    assert low == pytest.approx(psi - se * (_t(nu) - a * K), rel=EXACT)
    assert high == pytest.approx(psi + se * _t(nu), rel=EXACT)


def test_the_interval_contains_both_hull_members() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        psi, se = rng.normal(), rng.uniform(0.1, 2.0)
        a, nu = rng.normal(scale=0.05), rng.uniform(5.0, 60.0)
        low, high = skew_widened_ci(psi, se, df=nu, acceleration=a)
        t_low, t_high = wald_ci(psi, se, df=nu)
        cf_low, cf_high = psi - se * (_t(nu) - a * K), psi + se * (_t(nu) + a * K)
        assert low <= min(t_low, cf_low) + 1e-12 and high >= max(t_high, cf_high) - 1e-12
        assert low == pytest.approx(min(t_low, cf_low), rel=EXACT, abs=EXACT)
        assert high == pytest.approx(max(t_high, cf_high), rel=EXACT, abs=EXACT)


def test_the_margin_adds_to_each_side() -> None:
    low, high = skew_widened_ci(1.0, 0.5, df=12.0, acceleration=0.02, margin=0.1)
    base_low, base_high = skew_widened_ci(1.0, 0.5, df=12.0, acceleration=0.02)
    assert (low, high) == pytest.approx((base_low - 0.1, base_high + 0.1), rel=EXACT)


_POSITIVE = "test_a_positive_acceleration_moves_only_the_upper_limit"
_NEGATIVE = "test_a_negative_acceleration_moves_only_the_lower_limit"


@pytest.mark.parametrize(
    ("edit", "rejected_by"),
    [
        # The sign: widen the side the skew does not point to (one exchange of the two signs).
        (
            [
                ("critical - shift)", "critical @@ shift)"),
                ("critical + shift)", "critical - shift)"),
                ("critical @@ shift)", "critical + shift)"),
            ],
            {_POSITIVE, _NEGATIVE},
        ),
        # A two-sided shift instead of the hull: it narrows the lower side when a > 0, and
        # equals the hull's lower side when a < 0.
        ([("max(critical, critical - shift)", "(critical - shift)")], {_POSITIVE}),
        # No widening at all.
        (
            [("shift = float(acceleration) * cornish_fisher_factor(alpha)", "shift = 0.0")],
            {_POSITIVE, _NEGATIVE},
        ),
    ],
    ids=["sign", "two-sided shift", "no widening"],
)
def test_each_part_of_the_construction_is_needed(
    monkeypatch: pytest.MonkeyPatch, edit: list[tuple[str, str]], rejected_by: set[str]
) -> None:
    mutate(monkeypatch, delta_module, delta_module, "skew_widened_ci", edit)
    monkeypatch.setattr(
        "tests.unit.test_asymmetric_intervals.skew_widened_ci", delta_module.skew_widened_ci
    )
    failed = set()
    for name in (_POSITIVE, _NEGATIVE):
        try:
            globals()[name]()
        except AssertionError:
            failed.add(name)
    assert failed == rejected_by


def test_zero_acceleration_is_the_capped_t() -> None:
    assert skew_widened_ci(1.0, 0.5, df=12.0, acceleration=0.0) == pytest.approx(
        wald_ci(1.0, 0.5, df=12.0), rel=EXACT
    )


def test_an_estimate_with_three_df_reads_the_cap() -> None:
    curve = np.array([3.0, -1.0, -1.0, -1.0])
    estimate = make_estimate("ey1", 0.5, curve, n=4, rule_widened=False, outcome_family="gaussian")
    assert estimate.small_sample_df == pytest.approx(6.0)
    tiny = make_estimate(
        "ey1",
        0.5,
        np.array([10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        n=11,
        rule_widened=False,
        outcome_family="gaussian",
    )
    assert tiny.small_sample_df is not None and tiny.small_sample_df < 5.0
    assert tiny.effective_df == 5.0
    assert tiny.acceleration is not None
    low, high = skew_widened_ci(
        tiny.psi, tiny.std_error, df=tiny.effective_df, acceleration=tiny.acceleration
    )
    assert low == pytest.approx(tiny.psi - tiny.std_error * _t(5.0), rel=EXACT)
    assert high == pytest.approx(
        tiny.psi + tiny.std_error * (_t(5.0) + tiny.acceleration * K), rel=EXACT
    )


def test_non_finite_inputs_give_no_interval() -> None:
    assert all(math.isnan(x) for x in skew_widened_ci(1.0, math.nan, df=5.0, acceleration=0.1))
    assert all(math.isnan(x) for x in skew_widened_ci(1.0, 0.5, df=5.0, acceleration=math.nan))


# ------------------------------------------------------------------------------ T29 MOVER


def _symmetric(psi: float, se: float) -> tuple[float, float]:
    return (psi - Z * se, psi + Z * se)


@pytest.mark.parametrize("r", [-0.5, 0.0, 0.5])
def test_symmetric_arms_give_the_wald_interval(r: float) -> None:
    p1, s1, p0, s0 = 0.62, 0.05, 0.31, 0.03
    se = math.sqrt(s1**2 + s0**2 - 2 * r * s1 * s0)
    low, high = mover_ci(p1, _symmetric(p1, s1), p0, _symmetric(p0, s0), correlation=r)
    assert (low, high) == pytest.approx(wald_ci(p1 - p0, se), rel=EXACT)


def _asymmetric_arms() -> tuple[float, tuple[float, float], float, tuple[float, float]]:
    return 0.9, (0.78, 0.95), 0.2, (0.16, 0.27)


def test_a_correlated_fixture_matches_the_display() -> None:
    p1, (l1, u1), p0, (l0, u0) = _asymmetric_arms()
    r = 0.4
    low, high = mover_ci(p1, (l1, u1), p0, (l0, u0), correlation=r)
    d = p1 - p0
    expected_low = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2 - 2 * r * (p1 - l1) * (u0 - p0))
    expected_high = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2 - 2 * r * (u1 - p1) * (p0 - l0))
    assert (low, high) == pytest.approx((expected_low, expected_high), rel=EXACT)
    # Numbers written out: L = 0.7 - sqrt(0.0193 - 0.00672), U = 0.7 + sqrt(0.0041 - 0.0016).
    assert low == pytest.approx(0.7 - math.sqrt(0.01258), rel=1e-10)
    assert high == pytest.approx(0.7 + math.sqrt(0.0025), rel=1e-10)
    # The nonzero witness of the correlation term: it moves each limit.
    flat_low, flat_high = mover_ci(p1, (l1, u1), p0, (l0, u0), correlation=0.0)
    assert abs(flat_low - low) > 0.02 and abs(flat_high - high) > 0.005


def test_swapped_arm_sides_differ_from_the_display() -> None:
    p1, (l1, u1), p0, (l0, u0) = _asymmetric_arms()
    low, high = mover_ci(p1, (l1, u1), p0, (l0, u0), correlation=0.4)
    swapped = mover_ci(
        p1, (2 * p1 - u1, 2 * p1 - l1), p0, (2 * p0 - u0, 2 * p0 - l0), correlation=0.4
    )
    assert swapped[0] != pytest.approx(low, rel=1e-3)
    assert swapped[1] != pytest.approx(high, rel=1e-3)


@pytest.mark.parametrize(
    ("edit", "witness"),
    [
        (
            [
                ("- 2.0 * correlation * down_1 * up_0", ""),
                ("- 2.0 * correlation * up_1 * down_0", ""),
            ],
            "test_a_correlated_fixture_matches_the_display",
        ),
        (
            [
                (
                    "down_1, up_1 = minuend - low_1, high_1 - minuend",
                    "up_1, down_1 = minuend - low_1, high_1 - minuend",
                ),
                (
                    "down_0, up_0 = subtrahend - low_0, high_0 - subtrahend",
                    "up_0, down_0 = subtrahend - low_0, high_0 - subtrahend",
                ),
            ],
            "test_a_correlated_fixture_matches_the_display",
        ),
        (
            [("- 2.0 * correlation * down_1 * up_0", "+ 2.0 * correlation * down_1 * up_0")],
            "test_symmetric_arms_give_the_wald_interval",
        ),
    ],
    ids=["drop the correlation term", "swap each arm's sides", "flip the correlation sign"],
)
def test_each_part_of_mover_is_needed(
    monkeypatch: pytest.MonkeyPatch, edit: list[tuple[str, str]], witness: str
) -> None:
    mutate(monkeypatch, delta_module, delta_module, "mover_ci", edit)
    monkeypatch.setattr("tests.unit.test_asymmetric_intervals.mover_ci", delta_module.mover_ci)
    test = globals()[witness]
    if witness == "test_symmetric_arms_give_the_wald_interval":
        expect_failure(lambda: test(0.5))
    else:
        expect_failure(test)


def test_mover_refuses_an_impossible_correlation_or_interval() -> None:
    with pytest.raises(ValueError, match="correlation"):
        mover_ci(0.5, (0.4, 0.6), 0.3, (0.2, 0.4), correlation=1.2)
    with pytest.raises(ValueError, match="contain"):
        mover_ci(0.5, (0.55, 0.6), 0.3, (0.2, 0.4), correlation=0.0)
    assert all(
        math.isnan(x) for x in mover_ci(0.5, (math.nan, 0.6), 0.3, (0.2, 0.4), correlation=0.0)
    )


def test_the_implied_correlation_is_the_curve_covariance() -> None:
    rng = np.random.default_rng(2)
    n = 300
    one = rng.normal(size=n)
    zero = 0.6 * one + rng.normal(size=n)
    est_one = make_estimate("ey1", 0.0, one, n=n, rule_widened=False)
    est_zero = make_estimate("ey0", 0.0, zero, n=n, rule_widened=False)
    diff = make_estimate("ate", 0.0, one - zero, n=n, rule_widened=False)
    r = implied_correlation(diff.variance, est_one.variance, est_zero.variance)
    assert r == pytest.approx(np.corrcoef(one, zero)[0, 1], rel=1e-10)
    assert r > 0.3
    assert implied_correlation(1.0, 1.0, 1.0) == 0.5
    assert implied_correlation(10.0, 1.0, 1.0) == -1.0
    assert math.isnan(implied_correlation(1.0, 0.0, 1.0))


@pytest.fixture(scope="module")
def l1_fit() -> Any:
    return stratified_tmle.fit_cleverly(l1.sample(1_500, 20261113))


def test_symmetric_arms_of_a_fit_give_its_difference_interval(l1_fit: Any) -> None:
    for tag in ("[V=0]", "[V=1]", "[V=2]"):
        one, zero, diff = l1_fit[f"ey[1]{tag}"], l1_fit[f"ey[0]{tag}"], l1_fit[f"ate{tag}"]
        r = implied_correlation(diff.variance, one.variance, zero.variance)
        assert -1.0 < r < 1.0
        low, high = mover_ci(
            one.psi,
            _symmetric(one.psi, one.std_error),
            zero.psi,
            _symmetric(zero.psi, zero.std_error),
            correlation=r,
        )
        expected = wald_ci(diff.psi, diff.std_error)
        assert low == pytest.approx(expected[0], rel=1e-10, abs=1e-12)
        assert high == pytest.approx(expected[1], rel=1e-10, abs=1e-12)


def test_the_shipped_arm_intervals_make_mover_asymmetric(l1_fit: Any) -> None:
    one, zero, diff = l1_fit["ey[1][V=2]"], l1_fit["ey[0][V=2]"], l1_fit["ate[V=2]"]
    assert one.interval_scale == "range_logit"
    r = implied_correlation(diff.variance, one.variance, zero.variance)
    low, high = mover_ci(one.psi, one.ci, zero.psi, zero.ci, correlation=r)
    assert low < diff.psi < high
    assert abs((diff.psi - low) - (high - diff.psi)) > 1e-4
    # The difference's shipped interval is this MOVER interval (plan 7 item 8).
    assert diff.interval_scale == "mover"
    assert diff.ci == pytest.approx((low, high), rel=EXACT)
