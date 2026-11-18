"""The shipped asymmetric intervals of Fix-3 PR 3: the default flip and its tests (T30 to T33).

Study 2 (``asymmetric-wald``) passed every gated cell with no falsifier fired, so the default
reads the constructions that ``tests/unit/test_asymmetric_intervals.py`` witnesses (plan rev 6,
sections 9.2, 9.3 and 9.9):

* a bounded difference of two bounded means of one fit records them as ``components`` and takes
  correlated MOVER on their logit intervals (``interval_scale == "mover"``);
* an unclustered estimate of a continuous outcome takes the skew-widened t at
  ``max(satterthwaite_df, 5)`` with its curve's acceleration (``interval_scale == "skew_t"``);
* each p-value inverts its interval (T31), and the summaries print the df column and the notes
  (T33).

T30 checks the guards: a withheld arm keeps the difference withheld, a difference without
components keeps the identity interval and says so. T32 checks the scope: a binary-outcome
coefficient, a family-less contrast, a ratio and a clustered estimate keep their intervals.
Each component that vanishes at a symmetric or uncorrelated truth has a nonzero witness and a
mutation control (CLAUDE.md, Scientific changes).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.inference import (
    ParameterEstimate,
    implied_correlation,
    make_estimate,
    mover_ci,
    wald_ci,
    with_components,
)
from cleverly.inference import influence as influence_module
from cleverly.inference.influence import _range_logit_limits
from tests.studies import canonical_stratified_tmle as stratified_tmle
from tests.studies import stratified_law as l1
from tests.unit._source_mutation import mutate

EXACT = 1e-12


def _mean(name: str, psi: float, curve: np.ndarray) -> ParameterEstimate:
    return make_estimate(
        name,
        psi,
        curve,
        n=curve.size,
        scale="level",
        rule_widened=False,
        bounds=(0.0, 1.0),
        outcome_family="binomial",
    )


def _arms(
    seed: int, n: int = 300, p1: float = 0.92, p0: float = 0.4, share: float = 0.0
) -> tuple[ParameterEstimate, ParameterEstimate, ParameterEstimate]:
    """Two bounded arm means and their difference, curves sharing a ``share`` component."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n)
    one = share * common + rng.normal(size=n) * math.sqrt(p1 * (1 - p1) * n) / math.sqrt(n)
    zero = share * common + rng.normal(size=n) * math.sqrt(p0 * (1 - p0) * n) / math.sqrt(n)
    one, zero = one - one.mean(), zero - zero.mean()
    minuend, subtrahend = _mean("ey1", p1, one), _mean("ey0", p0, zero)
    difference = make_estimate(
        "ate",
        p1 - p0,
        one - zero,
        n=n,
        rule_widened=False,
        bounds=(-1.0, 1.0),
        outcome_family="binomial",
    )
    return minuend, subtrahend, with_components(difference, minuend, subtrahend)


def _mover_of(difference: ParameterEstimate, alpha: float | None = None) -> tuple[float, float]:
    """The display of plan 9.2 written out from the components."""
    assert difference.components is not None
    one, zero = difference.components
    level = difference.alpha if alpha is None else alpha
    l1_, u1 = _range_logit_limits(one.psi, one.std_error, level)
    l0, u0 = _range_logit_limits(zero.psi, zero.std_error, level)
    r = implied_correlation(difference.variance, one.variance, zero.variance)
    d = one.psi - zero.psi
    low = d - math.sqrt(
        (one.psi - l1_) ** 2 + (u0 - zero.psi) ** 2 - 2 * r * (one.psi - l1_) * (u0 - zero.psi)
    )
    high = d + math.sqrt(
        (u1 - one.psi) ** 2 + (zero.psi - l0) ** 2 - 2 * r * (u1 - one.psi) * (zero.psi - l0)
    )
    return low, high


def _gaussian(seed: int, n: int = 200, psi: float = 1.0) -> ParameterEstimate:
    rng = np.random.default_rng(seed)
    curve = rng.gamma(1.5, 1.0, n)
    curve = curve - curve.mean()
    return make_estimate(
        "ey1", psi, curve, n=n, scale="level", rule_widened=False, outcome_family="gaussian"
    )


# ------------------------------------------------------------------------ the default


def test_a_difference_with_components_takes_mover() -> None:
    _, _, difference = _arms(1, share=0.6)
    assert difference.components is not None
    assert difference.interval_scale == "mover"
    r = implied_correlation(*(e.variance for e in (difference, *difference.components)))
    # A nonzero correlation witness: the shared component makes r far from 0, so the term
    # the display carries is exercised.
    assert r > 0.2
    low, high = difference.ci
    assert (low, high) == pytest.approx(_mover_of(difference), rel=EXACT)
    one, zero = difference.components
    assert (low, high) == pytest.approx(
        mover_ci(one.psi, one.ci, zero.psi, zero.ci, correlation=r), rel=EXACT
    )
    # The arm near 1 makes the interval asymmetric, unlike the identity interval.
    assert abs((difference.psi - low) - (high - difference.psi)) > 1e-3


def test_dropping_the_correlation_term_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, difference = _arms(2, share=0.6)
    expected = _mover_of(difference)
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_mover_interval",
        [("correlation=correlation\n", "correlation=0.0\n")],
    )
    assert replace(difference).ci != pytest.approx(expected, rel=1e-6)


def test_swapping_the_arms_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, difference = _arms(3)
    expected = _mover_of(difference)
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_mover_interval",
        [("minuend, subtrahend = arms\n", "subtrahend, minuend = arms\n")],
    )
    with pytest.raises((AssertionError, ValueError)):
        assert replace(difference).ci == pytest.approx(expected, rel=1e-6)


def test_the_components_need_the_curve_identity() -> None:
    minuend, subtrahend, difference = _arms(4)
    assert difference.components is not None
    # The reverse order is not the difference, and a curve that differs is refused.
    bare = replace(difference, components=None)
    assert with_components(bare, subtrahend, minuend).components is None
    shifted = replace(bare, influence_curve=bare.influence_curve + 1e-3)
    assert with_components(shifted, minuend, subtrahend).components is None


def test_a_continuous_target_takes_the_skew_widened_t() -> None:
    estimate = _gaussian(5)
    assert estimate.interval_scale == "skew_t"
    assert estimate.acceleration is not None and estimate.acceleration > 0.01
    df = max(float(estimate.small_sample_df), 5.0)  # type: ignore[arg-type]
    t = float(stats.t.ppf(0.975, df))
    k = 2 * float(stats.norm.ppf(0.975)) ** 2 + 1
    low, high = estimate.ci
    assert low == pytest.approx(estimate.psi - estimate.std_error * t, rel=EXACT)
    assert high == pytest.approx(
        estimate.psi + estimate.std_error * (t + estimate.acceleration * k), rel=EXACT
    )


def test_dropping_the_widening_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    estimate = _gaussian(6)
    expected = estimate.ci
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_asymmetric_interval",
        [("acceleration=acceleration,\n", "acceleration=0.0,\n")],
    )
    assert replace(estimate).ci[1] < expected[1] - 1e-6


def test_the_widening_adds_the_stopping_rule_margin() -> None:
    estimate = replace(_gaussian(7), rule_widened=True)
    estimate = replace(estimate, influence_curve=estimate.influence_curve + 0.01)
    low, high = estimate.ci
    bare = replace(estimate, rule_widened=False)
    blow, bhigh = bare.ci
    margin = abs(estimate.score)
    assert margin > 0
    assert (blow - low, high - bhigh) == pytest.approx((margin, margin), rel=1e-9)


def test_a_fit_records_the_components_of_its_difference() -> None:
    fit = stratified_tmle.fit_cleverly(l1.sample(1_500, 20261114))
    for tag in ("", "[V=0]", "[V=1]", "[V=2]"):
        name = f"ate{tag}"
        if name not in fit:
            continue
        diff = fit[name]
        assert diff.interval_scale == "mover", name
        one, zero = diff.components
        assert one.psi == pytest.approx(fit[f"ey[1]{tag}"].psi, rel=EXACT)
        assert zero.psi == pytest.approx(fit[f"ey[0]{tag}"].psi, rel=EXACT)
        assert one.std_error == pytest.approx(fit[f"ey[1]{tag}"].std_error, rel=1e-10)


def _frame(seed: int, n: int = 500, binary: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * a + w - 1)))).astype(float)
    else:
        y = 0.5 * a + w + rng.gamma(2.0, 1.0, n)
    return pd.DataFrame({"W": w, "A": a, "Y": y})


def _point_fit(frame: pd.DataFrame, binary: bool) -> Any:
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))
    return study.identify(ATE()).estimate(
        outcome_learner=LogisticRegression() if binary else LinearRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=False,
        random_state=0,
    )


def test_a_point_treatment_fit_reads_both_rules() -> None:
    assert _point_fit(_frame(8, binary=True), True)["ate"].interval_scale == "mover"
    assert _point_fit(_frame(8), False)["ate"].interval_scale == "skew_t"


# ------------------------------------------------------------------------ T30 guards


def test_a_withheld_arm_keeps_the_difference_withheld(monkeypatch: pytest.MonkeyPatch) -> None:
    minuend, subtrahend, difference = _arms(9)
    withheld = replace(difference, inference="constant_arm_cell_plugin")
    calls: list[Any] = []
    original = influence_module._range_logit_limits

    def recorded(*args: Any) -> Any:
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(influence_module, "_range_logit_limits", recorded)
    low, high = withheld.plugin_interval
    assert math.isnan(low) and math.isnan(high)
    assert calls == []
    # An arm that is withheld or on its bound is never mapped; the difference falls back.
    on_bound = replace(difference, components=(replace(minuend, psi=1.0), subtrahend))
    assert on_bound.interval_scale == "identity"


def test_a_difference_without_components_says_identity() -> None:
    _, _, difference = _arms(10)
    bare = replace(difference, components=None)
    assert bare.interval_scale == "identity"
    assert bare.ci == wald_ci(bare.psi, bare.std_error, 0.05)


def test_stale_components_fall_back_to_identity() -> None:
    _, _, difference = _arms(11)
    moved = replace(difference, psi=difference.psi + 0.01)
    assert moved.interval_scale == "identity"


# ------------------------------------------------------------------------ T31 duality


ALPHAS = (0.01, 0.05, 0.10, 0.32)


def _fixtures() -> list[ParameterEstimate]:
    out: list[ParameterEstimate] = []
    for seed in range(100):
        out.append(_gaussian(100 + seed, n=60 + seed))
    rng = np.random.default_rng(7)
    for seed in range(100):
        p1 = float(rng.uniform(0.6, 0.97))
        p0 = float(rng.uniform(0.05, 0.6))
        # Some with a large correlation and unequal arms (plan 9.2, monotonicity).
        share = 0.7 if seed % 4 == 0 else float(rng.uniform(0.0, 0.5))
        out.append(_arms(200 + seed, n=80 + seed, p1=p1, p0=p0, share=share)[2])
    return out


@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_p_value_inverts_the_interval(alpha: float) -> None:
    large = 0
    for estimate in _fixtures():
        assert estimate.interval_scale in ("mover", "skew_t")
        if estimate.components is not None:
            one, zero = estimate.components
            r = implied_correlation(estimate.variance, one.variance, zero.variance)
            large += r > 0.7
        low, high = replace(estimate, alpha=alpha).ci
        span = high - low
        for null in np.linspace(low - 0.5 * span, high + 0.5 * span, 21):
            if min(abs(null - low), abs(null - high)) < 1e-9 * span:
                continue
            pvalue = estimate.wald_test(null=float(null)).pvalue
            assert (pvalue < alpha) == (not low <= null <= high), (estimate.name, null)
    assert large >= 10


def test_a_p_value_below_alpha_always_means_outside() -> None:
    # Near r = 1 with unequal arms a MOVER limit need not move monotonically with the level
    # (plan 9.2): above r of about 0.87 some fixtures turn. The inverted p-value is the
    # largest level at which the interval still holds the null, so a p-value below alpha
    # still means the null is outside the level-alpha interval.
    turned = 0
    for seed in range(40):
        difference = _arms(400 + seed, n=90, p1=0.95, p0=0.3, share=2.5)[2]
        lows = [replace(difference, alpha=a).ci[0] for a in np.linspace(0.005, 0.5, 40)]
        turned += bool(np.any(np.diff(lows) < -1e-12))
        for alpha in ALPHAS:
            low, high = replace(difference, alpha=alpha).ci
            for null in np.linspace(low - 0.2, high + 0.2, 25):
                if difference.wald_test(null=float(null)).pvalue < alpha:
                    assert not low <= null <= high
    assert turned > 0


def test_the_statistic_is_the_normal_quantile_of_the_p_value() -> None:
    estimate = _gaussian(12)
    test = estimate.wald_test(null=estimate.psi - 0.3)
    assert test.statistic == pytest.approx(stats.norm.isf(test.pvalue / 2), rel=1e-9)
    assert test.statistic > 0
    below = estimate.wald_test(null=estimate.psi + 0.3)
    assert below.statistic < 0
    assert test.interval_scale == "skew_t"


def test_the_skewed_side_needs_a_farther_null() -> None:
    # A nonzero witness that the test is asymmetric: at the same distance, the null on the
    # widened side has the larger p-value.
    estimate = _gaussian(13)
    assert estimate.acceleration is not None and estimate.acceleration > 0
    step = 2.0 * estimate.std_error
    assert (
        estimate.wald_test(null=estimate.psi + step).pvalue
        > estimate.wald_test(null=estimate.psi - step).pvalue
    )


# ------------------------------------------------------------------------ T32 scope


def test_a_binary_coefficient_keeps_z() -> None:
    curve = np.random.default_rng(14).gamma(1.5, 1.0, 200)
    curve -= curve.mean()
    coefficient = make_estimate(
        "msm[A]", 0.3, curve, n=200, scale="level", rule_widened=False, outcome_family="binomial"
    )
    assert coefficient.interval_scale == "identity"
    assert coefficient.ci == wald_ci(0.3, coefficient.std_error, 0.05)


def test_a_family_less_contrast_keeps_z() -> None:
    estimate = replace(_gaussian(15), outcome_family=None)
    assert estimate.interval_scale == "identity"
    assert estimate.ci == wald_ci(estimate.psi, estimate.std_error, 0.05)


def test_a_ratio_keeps_its_log_interval() -> None:
    base = _gaussian(16)
    ratio = replace(base, scale="ratio", psi=math.exp(0.2), log_psi=0.2)
    assert ratio.interval_scale == "log"
    low, high = wald_ci(0.2, ratio.std_error, 0.05)
    assert ratio.ci == pytest.approx((math.exp(low), math.exp(high)), rel=EXACT)


def test_a_clustered_estimate_keeps_its_reference() -> None:
    rng = np.random.default_rng(17)
    curve = rng.gamma(1.5, 1.0, 120)
    curve -= curve.mean()
    clustered = make_estimate(
        "ey1",
        1.0,
        curve,
        n=120,
        cluster=np.repeat(np.arange(20), 6),
        scale="level",
        rule_widened=False,
        reference_df=18,
        outcome_family="gaussian",
    )
    assert clustered.interval_scale == "identity"
    assert clustered.ci == wald_ci(1.0, clustered.std_error, 0.05, df=18)


# ------------------------------------------------------------------------ T33 summaries


def test_the_summary_prints_the_df_column_and_notes() -> None:
    fit = _point_fit(_frame(18), False)
    text = str(fit.summary())
    estimate = fit["ate"]
    assert " df" in text
    assert f"{float(estimate.effective_df):.1f}" in text
    assert "skew_widened" in fit.summary().note_codes or "widened further" in text
    one = str(estimate.summary(detail="full"))
    assert "skew-widened" in one


def test_the_summary_names_mover() -> None:
    fit = _point_fit(_frame(19, binary=True), True)
    assert "mover" in fit.summary().note_codes
    assert "MOVER (range-logit arms)" in str(fit["ate"].summary())
    assert "inverts it" in str(fit["ate"].wald_test(null=0.0).summary(detail="full"))


# ------------------------------------------------------------------------ parity with study 2


@pytest.mark.parametrize("label", ["student", "identity_msm", "boundary_2000", "ho_sparse_treat"])
def test_the_default_is_the_interval_study_two_measured(label: str) -> None:
    from tests.studies import asymmetric_wald as study

    group = study.GROUP_BY_LABEL[label]
    result = study.fit_group(group, study.sample(group.law, group.n, 2**42 + 1))
    for cell in group.cells:
        estimate = result[cell.estimand]
        names = study.arm_names(cell.estimand)
        arms = None
        if group.family == "binomial" and names is not None:
            arms = (result[names[0]], result[names[1]])
        if study.withheld_trigger(estimate, arms):
            continue
        low, high, construction = study.rule_interval(estimate, family=group.family, arms=arms)
        assert estimate.interval_scale == {"range_logit": "range_logit"}.get(
            construction, construction
        )
        assert estimate.ci == pytest.approx((low, high), rel=1e-10, abs=1e-12), cell.name


# ------------------------------------------------------------------------ m5: one scale


def test_a_bounded_continuous_copy_keeps_one_scale() -> None:
    """A hand-built copy with ``bounds`` (0, 1) on a continuous estimate takes the range-logit
    scale for its scale, its interval and its test alike."""
    estimate = replace(_gaussian(21, psi=0.4), bounds=(0.0, 1.0))
    assert estimate.interval_scale == "range_logit"
    low, high = estimate.ci
    expected = _range_logit_limits(estimate.psi, estimate.std_error, 0.05)
    assert (low, high) == pytest.approx(expected, rel=1e-12)
    assert estimate.wald_test(null=estimate.psi).pvalue == pytest.approx(1.0)


def test_dropping_the_range_check_splits_the_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: without the range check the interval is the skew-widened t again
    while the scale still reads range-logit, and the test of the estimate itself rejects."""
    estimate = replace(_gaussian(21, psi=0.4), bounds=(0.0, 1.0))
    expected = estimate.ci
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_skew_reference",
        [("or self._range_scale() is not None\n", "")],
    )
    broken = replace(estimate)
    assert broken.interval_scale == "range_logit"
    assert broken.ci != pytest.approx(expected, rel=1e-6)
    assert broken.wald_test(null=broken.psi).pvalue < 1e-6


# ------------------------------------------------------------------------ M3: model means and curves


def _skewed_levels(seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.integers(0, 3, n).astype(float)
    y = 0.5 * a + 0.3 * w + rng.lognormal(0.0, 0.9, n)
    return pd.DataFrame({"A": a, "W": w, "Y": y})


@pytest.fixture(scope="module")
def msm_fit() -> Any:
    from cleverly import MSMProjection
    from cleverly.msm import MSM

    study = CausalStudy(
        _skewed_levels(22), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    return study.identify(MSMProjection(MSM.linear())).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=0,
    )


def _hand_level_intervals(fit: Any) -> list[tuple[float, float, float]]:
    """Each level's skew-widened t of ``phi' IC``, built from the coefficients alone."""
    from cleverly.inference import curve_acceleration, satterthwaite_df, skew_widened_ci

    msm = fit.nuisance.msm
    names = [name for name in fit.estimates if name.startswith("msm[")]
    estimates = [fit.estimates[name] for name in names]
    beta = np.array([e.psi for e in estimates])
    covariance = np.asarray(fit.covariance(names), dtype=float)
    curves = np.column_stack([np.asarray(e.influence_curve, dtype=float) for e in estimates])
    out = []
    for phi in np.asarray(msm.design, dtype=float)[0]:
        curve = curves @ phi
        rows = curve != 0.0
        acceleration = curve_acceleration(curve, rows=rows, centered=True)
        low, high = skew_widened_ci(
            float(phi @ beta),
            float(np.sqrt(phi @ covariance @ phi)),
            0.05,
            df=max(satterthwaite_df(curve, rows=rows, centered=True), 5.0),
            acceleration=acceleration,
        )
        out.append((low, high, acceleration))
    return out


def test_the_working_model_means_read_the_skew_widened_t(msm_fit: Any) -> None:
    from cleverly.msm import working_model_means

    means = working_model_means(msm_fit)
    assert means.lower is not None and means.upper is not None
    intercept = msm_fit.estimates["msm[(intercept)]"]
    assert intercept.interval_scale == "skew_t"
    # Level 0 is the intercept itself, so its interval is the intercept's own.
    assert (means.lower[0], means.upper[0]) == pytest.approx(intercept.ci, rel=1e-12)
    hand = _hand_level_intervals(msm_fit)
    for j, (low, high, acceleration) in enumerate(hand):
        assert abs(acceleration) > 0.005, "nonzero witness: the widening must act"
        assert (means.lower[j], means.upper[j]) == pytest.approx((low, high), rel=1e-12)
        centre = float(means.estimate[j])
        assert abs((centre - means.lower[j]) - (means.upper[j] - centre)) > 1e-4


@pytest.mark.parametrize(
    "edit",
    [
        (
            "acceleration=curve_acceleration(curve, rows=rows, centered=centered),",
            "acceleration=0.0,",
        ),
        (
            "acceleration=curve_acceleration(curve, rows=rows, centered=centered),",
            "acceleration=-curve_acceleration(curve, rows=rows, centered=centered),",
        ),
        (
            'if all(estimate.interval_scale == "skew_t" for estimate in estimates):',
            "if False:",
        ),
    ],
    ids=["no-widening", "sign", "gate"],
)
def test_a_broken_model_mean_widening_is_caught(
    msm_fit: Any, monkeypatch: pytest.MonkeyPatch, edit: tuple[str, str]
) -> None:
    from cleverly import msm as msm_module

    hand = _hand_level_intervals(msm_fit)
    mutate(monkeypatch, msm_module, msm_module, "working_model_means", [edit])
    means = msm_module.working_model_means(msm_fit)
    assert means.lower is not None and means.upper is not None
    assert any(
        (means.lower[j], means.upper[j]) != pytest.approx((low, high), rel=1e-9)
        for j, (low, high, _) in enumerate(hand)
    )


def _curve_fit(binary: bool) -> Any:
    from cleverly import ConditionalContrast
    from cleverly.bases import Linear, Modifiers

    rng = np.random.default_rng(23)
    n = 600
    w = rng.uniform(0.0, 2.0, n)
    a = rng.binomial(1, 1 / (1 + np.exp(-(w - 1.0))))
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * a * w + 0.5 * w - 1.0)))).astype(float)
    else:
        y = a * (0.5 + 0.4 * w) + 0.3 * w + rng.lognormal(0.0, 0.9, n)
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))
    contrast = ConditionalContrast(Modifiers(["W"], domains={"W": (0.0, 2.0)}), model=Linear(["W"]))
    return study.identify(contrast).estimate(
        outcome_learner=LogisticRegression() if binary else LinearRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=False,
        random_state=0,
    )


@pytest.fixture(scope="module")
def curve_fit() -> Any:
    return _curve_fit(binary=False)


ROWS = {"W": np.array([0.1, 1.0, 1.9])}


def _hand_curve_rows(curve: Any) -> list[tuple[float, float, float]]:
    from cleverly.inference import curve_acceleration, satterthwaite_df, skew_widened_ci

    model = curve.model
    estimate, se = model.predict(ROWS), model.std_error(ROWS)
    influence = np.asarray(model.influence, dtype=float) @ np.asarray(model.design(ROWS)).T
    out = []
    for i in range(len(estimate)):
        column = influence[:, i]
        rows = column != 0.0
        acceleration = curve_acceleration(column, rows=rows, centered=True)
        low, high = skew_widened_ci(
            float(estimate[i]),
            float(se[i]),
            0.05,
            df=max(satterthwaite_df(column, rows=rows, centered=True), 5.0),
            acceleration=acceleration,
        )
        out.append((low, high, acceleration))
    return out


def test_an_effect_curve_row_reads_the_skew_widened_t(curve_fit: Any) -> None:
    curve = curve_fit.effect_curve(random_state=0)
    assert curve.model.skew_widened()
    frame = curve.predict(ROWS, interval=True)
    for i, (low, high, acceleration) in enumerate(_hand_curve_rows(curve)):
        assert abs(acceleration) > 0.003, "nonzero witness: the widening must act"
        assert (frame["ci_lower"][i], frame["ci_upper"][i]) == pytest.approx((low, high), rel=1e-12)


@pytest.mark.parametrize(
    "edit",
    [
        ("acceleration=acceleration,", "acceleration=0.0,"),
        ("acceleration=acceleration,", "acceleration=-acceleration,"),
        ("if not self.skew_widened():", "if True:"),
    ],
    ids=["no-widening", "sign", "gate"],
)
def test_a_broken_curve_widening_is_caught(
    curve_fit: Any, monkeypatch: pytest.MonkeyPatch, edit: tuple[str, str]
) -> None:
    from cleverly import effect_curves

    curve = curve_fit.effect_curve(random_state=0)
    hand = _hand_curve_rows(curve)
    mutate(monkeypatch, effect_curves, effect_curves.ProjectionCurve, "interval", [edit])
    low, high = curve.model.interval(ROWS)
    assert any(
        (low[i], high[i]) != pytest.approx((h_low, h_high), rel=1e-9)
        for i, (h_low, h_high, _) in enumerate(hand)
    )


def test_a_binary_curve_keeps_the_symmetric_half_width() -> None:
    curve = _curve_fit(binary=True).effect_curve(random_state=0)
    assert not curve.model.skew_widened()
    low, high = curve.model.interval(ROWS)
    centre = curve.model.predict(ROWS)
    np.testing.assert_allclose(centre - low, high - centre, rtol=1e-12)
