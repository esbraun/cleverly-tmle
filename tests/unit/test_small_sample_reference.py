"""The small-sample reference's inputs: the Kish count, the moment df, the estimate's df and family.

The tests here are the PR 1 witnesses of the Fix-3 rule. ``kish_count`` and ``satterthwaite_df``
are checked against hand values and against each other (T1), the conservative Kish df is a
mutation that the written-out values reject (T3), the estimate reads only the rows that carry
its curve (T5), the structural model check's one-split df is the shared function to a relative
1e-12 (T11), both counts are free of units and a ratio reads its log-scale curve (T13), the
reference quantile takes a fractional df (T14), the study rows record what a control needs
(T16), and the two counts move only as stated when zero rows are appended (T19).
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

from cleverly.estimators import TMLE
from cleverly.inference import (
    constant_outcome_cells,
    kish_count,
    make_estimate,
    reference_quantile,
    satterthwaite_df,
    wald_ci,
    wald_statistic,
)
from cleverly.inference import influence as influence_module
from cleverly.inference.cluster import cluster_sums
from cleverly.inference.delta import one_sided_quantile, two_sided_pvalue
from cleverly.inference.influence import ratio_estimates
from cleverly.longitudinal import structural as structural_module
from tests.studies import stratified_law as law
from tests.studies.evidence.properties import (
    SMALL_SAMPLE_COLUMNS,
    WITHHELD_TRIGGERS,
    small_sample_columns,
)
from tests.unit._source_mutation import mutate
from tests.unit._stratified_alternating_support import PerturbedOutcome, PerturbedTreatment

#: The relative tolerance of an identity that holds in exact arithmetic.
EXACT = 1e-12


def _three_ones() -> np.ndarray:
    return np.array([3.0, -1.0, -1.0, -1.0])


def _light_stratum(n: int = 2_000) -> np.ndarray:
    """Ten non-zero terms of centred kurtosis 3 in ``n`` rows: the moment df on them is 10.

    The terms are ``(c, -c, 1, -1, 1, -1, 1, -1, 1, -1)`` with ``c**2 = 6 + 5 sqrt(2)``, the
    root of ``c**4 - 12 c**2 - 14 = 0``, at which ``10 sum u^4 / (sum u^2)^2 = 3``.  Read over
    all ``n`` rows the kurtosis is ``3 n / 10`` and the df is ``2 n / (3 n / 10 - 1)``, 6.68 at
    2,000 rows.
    """
    c = math.sqrt(6.0 + 5.0 * math.sqrt(2.0))
    curve = np.zeros(n)
    curve[:10] = [c, -c, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    return curve


# ----------------------------------------------------------------------------------- T1


def check_written_out(df: Callable[..., float]) -> None:
    """Every hand value of T1 for a degrees-of-freedom function ``df``."""
    assert df(np.array([1.0, -1.0, 1.0, -1.0])) == math.inf
    # u^2 = (9, 1, 1, 1): sum 12, mean 3, squared deviations 36 + 3 * 4 = 48; 2 * 144 / 48.
    assert df(_three_ones()) == pytest.approx(6.0, rel=EXACT)
    m = 1_000
    dominant = np.zeros(m)
    dominant[0] = 1.0
    # Uncentred, kappa = m and nu = 2 m / (m - 1), the smallest value the formula takes.
    assert df(dominant, centered=False) == pytest.approx(2.0 * m / (m - 1.0), rel=EXACT)
    # Centred, 2 m (m - 1) / (m - 2)^2, within 0.3% of that bound at m = 1,000.
    assert df(dominant) == pytest.approx(2.0 * m * (m - 1.0) / (m - 2.0) ** 2, rel=EXACT)
    assert df(dominant) == pytest.approx(2.0 * m / (m - 1.0), rel=3e-3)
    # A normal curve has kurtosis 3, so nu = 2 m / 2 = m.
    normal = np.random.default_rng(20261112).normal(size=10_000)
    assert abs(df(normal) / normal.size - 1.0) < 0.05


def test_the_moment_df_takes_its_written_out_values() -> None:
    check_written_out(satterthwaite_df)


def test_the_kish_count_takes_its_written_out_values() -> None:
    assert kish_count(np.array([1.0, -1.0, 1.0, -1.0])) == 4.0
    assert kish_count(_three_ones()) == pytest.approx(144.0 / 84.0, rel=EXACT)
    assert kish_count(np.zeros(5)) == math.inf
    assert kish_count(np.full(5, 2.0)) == math.inf
    assert kish_count(np.full(5, 2.0), centered=False) == pytest.approx(5.0, rel=EXACT)


@pytest.mark.parametrize("seed", range(6))
def test_the_moment_df_is_two_kish_over_one_minus_kish_share(seed: int) -> None:
    """``nu = 2 K / (1 - K / m)`` on curves from light to heavy tails, centred or not."""
    rng = np.random.default_rng(seed)
    curve = rng.standard_t(2 + seed, size=50 + 40 * seed) * rng.uniform(0.1, 10.0)
    mask = rng.uniform(size=curve.size) < 0.6
    for rows in (None, mask):
        for centered in (True, False):
            m = curve.size if rows is None else int(mask.sum())
            k = kish_count(curve, rows=rows, centered=centered)
            expected = 2.0 * k / (1.0 - k / m)
            value = satterthwaite_df(curve, rows=rows, centered=centered)
            assert value == pytest.approx(expected, rel=1e-10)
            assert value >= 2.0 * m / (m - 1.0) * (1.0 - EXACT)


def test_integer_rows_read_the_same_terms_as_a_mask() -> None:
    curve = np.random.default_rng(3).normal(size=40)
    mask = np.arange(40) % 3 == 0
    assert satterthwaite_df(curve, rows=np.flatnonzero(mask)) == satterthwaite_df(curve, rows=mask)
    assert kish_count(curve, rows=np.flatnonzero(mask)) == kish_count(curve, rows=mask)


# ----------------------------------------------------------------------------------- T3


def test_kish_as_the_df_fails_the_written_out_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: the realised-term Kish count ``m / kappa`` in place of the moment df."""
    monkeypatch.setattr(influence_module, "satterthwaite_df", influence_module.kish_count)
    with pytest.raises(AssertionError):
        check_written_out(influence_module.satterthwaite_df)
    estimate = make_estimate(
        "ey1", 0.5, _three_ones(), n=4, rule_widened=False, outcome_family="gaussian"
    )
    # The estimate reads the module's function, so the mutation reaches it too.
    assert estimate.small_sample_df != pytest.approx(6.0)


# ----------------------------------------------------------------------------------- T5


def test_the_estimate_reads_only_the_rows_that_carry_its_curve() -> None:
    """Non-zero witness: a 10-row light stratum in 2,000 rows has df 10, not 6.68."""
    curve = _light_stratum()
    estimate = make_estimate(
        "ey1[V=2]", 0.5, curve, n=curve.size, rule_widened=False, outcome_family="gaussian"
    )
    assert estimate.small_sample_df == pytest.approx(10.0, rel=1e-10)
    assert estimate.effective_df == pytest.approx(10.0, rel=1e-10)
    unmasked = satterthwaite_df(curve)
    assert unmasked == pytest.approx(2.0 * 2_000 / (3.0 * 2_000 / 10.0 - 1.0), rel=1e-10)
    assert unmasked < 6.7


def test_dropping_the_mask_fails_the_light_stratum(monkeypatch: pytest.MonkeyPatch) -> None:
    mutate(
        monkeypatch,
        influence_module,
        influence_module.ParameterEstimate,
        "_small_sample_df",
        [("rows=curve != 0.0", "rows=None")],
    )
    with pytest.raises(AssertionError):
        test_the_estimate_reads_only_the_rows_that_carry_its_curve()


def _fixed_fit(frame: pd.DataFrame, *, stratum: int | None) -> Any:
    roles: dict[str, Any] = {"strata": ["V"]} if stratum is None else {}
    covariates = ["W", "V"] if stratum is None else ["W"]
    return (
        TMLE(
            outcome_learner=PerturbedOutcome(stratum),
            treatment_learner=PerturbedTreatment(stratum),
            cross_fit=False,
            estimands=("ey", "ate"),
            simultaneous=False,
            g_bounds=(0.01, 0.99),
            max_iter=100,
            tol=1e-12,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates, **roles)
        .single()
    )


@pytest.mark.parametrize("stem", ["ey[1]", "ey[0]", "ate"])
def test_a_stratum_estimate_has_the_df_of_the_fit_on_its_stratum(stem: str) -> None:
    """With fixed nuisances, the stratified estimate's df is the subset fit's.

    The stratum curve is the subset curve times ``n / n_s`` on the stratum's rows and 0 off
    them, and the df is scale-free, so the mask is what makes the two agree.
    """
    frame = law.sample(3_000, 20261112)
    stratified = _fixed_fit(frame, stratum=None)[f"{stem}[V=2]"]
    subset = _fixed_fit(frame.loc[frame["V"] == 2].reset_index(drop=True), stratum=2)[stem]
    curve = np.asarray(stratified.influence_curve)
    assert int(np.sum(curve != 0.0)) == int(np.sum(frame["V"] == 2))
    assert stratified.small_sample_df == pytest.approx(subset.small_sample_df, rel=1e-8)
    assert satterthwaite_df(curve) != pytest.approx(subset.small_sample_df, rel=1e-2)


# ----------------------------------------------------------------------------------- T11


def _one_split_before_the_refactor(
    curve: np.ndarray, cluster: Any, covariance: np.ndarray
) -> float:
    """The one-split branch of ``_satterthwaite_df`` as it stood before it called the helper."""
    unit = np.asarray(curve if cluster is None else cluster_sums(curve, cluster), dtype=float)
    centered = unit - unit.mean(axis=0)
    m = centered.shape[0]
    n = curve.shape[0]
    scale = m / ((m - 1) * n**2)
    diagonal = np.diag(covariance)
    terms = centered**2
    error = scale**2 * np.sum((terms - terms.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(error > 0.0, 2.0 * diagonal**2 / error, np.inf)
    return float(np.min(values))


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("heavy", [False, True])
def test_the_structural_one_split_df_is_unchanged_by_the_refactor(
    k: int, clustered: bool, heavy: bool
) -> None:
    rng = np.random.default_rng(100 * k + 10 * clustered + heavy)
    n = 600
    curve = rng.normal(size=(n, k))
    if heavy:
        curve[:3] *= 40.0
    cluster = rng.integers(0, 90, size=n) if clustered else None
    differences = curve.mean(axis=0)[None, :]
    covariance = structural_module._split_covariance(differences, [curve], cluster)
    value = structural_module._satterthwaite_df(differences, [curve], cluster, covariance)
    assert value == pytest.approx(
        _one_split_before_the_refactor(curve, cluster, covariance), rel=EXACT
    )


# ----------------------------------------------------------------------------------- T13


@pytest.mark.parametrize("factor", [1e-6, 3.0, -250.0])
def test_both_counts_are_free_of_units(factor: float) -> None:
    curve = np.random.default_rng(7).standard_t(4, size=300)
    mask = curve > -0.5
    assert kish_count(factor * curve) == pytest.approx(kish_count(curve), rel=EXACT)
    assert satterthwaite_df(factor * curve) == pytest.approx(satterthwaite_df(curve), rel=EXACT)
    assert satterthwaite_df(factor * curve, rows=mask) == pytest.approx(
        satterthwaite_df(curve, rows=mask), rel=EXACT
    )
    estimate = make_estimate("x", 1.0, curve, n=curve.size, rule_widened=False)
    assert replace(estimate, influence_curve=factor * curve).small_sample_df == pytest.approx(
        estimate.small_sample_df, rel=EXACT
    )


def test_a_ratio_reads_its_log_scale_curve() -> None:
    rng = np.random.default_rng(11)
    n = 400
    ic_one = rng.normal(size=n) * np.where(rng.uniform(size=n) < 0.05, 8.0, 1.0)
    ic_zero = rng.normal(size=n)
    psi_one, psi_zero = 0.2, 0.6
    ratio = ratio_estimates(
        psi_one, ic_one, psi_zero, ic_zero, n=n, rule_widened=False, which=("rr",)
    )["rr"]
    log_curve = ic_one / psi_one - ic_zero / psi_zero
    assert ratio.small_sample_df == pytest.approx(
        satterthwaite_df(log_curve, rows=log_curve != 0.0), rel=EXACT
    )
    # The witness: the difference-scale curve has another df, so reading it would be caught.
    assert satterthwaite_df(ic_one - ic_zero) != pytest.approx(ratio.small_sample_df, rel=0.05)


# ----------------------------------------------------------------------------------- T14


def test_the_reference_quantile_takes_a_fractional_df() -> None:
    assert reference_quantile(0.05, 17.5) == stats.t.ppf(0.975, 17.5)
    assert reference_quantile(0.05, math.inf) == pytest.approx(1.959963984540054, rel=EXACT)
    assert reference_quantile(0.05, math.inf) == reference_quantile(0.05)
    for refused in (0.5, 0.0, float("nan")):
        # The interval, the one-sided limit and the test refuse the same df alike.
        for call in (
            lambda df: reference_quantile(0.05, df),
            lambda df: one_sided_quantile(0.95, df),
            lambda df: wald_ci(1.0, 0.5, df=df),
            lambda df: wald_statistic(1.0, 0.0, 0.5, df=df),
            lambda df: two_sided_pvalue(1.0, 0.5, df=df),
        ):
            with pytest.raises(ValueError, match="at least 1 degree"):
                call(refused)
    assert one_sided_quantile(0.95, 17.5) == stats.t.ppf(0.95, 17.5)
    assert one_sided_quantile(0.95, math.inf) == one_sided_quantile(0.95)
    _, high = wald_ci(1.0, 0.5, df=17.5)
    assert high - 1.0 == pytest.approx(0.5 * stats.t.ppf(0.975, 17.5), rel=EXACT)
    assert wald_ci(1.0, 0.5, df=math.inf) == wald_ci(1.0, 0.5)
    _, pvalue = wald_statistic(1.0, 0.0, 0.5, df=17.5)
    assert pvalue == pytest.approx(2.0 * stats.t.sf(2.0, 17.5), rel=EXACT)
    assert wald_statistic(1.0, 0.0, 0.5, df=math.inf) == wald_statistic(1.0, 0.0, 0.5)
    assert two_sided_pvalue(1.0, 0.5, df=17.5) == pvalue


# ----------------------------------------------------------------------------------- the estimate


def test_the_t_is_opt_in_by_family() -> None:
    curve = _three_ones()
    for family in (None, "binomial"):
        assert (
            make_estimate(
                "x", 0.5, curve, n=4, rule_widened=False, outcome_family=family
            ).effective_df
            is None
        )
    gaussian = make_estimate("x", 0.5, curve, n=4, rule_widened=False, outcome_family="gaussian")
    assert gaussian.effective_df == 6.0
    heavy = np.zeros(400)
    heavy[0], heavy[1] = 1.0, -1.0
    heavy[2:] = 1e-4 * np.random.default_rng(0).normal(size=398)
    capped = make_estimate("x", 0.5, heavy, n=400, rule_widened=False, outcome_family="gaussian")
    assert capped.small_sample_df < 5.0
    assert capped.effective_df == 5.0


def test_the_df_is_none_where_it_has_no_meaning() -> None:
    curve = np.random.default_rng(1).normal(size=60)
    cluster = np.repeat(np.arange(20), 3)
    clustered = make_estimate(
        "x", 0.5, curve, n=60, rule_widened=False, cluster=cluster, outcome_family="gaussian"
    )
    assert clustered.small_sample_df is None
    assert clustered.effective_df == clustered.reference_df
    flat = make_estimate(
        "x", 0.5, np.zeros(10), n=10, rule_widened=False, outcome_family="gaussian"
    )
    assert flat.small_sample_df is None
    assert flat.effective_df is None


def test_the_df_never_sets_a_status_and_the_ci_does_not_read_it() -> None:
    heavy = 1e-3 * np.random.default_rng(4).normal(size=200)
    heavy[0] = 1.0
    estimate = make_estimate("x", 0.5, heavy, n=200, rule_widened=False, outcome_family="gaussian")
    assert estimate.small_sample_df < 2.1
    assert estimate.inference == "influence_curve"
    assert estimate.ci == wald_ci(0.5, estimate.std_error)


def test_the_second_moment_rule_reads_uncentred_terms() -> None:
    curve = np.array([2.0, 1.0, 0.5, 0.0, 0.0])
    estimate = make_estimate(
        "x", 0.5, curve, n=5, rule_widened=False, covariance_rule="second_moment"
    )
    assert estimate.small_sample_df == satterthwaite_df(curve, rows=curve != 0, centered=False)


def test_the_interval_scale_names_the_scale_of_the_shipped_interval() -> None:
    curve = np.random.default_rng(2).normal(size=50)
    assert make_estimate("x", 0.5, curve, n=50, rule_widened=False).interval_scale == "identity"
    ratio = make_estimate(
        "rr", 2.0, curve, n=50, rule_widened=False, scale="ratio", log_psi=math.log(2.0)
    )
    assert ratio.interval_scale == "log"


# ----------------------------------------------------------------------------------- cells


def test_a_constant_cell_is_an_arm_with_one_observed_value() -> None:
    outcome = np.array([1, 1, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    rows = np.array([True] * 6 + [False, False])
    assert constant_outcome_cells(outcome, treatment, rows) == frozenset({1})
    # The two rows outside the stratum break the tie only when they are read.
    assert constant_outcome_cells(outcome, treatment, np.ones(8, dtype=bool)) == frozenset()
    assert constant_outcome_cells(outcome, treatment, np.flatnonzero(rows)) == frozenset({1})


def test_weight_zero_and_unobserved_rows_do_not_break_a_tie() -> None:
    outcome = np.array([1.0, 1.0, 0.0, 0.0, 1.0, np.nan])
    treatment = np.array([1, 1, 1, 0, 0, 1])
    rows = np.ones(6, dtype=bool)
    weights = np.array([1.0, 2.0, 0.0, 1.0, 1.0, 1.0])
    observed = np.array([True, True, True, True, True, False])
    assert constant_outcome_cells(outcome, treatment, rows, observed=observed) == frozenset()
    assert constant_outcome_cells(
        outcome, treatment, rows, weights=weights, observed=observed
    ) == frozenset({1})
    # An arm with no qualifying row is not a constant cell.
    only = np.array([True, True, True, False, False, False])
    assert constant_outcome_cells(
        outcome, treatment, only, weights=weights, observed=observed
    ) == frozenset({1})


def test_a_continuous_arm_with_one_value_is_constant() -> None:
    outcome = np.array([0.0, 0.0, 0.0, 1.5, 2.5])
    treatment = np.array(["b", "b", "b", "a", "a"])
    assert constant_outcome_cells(outcome, treatment, np.ones(5, dtype=bool)) == frozenset({"b"})


# ----------------------------------------------------------------------------------- T16


def test_a_small_sample_row_records_what_a_control_needs() -> None:
    curve = _light_stratum()
    estimate = make_estimate("ey1[V=2]", 0.5, curve, n=curve.size, rule_widened=False)
    gaussian = small_sample_columns(
        estimate, outcome_family="gaussian", interval_scale="identity", withheld=""
    )
    assert tuple(gaussian) == SMALL_SAMPLE_COLUMNS
    assert gaussian["df"] == pytest.approx(10.0, rel=1e-10)
    assert gaussian["carrying_rows"] == 10
    assert gaussian["curve_kurtosis"] == pytest.approx(3.0, rel=1e-10)
    # The df written out from the two columns: 2 m / (kappa - 1).
    assert gaussian["df"] == pytest.approx(
        2.0 * gaussian["carrying_rows"] / (gaussian["curve_kurtosis"] - 1.0), rel=1e-10
    )
    binary = small_sample_columns(
        estimate, outcome_family="binomial", interval_scale="range_logit", withheld="constant_cell"
    )
    assert binary["df"] == math.inf
    assert (binary["interval_scale"], binary["outcome_family"], binary["withheld"]) == (
        "range_logit",
        "binomial",
        "constant_cell",
    )
    assert (
        small_sample_columns(
            estimate, outcome_family=None, interval_scale="identity", withheld="out_of_range"
        )["outcome_family"]
        == ""
    )
    assert set(WITHHELD_TRIGGERS) == {"", "constant_cell", "out_of_range"}
    with pytest.raises(ValueError, match="unknown withheld trigger"):
        small_sample_columns(estimate, outcome_family=None, interval_scale="identity", withheld="x")


# ----------------------------------------------------------------------------------- T19


def test_appended_zero_rows_move_only_the_unmasked_df() -> None:
    """The band's count and the interval's df read the same terms after zero rows join.

    A fitted curve is centred at its solved score, so the fixture is mean zero: centring over
    the longer curve then changes nothing, and the Kish count, which has no ``m``, is unchanged.
    The moment df has ``m``: read over all rows it moves, and read over the carrying rows it
    does not.
    """
    curve = np.random.default_rng(5).standard_t(3, size=200)
    curve -= curve.mean()
    longer = np.concatenate([curve, np.zeros(1_800)])
    assert kish_count(longer) == pytest.approx(kish_count(curve), rel=1e-10)
    assert satterthwaite_df(longer, rows=longer != 0.0) == pytest.approx(
        satterthwaite_df(curve, rows=curve != 0.0), rel=EXACT
    )
    assert kish_count(longer, rows=longer != 0.0) == pytest.approx(
        kish_count(curve, rows=curve != 0.0), rel=EXACT
    )
    assert satterthwaite_df(longer) < 0.95 * satterthwaite_df(curve)
    shorter = make_estimate("x", 0.0, curve, n=curve.size, rule_widened=False)
    extended = make_estimate("x", 0.0, longer, n=longer.size, rule_widened=False)
    assert extended.small_sample_df == pytest.approx(shorter.small_sample_df, rel=EXACT)
