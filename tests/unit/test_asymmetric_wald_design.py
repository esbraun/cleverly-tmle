"""The design of ``asymmetric-wald``, recomputed from the study and law modules.

Fix-3 plan rev 6, sections 9.5 to 9.7: every number the planning scripts printed
(``rows/Fix-3/exploratory/rev5/rev5_design.out``, ``width_predict.out`` and ``rev6_sizing.out``)
is recomputed from :mod:`tests.studies.asymmetric_wald_laws` and
:mod:`tests.studies.asymmetric_wald`, each compared at half a unit of its last printed digit.
The study's readings are checked on synthetic rows with known counts: the legs on the first
2,000 replicates, the tail clause, the width clause's bootstrap (nit R6-1), the witnesses, the
falsifiers and the failure accounting's candidates, each with a mutation the check rejects.

**No estimator is fitted on a held-out law here.**  The vector intervals are checked against
the library functions on fits of L1 and of two gaussian diagnosis groups, and the held-out
samplers are checked only by sample moments of observed variables (the covariates, the arm
shares, the arm means and the residuals) against the exact law, which reads no estimate and no
interval.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.inference import implied_correlation, mover_ci, skew_widened_ci
from tests.studies import asymmetric_wald as study
from tests.studies import asymmetric_wald_laws as heldout
from tests.studies import asymmetric_wald_properties as props
from tests.studies import canonical_stratified_tmle as stratified_tmle
from tests.studies import stratified_law as l1


def printed(value: float, expected: float, digits: int) -> bool:
    """``value`` prints as ``expected`` at ``digits`` decimals, within half a unit."""
    return abs(value - expected) <= 0.5 * 10.0**-digits + 1e-12


# ------------------------------------------------------------------ rev5_design.out

#: law, estimand, truth, nu_pop, kappa, skew, a_pop, eff SE, dist (or None), shares (c, 1, 0),
#: withheld (or None), as printed.
DESIGN = (
    ("d1", "ey[1][V=2]", 0.107487, 13.8, 87.96, 8.473, 0.0576, 0.03970, 2.71, (0.001, 0.999, 0.000), 1.12e-03),
    ("d1", "ey[0][V=2]", 0.303670, 1301.1, 1.92, 0.897, 0.0061, 0.01981, 15.33, (0.013, 0.000, 0.987), 1.35e-73),
    ("d1", "ate[V=2]", -0.196184, 21.4, 57.07, 6.011, 0.0409, 0.04431, 18.14, (0.001, 0.802, 0.197), 1.12e-03),
    ("d1", "ey[1][V=0]", 0.127122, 108.2, 28.72, 4.815, 0.0207, 0.01761, 7.22, (0.002, 0.998, 0.000), 2.57e-21),
    ("d1", "ey[0][V=0]", 0.344318, 3419.8, 1.88, 0.735, 0.0032, 0.01406, 24.49, (0.013, 0.000, 0.987), 1.80e-182),
    ("d1", "ate[V=0]", -0.217196, 292.5, 11.26, 2.139, 0.0092, 0.02248, 34.83, (0.001, 0.613, 0.386), 2.57e-21),
    ("d2", "ey[1]", 0.638121, 9961.0, 1.40, -0.585, -0.0022, 0.01105, 32.75, (0.049, 0.951, 0.000), 0.0),
    ("d2", "ey[0]", 0.923855, 13.6, 294.26, -15.354, -0.0572, 0.02474, 3.08, (0.001, 0.000, 0.999), 6.07e-05),
    ("d2", "ate", -0.285734, 19.3, 208.02, 11.722, 0.0437, 0.02702, 26.43, (0.004, 0.159, 0.837), 6.07e-05),
    ("c1", "ey[1][V=2]", 2.100000, 21.2, 29.27, 2.926, 0.0282, 0.25311, None, (0.006, 0.994, 0.000), None),
    ("c1", "ey[0][V=2]", 1.300000, 73.9, 9.12, 1.714, 0.0165, 0.10038, None, (0.040, 0.000, 0.960), None),
    ("c1", "ate[V=2]", 0.800000, 27.4, 22.91, 2.368, 0.0228, 0.27082, None, (0.000, 0.868, 0.132), None),
    ("c2", "ey[1][V=2]", 0.880000, 21.0, 29.57, -3.285, -0.0316, 0.01432, None, (0.033, 0.967, 0.000), None),
    ("c2", "ey[0][V=2]", 0.810000, 120.2, 5.99, -1.370, -0.0132, 0.01179, None, (0.049, 0.000, 0.951), None),
    ("c2", "ate[V=2]", 0.070000, 56.8, 11.57, -1.052, -0.0101, 0.01818, None, (0.000, 0.600, 0.400), None),
    ("c3", "ey[1][V=2]", 2.300000, 20.8, 29.78, -0.085, -0.0008, 0.18997, None, (0.008, 0.992, 0.000), None),
    ("c3", "ey[0][V=2]", 1.600000, 77.7, 8.72, 0.082, 0.0008, 0.08044, None, (0.043, 0.000, 0.957), None),
    ("c3", "ate[V=2]", 0.700000, 28.4, 22.15, -0.000, -0.0000, 0.20495, None, (0.000, 0.853, 0.147), None),
)  # fmt: skip


@pytest.mark.parametrize("line", DESIGN, ids=[f"{row[0]}-{row[1]}" for row in DESIGN])
def test_every_design_number_of_rev5(line: tuple[Any, ...]) -> None:
    law, estimand, truth, nu, kappa, skew, a_pop, se, dist, shares, withheld = line
    design = heldout.design(law, estimand)
    assert printed(design.truth, truth, 6)
    assert printed(design.nu_pop, nu, 1)
    assert printed(design.kappa, kappa, 2)
    assert printed(design.skewness, skew, 3)
    assert printed(design.a_pop, a_pop, 4)
    assert printed(design.efficiency_se, se, 5)
    if dist is not None:
        bounds = (-1.0, 1.0) if estimand.startswith("ate") else (0.0, 1.0)
        assert printed(design.distance(bounds), dist, 2)
    for key, value in zip(("c", "1", "0"), shares, strict=True):
        assert printed(design.shares[key], value, 3)
    if withheld is not None:
        exact = heldout.constant_cell(law, estimand)
        assert exact == pytest.approx(withheld, rel=5e-3, abs=1e-300)


def test_the_per_law_lines_of_rev5() -> None:
    d1 = heldout.arm_counts("d1", 2)
    assert printed(3_000 * d1["q_1"], 60.5, 1) and printed(3_000 * d1["q_11"], 6.79, 2)
    d1_v0 = heldout.arm_counts("d1", 0)
    assert printed(3_000 * d1_v0["q_1"], 353.8, 1) and printed(3_000 * d1_v0["q_11"], 47.04, 2)
    d2 = heldout.arm_counts("d2", None)
    assert printed(2_000 * d2["q_0"], 109.8, 1)
    assert printed(2_000 * (d2["q_0"] - d2["q_01"]), 9.69, 2)
    assert heldout.d2_default_bound_tail() == pytest.approx(9.69e-05, rel=5e-3)
    for law, treated in (("c1", 89.4), ("c2", 108.9), ("c3", 89.4)):
        pw, g = heldout._cells(law, 2)[:2]
        spec = heldout.LAWS[law]
        assert spec.p_v is not None
        rows = spec.n * spec.p_v[2]
        assert printed(rows, 300.0, 0)
        assert printed(rows * float(np.sum(np.asarray(pw) / np.sum(pw) * g)), treated, 1)
    gamma = heldout.gamma_central(2.0, 1.0)
    assert printed(gamma[3] / gamma[2] ** 1.5, 1.414, 3)
    for mu, skew in ((0.79, -1.0), (0.925, -2.26), (0.72, -0.69)):
        central = heldout.beta_central(mu, heldout.C2_PHI)
        assert printed(float(central[3] / central[2] ** 1.5), skew, 2)
    assert printed(float(heldout.c2_mu(0, 0, 0)), 0.660, 3)
    assert printed(float(heldout.c2_mu(1, 3, 2)), 0.925, 3)


def test_the_diagnosis_reference_block_of_rev5() -> None:
    lognormal = heldout.lognormal06_design("ey[1][V=2]")
    assert printed(lognormal.truth, 3.797217, 6) and printed(lognormal.nu_pop, 5.7, 1)
    assert printed(lognormal.skewness, 4.177, 3) and printed(lognormal.a_pop, 0.0492, 4)
    ate = heldout.lognormal06_design("ate[V=2]")
    assert printed(ate.skewness, 2.669, 3) and printed(ate.a_pop, 0.0315, 4)
    h1 = heldout.h1_design("ate")
    assert printed(h1.truth, 0.452176, 6) and printed(h1.skewness, -11.131, 3)
    assert printed(h1.a_pop, -0.0415, 4) and printed(h1.shares["1"], 0.909, 3)
    assert printed(h1.distance((-1.0, 1.0)), 14.67, 2)
    h1_mean = heldout.h1_design("ey[1]")
    assert printed(h1_mean.skewness, -12.888, 3) and printed(h1_mean.distance((0.0, 1.0)), 3.05, 2)


# ----------------------------------------------------------------- width_predict.out

#: label, estimand, a_pop, sd(a_hat), E|a_hat|, nu_pop, predicted ratio, as printed.
WIDTHS = (
    ("lognormal06", "ey[1][V=2]", 0.0492, 0.2582, 0.2097, 5.7, 1.367),
    ("lognormal06", "ate[V=2]", 0.0315, 0.1901, 0.1537, 8.4, 1.292),
    ("h2", "ey[1][V=2]", 0.0408, 0.0223, 0.0414, 14.9, 1.084),
    ("h2", "ate[V=2]", 0.0213, 0.0198, 0.0242, 24.4, 1.051),
    ("c1", "ey[1][V=2]", 0.0282, 0.0222, 0.0303, 21.2, 1.063),
    ("c1", "ate[V=2]", 0.0228, 0.0194, 0.0251, 27.4, 1.053),
    ("c2", "ey[1][V=2]", -0.0316, 0.0159, 0.0319, 21.0, 1.067),
    ("c2", "ate[V=2]", -0.0101, 0.0099, 0.0117, 56.8, 1.025),
    ("c3", "ey[1][V=2]", -0.0008, 0.0295, 0.0235, 20.8, 1.049),
    ("c3", "ate[V=2]", -0.0000, 0.0233, 0.0186, 28.4, 1.039),
)


@pytest.mark.parametrize("line", WIDTHS, ids=[f"{row[0]}-{row[1]}" for row in WIDTHS])
def test_every_width_prediction(line: tuple[Any, ...]) -> None:
    label, estimand, a_pop, sd, e_abs, nu, ratio = line
    if label == "lognormal06":
        design = heldout.lognormal06_design(estimand)
    elif label == "h2":
        design = heldout.h2_design(estimand)
    else:
        design = heldout.design(label, estimand)
    predicted = heldout.width_prediction(design.moments, design.m)
    assert printed(predicted["a_pop"], a_pop, 4)
    assert printed(predicted["sd"], sd, 4)
    assert printed(predicted["e_abs"], e_abs, 4)
    assert printed(predicted["nu_pop"], nu, 1)
    assert printed(predicted["ratio"], ratio, 3)


def test_the_declared_width_predictions_are_the_quadrature() -> None:
    for cell, (ratio, _) in study.WIDTH_PREDICTED.items():
        estimand = study.CELL_BY_NAME[cell].estimand
        design = heldout.design("c3", estimand)
        assert printed(heldout.width_prediction(design.moments, design.m)["ratio"], ratio, 3)


# ------------------------------------------------------------------- rev6_sizing.out

CALIBRATION = {
    "d1_ey1": 0.99642, "d1_ate": 0.99642, "d2_ey0": 0.99815, "d2_ate": 0.99815,
    "c1_ey1": 0.99822, "c1_ate": 0.99822, "c2_ey1": 0.99822, "c3_ey1": 0.99822,
    "c3_ate": 0.99822, "ho_sparse_treat_ate": 0.99799, "ho_hurdle_ey1": 0.99524,
    "lognormal06_ey1": 0.99822,
}  # fmt: skip
PREDICTED = {"ho_sparse_treat_ate": 0.99965, "ho_hurdle_ey1": 0.92322, "lognormal06_ey1": 0.99824}


def test_the_largest_passing_tail_counts() -> None:
    assert study.max_tail(4_000) == 193
    assert study.max_tail(2_000) == 103
    # The clause is the 99% CP lower end: 193 of 4,000 passes, 194 fails.
    assert props.tail_lower(193, 4_000) <= 0.040 < props.tail_lower(194, 4_000)


@pytest.mark.parametrize("cell", sorted(CALIBRATION))
def test_each_cell_pass_probability(cell: str) -> None:
    withheld = study.DECLARED_WITHHELD[cell]
    assert printed(study.split_pass(0.95, 0.025, 0.025, withheld), CALIBRATION[cell], 5)
    if cell in PREDICTED:
        coverage, below, above = study.PREDICTED[cell]
        mass = coverage + below + above
        value = study.split_pass(coverage / mass, below / mass, above / mass, withheld)
        assert printed(value, PREDICTED[cell], 5)


def test_the_declared_withheld_shares_are_the_exact_ones() -> None:
    for cell in ("d1_ey1", "d1_ate", "d2_ey0", "d2_ate"):
        group = study.GROUP_OF_CELL[cell]
        exact = heldout.constant_cell(group.law, study.CELL_BY_NAME[cell].estimand)
        assert exact == pytest.approx(study.DECLARED_WITHHELD[cell], rel=5e-3)
    from tests.studies import small_sample_laws as laws

    assert laws.constant_cell_share("h1", 2_000, "ate") == pytest.approx(1.9e-4, rel=0.05)
    hurdle = laws.h2_empty_cell(2_000) + laws.constant_cell_share("h2", 2_000, "ey[1][V=2]")
    assert hurdle == pytest.approx(1.6e-3, rel=0.05)


def test_the_family_and_the_designed_chance() -> None:
    gains = 1.0
    for b_rate, c_rate in study.GAIN_RATES.values():
        gains *= study.gain_power(b_rate, c_rate)
    assert printed(gains, 0.99996, 5)
    assert printed(study.family_pass(predicted=False), 0.9720, 4)
    assert printed(study.family_pass(predicted=True), 0.9032, 4)


TRADE = {
    (0.930, 0.055, 0.015): (0.8452, 0.9689, 0.9944, 0.8208, 0.9688),
    (0.940, 0.045, 0.015): (0.1923, 0.1518, 0.3084, 0.1848, 0.1517),
    (0.930, 0.035, 0.035): (0.8208, 0.5825, 0.8208, 0.8208, 0.0000),
}


@pytest.mark.parametrize("pattern", sorted(TRADE))
def test_the_trade_per_failure_pattern(pattern: tuple[float, float, float]) -> None:
    coverage, below, above = pattern
    single_2, single_4, split, legs, tails = TRADE[pattern]
    assert printed(1 - study.joint_single(coverage, below, above, 0.0, 2_000), single_2, 4)
    assert printed(1 - study.joint_single(coverage, below, above, 0.0, 4_000), single_4, 4)
    legs_value = study.legs_pass(coverage, 0.0)
    tails_value = study.tails_pass(below, above, 0.0)
    assert printed(1 - legs_value * tails_value, split, 4)
    assert printed(1 - legs_value, legs, 4)
    assert printed(1 - tails_value, tails, 4)


def test_the_single_r_families() -> None:
    shares = study.DECLARED_WITHHELD.values()
    two = float(np.prod([study.joint_single(0.95, 0.025, 0.025, w, 2_000) for w in shares]))
    four = float(np.prod([study.joint_single(0.95, 0.025, 0.025, w, 4_000) for w in shares]))
    assert printed(two, 0.9721, 4) and printed(four, 1.0000, 4)


TAIL_FIRE = {
    0.030: (0.0000, 0.0000), 0.035: (0.0000, 0.0001), 0.040: (0.0042, 0.0049),
    0.045: (0.1517, 0.0750), 0.050: (0.6783, 0.3548), 0.055: (0.9688, 0.7353),
    0.058: (0.9962, 0.8854),
}  # fmt: skip


@pytest.mark.parametrize("rate", sorted(TAIL_FIRE))
def test_the_tail_clause_firing_probability(rate: float) -> None:
    at_4000, at_2000 = TAIL_FIRE[rate]
    assert printed(study.tail_fire(rate), at_4000, 4)
    assert printed(study.tail_fire(rate, 2_000), at_2000, 4)


def test_the_witness_powers() -> None:
    rate, withheld = study.WITNESS_RATES["sign_flip_witness"]
    assert printed(study.tail_fire(rate, withheld=withheld), 0.99621, 5)
    rate, withheld = study.WITNESS_RATES["mover_swap_witness"]
    assert printed(study.tail_fire(rate, withheld=withheld), 1.00000, 5)


def test_the_width_clause_sizing() -> None:
    for cell, midpoint in (("c3_ey1", 1.0785), ("c3_ate", 1.0788)):
        ratio, sd = study.WIDTH_PREDICTED[cell]
        assert study.width_fire(ratio, sd) < 1e-12
        assert printed(study.width_midpoint(sd), midpoint, 4)
        assert study.width_fire(study.width_midpoint(sd), sd) == pytest.approx(0.5)


# ------------------------------------------------------------------------ the cells


def test_the_declared_cells() -> None:
    assert study.GATED_CELLS == (
        "d1_ey1", "d1_ate", "d2_ey0", "d2_ate", "c1_ey1", "c1_ate", "c2_ey1", "c3_ey1",
        "c3_ate", "ho_sparse_treat_ate", "ho_hurdle_ey1", "lognormal06_ey1",
    )  # fmt: skip
    assert set(study.DECLARED_WITHHELD) == set(study.GATED_CELLS)
    assert len(study.GROUPS) == 14
    assert all(group.replicates == 4_000 for group in study.GROUPS)
    assert sum(group.replicates for group in study.GROUPS) == 56_000
    heldout_cells = {
        cell.name for group in study.GROUPS for cell in group.cells if cell.family == study.HELDOUT
    }
    assert {study.GROUP_OF_CELL[name].law for name in heldout_cells} == set(heldout.HELDOUT_LAWS)
    assert not set(study.SMOKE_GROUPS) & {"d1", "d2", "c1", "c2", "c3"}
    assert study.STUDY.publication_policy == "reporting"
    assert study.SEED not in (2026111201, 2026111202)


def test_the_reported_cells_name_their_arms() -> None:
    assert study.arm_names("ate[V=2]") == ("ey[1][V=2]", "ey[0][V=2]")
    assert study.arm_names("ate") == ("ey[1]", "ey[0]")
    assert study.arm_names("ey[1]") is None


# ---------------------------------------------------------------- the intervals: parity


@pytest.fixture(scope="module")
def l1_rows() -> pd.DataFrame:
    """The rows of three L1 fits read as the boundary group, built by the study's own code."""
    group = study.GROUP_BY_LABEL["boundary_2000"]
    frames = []
    for seed in (101, 102, 103):
        result = stratified_tmle.fit_cleverly(l1.sample(2_000, seed))
        for estimand in ("ate[V=2]", "ate[V=1]", "ey[1][V=2]"):
            extra = study.rule_columns(result, estimand, group.family)
            frames.append(
                {
                    "truth": float(l1.TRUTH[estimand]),
                    "estimate": float(result[estimand].psi),
                    "std_error": float(result[estimand].plugin_std_error),
                    **extra,
                }
            )
    return pd.DataFrame(frames)


def test_the_vector_rule_is_the_library_rule(l1_rows: pd.DataFrame) -> None:
    low, high = props.candidate_interval(l1_rows, "rule")
    shown = props.withheld_triggers(l1_rows) == ""
    assert shown.all()
    np.testing.assert_allclose(low, l1_rows["rule_lower"], rtol=1e-12)
    np.testing.assert_allclose(high, l1_rows["rule_upper"], rtol=1e-12)
    assert set(l1_rows["construction"]) == {"mover", "range_logit"}


@pytest.fixture(scope="module")
def gaussian_rows() -> pd.DataFrame:
    """The cell rows of three fits each of two gaussian diagnosis groups, by the study's code.

    ``student`` carries a mean and a difference; ``identity_msm`` carries an MSM coefficient.
    Neither is a held-out law, and the seeds are not run streams.
    """
    frames = []
    for label in ("student", "identity_msm"):
        group = study.GROUP_BY_LABEL[label]
        assert group.law not in heldout.HELDOUT_LAWS
        for seed in (301, 302, 303):
            frame = study.sample(group.law, group.n, seed)
            frames.extend(props._replicate(label, seed, frame))
    rows = pd.DataFrame(frames)
    return rows.loc[rows["property"] != study.CONTROL].reset_index(drop=True)


def test_the_vector_skew_t_is_the_library_rule(gaussian_rows: pd.DataFrame) -> None:
    assert set(gaussian_rows["construction"]) == {"skew_t"}
    assert set(gaussian_rows["cell"]) == {"student_ey1", "student_ate", "identity_msm_v2_w"}
    assert (props.withheld_triggers(gaussian_rows) == "").all()
    # A witness that the acceleration term is live on these rows.
    assert (gaussian_rows["acceleration"].abs() > 0).all()
    low, high = props.candidate_interval(gaussian_rows, "rule")
    np.testing.assert_allclose(low, gaussian_rows["rule_lower"], rtol=1e-12)
    np.testing.assert_allclose(high, gaussian_rows["rule_upper"], rtol=1e-12)


def test_the_study_reads_the_shipped_arm_interval() -> None:
    result = stratified_tmle.fit_cleverly(l1.sample(2_000, 104))
    for name in ("ey[1][V=2]", "ey[0][V=2]", "ey[1][V=0]"):
        estimate = result[name]
        assert study._range_logit(estimate, 0.05) == pytest.approx(estimate.ci, rel=1e-12)


def _gaussian_rows(count: int = 50, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "truth": 1.0,
            "estimate": 1.0 + rng.normal(0, 0.1, count),
            "std_error": rng.uniform(0.05, 0.15, count),
            "nu": rng.uniform(3.0, 40.0, count),
            "acceleration": rng.normal(0, 0.04, count),
            "bound_low": math.nan,
            "bound_high": math.nan,
            "outcome_family": "gaussian",
            "construction": "skew_t",
            "withheld": "",
        }
    )


def test_the_vector_skew_interval_is_the_library_one() -> None:
    rows = _gaussian_rows()
    for alpha in (0.05, 0.2):
        low, high = props.candidate_interval(rows, "skew_t", alpha=alpha)
        flip_low, flip_high = props.candidate_interval(rows, "sign_flip", alpha=alpha)
        for i, row in rows.iterrows():
            df = max(float(row["nu"]), 5.0)
            expected = skew_widened_ci(
                row["estimate"], row["std_error"], alpha, df=df, acceleration=row["acceleration"]
            )
            flipped = skew_widened_ci(
                row["estimate"], row["std_error"], alpha, df=df, acceleration=-row["acceleration"]
            )
            assert (low[i], high[i]) == pytest.approx(expected, rel=1e-12)
            assert (flip_low[i], flip_high[i]) == pytest.approx(flipped, rel=1e-12)


def test_the_vector_mover_is_the_library_one() -> None:
    rng = np.random.default_rng(9)
    count = 40
    p1, p0 = rng.uniform(0.05, 0.95, count), rng.uniform(0.05, 0.95, count)
    s1, s0 = rng.uniform(0.01, 0.05, count), rng.uniform(0.01, 0.05, count)
    r = rng.uniform(-0.8, 0.8, count)
    rows = pd.DataFrame(
        {
            "truth": p1 - p0,
            "estimate": p1 - p0,
            "std_error": np.sqrt(s1**2 + s0**2 - 2 * r * s1 * s0),
            "nu": math.nan,
            "acceleration": 0.0,
            "bound_low": -1.0,
            "bound_high": 1.0,
            "arm1_estimate": p1,
            "arm1_std_error": s1,
            "arm0_estimate": p0,
            "arm0_std_error": s0,
            "arm_correlation": r,
            "outcome_family": "binomial",
            "construction": "mover",
            "withheld": "",
        }
    )
    from cleverly.inference import Transform, wald_ci

    scale = Transform.range_logit(0.0, 1.0)

    def arm(p: float, s: float) -> tuple[float, float]:
        low, high = wald_ci(float(scale.forward(p)), s * scale.slope(p))
        return float(scale.inverse(low)), float(scale.inverse(high))

    low, high = props.candidate_interval(rows, "mover")
    swap_low, swap_high = props.candidate_interval(rows, "mover_swap")
    flat_low, _ = props.candidate_interval(rows, "mover_rho0")
    for i in range(count):
        a1, a0 = arm(p1[i], s1[i]), arm(p0[i], s0[i])
        assert (low[i], high[i]) == pytest.approx(
            mover_ci(p1[i], a1, p0[i], a0, correlation=r[i]), rel=1e-12
        )
        mirrored = ((2 * p1[i] - a1[1], 2 * p1[i] - a1[0]), (2 * p0[i] - a0[1], 2 * p0[i] - a0[0]))
        assert (swap_low[i], swap_high[i]) == pytest.approx(
            mover_ci(p1[i], mirrored[0], p0[i], mirrored[1], correlation=r[i]), rel=1e-10
        )
        assert flat_low[i] == pytest.approx(mover_ci(p1[i], a1, p0[i], a0, correlation=0.0)[0])
    assert implied_correlation(1.0, 1.0, 1.0) == 0.5


# ---------------------------------------------------------------- synthetic readings


def synthetic_rows(
    cell: str,
    *,
    family: str = study.OWNER,
    role: str = "positive",
    requested: int = 4_000,
    lost: tuple[int, ...] = (),
    below: int = 0,
    above: int = 0,
    withheld: int = 0,
    early_misses: int = 0,
) -> pd.DataFrame:
    """Rows of one gated cell with known counts on a continuous target.

    Each row's skew-widened interval is ``psi +/- 0.2`` (acceleration 0, ``nu`` large); a
    covering row has ``psi = truth``, a ``below`` row an interval entirely below the truth, an
    ``above`` row one entirely above, and a withheld row a constant-cell trigger.  The misses
    sit at the end of the replicate order, except ``early_misses`` of the ``below`` ones, which
    sit among the first 2,000.  The replicates in ``lost`` are absent.
    """
    replicates = np.array([r for r in range(requested) if r not in set(lost)])
    present = replicates.size
    kinds = np.array(["cover"] * present, dtype=object)
    tail = [*(["below"] * (below - early_misses)), *(["above"] * above), *(["withheld"] * withheld)]
    if tail:
        kinds[present - len(tail) :] = tail
    kinds[:early_misses] = "below"
    truth = 3.0
    psi = np.where(kinds == "below", truth - 0.5, np.where(kinds == "above", truth + 0.5, truth))
    se = np.full(present, 0.2 / float(stats.t.ppf(0.975, 1e6)))
    shown = kinds != "withheld"
    covered = (kinds == "cover").astype(int)
    return pd.DataFrame(
        {
            "property": family,
            "cell": cell,
            "role": role,
            "replicate": replicates,
            "n": 2_000,
            "requested_replicates": requested,
            "failed_replicates": requested - present,
            "truth": truth,
            "estimate": psi,
            "std_error": se,
            "covered": covered,
            "rejected": 1 - covered,
            "df": 1e6,
            "curve_kurtosis": 5.0,
            "carrying_rows": 300,
            "interval_scale": "skew_t",
            "outcome_family": "gaussian",
            "withheld": np.where(shown, "", "constant_cell"),
            "nu": 1e6,
            "acceleration": 0.0,
            "bound_low": math.nan,
            "bound_high": math.nan,
            "arm1_estimate": math.nan,
            "arm1_std_error": math.nan,
            "arm0_estimate": math.nan,
            "arm0_std_error": math.nan,
            "arm_correlation": math.nan,
            "rule_lower": psi - 0.2,
            "rule_upper": psi + 0.2,
            "construction": "skew_t",
        }
    )


@pytest.mark.parametrize(("count", "passes"), [(193, True), (194, False)])
def test_the_tail_clause_either_side_of_its_threshold(count: int, passes: bool) -> None:
    rows = synthetic_rows("lognormal06_ey1", below=count)
    readings = props.cell_readings(rows)
    assert readings["below_count"] == count and readings["above_count"] == 0
    assert readings["tails_passed"] is passes
    mirrored = props.cell_readings(synthetic_rows("lognormal06_ey1", above=count))
    assert mirrored["above_count"] == count and mirrored["tails_passed"] is passes


def test_the_tail_clause_reads_all_4000_and_the_legs_only_the_first_2000() -> None:
    # 194 misses after replicate 2,000: the legs do not see them, the tail clause does.
    rows = synthetic_rows("lognormal06_ey1", below=194)
    readings = pd.Series(props.cell_readings(rows))
    assert readings["leg_replicates"] == 2_000 and readings["leg_coverage"] == 1.0
    assert not bool(readings["tails_passed"])
    # A mutation that read the tails on the first 2,000 only would pass this cell.
    first = rows.loc[rows["replicate"] < 2_000]
    assert props.cell_readings(first.assign(requested_replicates=2_000))["tails_passed"]


def test_the_legs_read_the_first_2000_with_lost_fits_as_misses() -> None:
    # 127 misses among the first 2,000 and 4 fits lost there.
    rows = synthetic_rows("ho_hurdle_ey1", below=127, early_misses=127, lost=(5, 6, 7, 8))
    readings = pd.Series(props.cell_readings(rows))
    assert readings["leg_survivors"] == 1_996
    complete = stats.beta.ppf(0.005, 1_869, 128)
    declared = stats.beta.ppf(0.005, 1_869, 132)
    assert readings["leg_complete_ci_lower"] == pytest.approx(complete, rel=1e-9)
    assert readings["lower_leg_ci_lower"] == pytest.approx(declared, rel=1e-9)
    assert complete >= 0.92 > declared
    summary = props.summarize_properties(rows)
    row = summary.loc[summary["cell"] == "ho_hurdle_ey1"].iloc[0]
    rule = props.pf.rule_for(study.OWNER, props)
    assert rule is not None
    ctx = props.pf.Context(props.STUDY, rows, summary)
    lower = rule.clause("coverage lower leg, first 2,000")
    assert lower.read(ctx, row).passed
    assert lower.candidates is not None
    judged = lower.candidates(ctx, row)
    assert len(judged) == 5 and not judged[0][1].passed and judged[-1][1].passed
    assert props.falsifiers(summary).empty is False


def test_the_tail_candidates_add_lost_fits_to_the_side() -> None:
    rows = synthetic_rows("lognormal06_ey1", below=192, lost=(10, 11))
    summary = props.summarize_properties(rows)
    row = summary.loc[summary["cell"] == "lognormal06_ey1"].iloc[0]
    rule = props.pf.rule_for(study.OWNER, props)
    assert rule is not None
    ctx = props.pf.Context(props.STUDY, rows, summary)
    tail = rule.clause("tail clause, below")
    assert tail.read(ctx, row).passed
    assert tail.candidates is not None
    passes = [judged.passed for _, judged in tail.candidates(ctx, row)]
    assert passes[0] and not passes[-1]


def test_the_upper_leg_reads_shown_fits_of_the_first_2000() -> None:
    rows = synthetic_rows("d1_ey1", family=study.HELDOUT, withheld=0)
    readings = props.cell_readings(rows)
    assert readings["leg_shown"] == 2_000 and readings["leg_shown_ci_upper"] > 0.98
    assert not props.leg_passes(pd.Series(readings))


def test_the_witness_passes_only_when_its_mutation_fails_the_tails() -> None:
    # The source cell's fits read under ``sign_flip``: an acceleration of 0 makes the mutation
    # the capped t itself, so with 194 one-sided misses it fails the tails and the witness
    # passes; with 193 it does not.
    for count, passes in ((194, True), (193, False)):
        rows = synthetic_rows("ho_hurdle_ey1", below=count)
        witness = props.witness_rows(rows, [*rows.columns, "passed", "property_passed"])
        sign_flip = next(row for row in witness if row["cell"] == "sign_flip_witness")
        assert sign_flip["below_count"] == count
        assert sign_flip["passed"] is passes


def test_the_width_bound_is_reproducible_and_paired() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0.9, 1.1, 4_000)
    wide = base * rng.normal(1.05, 0.03, 4_000)
    point, upper = props.width_ratio_bound(wide, base, seed=7)
    assert point == pytest.approx(wide.sum() / base.sum(), rel=1e-12)
    assert props.width_ratio_bound(wide, base, seed=7) == (point, upper)
    se = float(np.std(wide / base, ddof=1) / math.sqrt(base.size))
    # The percentile bound sits about 2.576 SEs above the point (the 0.995 quantile).
    assert upper - point == pytest.approx(2.576 * se, rel=0.15)
    # Paired resampling: unpaired widths would give a much wider bound.
    unpaired = props.width_ratio_bound(wide, rng.permutation(base), seed=7)[1]
    assert unpaired - point > 2.0 * (upper - point)


def test_the_width_falsifier_fires_above_its_threshold() -> None:
    summary = pd.DataFrame(
        [
            {"property": study.HELDOUT, "cell": "c3_ey1", "width_ratio_upper": 1.0801},
            {"property": study.HELDOUT, "cell": "c3_ate", "width_ratio_upper": 1.0799},
        ]
    )
    table = props.falsifiers(summary).set_index(["falsifier", "cell"])
    assert bool(table.loc[("harm (width)", "c3_ey1"), "fired"])
    assert not bool(table.loc[("harm (width)", "c3_ate"), "fired"])
    # Absent cells read nothing and fire nothing.
    assert not bool(table.loc[("mover", "d1_ate"), "fired"])


def _leg_row(cell: str, *, lower: float, upper: float, tails: bool) -> dict[str, Any]:
    return {
        "property": study.HELDOUT,
        "cell": cell,
        "lower_leg_ci_lower": lower,
        "leg_shown_ci_upper": upper,
        "tails_passed": tails,
        "shown_replicates": 4_000,
        "failed_replicates": 0,
        "below_count": 0 if tails else 400,
        "above_count": 0,
        "gain_b": 0,
        "gain_c": 0,
        "width_ratio_upper": 1.0,
    }


@pytest.mark.parametrize(
    ("lower", "upper", "tails", "fired_mover", "fired_means"),
    [
        (0.93, 0.97, True, False, False),
        (0.91, 0.97, True, True, True),
        (0.93, 0.99, True, False, True),
        (0.93, 0.97, False, True, True),
    ],
)
def test_the_falsifier_truth_table(
    lower: float, upper: float, tails: bool, fired_mover: bool, fired_means: bool
) -> None:
    summary = pd.DataFrame(
        [
            _leg_row(cell, lower=lower, upper=upper, tails=tails)
            for cell in ("d1_ate", "d2_ate", "d1_ey1", "d2_ey0", "c1_ey1", "c1_ate", "c2_ey1")
        ]
    )
    table = props.falsifiers(summary).set_index(["falsifier", "cell"])
    # MOVER and skew read the lower leg only (plan 9.6); the means read either leg.
    assert bool(table.loc[("mover", "d1_ate"), "fired"]) is fired_mover
    assert bool(table.loc[("skew", "c2_ey1"), "fired"]) is fired_mover
    assert bool(table.loc[("means", "d2_ey0"), "fired"]) is fired_means


def test_a_falsifier_counts_each_lost_fit_as_a_tail_miss() -> None:
    """Decision m4: the falsifiers' tail clause is the accounting's worst case, not complete case.

    192 misses below among 3,998 shown fits pass complete case; with the two lost fits added to
    each side (194 of 4,000) they fail, so the skew falsifier fires.
    """
    rows = synthetic_rows("c1_ey1", family=study.HELDOUT, below=192, lost=(10, 11))
    summary = props.summarize_properties(rows)
    row = summary.loc[summary["cell"] == "c1_ey1"].iloc[0]
    assert bool(row["tails_passed"]) and int(row["failed_replicates"]) == 2
    assert not props.lower_leg_fails(row)
    table = props.falsifiers(summary).set_index(["falsifier", "cell"])
    assert bool(table.loc[("skew", "c1_ey1"), "fired"])
    # The same misses with nothing lost do not fire it.
    complete = props.summarize_properties(synthetic_rows("c1_ey1", family=study.HELDOUT, below=192))
    assert not bool(
        props.falsifiers(complete).set_index(["falsifier", "cell"]).loc[("skew", "c1_ey1"), "fired"]
    )


def test_the_mover_against_identity_falsifier_reads_c_over_b_plus_c() -> None:
    summary = pd.DataFrame(
        [
            {**_leg_row("d1_ate", lower=0.93, upper=0.97, tails=True), "gain_b": 2, "gain_c": 30},
            {**_leg_row("d2_ate", lower=0.93, upper=0.97, tails=True), "gain_b": 1, "gain_c": 10},
        ]
    )
    row = props.falsifiers(summary).set_index("falsifier").loc["mover against identity"]
    assert row["value"] == pytest.approx(props.gain_lower(40, 3)) and bool(row["fired"])
    # The mutation that read b / (b + c) would not fire on MOVER losing to identity normal.
    assert props.gain_lower(3, 40) < 0.5


def test_the_paired_gain_clause() -> None:
    clause = props._paired_gain()
    ctx = props.pf.Context(props.STUDY, pd.DataFrame(), pd.DataFrame())
    row = pd.Series({"cell": "ho_hurdle_ey1", "gain_b": 8, "gain_c": 0, "failed_replicates": 1})
    assert clause.applies(ctx, row) and clause.read(ctx, row).passed
    assert clause.candidates is not None
    assert [judged.passed for _, judged in clause.candidates(ctx, row)] == [True, False]
    assert not clause.applies(ctx, pd.Series({"cell": "c1_ey1"}))


# --------------------------------------------------------------- the held-out samplers


@pytest.mark.parametrize("law", heldout.HELDOUT_LAWS)
def test_each_sampler_draws_its_law(law: str) -> None:
    """Sample moments of observed variables against the exact law; no estimate, no interval."""
    frame = heldout.sample(law, 200_000, 1)
    spec = heldout.LAWS[law]
    v = 2 if spec.p_v is not None else None
    rows = np.ones(len(frame), dtype=bool) if v is None else frame["V"].to_numpy() == v
    if spec.p_v is not None:
        assert rows.mean() == pytest.approx(spec.p_v[2], abs=4 * math.sqrt(0.25 / len(frame)))
    pw, g, q1, q0, central1, central0 = heldout._cells(law, 0 if v is None else v)
    pw = np.asarray(pw) / np.sum(pw)
    a = frame["A"].to_numpy()[rows]
    y = frame["Y"].to_numpy(dtype=float)[rows]
    treated = float(np.sum(pw * g))
    assert a.mean() == pytest.approx(treated, abs=5 * math.sqrt(treated / rows.sum()))
    for arm, q, central, weight in ((1, q1, central1, g), (0, q0, central0, 1 - g)):
        observed = y[a == arm]
        expected = float(np.sum(pw * weight * q) / np.sum(pw * weight))
        spread = math.sqrt(float(np.mean(central[2])) + float(np.var(np.asarray(q) + 0 * pw)))
        assert observed.mean() == pytest.approx(expected, abs=6 * spread / math.sqrt(observed.size))


@pytest.mark.parametrize("law", heldout.HELDOUT_LAWS)
def test_each_sampler_draws_its_covariate_law(law: str) -> None:
    """``P(V)`` at every stratum and the law of ``W`` within it; no estimate, no interval.

    A wrong covariate sampler would move every truth, which the exact truths cannot see.
    """
    frame = heldout.sample(law, 200_000, 3)
    spec = heldout.LAWS[law]
    w = frame["W"].to_numpy(dtype=float)
    if spec.p_v is None:
        strata = [(np.ones(len(frame), dtype=bool), None)]
    else:
        v = frame["V"].to_numpy()
        assert set(np.unique(v)) == {0, 1, 2}
        strata = [(v == k, k) for k in range(3)]
        for rows, k in strata:
            p = spec.p_v[k]
            assert rows.mean() == pytest.approx(p, abs=5 * math.sqrt(p * (1 - p) / len(frame)))
    for rows, k in strata:
        within = w[rows]
        m = within.size
        if law in ("d1", "c2"):
            table = l1.P_W_GIVEN_V if law == "d1" else heldout.C2_PW
            probabilities = np.asarray(table)[k]
            assert set(np.unique(within)) <= set(range(probabilities.size))
            for value, p in enumerate(probabilities):
                share = float(np.mean(within == value))
                assert share == pytest.approx(p, abs=5 * math.sqrt(p * (1 - p) / m))
        else:
            # Uniform(-1, 1) for C1 and C3, N(0, 1) for D2.
            variance = 1.0 if law == "d2" else 1.0 / 3.0
            fourth = 3.0 if law == "d2" else 1.0 / 5.0
            assert abs(within.mean()) < 5 * math.sqrt(variance / m)
            assert np.var(within) == pytest.approx(
                variance, abs=5 * math.sqrt((fourth - variance**2) / m)
            )
            if law != "d2":
                assert within.min() >= -1.0 and within.max() <= 1.0


def _q(law: str, a: Any, w: Any, v: Any) -> Any:
    return {
        "d1": lambda: heldout.d1_q(a, w, v),
        "d2": lambda: heldout.d2_q(a, w),
        "c1": lambda: heldout.c1_q(a, w, v),
        "c2": lambda: heldout.c2_mu(a, w, v),
        "c3": lambda: heldout.c3_q(a, w, v),
    }[law]()


@pytest.mark.parametrize("law", ("c1", "c2", "c3"))
def test_each_continuous_error_has_its_moments(law: str) -> None:
    """The residual ``Y - Q(A, W, V)`` per arm: its variance and skewness against the law."""
    frame = heldout.sample(law, 200_000, 2)
    a, w, v = (frame[name].to_numpy() for name in ("A", "W", "V"))
    residual = frame["Y"].to_numpy(dtype=float) - _q(law, a, w, v)
    for arm in (0, 1):
        error = residual[a == arm]
        if law == "c1":
            scale = heldout.C1_SCALE[arm]
            variance, skew = 2.0 * scale**2, 2.0 / math.sqrt(2.0)
        elif law == "c3":
            variance, skew = 2.0 * heldout.C3_SCALE[arm] ** 2, 0.0
        else:
            mu = heldout.c2_mu(arm, w[a == arm], v[a == arm])
            central = heldout.beta_central(mu, heldout.C2_PHI)
            variance = float(np.mean(central[2]))
            skew = float(np.mean(central[3])) / variance**1.5
        assert np.var(error) == pytest.approx(variance, rel=0.03)
        assert abs(error.mean()) < 4 * math.sqrt(variance / error.size)
        sample_skew = float(np.mean((error - error.mean()) ** 3) / np.var(error) ** 1.5)
        assert sample_skew == pytest.approx(skew, abs=0.08)


def test_the_failure_accounting_reads_the_declared_clauses() -> None:
    cell = "ho_hurdle_ey1"
    group, declared = study.GROUP_OF_CELL[cell], study.CELL_BY_NAME[cell]
    controls = study.controls_of(group, declared)
    frames = [synthetic_rows(cell, below=127, early_misses=127, lost=(5,))]
    frames += [
        synthetic_rows(f"{cell}__{name}", family=study.CONTROL, role=study.DIAGNOSTIC, lost=(5,))
        for name in controls
    ]
    rows = pd.concat(frames, ignore_index=True)
    summary = props.summarize_properties(rows)
    scenarios = ";".join(
        [f"{study.OWNER}/{cell}", *(f"{study.CONTROL}/{cell}__{name}" for name in controls)]
    )
    record = pd.DataFrame(
        [
            {
                "implementation": study.STUDY.implementation,
                "group_id": "group/ho_hurdle",
                "scenarios": scenarios,
                "replicates": "5",
                "attempt": 1,
                "class": "DataError",
                "stage": "fit",
                "call": "fit",
                "message": "empty_design_cell",
                "seed": 0,
            }
        ]
    )
    accounting = props.pf.property_accounting(
        rows, record, summary, record=study.STUDY, module=props
    )
    legs = list(accounting["leg"])
    assert legs[:6] == [
        "failure bound",
        "coverage lower leg, first 2,000",
        "coverage upper leg, first 2,000, shown only",
        "tail clause, below",
        "tail clause, above",
        "paired gain",
    ]
    lower = accounting.set_index("leg").loc["coverage lower leg, first 2,000"]
    # The lost fit sits among the first 2,000: the worst case counts it as a miss.
    assert lower["worst_case"] < lower["complete_case"]
