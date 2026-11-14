"""The design of ``projection-compositions`` (P1 plan rev 2.1, section 5.1), without a run.

Every number the plan computed for the laws, the controls and the strata learner is recomputed
here by quadrature at the factor learners' population limits, and the cells and arms are
checked against the plan's table.  The calibration ``R`` needs the 3,200-draw pilots, which
have not run; the declaration commit adds that computation.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest

from tests.studies import conditional_projection as cp
from tests.studies import projection_compositions as study
from tests.studies import projection_compositions_properties as properties


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


def test_the_arms_are_the_plans_twelve_with_45_gated_rows() -> None:
    sizes = numbers()["calibration_arms"]
    assert sizes == {
        "__wald_diff": 3,
        "__boot_diff": 3,
        "__wald_crr": 3,
        "__boot_crr": 3,
        "__mtreat": 2,
        "__mtreat_both": 2,
        "__mtreat_crr": 2,
        "__strata": 9,
        "__strata_crr": 9,
        "__rep3": 3,
        "__rep3_crr": 3,
        "__rep3_sl": 3,
    }
    assert sum(sizes.values()) == 45


def test_every_cell_is_declared_once_and_the_record_reads_them() -> None:
    cells = [cell for names in study.PROPERTY_CELLS.values() for cell in names]
    assert len(cells) == len(set(cells))
    assert study.STUDY.property_cells == study.PROPERTY_CELLS
    assert properties.PROPERTY_CELLS is study.PROPERTY_CELLS
    assert study.PROPERTY_CELLS["type_i_error"] == ("heterogeneity__rep3_sharp_null",)
    assert study.PROPERTY_CELLS["power"] == ("heterogeneity__rep3_slope_0.10",)


def test_the_declared_c_is_2_and_the_complete_case_control_clears_its_floor() -> None:
    # Plan 5.1: c = 2.0 unless the intercept's complete-case displacement is below 0.6 SD.
    assert numbers()["c"] == study.TREATMENT_RECORDING_C == 2.0
    intercept, slope = numbers()["complete_case_sd"]
    assert intercept == pytest.approx(0.825, abs=0.01)
    assert intercept >= study.COMPLETE_CASE_FLOOR
    assert abs(slope) < 0.05  # W1 is not a control row
    assert cp.bias_pass_probability(intercept, 1_000, control=True) > 0.9999


def test_the_dropped_treatment_observation_control_fails_on_the_intercept_only() -> None:
    intercept, slope = numbers()["no_treatment_observation_sd"]
    assert intercept == pytest.approx(-2.54, abs=0.03)
    assert cp.bias_pass_probability(intercept, 1_000, control=True) > 0.9999
    # W1 at about -0.51 SD is too near the 0.5 SD row rule to be declared.
    assert study.GROUP_BY_LABEL["mtreat_dr_4k"].controls == {
        "no_treatment_observation": ("(intercept)",)
    }
    assert slope == pytest.approx(-0.51, abs=0.03)


def test_the_targeted_mtreat_dr_point_is_unbiased_at_the_glm_limits() -> None:
    for bias in numbers()["mtreat_dr_bias_sd"]:
        assert abs(bias) < 0.05
        assert cp.bias_pass_probability(bias, 1_000, control=False) >= 0.985


def test_the_mtreat_factors_stay_positive_and_near_their_glm_limits() -> None:
    sd = numbers()["mtreat_sd"]
    assert sd == pytest.approx((0.0213, 0.0361), abs=5e-4)
    assert numbers()["smallest_g_c"] == pytest.approx(0.043, abs=0.002)
    # The plan's Monte Carlo read 3.7%; the quadrature limit is the reading.
    assert numbers()["factor_relative_error"] < 0.04


def test_each_stratum_differs_from_the_marginal_where_the_control_reads() -> None:
    intercept, w1, w3 = numbers()["stratum_gap_sd"]
    assert intercept == pytest.approx(2.80, abs=0.03)
    assert w1 == pytest.approx(-1.35, abs=0.03)
    assert abs(w3) < 0.01  # W3 is not a control row
    assert study.GROUP_BY_LABEL["strata_4k"].controls == {
        "marginal_as_stratum": ("(intercept)", "W1")
    }
    for gap in (intercept, w1):
        assert cp.bias_pass_probability(gap, study.BUDGETED_R, control=True) > 0.9999


def test_the_strata_learner_is_within_the_shipped_learners_approximation_error() -> None:
    # Root-mean-square error of the arm-1 regression at the population limit, under the law
    # of the arm-1 rows: 0.00739 for the strata learner and 0.00738 for the shipped spline GLM
    # (ratio 1.002); the strata law's own terms S and S W1 are in the strata learner's span.
    ratio = numbers()["strata_learner_error"] / numbers()["shipped_learner_error"]
    assert ratio <= 1.01


def test_the_repeated_power_cell_passes() -> None:
    assert numbers()["power"] == pytest.approx(0.933, abs=0.002)
    assert numbers()["power_noncentrality"] == pytest.approx(14.27, abs=0.02)
    assert cp.type_i_pass_probability(0.05, 2_400) > 0.999


def test_the_truths_are_the_full_data_projections() -> None:
    mtreat = study.truth("mtreat")
    assert set(mtreat) == {"cate_projection[(intercept)]", "cate_projection[W1]"}
    strata = study.truth("strata", "log_ratio")
    assert len(strata) == 9
    assert study.truth("observational") == cp.truth("observational")
