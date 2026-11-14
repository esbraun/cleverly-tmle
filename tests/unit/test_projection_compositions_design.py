"""The design of ``projection-compositions`` (P1 plan rev 2.2, section 5.1), without a run.

Every number the plan computed for the laws, the controls and the strata learner is recomputed
here by quadrature at the factor learners' population limits, and the cells and arms are
checked against the plan's table.  The calibration ``R`` is computed from the committed
3,200-draw pilot (:mod:`tests.studies.projection_compositions_pilot`) once it has run; until
then the rule is checked on stated ratios.  The summary step is run on synthetic rows of every
declared cell, and the cost gate reads the committed smoke timings.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies import conditional_projection as cp
from tests.studies import projection_compositions as study
from tests.studies import projection_compositions_pilot as pilot
from tests.studies import projection_compositions_properties as properties
from tests.studies import repeated_sizing
from tests.studies.evidence.property_verdicts import (
    COVERAGE_CONTROLS,
    DIAGNOSTIC_ROLE,
    SE_CONTROL_SIDES,
)


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


def test_the_arms_are_eleven_with_42_gated_rows() -> None:
    # Rev 2.2 (design review S1) dropped ``rep3_sl_2k`` and its arm: its displacement added 0
    # to 0.62% to the SE on eight measured fits, so it measured nothing the GLM repeats do not.
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
    }
    assert sum(sizes.values()) == 42
    assert "rep3_sl_2k" not in study.GROUP_BY_LABEL


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


# ------------------------------------------------------------------- design review rev 2.2


def synthetic_rows(seed: int = 0, replicates: int = 40) -> pd.DataFrame:
    """Rows of every declared cell with the published columns, drawn at a correct interval."""
    rng = np.random.default_rng(seed)
    diagnostic = {f"__{arm}" for arm in properties.DIAGNOSTIC_ARMS}
    rows = []
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            kind = cell.rsplit("__", 1)[-1] if "__" in cell else ""
            if f"__{kind}" in diagnostic:
                role = DIAGNOSTIC_ROLE
            elif kind in properties.POSITIVE_ARMS or family != study.CALIBRATION:
                role = "control" if any(c in cell for c in study.CONTROL_CLAIMS) else "positive"
            else:
                role = "control"
            if family in (study.TYPE_I, study.POWER):
                role = "positive"
            error = rng.normal(0.0, 1.0, replicates)
            rows += [
                {
                    "property": family,
                    "cell": cell,
                    "role": role,
                    "replicate": r,
                    "n": 2_000,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": float(error[r]),
                    "std_error": 1.0,
                    "covered": int(abs(error[r]) <= 1.96),
                    "rejected": int(abs(error[r]) > 1.96),
                    "bootstrap_requested": 200.0 if "boot" in cell else 0.0,
                    "bootstrap_failed": 0.0,
                    "se_without_displacement": 0.99 if "rep3" in cell else np.nan,
                }
                for r in range(replicates)
            ]
    return pd.DataFrame(rows)


def test_the_summary_reads_every_declared_cell_kind() -> None:
    """Design review B1: every calibration kind is one the shared verdict knows."""
    known = {*properties.POSITIVE_ARMS, *properties.DIAGNOSTIC_ARMS, *SE_CONTROL_SIDES}
    known |= set(COVERAGE_CONTROLS)
    for cell in study.PROPERTY_CELLS[study.CALIBRATION]:
        assert "__" in cell, cell
        assert cell.rsplit("__", 1)[-1] in known, cell
    summary = properties.summarize_properties(synthetic_rows())
    assert set(summary["cell"]) == {c for cells in study.PROPERTY_CELLS.values() for c in cells}
    diagnostic = summary.loc[summary["role"] == DIAGNOSTIC_ROLE]
    assert set(diagnostic["cell"]) == {
        c for c in study.PROPERTY_CELLS[study.CALIBRATION] if c.endswith("__boot_stacked")
    }
    assert diagnostic["passed"].all()
    repeated = summary.loc[
        summary["cell"].str.startswith("rep3_obs_4k") & (summary["role"] == "positive")
    ]
    assert repeated["se_ratio_without_displacement_ci_lower"].notna().all()
    assert (
        repeated["se_ratio_without_displacement_ci_lower"]
        <= repeated["se_ratio_without_displacement"]
    ).all()
    assert (
        repeated["se_ratio_without_displacement"]
        <= repeated["se_ratio_without_displacement_ci_upper"]
    ).all()


def test_each_stream_has_its_own_samples_and_folds_per_group() -> None:
    """Design review S2: pilot and smoke draws, and every group, use their own seeds."""
    groups = [group.label for group in study.GROUPS]
    declared = {properties.sample_seed(label, r) for label in groups for r in range(200)}
    folds = {properties.fold_seed(label, r) for label in groups for r in range(200)}
    assert len(declared) == len(folds) == 200 * len(groups)
    for stream in ("pilot", "smoke"):
        assert (
            not {properties.sample_seed(label, r, stream) for label in groups for r in range(200)}
            & declared
        )
        assert (
            not {properties.fold_seed(label, r, stream) for label in groups for r in range(200)}
            & folds
        )
    assert pilot.STREAM == "pilot"


def test_the_pilot_rule_sizes_r_and_the_budget_covers_it() -> None:
    """Design review S2: R comes from the pilot, through X32a's rule and the error correlations."""
    cells = pilot.positive_cells()
    flat = pd.DataFrame({"cell": list(cells), "se_ratio": 1.0, "replicates": pilot.REPLICATES})
    pairs = pd.DataFrame(
        [
            {"first": a, "second": b, "correlation": 0.5}
            for i, a in enumerate(cells)
            for b in cells[i + 1 :]
            if pilot._group_of(a) == pilot._group_of(b)
        ]
    )
    assert pilot.declared_replicates(flat, pairs) <= study.BUDGETED_R
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the pilot has not run; the declaration commit adds it (plan 5.0)")
    summary = pd.read_csv(pilot.PILOT_CSV)
    computed = pilot.declared_replicates(summary, pd.read_csv(pilot.CORRELATIONS_CSV))
    assert computed <= study.BUDGETED_R


def test_the_family_passes_at_the_budget_with_correct_intervals() -> None:
    margins = study.STUDY.margins
    single = float(repeated_sizing.calibration_pass(margins, study.BUDGETED_R, 1.0))
    assert single ** sum(numbers()["calibration_arms"].values()) > 0.99


def test_the_smoke_cost_stays_inside_the_cost_gate() -> None:
    """Design review S7 and N1: measured seconds, and the smoke read against 1.5 times them."""
    timings = pd.read_csv(study.SMOKE_TIMINGS)
    assert set(timings["group"]) == {group.label for group in study.GROUPS}
    for row in timings.itertuples(index=False):
        declared = study.GROUP_BY_LABEL[row.group].seconds
        assert row.seconds_per_fit <= study.COST_GATE * declared, row.group
    assert numbers()["cpu_hours"] == pytest.approx(study.run_cpu_hours())
    assert numbers()["cpu_hours"] + numbers()["pilot_cpu_hours"] < 100.0


def test_the_declaration_names_every_file_a_run_writes() -> None:
    """Design review S7: the files the registration commit commits."""
    from tests.canonical.declared_run import RECORD_FILES
    from tests.canonical.regenerate import ARTIFACT_NAMES, PROPERTY_FAILURE_FILES

    assert set(study.RUN_FILES) == {*ARTIFACT_NAMES, *RECORD_FILES, *study.STUDY.property_artifacts}
    assert set(study.CONDITIONAL_RUN_FILES) == set(PROPERTY_FAILURE_FILES)


def test_the_record_lists_the_median_rule_and_band_modules() -> None:
    """Design review N4."""
    for module in ("influence.py", "results.py", "multiplier.py"):
        assert f"src/cleverly/inference/{module}" in study.STUDY.modules
