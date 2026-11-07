"""The design numbers of the ``targeting-curves`` study, recomputed (X92 PR-5 plan, section 10).

The declaration's numbers are computed, not copied: these tests recompute each one from
:func:`tests.studies.targeting_curves.design_numbers` and the exact truths, and fail when a
declared constant drifts from what it was computed from.  The pilot tests read the committed
pilot CSV and skip until the pilot has run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import qmc

from cleverly import targeting_curves as tc
from tests.studies import targeting_curves as study


@pytest.fixture(scope="module")
def design() -> dict:
    return study.design_numbers()


def test_the_calibration_sizing_is_the_plan_s(design: dict) -> None:
    calibration = design["calibration"]
    assert calibration["rows"] == 34
    assert calibration["pass_at_1"] > calibration["pass_at_1_r3200"]
    assert calibration["family"] >= 0.96
    assert calibration["pass_at_0.97"] == pytest.approx(0.81, abs=0.01)


def test_every_control_is_detected(design: dict) -> None:
    for name, control in design["controls"].items():
        assert control["detection"] >= 0.90, name


def test_the_fixed_cut_ratios_match_the_plan_s_monte_carlo(design: dict) -> None:
    """1.011 / 1.242 / 1.714 at q = 0.1 / 0.5 / 0.9 from a 4-million-draw Monte Carlo."""
    controls = design["controls"]
    assert design["reported"]["H_toc_0.1__fixed_cut_ratio"] == pytest.approx(1.011, abs=0.01)
    assert controls["H_toc_0.5__fixed_cut"]["ratio"] == pytest.approx(1.242, abs=0.015)
    assert controls["H_toc_0.9__fixed_cut"]["ratio"] == pytest.approx(1.714, abs=0.02)


def test_the_quantile_influence_function_matches_a_monte_carlo() -> None:
    """The TOC's SD from :func:`influence_sd` against the estimator's own spread at n = 2,000.

    The draws are the oracle scores ``Gamma = tau + noise``, ranked by the score, so the
    estimator is the paper's (2.8) with true nuisances.
    """
    law = study.LAWS["H"]
    grid = study.law_grid(law)
    rng = np.random.default_rng(0)
    n, reps = 2_000, 1_500
    draws = []
    for _ in range(reps):
        w = rng.uniform(-1.0, 1.0, (n, 3))
        gamma = law.tau(w) + np.sqrt(law.gamma_variance(w)) * rng.normal(size=n)
        top = np.argsort(-law.score_of(w))[: n // 2]
        draws.append(gamma[top].mean() - gamma.mean())
    spread = float(np.std(draws, ddof=1) * math.sqrt(n))
    assert spread == pytest.approx(study.influence_sd(grid, "toc", 0.5), rel=0.06)


def test_the_fixed_rank_control_reads_the_side_its_ratio_is_on(design: dict) -> None:
    ratio = design["controls"]["H_autoc__fixed_ranks"]["ratio"]
    expected = "inflated_se_control" if ratio > 1.0 else "shrunken_se_control"
    assert expected == study.FIXED_RANKS_KIND


def test_the_power_slope_gives_the_declared_power() -> None:
    power = study.band_power(study.P_SLOPE, study.GROUP_BY_LABEL["P"].n)
    assert power == pytest.approx(0.90, abs=0.01)


def test_the_power_cell_passes_at_and_near_its_power(design: dict) -> None:
    assert design["power"]["pass_at_0.90"] > 0.999
    assert design["power"]["pass_at_0.87"] > 0.99


@pytest.mark.parametrize("label", ["H", "B", "K", "D"])
def test_the_exact_truths_match_a_sobol_average(label: str) -> None:
    """The quadrature TOC and group effects against the rank functionals of 2^20 points."""
    law = study.LAWS[label]
    curve = study.Curve(law)
    w = 2.0 * qmc.Sobol(3, scramble=True, seed=11).random(2**20) - 1.0
    score = law.score_of(w)
    order = np.argsort(-score, kind="stable")
    _, starts = np.unique(-score[order], return_index=True)
    mass = np.add.reduceat(np.ones(len(w)), starts)
    sums = np.add.reduceat(law.tau(w)[order], starts)
    weights = tc.block_weights(mass, (0.1, 0.5, 0.9), (0.2, 1.0))
    for q in (0.1, 0.5, 0.9):
        approx = float(np.sum(weights[("toc", q)] * sums) / len(w))
        assert approx == pytest.approx(curve.toc(q), abs=2e-3), q
    approx = float(np.sum(weights[("group_effect", 0.2)] * sums) / len(w))
    assert approx == pytest.approx(curve.group(0.0, 0.2), abs=2e-3)


def test_the_null_laws_have_a_flat_curve() -> None:
    for label in ("U", "N"):
        curve = study.Curve(study.LAWS[label])
        for q in (0.1, 0.5, 0.9):
            assert abs(curve.toc(q)) < 1e-10


def test_the_atomic_law_cuts_every_grid_point_inside_an_atom() -> None:
    """Cell D's claim: no grid point sits at an atom's edge (shares 1/8, 1/4, 1/4, 1/4, 1/8)."""
    edges = np.cumsum([0.125, 0.25, 0.25, 0.25, 0.125])
    for q in study.GRID[:-1]:
        assert np.min(np.abs(edges - q)) > 0.02


def test_the_declared_cells_are_the_plan_s() -> None:
    cells = study.PROPERTY_CELLS
    positives = [cell for cell in cells[study.CALIBRATION] if study.is_positive(cell)]
    assert len(positives) == 34
    assert set(cells[study.BIAS]) == {
        "D_toc_0.2__whole_block",
        "S_toc_0.1__in_sample",
        "S_autoc__in_sample",
    }
    assert cells[study.TYPE_I] == ("U_band__null", "N_band__null")
    assert cells[study.POWER] == ("P_band__slope",)


PILOT = study.pilot_rows()
needs_pilot = pytest.mark.skipif(not PILOT, reason="the targeting-curves pilot has not run")


@needs_pilot
def test_the_pilot_ratios_keep_the_declared_replicates() -> None:
    """Any pilot ratio more than 0.02 from 1 moves the design to R = 4,800 (plan, sizing)."""
    far = [
        key
        for key, row in PILOT.items()
        if key[1] == study.POSITIVE and abs(float(row["se_ratio"]) - 1.0) > 0.02
    ]
    assert not far or study.R_MAIN >= 4_800, far


@needs_pilot
def test_the_pilot_kurtosis_keeps_the_declared_replicates() -> None:
    worst = max(
        float(row["kurtosis"]) for key, row in PILOT.items() if key[1] in study.POSITIVE_KINDS
    )
    per_row = study.calibration_row_pass(study.R_MAIN, 1.0, kurtosis=max(3.0, worst))
    positives = [
        cell for cell in study.PROPERTY_CELLS[study.CALIBRATION] if study.is_positive(cell)
    ]
    assert per_row ** len(positives) >= 0.90


def test_the_untargeted_control_reads_the_side_its_ratio_is_on(design: dict) -> None:
    """Cell K's one-step draws are narrower than its targeted draws on AUTOC only.

    At TOC(0.5) and Qini the ratio is 1 to within the band, so those rows carry no control:
    a control row there could not leave the band.
    """
    ratio = design["controls"]["K_autoc__untargeted"]["ratio"]
    assert ratio == pytest.approx(0.80, abs=0.05)
    assert ("inflated" if ratio > 1.0 else "shrunken") == study.K_UNTARGETED_SIDE
    assert study.GROUP_BY_LABEL["K"].controls["untargeted"] == ("autoc",)
    low, high = study.STUDY.margins.calibration_se_ratio
    for name in ("K_toc_0.5__untargeted_ratio", "K_qini__untargeted_ratio"):
        assert low < design["reported"][name] < high, name


def test_the_pointwise_joint_coverage_is_the_exact_law_s(design: dict) -> None:
    """p0 = 0.754 from the TOC covariance on cell H (the review's computation)."""
    control = design["controls"]["H_band__pointwise_joint"]
    assert control["coverage"] == pytest.approx(0.754, abs=0.01)


def test_cell_k_s_positives_do_not_claim_a_correct_specification() -> None:
    cells = study.PROPERTY_CELLS[study.CALIBRATION]
    k_cells = [cell for cell in cells if cell.startswith("K_")]
    assert k_cells
    assert not [cell for cell in k_cells if cell.endswith(f"__{study.POSITIVE}")]
    assert "K_autoc__known_mechanism" in k_cells
    assert "K_untargeted_autoc__shrunken_se_control" in k_cells


@needs_pilot
def test_the_in_sample_control_is_detected() -> None:
    for name in study.GROUP_BY_LABEL["S"].controls["in_sample"]:
        shift = float(PILOT[("S", "in_sample", name)]["standardized_bias"])
        assert study.bias_pass_probability(shift, study.R_MAIN, control=True) >= 0.90, name


@needs_pilot
def test_the_split_truth_is_accurate_to_a_hundredth_of_an_se() -> None:
    for name in study.GROUP_BY_LABEL["S"].rows:
        error = float(PILOT[("S", "sobol_error", name)]["sobol_error"])
        assert error <= 0.01 * float(PILOT[("S", study.POSITIVE, name)]["mean_se"]), name
