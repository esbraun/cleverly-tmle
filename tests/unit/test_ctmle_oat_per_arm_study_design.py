"""The declared design of the registered study ``ctmle-oat-per-arm``.

Every number here was fixed before any registered run: the budgets, the seeds, the laws and
their truths, the power cell's exact-law power, the robustness control's population bias, the
joint cell's budget by the RM36 rule, the cells a run publishes and the revert flag's cells.
A replication that raises is never redrawn: the shared harness refuses a cell that lost one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from cleverly.estimators import ctmle as ctmle_module
from tests.studies import canonical_ctmle_oat_per_arm as study
from tests.studies import ctmle_oat_per_arm_properties as properties
from tests.studies import oat_per_arm_laws as laws
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.default_band_properties import control_power
from tests.studies.evidence.property_verdicts import (
    MINIMUM_POWER,
    design_power,
    power_cell_pass_probability,
)
from tests.studies.evidence.registry import Margins, registered

pytestmark = pytest.mark.xdist_group("ctmle_oat_per_arm_design")


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.calibration_efficiency_ratio is False
    assert study.RED_CELL_OWNER == "F19"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (800, 1_500)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261017, 2026101701)
    assert laws.G_BOUNDS == (0.01, 0.99)
    assert (
        properties.CALIBRATION_N,
        properties.CALIBRATION_REPLICATES,
        properties.REPEATS_REPLICATES,
        properties.NULL_N,
        properties.POWER_N,
        properties.NULL_REPLICATES,
        properties.LADDER,
        properties.LADDER_REPLICATES,
        properties.ROBUSTNESS_N,
        properties.ROBUSTNESS_REPLICATES,
        properties.JOINT_N,
        properties.JOINT_REPLICATES,
    ) == (
        2_000,
        2_000,
        2_000,
        1_000,
        2_000,
        800,
        (500, 2_000, 8_000),
        700,
        2_000,
        1_000,
        2_000,
        2_400,
    )


def test_the_study_is_registered_and_draws_from_its_own_seeds() -> None:
    assert study.STUDY.slug in {record.slug for record in registered()}
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not {study.SEED, study.RESAMPLING_SEED} & others
    seeds = list(properties.SEEDS.values())
    assert len(set(seeds)) == len(seeds)


def test_the_declared_cells_are_the_published_cells() -> None:
    """Every published cell is declared, and every declared cell is published."""
    declared = {(cell.property, cell.cell) for cell in properties.declared_cells()}
    derived = {
        ("interval_calibration", f"{label}__{kind}")
        for label in properties.CALIBRATION_LABELS
        for kind in ("shrunken_se_control", "noise_control")
    }
    rates = {("root_n_rate", "empirical_sd"), ("root_n_rate", "reported_se")}
    published = {(family, cell) for family, cells in study.PROPERTY_CELLS.items() for cell in cells}
    assert declared | derived | rates == published
    assert len(properties.declared_cells()) == len(declared)


def test_every_cell_reads_a_declared_law_and_an_estimand_it_reports() -> None:
    for cell in properties.cells():
        assert cell.dgp in laws.LAWS.values()
        assert cell.estimand in cell.dgp.truth()
    for names, law in (
        (study.BINARY_ESTIMANDS, laws.BINARY_ACTIVE),
        (study.THREE_ARM_ESTIMANDS, laws.THREE_ARM_ACTIVE),
    ):
        assert set(names) <= set(law.truth())
    assert set(properties.JOINT_NAMES) <= set(laws.THREE_ARM_ACTIVE.truth())


def test_the_revert_cells_are_declared_positive_cells() -> None:
    declared = {(cell.property, cell.cell): cell.role for cell in properties.declared_cells()}
    for family, cell in study.REVERT_CELLS:
        assert declared[family, cell] == "positive"
    assert study.REFERENCE_METADATA["revert_flag"].endswith("OAT_PER_ARM_INFERENTIAL")
    # The study's revert cells were red, so the declared revert is applied.
    assert ctmle_module.OAT_PER_ARM_INFERENTIAL is False


def test_the_expected_red_rows_are_declared_rows() -> None:
    """Each expected red row is a row the run publishes, and no oracle-design cell is in it."""
    declared = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
    assert set(study.EXPECTED_RED_PROPERTIES) <= declared
    assert not any("oracle_design" in row for row in study.EXPECTED_RED_PROPERTIES)
    assert all(
        cell.role == "positive"
        for cell in properties.declared_cells()
        if f"{cell.property}/{cell.cell}" in study.EXPECTED_RED_PROPERTIES
    )
    truth = {
        f"{implementation}/{scenario}/{estimand}"
        for implementation in (study.IMPLEMENTATION, study.REFERENCE)
        for scenario, estimands in study.STUDY.scenarios.items()
        for estimand in estimands
    }
    assert set(study.EXPECTED_RED_TRUTH) <= truth
    assert len(study.EXPECTED_RED_TRUTH) == len(truth) - 2


def test_the_joint_fit_builds_the_band_and_leaves_the_flag_unset() -> None:
    """Option (A): the joint cell sets the flag for its own fit and restores it."""
    frame, _ = laws.THREE_ARM_ACTIVE.sample(400, 5)
    result = properties.fit_joint(frame)
    assert result.inference_status == "influence_curve"
    assert result.simultaneous is not None
    assert ctmle_module.OAT_PER_ARM_INFERENTIAL is False
    # The control: the same fit without the context withholds and builds no band.
    withheld = properties._fit_joint(frame)
    assert withheld.inference_status == "generated_design_plugin"
    assert withheld.simultaneous is None
    assert withheld["ey[1]"].psi == result["ey[1]"].psi


@pytest.mark.parametrize("law", list(laws.LAWS.values()), ids=lambda law: law.name)
def test_each_truth_matches_a_monte_carlo(law: laws.OatLaw) -> None:
    """The quadrature truth against ``200,000`` draws, to three Monte Carlo errors."""
    rng = np.random.default_rng(20261017)
    w1, w2, w3 = rng.standard_normal((3, 200_000))
    weights = laws.weight(w1) if law.weighted else np.ones_like(w1)
    q = np.column_stack([law.outcome_probability(a, w1, w2) for a in range(law.k)])
    g = law.mechanism(w1, w3)
    truth = law.arm_means()
    for a in range(law.k):
        values = q[:, a]
        mean = np.average(values, weights=weights)
        error = np.sqrt(np.average((values - mean) ** 2, weights=weights) / values.size)
        assert abs(mean - truth[a]) < 3.0 * error * float(np.max(weights))
    observed = np.sum(g * q, axis=1)
    mean = np.average(observed, weights=weights)
    error = np.sqrt(np.var(observed) / observed.size) * float(np.max(weights))
    assert abs(mean - law.observed_mean()) < 3.0 * error


def test_the_sharp_null_twin_has_a_zero_contrast() -> None:
    truth = laws.BINARY_NULL.truth()
    assert truth["ate"] == pytest.approx(0.0, abs=1e-12)
    assert laws.BINARY_NULL.v == laws.BINARY_ACTIVE.v


@pytest.mark.parametrize("sd", ["POWER_CURVE_SD", "EIF_CURVE_SD"])
def test_the_power_cell_is_sized_from_the_exact_law(sd: str) -> None:
    """The cell clears the floor whether or not the superefficiency is realized at its size."""
    contrast = laws.BINARY_ACTIVE.truth()["ate"]
    power = design_power(contrast, getattr(properties, sd), properties.POWER_N)
    assert power >= MINIMUM_POWER
    assert power_cell_pass_probability(power, properties.NULL_REPLICATES) >= 0.95


def test_the_efficient_sd_is_the_per_arm_sd_over_the_root_efficiency_ratio() -> None:
    """The declared efficiency ratio of the binary ``ate`` at ``10^6`` draws is 0.4279."""
    ratio = (properties.POWER_CURVE_SD / properties.EIF_CURVE_SD) ** 2
    assert ratio == pytest.approx(0.4279, abs=5e-4)


@pytest.mark.parametrize(
    "cell",
    [
        cell
        for cell in properties.cells()
        if cell.role == "positive" and cell.property in ("interval_calibration", "generated_design")
    ],
    ids=lambda cell: cell.cell,
)
def test_each_calibration_cell_is_sized_for_its_pass_probability(cell: Any) -> None:
    """A perfectly calibrated cell passes the calibration rule with probability at least 0.90.

    The rule needs the 99% bootstrap interval of the SE ratio inside (0.93, 1.07) and the
    coverage interval inside (0.92, 0.98).  The generated-design pair answers to the same rule.
    """
    margins = Margins()
    probability = calibration_pass_probability(
        cell.replicates,
        true_se_ratio=1.0,
        confidence_level=margins.confidence_level,
        se_band=margins.calibration_se_ratio,
        coverage_band=margins.calibration_coverage,
    )
    assert probability >= 0.90, (cell.cell, cell.replicates, probability)


def test_the_robustness_control_can_fail() -> None:
    """Its population bias sits far outside the margin at its own size and budget."""
    margin = Margins().standardized_bias * properties.CONTROL_SD
    half_width = 2.576 * properties.CONTROL_SD / math.sqrt(properties.ROBUSTNESS_REPLICATES)
    assert properties.CONTROL_BIAS - half_width > 4.0 * margin


def test_the_joint_budget_follows_the_rm36_rule() -> None:
    assert control_power(properties.JOINT_P0, properties.JOINT_REPLICATES) >= 0.99


def test_the_law_reproduces_the_declared_design_numbers() -> None:
    """A recomputation of the law-level numbers at ``5,000`` draws.

    The declared constants come from ``10^6`` draws.  At ``5,000`` draws a curve standard
    deviation moves by about one percent, so three percent is the Monte Carlo tolerance.
    """
    declared = {
        "binary_active": (properties.CURVE_SD["binary_active"], properties.EIF_CURVE_SD),
        "three_arm_active": (properties.CURVE_SD["three_arm_active"], None),
    }
    assert properties.CURVE_SD["binary_active"] == properties.POWER_CURVE_SD
    for law in (laws.BINARY_ACTIVE, laws.THREE_ARM_ACTIVE):
        numbers = laws.design_numbers(law, 5_000, 20261006)
        for low, high in numbers["range"]:
            assert laws.G_BOUNDS[0] < low < high < laws.G_BOUNDS[1]
        assert min(numbers["gap"]) > 0.02
        # 0.792 at the declared 10^6 draws; 5,000 draws move it by a few hundredths.
        assert max(numbers["ratio"]) <= 0.85
        assert max(numbers["learner"]) < 0.01
        curve_sd, eif_sd = declared[law.name]
        assert numbers["ate_sd"][0] == pytest.approx(curve_sd, rel=0.03)
        if eif_sd is not None:
            assert numbers["ate_eif_sd"][0] == pytest.approx(eif_sd, rel=0.03)
