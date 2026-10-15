"""Design of the registered study ``ctmle-oat-per-arm-parametric`` (X58a).

The numbers here are fixed before any run.  Each test names the part of the plan it pins:
the cells of section 5.2 with the budgets of section 11, the power cell's size, the fresh
seeds, the failure bound, the two negative controls with their exact-law separation, and the
admission of every cell configuration by the support predicate.  No test fits more than a few
replicates, and none runs a study.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from cleverly.estimators.ctmle import per_arm_parametric_support
from tests.studies import canonical_ctmle_oat_per_arm_parametric as runner
from tests.studies import ctmle_oat_per_arm_parametric_properties as properties
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws
from tests.studies.evidence.property_verdicts import (
    MINIMUM_POWER,
    SE_CONTROL_SIDES,
    design_power,
    power_cell_pass_probability,
)
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: Plan 5.2 cell -> (replicates, n).  The replicates are the budgets of plan section 11, except
#: the three cells that section does not size (see ``runner.BUDGETS``).
PLAN = {
    "binary_active__corrected": (6_800, 1_500),
    "shipped_curve_control": (6_800, 1_500),
    "n_500": (14_400, 500),
    "n_2000": (4_400, 2_000),
    "n_8000": (4_400, 8_000),
    "shared_prognosis__corrected": (4_800, 1_500),
    "cross_term_dropped_control": (4_800, 1_500),
    "shared_prognosis__ratios": (4_800, 1_500),
    "kappa_law__corrected": (8_400, 1_500),
    "bounds_active__corrected": (5_200, 1_500),
    "auto_bounds__corrected": (4_400, 1_500),
    "bounded_outcome__corrected": (6_800, 1_500),
    "pseudo_true__corrected": (4_400, 1_500),
    "oracle_design_against_estimated": (4_400, 1_500),
    "sharp_null": (800, 1_000),
    "alternative": (1_200, 800),
}
#: The families the harness classifies and this study uses.
FAMILIES = {"interval_calibration", "generated_design", "type_i_error", "power"}
#: The calibration rows the pooled-tau budget rule gates (plan section 11).
GATED_ROWS = 33


def _study() -> StudyRecord:
    (record,) = (study for study in registered() if study.slug == "ctmle-oat-per-arm-parametric")
    return record


def test_the_record_registers_with_the_declared_design() -> None:
    record = _study()
    assert record is runner.STUDY
    assert record.reference is None
    assert record.comparator_state == "none"
    assert record.publication_policy == "gated"
    assert record.calibration_efficiency_ratio is False
    assert record.implementation == runner.IMPLEMENTATION
    assert record.replicates == runner.PRIMARY_REPLICATES == 800
    assert record.n == runner.PRIMARY_N == 1_500
    assert record.scenarios == {"binary_active": ("ey0", "ey1", "ate", "rr", "or")}
    assert set(record.property_cells) == FAMILIES
    assert record.artifacts == ROOT / "tests" / "canonical" / "ctmle_oat_per_arm_parametric"
    assert (record.artifacts / "__init__.py").is_file()
    assert (record.artifacts / "regenerate.py").is_file()
    assert record.runner() is runner
    assert record.properties() is properties
    assert record.property_cells == runner.property_cells()


def test_every_plan_cell_exists_with_its_budget_and_size() -> None:
    assert set(runner.PLAN_CELLS) == set(PLAN)
    for plan, (replicates, n) in PLAN.items():
        group = runner.GROUPS[runner.PLAN_CELLS[plan]]
        assert (group.replicates, group.n) == (replicates, n), plan
        cells = [cell for cell in runner.CELLS if cell.plan == plan]
        assert cells, f"the plan cell {plan} publishes nothing"
        assert {cell.group for cell in cells} == {group.name}, plan
    assert {cell.plan for cell in runner.CELLS} == set(PLAN)
    assert set(runner.BUDGETS) == set(runner.GROUPS)


def test_the_cells_the_summary_must_contain_are_the_declared_ones() -> None:
    declared = {name for names in _study().property_cells.values() for name in names}
    assert declared == {cell.name for cell in runner.CELLS} | set(runner.DERIVED_CONTROLS)
    assert len(declared) == len(runner.CELLS) + len(runner.DERIVED_CONTROLS)
    positive = [
        cell
        for cell in runner.CELLS
        if cell.family == "interval_calibration" and cell.role == "positive"
    ]
    assert len(positive) == GATED_ROWS
    assert all(cell.name.endswith("__corrected") for cell in positive)
    assert {cell.family for cell in runner.CELLS} == FAMILIES


def test_the_two_negative_controls_are_declared_and_read_the_right_side() -> None:
    controls = [cell for cell in runner.CELLS if cell.role == "control"]
    assert {cell.name for cell in controls} == set(runner.NEGATIVE_CONTROLS)
    for cell in controls:
        kind = cell.name.rsplit("__", 1)[-1]
        assert SE_CONTROL_SIDES[kind] == runner.NEGATIVE_CONTROLS[cell.name]
        assert cell.family == "interval_calibration"
    kinds = {runner.NEGATIVE_CONTROLS[cell.name] for cell in controls}
    assert kinds == {"below", "above"}
    reading = runner.CONFIGURATION["declared_reading"]
    assert "shipped_curve_control" in reading and "cross_term_dropped_control" in reading
    assert "must be red" in reading
    assert "*must* be red" in (runner.__doc__ or "")
    assert runner.STUDY.publication_policy == "gated"


def test_each_negative_control_leaves_the_band_at_its_declared_size() -> None:
    """The exact-law SE ratio of each control, and the interval it reads at its budget.

    A control passes by leaving the calibration band.  The 99% interval of an SE ratio has the
    relative half-width ``z / sqrt(2 (R - 1))``, so a control whose exact ratio sits that far
    from the band edge cannot be read inside it.
    """
    low, high = runner.STUDY.margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + runner.STUDY.margins.confidence_level / 2.0))
    fit = quad.population_fit(quad.BINARY_ACTIVE, nodes=20)
    shipped = quad.shipped_curves(fit)
    mass = fit.grid.mass
    cov = (shipped * mass[:, None]).T @ shipped
    shipped_ratio = np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]) / quad.ate_sd(fit)
    shared = quad.population_fit(quad.SHARED_PROGNOSIS, nodes=20)
    arms = quad.covariance(shared)
    dropped_ratio = np.sqrt(arms[0, 0] + arms[1, 1]) / quad.ate_sd(shared)
    assert shipped_ratio == pytest.approx(0.845, abs=0.01)
    assert dropped_ratio == pytest.approx(1.228, abs=0.01)
    r_shipped = runner.BUDGETS["binary_active"]
    r_dropped = runner.BUDGETS["shared_prognosis"]
    assert shipped_ratio * (1.0 + z / np.sqrt(2.0 * (r_shipped - 1))) < low
    assert dropped_ratio * (1.0 - z / np.sqrt(2.0 * (r_dropped - 1))) > high


def test_the_power_cell_runs_at_the_size_whose_exact_law_power_is_in_range() -> None:
    n = runner.GROUPS["power"].n
    assert n == 800
    ate = runner.truth("binary_active", "ate")
    sd = runner.curve_sd()["ate"]
    assert sd == pytest.approx(1.1234, abs=1e-3)
    power = design_power(ate, sd, n)
    assert 0.80 <= power <= 0.90
    # n = 700 is below the floor, which is why the cell moved to 800.
    assert design_power(ate, sd, 700) < MINIMUM_POWER
    declared = runner.GROUPS["power"].replicates
    assert power_cell_pass_probability(power, declared) >= 0.95
    assert power_cell_pass_probability(power, declared - 400) < 0.95


def test_the_seeds_are_fresh() -> None:
    others = [study for study in registered() if study is not runner.STUDY]
    taken = {study.seed for study in others} | {
        study.resampling_seed for study in others if study.resampling_seed is not None
    }
    assert runner.SEED not in taken
    assert runner.RESAMPLING_SEED not in taken
    assert runner.SEED != runner.RESAMPLING_SEED
    assert runner.STUDY.seed == runner.SEED
    assert runner.STUDY.resampling_seed == runner.RESAMPLING_SEED
    # No sample seed of this study is any other study's primary seed, for the first replicates.
    primary = {
        replicate_seed(study, scenario, replicate)
        for study in others
        for scenario in study.scenarios
        for replicate in range(50)
    }
    own = {replicate_seed(runner.STUDY, runner.SCENARIO, replicate) for replicate in range(50)}
    own |= {
        stream_seed(runner.STUDY, "property_sample", group, replicate)
        for group in runner.GROUPS
        for replicate in range(50)
    }
    assert not own & primary
    assert len(own) == 50 + 50 * len(runner.GROUPS)


def test_the_failure_bound_is_one_percent_in_every_module() -> None:
    assert runner.FAILURE_BOUND == 0.01
    assert properties.FAILURE_BOUND == 0.01
    assert not runner.STUDY.failure_bound


@pytest.mark.parametrize(
    ("group", "fit"),
    [(group.name, fit) for group in runner.GROUPS.values() for fit in group.fits],
)
def test_each_cell_configuration_is_admitted_by_the_support_predicate(
    group: str, fit: runner.Fit
) -> None:
    spec = runner.GROUPS[group]
    frame = runner.sample(spec.law, 300, 17)
    subject = runner.estimator(fit)
    data = subject._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=list(fit.covariates),
        delta=None,
        weights=None,
        id=None,
        intermediate=None,
        strata=None,
    )
    verdict = per_arm_parametric_support(subject, data)
    if fit.outcome == "parametric":
        assert verdict.admitted, verdict.detail
    else:
        # The oracle arm pins Qbar_0, so the predicate refuses it and the fit keeps the shipped
        # curve: the pair compares the stacked curve against that curve.
        assert not verdict.admitted
        assert verdict.axis == "outcome_learner"


def test_the_reported_pseudo_true_coverage_has_a_nonzero_gap_to_report() -> None:
    star = runner.truth("pseudo_true", "ate")
    true = runner.truth("binary_active", "ate")
    assert abs(star - true) > 0.05
    assert runner.truth("bounded", "ate") == pytest.approx(8.0 * true)
    assert runner.truth("bounded", "ey0") == pytest.approx(
        2.0 + 8.0 * runner.truth("binary_active", "ey0")
    )
    assert runner.truth("binary_null", "ate") == pytest.approx(0.0, abs=1e-12)
    for estimand in ("log_rr", "log_or"):
        assert runner.truth("shared_prognosis", estimand) != 0.0


def test_a_replicate_reads_every_cell_of_its_group_and_the_shipped_curve_is_narrower() -> None:
    key = ("binary_active", 0)
    measured = properties._measure(key, properties._draw(key))
    names = [item["cell"] for item in measured]
    assert names == [cell.name for cell in runner.cells_of("binary_active")]
    se = {item["cell"]: item["std_error"] for item in measured}
    assert (
        se["binary_active__ate__shipped_curve__shrunken_se_control"]
        < se["binary_active__ate__corrected"]
    )
    rows = properties._rows(runner.GROUPS["binary_active"], 0, measured, 6_800)
    assert [row["cell"] for row in rows] == names
    assert {row["requested_replicates"] for row in rows} == {6_800}
    assert {row["failed_replicates"] for row in rows} == {0}
    roles = {row["cell"]: row["role"] for row in rows}
    assert roles["binary_active__ate__shipped_curve__shrunken_se_control"] == "control"
    assert roles["binary_active__ate__corrected"] == "positive"


def test_the_primary_phase_draws_from_the_published_seed() -> None:
    frame, truths = runner.draw_scenario(runner.SCENARIO, 200, 3)
    again, _ = runner.draw_from_seed(
        runner.SCENARIO, 200, replicate_seed(runner.STUDY, runner.SCENARIO, 3)
    )
    assert frame.equals(again)
    assert set(truths) == set(runner.BINARY_ESTIMANDS)
    assert truths["ate"] == pytest.approx(laws.BINARY_ACTIVE.truth()["ate"])
