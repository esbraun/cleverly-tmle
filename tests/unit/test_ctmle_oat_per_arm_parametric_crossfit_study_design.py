"""Design of the registered study ``ctmle-oat-per-arm-parametric-crossfit`` (X58b1).

The numbers here are fixed before any run.  Each test names the part of the plan it pins:
the groups, laws and budgets of section 13.6, the unit count ``G = 9`` and its per-unit target,
the type I size, the fresh seeds, the failure bound, the two negative controls with their
exact-law separation, the stored shipped standard error that the repeated control reads, and
the admission of every configuration by the support predicate.  No test fits more than a few
replicates, and none runs a study.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom, norm

from cleverly.estimators.ctmle import per_arm_parametric_support
from tests.studies import canonical_ctmle_oat_per_arm_parametric_crossfit as runner
from tests.studies import ctmle_oat_per_arm_parametric_crossfit_properties as properties
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.property_verdicts import SE_CONTROL_SIDES
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: Plan 13.6 group -> (replicates, n).  The replicates are the rev 3.1 expected budgets.
PLAN = {
    "crossfit_1500": (15_200, 1_500),
    "crossfit_500": (64_800, 500),
    "crossfit_8000": (10_000, 8_000),
    "crossfit_pseudo_true": (10_000, 1_500),
    "crossfit_generated_design": (10_800, 1_500),
    "repeats": (10_800, 1_500),
    "crossfit_null": (1_200, 1_000),
}
#: The families the harness classifies and this study uses.  The study has no power cell.
FAMILIES = {"interval_calibration", "generated_design", "type_i_error"}
#: The positive calibration rows: five at ``n = 1,500``, three at each of the other four groups.
GATED_ROWS = 5 + 3 + 3 + 3 + 3
#: The study-wise units of plan 13.6: six calibration groups, one type I group, no primary phase
#: and two negative controls.
UNITS = 9


def _study() -> StudyRecord:
    (record,) = (
        study for study in registered() if study.slug == "ctmle-oat-per-arm-parametric-crossfit"
    )
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
    assert (
        record.artifacts == ROOT / "tests" / "canonical" / "ctmle_oat_per_arm_parametric_crossfit"
    )
    assert (record.artifacts / "__init__.py").is_file()
    assert (record.artifacts / "regenerate.py").is_file()
    assert record.runner() is runner
    assert record.properties() is properties
    assert record.property_cells == runner.property_cells()


def test_every_group_exists_with_its_budget_and_size() -> None:
    assert set(runner.PLAN_CELLS) == set(PLAN)
    for name, (replicates, n) in PLAN.items():
        group = runner.GROUPS[runner.PLAN_CELLS[name]]
        assert (group.replicates, group.n) == (replicates, n), name
        assert replicates % 400 == 0
        assert [cell for cell in runner.CELLS if cell.plan == name], name
    assert {cell.plan for cell in runner.CELLS} == set(PLAN)
    assert set(runner.BUDGETS) == set(runner.GROUPS)
    assert sum(runner.BUDGETS.values()) == 122_800


def test_every_group_is_cross_fitted_at_the_studied_values() -> None:
    for group in runner.GROUPS.values():
        for fit in group.fits:
            subject = runner.estimator(fit)
            assert subject.cross_fit is True
            assert subject.n_folds == runner.FOLDS == 5
            assert subject.simultaneous is False
            assert subject.repeats == (3 if group.name == "repeats" else 1)


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
    # Plan 13.6: the gated rows of crossfit_1500 are ey0, ey1, ate, log_rr and log_or.
    rows = {cell.estimand for cell in runner.cells_of("crossfit_1500") if cell.role == "positive"}
    assert rows == {"ey0", "ey1", "ate", "log_rr", "log_or"}
    # Every group's ate row has the two derived controls; the null group has none.
    assert {label.removesuffix("__ate") for label in runner.CONTROL_LABELS} == {
        "crossfit_1500",
        "crossfit_500",
        "crossfit_8000",
        "crossfit_pseudo_true",
        "repeats",
    }


def test_the_study_has_nine_units_and_each_is_sized_to_the_per_unit_target() -> None:
    assert dict(runner.UNITS) == {
        "calibration_groups": 6,
        "type_i_groups": 1,
        "primary_phase": 0,
        "negative_controls": 2,
    }
    assert runner.UNIT_COUNT == UNITS
    calibration = {
        cell.group
        for cell in runner.CELLS
        if cell.family in ("interval_calibration", "generated_design")
    }
    assert len(calibration) == runner.UNITS["calibration_groups"]
    assert len(runner.NEGATIVE_CONTROLS) == runner.UNITS["negative_controls"]
    target = 0.95 ** (1.0 / UNITS)
    assert target == pytest.approx(0.99432, abs=1e-5)
    assert target < 0.95 ** (1.0 / 19) < 1.0


def test_the_type_i_group_passes_at_its_budget_with_the_declared_probability() -> None:
    """At 1,200 replicates at most 93 rejections pass; the pass probability clears the target."""
    margins = runner.STUDY.margins
    replicates = runner.BUDGETS["crossfit_null"]
    ceiling = margins.alpha + margins.type_i_margin
    allowed = max(
        k
        for k in range(replicates)
        if clopper_pearson(k, replicates, confidence_level=margins.confidence_level).high <= ceiling
    )
    assert allowed == 93
    target = 0.95 ** (1.0 / UNITS)
    at_null = float(binom.cdf(allowed, replicates, 0.05))
    at_inflated = float(binom.cdf(allowed, replicates, 0.055))
    assert at_null == pytest.approx(0.99998, abs=2e-5)
    assert at_inflated == pytest.approx(0.99953, abs=2e-5)
    assert at_inflated > target
    # At 800 replicates the same rule fails the target at a true rate of 0.055.
    short = max(
        k
        for k in range(800)
        if clopper_pearson(k, 800, confidence_level=margins.confidence_level).high <= ceiling
    )
    assert short == 58
    assert float(binom.cdf(short, 800, 0.055)) < target


def test_the_two_negative_controls_are_declared_and_read_the_right_side() -> None:
    controls = [cell for cell in runner.CELLS if cell.role == "control"]
    assert {cell.name for cell in controls} == set(runner.NEGATIVE_CONTROLS)
    assert {cell.group for cell in controls} == {"crossfit_1500", "repeats"}
    for cell in controls:
        kind = cell.name.rsplit("__", 1)[-1]
        assert SE_CONTROL_SIDES[kind] == runner.NEGATIVE_CONTROLS[cell.name] == "below"
        assert cell.family == "interval_calibration"
        assert cell.recipe == "shipped_curve"
        assert cell.estimand == "ate"
    reading = runner.CONFIGURATION["declared_reading"]
    assert "shipped_curve_control" in reading and "shipped_curve_median_control" in reading
    assert "must be red" in reading
    assert "*must* be red" in (runner.__doc__ or "")
    assert runner.STUDY.publication_policy == "gated"


def test_each_negative_control_leaves_the_band_at_its_declared_size() -> None:
    """The exact-law SE ratio of the shipped curve, and the interval it reads at each budget.

    A control passes by leaving the calibration band.  The 99% interval of an SE ratio has the
    relative half-width ``z / sqrt(2 (R - 1))``, so a control whose exact ratio sits that far
    from the band edge cannot be read inside it.  The cross-fitted curve has the in-sample
    limit (plan 13.3, CW4), so the in-sample quadrature gives the ratio.
    """
    low, _ = runner.STUDY.margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + runner.STUDY.margins.confidence_level / 2.0))
    fit = quad.population_fit(quad.BINARY_ACTIVE, nodes=20)
    shipped = quad.shipped_curves(fit)
    mass = fit.grid.mass
    cov = (shipped * mass[:, None]).T @ shipped
    ratio = np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]) / quad.ate_sd(fit)
    assert ratio == pytest.approx(0.845, abs=0.01)
    for group in ("crossfit_1500", "repeats"):
        replicates = runner.BUDGETS[group]
        assert ratio * (1.0 + z / np.sqrt(2.0 * (replicates - 1))) < low, group


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
    assert dict(runner.STUDY.failure_bound) == {runner.IMPLEMENTATION: 0.01}
    assert properties.PROPERTY_WITNESS is property_verdicts.SAMPLE_WITNESS
    assert len(properties.PROPERTY_RULES) == 1


@pytest.mark.parametrize(
    ("group", "fit"),
    [(group.name, fit) for group in runner.GROUPS.values() for fit in group.fits],
)
def test_each_configuration_is_admitted_by_the_support_predicate(
    group: str, fit: runner.Fit
) -> None:
    spec = runner.GROUPS[group]
    frame = runner.sample(spec.law, 600, 17)
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


def test_the_pseudo_true_cell_has_a_nonzero_gap_to_report() -> None:
    star = runner.truth("pseudo_true", "ate")
    true = runner.truth("binary_active", "ate")
    assert abs(star - true) > 0.05
    assert runner.truth("binary_null", "ate") == pytest.approx(0.0, abs=1e-12)
    for estimand in ("log_rr", "log_or"):
        assert runner.truth("binary_active", estimand) != 0.0
    assert runner.curve_sd()["ate"] == pytest.approx(1.1234, abs=1e-3)


def test_the_noise_control_adds_one_standard_deviation_of_each_groups_own_estimate() -> None:
    bounds = runner.control_bounds()
    sizes = {group.name: group.n for group in runner.GROUPS.values()}
    assert set(bounds) == set(runner.CONTROL_LABELS)
    for label, bound in bounds.items():
        n = sizes[label.removesuffix("__ate")]
        assert bound / np.sqrt(runner.CALIBRATION_N) == pytest.approx(
            runner.curve_sd()["ate"] / np.sqrt(n)
        )


def test_the_repeated_shipped_control_reads_the_reported_rule_on_stored_standard_errors() -> None:
    """Swap the stored shipped SE for the reported SE and the rule returns the reported SE.

    That identity pins the median rule the control uses.  The shipped SE of each draw is below
    its reported SE on ``binary_active``, so the control is the nonzero term the rule must keep.
    """
    group = runner.GROUPS["repeats"]
    frame = runner.sample(group.law, 600, 23)
    result = runner.fit_cleverly(frame, group.fits[0])
    assert len(result.repeats) == 3
    points = np.array([draw.psi["ate"] for draw in result.repeats])
    reported = np.array([draw.inference["ate"].se for draw in result.repeats])
    rule = np.sqrt(np.median(reported**2 + (points - np.median(points)) ** 2))
    assert rule == pytest.approx(result["ate"].plugin_std_error, rel=1e-12)
    shipped = properties.shipped_ate_std_error(result)
    assert 0.0 < shipped < result["ate"].plugin_std_error


def test_a_replicate_reads_every_cell_of_its_group_and_the_shipped_curve_is_narrower() -> None:
    for name in ("crossfit_1500", "repeats"):
        key = (name, 0)
        measured = properties._measure(key, properties._draw(key))
        names = [item["cell"] for item in measured]
        assert names == [cell.name for cell in runner.cells_of(name)]
        se = {item["cell"]: item["std_error"] for item in measured}
        control = f"{name}__ate__shipped_curve__shrunken_se_control"
        assert se[control] < se[f"{name}__ate__corrected"]
        rows = properties._rows(runner.GROUPS[name], 0, measured, runner.BUDGETS[name])
        assert [row["cell"] for row in rows] == names
        assert {row["requested_replicates"] for row in rows} == {runner.BUDGETS[name]}
        assert {row["failed_replicates"] for row in rows} == {0}
        roles = {row["cell"]: row["role"] for row in rows}
        assert roles[control] == "control"
        assert roles[f"{name}__ate__corrected"] == "positive"
    spread = [item for item in measured if "draw_spread" in item]
    assert spread and all(item["draw_spread"] > 0.0 for item in spread)


def test_a_key_names_its_cells_and_the_controls_derived_from_them() -> None:
    names = properties._cells_of(("crossfit_1500", 0, 5))
    positives = [f"{cell.family}/{cell.name}" for cell in runner.cells_of("crossfit_1500")]
    derived = [
        f"interval_calibration/crossfit_1500__ate__{kind}"
        for kind in ("shrunken_se_control", "noise_control")
    ]
    assert names == [*positives, *derived]
    assert {name for controls in properties.DERIVED_CELLS.values() for name in controls} == {
        f"interval_calibration/{label}__{kind}"
        for label in runner.CONTROL_LABELS
        for kind in ("shrunken_se_control", "noise_control")
    }
    assert set(properties.DERIVED_CELLS) <= {f"{cell.family}/{cell.name}" for cell in runner.CELLS}
    assert properties._cells_of(("crossfit_null", 0, 5)) == ["type_i_error/sharp_null"]


def test_the_property_loop_runs_through_fit_properties_and_writes_the_cells_of_its_keys() -> None:
    rows = properties.group_rows(runner.GROUPS["crossfit_null"], n_jobs=1, replicates=2)
    assert set(rows["cell"]) == {"sharp_null"}
    assert rows["replicate"].tolist() == [0, 1]
    assert set(rows["requested_replicates"]) == {2}


def test_the_primary_phase_draws_from_the_published_seed() -> None:
    frame, truths = runner.draw_scenario(runner.SCENARIO, 200, 3)
    again, _ = runner.draw_from_seed(
        runner.SCENARIO, 200, replicate_seed(runner.STUDY, runner.SCENARIO, 3)
    )
    assert frame.equals(again)
    assert set(truths) == set(runner.BINARY_ESTIMANDS)
    assert truths["ate"] == pytest.approx(laws.BINARY_ACTIVE.truth()["ate"])
