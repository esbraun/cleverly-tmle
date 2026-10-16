"""Design of the registered study ``ctmle-oat-per-arm-parametric-crossfit`` (X58b1).

The numbers here are fixed before any run.  Each test names the part of the plan it pins:
the groups, laws and budgets of section 13.6, the unit count ``G = 10`` and its per-unit target,
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
#: The study-wise units of plan 13.6: six calibration groups, one type I group, the primary phase
#: (design review M1) and two negative controls.
UNITS = 10
#: The primary phase runs 1,200 replicates (design review M1).
PRIMARY_REPLICATES = 1_200

#: The declared-size pilot (plans repository ``reviews/x58/x58b/pilot/declared_report.log``,
#: pilot commit ``0cb5fd5b``, pilot-only seed ``PILOT_SEED``).  Each row is the pilot SE ratio, the
#: 99% interval of that ratio and the rejection rate.  The design test pins the figures that plan
#: 13.6 asks it to compute from the pilot.  They are constants, because the pilot rows are not
#: committed.
PILOT_SEED = 9_158_000_058
PILOT_CONTROLS = {
    "crossfit_1500__ate__shipped_curve__shrunken_se_control": (0.8521, 0.8395, 0.8647),
    "repeats__ate__shipped_curve__shrunken_se_control": (0.8526, 0.8376, 0.8675),
}
#: 58 rejections in 1,200 replicates, the 0.0483 of the pilot report.
PILOT_TYPE_I_RATE = 58 / 1_200
#: The pass probability of each unit at its declared budget, from the pilot report.
PILOT_UNITS = {
    "crossfit_1500": 1.00000,
    "crossfit_500": 1.00000,
    "crossfit_8000": 0.99997,
    "crossfit_pseudo_true": 0.99999,
    "repeats": 0.99998,
    "crossfit_generated_design": 0.99991,
    "crossfit_1500__ate__shipped_curve__shrunken_se_control": 1.00000,
    "repeats__ate__shipped_curve__shrunken_se_control": 1.00000,
    "sharp_null": 0.99998,
}
#: The primary phase is not in the declared-size pilot.  Its pass probability comes from 20,000
#: bootstrap resamples of the 15,200 ``crossfit_1500`` pilot replicates (same law, ``n``, folds
#: and estimands), through the primary verdict rules (design review M1).
PILOT_PRIMARY = {800: 0.9915, 1_200: 0.99985}


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
    assert record.replicates == runner.PRIMARY_REPLICATES == PRIMARY_REPLICATES
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


def test_the_study_has_ten_units_and_each_is_sized_to_the_per_unit_target() -> None:
    assert dict(runner.UNITS) == {
        "calibration_groups": 6,
        "type_i_groups": 1,
        "primary_phase": 1,
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
    assert target == pytest.approx(0.99488, abs=1e-5)
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
    docstring = runner.__doc__ or ""
    for name in runner.NEGATIVE_CONTROLS:
        assert name in reading
        assert name in docstring
    assert "shipped_curve_median_control" not in reading + docstring
    assert "must be red" in reading
    assert "*must* be red" in docstring
    # Design review m2: both controls declare an exact-law ratio and the pilot ratio beside it.
    expected = runner.CONFIGURATION["negative_control_expected_se_ratio"]
    assert set(expected) == set(runner.NEGATIVE_CONTROLS)
    for name, values in expected.items():
        assert values["exact"] == 0.845
        assert values["pilot"] == pytest.approx(PILOT_CONTROLS[name][0], abs=1e-3)
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
    assert star == pytest.approx(0.595915 - 0.404085, abs=1e-6)
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


def test_the_coverage_of_psi_0_is_declared_at_the_quadrature_values() -> None:
    """Design review m3: the pseudo-true cells target ``psi*``, so ``psi_0`` is covered rarely.

    The coverage of ``psi_0`` at ``n`` is ``Phi(1.96 - b) - Phi(-1.96 - b)`` with ``b`` the
    standardized gap ``(psi* - psi_0) sqrt(n) / SD``.  The quadrature gives the declared values
    to the tolerance of their four decimals.  The pilot read 0.571, 0.562 and 0.303.
    """
    declared = runner.CONFIGURATION["psi0_coverage_expected"]
    assert declared == {"ey0": 0.5646, "ey1": 0.5646, "ate": 0.2963}
    wrong = quad.population_fit(quad.BINARY_ACTIVE, covariates=(1, 2))
    right = quad.population_fit(quad.BINARY_ACTIVE)
    variance = quad.covariance(wrong)
    sd = {
        "ey0": float(np.sqrt(variance[0, 0])),
        "ey1": float(np.sqrt(variance[1, 1])),
        "ate": quad.ate_sd(wrong),
    }
    gap = {
        "ey0": wrong.psi_star[0] - right.psi_0[0],
        "ey1": wrong.psi_star[1] - right.psi_0[1],
        "ate": (wrong.psi_star[1] - wrong.psi_star[0]) - (right.psi_0[1] - right.psi_0[0]),
    }
    for estimand, value in declared.items():
        b = gap[estimand] * np.sqrt(runner.CALIBRATION_N) / sd[estimand]
        coverage = float(norm.cdf(1.96 - b) - norm.cdf(-1.96 - b))
        assert coverage == pytest.approx(value, abs=1e-4), estimand
    # The pilot reads sit within 0.01 of the prediction.
    for estimand, pilot in {"ey0": 0.571, "ey1": 0.562, "ate": 0.303}.items():
        assert pilot == pytest.approx(declared[estimand], abs=0.01)


def test_the_pilot_figures_that_plan_13_6_asks_the_design_test_to_compute() -> None:
    """Design review m4: the type I, control, primary-phase and study-wise figures from the pilot."""
    target = 0.95 ** (1.0 / UNITS)
    margins = runner.STUDY.margins
    replicates = runner.BUDGETS["crossfit_null"]
    ceiling = margins.alpha + margins.type_i_margin
    allowed = max(
        k
        for k in range(replicates)
        if clopper_pearson(k, replicates, confidence_level=margins.confidence_level).high <= ceiling
    )
    lost = int(runner.FAILURE_BOUND * replicates)
    assert (allowed, lost) == (93, 12)
    # Type I at the pilot rate, and with every lost replicate counted as a rejection.
    at_pilot = float(binom.cdf(allowed, replicates, PILOT_TYPE_I_RATE))
    worst = float(binom.cdf(allowed - lost, replicates, PILOT_TYPE_I_RATE))
    assert at_pilot == pytest.approx(0.99999, abs=1e-5)
    assert worst == pytest.approx(0.99868, abs=1e-5)
    assert min(at_pilot, worst) > target
    # Each shipped-curve control is red at its budget with the probability the pilot gives.
    low, _ = margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + margins.confidence_level / 2.0))
    for name, (ratio, lower, upper) in PILOT_CONTROLS.items():
        sd = (upper - lower) / (2.0 * z)
        red = float(norm.cdf((low - z * sd - ratio) / sd))
        assert red == pytest.approx(1.0, abs=5e-4), name
        assert upper < low, name
    # The primary phase: the pilot bootstrap passes 1,200 replicates and fails 800 replicates.
    assert PILOT_PRIMARY[800] < 0.95 ** (1.0 / UNITS) < PILOT_PRIMARY[1_200]
    assert runner.PRIMARY_REPLICATES == 1_200
    # The study-wise figure with every unit.
    study_wise = float(np.prod(list(PILOT_UNITS.values()))) * PILOT_PRIMARY[1_200]
    assert len(PILOT_UNITS) + 1 == UNITS
    assert min(PILOT_UNITS.values()) > target
    assert study_wise >= 0.95
    assert study_wise == pytest.approx(0.9997, abs=2e-4)


def test_the_primary_phase_has_the_per_estimand_headroom_at_1200_and_not_at_800() -> None:
    """Design review M1: the covered count the 99% lower bound needs, and the pass probability.

    One estimand at true coverage 0.95 passes with probability 0.9977 at 800 replicates.  The
    five estimands of the primary phase share one pass event, and the pilot bootstrap gives the
    joint figure (``PILOT_PRIMARY``).
    """
    margins = runner.STUDY.margins
    need = {
        replicates: min(
            k
            for k in range(replicates + 1)
            if clopper_pearson(k, replicates, confidence_level=margins.confidence_level).low
            >= margins.coverage_floor
        )
        for replicates in (800, 1_200)
    }
    assert need[800] == 742
    one = {r: float(binom.sf(k - 1, r, 0.95)) for r, k in need.items()}
    assert one[800] == pytest.approx(0.9977, abs=2e-4)
    assert one[1_200] > one[800]
    # Five estimands that do not move together would pass with one[800] ** 5 = 0.989.  The
    # estimands share the same fits, and the pilot bootstrap reads 0.9915 at 800 replicates.
    assert one[800] ** 5 == pytest.approx(0.9886, abs=1e-3)
    assert one[800] ** 5 < PILOT_PRIMARY[800] < 0.95 ** (1.0 / UNITS)


def test_the_exact_law_shipped_ratios_pin_what_the_cells_see() -> None:
    """Design review m4: the shipped curve is narrower on every law the cells draw from.

    The sharp-null law gives 0.833 for the ATE, so the type I cell sees the stack term: the
    shipped curve would reject at about 0.10.  The pseudo-true fit gives 0.855 for the arm means
    and 0.861 for the ATE.
    """

    def shipped_ratio(fit: quad.PopulationFit) -> tuple[float, float]:
        shipped = quad.shipped_curves(fit)
        mass = fit.grid.mass
        cov = (shipped * mass[:, None]).T @ shipped
        arm = float(np.sqrt(cov[0, 0] / quad.covariance(fit)[0, 0]))
        ate = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]) / quad.ate_sd(fit))
        return arm, ate

    null = quad.QuadratureLaw(
        "binary_null",
        lambda w: np.asarray(laws.BINARY_NULL.mechanism(w[:, 0], w[:, 2])[:, 1]),
        lambda a, w: laws.BINARY_NULL.outcome_probability(a, w[:, 0], w[:, 1]),
    )
    _, null_ate = shipped_ratio(quad.population_fit(null))
    assert null_ate == pytest.approx(0.833, abs=5e-4)
    # A shipped interval with the SD ratio r < 1 rejects at 2 (1 - Phi(1.96 r)).
    shipped_rejection = 2.0 * float(norm.sf(1.96 * null_ate))
    assert shipped_rejection == pytest.approx(0.10, abs=0.005)
    assert shipped_rejection > 0.05 + runner.STUDY.margins.type_i_margin - 0.005
    arm, ate = shipped_ratio(quad.population_fit(quad.BINARY_ACTIVE, covariates=(1, 2)))
    assert arm == pytest.approx(0.855, abs=5e-4)
    assert ate == pytest.approx(0.861, abs=5e-4)
    arm_active, ate_active = shipped_ratio(quad.population_fit(quad.BINARY_ACTIVE))
    assert arm_active == pytest.approx(0.845, abs=5e-4)
    assert ate_active == pytest.approx(0.845, abs=5e-4)
