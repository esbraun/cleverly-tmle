"""Design of the registered study ``ctmle-oat-per-arm-parametric-extensions`` (X58b2).

The numbers here are fixed before any run.  Each test names the part of the plan it pins: the
groups, laws and budgets of section 13.6, the unit count ``G = 19`` and its per-unit target, the
type I size, the fresh seeds, the failure bound, the nine negative controls with their exact-law
separation, the reference arm of each group, and the admission of every configuration by the
support predicate.  No test fits more than a few replicates, and none runs a study.  The
declared-size pilot has not run, so the pilot-derived figures that the X58b1 test pins
(``tests/unit/test_ctmle_oat_per_arm_parametric_crossfit_study_design.py``) are absent here.  The
exact-law figures come from the quadrature of :mod:`tests.studies.oat_multi_arm_quadrature`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from scipy.stats import binom, norm

from cleverly.estimators.ctmle import per_arm_parametric_support
from tests.studies import canonical_ctmle_oat_per_arm_parametric_extensions as runner
from tests.studies import ctmle_oat_per_arm_parametric_extensions_properties as properties
from tests.studies import oat_multi_arm_quadrature as mq
from tests.studies import oat_per_arm_laws as laws
from tests.studies.evidence import descriptions, property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.property_verdicts import SE_CONTROL_SIDES
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: Plan 13.6 group -> (replicates, n).  The replicates are the rev 3.1 expected budgets.
PLAN = {
    "three_arm": (29_200, 1_500),
    "three_arm_shared": (14_000, 1_500),
    "binary_population": (9_600, 1_500),
    "three_arm_crossfit": (17_600, 1_500),
    "three_arm_generated_design": (11_600, 1_500),
    "weighted": (17_600, 1_500),
    "weighted_crossfit": (11_600, 1_500),
    "composition": (17_600, 1_500),
    "three_arm_null": (1_600, 1_000),
}
#: The law, cross-fitting and reference arm of each group (plan 13.6, m11).
PLAN_FITS = {
    "three_arm": ("three_arm_active", False, "1"),
    "three_arm_shared": ("three_arm_shared", False, "1"),
    "binary_population": ("binary_active", False, None),
    "three_arm_crossfit": ("three_arm_active", True, "0"),
    "three_arm_generated_design": ("three_arm_active", True, "0"),
    "weighted": ("binary_active_weighted", False, None),
    "weighted_crossfit": ("binary_active_weighted", True, None),
    "composition": ("three_arm_active_weighted", True, "1"),
    "three_arm_null": ("three_arm_null", False, "1"),
}
#: The families the harness classifies and this study uses.  The study has no power cell.
FAMILIES = {"interval_calibration", "generated_design", "type_i_error", "simultaneous_coverage"}
#: The positive calibration rows by group (plan 13.6, "gated rows").
GATED_ROWS = {
    "three_arm": 12,
    "three_arm_shared": 5,
    "binary_population": 3,
    "three_arm_crossfit": 5,
    "weighted": 8,
    "weighted_crossfit": 3,
    "composition": 5,
}
#: The study-wise units of plan 13.6: eight calibration groups, one type I group, the primary
#: phase and nine negative controls.
UNITS = 19
TARGET = 0.95 ** (1.0 / UNITS)
#: The primary phase runs 1,600 replicates per law, 3,200 in all (plan 13.6, "Sizing").
PRIMARY_REPLICATES = 1_600


def _study() -> StudyRecord:
    (record,) = (
        study for study in registered() if study.slug == "ctmle-oat-per-arm-parametric-extensions"
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
    assert set(record.property_cells) == FAMILIES
    assert (
        record.artifacts == ROOT / "tests" / "canonical" / "ctmle_oat_per_arm_parametric_extensions"
    )
    assert (record.artifacts / "__init__.py").is_file()
    assert (record.artifacts / "regenerate.py").is_file()
    assert record.runner() is runner
    assert record.properties() is properties
    assert record.property_cells == runner.property_cells()


def test_the_primary_phase_re_measures_the_old_rows_that_x58a_did_not_take() -> None:
    """Plan 13.6: every estimand of the old three-arm primary and ``par``, ``paf`` of the binary."""
    assert dict(runner.STUDY.scenarios) == {
        "binary_active": ("par", "paf"),
        "three_arm_active": (
            "ey[0]",
            "ey[1]",
            "ey[2]",
            "ate[1 vs 0]",
            "ate[2 vs 0]",
            "rr[1 vs 0]",
            "rr[2 vs 0]",
            "or[1 vs 0]",
            "or[2 vs 0]",
        ),
    }
    old = pytest.importorskip("tests.studies.canonical_ctmle_oat_per_arm")
    assert runner.THREE_ARM_PRIMARY == old.THREE_ARM_ESTIMANDS
    assert set(runner.BINARY_PRIMARY) <= set(old.BINARY_ESTIMANDS)
    # The three-arm primary reads the old cells' reference arm.
    assert runner.PRIMARY_FITS["three_arm_active"].reference == "0"
    frame, truths = runner.draw_scenario("three_arm_active", 200, 3)
    again, _ = runner.draw_from_seed(
        "three_arm_active", 200, replicate_seed(runner.STUDY, "three_arm_active", 3)
    )
    assert frame.equals(again)
    assert set(truths) == set(runner.THREE_ARM_PRIMARY)
    exact = laws.THREE_ARM_ACTIVE.truth()
    assert all(truths[name] == pytest.approx(exact[name]) for name in truths)
    binary = runner.draw_scenario("binary_active", 200, 3)[1]
    assert binary == pytest.approx({"par": 0.0592956, "paf": binary["paf"]}, rel=1e-5)


def test_every_group_exists_with_its_budget_size_law_and_reference() -> None:
    assert set(runner.PLAN_CELLS) == set(PLAN) == set(runner.BUDGETS) == set(runner.GROUPS)
    for name, (replicates, n) in PLAN.items():
        group = runner.GROUPS[runner.PLAN_CELLS[name]]
        assert (group.replicates, group.n) == (replicates, n), name
        assert replicates % 400 == 0
        assert [cell for cell in runner.CELLS if cell.plan == name], name
        law, cross_fit, reference = PLAN_FITS[name]
        assert group.law == law, name
        for fit in group.fits:
            assert fit.cross_fit is cross_fit, (name, fit.name)
            assert fit.reference == reference, (name, fit.name)
    assert {cell.plan for cell in runner.CELLS} == set(PLAN)
    # Plan 13.6 with the type I group at 1,600 (design review m1): the nine groups sum to
    # 130,400 replicates and the primary phase adds 3,200.
    assert sum(runner.BUDGETS.values()) == 130_400
    assert sum(runner.BUDGETS.values()) + 2 * runner.PRIMARY_REPLICATES == 133_600


def test_every_cross_fitted_group_uses_five_folds_and_one_draw() -> None:
    for group in runner.GROUPS.values():
        for fit in group.fits:
            subject = runner.estimator(fit)
            assert subject.cross_fit is fit.cross_fit
            assert subject.n_folds == runner.FOLDS == 5
            assert subject.repeats == 1
            assert subject.reference == fit.reference
            assert subject.outcome_learner is not None
            if fit.outcome != "oracle":
                assert subject.outcome_learner.arm_columns == fit.k - 1


def test_the_weighted_groups_and_the_laws_carry_the_declared_weight() -> None:
    weighted_groups = {"weighted", "weighted_crossfit", "composition"}
    for group in runner.GROUPS.values():
        assert all(fit.weighted == (group.name in weighted_groups) for fit in group.fits), (
            group.name
        )
        frame = runner.sample(group.law, 50, 1)
        assert ("wt" in frame) == (group.name in weighted_groups), group.name
    frame = runner.sample("binary_active_weighted", 50, 1)
    assert set(frame["wt"]) <= {0.5, 1.5}
    tilted = runner.sample("three_arm_active_weighted", 50, 1)
    assert np.array_equal(tilted["wt"].to_numpy(), laws.weight(tilted["W1"].to_numpy()))


def test_the_new_laws_have_the_declared_parameters() -> None:
    shared = laws.THREE_ARM_SHARED
    assert shared.c == laws.THREE_ARM_ACTIVE.c == (-0.4, 0.0, 0.4)
    assert shared.u == ((3.5, 0.8),) * 3
    assert shared.v == laws.THREE_ARM_ACTIVE.v
    null = laws.THREE_ARM_NULL
    assert null.c == (-0.3,) * 3 and null.u == ((0.8, -0.8),) * 3
    assert null.v == laws.THREE_ARM_ACTIVE.v
    weighted = laws.THREE_ARM_ACTIVE_WEIGHTED
    assert weighted.weighted and (weighted.c, weighted.u, weighted.v) == (
        laws.THREE_ARM_ACTIVE.c,
        laws.THREE_ARM_ACTIVE.u,
        laws.THREE_ARM_ACTIVE.v,
    )
    # The quadrature module reads the same shared law, so the SDs of this study are its.
    assert mq.THREE_ARM_SHARED.name == "three_arm_shared"
    assert set(laws.LAWS) >= {
        "three_arm_shared",
        "three_arm_null",
        "three_arm_active_weighted",
    }
    # Plan 13.2: psi of the shared law.
    assert laws.THREE_ARM_SHARED.arm_means() == pytest.approx([0.4602, 0.5, 0.5398], abs=5e-5)
    # The sharp null has equal arm means, so every contrast is zero to quadrature precision.
    means = laws.THREE_ARM_NULL.arm_means()
    assert np.ptp(means) < 1e-12
    for contrast in ("ate[0 vs 1]", "ate[2 vs 1]"):
        assert runner.truth("three_arm_null", contrast) == pytest.approx(0.0, abs=1e-12)


def test_the_cells_the_summary_must_contain_are_the_declared_ones() -> None:
    declared = {name for names in _study().property_cells.values() for name in names}
    assert declared == {cell.name for cell in runner.CELLS} | set(runner.DERIVED_CONTROLS)
    assert len(declared) == len(runner.CELLS) + len(runner.DERIVED_CONTROLS)
    positive = [
        cell
        for cell in runner.CELLS
        if cell.family == "interval_calibration" and cell.role == "positive"
    ]
    assert len(positive) == sum(GATED_ROWS.values()) == 41
    assert all(cell.name.endswith("__corrected") for cell in positive)
    for group, rows in GATED_ROWS.items():
        assert len([cell for cell in positive if cell.group == group]) == rows, group
    assert {cell.family for cell in runner.CELLS} == FAMILIES
    # Plan 13.6: the gated rows of the in-sample three-arm group.
    assert {cell.estimand for cell in positive if cell.group == "three_arm"} == {
        "ey[0]",
        "ey[1]",
        "ey[2]",
        "ate[0 vs 1]",
        "ate[2 vs 1]",
        "log_rr[0 vs 1]",
        "log_rr[2 vs 1]",
        "log_or[0 vs 1]",
        "log_or[2 vs 1]",
        "par[1]",
        "paf[1]",
        "ey_obs",
    }
    assert {cell.estimand for cell in positive if cell.group == "binary_population"} == {
        "par",
        "paf",
        "ey_obs",
    }
    assert {cell.estimand for cell in positive if cell.group == "weighted"} == {
        "ey0",
        "ey1",
        "ate",
        "log_rr",
        "log_or",
        "par",
        "paf",
        "ey_obs",
    }
    # The generated-design pair reads both contrasts, two arms each.
    pair = [cell for cell in runner.CELLS if cell.family == "generated_design"]
    assert {cell.estimand for cell in pair} == {"ate[1 vs 0]", "ate[2 vs 0]"}
    assert len(pair) == 4
    # The type I group reads the two contrasts against the reference arm 1.
    null = [cell for cell in runner.CELLS if cell.family == "type_i_error"]
    assert {cell.estimand for cell in null} == {"ate[0 vs 1]", "ate[2 vs 1]"}
    # Every group but the pair and the null has its contrast row's two derived controls.
    assert {label.rsplit("__", 1)[0] for label in runner.CONTROL_LABELS} == set(GATED_ROWS)
    for label in runner.CONTROL_LABELS:
        assert f"{label}__corrected" in {cell.name for cell in runner.CELLS}


def test_ey_obs_is_gated_in_the_three_groups_that_fit_it() -> None:
    """Design review m2: each support row that licenses ``ey_obs`` (family P4) has a gated cell."""
    cells = [cell for cell in runner.CELLS if cell.estimand == "ey_obs"]
    assert {cell.name for cell in cells} == {
        "three_arm__ey_obs__corrected",
        "binary_population__ey_obs__corrected",
        "weighted__ey_obs__corrected",
    }
    assert all(cell.role == "positive" and not cell.report for cell in cells)
    for cell in cells:
        fit = next(f for f in runner.GROUPS[cell.group].fits if f.name == cell.fit)
        assert "ey_obs" in fit.estimands
    # No group that fits ey_obs leaves it ungated, and no extra column records it.
    fitted = {
        group.name
        for group in runner.GROUPS.values()
        if any("ey_obs" in fit.estimands for fit in group.fits if fit.outcome == "parametric")
    }
    assert fitted == {cell.group for cell in cells}
    columns = (*properties.EXTRA_COLUMNS, *properties.SUMMARY_COLUMNS)
    assert not [name for name in columns if "obs" in name]
    # The support rows that license P4 under the Wald kind list exactly these cells.
    from cleverly._support_rows import POINT_ROWS

    listed = {
        cell
        for row in POINT_ROWS
        if row.id in ("C11.binary", "C11.k_arms", "C11.weighted")
        for cell in row.cells
        if cell.endswith("__ey_obs__corrected")
    }
    assert listed == {cell.name for cell in cells}


def test_the_study_has_nineteen_units_and_each_is_sized_to_the_per_unit_target() -> None:
    assert dict(runner.UNITS) == {
        "calibration_groups": 8,
        "type_i_groups": 1,
        "primary_phase": 1,
        "negative_controls": 9,
    }
    assert runner.UNIT_COUNT == UNITS
    calibration = {
        cell.group
        for cell in runner.CELLS
        if cell.family in ("interval_calibration", "generated_design")
    }
    assert len(calibration) == runner.UNITS["calibration_groups"]
    assert len(runner.NEGATIVE_CONTROLS) == runner.UNITS["negative_controls"]
    assert pytest.approx(0.99730, abs=1e-5) == TARGET
    assert 0.95 ** (1.0 / 10) < TARGET < 0.95 ** (1.0 / 28) < 1.0


def test_the_type_i_group_passes_at_its_budget_with_the_declared_probability() -> None:
    """At 1,600 replicates at most 129 rejections pass, and the target holds at every lost count."""
    margins = runner.STUDY.margins
    replicates = runner.BUDGETS["three_arm_null"]
    ceiling = margins.alpha + margins.type_i_margin

    def allowed(size: int) -> int:
        return max(
            k
            for k in range(size)
            if clopper_pearson(k, size, confidence_level=margins.confidence_level).high <= ceiling
        )

    def passes(size: int, lost: int, rate: float) -> float:
        # The kept replicates are ``size - lost``; each lost replicate counts as a rejection, so
        # at most ``allowed(size) - lost`` of the kept ones may reject.  The two rows share one
        # fit and so lose together; the product is the independent (lower) bound.
        return float(binom.cdf(allowed(size) - lost, size - lost, rate)) ** 2

    assert allowed(replicates) == 129
    lost = int(runner.FAILURE_BOUND * replicates)
    assert lost == 16
    # Clean and with every allowed lost replicate counted as a rejection, at a true rate of
    # 0.05 and at 0.055.
    assert passes(replicates, 0, 0.05) > 0.99999
    assert passes(replicates, 0, 0.055) == pytest.approx(0.99998, abs=2e-5)
    assert passes(replicates, lost, 0.05) == pytest.approx(0.99982, abs=2e-5)
    assert passes(replicates, lost, 0.055) == pytest.approx(0.99490, abs=2e-5)
    # The target holds at every allowed loss count at 0.05, and at 13 of 16 at 0.055.
    assert all(passes(replicates, k, 0.05) > TARGET for k in range(lost + 1))
    assert max(k for k in range(lost + 1) if passes(replicates, k, 0.055) > TARGET) == 13
    # At 1,200 replicates the worst case at the bound (12) misses the target: 0.99506 at 0.05
    # holds only up to ten lost replicates, and 0.95525 at 0.055 up to two.  Hence 1,600.
    assert allowed(1_200) == 93
    assert passes(1_200, 0, 0.05) == pytest.approx(0.99996, abs=2e-5)
    assert passes(1_200, 12, 0.05) == pytest.approx(0.99506, abs=2e-5)
    assert passes(1_200, 12, 0.055) == pytest.approx(0.95525, abs=2e-5)
    assert passes(1_200, 12, 0.05) < TARGET
    # At 800 replicates the same two rows give 0.9955 at 0.05, below the target.
    assert allowed(800) == 58
    assert passes(800, 0, 0.05) == pytest.approx(0.9955, abs=1e-4)
    assert passes(800, 0, 0.05) < TARGET


def test_the_primary_phase_has_the_per_estimand_headroom() -> None:
    """The lower 99% bound of coverage must reach 0.90: 1,107 of 1,200, 1,471 of 1,600."""
    margins = runner.STUDY.margins
    need = {
        size: min(
            k
            for k in range(size + 1)
            if clopper_pearson(k, size, confidence_level=margins.confidence_level).low
            >= margins.coverage_floor
        )
        for size in (800, 1_200, 1_600)
    }
    assert need == {800: 742, 1_200: 1_107, 1_600: 1_471}
    rows = len(runner.BINARY_PRIMARY) + len(runner.THREE_ARM_PRIMARY)
    assert rows == 11
    one = {size: float(binom.sf(k - 1, size, 0.95)) for size, k in need.items()}
    # Eleven rows, taken as independent, pass with these probabilities at a true coverage 0.95.
    assert one[1_600] ** rows > TARGET
    assert one[800] ** rows < TARGET
    # Design review m6: 1,200 is the smallest multiple of 400 at the target at a coverage of 0.95.
    # 1,600 is kept: it holds the target down to a coverage of 0.945 on every row and gives
    # 0.99592 at 0.94 (the review wrote that it holds the target at 0.94; the figure is below it).
    assert one[1_200] ** rows == pytest.approx(0.99981, abs=2e-5)
    assert one[1_200] ** rows > TARGET
    at = {
        (size, coverage): float(binom.sf(k - 1, size, coverage)) ** rows
        for size, k in need.items()
        for coverage in (0.945, 0.94)
    }
    assert at[1_200, 0.94] == pytest.approx(0.93764, abs=2e-5)
    assert at[1_200, 0.94] < TARGET
    assert at[1_600, 0.945] == pytest.approx(0.99990, abs=2e-5)
    assert at[1_200, 0.945] == pytest.approx(0.99478, abs=2e-5)
    assert at[1_200, 0.945] < TARGET < at[1_600, 0.945]
    assert at[1_600, 0.94] == pytest.approx(0.99592, abs=2e-5)
    assert at[1_600, 0.94] < TARGET
    assert runner.PRIMARY_REPLICATES == 1_600
    assert "0.945" in runner.CONFIGURATION["budget_rule"]


# ----------------------------------------------------------------------- the negative controls


def _controls() -> list[runner.Cell]:
    return [cell for cell in runner.CELLS if cell.role == "control"]


def test_the_nine_negative_controls_are_declared_and_read_the_right_side() -> None:
    controls = _controls()
    assert {cell.name for cell in controls} == set(runner.NEGATIVE_CONTROLS)
    by_group: dict[str, int] = {}
    for cell in controls:
        by_group[cell.group] = by_group.get(cell.group, 0) + 1
    # Plan 13.6: three_arm 3, three_arm_shared 1, binary_population 3, three_arm_crossfit 1,
    # weighted 1.
    assert by_group == {
        "three_arm": 3,
        "three_arm_shared": 1,
        "binary_population": 3,
        "three_arm_crossfit": 1,
        "weighted": 1,
    }
    assert sum(by_group.values()) == runner.UNITS["negative_controls"] == 9
    for cell in controls:
        kind = cell.name.rsplit("__", 1)[-1]
        if cell.family == "interval_calibration":
            assert SE_CONTROL_SIDES[kind] == runner.NEGATIVE_CONTROLS[cell.name], cell.name
        else:
            assert cell.family == "simultaneous_coverage"
            assert kind in ("pointwise_joint_control", "shipped_curve_band_control")
            assert runner.NEGATIVE_CONTROLS[cell.name] == "below"
    shipped = {c.group: c for c in controls if c.recipe == "shipped_curve"}
    assert {g: c.estimand for g, c in shipped.items()} == {
        "three_arm": "ey[1]",
        "binary_population": "par",
        "three_arm_crossfit": "ey[1]",
        "weighted": "ey0",
    }
    (dropped,) = (c for c in controls if c.recipe == "cross_term_dropped")
    assert (dropped.group, dropped.estimand) == ("three_arm_shared", "ate[0 vs 1]")
    # The band groups: the in-sample groups declare a pointwise and a shipped-band control, the
    # cross-fitted three-arm group declares the band alone (it has one control, plan 13.6).
    bands = {band.label: band for group in runner.GROUPS.values() for band in group.bands}
    assert {label: (b.pointwise, b.shipped is not None) for label, b in bands.items()} == {
        "three_arm": (True, True),
        "binary_population": (True, True),
        "three_arm_crossfit": (False, False),
    }
    reading = runner.CONFIGURATION["declared_reading"]
    docstring = runner.__doc__ or ""
    assert "must be red" in reading
    assert "*must* be red" in docstring
    assert set(runner.CONFIGURATION["negative_control_expected"]) == set(runner.NEGATIVE_CONTROLS)
    assert runner.STUDY.publication_policy == "gated"


@cache
def _ratios() -> dict[str, float]:
    """The exact-law SD of the shipped curve over the stacked curve, by control."""

    def ratio(law: mq.ArmLaw, nodes: tuple[np.ndarray, np.ndarray], name: str, ref: int) -> float:
        fit = mq.population_fit(law, *nodes)
        stacked = mq.contrast_curves(fit, mq.arm_curves(fit), ref)[name]
        shipped = mq.contrast_curves(fit, mq.shipped_curves(fit), ref)[name]
        return mq.sd(fit, shipped) / mq.sd(fit, stacked)

    grid = mq.hermite_nodes(20)
    return {
        "three_arm__ey_1__shipped_curve__shrunken_se_control": ratio(
            mq.THREE_ARM_ACTIVE, grid, "ey[1]", 1
        ),
        "three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control": ratio(
            mq.THREE_ARM_ACTIVE, grid, "ey[1]", 0
        ),
        "binary_population__par__shipped_curve__shrunken_se_control": ratio(
            mq.BINARY_ACTIVE, grid, "par[0]", 0
        ),
        "weighted__ey0__shipped_curve__shrunken_se_control": ratio(
            mq.BINARY_ACTIVE_WEIGHTED, mq.half_line_nodes(30, 12), "ey[0]", 0
        ),
    }


@cache
def _cross_term_ratio() -> float:
    fit = mq.population_fit(mq.THREE_ARM_SHARED, *mq.hermite_nodes(20))
    curves = mq.contrast_curves(fit, mq.arm_curves(fit), 1)
    cov = mq.covariance(
        fit, np.column_stack([curves["ey[0]"], curves["ey[1]"], curves["ate[0 vs 1]"]])
    )
    return float(np.sqrt(cov[0, 0] + cov[1, 1]) / np.sqrt(cov[2, 2]))


def test_each_se_control_leaves_the_band_at_its_declared_size() -> None:
    """The exact-law ratio of each SE control, and the interval it reads at its budget.

    A control passes by leaving the calibration band.  The 99% interval of an SE ratio has the
    relative half-width ``z / sqrt(2 (R - 1))``, so a control whose exact ratio sits that far
    from the band edge cannot be read inside it.  Plan 13.6 gives the ratios 0.780, 0.766,
    0.819, 0.780 and 1.163.
    """
    low, high = runner.STUDY.margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + runner.STUDY.margins.confidence_level / 2.0))
    expected = {
        "three_arm__ey_1__shipped_curve__shrunken_se_control": 0.780,
        "three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control": 0.780,
        "binary_population__par__shipped_curve__shrunken_se_control": 0.766,
        "weighted__ey0__shipped_curve__shrunken_se_control": 0.819,
    }
    for name, ratio in _ratios().items():
        assert ratio == pytest.approx(expected[name], abs=1.5e-3), name
        assert ratio < 0.93, name
        group = next(cell.group for cell in runner.CELLS if cell.name == name)
        replicates = runner.BUDGETS[group]
        assert ratio * (1.0 + z / np.sqrt(2.0 * (replicates - 1))) < low, name
    shared = _cross_term_ratio()
    assert shared == pytest.approx(1.163, abs=1e-3)
    assert shared > 1.07
    replicates = runner.BUDGETS["three_arm_shared"]
    assert shared * (1.0 - z / np.sqrt(2.0 * (replicates - 1))) > high
    # The expected values the configuration declares are the exact-law ones.
    declared = runner.CONFIGURATION["negative_control_expected"]
    for name, value in expected.items():
        assert declared[name]["se_ratio"] == value
    assert declared["three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control"][
        "se_ratio"
    ] == pytest.approx(shared, abs=1e-3)


def _root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return vectors * np.sqrt(np.clip(values, 0.0, None))


def exact_joint(
    law: mq.ArmLaw, names: tuple[str, ...], reference: int, *, draws: int = 1_000_000
) -> dict[str, float]:
    """The joint coverage of a band, a shipped band and the pointwise intervals, by quadrature.

    A fit of the stacked curves has covariance ``C``.  The band is ``c sd_j``, with ``c`` the 95%
    point of ``max |Z_j| / sd_j`` for ``Z ~ N(0, C)``.  The shipped band uses the covariance
    ``C_s`` of the shipped curves for ``c`` and ``sd_j``, and covers ``Z``.  The estimates with a
    singular covariance (an ``ate`` is a difference of two arm means) take the eigen root.
    """
    fit = mq.population_fit(law, *mq.hermite_nodes(20))

    def curves(arms: np.ndarray) -> np.ndarray:
        named = mq.contrast_curves(fit, arms, reference)
        named["ey_obs"] = mq.observed_curve(fit)
        return np.column_stack([named[name] for name in names])

    stacked = mq.covariance(fit, curves(mq.arm_curves(fit)))
    shipped = mq.covariance(fit, curves(mq.shipped_curves(fit)))
    g = np.random.default_rng(7).standard_normal((draws, len(names)))

    def critical(cov: np.ndarray) -> float:
        sd = np.sqrt(np.diag(cov))
        statistic = np.max(np.abs(g @ _root(cov / np.outer(sd, sd)).T), axis=1)
        return float(np.quantile(statistic, 0.95))

    z = g @ _root(stacked).T
    sd_stacked, sd_shipped = np.sqrt(np.diag(stacked)), np.sqrt(np.diag(shipped))

    def covered(half: np.ndarray) -> float:
        return float(np.mean(np.all(np.abs(z) <= half, axis=1)))

    return {
        "stacked": covered(critical(stacked) * sd_stacked),
        "shipped": covered(critical(shipped) * sd_shipped),
        "pointwise": covered(1.96 * sd_stacked),
    }


@cache
def _joint(label: str) -> dict[str, float]:
    if label == "three_arm":
        return exact_joint(mq.THREE_ARM_ACTIVE, runner.BAND_REF1, 1)
    if label == "three_arm_crossfit":
        return exact_joint(mq.THREE_ARM_ACTIVE, runner.BAND_REF0, 0)
    names = ("ey[0]", "ey[1]", "ate[1 vs 0]", "ey_obs", "par[0]", "paf[0]")
    return exact_joint(mq.BINARY_ACTIVE, names, 0)


def test_the_joint_controls_are_red_and_the_band_is_calibrated_on_the_exact_law() -> None:
    """Plan 13.2 and 13.6: 0.826 pointwise, 0.861 and 0.867 shipped, 0.9505 and 0.9497 stacked.

    The binary population band set (``ey0``, ``ey1``, ``ate``, ``ey_obs``, ``par``, ``paf``) has
    no figure in the plan.  The quadrature gives 0.841 pointwise and 0.872 for the shipped band,
    and CONFIGURATION declares both.  Every control is far enough below 0.95 that the upper 99%
    bound at its budget stays below it.
    """
    z = float(norm.ppf(0.995))
    expected = {
        "three_arm": {"pointwise": 0.826, "shipped": 0.861},
        "three_arm_crossfit": {"pointwise": 0.826, "shipped": 0.867},
        "binary_population": {"pointwise": 0.841, "shipped": 0.872},
    }
    for label, figures in expected.items():
        joint = _joint(label)
        assert joint["stacked"] == pytest.approx(0.95, abs=2e-3), label
        for kind, value in figures.items():
            assert joint[kind] == pytest.approx(value, abs=2e-3), (label, kind)
            group = label
            replicates = runner.BUDGETS[group]
            bound = value + z * np.sqrt(value * (1.0 - value) / replicates)
            assert bound < 0.95, (label, kind)
    # CONFIGURATION declares the exact values, as it does for the SE controls.
    declared = runner.CONFIGURATION["negative_control_expected"]
    for label in ("three_arm", "binary_population"):
        figures = expected[label]
        pointwise = declared[f"{label}__pointwise_joint_control"]
        shipped = declared[f"{label}__shipped_curve_band_control"]
        assert pointwise["joint_coverage"] == figures["pointwise"]
        assert shipped["joint_coverage"] == figures["shipped"]
    assert "pilot" not in str(declared)
    # The stacked band covers 0.9505, 0.9497 and 0.950 (design review m5).
    assert _joint("three_arm")["stacked"] == pytest.approx(0.9505, abs=1e-3)
    assert _joint("three_arm_crossfit")["stacked"] == pytest.approx(0.9497, abs=1e-3)
    assert _joint("binary_population")["stacked"] == pytest.approx(0.950, abs=1e-3)
    # The band set of each band is the declared one, and each holds only stacked estimates.
    sets = {band.label: band.names for g in runner.GROUPS.values() for band in g.bands}
    assert sets["three_arm"] == ("ey[0]", "ey[1]", "ey[2]", "ate[0 vs 1]", "ate[2 vs 1]")
    assert sets["three_arm_crossfit"] == ("ey[0]", "ey[1]", "ey[2]", "ate[1 vs 0]", "ate[2 vs 0]")
    assert sets["binary_population"] == ("ey0", "ey1", "ate", "ey_obs", "par", "paf")


@pytest.mark.parametrize(
    ("group", "band_fit", "shipped_fit"),
    [
        ("three_arm", "band", "shipped_band"),
        ("binary_population", "main", "shipped_band"),
    ],
)
def test_the_shipped_band_control_reads_the_stacked_estimates_with_the_shipped_se(
    group: str, band_fit: str, shipped_fit: str
) -> None:
    """Design review n1: the refused ``ShippedOutcome`` fit differs from the stack in the curve only.

    The control relabels the refused fit ``influence_curve`` to build the shipped band.  If a
    later predicate admitted the subclass, the control would turn green and void the study at run
    time.  This test fails first.  Every name of the declared band set must carry the stacked
    ``psi`` exactly, the stored shipped standard error to ``1e-12``, and the plug-in status.
    """
    key = (group, 0, 5)
    _, results = properties._measure(key, properties._draw(key))
    (band,) = runner.GROUPS[group].bands
    stacked, shipped = results[band_fit], results[shipped_fit]
    for name in band.names:
        assert shipped[name].psi == stacked[name].psi, name
        assert shipped[name].plugin_std_error == pytest.approx(
            stacked.repeats[0].inference[name].shipped_se, rel=1e-12
        ), name
        assert stacked[name].inference == "influence_curve", name
        assert shipped[name].inference == "generated_design_plugin", name


def test_the_band_cells_pass_at_their_budget_with_the_declared_probability() -> None:
    """The joint-coverage interval of the band must lie in the calibration coverage band."""
    margins = runner.STUDY.margins
    low, high = margins.calibration_coverage
    for label in ("three_arm", "binary_population", "three_arm_crossfit"):
        replicates = runner.BUDGETS[label]
        p = _joint(label)["stacked"]
        inside = [
            k
            for k in range(int(replicates * (low - 0.01)), int(replicates * (high + 0.01)))
            if clopper_pearson(k, replicates, confidence_level=margins.confidence_level).within(
                low, high
            )
        ]
        passes = float(
            binom.cdf(max(inside), replicates, p) - binom.cdf(min(inside) - 1, replicates, p)
        )
        assert passes > TARGET, label


# ------------------------------------------------------------------------------ the quadrature


def test_the_exact_law_sds_pin_the_plan_tables() -> None:
    """Plan 13.2: the three-arm stacked SDs, and plan 13.6 the weighted and shared SDs."""
    three = runner.exact_sds("three_arm_active", 0)
    assert three["ey[0]"] == pytest.approx(0.9404, abs=2e-4)
    assert three["ey[1]"] == pytest.approx(1.0176, abs=2e-4)
    assert three["ey[2]"] == pytest.approx(1.0332, abs=2e-4)
    assert three["ate[1 vs 0]"] == pytest.approx(1.3855, abs=2e-4)
    assert three["ate[2 vs 0]"] == pytest.approx(1.4033, abs=2e-4)
    assert three["par[0]"] == pytest.approx(0.8455, abs=2e-4)
    assert three["paf[0]"] == pytest.approx(1.5489, abs=2e-4)
    assert runner.exact_sds("three_arm_active", 1)["par[1]"] == pytest.approx(0.9306, abs=2e-4)
    binary = runner.exact_sds("binary_active")
    assert binary["par"] == pytest.approx(0.6605, abs=2e-4)
    assert binary["paf"] == pytest.approx(1.3164, abs=2e-4)
    assert binary["ate"] == pytest.approx(1.1234, abs=2e-4)
    weighted = runner.exact_sds("binary_active_weighted")
    assert weighted["ey0"] == pytest.approx(0.9500, abs=2e-4)
    assert weighted["ate"] == pytest.approx(1.2598, abs=2e-4)
    assert weighted["par"] == pytest.approx(0.8050, abs=2e-4)
    shared = runner.exact_sds("three_arm_shared", 1)
    # The strong shared index converges slowly in the Hermite rule: 0.9187, 0.9203, 0.9209 and
    # 0.9210 at 16, 20, 26 and 30 nodes.  The plan's log reads 0.9208.
    assert shared["ate[0 vs 1]"] == pytest.approx(0.9208, abs=1e-3)


def test_the_noise_control_adds_one_standard_deviation_of_each_labels_own_estimate() -> None:
    bounds = runner.control_bounds()
    assert set(bounds) == set(runner.CONTROL_LABELS)
    assert bounds["three_arm_crossfit__ate_1_vs_0"] == pytest.approx(1.3855, abs=2e-4)
    assert bounds["binary_population__par"] == pytest.approx(0.6605, abs=2e-4)
    assert bounds["weighted__ate"] == pytest.approx(1.2598, abs=2e-4)
    assert bounds["three_arm_shared__ate_0_vs_1"] == pytest.approx(0.9208, abs=1e-3)
    assert all(value > 0.0 for value in bounds.values())
    sizes = {cell.name: runner.GROUPS[cell.group].n for cell in runner.CELLS}
    assert {sizes[f"{label}__corrected"] for label in bounds} == {runner.CALIBRATION_N}


def test_the_truth_follows_the_estimate_names_and_the_reference_arm() -> None:
    exact = laws.THREE_ARM_ACTIVE.truth()
    assert runner.truth("three_arm_active", "par[0]") == pytest.approx(exact["par[0]"])
    means = laws.THREE_ARM_ACTIVE.arm_means()
    observed = laws.THREE_ARM_ACTIVE.observed_mean()
    assert runner.truth("three_arm_active", "par[1]") == pytest.approx(observed - means[1])
    assert runner.truth("three_arm_active", "ate[0 vs 1]") == pytest.approx(means[0] - means[1])
    assert runner.truth("three_arm_active", "paf[1]") == pytest.approx(1.0 - means[1] / observed)
    assert runner.truth("three_arm_active", "log_rr[2 vs 1]") == pytest.approx(
        np.log(means[2] / means[1])
    )
    odds = means / (1.0 - means)
    assert runner.truth("three_arm_active", "log_or[0 vs 1]") == pytest.approx(
        np.log(odds[0] / odds[1])
    )
    # The weighted truth is the half-line quadrature, the tilted law (plan 13.4).
    weighted = laws.BINARY_WEIGHTED.truth()
    for name in ("ey0", "ey1", "ate", "par", "paf", "ey_obs"):
        assert runner.truth("binary_active_weighted", name) == pytest.approx(weighted[name])
    assert runner.truth("binary_active_weighted", "ey0") == pytest.approx(0.5052, abs=5e-5)
    assert runner.truth("binary_active_weighted", "ey1") == pytest.approx(0.6238, abs=5e-5)
    assert runner.truth("binary_active_weighted", "ey_obs") == pytest.approx(0.5681, abs=5e-5)
    assert runner.truth("binary_active_weighted", "ey0") != pytest.approx(
        runner.truth("binary_active", "ey0"), abs=0.01
    )
    with pytest.raises(KeyError):
        runner.truth("binary_active", "ey")
    with pytest.raises(KeyError):
        runner.truth("binary_active", "nonsense")


# ------------------------------------------------------------------------- seeds and bounds


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
    own = {
        replicate_seed(runner.STUDY, scenario, replicate)
        for scenario in runner.SCENARIOS
        for replicate in range(50)
    }
    own |= {
        stream_seed(runner.STUDY, "property_sample", group, replicate)
        for group in runner.GROUPS
        for replicate in range(50)
    }
    assert not own & primary
    assert len(own) == 50 * len(runner.SCENARIOS) + 50 * len(runner.GROUPS)


def test_the_failure_bound_is_one_percent_in_every_module() -> None:
    assert runner.FAILURE_BOUND == 0.01
    assert properties.FAILURE_BOUND == 0.01
    assert dict(runner.STUDY.failure_bound) == {runner.IMPLEMENTATION: 0.01}
    assert properties.PROPERTY_WITNESS is property_verdicts.SAMPLE_WITNESS
    assert len(properties.PROPERTY_RULES) == 1


def _prepared(group: str, fit: runner.Fit) -> tuple[object, object]:
    spec = runner.GROUPS[group]
    frame = runner.sample(spec.law, 600, 17)
    subject = runner.estimator(fit)
    data = subject._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=list(fit.covariates),
        delta=None,
        weights="wt" if fit.weighted else None,
        id=None,
        intermediate=None,
        strata=None,
    )
    return subject, data


@pytest.mark.parametrize(
    ("group", "fit"),
    [(group.name, fit) for group in runner.GROUPS.values() for fit in group.fits],
)
def test_each_configuration_is_admitted_or_refused_by_the_support_predicate(
    group: str, fit: runner.Fit
) -> None:
    subject, data = _prepared(group, fit)
    verdict = per_arm_parametric_support(subject, data)  # type: ignore[arg-type]
    if fit.outcome == "parametric":
        assert verdict.admitted, verdict.detail
    else:
        # The oracle and the shipped-band fit keep the shipped curve: the pair compares the
        # stacked curve against it, and the band control reads its curves.
        assert not verdict.admitted
        assert verdict.axis == "outcome_learner"


# --------------------------------------------------------------------------- one replicate


def test_a_replicate_reads_every_cell_of_its_group() -> None:
    """Every group yields one row per declared cell, in the declared roles."""
    for name, group in runner.GROUPS.items():
        key = (name, 0, runner.BUDGETS[name])
        rows = properties._replication(key, properties._draw(key))
        declared = [cell.name for cell in runner.cells_of(name)]
        assert sorted(row["cell"] for row in rows) == sorted(declared), name
        assert {row["requested_replicates"] for row in rows} == {runner.BUDGETS[name]}
        assert {row["failed_replicates"] for row in rows} == {0}
        roles = {row["cell"]: row["role"] for row in rows}
        assert roles == {cell.name: cell.role for cell in runner.cells_of(name)}, name
        assert {row["n"] for row in rows} == {group.n}
        assert {row["property"] for row in rows} == {cell.family for cell in runner.cells_of(name)}


def test_a_shipped_curve_control_reads_a_narrower_standard_error_than_the_stack() -> None:
    for name in ("three_arm", "binary_population", "three_arm_crossfit", "weighted"):
        key = (name, 1, 5)
        measured, _ = properties._measure(key, properties._draw(key))
        se = {item["cell"]: item["std_error"] for item in measured}
        for cell in runner.cells_of(name):
            if cell.recipe == "shipped_curve":
                corrected = cell.name.split("__shipped_curve__")[0] + "__corrected"
                assert 0.0 < se[cell.name] < se[corrected], cell.name


def test_the_cross_term_dropped_control_adds_the_two_arm_variances() -> None:
    key = ("three_arm_shared", 1, 5)
    measured, results = properties._measure(key, properties._draw(key))
    result = results["main"]
    se = {item["cell"]: item["std_error"] for item in measured}
    control = "three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control"
    arms = [result[f"ey[{a}]"].plugin_std_error for a in (0, 1)]
    assert se[control] == pytest.approx(float(np.hypot(*arms)), rel=1e-12)
    # The arm curves of this law correlate positively, so dropping the term widens the interval.
    assert se[control] > se["three_arm_shared__ate_0_vs_1__corrected"]
    item = next(m for m in measured if m["cell"] == "three_arm_shared__ate_0_vs_1__corrected")
    assert item["reported_cov01"] > 0.0
    assert item["arm0_estimate"] == pytest.approx(float(result["ey[0]"].psi))


def test_the_band_rows_read_the_declared_band_set_and_the_shipped_band() -> None:
    key = ("three_arm", 2, 5)
    _, results = properties._measure(key, properties._draw(key))
    group = runner.GROUPS["three_arm"]
    (band,) = group.bands
    stacked = results["band"].simultaneous
    assert set(stacked.bands) == set(band.names)
    shipped = properties.shipped_banded(results["shipped_band"])
    assert set(shipped.simultaneous.bands) == set(band.names)
    # Same estimates, a different curve: the shipped band is narrower on the arm that omits the
    # generated-design term.
    for name in ("ey[1]", "ey[2]"):
        stacked_width = np.diff(stacked.bands[name])[0]
        shipped_width = np.diff(shipped.simultaneous.bands[name])[0]
        assert shipped_width < stacked_width, name
    assert shipped.estimates["ey[1]"].psi == pytest.approx(results["band"]["ey[1]"].psi)
    rows = properties._joint_rows(group, 2, results, 5)
    assert [row["cell"] for row in rows] == [
        "three_arm__simultaneous_band",
        "three_arm__pointwise_joint_control",
        "three_arm__shipped_curve_band_control",
    ]
    assert [row["role"] for row in rows] == ["positive", "control", "control"]
    # The cross-fitted three-arm group declares the band alone.
    key = ("three_arm_crossfit", 2, 5)
    _, results = properties._measure(key, properties._draw(key))
    rows = properties._joint_rows(runner.GROUPS["three_arm_crossfit"], 2, results, 5)
    assert [row["cell"] for row in rows] == ["three_arm_crossfit__simultaneous_band"]


def test_a_band_replicate_is_lost_when_any_estimate_of_its_set_is_not_stacked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared band set rule: another status than ``influence_curve`` fails the replicate."""
    from cleverly.estimators import _oat_stack
    from cleverly.exceptions import DataError

    original = _oat_stack._arm_failure

    def mutant(stack: object, rconds: dict[str, float], j: int) -> str | None:
        return "forced" if j == 2 else original(stack, rconds, j)  # type: ignore[arg-type]

    monkeypatch.setattr(_oat_stack, "_arm_failure", mutant)
    group = runner.GROUPS["three_arm"]
    frame = runner.sample(group.law, group.n, 5)
    with pytest.raises(DataError, match="stack_failure_plugin"):
        runner.fit_cleverly(frame, group.fits[1])


def test_a_key_names_its_cells_and_the_controls_derived_from_them() -> None:
    names = properties._cells_of(("three_arm", 0, 5))
    positives = [f"{cell.family}/{cell.name}" for cell in runner.cells_of("three_arm")]
    derived = [
        f"interval_calibration/three_arm__ate_0_vs_1__{kind}"
        for kind in ("shrunken_se_control", "noise_control")
    ]
    assert names == [*positives, *derived]
    assert {name for controls in properties.DERIVED_CELLS.values() for name in controls} == {
        f"interval_calibration/{label}__{kind}"
        for label in runner.CONTROL_LABELS
        for kind in ("shrunken_se_control", "noise_control")
    }
    assert set(properties.DERIVED_CELLS) <= {f"{cell.family}/{cell.name}" for cell in runner.CELLS}
    assert properties._cells_of(("three_arm_null", 0, 5)) == [
        "type_i_error/three_arm_null__ate_0_vs_1",
        "type_i_error/three_arm_null__ate_2_vs_1",
    ]
    # The pair writes no derived control.
    assert not [
        n for n in properties._cells_of(("three_arm_generated_design", 0, 5)) if "control" in n
    ]


def test_the_property_loop_runs_through_fit_properties_and_writes_the_cells_of_its_keys() -> None:
    rows = properties.group_rows(runner.GROUPS["three_arm_null"], n_jobs=1, replicates=2)
    assert set(rows["cell"]) == {"three_arm_null__ate_0_vs_1", "three_arm_null__ate_2_vs_1"}
    assert sorted(rows["replicate"].tolist()) == [0, 0, 1, 1]
    assert set(rows["requested_replicates"]) == {2}


def test_the_generated_design_verdict_reads_both_contrasts_of_both_arms() -> None:
    """Thirty replicates of each cell go through the shared verdict without error."""
    rows = properties.group_rows(
        runner.GROUPS["three_arm_generated_design"], n_jobs=1, replicates=30
    )
    summary, _ = property_verdicts.apply_shared_verdicts(
        rows, runner.STUDY, extra_columns=properties.SUMMARY_COLUMNS, rate_labels=()
    )
    properties._generated_design_verdicts(summary, rows)
    pair = summary.loc[summary["property"] == "generated_design"]
    assert set(pair["cell"]) == {
        "three_arm_active__oracle_design__ate_1_vs_0",
        "three_arm_active__oracle_design__ate_2_vs_0",
        "three_arm_active__estimated__ate_1_vs_0",
        "three_arm_active__estimated__ate_2_vs_0",
    }
    assert pair["se_ratio_deficit_lower"].notna().all()
    assert pair["property_passed"].nunique() == 1


def test_the_primary_phase_fits_the_names_it_transcribes() -> None:
    for scenario, estimands in runner.SCENARIOS.items():
        frame, _ = runner.draw_scenario(scenario, 400, 1)
        result = runner.fit_cleverly(frame, runner.PRIMARY_FITS[scenario])
        assert set(estimands) <= set(result.estimates), scenario
        initials = runner.initials(result, estimands)
        assert set(initials) == set(estimands)
        assert all(np.isfinite(value) for value in initials.values())


def test_every_cell_has_a_description() -> None:
    for family, names in runner.STUDY.property_cells.items():
        for name in names:
            tested, required = descriptions.cell(family, name, slug=runner.STUDY.slug)
            assert tested and required, name
    assert descriptions.implementation(runner.IMPLEMENTATION)


def test_the_clause_bindings_name_this_slug_and_leave_the_binding_search_to_the_run() -> None:
    import pandas as pd

    table = pd.read_csv(
        ROOT / "tests" / "canonical" / "clause-bindings.csv", dtype=str, keep_default_na=False
    )
    own = table.loc[table["subject"] == runner.STUDY.slug]
    pairs = set(zip(own["property"], own["clause"], strict=True))
    assert len(own) == len(pairs) == 11
    assert ("interval_calibration", "se ratio above band") in pairs
    assert {prop for prop, _ in pairs} == FAMILIES
    # The two joint rows carry a precedent until the binding search runs.
    joint = own.loc[own["property"] == "simultaneous_coverage"]
    assert set(joint["perturbation"]) == {"se_low"}
    assert (
        "TODO(binding search)"
        in (
            ROOT / "tests" / "studies" / "canonical_ctmle_oat_per_arm_parametric_extensions.py"
        ).read_text()
    )


# ------------------------------------------------------------------------------ the pilot

#: The declared-size pilot (plans repository ``reviews/x58/x58b/pilot/``: ``x58b2_step2_report.log``
#: at 5,000 replicates per group on seed 9_258_000_058, ``x58b2_declared_report.log`` at the
#: declared budgets on seed 9_558_000_058, ``x58b2_primary_report.log`` for the primary phase on
#: seed 9_458_000_058; code commit ``7880f787``).  The pilot rows are not committed, so the figures
#: are constants.  Each value is ``(lowest SE ratio, highest SE ratio, highest upper 99% bound)``
#: over the group's positive calibration rows, at the declared budget (step 4).
PILOT_STEP4_RATIOS = {
    "three_arm": (0.9929, 1.0015, 1.0122),
    "three_arm_shared": (0.9887, 0.9950, 1.0103),
    "binary_population": (0.9881, 1.0107, 1.0295),
    "three_arm_crossfit": (1.0084, 1.0138, 1.0278),
    "three_arm_generated_design": (1.0145, 1.0355, 1.0531),
    "weighted": (0.9903, 0.9966, 1.0103),
    "weighted_crossfit": (0.9907, 1.0018, 1.0187),
    "composition": (1.0090, 1.0158, 1.0298),
}
#: Step 2 (5,000 replicates per group), for the same groups: lowest and highest SE ratio.
PILOT_STEP2_RATIOS = {
    "three_arm": (0.9821, 1.0031),
    "three_arm_shared": (0.9837, 1.0048),
    "binary_population": (0.9854, 0.9951),
    "three_arm_crossfit": (1.0015, 1.0329),
    "three_arm_generated_design": (1.0061, 1.0277),
    "weighted": (0.9825, 1.0098),
    "weighted_crossfit": (0.9908, 1.0140),
    "composition": (0.9971, 1.0117),
}
#: Joint coverage of the band cells at the declared budget (step 4) and at step 2.
PILOT_BANDS = {
    "three_arm": (0.9439, 0.9402),
    "binary_population": (0.9467, 0.9488),
    "three_arm_crossfit": (0.9484, 0.9496),
}
#: The negative controls at the declared budget: the SE ratio or the joint coverage.
PILOT_CONTROLS = {
    "three_arm__ey_1__shipped_curve__shrunken_se_control": 0.7767,
    "three_arm__pointwise_joint_control": 0.8185,
    "three_arm__shipped_curve_band_control": 0.8527,
    "three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control": 1.1560,
    "binary_population__par__shipped_curve__shrunken_se_control": 0.7753,
    "binary_population__pointwise_joint_control": 0.8373,
    "binary_population__shipped_curve_band_control": 0.8707,
    "three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control": 0.7952,
    "weighted__ey0__shipped_curve__shrunken_se_control": 0.8130,
}
#: Rejections of the two type I rows over 1,600 replicates (step 4), and the primary phase
#: figure: all 11 rows pass at 1,600 per law, and 200 bootstrap resamples all pass.
PILOT_TYPE_I_REJECTIONS = (0.0450, 0.0525)
PILOT_PRIMARY = 1.0000
#: The predictive model's unit probability at the declared budgets, step 4 (study-wise 0.7926 with
#: all 19 units).  Step 2 gave 0.7675 over 18 units.  The model carries the pilot's own sampling
#: error, so it cannot reach the target for the cross-fitted groups at any budget (step 2), and
#: three units sit below it at step 4.  The budgets stay at the plan values on the X58b1 precedent;
#: the declared reading passes every row at both steps.
PILOT_STEP4_UNITS = {
    "three_arm": 0.99993,
    "three_arm_shared": 0.99846,
    "binary_population": 0.98897,
    "three_arm_crossfit": 0.99736,
    "three_arm_generated_design": 0.80937,
    "weighted": 0.99949,
    "weighted_crossfit": 0.99842,
    "composition": 0.99653,
}
PILOT_STUDY_WISE = 0.7926


def test_the_pilot_reads_every_gated_row_inside_the_calibration_band_at_the_declared_size() -> None:
    low, high = runner.STUDY.margins.calibration_se_ratio
    assert (
        set(PILOT_STEP4_RATIOS)
        == set(PILOT_STEP2_RATIOS)
        == set(GATED_ROWS) | {"three_arm_generated_design"}
    )
    for group, (lowest, highest, bound) in PILOT_STEP4_RATIOS.items():
        assert low < lowest <= highest < bound < high, group
        assert low < PILOT_STEP2_RATIOS[group][0] <= PILOT_STEP2_RATIOS[group][1] < high, group
    # The pair is the closest row: its upper 99% bound sits 0.017 under the band edge.
    assert (
        max(bound for *_, bound in PILOT_STEP4_RATIOS.values())
        == (PILOT_STEP4_RATIOS["three_arm_generated_design"][2])
    )
    assert high - PILOT_STEP4_RATIOS["three_arm_generated_design"][2] > 0.016


def test_the_pilot_band_cells_and_the_type_i_and_primary_figures_pass() -> None:
    margins = runner.STUDY.margins
    low, high = margins.calibration_coverage
    for label, figures in PILOT_BANDS.items():
        replicates = runner.BUDGETS[label]
        for coverage in figures:
            assert low < coverage < high, label
        step4 = figures[0]
        interval = clopper_pearson(
            round(step4 * replicates), replicates, confidence_level=margins.confidence_level
        )
        assert interval.within(low, high), label
    # The in-sample three-arm band reads lowest, 0.9402 at step 2 against the exact 0.9505.
    assert min(f[1] for f in PILOT_BANDS.values()) == PILOT_BANDS["three_arm"][1]
    replicates = runner.BUDGETS["three_arm_null"]
    ceiling = margins.alpha + margins.type_i_margin
    for rate in PILOT_TYPE_I_REJECTIONS:
        covered = round(rate * replicates)
        assert (
            clopper_pearson(covered, replicates, confidence_level=margins.confidence_level).high
            <= ceiling
        )
    assert PILOT_PRIMARY > TARGET


def test_the_pilot_controls_are_red_on_their_declared_side() -> None:
    low, high = runner.STUDY.margins.calibration_se_ratio
    assert set(PILOT_CONTROLS) == set(runner.NEGATIVE_CONTROLS)
    declared = runner.CONFIGURATION["negative_control_expected"]
    for name, value in PILOT_CONTROLS.items():
        side = runner.NEGATIVE_CONTROLS[name]
        figure = declared[name]
        if "se_ratio" in figure:
            assert abs(value - figure["se_ratio"]) < 0.04, name
            assert (value < low) if side == "below" else (value > high), name
        else:
            assert abs(value - figure["joint_coverage"]) < 0.01, name
            assert value < 0.95, name


def test_the_pilot_study_wise_figure_is_recorded_with_its_shortfall() -> None:
    """The predictive model gives 0.7926 over 19 units: three groups sit under the target."""
    below = {g for g, v in PILOT_STEP4_UNITS.items() if v < TARGET}
    assert below == {"binary_population", "three_arm_generated_design", "composition"}
    assert (
        pytest.approx(float(np.prod(list(PILOT_STEP4_UNITS.values()))), abs=1e-3)
        == PILOT_STUDY_WISE
    )
    assert len(PILOT_STEP4_UNITS) + runner.UNITS["negative_controls"] + 2 == UNITS
