"""The design of ``projection-evidence`` (P1 plan rev 3, section 5), without a run.

Every number plan 5.1 computed for the ``cde_obs`` law, its controls and its bands is recomputed
here by quadrature, and the arms and cells are checked against the plan's table.  The pilot rule,
the point-reading fallback and the ``pilot_2`` selection are checked on stated ratios until the
pilot has run.  The summary step is run on synthetic rows of every declared cell, and the cost
gate reads the committed smoke timings.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.studies import conditional_projection as cp
from tests.studies import point_reading, repeated_sizing
from tests.studies import projection_evidence as study
from tests.studies import projection_evidence_law as law
from tests.studies import projection_evidence_pilot as pilot
from tests.studies import projection_evidence_properties as properties
from tests.studies.evidence.property_verdicts import DIAGNOSTIC_ROLE
from tests.studies.evidence.registry import registered


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


# --------------------------------------------------------------------------- arms and cells


def test_the_arms_are_six_with_14_gated_rows() -> None:
    sizes = numbers()["calibration_arms"]
    assert sizes == {
        "__cde_z0": 2,
        "__cde_z1": 2,
        "__cde_crr_z0": 2,
        "__cde_crr_z1": 2,
        "__pooled": 3,
        "__pooled_crr": 3,
    }
    assert sum(sizes.values()) == 14


def test_every_cell_is_declared_once_and_the_record_reads_them() -> None:
    cells = [cell for names in study.PROPERTY_CELLS.values() for cell in names]
    assert len(cells) == len(set(cells))
    assert study.STUDY.property_cells == study.PROPERTY_CELLS
    assert properties.PROPERTY_CELLS is study.PROPERTY_CELLS
    assert set(study.PROPERTY_CELLS) == {study.CALIBRATION, study.BIAS, study.SIMULTANEOUS}
    controls = [c for c in study.PROPERTY_CELLS[study.BIAS] if not c.startswith("cde_dr_4k")]
    assert len(controls) == 10
    gated = [g.label for g in study.GROUPS if g.band == "gated"]
    assert gated == ["band_obs_4k", "band_crr_4k", "band_clustered_4k", "band_rep3_4k"]
    joint = study.PROPERTY_CELLS[study.SIMULTANEOUS]
    for label in gated:
        assert f"{label}__{study.BAND_KIND}" in joint
        assert f"{label}__{study.JOINT_CONTROL_KIND}" in joint
    # The free readings publish no pointwise control.
    for label in ("cde_4k_z0", "cde_4k_z1", "pooled_4k"):
        assert f"{label}__{study.BAND_KIND}" in joint
        assert f"{label}__{study.JOINT_CONTROL_KIND}" not in joint


def test_the_study_is_declared_and_not_registered() -> None:
    assert study.SLUG not in {record.slug for record in registered()}


def test_the_seeds_are_unique_against_every_registered_study() -> None:
    # The plan's 20261501 and 20261502 are held by ``calibration-slope-warning``.
    taken = {r.seed for r in registered()} | {r.resampling_seed for r in registered()}
    assert study.SEED not in taken
    assert study.RESAMPLING_SEED not in taken
    assert study.SEED != study.RESAMPLING_SEED


# ------------------------------------------------------------------------ plan 5.1 numbers


def test_the_cde_truths_and_efficient_sds_are_the_plans() -> None:
    cde = numbers()["cde"]
    assert cde[0]["beta"] == pytest.approx((0.0715, 0.1101), abs=1e-4)
    assert cde[1]["beta"] == pytest.approx((-0.1358, 0.1007), abs=1e-4)
    assert cde[0]["sd"] == pytest.approx((0.0253, 0.0433), abs=1e-4)
    assert cde[1]["sd"] == pytest.approx((0.0224, 0.0384), abs=1e-4)
    assert numbers()["smallest_density_product"] == pytest.approx(0.0355, abs=2e-4)
    assert law.quadrature_error() < 1e-10


def test_the_targeted_limit_is_unbiased_and_each_mutant_is_not() -> None:
    cde = numbers()["cde"]
    for level in study.LEVELS:
        assert np.abs(cde[level]["tmle_full"]).max() < 1e-8
    assert cde[0]["tmle_no_density"] == pytest.approx((3.15, -0.59), abs=0.01)
    assert cde[1]["tmle_no_density"] == pytest.approx((-3.06, -0.39), abs=0.01)
    assert cde[0]["tmle_no_selection"] == pytest.approx((-3.71, -0.28), abs=0.01)
    assert cde[1]["tmle_no_selection"] == pytest.approx((2.99, -0.18), abs=0.01)


def test_the_bias_controls_read_the_plans_displacements() -> None:
    cde = numbers()["cde"]
    assert cde[0]["plugin_stratum"] == pytest.approx((-0.71, 0.05), abs=0.01)
    assert cde[1]["plugin_stratum"] == pytest.approx((0.65, 0.11), abs=0.01)
    assert cde[0]["total_effect"] == pytest.approx((-4.34, 0.30), abs=0.01)
    assert cde[1]["total_effect"] == pytest.approx((4.36, 0.58), abs=0.01)
    assert cde[0]["plugin_wrong"] == pytest.approx((-0.07, -1.06), abs=0.01)
    assert cde[1]["plugin_wrong"] == pytest.approx((0.75, -1.06), abs=0.01)


def test_every_declared_control_detects_its_displacement() -> None:
    """Each control row reads a term whose displacement fails the rule with near certainty."""
    cde = numbers()["cde"]
    readings = {
        "plugin_stratum": "plugin_stratum",
        "no_selection": "tmle_no_selection",
        "no_intermediate_weight": "tmle_no_density",
        "plugin_wrong": "plugin_wrong",
        "total_effect": "total_effect",
    }
    for group in study.GROUPS:
        for control, terms in group.controls.items():
            if control in study.CALIBRATION_CONTROL_KINDS:
                continue
            index = [study.CDE_TERMS.index(term) for term in terms]
            for level in study.control_levels(control, group):
                for i in index:
                    gap = float(cde[level][readings[control]][i])
                    detected = cp.bias_pass_probability(gap, group.replicates, control=True)
                    assert detected > 0.999, (control, level, gap)
    # The rows left out are those too near the 0.25 SD margin to be declared.
    assert study.GROUP_BY_LABEL["cde_dr_4k"].controls["plugin_wrong"] == ("W1",)


def test_the_positive_bias_rows_pass_at_the_targeted_limit() -> None:
    group = study.GROUP_BY_LABEL["cde_dr_4k"]
    for level in study.LEVELS:
        for bias in numbers()["cde"][level]["tmle_full"]:
            assert cp.bias_pass_probability(float(bias), group.replicates, control=False) > 0.99


def test_the_shrunken_control_lies_below_the_band() -> None:
    margins = study.STUDY.margins
    probability = repeated_sizing.calibration_control_probability(
        margins, study.BUDGETED_R, study.SHRUNKEN_FACTOR
    )
    assert probability > 0.9999


def test_the_band_critical_values_and_pointwise_joint_coverages_are_the_plans() -> None:
    cde_joint, cde_critical = numbers()["cde_band"]
    proj_joint, proj_critical = numbers()["proj_band"]
    assert cde_joint == pytest.approx(0.871, abs=0.005)
    assert cde_critical == pytest.approx(2.377, abs=0.01)
    assert proj_joint == pytest.approx(0.779, abs=0.005)
    assert proj_critical == pytest.approx(2.63, abs=0.01)


def test_a_correct_band_passes_at_2400_and_the_pointwise_control_fails_there() -> None:
    margins = study.STUDY.margins
    assert repeated_sizing.joint_band_pass_probability(margins, study.BAND_R, 0.95) > 0.99
    # Plan 5.4: a band at 0.945 passes with 0.990 (0.9897 exactly).
    low = repeated_sizing.joint_band_pass_probability(margins, study.BAND_R, 0.945)
    assert low == pytest.approx(0.990, abs=0.001)
    # The pointwise control passes when the 99% exact upper end of its coverage is below 0.95.
    proj_joint, _ = numbers()["proj_band"]
    counts = np.arange(study.BAND_R)
    upper = stats.beta.ppf(0.995, counts + 1, study.BAND_R - counts)
    largest = int(counts[upper < 1.0 - margins.alpha].max())
    assert float(stats.binom.cdf(largest, study.BAND_R, proj_joint)) > 0.9999


def test_the_family_passes_at_the_budget_with_correct_intervals() -> None:
    margins = study.STUDY.margins
    single = float(repeated_sizing.calibration_pass(margins, study.BUDGETED_R, np.array(1.0)))
    assert single ** sum(numbers()["calibration_arms"].values()) > 0.99


def test_the_primary_is_the_cde_intercepts_at_2400() -> None:
    assert study.STUDY.scenarios == {
        "cde_4k": ("z0:cate_projection[(intercept)]", "z1:cate_projection[(intercept)]")
    }
    covered = int(0.95 * pilot.REPLICATES)
    summary = pd.DataFrame(
        {
            "cell": [
                study.calibration_cell(study.GROUP_BY_LABEL["cde_4k"], arm, "(intercept)")
                for arm in ("__cde_z0", "__cde_z1")
            ],
            "covered": covered,
            "replicates": pilot.REPLICATES,
        }
    )
    assert pilot.primary_replicates(summary) <= study.PRIMARY_REPLICATES == 2_400
    assert study.PRIMARY_COVERAGE_FLOOR == 0.90


def test_the_truths_name_every_reported_coefficient() -> None:
    for group in study.GROUPS:
        truths = study.truth(group)
        for level in group.levels:
            assert set(group.names(level)) <= set(truths), group.label
    assert study.truth(study.GROUP_BY_LABEL["pooled_4k"]) == cp.truth("observational")


# --------------------------------------------------------------------------- the summary


def synthetic_rows(seed: int = 0, replicates: int = 400) -> pd.DataFrame:
    """Rows of every declared cell, drawn at a correct interval, a failing control, a band."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            group = study.GROUP_BY_LABEL[pilot.group_of(cell)] if family != study.BIAS else None
            if family == study.SIMULTANEOUS:
                gated = group is not None and group.band == "gated"
                control = cell.endswith(study.JOINT_CONTROL_KIND)
                role = "control" if control else ("positive" if gated else DIAGNOSTIC_ROLE)
                coverage = 0.78 if control else 0.95
                covered = rng.random(replicates) < coverage
                estimate = np.where(covered, 1.0, 3.0)
                se, truth = np.full(replicates, 2.0), np.zeros(replicates)
            else:
                role = "positive"
                shift = 0.0
                if family == study.BIAS and not cell.startswith("cde_dr_4k"):
                    role, shift = "control", 3.0
                if cell.endswith("__shrunken_se_control"):
                    role = "control"
                truth = np.zeros(replicates)
                estimate = rng.normal(shift, 1.0, replicates)
                se = np.full(replicates, 0.7 if role == "control" and family != study.BIAS else 1.0)
                covered = np.abs(estimate) <= 1.96 * se
            rows += [
                {
                    "property": family,
                    "cell": cell,
                    "role": role,
                    "replicate": r,
                    "n": 4_000,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": float(truth[r]),
                    "estimate": float(estimate[r]),
                    "std_error": float(se[r]),
                    "covered": int(covered[r]),
                    "rejected": int(not covered[r]),
                }
                for r in range(replicates)
            ]
    frame = pd.DataFrame(rows)
    for column in properties.REPLICATE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def test_the_summary_reads_every_declared_cell() -> None:
    summary = properties.summarize_properties(synthetic_rows())
    assert set(summary["cell"]) == {c for cells in study.PROPERTY_CELLS.values() for c in cells}
    diagnostic = summary.loc[summary["role"] == DIAGNOSTIC_ROLE]
    assert set(diagnostic["cell"]) == {
        "cde_4k_z0__simultaneous_band",
        "cde_4k_z1__simultaneous_band",
        "pooled_4k__simultaneous_band",
    }
    assert diagnostic["passed"].all()
    controls = summary.loc[summary["role"] == "control"]
    assert controls["passed"].all(), controls.loc[~controls["passed"], "cell"].tolist()


# ------------------------------------------------------------------- streams and the pilot


def test_each_stream_has_its_own_samples_and_folds_per_group() -> None:
    groups = [group.label for group in study.GROUPS]
    seen: set[int] = set()
    for stream in properties.STREAMS:
        samples = {properties.sample_seed(g, r, stream) for g in groups for r in range(100)}
        folds = {properties.fold_seed(g, r, stream) for g in groups for r in range(100)}
        assert len(samples) == len(folds) == 100 * len(groups)
        assert not (samples | folds) & seen
        assert not samples & folds
        seen |= samples | folds
    assert set(properties.STREAMS) == {"property", "pilot", "pilot_2", "smoke"}


def _pilot_frame(ratios: Any, se: float = 0.01, replicates: int = pilot.REPLICATES) -> pd.DataFrame:
    cells = pilot.positive_cells()
    return pd.DataFrame(
        {
            "cell": list(cells),
            "group": [pilot.group_of(cell) for cell in cells],
            "arm": [pilot.arm_of(cell) for cell in cells],
            "replicates": replicates,
            "se_ratio": ratios,
            "se_ratio_se": se,
            "covered": int(0.95 * pilot.REPLICATES),
        }
    )


NO_PAIRS = pd.DataFrame(columns=["first", "second", "correlation"])

#: Both pilots' draws: the fallback runs only after ``pilot_2`` (plan 5.0 rule 4).
BOTH = pilot.REPLICATES + pilot.EXTENSION


def _alternating(spread: float) -> np.ndarray:
    index = np.arange(len(pilot.positive_cells()))
    return np.where(index % 2 == 0, 1.0 - spread, 1.0 + spread)


def test_the_pilot_rule_sizes_r_by_x32a_on_correct_ratios() -> None:
    assert pilot.declared_replicates(_pilot_frame(1.0), NO_PAIRS) == (study.BUDGETED_R, "x32a")
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the pilot has not run; the declaration commit adds it (plan 5.0)")
    summary = pd.read_csv(pilot.PILOT_CSV)
    computed, _ = pilot.declared_replicates(summary, pd.read_csv(pilot.CORRELATIONS_CSV))
    assert computed >= study.BUDGETED_R
    assert max(computed, study.BUDGETED_R) == study.DECLARED_R == study.R


def test_the_point_reading_fallback_runs_where_x32a_cannot() -> None:
    frame = _pilot_frame(_alternating(0.02), replicates=BOTH)
    with pytest.raises(ValueError, match="calibration rule"):
        pilot.x32a_replicates(frame, NO_PAIRS)
    replicates, rule = pilot.declared_replicates(frame, NO_PAIRS)
    assert rule == "point_reading"
    assert replicates == study.BUDGETED_R
    ratios = dict(zip(frame["cell"], frame["se_ratio"], strict=True))
    errors = dict(zip(frame["cell"], frame["se_ratio_se"], strict=True))
    found = point_reading.point_reading_replicates(
        study.STUDY.margins, ratios, errors, target=study.POINT_READING_TARGET
    )
    assert found % repeated_sizing.BUDGET_STEP == 0
    margins = study.STUDY.margins
    assert point_reading.point_reading_pass(margins, ratios, found) >= 0.95
    if found > repeated_sizing.BUDGET_STEP:
        below = found - repeated_sizing.BUDGET_STEP
        assert point_reading.point_reading_pass(margins, ratios, below) < 0.95


def test_the_point_reading_fallback_refuses_a_defect_suspect() -> None:
    # Outside the absorb band: refused even when diagnosed.
    frame = _pilot_frame(_alternating(0.04), replicates=BOTH)
    with pytest.raises(ValueError, match="outside"):
        pilot.declared_replicates(frame, NO_PAIRS, diagnosed=pilot.positive_cells())
    # Inside the band but with a 99% interval that excludes 1: refused until diagnosed.
    tight = _pilot_frame(_alternating(0.02), se=0.002, replicates=BOTH)
    with pytest.raises(ValueError, match="no diagnosis"):
        pilot.declared_replicates(tight, NO_PAIRS)
    assert pilot.declared_replicates(tight, NO_PAIRS, diagnosed=pilot.positive_cells())[1] == (
        "point_reading"
    )
    # Both sides of 1 count.
    assert point_reading.excludes_one(1.02, 0.002)
    assert point_reading.excludes_one(0.98, 0.002)
    assert not point_reading.excludes_one(0.98, 0.01)


def test_pilot_2_draws_each_arm_whose_plug_in_pass_is_low() -> None:
    assert pilot.pilot_2_arms(_pilot_frame(1.0)) == ()
    ratios = np.ones(len(pilot.positive_cells()))
    low = [i for i, cell in enumerate(pilot.positive_cells()) if pilot.arm_of(cell) == "__pooled"]
    ratios[low[0]] = 0.95
    assert pilot.pilot_2_arms(_pilot_frame(ratios)) == ("__pooled",)


def test_a_low_band_pilot_rebuilds_its_group_at_4000() -> None:
    bands = pd.DataFrame(
        [
            {"cell": f"{g.label}__{study.BAND_KIND}", "group": g.label, "coverage": 0.95}
            for g in study.GROUPS
            if g.band
        ]
    )
    bands.loc[bands["group"] == "band_rep3_4k", "coverage"] = 0.93
    sized = pilot.band_replicates(bands)
    assert set(sized) == {g.label for g in study.GROUPS if g.band == "gated"}
    assert sized["band_rep3_4k"] == study.BAND_REBUILT_R
    assert sized["band_obs_4k"] == study.BAND_R


# ---------------------------------------------------------------------- the cost and files


def test_the_smoke_cost_stays_inside_the_cost_gate() -> None:
    timings = pd.read_csv(study.SMOKE_TIMINGS)
    assert set(timings["group"]) == {group.label for group in study.GROUPS}
    for row in timings.itertuples(index=False):
        declared = study.GROUP_BY_LABEL[row.group].seconds
        assert row.seconds_per_fit <= study.COST_GATE * declared, row.group
    assert numbers()["cpu_hours"] == pytest.approx(study.run_cpu_hours())
    assert study.run_cpu_hours() == pytest.approx(6.35, abs=0.05)
    assert study.pilot_cpu_hours() == pytest.approx(2.65, abs=0.05)


def test_the_declaration_names_every_file_a_run_writes() -> None:
    from tests.canonical.declared_run import RECORD_FILES
    from tests.canonical.regenerate import ARTIFACT_NAMES, PROPERTY_FAILURE_FILES

    assert set(study.RUN_FILES) == {*ARTIFACT_NAMES, *RECORD_FILES, *study.STUDY.property_artifacts}
    assert set(study.CONDITIONAL_RUN_FILES) == set(PROPERTY_FAILURE_FILES)


# ------------------------------------------------------------------------ the fitted mutants


@cache
def _fit() -> Any:
    group = study.GROUP_BY_LABEL["cde_dr_4k"]
    frame, _ = study.draw_from_seed(group.label, group.n, 7)
    return study.fit_cde(group, frame, random_state=7)


@pytest.mark.parametrize("level", study.LEVELS)
def test_the_full_mutant_is_the_fit_and_each_mutation_moves_it(level: int) -> None:
    result = _fit()[float(level)]
    group = study.GROUP_BY_LABEL["cde_dr_4k"]
    reported = np.array([float(result.estimates[f"{group.stem}[{t}]"].psi) for t in group.terms])
    full = study.mutant_tmle(result, level, "full")
    assert full == pytest.approx(reported, abs=1e-8)
    for covariate in ("no_density", "no_selection"):
        moved = study.mutant_tmle(result, level, covariate)
        assert np.abs(moved - full).max() > 1e-3, covariate


# ------------------------------------------------------- the reading paths (review S5)


@pytest.mark.parametrize("label", ["cde_4k", "cde_crr_4k"])
def test_the_engine_fit_is_the_study_route_at_both_levels(label: str) -> None:
    """The CDE groups measure ``fit_cde``; the registered row lifts the study route."""
    from cleverly import CausalStudy, ConditionalContrast, ControlledDirectEffect, PointTreatment

    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(label, 1_000, 5)
    engine = study.fit_cde(group, frame, 11)
    route = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), intermediate="Z"
        ),
    )
    for level in study.LEVELS:
        identified = route.identify(
            ControlledDirectEffect(
                float(level),
                ConditionalContrast(study.CDE_MODIFIERS, scale=group.scale, model=study.CDE_BASIS),  # type: ignore[arg-type]
            )
        )
        fit = study._quiet(identified.estimate, **study.learners(group), random_state=11)
        ours = [(e.psi, e.std_error) for e in engine[float(level)].estimates.values()]
        theirs = [(e.psi, e.std_error) for e in fit.estimates.values()]
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-12)
        assert list(engine[float(level)].estimates) == list(fit.estimates)


#: The support row each gated band group's coefficients must read (plan 2.3).
BAND_ROWS = {
    "band_obs_4k": "T49",
    "band_crr_4k": "T50",
    "band_clustered_4k": "T49.clustered",
    "band_rep3_4k": "T49.repeats",
}


@pytest.mark.parametrize("label", sorted(BAND_ROWS))
def test_each_band_group_reads_the_row_its_band_evidence_names(label: str) -> None:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(label, group.n, 3)
    fit = study.fit_projection(group, frame, 3)
    assert {(e.support.label, e.support.row) for e in fit.estimates.values()} == {
        ("supported", BAND_ROWS[label])
    }
    assert {g.label for g in study.GROUPS if g.band == "gated"} == set(BAND_ROWS)


def test_the_plugin_stratum_control_is_the_stratum_projection_on_the_exact_law() -> None:
    """On the discrete law the control equals ``law.plugin_stratum(z)``, not the CDE."""
    from tests import discrete_law_cde_projection as discrete
    from tests.unit import test_cde_projection as exact

    frame = discrete.frame().rename(columns={"W": "W1"})
    from cleverly import CausalStudy, ConditionalContrast, ControlledDirectEffect, PointTreatment
    from cleverly.bases import Linear, Modifiers

    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1",), missingness="Delta", intermediate="Z"
    )
    contrast = ConditionalContrast(
        Modifiers(["W1"], domains={"W1": (0.0, 2.0)}),
        model=Linear(["W1"]),  # type: ignore[arg-type]
    )
    for level in study.LEVELS:
        identified = CausalStudy(frame, design=design).identify(
            ControlledDirectEffect(float(level), contrast)
        )
        fit = study._quiet(identified.estimate, **exact.oracle())
        np.testing.assert_allclose(
            study.plugin_stratum(fit, level), discrete.plugin_stratum(level), atol=1e-10
        )
        assert np.max(np.abs(discrete.plugin_stratum(level) - discrete.truth(level))) > 0.02


def test_the_declared_wrong_outcome_is_the_logit_shift_at_both_arms_and_levels() -> None:
    from scipy.special import expit

    group = study.GROUP_BY_LABEL["cde_dr_4k"]
    frame, _ = study.draw_from_seed(group.label, 1_000, 7)
    fits = study.fit_cde(group, frame, 7)
    w = frame[["W1", "W2", "W3"]].to_numpy()
    for level in study.LEVELS:
        arms = fits[float(level)].repeats[0].nuisance.outcome.arms
        for a in (0.0, 1.0):
            expected = expit(law.outcome_logit(w, a, level) + law.declared_error(w, a))
            np.testing.assert_allclose(arms[a], expected, rtol=0, atol=1e-10)
        assert np.max(np.abs(law.declared_error(w, 1.0) - law.declared_error(w, 0.0))) > 0.05


def test_the_fallback_answers_only_the_no_budget_case_after_pilot_2() -> None:
    """Review S6: another sizing error stops the declaration, and so do missing extensions."""
    low = np.ones(len(pilot.positive_cells()))
    low[[pilot.arm_of(c) == "__pooled" for c in pilot.positive_cells()]] = 0.945
    assert pilot.pilot_2_arms(_pilot_frame(low)) == ("__pooled",)
    with pytest.raises(ValueError, match="only after pilot_2"):
        pilot.declared_replicates(_pilot_frame(low), NO_PAIRS)
    nan = _pilot_frame(_alternating(0.02), replicates=BOTH)
    cells = list(pilot.positive_cells())
    pairs = pd.DataFrame([{"first": cells[0], "second": cells[1], "correlation": float("nan")}])
    with pytest.raises(ValueError) as caught:
        pilot.declared_replicates(nan, pairs)
    assert pilot.NO_BUDGET not in str(caught.value)
    assert "pilot_2" not in str(caught.value)
