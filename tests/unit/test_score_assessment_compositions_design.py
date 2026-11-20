"""The design of ``score-assessment-compositions`` (P1 plan rev 3, section 5B), without a run.

The arms and cells are checked against the plan's table, the sizing scenarios of plan 5B.3 are
recomputed from the shipped ``score-assessment`` pilot, and the controls' detection
probabilities from the shipped clustered row-level ratios.  Each group's fit is checked to read
exactly the composition it measures.  The pilot rule, the point-reading fallback and the
row-variance precondition are checked on stated ratios until the pilot has run, the summary
step runs on synthetic rows of every declared cell, and the cost gate reads the committed smoke
timings.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import effect_curves as ec
from tests.studies import conditional_projection as cp
from tests.studies import point_reading, repeated_sizing
from tests.studies import score_assessment as sa
from tests.studies import score_assessment_compositions as study
from tests.studies import score_assessment_compositions_pilot as pilot
from tests.studies import score_assessment_compositions_properties as properties
from tests.studies.evidence.registry import registered


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


# --------------------------------------------------------------------------- arms and cells


def test_the_arms_are_ten_with_35_gated_rows() -> None:
    sizes = numbers()["calibration_arms"]
    assert len(sizes) == 10
    assert sum(sizes.values()) == 35
    for group in study.GROUPS:
        assert sizes[group.risk_arm] == 3
        assert sizes[group.calibration_arm] == 4
    assert [g.label for g in study.GROUPS] == [
        "assess_weighted_2k",
        "assess_clustered_4k",
        "assess_missing_4k",
        "assess_arms3_4k",
        "assess_insample_2k",
    ]


def test_every_cell_is_declared_once_and_the_record_reads_them() -> None:
    cells = [cell for names in study.PROPERTY_CELLS.values() for cell in names]
    assert len(cells) == len(set(cells)) == 35 + 3
    assert study.STUDY.property_cells == study.PROPERTY_CELLS
    assert properties.PROPERTY_CELLS is study.PROPERTY_CELLS
    controls = [cell for cell in cells if cell.endswith("_control")]
    assert controls == [
        "assess_weighted_2k_S_risk_difference__shrunken_se_control",
        "assess_clustered_4k_S_risk_difference__row_variance_control",
        "assess_clustered_4k_S_calibration_slope__row_variance_control",
    ]


def test_the_groups_measure_each_listed_composition_once() -> None:
    # Plan section 8: weights, clusters, missing, multi_arm and not_cross_fitted.
    codes = [group.code for group in study.GROUPS]
    assert sorted(codes) == sorted(
        ["weights", "clusters", "missing", "multi_arm", "not_cross_fitted"]
    )
    assert set(codes) <= set(ec.SCORE_REASONS)


def test_the_study_is_declared_and_not_registered() -> None:
    assert study.SLUG not in {record.slug for record in registered()}


def test_the_seeds_are_unique_against_every_registered_study() -> None:
    from tests.studies import projection_evidence

    taken = {r.seed for r in registered()} | {r.resampling_seed for r in registered()}
    taken |= {projection_evidence.SEED, projection_evidence.RESAMPLING_SEED}
    assert study.SEED not in taken
    assert study.RESAMPLING_SEED not in taken


# ------------------------------------------------------------------------------ the truths


def test_the_truths_are_the_paired_designs_on_each_law() -> None:
    truths = numbers()["truths"]
    observational = sa.truth("observational", "difference", study.KIND)
    # The pair (2, 0), the MAR outcome and the in-sample fit keep the observational truth.
    for law in ("multi_arm", "missing", "observational"):
        assert truths[law] == pytest.approx(observational, abs=1e-12), law
    assert set(observational) == set(sa.parameter_labels(True))
    # The target population of the weighted law is tilted: its truth moves.
    moved = max(abs(truths["weighted"][k] - observational[k]) for k in observational)
    assert moved > 0.01
    # The clustered arm regressions are marginal over the cluster effect.
    assert truths["clustered"]["paired"] != pytest.approx(observational["paired"], abs=1e-3)
    assert sa.quadrature_error("observational", "difference", study.KIND) < 1e-8


def test_the_weighted_truth_is_the_tilted_quadrature() -> None:
    w, mass = cp._grid(32)
    tilted = cp.law_mass("weighted", w, mass)
    assert tilted == pytest.approx(mass * cp.sampling_weight(w) / (mass @ cp.sampling_weight(w)))
    coarse = study.truth("weighted", nodes=32)
    assert coarse == pytest.approx(numbers()["truths"]["weighted"], abs=1e-8)


# ------------------------------------------------------------------------- sizing (5B.3)


def _shipped(group: str) -> tuple[list[float], list[float]]:
    """The shipped pilot's risk and calibration ratios of one ``score-assessment`` group."""
    rows = pd.read_csv(sa.PILOT_CSV).set_index(["group", "label"])
    risk = [float(rows.loc[(group, label), "se_ratio"]) for label in study.RISK_ROWS]
    calibration = [float(rows.loc[(group, label), "se_ratio"]) for label in study.CALIBRATION_ROWS]
    return risk, calibration


def _flat(rho: float) -> repeated_sizing.Correlations:
    def correlations(labels: Any) -> np.ndarray:
        k = len(labels)
        return np.full((k, k), rho) + (1.0 - rho) * np.eye(k)

    return correlations


def _rule(arms: list[list[float]]) -> int:
    ratios = {f"c{i}__arm{j}": v for j, values in enumerate(arms) for i, v in enumerate(values)}
    return repeated_sizing.calibration_replicates(
        study.STUDY.margins, ratios, study.PILOT_DRAWS, _flat(0.5)
    )


def test_the_sizing_scenarios_are_the_plans() -> None:
    """Plan 5B.3, at a within-arm error correlation of 0.5 as the plan's prior read it."""
    risk, cal = _shipped("paired_2k")
    trial_risk, trial_cal = _shipped("paired_trial_2k")
    assert _rule([[1.0] * 3, [1.0] * 4] * 5) == study.BUDGETED_R == 4_800
    assert _rule([risk, cal] * 5) == 9_200
    assert _rule([trial_risk, trial_cal] * 5) == study.PLANNED_R == 13_200
    alternating = [risk, cal, trial_risk, trial_cal] * 2 + [risk, cal]
    assert _rule(alternating) == 9_200


def test_the_family_passes_at_the_floor_with_correct_intervals() -> None:
    margins = study.STUDY.margins
    single = float(repeated_sizing.calibration_pass(margins, study.BUDGETED_R, np.array(1.0)))
    assert single ** sum(numbers()["calibration_arms"].values()) > 0.95


def test_the_controls_detect_at_the_floor() -> None:
    margins = study.STUDY.margins
    detect = repeated_sizing.calibration_control_probability
    assert detect(margins, study.BUDGETED_R, study.SHRUNKEN_FACTOR) > 0.9999
    # Plan 5B.2: the projection's clustered row-level ratios, and the precondition ceiling.
    assert detect(margins, 500, 0.785) == pytest.approx(0.9994, abs=1e-4)
    assert detect(margins, 500, 0.772) == pytest.approx(0.9999, abs=1e-4)
    assert detect(margins, 500, 0.85) == pytest.approx(0.65, abs=0.01)
    assert detect(margins, 2_400, 0.90) == pytest.approx(0.39, abs=0.01)
    assert detect(margins, study.BUDGETED_R, study.ROW_VARIANCE_CEILING) > 0.999


def test_the_primary_is_the_weighted_paired_row_at_2400() -> None:
    assert study.STUDY.scenarios == {"assess_weighted_2k": ("risk_difference[S - SB]",)}
    group = study.GROUP_BY_LABEL[study.PRIMARY_GROUP]
    summary = pd.DataFrame(
        {
            "cell": [study.calibration_cell(group, group.risk_arm, "paired")],
            "covered": [int(0.95 * pilot.REPLICATES)],
            "replicates": [pilot.REPLICATES],
        }
    )
    assert pilot.primary_replicates(summary) == study.PRIMARY_REPLICATES == 2_400
    assert study.PRIMARY_COVERAGE_FLOOR == 0.90


# --------------------------------------------------------------------- the fits (witnesses)


@cache
def _fitted(label: str) -> Any:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(label, group.n, 3)
    return frame


@pytest.mark.parametrize("label", [group.label for group in study.GROUPS])
def test_each_group_reads_exactly_its_own_composition(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    """The fit is experimental for its code alone: listing that code lifts it, another does not."""
    group = study.GROUP_BY_LABEL[label]
    frame = _fitted(label)
    assert study.assess(group, frame, 0).status == "experimental"
    key = ("difference", "binary", "estimated")
    others = {code for g in study.GROUPS for code in (g.code,)} - {group.code}
    monkeypatch.setattr(ec, "SCORE_COMPOSITIONS_MEASURED", {(key, code): "s: g" for code in others})
    assert study.assess(group, frame, 0).status == "experimental"
    monkeypatch.setattr(ec, "SCORE_COMPOSITIONS_MEASURED", {(key, group.code): "s: g"})
    lifted = study.assess(group, frame, 0)
    assert lifted.status == "supported", lifted.status_reason


def test_the_row_variance_control_reads_the_row_level_se() -> None:
    group = study.GROUP_BY_LABEL["assess_clustered_4k"]
    fitted = study.fit_group(group, _fitted(group.label), random_state=0)
    for row in group.controls["row_variance"]:
        estimate = fitted["estimates"][row]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        row_level = float(np.std(curve, ddof=1) / np.sqrt(len(curve)))
        assert fitted["controls"]["row_variance"][row] == pytest.approx(row_level)
        # The cluster sum is wider on this law: the control is below the reported SE.
        assert row_level < 0.9 * float(estimate.std_error)


# --------------------------------------------------------------------------- the summary


def synthetic_rows(seed: int = 0, replicates: int = 400) -> pd.DataFrame:
    """Rows of every declared cell, drawn at a correct interval; controls at a shrunken SE."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            control = cell.endswith("_control")
            se = 0.7 if control else 1.0
            estimate = rng.normal(0.0, 1.0, replicates)
            covered = np.abs(estimate) <= 1.96 * se
            rows += [
                {
                    "property": family,
                    "cell": cell,
                    "role": "control" if control else "positive",
                    "replicate": r,
                    "n": 4_000,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": float(estimate[r]),
                    "std_error": se,
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
    assert set(summary["cell"]) == set(study.PROPERTY_CELLS[study.CALIBRATION])
    controls = summary.loc[summary["role"] == "control"]
    assert len(controls) == 3
    assert controls["passed"].all()


# ------------------------------------------------------------------- streams and the pilot


def test_each_stream_has_its_own_samples_and_folds_per_group() -> None:
    groups = [group.label for group in study.GROUPS]
    seen: set[int] = set()
    for stream in properties.STREAMS:
        samples = {properties.sample_seed(g, r, stream) for g in groups for r in range(100)}
        folds = {properties.fold_seed(g, r, stream) for g in groups for r in range(100)}
        assert len(samples) == len(folds) == 100 * len(groups)
        assert not (samples | folds) & seen
        seen |= samples | folds


def _pilot_frame(
    ratios: Any, se: float = 0.01, controls: float = 0.79, replicates: int = pilot.REPLICATES
) -> pd.DataFrame:
    cells = list(pilot.positive_cells())
    control_cells = [c for c in study.PROPERTY_CELLS[study.CALIBRATION] if c.endswith("_control")]
    ratio = np.broadcast_to(np.asarray(ratios, dtype=float), (len(cells),))
    return pd.DataFrame(
        {
            "cell": cells + control_cells,
            "group": [pilot.group_of(c) for c in cells + control_cells],
            "arm": [pilot.arm_of(c) for c in cells + control_cells],
            "role": ["positive"] * len(cells) + ["control"] * len(control_cells),
            "replicates": replicates,
            "se_ratio": [*ratio, *([controls] * len(control_cells))],
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
    assert pilot.row_variance_failures(summary) == []


def test_the_point_reading_fallback_runs_where_x32a_cannot_and_refuses_suspects() -> None:
    frame = _pilot_frame(_alternating(0.025), replicates=BOTH)
    with pytest.raises(ValueError, match="calibration rule"):
        pilot.x32a_replicates(frame, NO_PAIRS)
    replicates, rule = pilot.declared_replicates(frame, NO_PAIRS)
    assert rule == "point_reading"
    positives = frame.loc[frame["role"] == "positive"]
    ratios = dict(zip(positives["cell"], positives["se_ratio"], strict=True))
    assert point_reading.point_reading_pass(study.STUDY.margins, ratios, replicates) >= 0.95
    with pytest.raises(ValueError, match="outside"):
        pilot.declared_replicates(
            _pilot_frame(_alternating(0.04), replicates=BOTH),
            NO_PAIRS,
            diagnosed=pilot.positive_cells(),
        )
    tight = _pilot_frame(_alternating(0.025), se=0.002, replicates=BOTH)
    with pytest.raises(ValueError, match="no diagnosis"):
        pilot.declared_replicates(tight, NO_PAIRS)
    assert pilot.declared_replicates(tight, NO_PAIRS, diagnosed=pilot.positive_cells())[1] == (
        "point_reading"
    )


def test_pilot_2_draws_each_arm_whose_plug_in_pass_is_low() -> None:
    assert pilot.pilot_2_arms(_pilot_frame(1.0)) == ()
    ratios = np.ones(len(pilot.positive_cells()))
    first = next(i for i, c in enumerate(pilot.positive_cells()) if pilot.arm_of(c) == "__k3_cal")
    ratios[first] = 1.06
    assert pilot.pilot_2_arms(_pilot_frame(ratios)) == ("__k3_cal",)


def test_the_row_variance_precondition_reads_the_ceiling() -> None:
    assert pilot.row_variance_failures(_pilot_frame(1.0, controls=0.79)) == []
    failures = pilot.row_variance_failures(_pilot_frame(1.0, controls=0.86))
    assert len(failures) == 2
    assert all("row-level SE ratio 0.860" in failure for failure in failures)


# ---------------------------------------------------------------------- the cost and files


def test_the_smoke_cost_stays_inside_the_cost_gate() -> None:
    timings = pd.read_csv(study.SMOKE_TIMINGS)
    assert set(timings["group"]) == {group.label for group in study.GROUPS}
    for row in timings.itertuples(index=False):
        declared = study.GROUP_BY_LABEL[row.group].seconds
        assert row.seconds_per_fit <= study.COST_GATE * declared, row.group
    assert numbers()["cpu_hours"] == pytest.approx(study.run_cpu_hours())


def test_the_declaration_names_every_file_a_run_writes() -> None:
    from tests.canonical.declared_run import RECORD_FILES
    from tests.canonical.regenerate import ARTIFACT_NAMES, PROPERTY_FAILURE_FILES

    assert set(study.RUN_FILES) == {*ARTIFACT_NAMES, *RECORD_FILES, *study.STUDY.property_artifacts}
    assert set(study.CONDITIONAL_RUN_FILES) == set(PROPERTY_FAILURE_FILES)


def test_the_fallback_answers_only_the_no_budget_case_after_pilot_2() -> None:
    """Review S6: another sizing error stops the declaration, and so do missing extensions."""
    with pytest.raises(ValueError, match="only after pilot_2"):
        pilot.declared_replicates(_pilot_frame(_alternating(0.025)), NO_PAIRS)
    nan = _pilot_frame(_alternating(0.025), replicates=BOTH)
    cells = list(pilot.positive_cells())
    pairs = pd.DataFrame([{"first": cells[0], "second": cells[1], "correlation": float("nan")}])
    with pytest.raises(ValueError) as caught:
        pilot.declared_replicates(nan, pairs)
    assert pilot.NO_BUDGET not in str(caught.value)
    assert "pilot_2" not in str(caught.value)
