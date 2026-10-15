"""The design of the repeated-outcome IPSI study, checked before any run.

The pilots P4, P5 and P6 have not run.  The pilot tables in
:mod:`tests.studies.ipsi_repeated_properties` are placeholders, and these tests state that.
They assert that each declared size equals the declared rule applied to the placeholder tables,
that the declared run refuses to start, and that every control the study declares separates from
the truth by a nonzero amount.  The framework tests in ``tests/unit/test_method_evidence.py``
recompute committed verdicts when artifacts exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical.ipsi_repeated import pilot
from tests.studies import canonical_ipsi_repeated as study
from tests.studies import ipsi_repeated_properties as properties
from tests.studies import ltmle_repeated_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import replication
from tests.studies.evidence.registry import registered
from tests.studies.fractional_glm import QuasiBinomialGLM

pytestmark = pytest.mark.filterwarnings("ignore")

#: The fast sample size of a wiring check.
SMALL_N = 600


def _gap_in_se(limit: float, truth: float, name: str, n: int) -> float:
    """The distance of a limit from the truth in standard errors of the efficient curve."""
    return (limit - truth) / (properties.EFFICIENCY_SD[name] / np.sqrt(n))


# ---------------------------------------------------------------------------- the record


def test_the_record_declares_no_comparator() -> None:
    """The record is not in ``registered()`` yet: registration needs the committed artifacts."""
    assert study.STUDY.slug == "longitudinal-repeated-ipsi"
    assert study.STUDY.reference is None
    assert study.STUDY.pending_reference is None
    assert study.STUDY not in registered()


def test_the_study_seed_is_unique() -> None:
    seeds = [record.seed for record in registered()]
    assert study.STUDY.seed not in seeds


def test_the_study_declares_fifteen_estimands_in_both_scenarios() -> None:
    assert len(study.LABELS) == 15
    assert len(study.ESTIMANDS) == 15
    assert len(study.GATED_LABELS) == 5
    assert set(study.TRUTH) == set(study.ESTIMANDS)
    assert set(study.SCENARIOS) == {"repeated_folds_5", "repeated_folds_1"}
    assert study.SCENARIO_FOLDS == {"repeated_folds_5": 5, "repeated_folds_1": 1}


def test_the_study_record_declares_the_cells_each_family_emits() -> None:
    cells = study.STUDY.property_cells
    sizes = {family: len(names) for family, names in cells.items()}
    assert sizes == {
        "interval_calibration": 55,
        "booster_calibration": 5,
        "small_sample_calibration": 10,
        "simultaneous_coverage": 2,
        "single_robustness": 8,
        "repeated_outcome_necessity": 12,
        "outcome_history_necessity": 4,
    }


# ---------------------------------------------------------------------------- the truths


@pytest.mark.parametrize("delta", study.DELTAS)
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_three_independent_routes_give_the_same_truth(delta: float, horizon: int) -> None:
    counterfactual = law.BINARY.tilt_counterfactual(delta, horizon)
    functional = law.BINARY.tilt_functional(delta, horizon)
    limit = law.BINARY.ipsi_limit(delta, horizon)
    assert functional == pytest.approx(counterfactual, abs=1e-10)
    assert limit == pytest.approx(counterfactual, abs=1e-10)


def test_the_declared_truth_table_matches_the_law() -> None:
    for name in study.LABELS:
        delta, reference, horizon = study.label_parts(name)
        value = law.BINARY.tilt_counterfactual(delta, horizon)
        if reference is not None:
            value -= law.BINARY.tilt_counterfactual(reference, horizon)
        assert study.TRUTH[study.label_estimand(name)] == pytest.approx(value, abs=1e-12)


def test_the_tilts_differ_so_a_wrong_tilt_cannot_pass() -> None:
    for horizon in (1, 2, 3):
        values = [law.BINARY.tilt_counterfactual(delta, horizon) for delta in study.DELTAS]
        assert values[0] > values[1] > values[2]
        assert min(values[0] - values[1], values[1] - values[2]) > 0.01 * horizon


# ---------------------------------------------------------------------------- the controls


def test_single_robustness_limits_are_nonzero_at_the_declared_size() -> None:
    gaps = {
        name: _gap_in_se(
            properties.ONE_STEP_LIMIT[name],
            study.TRUTH[study.label_estimand(name)],
            name,
            properties.N,
        )
        for name in study.SINGLE_ROBUSTNESS_LABELS
    }
    means = [gap for name, gap in gaps.items() if study.label_parts(name)[1] is None]
    contrasts = [gap for name, gap in gaps.items() if study.label_parts(name)[1] is not None]
    # The plug-in without the correction falls below the truth at the tilt 0.5 and above it
    # at the tilt 2: a sign error in the correction cannot hide behind a one-sided gap.
    assert min(means) < -1.9 < 1.9 < max(means)
    assert min(contrasts) < -4.0 < 4.0 < max(contrasts)


def test_the_one_step_limit_is_the_plug_in_of_the_wrong_regression() -> None:
    for name, value in properties.ONE_STEP_LIMIT.items():
        assert value == pytest.approx(
            properties._plan_value(law.BINARY, name, "ipsi_plug_in_limit"), abs=1e-12
        )
        # The one-step limit with the wrong regression and the right mechanisms is the truth.
        delta, reference, horizon = study.label_parts(name)
        exact = law.BINARY.ipsi_limit(delta, horizon, m_kind="baseline")
        if reference is not None:
            exact -= law.BINARY.ipsi_limit(reference, horizon, m_kind="baseline")
        assert exact == pytest.approx(study.TRUTH[study.label_estimand(name)], abs=1e-10)


def test_the_absorbing_reading_separates_from_the_truth_at_the_declared_size() -> None:
    gaps = [
        abs(
            _gap_in_se(
                properties.ABSORBING_LIMIT[name],
                study.TRUTH[study.label_estimand(name)],
                name,
                properties.ABSORBING_N,
            )
        )
        for name in study.ABSORBING_LABELS
    ]
    assert min(gaps) > 20.0
    assert max(gaps) > 30.0


def test_the_history_free_limit_is_a_nonzero_distance_from_the_truth_on_both_sides() -> None:
    gaps = [
        _gap_in_se(
            properties.HISTORY_FREE_LIMIT[name],
            study.TRUTH[study.label_estimand(name)],
            name,
            properties.HISTORY_FREE_N,
        )
        for name in study.HISTORY_FREE_LABELS
    ]
    assert min(gaps) < -1.7
    assert max(gaps) > 0.8
    for name in study.HISTORY_FREE_LABELS:
        delta, _, horizon = study.label_parts(name)
        assert properties.HISTORY_FREE_LIMIT[name] == pytest.approx(
            law.BINARY.ipsi_limit(delta, horizon, m_kind="free", pi_kind="free"), abs=1e-10
        )


def test_the_longhand_plug_in_equals_the_law_on_the_weighted_paths() -> None:
    paths = law.BINARY.paths
    weights = paths["p"].to_numpy()
    for delta in (0.5, 2.0):
        longhand = properties.tilt_plug_ins(paths, (delta,), 3, weights=weights)[delta]
        assert longhand == pytest.approx(
            law.BINARY.ipsi_plug_in_limit(delta, 3, m_kind="baseline"), abs=1e-9
        )
        # Nonzero witness: the plug-in is not the truth, so a function that returned the
        # truth would fail the line above.
        assert abs(longhand - law.BINARY.tilt_counterfactual(delta, 3)) > 0.01


def test_the_fallback_mass_is_a_share_and_is_nonzero_on_the_continuous_arm() -> None:
    frame = properties.sample("continuous", seed=1, n=2_000)
    mass = properties.fallback_mass(frame, 2.0)
    assert 0.0 < mass < 1.0


def test_the_baseline_and_arm_learner_reads_the_first_and_last_column_only() -> None:
    rng = np.random.default_rng(0)
    design = rng.normal(size=(300, 4))
    design[:, -1] = rng.integers(0, 2, 300)
    y = (design[:, 0] + design[:, -1] + rng.normal(size=300) > 0.5).astype(float)
    learner = properties.BaselineAndArm(QuasiBinomialGLM()).fit(design, y)
    shifted = design.copy()
    shifted[:, 1:3] += 10.0
    assert np.allclose(learner.predict(design), learner.predict(shifted))
    # Nonzero witness: the two kept columns do move the prediction.
    moved = design.copy()
    moved[:, 0] += 1.0
    assert not np.allclose(learner.predict(design), learner.predict(moved))


def test_the_reported_arms_draw_the_declared_replicates() -> None:
    assert properties.BOOSTER_REPLICATES == properties.SMALL_SAMPLE_REPLICATES == 2_000


@pytest.mark.parametrize("horizon", [3])
def test_the_control_fits_report_the_declared_estimand_names(horizon: int) -> None:
    frame = properties.sample("in_sample", seed=5, n=SMALL_N)
    absorbing = properties.absorbing_fit(frame, horizon)
    history_free = properties.history_free_fit(frame, horizon)
    for result, labels in (
        (absorbing, study.ABSORBING_LABELS),
        (history_free, study.HISTORY_FREE_LABELS),
    ):
        for name in labels:
            fitted = result[properties._end_of_study_name(name)]
            assert np.isfinite(fitted.psi)
            assert fitted.std_error > 0.0


# ---------------------------------------------------------------------------- the sizes


def test_the_pilot_tables_are_placeholders() -> None:
    """The tables are uniform planning values.  No pilot has run."""
    assert study.PILOT_STATE == "placeholder"
    assert set(properties.PRIMARY_PILOT_COVERED.values()) == {1_900}
    assert set(properties.PRIMARY_PILOT_BIASES.values()) == {0.0}
    assert set(properties.CALIBRATION_PILOT_RATIOS.values()) == {1.0}
    assert set(properties.CALIBRATION_PILOT_BIASES.values()) == {0.0}
    assert set(properties.PROPERTY_PILOT_BIASES.values()) == {0.0}
    assert properties.CALIBRATION_PILOT_JOINT_COVERAGE == 0.95
    assert properties.C6_FALLBACK_MASS == {}


def test_each_declared_size_is_the_rule_applied_to_the_placeholder_pilots() -> None:
    derived = properties.declared_sizes()
    assert derived["primary_replicates"] == study.PRIMARY_REPLICATES
    assert derived["calibration_replicates"] == properties.CALIBRATION_REPLICATES
    assert derived["calibration_budget_met"] == properties.CALIBRATION_BUDGET_MET
    assert derived["single_robustness"] == properties.SINGLE_ROBUSTNESS_REPLICATES
    assert derived["repeated_outcome_necessity"] == properties.ABSORBING_REPLICATES
    assert derived["outcome_history_necessity"] == properties.HISTORY_FREE_REPLICATES
    # No rule fires on placeholder tables, so no size has doubled.
    assert not any(
        derived[key] for key in ("primary_n_rule", "c6_n_rule", "n1_n_rule", "n2_n_rule")
    )
    assert study.PRIMARY_N == 8_000
    assert properties.CALIBRATION_N == 8_000
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES
    assert study.STUDY.n == study.PRIMARY_N


def test_the_property_sizes_cover_the_three_control_families() -> None:
    sizes = properties.property_replicates(
        properties.PROPERTY_PILOT_BIASES, properties.PROPERTY_PILOT_REPLICATES
    )
    assert sizes == {
        "single_robustness": properties.SINGLE_ROBUSTNESS_REPLICATES,
        "repeated_outcome_necessity": properties.ABSORBING_REPLICATES,
        "outcome_history_necessity": properties.HISTORY_FREE_REPLICATES,
    }


def test_the_pilot_rules_applied_to_the_placeholder_tables_give_the_declared_sizes() -> None:
    primary = (
        properties.PRIMARY_PILOT_COVERED,
        properties.PRIMARY_PILOT_BIASES,
        properties.PRIMARY_PILOT_REPLICATES,
    )
    verdicts = properties.apply_pilot_rules(
        primary=primary,
        calibration={
            "ratios": properties.CALIBRATION_PILOT_RATIOS,
            "biases": properties.CALIBRATION_PILOT_BIASES,
            "replicates": properties.CALIBRATION_PILOT_REPLICATES,
            "joint_coverage": properties.CALIBRATION_PILOT_JOINT_COVERAGE,
        },
        properties=(properties.PROPERTY_PILOT_BIASES, properties.PROPERTY_PILOT_REPLICATES),
    )
    assert verdicts["primary_replicates"] == study.PRIMARY_REPLICATES
    assert verdicts["calibration_replicates"] == properties.CALIBRATION_REPLICATES
    assert verdicts["property_replicates"]["single_robustness"] == 600
    assert not verdicts["primary_n_rule_fires"]
    assert not verdicts["c6_n_rule_fires"]


def test_the_discrimination_checks_hold_their_share() -> None:
    k = 55
    assert k == 8 + 12 + 4 + 30 + 1
    assert sizing.family_share(3) == pytest.approx(0.994556, abs=1e-6)


def test_the_declared_primary_run_is_refused_while_the_pilots_are_placeholders() -> None:
    with pytest.raises(study.PlaceholderPilotsError, match="placeholder pilot tables"):
        study.draw_and_fit(replicates=study.PRIMARY_REPLICATES, n=study.PRIMARY_N, n_jobs=1)


def test_the_declared_property_run_is_refused_while_the_pilots_are_placeholders() -> None:
    with pytest.raises(study.PlaceholderPilotsError, match="placeholder pilot tables"):
        properties.generate_property_rows(n_jobs=1)


# ---------------------------------------------------------------------------- the drivers


def test_draw_and_fit_returns_a_phase_result_with_every_estimand() -> None:
    result = study.draw_and_fit(replicates=1, n=SMALL_N, n_jobs=1)
    assert isinstance(result, replication.PhaseResult)
    assert result.samples is None
    rows = result.rows
    assert set(rows["scenario"]) == set(study.SCENARIOS)
    for scenario in study.SCENARIOS:
        assert set(rows.loc[rows["scenario"] == scenario, "estimand"]) == set(study.ESTIMANDS)


def test_the_fit_sets_emit_exactly_the_declared_cells() -> None:
    outcomes = [
        properties._replication((family, configuration, replicate, 2, 11 + replicate, SMALL_N))
        for family, configuration, _ in properties.FIT_SETS
        for replicate in range(2)
    ]
    rows = properties.assemble_property_rows(outcomes)
    emitted: dict[str, set[str]] = {}
    for family, cell in set(zip(rows["property"], rows["cell"], strict=True)):
        emitted.setdefault(family, set()).add(cell)
    assert {family: set(cells) for family, cells in study.STUDY.property_cells.items()} == emitted


# ---------------------------------------------------------------------------- the pilot driver


def test_the_pilot_task_lists_cover_every_cell() -> None:
    assert len(pilot.task_list("P4", 3, None)) == 2 * 3
    assert len(pilot.task_list("P5", 3, None)) == 3 * 3
    assert len(pilot.task_list("P6", 3, None)) == 3 * 3
    assert {task[2] for task in pilot.task_list("P6", 1, None)} == {
        properties.N,
        properties.ABSORBING_N,
        properties.HISTORY_FREE_N,
    }


def test_the_pilot_driver_writes_parts_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "p4"
    written = pilot.run("P4", out, jobs=1, chunk=2, replicates=2, n=SMALL_N)
    assert written == 2
    assert len(list(out.glob("part-*.json"))) == 2
    first = capsys.readouterr().out
    assert first.count("[P4] part") == 2
    again = pilot.run("P4", out, jobs=1, chunk=2, replicates=2, n=SMALL_N)
    assert again == 0
    assert "skipped" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pilot.run("P4", out, jobs=1, chunk=2, replicates=3, n=SMALL_N)
    parts = pilot.load_parts("P4", out)
    assert len(parts) == 4
    assert json.loads((out / "plan.json").read_text())["pilot"] == "P4"


def test_the_pilot_driver_runs_the_tail_down_to_an_index(tmp_path: Path) -> None:
    out = tmp_path / "p6"
    pilot.run("P6", out, jobs=1, chunk=1, replicates=1, n=SMALL_N, start=2, descending=True)
    assert [p.name for p in sorted(out.glob("part-*.json"))] == ["part-00000002.json"]
    with pytest.raises(SystemExit, match="missing"):
        pilot.load_parts("P6", out)


def test_the_pilot_summaries_read_synthetic_rows() -> None:
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "replicate": r,
                "label": name,
                "estimate": study.TRUTH[study.label_estimand(name)] + rng.normal(0, 0.01),
                "std_error": 0.01,
                "covered": int(rng.random() < 0.95),
            }
            for scenario in study.SCENARIOS
            for name in study.LABELS
            for r in range(40)
        ]
    )
    covered, biases, replicates = properties.summarize_primary_pilot(rows)
    assert replicates == 40
    assert set(covered) == {f"{s}/{n}" for s in study.SCENARIOS for n in study.LABELS}
    assert max(abs(value) for value in biases.values()) > 0.0
