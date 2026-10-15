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
from scipy.stats import norm

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


def _arm_spread(family: str, cell: str, n: int) -> float:
    """The spread of a cell's estimate: the pilot's once recorded, else the law's own."""
    if study.PILOT_STATE == "recorded":
        return properties.PROPERTY_PILOT_SPREADS[f"{family}/{cell}"]
    name = cell.split("__", 1)[0]
    return properties.EFFICIENCY_SD[name] / np.sqrt(n)


def _joint_control_probability() -> float:
    """P(the pointwise joint control's 99% interval lies below the nominal rate).

    The control's exact limit is the joint coverage of fifteen pointwise intervals under the
    law's correlation of the influence curves (plan 5.4).  A fixed-seed draw of the normal
    vector gives that limit, and the interval is the Wald interval of the count.
    """
    correlation = properties.label_correlations(study.LABELS)
    draws = np.random.default_rng(0).multivariate_normal(
        np.zeros(len(study.LABELS)), correlation, size=200_000
    )
    critical = float(norm.ppf(1.0 - properties.MARGINS.alpha / 2.0))
    limit = float(np.mean(np.all(np.abs(draws) <= critical, axis=1)))
    replicates = properties.CALIBRATION_REPLICATES
    z = float(norm.ppf(0.5 + properties.MARGINS.confidence_level / 2.0))
    se = np.sqrt(limit * (1.0 - limit) / replicates)
    nominal = 1.0 - properties.MARGINS.alpha
    return float(norm.cdf((nominal - limit) / se - z))


def _discrimination_probabilities() -> dict[str, float]:
    """The pass probability of every control that must miss its truth, at the declared sizes."""
    margins = properties.MARGINS
    out: dict[str, float] = {}
    controls = (
        ("single_robustness", properties.ONE_STEP_LIMIT, study.SINGLE_ROBUSTNESS_ARMS[1]),
        ("repeated_outcome_necessity", properties.ABSORBING_LIMIT, study.ABSORBING_ARMS[1]),
        ("outcome_history_necessity", properties.HISTORY_FREE_LIMIT, study.HISTORY_FREE_ARMS[1]),
    )
    replicates = {
        "single_robustness": properties.SINGLE_ROBUSTNESS_REPLICATES,
        "repeated_outcome_necessity": properties.ABSORBING_REPLICATES,
        "outcome_history_necessity": properties.HISTORY_FREE_REPLICATES,
    }
    for family, limits, arm in controls:
        for name, limit in limits.items():
            cell = f"{name}__{arm}"
            spread = _arm_spread(family, cell, properties.family_n(family))
            gap = (limit - study.TRUTH[study.label_estimand(name)]) / spread
            out[f"{family}/{cell}"] = sizing.bias_discriminated_probability(
                margins, gap, replicates[family]
            )
    for name in study.LABELS:
        cell = f"{name}__in_sample"
        ratio = properties.CALIBRATION_PILOT_RATIOS[cell]
        for control, factor in (
            ("shrunken_se_control", properties.SHRUNKEN_SE_FACTOR),
            ("noise_control", 2.0**-0.5),
        ):
            out[f"interval_calibration/{name}__{control}"] = sizing.calibration_control_probability(
                margins, properties.CALIBRATION_REPLICATES, factor * ratio
            )
    out["simultaneous_coverage/pointwise_joint_control"] = _joint_control_probability()
    return out


def test_every_control_is_discriminated_at_the_declared_sizes() -> None:
    """The union bound gives each of 55 checks the pass probability 1 - 0.001 / 55.

    The law gaps stand in for the pilot spreads while ``PILOT_STATE`` is "placeholder".  The
    test reads the pilot spreads and ratios from the recorded tables once the state changes.
    """
    probabilities = _discrimination_probabilities()
    floor = sizing.discrimination_floor(55)
    assert floor == pytest.approx(1.0 - 0.001 / 55)
    assert len(probabilities) == 4 + 6 + 2 + 30 + 1
    assert len(probabilities) <= 55
    below = {key: value for key, value in probabilities.items() if value < floor}
    assert not below, below
    assert sizing.family_share(3) == pytest.approx(0.994556, abs=1e-6)


def test_the_discrimination_check_fails_a_control_that_sits_on_the_truth() -> None:
    """A mutation control: a gap of zero must not clear the floor, or the check is vacuous."""
    floor = sizing.discrimination_floor(55)
    probability = sizing.bias_discriminated_probability(
        properties.MARGINS, 0.0, properties.SINGLE_ROBUSTNESS_REPLICATES
    )
    assert probability < floor
    assert (
        sizing.calibration_control_probability(
            properties.MARGINS, properties.CALIBRATION_REPLICATES, 1.0
        )
        < floor
    )


def test_each_pilot_table_names_its_size_and_its_replicates() -> None:
    """A recorded table must come from the declared pilot, at the current declared sizes."""
    assert properties.PRIMARY_PILOT_N > 0
    assert set(properties.CALIBRATION_PILOT_N) == set(properties.PILOT_UNITS["P5"])
    assert set(properties.PROPERTY_PILOT_N) == set(properties.PILOT_UNITS["P6"])
    if study.PILOT_STATE != "recorded":
        assert study.PILOT_STATE == "placeholder"
        return
    assert properties.PRIMARY_PILOT_REPLICATES == 2_000
    assert properties.CALIBRATION_PILOT_REPLICATES == 8_000
    assert properties.PROPERTY_PILOT_REPLICATES == 2_000
    for pilot_name, recorded in (
        ("P4", dict.fromkeys(properties.PILOT_UNITS["P4"], properties.PRIMARY_PILOT_N)),
        ("P5", properties.CALIBRATION_PILOT_N),
        ("P6", properties.PROPERTY_PILOT_N),
    ):
        for unit, size in recorded.items():
            assert size == properties.pilot_declared_n(pilot_name, unit)
            assert size == properties.pilot_recorded_n(pilot_name, unit)


def test_the_pins_cover_every_cell_the_pilots_report() -> None:
    primary = {f"{s}/{n}" for s in study.SCENARIOS for n in study.LABELS}
    assert set(properties.PRIMARY_PILOT_RATIOS) == primary == set(properties.PRIMARY_PILOT_COVERED)
    cells = set(properties.CALIBRATION_CELLS)
    assert set(properties.CALIBRATION_PILOT_COVERED) == cells
    assert set(properties.CALIBRATION_PILOT_SPREADS) == cells
    declared = {key for names in properties.property_cells().values() for key in names}
    for table in (
        properties.PROPERTY_PILOT_BIASES,
        properties.PROPERTY_PILOT_RATIOS,
        properties.PROPERTY_PILOT_COVERED,
        properties.PROPERTY_PILOT_SPREADS,
    ):
        assert set(table) == declared


def test_the_c6_arm_has_its_own_declared_size() -> None:
    assert properties.CONTINUOUS_N == properties.CALIBRATION_N == 8_000
    assert properties.fit_n("interval_calibration", "continuous") == properties.CONTINUOUS_N
    assert properties.fit_n("interval_calibration", "crossfit") == properties.CALIBRATION_N
    payload_sizes = {
        configuration: payload[0][5]
        for payload in properties._payloads()
        for family, configuration in [(payload[0][0], payload[0][1])]
        if family == "interval_calibration"
    }
    assert payload_sizes["continuous"] == properties.CONTINUOUS_N


def test_doubling_the_c6_size_leaves_the_other_arms_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(properties, "CONTINUOUS_N", 16_000)
    assert properties.fit_n("interval_calibration", "continuous") == 16_000
    assert properties.fit_n("interval_calibration", "in_sample") == 8_000
    assert properties.pilot_declared_n("P5", "continuous") == 16_000
    assert properties.pilot_declared_n("P5", "crossfit") == 8_000
    assert properties.fit_n("booster_calibration", "crossfit_boosted") == 8_000


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


@pytest.fixture
def small_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare a small size everywhere, so a pilot unit fits in a fraction of a second."""
    for name in (
        "PRIMARY_N",
        "CALIBRATION_N",
        "CONTINUOUS_N",
        "ABSORBING_N",
        "HISTORY_FREE_N",
        "N",
    ):
        monkeypatch.setattr(properties, name, SMALL_N)


def test_the_pilot_task_lists_cover_every_unit() -> None:
    sizes = {unit: properties.pilot_declared_n("P6", unit) for unit in properties.PILOT_UNITS["P6"]}
    tasks = pilot.task_list("P6", 3, sizes)
    assert len(tasks) == 3 * 3
    assert {task[2] for task in tasks} == {
        properties.N,
        properties.ABSORBING_N,
        properties.HISTORY_FREE_N,
    }
    assert pilot.resolve_n("P5", ["in_sample", "continuous"], None, {}) == {
        "in_sample": properties.CALIBRATION_N,
        "continuous": properties.CONTINUOUS_N,
    }
    assert pilot.resolve_n("P5", ["in_sample", "continuous"], 100, {"continuous": 200}) == {
        "in_sample": 100,
        "continuous": 200,
    }


@pytest.mark.usefixtures("small_declared")
def test_the_pilot_driver_writes_parts_records_the_plan_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "p4"
    written = pilot.run("P4", out, jobs=1, chunk=2, replicates=2)
    assert written == 2
    assert len(list(out.glob("part-*.json"))) == 2
    assert not list(out.glob("*.tmp"))
    assert capsys.readouterr().out.count("[P4] part") == 2
    plan = json.loads((out / "plan.json").read_text())
    assert plan["resolved_n"] == dict.fromkeys(study.SCENARIOS, SMALL_N)
    assert set(plan["modules"]) == set(pilot.study_files())
    assert plan["head"]
    assert pilot.run("P4", out, jobs=1, chunk=2, replicates=2) == 0
    assert "skipped" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pilot.run("P4", out, jobs=1, chunk=2, replicates=3)
    with pytest.raises(SystemExit):
        pilot.run("P4", out, jobs=1, chunk=2, replicates=2, n=SMALL_N + 1)
    _, by_unit = pilot.load_parts("P4", out)
    assert {unit: len(results) for unit, results in by_unit.items()} == dict.fromkeys(
        study.SCENARIOS, 2
    )
    row = by_unit["repeated_folds_1"][0][0]
    assert row["n"] == SMALL_N


def test_a_part_that_exists_is_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "part-00000000.json"
    assert pilot._write_once(path, [1])
    assert not pilot._write_once(path, [2])
    assert json.loads(path.read_text()) == [1]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.usefixtures("small_declared")
def test_the_pilot_driver_runs_the_tail_down_to_an_index(tmp_path: Path) -> None:
    out = tmp_path / "p6"
    pilot.run("P6", out, jobs=1, chunk=1, replicates=1, start=2, descending=True)
    assert [p.name for p in sorted(out.glob("part-*.json"))] == ["part-00000002.json"]
    with pytest.raises(SystemExit, match="missing"):
        pilot.load_parts("P6", out)


@pytest.mark.usefixtures("small_declared")
def test_a_unit_drawn_at_an_old_size_is_refused_until_it_is_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The N2 path: the n rule doubles HISTORY_FREE_N, only N2 is piloted again, and the
    assembly reads the old directory for the other two families."""
    first = tmp_path / "p6"
    pilot.run("P6", first, jobs=1, chunk=3, replicates=2)
    chosen, sizes = pilot.collect("P6", [first])
    assert sizes["outcome_history_necessity"] == SMALL_N
    monkeypatch.setattr(properties, "HISTORY_FREE_N", 2 * SMALL_N)
    with pytest.raises(SystemExit, match="no directory supplies"):
        pilot.collect("P6", [first])
    assert "skipped outcome_history_necessity" in capsys.readouterr().out
    second = tmp_path / "p6-n2"
    pilot.run("P6", second, jobs=1, chunk=3, replicates=2, units=["outcome_history_necessity"])
    chosen, sizes = pilot.collect("P6", [first, second])
    assert sizes == {
        "single_robustness": SMALL_N,
        "repeated_outcome_necessity": SMALL_N,
        "outcome_history_necessity": 2 * SMALL_N,
    }
    rows = pilot._flatten("P6", [r for unit in properties.PILOT_UNITS["P6"] for r in chosen[unit]])
    table = properties.property_cell_table(rows)
    assert set(table.index) == {k for names in properties.property_cells().values() for k in names}
    assert set(properties.pilot_row_sizes("P6", rows)["outcome_history_necessity"]) == {2 * SMALL_N}
    # The directory order does not matter for a unit that only one directory supplies.
    assert pilot.collect("P6", [second, first])[1] == sizes


@pytest.mark.usefixtures("small_declared")
def test_rows_at_the_wrong_size_stop_the_assembly(tmp_path: Path) -> None:
    out = tmp_path / "p4"
    pilot.run("P4", out, jobs=1, chunk=2, replicates=1)
    for part in out.glob("part-*.json"):
        results = json.loads(part.read_text())
        for result in results:
            for row in result:
                row["n"] = SMALL_N + 5
        part.write_text(json.dumps(results))
    with pytest.raises(SystemExit, match="the rows of"):
        pilot.collect("P4", [out])


@pytest.mark.usefixtures("small_declared")
def test_an_older_plan_is_stamped_once_and_only_without_an_n_override(tmp_path: Path) -> None:
    out = tmp_path / "old"
    pilot.run("P4", out, jobs=1, chunk=2, replicates=2)
    new_plan = json.loads((out / "plan.json").read_text())
    legacy = {"pilot": "P4", "replicates": 2, "n": None, "chunk": 2, "tasks": 4}
    (out / "plan.json").write_text(json.dumps(legacy))
    with pytest.raises(SystemExit, match="stamp"):
        pilot.load_parts("P4", out)
    stamped = pilot.stamp("P4", out)
    assert stamped["resolved_n"] == new_plan["resolved_n"]
    assert stamped["stamped_from_legacy"] is True
    with pytest.raises(SystemExit, match="already stamped"):
        pilot.stamp("P4", out)
    assert pilot.collect("P4", [out])[1] == new_plan["resolved_n"]
    (out / "plan.json").write_text(json.dumps({**legacy, "n": 500}))
    with pytest.raises(SystemExit, match="override"):
        pilot.stamp("P4", out)


@pytest.mark.usefixtures("small_declared")
def test_the_p5_arms_run_one_at_a_time(tmp_path: Path) -> None:
    out = tmp_path / "p5-continuous"
    pilot.run("P5", out, jobs=1, chunk=2, replicates=2, units=["continuous"])
    plan = json.loads((out / "plan.json").read_text())
    assert plan["units"] == ["continuous"]
    _, by_unit = pilot.load_parts("P5", out)
    assert by_unit["continuous"][0]["mass"]


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
                "n": 1_000,
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
    table = properties.primary_cell_table(rows)
    assert list(table.columns) == ["draws", "covered", "ratio", "bias", "spread"]
    assert (table["ratio"] > 0.5).all()
    assert properties.pilot_row_sizes("P4", rows) == {
        scenario: {1_000} for scenario in study.SCENARIOS
    }


def test_each_declared_n_is_its_base_times_the_recorded_doublings() -> None:
    """The n rules double a size at most twice, and the placeholder pilots fire none."""
    assert study.PRIMARY_N == study.PRIMARY_N_BASE * 2**study.PRIMARY_N_DOUBLINGS
    for name in ("CONTINUOUS", "ABSORBING", "HISTORY_FREE"):
        base = getattr(properties, f"{name}_N_BASE")
        doublings = getattr(properties, f"{name}_N_DOUBLINGS")
        assert getattr(properties, f"{name}_N") == base * 2**doublings
        assert doublings in (0, 1, 2)
    if study.PILOT_STATE == "placeholder":
        assert study.PRIMARY_N_DOUBLINGS == 0
        assert properties.CONTINUOUS_N_DOUBLINGS == 0
        assert properties.ABSORBING_N_DOUBLINGS == 0
        assert properties.HISTORY_FREE_N_DOUBLINGS == 0


def test_the_drift_check_names_a_study_file_that_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A recorded hash that differs from the tree prints a warning that names the file."""
    recorded = pilot.module_hashes()
    assert pilot.drifted_files(recorded) == []
    name = "tests/studies/repeated_sizing.py"
    assert name in recorded
    altered = {**recorded, name: "0" * 64, "src/cleverly/longitudinal/removed.py": "1" * 64}
    expected = sorted([name, "src/cleverly/longitudinal/removed.py"])
    assert pilot.drifted_files(altered) == expected
    plan = {"modules": altered, "head": pilot.git_head()}
    assert pilot.warn_on_drift(plan, tmp_path) == expected
    printed = capsys.readouterr().out
    assert f"differs: {name}" in printed
    assert pilot.warn_on_drift({"modules": recorded, "head": pilot.git_head()}, tmp_path) == []
    assert "WARNING" not in capsys.readouterr().out
