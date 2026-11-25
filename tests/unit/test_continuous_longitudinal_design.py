"""The declaration of continuous-time-longitudinal (X6 plan rev 7, section 5.7; X6 part (e), E2)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical.continuous_time_longitudinal import regenerate
from tests.studies import continuous_time_longitudinal as study
from tests.studies import continuous_time_longitudinal_law as laws
from tests.studies import continuous_time_longitudinal_pilot as pilot
from tests.studies import continuous_time_longitudinal_readings as readings
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import replicate_seed

TESTS = Path(__file__).resolve().parents[1]
PILOT = pilot.read()


def test_the_declaration_is_the_plans() -> None:
    record = study.STUDY
    assert record.replicates == study.PRIMARY_REPLICATES
    assert study.PRIMARY_REPLICATES in study.PRIMARY_RUNGS == (2_400, 3_200, 4_800)
    assert study.MECHANISM_REPLICATES in study.MECHANISM_RUNGS == (1_600, 2_400, 3_200)
    assert study.REPORTED_REPLICATES == 800 and study.PILOT_REPLICATES == 400
    assert min(study.PRIMARY_RUNGS) >= study.MECHANISM_REPLICATES
    assert record.n == study.PRIMARY_N == 1_000
    assert study.FAILURE_BOUND == 0.01 and study.SHRUNKEN_SE_FACTOR == 0.85
    assert study.MARGINS.standardized_bias == 0.2
    assert record.reference is None and record.property_cells == {}
    assert record.publication_policy == "gated" and record.reported_scenarios == frozenset()
    assert study.N_FOLDS == 10 and study.LEARNER_FOLDS == 5
    assert laws.TIMES == (10.0, 20.0, 30.0) and laws.END == 30.0
    assert (record.artifacts / "regenerate.py").exists()
    assert set(record.extra_artifacts) == {
        study.ROWS_FILE,
        study.SECONDARY_FILE,
        study.CALIBRATION_FILE,
        study.MECHANISM_FILE,
        study.MISSPECIFIED_FILE,
        study.PAIRING_FILE,
        study.READINGS_FILE,
    }


def test_the_rows_are_the_plans() -> None:
    assert len(study.DAY_ROWS) == 30 and len(set(study.DAY_ROWS)) == 30
    assert sum("rule" in name for name in study.DAY_ROWS) == 12
    assert len(study.CONT_ROWS) == 18 and set(study.CONT_ROWS) <= set(study.DAY_ROWS)
    assert len(study.ROBUST_ROWS) == 12
    assert all("@ t=10]" not in name for name in study.ROBUST_ROWS)
    assert len(study.DAY_ROWS) + len(study.CONT_ROWS) + len(study.ROBUST_ROWS) == 60
    assert len(study.PLUG_IN_ROWS) == 4 and set(study.PLUG_IN_ROWS) <= set(study.ROBUST_ROWS)
    assert all("@ t=30]" in name and "ate_" not in name for name in study.PLUG_IN_ROWS)
    assert study.CALIBRATION_ROW == "ate_regimen[always vs never, cause1 @ t=20]"
    assert study.SPARSE_ROW == "ate_regimen[always vs never, cause2 @ t=10]"
    assert all(study.CALIBRATION_ROW in study.ESTIMANDS[cell] for cell in study.CALIBRATION_CELLS)
    for _, count, names in study.SECONDARY.values():
        assert set(names) <= set(study.DAY_ROWS) and count <= min(study.PRIMARY_RUNGS)
    assert len(study.PAIRED_ROWS) == 18 and study.SECONDARY["ltmle_daily"][1] == 400
    assert study.GATED_SECONDARY == ("mechanism_correct",)


def test_the_seeds_are_used_nowhere_else_in_tests() -> None:
    mine = {str(study.SEED), str(study.RESAMPLING_SEED)}
    assert len(mine) == 2
    own = {Path(study.__file__).name, Path(__file__).name}
    for path in TESTS.rglob("*.py"):
        if path.name in own:
            continue
        numbers = set(re.findall(r"\d{7,}", path.read_text(encoding="utf-8").replace("_", "")))
        assert not mine & numbers, path


def test_the_streams_are_disjoint() -> None:
    streams: dict[str, list[int]] = {
        **{
            f"primary/{cell}": [
                replicate_seed(study.STUDY, cell, r) for r in range(max(study.PRIMARY_RUNGS))
            ]
            for cell in study.CELLS
        },
        **{
            f"secondary/{cell}": [
                study.secondary_seed(cell, r) for r in range(max(study.MECHANISM_RUNGS))
            ]
            for cell in study.SECONDARY
            if cell != "ltmle_daily"
        },
        **{
            f"pilot/{cell}": [pilot.pilot_seed(cell, r) for r in range(pilot.EXTENSION_DRAWS)]
            for cell in pilot.CELLS
        },
        **{
            f"truth/{law}/{regime}": pilot.truth_seeds(law, regime)
            for law in laws.SAMPLERS
            for regime in laws.REGIMES
        },
        "plugin_limit": [pilot.plugin_seed()],
        "feasibility": [pilot.feasibility_seed()],
    }
    seen: set[int] = set()
    for name, seeds in streams.items():
        assert len(set(seeds)) == len(seeds), name
        assert not set(seeds) & seen, name
        seen |= set(seeds)
    # The pairing reads day_cf's own draws, by design.
    assert study.secondary_seed("ltmle_daily", 3) == replicate_seed(study.STUDY, "day_cf", 3)


def _unit_rows(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [rows for _, rows in frame.groupby("id", sort=True)]


@pytest.mark.parametrize("law", ["ct", "ct_day"])
def test_a_draw_is_a_counting_process_frame(law: str) -> None:
    frame = laws.SAMPLERS[law](400, 7)
    assert frame["id"].nunique() == 400
    for rows in _unit_rows(frame)[:200]:
        starts, stops = rows["start"].to_numpy(), rows["stop"].to_numpy()
        assert starts[0] == 0 and np.all(starts[1:] == stops[:-1])
        assert (rows["event"].iloc[:-1] == 0).all() and stops[-1] <= laws.END
        assert (rows["W1"].nunique(), rows["W2"].nunique()) == (1, 1)
        if law == "ct_day":
            assert np.all(stops == np.round(stops))
            # Only a censoring on the day of a visit leaves a zero-length (last) row.
            zero = stops == starts
            assert not zero[:-1].any() and (not zero[-1] or rows["event"].iloc[-1] == 0)
        else:
            assert np.all(stops > starts)
    from cleverly.survival._longitudinal_data import ContinuousLongitudinalData

    data = ContinuousLongitudinalData.from_frame(frame, **laws.COLUMNS, causes=laws.CAUSES)
    assert data.dropped_visits == 0


def test_the_true_weights_reproduce_the_truth_and_a_dropped_censoring_weight_does_not() -> None:
    """A nonzero witness: the true-weight IPW mean equals the regime's risk, and the IPW with
    the censoring factor removed misses it, so a wrong weight cannot pass."""
    paths = laws.day_paths(laws.sample_day(300_000, 11))
    risks = laws.monte_carlo_risks("ct_day", "never", [12], chunk=300_000)
    for cause in laws.CAUSES:
        values = laws.ipw_contributions(paths, "never", cause, 30.0)
        se = float(np.std(values, ddof=1) / np.sqrt(values.size))
        truth, _ = risks[(cause, 30.0)]
        assert abs(values.mean() - truth) < 4.5 * np.hypot(se, np.sqrt(truth / 300_000))
    mutated = laws.DayPaths(
        paths.time,
        paths.event,
        paths.baseline_follow,
        paths.followed,
        paths.log_pi,
        np.zeros_like(paths.log_sc),
    )
    biased = laws.ipw_contributions(mutated, "never", 1, 30.0).mean()
    assert abs(biased - risks[(1, 30.0)][0]) > 0.02


def test_the_truths_are_the_monte_carlo_risks() -> None:
    table = laws.TRUTH_RISKS
    assert set(table) == {"ct", "ct_day"} and all(
        set(t) == set(laws.REGIMES) for t in table.values()
    )
    assert all(v > 0 for t in table.values() for r in t.values() for v in r.values())
    for law in table:
        for regime in laws.REGIMES:
            fresh = laws.monte_carlo_risks(law, regime, [31, 32], chunk=200_000)
            for key, (value, units) in fresh.items():
                se = np.sqrt(value * (1 - value) / units)
                assert abs(table[law][regime][key] - value) < 5.0 * se, (law, regime, key)
    assert set(laws.TRUTHS["ct_day"]) >= set(study.DAY_ROWS)
    assert set(laws.TRUTHS["ct"]) >= set(study.CONT_ROWS)
    if not PILOT.empty and (PILOT["kind"] == "truth").any():
        assert pilot.truth_table(PILOT) == laws.TRUTH_RISKS


def test_the_misspecified_learners_drop_l() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 7))
    X[:, 0] = rng.integers(0, 2, 400)
    time = rng.exponential(1.0, 400)
    event = rng.integers(0, 2, 400)
    model = laws.CoxWithout().fit(X, time, event)
    moved = X.copy()
    moved[:, laws.HAZARD_L] += 5.0
    grid = np.array([0.5, 1.0])
    assert np.array_equal(model.cumulative_hazard(X, grid), model.cumulative_hazard(moved, grid))
    y = (rng.random(400) < 0.5).astype(int)
    classifier = laws.OmitColumn(laws.main_logistic(), laws.TREATMENT_L).fit(X[:, 1:], y)
    shifted = X[:, 1:].copy()
    shifted[:, laws.TREATMENT_L] += 3.0
    assert np.array_equal(classifier.predict_proba(X[:, 1:]), classifier.predict_proba(shifted))
    sides = laws.CONFIGURATIONS
    assert set(sides) == {"default", "mechanism_correct", "outcome_correct", "both_wrong"}
    assert sides["default"]() == {}
    assert set(sides["mechanism_correct"]()) == {
        "hazard_library",
        "visit_library",
        "continuation_learner",
    }
    assert set(sides["outcome_correct"]()) == {"treatment_learner", "censoring_library"}
    assert set(sides["both_wrong"]()) == set(sides["mechanism_correct"]()) | set(
        sides["outcome_correct"]()
    )


def test_the_panel_ends_its_nodes_on_the_target_days() -> None:
    assert study.NODE_ENDS == (3, 6, 10, 13, 16, 20, 23, 26, 30)
    assert [study.x70_node(t) for t in laws.TIMES] == [3, 6, 9]
    frame = pd.DataFrame(
        {
            "id": [0, 0, 1, 2, 2],
            "start": [0.0, 4.0, 0.0, 0.0, 11.0],
            "stop": [4.0, 8.0, 12.0, 11.0, 11.0],
            "event": [0, 2, 0, 0, 0],
            "W1": [1.0, 1.0, 0.0, 0.0, 0.0],
            "W2": [3.0, 3.0, 5.0, 2.0, 2.0],
            "L": [0.0, 1.0, 1.0, 0.0, 1.0],
            "A": [1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    wide = study.panel(frame)
    # Unit 0: a visit on day 4 (node 2), death of cause 2 on day 8 (node 3).
    assert wide.loc[0, ["L2", "A2", "V2"]].tolist() == [0.0, 1.0, 0.0]
    assert wide.loc[0, ["L3", "A3", "V3", "VA3"]].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert wide.loc[0, ["C3", "R3", "D3"]].tolist() == [1.0, 0.0, 1.0] and np.isnan(
        wide.loc[0, "L4"]
    )
    # Unit 1: censored on day 12 (node 4), its events missing there.
    assert wide.loc[1, "C4"] == 0.0 and np.isnan(wide.loc[1, "R4"]) and np.isnan(wide.loc[1, "L5"])
    # Unit 2: a visit and a censoring on day 11: censored in node 4, before its outcomes.
    assert wide.loc[2, "C4"] == 0.0 and np.isnan(wide.loc[2, "D4"])
    assert study.x70_name(study.CALIBRATION_ROW) == "ate_regimen[always vs never, cause1 @ t=6]"


def _synthetic_rows(
    replicates: int, rng: np.random.Generator, se_scale: float = 1.0
) -> pd.DataFrame:
    """Primary rows whose estimates are normal around the truth with their SE."""
    records = []
    for scenario, names in study.ESTIMANDS.items():
        truth = laws.TRUTHS[study.law_of(scenario)]
        for r in range(replicates):
            for name in names:
                se = 0.02
                estimate = truth[name] + rng.normal(0.0, se)
                record = {
                    "scenario": scenario,
                    "replicate": r,
                    "estimand": name,
                    "truth": truth[name],
                    "estimate": estimate,
                    "std_error": se * se_scale,
                    "covered": int(abs(estimate - truth[name]) <= 1.96 * se * se_scale),
                    "initial_estimate": estimate,
                    "variance_rel_sd": 0.05,
                    "event_share": 0.1,
                    "followers_at_last": 100.0,
                    "past_cap_share": 0.0,
                    "steps": 10.0,
                    "converged": 1.0,
                    "bounded_share": 0.0,
                    "jump_bound_share": 0.0,
                    "visit_bound_share": 0.0,
                    "largest_cap": 4.0,
                    "seconds": 1.0,
                }
                for cell, (_, count, rows) in study.SECONDARY.items():
                    run = scenario == "day_cf" and name in rows and r < count
                    value = truth[name] + rng.normal(0.0, se)
                    record.update(
                        {
                            f"{cell}_estimate": value if run else np.nan,
                            f"{cell}_std_error": se if run else np.nan,
                            f"{cell}_covered": float(abs(value - truth[name]) <= 1.96 * se)
                            if run
                            else np.nan,
                            f"{cell}_initial": value + 0.01 if run else np.nan,
                            f"{cell}_failed": 0.0 if run else np.nan,
                            f"{cell}_interval_refused": 0.0 if run else np.nan,
                            f"{cell}_seconds": 1.0 if run else np.nan,
                        }
                    )
                records.append(record)
    return pd.DataFrame.from_records(records)


@pytest.fixture(scope="module")
def synthetic() -> dict[str, pd.DataFrame]:
    rows = _synthetic_rows(study.PRIMARY_REPLICATES, np.random.default_rng(5))
    return {"rows": rows, **readings.artifacts(rows)}


def test_the_gates_pass_a_calibrated_synthetic_run(synthetic: dict[str, pd.DataFrame]) -> None:
    gates = readings.gates(synthetic)
    assert {name: len(frame) for name, frame in gates.items()} == dict.fromkeys(gates, 0)
    calibrated = synthetic[study.CALIBRATION_FILE]
    assert set(calibrated["cell"]) == {"calibration", "shrunk_se", "sparse_calibration"}
    assert len(synthetic[study.MECHANISM_FILE]) == 12 + 4
    assert set(synthetic[study.MISSPECIFIED_FILE]["cell"]) == set(readings.MISSPECIFIED)
    pairing = synthetic[study.PAIRING_FILE]
    assert len(pairing) == 18 and (pairing["pairs"] == study.PAIRED_REPLICATES).all()
    assert not pairing["diagnose"].any()
    assert study.account_failures(None, synthetic)["passed"].all()


def test_each_gate_fails_its_mutation(synthetic: dict[str, pd.DataFrame]) -> None:
    """Deliberate mutations: shrunk SEs, a biased mechanism_correct row and a plug-in at truth."""
    rows = synthetic["rows"].copy()
    calibration = rows["estimand"] == study.CALIBRATION_ROW
    rows.loc[calibration, "std_error"] *= 0.8
    rows["covered"] = ((rows["estimate"] - rows["truth"]).abs() <= 1.96 * rows["std_error"]).astype(
        int
    )
    target = rows["estimand"] == study.ROBUST_ROWS[0]
    rows.loc[target, "mechanism_correct_estimate"] += 0.012
    for name in study.PLUG_IN_ROWS:
        chosen = rows["estimand"] == name
        rows.loc[chosen, "mechanism_correct_initial"] = rows.loc[
            chosen, "mechanism_correct_estimate"
        ]
    gates = readings.gates(readings.artifacts(rows))
    assert len(gates[readings.CALIBRATION_GATE]) == 2
    assert len(gates[readings.MECHANISM_GATE]) == 1
    assert len(gates[readings.PLUG_IN_GATE]) == 4
    assert len(gates[readings.SHRUNK_GATE]) == 0


def test_the_sizing_arithmetic_is_the_plans() -> None:
    margins = study.MARGINS
    # The coverage clause alone at R = 2,400 (plan 5.7, recomputed for rev 7).
    for coverage, expected in ((0.92, 0.786), (0.925, 0.958), (0.93, 0.996), (0.94, 1.000)):
        assert sizing.coverage_floor_pass_probability(margins, 2_400, coverage) == pytest.approx(
            expected, abs=0.0015
        )
    # The calibration pair at target: 0.977 per row and 0.954 for the pair at 2,400.
    one = float(sizing.calibration_pass(margins, 2_400, np.array([1.0]), 0.95)[0])
    assert one == pytest.approx(0.977, abs=0.0015)
    assert pilot.at_target_pass(2_400) == pytest.approx(0.954, abs=0.002)
    assert pilot.at_target_pass(2_400) >= sizing.FAMILY_WISE_PASS
    # The bias clause at 2,400 passes with about 0.9993 at 0.08 SD; the shrunk control fires.
    assert float(sizing.bias_equivalent_probability(margins, 0.08, 2_400)) > 0.999
    assert sizing.calibration_control_probability(margins, 2_400, 0.85) > 0.999
    # A plug-in separated by 0.5 SD fires its control at 1,600 with probability above 0.999.
    assert sizing.bias_discriminated_probability(margins, 0.5, 1_600) > 0.999
    # A row at the design point passes at every rung, and the family rule is monotone.
    assert pilot.row_pass(2_400, 0.95, 1.0, 0.0) > 0.999
    assert pilot.row_pass(2_400, 0.925, 1.0, 0.0) < pilot.row_pass(4_800, 0.925, 1.0, 0.0)


def test_the_regeneration_refuses_until_the_pilot_agrees() -> None:
    reasons = regenerate.pilot_refusals()
    if PILOT.empty or pilot.draws(PILOT, "cont_cf") < pilot.DRAWS:
        assert reasons, "a declared run must wait for the committed pilot"
    else:
        assert reasons == [], reasons


def test_the_budget_table_is_filled() -> None:
    assert "SMOKE_" not in (study.__doc__ or "")
    diagnosis = (study.STUDY.artifacts / "pilot-diagnosis.md").read_text(encoding="utf-8")
    assert "SMOKE_" not in diagnosis and "--jobs" in diagnosis


def test_the_support_exemption_is_ready_for_registration() -> None:
    """The slug joins ``EXEMPT_STUDIES`` when the record registers (a pending study is not)."""
    reason = study.EXEMPT_REASON
    assert reason.strip().endswith(".") and "ContinuousLongitudinalTMLE" in reason
