"""The declaration of continuous-time-longitudinal (X6 plan rev 7, section 5.7; X6 part (e), E2)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.survival._longitudinal_nuisance import by_baseline
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
    assert study.PRIMARY_REPLICATES >= study.MECHANISM_REPLICATES
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
        study.VISIT_CAP_FILE,
        study.FLOORS_FILE,
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
        assert set(names) <= set(study.DAY_ROWS) and count <= study.PRIMARY_REPLICATES
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
        **{
            f"truth_capped/{law}/{regime}/{cap}": pilot.capped_seeds(law, regime, cap)
            for law in laws.SAMPLERS
            for regime in laws.REGIMES
            for cap in laws.CAPS
        },
        **{f"truth_panel/{regime}": pilot.panel_seeds(regime) for regime in laws.STATIC},
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


@pytest.fixture(scope="module")
def paths() -> laws.DayPaths:
    return laws.day_paths(laws.sample_day(300_000, 11))


@pytest.mark.parametrize("regime", ["never", "always"])
def test_the_true_weights_reproduce_the_truth_and_a_dropped_factor_does_not(
    paths: laws.DayPaths, regime: str
) -> None:
    """A nonzero witness: the true-weight IPW mean equals the regime's risk, and the IPW with the
    censoring factor or the treatment factor removed misses it, so a wrong weight cannot pass
    (review m4).  Always carries the plug-in separation of 0.53."""
    risks = laws.monte_carlo_risks("ct_day", regime, [12], chunk=300_000)
    for cause in laws.CAUSES:
        values = laws.ipw_contributions(paths, regime, cause, 30.0)
        se = float(np.std(values, ddof=1) / np.sqrt(values.size))
        truth, _ = risks[(cause, 30.0)]
        assert abs(values.mean() - truth) < 4.5 * np.hypot(se, np.sqrt(truth / 300_000))
    for field in ("log_sc", "log_pi"):
        mutated = laws.DayPaths(
            paths.time,
            paths.event,
            paths.baseline_follow,
            paths.followed,
            np.zeros_like(paths.log_pi) if field == "log_pi" else paths.log_pi,
            np.zeros_like(paths.log_sc) if field == "log_sc" else paths.log_sc,
        )
        biased = laws.ipw_contributions(mutated, regime, 1, 30.0).mean()
        assert abs(biased - risks[(1, 30.0)][0]) > 0.01, field


def test_the_capped_risks_close_on_the_full_risk_and_open_at_a_small_cap() -> None:
    r"""A nonzero witness for :math:`\psi^{(K)}` (review S1): at a cap no unit reaches it is the
    full risk, at :math:`K = 2` never's cause-1 day-30 risk moves by more than 0.006 (about
    -0.009), and the committed table is the pilot's and holds the declared caps."""
    seeds = [41]
    full = laws.monte_carlo_risks("ct_day", "never", seeds, chunk=1_000_000)
    wide = laws.monte_carlo_risks("ct_day", "never", seeds, chunk=1_000_000, cap=12)
    tight = laws.monte_carlo_risks("ct_day", "never", seeds, chunk=1_000_000, cap=2)
    se = np.sqrt(2 * 0.37 * 0.63 / 1_000_000)
    assert abs(wide[(1, 30.0)][0] - full[(1, 30.0)][0]) < 4.0 * se
    assert tight[(1, 30.0)][0] - full[(1, 30.0)][0] < -0.006
    table = laws.CAPPED_RISKS
    assert set(table) == {"ct", "ct_day"}
    assert all(set(table[law][r]) == set(laws.CAPS) for law in table for r in laws.REGIMES)
    if not PILOT.empty and (PILOT["kind"] == "truth_capped").any():
        assert pilot.capped_table(PILOT) == table
    name = "ate_regimen[always vs never, cause1 @ t=30]"
    expected = table["ct_day"]["always"][4][(1, 30.0)] - table["ct_day"]["never"][3][(1, 30.0)]
    assert laws.capped_truth("ct_day", name, 4.0, 3.0) == pytest.approx(expected)
    assert np.isnan(laws.capped_truth("ct_day", name, 4.0, 99.0))


def test_the_panel_risk_is_the_coarsened_regime() -> None:
    """The panel's risk (review S5): its resets are the node starts, a fresh draw reproduces the
    committed value, and it differs from the full risk (a nonzero witness)."""
    assert (*laws.PANEL_RESETS, 30) == study.NODE_ENDS
    fresh = laws.monte_carlo_risks(
        "ct_day", "never", [43], chunk=1_000_000, resets=laws.PANEL_RESETS
    )
    value, units = fresh[(1, 30.0)]
    assert abs(laws.PANEL_RISKS["never"][(1, 30.0)] - value) < 4.5 * np.sqrt(
        value * (1 - value) / units
    )
    assert laws.PANEL_RISKS["never"][(1, 30.0)] - laws.TRUTH_RISKS["ct_day"]["never"][(1, 30.0)] < (
        -0.003
    )
    assert set(laws.PANEL_TRUTHS) == set(study.PAIRED_ROWS)
    if not PILOT.empty and (PILOT["kind"] == "truth_panel").any():
        assert pilot.panel_table(PILOT) == laws.PANEL_RISKS


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
    treatment = laws.CONFIGURATIONS["outcome_correct"]()["treatment_learner"]
    TX = X[:, 1:].copy()
    TX[:, -2] = rng.integers(0, 3, 400)  # the visit count: zero at baseline
    classifier = treatment.fit(TX, y)
    shifted = TX.copy()
    shifted[:, laws.TREATMENT_L] += 3.0
    assert np.array_equal(classifier.predict_proba(TX), classifier.predict_proba(shifted))
    # It is the default's stratified main-terms member without L, not a pooled fit.
    reduced = np.delete(TX, laws.TREATMENT_L, axis=1)
    by_hand = laws.main_logistic().fit(by_baseline(reduced, reduced[:, -2] == 0), y)
    np.testing.assert_allclose(
        classifier.predict_proba(TX),
        by_hand.predict_proba(by_baseline(reduced, reduced[:, -2] == 0)),
        atol=1e-10,
    )
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


_CAP_KEYS = ("cap", "reference_cap", "capped_truth")


def _caps(law: str, name: str) -> dict[str, float]:
    """Every regime at cap 4, the modal :math:`K^*` (``pilot-diagnosis.md``)."""
    reference = 4.0 if name.startswith("ate_regimen") else np.nan
    return {
        "cap": 4.0,
        "reference_cap": reference,
        "capped_truth": laws.capped_truth(law, name, 4.0, reference),
    }


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
                    "implementation": study.SUBJECT,
                    "inference_scale": "identity",
                    "scenario": scenario,
                    "replicate": r,
                    "estimand": name,
                    "truth": truth[name],
                    "estimate": estimate,
                    "inference_estimate": estimate,
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
                    **_caps(study.law_of(scenario), name),
                }
                for cell, (_, count, rows) in study.SECONDARY.items():
                    run = scenario == "day_cf" and name in rows and r < count
                    paired = cell == "ltmle_daily"
                    centre = laws.PANEL_TRUTHS.get(name, np.nan) if paired else truth[name]
                    value = centre + rng.normal(0.0, se)
                    caps = _caps("ct_day", name) if not paired else dict.fromkeys(_CAP_KEYS, np.nan)
                    record.update(
                        {
                            f"{cell}_estimate": value if run else np.nan,
                            f"{cell}_std_error": se if run else np.nan,
                            f"{cell}_covered": float(abs(value - centre) <= 1.96 * se)
                            if run
                            else np.nan,
                            f"{cell}_initial": value + 0.01 if run else np.nan,
                            f"{cell}_failed": 0.0 if run else np.nan,
                            f"{cell}_interval_refused": 0.0 if run else np.nan,
                            f"{cell}_seconds": 1.0 if run else np.nan,
                            **{
                                f"{cell}_{key}": value if run else np.nan
                                for key, value in caps.items()
                            },
                            f"{cell}_error": "",
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
    assert (pairing["expected_difference"] == pairing["panel_truth"] - pairing["truth"]).all()
    assert study.account_failures(None, synthetic)["passed"].all()
    caps = synthetic[study.VISIT_CAP_FILE]
    assert set(caps["cell"]) == {*study.CELLS, "mechanism_correct", *readings.MISSPECIFIED}
    assert (caps["fits_without_capped_truth"] == 0).all() and (caps["caps"] != "").all()
    floors = synthetic[study.FLOORS_FILE]
    assert len(floors) == len(study.DAY_ROWS) + len(study.CONT_ROWS)
    assert floors["rule_passed"].all()
    values = readings.floor_values(floors)
    assert values["longitudinal_rel_sd"] == 0.05 and values["rules_supported"]
    assert values["fewest_followers"] == 100.0 and not values["jump_bound_acted"]


def test_a_conservative_calibration_row_does_not_void_through_its_control(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """Review B1: at a true SE ratio of 1.08, the direction plan 5.7 expects, the calibration
    rows read red (and then conservative, post-run), but each shrunk control is read against its
    row's own interval and passes, where the band reading would have failed it."""
    rows = synthetic["rows"].copy()
    calibration = rows["estimand"] == study.CALIBRATION_ROW
    rows.loc[calibration, "std_error"] *= 1.08
    rows["covered"] = ((rows["estimate"] - rows["truth"]).abs() <= 1.96 * rows["std_error"]).astype(
        int
    )
    table = readings.calibration(rows)
    gates = readings.gates({**readings.artifacts(rows)})
    assert len(gates[readings.CALIBRATION_GATE]) == 2
    assert len(gates[readings.SHRUNK_GATE]) == 0
    shrunk = table.loc[table["cell"] == "shrunk_se"]
    assert (shrunk["reading"] == "below the row's interval").all()
    assert (shrunk["se_ratio_ci_upper"] > study.MARGINS.calibration_se_ratio[0]).all()


def test_the_counts_never_put_mechanism_correct_above_the_primary_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review S2: mechanism_correct runs inside day_cf's replicates."""
    monkeypatch.setattr(pilot, "family_pass", lambda rows, primary, mechanism: mechanism >= 3_200)
    monkeypatch.setattr(pilot, "cost", lambda rows, primary, mechanism: primary + mechanism)
    assert pilot.counts(pd.DataFrame()) == (3_200, 3_200)
    for primary in study.PRIMARY_RUNGS:
        for mechanism in study.MECHANISM_RUNGS:
            monkeypatch.setattr(
                pilot, "family_pass", lambda rows, p, m, q=primary, k=mechanism: (p, m) == (q, k)
            )
            chosen = pilot.counts(pd.DataFrame())
            assert chosen is None or chosen[1] <= chosen[0]


def _pilot_rows(cell: str, draws: int, ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    truth = laws.TRUTHS[pilot.CELLS[cell][0]][study.CALIBRATION_ROW]
    estimate = truth + rng.normal(0.0, 0.03, draws)
    se = 0.03 * ratio
    return pd.DataFrame(
        {
            "kind": "primary",
            "label": cell,
            "replicate": np.arange(draws),
            "estimand": study.CALIBRATION_ROW,
            "estimate": estimate,
            "std_error": se,
            "covered": (np.abs(estimate - truth) <= 1.96 * se).astype(int),
            "refusal": "",
        }
    )


def test_only_a_confirmed_calibration_departure_needs_a_diagnosis() -> None:
    """Review S4: a calibrated 800-draw pilot is a chance reading; a ratio of 1.3 departs."""
    rng = np.random.default_rng(3)
    assert not pilot.calibration_departure(_pilot_rows("cont_cf", 800, 1.0, rng), "cont_cf")
    assert pilot.calibration_departure(_pilot_rows("cont_cf", 800, 1.3, rng), "cont_cf")
    # The at-target shrunk power, the refusal's input since review B1, needs no pilot row.
    assert pilot.shrunken_power(study.PRIMARY_REPLICATES) >= regenerate.SHRUNK_POWER


def test_a_secondary_fit_records_only_a_recordable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review S6: a DataError is a recorded failure with its class; a TypeError stops the run."""
    from cleverly.exceptions import DataError

    def refuse(frame: pd.DataFrame, subject: object) -> None:
        raise DataError("no event of cause 1 before the first target time")

    monkeypatch.setattr(study, "fit_with", refuse)
    out = study._secondary("both_wrong", 0, pd.DataFrame())
    assert {row["error"] for row in out.values()} == {"DataError"}
    assert all(row["failed"] == 1.0 for row in out.values())

    def defect(frame: pd.DataFrame, subject: object) -> None:
        raise TypeError("a defect")

    monkeypatch.setattr(study, "fit_with", defect)
    with pytest.raises(TypeError):
        study._secondary("both_wrong", 0, pd.DataFrame())


def test_the_floor_rules() -> None:
    """Review S7: the rung rule stops at the first red row; the window stops short of reds."""
    assert readings.rel_sd_floor([0.1, 0.2, 0.3], [True, True, True]) == 0.3
    assert readings.rel_sd_floor([0.1, 0.2, 0.3], [True, False, True]) == 0.1
    assert readings.rel_sd_floor([0.1, 0.2], [False, True]) == 0.0
    shares = [0.02, 0.05, 0.1, 0.2, 0.3]
    assert readings.share_window(shares, [True] * 5) == (0.02, 0.3)
    assert readings.share_window(shares, [False, True, True, True, False]) == (0.05, 0.2)
    table = pd.DataFrame(
        {
            "regime_kind": ["static", "static", "rule"],
            "median_rel_sd": [0.1, 0.2, 0.15],
            "median_event_share": [0.1, 0.2, 0.15],
            "fewest_followers": [80.0, 60.0, 40.0],
            "largest_past_cap_share": [0.05, 0.09, 0.2],
            "q99_past_cap_share": [0.04, 0.07, 0.1],
            "largest_bounded_share": [0.0, 0.001, 0.0],
            "largest_jump_bound_share": [0.0, 0.0, 0.0],
            "rule_passed": [True, True, False],
        }
    )
    values = readings.floor_values(table)
    assert values["longitudinal_rel_sd"] == 0.2 and not values["rules_supported"]
    assert values["rules_rel_sd"] is None and values["fewest_followers"] == 60.0
    assert values["largest_past_cap_share"] == 0.09


def test_the_omitted_columns_are_l_in_the_engines_designs() -> None:
    """Review m5: ``HAZARD_L`` and ``TREATMENT_L`` are the engine's columns of :math:`L(s_k)`."""
    from cleverly.survival import _longitudinal_nuisance as nuisance
    from cleverly.survival._longitudinal_data import ContinuousLongitudinalData

    frame = pd.DataFrame(
        {
            "id": [0, 0, 1],
            "start": [0.0, 3.0, 0.0],
            "stop": [3.0, 9.0, 7.0],
            "event": [0, 1, 2],
            "W1": [1.0, 1.0, 0.0],
            "W2": [4.0, 4.0, 2.0],
            "L": [0.0, 7.0, 5.0],
            "A": [1, 0, 1],
        }
    )
    data = ContinuousLongitudinalData.from_frame(frame, **laws.COLUMNS, causes=laws.CAUSES)
    coded = np.asarray(frame["A"], dtype=int)
    visits = np.arange(len(frame))
    hazard = nuisance.design(data, coded, visits)
    treatment = nuisance.treatment_design(data, coded, visits)
    assert hazard[:, laws.HAZARD_L].tolist() == [0.0, 7.0, 5.0]
    assert treatment[:, laws.TREATMENT_L].tolist() == [0.0, 7.0, 5.0]


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
    if not PILOT.empty and not any(pilot.draws(PILOT, cell) for cell in pilot.CELLS):
        # A fresh primary pilot: every row it keeps still agrees with the law module.
        assert reasons == [f"{regenerate.FRESH_PILOT}; run --kinds primary"], reasons
    elif PILOT.empty or pilot.draws(PILOT, "cont_cf") < pilot.DRAWS:
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
